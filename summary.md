# Meeting Summarizer Redesign — richer output for a 128k-context gateway model

## Why the current design is thin

Two independent caps are strangling this pipeline well below what the gateway can do:

- `transcript.to_prompt_text(max_chars=6000)` — truncates the transcript at **6,000 characters**
  before it ever reaches the model. For a real meeting this is a couple of minutes of dialogue.
- `SUMMARY_MAX_TOKENS=4096` — the *output* budget, shared with qwen's internal thinking trace,
  which is why `EmptySummaryError` exists at all (the model burns the whole budget reasoning and
  never emits a final `text` block).
- The schema itself asks for three arrays of **bare strings** (`topics`, `decisions`,
  `action_items`) — by construction there's nowhere for the model to put context, rationale, or
  nuance even if it wanted to.

None of these match a 128k-context model. The fix touches four files: `summarizer.py` (prompt +
budgets), `summary_parse.py` (schema), `poster.py` (how the richer content survives Discord's
limits), `config.py` (new/missing fields).

## 1. New output schema

Keep JSON as the only channel (don't regress the reliability work in `summary_parse.py`), but
give every bucket a place to carry context instead of a bare label:

```json
{
  "overview": "ย่อหน้าสรุปภาพรวมการประชุม 1 ย่อหน้า",
  "topics": [
    {"title": "...", "detail": "..."}
  ],
  "decisions": [
    {"decision": "...", "rationale": "..."}
  ],
  "action_items": [
    {"action": "...", "owner": "...", "due": "..."}
  ],
  "open_questions": ["..."]
}
```

New system prompt (`_SYSTEM_PROMPT` in `summarizer.py`):

```python
_SYSTEM_PROMPT = """\
คุณคือผู้ช่วยสรุปการประชุมภาษาไทยที่ละเอียดและมีบริบทครบถ้วน
จงตอบเป็นภาษาไทยเท่านั้น และให้ส่งออกเฉพาะ JSON ตามโครงสร้างต่อไปนี้
โดยไม่มีเครื่องหมาย markdown fence และไม่มีข้อความอื่นใดนอกเหนือจาก JSON:

{
  "overview": "...",
  "topics": [{"title": "...", "detail": "..."}],
  "decisions": [{"decision": "...", "rationale": "..."}],
  "action_items": [{"action": "...", "owner": "...", "due": "..."}],
  "open_questions": ["..."]
}

ความหมายของแต่ละช่อง:
- overview: ย่อหน้าสรุปภาพรวมการประชุม 3-6 ประโยค ครอบคลุมบริบท ลำดับเหตุการณ์
  และน้ำเสียงโดยรวมของการสนทนา
- topics: หัวข้อที่พูดคุยในการประชุม แต่ละหัวข้อมี "title" (ชื่อหัวข้อสั้น ๆ)
  และ "detail" (อธิบายเนื้อหาการสนทนาในหัวข้อนั้นอย่างละเอียด 2-4 ประโยค
  รวมถึงมุมมองต่าง ๆ ที่ถูกพูดถึง)
- decisions: การตัดสินใจที่เกิดขึ้น แต่ละรายการมี "decision" (สิ่งที่ตัดสินใจ)
  และ "rationale" (เหตุผลหรือบริบทที่นำไปสู่การตัดสินใจนั้น ถ้าไม่มีให้ใช้ "")
- action_items: รายการสิ่งที่ต้องทำ โดยระบุ "owner" หากทราบผู้รับผิดชอบ
  (ถ้าไม่ทราบให้ใช้ null) และ "due" หากมีการระบุกำหนดเวลา (ถ้าไม่มีให้ใช้ null)
- open_questions: ประเด็นหรือคำถามที่ถูกพูดถึงแต่ยังไม่ได้ข้อสรุปในที่ประชุม
  (ถ้าไม่มีให้ใช้ [])

ห้ามละเว้นบริบทที่สำคัญ จงสรุปให้ครบถ้วนและมีรายละเอียดเพียงพอที่จะเข้าใจ
การประชุมได้โดยไม่ต้องฟังเทปซ้ำ
"""
```

`_USER_SUFFIX` gets the null/empty-type reminder extended:

```python
_USER_SUFFIX = (
    "\n\nโปรดตอบเฉพาะ JSON ตามรูปแบบที่กำหนดเท่านั้น "
    "ห้ามมีข้อความอื่นนอกจาก JSON และถ้าส่วนใดไม่มีเนื้อหาให้ใช้ [] หรือ \"\" "
    "หรือ null ตามชนิดของช่องนั้น"
)
```

## 2. Budget changes (this is the actual "128k context" part)

| Field | Old | New default | Why |
|---|---|---|---|
| `max_prompt_chars` | 6000 | **48000** | Thai/qwen tokenizer ratio is unverified, so this stays conservative rather than maxing the window: ~48k chars is comfortably under 128k tokens even at a pessimistic ~1.5 chars/token, leaving headroom for system prompt + thinking + output. Raise further once you've confirmed the real ratio against the gateway. |
| `summary_max_tokens` | 4096 | **8192** | The richer schema (overview + per-item detail/rationale + open_questions) needs meaningfully more output tokens, and qwen spends part of the budget on thinking before any of that appears — this is the direct fix for `EmptySummaryError` becoming *more* likely, not less, if you only widen the schema. |
| `summarize_timeout_seconds` | 90 | **180** | Longer input + longer output legitimately takes longer; this is the SDK-level ceiling, independent of the new stall guard below. |
| `stall_timeout_seconds` *(new — required by the pasted streaming code, currently missing from `config.py`)* | — | **20.0** | Per-event "no progress" guard inside the stream loop. |
| `repetition_window_chars` *(new, same reason)* | — | **300** | Matches the "multi-hundred character window" the docstring already describes. |
| `repetition_min_repeats` *(new, same reason)* | — | **3** | Three identical consecutive windows before declaring a loop — avoids false positives on legitimately repetitive schema output (e.g. several `"owner":` keys in a row). |

All six stay overridable via `.env` — same pattern as the existing `_OPTIONAL_FLOAT_ENV` tuple.

## 3. `summary_parse.py` — new shape, still never-raising

```python
@dataclass
class TopicItem:
    title: str
    detail: str = ""

@dataclass
class DecisionItem:
    decision: str
    rationale: str = ""

@dataclass
class ActionItem:
    action: str
    owner: str | None = None
    due: str | None = None          # new field

@dataclass
class Summary:
    overview: str
    topics: list[TopicItem]
    decisions: list[DecisionItem]
    action_items: list[ActionItem]
    open_questions: list[str]
    raw: str
```

Parsing rules to preserve the existing tolerance:
- **Backward compatible inputs**: if the model (or a stale gateway) reverts to plain strings for
  `topics`/`decisions`, wrap them as `TopicItem(title=s, detail="")` /
  `DecisionItem(decision=s, rationale="")` rather than failing — same spirit as today's
  `_as_str_list` handling both strings and dicts.
- Add Thai key aliases: `overview` → `ภาพรวม`/`สรุปภาพรวม`; `open_questions` →
  `คำถามที่ยังไม่ได้ข้อสรุป`/`ประเด็นค้าง`; `decision`/`rationale` →
  `การตัดสินใจ`/`เหตุผล`; `detail` → `รายละเอียด`.
- Markdown fallback gets two new headers (`ภาพรวม`/`Overview`,
  `คำถามที่ยังไม่ได้ข้อสรุป`/`Open Questions`) and reuses the existing
  `"label: value"` split regex to pull `title`/`detail` and `decision`/`rationale` apart when a
  bullet contains a separator, falling back to a bare title/decision with empty detail otherwise.
- **Raw last-resort fallback** changes from `topics=[text]` to `overview=text` — a blob of
  unstructured text is semantically an overview, not a topic list.

## 4. `poster.py` — compact embed + full detail as an attached file

The richer content **will not fit** in Discord embed fields (1024 chars/field, 6000/embed total).
Rather than truncating and losing the "more context" you asked for, split the output:

- **Embed** (unchanged shape, richer content): description = `overview`; topic/decision fields
  render `• **{title}** — {detail}` bullets, hard-truncated per field with a
  `"…ดูรายละเอียดเพิ่มเติมในไฟล์แนบ"` note if they'd exceed 1024 chars; add a new
  `"คำถามที่ยังไม่ได้ข้อสรุป"` field for `open_questions`.
- **Attached file** (new): a generated Markdown document with the *complete* summary — full
  `detail`/`rationale` text for every topic and decision, the full action item table, all open
  questions — sent via `discord.File` in the same `channel.send(embed=..., file=...)` call. This
  is where "more summarized context" actually lives; the embed is just a scannable index into it.

## 5. `.env.example` / spec sync

`meeting_bot_spec.md`'s acceptance criteria require `.env.example` keys to match `config.py`
exactly — add the three new keys there with the same comment style as the existing ones, and
update `MAX_PROMPT_CHARS`/`SUMMARY_MAX_TOKENS`/`SUMMARIZE_TIMEOUT_SECONDS` defaults and comments.

---

## Prompt for Claude Code

Paste this into Claude Code at the repo root (`sumbit/`):

```
Redesign the meeting summarizer to produce richer, more contextual summaries instead of
terse bullet lists, and to take advantage of the gateway's 128k context window instead of
truncating transcripts at 6000 characters. Read CLAUDE.md first for project conventions
(pure-module import rule, never-raising parser philosophy, Thai-language conventions).

1. meeting_bot/summarizer.py
   - Replace this file with the streaming version below (it already has stall/repetition
     detection wired to self.cfg.stall_timeout_seconds / repetition_window_chars /
     repetition_min_repeats — those config fields don't exist yet, add them in step 4):
     [PASTE THE DOCUMENT I GAVE YOU HERE]
   - Replace _SYSTEM_PROMPT and _USER_SUFFIX with the versions in
     summarizer_redesign_spec.md section 1 (new schema: overview, topics with
     title+detail, decisions with decision+rationale, action_items with owner+due,
     open_questions).

2. meeting_bot/summary_parse.py
   - Replace the flat topics/decisions: list[str] shape with TopicItem, DecisionItem
     dataclasses per summarizer_redesign_spec.md section 3. Add `overview: str` and
     `open_questions: list[str]` to Summary. Add `due: str | None` to ActionItem.
   - Keep the JSON -> markdown -> raw fallback chain and the "never raises" contract intact.
   - Backward-compat: if topics/decisions arrive as plain strings (not dicts), wrap them
     rather than dropping data or erroring.
   - Raw last-resort fallback should set `overview=text`, not `topics=[text]`.
   - Add Thai key aliases as specified.

3. meeting_bot/poster.py
   - Keep build_embed() but adapt field rendering to the new TopicItem/DecisionItem shape:
     "• **{title}** — {detail}" bullets, truncated to fit Discord's 1024-char field limit
     with a "…ดูรายละเอียดเพิ่มเติมในไฟล์แนบ" note when truncated. Add a field for
     open_questions.
   - Add a new function that renders the full Summary (including untruncated detail/
     rationale) as a Markdown document, and have Poster.post() attach it via discord.File
     alongside the embed in the same channel.send() call.

4. meeting_bot/config.py
   - Add stall_timeout_seconds (default 20.0), repetition_window_chars (default 300),
     repetition_min_repeats (default 3) to the Config dataclass and _OPTIONAL_FLOAT_ENV /
     load_config, following the existing pattern exactly.
   - Change defaults: max_prompt_chars 6000 -> 48000, summary_max_tokens 4096 -> 8192,
     summarize_timeout_seconds 90 -> 180.

5. .env.example
   - Add STALL_TIMEOUT_SECONDS, REPETITION_WINDOW_CHARS, REPETITION_MIN_REPEATS with
     comments explaining what they guard against (streaming stall vs. exact-repeat loop
     detection in qwen's thinking trace).
   - Update MAX_PROMPT_CHARS, SUMMARY_MAX_TOKENS, SUMMARIZE_TIMEOUT_SECONDS defaults/
     comments to match config.py, and note the "128k context, conservative chars/token
     estimate" reasoning inline so a future editor doesn't silently revert it.

6. tests/test_summary_parse.py
   - Rewrite fixtures for the new schema (overview, topics as {title, detail}, decisions
     as {decision, rationale}, action_items with due, open_questions).
   - Add a backward-compat test: plain-string topics/decisions still parse into
     TopicItem/DecisionItem with empty detail/rationale.
   - Update the markdown-fallback test to cover the new overview/open_questions headers.
   - Update the garbage/last-resort test to assert overview == the raw text and
     topics == [].

7. CLAUDE.md and meeting_bot_spec.md
   - Update the "Summarization" and "Summary parser" bullets under Meeting Summary
     Discord Bot to describe the new schema and the embed+file-attachment split.
   - Update meeting_bot_spec.md's summary_parse.py / summarizer.py / poster.py module
     contracts and the .env.example section to match, since build_bot_task.py feeds this
     file to a jailed agent as ground truth -- if it's stale, the next Shepherd build
     regenerates the old terse version.

Run pytest -q when done and fix any failures. Do not touch audio.py, chunker.py,
transcriber.py, sink.py, bot.py's voice-trigger logic, or the streaming stall/repetition
detection algorithm itself -- only wire config into it.
```

**Before you run this**: swap in your own numbers for the table in section 2 if 48000/8192/180
don't match what you've actually seen the gateway handle — those are conservative starting
points, not measured limits. Once you confirm the gateway streams real `content_block_delta`
events (the open assumption already flagged in your pasted docstring), it's worth a quick probe
to find the real Thai chars-per-token ratio so `max_prompt_chars` can be sized precisely instead
of padded for safety.