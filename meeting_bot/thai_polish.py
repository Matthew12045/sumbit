"""Thai-writing polish pass for meeting summaries.

Uses the **kien-thai** skill's audit+fix loop to iteratively improve the
Thai prose in a meeting summary until the model reports zero edits needed.
Only ``overview``, ``topics[].detail``, and ``decisions[].rationale`` are
polished — everything else is passed through byte-identical.

Uses **OpenTyphoon**'s OpenAI-compatible API (``openai`` SDK) with
``typhoon-v2.5-30b-a3b-instruct``.  If convergence is not reached or any
API failure occurs, the **original** summary is returned unmodified.

**Never raises past this module** — all exceptions are caught and logged.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from meeting_bot.summary_parse import ActionItem, DecisionItem, Summary, TopicItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill bundle loading
# ---------------------------------------------------------------------------

_SKILL_DIR = Path(__file__).parent / "thai_skill"


def _load_skill_bundle() -> str:
    """Concatenate all skill reference files into one system-prompt bundle."""
    parts: list[str] = []

    skill_md = _SKILL_DIR / "SKILL.md"
    if skill_md.exists():
        parts.append(skill_md.read_text(encoding="utf-8"))

    refs_dir = _SKILL_DIR / "references"
    if refs_dir.is_dir():
        for ref_file in sorted(refs_dir.iterdir()):
            if ref_file.suffix == ".md":
                parts.append(ref_file.read_text(encoding="utf-8"))

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class _PolishInput:
    """JSON-serialisable snapshot of the fields to polish."""

    overview: str
    topics: list[dict[str, str]]  # [{title, detail}, ...]
    decisions: list[dict[str, str]]  # [{decision, rationale}, ...]
    action_items: list[dict[str, str | None]]  # [{action, owner, due}, ...]
    open_questions: list[str]


@dataclass
class _PolishOutput:
    """Parsed output from the model's audit+fix pass."""

    overview: str
    topics: list[dict[str, str]]
    decisions: list[dict[str, str]]
    action_items: list[dict[str, str | None]]
    open_questions: list[str]
    edits_needed: bool
    edit_notes: list[str]


def _to_input(summary: Summary) -> _PolishInput:
    """Convert a Summary to the flat polish input."""
    return _PolishInput(
        overview=summary.overview or "",
        topics=[{"title": t.title, "detail": t.detail} for t in summary.topics],
        decisions=[{"decision": d.decision, "rationale": d.rationale} for d in summary.decisions],
        action_items=[
            {"action": a.action, "owner": a.owner, "due": a.due}
            for a in summary.action_items
        ],
        open_questions=list(summary.open_questions) if summary.open_questions else [],
    )


def _to_output(
    data: Any,
    pass_num: int = 0,
    expected: "_PolishInput | None" = None,
) -> _PolishOutput:
    """Validate the model's JSON payload and parse it into ``_PolishOutput``.

    Raises ``ValueError`` (never coerces silently) on: non-object payloads,
    missing/wrong-typed required fields (``overview``, ``edits_needed``,
    list-shaped ``topics``/``decisions`` with dict items), or a
    topics/decisions array whose length differs from the input the model was
    given. Callers catch the error and fall back to the original summary —
    an unparseable pass must NEVER masquerade as an empty "no edits" result
    (that was the data-loss bug this module used to have).
    """
    where = f"thai_polish pass {pass_num}" if pass_num else "thai_polish"
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(data).__name__}")

    def _required_str(key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str):
            raise ValueError(
                f"{where}: field {key!r} must be a string, got {type(value).__name__}"
            )
        return value

    def _list_of_dicts(key: str) -> list:
        value = data.get(key)
        if not isinstance(value, list):
            raise ValueError(
                f"{where}: field {key!r} must be a list, got {type(value).__name__}"
            )
        for item in value:
            if not isinstance(item, dict):
                raise ValueError(
                    f"{where}: field {key!r} must contain objects, "
                    f"got {type(item).__name__} item"
                )
        return value

    overview = _required_str("overview")

    edits_raw = data.get("edits_needed")
    if not isinstance(edits_raw, bool):
        raise ValueError(
            f"{where}: field 'edits_needed' must be a boolean, got "
            f"{type(edits_raw).__name__}"
        )

    topics = _list_of_dicts("topics")
    decisions = _list_of_dicts("decisions")

    if expected is not None:
        if len(topics) != len(expected.topics):
            raise ValueError(
                f"{where}: got {len(topics)} topics for "
                f"{len(expected.topics)} input topics — refusing to coerce"
            )
        if len(decisions) != len(expected.decisions):
            raise ValueError(
                f"{where}: got {len(decisions)} decisions for "
                f"{len(expected.decisions)} input decisions — refusing to coerce"
            )

    action_items = data.get("action_items", [])
    if not isinstance(action_items, list):
        raise ValueError(f"{where}: field 'action_items' must be a list")
    open_questions = data.get("open_questions", [])
    if not isinstance(open_questions, list):
        raise ValueError(f"{where}: field 'open_questions' must be a list")
    edit_notes = data.get("edit_notes", [])
    if not isinstance(edit_notes, list):
        raise ValueError(f"{where}: field 'edit_notes' must be a list")

    return _PolishOutput(
        overview=overview,
        topics=topics,
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        edits_needed=edits_raw,
        edit_notes=edit_notes,
    )


def _reconstruct(original: Summary, polished: _PolishOutput) -> Summary:
    """Rebuild a Summary from the polished prose fields.

    Protected fields are taken from the **original** Summary regardless of
    what the model returned: topic titles and decision labels pair
    positionally with the polished details/rationales (lengths are enforced
    by :func:`_to_output`), and ``action_items``/``open_questions`` are
    passed through byte-identical.
    """
    topics = [
        TopicItem(title=orig.title, detail=pol.get("detail", ""))
        for orig, pol in zip(original.topics, polished.topics)
    ]
    decisions = [
        DecisionItem(decision=orig.decision, rationale=pol.get("rationale", ""))
        for orig, pol in zip(original.decisions, polished.decisions)
    ]
    action_items = list(original.action_items)
    open_questions = list(original.open_questions)
    raw = "\n".join(
        (
            f"**ภาพรวม**\n\n{polished.overview}" if polished.overview else "",
            (
                "**หัวข้อ**\n\n"
                + "\n".join(
                    (
                        f"- **{t.title}** — {t.detail}"
                        if t.detail
                        else f"- **{t.title}**"
                    )
                    for t in topics
                )
                if topics
                else ""
            ),
            (
                "**การตัดสินใจ**\n\n"
                + "\n".join(
                    (
                        f"- **{d.decision}** — {d.rationale}"
                        if d.rationale
                        else f"- **{d.decision}**"
                    )
                    for d in decisions
                )
                if decisions
                else ""
            ),
        )
    )
    return Summary(
        overview=polished.overview or "",
        topics=topics,
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

# Only the polished prose participates in convergence detection; protected
# fields are structurally unchanged (see _reconstruct).
_Fields = tuple[str, tuple[str, ...], tuple[str, ...]]


def _fields(data: _PolishInput) -> _Fields:
    """Snapshot of the polisable fields: (overview, topic details, rationales)."""
    return (
        data.overview,
        tuple(t.get("detail", "") for t in data.topics),
        tuple(d.get("rationale", "") for d in data.decisions),
    )


def _blanks_text(prev: _Fields, nxt: _Fields) -> bool:
    """True when the model emptied a previously-nonempty polished field."""
    prev_overview, prev_topics, prev_decisions = prev
    next_overview, next_topics, next_decisions = nxt
    if prev_overview and not next_overview:
        return True
    if any(a and not b for a, b in zip(prev_topics, next_topics)):
        return True
    return any(a and not b for a, b in zip(prev_decisions, next_decisions))


def _input_from_output(result: _PolishOutput) -> _PolishInput:
    """Next-loop input built from a pass's polished fields."""
    return _PolishInput(
        overview=result.overview,
        topics=[dict(t) for t in result.topics],
        decisions=[dict(d) for d in result.decisions],
        action_items=list(result.action_items),
        open_questions=list(result.open_questions),
    )


# ---------------------------------------------------------------------------
# ThaiPolisher
# ---------------------------------------------------------------------------

class ThaiPolisher:
    """Audit+fix loop for Thai meeting-summary prose.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible API base URL (default: OpenTyphoon).
    auth_token : str
        API key for the OpenAI-compatible endpoint.
    model : str
        Model name (default: ``typhoon-v2.5-30b-a3b-instruct``).
    max_passes : int
        Safety cap on audit+fix iterations (default: 20).
    timeout_seconds : float
        Per-pass timeout in seconds (default: 120.0).
    """

    def __init__(
        self,
        base_url: str = "https://api.opentyphoon.ai/v1",
        auth_token: str = "",
        model: str = "typhoon-v2.5-30b-a3b-instruct",
        max_passes: int = 20,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.auth_token = auth_token
        self.model = model
        self.max_passes = max_passes
        self.timeout_seconds = timeout_seconds
        self._skill_bundle = _load_skill_bundle()
        self._client = OpenAI(
            base_url=base_url,
            api_key=auth_token,
        )
        # Outcome of the most recent polish() call: {passes, outcome} where
        # outcome ∈ converged | final_edit | blanked | cap | hard_failure |
        # skipped. For log/probe visibility (kienthai.md reporting).
        self.last_stats: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def polish(self, summary: Summary) -> Summary:
        """Run the audit+fix loop and return the polished Summary object.

        If convergence is not reached or any error occurs, returns the
        **original** summary unchanged. Never raises.
        """
        if not summary.raw or not summary.raw.strip():
            self.last_stats = {"passes": 0, "outcome": "skipped"}
            return summary

        current = _to_input(summary)

        for pass_num in range(1, self.max_passes + 1):
            try:
                result = self._one_pass(current, pass_num)
            except Exception:
                logger.warning(
                    "thai_polish: hard failure on pass %d — returning original",
                    pass_num,
                    exc_info=True,
                )
                self.last_stats = {"passes": pass_num, "outcome": "hard_failure"}
                return summary

            next_input = _input_from_output(result)
            next_fields = _fields(next_input)
            current_fields = _fields(current)

            if next_fields == current_fields and not result.edits_needed:
                logger.info(
                    "thai_polish: converged after %d pass(es) (zero diffs)",
                    pass_num,
                )
                self.last_stats = {"passes": pass_num, "outcome": "converged"}
                return _reconstruct(summary, result)
            if next_fields == current_fields:
                # Text is stable even though the model still claims edits are
                # needed — further passes would just repeat it.
                logger.info(
                    "thai_polish: converged after %d pass(es) (stable text)", pass_num
                )
                self.last_stats = {"passes": pass_num, "outcome": "converged"}
                return _reconstruct(summary, result)
            if not result.edits_needed:
                if _blanks_text(current_fields, next_fields):
                    logger.warning(
                        "thai_polish: pass %d emptied polished text — "
                        "returning original",
                        pass_num,
                    )
                    self.last_stats = {"passes": pass_num, "outcome": "blanked"}
                    return summary
                logger.info(
                    "thai_polish: accepted final edit after %d pass(es) "
                    "(edits_needed=false)",
                    pass_num,
                )
                self.last_stats = {"passes": pass_num, "outcome": "final_edit"}
                return _reconstruct(summary, result)

            # Feed the edited version back for the next pass
            current = next_input

        logger.warning(
            "thai_polish: hit safety cap (%d passes) — returning original",
            self.max_passes,
        )
        self.last_stats = {"passes": self.max_passes, "outcome": "cap"}
        return summary

    # ------------------------------------------------------------------
    # Single pass
    # ------------------------------------------------------------------

    def _one_pass(self, data: _PolishInput, pass_num: int) -> _PolishOutput:
        """Send one audit+fix request and parse the response."""
        system_prompt = (
            "You are a Thai-writing editor. Your job is to audit the Thai prose "
            "in a meeting summary and fix any issues according to the skill "
            "guidelines below. Follow the audit+fix workflow exactly.\n\n"
            "Return a JSON object with these fields:\n"
            "- `overview`: polished overview text\n"
            "- `topics`: array of {title, detail}\n"
            "- `decisions`: array of {decision, rationale}\n"
            "- `action_items`: array of {action, owner, due}\n"
            "- `open_questions`: array of strings\n"
            "- `edits_needed`: boolean — true if you made changes that need "
            "another pass, false when the text is polished\n"
            "- `edit_notes`: array of short strings describing what you changed\n\n"
            "IMPORTANT: Always include ALL fields in your JSON output, even if "
            "you made no changes. Set `edits_needed` to false when you believe "
            "the text is now correct."
        )

        user_content = (
            f"=== SKILL GUIDELINES ===\n\n{self._skill_bundle}\n\n"
            f"=== MEETING SUMMARY TO POLISH (Pass {pass_num}) ===\n\n"
            f"{json.dumps(data.__dict__, ensure_ascii=False, indent=2)}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
            timeout=self.timeout_seconds,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from polish model")

        return _parse_polished(content, pass_num, expected=data)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def skill_bundle_size(self) -> int:
        """Return the size of the skill bundle in bytes."""
        return len(self._skill_bundle.encode("utf-8"))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_polished(
    raw: str,
    pass_num: int,
    expected: "_PolishInput | None" = None,
) -> _PolishOutput:
    """Extract JSON from the model's response and parse it.

    Handles markdown code fences and strips leading/trailing whitespace.
    Raises ``ValueError`` on malformed JSON or shape violations (see
    :func:`_to_output`) — callers fall back to the original summary.
    """
    text = raw.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"thai_polish pass {pass_num}: malformed JSON from polish "
            f"model ({exc.msg} at line {exc.lineno} column {exc.colno})"
        ) from exc

    return _to_output(data, pass_num=pass_num, expected=expected)
