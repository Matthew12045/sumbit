"""Robust parsing of the gateway's structured summary (pure stdlib).

The qwen gateway model has unreliable structured output, so this parser never
raises: it tries JSON, falls back to markdown sections, then to a raw
catch-all.

Schema (2026-08-10, richer output): the summary carries context, not just
labels -- an ``overview`` paragraph, topics with ``title`` + ``detail``,
decisions with ``decision`` + ``rationale``, action items with optional
``owner``/``due``, and ``open_questions``.  Backward compatibility is kept:
plain-string topics/decisions (the old schema) still parse into
TopicItem/DecisionItem with empty detail/rationale.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

__all__ = ["ActionItem", "TopicItem", "DecisionItem", "Summary", "parse_summary"]

_OVERVIEW_KEYS = ("overview", "ภาพรวม", "สรุปภาพรวม")
_TOPIC_KEYS = ("topics", "หัวข้อ")
_DECISION_KEYS = ("decisions", "การตัดสินใจ")
_ACTION_ITEM_KEYS = ("action_items", "รายการที่ต้องทำ", "สิ่งที่ต้องทำ")
_OPEN_QUESTION_KEYS = ("open_questions", "คำถามที่ยังไม่ได้ข้อสรุป", "ประเด็นค้าง")

_OVERVIEW_HEADERS = ("ภาพรวม", "Overview")
_TOPIC_HEADERS = ("หัวข้อ", "Topics")
_DECISION_HEADERS = ("การตัดสินใจ", "Decisions")
_ACTION_ITEM_HEADERS = ("สิ่งที่ต้องทำ", "Action Items", "รายการที่ต้องทำ")
_OPEN_QUESTION_HEADERS = ("คำถามที่ยังไม่ได้ข้อสรุป", "Open Questions")

_HEADER_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[*_]+)?"
    r"(ภาพรวม|Overview|หัวข้อ|Topics|การตัดสินใจ|Decisions|สิ่งที่ต้องทำ|"
    r"Action Items|รายการที่ต้องทำ|คำถามที่ยังไม่ได้ข้อสรุป|Open Questions)"
    r"(?:[*_]+)?\s*[:：]?\s*$"
)

# Pulls a label apart from its detail/rationale inside a bullet.  Matches a
# colon (existing behaviour) plus the em/en-dash the poster itself renders as
# "{title} — {detail}" / "{decision} — {rationale}".
_LABEL_SPLIT = re.compile(r"^(.*?)\s*(?:[:：]|—|–)\s*(.+)$")


@dataclass
class ActionItem:
    action: str
    owner: str | None = None
    due: str | None = None

    @classmethod
    def parse(cls, obj) -> "ActionItem":
        if isinstance(obj, dict):
            action = obj.get("action") or obj.get("สิ่งที่ต้องทำ")
            owner = obj.get("owner") or obj.get("ผู้รับผิดชอบ")
            due = obj.get("due") or obj.get("กำหนดเวลา") or obj.get("ครบกำหนด")
            return cls(
                action=str(action).strip() if action else "",
                owner=str(owner).strip() if owner else None,
                due=str(due).strip() if due else None,
            )
        return cls(action=str(obj).strip())


@dataclass
class TopicItem:
    title: str
    detail: str = ""


@dataclass
class DecisionItem:
    decision: str
    rationale: str = ""


@dataclass
class Summary:
    overview: str
    topics: list[TopicItem]
    decisions: list[DecisionItem]
    action_items: list[ActionItem]
    open_questions: list[str]
    raw: str


def parse_summary(text: str) -> Summary:
    """Parse the model's summary into a :class:`Summary`. Never raises."""
    text = (text or "").strip()
    if not text:
        return Summary(
            overview="", topics=[], decisions=[], action_items=[], open_questions=[], raw=""
        )
    try:
        summary = _try_json(text)
        if summary is not None:
            return summary
        summary = _try_markdown(text)
        if summary is not None:
            return summary
    except Exception:  # noqa: BLE001 - the last resort below must always run
        pass
    # Unstructured text is semantically an overview, not a topic list.
    return Summary(
        overview=text, topics=[], decisions=[], action_items=[], open_questions=[], raw=text
    )


# -- JSON attempt ---------------------------------------------------------


def _try_json(text: str) -> Summary | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return Summary(
        overview=_as_overview_str(_first_key(obj, _OVERVIEW_KEYS)),
        topics=_parse_topics(_first_key(obj, _TOPIC_KEYS)),
        decisions=_parse_decisions(_first_key(obj, _DECISION_KEYS)),
        action_items=_parse_action_items(_first_key(obj, _ACTION_ITEM_KEYS)),
        open_questions=_as_str_list(_first_key(obj, _OPEN_QUESTION_KEYS)),
        raw=text,
    )


def _first_key(obj: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def _as_overview_str(value) -> str:
    """Coerce the overview to one string: str, a list of sentences, or None."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(p).strip() for p in value if str(p).strip()]
        return " ".join(parts)
    return str(value).strip()


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n;]", value)
        return [re.sub(r"^[-*•]\s*", "", p).strip() for p in parts if p.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "name", "title"):
                    if item.get(key):
                        out.append(str(item[key]).strip())
                        break
            elif item is not None:
                out.append(str(item).strip())
        return [x for x in out if x]
    if isinstance(value, (int, float)):
        return [str(value)]
    return []


def _parse_topics(value) -> list[TopicItem]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[TopicItem] = []
        for item in value:
            if isinstance(item, dict):
                title = item.get("title") or item.get("หัวข้อ")
                detail = item.get("detail") or item.get("รายละเอียด")
                out.append(
                    TopicItem(
                        title=str(title).strip() if title else "",
                        detail=str(detail).strip() if detail else "",
                    )
                )
            elif isinstance(item, str):
                # Backward-compat: old schema sent bare strings.
                out.append(TopicItem(title=item.strip()))
            elif item is not None:
                out.append(TopicItem(title=str(item).strip()))
        return [t for t in out if t.title]
    if isinstance(value, str):
        out = []
        for part in re.split(r"[\n;]", value):
            part = re.sub(r"^[-*•]\s*", "", part).strip()
            if not part:
                continue
            out.append(TopicItem(title=part))
        return out
    return []


def _parse_decisions(value) -> list[DecisionItem]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[DecisionItem] = []
        for item in value:
            if isinstance(item, dict):
                decision = item.get("decision") or item.get("การตัดสินใจ")
                rationale = item.get("rationale") or item.get("เหตุผล")
                out.append(
                    DecisionItem(
                        decision=str(decision).strip() if decision else "",
                        rationale=str(rationale).strip() if rationale else "",
                    )
                )
            elif isinstance(item, str):
                # Backward-compat: old schema sent bare strings.
                out.append(DecisionItem(decision=item.strip()))
            elif item is not None:
                out.append(DecisionItem(decision=str(item).strip()))
        return [d for d in out if d.decision]
    if isinstance(value, str):
        out = []
        for part in re.split(r"[\n;]", value):
            part = re.sub(r"^[-*•]\s*", "", part).strip()
            if not part:
                continue
            out.append(DecisionItem(decision=part))
        return out
    return []


def _parse_action_items(value) -> list[ActionItem]:
    if value is None:
        return []
    if isinstance(value, list):
        return [ActionItem.parse(item) for item in value if _nonempty(item)]
    if isinstance(value, str):
        items: list[ActionItem] = []
        for line in re.split(r"\n", value):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*•]\s*", "", line)
            match = re.match(r"^(.*?)[:：]\s*(.+)$", line)
            if match:
                items.append(
                    ActionItem(action=match.group(2).strip(), owner=match.group(1).strip())
                )
            else:
                items.append(ActionItem(action=line))
        return items
    return []


def _nonempty(obj) -> bool:
    if isinstance(obj, dict):
        return bool(
            obj.get("action")
            or obj.get("สิ่งที่ต้องทำ")
            or obj.get("owner")
            or obj.get("ผู้รับผิดชอบ")
            or obj.get("due")
            or obj.get("กำหนดเวลา")
            or obj.get("ครบกำหนด")
        )
    return bool(str(obj).strip())


# -- Markdown fallback ------------------------------------------------------


def _strip_bold(s: str) -> str:
    """Drop ``**...**`` markers a model may put around a bold title."""
    return re.sub(r"^\*\*(.+?)\*\*$", r"\1", s.strip()).strip()


def _parse_topic_bullet(content: str) -> TopicItem:
    match = _LABEL_SPLIT.match(content)
    if match:
        return TopicItem(
            title=_strip_bold(match.group(1)),
            detail=match.group(2).strip(),
        )
    return TopicItem(title=_strip_bold(content))


def _parse_decision_bullet(content: str) -> DecisionItem:
    match = _LABEL_SPLIT.match(content)
    if match:
        return DecisionItem(
            decision=_strip_bold(match.group(1)),
            rationale=match.group(2).strip(),
        )
    return DecisionItem(decision=_strip_bold(content))


def _try_markdown(text: str) -> Summary | None:
    overview_lines: list[str] = []
    topics: list[TopicItem] = []
    decisions: list[DecisionItem] = []
    action_items: list[ActionItem] = []
    open_questions: list[str] = []
    current: str | None = None

    for line in text.splitlines():
        header = _HEADER_PATTERN.match(line)
        if header:
            label = header.group(1)
            if label in _OVERVIEW_HEADERS:
                current = "overview"
            elif label in _TOPIC_HEADERS:
                current = "topics"
            elif label in _DECISION_HEADERS:
                current = "decisions"
            elif label in _OPEN_QUESTION_HEADERS:
                current = "open_questions"
            else:
                current = "action_items"
            continue
        if current is None:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if current == "overview":
            # Overview is a paragraph, not bullets: collect verbatim lines.
            if not re.match(r"^[-*•]\s+", stripped):
                overview_lines.append(stripped)
            continue
        bullet = re.match(r"^[-*•]\s+(.*)$", stripped)
        if not bullet:
            continue
        content = bullet.group(1).strip()
        if current == "topics":
            topics.append(_parse_topic_bullet(content))
        elif current == "decisions":
            decisions.append(_parse_decision_bullet(content))
        elif current == "action_items":
            match = re.match(r"^(.*?)[:：]\s*(.+)$", content)
            if match:
                action_items.append(
                    ActionItem(action=match.group(2).strip(), owner=match.group(1).strip())
                )
            else:
                action_items.append(ActionItem(action=content))
        else:  # open_questions
            open_questions.append(content)

    overview = " ".join(overview_lines).strip()
    if overview or topics or decisions or action_items or open_questions:
        return Summary(
            overview=overview,
            topics=topics,
            decisions=decisions,
            action_items=action_items,
            open_questions=open_questions,
            raw=text,
        )
    return None
