"""Robust parsing of the gateway's structured summary (pure stdlib).

The qwen gateway model has unreliable structured output, so this parser never
raises: it tries JSON, falls back to markdown sections, then to a raw
catch-all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

__all__ = ["ActionItem", "Summary", "parse_summary"]

_TOPIC_KEYS = ("topics", "หัวข้อ")
_DECISION_KEYS = ("decisions", "การตัดสินใจ")
_ACTION_ITEM_KEYS = ("action_items", "รายการที่ต้องทำ", "สิ่งที่ต้องทำ")

_TOPIC_HEADERS = ("หัวข้อ", "Topics")
_DECISION_HEADERS = ("การตัดสินใจ", "Decisions")
_ACTION_ITEM_HEADERS = ("สิ่งที่ต้องทำ", "Action Items", "รายการที่ต้องทำ")

_HEADER_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[*_]+)?"
    r"(หัวข้อ|Topics|การตัดสินใจ|Decisions|สิ่งที่ต้องทำ|Action Items|รายการที่ต้องทำ)"
    r"(?:[*_]+)?\s*[:：]?\s*$"
)


@dataclass
class ActionItem:
    action: str
    owner: str | None = None

    @classmethod
    def parse(cls, obj) -> "ActionItem":
        if isinstance(obj, dict):
            action = obj.get("action") or obj.get("สิ่งที่ต้องทำ")
            owner = obj.get("owner") or obj.get("ผู้รับผิดชอบ")
            return cls(
                action=str(action).strip() if action else "",
                owner=str(owner).strip() if owner else None,
            )
        return cls(action=str(obj).strip())


@dataclass
class Summary:
    topics: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    raw: str


def parse_summary(text: str) -> Summary:
    """Parse the model's summary into a :class:`Summary`. Never raises."""
    text = (text or "").strip()
    if not text:
        return Summary(topics=[], decisions=[], action_items=[], raw="")
    try:
        summary = _try_json(text)
        if summary is not None:
            return summary
        summary = _try_markdown(text)
        if summary is not None:
            return summary
    except Exception:  # noqa: BLE001 - the last resort below must always run
        pass
    return Summary(topics=[text], decisions=[], action_items=[], raw=text)


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
        topics=_as_str_list(_first_key(obj, _TOPIC_KEYS)),
        decisions=_as_str_list(_first_key(obj, _DECISION_KEYS)),
        action_items=_parse_action_items(_first_key(obj, _ACTION_ITEM_KEYS)),
        raw=text,
    )


def _first_key(obj: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in obj:
            return obj[key]
    return None


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
                items.append(ActionItem(action=match.group(2).strip(), owner=match.group(1).strip()))
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
        )
    return bool(str(obj).strip())


# -- Markdown fallback ------------------------------------------------------


def _try_markdown(text: str) -> Summary | None:
    topics: list[str] = []
    decisions: list[str] = []
    action_items: list[ActionItem] = []
    current: str | None = None

    for line in text.splitlines():
        header = _HEADER_PATTERN.match(line)
        if header:
            label = header.group(1)
            if label in _TOPIC_HEADERS:
                current = "topics"
            elif label in _DECISION_HEADERS:
                current = "decisions"
            else:
                current = "action_items"
            continue
        if current is None:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        bullet = re.match(r"^[-*•]\s+(.*)$", stripped)
        if not bullet:
            continue
        content = bullet.group(1).strip()
        if current == "action_items":
            match = re.match(r"^(.*?)[:：]\s*(.+)$", content)
            if match:
                action_items.append(
                    ActionItem(action=match.group(2).strip(), owner=match.group(1).strip())
                )
            else:
                action_items.append(ActionItem(action=content))
        elif current == "topics":
            topics.append(content)
        else:
            decisions.append(content)

    if topics or decisions or action_items:
        return Summary(topics=topics, decisions=decisions, action_items=action_items, raw=text)
    return None
