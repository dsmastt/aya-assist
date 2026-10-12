"""Persistent todo and reminder tools backed by a single JSON state file.

Stdlib only. All tool functions are async, take ``(ctx, **params)``, and
return JSON-serializable dicts; expected failures return ``{"error": ...}``
instead of raising.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_STATE_FILE = "~/.local/share/assistant/state.json"

_LOCK = threading.Lock()

_IN_RE = re.compile(
    r"^in\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?\s*$", re.IGNORECASE
)
_AT_RE = re.compile(r"^(?:at\s+)?(\d{1,2}):(\d{2})\s*$", re.IGNORECASE)
_TOMORROW_RE = re.compile(
    r"^tomorrow(?:\s+(?:(\d{1,2})\s*(am|pm)|(\d{1,2}):(\d{2})))?\s*$",
    re.IGNORECASE,
)


def parse_when(when: str, now: datetime | None = None) -> datetime:
    """Parse a human time expression into a datetime.

    Supports ``in 30m`` / ``in 2h`` / ``in 45s`` / ``in 1h30m`` (compound),
    ``at 18:30`` / ``18:30`` (today, or tomorrow when already past),
    ``tomorrow`` (09:00 default) and ``tomorrow 9am`` / ``tomorrow 09:00``.
    Raises :class:`ValueError` for anything unrecognised.
    """
    if now is None:
        now = datetime.now()
    text = str(when).strip()

    match = _IN_RE.match(text)
    if match and any(match.groups()):
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return now + timedelta(hours=hours, minutes=minutes, seconds=seconds)

    match = _AT_RE.match(text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23 or minute > 59:
            raise ValueError(f"invalid clock time: {when!r}")
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    match = _TOMORROW_RE.match(text)
    if match:
        base = now + timedelta(days=1)
        if match.group(3) is not None:
            hour, minute = int(match.group(3)), int(match.group(4))
            if hour > 23 or minute > 59:
                raise ValueError(f"invalid clock time: {when!r}")
            return base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if match.group(1) is not None:
            hour = int(match.group(1))
            ampm = (match.group(2) or "am").casefold()
            if hour > 12 or hour < 1:
                raise ValueError(f"invalid clock time: {when!r}")
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            return base.replace(hour=hour, minute=0, second=0, microsecond=0)
        return base.replace(hour=9, minute=0, second=0, microsecond=0)

    raise ValueError(f"cannot parse time expression: {when!r}")


_DURATION_HINT = (
    "accepted formats: '20m' / '20 min' / '20 minutes' / '20mins', '2h', "
    "'1h30m' (compound), '90s' / '45 sec' / '45 seconds', '1:30' or '20:00' "
    "(mm:ss), optionally prefixed with 'in '"
)

_DUR_PART_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(h|hrs?|hours?|m|mins?|minutes?|s|secs?|seconds?)(?=$|\s|\d)",
    re.IGNORECASE,
)


def _parse_duration(text: str) -> float:
    """Parse a duration expression into seconds. Raises :class:`ValueError`.

    Accepts ``20m`` / ``20 min`` / ``20 minutes`` / ``20mins``, ``2h``,
    compound forms like ``1h30m``, ``90s`` / ``45 sec`` / ``45 seconds``,
    ``mm:ss`` clock forms like ``1:30`` or ``20:00``, and an optional
    leading ``in ``. Bare numbers are rejected.
    """
    raw = str(text).strip()
    stripped = re.sub(r"^in\s+", "", raw, flags=re.IGNORECASE).strip()
    if not stripped:
        raise ValueError("empty duration; give a value like '20m' or '5 minutes'")
    if re.fullmatch(r"\d{1,2}:\d{2}", stripped):
        minutes, seconds = (int(part) for part in stripped.split(":"))
        return minutes * 60 + seconds
    if re.fullmatch(r"\d+(?:\.\d+)?", stripped):
        raise ValueError(
            f"bare number {stripped!r} needs a unit (e.g. '20m') or mm:ss form "
            "(e.g. '20:00')"
        )
    total = 0.0
    pos = 0
    for match in _DUR_PART_RE.finditer(stripped):
        if stripped[pos : match.start()].strip():
            raise ValueError(f"cannot parse duration: {text!r}")
        value = float(match.group(1))
        unit = match.group(2)[0].casefold()
        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        else:
            total += value
        pos = match.end()
    if not pos or stripped[pos:].strip():
        raise ValueError(f"cannot parse duration: {text!r}")
    return total


def _state_path(ctx: Any) -> Path:
    cfg = getattr(ctx, "config", None)
    raw = getattr(getattr(cfg, "paths", None), "state_file", None) or DEFAULT_STATE_FILE
    return Path(raw).expanduser()


def _load(path: Path) -> dict:
    with _LOCK:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("todos", [])
        data.setdefault("reminders", [])
        data["timers"] = data.get("timers", [])
        return data


def _save(path: Path, data: dict) -> None:
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)


def _next_id(items: list[dict]) -> int:
    return max((int(item.get("id", 0)) for item in items), default=0) + 1


async def todo_add(ctx: Any, title: str) -> dict:
    title = str(title or "").strip()
    if not title:
        return {"error": "todo title must be a non-empty string"}
    path = _state_path(ctx)
    data = _load(path)
    todo = {
        "id": _next_id(data["todos"]),
        "title": title,
        "done": False,
        "created_at": datetime.now().isoformat(),
    }
    data["todos"].append(todo)
    _save(path, data)
    return {"id": todo["id"], "title": todo["title"]}


async def todo_list(ctx: Any, include_done: bool = False) -> dict:
    data = _load(_state_path(ctx))
    todos = [t for t in data["todos"] if include_done or not t.get("done")]
    return {"todos": todos}


async def todo_toggle(
    ctx: Any,
    todo_id: int | None = None,
    query: str | None = None,
    done: bool | None = None,
) -> dict:
    path = _state_path(ctx)
    data = _load(path)
    target = None
    if todo_id is not None:
        for todo in data["todos"]:
            if todo.get("id") == todo_id:
                target = todo
                break
    elif query:
        needle = query.casefold()
        for todo in data["todos"]:
            if needle in todo.get("title", "").casefold():
                target = todo
                break
    if target is None:
        return {"error": "no matching todo"}
    target["done"] = (not target.get("done")) if done is None else bool(done)
    _save(path, data)
    return {"todo": target}


async def reminder_add(ctx: Any, message: str, when: str) -> dict:
    message = str(message or "").strip()
    if not message:
        return {"error": "reminder message must be a non-empty string"}
    try:
        at = parse_when(str(when))
    except ValueError as exc:
        return {"error": str(exc)}
    path = _state_path(ctx)
    data = _load(path)
    reminder = {
        "id": _next_id(data["reminders"]),
        "message": message,
        "at": at.isoformat(),
        "created_at": datetime.now().isoformat(),
    }
    data["reminders"].append(reminder)
    _save(path, data)
    return {
        "id": reminder["id"],
        "message": message,
        "at": reminder["at"],
        "parsed_note": at.strftime("%Y-%m-%d %H:%M"),
    }


async def reminder_list(ctx: Any) -> dict:
    data = _load(_state_path(ctx))
    reminders = sorted(data["reminders"], key=lambda r: r.get("at", ""))
    return {"reminders": reminders}


async def reminder_cancel(ctx: Any, reminder_id: int) -> dict:
    path = _state_path(ctx)
    data = _load(path)
    for index, reminder in enumerate(data["reminders"]):
        if reminder.get("id") == reminder_id:
            del data["reminders"][index]
            _save(path, data)
            return {"cancelled": True}
    return {"error": f"no reminder with id {reminder_id}"}

async def timer_set(ctx: Any, label: str = "timer", duration: str = "20m") -> dict:
    try:
        seconds = _parse_duration(duration)
    except ValueError as exc:
        return {"error": f"invalid duration: {exc} — {_DURATION_HINT}"}
    label = str(label or "timer").strip() or "timer"
    now = datetime.now().astimezone()
    path = _state_path(ctx)
    data = _load(path)
    timer = {
        "id": _next_id(data["timers"]),
        "label": label,
        "end_at": (now + timedelta(seconds=seconds)).isoformat(timespec="seconds"),
        "created_at": now.isoformat(timespec="seconds"),
    }
    data["timers"].append(timer)
    _save(path, data)
    return {
        "id": timer["id"],
        "label": timer["label"],
        "duration_seconds": seconds,
        "end_at": timer["end_at"],
        "note": "timer set; I'll notify you when it ends",
    }


async def timer_status(ctx: Any, _now: datetime | None = None) -> dict:
    """List timers with remaining time. ``_now`` is a test hook only."""
    if _now is None:
        _now = datetime.now().astimezone()
    data = _load(_state_path(ctx))
    timers = []
    for timer in data.get("timers", []):
        try:
            end_at = datetime.fromisoformat(timer.get("end_at", ""))
        except (TypeError, ValueError):
            continue
        if end_at.tzinfo is None and _now.tzinfo is not None:
            end_at = end_at.replace(tzinfo=_now.tzinfo)
        elif end_at.tzinfo is not None and _now.tzinfo is None:
            end_at = end_at.replace(tzinfo=None)
        remaining = (end_at - _now).total_seconds()
        timers.append(
            {
                "id": timer.get("id"),
                "label": timer.get("label", "timer"),
                "end_at": timer.get("end_at"),
                "remaining_seconds": remaining,
                "status": "running" if remaining >= 0 else "done",
            }
        )
    timers.sort(key=lambda t: t["end_at"])
    next_timer = next((t for t in timers if t["status"] == "running"), None)
    return {
        "timers": timers,
        "next": next_timer,
        "note": "no timers" if not timers else None,
    }


def _running_timers(data: dict) -> list[dict]:
    """Timers whose ``end_at`` has not passed yet (matches timer_status)."""
    now = datetime.now().astimezone()
    running = []
    for timer in data.get("timers", []):
        try:
            end_at = datetime.fromisoformat(timer.get("end_at", ""))
        except (TypeError, ValueError):
            continue
        if end_at.tzinfo is None:
            end_at = end_at.replace(tzinfo=now.tzinfo)
        if end_at >= now:
            running.append(timer)
    return running


def _running_timer_summary(running: list[dict]) -> str:
    parts = [
        f"{timer.get('id')}: {str(timer.get('label', 'timer')).strip() or 'timer'}"
        for timer in running
    ]
    return "[" + ", ".join(parts) + "]"


async def timer_cancel(ctx: Any, timer_id: int | None = None) -> dict:
    path = _state_path(ctx)
    data = _load(path)
    running = _running_timers(data)
    if timer_id is None:
        if len(running) == 1:
            timer = running[0]
            data["timers"] = [
                t for t in data["timers"] if t.get("id") != timer.get("id")
            ]
            _save(path, data)
            return {"cancelled": timer.get("id")}
        return {
            "error": (
                "no timer id given; running timers: "
                f"{_running_timer_summary(running)}"
            )
        }
    for index, timer in enumerate(data["timers"]):
        if timer.get("id") == timer_id:
            del data["timers"][index]
            _save(path, data)
            return {"cancelled": timer_id}
    return {
        "error": (
            f"no timer with id {timer_id}; running timers: "
            f"{_running_timer_summary(running)}"
        )
    }


async def timer_drain(ctx: Any, _now: datetime | None = None) -> list[str]:
    """Pop all finished timers and return their labels.

    Used by the notification scheduler to announce ``Timer done: <label>``.
    ``_now`` is a test hook only.
    """
    if _now is None:
        _now = datetime.now().astimezone()
    path = _state_path(ctx)
    data = _load(path)
    due_labels = []
    remaining = []
    for timer in data["timers"]:
        try:
            end_at = datetime.fromisoformat(timer.get("end_at", ""))
        except (TypeError, ValueError):
            remaining.append(timer)
            continue
        if end_at.tzinfo is None and _now.tzinfo is not None:
            end_at = end_at.replace(tzinfo=_now.tzinfo)
        elif end_at.tzinfo is not None and _now.tzinfo is None:
            end_at = end_at.replace(tzinfo=None)
        if end_at <= _now:
            due_labels.append(str(timer.get("label", "timer")))
        else:
            remaining.append(timer)
    data["timers"] = remaining
    _save(path, data)
    return due_labels
