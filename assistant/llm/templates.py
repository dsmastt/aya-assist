"""Small message/template helpers for the LLM chat layer."""

from __future__ import annotations

import json
from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token, minimum 1."""
    return max(1, len(text) // 4)


def system_message(content: str) -> dict[str, str]:
    """Build a chat message with role 'system'."""
    return {"role": "system", "content": content}


def user_message(content: str) -> dict[str, str]:
    """Build a chat message with role 'user'."""
    return {"role": "user", "content": content}


def assistant_message(
    content: str, tool_calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build a chat message with role 'assistant' (and optional tool_calls)."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def tool_result_message(tool_call_id: str, content: str) -> dict[str, str]:
    """Build a chat message with role 'tool' carrying a tool result."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def format_tool_result(name: str, result_dict: dict[str, Any]) -> str:
    """Compact JSON rendering of a tool result for feeding back to the model.

    Truncated to 4000 characters with a ``[truncated]`` marker.
    """
    text = json.dumps(result_dict, ensure_ascii=False, default=str)
    out = f"{name}: {text}"
    if len(out) > 4000:
        out = out[:3989] + "\n[truncated]"
    return out
