"""Memory tools: add, recall, list, and (approved) delete of agent facts.

Stdlib only. ``ctx.memory`` is duck-typed; approval failures raise the
project's :class:`NeedsApproval` (imported lazily) so the agent loop can
handle the approval handshake.
"""

from __future__ import annotations

from typing import Any

_RECALL_KEYS = ("id", "text", "kind", "score", "updated_at", "last_accessed")


async def memory_add(
    ctx: Any, fact: str, kind: str = "fact", confidence: float = 1.0
) -> dict:
    try:
        fid = ctx.memory.add_fact(
            str(fact), kind=kind, source="agent", confidence=float(confidence)
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return {"id": int(fid), "stored": True}


async def memory_recall(ctx: Any, query: str, k: int = 5) -> dict:
    try:
        results = ctx.memory.search_facts(str(query), k)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    matches = [{key: result.get(key) for key in _RECALL_KEYS} for result in results]
    return {"matches": matches}


async def memory_list(ctx: Any, limit: int = 20, kind: str | None = None) -> dict:
    try:
        facts = ctx.memory.list_facts(limit)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    if kind is not None:
        facts = [fact for fact in facts if fact.get("kind") == kind]
    return {"facts": facts}


async def memory_delete(
    ctx: Any, fact_id: int | None = None, query: str | None = None
) -> dict:
    if fact_id is None and not query:
        return {"error": "give an id or a text query"}
    try:
        from assistant.tools.registry import NeedsApproval as _NeedsApproval
    except ImportError:  # pragma: no cover - fallback without the registry
        class _NeedsApproval(Exception):  # noqa: N801
            def __init__(self, label: str, detail: str) -> None:
                self.label = label
                self.detail = detail
                super().__init__(f"approval required for {label}: {detail}")
    try:
        approved = ctx.approve("memory delete", f"delete fact {fact_id or query}")
    except _NeedsApproval:
        raise
    if approved is False:
        return {"error": "denied by user"}
    try:
        ids = ctx.memory.delete_fact(fact_id=fact_id, text_match=query)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return {"deleted": list(ids)}
