"""Tool registry: schema generation, approval gating, and safe invocation.

This module deliberately imports only the standard library so it can be used
without pulling in any other ``assistant.*`` modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class ToolResult:
    """Outcome of a single tool invocation."""

    ok: bool = True
    data: Any = None
    error: str | None = None
    note: str | None = None


class NeedsApproval(Exception):
    """Raised when a tool call requires interactive user approval.

    Attributes:
        label: short, human-readable name of the action.
        detail: extra context (usually a JSON rendering of the arguments).
    """

    def __init__(self, label: str, detail: str) -> None:
        self.label = label
        self.detail = detail
        super().__init__(f"approval required for {label}: {detail}")


@dataclass
class ToolContext:
    """Per-invocation context handed to tool functions."""

    config: Any = None
    console: Any = None
    session: Any = None
    runner: Any = None
    memory: Any = None

    def approve(self, label: str, detail: str) -> bool:
        """Ask the user to approve an action.

        Delegates to ``console.ask_yes_no`` when a console with that method is
        available; otherwise raises :class:`NeedsApproval`.
        """
        if self.console is not None and hasattr(self.console, "ask_yes_no"):
            return bool(self.console.ask_yes_no(label, detail))
        raise NeedsApproval(label, detail)


@dataclass
class ToolEntry:
    """A registered tool: name, async function, and schema metadata."""

    name: str
    func: Callable[..., Awaitable[Any]]
    description: str = ""
    category: str = "info"
    params: dict[str, Any] = field(default_factory=dict)

    async def run(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        """Invoke the tool, converting exceptions into failed ToolResults."""
        try:
            result = await self.func(ctx, arguments)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(ok=True, data=result)
        except NeedsApproval:
            raise
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=str(exc))


class ToolRegistry:
    """Name -> ToolEntry registry with OpenAI-style schema export."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Awaitable[Any]] | None = None,
        *,
        description: str = "",
        category: str = "info",
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Register a tool; usable as a decorator or a direct call."""

        def _decorate(f: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            self._tools[name] = ToolEntry(
                name=name,
                func=f,
                description=description,
                category=category,
                params=dict(params) if params else {},
            )
            return f

        if func is None:
            return _decorate
        return _decorate(func)

    def get(self, name: str) -> ToolEntry | None:
        """Return the entry for ``name``, or None."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """All registered tool names, in registration order."""
        return list(self._tools)

    def entries(self) -> list[ToolEntry]:
        """All registered entries, in registration order."""
        return list(self._tools.values())

    def openai_schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas for every registered tool.

        ``params`` values are JSON-schema-style property dicts. A ``required``
        list may be given inside ``params``; otherwise properties flagged
        ``"required": true`` in their own schema are collected.
        """
        schemas: list[dict[str, Any]] = []
        for entry in self._tools.values():
            props = dict(entry.params)
            explicit = props.pop("required", None)
            properties = {k: v for k, v in props.items() if k != "required"}
            if explicit is not None:
                required = list(explicit)
            else:
                required = [
                    k
                    for k, v in properties.items()
                    if isinstance(v, dict) and v.get("required")
                ]
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": properties,
            }
            if required:
                parameters["required"] = required
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": entry.name,
                        "description": entry.description,
                        "parameters": parameters,
                    },
                }
            )
        return schemas

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        ctx: ToolContext,
        force: bool = False,
    ) -> ToolResult:
        """Invoke a tool, gated on the configured auto-approve categories.

        Unless ``force`` is true, tools whose category is not in
        ``ctx.config.safety.auto_approve_categories`` require interactive
        approval via ``ctx.approve``. A raised :class:`NeedsApproval`
        propagates to the caller; ``False`` becomes a denied ToolResult.
        """
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(ok=False, error=f"unknown tool: {name!r}")

        if not force:
            safety = getattr(ctx.config, "safety", None) if ctx is not None else None
            categories = getattr(safety, "auto_approve_categories", None)
            if not categories or entry.category not in categories:
                detail = json.dumps(arguments or {}, default=str)[:200]
                approved = ctx.approve(entry.name, detail)
                if approved is False:
                    return ToolResult(ok=False, error="denied by user")

        return await entry.run(ctx, arguments or {})


#: Module-level singleton registry used across the project.
TOOLS = ToolRegistry()
