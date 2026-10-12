"""Conversation session: in-memory history backed by a :class:`MemoryStore`.

One session = one SQLite ``sessions`` row plus its ``messages``. The
in-memory list mirrors what is persisted so the agent loop can trim the
token budget without re-reading the database.
"""

from __future__ import annotations

from datetime import datetime

from ..llm.templates import estimate_tokens as _default_estimate
from ..memory.store import MemoryStore


class Session:
    """A single chat session tied to a persistent memory store."""

    def __init__(self, config, store: MemoryStore) -> None:
        self.config = config
        self.store = store
        self._session_id: int | None = None
        self._messages: list[dict] = []

    # ---------------------------------------------------------------- id

    @property
    def session_id(self) -> int | None:
        """The id of the current session row, or ``None`` before :meth:`start`."""
        return self._session_id

    # ----------------------------------------------------------- lifecycle

    def start(self) -> int:
        """Create a new session (``session <date>``) and reset in-memory history."""
        title = f"session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self._session_id = self.store.create_session(title)
        self._messages = []
        return self._session_id

    def clear(self) -> None:
        """Start a fresh session (the old one is kept in the store)."""
        self.start()

    # ------------------------------------------------------------ messages

    def add_message(self, role: str, content: str) -> None:
        """Persist a message and append it to the in-memory history.

        Lazily starts a session when none exists yet.
        """
        if self._session_id is None:
            self.start()
        content = str(content)
        self.store.add_message(self._session_id, role, content)
        self._messages.append({"role": role, "content": content})

    def history(self, limit: int = 40) -> list[dict]:
        """The last ``limit`` messages, chronological, as ``{role, content}``."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._messages[-limit:]
        ]

    def trim_to_tokens(
        self, max_tokens: int, estimate=_default_estimate
    ) -> list[dict]:
        """Drop the oldest in-memory messages while over ``max_tokens``.

        The newest message is always kept. Returns the trimmed history.
        """
        while len(self._messages) > 1:
            total = sum(
                estimate(str(m.get("content") or "")) for m in self._messages
            )
            if total <= max_tokens:
                break
            self._messages.pop(0)
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._messages
        ]
