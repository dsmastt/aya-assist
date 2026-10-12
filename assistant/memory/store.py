"""Persistent memory storage for the local AI desktop agent.

Provides a thread-safe, SQLite-backed store for facts (with FTS5 full-text
search and a LIKE-based fallback when FTS5 is unavailable), chat sessions,
and messages. Uses only the Python standard library.

Search scoring: FTS5 rows are fetched ordered by bm25 (best first, capped
at ~50 candidates), then a combined score is computed in Python:

    final = -bm25 * (1.0 + 0.5 * exp(-age_days / 30))

which boosts fresher facts (updated_at within the last weeks). Candidates
are re-sorted by this score and the top k returned.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sqlite3
import threading
from collections.abc import Iterator

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    source TEXT NOT NULL DEFAULT 'agent',
    confidence REAL NOT NULL DEFAULT 1.0,
    meta TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    last_accessed TEXT,
    access_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    meta TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT DEFAULT '',
    started_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TRIGGER IF NOT EXISTS trg_messages_touch
AFTER INSERT ON messages
BEGIN
    UPDATE sessions SET updated_at = datetime('now') WHERE id = NEW.session_id;
END;
"""

_FTS_TRIGGERS = (
    "CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN "
    "INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text); END",
    "CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN "
    "INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text); END",
    "CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN "
    "INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text); "
    "INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text); END",
)


class MemoryStore:
    """Thread-safe SQLite-backed memory store.

    All database access is guarded by a ``threading.RLock``, so a single
    instance may be shared across threads. The connection is opened with
    ``check_same_thread=False``; the lock, not the connection, provides the
    thread-safety guarantee.

    Schema versioning is tracked with ``PRAGMA user_version``. Version 1
    creates the ``facts``, ``messages`` and ``sessions`` tables plus an
    FTS5 index (``facts_fts``) kept in sync via triggers. If FTS5 is not
    compiled into the SQLite library, ``fts_available`` is set to ``False``
    and searches fall back to SQL ``LIKE``.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        """Open (creating if needed) the store at ``db_path``.

        Creates parent directories, opens the SQLite connection, and runs
        the schema migration.
        """
        self.db_path = os.fspath(db_path)
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self.fts_available = False
        self._migrate()
        self._setup_fts()

    # ------------------------------------------------------------------ db

    def _migrate(self) -> None:
        """Apply schema migrations based on ``PRAGMA user_version``."""
        with self._lock:
            conn = self._conn
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version >= 1:
                return
            conn.executescript(_SCHEMA_V1)
            conn.commit()
            conn.execute("PRAGMA user_version = 1")
            conn.commit()

    def _setup_fts(self) -> None:
        """Create the FTS5 index and sync triggers, if FTS5 is available.

        Falls back to ``fts_available = False`` (LIKE-based search) when the
        SQLite build lacks FTS5. Rebuilding on every open keeps the index
        consistent even if it was created after facts already existed.
        """
        try:
            with self._lock, self._tx():
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts "
                    "USING fts5(text, content='facts', content_rowid='id')"
                )
                for stmt in _FTS_TRIGGERS:
                    self._conn.execute(stmt)
                self._conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
            self.fts_available = True
        except sqlite3.Error:
            self.fts_available = False

    @contextlib.contextmanager
    def _tx(self) -> Iterator[None]:
        """Commit on success, roll back on any exception."""
        try:
            yield
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    # ---------------------------------------------------------------- facts

    def add_fact(
        self,
        text: str,
        kind: str = "fact",
        source: str = "agent",
        confidence: float = 1.0,
        meta: dict | None = None,
    ) -> int:
        """Insert a fact and return its id.

        Raises ``ValueError`` if ``text`` is empty. ``meta`` is stored as a
        JSON string.
        """
        text = str(text).strip()
        if not text:
            raise ValueError("fact text must be a non-empty string")
        meta_json = json.dumps(meta) if meta is not None else None
        with self._lock, self._tx():
            cur = self._conn.execute(
                "INSERT INTO facts (text, kind, source, confidence, meta) "
                "VALUES (?, ?, ?, ?, ?)",
                (text, kind, source, float(confidence), meta_json),
            )
            return int(cur.lastrowid)

    def search_facts(
        self,
        query: str,
        k: int = 5,
        kinds: list[str] | None = None,
    ) -> list[dict]:
        """Full-text search for facts, best matches first.

        The query is tokenized on whitespace and each token sanitized of
        FTS5-special characters, producing a prefix-phrase match string like
        ``"tok1"* OR "tok2"*``. The top ~50 bm25 candidates are combined
        with a recency boost in Python and the top ``k`` returned. An empty
        query returns the most recently updated facts. Results touch
        ``last_accessed``/``access_count`` of every returned fact.
        """
        if k <= 0:
            return []
        tokens = self._tokenize(query)
        with self._lock:
            if not tokens:
                kind_sql, kind_params = self._kind_clause(kinds, "WHERE")
                candidates = self._recent_facts(k, kind_sql, kind_params)
            elif self.fts_available:
                kind_sql, kind_params = self._kind_clause(kinds, "AND")
                candidates = self._search_fts(tokens, kind_sql, kind_params)
            else:
                kind_sql, kind_params = self._kind_clause(kinds, "AND")
                candidates = self._search_like(tokens, kind_sql, kind_params)
            candidates.sort(key=lambda r: r["score"], reverse=True)
            top = candidates[:k]
            if not top:
                return []
            return self._touch_and_return(top)

    def get_fact(self, fact_id: int) -> dict | None:
        """Return a single fact by id, or ``None`` if it does not exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE id = ?", (fact_id,)
            ).fetchone()
            return self._fact_to_dict(row) if row is not None else None

    def update_fact(
        self,
        fact_id: int,
        text: str | None = None,
        confidence: float | None = None,
        kind: str | None = None,
    ) -> bool:
        """Update fields of a fact; returns whether the fact existed.

        ``updated_at`` is refreshed whenever a field actually changes.
        """
        sets: list[str] = []
        params: list[object] = []
        if text is not None:
            text = str(text).strip()
            if not text:
                raise ValueError("fact text must be a non-empty string")
            sets.append("text = ?")
            params.append(text)
        if confidence is not None:
            sets.append("confidence = ?")
            params.append(float(confidence))
        if kind is not None:
            sets.append("kind = ?")
            params.append(kind)
        with self._lock:
            if not sets:
                exists = self._conn.execute(
                    "SELECT 1 FROM facts WHERE id = ?", (fact_id,)
                ).fetchone()
                return exists is not None
            sets.append("updated_at = datetime('now')")
            with self._tx():
                cur = self._conn.execute(
                    f"UPDATE facts SET {', '.join(sets)} WHERE id = ?",
                    (*params, fact_id),
                )
                return cur.rowcount > 0

    def delete_fact(
        self,
        fact_id: int | None = None,
        text_match: str | None = None,
    ) -> list[int]:
        """Delete facts and return the ids that were removed.

        Deletes by id when ``fact_id`` is given, otherwise by
        ``text LIKE '%text_match%'`` when ``text_match`` is given. ``%``,
        ``_`` and ``\\`` inside ``text_match`` are escaped. FTS sync is
        handled by the delete trigger.
        """
        with self._lock, self._tx():
            conn = self._conn
            if fact_id is not None:
                ids = [
                    r["id"]
                    for r in conn.execute(
                        "SELECT id FROM facts WHERE id = ?", (fact_id,)
                    ).fetchall()
                ]
                if ids:
                    conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            elif text_match is not None and text_match.strip():
                pattern = f"%{self._escape_like(text_match)}%"
                ids = [
                    r["id"]
                    for r in conn.execute(
                        "SELECT id FROM facts WHERE text LIKE ? ESCAPE '\\'",
                        (pattern,),
                    ).fetchall()
                ]
                if ids:
                    conn.execute(
                        "DELETE FROM facts WHERE text LIKE ? ESCAPE '\\'",
                        (pattern,),
                    )
            else:
                ids = []
            return ids

    def list_facts(self, limit: int = 50, kind: str | None = None) -> list[dict]:
        """List facts, newest first, optionally filtered by ``kind``."""
        with self._lock:
            if kind is None:
                rows = self._conn.execute(
                    "SELECT * FROM facts ORDER BY created_at DESC, id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM facts WHERE kind = ? "
                    "ORDER BY created_at DESC, id DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            return [self._fact_to_dict(r) for r in rows]

    # ------------------------------------------------------------- sessions

    def create_session(self, title: str = "") -> int:
        """Create a session and return its id."""
        with self._lock, self._tx():
            cur = self._conn.execute(
                "INSERT INTO sessions (title) VALUES (?)", (title,)
            )
            return int(cur.lastrowid)

    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        meta: dict | None = None,
    ) -> int:
        """Append a message to a session and return its id.

        Raises ``ValueError`` if ``session_id`` does not exist. The
        session's ``updated_at`` is refreshed by trigger.
        """
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise ValueError(f"unknown session_id: {session_id}")
            meta_json = json.dumps(meta) if meta is not None else None
            with self._tx():
                cur = self._conn.execute(
                    "INSERT INTO messages (session_id, role, content, meta) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, role, content, meta_json),
                )
                return int(cur.lastrowid)

    def recent_messages(self, session_id: int, n: int = 50) -> list[dict]:
        """Return the last ``n`` messages of a session in chronological order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, n),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "role": r["role"],
                    "content": r["content"],
                    "created_at": r["created_at"],
                }
                for r in reversed(rows)
            ]

    def list_sessions(self, limit: int = 10) -> list[dict]:
        """List sessions, most recently updated first, with message counts."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.id, s.title, s.started_at, s.updated_at, "
                "(SELECT COUNT(*) FROM messages m "
                " WHERE m.session_id = s.id) AS message_count "
                "FROM sessions s ORDER BY s.updated_at DESC, s.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def set_session_title(self, session_id: int, title: str) -> None:
        """Set the title of a session."""
        with self._lock, self._tx():
            self._conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
            )

    # --------------------------------------------------------------- misc

    def stats(self) -> dict:
        """Return store statistics (fact count by kind, messages, sessions)."""
        with self._lock:
            fact_rows = self._conn.execute(
                "SELECT kind, COUNT(*) AS n FROM facts GROUP BY kind ORDER BY kind"
            ).fetchall()
            message_count = self._conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            session_count = self._conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]
        return {
            "facts": {r["kind"]: r["n"] for r in fact_rows},
            "messages": message_count,
            "sessions": session_count,
            "fts_available": self.fts_available,
        }

    def close(self) -> None:
        """Commit and close the underlying connection. Safe to call twice."""
        with self._lock:
            conn = self._conn
            if conn is not None:
                self._conn = None
                try:
                    conn.commit()
                finally:
                    conn.close()

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        """Split on whitespace, stripping FTS5-special characters."""
        if not query or not query.strip():
            return []
        tokens = []
        for tok in query.split():
            clean = "".join(ch for ch in tok if ch not in '()"*')
            if clean:
                tokens.append(clean)
        return tokens

    @staticmethod
    def _kind_clause(
        kinds: list[str] | None, prefix: str
    ) -> tuple[str, tuple[str, ...]]:
        """Build ``<prefix> f.kind IN (...)` or an empty clause."""
        if not kinds:
            return "", ()
        kinds = list(kinds)
        placeholders = ", ".join("?" for _ in kinds)
        return f"{prefix} f.kind IN ({placeholders})", tuple(kinds)

    @staticmethod
    def _escape_like(text: str) -> str:
        """Escape LIKE wildcards for use with ``ESCAPE '\\'``."""
        return (
            str(text)
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    @staticmethod
    def _parse_meta(meta: str | None) -> dict | None:
        if not meta:
            return None
        try:
            value = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _recency_factor(age_days: float) -> float:
        """1.0 + 0.5 * exp(-age_days / 30): 1.5 for fresh, ~1.0 for old."""
        age = max(float(age_days or 0.0), 0.0)
        return 1.0 + 0.5 * math.exp(-age / 30.0)

    def _fact_to_dict(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "text": row["text"],
            "kind": row["kind"],
            "source": row["source"],
            "confidence": row["confidence"],
            "meta": self._parse_meta(row["meta"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_accessed": row["last_accessed"],
            "access_count": row["access_count"],
        }

    @staticmethod
    def _search_row(row: sqlite3.Row, score: float) -> dict:
        return {
            "id": row["id"],
            "text": row["text"],
            "kind": row["kind"],
            "source": row["source"],
            "confidence": row["confidence"],
            "score": score,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_accessed": row["last_accessed"],
            "access_count": row["access_count"],
        }

    def _search_fts(
        self, tokens: list[str], kind_sql: str, kind_params: tuple[str, ...]
    ) -> list[dict]:
        match = " OR ".join(f'"{t}"*' for t in tokens)
        sql = (
            "SELECT f.id, f.text, f.kind, f.source, f.confidence, "
            "f.created_at, f.updated_at, f.last_accessed, f.access_count, "
            "bm25(facts_fts) AS bm25, "
            "julianday('now') - julianday(f.updated_at) AS age_days "
            "FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
            f"WHERE facts_fts MATCH ? {kind_sql} ORDER BY bm25 LIMIT 50"
        )
        rows = self._conn.execute(sql, (match, *kind_params)).fetchall()
        return [
            self._search_row(r, -r["bm25"] * self._recency_factor(r["age_days"]))
            for r in rows
        ]

    def _search_like(
        self, tokens: list[str], kind_sql: str, kind_params: tuple[str, ...]
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[str] = []
        for tok in tokens:
            clauses.append("f.text LIKE ? ESCAPE '\\'")
            params.append(f"%{self._escape_like(tok)}%")
        where = "(" + " OR ".join(clauses) + ")"
        sql = (
            "SELECT f.id, f.text, f.kind, f.source, f.confidence, "
            "f.created_at, f.updated_at, f.last_accessed, f.access_count, "
            "julianday('now') - julianday(f.updated_at) AS age_days "
            f"FROM facts f WHERE {where} {kind_sql} "
            "ORDER BY f.updated_at DESC, f.id DESC LIMIT 100"
        )
        rows = self._conn.execute(sql, (*params, *kind_params)).fetchall()
        out = []
        for r in rows:
            low = r["text"].lower()
            matches = sum(1 for tok in tokens if tok.lower() in low)
            out.append(
                self._search_row(r, matches * self._recency_factor(r["age_days"]))
            )
        return out

    def _recent_facts(
        self, k: int, kind_sql: str, kind_params: tuple[str, ...]
    ) -> list[dict]:
        sql = (
            "SELECT f.id, f.text, f.kind, f.source, f.confidence, "
            "f.created_at, f.updated_at, f.last_accessed, f.access_count, "
            "julianday('now') - julianday(f.updated_at) AS age_days "
            f"FROM facts f {kind_sql} ORDER BY f.updated_at DESC, f.id DESC LIMIT ?"
        )
        rows = self._conn.execute(sql, (*kind_params, k)).fetchall()
        return [
            self._search_row(r, self._recency_factor(r["age_days"])) for r in rows
        ]

    def _touch_and_return(self, top: list[dict]) -> list[dict]:
        ids = [r["id"] for r in top]
        placeholders = ", ".join("?" for _ in ids)
        with self._tx():
            self._conn.execute(
                "UPDATE facts SET last_accessed = datetime('now'), "
                "access_count = access_count + 1 "
                f"WHERE id IN ({placeholders})",
                ids,
            )
        rows = self._conn.execute(
            "SELECT id, text, kind, source, confidence, created_at, "
            "updated_at, last_accessed, access_count FROM facts "
            f"WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        fresh = {r["id"]: r for r in rows}
        return [
            {
                "id": f["id"],
                "text": f["text"],
                "kind": f["kind"],
                "source": f["source"],
                "confidence": f["confidence"],
                "score": r["score"],
                "created_at": f["created_at"],
                "updated_at": f["updated_at"],
                "last_accessed": f["last_accessed"],
                "access_count": f["access_count"],
            }
            for r in top
            for f in [fresh[r["id"]]]
        ]


__all__ = ["MemoryStore"]
