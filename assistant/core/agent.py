"""The agentic loop: context building, tool execution, final answers.

``Agent.run_once`` drives one full turn: persist the user message, build the
system prompt (persona + injected context: local time, relevant memories,
open todos, upcoming reminders, system status), then run up to 8 rounds of
engine chat + tool calls before producing the final answer.

Tool-call history is shaped the way llama-cpp's chat handlers expect:
assistant messages carry ``tool_calls`` as a list of
``{"type": "function", "function": {"name", "arguments"}, "id"}`` objects
with ``arguments`` serialized to a JSON string.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from datetime import datetime
from typing import Any

from ..config import Config
from ..llm.engine import EngineError
from ..llm.templates import (
    assistant_message,
    format_tool_result,
    system_message,
    tool_result_message,
    user_message,
)
from ..persona import build_system_prompt
from ..tools import builtin as _builtin  # noqa: F401 — registers the tools
from ..tools.registry import TOOLS, NeedsApproval, ToolContext, ToolResult
from .resources import system_status

log = logging.getLogger("assistant")

_MAX_TOOL_ROUNDS = 8

#: Context tokens kept in reserve for the system prompt + injected tools
#: block + generation headroom when trimming history to the model budget.
_CONTEXT_RESERVE = 2600

_ACTION_CLAIM_RE = re.compile(
    r"\b(?:set|setting|started|starting|start|opened|opening|open|"
    r"launched|launching|launch|playing|played|play|playback|"
    r"streaming|stream|scheduled|switching|switched|switch|"
    r"moved|move|checking|checked|check|created|will start)\b|"
    r"you'?ll be notified|it'?s running|now playing|system status",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"can'?t|cannot|couldn'?t|didn'?t|did not|haven'?t|have not|hasn'?t|"
    r"has not|won'?t|will not|no active|there are no|don'?t have|failed|denied",
    re.IGNORECASE,
)


def _looks_like_action_claim(text: str) -> bool:
    """Heuristic: does ``text`` read like a claim that an action happened?

    False for questions and for text carrying negations/absence markers, so
    honest "nothing happened" answers never trigger the audit round.
    """
    text = (text or "").strip()
    if not text:
        return False
    if text.endswith("?"):
        return False
    if _NEGATION_RE.search(text):
        return False
    return bool(_ACTION_CLAIM_RE.search(text) or _TIME_CLAIM_RE.search(text))


_TIME_CLAIM_RE = re.compile(
    r"\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+(?:left|remaining)\b"
    r"|\bremaining\s+(?:time|minutes?|seconds?|hours?)\b"
    r"|\b(?:elapsed|remaining)\s+time\b"
    r"|\bhas\s+been\s+running\s+for\b"
    r"|\btimer\s+(?:status|left|remaining)\b"
    r"|\btime\s+(?:on|of|for)\s+(?:the\s+)?timer\b"
    r"|\btimer\b.*?\b(?:status|left|remaining|time|reading|readout|countdown)\b"
    r"|\b(?:status|left|remaining|time|reading|readout|countdown)\b.*?\btimer\b",
    re.IGNORECASE,
)


_CLAIM_VERB_TOOLS = {
    "play": {"play_video", "ani_play"},
    "playing": {"play_video", "ani_play"},
    "played": {"play_video", "ani_play"},
    "playback": {"play_video", "ani_play"},
    "stream": {"play_video", "ani_play"},
    "streaming": {"play_video", "ani_play"},
    "starting": {"play_video", "ani_play"},
    "open": {"open_url", "open_app", "open_path", "open_project", "play_video"},
    "opening": {"open_url", "open_app", "open_path", "open_project", "play_video"},
    "opened": {"open_url", "open_app", "open_path", "open_project", "play_video"},
    "launch": {"open_url", "open_app", "open_path", "open_project", "play_video"},
    "launching": {"open_url", "open_app", "open_path", "open_project", "play_video"},
    "launched": {"open_url", "open_app", "open_path", "open_project", "play_video"},
    "set": {"timer_set", "reminder_add", "reminder_cancel"},
    "setting": {"timer_set", "reminder_add", "reminder_cancel"},
    "scheduled": {"timer_set", "reminder_add", "reminder_cancel"},
    "notified": {"timer_set", "reminder_add", "reminder_cancel"},
    "you'll be notified": {"timer_set", "reminder_add", "reminder_cancel"},
    "switch": {"move_window", "focus_workspace", "organize_workspaces"},
    "switched": {"move_window", "focus_workspace", "organize_workspaces"},
    "move": {"move_window", "focus_workspace", "organize_workspaces"},
    "moved": {"move_window", "focus_workspace", "organize_workspaces"},
    "checking": {"system_status", "hypr_workspaces", "hypr_clients", "timer_status"},
    "checked": {"system_status", "hypr_workspaces", "hypr_clients", "timer_status"},
    "check": {"system_status", "hypr_workspaces", "hypr_clients", "timer_status"},
    "system status": {"system_status", "hypr_workspaces", "hypr_clients", "timer_status"},
}

_CLAIM_VERB_RES = {
    verb: (
        re.compile(rf"\b{verb}(?:s|ed|ing|d)?\b", re.IGNORECASE)
        if re.fullmatch(r"[a-z]+", verb)
        else re.compile(re.escape(verb), re.IGNORECASE)
    )
    for verb in _CLAIM_VERB_TOOLS
}

_SITE_KEYWORDS = [
    "youtube",
    "chatgpt",
    "chat gpt",
    "openai",
    "browser",
    "website",
    "web page",
    "url",
    "github",
    "reddit",
    "netflix",
    "twitch",
    "spotify web",
    "google",
]

_SITE_KEYWORD_RES = [
    re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    for keyword in _SITE_KEYWORDS
]

_AUDIT_GENERIC = (
    "[audit] You claimed to have performed an action, but you called no tools "
    "this turn. Either call the appropriate tool now (e.g. timer_set, open_url, "
    "play_video, open_project) or say plainly what you can actually do instead. "
    "Never claim an action without calling its tool."
)


def _claim_tool_hint(text: str, called_tool_names: list[str]) -> list[str]:
    """Tool names for action families claimed by ``text`` but never called.

    Applies the same question/negation guards as the claim detectors; a
    family contributes its tool names only when none of them were called.
    Claims mentioning a site (youtube, chatgpt, browser, website, url, ...)
    additionally require ``open_url`` (or ``play_video``, which opens pages
    in the browser) — the open verb family alone is not enough.
    """
    text = (text or "").strip()
    if not text:
        return []
    if text.endswith("?"):
        return []
    if _NEGATION_RE.search(text):
        return []
    called = set(called_tool_names or [])
    hints: list[str] = []
    for verb, tools in _CLAIM_VERB_TOOLS.items():
        if _CLAIM_VERB_RES[verb].search(text) and not (tools & called):
            for tool in sorted(tools):
                if tool not in hints:
                    hints.append(tool)
    if _TIME_CLAIM_RE.search(text) and "timer_status" not in called:
        if "timer_status" not in hints:
            hints.append("timer_status")
    site_mention = any(pattern.search(text) for pattern in _SITE_KEYWORD_RES)
    if site_mention and not ({"open_url", "play_video"} & called):
        if "open_url" not in hints:
            hints.append("open_url")
    return hints


def _claims_unsupported_action(text: str, called_tool_names: list[str]) -> bool:
    """Does ``text`` claim an action whose tools were not called this turn?

    True either when a claimed verb family has no called tool, or when the
    text mentions a site keyword without ``open_url`` (or ``play_video``)
    having been called — the "open" verb family alone does not satisfy a
    site mention.
    """
    return bool(_claim_tool_hint(text, called_tool_names))


def _detect_claim(
    text: str, tools_called: bool, called_tool_names: list[str]
) -> bool:
    """Pick the right claim detector for the current stage of the turn."""
    if not tools_called:
        return _looks_like_action_claim(text)
    return _claims_unsupported_action(text, called_tool_names)


def _claim_excerpt(text: str, limit: int = 60) -> str:
    """A short single-line excerpt of ``text`` for audit messages."""
    excerpt = (text or "").strip().replace("\n", " ")
    if len(excerpt) > limit:
        excerpt = excerpt[:limit].rstrip() + "…"
    return excerpt


def _run_coro(coro: Any) -> Any:
    """Run a coroutine to completion, tolerating an already-running loop.

    Used only by sync callers of :meth:`Agent._build_system_prompt`; the
    normal path (inside ``run_once``) awaits the context coroutines directly.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        box: dict[str, Any] = {}

        def _go() -> None:
            try:
                box["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc

        thread = threading.Thread(target=_go, daemon=True)
        thread.start()
        thread.join(timeout=15.0)
        if "error" in box:
            log.debug("context coroutine failed: %s", box["error"])
            return None
        return box.get("value")


class Agent:
    """Owns one conversation turn: prompt building and the tool loop."""

    def __init__(
        self,
        config: Config,
        console,
        engine,
        store,
        runner=None,
        session=None,
    ) -> None:
        self.config = config
        self.console = console
        self.engine = engine
        self.store = store
        self.runner = runner
        if session is None:
            from .session import Session

            session = Session(config, store)
        self.session = session

    # ----------------------------------------------------------- context

    def _context_tools(self) -> ToolContext:
        return ToolContext(
            self.config, self.console, self.session, self.runner, self.store
        )

    async def _fetch_tasks(self) -> tuple[list, list]:
        """Open todos (top 5) and upcoming reminders (next 3) for the prompt."""
        from ..tools import tasks

        ctx = self._context_tools()
        todos: list = []
        reminders: list = []
        try:
            result = await tasks.todo_list(ctx, include_done=False)
            todos = list((result or {}).get("todos", []) or [])[:5]
        except Exception:  # noqa: BLE001 — context must never kill the turn
            log.debug("todo fetch failed", exc_info=True)
        try:
            result = await tasks.reminder_list(ctx)
            reminders = list((result or {}).get("reminders", []) or [])
            reminders.sort(key=lambda r: str(r.get("at", "")))
            reminders = reminders[:3]
        except Exception:  # noqa: BLE001
            log.debug("reminder fetch failed", exc_info=True)
        return todos, reminders

    def _build_system_prompt(
        self,
        user_text: str,
        todos: list | None = None,
        reminders: list | None = None,
    ) -> str:
        """Persona prompt plus a plain-text injected context block.

        Context: local time, relevant durable memories (k=4), open todos
        (top 5), upcoming reminders (next 3), and a one-line system status
        (RAM available, VRAM free, load average).
        """
        prompt = build_system_prompt(self.config)
        if todos is None or reminders is None:
            fetched = _run_coro(self._fetch_tasks())
            if fetched:
                todos, reminders = fetched
            else:
                todos, reminders = [], []

        lines = [
            "",
            "# Injected context",
            "",
            f"- Local time: {datetime.now().strftime('%A %Y-%m-%d %H:%M')}",
        "- You are the assistant project: your own source code lives at "
        "/mnt/Data/app/assistant (project name 'assistant').",
        "If the user refers to 'the project where you live / were made / "
        "'are running from', open that path.",
        ]
        try:
            facts = self.store.search_facts(str(user_text), k=4)
        except Exception:  # noqa: BLE001
            facts = []
        if facts:
            lines.append("")
            lines.append("Relevant memories (from durable memory):")
            for fact in facts:
                lines.append(f"- {str(fact.get('text', '')).strip()}")
        if todos:
            lines.append("")
            lines.append("Open todos:")
            for todo in todos:
                title = str(todo.get("title", "")).strip()
                if title:
                    lines.append(f"- [{todo.get('id')}] {title}")
        if reminders:
            lines.append("")
            lines.append("Upcoming reminders:")
            for reminder in reminders:
                message = str(reminder.get("message", "")).strip()
                at = str(reminder.get("at", "")).strip()
                if message:
                    lines.append(f"- {at}: {message}")

        status = system_status()
        ram = status.get("ram") or {}
        gpu = status.get("gpu") or {}
        load = status.get("load_avg") or []
        if gpu.get("available"):
            vram = f"{gpu.get('vram_free_mb')} MB VRAM free"
        else:
            vram = "no GPU info"
        load_str = "/".join(f"{x:.2f}" for x in load) if load else "n/a"
        lines.append("")
        lines.append(
            f"System status: {ram.get('avail_mb')} MB RAM available, "
            f"{vram}, load {load_str}."
        )
        return prompt + "\n".join(lines)

    # --------------------------------------------------------------- chat

    async def _chat(self, messages: list[dict], tools: list | None = None) -> dict | None:
        """One engine call with streaming; returns None on error/interrupt."""
        try:
            with self.console.stream() as sh:
                return self.engine.chat(messages, tools=tools, stream_cb=sh.append)
        except KeyboardInterrupt:
            self.console.warn("interrupted")
            return None
        except EngineError as exc:
            self.console.error(f"model error: {exc}")
            log.exception("engine error during chat")
            return None
        except Exception as exc:  # noqa: BLE001
            self.console.error(f"unexpected error: {exc}")
            log.exception("unexpected error during chat")
            return None

    async def _chat_with_context_retry(
        self, messages: list[dict], tools: list | None = None
    ) -> dict | None:
        """One chat call with a single context-overflow retry.

        When the engine raises an :class:`EngineError` mentioning context,
        the oldest ~40% of the message history (keeping the system message)
        is dropped and the SAME call is retried once; a second failure is
        reported to the console and the round aborts.
        """
        for attempt in range(2):
            try:
                with self.console.stream() as sh:
                    return self.engine.chat(messages, tools=tools, stream_cb=sh.append)
            except KeyboardInterrupt:
                self.console.warn("interrupted")
                return None
            except EngineError as exc:
                if "context" not in str(exc).casefold() or attempt == 1:
                    self.console.error(f"model error: {exc}")
                    log.exception("engine error during chat")
                    return None
                log.warning(
                    "context overflow detected (%s): dropping oldest history "
                    "and retrying",
                    exc,
                )
                history_end = len(messages) - 1  # newest message is kept
                drop = max(1, int((history_end - 1) * 0.4))
                if drop > 0:
                    del messages[1:1 + min(drop, history_end - 1)]
        return None

    # ------------------------------------------------------------ turn

    async def run_once(self, user_text: str) -> None:
        """Run one full agentic turn for ``user_text``."""
        user_text = str(user_text or "")
        self.session.add_message("user", user_text)
        todos, reminders = await self._fetch_tasks()
        system_prompt = self._build_system_prompt(
            user_text, todos=todos, reminders=reminders
        )

        budget = max(400, self.config.model.n_ctx - _CONTEXT_RESERVE)
        if getattr(self.engine, "loaded", False) and callable(
            getattr(self.engine, "token_count", None)
        ):
            estimate_tokens_ctx = self.engine.token_count
        else:
            estimate_tokens_ctx = lambda text: max(1, int(len(text) / 3.2))  # noqa: E731
        history = self.session.trim_to_tokens(budget, estimate=estimate_tokens_ctx)
        if (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content") == user_text
        ):
            history = history[:-1]  # the current message is appended below

        messages = [system_message(system_prompt)]
        messages.extend(history)
        messages.append(user_message(user_text))

        tools_called = False
        called_tool_names: list[str] = []
        for _ in range(_MAX_TOOL_ROUNDS):
            response = await self._chat_with_context_retry(
                messages, tools=TOOLS.openai_schemas()
            )
            if response is None:
                return
            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                final = response.get("content") or ""
                if not _detect_claim(final, tools_called, called_tool_names):
                    self.session.add_message("assistant", final)
                    return
                messages.append(assistant_message(final))
                for audit_round in range(2):
                    if audit_round == 0:
                        hints = _claim_tool_hint(final, called_tool_names)
                        audit_text = (
                            "[audit] The user's request needs an action. If "
                            "you have not called the tool yet, call it now: "
                            f"{', '.join(hints)}. If you already performed it "
                            "via a tool in a previous turn, briefly summarize "
                            "that result. Do not claim an action that no tool "
                            "result proves."
                            if hints
                            else _AUDIT_GENERIC
                        )
                    else:
                        audit_text = (
                            f"[audit] You AGAIN claimed '{_claim_excerpt(final)}' "
                            "without calling a tool. The user's computer did "
                            "not change. Either call the tool now or retract "
                            "the claim explicitly."
                        )
                    messages.append(user_message(audit_text))
                    response = await self._chat(
                        messages, tools=TOOLS.openai_schemas()
                    )
                    if response is None:
                        return
                    tool_calls = response.get("tool_calls") or []
                    if tool_calls:
                        break
                    final = response.get("content") or ""
                    messages.append(assistant_message(final))
                    if not _detect_claim(final, tools_called, called_tool_names):
                        self.session.add_message("assistant", final)
                        return
                else:
                    self.console.warn(
                        "note: I couldn't verify that — no tool confirmed "
                        "the action"
                    )
                    self.session.add_message("assistant", final)
                    return
                # fall through: run the audit round's tool calls below

            shaped = [
                {
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(
                            tc.get("arguments") or {}, default=str
                        ),
                    },
                    "id": tc.get("id"),
                }
                for tc in tool_calls
            ]
            messages.append(
                assistant_message(response.get("content") or "", tool_calls=shaped)
            )

            for tc in tool_calls:
                name = tc.get("name", "")
                called_tool_names.append(name)
                arguments = tc.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                self.console.tool(f"{name} {json.dumps(arguments, default=str)[:160]}")
                ctx = self._context_tools()
                try:
                    result = await TOOLS.call(name, arguments, ctx)
                except NeedsApproval as exc:
                    granted = self.console.ask_yes_no(
                        f"approve: {exc.label} — {str(exc.detail)[:200]}?",
                        default=False,
                    )
                    if granted:
                        result = await TOOLS.call(name, arguments, ctx, force=True)
                    else:
                        result = ToolResult(ok=False, error="denied by user")
                content = format_tool_result(
                    name,
                    {
                        "ok": result.ok,
                        "data": result.data,
                        "error": result.error,
                        "note": result.note,
                    },
                )
                messages.append(
                    tool_result_message(str(tc.get("id") or ""), content)
                )

            tools_called = True

        summary = await self._chat(
            messages
            + [user_message("Summarize what you just did for the user, briefly.")],
            tools=None,
        )
        if summary is not None:
            final = summary.get("content") or ""
            self.session.add_message("assistant", final)
