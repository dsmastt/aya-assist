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


#: Action verbs -> the action-aware proof each family needs.
#: Families: play/stream/starting -> video with a real play call (a search
#: alone is not enough); open/launch -> launch with the action the claim
#: implies (url for sites, app for apps, project/path for projects);
#: set/schedule -> the timer set/status/cancel mapping; move/switch ->
#: workspace; check/status -> system_status/workspace/timer. Families whose
#: tools carry no action keep the name-level check.
_PLAY_VERBS = (
    "play",
    "playing",
    "played",
    "playback",
    "stream",
    "streaming",
    "starting",
)
_OPEN_VERBS = (
    "open",
    "opening",
    "opened",
    "launch",
    "launching",
    "launched",
)
_SET_VERBS = ("set", "setting", "scheduled")
_NOTIFY_VERBS = ("notified", "you'll be notified")
_MOVE_VERBS = ("switch", "switched", "move", "moved")
_CHECK_VERBS = ("checking", "checked", "check", "system status")


def _verb_res(verbs: tuple[str, ...]) -> list[re.Pattern]:
    """Compiled regexes for a verb family (word boundary + light suffixes)."""
    return [
        re.compile(rf"\b{verb}(?:s|ed|ing|d)?\b", re.IGNORECASE)
        if re.fullmatch(r"[a-z]+", verb)
        else re.compile(re.escape(verb), re.IGNORECASE)
        for verb in verbs
    ]


_PLAY_VERB_RES = _verb_res(_PLAY_VERBS)
_OPEN_VERB_RES = _verb_res(_OPEN_VERBS)
_SET_VERB_RES = _verb_res(_SET_VERBS)
_NOTIFY_VERB_RES = _verb_res(_NOTIFY_VERBS)
_MOVE_VERB_RES = _verb_res(_MOVE_VERBS)
_CHECK_VERB_RES = _verb_res(_CHECK_VERBS)

#: (tool, action) pairs proving a play claim: an actual video play (or
#: anime) call — ``video search`` alone never proves playback.
_VIDEO_PLAY_ACTIONS = {("video", "play"), ("video", "anime")}
#: (tool, action) pairs proving an "opened the project/folder" claim.
_LAUNCH_PROJECT_ACTIONS = {("launch", "project"), ("launch", "path")}

_SITE_KEYWORDS = [
    "youtube",
    "chatgpt",
    "chat gpt",
    "openai",
    "browser",
    "website",
    "web page",
    "url",
    "site",
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

#: The site keywords counted for multi-site claims. A claim mentioning
#: two distinct sites ("YouTube on ws 3 and ChatGPT on ws 5") needs one
#: ``launch (action: url)`` per site to be proven.
_SITE_COUNT_KEYWORDS = [
    "youtube",
    "chatgpt",
    "chat gpt",
    "openai",
    "github",
    "reddit",
    "netflix",
    "twitch",
    "google",
    "browser",
    "website",
    "url",
]

_SITE_COUNT_RES = [
    re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    for keyword in _SITE_COUNT_KEYWORDS
]

_PROJECT_KEYWORD_RES = [
    re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    for keyword in ("project", "folder")
]
_APP_WORD_RE = re.compile(r"\bapp(?:lication)?s?\b", re.IGNORECASE)
_TIMER_WORD_RE = re.compile(r"\btimers?\b", re.IGNORECASE)
_CANCEL_VERB_RE = re.compile(r"\bcancel(?:led|ed|ling|lation)?\b", re.IGNORECASE)

_AUDIT_GENERIC = (
    "[audit] You claimed to have performed an action, but you called no tools "
    "this turn. Either call the appropriate tool now (e.g. timer, launch, "
    "video, start_project) or say plainly what you can actually do instead. "
    "Never claim an action without calling its tool."
)


def _distinct_site_mentions(text: str) -> int:
    """Number of DISTINCT site keywords mentioned in ``text``.

    Sizes the launch (action: url) proof against the claim: a claim
    naming two sites is only proven by one url launch per site.
    """
    text = (text or "").casefold()
    return sum(1 for res in _SITE_COUNT_RES if res.search(text))


def _claim_tool_hint(
    text: str, called_actions: list[tuple[str, str] | str]
) -> list[str]:
    """Tool hints for action families claimed by ``text`` but not proven.

    ``called_actions`` are the ``(tool, action)`` pairs recorded while the
    turn's tool calls executed (a bare tool name counts as no action).
    Action-aware families require the right action, not just the tool
    name: play/stream/starting claims need a real video play call (search
    alone is not enough); open claims need the launch action the claim
    implies (url for sites, app for apps, project/path for projects);
    timer claims need the matching set/status/cancel call. Families whose
    tools have no actions keep the name-level check. Hints carry the
    expected action as a parenthetical, e.g. "video (action: play)", so
    the audit message can name both the tool and the action.
    """
    text = (text or "").strip()
    if not text:
        return []
    if text.endswith("?"):
        return []
    if _NEGATION_RE.search(text):
        return []
    raw_actions = [
        tuple(item) if isinstance(item, (tuple, list)) else (str(item), None)
        for item in (called_actions or [])
    ]
    called = set(raw_actions)
    called_names = {name for name, _ in called}
    hints: list[str] = []

    def add(hint: str) -> None:
        if hint not in hints:
            hints.append(hint)

    def any_res(res_list: list[re.Pattern]) -> bool:
        return any(res.search(text) for res in res_list)

    if any_res(_PLAY_VERB_RES) and not (_VIDEO_PLAY_ACTIONS & called):
        add("video (action: play)")

    if any_res(_OPEN_VERB_RES):
        if any_res(_SITE_KEYWORD_RES):
            if ("launch", "url") not in called and "video" not in called_names:
                add("launch (action: url)")
        elif any_res(_PROJECT_KEYWORD_RES):
            if (
                not (_LAUNCH_PROJECT_ACTIONS & called)
                and "start_project" not in called_names
            ):
                add("launch (action: project)")
        elif _APP_WORD_RE.search(text):
            if ("launch", "app") not in called:
                add("launch (action: app)")
        elif not ({"launch", "start_project"} & called_names):
            add("launch")

    if any_res(_SET_VERB_RES):
        if _TIMER_WORD_RE.search(text):
            if ("timer", "set") not in called:
                add("timer (action: set)")
        elif not ({"timer", "reminder", "todo"} & called_names):
            for tool in ("timer", "reminder", "todo"):
                add(tool)

    if _CANCEL_VERB_RE.search(text) and _TIMER_WORD_RE.search(text):
        if ("timer", "cancel") not in called:
            add("timer (action: cancel)")

    if any_res(_NOTIFY_VERB_RES) and not (
        {"timer", "reminder", "todo"} & called_names
    ):
        for tool in ("timer", "reminder", "todo"):
            add(tool)

    if any_res(_MOVE_VERB_RES) and "workspace" not in called_names:
        add("workspace")

    if any_res(_CHECK_VERB_RES) and not (
        {"system_status", "workspace", "timer"} & called_names
    ):
        for tool in ("system_status", "workspace", "timer"):
            add(tool)

    if _TIME_CLAIM_RE.search(text) and ("timer", "status") not in called:
        add("timer (action: status)")

    site_mention = any_res(_SITE_KEYWORD_RES)
    if site_mention and "video" not in called_names:
        url_calls = sum(
            1 for name, action in raw_actions
            if name == "launch" and action == "url"
        )
        site_count = _distinct_site_mentions(text)
        if ("launch", "url") not in called:
            add("launch (action: url)")
        elif site_count >= 2 and url_calls < site_count:
            add("launch (action: url)")

    return hints


def _claims_unsupported_action(
    text: str, called_actions: list[tuple[str, str] | str]
) -> bool:
    """Does ``text`` claim an action whose (tool, action) calls were not
    made this turn?

    True either when a claimed verb family has no proving call, or when the
    text mentions a site keyword without a ``launch url`` (or a ``video``
    call, which opens pages in the browser) having been made.
    """
    return bool(_claim_tool_hint(text, called_actions))


def _detect_claim(
    text: str, tools_called: bool, called_actions: list[tuple[str, str] | str]
) -> bool:
    """Pick the right claim detector for the current stage of the turn."""
    if not tools_called:
        return _looks_like_action_claim(text)
    return _claims_unsupported_action(text, called_actions)


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
        """Persona prompt plus a compact injected context block.

        Context: local time, top-3 relevant durable memories, and a one-line
        system status (RAM available, VRAM free, load average). Todos and
        reminders are left for the model to fetch via tools on demand.
        """
        prompt = build_system_prompt(self.config)
        try:
            facts = self.store.search_facts(str(user_text), k=3)
        except Exception:  # noqa: BLE001
            facts = []

        lines = [
            "",
            "# Injected context",
            "",
            f"- Local time: {datetime.now().strftime('%A %Y-%m-%d %H:%M')}",
            "- You are the assistant project: your own source code lives at "
            "/mnt/Data/app/assistant (project name 'assistant'). If the user "
            "refers to 'the project where you live / were made / are running "
            "from', open that path.",
        ]
        for fact in facts:
            text = str(fact.get("text", "")).strip()
            if text:
                lines.append(f"- {text}")

        status = system_status()
        ram = status.get("ram") or {}
        gpu = status.get("gpu") or {}
        load = status.get("load_avg") or []
        if gpu.get("available"):
            vram = f"{gpu.get('vram_free_mb')} MB VRAM free"
        else:
            vram = "no GPU info"
        load_str = "/".join(f"{x:.2f}" for x in load) if load else "n/a"
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
        called_actions: list[tuple[str, str]] = []
        for _ in range(_MAX_TOOL_ROUNDS):
            response = await self._chat_with_context_retry(
                messages, tools=TOOLS.openai_schemas()
            )
            if response is None:
                return
            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                final = response.get("content") or ""
                if not _detect_claim(final, tools_called, called_actions):
                    self.session.add_message("assistant", final)
                    return
                messages.append(assistant_message(final))
                for audit_round in range(2):
                    if audit_round == 0:
                        hints = _claim_tool_hint(final, called_actions)
                        # hints carry "(action: ...)" parentheticals, so
                        # the audit message names both tool and action.
                        audit_text = (
                            "[audit] The user's request needs an action. If "
                            "you have not called the tool yet, call it now "
                            "(tool and expected action): "
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
                    if not _detect_claim(final, tools_called, called_actions):
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
                arguments = tc.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                action = str(arguments.get("action") or name)
                called_actions.append((name, action))
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
