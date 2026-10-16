"""Registration of every built-in tool into the global ``TOOLS`` registry.

Each tool is a thin async wrapper around the existing implementation
modules. Wrappers receive ``(ctx, **params)`` and always return a
JSON-serializable dict; expected failures (DesktopError, HyprError,
MediaError, WebError, ...) are converted to ``{"error": str(exc)}``.

The registry invokes tool functions as ``func(ctx, arguments_dict)``, so
every wrapper is adapted through :func:`_adapt`, which unpacks that dict
into keyword parameters.
"""

from __future__ import annotations

import functools
from typing import Any

from ..core import resources
from . import desktop, hypr, media, memory_tools, notify, shell, tasks, web
from .registry import TOOLS, NeedsApproval
from .safety import SafetyPolicy


def _adapt(fn):
    """Wrap a ``(ctx, **params)`` tool so it accepts ``(ctx, arguments)``."""

    @functools.wraps(fn)
    async def wrapper(ctx, arguments=None, **kwargs):
        params = dict(arguments or {})
        params.update(kwargs)
        return await fn(ctx, **params)

    return wrapper


def _error(exc: BaseException) -> dict:
    return {"error": str(exc)}


_SITES = {
    "chatgpt": "https://chatgpt.com",
    "youtube": "https://www.youtube.com",
    "github": "https://github.com",
    "reddit": "https://www.reddit.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "spotify": "https://open.spotify.com",
    "wikipedia": "https://en.wikipedia.org",
    "stackoverflow": "https://stackoverflow.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "amazon": "https://www.amazon.com",
    "netflix": "https://www.netflix.com",
    "twitch": "https://www.twitch.tv",
    "discord": "https://discord.com/app",
}


def _resolve_site_name(url: str) -> str | None:
    if "://" in url or "/" in url:
        return None
    key = url.lower()
    if key in _SITES:
        return _SITES[key]
    if "." not in key and key.replace("-", "").isalpha():
        return f"https://www.{key}.com"
    return None


# ---------------------------------------------------------------- shell

@TOOLS.register(
    "run_shell",
    description=(
        "Run a shell command and return stdout, stderr and the exit status. "
        "Privileged commands (sudo/doas) and destructive ones (mkfs, dd, "
        "shutdown, rm -rf of protected paths) are refused outright; commands "
        "that write outside the home directory, install packages, or signal "
        "processes require interactive user approval. Output is capped at "
        "about 20 KB."
    ),
    category="shell",
    params={
        "command": {
            "type": "string",
            "description": "The shell command to run.",
            "required": True,
        },
        "timeout": {
            "type": "integer",
            "default": 120,
            "description": "Timeout in seconds.",
        },
        "cwd": {"type": "string", "description": "Working directory."},
    },
)
@_adapt
async def _run_shell(ctx, command=None, timeout=120, cwd=None, **extra):
    if not command or not str(command).strip():
        return {"error": "command must be a non-empty string"}
    policy = SafetyPolicy(
        approve_write_paths=list(ctx.config.safety.write_approve_paths),
        forbidden_paths=list(ctx.config.safety.forbidden_paths),
    )

    def approve(label, detail):
        try:
            return bool(ctx.approve(label, detail))
        except NeedsApproval:
            return False

    try:
        return shell.run_checked(
            str(command),
            policy=policy,
            cwd=cwd or None,
            timeout=int(timeout or 120),
            approve=approve,
        )
    except shell.ShellError as exc:
        return _error(exc)


# ----------------------------------------------------------- app-launch

@TOOLS.register(
    "launch",
    description=(
        "app — launch an installed application by name (codium, spotify, zen, "
        "foot, mpv...); url — open a website in the browser, accepts site "
        "names like 'chatgpt' or full URLs; path — open a file or folder with "
        "the right app; project — open a project by fuzzy name in the editor. "
        "Workspace optional; launches never steal focus."
    ),
    category="app-launch",
    params={
        "action": {
            "type": "string",
            "enum": ["app", "url", "path", "project"],
            "required": True,
        },
        "target": {
            "type": "string",
            "description": "App name, URL/site name, path, or project name.",
            "required": True,
        },
        "workspace": {"type": "integer", "description": "Workspace number."},
        "args": {
            "type": "string",
            "description": "Extra command-line arguments (action 'app' only).",
        },
    },
)
@_adapt
async def _launch(ctx, action=None, target=None, workspace=None, args="", **extra):
    if not target or not str(target).strip():
        return {"error": "target is required"}
    target = str(target).strip()
    try:
        if action == "app":
            return desktop.launch_app(
                target,
                str(args or ""),
                workspace=workspace,
                silent=True,
                config=ctx.config,
            )
        if action == "url":
            resolved = _resolve_site_name(target)
            if target == target and resolved is None and "://" not in target and "/" not in target:
                return {"error": "give a full URL (https://...)"}
            result = desktop.open_url(resolved or target, workspace, ctx.config)
            if isinstance(result, dict):
                result.setdefault(
                    "note", "opened URL in the browser (no media playback)"
                )
            return result
        if action == "path":
            return desktop.open_path(target, workspace, ctx.config)
        if action == "project":
            return desktop.open_project(target, workspace, ctx.config)
        return {"error": "unknown action (expected app, url, path or project)"}
    except (desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)


@TOOLS.register(
    "start_project",
    description=(
        "Start a project: finds its folder by name under the app directories, "
        "reads the README to extract the run command (e.g. 'npm run dev'), and "
        "launches it in a terminal on the agent workspace. Use this when the "
        "user says 'start/run project X'. If the command needs approval you "
        "will be asked."
    ),
    category="app-launch",
    params={
        "name_or_path": {
            "type": "string",
            "description": "Project name or exact path.",
            "required": True,
        },
        "workspace": {
            "type": "integer",
            "description": "Workspace number (defaults to the agent workspace).",
        },
    },
)
@_adapt
async def _start_project(ctx, name_or_path=None, workspace=None, **extra):
    if not name_or_path or not str(name_or_path).strip():
        return {"error": "name_or_path is required"}
    try:
        result = desktop.start_project(
            str(name_or_path), workspace=workspace, config=ctx.config
        )
    except (desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)
    if isinstance(result, dict) and result.get("status") == "needs-approval":
        ctx.approve(
            "start project",
            f"run {result['command']} for {result['project']}?",
        )
        result = desktop.start_project(
            str(name_or_path), workspace=workspace, config=ctx.config, approved=True
        )
    return result


# ------------------------------------------------------------ workspace

@TOOLS.register(
    "workspace",
    description=(
        "list — show workspaces; clients — show windows; move — silently move "
        "a window (address:0x... or class:...) to a workspace; organize — "
        "rearrange per the user's layout (editors/terminals → 1-2, everything "
        "else → 3-5, focused workspace untouched, nothing closed)."
    ),
    category="workspace",
    params={
        "action": {
            "type": "string",
            "enum": ["list", "clients", "move", "organize"],
            "required": True,
        },
        "target": {
            "type": "string",
            "description": "Window selector (address:... or class:...) for 'move'.",
        },
        "workspace": {
            "type": "integer",
            "description": "Destination workspace number for 'move'.",
        },
    },
)
@_adapt
async def _workspace(ctx, action=None, target=None, workspace=None, **extra):
    try:
        if action == "list":
            try:
                active = hypr.active_workspace()
            except hypr.HyprError:
                active = None
            return {"active": active, "workspaces": hypr.workspaces()}
        if action == "clients":
            return {"clients": hypr.clients()}
        if action == "move":
            if not target or workspace is None:
                return {"error": "action 'move' requires target and workspace"}
            return desktop.move_window(str(target), workspace, silent=True)
        if action == "organize":
            return desktop.organize_workspaces(config=ctx.config)
        return {"error": "unknown action (expected list, clients, move or organize)"}
    except (desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)


# -------------------------------------------------------------- network

@TOOLS.register(
    "web",
    description=(
        "search — search the web (DuckDuckGo) and return titles, URLs and "
        "snippets; fetch — fetch a URL and return its title and extracted "
        "plain text."
    ),
    category="network",
    params={
        "action": {
            "type": "string",
            "enum": ["search", "fetch"],
            "required": True,
        },
        "query": {"type": "string", "description": "Search query."},
        "url": {"type": "string", "description": "Page URL to fetch."},
        "max_results": {
            "type": "integer",
            "default": 6,
            "description": "Maximum number of search results.",
        },
        "max_chars": {
            "type": "integer",
            "default": 8000,
            "description": "Maximum characters of text to return.",
        },
    },
)
@_adapt
async def _web(ctx, action=None, query=None, url=None, max_results=6, max_chars=8000, **extra):
    try:
        if action == "search":
            if not query or not str(query).strip():
                return {"error": "action 'search' requires query"}
            return web.search_duckduckgo(
                str(query), max_results=int(max_results or 6)
            )
        if action == "fetch":
            if not url or not str(url).strip():
                return {"error": "action 'fetch' requires url"}
            return web.fetch_page(str(url), max_chars=int(max_chars or 8000))
        return {"error": "unknown action (expected search or fetch)"}
    except web.WebError as exc:
        return _error(exc)


# ------------------------------------------------------------ app-launch

@TOOLS.register(
    "video",
    description=(
        "search — find YouTube videos (metadata only); play — actually play a "
        "URL/id/query (browser by default, mpv only if asked); anime — play "
        "an anime episode via ani-cli; next_episode — next unwatched episode "
        "from watch history. When the user wants to watch something: search "
        "then play, without asking which one."
    ),
    category="app-launch",
    params={
        "action": {
            "type": "string",
            "enum": ["search", "play", "anime", "next_episode"],
            "required": True,
        },
        "query": {
            "type": "string",
            "description": "Search query / anime title.",
        },
        "video": {
            "type": "string",
            "description": "Video URL, YouTube id, or search text for 'play'.",
        },
        "player": {
            "type": "string",
            "description": "'browser' (default) or 'mpv' (only when the user explicitly asks).",
        },
        "episode": {"type": "integer", "description": "Episode number."},
        "select_nth": {
            "type": "integer",
            "description": "Which anime search result to play.",
        },
        "dub": {"type": "boolean", "description": "Prefer the dubbed version."},
        "workspace": {"type": "integer", "description": "Workspace number."},
    },
)
@_adapt
async def _video(
    ctx,
    action=None,
    query=None,
    video=None,
    player=None,
    episode=None,
    select_nth=1,
    dub=False,
    workspace=None,
    **extra,
):
    try:
        if action == "search":
            if not query or not str(query).strip():
                return {"error": "action 'search' requires query"}
            api_key = (
                getattr(getattr(ctx.config, "web", None), "youtube_api_key", "") or ""
            )
            return media.yt_search(str(query), api_key=api_key)
        if action == "play":
            if not video or not str(video).strip():
                return {"error": "action 'play' requires video"}
            return media.play_video(
                str(video), player=player or None, workspace=workspace,
                config=ctx.config,
            )
        if action == "anime":
            if not query or not str(query).strip():
                return {"error": "action 'anime' requires query"}
            return media.ani_play(
                str(query),
                episode=episode,
                select_nth=int(select_nth or 1),
                dub=bool(dub),
                workspace=workspace,
                config=ctx.config,
            )
        if action == "next_episode":
            return media.ani_next(
                query=str(query) if query else None, config=ctx.config
            )
        return {
            "error": "unknown action (expected search, play, anime or next_episode)"
        }
    except (media.MediaError, desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)


# -------------------------------------------------------------- memory

@TOOLS.register(
    "memory",
    description=(
        "add — store a durable fact about the user (preferences, projects, "
        "decisions); recall — search memory for facts related to a query; "
        "list — list stored facts newest first, optionally by kind; delete — "
        "delete a fact by id or text match (needs approval)."
    ),
    category="memory",
    params={
        "action": {
            "type": "string",
            "enum": ["add", "recall", "list", "delete"],
            "required": True,
        },
        "fact": {"type": "string", "description": "The fact to remember."},
        "query": {
            "type": "string",
            "description": "What to search for / text match to delete.",
        },
        "kind": {
            "type": "string",
            "description": "Category: fact, preference, project, ...",
        },
        "fact_id": {"type": "integer", "description": "Fact id to delete."},
        "k": {
            "type": "integer",
            "default": 5,
            "description": "Maximum number of recall matches.",
        },
        "limit": {
            "type": "integer",
            "default": 20,
            "description": "Maximum number of facts to list.",
        },
        "confidence": {
            "type": "number",
            "description": "How sure you are (0-1).",
        },
    },
)
@_adapt
async def _memory(
    ctx,
    action=None,
    fact=None,
    query=None,
    kind=None,
    fact_id=None,
    k=5,
    limit=20,
    confidence=1.0,
    **extra,
):
    if action == "add":
        if not fact or not str(fact).strip():
            return {"error": "action 'add' requires fact"}
        return await memory_tools.memory_add(
            ctx,
            str(fact),
            kind=str(kind or "fact"),
            confidence=float(confidence or 1.0),
        )
    if action == "recall":
        if not query or not str(query).strip():
            return {"error": "action 'recall' requires query"}
        return await memory_tools.memory_recall(ctx, str(query), k=int(k or 5))
    if action == "list":
        return await memory_tools.memory_list(
            ctx, limit=int(limit or 20), kind=str(kind) if kind else None
        )
    if action == "delete":
        if fact_id is None and not query:
            return {"error": "action 'delete' requires fact_id or query"}
        return await memory_tools.memory_delete(
            ctx, fact_id=fact_id, query=str(query) if query else None
        )
    return {"error": "unknown action (expected add, recall, list or delete)"}


# --------------------------------------------------------------- tasks

@TOOLS.register(
    "todo",
    description=(
        "add — add an open todo to the persistent list; list — list todos "
        "(open by default, include_done shows all); toggle — toggle or set "
        "the done state of a todo by id or title text."
    ),
    category="tasks",
    params={
        "action": {
            "type": "string",
            "enum": ["add", "list", "toggle"],
            "required": True,
        },
        "title": {"type": "string", "description": "Todo text."},
        "query": {"type": "string", "description": "Title text to match."},
        "todo_id": {"type": "integer", "description": "Todo id."},
        "include_done": {
            "type": "boolean",
            "default": False,
            "description": "Also show completed todos.",
        },
        "done": {"type": "boolean", "description": "Force done/not done."},
    },
)
@_adapt
async def _todo(
    ctx, action=None, title=None, query=None, todo_id=None,
    include_done=False, done=None, **extra,
):
    if action == "add":
        if not title or not str(title).strip():
            return {"error": "action 'add' requires title"}
        return await tasks.todo_add(ctx, str(title))
    if action == "list":
        return await tasks.todo_list(ctx, include_done=bool(include_done))
    if action == "toggle":
        if todo_id is None and not query:
            return {"error": "action 'toggle' requires todo_id or query"}
        return await tasks.todo_toggle(
            ctx,
            todo_id=todo_id,
            query=str(query) if query else None,
            done=done,
        )
    return {"error": "unknown action (expected add, list or toggle)"}


@TOOLS.register(
    "reminder",
    description=(
        "add — schedule a reminder ('in 30m', 'at 18:30', 'tomorrow 9am'); "
        "list — list scheduled reminders, soonest first; cancel — cancel a "
        "reminder by id."
    ),
    category="tasks",
    params={
        "action": {
            "type": "string",
            "enum": ["add", "list", "cancel"],
            "required": True,
        },
        "message": {"type": "string", "description": "Reminder text."},
        "when": {"type": "string", "description": "Human time expression."},
        "reminder_id": {"type": "integer", "description": "Reminder id."},
    },
)
@_adapt
async def _reminder(
    ctx, action=None, message=None, when=None, reminder_id=None, **extra
):
    if action == "add":
        if not message or not when:
            return {"error": "action 'add' requires message and when"}
        return await tasks.reminder_add(ctx, str(message), str(when))
    if action == "list":
        return await tasks.reminder_list(ctx)
    if action == "cancel":
        if reminder_id is None:
            return {"error": "action 'cancel' requires reminder_id"}
        return await tasks.reminder_cancel(ctx, int(reminder_id))
    return {"error": "unknown action (expected add, list or cancel)"}


@TOOLS.register(
    "timer",
    description=(
        "set — start a countdown timer ('20m', '1h30m', '90s', '1:30'); "
        "status — report remaining time on all running timers; cancel — "
        "cancel a timer by id, or the only running one when omitted."
    ),
    category="tasks",
    params={
        "action": {
            "type": "string",
            "enum": ["set", "status", "cancel"],
            "required": True,
        },
        "label": {"type": "string", "description": "Timer label."},
        "duration": {
            "type": "string",
            "description": "Countdown duration (e.g. '20m', '1h30m', '90s').",
        },
        "timer_id": {
            "type": "integer",
            "description": "Timer id to cancel.",
        },
    },
)
@_adapt
async def _timer(ctx, action=None, label="timer", duration=None, timer_id=None, **extra):
    if action == "set":
        if duration is None:
            return {"error": "action 'set' requires duration"}
        return await tasks.timer_set(
            ctx, label=str(label or "timer"), duration=str(duration)
        )
    if action == "status":
        return await tasks.timer_status(ctx)
    if action == "cancel":
        if timer_id is None:
            return await tasks.timer_cancel(ctx)
        return await tasks.timer_cancel(ctx, int(timer_id))
    return {"error": "unknown action (expected set, status or cancel)"}


# ---------------------------------------------------------------- info

@TOOLS.register(
    "notify",
    description="Pop a desktop toast notification addressed to the user.",
    category="info",
    params={
        "message": {
            "type": "string",
            "description": "Notification text.",
            "required": True,
        }
    },
)
@_adapt
async def _notify(ctx, message=None, **extra):
    if not message or not str(message).strip():
        return {"error": "message is required"}
    return await notify.notify_user(ctx, str(message))


@TOOLS.register(
    "system_status",
    description=(
        "Report RAM/swap, load average, GPU VRAM and the top memory-hungry "
        "processes. Check before heavy work (model loads, builds, ffmpeg)."
    ),
    category="info",
    params={},
)
@_adapt
async def _system_status(ctx, **extra):
    return resources.system_status()


__all__ = ["TOOLS"]
