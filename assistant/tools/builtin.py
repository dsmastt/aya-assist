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
    "open_app",
    description=(
        "Launch a desktop application, optionally onto a specific Hyprland "
        "workspace. Uses an inline windowrule so the user's current workspace "
        "never loses focus. Use this for launching desktop applications by "
        "name — not for websites (open_url) and not for videos (play_video). "
        "Known installed apps include: codium (editor), code, zen (browser), "
        "firefox, foot (terminal), kitty, mpv (video player), vlc, spotify "
        "(music player — it IS installed), discord, steam, btop, htop, "
        "thunar, dolphin, qbittorrent, ani-cli, notify-send. If you are "
        "unsure whether an app is installed, call it anyway — the tool "
        "reports clearly when an app is missing. Websites are NOT apps — "
        "use open_url for sites. To open a project or folder by name, use "
        "open_project instead."
    ),
    category="app-launch",
    params={
        "app": {
            "type": "string",
            "description": "App name (codium, zen, foot, mpv, ...).",
            "required": True,
        },
        "args": {
            "type": "string",
            "default": "",
            "description": "Extra command-line arguments.",
        },
        "workspace": {
            "type": "integer",
            "description": "Workspace number to open the app on.",
        },
        "silent": {
            "type": "boolean",
            "default": True,
            "description": "Do not focus the new window.",
        },
    },
)
@_adapt
async def _open_app(ctx, app=None, args="", workspace=None, silent=True, **extra):
    if not app or not str(app).strip():
        return {"error": "app name is required"}
    try:
        return desktop.launch_app(
            str(app),
            str(args or ""),
            workspace=workspace,
            silent=True,
            config=ctx.config,
        )
    except (desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)


@TOOLS.register(
    "open_path",
    description=(
        "Open a file or directory with the right application: xdg-open for "
        "files, the editor for project folders, Dolphin otherwise. The path "
        "must already exist."
    ),
    category="app-launch",
    params={
        "path": {
            "type": "string",
            "description": "Filesystem path to open.",
            "required": True,
        },
        "workspace": {"type": "integer", "description": "Workspace number."},
    },
)
@_adapt
async def _open_path(ctx, path=None, workspace=None, **extra):
    if not path or not str(path).strip():
        return {"error": "path is required"}
    try:
        return desktop.open_path(str(path), workspace=workspace, config=ctx.config)
    except (desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)


@TOOLS.register(
    "open_project",
    description=(
        "Open a project in the editor, matched by name against the projects "
        "root, or open an exact path directly. Returns an 'ambiguous' result "
        "when several projects match. To RUN/start a project use start_project "
        "instead."
    ),
    category="app-launch",
    params={
        "name_or_path": {
            "type": "string",
            "description": "Project name or exact path.",
            "required": True,
        },
        "workspace": {"type": "integer", "description": "Workspace number."},
    },
)
@_adapt
async def _open_project(ctx, name_or_path=None, workspace=None, **extra):
    if not name_or_path or not str(name_or_path).strip():
        return {"error": "name_or_path is required"}
    try:
        return desktop.open_project(
            str(name_or_path), workspace=workspace, silent=True, config=ctx.config
        )
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


@TOOLS.register(
    "open_url",
    description=(
        "Open a URL in the user's web browser, optionally on a specific "
        "workspace without stealing focus. Accepts a full URL (https://...) "
        "OR a site name like 'chatgpt', 'youtube', 'github', 'spotify' — "
        "site names are resolved to their real URL automatically. USE THIS "
        "whenever the user wants to visit or open a website — not web_search. "
        "This only opens a page in the browser — it does NOT play media or "
        "run anything. For YouTube videos, music or lofi streams use "
        "find_video + play_video instead."
    ),
    category="app-launch",
    params={
        "url": {
            "type": "string",
            "description": "Full URL or a site name (e.g. 'chatgpt', 'youtube').",
            "required": True,
        },
        "workspace": {"type": "integer", "description": "Workspace number."},
    },
)
@_adapt
async def _open_url(ctx, url=None, workspace=None, **extra):
    if not url or not str(url).strip():
        return {"error": "url is required"}
    url = str(url).strip()
    resolved = None
    if "://" not in url and "/" not in url:
        key = url.lower()
        known_sites = {
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
        if key in known_sites:
            resolved = known_sites[key]
        elif "." not in key and key.replace("-", "").isalpha():
            resolved = f"https://www.{key}.com"
        else:
            return {"error": "give a full URL (https://...)"}
    try:
        result = desktop.open_url(
            resolved or url, workspace=workspace, silent=True, config=ctx.config
        )
    except (desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)
    if isinstance(result, dict):
        result.setdefault("note", "opened URL in the browser (no media playback)")
        if resolved:
            result["note"] = f"opened {resolved} in the browser (no media playback)"
    return result


@TOOLS.register(
    "play_video",
    description=(
        "Actually start playing a video NOW: resolves a URL, YouTube id, or "
        "search query and opens the watch page in the browser on the given "
        "workspace (the default player is 'browser'). Only use player='mpv' "
        "when the user explicitly asks for mpv. Use this when the user wants "
        "to watch something — after find_video picks a candidate, call "
        "play_video with its URL or id. Optionally pass workspace to keep "
        "playback off the user's active workspace. Returns the resolved URL "
        "and how it was launched. If the user asked to watch something "
        "without naming an exact video, use the top find_video result and "
        "call play_video with its URL or id — do not ask the user which one. "
        "Also the correct tool for YouTube music, lofi streams and "
        "background audio. Never pass a made-up URL — use a URL/id from "
        "find_video results or plain search text."
    ),
    category="app-launch",
    params={
        "video": {
            "type": "string",
            "description": "Video URL, YouTube id, or search query.",
            "required": True,
        },
        "player": {
            "type": "string",
            "description": "'browser' (default) or 'mpv' (only when the user explicitly asks for mpv).",
        },
        "workspace": {"type": "integer", "description": "Workspace number."},
    },
)
@_adapt
async def _play_video(ctx, video=None, player=None, workspace=None, **extra):
    if not video or not str(video).strip():
        return {"error": "video is required"}
    try:
        return media.play_video(
            str(video), player=player or None, workspace=workspace,
            config=ctx.config,
        )
    except (media.MediaError, desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)


@TOOLS.register(
    "ani_play",
    description=(
        "Start playing an anime with ani-cli (quality 'best', streaming via "
        "mpv). select_nth picks the Nth search result when several match; "
        "episode pins a specific episode. Optionally lands on a workspace. "
        "Actually starts playback via ani-cli and mpv. When the user asks for the "
        "latest/next episode, call ani_next first and pass the episode number it "
        "returns."
    ),
    category="app-launch",
    params={
        "query": {
            "type": "string",
            "description": "Anime title to search for.",
            "required": True,
        },
        "episode": {"type": "integer", "description": "Episode number."},
        "select_nth": {
            "type": "integer",
            "default": 1,
            "description": "Which search result to play.",
        },
        "dub": {
            "type": "boolean",
            "default": False,
            "description": "Prefer the dubbed version.",
        },
        "workspace": {"type": "integer", "description": "Workspace number."},
    },
)
@_adapt
async def _ani_play(
    ctx, query=None, episode=None, select_nth=1, dub=False, workspace=None, **extra
):
    if not query or not str(query).strip():
        return {"error": "query is required"}
    try:
        return media.ani_play(
            str(query),
            episode=episode,
            select_nth=int(select_nth or 1),
            dub=bool(dub),
            workspace=workspace,
            config=ctx.config,
        )
    except (media.MediaError, desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)


# ------------------------------------------------------------ workspace

@TOOLS.register(
    "hypr_workspaces",
    description=(
        "List all Hyprland workspaces plus the currently active one. Fails "
        "with an error when the agent is not inside a Hyprland session."
    ),
    category="workspace",
    params={},
)
@_adapt
async def _hypr_workspaces(ctx, **extra):
    try:
        active = hypr.active_workspace()
    except hypr.HyprError:
        active = None
    try:
        workspaces = hypr.workspaces()
    except hypr.HyprError as exc:
        return _error(exc)
    return {"active": active, "workspaces": workspaces}


@TOOLS.register(
    "hypr_clients",
    description=(
        "List all open Hyprland windows with address, class and title. Fails "
        "with an error outside a Hyprland session."
    ),
    category="workspace",
    params={},
)
@_adapt
async def _hypr_clients(ctx, **extra):
    try:
        return {"clients": hypr.clients()}
    except hypr.HyprError as exc:
        return _error(exc)


@TOOLS.register(
    "organize_workspaces",
    description=(
        "Rearrange Hyprland workspaces per the user's layout: "
        "editor/terminal windows move to workspaces 1-2, everything else to "
        "3-5; the currently focused workspace is left alone; moves are "
        "silent and nothing is ever closed. Returns a summary of what moved."
    ),
    category="workspace",
    params={},
)
@_adapt
async def _organize_workspaces(ctx, **extra):
    try:
        return desktop.organize_workspaces(config=ctx.config)
    except (desktop.DesktopError, hypr.HyprError) as exc:
        return _error(exc)


@TOOLS.register(
    "move_window",
    description=(
        "Move a window (selector like 'address:0x...' or 'class:firefox') to "
        "a workspace without stealing focus."
    ),
    category="workspace",
    params={
        "target": {
            "type": "string",
            "description": "Window selector (address:... or class:...).",
            "required": True,
        },
        "workspace": {
            "type": "integer",
            "description": "Destination workspace number.",
            "required": True,
        },
        "silent": {
            "type": "boolean",
            "default": True,
            "description": "Keep focus where it is.",
        },
    },
)
@_adapt
async def _move_window(ctx, target=None, workspace=None, silent=True, **extra):
    if not target or workspace is None:
        return {"error": "target and workspace are required"}
    try:
        return desktop.move_window(str(target), workspace, silent=bool(silent))
    except hypr.HyprError as exc:
        return _error(exc)


# -------------------------------------------------------------- network

@TOOLS.register(
    "web_search",
    description=(
        "Search the web (DuckDuckGo) for information and return result "
        "titles, URLs and snippets. Use for research, facts and summaries "
        "— NOT for opening a site the user named (use open_url) and NOT "
        "for playing media (use find_video + play_video)."
    ),
    category="network",
    params={
        "query": {
            "type": "string",
            "description": "Search query.",
            "required": True,
        },
        "max_results": {
            "type": "integer",
            "default": 6,
            "description": "Maximum number of results.",
        },
    },
)
@_adapt
async def _web_search(ctx, query=None, max_results=6, **extra):
    if not query or not str(query).strip():
        return {"error": "query is required"}
    try:
        return web.search_duckduckgo(str(query), max_results=int(max_results or 6))
    except web.WebError as exc:
        return _error(exc)


@TOOLS.register(
    "fetch_page",
    description=(
        "Fetch a URL and return its title and extracted plain text (scripts, "
        "styles and navigation stripped). Truncated to max_chars."
    ),
    category="network",
    params={
        "url": {"type": "string", "description": "Page URL.", "required": True},
        "max_chars": {
            "type": "integer",
            "default": 8000,
            "description": "Maximum characters of text to return.",
        },
    },
)
@_adapt
async def _fetch_page(ctx, url=None, max_chars=8000, **extra):
    if not url or not str(url).strip():
        return {"error": "url is required"}
    try:
        return web.fetch_page(str(url), max_chars=int(max_chars or 8000))
    except web.WebError as exc:
        return _error(exc)


@TOOLS.register(
    "find_video",
    description=(
        "Search YouTube for videos matching a query and return candidates "
        "(id, title, URL, duration, channel) WITHOUT playing anything. Use "
        "this to find what to watch, then call play_video with the chosen "
        "URL/id. (uses the YouTube API when a key is configured in the "
        "config, otherwise yt-dlp metadata)"
    ),
    category="network",
    params={
        "query": {
            "type": "string",
            "description": "Search query.",
            "required": True,
        },
        "max_results": {
            "type": "integer",
            "default": 5,
            "description": "Maximum number of results.",
        },
    },
)
@_adapt
async def _find_video(ctx, query=None, max_results=5, **extra):
    if not query or not str(query).strip():
        return {"error": "query is required"}
    try:
        api_key = getattr(getattr(ctx.config, "web", None), "youtube_api_key", "") or ""
        return media.yt_search(
            str(query), max_results=int(max_results or 5), api_key=api_key
        )
    except media.MediaError as exc:
        return _error(exc)


@TOOLS.register(
    "ani_next",
    description=(
        "Report the next unwatched episode for an anime from the ani-cli "
        "watch history. Without a query, returns the whole history."
    ),
    category="info",
    params={
        "query": {
            "type": "string",
            "description": "Title substring to look up (omit for full history).",
        }
    },
)
@_adapt
async def _ani_next(ctx, query=None, **extra):
    try:
        return media.ani_next(query=str(query) if query else None, config=ctx.config)
    except media.MediaError as exc:
        return _error(exc)


# -------------------------------------------------------------- memory

@TOOLS.register(
    "memory_add",
    description=(
        "Store a durable fact about the user, their projects or preferences "
        "in local memory. Use for preferences, recurring decisions, dislikes."
    ),
    category="memory",
    params={
        "fact": {
            "type": "string",
            "description": "The fact to remember.",
            "required": True,
        },
        "kind": {
            "type": "string",
            "default": "fact",
            "description": "Category: fact, preference, project, ...",
        },
        "confidence": {
            "type": "number",
            "default": 1.0,
            "description": "How sure you are (0-1).",
        },
    },
)
@_adapt
async def _memory_add(ctx, fact=None, kind="fact", confidence=1.0, **extra):
    if not fact or not str(fact).strip():
        return {"error": "fact is required"}
    return await memory_tools.memory_add(
        ctx, str(fact), kind=str(kind or "fact"), confidence=float(confidence or 1.0)
    )


@TOOLS.register(
    "memory_recall",
    description=(
        "Search durable memory for facts related to a query. Recall before "
        "answering questions about user preferences, projects or past asks."
    ),
    category="memory",
    params={
        "query": {
            "type": "string",
            "description": "What to search for.",
            "required": True,
        },
        "k": {
            "type": "integer",
            "default": 5,
            "description": "Maximum number of matches.",
        },
    },
)
@_adapt
async def _memory_recall(ctx, query=None, k=5, **extra):
    if not query or not str(query).strip():
        return {"error": "query is required"}
    return await memory_tools.memory_recall(ctx, str(query), k=int(k or 5))


@TOOLS.register(
    "memory_list",
    description=(
        "List stored facts, newest first, optionally filtered by kind "
        "(e.g. 'preference')."
    ),
    category="memory",
    params={
        "limit": {
            "type": "integer",
            "default": 20,
            "description": "Maximum number of facts.",
        },
        "kind": {"type": "string", "description": "Only this kind of fact."},
    },
)
@_adapt
async def _memory_list(ctx, limit=20, kind=None, **extra):
    return await memory_tools.memory_list(
        ctx, limit=int(limit or 20), kind=str(kind) if kind else None
    )


@TOOLS.register(
    "memory_delete",
    description=(
        "Delete a stored fact by id or by text match. Deleting memories "
        "requires the user's approval."
    ),
    category="memory",
    params={
        "fact_id": {"type": "integer", "description": "Fact id to delete."},
        "query": {
            "type": "string",
            "description": "Text match — deletes facts containing it.",
        },
    },
)
@_adapt
async def _memory_delete(ctx, fact_id=None, query=None, **extra):
    return await memory_tools.memory_delete(
        ctx, fact_id=fact_id, query=str(query) if query else None
    )


# --------------------------------------------------------------- tasks

@TOOLS.register(
    "todo_add",
    description="Add an open todo to the persistent todo list.",
    category="tasks",
    params={
        "title": {
            "type": "string",
            "description": "Todo text.",
            "required": True,
        }
    },
)
@_adapt
async def _todo_add(ctx, title=None, **extra):
    return await tasks.todo_add(ctx, str(title or ""))


@TOOLS.register(
    "todo_list",
    description="List todos (open ones by default; include_done shows all).",
    category="tasks",
    params={
        "include_done": {
            "type": "boolean",
            "default": False,
            "description": "Also show completed todos.",
        }
    },
)
@_adapt
async def _todo_list(ctx, include_done=False, **extra):
    return await tasks.todo_list(ctx, include_done=bool(include_done))


@TOOLS.register(
    "todo_toggle",
    description=(
        "Toggle (or explicitly set) the done state of a todo, found by id or "
        "by a text match on its title."
    ),
    category="tasks",
    params={
        "todo_id": {"type": "integer", "description": "Todo id."},
        "query": {"type": "string", "description": "Title text to match."},
        "done": {"type": "boolean", "description": "Force done/not done."},
    },
)
@_adapt
async def _todo_toggle(ctx, todo_id=None, query=None, done=None, **extra):
    return await tasks.todo_toggle(
        ctx, todo_id=todo_id, query=str(query) if query else None, done=done
    )


@TOOLS.register(
    "reminder_add",
    description=(
        "Schedule a reminder. 'when' accepts expressions like 'in 30m', "
        "'in 1h30m', 'at 18:30', 'tomorrow' or 'tomorrow 9am'."
    ),
    category="tasks",
    params={
        "message": {
            "type": "string",
            "description": "Reminder text.",
            "required": True,
        },
        "when": {
            "type": "string",
            "description": "Human time expression.",
            "required": True,
        },
    },
)
@_adapt
async def _reminder_add(ctx, message=None, when=None, **extra):
    return await tasks.reminder_add(ctx, str(message or ""), str(when or ""))


@TOOLS.register(
    "reminder_list",
    description="List scheduled reminders, soonest first.",
    category="tasks",
    params={},
)
@_adapt
async def _reminder_list(ctx, **extra):
    return await tasks.reminder_list(ctx)


@TOOLS.register(
    "reminder_cancel",
    description="Cancel a scheduled reminder by id.",
    category="tasks",
    params={
        "reminder_id": {
            "type": "integer",
            "description": "Reminder id.",
            "required": True,
        }
    },
)
@_adapt
async def _reminder_cancel(ctx, reminder_id=None, **extra):
    if reminder_id is None:
        return {"error": "reminder_id is required"}
    return await tasks.reminder_cancel(ctx, int(reminder_id))


@TOOLS.register(
    "timer_set",
    description=(
        "Start a countdown timer (this machine has no stopwatch app — this IS "
        "the stopwatch). Accepts '20m', '20 minutes', '2h', '1h30m', '90s', "
        "'1:30', or 'in 20m'. You'll be notified when it ends; the user can "
        "ask for the remaining time anytime."
    ),
    category="tasks",
    params={
        "label": {
            "type": "string",
            "default": "timer",
            "description": "Timer label.",
        },
        "duration": {
            "type": "string",
            "description": "Countdown duration (e.g. '20m', '1h30m', '90s').",
            "required": True,
        },
    },
)
@_adapt
async def _timer_set(ctx, label="timer", duration=None, **extra):
    if duration is None:
        return {"error": "duration is required"}
    return await tasks.timer_set(ctx, label=str(label or "timer"), duration=str(duration))


@TOOLS.register(
    "timer_status",
    description="Report remaining time on all running timers (the stopwatch readout).",
    category="tasks",
    params={},
)
@_adapt
async def _timer_status(ctx, **extra):
    return await tasks.timer_status(ctx)


@TOOLS.register(
    "timer_cancel",
    description="Cancel a timer by id, or omit to cancel the only running timer.",
    category="tasks",
    params={
        "timer_id": {
            "type": "integer",
            "description": "Timer id to cancel; omit to cancel the only running timer.",
        }
    },
)
@_adapt
async def _timer_cancel(ctx, timer_id=None, **extra):
    if timer_id is None:
        return await tasks.timer_cancel(ctx)
    return await tasks.timer_cancel(ctx, int(timer_id))


# ---------------------------------------------------------------- info

@TOOLS.register(
    "notify_user",
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
async def _notify_user(ctx, message=None, **extra):
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
