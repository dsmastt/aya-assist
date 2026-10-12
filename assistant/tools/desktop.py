"""Desktop integration: app launching, window placement, project opening.

Built on top of the Hyprland IPC helpers in :mod:`assistant.tools.hypr`.
All functions are plain, return JSON-serializable dicts, and raise
:class:`DesktopError` (or propagate :class:`hypr.HyprError`) on failure.

Focus is never stolen: launching always goes through ``exec`` (with an
inline window rule when a workspace is requested) and windows are moved with
``movetoworkspacesilent``; the bare ``workspace`` dispatcher is only used by
:func:`focus_workspace`, which is called sparingly.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import time
import tomllib
import urllib.parse
from pathlib import Path
from typing import Any

from . import hypr

_TOKEN_SPLIT = re.compile(r"[-_ .]+")

_ZEN_FALLBACK = "/opt/zen-browser-bin/zen-bin"

_NEW_WINDOW_FLAGS = {
    "codium": "-n",
    "code": "-n",
    "vscode": "-n",
    "zen": "--new-window",
    "zen-bin": "--new-window",
    "firefox": "--new-window",
}

APP_WINDOW_CLASSES: dict[str, list[str]] = {
    "codium": ["codium", "vscodium"],
    "code": ["code", "vscodium"],
    "vscode": ["vscodium", "code"],
    "zen": ["zen", "zen-bin"],
    "zen-bin": ["zen", "zen-bin"],
    "browser": ["zen", "zen-bin"],
    "firefox": ["firefox", "navigator"],
    "foot": ["foot"],
    "kitty": ["kitty"],
    "mpv": ["mpv"],
    "vlc": ["vlc"],
    "thunar": ["thunar"],
    "dolphin": ["dolphin"],
    "spotify": ["spotify"],
    "discord": ["discord"],
    "steam": ["steam"],
}

_BACKUP_PROJECTS_ROOT = "/mnt/Data/bkp/app"


class DesktopError(RuntimeError):
    """Raised when a desktop action (launch, open, move) cannot be performed."""


APP_LAUNCHERS: dict[str, list[str]] = {
    "codium": ["codium"],
    "code": ["code"],
    "vscode": ["codium"],
    "firefox": ["firefox"],
    "zen": ["zen-bin", "zen"],
    "browser": ["zen-bin", "zen"],
    "foot": ["foot"],
    "kitty": ["kitty"],
    "thunar": ["thunar"],
    "dolphin": ["dolphin"],
    "mpv": ["mpv"],
    "vlc": ["vlc"],
    "btop": ["btop"],
    "htop": ["htop"],
    "nvtop": ["nvtop"],
    "discord": ["discord"],
    "spotify": ["spotify"],
    "qbittorrent": ["qbittorrent"],
    "steam": ["steam"],
    "ani-cli": ["ani-cli"],
    "editor": ["codium"],
    "terminal": ["foot"],
    "xdg-open": ["xdg-open"],
}

_resolve_cache: dict[str, str] = {}


def _config_value(config: Any, section: str, key: str, default: Any) -> Any:
    """Read ``config.<section>.<key>`` with graceful fallback to ``default``."""
    if config is None:
        return default
    sec = getattr(config, section, None)
    if sec is None:
        return default
    value = getattr(sec, key, None)
    return value if value is not None else default


def resolve_binary(app: str) -> str:
    """Resolve an app name to an absolute path on PATH (with caching).

    The lookup is case-insensitive and uses the :data:`APP_LAUNCHERS`
    candidate table; unknown names fall back to the name itself. Zen browser
    is additionally probed at its default Arch install location since its
    binary usually lives outside PATH.

    Raises :class:`DesktopError` when no candidate is found.
    """
    key = app.lower()
    cached = _resolve_cache.get(key)
    if cached:
        return cached
    candidates = list(APP_LAUNCHERS.get(key, [app]))
    if key in ("zen", "browser") and os.access(_ZEN_FALLBACK, os.X_OK):
        candidates.append(_ZEN_FALLBACK)
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            _resolve_cache[key] = found
            return found
    raise DesktopError(f"unknown app: {app!r} (none of {candidates!r} on PATH)")


def _find_new_client(
    app: str, hint: str | None = None, timeout: float = 12.0
) -> str | None:
    """Poll Hyprland clients for a newly appeared window of ``app``.

    ``app`` is an app name (or binary path); the class needles come from
    :data:`APP_WINDOW_CLASSES`, falling back to the base name. A new client
    matches when any needle is a substring of its class or title (casefold)
    — the binary name itself is not used, because window classes often
    differ from the binary (e.g. ``zen-bin`` windows report class ``zen``).

    When ``hint`` is given, a new window whose title contains the hint is
    preferred; the return value is the first hint-match, else the first
    class-match, else ``None`` when nothing shows up within ``timeout``
    seconds.
    """
    key = Path(app).name.casefold()
    needles = APP_WINDOW_CLASSES.get(key, [key])
    hint_cf = hint.casefold() if hint else None
    try:
        before = {c.get("address") for c in hypr.clients()}
    except hypr.HyprError:
        before = set()
    deadline = time.monotonic() + timeout
    class_match: str | None = None
    while time.monotonic() < deadline:
        try:
            for client in hypr.clients():
                address = client.get("address")
                if not address or address in before:
                    continue
                cls = (client.get("class") or "").casefold()
                title = (client.get("title") or "").casefold()
                if not any(n in cls or n in title for n in needles):
                    continue
                if hint_cf:
                    if hint_cf in title:
                        return address
                    if class_match is None:
                        class_match = address
                else:
                    return address
        except hypr.HyprError:
            pass
        time.sleep(0.5)
    return class_match


def _existing_window_on_workspace(app: str, workspace: int) -> str | None:
    """Address of an existing window of ``app`` already on ``workspace``.

    Class needles come from :data:`APP_WINDOW_CLASSES` (same matching rules
    as :func:`_find_new_client`). Returns ``None`` when the IPC is
    unreachable or no matching window sits on the target workspace.
    """
    key = Path(app).name.casefold()
    needles = APP_WINDOW_CLASSES.get(key, [key])
    try:
        clients = hypr.clients()
    except hypr.HyprError:
        return None
    for client in clients:
        if not isinstance(client, dict):
            continue
        cls = (client.get("class") or "").casefold()
        title = (client.get("title") or "").casefold()
        ws = client.get("workspace")
        ws_id = ws.get("id") if isinstance(ws, dict) else None
        if ws_id != workspace:
            continue
        if any(n in cls or n in title for n in needles):
            return client.get("address")
    return None


_lua_config: bool | None = None


def _uses_lua_config() -> bool:
    """Cache whether the running Hyprland config provider is Lua (0.56+)."""
    global _lua_config
    if _lua_config is None:
        try:
            _lua_config = hypr.hyprctl_json("status").get("configProvider") == "lua"
        except hypr.HyprError:
            _lua_config = False
    return _lua_config


def _dispatch(args: str) -> str:
    """Send a ``dispatch`` request, translating the legacy ``exec``,
    ``movetoworkspacesilent`` and ``movetoworkspace`` dispatchers to
    ``hl.dsp`` Lua calls when the running Hyprland config is Lua (0.56+)."""
    if not _uses_lua_config():
        return hypr.dispatch(args)
    if args.startswith("exec "):
        payload = args[len("exec "):]
        if len(payload) >= 2 and payload.startswith('"') and payload.endswith('"'):
            payload = payload[1:-1]
        escaped = payload.replace("\\", "\\\\").replace('"', '\\"')
        return hypr.dispatch(f'hl.dsp.exec_cmd("{escaped}")')
    for verb in ("movetoworkspacesilent ", "movetoworkspace "):
        if args.startswith(verb):
            workspace_part, target = args[len(verb):].split(",", 1)
            selector = target if ":" in target else f"address:{target}"
            return hypr.dispatch(
                f'hl.dsp.window.move({{window = "{selector}", workspace = "{workspace_part}", follow = false}})'
            )
    return hypr.dispatch(args)


def launch_app(
    app: str,
    args: str = "",
    workspace: int | None = None,
    silent: bool = True,
    config: Any = None,
    hint: str | None = None,
) -> dict:
    """Launch ``app`` via the Hyprland ``exec`` dispatcher.

    ``args`` is appended to the resolved binary verbatim (quote it yourself
    with :func:`shlex.quote` when needed).

    When ``workspace`` is given, an inline window rule
    (``exec "[workspace <workspace> silent] <cmd>"``) is attempted first so the new
    window opens on that workspace without ever stealing focus. If the
    windowrule form is rejected or the IPC is unavailable, the app is
    launched plainly and its new window is located via :func:`hypr.clients`
    polling and moved with ``movetoworkspacesilent``.

    Single-instance apps (codium, zen, firefox, ...) get their new-window
    flag prepended when a workspace is requested, and are launched with a
    plain ``exec`` followed by locating and moving the new window with
    ``movetoworkspacesilent`` — inline window rules never place windows
    created by the running instance. ``hint`` (a title substring) helps
    pick the right new window when several appear at once.

    The bare ``workspace`` dispatcher is never used here — focus is not
    touched. Returns launch metadata including the method actually used and
    an ``ok`` field: ``True`` when the app actually launched, ``False`` when
    the dispatch failed (with a ``note`` explaining what did happen).
    """
    binary = resolve_binary(app)
    normalized = app.strip().casefold()
    flag_key = "zen" if normalized == "browser" else normalized
    single_instance = flag_key in _NEW_WINDOW_FLAGS
    if workspace is not None:
        flag = _NEW_WINDOW_FLAGS.get(flag_key)
        if flag and flag not in args.split():
            args = f"{flag} {args}".strip()
    command = binary + (f" {args}" if args else "")

    def _result(method, reply, note, ok, command_used):
        return {
            "app": app,
            "binary": binary,
            "command": command_used,
            "workspace": workspace,
            "silent": silent,
            "hint": hint,
            "method": method,
            "note": note,
            "reply": reply,
            "ok": ok,
        }

    if workspace is None:
        try:
            reply = _dispatch(f"exec {command}")
        except hypr.HyprError as exc:
            return _result("exec", "", f"launch failed: {exc}", False, command)
        lowered = reply.lower()
        if "error" in lowered or "invalid" in lowered:
            return _result(
                "exec", reply, f"dispatch rejected the launch: {reply!r}",
                False, command,
            )
        return _result("exec", reply, None, True, command)

    if single_instance:
        # Inline window rules only match windows spawned by the exec'd
        # process; a single-instance app hands off to its running instance,
        # so launch plainly and move the new window once it appears.
        try:
            exec_reply = _dispatch(f"exec {command}")
        except hypr.HyprError as exc:
            return _result("exec", "", f"launch failed: {exc}", False, command)
        exec_lowered = exec_reply.lower()
        if "error" in exec_lowered or "invalid" in exec_lowered:
            return _result(
                "exec", exec_reply,
                f"dispatch rejected the launch: {exec_reply!r}",
                False, command,
            )
        address = _find_new_client(app, hint)
        if not address:
            existing = _existing_window_on_workspace(app, workspace)
            if existing:
                return _result(
                    "exec+tab", exec_reply,
                    f"opened as a tab in the existing browser window already "
                    f"on workspace {workspace}",
                    True, command,
                )
            return _result(
                "exec+move", exec_reply,
                f"launched, but no window could be placed on workspace "
                f"{workspace} — the new window could not be located "
                "(it may have opened inside the existing instance's window)",
                False, command,
            )
        move_reply = _dispatch(
            f"movetoworkspacesilent {workspace},address:{address}"
        )
        return _result(
            "exec+move", move_reply,
            f"new window moved to workspace {workspace} ({address})",
            True, command,
        )

    rule = f"workspace {workspace}" + (" silent" if silent else "")
    windowrule_command = f'exec "[{rule}] {command}"'
    failure_reason = ""
    try:
        reply = _dispatch(windowrule_command)
    except hypr.HyprError as exc:
        reply = ""
        failure_reason = str(exc)
    lowered = reply.lower()
    if not failure_reason and "error" not in lowered and "invalid" not in lowered:
        return _result("windowrule", reply, None, True, windowrule_command)

    # Fallback: launch normally, then move the new window to the workspace.
    reason = failure_reason or reply or "windowrule rejected"
    try:
        exec_reply = _dispatch(f"exec {command}")
    except hypr.HyprError as exc:
        return _result(
            "exec", "",
            f"launch failed: inline rule unavailable ({reason}) and plain exec also failed ({exc})",
            False, command,
        )
    exec_lowered = exec_reply.lower()
    if "error" in exec_lowered or "invalid" in exec_lowered:
        return _result(
            "exec", exec_reply,
            f"launch failed: inline rule unavailable ({reason}) and plain exec replied: {exec_reply!r}",
            False, command,
        )
    address = _find_new_client(app, hint, timeout=5.0)
    if address:
        move_reply = _dispatch(f"movetoworkspacesilent {workspace},address:{address}")
        note = (
            f"launched via exec and moved to workspace {workspace} ({address})"
            f" — inline rule unavailable ({reason})"
        )
        if move_reply and "error" in str(move_reply).lower():
            note = (
                f"launched via exec (inline rule unavailable: {reason}); "
                f"moving to workspace {workspace} failed: {move_reply!r}"
            )
        return _result("exec+move", move_reply, note, True, command)
    return _result(
        "exec+move", exec_reply,
        "app launched (inline workspace rule unavailable; window could not be located to move)",
        True, command,
    )


def _url_hint(url: str) -> str | None:
    """Derive a window-title hint from a URL.

    YouTube hosts yield "youtube"; otherwise the longest dot-label of the
    netloc that is at least 4 characters long (e.g. ``chat.openai.com``
    yields "openai"), or ``None`` when the URL has no usable hostname.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com"):
        return "youtube"
    labels = [label for label in host.split(".") if len(label) >= 4]
    return max(labels, key=len) if labels else None


def open_url(
    url: str, workspace: int | None = None, silent: bool = True, config: Any = None
) -> dict:
    """Open ``url`` in the configured browser (default ``zen``).

    The URL-derived ``hint`` helps :func:`launch_app` locate the right new
    browser window among several that may appear at once.
    """
    browser = _config_value(config, "tools", "browser", "zen")
    hint = _url_hint(url)
    result = launch_app(
        browser, shlex.quote(url), workspace=workspace, silent=silent,
        config=config, hint=hint,
    )
    result["url"] = url
    result["hint"] = hint
    return result


def open_path(path: str, workspace: int | None = None, config: Any = None) -> dict:
    """Open a file or directory with the right application.

    Files go through ``xdg-open``; directories containing ``.git/``, a
    ``*.code-workspace`` file, or ``package.json`` open in the editor;
    everything else opens in Dolphin. Raises :class:`DesktopError` when the
    path does not exist.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise DesktopError(f"path does not exist: {path!r}")
    if p.is_file():
        result = launch_app("xdg-open", shlex.quote(str(p)), workspace=workspace, config=config)
        return {"method": "xdg-open", "path": str(p), **result}
    is_project = (
        (p / ".git").is_dir()
        or any(p.glob("*.code-workspace"))
        or (p / "package.json").is_file()
        or (p / "pyproject.toml").is_file()
        or (p / "Cargo.toml").is_file()
        or (p / "requirements.txt").is_file()
        or (p / "go.mod").is_file()
    )
    if is_project:
        editor = _config_value(config, "tools", "editor", "codium")
        result = launch_app(
            editor, shlex.quote(str(p)), workspace=workspace, config=config,
            hint=p.name,
        )
        return {"method": "editor", "path": str(p), **result}
    result = launch_app(
        "dolphin", shlex.quote(str(p)), workspace=workspace, config=config,
        hint=p.name,
    )
    return {"method": "file-manager", "path": str(p), **result}


def _project_roots(config: Any) -> list[Path]:
    """Project search roots: ``config.paths.projects_root`` first, plus the
    backup root if it exists (deduped, expanduser'd, existing dirs only)."""
    primary = _config_value(config, "paths", "projects_root", "/mnt/Data/app")
    roots: list[Path] = []
    for candidate in (primary, _BACKUP_PROJECTS_ROOT):
        p = Path(str(candidate)).expanduser()
        if p.is_dir() and p not in roots:
            roots.append(p)
    return roots


def _score_name(query: str, name: str) -> int:
    """Fuzzy score between a query and a project directory name.

    2 per token that matches exactly, a weighted score per token pair sharing
    a common prefix of length >= 3 (either direction): max(1, length - 2), so
    longer shared prefixes rank higher ("note" -> 2, "not" -> 1). One full
    name being a substring of the other (length >= 3) adds 1. A score >= 1
    counts as a match.
    """
    q_tokens = [t for t in _TOKEN_SPLIT.split(query.casefold()) if t]
    n_tokens = [t for t in _TOKEN_SPLIT.split(name.casefold()) if t]
    score = 0
    for qt in q_tokens:
        for nt in n_tokens:
            if qt == nt:
                score += 2
    for qt in q_tokens:
        for nt in n_tokens:
            if qt == nt:
                continue
            common = os.path.commonprefix([qt, nt])
            if len(common) >= 3:
                score += max(1, len(common) - 2)
    qf = query.casefold()
    nf = name.casefold()
    if len(qf) >= 3 and len(nf) >= 3 and (qf in nf or nf in qf):
        score += 1
    return score


def _manifest_names(entry: Path) -> list[str]:
    """Project names declared in ``entry``'s package.json / pyproject.toml.

    package.json is read with json.loads; pyproject.toml with tomllib under
    ``[project]``. Missing files, unparsable content, and missing/invalid
    ``name`` fields all contribute nothing (empty list).
    """
    names: list[str] = []
    pkg = entry / "package.json"
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict) and isinstance(data.get("name"), str):
                names.append(data["name"].strip())
        except (OSError, ValueError):
            pass
    pyproject = entry / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(
                pyproject.read_text(encoding="utf-8", errors="replace")
            )
            project = data.get("project")
            if isinstance(project, dict) and isinstance(project.get("name"), str):
                names.append(project["name"].strip())
        except (OSError, ValueError, TypeError):
            pass
    return [n for n in names if n]


def _find_project_match(name_or_path: str, config: Any) -> tuple[list[str], list[str]]:
    """Score project dirs across all :func:`_project_roots` against a name.

    A directory's declared manifest name (package.json / pyproject.toml) adds
    its own ``_score_name`` on top of the directory-name score, so a package
    named e.g. "notemind" breaks a directory-name tie. Returns ``(best,
    visible)`` where ``best`` holds the top-scoring matches (possibly empty)
    and ``visible`` every directory name seen.
    """
    best_score = 0
    best: list[str] = []
    visible: list[str] = []
    for root_path in _project_roots(config):
        entries = sorted(e for e in root_path.iterdir() if e.is_dir())
        for entry in entries:
            visible.append(entry.name)
            score = _score_name(name_or_path, entry.name)
            if score > 0 and any(
                (entry / marker).exists()
                for marker in (
                    "README.md",
                    "readme.md",
                    "README",
                    "package.json",
                    "pyproject.toml",
                    "Cargo.toml",
                    ".git",
                )
            ):
                score += 3
            for manifest_name in _manifest_names(entry):
                score += _score_name(name_or_path, manifest_name)
            if score > best_score:
                best_score = score
                best = [str(entry)]
            elif score > 0 and score == best_score:
                best.append(str(entry))
    return best, visible


def open_project(
    name_or_path: str,
    workspace: int | None = None,
    silent: bool = True,
    config: Any = None,
) -> dict:
    """Open a project in the editor.

    If ``name_or_path`` is an existing path it is opened directly. Otherwise
    the top-level directories under the project roots (``config.paths.
    projects_root`` plus the backup root) are scored by fuzzy token match
    (casefold, split on ``- _ .`` and spaces): a single strong match opens in
    the editor, no match raises :class:`DesktopError` (with the visible
    directory list), and a tie returns an ``ambiguous`` result so the caller
    can disambiguate. To RUN a project rather than open it, use
    :func:`start_project`.
    """
    p = Path(name_or_path).expanduser()
    if p.exists():
        editor = _config_value(config, "tools", "editor", "codium")
        result = launch_app(
            editor, shlex.quote(str(p)), workspace=workspace, silent=silent,
            config=config, hint=p.name,
        )
        return {"status": "found", "method": "editor", "path": str(p), **result}

    best, visible = _find_project_match(name_or_path, config)
    if not best:
        roots = ", ".join(str(r) for r in _project_roots(config))
        shown = ", ".join(visible) or "(no directories)"
        raise DesktopError(
            f"no project matching {name_or_path!r} under {roots}; "
            f"visible projects: {shown}"
        )
    if len(best) == 1:
        editor = _config_value(config, "tools", "editor", "codium")
        result = launch_app(
            editor, shlex.quote(best[0]), workspace=workspace, silent=silent,
            config=config, hint=Path(best[0]).name,
        )
        return {"status": "found", "method": "editor", "path": best[0], **result}
    return {
        "status": "ambiguous",
        "candidates": best,
        "note": "multiple projects match; pass an exact path or a more specific name",
    }


_PRIMARY_CLASSES = (
    "codium",
    "code",
    "vscode",
    "vscodium",
    "zed",
    "foot",
    "kitty",
    "alacritty",
    "xterm",
    "ghostty",
    "wezterm",
)


def organize_workspaces(config: Any = None) -> dict:
    """Rearrange windows onto the user's workspace layout.

    Editor/terminal windows (class matching :data:`_PRIMARY_CLASSES`) move
    round-robin to workspaces 1 and 2; everything else moves round-robin to
    workspaces 3, 4 and 5. Windows on the currently focused workspace are
    left untouched, and all moves use ``movetoworkspacesilent`` — nothing is
    ever closed and focus is never stolen. Returns a summary of what moved.

    Raises :class:`hypr.HyprError` when the Hyprland IPC is unreachable.
    """
    clients = hypr.clients()
    active = hypr.active_workspace()
    focused_ws = active.get("id") if isinstance(active, dict) else None

    primary_ws = (1, 2)
    secondary_ws = (3, 4, 5)
    primary_i = 0
    secondary_i = 0
    moves: list[dict] = []
    kept: list[dict] = []

    for client in clients:
        if not isinstance(client, dict):
            continue
        cls = client.get("class")
        title = client.get("title") or ""
        address = client.get("address")
        ws = client.get("workspace")
        from_ws = ws.get("id") if isinstance(ws, dict) else None

        if address is None or from_ws is None or from_ws == focused_ws:
            kept.append(
                {"class": cls, "title": title, "workspace": from_ws}
            )
            continue

        is_primary = any(
            pattern in (cls or "").casefold() for pattern in _PRIMARY_CLASSES
        )
        if is_primary:
            to_ws = primary_ws[primary_i % len(primary_ws)]
            primary_i += 1
        else:
            to_ws = secondary_ws[secondary_i % len(secondary_ws)]
            secondary_i += 1

        if to_ws == from_ws:
            kept.append(
                {"class": cls, "title": title, "workspace": from_ws}
            )
            continue

        reply = _dispatch(f"movetoworkspacesilent {to_ws},address:{address}")
        if reply and "error" in str(reply).lower():
            raise hypr.HyprError(reply)
        moves.append(
            {"class": cls, "title": title, "from_ws": from_ws, "to_ws": to_ws}
        )

    return {
        "moves": moves,
        "kept": kept,
        "note": "no windows were closed",
    }


def move_window(target: str, workspace: int | str, silent: bool = True) -> dict:
    """Move a window to a workspace without (by default) focusing it.

    ``target`` is a window selector such as ``address:0x...`` or
    ``class:firefox``.
    """
    verb = "movetoworkspacesilent" if silent else "movetoworkspace"
    reply = _dispatch(f"{verb} {workspace},{target}")
    if reply and "error" in str(reply).lower():
        raise hypr.HyprError(reply)
    return {"target": target, "workspace": workspace, "silent": silent, "reply": reply}


def focus_workspace(workspace: int | str) -> dict:
    """Switch focus to a workspace. Use sparingly — it does steal focus."""
    reply = hypr.dispatch(f"workspace {workspace}")
    return {"workspace": workspace, "reply": reply}

# ----------------------------------------------------------- project start

# (pattern, priority): 0 dev, 1 start, 2 serve/watch, 3 run, 4 other
_RUN_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"npm run dev\b"), 0),
    (re.compile(r"yarn dev\b"), 0),
    (re.compile(r"pnpm dev\b"), 0),
    (re.compile(r"bun(?: run)? dev\b"), 0),
    (re.compile(r"next dev\b"), 0),
    (re.compile(r"make dev\b"), 0),
    (re.compile(r"npm (?:run )?start\b"), 1),
    (re.compile(r"yarn start\b"), 1),
    (re.compile(r"pnpm start\b"), 1),
    (re.compile(r"npm run (?:serve|watch)\b"), 2),
    (re.compile(r"\bvite\b"), 2),
    (re.compile(r"cargo run\b"), 3),
    (re.compile(r"uv run\b"), 3),
    (re.compile(r"make run\b"), 3),
    (re.compile(r"\./run\.sh\b"), 3),
    (re.compile(r"flask run\b"), 3),
    (re.compile(r"\buvicorn\b"), 4),
    (re.compile(r"python3?\s+(main\.py|app\.py|manage\.py)\b"), 4),
    (re.compile(r"docker compose up\b"), 4),
]

_GENERIC_RUN = re.compile(
    r"^\s*(npm|yarn|pnpm|bun|cargo|uv|make)\s+[a-z-]*(dev|start|serve|run)\b"
)
_GENERIC_PRIORITY = {"dev": 0, "start": 1, "serve": 2, "run": 3}


def extract_run_commands(readme_text: str) -> list[str]:
    """Extract run commands from README text.

    Scans line by line; matches inside fenced code blocks are preferred over
    prose mentions. Known command shapes (npm/yarn/pnpm/bun dev scripts,
    cargo/uv/make run, uvicorn, flask, python entry points, docker compose,
    vite, ...) are matched anywhere, plus a generic dev/start/serve/run
    fallback inside code blocks only. Returns a unique list ordered by
    priority: dev > start > serve/watch > run > other.
    """
    found: list[tuple[int, int, str]] = []
    in_code = False
    for line in (readme_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            continue
        if not stripped:
            continue
        code_rank = 0 if in_code else 1
        matched = False
        for pattern, priority in _RUN_PATTERNS:
            m = pattern.search(line)
            if m:
                found.append((priority, code_rank, m.group(0).strip()))
                matched = True
                break
        if matched or not in_code:
            continue
        m = _GENERIC_RUN.match(line)
        if m:
            priority = _GENERIC_PRIORITY[m.group(2)]
            found.append((priority, code_rank, m.group(0).strip()))

    seen: set[str] = set()
    commands: list[str] = []
    for _, _, raw in sorted(found, key=lambda item: (item[0], item[1])):
        cmd = raw.strip().strip("`\"'")
        if not cmd:
            continue
        if cmd not in seen:
            seen.add(cmd)
            commands.append(cmd)
    return commands


def _find_readme(project_dir: Path) -> Path | None:
    """Locate a README file (case-insensitive) in ``project_dir``."""
    if not project_dir.is_dir():
        return None
    candidates = sorted(
        e for e in project_dir.iterdir()
        if e.is_file() and e.name.casefold().startswith("readme")
    )
    return candidates[0] if candidates else None


def start_project(
    name_or_path: str,
    workspace: int | None = None,
    config: Any = None,
    approved: bool = False,
) -> dict:
    """Find a project by name/path, read its README, and run its command.

    Resolves ``name_or_path`` as an existing directory or by fuzzy match
    against the project roots; reads the README (case-insensitive) and picks
    the first candidate from :func:`extract_run_commands`. The command is
    safety-classified: DENY is refused, APPROVE needs ``approved=True`` (or
    user approval via the caller), and anything else launches in a terminal
    on ``workspace`` (default: the agent workspace).
    """
    from . import safety

    p = Path(name_or_path).expanduser()
    if p.is_dir():
        project_dir = p
    else:
        best, _visible = _find_project_match(name_or_path, config)
        if not best:
            roots = ", ".join(str(r) for r in _project_roots(config))
            raise DesktopError(
                f"no project matching {name_or_path!r} under {roots}"
            )
        if len(best) > 1:
            return {
                "status": "ambiguous",
                "candidates": best,
                "note": "multiple projects match; pass an exact path or a more specific name",
            }
        project_dir = Path(best[0])

    readme = _find_readme(project_dir)
    if readme is None:
        return {
            "status": "no-command",
            "project": str(project_dir),
            "note": "couldn't find a run command in README; open it with open_project or tell me the command",
        }
    commands = extract_run_commands(readme.read_text(errors="replace"))
    if not commands:
        return {
            "status": "no-command",
            "project": str(project_dir),
            "note": "couldn't find a run command in README; open it with open_project or tell me the command",
        }
    command = commands[0]

    verdict = safety.classify(command)
    if verdict.action == safety.Action.DENY:
        return {"status": "refused", "command": command, "reasons": verdict.reasons}
    if verdict.action == safety.Action.APPROVE and not approved:
        return {
            "status": "needs-approval",
            "command": command,
            "project": str(project_dir),
            "reasons": verdict.reasons,
        }

    terminal = _config_value(config, "tools", "terminal", "foot")
    if workspace is None:
        workspace = int(_config_value(config, "tools", "agent_workspace", 4))
    full = f"cd {shlex.quote(str(project_dir))} && {command}"
    args = f"sh -c {shlex.quote(full)}"
    launch = launch_app(terminal, args, workspace=workspace, silent=True, config=config)
    return {
        "status": "started",
        "project": str(project_dir),
        "command": command,
        "workspace": workspace,
        "terminal": terminal,
        "launch": launch,
    }
