"""Media tooling: video search/playback (mpv, yt-dlp) and ani-cli history.

Stdlib only. yt-dlp may live only in the project venv; helper functions
prepend its directory to ``PATH`` when spawning mpv so mpv's ``ytdl_hook``
can resolve YouTube streams. All functions return JSON-serializable dicts
(or plain values for pure parsers) and raise :class:`MediaError` on failure.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import desktop

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_WATCH_HOSTS = (
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
)
_SUSPICIOUS_URL_MESSAGE = (
    "suspicious/invalid video URL — pass a real URL or id from "
    "find_video results, or plain search text"
)


def _youtube_watch_id(url: str) -> str | None:
    """Extract the video id from a YouTube watch URL, or ``None`` if not one.

    Returns an empty string when the URL is a YouTube watch URL but carries
    no usable id, so the caller can reject it.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if host == "youtu.be":
        return parts.path.strip("/")
    if host in _YOUTUBE_WATCH_HOSTS and parts.path == "/watch":
        values = urllib.parse.parse_qs(parts.query).get("v") or []
        return values[0] if values else ""
    return None
_EPISODE_RE = re.compile(r"episode\s+(\d+)\s*$", re.IGNORECASE)
_TRAILING_NUMBER_RE = re.compile(r"(\d+)\s*$")
_TRAILING_SEPARATOR_RE = re.compile(r"[\s:\-—–,]+$")

_DEFAULT_ANI_HISTORY = "~/.local/state/ani-cli/history.txt"


class MediaError(RuntimeError):
    """Raised when a media operation fails."""


def _venv_bin() -> str | None:
    """Return ``<project>/.venv/bin`` when it exists, else ``None``."""
    project = Path(__file__).resolve().parents[2]
    candidate = project / ".venv" / "bin"
    if candidate.is_dir():
        return str(candidate)
    return None


def _run(argv: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
    """Run ``argv`` without a shell, capturing output; venv bin prepended to PATH.

    Returns ``(returncode, stdout, stderr)``. Raises :class:`MediaError` on
    timeout; ``FileNotFoundError`` propagates for callers that want a custom
    "not installed" message.
    """
    env = dict(os.environ)
    venv = _venv_bin()
    if venv:
        env["PATH"] = f"{venv}:{env.get('PATH', '')}"
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{argv[0]!r} timed out after {timeout}s") from exc
    return proc.returncode, proc.stdout, proc.stderr


def parse_yt_lines(stdout: str) -> list[dict]:
    """Parse yt-dlp ``--print`` output lines into result dicts.

    Each line is tab-separated ``id, title, duration, channel, url``; the URL
    column is optional and is synthesized from the video id when absent.
    Lines with fewer than four columns are skipped. Pure function.
    """
    results: list[dict] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        while len(parts) < 5:
            parts.append("")
        video_id, title, duration, channel, url = (p.strip() for p in parts[:5])
        if not video_id:
            continue
        if not url:
            url = f"https://www.youtube.com/watch?v={video_id}"
        results.append(
            {
                "id": video_id,
                "title": title,
                "duration": duration,
                "channel": channel,
                "url": url,
            }
        )
    return results


_YOUTUBE_API_SEARCH = "https://www.googleapis.com/youtube/v3/search"


def parse_youtube_api(json_text: str) -> list[dict]:
    """Parse a YouTube Data API v3 ``search`` JSON reply into result dicts.

    Each item yields ``{"id", "title", "channel", "url", "description"}``
    with the description truncated to 200 characters; items without a
    usable videoId are skipped. Pure function.
    """
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    results: list[dict] = []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        video_id = None
        if isinstance(item_id, dict):
            video_id = item_id.get("videoId")
        elif isinstance(item_id, str):
            video_id = item_id
        if not video_id:
            continue
        snippet = item.get("snippet")
        snippet = snippet if isinstance(snippet, dict) else {}
        title = str(snippet.get("title") or "")
        channel = str(snippet.get("channelTitle") or "")
        description = str(snippet.get("description") or "")[:200]
        results.append(
            {
                "id": video_id,
                "title": title,
                "channel": channel,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "description": description,
            }
        )
    return results


def _yt_api_search(
    query: str, max_results: int, api_key: str, timeout: float
) -> list[dict]:
    """Query the official YouTube Data API search endpoint via urllib."""
    params = urllib.parse.urlencode(
        {
            "part": "snippet",
            "type": "video",
            "q": query,
            "maxResults": max_results,
            "key": api_key,
        }
    )
    url = f"{_YOUTUBE_API_SEARCH}?{params}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "assistant/1.0 (+local desktop agent)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return parse_youtube_api(body)


def yt_search(
    query: str,
    max_results: int = 5,
    timeout: float = 60.0,
    api_key: str | None = None,
) -> dict:
    """Search YouTube via the official API when ``api_key`` is set, else yt-dlp.

    With a key, the YouTube Data API v3 search endpoint is used (results
    include a truncated description); on any API error the request falls back
    to the yt-dlp flat-playlist metadata search silently.
    """
    if api_key:
        try:
            results = _yt_api_search(query, max_results, api_key, timeout)
            if results:
                return {"query": query, "results": results, "source": "youtube-api"}
        except (OSError, ValueError):
            pass
    argv = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--flat-playlist",
        "--print",
        "%(id)s\t%(title)s\t%(duration)s\t%(channel)s",
        "--no-warnings",
        "--skip-download",
        "--no-playlist",
    ]
    try:
        code, stdout, stderr = _run(argv, timeout=timeout)
    except FileNotFoundError as exc:
        raise MediaError("yt-dlp not found — run scripts/install.sh") from exc
    if code != 0:
        raise MediaError(f"yt-dlp failed (exit {code}): {stderr.strip()[-300:]}")
    return {"query": query, "results": parse_yt_lines(stdout)}


def resolve_video_url(video: str, searcher: Any = None) -> str:
    """Resolve ``video`` to a playable URL.

    HTTP(S) URLs pass through unchanged, 11-character YouTube ids become
    watch URLs, and anything else is looked up with ``searcher`` (defaulting
    to :func:`yt_search`) using the top result's URL. Raises
    :class:`MediaError` when the search yields no usable result.
    """
    value = video.strip()
    lowered = value.lower()
    if lowered.startswith(("http://", "https://")):
        video_id = _youtube_watch_id(value)
        if video_id is not None and not _YOUTUBE_ID_RE.match(video_id):
            raise MediaError(_SUSPICIOUS_URL_MESSAGE)
        return value
    if _YOUTUBE_ID_RE.match(value):
        return f"https://www.youtube.com/watch?v={value}"

    searcher = searcher or yt_search
    search_result = searcher(value)
    results = search_result.get("results", []) if isinstance(search_result, dict) else []
    if not results:
        raise MediaError(f"no results for {video!r}")
    first = results[0]
    url = first.get("url") if isinstance(first, dict) else None
    if not url:
        raise MediaError(f"top result for {video!r} has no usable URL")
    return url


def _mpv_command(url: str, config: Any = None) -> str:
    """Build the mpv command line for ``url``.

    When yt-dlp is not on the system PATH but exists in the project venv, the
    command prefixes the venv ``bin`` directory to ``PATH`` via ``env`` so
    mpv's ``ytdl_hook`` can find yt-dlp (this form works under a shell,
    e.g. Hyprland's ``exec`` dispatcher).
    """
    quoted = shlex.quote(url)
    if shutil.which("yt-dlp"):
        return f"mpv {quoted}"
    venv = _venv_bin()
    if venv:
        return f"env PATH={shlex.quote(venv)}:$PATH mpv {quoted}"
    return f"mpv {quoted}"


def play_video(
    video: str,
    player: str | None = None,
    workspace: int | None = None,
    config: Any = None,
) -> dict:
    """Play ``video`` in the browser (default) or with mpv.

    The default player comes from ``config.tools.media_player`` ("browser");
    pass ``player="mpv"`` to use mpv instead. The browser path opens the
    resolved watch URL via :func:`desktop.open_url` on ``workspace``, which
    places the new window itself (exec+move for single-instance browsers);
    its ``note`` is surfaced here when the launch does not fully succeed.

    For mpv with no workspace, mpv is spawned directly (detached, output
    suppressed) and its pid returned; with a workspace, it is launched
    through the Hyprland ``exec`` dispatcher so the window lands on the
    right workspace.
    """
    if player is None:
        player = desktop._config_value(config, "tools", "media_player", "browser")
    url = resolve_video_url(video)
    if player not in ("mpv", "browser"):
        raise MediaError(f"unsupported player: {player!r} (supported: 'mpv', 'browser')")

    venv = _venv_bin()
    has_system_ytdlp = shutil.which("yt-dlp") is not None
    note = None
    if player == "mpv" and not has_system_ytdlp and venv:
        note = (
            "yt-dlp exists only in the project venv; its bin dir is prepended "
            "to PATH for mpv so ytdl_hook can resolve the stream"
        )

    if player == "browser":
        result = desktop.open_url(url, workspace=workspace, config=config)
        note = result.get("note") if result.get("ok") is False else None
        return {
            "url": url,
            "title": video,
            "player": player,
            "workspace": workspace,
            "pid": None,
            "method": result.get("method", "browser"),
            "note": note,
        }

    if workspace is None:
        command = _mpv_command(url, config=config)
        if "$PATH" in command:
            command = command.replace("$PATH", os.environ.get("PATH", ""))
        proc = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {
            "url": url,
            "title": video,
            "player": player,
            "workspace": None,
            "pid": proc.pid,
            "method": "direct",
            "note": note,
        }

    result = desktop.launch_app(
        "mpv", shlex.quote(url), workspace=workspace, config=config
    )
    return {
        "url": url,
        "title": video,
        "player": player,
        "workspace": workspace,
        "pid": None,
        "method": result.get("method"),
        "note": note,
    }


def parse_ani_history(text: str) -> list[dict]:
    """Parse ani-cli watch history lines into ``{"title", "episode"}`` dicts.

    Lenient: the episode is the trailing number (preferring an explicit
    ``episode <N>`` form), and the title is the remainder with trailing
    separators (``: - — – ,`` and whitespace) stripped. Lines without a
    trailing number are skipped. Pure function.
    """
    entries: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _EPISODE_RE.search(line)
        if match:
            title = line[: match.start()]
            episode = int(match.group(1))
        else:
            match = _TRAILING_NUMBER_RE.search(line)
            if not match:
                continue
            title = line[: match.start()]
            episode = int(match.group(1))
        title = _TRAILING_SEPARATOR_RE.sub("", title).strip()
        if title:
            entries.append({"title": title, "episode": episode})
    return entries


def _ani_history_path(config: Any) -> Path:
    """Resolve the ani-cli history file path from config (or the default)."""
    value = desktop._config_value(config, "tools", "ani_history", _DEFAULT_ANI_HISTORY)
    return Path(str(value)).expanduser()


def _read_ani_history(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise MediaError(f"cannot read ani-cli history {path}: {exc}") from exc


def ani_play(
    query: str,
    episode: int | None = None,
    select_nth: int = 1,
    dub: bool = False,
    workspace: int | None = None,
    config: Any = None,
) -> dict:
    """Start playing ``query`` with ani-cli (quality "best", streaming via mpv).

    With a workspace, ani-cli is launched through the Hyprland ``exec``
    dispatcher; otherwise it is spawned directly and its pid returned.
    """
    argv = ["ani-cli", "-S", str(select_nth), "-q", "best"]
    if episode is not None:
        argv += ["-e", str(episode)]
    if dub:
        argv.append("--dub")
    argv.append(query)
    command = " ".join(argv)

    if workspace is not None:
        result = desktop.launch_app(
            "ani-cli", shlex.join(argv[1:]), workspace=workspace, config=config
        )
        return {
            "command": command,
            "pid": None,
            "workspace": workspace,
            "method": result.get("method"),
            "note": "playing via mpv",
        }
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise MediaError("ani-cli not found — install it with your package manager") from exc
    return {
        "command": command,
        "pid": proc.pid,
        "workspace": None,
        "method": "direct",
        "note": "playing via mpv",
    }


def ani_next(query: str | None = None, config: Any = None) -> dict:
    """Report what episode comes next for an anime watched with ani-cli.

    Without ``query``, returns the full history (latest entries first). With
    a query, returns the best (most recent) title whose name casefold-
    contains the query, plus its next episode number. A missing history file
    yields ``{"status": "no-history", ...}``.
    """
    path = _ani_history_path(config)
    if not path.exists():
        return {
            "status": "no-history",
            "note": f"no ani-cli history at {path} — watch something with ani-cli first",
        }
    entries = parse_ani_history(_read_ani_history(path))
    if query is not None:
        needle = query.casefold()
        match = None
        for entry in entries:
            if needle in entry["title"].casefold():
                match = entry
        if match is None:
            return {
                "status": "not-found",
                "query": query,
                "note": f"no watched title matching {query!r}",
            }
        return {
            "status": "found",
            "title": match["title"],
            "last_episode": match["episode"],
            "next_episode": match["episode"] + 1,
        }
    return {"status": "ok", "history": list(reversed(entries))}


def list_ani_history(config: Any = None) -> dict:
    """Return the full parsed ani-cli watch history (latest entries first)."""
    path = _ani_history_path(config)
    if not path.exists():
        return {
            "status": "no-history",
            "note": f"no ani-cli history at {path}",
        }
    entries = parse_ani_history(_read_ani_history(path))
    return {"status": "ok", "path": str(path), "history": list(reversed(entries))}
