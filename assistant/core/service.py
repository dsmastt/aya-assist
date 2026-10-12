"""Standalone background service: timers, reminders, notifications.

Runs independently of the LLM/repl as its own process (``assist service
run`` in the foreground, ``assist service start`` detached, or a systemd
user unit via ``assist service install``). Every 5 seconds it drains
finished timers and fires due reminders as desktop toasts, each with a
tsundere-flavored suffix.

The repl no longer schedules reminders itself (no double-firing); the
service owns the ``tasks`` state file for notification purposes.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import secrets
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import Config
from ..tools.registry import ToolContext

POLL_SECONDS = 5.0

_FLAVOR_SUFFIXES = [
    "(or else I'll be pretty angry, you know.)",
    "— not that I care or anything. 😏",
    "You'd better be done. I'll be checking.",
    "(*pouts*) ...fine, I'm just saying.",
    "Don't make me remind you twice.",
]


def flavored(base: str) -> str:
    """Append one tsundere suffix to ``base`` (keeps its prefix intact)."""
    return f"{base} {secrets.choice(_FLAVOR_SUFFIXES)}"


# ------------------------------------------------------------ pid file

def _pid_file(config: Any = None) -> Path:
    if config is None:
        config = Config()
    return Path(config.paths.state_file).expanduser().parent / "service.pid"


def _remove_pidfile(pidfile: Path) -> None:
    try:
        if pidfile.exists():
            pidfile.unlink()
    except OSError:
        pass


def service_pid(config: Any = None) -> int | None:
    """The pid recorded in the pidfile, or None."""
    try:
        return int(_pid_file(config).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def service_running(config: Any = None) -> bool:
    """True when the pidfile exists and its pid answers ``kill(pid, 0)``."""
    pidfile = _pid_file(config)
    if not pidfile.exists():
        return False
    pid = service_pid(config)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


# ------------------------------------------------------------ notify

def notify(title: str, message: str) -> str | None:
    """Pop a desktop toast; returns the backend that delivered it (or None)."""
    title, message = str(title), str(message)
    attempts = (
        ("caelestia", ["caelestia", "shell", "toaster", "info", title, message]),
        ("notify-send", ["notify-send", title, message]),
    )
    for sent_via, argv in attempts:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            return sent_via
    return None


# ------------------------------------------------------------ tick loop

def _tick(config: Any, log: logging.Logger) -> None:
    from ..tools import tasks

    ctx = ToolContext(config=config, console=None)

    for label in asyncio.run(tasks.timer_drain(ctx)):
        notify("Timer done", flavored(f"Timer done: {label}"))

    listing = asyncio.run(tasks.reminder_list(ctx))
    reminders = listing.get("reminders", []) if isinstance(listing, dict) else []
    now = datetime.now()
    for reminder in reminders:
        try:
            at = datetime.fromisoformat(str(reminder.get("at", "")))
        except (TypeError, ValueError):
            continue
        if at <= now:
            message = str(reminder.get("message", "") or "")
            notify("Reminder", flavored(f"Reminder: {message}"))
            try:
                asyncio.run(tasks.reminder_cancel(ctx, int(reminder.get("id"))))
            except (TypeError, ValueError):
                continue


def run_forever(config: Any = None, log: logging.Logger | None = None) -> None:
    """Foreground loop: drain timers / fire reminders every 5 seconds.

    Writes the pidfile on start and removes it on exit (atexit + finally).
    SIGINT/SIGTERM stop the loop for a clean exit.
    """
    cfg = Config.load() if config is None else config
    if log is None:
        log_dir = Path(cfg.paths.log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_dir / "service.log"),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        log = logging.getLogger("assistant.service")

    pidfile = _pid_file(cfg)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(_remove_pidfile, pidfile)

    _stop = threading.Event()

    def _on_signal(signum, frame) -> None:  # noqa: ARG001
        _stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            pass

    try:
        while not _stop.is_set():
            try:
                _tick(cfg, log)
            except Exception:  # noqa: BLE001 — keep the service alive
                log.exception("service tick failed")
            _stop.wait(POLL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        _remove_pidfile(pidfile)


# ------------------------------------------------------------ daemon

def start_daemon(config: Any = None) -> int:
    """Spawn ``python -m assistant service run`` detached; return its pid."""
    pidfile = _pid_file(config)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "assistant", "service", "run"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.pid


def stop_daemon(config: Any = None) -> bool:
    """SIGTERM the running service via its pidfile; True when sent."""
    pid = service_pid(config)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True
