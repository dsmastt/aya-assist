"""Background threads: background jobs, model keep-alive, reminders.

Three daemon threads keep the agent ticking between turns:

- :class:`BackgroundRunner` polls spawned background processes and pops a
  desktop notification when one finishes.
- :class:`KeepAlive` unloads the LLM when it idles past the configured
  keep-alive window, freeing VRAM/RAM.
- :class:`ReminderScheduler` fires due reminders and finished timers as
  desktop notifications.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import threading
import uuid
from datetime import datetime

from ..tools import shell
from ..tools.registry import ToolContext

log = logging.getLogger("assistant")


class BackgroundRunner:
    """Tracks background processes spawned via :func:`assistant.tools.shell.spawn`.

    A daemon thread polls every 2 seconds; when a process exits it is
    collected once and ``notify-send`` announces the completion.
    """

    POLL_SECONDS = 2.0

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="bg-jobs", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------- jobs

    def spawn(self, cmd: str, label: str = "") -> str:
        """Spawn ``cmd`` in the background and return its job id."""
        proc = shell.spawn(cmd)
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "label": label or cmd[:40], "proc": proc, "done": False}
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def list_jobs(self) -> list[dict]:
        """Snapshot of all jobs with running state."""
        with self._lock:
            return [
                {
                    "id": job["id"],
                    "label": job["label"],
                    "running": not job["done"] and job["proc"].is_running(),
                    "done": job["done"],
                }
                for job in self._jobs.values()
            ]

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        return not job["done"] and job["proc"].is_running()

    def stop_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job["done"]:
            return False
        return job["proc"].kill()

    # -------------------------------------------------------------- loop

    def _loop(self) -> None:
        while not self._stop.is_set():
            finished: list[dict] = []
            with self._lock:
                for job in self._jobs.values():
                    if job["done"] or job["proc"].is_running():
                        continue
                    job["done"] = True
                    finished.append(job)
            for job in finished:
                try:
                    job["proc"].wait(timeout=0)
                except Exception:  # noqa: BLE001 — collection is best-effort
                    pass
                self._notify(job["label"])
            self._stop.wait(self.POLL_SECONDS)

    @staticmethod
    def _notify(label: str) -> None:
        """Desktop notification that a background job finished (fired once)."""
        try:
            subprocess.Popen(
                ["notify-send", "Assist", f"{label} finished"],
                env=shell.default_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            log.debug("notify-send failed: %s", exc)


class KeepAlive:
    """Unloads the model when it idles past ``cfg.model.keep_alive_seconds``."""

    POLL_SECONDS = 60.0

    def __init__(self, config, console, engine) -> None:
        self.config = config
        self.console = console
        self.engine = engine
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="keep-alive", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.POLL_SECONDS)
            if self._stop.is_set():
                return
            try:
                if self.engine.loaded and (
                    self.engine.idle_seconds() > self.config.model.keep_alive_seconds
                ):
                    if self.console is not None:
                        self.console.info(
                            f"model idle for {int(self.engine.idle_seconds())}s "
                            "— unloading to free resources"
                        )
                    self.engine.unload()
            except Exception:  # noqa: BLE001 — keep the thread alive
                log.exception("keep-alive tick failed")


class ReminderScheduler:
    """Fires due reminders and finished timers every 10 seconds.

    Reminders: notify once, then cancel. Timers: drain all finished ones
    and pop a ``Timer done: <label>`` notification for each.
    """

    POLL_SECONDS = 10.0

    def __init__(self, config, console) -> None:
        self.config = config
        self.console = console
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="reminders", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.POLL_SECONDS)
            if self._stop.is_set():
                return
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 — fail silently, log only
                print(f"reminder scheduler error: {exc}", file=sys.stderr)

    def _tick(self) -> None:
        from ..tools import notify, tasks

        ctx = ToolContext(config=self.config, console=self.console)
        listing = asyncio.run(tasks.reminder_list(ctx))
        now = datetime.now()
        reminders = listing.get("reminders", []) if isinstance(listing, dict) else []
        for reminder in reminders:
            try:
                at = datetime.fromisoformat(str(reminder.get("at", "")))
            except (TypeError, ValueError):
                continue
            if at <= now:
                asyncio.run(notify.notify_user(ctx, str(reminder.get("message", ""))))
                try:
                    asyncio.run(tasks.reminder_cancel(ctx, int(reminder.get("id"))))
                except (TypeError, ValueError):
                    continue
        for label in asyncio.run(tasks.timer_drain(ctx)):
            asyncio.run(notify.notify_user(ctx, f"Timer done: {label}"))
