"""Desktop notifications: caelestia-toaster with a notify-send fallback."""

from __future__ import annotations

import subprocess
from typing import Any


async def notify_user(ctx: Any, message: str, title: str = "Assist") -> dict:
    """Pop a desktop toast; returns which backend delivered it."""
    message = str(message)
    title = str(title or "Assist")
    attempts = (
        (
            "caelestia-toaster",
            ["caelestia", "shell", "toaster", "info", title, message],
        ),
        ("notify-send", ["notify-send", title, message]),
    )
    last_error = ""
    for sent_via, argv in attempts:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
            continue
        if proc.returncode == 0:
            return {"sent_via": sent_via, "message": message}
        last_error = (
            (proc.stderr or proc.stdout).strip()
            or f"exit code {proc.returncode}"
        )
    return {"sent_via": None, "error": last_error}
