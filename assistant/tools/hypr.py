"""Hyprland IPC helpers.

Talks to the Hyprland command/IPC socket directly over ``AF_UNIX`` so the
agent works inside the desktop session without shelling out to ``hyprctl``.

Everything here is plain functions returning JSON-serializable data. Any
failure raises :class:`HyprError`; success is never fabricated.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

_UNREACHABLE_MSG = (
    "cannot reach Hyprland IPC — is the agent running inside your desktop session?"
)


class HyprError(RuntimeError):
    """Raised when the Hyprland IPC socket is unreachable or misbehaves."""


def _socket_path() -> str:
    """Return the Hyprland IPC socket path from the session environment.

    Raises :class:`HyprError` when ``XDG_RUNTIME_DIR`` or
    ``HYPRLAND_INSTANCE_SIGNATURE`` is unset (i.e. not inside a Hyprland
    session).
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    if not runtime or not signature:
        raise HyprError(
            "not in a Hyprland session: XDG_RUNTIME_DIR and/or "
            "HYPRLAND_INSTANCE_SIGNATURE are unset"
        )
    return os.path.join(runtime, "hypr", signature, ".socket.sock")


def hyprctl_raw(payload: str, timeout: float = 5.0) -> str:
    """Send a raw payload to the Hyprland IPC socket and read the full reply.

    The connection is an ``AF_UNIX`` stream socket; the payload is sent
    verbatim and the reply is read until EOF. Raises :class:`HyprError` when
    the socket cannot be reached or the request fails.
    """
    path = _socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        try:
            sock.connect(path)
        except (ConnectionRefusedError, FileNotFoundError) as exc:
            raise HyprError(_UNREACHABLE_MSG) from exc
        sock.sendall(payload.encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")
    except HyprError:
        raise
    except (socket.timeout, TimeoutError, OSError) as exc:
        raise HyprError(f"Hyprland IPC request failed: {exc}") from exc
    finally:
        sock.close()


def hyprctl_json(request: str, timeout: float = 5.0) -> Any:
    """Send a ``j/<request>`` payload and parse the reply as JSON.

    A parse failure raises :class:`HyprError` carrying the head of the raw
    reply for diagnostics.
    """
    raw = hyprctl_raw("j/" + request, timeout=timeout)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HyprError(f"invalid JSON reply from Hyprland IPC: {raw[:200]!r}") from exc


def dispatch(args: str, timeout: float = 5.0) -> str:
    """Send a ``dispatch <args>`` command and return the raw reply string."""
    return hyprctl_raw("dispatch " + args, timeout=timeout).strip()


def version() -> dict:
    """Hyprland version information."""
    return hyprctl_json("version")


def workspaces() -> list[dict]:
    """All workspaces with their metadata."""
    return hyprctl_json("workspaces")


def active_workspace() -> dict:
    """The currently focused workspace."""
    return hyprctl_json("activeworkspace")


def clients() -> list[dict]:
    """All open windows (clients) with address/class/title metadata."""
    return hyprctl_json("clients")


def monitors() -> list[dict]:
    """All monitors and their geometry."""
    return hyprctl_json("monitors")


def active_window() -> dict:
    """The currently focused window."""
    return hyprctl_json("activewindow")
