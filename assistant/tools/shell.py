"""Shell command execution for the assistant agent.

Provides synchronous runs with timeouts and output caps, background
processes, environment scrubbing, and an optional safety-check hook.
"""

from __future__ import annotations

import getpass
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

OUTPUT_CAP = 20_000
TRUNCATED_MARKER = "…[output truncated]"

_COPY_KEYS = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "HYPRLAND_INSTANCE_SIGNATURE",
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
    "DBUS_SESSION_BUS_ADDRESS",
    "QT_QPA_PLATFORM",
    "GDK_BACKEND",
    "MOZ_ENABLE_WAYLAND",
    "NO_COLOR",
)

_FORBIDDEN_PREFIXES = ("SUDO_", "PASS", "AWS_", "GITHUB_", "OPENAI_", "ANTHROPIC_")
_FORBIDDEN_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD")
_FORBIDDEN_KEYS = {"SSH_AUTH_SOCK", "SSH_AGENT_PID", "GNUPGHOME", "HISTFILE"}


class ShellError(RuntimeError):
    pass


@dataclass
class CommandResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


def _is_forbidden_key(key_upper: str) -> bool:
    if key_upper in _FORBIDDEN_KEYS:
        return True
    if any(key_upper.startswith(p) for p in _FORBIDDEN_PREFIXES):
        return True
    if any(s in key_upper for s in _FORBIDDEN_SUBSTRINGS):
        return True
    return False


def default_env(extra: dict | None = None) -> dict:
    """Build a scrubbed environment for shell commands."""
    home = os.environ.get("HOME") or os.path.expanduser("~") or "/"
    try:
        user = os.environ.get("USER") or getpass.getuser()
    except Exception:
        user = ""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": home,
        "LANG": "C.UTF-8",
        "TERM": "xterm-256color",
        "SHELL": os.environ.get("SHELL", "/bin/sh"),
    }
    if user:
        env["USER"] = user
    for key in _COPY_KEYS:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    if extra:
        env.update(extra)
    for key in list(env):
        if _is_forbidden_key(key.upper()):
            env.pop(key, None)
    return env


def _cap_output(data: str) -> tuple[str, bool]:
    if len(data) > OUTPUT_CAP:
        return data[:OUTPUT_CAP] + TRUNCATED_MARKER, True
    return data, False


def _kill_group(pgid: int, sig: int = signal.SIGTERM) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run(
    command: str,
    cwd: str | None = None,
    timeout: float = 120.0,
    env: dict | None = None,
) -> CommandResult:
    if env is None:
        env = default_env()
    try:
        proc = subprocess.Popen(
            ["/bin/sh", "-c", command],
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, ValueError) as exc:
        raise ShellError(str(exc)) from exc

    timed_out = False
    try:
        out_b, err_b = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc.pid, signal.SIGTERM)
        try:
            out_b, err_b = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            _kill_group(proc.pid, signal.SIGKILL)
            out_b, err_b = proc.communicate()

    stdout = out_b.decode("utf-8", errors="replace")
    stderr = err_b.decode("utf-8", errors="replace")
    stdout, out_trunc = _cap_output(stdout)
    stderr, err_trunc = _cap_output(stderr)
    return CommandResult(
        returncode=None if timed_out else proc.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        truncated=out_trunc or err_trunc,
    )


@dataclass
class BackgroundProc:
    pid: int
    cmd: str
    _proc: subprocess.Popen = field(repr=False)
    _out_path: str = field(repr=False)
    _err_path: str = field(repr=False)
    _stdout: str = field(repr=False, default="")
    _stderr: str = field(repr=False, default="")

    def poll(self) -> int | None:
        return self._proc.poll()

    def is_running(self) -> bool:
        return self._proc.poll() is None

    def kill(self, grace: float = 5.0) -> bool:
        """Terminate the process group; return True if it was running."""
        if self._proc.poll() is not None:
            return False
        _kill_group(self.pid, signal.SIGTERM)
        try:
            self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            _kill_group(self.pid, signal.SIGKILL)
            self._proc.wait()
        return True

    def _read_tail(self, path: str, tail: int) -> str:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            return ""
        text = data.decode("utf-8", errors="replace")
        if len(text) > tail:
            text = text[-tail:]
        return text

    def output(self, tail: int = 2000) -> str:
        out = self._stdout
        err = self._stderr
        if os.path.exists(self._out_path):
            out = self._read_tail(self._out_path, tail)
        if os.path.exists(self._err_path):
            err = self._read_tail(self._err_path, tail)
        if err:
            return f"{out}\n[stderr]\n{err}" if out else err
        return out

    def _collect(self) -> None:
        for path, attr in ((self._out_path, "_stdout"), (self._err_path, "_stderr")):
            try:
                with open(path, "rb") as fh:
                    text = fh.read().decode("utf-8", errors="replace")
                os.unlink(path)
            except OSError:
                text = getattr(self, attr)
            setattr(self, attr, text)

    def wait(self, timeout: float | None = None) -> int | None:
        try:
            returncode = self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        self._collect()
        return returncode


def spawn(
    command: str,
    cwd: str | None = None,
    env: dict | None = None,
) -> BackgroundProc:
    if env is None:
        env = default_env()
    out_fd, out_path = tempfile.mkstemp(prefix="shell-out-", text=True)
    err_fd, err_path = tempfile.mkstemp(prefix="shell-err-", text=True)
    try:
        proc = subprocess.Popen(
            ["/bin/sh", "-c", command],
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdout=out_fd,
            stderr=err_fd,
        )
    except (OSError, ValueError) as exc:
        os.close(out_fd)
        os.close(err_fd)
        os.unlink(out_path)
        os.unlink(err_path)
        raise ShellError(str(exc)) from exc
    os.close(out_fd)
    os.close(err_fd)
    return BackgroundProc(pid=proc.pid, cmd=command, _proc=proc, _out_path=out_path, _err_path=err_path)


def run_checked(
    command: str,
    policy: Any = None,
    cwd: str | None = None,
    timeout: float = 120.0,
    approve: Optional[Callable[[str, str], bool]] = None,
    env: dict | None = None,
) -> dict:
    if policy is None:
        try:
            from .safety import default_policy
        except ImportError:
            default_policy = None
        policy = default_policy() if default_policy is not None else None

    if policy is not None:
        verdict = policy.classify(command)
        action = getattr(verdict, "action", None)
        action_name = getattr(action, "name", None)
        category = getattr(verdict, "category", "")
        reasons = list(getattr(verdict, "reasons", []) or [])
        if action_name == "DENY":
            return {"status": "denied", "verdict_category": category, "reasons": reasons}
        if action_name == "APPROVE":
            detail = (
                f"{getattr(verdict, 'summary', '')}; reasons: "
                + "; ".join(reasons)
                + "; command: "
                + command[:300]
            )
            if approve is None or not approve("shell command needs approval", detail):
                return {"status": "not-approved", "verdict_category": category, "reasons": reasons}

    try:
        result = run(command, cwd=cwd, timeout=timeout, env=env)
    except ShellError as exc:
        return {"status": "error", "error": str(exc)}
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
    }
