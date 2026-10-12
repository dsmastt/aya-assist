"""Console UI: the assistant's only user-facing output surface.

Renders markdown and tool activity with rich, streams partial LLM output
through a Live view, and reads input / confirmations with prompt_toolkit.
All interactive classes degrade to plain ``print``/``input`` if rich or
prompt_toolkit are unavailable.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from ..config import Config
from .. import __version__

try:
    from rich.console import Console as _RichConsole
    from rich.console import Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text

    _RICH = True
except Exception:  # pragma: no cover - depends on environment
    _RICH = False

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.validation import ValidationError, Validator

    _PT = True
except Exception:  # pragma: no cover - depends on environment
    _PT = False

if TYPE_CHECKING:
    from rich.live import Live as LiveType
    from rich.status import Status

_STREAM_THROTTLE_S = 0.05
_STREAM_MAX_LINES = 40


class _YesNoValidator(Validator):
    """Accept y/n/yes/no or empty (empty means "use the default")."""

    def validate(self, document) -> None:
        text = document.text.strip().lower()
        if text in ("", "y", "n", "yes", "no"):
            return
        raise ValidationError(message="Please answer y or n", cursor_position=0)


class StreamHandle:
    """Accumulates streamed text and renders it inside a rich Live view.

    Use as a context manager::

        with console.stream() as sh:
            sh.append(chunk)

    Partial markdown is re-rendered at most every 50 ms; on exit the final
    content is printed exactly once so it stays on screen.
    """

    def __init__(
        self, rc: "_RichConsole | None", use_live: bool = True
    ) -> None:
        self._rc = rc
        self._use_live = use_live
        self._buf: list[str] = []
        self._live: "LiveType | None" = None
        self._last_update = 0.0
        self._dirty = False

    def __enter__(self) -> "StreamHandle":
        if self._use_live and self._rc is not None:
            self._live = Live(
                Markdown("", code_theme="monokai"),
                console=self._rc,
                auto_refresh=False,
                transient=True,
            )
            self._live.start()
        return self

    def append(self, text: str) -> None:
        self._buf.append(text)
        self._dirty = True
        if self._live is None:
            return
        now = time.monotonic()
        if now - self._last_update >= _STREAM_THROTTLE_S:
            self._render()
            self._last_update = now

    def _render(self) -> None:
        if self._live is None or not self._dirty:
            return
        body = "".join(self._buf)
        lines = body.splitlines()
        if len(lines) > _STREAM_MAX_LINES:
            body = "\n".join(lines[-_STREAM_MAX_LINES:])
        self._live.update(Markdown(body, code_theme="monokai"), refresh=True)
        self._dirty = False

    def __exit__(self, *exc_info) -> None:
        if self._live is not None:
            self._render()
            self._live.stop()
            self._live = None
        body = "".join(self._buf)
        if not body:
            return
        if self._rc is not None:
            self._rc.print(Markdown(body, code_theme="monokai"))
        else:
            print(body)


class Console:
    """Terminal UI for the assistant (rich + prompt_toolkit)."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._name = (config.persona.name or "Aya").strip() or "Aya"
        self._rc: "_RichConsole | None" = _RichConsole() if _RICH else None
        self._session = None
        self._yn_session = None
        self._interactive = sys.stdin.isatty()
        if _PT:
            history_path = Path(config.paths.history_file).expanduser()
            try:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                self._session = PromptSession(
                    history=FileHistory(str(history_path))
                )
                self._yn_session = PromptSession(validator=_YesNoValidator())
            except Exception:  # pragma: no cover - filesystem issues
                self._session = None
                self._yn_session = None
        if self._rc is None or self._session is None:
            self.warn(
                "rich/prompt_toolkit unavailable; falling back to plain text I/O"
            )

    # ------------------------------------------------------------- display

    def welcome(self, status_line: str = "") -> None:
        if self._rc is None:
            line = f"[{self._name}] assist v{__version__}"
            if status_line:
                line += f"  {status_line}"
            print(line)
            print("type /help for commands")
            return
        title = Text()
        title.append(f"[{self._name}]", style="bold magenta")
        title.append(f" assist v{__version__}", style="bold red")
        parts = [title]
        if status_line:
            parts.append(Text(status_line, style="italic dim"))
        parts.append(Text("type /help for commands", style="dim"))
        self._rc.print(Panel(Group(*parts), border_style="red", expand=False))

    def markdown(self, text: str) -> None:
        if self._rc is None:
            print(text)
            return
        self._rc.print(Markdown(text, code_theme="monokai"))

    def tool(self, label: str) -> None:
        if self._rc is None:
            print(f"• {label}")
            return
        self._rc.print(Text(f"• {label}", style="dim italic grey58"))

    def info(self, msg: str) -> None:
        if self._rc is None:
            print(f"· {msg}")
            return
        self._rc.print(Text(f"ℹ {msg}", style="cyan"))

    def warn(self, msg: str) -> None:
        if self._rc is None:
            print(f"warning: {msg}")
            return
        self._rc.print(Text(f"⚠ {msg}", style="yellow"))

    def error(self, msg: str) -> None:
        if self._rc is None:
            print(f"error: {msg}")
            return
        self._rc.print(Text(f"✗ {msg}", style="bold red"))

    def hr(self) -> None:
        if self._rc is None:
            print("-" * 40)
            return
        self._rc.print(Rule(style="dim"))

    # ------------------------------------------------------------- input

    def prompt(self) -> str:
        if self._session is None or not self._interactive:
            try:
                return input("❯ ").strip()
            except EOFError:
                return "/exit"
        try:
            text = self._session.prompt(
                "❯ ", mouse_support=False, enable_history_search=True
            )
        except EOFError:
            return "/exit"
        return text.strip()

    def ask_yes_no(self, question: str, default: bool = False) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        if self._yn_session is None or not self._interactive:
            try:
                raw = input(f"{question} {hint} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
        else:
            try:
                raw = self._yn_session.prompt(
                    HTML(f"<yellow>{html_escape(question)}</yellow> {hint} "),
                    mouse_support=False,
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
        if raw == "":
            return default
        return raw in ("y", "yes")

    # ------------------------------------------------------------- async

    def stream(self) -> "StreamHandle":
        use_live = bool(self._rc is not None and self._rc.is_terminal)
        return StreamHandle(self._rc, use_live=use_live)

    @contextmanager
    def spinner(self, label: str) -> Iterator["Status | None"]:
        if self._rc is None:
            print(label)
            yield None
            return
        with self._rc.status(label, spinner="dots") as status:
            yield status
