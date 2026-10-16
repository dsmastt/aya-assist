"""Command-line entry point for the assistant agent.

Subcommands:

- ``assist`` / ``assist repl`` — interactive agent
- ``assist chat "message..."`` — one-shot, non-interactive turn
- ``assist status`` — system + model status, no load
- ``assist setup`` — create config dirs, print config path and hints
- ``assist selftest`` — self-test battery (launches real apps)

The model is loaded lazily on the first user message (RAM friendly), never
at startup.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from assistant import __version__
from assistant.config import Config
from assistant.core.agent import Agent
from assistant.core.background import BackgroundRunner, KeepAlive
from assistant.core import service
from assistant.core.resources import system_status
from assistant.core.session import Session
from assistant.llm.engine import EngineError, LLMEngine
from assistant.memory.store import MemoryStore
from assistant.tools import builtin as _builtin  # noqa: F401 — registers tools
from assistant.tools import tasks as tasks_mod
from assistant.tools.registry import ToolContext
from assistant.ui.console import Console

log = logging.getLogger("assistant")

_HELP_TEXT = """\
Available commands:
  /help                     show this list
  /exit, /quit              leave (the old data stays safe, of course)
  /clear                    start a fresh session
  /model load|unload|status manage the model (loads lazily otherwise)
  /memory list              list stored facts
  /memory search <text>     full-text search over memories
  /memory forget <id|text>  delete a fact by id or text match
  /tasks                    show open todos
  /reminders                show scheduled reminders
  /timers                   show running countdown timers
  /history                  print recent conversation
  /status                   system + model status
Anything else is sent to the agent.

Timers and reminders are fired by the background service — run
`assist service install` (or `assist service start` for one session)."""

_GOODBYE = (
    "Hmph! Fine, leaving already? It's not like I wanted to keep helping "
    "you or anything! ...Take care. 😤"
)


def _configure_logging(cfg: Config) -> None:
    log_dir = Path(cfg.paths.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / "assistant.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_runtime(cfg: Config):
    """Console, memory store, engine and session for a run."""
    console = Console(cfg)
    store = MemoryStore(cfg.paths.memory_db)
    engine = LLMEngine(cfg, console)
    session = Session(cfg, store)
    return console, store, engine, session


def _model_status(cfg: Config, engine: LLMEngine) -> str:
    state = "loaded" if engine.loaded else "not loaded"
    est = engine.estimate_vram()
    return (
        f"model: {cfg.model.file} ({state}) — est. VRAM {int(est['total_mb'])} MB "
        f"(weights {int(est['weights_mb'])} + KV {int(est['kv_mb'])})"
    )


def _print_load_hint(console: Console) -> None:
    status = system_status()
    ram = (status.get("ram") or {}).get("avail_mb")
    console.error(
        f"hints: only ~{ram} MB RAM free. Close some apps, pick a smaller "
        "model (see config.toml [model] file/n_ctx/n_gpu_layers/kv_quant), "
        "or check /model status."
    )


def _ensure_engine_loaded(console: Console, engine: LLMEngine) -> bool:
    """Lazily load the model; report a helpful error on failure."""
    if engine.loaded:
        return True
    try:
        with console.spinner("loading model (HDD — this takes a while)"):
            ok = engine.load()
    except EngineError as exc:
        message = f"model load failed: {exc}"
        if "vram" in str(exc).lower():
            message += (
                "\nIf another assist session is running, close it — each "
                "session holds the model in VRAM until its idle timeout."
            )
        console.error(message)
        _print_load_hint(console)
        return False
    if not ok:
        console.error("model load refused by the resource check")
        _print_load_hint(console)
        return False
    return True


# ------------------------------------------------------------ subcommands

def cmd_repl(args: argparse.Namespace) -> int:
    cfg = Config.load()
    cfg.ensure_dirs()
    _configure_logging(cfg)
    console, store, engine, session = _build_runtime(cfg)
    runner = BackgroundRunner()
    keepalive = KeepAlive(cfg, console, engine)
    runner.start()
    keepalive.start()
    agent = Agent(cfg, console, engine, store, runner=runner, session=session)
    console.welcome(_model_status(cfg, engine))
    if not service.service_running(cfg):
        console.warn(
            "timers/reminders need the background service — run "
            "`assist service install` (or `assist service start` for this "
            "session)"
        )
    try:
        while True:
            try:
                text = console.prompt()
            except (KeyboardInterrupt, EOFError):
                console.markdown(_GOODBYE)
                break
            if not text:
                continue
            if text.startswith("/"):
                if _handle_command(text, cfg, console, store, engine, session, agent):
                    break
                continue
            if not _ensure_engine_loaded(console, engine):
                continue
            try:
                asyncio.run(agent.run_once(text))
            except KeyboardInterrupt:
                console.warn("interrupted")
    finally:
        runner.stop()
        keepalive.stop()
        store.close()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = Config.load()
    cfg.ensure_dirs()
    _configure_logging(cfg)
    console, store, engine, session = _build_runtime(cfg)
    agent = Agent(cfg, console, engine, store, session=session)
    try:
        if not _ensure_engine_loaded(console, engine):
            return 1
        asyncio.run(agent.run_once(" ".join(args.message)))
    finally:
        store.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = Config.load()
    _configure_logging(cfg)
    engine = LLMEngine(cfg, None)
    status = system_status()
    ram = status.get("ram") or {}
    gpu = status.get("gpu") or {}
    load = status.get("load_avg") or []
    est = engine.estimate_vram()
    print(f"assist v{__version__}")
    print(f"config: {cfg.default_path()}")
    print(f"model file: {cfg.model.file}")
    print(f"model path: {engine.model_path}")
    print(f"model loaded: {engine.loaded}")
    print(
        f"vram estimate: weights {int(est['weights_mb'])} MB + KV "
        f"{int(est['kv_mb'])} MB = {int(est['total_mb'])} MB"
    )
    print(f"ram: {ram.get('avail_mb')} MB available of {ram.get('total_mb')} MB")
    if gpu.get("available"):
        print(
            f"gpu: {gpu.get('name')} — {gpu.get('vram_free_mb')} MB VRAM free "
            f"of {gpu.get('vram_total_mb')} MB"
        )
    else:
        print(f"gpu: {gpu.get('note', 'unavailable')}")
    print(f"load: {'/'.join(f'{x:.2f}' for x in load) if load else 'n/a'}")
    if service.service_running(cfg):
        print(f"service: running (pid {service.service_pid(cfg)})")
    else:
        print("service: not running")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    cfg = Config.load()
    cfg.ensure_dirs()
    path = cfg.default_path()
    print(f"config: {path}")
    print("directories ensured: models, logs, data files")
    if not path.exists():
        cfg.save(path)
        print("wrote a default config.toml")
    else:
        print("config.toml already exists — left untouched")
    print(f"model file: {cfg.model.file}")
    print(f"model path: {cfg.model.model_path}")
    print("next steps:")
    print("  bash scripts/download-model.sh   # fetch the model (if missing)")
    print("  assist                           # interactive agent")
    print('  assist chat "hello"              # one-shot turn')
    print("  assist status                    # check hardware + model")
    return 0


def cmd_service(args: argparse.Namespace) -> int:
    """Manage the background timers/reminders/notifications service."""
    cfg = Config.load()
    sub = args.service_command or "status"
    if sub == "run":
        service.run_forever(config=cfg)
        return 0
    if sub == "start":
        if service.service_running(cfg):
            print(f"already running (pid {service.service_pid(cfg)})")
        else:
            pid = service.start_daemon(cfg)
            print(f"assistant service started (pid {pid})")
        return 0
    if sub == "stop":
        if service.stop_daemon(cfg):
            print("assistant service stopped")
        else:
            print("service not running")
        return 0
    if sub == "restart":
        service.stop_daemon(cfg)
        pid = service.start_daemon(cfg)
        print(f"assistant service restarted (pid {pid})")
        return 0
    if sub == "status":
        if service.service_running(cfg):
            print(f"running (pid {service.service_pid(cfg)})")
        else:
            print("not running")
        return 0
    if sub == "install":
        return _service_install(cfg)
    print("usage: assist service start|stop|restart|status|run|install", file=sys.stderr)
    return 2


def _service_install(cfg: Config) -> int:
    """Write a systemd user unit and enable/start it."""
    unit_dir = Path("~/.config/systemd/user").expanduser()
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / "assistant-service.service"
    unit_path.write_text(
        "[Unit]\n"
        "Description=Assistant timers & reminders service\n"
        "\n"
        "[Service]\n"
        f'ExecStart="{sys.executable}" -m assistant service run\n'
        "Restart=on-failure\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n",
        encoding="utf-8",
    )
    print(f"wrote {unit_path}")
    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "assistant-service"],
            check=True, capture_output=True, text=True,
        )
    except Exception:
        print("systemctl unavailable or failed — start the service manually:")
        print("  systemctl --user daemon-reload")
        print("  systemctl --user enable --now assistant-service")
        print("  # or for this session only:")
        print("  assist service start")
        return 0
    print("enabled and started assistant-service (systemd user unit)")
    return 0


# ----------------------------------------------------------- slash handlers

def _handle_command(
    text: str,
    cfg: Config,
    console: Console,
    store: MemoryStore,
    engine: LLMEngine,
    session: Session,
    agent: Agent,
) -> bool:
    """Handle a slash command; return True when the repl should exit."""
    parts = text.split()
    cmd = parts[0].lstrip("/").lower()
    rest = parts[1:]

    if cmd in ("exit", "quit", "q"):
        console.markdown(_GOODBYE)
        return True
    if cmd == "help":
        console.markdown(_HELP_TEXT)
    elif cmd == "clear":
        session.clear()
        console.info("started a fresh session")
    elif cmd == "history":
        history = session.history()
        if not history:
            console.info("nothing in this session yet")
        for message in history:
            console.markdown(f"**{message['role']}:** {message['content']}")
    elif cmd == "model":
        sub = rest[0].lower() if rest else "status"
        if sub == "load":
            _ensure_engine_loaded(console, engine)
        elif sub == "unload":
            engine.unload()
        elif sub == "status":
            console.markdown(f"**{_model_status(cfg, engine)}**")
        else:
            console.warn("usage: /model load|unload|status")
    elif cmd == "memory":
        _memory_command(rest, console, store)
    elif cmd == "tasks":
        ctx = ToolContext(cfg, console, session, None, store)
        result = asyncio.run(tasks_mod.todo_list(ctx, include_done=False))
        todos = (result or {}).get("todos", [])
        if not todos:
            console.info("no open todos")
        for todo in todos:
            console.markdown(f"- [{todo.get('id')}] {todo.get('title')}")
    elif cmd == "reminders":
        ctx = ToolContext(cfg, console, session, None, store)
        result = asyncio.run(tasks_mod.reminder_list(ctx))
        reminders = (result or {}).get("reminders", [])
        if not reminders:
            console.info("no reminders scheduled")
        for reminder in reminders:
            console.markdown(
                f"- [{reminder.get('id')}] {reminder.get('at')}: "
                f"{reminder.get('message')}"
            )
    elif cmd == "timers":
        ctx = ToolContext(cfg, console, session, None, store)
        result = asyncio.run(tasks_mod.timer_status(ctx))
        timers = (result or {}).get("timers", [])
        if not timers:
            console.info("no timers running")
        for timer in timers:
            remaining = max(0, int(float(timer.get("remaining_seconds", 0))))
            minutes, seconds = divmod(remaining, 60)
            console.markdown(
                f"- [{timer.get('id')}] {timer.get('label', 'timer')}: "
                f"{minutes}m {seconds}s left"
            )
    elif cmd == "status":
        status = system_status()
        ram = status.get("ram") or {}
        gpu = status.get("gpu") or {}
        lines = [f"**{_model_status(cfg, engine)}**"]
        lines.append(
            f"- RAM: {ram.get('avail_mb')} MB free / {ram.get('total_mb')} MB"
        )
        if gpu.get("available"):
            lines.append(
                f"- GPU: {gpu.get('name')}, {gpu.get('vram_free_mb')} MB VRAM free"
            )
        else:
            lines.append(f"- GPU: {gpu.get('note', 'unavailable')}")
        console.markdown("\n".join(lines))
    else:
        console.warn(f"unknown command {cmd!r} — try /help")
    return False


def _memory_command(rest: list[str], console: Console, store: MemoryStore) -> None:
    if not rest:
        console.warn("usage: /memory list | search <text> | forget <id|text>")
        return
    sub = rest[0].lower()
    if sub == "list":
        facts = store.list_facts(limit=50)
        if not facts:
            console.info("no memories stored yet")
            return
        for fact in facts:
            console.markdown(
                f"**[{fact['id']}]** ({fact['kind']}, conf {fact['confidence']:.2f}) "
                f"{fact['text']}"
            )
    elif sub == "search":
        query = " ".join(rest[1:])
        if not query:
            console.warn("usage: /memory search <text>")
            return
        facts = store.search_facts(query, k=10)
        if not facts:
            console.info(f"nothing matches {query!r}")
            return
        for fact in facts:
            console.markdown(f"**[{fact['id']}]** {fact['text']}")
    elif sub in ("forget", "delete"):
        target = " ".join(rest[1:])
        if not target:
            console.warn("usage: /memory forget <id|text>")
            return
        try:
            ids = store.delete_fact(fact_id=int(target))
        except ValueError:
            ids = store.delete_fact(text_match=target)
        console.info(f"deleted {len(ids)} fact(s)" if ids else "nothing matched")
    else:
        console.warn("usage: /memory list | search <text> | forget <id|text>")


# ---------------------------------------------------------------- entry

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assist", description="Fully-local AI desktop agent"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("repl", help="start the interactive agent (default)")
    chat = sub.add_parser("chat", help="one-shot, non-interactive turn")
    chat.add_argument("message", nargs="+", help="message text to send")
    sub.add_parser("status", help="print system and model status (no model load)")
    sub.add_parser("setup", help="create config dirs and a default config")
    sub.add_parser(
        "selftest", help="run the self-test verification battery (launches real apps)"
    )
    service_parser = sub.add_parser(
        "service", help="manage the background timers/reminders service"
    )
    service_sub = service_parser.add_subparsers(dest="service_command")
    service_sub.add_parser("start", help="start the service as a daemon")
    service_sub.add_parser("stop", help="stop the running service")
    service_sub.add_parser("restart", help="stop then start the service")
    service_sub.add_parser("status", help="report whether the service runs")
    service_sub.add_parser("run", help="run the service in the foreground")
    service_sub.add_parser("install", help="install a systemd user unit and enable it")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``assist`` console script."""
    if argv is None:
        argv = sys.argv[1:]
    # ``selftest`` parses its own arguments (--scenario/--list/--json),
    # which argparse subparsers cannot forward faithfully.
    if argv and argv[0] == "selftest":
        from assistant.selftest import main as selftest_main

        return selftest_main(argv[1:])
    args = _build_parser().parse_args(argv)
    command = args.command or "repl"
    try:
        if command == "chat":
            return cmd_chat(args)
        if command == "status":
            return cmd_status(args)
        if command == "setup":
            return cmd_setup(args)
        if command == "service":
            return cmd_service(args)
        return cmd_repl(args)
    except KeyboardInterrupt:
        print(_GOODBYE)
        return 130


if __name__ == "__main__":
    sys.exit(main())
