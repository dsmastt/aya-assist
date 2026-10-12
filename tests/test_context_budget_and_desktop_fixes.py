"""Tests for the context-budget/retry fixes in agent.py and the honest
terminal/tab fixes in desktop.py.

- ``start_project`` always wraps the run command as ``sh -c <full>`` (foot
  AND kitty both receive argv directly, no shell).
- ``open_path`` treats pyproject.toml / Cargo.toml / requirements.txt /
  go.mod as project markers, routing such dirs to the editor.
- ``launch_app`` reports an honest "exec+tab" result when the new window
  lands as a tab in an existing window already on the target workspace.
- ``run_once`` computes the trim budget as n_ctx - 2600 with the engine's
  real token counter when loaded, and retries a context-overflow round once.
"""

import asyncio
import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.config import Config  # noqa: E402
from assistant.core.agent import Agent, _CONTEXT_RESERVE  # noqa: E402
from assistant.core.session import Session  # noqa: E402
from assistant.llm.engine import EngineError  # noqa: E402
from assistant.memory import MemoryStore  # noqa: E402
from assistant.tools import desktop, hypr  # noqa: E402


class _NullStream:
    def __init__(self, callback=None):
        self.callback = callback
        self.chunks = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if self.callback:
            self.callback("".join(self.chunks))
        return False

    def append(self, text):
        self.chunks.append(text)


class FakeConsole:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.tool_lines = []
        self.streams = []

    def stream(self):
        return _NullStream(callback=self.streams.append)

    def tool(self, label):
        self.tool_lines.append(label)

    def markdown(self, text):
        pass

    def warn(self, msg):
        self.warnings.append(msg)

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        pass

    def ask_yes_no(self, question, default=False):
        return True

    def spinner(self, label):
        return contextlib.nullcontext()

    def hr(self):
        pass


class _CannedEngine:
    """Returns canned responses in sequence; records the messages sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, stream_cb=None, max_tokens=None):
        self.calls.append(list(messages))
        response = (
            dict(self.responses.pop(0))
            if self.responses
            else {"role": "assistant", "content": "", "tool_calls": []}
        )
        if stream_cb and response.get("content"):
            stream_cb(response["content"])
        return response


class _FlakyContextEngine:
    """Raises a context EngineError once, then answers normally."""

    def __init__(self):
        self.calls = []
        self.failures = 0

    def chat(self, messages, tools=None, stream_cb=None, max_tokens=None):
        self.calls.append(list(messages))
        if self.failures == 0:
            self.failures += 1
            raise EngineError(
                "chat completion failed: prompt exceeds n_ctx (context "
                "window exceeded). raise model.n_ctx"
            )
        response = {"role": "assistant", "content": "Done.", "tool_calls": []}
        if stream_cb:
            stream_cb(response["content"])
        return response


class _AlwaysContextEngine:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None, stream_cb=None, max_tokens=None):
        self.calls.append(list(messages))
        raise EngineError("context window exceeded: prompt too long")


class _AgentBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.cfg.paths.state_file = os.path.join(self.tmpdir.name, "state.json")
        self.store = MemoryStore(os.path.join(self.tmpdir.name, "memory.db"))

    def tearDown(self):
        self.store.close()
        self.tmpdir.cleanup()


class ContextBudgetTest(_AgentBase):
    def test_budget_is_n_ctx_minus_reserve_with_loaded_token_counter(self):
        engine = _CannedEngine(
            [{"role": "assistant", "content": "ok", "tool_calls": []}]
        )
        engine.loaded = True
        engine.token_count = lambda text: 4 * len(text)
        console = FakeConsole()
        session = Session(self.cfg, self.store)
        captured = {}
        original = session.trim_to_tokens

        def spy(max_tokens, estimate=None):
            captured["max_tokens"] = max_tokens
            captured["estimate"] = estimate
            return original(max_tokens, estimate=estimate)

        session.trim_to_tokens = spy
        agent = Agent(self.cfg, console, engine, self.store, session=session)
        asyncio.run(agent.run_once("hello"))
        self.assertEqual(captured["max_tokens"], self.cfg.model.n_ctx - _CONTEXT_RESERVE)
        self.assertIs(captured["estimate"], engine.token_count)

    def test_budget_floor_of_400_without_loaded_engine(self):
        engine = _CannedEngine(
            [{"role": "assistant", "content": "ok", "tool_calls": []}]
        )
        console = FakeConsole()
        session = Session(self.cfg, self.store)
        self.cfg.model.n_ctx = 500
        captured = {}
        original = session.trim_to_tokens

        def spy(max_tokens, estimate=None):
            captured["max_tokens"] = max_tokens
            return original(max_tokens, estimate=estimate)

        session.trim_to_tokens = spy
        agent = Agent(self.cfg, console, engine, self.store, session=session)
        asyncio.run(agent.run_once("hello"))
        self.assertEqual(captured["max_tokens"], 400)


class ContextRetryTest(_AgentBase):
    def test_context_error_retries_same_round_and_completes(self):
        engine = _FlakyContextEngine()
        console = FakeConsole()
        session = Session(self.cfg, self.store)
        session.start()
        for index in range(5):
            role = "user" if index % 2 == 0 else "assistant"
            session.add_message(role, f"history message {index}")
        agent = Agent(self.cfg, console, engine, self.store, session=session)
        asyncio.run(agent.run_once("hello there"))

        self.assertEqual(len(engine.calls), 2)
        self.assertLess(len(engine.calls[1]), len(engine.calls[0]))
        self.assertEqual(console.errors, [])
        persisted = self.store.recent_messages(session.session_id, 10)
        self.assertEqual(persisted[-1]["content"], "Done.")

    def test_second_context_error_is_fatal(self):
        engine = _AlwaysContextEngine()
        console = FakeConsole()
        session = Session(self.cfg, self.store)
        session.start()
        for index in range(5):
            session.add_message("user", f"history message {index}")
        agent = Agent(self.cfg, console, engine, self.store, session=session)
        asyncio.run(agent.run_once("hello there"))

        self.assertEqual(len(engine.calls), 2)
        self.assertTrue(console.errors)
        self.assertIn("model error", console.errors[0])


class StartProjectShellWrapTest(unittest.TestCase):
    def _cfg(self, terminal):
        return type(
            "C",
            (),
            {
                "tools": type(
                    "T", (), {"terminal": terminal, "agent_workspace": 4}
                )()
            },
        )()

    def _run(self, terminal):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(
                "# App\n\n```bash\nnpm run dev\n```\n"
            )
            calls = []

            def fake_launch(app, args="", workspace=None, silent=True, config=None):
                calls.append({"app": app, "args": args, "workspace": workspace})
                return {"ok": True, "method": "windowrule", "app": app}

            with mock.patch.object(desktop, "launch_app", side_effect=fake_launch):
                result = desktop.start_project(
                    str(root), config=self._cfg(terminal)
                )
        return result, calls

    def test_foot_uses_sh_c_wrapper(self):
        result, calls = self._run("foot")
        self.assertEqual(result["status"], "started")
        self.assertEqual(result["terminal"], "foot")
        self.assertEqual(result["command"], "npm run dev")
        self.assertEqual(result["workspace"], 4)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["args"].startswith("sh -c"))
        self.assertIn("cd ", calls[0]["args"])
        self.assertIn("npm run dev", calls[0]["args"])

    def test_kitty_uses_sh_c_wrapper_too(self):
        result, calls = self._run("kitty")
        self.assertEqual(result["status"], "started")
        self.assertEqual(result["terminal"], "kitty")
        self.assertTrue(calls[0]["args"].startswith("sh -c"))
        self.assertIn("cd ", calls[0]["args"])
        self.assertIn("npm run dev", calls[0]["args"])


class OpenPathMarkerTest(unittest.TestCase):
    def _open(self, root):
        calls = []

        def fake_launch(
            app, args="", workspace=None, silent=True, config=None, hint=None
        ):
            calls.append({"app": app, "args": args})
            return {"ok": True, "app": app}

        with mock.patch.object(desktop, "launch_app", side_effect=fake_launch):
            result = desktop.open_path(str(root))
        return result, calls

    def test_pyproject_toml_routes_to_editor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname = 'x'\n")
            result, calls = self._open(root)
        self.assertEqual(result["method"], "editor")
        self.assertEqual(calls[0]["app"], "codium")

    def test_other_project_markers_route_to_editor(self):
        for marker in ("Cargo.toml", "requirements.txt", "go.mod"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / marker).write_text("x\n")
                result, _ = self._open(root)
            self.assertEqual(result["method"], "editor", marker)

    def test_loose_files_open_file_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.txt").write_text("hi\n")
            result, calls = self._open(root)
        self.assertEqual(result["method"], "file-manager")
        self.assertEqual(calls[0]["app"], "dolphin")


def client(address, cls, title, ws_id):
    return {
        "address": address,
        "class": cls,
        "title": title,
        "workspace": {"id": ws_id},
    }


class SingleInstanceTabFallbackTest(unittest.TestCase):
    def test_existing_window_on_target_workspace_is_honest_tab(self):
        existing = [client("0xzen0", "zen", "OpenAI Chat", 3)]
        with mock.patch.object(desktop, "resolve_binary", return_value="/usr/bin/zen-bin"), \
                mock.patch.object(desktop, "_find_new_client", return_value=None), \
                mock.patch.object(hypr, "clients", return_value=existing), \
                mock.patch.object(desktop, "_dispatch", return_value="ok"):
            result = desktop.launch_app(
                "zen", "https://example.com", workspace=3
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["method"], "exec+tab")
        self.assertIn(
            "existing browser window already on workspace 3", result["note"]
        )
        self.assertEqual(result["workspace"], 3)

    def test_window_on_other_workspace_is_not_enough(self):
        elsewhere = [client("0xzen0", "zen", "OpenAI Chat", 4)]
        with mock.patch.object(desktop, "resolve_binary", return_value="/usr/bin/zen-bin"), \
                mock.patch.object(desktop, "_find_new_client", return_value=None), \
                mock.patch.object(hypr, "clients", return_value=elsewhere), \
                mock.patch.object(desktop, "_dispatch", return_value="ok"):
            result = desktop.launch_app(
                "zen", "https://example.com", workspace=3
            )
        self.assertFalse(result["ok"])
        self.assertIn(
            "no window could be placed on workspace 3", result["note"]
        )

    def test_no_existing_window_is_honest_failure(self):
        with mock.patch.object(desktop, "resolve_binary", return_value="/usr/bin/zen-bin"), \
                mock.patch.object(desktop, "_find_new_client", return_value=None), \
                mock.patch.object(hypr, "clients", return_value=[]), \
                mock.patch.object(desktop, "_dispatch", return_value="ok"):
            result = desktop.launch_app(
                "zen", "https://example.com", workspace=3
            )
        self.assertFalse(result["ok"])
        self.assertIn(
            "no window could be placed on workspace 3", result["note"]
        )


if __name__ == "__main__":
    unittest.main()
