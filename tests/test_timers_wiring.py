"""Tests for the timer tool wiring: registry, scheduler drain, CLI command.

Covers the plumbing added around ``tasks.timer_*``: the merged ``timer``
registry entry (actions set/status/cancel) in ``builtin``, the
``ReminderScheduler`` timer drain + notification path, and the ``/timers``
slash command. No GPU, no model load.
"""

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.config import Config  # noqa: E402
from assistant.core.background import ReminderScheduler  # noqa: E402
from assistant.tools import builtin, notify, tasks  # noqa: E402, F401
from assistant.tools.registry import TOOLS, ToolContext  # noqa: E402


def run(coro):
    return asyncio.run(coro)


class TimerToolRegistrationTest(unittest.TestCase):
    def test_timer_tool_registered_in_tasks_category(self):
        entry = TOOLS.get("timer")
        self.assertIsNotNone(entry, "timer not registered")
        self.assertEqual(entry.category, "tasks")

    def test_timer_set_schema(self):
        schema = {
            f["function"]["name"]: f["function"]
            for f in TOOLS.openai_schemas()
        }["timer"]
        self.assertIn("countdown timer", schema["description"])
        props = schema["parameters"]["properties"]
        self.assertEqual(props["action"]["enum"], ["set", "status", "cancel"])
        self.assertEqual(props["label"]["type"], "string")
        self.assertEqual(props["duration"]["type"], "string")
        self.assertEqual(schema["parameters"]["required"], ["action"])

    def test_timer_status_schema_requires_only_action(self):
        schema = {
            f["function"]["name"]: f["function"]
            for f in TOOLS.openai_schemas()
        }["timer"]
        props = schema["parameters"]["properties"]
        self.assertIn("status", props["action"]["enum"])
        required = schema["parameters"].get("required", [])
        self.assertEqual(required, ["action"])

    def test_timer_cancel_schema(self):
        schema = {
            f["function"]["name"]: f["function"]
            for f in TOOLS.openai_schemas()
        }["timer"]
        self.assertEqual(
            schema["parameters"]["properties"]["timer_id"]["type"], "integer"
        )
        self.assertNotIn("timer_id", schema["parameters"].get("required", []))


class TimerToolInvocationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.cfg.paths.state_file = os.path.join(self.tmpdir.name, "state.json")
        self.ctx = ToolContext(config=self.cfg)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_timer_round_trip_through_registry(self):
        result = run(
            TOOLS.get("timer").run(
                self.ctx, {"action": "set", "label": "tea", "duration": "90s"}
            )
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.data["label"], "tea")
        self.assertEqual(result.data["duration_seconds"], 90)

        status = run(TOOLS.get("timer").run(self.ctx, {"action": "status"}))
        self.assertTrue(status.ok)
        timers = status.data["timers"]
        self.assertEqual(len(timers), 1)
        self.assertEqual(timers[0]["label"], "tea")
        self.assertEqual(timers[0]["status"], "running")

        cancelled = run(
            TOOLS.get("timer").run(
                self.ctx, {"action": "cancel", "timer_id": timers[0]["id"]}
            )
        )
        self.assertTrue(cancelled.ok)
        self.assertEqual(cancelled.data, {"cancelled": timers[0]["id"]})

        after = run(TOOLS.get("timer").run(self.ctx, {"action": "status"}))
        self.assertEqual(after.data["timers"], [])

    def test_timer_set_requires_duration(self):
        result = run(
            TOOLS.get("timer").run(self.ctx, {"action": "set", "label": "x"})
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"error": "action 'set' requires duration"})


class _NullConsole:
    def __init__(self):
        self.infos = []
        self.markdown_lines = []
        self.warnings = []

    def info(self, msg):
        self.infos.append(msg)

    def markdown(self, text):
        self.markdown_lines.append(text)

    def warn(self, msg):
        self.warnings.append(msg)

    def spinner(self, label):
        return contextlib.nullcontext()


class SchedulerTimerDrainTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.cfg.paths.state_file = os.path.join(self.tmpdir.name, "state.json")
        self.console = _NullConsole()
        self.scheduler = ReminderScheduler(self.cfg, self.console)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_state(self, timers):
        data = {"todos": [], "reminders": [], "timers": timers}
        path = os.path.join(self.tmpdir.name, "state.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def _read_state(self):
        path = os.path.join(self.tmpdir.name, "state.json")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def test_poll_interval_is_10_seconds(self):
        self.assertEqual(ReminderScheduler.POLL_SECONDS, 10.0)

    def test_due_timers_drained_and_notified(self):
        now = datetime.now().astimezone()
        self._write_state(
            [
                {
                    "id": 1,
                    "label": "test-ping",
                    "end_at": (now - timedelta(seconds=1)).isoformat(timespec="seconds"),
                    "created_at": (now - timedelta(minutes=1)).isoformat(timespec="seconds"),
                },
                {
                    "id": 2,
                    "label": "later",
                    "end_at": (now + timedelta(minutes=5)).isoformat(timespec="seconds"),
                    "created_at": now.isoformat(timespec="seconds"),
                },
            ]
        )
        sent = []

        async def fake_notify(ctx, message):
            sent.append(message)
            return {"sent_via": "test"}

        with mock.patch.object(notify, "notify_user", side_effect=fake_notify):
            self.scheduler._tick()

        self.assertEqual(sent, ["Timer done: test-ping"])
        remaining = self._read_state()["timers"]
        self.assertEqual([t["id"] for t in remaining], [2])

    def test_no_due_timers_fires_nothing(self):
        now = datetime.now().astimezone()
        self._write_state(
            [
                {
                    "id": 3,
                    "label": "later",
                    "end_at": (now + timedelta(minutes=5)).isoformat(timespec="seconds"),
                    "created_at": now.isoformat(timespec="seconds"),
                }
            ]
        )
        sent = []

        async def fake_notify(ctx, message):
            sent.append(message)
            return {"sent_via": "test"}

        with mock.patch.object(notify, "notify_user", side_effect=fake_notify):
            self.scheduler._tick()

        self.assertEqual(sent, [])
        self.assertEqual(len(self._read_state()["timers"]), 1)


class CliTimersCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.cfg.paths.state_file = os.path.join(self.tmpdir.name, "state.json")
        self.console = _NullConsole()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_timers_command_reports_remaining_time(self):
        import assistant.cli as cli

        ctx = ToolContext(config=self.cfg)
        run(tasks.timer_set(ctx, label="coffee", duration="2m"))

        handled = cli._handle_command(
            "/timers", self.cfg, self.console, None, None, None, None
        )
        self.assertFalse(handled)
        self.assertEqual(len(self.console.markdown_lines), 1)
        self.assertIn("coffee", self.console.markdown_lines[0])
        self.assertIn("left", self.console.markdown_lines[0])

    def test_timers_command_with_no_timers(self):
        import assistant.cli as cli

        cli._handle_command("/timers", self.cfg, self.console, None, None, None, None)
        self.assertEqual(self.console.markdown_lines, [])
        self.assertIn("no timers running", self.console.infos)

    def test_help_text_mentions_timers(self):
        import assistant.cli as cli

        self.assertIn("/timers", cli._HELP_TEXT)


if __name__ == "__main__":
    unittest.main()
