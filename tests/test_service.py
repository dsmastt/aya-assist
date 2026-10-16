"""Tests for the standalone timers/reminders/notifications service.

Covers the flavor suffix pool, the notify backend selection (caelestia
preferred, notify-send fallback, both-fail -> None), the timer-query claim
regex in the agent audit pipeline, and CLI service function smoke checks.
No GPU, no model load, no spawned daemons.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.config import Config  # noqa: E402
from assistant.core import service  # noqa: E402
from assistant.core.agent import (  # noqa: E402
    _TIME_CLAIM_RE,
    _claim_tool_hint,
    _looks_like_action_claim,
)


class FlavorTest(unittest.TestCase):
    def test_flavored_keeps_base_prefix_and_pool_suffix(self):
        bases = (
            "Timer done: tea",
            "Reminder: finish your assignment today",
        )
        for base in bases:
            for _ in range(20):
                out = service.flavored(base)
                self.assertTrue(out.startswith(base + " "), out)
                suffix = out[len(base) + 1:]
                self.assertIn(suffix, service._FLAVOR_SUFFIXES, out)
            self.assertTrue(service.flavored(base).startswith(base.split(":")[0] + ":"))

    def test_flavored_is_always_str(self):
        self.assertIsInstance(service.flavored("Timer done: x"), str)


class NotifyPathTest(unittest.TestCase):
    def test_caelestia_preferred_when_successful(self):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((list(argv), kwargs))
            result = mock.Mock()
            result.returncode = 0
            return result

        with mock.patch.object(service.subprocess, "run", side_effect=fake_run):
            sent_via = service.notify("Reminder", "finish it")
        self.assertEqual(sent_via, "caelestia")
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0][0],
            ["caelestia", "shell", "toaster", "info", "Reminder", "finish it"],
        )
        self.assertEqual(calls[0][1]["timeout"], 5)

    def test_falls_back_to_notify_send(self):
        def fake_run(argv, **kwargs):
            if argv[0] == "caelestia":
                raise OSError("no caelestia")
            result = mock.Mock()
            result.returncode = 0
            return result

        with mock.patch.object(service.subprocess, "run", side_effect=fake_run):
            sent_via = service.notify("Timer done", "tea")
        self.assertEqual(sent_via, "notify-send")

    def test_both_paths_fail_returns_none(self):
        def fake_run(argv, **kwargs):
            if argv[0] == "caelestia":
                raise OSError("gone")
            result = mock.Mock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "boom"
            return result

        with mock.patch.object(service.subprocess, "run", side_effect=fake_run):
            self.assertIsNone(service.notify("x", "y"))


class AgentTimeClaimTest(unittest.TestCase):
    def test_time_remaining_patterns_hint_timer(self):
        for text in (
            "The timer has 29 minutes left",
            "it has 5 minutes remaining",
            "what's the time on the timer",
            "has been running for 10 minutes",
            "remaining time is 3 minutes",
            "elapsed time so far is 2 minutes",
            "the timer status looks fine",
        ):
            hints = _claim_tool_hint(text, [])
            self.assertTrue(any("timer" in hint for hint in hints), text)

    def test_plain_fact_gets_no_hint(self):
        self.assertEqual(_claim_tool_hint("Paris is the capital", []), [])

    def test_regex_matches_time_claims_directly(self):
        self.assertTrue(_TIME_CLAIM_RE.search("The timer has 29 minutes left"))
        self.assertTrue(_TIME_CLAIM_RE.search("it has 5 minutes remaining"))
        self.assertTrue(_TIME_CLAIM_RE.search("what's the time on the timer"))
        self.assertFalse(_TIME_CLAIM_RE.search("Paris is the capital"))

    def test_looks_like_action_claim_detects_time_claims(self):
        self.assertTrue(_looks_like_action_claim("The timer has 29 minutes left"))
        self.assertFalse(_looks_like_action_claim("Paris is the capital of France"))


class CliServiceSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.cfg.paths.state_file = os.path.join(self.tmpdir.name, "state.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_service_running_returns_bool(self):
        self.assertIsInstance(service.service_running(self.cfg), bool)
        self.assertIsInstance(service.service_running(), bool)

    def test_service_running_false_without_pidfile(self):
        self.assertFalse(service.service_running(self.cfg))
        self.assertIsNone(service.service_pid(self.cfg))

    def test_service_running_true_for_live_pid(self):
        pidfile = Path(self.cfg.paths.state_file).parent / "service.pid"
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text(str(os.getpid()), encoding="utf-8")
        self.assertTrue(service.service_running(self.cfg))
        self.assertEqual(service.service_pid(self.cfg), os.getpid())

    def test_service_running_false_for_dead_pid(self):
        pidfile = Path(self.cfg.paths.state_file).parent / "service.pid"
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("99999999", encoding="utf-8")
        self.assertFalse(service.service_running(self.cfg))

    def test_stop_daemon_without_pidfile_returns_false(self):
        self.assertFalse(service.stop_daemon(self.cfg))

    def test_cli_service_status_subcommand_exists(self):
        import assistant.cli as cli

        parser = cli._build_parser()
        args = parser.parse_args(["service", "status"])
        self.assertEqual(args.service_command, "status")
        for name in ("start", "stop", "restart", "run", "install"):
            parsed = parser.parse_args(["service", name])
            self.assertEqual(parsed.service_command, name)


if __name__ == "__main__":
    unittest.main()
