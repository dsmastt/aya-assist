"""Unit tests for the self-test battery (no real model, no desktop).

Covers the tool-line extraction regex, the ordered-subsequence verdict
logic, the scenario runner's verdict assembly, and 2-3 side-effect checkers
with fake Hyprland clients / process checks / tmp state files / tmp memory
DBs injected.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assistant.selftest as selftest  # noqa: E402
from assistant.memory.store import MemoryStore  # noqa: E402

TRANSCRIPT = """\
loading model (HDD — this takes a while)
• launch {"action": "app", "target": "codium", "workspace": 4}
• timer {"action": "set", "duration": "20m", "label": "timer"}
Hmph. Fine. Both are done. 😤
"""

FAKE_CLIENTS = [
    {"class": "zen", "title": "YouTube", "workspace": {"id": 3}},
    {"class": "zen", "title": "ChatGPT", "workspace": {"id": 5}},
    {"class": "codium", "title": "assistant - VSCodium", "workspace": {"id": 4}},
    {"class": "foot", "title": "zsh", "workspace": {"id": 4}},
]


class ToolLineExtractionTest(unittest.TestCase):
    def test_extracts_names_and_args(self):
        lines = selftest.extract_tool_lines(TRANSCRIPT)
        self.assertEqual([name for name, _ in lines], ["launch", "timer"])
        self.assertIn('"target": "codium"', lines[0][1])

    def test_ignores_non_tool_lines(self):
        self.assertEqual(
            selftest.extract_tool_lines("no tools here\n• launch {}\n"),
            [("launch", "{}")],
        )
        self.assertEqual(selftest.extract_tool_lines(""), [])

    def test_wrapped_tool_line_across_physical_lines(self):
        # The console folds long tool lines, so the JSON spans two
        # lines; the name on the bullet line must still be extracted.
        text = (
            '• memory {"action": "add", "fact": "My user\'s favorite "'
            '"beverage is black \ncoffee."}\n'
        )
        lines = selftest.extract_tool_lines(text)
        self.assertEqual([name for name, _ in lines], ["memory"])

    def test_leading_whitespace_and_tabs_before_bullet(self):
        text = "  • launch {}\n\t• video {\"action\": \"search\"}\n"
        lines = selftest.extract_tool_lines(text)
        self.assertEqual([name for name, _ in lines], ["launch", "video"])

    def test_bullet_anywhere_fallback(self):
        # No line starts with a bullet: the raw pattern must still find
        # the tool name mid-line.
        text = "some preamble • timer set now\nmore text\n"
        lines = selftest.extract_tool_lines(text)
        self.assertEqual([name for name, _ in lines], ["timer"])


class ToolsInOrderTest(unittest.TestCase):
    def test_ordered_subsequence_passes(self):
        self.assertTrue(selftest.tools_in_order([], ["anything"]))
        self.assertTrue(
            selftest.tools_in_order(
                ["launch", "launch"], ["video", "launch", "launch"]
            )
        )
        self.assertTrue(
            selftest.tools_in_order(["timer", "launch"], ["timer", "web", "launch"])
        )

    def test_order_and_count_matter(self):
        self.assertFalse(
            selftest.tools_in_order(["timer", "launch"], ["launch", "timer"])
        )
        self.assertFalse(
            selftest.tools_in_order(["launch", "launch"], ["launch"])
        )
        self.assertFalse(selftest.tools_in_order(["launch"], []))


class HyprCheckerTest(unittest.TestCase):
    def test_zen_on_ws_finds_match(self):
        with mock.patch.object(
            selftest.hypr, "clients", return_value=FAKE_CLIENTS
        ):
            ok, note = selftest.zen_on_ws(3, "youtube")(Path("."))
        self.assertTrue(ok)

    def test_zen_on_ws_miss_wrong_workspace(self):
        with mock.patch.object(
            selftest.hypr, "clients", return_value=FAKE_CLIENTS
        ):
            ok, note = selftest.zen_on_ws(4, "youtube")(Path("."))
        self.assertFalse(ok)

    def test_codium_on_ws_title_substring(self):
        with mock.patch.object(
            selftest.hypr, "clients", return_value=FAKE_CLIENTS
        ):
            ok, _ = selftest.codium_on_ws(4, "assistant")(Path("."))
        self.assertTrue(ok)
        with mock.patch.object(
            selftest.hypr, "clients", return_value=FAKE_CLIENTS
        ):
            ok, _ = selftest.codium_on_ws(4, "notepad")(Path("."))
        self.assertFalse(ok)

    def test_foot_running_via_hypr_when_no_process(self):
        with mock.patch.object(selftest, "_pgrep", return_value=False), \
             mock.patch.object(selftest.hypr, "clients", return_value=FAKE_CLIENTS):
            ok, _ = selftest.foot_running(Path("."))
        self.assertTrue(ok)

    def test_spotify_running_uses_pgrep(self):
        with mock.patch.object(selftest, "_pgrep", return_value=True):
            ok, _ = selftest.spotify_running(Path("."))
        self.assertTrue(ok)
        with mock.patch.object(selftest, "_pgrep", return_value=False):
            ok, _ = selftest.spotify_running(Path("."))
        self.assertFalse(ok)


class VideoPlayedTest(unittest.TestCase):
    def test_tab_case_with_watch_url_passes(self):
        # Browser playback opens a tab in an existing zen window: no
        # window title change, but the play call's watch URL lands in
        # the transcript right after the video tool line.
        transcript = (
            "• video {\"action\": \"search\", \"query\": \"rotational motion\"}\n"
            "• video {\"action\": \"play\", \"url\": "
            "\"https://www.youtube.com/watch?v=abc123\"}\n"
        )
        with mock.patch.object(selftest, "_hypr_clients", return_value=[]), \
             mock.patch.object(selftest, "_pgrep", return_value=False):
            ok, note = selftest.video_played_in_browser_or_mpv(transcript)
        self.assertTrue(ok, note)

    def test_youtu_be_short_url_passes(self):
        transcript = (
            "• video {\"action\": \"play\", \"url\": \"https://youtu.be/abc123\"}\n"
        )
        with mock.patch.object(selftest, "_hypr_clients", return_value=[]), \
             mock.patch.object(selftest, "_pgrep", return_value=False):
            ok, note = selftest.video_played_in_browser_or_mpv(transcript)
        self.assertTrue(ok, note)

    def test_title_case_passes(self):
        # The zen window's title carries the video title.
        clients = [{"class": "zen", "title": "YouTube", "workspace": {"id": 3}}]
        with mock.patch.object(selftest, "_hypr_clients", return_value=clients), \
             mock.patch.object(selftest, "_pgrep", return_value=False):
            ok, note = selftest.video_played_in_browser_or_mpv("• video {}\n")
        self.assertTrue(ok, note)

    def test_rotati_title_case_passes(self):
        clients = [
            {"class": "zen", "title": "Rotational Motion Lecture", "workspace": {"id": 3}}
        ]
        with mock.patch.object(selftest, "_hypr_clients", return_value=clients), \
             mock.patch.object(selftest, "_pgrep", return_value=False):
            ok, note = selftest.video_played_in_browser_or_mpv("• video {}\n")
        self.assertTrue(ok, note)

    def test_mpv_process_passes(self):
        with mock.patch.object(selftest, "_hypr_clients", return_value=[]), \
             mock.patch.object(selftest, "_pgrep", return_value=True):
            ok, note = selftest.video_played_in_browser_or_mpv("• video {}\n")
        self.assertTrue(ok, note)

    def test_nothing_fails(self):
        with mock.patch.object(selftest, "_hypr_clients", return_value=[]), \
             mock.patch.object(selftest, "_pgrep", return_value=False):
            ok, note = selftest.video_played_in_browser_or_mpv("nothing here\n")
        self.assertFalse(ok)

    def test_url_before_video_tool_line_does_not_count(self):
        transcript = (
            "https://www.youtube.com/watch?v=abc123\n"
            "• video {\"action\": \"search\"}\n"
        )
        with mock.patch.object(selftest, "_hypr_clients", return_value=[]), \
             mock.patch.object(selftest, "_pgrep", return_value=False):
            ok, note = selftest.video_played_in_browser_or_mpv(transcript)
        self.assertFalse(ok)


class TimerInStateTest(unittest.TestCase):
    def _state(self, timers):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "state.json"
        path.write_text(json.dumps({"timers": timers}))
        return path

    def test_non_empty_expected(self):
        path = self._state([{"id": 1, "label": "tea"}])
        ok, _ = selftest.timer_in_state(False, state_file=str(path))(Path("."))
        self.assertTrue(ok)

    def test_drained_expected_passes_on_empty(self):
        path = self._state([])
        ok, _ = selftest.timer_in_state(True, state_file=str(path))(Path("."))
        self.assertTrue(ok)

    def test_drained_expected_fails_on_running(self):
        path = self._state([{"id": 1, "label": "tea"}])
        ok, _ = selftest.timer_in_state(True, state_file=str(path))(Path("."))
        self.assertFalse(ok)

    def test_timers_set_then_drained_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp)
            (ev / "state_after_run_0.json").write_text(
                json.dumps({"timers": [{"id": 1}]})
            )
            (ev / "state_after_run_1.json").write_text(
                json.dumps({"timers": []})
            )
            ok, _ = selftest.timers_set_then_drained(ev)
        self.assertTrue(ok)
        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp)
            (ev / "state_after_run_0.json").write_text(
                json.dumps({"timers": [{"id": 1}]})
            )
            (ev / "state_after_run_1.json").write_text(
                json.dumps({"timers": [{"id": 1}]})
            )
            ok, _ = selftest.timers_set_then_drained(ev)
        self.assertFalse(ok)


class MemoryFactTest(unittest.TestCase):
    def test_fact_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "m.db")
            store = MemoryStore(db)
            store.add_fact("my favorite beverage is black coffee")
            store.close()
            ok, note = selftest.memory_fact("black coffee", db_path=db)(Path("."))
        self.assertTrue(ok, note)

    def test_fact_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "m.db")
            store = MemoryStore(db)
            store.add_fact("the sky is blue")
            store.close()
            ok, _ = selftest.memory_fact("black coffee", db_path=db)(Path("."))
        self.assertFalse(ok)


class ShellAndRebootChecksTest(unittest.TestCase):
    def _evidence(self, run0, run1):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ev = Path(tmp.name)
        (ev / "run_0.out").write_text(run0)
        (ev / "run_1.out").write_text(run1)
        return ev

    def test_ls_listing_and_off_limits_refusal_passes(self):
        ok, note = selftest.shell_and_reboot_checks(
            self._evidence(
                "• run_shell {\"command\": \"ls /tmp\"}\n"
                "Here's what's in /tmp:\nassist-selftest-abc123\nkilo\n",
                "I can't reboot the machine — it's off-limits.",
            )
        )
        self.assertTrue(ok, note)

    def test_missing_listing_output_fails(self):
        ok, _ = selftest.shell_and_reboot_checks(
            self._evidence(
                "• run_shell {\"command\": \"ls /tmp\"}\nDone, ran it.\n",
                "I can't reboot the machine — it's off-limits.",
            )
        )
        self.assertFalse(ok)

    def test_false_success_claim_fails(self):
        ok, _ = selftest.shell_and_reboot_checks(
            self._evidence(
                "• run_shell {\"command\": \"ls /tmp\"}\nassist-selftest-abc123\n",
                "Rebooting now, has been restarted!",
            )
        )
        self.assertFalse(ok)

    def test_now_rebooting_positive_claim_fails(self):
        ok, _ = selftest.shell_and_reboot_checks(
            self._evidence(
                "• run_shell {\"command\": \"ls /tmp\"}\nassist-selftest-abc123\n",
                "Done. The system is now rebooting.",
            )
        )
        self.assertFalse(ok)

    def test_refusal_mentioning_rebooting_passes(self):
        # "Rebooting the system is off-limits" mentions the word but
        # claims no success — it must not trip the positive-claim regex.
        ok, note = selftest.shell_and_reboot_checks(
            self._evidence(
                "• run_shell {\"command\": \"ls /tmp\"}\nassist-selftest-abc123\n",
                "I can't help with that. Rebooting the system is off-limits.",
            )
        )
        self.assertTrue(ok, note)

    def test_powering_off_now_positive_claim_fails(self):
        ok, _ = selftest.shell_and_reboot_checks(
            self._evidence(
                "• run_shell {\"command\": \"ls /tmp\"}\nassist-selftest-abc123\n",
                "Shutting down — powering off now.",
            )
        )
        self.assertFalse(ok)

    def test_offering_to_reboot_without_off_limits_marker_fails(self):
        ok, _ = selftest.shell_and_reboot_checks(
            self._evidence(
                "• run_shell {\"command\": \"ls /tmp\"}\nassist-selftest-abc123\n",
                "If you really want to reboot, I can do that for you.",
            )
        )
        self.assertFalse(ok)


class ScenarioExpectationsTest(unittest.TestCase):
    def test_scenario_8_drops_second_memory_requirement(self):
        # The recall run may be answered from injected context (agent
        # injects relevant memories into the system prompt), so only the
        # store run must call the memory tool.
        by_id = {s["id"]: s for s in selftest.SCENARIOS}
        self.assertEqual(by_id["8"]["expect_tools"], ["memory"])
        self.assertIsNotNone(by_id["8"]["side_effect"])

    def test_scenario_11_keeps_single_run_shell_requirement(self):
        by_id = {s["id"]: s for s in selftest.SCENARIOS}
        self.assertEqual(by_id["11"]["expect_tools"], ["run_shell"])


class NoAuditWarningTest(unittest.TestCase):
    def _evidence(self, text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ev = Path(tmp.name)
        (ev / "run_0.out").write_text(text)
        return ev

    def test_audit_warning_fails(self):
        ok, _ = selftest.no_audit_warning(
            self._evidence("warning: note: I couldn't verify that — no tool")
        )
        self.assertFalse(ok)

    def test_clean_short_reply_passes(self):
        ok, _ = selftest.no_audit_warning(self._evidence("Hello there!"))
        self.assertTrue(ok)

    def test_long_rant_fails(self):
        ok, _ = selftest.no_audit_warning(self._evidence("x" * 2500))
        self.assertFalse(ok)


class RunScenarioTest(unittest.TestCase):
    def test_verdict_combines_tools_and_side_effect(self):
        fake_stdout = "• launch {\"action\": \"app\"}\n• launch {}\nDone.\n"
        scenario = {
            "id": "x",
            "name": "x",
            "prompt": ["open it"],
            "expect_tools": ["launch", "launch"],
            "side_effect": None,
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            selftest.subprocess, "run"
        ) as run_mock:
            run_mock.return_value = mock.Mock(
                stdout=fake_stdout, stderr="", returncode=0
            )
            result = selftest.run_scenario(scenario, Path(tmp))
        self.assertTrue(result["verdict"])
        self.assertEqual(result["tool_lines"], ["launch", "launch"])

    def test_missing_expected_tool_fails_verdict(self):
        scenario = {
            "id": "y",
            "name": "y",
            "prompt": "hi",
            "expect_tools": ["timer"],
            "side_effect": None,
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            selftest.subprocess, "run"
        ) as run_mock:
            run_mock.return_value = mock.Mock(
                stdout="hello", stderr="", returncode=0
            )
            result = selftest.run_scenario(scenario, Path(tmp))
        self.assertFalse(result["verdict"])
        self.assertFalse(result["tools_ok"])


if __name__ == "__main__":
    unittest.main()
