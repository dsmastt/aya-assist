"""Tests for the action-aware claim audit.

Covers the open family (launch), the move family (workspace), and the
(tool, action) pair requirements introduced for play/stream claims
(video play, not search), site/app/project opens (launch url/app/
project), and timer claims (timer set/status/cancel).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.core.agent import _claim_tool_hint, _claims_unsupported_action  # noqa: E402


class ClaimVerbToolSetsTest(unittest.TestCase):
    def test_launch_satisfies_open_family(self):
        self.assertFalse(
            _claims_unsupported_action(
                "opened it in your browser", [("launch", "url")]
            )
        )
        self.assertFalse(
            _claims_unsupported_action("I opened it for you", [("launch", "app")])
        )

    def test_workspace_satisfies_move_family(self):
        self.assertFalse(
            _claims_unsupported_action("moved things around", [("workspace", "move")])
        )

    def test_move_claim_without_move_tool_still_flagged(self):
        self.assertTrue(
            _claims_unsupported_action("moved things around", [("video", "search")])
        )


class ActionAwareClaimTest(unittest.TestCase):
    def test_play_claim_needs_video_play_action(self):
        self.assertTrue(
            _claims_unsupported_action("Playing X now", [("video", "search")])
        )
        hints = _claim_tool_hint("Playing X now", [("video", "search")])
        self.assertTrue(any("video" in hint for hint in hints))
        self.assertFalse(
            _claims_unsupported_action("Playing X now", [("video", "play")])
        )

    def test_site_claim_needs_launch_url_action(self):
        self.assertTrue(
            _claims_unsupported_action(
                "opened the site in the browser", [("launch", "app")]
            )
        )
        hints = _claim_tool_hint(
            "opened the site in the browser", [("launch", "app")]
        )
        self.assertTrue(any("url" in hint for hint in hints))
        self.assertFalse(
            _claims_unsupported_action("opened the site", [("launch", "url")])
        )

    def test_app_claim_needs_launch_app_action(self):
        self.assertTrue(
            _claims_unsupported_action("opened the app for you", [("launch", "url")])
        )
        self.assertFalse(
            _claims_unsupported_action("opened the app for you", [("launch", "app")])
        )

    def test_project_claim_needs_project_action(self):
        self.assertTrue(
            _claims_unsupported_action("opened the project", [("launch", "app")])
        )
        self.assertFalse(
            _claims_unsupported_action("opened the project", [("launch", "project")])
        )
        self.assertFalse(
            _claims_unsupported_action("opened the project", [("launch", "path")])
        )

    def test_timer_set_claim_needs_timer_set_action(self):
        self.assertTrue(
            _claims_unsupported_action(
                "Setting a 20-minute timer for you", [("timer", "status")]
            )
        )
        self.assertFalse(
            _claims_unsupported_action(
                "Setting a 20-minute timer for you", [("timer", "set")]
            )
        )

    def test_timer_time_claim_needs_timer_status_action(self):
        self.assertTrue(
            _claims_unsupported_action(
                "The timer has 29 minutes left", [("timer", "set")]
            )
        )
        self.assertFalse(
            _claims_unsupported_action(
                "The timer has 29 minutes left", [("timer", "status")]
            )
        )


class SiteCountAuditTest(unittest.TestCase):
    """Multi-site claims need one launch (action: url) per site."""

    def test_two_sites_one_url_call_flagged(self):
        text = "Opened YouTube on workspace 3 and ChatGPT on workspace 5"
        self.assertTrue(
            _claims_unsupported_action(text, [("launch", "url")])
        )
        hints = _claim_tool_hint(text, [("launch", "url")])
        self.assertTrue(any("url" in hint for hint in hints))

    def test_two_sites_two_url_calls_ok(self):
        text = "Opened YouTube on workspace 3 and ChatGPT on workspace 5"
        self.assertFalse(
            _claims_unsupported_action(
                text, [("launch", "url"), ("launch", "url")]
            )
        )

    def test_one_site_one_url_call_ok(self):
        self.assertFalse(
            _claims_unsupported_action(
                "Opened YouTube on workspace 3", [("launch", "url")]
            )
        )

    def test_two_sites_no_url_call_flagged(self):
        text = "YouTube on ws 3 and ChatGPT on ws 5"
        self.assertTrue(_claims_unsupported_action(text, []))

    def test_distinct_counting_chatgpt_vs_chat_gpt(self):
        # "chatgpt" and "chat gpt" are distinct keywords, not one site.
        text = "Opened chatgpt and chat gpt on workspaces 3 and 5"
        self.assertTrue(
            _claims_unsupported_action(text, [("launch", "url")])
        )


if __name__ == "__main__":
    unittest.main()
