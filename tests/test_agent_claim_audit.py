"""Tests for the _CLAIM_VERB_TOOLS audit mappings (play_video in the open
family, organize_workspaces in the move family)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.core.agent import _claims_unsupported_action  # noqa: E402


class ClaimVerbToolSetsTest(unittest.TestCase):
    def test_play_video_satisfies_open_family(self):
        self.assertFalse(
            _claims_unsupported_action("opened it in your browser", ["play_video"])
        )
        self.assertFalse(
            _claims_unsupported_action("I opened it for you", ["play_video"])
        )

    def test_organize_workspaces_satisfies_move_family(self):
        self.assertFalse(
            _claims_unsupported_action("moved things around", ["organize_workspaces"])
        )

    def test_move_claim_without_move_tool_still_flagged(self):
        self.assertTrue(
            _claims_unsupported_action("moved things around", ["find_video"])
        )


if __name__ == "__main__":
    unittest.main()
