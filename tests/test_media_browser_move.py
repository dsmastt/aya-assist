"""Tests for the browser path of play_video.

open_url now places the new browser window itself (exec+move for
single-instance browsers), so play_video no longer calls move_window: it
passes the workspace through to open_url and surfaces open_url's note when
the launch did not fully succeed (ok=False).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.tools import media  # noqa: E402


class _FakeConfig:
    def __init__(self, browser=None, media_player="browser"):
        self.tools = type("Tools", (), {"browser": browser, "media_player": media_player})()


class PlayVideoBrowserOpenUrlTest(unittest.TestCase):
    def test_workspace_passed_through_to_open_url(self):
        cfg = _FakeConfig(browser="zen")
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ), mock.patch.object(
            media.desktop, "open_url",
            return_value={
                "method": "exec+move",
                "ok": True,
                "note": "new window moved to workspace 2 (0x1)",
            },
        ) as open_url:
            result = media.play_video("v", workspace=2, config=cfg)
        open_url.assert_called_once_with("https://x.example/v", workspace=2, config=cfg)
        self.assertEqual(result["workspace"], 2)
        self.assertEqual(result["method"], "exec+move")

    def test_browser_path_never_calls_move_window(self):
        cfg = _FakeConfig(browser="zen")
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ), mock.patch.object(
            media.desktop, "open_url", return_value={"method": "exec+move", "ok": True}
        ), mock.patch.object(media.desktop, "move_window") as move:
            media.play_video("v", workspace=1, config=cfg)
        move.assert_not_called()

    def test_open_url_failure_note_surfaced(self):
        cfg = _FakeConfig(browser="zen")
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ), mock.patch.object(
            media.desktop, "open_url",
            return_value={
                "method": "exec+move",
                "ok": False,
                "note": (
                    "launched, but the new window could not be located "
                    "(it may have opened inside the existing instance's window)"
                ),
            },
        ):
            result = media.play_video("v", workspace=3, config=cfg)
        self.assertIn("new window could not be located", result["note"])

    def test_successful_open_keeps_note_none(self):
        cfg = _FakeConfig(browser="zen")
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ), mock.patch.object(
            media.desktop, "open_url",
            return_value={"method": "exec+move", "ok": True, "note": "moved"},
        ):
            result = media.play_video("v", workspace=3, config=cfg)
        self.assertIsNone(result["note"])

    def test_no_workspace_still_opens_url(self):
        cfg = _FakeConfig(browser="zen")
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ), mock.patch.object(
            media.desktop, "open_url", return_value={"method": "exec", "ok": True}
        ) as open_url:
            result = media.play_video("v", config=cfg)
        open_url.assert_called_once_with(
            "https://x.example/v", workspace=None, config=cfg
        )
        self.assertIsNone(result["note"])


if __name__ == "__main__":
    unittest.main()
