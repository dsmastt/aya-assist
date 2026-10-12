"""Tests for the Hyprland workspace organization policy.

Covers :func:`assistant.tools.desktop.organize_workspaces`: primary
(editor/terminal) vs secondary classification, round-robin placement onto
workspaces 1-2 / 3-5, untouched focused workspace, silent dispatches only,
and HyprError propagation. The ``hypr`` module functions are monkeypatched —
no IPC, no socket.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.tools import builtin, desktop, hypr  # noqa: E402, F401
from assistant.tools.registry import TOOLS  # noqa: E402


def client(address, cls, title, ws_id):
    return {"address": address, "class": cls, "title": title, "workspace": {"id": ws_id}}


class OrganizeWorkspacesTest(unittest.TestCase):
    def run_organize(self, clients, active_id, dispatches=None):
        dispatches = [] if dispatches is None else dispatches
        with mock.patch.object(hypr, "clients", return_value=clients), mock.patch.object(
            hypr, "active_workspace", return_value={"id": active_id}
        ), mock.patch.object(
            hypr, "dispatch", side_effect=lambda args: dispatches.append(args)
        ), mock.patch.object(
            desktop, "_uses_lua_config", return_value=False
        ):
            return desktop.organize_workspaces()

    def test_classification_and_round_robin(self):
        clients = [
            client("0x1", "Codium", "main.go", 6),
            client("0x2", "foot", "zsh", 7),
            client("0x3", "zen", "YouTube", 6),
            client("0x4", "Mpv", "video", 8),
            client("0x5", "kitty", "notes", 2),
            client("0x6", "firefox", "docs", 3),
        ]
        dispatches = []
        result = self.run_organize(clients, active_id=2, dispatches=dispatches)
        moves = {(m["class"], m["from_ws"], m["to_ws"]) for m in result["moves"]}
        self.assertIn(("Codium", 6, 1), moves)
        self.assertIn(("foot", 7, 2), moves)
        self.assertIn(("zen", 6, 3), moves)
        self.assertIn(("Mpv", 8, 4), moves)
        self.assertIn(("firefox", 3, 5), moves)
        self.assertEqual(len(dispatches), 5)
        self.assertEqual(result["note"], "no windows were closed")

    def test_primary_editor_and_terminal_classes_recognized(self):
        for cls in (
            "codium", "code", "vscode", "vscodium", "zed",
            "foot", "kitty", "alacritty", "xterm", "ghostty", "wezterm",
        ):
            dispatches = []
            result = self.run_organize(
                [client("0xa", cls, "t", 6)], active_id=9, dispatches=dispatches
            )
            self.assertEqual(result["moves"], [{"class": cls, "title": "t", "from_ws": 6, "to_ws": 1}], cls)
            self.assertEqual(dispatches, ["movetoworkspacesilent 1,address:0xa"])

    def test_focused_workspace_windows_untouched(self):
        clients = [
            client("0x1", "codium", "a", 2),
            client("0x2", "zen", "b", 2),
        ]
        dispatches = []
        result = self.run_organize(clients, active_id=2, dispatches=dispatches)
        self.assertEqual(result["moves"], [])
        self.assertEqual(dispatches, [])
        self.assertEqual(len(result["kept"]), 2)

    def test_window_already_on_target_kept(self):
        clients = [
            client("0x1", "codium", "a", 1),
            client("0x2", "codium", "b", 4),
        ]
        dispatches = []
        result = self.run_organize(clients, active_id=3, dispatches=dispatches)
        self.assertEqual(
            result["moves"], [{"class": "codium", "title": "b", "from_ws": 4, "to_ws": 2}]
        )
        self.assertEqual(dispatches, ["movetoworkspacesilent 2,address:0x2"])
        self.assertEqual(len(result["kept"]), 1)

    def test_only_silent_dispatch_commands(self):
        clients = [
            client("0x1", "zed", "a", 6),
            client("0x2", "spotify", "s", 6),
        ]
        dispatches = []
        self.run_organize(clients, active_id=1, dispatches=dispatches)
        self.assertTrue(dispatches)
        for d in dispatches:
            self.assertTrue(d.startswith("movetoworkspacesilent "), d)
        self.assertFalse(any("closewindow" in d or "kill" in d for d in dispatches))
        self.assertFalse(any(d.startswith("workspace ") for d in dispatches))

    def test_hypr_error_propagates(self):
        with mock.patch.object(hypr, "clients", side_effect=hypr.HyprError("boom")):
            with self.assertRaises(hypr.HyprError):
                desktop.organize_workspaces()


class OrganizeWorkspacesRegistrationTest(unittest.TestCase):
    def test_tool_registered_in_workspace_category(self):
        entry = TOOLS.get("organize_workspaces")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.category, "workspace")

    def test_schema_has_no_params(self):
        schema = {
            f["function"]["name"]: f["function"] for f in TOOLS.openai_schemas()
        }["organize_workspaces"]
        self.assertEqual(schema["parameters"]["properties"], {})
        self.assertNotIn("required", schema["parameters"])


if __name__ == "__main__":
    unittest.main()
