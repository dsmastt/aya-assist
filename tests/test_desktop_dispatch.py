"""Tests for Lua-aware window-move dispatch routing in desktop tools.

Covers :func:`assistant.tools.desktop._dispatch` translation of
``movetoworkspacesilent``/``movetoworkspace`` into ``hl.dsp.window.move`` Lua
calls, error-reply raising in ``move_window``/``organize_workspaces``, and
that both tools route through the translated dispatcher. No IPC, no socket.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.tools import desktop, hypr  # noqa: E402


def client(address, cls, title, ws_id):
    return {"address": address, "class": cls, "title": title, "workspace": {"id": ws_id}}


class DispatchTranslationTest(unittest.TestCase):
    def test_movetoworkspacesilent_lua_form(self):
        calls = []
        with mock.patch.object(desktop, "_uses_lua_config", return_value=True), \
                mock.patch.object(hypr, "dispatch", side_effect=lambda a: calls.append(a) or "ok"):
            reply = desktop._dispatch("movetoworkspacesilent 2,address:0xabc")
        self.assertEqual(reply, "ok")
        self.assertEqual(
            calls,
            ['hl.dsp.window.move({window = "address:0xabc", workspace = "2", follow = false})'],
        )

    def test_movetoworkspace_lua_form(self):
        calls = []
        with mock.patch.object(desktop, "_uses_lua_config", return_value=True), \
                mock.patch.object(hypr, "dispatch", side_effect=lambda a: calls.append(a) or "ok"):
            desktop._dispatch("movetoworkspace 4,class:firefox")
        self.assertEqual(
            calls,
            ['hl.dsp.window.move({window = "class:firefox", workspace = "4", follow = false})'],
        )

    def test_class_selector_kept_not_address_prefixed(self):
        calls = []
        with mock.patch.object(desktop, "_uses_lua_config", return_value=True), \
                mock.patch.object(hypr, "dispatch", side_effect=lambda a: calls.append(a) or "ok"):
            desktop._dispatch("movetoworkspacesilent 3,class:zen")
        self.assertEqual(
            calls,
            ['hl.dsp.window.move({window = "class:zen", workspace = "3", follow = false})'],
        )

    def test_non_lua_passthrough(self):
        with mock.patch.object(desktop, "_uses_lua_config", return_value=False), \
                mock.patch.object(hypr, "dispatch", return_value="ok") as dispatch:
            desktop._dispatch("movetoworkspacesilent 1,address:0x1")
        dispatch.assert_called_once_with("movetoworkspacesilent 1,address:0x1")


class MoveWindowDispatchTest(unittest.TestCase):
    def test_routes_through_translated_dispatch(self):
        with mock.patch.object(desktop, "_dispatch", return_value="ok") as dispatch:
            result = desktop.move_window("class:zen", 3, silent=True)
        dispatch.assert_called_once_with("movetoworkspacesilent 3,class:zen")
        self.assertEqual(result["target"], "class:zen")
        self.assertEqual(result["workspace"], 3)

    def test_non_silent_verb(self):
        with mock.patch.object(desktop, "_dispatch", return_value="ok") as dispatch:
            desktop.move_window("address:0x1", 2, silent=False)
        dispatch.assert_called_once_with("movetoworkspace 2,address:0x1")

    def test_error_reply_raises(self):
        with mock.patch.object(desktop, "_dispatch", return_value="command error: failed"):
            with self.assertRaises(hypr.HyprError):
                desktop.move_window("class:zen", 3)

    def test_ok_reply_no_raise(self):
        with mock.patch.object(desktop, "_dispatch", return_value="ok"):
            result = desktop.move_window("class:zen", 3)
        self.assertEqual(result["reply"], "ok")


class OrganizeWorkspacesDispatchTest(unittest.TestCase):
    def test_moves_route_through_translated_dispatch(self):
        calls = []
        with mock.patch.object(hypr, "clients", return_value=[client("0x1", "codium", "t", 6)]), \
                mock.patch.object(hypr, "active_workspace", return_value={"id": 9}), \
                mock.patch.object(desktop, "_dispatch", side_effect=lambda a: calls.append(a) or "ok"):
            result = desktop.organize_workspaces()
        self.assertEqual(
            result["moves"], [{"class": "codium", "title": "t", "from_ws": 6, "to_ws": 1}]
        )
        self.assertEqual(calls, ["movetoworkspacesilent 1,address:0x1"])

    def test_lua_dispatch_payload(self):
        calls = []
        with mock.patch.object(desktop, "_uses_lua_config", return_value=True), \
                mock.patch.object(hypr, "clients", return_value=[client("0x2", "spotify", "s", 6)]), \
                mock.patch.object(hypr, "active_workspace", return_value={"id": 9}), \
                mock.patch.object(hypr, "dispatch", side_effect=lambda a: calls.append(a) or "ok"):
            desktop.organize_workspaces()
        self.assertEqual(
            calls,
            ['hl.dsp.window.move({window = "address:0x2", workspace = "3", follow = false})'],
        )

    def test_error_reply_raises(self):
        with mock.patch.object(hypr, "clients", return_value=[client("0x1", "codium", "t", 6)]), \
                mock.patch.object(hypr, "active_workspace", return_value={"id": 9}), \
                mock.patch.object(desktop, "_dispatch", return_value="error moving window"):
            with self.assertRaises(hypr.HyprError):
                desktop.organize_workspaces()


class FindNewClientTest(unittest.TestCase):
    def test_matches_window_class_not_binary_name(self):
        new = [client("0x1", "zen", "New Tab", 1)]
        with mock.patch.object(hypr, "clients", side_effect=[[], new]):
            address = desktop._find_new_client("zen-bin", timeout=1.0)
        self.assertEqual(address, "0x1")

    def test_hint_prefers_matching_title(self):
        new = [
            client("0x2", "zen", "New Tab", 1),
            client("0x1", "zen", "OpenAI Chat", 1),
        ]
        with mock.patch.object(hypr, "clients", side_effect=[[], new]):
            address = desktop._find_new_client("zen-bin", hint="openai", timeout=1.0)
        self.assertEqual(address, "0x1")

    def test_no_hint_returns_first_class_match(self):
        new = [
            client("0x2", "zen", "New Tab", 1),
            client("0x1", "zen", "OpenAI Chat", 1),
        ]
        with mock.patch.object(hypr, "clients", side_effect=[[], new]):
            address = desktop._find_new_client("zen-bin", timeout=1.0)
        self.assertEqual(address, "0x2")

    def test_preexisting_clients_ignored(self):
        calls = {"n": 0}

        def fake_clients():
            calls["n"] += 1
            if calls["n"] == 1:
                return [client("0x0", "zen", "Old Tab", 1)]
            return []

        with mock.patch.object(hypr, "clients", side_effect=fake_clients):
            address = desktop._find_new_client("zen", timeout=1.0)
        self.assertIsNone(address)

    def test_returns_none_when_no_window_appears(self):
        with mock.patch.object(hypr, "clients", return_value=[]):
            address = desktop._find_new_client("zen", timeout=1.0)
        self.assertIsNone(address)


class SingleInstanceLaunchTest(unittest.TestCase):
    def test_zen_workspace_uses_exec_plus_move(self):
        calls = []
        with mock.patch.object(desktop, "resolve_binary", return_value="/usr/bin/zen-bin"), \
                mock.patch.object(desktop, "_find_new_client", return_value="0xzen1"), \
                mock.patch.object(desktop, "_dispatch", side_effect=lambda a: calls.append(a) or "ok"):
            result = desktop.launch_app("zen", "https://example.com", workspace=3)
        self.assertEqual(result["method"], "exec+move")
        self.assertTrue(result["ok"])
        self.assertIn("new window moved to workspace 3 (0xzen1)", result["note"])
        self.assertEqual(
            calls,
            [
                "exec /usr/bin/zen-bin --new-window https://example.com",
                "movetoworkspacesilent 3,address:0xzen1",
            ],
        )
        self.assertFalse(any("[workspace" in c for c in calls))

    def test_codium_workspace_uses_exec_plus_move(self):
        with mock.patch.object(desktop, "resolve_binary", return_value="/usr/bin/codium"), \
                mock.patch.object(desktop, "_find_new_client", return_value="0x1"), \
                mock.patch.object(desktop, "_dispatch", return_value="ok") as dispatch:
            result = desktop.launch_app("codium", "/mnt/Data/app/x", workspace=4)
        dispatch.assert_any_call("exec /usr/bin/codium -n /mnt/Data/app/x")
        dispatch.assert_any_call("movetoworkspacesilent 4,address:0x1")
        self.assertEqual(result["method"], "exec+move")
        self.assertTrue(result["ok"])

    def test_ok_false_when_no_window_found(self):
        with mock.patch.object(desktop, "resolve_binary", return_value="/usr/bin/zen-bin"), \
                mock.patch.object(desktop, "_find_new_client", return_value=None), \
                mock.patch.object(desktop, "_dispatch", return_value="ok"):
            result = desktop.launch_app("zen", "https://example.com", workspace=3)
        self.assertFalse(result["ok"])
        self.assertIn("new window could not be located", result["note"])

    def test_hint_forwarded_to_finder(self):
        with mock.patch.object(desktop, "resolve_binary", return_value="/usr/bin/zen-bin"), \
                mock.patch.object(desktop, "_find_new_client", return_value="0x1") as finder, \
                mock.patch.object(desktop, "_dispatch", return_value="ok"):
            desktop.launch_app(
                "zen", "https://chat.openai.com", workspace=2, hint="openai"
            )
        finder.assert_called_once_with("zen", "openai")

    def test_foot_still_uses_windowrule_first(self):
        with mock.patch.object(desktop, "resolve_binary", return_value="/usr/bin/foot"), \
                mock.patch.object(desktop, "_dispatch", return_value="ok") as dispatch:
            result = desktop.launch_app("foot", "-e htop", workspace=2)
        dispatch.assert_called_once_with('exec "[workspace 2 silent] /usr/bin/foot -e htop"')
        self.assertEqual(result["method"], "windowrule")
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
