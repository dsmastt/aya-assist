"""Tests for assistant.tools.hypr using a fake in-process Unix socket server."""

import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.tools import hypr  # noqa: E402


class FakeHyprServer:
    """AF_UNIX server that answers each request with a fixed response."""

    def __init__(self, directory: str) -> None:
        self.path = os.path.join(directory, "test.sock")
        self.responses: dict[str, str] = {}
        self.default = "ok"
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.path)
        self._sock.listen(4)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(0.5)
                data = b""
                while True:
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        break
                    if not chunk:
                        break
                    data += chunk
                payload = data.decode("utf-8", "replace")
                response = self.default
                for prefix, resp in self.responses.items():
                    if payload.startswith(prefix):
                        response = resp
                        break
                try:
                    conn.sendall(response.encode("utf-8"))
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class HyprIpcTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.server = FakeHyprServer(self.tmpdir.name)
        self.server.start()

    def tearDown(self) -> None:
        self.server.stop()
        self.tmpdir.cleanup()

    def patch_socket(self):
        return mock.patch.object(hypr, "_socket_path", return_value=self.server.path)

    def test_socket_path_from_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"XDG_RUNTIME_DIR": "/run/user/1000", "HYPRLAND_INSTANCE_SIGNATURE": "abc"},
        ):
            self.assertEqual(
                hypr._socket_path(), "/run/user/1000/hypr/abc/.socket.sock"
            )

    def test_socket_path_raises_without_env(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("XDG_RUNTIME_DIR", None)
            os.environ.pop("HYPRLAND_INSTANCE_SIGNATURE", None)
            with self.assertRaises(hypr.HyprError):
                hypr._socket_path()

    def test_hyprctl_json_parses_list(self) -> None:
        with self.patch_socket():
            self.server.responses["j/workspaces"] = json.dumps(
                [{"id": 1, "name": "1"}, {"id": 2, "name": "2"}]
            )
            parsed = hypr.hyprctl_json("workspaces")
            self.assertEqual(parsed, [{"id": 1, "name": "1"}, {"id": 2, "name": "2"}])

    def test_workspaces_returns_list(self) -> None:
        with self.patch_socket():
            self.server.responses["j/workspaces"] = json.dumps([{"id": 1, "name": "1"}])
            self.assertIsInstance(hypr.workspaces(), list)
            self.assertEqual(hypr.workspaces(), [{"id": 1, "name": "1"}])

    def test_hyprctl_json_invalid_json_raises(self) -> None:
        with self.patch_socket():
            self.server.responses["j/version"] = "this is not json at all"
            with self.assertRaises(hypr.HyprError) as ctx:
                hypr.hyprctl_json("version")
            self.assertIn("invalid JSON", str(ctx.exception))

    def test_dispatch_returns_reply(self) -> None:
        with self.patch_socket():
            self.server.default = "ok"
            self.assertEqual(hypr.dispatch("exec mpv"), "ok")

    def test_conn_refused_raises_friendly_error(self) -> None:
        stale = os.path.join(self.tmpdir.name, "stale.sock")
        with open(stale, "w", encoding="utf-8") as fh:
            fh.write("not a socket")
        with mock.patch.object(hypr, "_socket_path", return_value=stale):
            with self.assertRaises(hypr.HyprError) as ctx:
                hypr.hyprctl_raw("dispatch exec mpv")
        self.assertIn("cannot reach Hyprland IPC", str(ctx.exception))

    def test_missing_socket_file_raises_friendly_error(self) -> None:
        missing = os.path.join(self.tmpdir.name, "missing.sock")
        with mock.patch.object(hypr, "_socket_path", return_value=missing):
            with self.assertRaises(hypr.HyprError) as ctx:
                hypr.hyprctl_raw("dispatch exec mpv")
        self.assertIn("cannot reach Hyprland IPC", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
