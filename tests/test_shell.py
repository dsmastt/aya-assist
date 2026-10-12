"""Tests for assistant.tools.shell."""

import os
import sys
import tempfile
import unittest
from enum import Enum
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.tools import shell  # noqa: E402

try:
    from assistant.tools import safety  # noqa: F401

    HAVE_SAFETY = True
except ImportError:
    HAVE_SAFETY = False


class _Action(Enum):
    ALLOW = "allow"
    APPROVE = "approve"
    DENY = "deny"


class _FakePolicy:
    def __init__(self, action):
        self._action = action

    def classify(self, command):
        return SimpleNamespace(
            action=self._action,
            category="fake",
            reasons=["fake reason"],
            summary="fake summary",
        )


class TestRun(unittest.TestCase):
    def test_echo_hello(self):
        result = shell.run("echo hello")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "hello\n")

    def test_stderr(self):
        result = shell.run("echo err >&2")
        self.assertIn("err", result.stderr)

    def test_timeout(self):
        result = shell.run("sleep 10", timeout=1)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.returncode)

    def test_output_cap(self):
        result = shell.run("printf 'x%.0s' {1..30000}")
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.stdout), shell.OUTPUT_CAP + len(shell.TRUNCATED_MARKER))


class TestRunChecked(unittest.TestCase):
    def test_no_policy(self):
        result = shell.run_checked("echo hi")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["returncode"], 0)

    @unittest.skipUnless(HAVE_SAFETY, "assistant.tools.safety not available")
    def test_safety_denies_sudo(self):
        result = shell.run_checked("sudo id")
        self.assertEqual(result["status"], "denied")

    def test_deny_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "marker")
            result = shell.run_checked(
                f"touch {marker}", policy=_FakePolicy(_Action.DENY)
            )
            self.assertEqual(result["status"], "denied")
            self.assertIsNone(result.get("returncode"))
            self.assertFalse(os.path.exists(marker))

    def test_approve_rejected(self):
        result = shell.run_checked(
            "echo hi", policy=_FakePolicy(_Action.APPROVE), approve=lambda label, detail: False
        )
        self.assertEqual(result["status"], "not-approved")
        self.assertIsNone(result.get("returncode"))

    def test_approve_accepted(self):
        result = shell.run_checked(
            "echo approved",
            policy=_FakePolicy(_Action.APPROVE),
            approve=lambda label, detail: True,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "approved\n")

    def test_allow_executes(self):
        result = shell.run_checked("echo allowed", policy=_FakePolicy(_Action.ALLOW))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stdout"], "allowed\n")


class TestSpawn(unittest.TestCase):
    def tearDown(self):
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.is_running():
            proc.kill()
            proc.wait()

    def test_spawn_kill_wait(self):
        proc = shell.spawn("sleep 30")
        self._proc = proc
        self.assertTrue(proc.is_running())
        self.assertIsNone(proc.poll())
        self.assertTrue(proc.kill())
        returncode = proc.wait()
        self.assertIsNotNone(returncode)
        self.assertFalse(proc.is_running())


class TestDefaultEnv(unittest.TestCase):
    def test_scrubbing(self):
        env = shell.default_env({"SSH_AUTH_SOCK": "x", "MY_EXTRA": "1"})
        self.assertNotIn("SSH_AUTH_SOCK", env)
        self.assertIn("PATH", env)
        self.assertIn("MY_EXTRA", env)


if __name__ == "__main__":
    unittest.main()
