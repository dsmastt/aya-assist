"""Tests for project startup and new-window launches in desktop tools.

Covers :func:`assistant.tools.desktop.extract_run_commands` ordering,
fuzzy project-name matching (``_score_name`` plus the project-marker
bonus), :func:`start_project` (wire-level, with monkeypatched
launch/safety), and the new-window flag injection / exec+move placement in
:func:`launch_app` / :func:`open_url` (including URL-derived hints).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.tools import desktop, safety  # noqa: E402


README_FIXTURE = """# Demo

Run the development server:

```bash
npm run dev
```

Or start the production build:

```sh
npm start
```

You can also use `vite` directly in prose.

```bash
npm run serve
```

./run.sh works too.
"""


class ExtractRunCommandsTest(unittest.TestCase):
    def test_orders_dev_before_others(self):
        commands = desktop.extract_run_commands(README_FIXTURE)
        self.assertEqual(
            commands,
            ["npm run dev", "npm start", "npm run serve", "vite", "./run.sh"],
        )

    def test_generic_fallback_only_in_code_blocks(self):
        text = """prose says npm serve somewhere

```bash
pnpm serve
```
"""
        self.assertEqual(desktop.extract_run_commands(text), ["pnpm serve"])

    def test_no_commands_returns_empty(self):
        self.assertEqual(desktop.extract_run_commands("No commands here at all."), [])

    def test_empty_input(self):
        self.assertEqual(desktop.extract_run_commands(""), [])


class FuzzyMatchTest(unittest.TestCase):
    def test_notemind_matches_ai_notemaker(self):
        self.assertGreater(desktop._score_name("notemind", "ai-notemaker"), 0)

    def test_ear_training_matches_larp_project(self):
        self.assertGreater(
            desktop._score_name("ear training", "larp-perfect-pitch-ear-trainer"), 0
        )

    def test_exact_tokens_outrank_prefixes(self):
        exact = desktop._score_name("notemind", "notemind-app")
        prefix = desktop._score_name("notemind", "notegram")
        self.assertGreater(exact, prefix)

    def test_unrelated_names_score_zero(self):
        self.assertEqual(desktop._score_name("notemind", "video-downloader"), 0)


class StartProjectTest(unittest.TestCase):
    def _launch_capture(self, calls):
        def fake(app, args="", workspace=None, silent=True, config=None):
            calls.append(
                {"app": app, "args": args, "workspace": workspace, "silent": silent}
            )
            return {"ok": True, "method": "windowrule", "app": app}

        return fake

    def test_starts_project_from_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(
                "# App\n\n```bash\nnpm run dev\n```\n"
            )
            calls = []
            with mock.patch.object(
                desktop, "launch_app", side_effect=self._launch_capture(calls)
            ):
                result = desktop.start_project(str(root))
        self.assertEqual(result["status"], "started")
        self.assertEqual(result["command"], "npm run dev")
        self.assertEqual(result["workspace"], 4)
        self.assertEqual(result["terminal"], "foot")
        self.assertEqual(len(calls), 1)
        self.assertIn("cd ", calls[0]["args"])
        self.assertIn("npm run dev", calls[0]["args"])
        self.assertEqual(calls[0]["workspace"], 4)
        self.assertTrue(calls[0]["silent"])

    def test_no_readme_returns_no_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = desktop.start_project(str(tmp))
        self.assertEqual(result["status"], "no-command")
        self.assertIn("couldn't find a run command", result["note"])

    def test_needs_approval_then_approved(self):
        verdict = safety.Verdict(
            safety.Action.APPROVE, safety.SHELL, ["needs approval"], "test"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# App\n\n```bash\nnpm start\n```\n")
            calls = []
            with mock.patch.object(
                safety, "classify", return_value=verdict
            ), mock.patch.object(
                desktop, "launch_app", side_effect=self._launch_capture(calls)
            ):
                pending = desktop.start_project(str(root), approved=False)
                started = desktop.start_project(str(root), approved=True)
        self.assertEqual(pending["status"], "needs-approval")
        self.assertEqual(pending["command"], "npm start")
        self.assertEqual(started["status"], "started")
        self.assertEqual(len(calls), 1)

    def test_denied_verdict_refused(self):
        verdict = safety.Verdict(safety.Action.DENY, safety.FORBIDDEN, ["nope"], "x")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# App\n\n```bash\nnpm run dev\n```\n")
            with mock.patch.object(safety, "classify", return_value=verdict):
                result = desktop.start_project(str(root), approved=True)
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["command"], "npm run dev")


class NewWindowFlagTest(unittest.TestCase):
    def test_zen_gets_new_window_flag(self):
        calls = []
        with mock.patch.object(
            desktop, "resolve_binary", return_value="/usr/bin/zen-bin"
        ), mock.patch.object(
            desktop, "_find_new_client", return_value="0x1"
        ), mock.patch.object(
            desktop, "_dispatch", side_effect=lambda a: calls.append(a) or "ok"
        ):
            result = desktop.launch_app("zen", "https://example.com", workspace=3)
        self.assertEqual(result["method"], "exec+move")
        self.assertTrue(result["ok"])
        self.assertIn("--new-window https://example.com", result["command"])
        self.assertIn("exec /usr/bin/zen-bin --new-window https://example.com", calls)
        self.assertTrue(any("[workspace" in c for c in calls) is False)
        self.assertIn("new window moved to workspace 3 (0x1)", result["note"])

    def test_codium_gets_dash_n(self):
        with mock.patch.object(
            desktop, "resolve_binary", return_value="/usr/bin/codium"
        ), mock.patch.object(
            desktop, "_find_new_client", return_value="0x1"
        ), mock.patch.object(desktop, "_dispatch", return_value="ok"):
            result = desktop.launch_app("codium", "/mnt/Data/app/x", workspace=4)
        self.assertIn("-n", result["command"].split())
        self.assertLess(
            result["command"].index("-n"),
            result["command"].index("/mnt/Data/app/x"),
        )

    def test_no_injection_without_workspace(self):
        with mock.patch.object(
            desktop, "resolve_binary", return_value="/usr/bin/zen-bin"
        ), mock.patch.object(desktop, "_dispatch", return_value="ok"):
            result = desktop.launch_app("zen", "https://example.com")
        self.assertNotIn("--new-window", result["command"])
        self.assertTrue(result["ok"])

    def test_flag_not_duplicated(self):
        with mock.patch.object(
            desktop, "resolve_binary", return_value="/usr/bin/zen-bin"
        ), mock.patch.object(
            desktop, "_find_new_client", return_value="0x1"
        ), mock.patch.object(desktop, "_dispatch", return_value="ok"):
            result = desktop.launch_app(
                "zen", "--new-window https://example.com", workspace=3
            )
        self.assertEqual(
            result["command"].count("--new-window"), 1
        )

    def test_failed_dispatch_reports_ok_false(self):
        with mock.patch.object(
            desktop, "resolve_binary", return_value="/usr/bin/zen-bin"
        ), mock.patch.object(desktop, "_dispatch", return_value="error: nope"):
            result = desktop.launch_app("zen", "https://example.com")
        self.assertFalse(result["ok"])
        self.assertIn("dispatch rejected", result["note"])


class OpenUrlNewWindowTest(unittest.TestCase):
    def test_open_url_uses_new_window_on_workspace(self):
        with mock.patch.object(
            desktop, "resolve_binary", return_value="/usr/bin/zen-bin"
        ), mock.patch.object(
            desktop, "_find_new_client", return_value="0x1"
        ), mock.patch.object(desktop, "_dispatch", return_value="ok"):
            result = desktop.open_url("https://example.com", workspace=3)
        self.assertIn("--new-window", result["command"])
        self.assertIn("https://example.com", result["command"])
        self.assertEqual(result["url"], "https://example.com")
        self.assertEqual(result["hint"], "example")


class UrlHintTest(unittest.TestCase):
    def _open(self, url):
        with mock.patch.object(
            desktop, "resolve_binary", return_value="/usr/bin/zen-bin"
        ), mock.patch.object(
            desktop, "_find_new_client", return_value="0x1"
        ), mock.patch.object(desktop, "_dispatch", return_value="ok"):
            return desktop.open_url(url, workspace=3)

    def test_youtube_watch_host(self):
        self.assertEqual(
            self._open("https://www.youtube.com/watch?v=abc")["hint"], "youtube"
        )

    def test_youtu_be_short_url(self):
        self.assertEqual(
            self._open("https://youtu.be/abc123")["hint"], "youtube"
        )

    def test_longest_netloc_label(self):
        self.assertEqual(
            self._open("https://chat.openai.com/c/123")["hint"], "openai"
        )

    def test_short_labels_yield_none(self):
        self.assertEqual(self._open("https://x.yz/")["hint"], None)


class ProjectMarkerBonusTest(unittest.TestCase):
    def test_project_marker_breaks_prefix_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "ai-notemaker"
            project.mkdir()
            (project / "README.md").write_text("# x\n")
            junk = root / "nothing"
            junk.mkdir()
            config = type(
                "Cfg", (), {"paths": type("Paths", (), {"projects_root": str(root)})()}
            )()
            with mock.patch.object(desktop, "_BACKUP_PROJECTS_ROOT", str(root)):
                best, _visible = desktop._find_project_match("notemind", config)
        self.assertEqual(best, [str(project)])

    def test_junk_without_markers_loses_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "ai-notemaker"
            project.mkdir()
            (project / "package.json").write_text("{}\n")
            junk = root / "nothing"
            junk.mkdir()
            config = type(
                "Cfg", (), {"paths": type("Paths", (), {"projects_root": str(root)})()}
            )()
            with mock.patch.object(desktop, "_BACKUP_PROJECTS_ROOT", str(root)):
                best, _visible = desktop._find_project_match("notemind", config)
        self.assertEqual(best, [str(project)])


if __name__ == "__main__":
    unittest.main()
