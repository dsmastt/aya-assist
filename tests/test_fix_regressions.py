"""Regression tests for three live-diagnosed fixes.

1. ``_score_name`` prefix weighting and package.json/pyproject.toml name
   scoring in ``_find_project_match`` (desktop).
2. Site-mention audit: site keywords require ``launch`` or ``video`` (agent).
3. Launch wrappers force ``silent=True`` (builtin).
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.core.agent import _claim_tool_hint, _claims_unsupported_action  # noqa: E402
from assistant.tools import builtin, desktop  # noqa: E402
from assistant.tools.registry import TOOLS, ToolContext  # noqa: E402


def _config(root: Path):
    return type(
        "Cfg", (), {"paths": type("Paths", (), {"projects_root": str(root)})()}
    )()


class PrefixWeightingTest(unittest.TestCase):
    def test_longer_shared_prefix_ranks_higher(self):
        self.assertGreater(
            desktop._score_name("notemind", "ai-notemaker"),
            desktop._score_name("notemind", "nothing"),
        )

    def test_five_char_common_prefix_scores_three(self):
        self.assertEqual(desktop._score_name("notemind", "ai-notemaker"), 3)

    def test_not_prefix_scores_one(self):
        self.assertEqual(desktop._score_name("notemind", "nothing"), 1)


class ManifestNameMatchTest(unittest.TestCase):
    def test_package_json_name_breaks_directory_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "ai-notemaker"
            real.mkdir()
            (real / "package.json").write_text(json.dumps({"name": "notemind"}))
            junk = root / "nothing"
            junk.mkdir()
            (junk / "package.json").write_text(json.dumps({"name": "x"}))
            with mock.patch.object(desktop, "_BACKUP_PROJECTS_ROOT", str(root)):
                best, _visible = desktop._find_project_match("notemind", _config(root))
        self.assertEqual(best, [str(real)])

    def test_invalid_package_json_contributes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            junk = root / "nothing"
            junk.mkdir()
            (junk / "package.json").write_text("not json at all")
            real = root / "ai-notemaker"
            real.mkdir()
            (real / "package.json").write_text(json.dumps({"name": "notemind"}))
            with mock.patch.object(desktop, "_BACKUP_PROJECTS_ROOT", str(root)):
                best, _visible = desktop._find_project_match("notemind", _config(root))
        self.assertEqual(best, [str(real)])

    def test_pyproject_name_scores_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "ai-notemaker"
            real.mkdir()
            (real / "pyproject.toml").write_text('[project]\nname = "notemind"\n')
            junk = root / "nothing"
            junk.mkdir()
            (junk / "package.json").write_text(json.dumps({"name": "x"}))
            with mock.patch.object(desktop, "_BACKUP_PROJECTS_ROOT", str(root)):
                best, _visible = desktop._find_project_match("notemind", _config(root))
        self.assertEqual(best, [str(real)])


class SiteMentionAuditTest(unittest.TestCase):
    def test_site_mention_without_launch_flagged_with_hint(self):
        self.assertTrue(
            _claims_unsupported_action(
                "I opened YouTube on workspace 3",
                [("system_status", "system_status")],
            )
        )
        hints = _claim_tool_hint(
            "I opened YouTube on workspace 3",
            [("system_status", "system_status")],
        )
        self.assertTrue(any("launch" in hint for hint in hints))

    def test_site_mention_with_launch_passes(self):
        # Action-aware: only a launch with action "url" proves a site open;
        # a generic launch name without a matching action still audits.
        self.assertFalse(
            _claims_unsupported_action(
                "I opened YouTube on workspace 3", [("launch", "url")]
            )
        )

    def test_play_claim_without_play_tool_still_flagged(self):
        self.assertTrue(
            _claims_unsupported_action("Playing some lofi now", [("web", "search")])
        )

    def test_negated_site_mention_passes(self):
        self.assertFalse(
            _claims_unsupported_action("I didn't open YouTube", [("launch", "url")])
        )


class SilentLaunchWrapperTest(unittest.TestCase):
    def _run(self, entry, arguments):
        return asyncio.run(entry.func(ToolContext(), arguments))

    def test_launch_app_action_forces_silent_true(self):
        captured = {}

        def fake_launch(app, args="", workspace=None, silent=True, config=None, hint=None):
            captured["silent"] = silent
            return {"ok": True, "app": app}

        with mock.patch.object(
            builtin.desktop, "launch_app", side_effect=fake_launch
        ):
            result = self._run(
                TOOLS.get("launch"),
                {"action": "app", "target": "codium", "silent": False},
            )
        self.assertTrue(captured["silent"])
        self.assertEqual(result["app"], "codium")

    def test_launch_url_action_ignores_caller_silent(self):
        captured = {}

        def fake_open_url(url, workspace=None, silent=True, config=None):
            # The merged launch wrapper swallows the caller's "silent"
            # argument instead of forwarding it: a registry launch can never
            # request focus stealing.
            captured["silent"] = silent
            return {"ok": True, "url": url}

        with mock.patch.object(
            builtin.desktop, "open_url", side_effect=fake_open_url
        ):
            result = self._run(
                TOOLS.get("launch"),
                {"action": "url", "target": "youtube", "silent": False},
            )
        self.assertNotEqual(captured["silent"], False)
        self.assertIn("youtube.com", result["url"])

    def test_launch_project_action_ignores_caller_silent(self):
        captured = {}

        def fake_open_project(name_or_path, workspace=None, silent=True, config=None):
            # The merged launch wrapper swallows the caller's "silent"
            # argument instead of forwarding it.
            captured["silent"] = silent
            return {"status": "found", "path": name_or_path}

        with mock.patch.object(
            builtin.desktop, "open_project", side_effect=fake_open_project
        ):
            result = self._run(
                TOOLS.get("launch"),
                {"action": "project", "target": "assistant", "silent": False},
            )
        self.assertNotEqual(captured["silent"], False)
        self.assertEqual(result["status"], "found")


if __name__ == "__main__":
    unittest.main()
