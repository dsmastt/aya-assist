"""Regression tests for three live-diagnosed fixes.

1. ``_score_name`` prefix weighting and package.json/pyproject.toml name
   scoring in ``_find_project_match`` (desktop).
2. Site-mention audit: site keywords require ``open_url`` (agent).
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
    def test_site_mention_without_open_url_flagged_with_hint(self):
        self.assertTrue(
            _claims_unsupported_action("I opened YouTube on workspace 3", ["open_app"])
        )
        hints = _claim_tool_hint("I opened YouTube on workspace 3", ["open_app"])
        self.assertIn("open_url", hints)

    def test_site_mention_with_open_url_passes(self):
        self.assertFalse(
            _claims_unsupported_action("I opened YouTube on workspace 3", ["open_url"])
        )

    def test_play_claim_without_play_tool_still_flagged(self):
        self.assertTrue(
            _claims_unsupported_action("Playing some lofi now", ["find_video"])
        )

    def test_negated_site_mention_passes(self):
        self.assertFalse(
            _claims_unsupported_action("I didn't open YouTube", ["open_app"])
        )


class SilentLaunchWrapperTest(unittest.TestCase):
    def _run(self, entry, arguments):
        return asyncio.run(entry.func(ToolContext(), arguments))

    def test_open_app_forces_silent_true(self):
        captured = {}

        def fake_launch(app, args="", workspace=None, silent=True, config=None, hint=None):
            captured["silent"] = silent
            return {"ok": True, "app": app}

        with mock.patch.object(
            builtin.desktop, "launch_app", side_effect=fake_launch
        ):
            result = self._run(
                TOOLS.get("open_app"), {"app": "codium", "silent": False}
            )
        self.assertTrue(captured["silent"])
        self.assertEqual(result["app"], "codium")

    def test_open_url_forces_silent_true(self):
        captured = {}

        def fake_open_url(url, workspace=None, silent=True, config=None):
            captured["silent"] = silent
            return {"ok": True, "url": url}

        with mock.patch.object(
            builtin.desktop, "open_url", side_effect=fake_open_url
        ):
            result = self._run(TOOLS.get("open_url"), {"url": "youtube", "silent": False})
        self.assertTrue(captured["silent"])
        self.assertIn("youtube.com", result["url"])

    def test_open_project_forces_silent_true(self):
        captured = {}

        def fake_open_project(name_or_path, workspace=None, silent=True, config=None):
            captured["silent"] = silent
            return {"status": "found", "path": name_or_path}

        with mock.patch.object(
            builtin.desktop, "open_project", side_effect=fake_open_project
        ):
            result = self._run(
                TOOLS.get("open_project"), {"name_or_path": "assistant", "silent": False}
            )
        self.assertTrue(captured["silent"])
        self.assertEqual(result["status"], "found")


if __name__ == "__main__":
    unittest.main()
