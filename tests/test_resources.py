"""Tests for assistant.core.resources (stdlib + psutil; no GPU required)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.config import Config
from assistant.core import resources  # noqa: E402


class SystemRamTest(unittest.TestCase):
    def test_sane_keys(self) -> None:
        ram = resources.system_ram()
        self.assertGreater(ram["total_mb"], 0)
        self.assertGreaterEqual(ram["avail_mb"], 0)
        self.assertGreaterEqual(ram["used_mb"], 0)
        self.assertGreaterEqual(ram["percent"], 0.0)
        self.assertLessEqual(ram["percent"], 100.0)


class LoadAvgTest(unittest.TestCase):
    def test_tuple_of_three_floats(self) -> None:
        load = resources.load_avg()
        self.assertIsInstance(load, tuple)
        self.assertEqual(len(load), 3)
        self.assertTrue(all(isinstance(v, float) for v in load))


class GpuInfoTest(unittest.TestCase):
    def test_never_raises_and_well_formed(self) -> None:
        info = resources.gpu_info()  # must not raise
        self.assertIsInstance(info, dict)
        if info.get("available"):
            for key in (
                "name",
                "vram_total_mb",
                "vram_used_mb",
                "vram_free_mb",
                "util_percent",
                "temp_c",
            ):
                self.assertIn(key, info)
            self.assertGreater(info["vram_total_mb"], 0)
        else:
            self.assertIn("note", info)


class VramFreeTest(unittest.TestCase):
    def test_int_or_none(self) -> None:
        free = resources.vram_free_mb()
        self.assertTrue(free is None or isinstance(free, int))


class SystemStatusTest(unittest.TestCase):
    def test_expected_top_level_keys(self) -> None:
        status = resources.system_status()
        for key in ("ram", "load_avg", "gpu", "top_processes"):
            self.assertIn(key, status)
        self.assertEqual(len(status["load_avg"]), 3)
        self.assertIsInstance(status["top_processes"], list)
        self.assertEqual(len(status["top_processes"]), 5)


class CheckBeforeHeavyTest(unittest.TestCase):
    def test_returns_ok_bool_and_message(self) -> None:
        ok, msg = resources.check_before_heavy("test-label", Config())
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(msg, str)

    def test_fails_below_minimum(self) -> None:
        cfg = Config()
        cfg.resources.min_free_ram_mb = 2**40  # unreachable minimum
        ok, msg = resources.check_before_heavy("test", cfg)
        self.assertFalse(ok)
        self.assertIn("test", msg)


if __name__ == "__main__":
    unittest.main()
