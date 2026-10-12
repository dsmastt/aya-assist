"""Tests for assistant.llm.engine (pure logic; no model load required)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.config import Config
from assistant.llm.engine import (  # noqa: E402
    MODEL_PRESETS,
    EngineError,
    LLMEngine,
    resolve_chat_format,
)


class EstimateVramTest(unittest.TestCase):
    def test_keys_and_positive_values(self) -> None:
        est = LLMEngine(Config()).estimate_vram()
        for key in ("weights_mb", "kv_mb", "total_mb"):
            self.assertIn(key, est)
            self.assertGreaterEqual(est[key], 0)
        self.assertAlmostEqual(est["total_mb"], est["weights_mb"] + est["kv_mb"])

    def test_formula_with_known_file_size(self) -> None:
        cfg = Config()
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as fh:
            fh.write(b"x" * (1024 * 1024))  # exactly 1 MiB
            path = fh.name
        try:
            cfg.model.file = path  # absolute path is used as-is
            cfg.model.n_ctx = 1000
            cfg.model.kv_quant = "q8_0"
            est = LLMEngine(cfg).estimate_vram()
            self.assertAlmostEqual(est["weights_mb"], 1.15, places=3)
            self.assertAlmostEqual(est["kv_mb"], 1000 * 0.035, places=3)
            self.assertAlmostEqual(
                est["total_mb"], 1.15 + 1000 * 0.035, places=3
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_file_zero_weights_with_note(self) -> None:
        cfg = Config()
        cfg.model.file = "/nonexistent/definitely-not-here.gguf"
        est = LLMEngine(cfg).estimate_vram()
        self.assertEqual(est["weights_mb"], 0)
        self.assertIn("note", est)


class ResolveChatFormatTest(unittest.TestCase):
    def test_auto_prefers_qwen2_5_when_available(self) -> None:
        self.assertEqual(
            resolve_chat_format("auto", ["chatml", "qwen2.5", "llama-3"]),
            "qwen2.5",
        )

    def test_auto_falls_back_to_chatml(self) -> None:
        self.assertEqual(
            resolve_chat_format("auto", ["chatml", "llama-3"]), "chatml"
        )

    def test_explicit_value_passes_through(self) -> None:
        self.assertEqual(resolve_chat_format("chatml"), "chatml")
        self.assertEqual(resolve_chat_format("llama-3"), "llama-3")

    def test_auto_without_llama_cpp_environment(self) -> None:
        result = resolve_chat_format("auto")
        self.assertIn(result, ("chatml", "qwen2.5"))


class ModelPresetsTest(unittest.TestCase):
    def test_qwen2_5_preset_has_expected_keys(self) -> None:
        preset = MODEL_PRESETS["qwen2.5-7b-instruct-q4_k_m"]
        for key in ("file", "url", "notes", "min_vram_mb"):
            self.assertIn(key, preset)
        self.assertTrue(
            preset["url"].startswith("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/")
        )


class EngineUnloadedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = LLMEngine(Config())

    def test_not_loaded_initially(self) -> None:
        self.assertFalse(self.engine.loaded)
        self.assertEqual(self.engine.idle_seconds(), 0.0)

    def test_token_count_heuristic_when_not_loaded(self) -> None:
        text = "hello world, this is a test sentence!"
        self.assertEqual(self.engine.token_count(text), len(text) // 4)

    def test_prompt_token_estimate_sums_contents(self) -> None:
        msgs = [
            {"role": "user", "content": "a" * 40},
            {"role": "assistant", "content": "b" * 20},
        ]
        expected = 40 // 4 + 20 // 4 + 4 * 2
        self.assertEqual(self.engine.prompt_token_estimate(msgs), expected)

    def test_chat_raises_when_not_loaded(self) -> None:
        with self.assertRaises(EngineError) as ctx:
            self.engine.chat([{"role": "user", "content": "hi"}])
        self.assertIn("not loaded", str(ctx.exception))

    def test_unload_is_noop_when_not_loaded(self) -> None:
        self.engine.unload()  # must not raise
        self.assertFalse(self.engine.loaded)


if __name__ == "__main__":
    unittest.main()
