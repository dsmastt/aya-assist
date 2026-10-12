"""Tests for the Qwen2.5-native tool-call pipeline in assistant.llm.engine.

Pure logic: no GPU and no model load required.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.config import Config  # noqa: E402
from assistant.llm.engine import (  # noqa: E402
    LLMEngine,
    _ToolCallStreamFilter,
    _adapt_messages,
    _extract_tool_calls,
    _inject_tools,
)


class ExtractToolCallsTest(unittest.TestCase):
    def test_plain_text_passes_through(self):
        text, calls = _extract_tool_calls("just a normal answer")
        self.assertEqual(text, "just a normal answer")
        self.assertEqual(calls, [])

    def test_single_tool_call_extracted_and_removed(self):
        text, calls = _extract_tool_calls(
            '<tool_call>\n{"name": "system_status", "arguments": {}}\n</tool_call>'
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "system_status")
        self.assertEqual(calls[0]["arguments"], {})
        self.assertNotIn("<tool_call>", text)

    def test_multiple_calls_and_prose(self):
        raw = (
            "Let me check.\n"
            '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>\n'
            '<tool_call>\n{"name": "b", "arguments": {"y": "z"}}\n</tool_call>'
        )
        text, calls = _extract_tool_calls(raw)
        self.assertEqual([c["name"] for c in calls], ["a", "b"])
        self.assertEqual(calls[0]["arguments"], {"x": 1})
        self.assertIn("Let me check.", text)

    def test_malformed_body_is_ignored(self):
        text, calls = _extract_tool_calls(
            "<tool_call>\nnot json at all\n</tool_call>"
        )
        self.assertEqual(calls, [])

    def test_unterminated_block_salvaged_valid_json(self):
        raw = (
            '<tool_call>{"name": "play_video", "arguments": '
            '{"video": "https://www.youtube.com/watch?v=AbCdEfGhIjK"}}'
        )
        text, calls = _extract_tool_calls(raw)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "play_video")
        self.assertEqual(
            calls[0]["arguments"]["video"],
            "https://www.youtube.com/watch?v=AbCdEfGhIjK",
        )
        self.assertEqual(text, "")

    def test_unterminated_block_salvaged_malformed_kept(self):
        raw = '<tool_call>{"name": "play_video", "arguments": {'
        text, calls = _extract_tool_calls(raw)
        self.assertEqual(calls, [])
        self.assertEqual(text, raw)

    def test_unterminated_block_salvaged_with_prose(self):
        raw = (
            'Let me find that.\n<tool_call>{"name": "play_video", "arguments": '
            '{"video": "https://www.youtube.com/watch?v=AbCdEfGhIjK"}}'
        )
        text, calls = _extract_tool_calls(raw)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "play_video")
        self.assertEqual(text, "")

    def test_string_arguments_are_parsed(self):
        text, calls = _extract_tool_calls(
            '<tool_call>\n{"name": "a", "arguments": "{\\"k\\": 2}"}\n</tool_call>'
        )
        self.assertEqual(calls[0]["arguments"], {"k": 2})


class AdaptMessagesTest(unittest.TestCase):
    def test_tool_role_becomes_user_text(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "tool_call_id": "7", "content": "result 42"},
        ]
        out = _adapt_messages(messages)
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[1]["role"], "user")
        self.assertIn("result 42", out[1]["content"])
        self.assertIn("id=7", out[1]["content"])

    def test_other_roles_untouched(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(_adapt_messages(messages), messages)


class InjectToolsTest(unittest.TestCase):
    def test_system_message_gains_tools_block(self):
        messages = [{"role": "system", "content": "base"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "t", "description": "d", "parameters": {}},
            }
        ]
        out = _inject_tools(messages, tools)
        self.assertIn("base", out[0]["content"])
        self.assertIn("<tools>", out[0]["content"])
        self.assertIn('"name": "t"', out[0]["content"])
        self.assertEqual(out[0]["role"], "system")

    def test_caller_messages_not_mutated(self):
        messages = [{"role": "system", "content": "base"}]
        _inject_tools(messages, [])
        self.assertEqual(messages[0]["content"], "base")

    def test_system_message_inserted_when_missing(self):
        out = _inject_tools([{"role": "user", "content": "hi"}], [])
        self.assertEqual(out[0]["role"], "system")


class ToolCallStreamFilterTest(unittest.TestCase):
    def test_passthrough_without_tool_calls(self):
        pieces = []
        filt = _ToolCallStreamFilter(pieces.append)
        for chunk in ("hello ", "wor", "ld"):
            filt.feed(chunk)
        filt.flush()
        self.assertEqual("".join(pieces), "hello world")

    def test_tool_call_block_is_dropped(self):
        pieces = []
        filt = _ToolCallStreamFilter(pieces.append)
        stream = (
            "before "
            '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>'
            " after"
        )
        for i in range(0, len(stream), 5):
            filt.feed(stream[i:i + 5])
        filt.flush()
        joined = "".join(pieces)
        self.assertIn("before", joined)
        self.assertIn("after", joined)
        self.assertNotIn("<tool_call>", joined)

    def test_partial_prefix_held_back(self):
        pieces = []
        filt = _ToolCallStreamFilter(pieces.append)
        filt.feed("text <tool_ca")
        self.assertEqual(pieces, ["tex"])
        filt.feed("ll>\nboom\n</tool_call> rest")
        filt.flush()
        joined = "".join(pieces)
        self.assertNotIn("boom", joined)
        self.assertIn("rest", joined)


_PLAY_VIDEO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_video",
            "description": "Play a video in a workspace",
            "parameters": {},
        },
    }
]


class BareJsonToolCallTest(unittest.TestCase):
    def test_fallback_a_named_json_in_prose(self):
        raw = (
            "I called play_video "
            '{"name": "play_video", "arguments": '
            '{"video": "https://www.youtube.com/watch?v=AbCdEfGhIjK", "workspace": 3}}'
        )
        text, calls = _extract_tool_calls(raw, _PLAY_VIDEO_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "play_video")
        self.assertEqual(
            calls[0]["arguments"]["video"],
            "https://www.youtube.com/watch?v=AbCdEfGhIjK",
        )
        self.assertEqual(calls[0]["arguments"]["workspace"], 3)
        self.assertIn("I called play_video", text)
        self.assertNotIn('"video"', text)
        self.assertNotIn("{", text)

    def test_fallback_b_bare_json_after_mention(self):
        raw = (
            'I called play_video {"video": '
            '"https://www.youtube.com/watch?v=AbCdEfGhIjK", "player": "mpv"}'
        )
        text, calls = _extract_tool_calls(raw, _PLAY_VIDEO_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "play_video")
        self.assertEqual(
            calls[0]["arguments"],
            {
                "video": "https://www.youtube.com/watch?v=AbCdEfGhIjK",
                "player": "mpv",
            },
        )
        self.assertNotIn('"mpv"', text)
        self.assertIn("I called play_video", text)

    def test_unknown_name_stays_in_content(self):
        raw = '{"name": "not_a_tool", "arguments": {"x": 1}}'
        text, calls = _extract_tool_calls(raw, _PLAY_VIDEO_TOOLS)
        self.assertEqual(calls, [])
        self.assertEqual(text, raw)

    def test_tools_none_disables_fallbacks(self):
        raw = 'I called play_video {"name": "play_video", "arguments": {"video": "x"}}'
        text, calls = _extract_tool_calls(raw, None)
        self.assertEqual(calls, [])
        self.assertEqual(text, raw)

    def test_nested_json_arguments_extracted(self):
        raw = '{"name": "play_video", "arguments": {"video": "v", "opts": {"k": 1}}}'
        text, calls = _extract_tool_calls(raw, _PLAY_VIDEO_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["opts"], {"k": 1})
        self.assertNotIn("opts", text)


class BareJsonStreamFilterTest(unittest.TestCase):
    def _stream(self, full, tools):
        pieces = []
        filt = _ToolCallStreamFilter(pieces.append, tools)
        for i in range(0, len(full), 3):
            filt.feed(full[i:i + 3])
        filt.flush()
        return "".join(pieces)

    def test_bare_json_call_not_emitted(self):
        full = (
            'I called play_video {"video": '
            '"https://www.youtube.com/watch?v=AbCdEfGhIjK", "player": "mpv"}'
        )
        joined = self._stream(full, _PLAY_VIDEO_TOOLS)
        self.assertNotIn('"mpv"', joined)
        self.assertNotIn('"video"', joined)
        self.assertIn("I called play_video", joined)
        _, calls = _extract_tool_calls(full, _PLAY_VIDEO_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "play_video")

    def test_xml_block_split_across_chunks(self):
        full = (
            "before "
            '<tool_call>{"name": "play_video", "arguments": {"video": "v"}}</tool_call>'
            " after"
        )
        joined = self._stream(full, _PLAY_VIDEO_TOOLS)
        self.assertNotIn("<tool_call>", joined)
        self.assertNotIn("</tool_call>", joined)
        self.assertNotIn('"video"', joined)
        self.assertIn("before", joined)
        self.assertIn("after", joined)
        self.assertEqual(joined, _extract_tool_calls(full, _PLAY_VIDEO_TOOLS)[0])

    def test_emitted_matches_stripped_final_content(self):
        full = (
            "I'll play it now. "
            '{"name": "play_video", "arguments": '
            '{"video": "https://www.youtube.com/watch?v=AbCdEfGhIjK"}} '
            "done."
        )
        joined = self._stream(full, _PLAY_VIDEO_TOOLS)
        self.assertEqual(joined, _extract_tool_calls(full, _PLAY_VIDEO_TOOLS)[0])


_TIMER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "timer_set",
            "description": "set a timer",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timer_status",
            "description": "check a timer",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_video",
            "description": "find a video",
            "parameters": {},
        },
    },
]


class PseudoCodeCallTest(unittest.TestCase):
    def test_pseudo_code_call_extracted(self):
        raw = "Sure!\n\ntimer_set(duration='30m', label='study')"
        text, calls = _extract_tool_calls(raw, _TIMER_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "timer_set")
        self.assertEqual(
            calls[0]["arguments"], {"duration": "30m", "label": "study"}
        )
        self.assertEqual(text.strip(), "Sure!")

    def test_pseudo_code_numbers_and_double_quotes(self):
        raw = 'find_video(query="lofi", max_results=5)'
        text, calls = _extract_tool_calls(raw, _TIMER_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "find_video")
        self.assertEqual(calls[0]["arguments"], {"query": "lofi", "max_results": 5})
        self.assertEqual(text, "")

    def test_pseudo_code_empty_parens(self):
        text, calls = _extract_tool_calls("timer_status()", _TIMER_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "timer_status")
        self.assertEqual(calls[0]["arguments"], {})
        self.assertEqual(text, "")

    def test_pseudo_code_unknown_name_ignored(self):
        raw = "mystery_tool(x=1)"
        text, calls = _extract_tool_calls(raw, _TIMER_TOOLS)
        self.assertEqual(calls, [])
        self.assertEqual(text, raw)


class PseudoCodeStreamFilterTest(unittest.TestCase):
    def test_stream_pseudo_code_held_back(self):
        pieces = []
        filt = _ToolCallStreamFilter(pieces.append, _TIMER_TOOLS)
        for chunk in ("Sure!", " timer_set", "(duration=", "'30m')"):
            filt.feed(chunk)
        filt.flush()
        joined = "".join(pieces)
        self.assertNotIn("timer_set", joined)
        raw = "Sure! timer_set(duration='30m')"
        self.assertEqual(joined, _extract_tool_calls(raw, _TIMER_TOOLS)[0])


class _StubModel:
    def __init__(self):
        self.kwargs = []

    def create_chat_completion(self, **kwargs):
        self.kwargs.append(kwargs)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok", "tool_calls": []},
                    "finish_reason": "stop",
                }
            ]
        }


class StopSequenceTest(unittest.TestCase):
    def _run_chat(self, tools):
        engine = LLMEngine(Config())
        stub = _StubModel()
        engine._model = stub
        engine.chat([{"role": "user", "content": "hi"}], tools=tools)
        return stub.kwargs[-1]

    def test_stop_sequence_passed_with_tools(self):
        kwargs = self._run_chat(_PLAY_VIDEO_TOOLS)
        self.assertEqual(kwargs["stop"], ["</tool_call>"])

    def test_stop_sequence_absent_without_tools(self):
        kwargs = self._run_chat(None)
        self.assertNotIn("stop", kwargs)


if __name__ == "__main__":
    unittest.main()
