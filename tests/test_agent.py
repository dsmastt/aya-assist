"""Tests for the agent loop, session trimming, and tool wiring.

Mock-based: a FakeEngine returns canned responses, a FakeConsole records
output, and a real (temporary) MemoryStore/Config exercise the real tool
registry, memory and resource plumbing. No GPU, no model load.
"""

import asyncio
import contextlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.config import Config  # noqa: E402
from assistant.core.agent import (  # noqa: E402
    Agent,
    _claim_tool_hint,
    _claims_unsupported_action,
    _looks_like_action_claim,
)
from assistant.core.session import Session  # noqa: E402
from assistant.memory import MemoryStore  # noqa: E402
from assistant.tools import builtin  # noqa: E402, F401 — registers tools
from assistant.tools.registry import TOOLS  # noqa: E402


class FakeEngine:
    """Records chat calls and returns canned responses in sequence."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, stream_cb=None, max_tokens=None):
        self.calls.append(list(messages))
        response = (
            dict(self.responses.pop(0))
            if self.responses
            else {"role": "assistant", "content": "", "tool_calls": []}
        )
        if stream_cb and response.get("content"):
            stream_cb(response["content"])
        return response


class _NullStream:
    def __init__(self, callback=None):
        self.callback = callback
        self.chunks = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if self.callback:
            self.callback("".join(self.chunks))
        return False

    def append(self, text):
        self.chunks.append(text)


class FakeConsole:
    def __init__(self, ask_yes_no_result=True):
        self.ask_yes_no_result = ask_yes_no_result
        self.tool_lines = []
        self.markdown_lines = []
        self.warnings = []
        self.errors = []
        self.infos = []
        self.streams = []

    def stream(self):
        return _NullStream(callback=self.streams.append)

    def spinner(self, label):
        return contextlib.nullcontext()

    def tool(self, label):
        self.tool_lines.append(label)

    def markdown(self, text):
        self.markdown_lines.append(text)

    def warn(self, msg):
        self.warnings.append(msg)

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        self.infos.append(msg)

    def ask_yes_no(self, question, default=False):
        return self.ask_yes_no_result

    def hr(self):
        pass


class _AgentBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.cfg.paths.state_file = os.path.join(self.tmpdir.name, "state.json")
        self.store = MemoryStore(os.path.join(self.tmpdir.name, "memory.db"))

    def tearDown(self):
        self.store.close()
        self.tmpdir.cleanup()

    def _agent(self, responses, ask_yes_no=True):
        engine = FakeEngine(responses)
        console = FakeConsole(ask_yes_no_result=ask_yes_no)
        session = Session(self.cfg, self.store)
        agent = Agent(self.cfg, console, engine, self.store, session=session)
        return agent, engine, console, session


class AgentRunOnceTest(_AgentBase):
    def test_tools_registered(self):
        for name in (
            "run_shell",
            "system_status",
            "memory_delete",
            "todo_add",
            "reminder_list",
            "play_video",
        ):
            self.assertIn(name, TOOLS.names())

    def test_tool_round_then_final_answer(self):
        agent, engine, console, session = self._agent(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "1", "name": "system_status", "arguments": {}}
                    ],
                },
                {
                    "role": "assistant",
                    "content": "All good.",
                    "tool_calls": [],
                },
            ]
        )
        asyncio.run(agent.run_once("check system"))

        self.assertEqual(console.markdown_lines, [])
        self.assertEqual(len(engine.calls), 2)

        tool_messages = [m for m in engine.calls[1] if m.get("role") == "tool"]
        self.assertTrue(tool_messages, "expected a tool result message")
        self.assertIn("ram", tool_messages[0]["content"])

        persisted = self.store.recent_messages(session.session_id, 10)
        self.assertEqual(
            [m["content"] for m in persisted], ["check system", "All good."]
        )
        self.assertEqual([m["role"] for m in persisted], ["user", "assistant"])

    def test_assistant_tool_call_shape_is_llama_cpp_style(self):
        agent, engine, *_ = self._agent(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "1", "name": "system_status", "arguments": {}}
                    ],
                },
                {"role": "assistant", "content": "ok", "tool_calls": []},
            ]
        )
        asyncio.run(agent.run_once("check system"))
        assistant_msgs = [m for m in engine.calls[1] if m.get("role") == "assistant"]
        self.assertTrue(assistant_msgs)
        shaped = assistant_msgs[0]["tool_calls"][0]
        self.assertEqual(shaped["type"], "function")
        self.assertEqual(shaped["function"]["name"], "system_status")
        self.assertEqual(shaped["function"]["arguments"], "{}")
        self.assertEqual(shaped["id"], "1")

    def test_denied_approval_yields_denied_tool_result(self):
        agent, engine, console, session = self._agent(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "1", "name": "memory_delete", "arguments": {"fact_id": 999}}
                    ],
                },
                {
                    "role": "assistant",
                    "content": "I could not delete it.",
                    "tool_calls": [],
                },
            ],
            ask_yes_no=False,
        )
        asyncio.run(agent.run_once("forget that"))
        tool_messages = [m for m in engine.calls[1] if m.get("role") == "tool"]
        self.assertTrue(tool_messages)
        self.assertIn("denied by user", tool_messages[0]["content"])
        self.assertEqual(console.markdown_lines, [])

    def test_claim_without_tools_triggers_audit_round(self):
        agent, engine, console, session = self._agent(
            [
                {
                    "role": "assistant",
                    "content": "Setting a 20-minute timer for you",
                    "tool_calls": [],
                },
                {
                    "role": "assistant",
                    "content": (
                        "I did not set a timer — I can only answer questions, "
                        "so use timer_set if you want one."
                    ),
                    "tool_calls": [],
                },
            ]
        )
        asyncio.run(agent.run_once("set a stopwatch of 20 minutes"))

        self.assertEqual(len(engine.calls), 2)
        audit_msgs = [
            m
            for m in engine.calls[1]
            if m.get("role") == "user" and "[audit]" in m.get("content", "")
        ]
        self.assertTrue(audit_msgs, "expected an audit message in round two")
        self.assertTrue(console.streams, "expected the retraction to be streamed")
        self.assertIn("timer_set", "".join(console.streams))
        persisted = self.store.recent_messages(session.session_id, 10)
        self.assertEqual(
            [m["content"] for m in persisted],
            ["set a stopwatch of 20 minutes",
             "I did not set a timer — I can only answer questions, "
             "so use timer_set if you want one."],
        )


    def test_two_round_audit_flow_executes_round_two_tool(self):
        agent, engine, console, session = self._agent(
            [
                {
                    "role": "assistant",
                    "content": "Starting playback of the lecture",
                    "tool_calls": [],
                },
                {
                    "role": "assistant",
                    "content": "I apologize, the playback has started",
                    "tool_calls": [],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "2",
                            "name": "timer_set",
                            "arguments": {"label": "test", "duration": "1m"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "The timer is set.",
                    "tool_calls": [],
                },
            ]
        )
        asyncio.run(agent.run_once("start playback"))

        self.assertEqual(len(engine.calls), 4)
        audit_msgs = {
            m["content"]
            for call in engine.calls
            for m in call
            if m.get("role") == "user" and "[audit]" in m.get("content", "")
        }
        self.assertEqual(len(audit_msgs), 2)
        self.assertTrue(any("play_video" in m for m in audit_msgs))
        self.assertTrue(any("AGAIN claimed" in m for m in audit_msgs))
        self.assertTrue(
            any("timer_set" in line for line in console.tool_lines),
            "expected the round-two tool call to execute",
        )
        persisted = self.store.recent_messages(session.session_id, 10)
        self.assertEqual(persisted[-1]["content"], "The timer is set.")

    def test_exhausted_audit_rounds_warns_and_saves(self):
        agent, engine, console, session = self._agent(
            [
                {
                    "role": "assistant",
                    "content": "Starting playback now",
                    "tool_calls": [],
                },
                {
                    "role": "assistant",
                    "content": "The playback has started, enjoy",
                    "tool_calls": [],
                },
                {
                    "role": "assistant",
                    "content": "Playback is playing right now",
                    "tool_calls": [],
                },
            ]
        )
        asyncio.run(agent.run_once("play something"))

        self.assertEqual(len(engine.calls), 3)
        self.assertTrue(console.warnings)
        self.assertIn("no tool confirmed", console.warnings[0])
        persisted = self.store.recent_messages(session.session_id, 10)
        self.assertEqual(persisted[-1]["content"], "Playback is playing right now")


class LooksLikeActionClaimTest(unittest.TestCase):
    def test_true_for_action_claims(self):
        for text in (
            "Setting a 20-minute timer for you",
            "I opened the browser",
            "It's playing now",
            "You'll be notified when it ends",
            "Starting playback of the lecture",
            "Workspace 4 has been switched to the website",
            "Checking workspace status...",
            "Opening the ChatGPT website",
            "I will start the video soon",
        ):
            self.assertTrue(_looks_like_action_claim(text), text)

    def test_false_for_negations_questions_and_facts(self):
        for text in (
            "It seems there was no active timer",
            "Paris is the capital of France",
            "Which one would you like?",
            "I can't do that",
            "I did not start the playback",
            "I haven't opened anything",
            "The window did not move",
        ):
            self.assertFalse(_looks_like_action_claim(text), text)


class ClaimToolHintTest(unittest.TestCase):
    def test_missing_family_tools_listed(self):
        hints = _claim_tool_hint("Playing lofi now", ["find_video"])
        self.assertIn("play_video", hints)
        self.assertIn("ani_play", hints)

    def test_called_family_tool_gives_no_hint(self):
        self.assertEqual(_claim_tool_hint("Playing lofi now", ["play_video"]), [])

    def test_question_and_negation_guards(self):
        self.assertEqual(_claim_tool_hint("Which one would you like?", []), [])
        self.assertEqual(_claim_tool_hint("I did not open it", []), [])


class ClaimsUnsupportedActionTest(unittest.TestCase):
    def test_true_when_claimed_verb_has_no_matching_tool(self):
        for text, called in (
            ("Playing some lofi in the background", ["find_video"]),
            ("Opening the project", ["find_video"]),
        ):
            self.assertTrue(_claims_unsupported_action(text, called), text)

    def test_false_when_matching_tool_was_called(self):
        for text, called in (
            ("Playing some lofi", ["play_video"]),
            ("Opening the project", ["open_project"]),
        ):
            self.assertFalse(_claims_unsupported_action(text, called), text)

    def test_false_for_negations_and_questions(self):
        for text in (
            "It seems there was no active timer",
            "I can't do that",
            "Which one would you like?",
        ):
            self.assertFalse(_claims_unsupported_action(text, []), text)


class SessionTrimTest(_AgentBase):
    def test_trim_to_tokens_drops_oldest(self):
        session = Session(self.cfg, self.store)
        session.start()
        for index in range(10):
            session.add_message("user", f"message number {index} " + "x" * 200)
        session.trim_to_tokens(60)
        remaining = session.history(100)
        self.assertLess(len(remaining), 10)
        self.assertIn("message number 9", remaining[-1]["content"])

    def test_history_shapes_and_limits(self):
        session = Session(self.cfg, self.store)
        session.start()
        for index in range(5):
            session.add_message("user", f"msg {index}")
        history = session.history(limit=2)
        self.assertEqual(history, [{"role": "user", "content": "msg 3"},
                                   {"role": "user", "content": "msg 4"}])
        session.clear()
        self.assertNotEqual(session.session_id, None)
        self.assertEqual(session.history(), [])


if __name__ == "__main__":
    unittest.main()
