"""Tests for tasks/reminder tools and the memory tool wrappers."""

import asyncio
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.tools import tasks  # noqa: E402
from assistant.tools.memory_tools import (  # noqa: E402
    memory_add,
    memory_delete,
    memory_list,
    memory_recall,
)
from assistant.tools.registry import NeedsApproval  # noqa: E402


def run(coro):
    return asyncio.run(coro)


class ParseWhenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 10, 10, 12, 0, 0)

    def test_in_minutes(self) -> None:
        self.assertEqual(
            tasks.parse_when("in 30m", self.now),
            self.now + timedelta(minutes=30),
        )

    def test_in_hours(self) -> None:
        self.assertEqual(
            tasks.parse_when("in 2h", self.now), self.now + timedelta(hours=2)
        )

    def test_in_seconds(self) -> None:
        self.assertEqual(
            tasks.parse_when("in 45s", self.now), self.now + timedelta(seconds=45)
        )

    def test_in_compound(self) -> None:
        self.assertEqual(
            tasks.parse_when("in 1h30m", self.now),
            self.now + timedelta(hours=1, minutes=30),
        )

    def test_at_future_is_same_day(self) -> None:
        result = tasks.parse_when("at 18:30", self.now)
        self.assertEqual(result.date(), self.now.date())
        self.assertEqual((result.hour, result.minute), (18, 30))

    def test_at_past_rolls_to_tomorrow(self) -> None:
        now = datetime(2026, 10, 10, 20, 0, 0)
        result = tasks.parse_when("at 18:30", now)
        self.assertEqual(result.date() - now.date(), timedelta(days=1))
        self.assertEqual((result.hour, result.minute), (18, 30))
        bare = tasks.parse_when("18:30", now)
        self.assertEqual(bare.date() - now.date(), timedelta(days=1))

    def test_tomorrow_9am(self) -> None:
        result = tasks.parse_when("tomorrow 9am", self.now)
        self.assertEqual(result.date() - self.now.date(), timedelta(days=1))
        self.assertEqual((result.hour, result.minute), (9, 0))

    def test_tomorrow_0900(self) -> None:
        result = tasks.parse_when("tomorrow 09:00", self.now)
        self.assertEqual(result.date() - self.now.date(), timedelta(days=1))
        self.assertEqual((result.hour, result.minute), (9, 0))

    def test_tomorrow_defaults_to_9am(self) -> None:
        result = tasks.parse_when("tomorrow", self.now)
        self.assertEqual(result.date() - self.now.date(), timedelta(days=1))
        self.assertEqual((result.hour, result.minute), (9, 0))

    def test_garbage_raises_value_error(self) -> None:
        for bad in ("garbage", "in", "at 25:99", "tomorrow 13pm"):
            with self.assertRaises(ValueError):
                tasks.parse_when(bad, self.now)


class ToolCtxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state_file = os.path.join(self.tmpdir.name, "state.json")
        self.ctx = SimpleNamespace(
            config=SimpleNamespace(
                paths=SimpleNamespace(state_file=self.state_file)
            )
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()


class TodoToolsTest(ToolCtxTest):
    def test_add_and_list_roundtrip(self) -> None:
        added = run(tasks.todo_add(self.ctx, "buy milk"))
        self.assertEqual(added["id"], 1)
        self.assertEqual(added["title"], "buy milk")
        listed = run(tasks.todo_list(self.ctx))
        self.assertEqual([t["title"] for t in listed["todos"]], ["buy milk"])
        self.assertFalse(listed["todos"][0]["done"])

    def test_list_hides_done_by_default(self) -> None:
        run(tasks.todo_add(self.ctx, "one"))
        run(tasks.todo_add(self.ctx, "two"))
        run(tasks.todo_toggle(self.ctx, todo_id=1))
        visible = run(tasks.todo_list(self.ctx))
        self.assertEqual([t["title"] for t in visible["todos"]], ["two"])
        all_todos = run(tasks.todo_list(self.ctx, include_done=True))
        self.assertEqual(len(all_todos["todos"]), 2)

    def test_toggle_by_id_and_query(self) -> None:
        run(tasks.todo_add(self.ctx, "write report"))
        toggled = run(tasks.todo_toggle(self.ctx, todo_id=1))
        self.assertTrue(toggled["todo"]["done"])
        toggled = run(tasks.todo_toggle(self.ctx, query="REPORT"))
        self.assertFalse(toggled["todo"]["done"])

    def test_toggle_set_done_explicitly(self) -> None:
        run(tasks.todo_add(self.ctx, "pay bills"))
        result = run(tasks.todo_toggle(self.ctx, todo_id=1, done=True))
        self.assertTrue(result["todo"]["done"])
        result = run(tasks.todo_toggle(self.ctx, todo_id=1, done=True))
        self.assertTrue(result["todo"]["done"])

    def test_toggle_no_match(self) -> None:
        run(tasks.todo_add(self.ctx, "existing"))
        self.assertEqual(
            run(tasks.todo_toggle(self.ctx, todo_id=99)),
            {"error": "no matching todo"},
        )
        self.assertEqual(
            run(tasks.todo_toggle(self.ctx, query="zzz")),
            {"error": "no matching todo"},
        )

    def test_corrupt_state_starts_fresh(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as fh:
            fh.write("{not valid json")
        listed = run(tasks.todo_list(self.ctx))
        self.assertEqual(listed["todos"], [])

    def test_persistence_across_calls(self) -> None:
        run(tasks.todo_add(self.ctx, "remember me"))
        listed = run(tasks.todo_list(self.ctx))
        self.assertEqual(len(listed["todos"]), 1)


class ReminderToolsTest(ToolCtxTest):
    def test_add_returns_parsed_time(self) -> None:
        before = datetime.now()
        result = run(tasks.reminder_add(self.ctx, "drink water", "in 30m"))
        after = datetime.now() + timedelta(minutes=30)
        self.assertEqual(result["id"], 1)
        self.assertEqual(result["message"], "drink water")
        parsed = datetime.fromisoformat(result["at"])
        self.assertLessEqual(before + timedelta(minutes=30), parsed)
        self.assertLessEqual(parsed, after)
        self.assertIn("parsed_note", result)

    def test_add_bad_when_returns_error(self) -> None:
        result = run(tasks.reminder_add(self.ctx, "x", "garbage"))
        self.assertIn("error", result)

    def test_list_sorted_by_at(self) -> None:
        run(tasks.reminder_add(self.ctx, "later", "in 2h"))
        run(tasks.reminder_add(self.ctx, "sooner", "in 30m"))
        listed = run(tasks.reminder_list(self.ctx))
        messages = [r["message"] for r in listed["reminders"]]
        self.assertEqual(messages, ["sooner", "later"])

    def test_cancel_roundtrip(self) -> None:
        run(tasks.reminder_add(self.ctx, "doomed", "in 1h"))
        result = run(tasks.reminder_cancel(self.ctx, 1))
        self.assertEqual(result, {"cancelled": True})
        self.assertEqual(run(tasks.reminder_list(self.ctx))["reminders"], [])
        self.assertIn(
            "error", run(tasks.reminder_cancel(self.ctx, 999))
        )


class FakeMemory:
    def __init__(self) -> None:
        self.facts: list[dict] = []
        self.next_id = 1

    def add_fact(self, text, kind="fact", source="agent", confidence=1.0):
        fid = self.next_id
        self.next_id += 1
        self.facts.append(
            {
                "id": fid,
                "text": text,
                "kind": kind,
                "source": source,
                "confidence": confidence,
            }
        )
        return fid

    def search_facts(self, query, k=5):
        return [
            {
                "id": f["id"],
                "text": f["text"],
                "kind": f["kind"],
                "score": 1.0,
                "updated_at": "2026-10-10T12:00:00",
                "last_accessed": None,
            }
            for f in self.facts
            if query.casefold() in f["text"].casefold()
        ][:k]

    def list_facts(self, limit=50):
        return self.facts[:limit]

    def delete_fact(self, fact_id=None, text_match=None):
        removed: list[int] = []
        kept: list[dict] = []
        for fact in self.facts:
            if fact_id is not None and fact["id"] == fact_id:
                removed.append(fact["id"])
            elif text_match and text_match.casefold() in fact["text"].casefold():
                removed.append(fact["id"])
            else:
                kept.append(fact)
        self.facts = kept
        return removed


class MemoryToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = FakeMemory()
        self.approvals: list[tuple[str, str]] = []
        self.ctx = SimpleNamespace(memory=self.memory, approve=self._approve)

    def _approve(self, label, detail):
        self.approvals.append((label, detail))
        return True

    def test_add(self) -> None:
        result = run(memory_add(self.ctx, "user likes coffee"))
        self.assertEqual(result, {"id": 1, "stored": True})
        self.assertEqual(self.memory.facts[0]["kind"], "fact")
        self.assertEqual(self.memory.facts[0]["source"], "agent")

    def test_recall_projects_keys(self) -> None:
        run(memory_add(self.ctx, "user likes coffee"))
        result = run(memory_recall(self.ctx, "coffee"))
        self.assertEqual(
            set(result["matches"][0].keys()),
            {"id", "text", "kind", "score", "updated_at", "last_accessed"},
        )
        self.assertEqual(result["matches"][0]["id"], 1)

    def test_list_and_kind_filter(self) -> None:
        run(memory_add(self.ctx, "fact one"))
        run(memory_add(self.ctx, "pref one", kind="preference"))
        self.assertEqual(len(run(memory_list(self.ctx))["facts"]), 2)
        filtered = run(memory_list(self.ctx, kind="preference"))
        self.assertEqual([f["kind"] for f in filtered["facts"]], ["preference"])

    def test_delete_needs_id_or_query(self) -> None:
        result = run(memory_delete(self.ctx))
        self.assertEqual(result, {"error": "give an id or a text query"})

    def test_delete_approved_by_id(self) -> None:
        run(memory_add(self.ctx, "user likes cats"))
        result = run(memory_delete(self.ctx, fact_id=1))
        self.assertEqual(result, {"deleted": [1]})
        self.assertEqual(self.memory.facts, [])
        self.assertEqual(self.approvals[0][0], "memory delete")

    def test_delete_approved_by_query(self) -> None:
        run(memory_add(self.ctx, "user likes dogs"))
        result = run(memory_delete(self.ctx, query="dogs"))
        self.assertEqual(result, {"deleted": [1]})

    def test_delete_denied(self) -> None:
        run(memory_add(self.ctx, "secret fact"))
        self.ctx.approve = lambda label, detail: False
        result = run(memory_delete(self.ctx, fact_id=1))
        self.assertIn("error", result)
        self.assertEqual(len(self.memory.facts), 1)

    def test_delete_needs_approval_propagates(self) -> None:
        run(memory_add(self.ctx, "guarded fact"))

        def raise_approval(label, detail):
            raise NeedsApproval(label, detail)

        self.ctx.approve = raise_approval
        with self.assertRaises(NeedsApproval):
            run(memory_delete(self.ctx, fact_id=1))


class ParseDurationTest(unittest.TestCase):
    def test_minutes_forms(self) -> None:
        for text in ("20m", "20 min", "20 minutes", "20mins", "20M"):
            self.assertEqual(tasks._parse_duration(text), 1200)

    def test_hours(self) -> None:
        self.assertEqual(tasks._parse_duration("2h"), 7200)

    def test_compound(self) -> None:
        self.assertEqual(tasks._parse_duration("1h30m"), 5400)

    def test_seconds_forms(self) -> None:
        self.assertEqual(tasks._parse_duration("90s"), 90)
        self.assertEqual(tasks._parse_duration("45 sec"), 45)
        self.assertEqual(tasks._parse_duration("45 seconds"), 45)

    def test_leading_in(self) -> None:
        self.assertEqual(tasks._parse_duration("in 20m"), 1200)

    def test_mmss_forms(self) -> None:
        self.assertEqual(tasks._parse_duration("1:30"), 90)
        self.assertEqual(tasks._parse_duration("20:00"), 1200)

    def test_bad_values_raise(self) -> None:
        for bad in ("", "abc", "20"):
            with self.assertRaises(ValueError):
                tasks._parse_duration(bad)


class TimerToolsTest(ToolCtxTest):
    def test_set_status_cancel_roundtrip(self) -> None:
        result = run(tasks.timer_set(self.ctx, label="tea", duration="90s"))
        self.assertEqual(result["id"], 1)
        self.assertEqual(result["label"], "tea")
        self.assertEqual(result["duration_seconds"], 90)
        self.assertIn("end_at", result)

        status = run(tasks.timer_status(self.ctx))
        self.assertEqual(len(status["timers"]), 1)
        timer = status["timers"][0]
        self.assertEqual(timer["id"], 1)
        self.assertEqual(timer["label"], "tea")
        self.assertEqual(timer["status"], "running")
        self.assertGreaterEqual(timer["remaining_seconds"], 0)
        self.assertEqual(status["next"]["id"], 1)

        self.assertEqual(run(tasks.timer_cancel(self.ctx, 1)), {"cancelled": 1})
        self.assertEqual(run(tasks.timer_status(self.ctx))["timers"], [])
        self.assertEqual(
            run(tasks.timer_cancel(self.ctx, 1)),
            {"error": "no timer with id 1; running timers: []"},
        )

    def test_cancel_without_id_cancels_sole_running_timer(self) -> None:
        result = run(tasks.timer_set(self.ctx, label="tea", duration="90s"))
        self.assertEqual(run(tasks.timer_cancel(self.ctx)), {"cancelled": result["id"]})
        self.assertEqual(run(tasks.timer_status(self.ctx))["timers"], [])

    def test_cancel_without_id_and_two_timers_lists_both(self) -> None:
        run(tasks.timer_set(self.ctx, label="tea", duration="90s"))
        run(tasks.timer_set(self.ctx, label="coffee", duration="2h"))
        result = run(tasks.timer_cancel(self.ctx))
        self.assertIn("error", result)
        self.assertIn("no timer id given", result["error"])
        self.assertIn("1: tea", result["error"])
        self.assertIn("2: coffee", result["error"])

    def test_unknown_id_error_lists_running_timers(self) -> None:
        run(tasks.timer_set(self.ctx, label="tea", duration="90s"))
        result = run(tasks.timer_cancel(self.ctx, 999))
        self.assertIn("no timer with id 999", result["error"])
        self.assertIn("1: tea", result["error"])

    def test_status_done_after_end(self) -> None:
        result = run(tasks.timer_set(self.ctx, label="beep", duration="20s"))
        end = datetime.fromisoformat(result["end_at"])
        status = run(tasks.timer_status(self.ctx, _now=end + timedelta(seconds=1)))
        self.assertEqual(status["timers"][0]["status"], "done")
        self.assertLess(status["timers"][0]["remaining_seconds"], 0)
        self.assertIsNone(status["next"])

    def test_set_bad_duration_returns_error(self) -> None:
        for bad in ("20", "abc", ""):
            result = run(tasks.timer_set(self.ctx, duration=bad))
            self.assertIn("error", result)
            self.assertIn("accepted formats", result["error"])

    def test_drain_returns_due_labels_and_empties_state(self) -> None:
        result = run(tasks.timer_set(self.ctx, label="break", duration="30s"))
        end = datetime.fromisoformat(result["end_at"])
        drained = run(tasks.timer_drain(self.ctx, _now=end + timedelta(seconds=1)))
        self.assertEqual(drained, ["break"])
        self.assertEqual(run(tasks.timer_status(self.ctx))["timers"], [])
        self.assertEqual(run(tasks.timer_drain(self.ctx)), [])

    def test_drain_keeps_future_timers(self) -> None:
        run(tasks.timer_set(self.ctx, label="soon", duration="10m"))
        drained = run(tasks.timer_drain(self.ctx))
        self.assertEqual(drained, [])
        status = run(tasks.timer_status(self.ctx))
        self.assertEqual(len(status["timers"]), 1)

    def test_state_without_timers_key_loads(self) -> None:
        with open(self.state_file, "w", encoding="utf-8") as fh:
            fh.write('{"todos": [{"id": 1, "title": "old", "done": false}]}')
        status = run(tasks.timer_status(self.ctx))
        self.assertEqual(status["timers"], [])
        self.assertIsNone(status["next"])
        self.assertEqual(status["note"], "no timers")
        listed = run(tasks.todo_list(self.ctx))
        self.assertEqual([t["title"] for t in listed["todos"]], ["old"])


if __name__ == "__main__":
    unittest.main()
