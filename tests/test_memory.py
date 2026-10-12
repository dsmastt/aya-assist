"""Tests for assistant.memory.store.MemoryStore."""

import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.memory import MemoryStore  # noqa: E402


class MemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "nested", "dir", "memory.db")
        self.store = MemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmpdir.cleanup()

    def test_add_and_get_fact(self) -> None:
        fid = self.store.add_fact("user drinks coffee")
        self.assertIsInstance(fid, int)
        fact = self.store.get_fact(fid)
        self.assertIsNotNone(fact)
        assert fact is not None
        self.assertEqual(fact["text"], "user drinks coffee")
        self.assertEqual(fact["kind"], "fact")
        self.assertEqual(fact["source"], "agent")
        self.assertEqual(fact["confidence"], 1.0)
        self.assertIsNone(fact["meta"])
        self.assertEqual(fact["access_count"], 0)
        self.assertIsNone(fact["last_accessed"])
        self.assertIsNotNone(fact["created_at"])
        self.assertIsNotNone(fact["updated_at"])
        fid2 = self.store.add_fact(
            "user likes tea",
            kind="preference",
            source="chat",
            confidence=0.9,
            meta={"topic": "drinks"},
        )
        fact2 = self.store.get_fact(fid2)
        self.assertIsNotNone(fact2)
        assert fact2 is not None
        self.assertEqual(fact2["kind"], "preference")
        self.assertEqual(fact2["source"], "chat")
        self.assertEqual(fact2["confidence"], 0.9)
        self.assertEqual(fact2["meta"], {"topic": "drinks"})
        self.assertIsNone(self.store.get_fact(999_999))

    def test_search_relevance(self) -> None:
        ear_id = self.store.add_fact(
            "user's ear-training project lives in /mnt/Data/app/ear-training"
        )
        self.store.add_fact("user likes cats")
        self.store.add_fact("user drinks coffee")
        results = self.store.search_facts("ear training")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], ear_id)
        self.assertIn("ear-training", results[0]["text"])
        expected_keys = {
            "id",
            "text",
            "kind",
            "source",
            "confidence",
            "score",
            "created_at",
            "updated_at",
            "last_accessed",
            "access_count",
        }
        self.assertEqual(set(results[0].keys()), expected_keys)
        no_match = self.store.search_facts("zzzzqqqqx")
        self.assertEqual(no_match, [])

    def test_search_empty_query_returns_recent(self) -> None:
        self.store.add_fact("an older fact about cats")
        newer_id = self.store.add_fact("a newer fact about coffee")
        results = self.store.search_facts("", k=10)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], newer_id)
        self.assertEqual(results[0]["text"], "a newer fact about coffee")

    def test_update_fact(self) -> None:
        fid = self.store.add_fact("original text")
        self.assertTrue(self.store.update_fact(fid, text="updated text", confidence=0.5))
        fact = self.store.get_fact(fid)
        self.assertIsNotNone(fact)
        assert fact is not None
        self.assertEqual(fact["text"], "updated text")
        self.assertEqual(fact["confidence"], 0.5)
        self.assertTrue(self.store.update_fact(fid, kind="preference"))
        self.assertEqual(self.store.get_fact(fid)["kind"], "preference")
        self.assertFalse(self.store.update_fact(999_999, text="nope"))
        search = self.store.search_facts("updated")
        self.assertTrue(any(f["id"] == fid for f in search))

    def test_delete_by_id_and_by_text(self) -> None:
        fid1 = self.store.add_fact("user likes cats")
        fid2 = self.store.add_fact("user likes dogs")
        self.assertEqual(self.store.delete_fact(fact_id=fid1), [fid1])
        self.assertIsNone(self.store.get_fact(fid1))
        self.assertEqual(self.store.delete_fact(text_match="dogs"), [fid2])
        self.assertIsNone(self.store.get_fact(fid2))
        self.assertEqual(self.store.delete_fact(fact_id=123_456), [])
        self.assertEqual(self.store.delete_fact(), [])
        self.assertEqual(self.store.search_facts("dogs"), [])

    def test_kind_filter(self) -> None:
        self.store.add_fact("user likes coffee", kind="preference")
        self.store.add_fact("user likes cats", kind="fact")
        results = self.store.search_facts("likes", kinds=["preference"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["kind"], "preference")
        results2 = self.store.search_facts("", kinds=["preference"])
        self.assertEqual(len(results2), 1)
        self.assertEqual(results2[0]["kind"], "preference")

    def test_access_tracking(self) -> None:
        fid = self.store.add_fact("access tracking fact")
        self.assertEqual(self.store.get_fact(fid)["access_count"], 0)
        self.store.search_facts("tracking")
        fact = self.store.get_fact(fid)
        self.assertEqual(fact["access_count"], 1)
        self.assertIsNotNone(fact["last_accessed"])
        self.store.search_facts("tracking")
        self.assertEqual(self.store.get_fact(fid)["access_count"], 2)

    def test_sessions_and_messages_roundtrip(self) -> None:
        sid = self.store.create_session("planning")
        self.assertIsInstance(sid, int)
        self.store.add_message(sid, "user", "hello")
        self.store.add_message(sid, "assistant", "hi there")
        self.store.add_message(sid, "user", "do the thing")
        recent = self.store.recent_messages(sid, 10)
        self.assertEqual(
            [m["content"] for m in recent],
            ["hello", "hi there", "do the thing"],
        )
        self.assertEqual(set(recent[0].keys()), {"id", "role", "content", "created_at"})
        sessions = self.store.list_sessions()
        self.assertTrue(
            any(s["id"] == sid and s["message_count"] == 3 for s in sessions)
        )
        sess = next(s for s in sessions if s["id"] == sid)
        self.assertEqual(sess["title"], "planning")
        self.store.set_session_title(sid, "renamed")
        sess = next(s for s in self.store.list_sessions() if s["id"] == sid)
        self.assertEqual(sess["title"], "renamed")
        with self.assertRaises(ValueError):
            self.store.add_message(999_999, "user", "orphan")

    def test_persistence(self) -> None:
        fid = self.store.add_fact("persistent fact about rockets")
        self.store.close()
        store2 = MemoryStore(self.db_path)
        self.addCleanup(store2.close)
        results = store2.search_facts("rockets")
        self.assertTrue(any(r["id"] == fid for r in results))

    def test_fts_special_chars_sanitized(self) -> None:
        self.store.add_fact("user's ear-training project")
        self.store.add_fact("C++ pointer arithmetic facts")
        for query in ("user's", "C++", '"quoted" (parens) *star*', "-dash: colon", "100% sure"):
            results = self.store.search_facts(query)
            self.assertIsInstance(results, list)
        starred = self.store.search_facts("***")
        blank = self.store.search_facts("")
        self.assertEqual([f["id"] for f in starred], [f["id"] for f in blank])

    def test_concurrent_smoke(self) -> None:
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                for n in range(10):
                    self.store.add_fact(f"thread {i} fact number {n}")
                    self.store.search_facts(f"thread {i}")
                    self.store.list_facts(5)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_stats(self) -> None:
        self.store.add_fact("fact one")
        self.store.add_fact("preference one", kind="preference")
        sid = self.store.create_session()
        self.store.add_message(sid, "user", "hi")
        stats = self.store.stats()
        self.assertEqual(stats["facts"].get("fact"), 1)
        self.assertEqual(stats["facts"].get("preference"), 1)
        self.assertEqual(stats["messages"], 1)
        self.assertEqual(stats["sessions"], 1)


if __name__ == "__main__":
    unittest.main()
