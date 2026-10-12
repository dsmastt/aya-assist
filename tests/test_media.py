"""Tests for assistant.tools.media pure helpers (no network, no subprocess)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.tools import media  # noqa: E402


YT_LINES = (
    "dQw4w9WgXcQ\tRick Astley - Never Gonna Give You Up\t212\tRick Astley\t"
    "https://youtube.com/watch?v=dQw4w9WgXcQ\n"
    "abcdefghijk\tSome Title\t0:42\tSome Channel\n"
    "\n"
    "short\ttoo few columns\n"
)


class ParseYtLinesTest(unittest.TestCase):
    def test_parses_columns(self) -> None:
        results = media.parse_yt_lines(YT_LINES)
        self.assertEqual(len(results), 2)
        first = results[0]
        self.assertEqual(first["id"], "dQw4w9WgXcQ")
        self.assertEqual(first["title"], "Rick Astley - Never Gonna Give You Up")
        self.assertEqual(first["duration"], "212")
        self.assertEqual(first["channel"], "Rick Astley")
        self.assertEqual(
            first["url"], "https://youtube.com/watch?v=dQw4w9WgXcQ"
        )

    def test_synthesizes_watch_url_from_id(self) -> None:
        results = media.parse_yt_lines(YT_LINES)
        self.assertEqual(
            results[1]["url"], "https://www.youtube.com/watch?v=abcdefghijk"
        )

    def test_skips_blank_and_short_lines(self) -> None:
        results = media.parse_yt_lines(YT_LINES)
        ids = [r["id"] for r in results]
        self.assertNotIn("short", ids)

    def test_empty_input(self) -> None:
        self.assertEqual(media.parse_yt_lines(""), [])


class ResolveVideoUrlTest(unittest.TestCase):
    def test_http_passthrough(self) -> None:
        self.assertEqual(
            media.resolve_video_url("https://example.com/v.mp4"),
            "https://example.com/v.mp4",
        )
        self.assertEqual(
            media.resolve_video_url("http://example.com/v"), "http://example.com/v"
        )

    def test_youtube_id_becomes_watch_url(self) -> None:
        self.assertEqual(
            media.resolve_video_url("dQw4w9WgXcQ"),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )

    def test_searcher_top_result_used(self) -> None:
        def fake_searcher(query: str) -> dict:
            self.assertEqual(query, "some show episode 3")
            return {"results": [{"title": "t", "url": "https://found.example/v"}]}

        self.assertEqual(
            media.resolve_video_url("some show episode 3", searcher=fake_searcher),
            "https://found.example/v",
        )

    def test_no_results_raises_media_error(self) -> None:
        fake_searcher = lambda q: {"results": []}  # noqa: E731
        with self.assertRaises(media.MediaError):
            media.resolve_video_url("nothing matches", searcher=fake_searcher)

    def test_result_without_url_raises_media_error(self) -> None:
        fake_searcher = lambda q: {"results": [{"title": "no url"}]}  # noqa: E731
        with self.assertRaises(media.MediaError):
            media.resolve_video_url("broken result", searcher=fake_searcher)

    def test_fabricated_watch_url_raises_media_error(self) -> None:
        with self.assertRaises(media.MediaError):
            media.resolve_video_url(
                "https://www.youtube.com/watch?v=example_video_id"
            )

    def test_bare_youtube_watch_url_raises_media_error(self) -> None:
        with self.assertRaises(media.MediaError):
            media.resolve_video_url("https://www.youtube.com/watch")

    def test_fabricated_youtu_be_url_raises_media_error(self) -> None:
        with self.assertRaises(media.MediaError):
            media.resolve_video_url("https://youtu.be/example_video_id")

    def test_valid_youtube_watch_url_passes_through(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(media.resolve_video_url(url), url)

    def test_valid_youtu_be_url_passes_through(self) -> None:
        url = "https://youtu.be/dQw4w9WgXcQ"
        self.assertEqual(media.resolve_video_url(url), url)

    def test_non_youtube_http_url_passes_through(self) -> None:
        url = "https://vimeo.com/123456789"
        self.assertEqual(media.resolve_video_url(url), url)

    def test_plain_text_still_uses_searcher(self) -> None:
        def fake_searcher(query: str) -> dict:
            self.assertEqual(query, "lofi beats")
            return {"results": [{"title": "lofi", "url": "https://found.example/v"}]}

        self.assertEqual(
            media.resolve_video_url("lofi beats", searcher=fake_searcher),
            "https://found.example/v",
        )


class ParseAniHistoryTest(unittest.TestCase):
    def test_episode_word_form(self) -> None:
        out = media.parse_ani_history(
            "Mushoku Tensei: Jobless Reincarnation Part 2 — episode 4\n"
        )
        self.assertEqual(
            out,
            [{"title": "Mushoku Tensei: Jobless Reincarnation Part 2", "episode": 4}],
        )

    def test_trailing_number_form(self) -> None:
        out = media.parse_ani_history("Steel Ball Run:4\n")
        self.assertEqual(out, [{"title": "Steel Ball Run", "episode": 4}])

    def test_skips_blank_and_numberless_lines(self) -> None:
        out = media.parse_ani_history("\nno episode here\nFrieren 12\n\n")
        self.assertEqual(out, [{"title": "Frieren", "episode": 12}])

    def test_multiple_lines_keep_order(self) -> None:
        out = media.parse_ani_history("One Piece episode 1101\nBocchi the Rock:5\n")
        self.assertEqual(
            out,
            [
                {"title": "One Piece", "episode": 1101},
                {"title": "Bocchi the Rock", "episode": 5},
            ],
        )

    def test_empty_input(self) -> None:
        self.assertEqual(media.parse_ani_history(""), [])


class VenvBinTest(unittest.TestCase):
    def test_venv_bin_detected(self) -> None:
        result = media._venv_bin()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.endswith(os.path.join(".venv", "bin")))


if __name__ == "__main__":
    unittest.main()


import json  # noqa: E402
from unittest import mock  # noqa: E402


API_JSON = json.dumps(
    {
        "kind": "youtube#searchListResponse",
        "items": [
            {
                "id": {"kind": "youtube#video", "videoId": "dQw4w9WgXcQ"},
                "snippet": {
                    "title": "Never Gonna Give You Up",
                    "channelTitle": "Rick Astley",
                    "description": "x" * 250,
                },
            },
            {
                "id": {"kind": "youtube#channel", "channelId": "UC123"},
                "snippet": {"title": "not a video"},
            },
            {"snippet": {"title": "no id at all"}},
            "not-a-dict",
        ],
    }
)


class ParseYoutubeApiTest(unittest.TestCase):
    def test_parses_video_items(self):
        results = media.parse_youtube_api(API_JSON)
        self.assertEqual(len(results), 1)
        first = results[0]
        self.assertEqual(first["id"], "dQw4w9WgXcQ")
        self.assertEqual(first["title"], "Never Gonna Give You Up")
        self.assertEqual(first["channel"], "Rick Astley")
        self.assertEqual(
            first["url"], "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        self.assertEqual(len(first["description"]), 200)

    def test_skips_non_video_items(self):
        results = media.parse_youtube_api(API_JSON)
        ids = [r["id"] for r in results]
        self.assertEqual(ids, ["dQw4w9WgXcQ"])

    def test_invalid_json_yields_empty(self):
        self.assertEqual(media.parse_youtube_api("not json"), [])

    def test_non_dict_payload_yields_empty(self):
        self.assertEqual(media.parse_youtube_api('["a", "b"]'), [])

    def test_empty_items_yields_empty(self):
        self.assertEqual(media.parse_youtube_api('{"items": []}'), [])


class YtSearchApiTest(unittest.TestCase):
    def test_api_success_used(self):
        with mock.patch.object(
            media, "_yt_api_search", return_value=[{"id": "v"}]
        ) as api, mock.patch.object(media, "_run") as run:
            result = media.yt_search("q", api_key="key")
        api.assert_called_once()
        run.assert_not_called()
        self.assertEqual(result["source"], "youtube-api")

    def test_api_error_falls_back_to_ytdlp(self):
        with mock.patch.object(
            media, "_yt_api_search", side_effect=OSError("boom")
        ), mock.patch.object(
            media, "_run", return_value=(0, "id123\tt\td\tc\n", "")
        ):
            result = media.yt_search("q", api_key="key")
        self.assertEqual(result["results"][0]["id"], "id123")
        self.assertNotIn("source", result)

    def test_empty_key_skips_api(self):
        with mock.patch.object(media, "_yt_api_search") as api, mock.patch.object(
            media, "_run", return_value=(0, "", "")
        ):
            media.yt_search("q")
        api.assert_not_called()


class _FakeConfig:
    def __init__(self, media_player):
        self.tools = type("Tools", (), {"media_player": media_player})()


class PlayVideoBrowserDefaultTest(unittest.TestCase):
    def test_default_player_read_from_config(self):
        cfg = _FakeConfig("browser")
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://www.youtube.com/watch?v=abc"
        ), mock.patch.object(
            media.desktop, "open_url", return_value={"method": "browser"}
        ) as open_url:
            result = media.play_video("some video", workspace=3, config=cfg)
        open_url.assert_called_once_with(
            "https://www.youtube.com/watch?v=abc", workspace=3, config=cfg
        )
        self.assertEqual(result["player"], "browser")
        self.assertEqual(result["workspace"], 3)

    def test_no_config_defaults_to_browser(self):
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ), mock.patch.object(
            media.desktop, "open_url", return_value={"method": "browser"}
        ):
            result = media.play_video("v")
        self.assertEqual(result["player"], "browser")

    def test_unset_config_player_defaults_to_browser(self):
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ), mock.patch.object(
            media.desktop, "open_url", return_value={"method": "browser"}
        ):
            result = media.play_video("v", config=_FakeConfig(None))
        self.assertEqual(result["player"], "browser")

    def test_config_player_honored(self):
        cfg = _FakeConfig("mpv")
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ), mock.patch.object(
            media, "_mpv_command", return_value="mpv https://x.example/v"
        ), mock.patch("assistant.tools.media.subprocess.Popen") as popen:
            result = media.play_video("v", config=cfg)
        popen.assert_called_once()
        self.assertEqual(result["player"], "mpv")

    def test_explicit_browser_player_wins(self):
        cfg = _FakeConfig("mpv")
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ), mock.patch.object(
            media.desktop, "open_url", return_value={"method": "browser"}
        ) as open_url:
            result = media.play_video("v", player="browser", config=cfg)
        open_url.assert_called_once()
        self.assertEqual(result["player"], "browser")

    def test_invalid_player_raises(self):
        with mock.patch.object(
            media, "resolve_video_url", return_value="https://x.example/v"
        ):
            with self.assertRaises(media.MediaError):
                media.play_video("v", player="vlc", config=_FakeConfig("browser"))
