"""Tests for assistant.tools.web parsers (pure functions, no network)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.tools import web  # noqa: E402


DDG_HTML = """<!DOCTYPE html>
<html><body>
<div class="result results_links results_links_deep web-result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fhello%3Fworld%3D1%26x%3D2&amp;rut=abc123">Example &amp; Site</a>
  <a class="result__snippet">This is a <b>snippet</b> about example.com</a>
</div>
<div class="result">
  <a class="result__a" rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpython.org&amp;rut=def456">Python <em>Programming</em></a>
  <a class="result__snippet">Official Python docs</a>
</div>
</body></html>
"""

DDG_LITE = """<html><body>
<table><tbody>
<tr>
  <td><a rel="nofollow" href="https://example.org" class="result-link">Example Org</a></td>
  <td class="result-snippet">A lite snippet.</td>
</tr>
<tr>
  <td><a class="result-link" rel="nofollow" href="https://example.net">Example Net</a></td>
  <td class="result-snippet">Another snippet.</td>
</tr>
</tbody></table>
</body></html>
"""


class ParseDuckDuckGoHtmlTest(unittest.TestCase):
    def test_parses_results_and_decodes_uddg(self) -> None:
        results = web.parse_duckduckgo_html(DDG_HTML)
        self.assertEqual(len(results), 2)
        first = results[0]
        self.assertEqual(first["url"], "https://example.com/hello?world=1&x=2")
        self.assertEqual(first["title"], "Example & Site")
        self.assertEqual(first["snippet"], "This is a snippet about example.com")
        self.assertEqual(results[1]["url"], "https://python.org")
        self.assertEqual(results[1]["title"], "Python Programming")
        self.assertEqual(results[1]["snippet"], "Official Python docs")

    def test_empty_html_yields_no_results(self) -> None:
        self.assertEqual(web.parse_duckduckgo_html("<html><body>nothing</body></html>"), [])

    def test_uddg_redirect_decoding(self) -> None:
        href = (
            "//duckduckgo.com/l/?uddg="
            "https%3A%2F%2Fexample.com%2Fpath%3Fq%3Dhello%2520world"
            "&amp;rut=xyz"
        )
        html = f'<a class="result__a" href="{href}">T</a>'
        results = web.parse_duckduckgo_html(html)
        self.assertEqual(results[0]["url"], "https://example.com/path?q=hello%20world")


class ParseDuckDuckGoLiteTest(unittest.TestCase):
    def test_parses_links_and_snippets(self) -> None:
        results = web.parse_duckduckgo_lite(DDG_LITE)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["url"], "https://example.org")
        self.assertEqual(results[0]["title"], "Example Org")
        self.assertEqual(results[0]["snippet"], "A lite snippet.")
        self.assertEqual(results[1]["url"], "https://example.net")
        self.assertEqual(results[1]["title"], "Example Net")
        self.assertEqual(results[1]["snippet"], "Another snippet.")

    def test_empty_lite_yields_no_results(self) -> None:
        self.assertEqual(web.parse_duckduckgo_lite("<html><body>x</body></html>"), [])


class ExtractTitleTest(unittest.TestCase):
    def test_title_tag(self) -> None:
        html = "<html><head><title>My &amp; Page</title></head><body></body></html>"
        self.assertEqual(web.extract_title(html), "My & Page")

    def test_fallback_to_h1(self) -> None:
        self.assertEqual(web.extract_title("<html><body><h1>Heading</h1></body></html>"), "Heading")

    def test_no_title(self) -> None:
        self.assertEqual(web.extract_title("<html><body>text</body></html>"), "")


class ExtractTextTest(unittest.TestCase):
    HTML = (
        "<html><head><title>skip me</title>"
        "<script>var a = 1;</script><style>.x{color:red}</style></head>"
        "<body><nav>menu items</nav>"
        "<h1>Hello</h1>"
        "<p>First &amp; para.<br>Second line.</p>"
        "<ul><li>item   one</li><li>item two</li></ul>"
        "<footer>bye bye</footer></body></html>"
    )

    def test_strips_scripts_styles_nav_footer(self) -> None:
        text = web.extract_text(self.HTML)
        self.assertNotIn("var a", text)
        self.assertNotIn(".x{color", text)
        self.assertNotIn("menu items", text)
        self.assertNotIn("bye bye", text)
        self.assertNotIn("skip me", text)

    def test_keeps_visible_text_with_newlines(self) -> None:
        text = web.extract_text(self.HTML)
        self.assertIn("Hello", text)
        self.assertIn("First & para.", text)
        self.assertIn("Second line.", text)
        self.assertIn("item one", text)
        self.assertIn("item two", text)

    def test_collapses_whitespace(self) -> None:
        text = web.extract_text(self.HTML)
        self.assertNotIn("item   one", text)
        self.assertNotIn("\n\n\n", text)

    def test_unescapes_entities(self) -> None:
        text = web.extract_text("<p>a &amp; b &lt; c &gt; d</p>")
        self.assertEqual(text, "a & b < c > d")


if __name__ == "__main__":
    unittest.main()
