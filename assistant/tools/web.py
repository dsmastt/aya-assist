"""Web fetching and DuckDuckGo search, stdlib only.

Uses :mod:`urllib.request` (no third-party HTTP clients). Search results
come from DuckDuckGo's HTML endpoints; page fetching extracts plain text
with an :class:`html.parser.HTMLParser`-based scraper.

All functions return JSON-serializable dicts (or plain values for the pure
parsers) and raise :class:`WebError` on hard failures; success is never
fabricated.
"""

from __future__ import annotations

import http.client
import re
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from html.parser import HTMLParser
from typing import Any

DEFAULT_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:156.0) Gecko/20100101 Firefox/156.0"

_SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "head",
    "nav",
    "footer",
    "template",
    "iframe",
    "svg",
}

_BLOCK_TAGS = {
    "br",
    "p",
    "div",
    "li",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "blockquote",
    "table",
    "section",
    "article",
    "header",
    "main",
}

_RESULT_A_RE = re.compile(r'<a\b[^>]*class="result__a"[^>]*>(.*?)</a>', re.S | re.I)
_SNIPPET_A_RE = re.compile(r'<a\b[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.S | re.I)
_LITE_LINK_RE = re.compile(r'<a\b[^>]*class="result-link"[^>]*>(.*?)</a>', re.S | re.I)
_LITE_SNIPPET_RE = re.compile(
    r'<td\b[^>]*class="result-snippet"[^>]*>(.*?)</td>', re.S | re.I
)
_HREF_RE = re.compile(r'href="([^"]+)"', re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SNIFF = (b"<!doctype html", b"<html", b"<head", b"<body", b"<title")


class WebError(RuntimeError):
    """Raised when a network fetch or search fails hard."""


def _strip_tags(fragment: str) -> str:
    """Remove all tags from an HTML fragment and unescape entities."""
    return unescape(_TAG_RE.sub("", fragment)).strip()


def _resolve_uddg(href: str) -> str:
    """Turn a DuckDuckGo ``//duckduckgo.com/l/?uddg=...`` link into the real URL."""
    target = href
    if target.startswith("//"):
        target = "https:" + target
    if target.lower().startswith(("http://", "https://")):
        parsed = urllib.parse.urlsplit(target)
        for part in parsed.query.split("&"):
            if part.startswith("uddg="):
                return urllib.parse.unquote(part[len("uddg=") :])
    return href


def _fetch_meta(url: str, timeout: float = 15.0, max_bytes: int = 2_000_000) -> tuple[bytes, str]:
    """Fetch ``url``, returning the body (truncated to ``max_bytes``) and content-type.

    Truncation is not an error. Raises :class:`WebError` on HTTP/network
    failures and timeouts, with the status and URL in the message.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        },
    )
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as resp:
            content_type = resp.headers.get("Content-Type", "")
            chunks: list[bytes] = []
            total = 0
            while total < max_bytes:
                try:
                    chunk = resp.read(65536)
                except http.client.IncompleteRead as exc:
                    chunk = exc.partial
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            return b"".join(chunks), content_type
    except urllib.error.HTTPError as exc:
        raise WebError(f"HTTP error {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise WebError(f"timed out fetching {url}") from exc
        raise WebError(f"cannot fetch {url}: {reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise WebError(f"timed out fetching {url}") from exc


def _fetch(url: str, timeout: float = 15.0, max_bytes: int = 2_000_000) -> bytes:
    """Fetch ``url`` and return the raw body bytes."""
    return _fetch_meta(url, timeout=timeout, max_bytes=max_bytes)[0]


def parse_duckduckgo_html(html: str) -> list[dict]:
    """Parse DuckDuckGo's html endpoint into ``{"title", "url", "snippet"}`` dicts.

    Links are ``//duckduckgo.com/l/?uddg=<urlencoded>`` redirects; the
    ``uddg`` parameter is decoded to the real URL. Pure function, no network.
    """
    matches = list(_RESULT_A_RE.finditer(html))
    snippets = _SNIPPET_A_RE.findall(html)
    results: list[dict] = []
    for index, match in enumerate(matches):
        open_tag = match.group(0)[: match.group(0).find(">") + 1]
        href_match = _HREF_RE.search(open_tag)
        if not href_match:
            continue
        title = _strip_tags(match.group(1))
        snippet = _strip_tags(snippets[index]) if index < len(snippets) else ""
        if not title and not snippet:
            continue
        results.append(
            {
                "title": title,
                "url": _resolve_uddg(href_match.group(1)),
                "snippet": snippet,
            }
        )
    return results


def parse_duckduckgo_lite(html: str) -> list[dict]:
    """Parse DuckDuckGo's lite endpoint into ``{"title", "url", "snippet"}`` dicts.

    Pure function, no network.
    """
    matches = list(_LITE_LINK_RE.finditer(html))
    snippets = _LITE_SNIPPET_RE.findall(html)
    results: list[dict] = []
    for index, match in enumerate(matches):
        open_tag = match.group(0)[: match.group(0).find(">") + 1]
        href_match = _HREF_RE.search(open_tag)
        if not href_match:
            continue
        title = _strip_tags(match.group(1))
        snippet = _strip_tags(snippets[index]) if index < len(snippets) else ""
        if not title and not snippet:
            continue
        results.append(
            {
                "title": title,
                "url": href_match.group(1),
                "snippet": snippet,
            }
        )
    return results


def search_duckduckgo(query: str, max_results: int = 6, timeout: float = 15.0) -> dict:
    """Search DuckDuckGo, falling back from the html endpoint to the lite one.

    The lite endpoint is tried when the html endpoint yields no results or
    fails hard. A :class:`WebError` is raised only when both endpoints fail;
    an empty result set with both endpoints reachable is reported honestly as
    ``results: []``.
    """
    html_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    lite_url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote_plus(query)

    results: list[dict] = []
    html_error: WebError | None = None
    try:
        body = _fetch(html_url, timeout=timeout)
        results = parse_duckduckgo_html(body.decode("utf-8", errors="replace"))
    except WebError as exc:
        html_error = exc
        results = []

    lite_error: WebError | None = None
    if not results:
        try:
            body = _fetch(lite_url, timeout=timeout)
            results = parse_duckduckgo_lite(body.decode("utf-8", errors="replace"))
        except WebError as exc:
            lite_error = exc
            results = []

    if not results and html_error is not None and lite_error is not None:
        raise WebError(
            f"duckduckgo search failed on both endpoints — html: {html_error}; "
            f"lite: {lite_error}"
        )

    truncated = len(results) > max_results
    return {
        "query": query,
        "engine": "duckduckgo",
        "results": results[:max_results],
        "truncated": truncated,
    }


class _TitleParser(HTMLParser):
    """Collects text from ``<title>`` with an ``<h1>`` fallback."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self._in_h1 = False
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "title":
            self._in_title = True
        elif tag == "h1":
            self._in_h1 = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif self._in_h1:
            self.h1_parts.append(data)


class _TextParser(HTMLParser):
    """Extracts visible text, skipping non-content tags, marking block breaks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)


def _collapse(text: str) -> str:
    """Collapse horizontal whitespace runs and 3+ newlines, then strip."""
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title(html: str) -> str:
    """Extract the page title from ``<title>`` (falling back to ``<h1>``)."""
    parser = _TitleParser()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 — malformed HTML must never kill extraction
        pass
    title = _collapse(" ".join(parser.title_parts))
    if title:
        return title
    return _collapse(" ".join(parser.h1_parts))


def extract_text(html: str) -> str:
    """Extract visible text from HTML.

    Script/style/noscript/head/nav/footer/template (and their content) are
    skipped; ``br``/``p``/``div``/``li`` become newlines; entities are
    unescaped and whitespace is collapsed.
    """
    parser = _TextParser()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001
        pass
    return _collapse("".join(parser.parts))


def fetch_page(url: str, timeout: float = 15.0, max_chars: int = 8000) -> dict:
    """Fetch a page and return its title and plain text.

    HTML pages are parsed with :func:`extract_title`/:func:`extract_text`
    and the text truncated to ``max_chars`` with a marker; non-HTML content
    is decoded as text (errors replaced) and truncated the same way.
    """
    data, content_type = _fetch_meta(url, timeout=timeout)
    ctype = content_type.lower()
    is_html = "text/html" in ctype
    if not ctype:
        head = data[:512].lstrip().lower()
        is_html = head.startswith(_HTML_SNIFF)
    if is_html:
        html = data.decode("utf-8", errors="replace")
        title = extract_title(html)
        text = extract_text(html)
    else:
        title = ""
        text = data.decode("utf-8", errors="replace")

    full_length = len(text)
    truncated = full_length > max_chars
    if truncated:
        text = text[:max_chars] + "…[truncated]"
    return {
        "url": url,
        "title": title,
        "text": text,
        "truncated": truncated,
        "length": full_length,
    }
