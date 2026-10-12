"""The streamed final answer must render exactly once on exit."""

from assistant.ui.console import StreamHandle


class _StubConsole:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.is_terminal = True

    def print(self, obj) -> None:
        self.calls.append(getattr(obj, "markup", str(obj)))


def test_stream_prints_once() -> None:
    rc = _StubConsole()
    with StreamHandle(rc, use_live=False) as sh:
        sh.append("hello ")
        sh.append("world")
    assert len(rc.calls) == 1
    assert "hello world" in rc.calls[0]


def test_empty_stream_prints_nothing() -> None:
    rc = _StubConsole()
    with StreamHandle(rc, use_live=False) as sh:
        pass
    assert rc.calls == []
