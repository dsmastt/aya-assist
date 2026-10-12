"""Local LLM inference via llama-cpp-python.

``LLMEngine`` owns the lifecycle of a single ``Llama`` instance: VRAM
estimation before load, pre-load resource checks (delegated to
``assistant.core.resources``), chat completions (streaming or not) with
OpenAI-style tool-call support, and idle tracking so the agent loop can
unload an unused model.

VRAM estimate heuristic
-----------------------
- ``weights_mb``: GGUF file size (MiB) * 1.15. The 15% headroom covers
  runtime tensor repacking, the CUDA context and small per-tensor overhead
  that is not present in the on-disk size.
- ``kv_mb``: per-token KV-cache cost for a 7-8B GQA model: ``n_ctx * 0.035``
  MB when KV-cache quantization (``kv_quant == "q8_0"``) is enabled, else
  ``n_ctx * 0.06`` MB (F16 approximation). Basis: ~30 KB/token with Q8_0 KV,
  ~55 KB/token with F16, plus ~15% weights overhead on ``weights_mb``.
"""

from __future__ import annotations

import gc
import inspect
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Config
from ..core.resources import check_model_load, gpu_info


class EngineError(RuntimeError):
    """Raised when the model cannot be loaded or inference fails."""


#: Known-good models for the setup CLI: name -> {file, url, notes, min_vram_mb}
MODEL_PRESETS: dict[str, dict] = {
    "qwen2.5-7b-instruct-q4_k_m": {
        "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "url": (
            "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/"
            "resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
        ),
        "notes": (
            "Qwen2.5-7B-Instruct, Q4_K_M quantization, ~4.7 GB on disk. "
            "Fits an 8 GB card with Q8_0 KV cache quantization."
        ),
        "min_vram_mb": 6200,
    },
}


def available_chat_formats() -> list[str]:
    """Names of the built-in chat formats registered with llama_cpp."""
    try:
        import llama_cpp.llama_chat_format as llama_chat_format

        registry = llama_chat_format.LlamaChatCompletionHandlerRegistry()
        handlers = getattr(registry, "_chat_handlers", None)
        if handlers:
            return sorted(handlers.keys())
    except Exception:
        pass
    return []


def resolve_chat_format(requested: str, available: list[str] | None = None) -> str:
    """Map ``cfg.model.chat_format`` onto a concrete llama_cpp format name.

    ``"auto"`` prefers a built-in ``"qwen2.5"`` handler when llama_cpp
    registers one, otherwise falls back to ``"chatml"`` (Qwen2.5's template
    is ChatML-based, and llama_cpp's chatml handler supports OpenAI-style
    tool calls). Explicit values are used as-is.
    """
    if requested and requested != "auto":
        return requested
    if available is None:
        available = available_chat_formats()
    return "qwen2.5" if "qwen2.5" in available else "chatml"


def _kv_cache_q8_value() -> int | None:
    """ctypes value enabling the Q8_0 KV-cache type, or None if unsupported.

    Newer llama_cpp builds may expose ``llama_cpp.llama_kv_cache_type``
    (Q8_0 == 1). The bundled backend (llama.cpp 0.5.0-dev) instead exposes
    ``type_k``/``type_v`` on the ``Llama`` constructor as raw ``ggml_type``
    values, where ``GGML_TYPE_Q8_0 == 8`` (``GGML_TYPE_F16 == 1`` default).
    """
    try:
        import llama_cpp

        enum = getattr(llama_cpp, "llama_kv_cache_type", None)
        if enum is not None:
            return int(getattr(enum, "LLAMA_KV_CACHE_TYPE_Q8_0", 1))
    except Exception:
        pass
    return 8  # GGML_TYPE_Q8_0


def _normalize_tool_calls(raw: Any) -> list[dict]:
    """Normalize OpenAI-style tool calls into ``{id, name, arguments}``."""
    calls: list[dict] = []
    if not raw:
        return calls
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
        elif not isinstance(args, dict):
            args = {}
        calls.append(
            {
                "id": tc.get("id"),
                "name": fn.get("name", ""),
                "arguments": args,
            }
        )
    return calls


def _accumulate_stream_tool_calls(acc: list[dict], fragments: Any) -> None:
    """Merge incremental ``delta.tool_calls`` fragments by index."""
    if not fragments:
        return
    for frag in fragments:
        if not isinstance(frag, dict):
            continue
        idx = frag.get("index", 0)
        while len(acc) <= idx:
            acc.append({"id": None, "function": {"name": "", "arguments": ""}})
        slot = acc[idx]
        if frag.get("id"):
            slot["id"] = frag["id"]
        fn = frag.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]


_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_TOOLS_BLOCK_HEADER = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML "
    "tags:\n<tools>\n"
)
_TOOLS_BLOCK_FOOTER = (
    "</tools>\n\n"
    "To call a tool you MUST emit exactly this on its own line:\n"
    '<tool_call>{"name": "<tool_name>", "arguments": {<json>}}</tool_call>\n\n'
    "Any JSON outside these tags is NOT executed — it is just text. "
    "Never print tool-call JSON in your spoken reply."
)


def _tool_names(tools: Any) -> list[str]:
    """Collect valid tool names from an OpenAI-style tools list."""
    names: list[str] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            fn = tool
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _make_call(idx: int, name: str, arguments: Any) -> dict:
    """Build a normalized ``{id, name, arguments}`` tool call."""
    args = arguments or {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"id": f"call_{idx}_{name}", "name": name, "arguments": args}


def _skip_string(text: str, pos: int) -> int:
    """Index just past the closing quote of the string at ``pos`` (a ``"``),
    or ``len(text)`` when unterminated."""
    i = pos + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            return i + 1
        i += 1
    return len(text)


def _match_brace(text: str, start: int) -> int | None:
    """Index of the brace matching ``text[start] == '{'`` (string/escape
    aware), or None when the object is never closed."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _iter_json_objects(text: str):
    """Yield ``(start, end, obj)`` for every dict parseable from a ``{...}``
    span in ``text`` (string/escape aware, nested objects supported)."""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            i = _skip_string(text, i)
            continue
        if ch != "{":
            i += 1
            continue
        end = _match_brace(text, i)
        if end is None:
            i += 1
            continue
        try:
            obj = json.loads(text[i:end + 1])
        except (json.JSONDecodeError, TypeError):
            i = end + 1
            continue
        if isinstance(obj, dict):
            yield i, end, obj
        i = end + 1


def _mentioned_tool(
    text: str, start: int, names: list[str], window: int = 200
) -> str | None:
    """Name of a known tool mentioned within ``window`` chars before
    ``start`` (the nearest mention wins)."""
    if start <= 0 or not names:
        return None
    window_text = text[max(0, start - window):start]
    if not window_text:
        return None
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(name) for name in names) + r")(?![A-Za-z0-9_])"
    )
    match = None
    for m in pattern.finditer(window_text):
        match = m
    return match.group(1) if match else None


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove the given ``[start, end]`` spans from ``text``."""
    out: list[str] = []
    pos = 0
    for start, end in sorted(spans):
        if start < pos:
            continue
        out.append(text[pos:start])
        pos = end + 1
    out.append(text[pos:])
    return "".join(out)


_PARSE_FAIL = object()


def _split_top_level_commas(text: str) -> list[str]:
    """Split on commas at depth 0, respecting quotes."""
    parts: list[str] = []
    cur: list[str] = []
    quote = None
    depth = 0
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return parts


def _parse_pseudo_value(raw: str):
    raw = raw.strip()
    if not raw:
        return _PARSE_FAIL
    if raw[0] in "\"'":
        if len(raw) >= 2 and raw[-1] == raw[0]:
            return raw[1:-1]
        return _PARSE_FAIL
    if raw in ("true", "false"):
        return raw == "true"
    if raw in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw):
        return raw
    return _PARSE_FAIL


def _find_matching_paren(text: str, open_pos: int) -> int | None:
    depth = 0
    quote = None
    i = open_pos
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote and text[i - 1] != "\\":
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _parse_pseudo_args(inner: str) -> dict | None:
    if not inner.strip():
        return {}
    args: dict = {}
    for part in _split_top_level_commas(inner):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            return None
        key, _, value = part.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            return None
        parsed = _parse_pseudo_value(value)
        if parsed is _PARSE_FAIL:
            return None
        args[key] = parsed
    return args


def _extract_pseudo_calls(
    text: str, names: list[str]
) -> tuple[list[tuple[int, int]], list[dict]] | None:
    """Extract pseudo-code tool calls like ``timer_set(duration='30m')``."""
    spans: list[tuple[int, int]] = []
    calls: list[dict] = []
    for name in names:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\("
        )
        for m in pattern.finditer(text):
            open_pos = text.find("(", m.start())
            end = _find_matching_paren(text, open_pos)
            if end is None:
                continue
            args = _parse_pseudo_args(text[open_pos + 1:end])
            if args is None:
                continue
            spans.append((m.start(), end))
            calls.append(_make_call(len(calls), name, args))
    if not spans:
        return None
    return spans, calls


def _extract_xml_calls(text: str) -> tuple[str, list[dict]]:
    """Primary path: split ``<tool_call>`` blocks out of completion text.

    Returns ``(cleaned_text, calls)``; each call is
    ``{"id", "name", "arguments"}`` with ``arguments`` always a dict.
    An opener whose closing tag is missing at the END of the text is
    leniently treated as a call when its body parses to a named object
    (the model frequently stops right after the JSON).
    """
    calls: list[dict] = []
    kept: list[str] = []
    pos = 0
    while True:
        start = text.find(_TOOL_CALL_OPEN, pos)
        if start == -1:
            kept.append(text[pos:])
            break
        end = text.find(_TOOL_CALL_CLOSE, start + len(_TOOL_CALL_OPEN))
        if end == -1:
            body = text[start + len(_TOOL_CALL_OPEN):].strip()
            payload = None
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                pass
            if isinstance(payload, dict) and payload.get("name"):
                calls.append(
                    _make_call(len(calls), str(payload["name"]), payload.get("arguments"))
                )
                break
            kept.append(text[pos:])
            break
        kept.append(text[pos:start])
        body = text[start + len(_TOOL_CALL_OPEN):end].strip()
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict) and payload.get("name"):
            calls.append(
                _make_call(len(calls), str(payload["name"]), payload.get("arguments"))
            )
        pos = end + len(_TOOL_CALL_CLOSE)
    return "".join(kept), calls


def _extract_tool_calls(
    text: str, tools: list[dict] | None = None
) -> tuple[str, list[dict]]:
    """Split tool calls out of completion text and return the cleaned text.

    Primary: ``<tool_call>`` XML blocks (with lenient unclosed handling).
    When no XML calls were found and valid tool names are known, fall back
    to: (A) bare-JSON objects carrying a known ``"name"``, (B) objects
    without a ``"name"`` that follow a mention of a known tool within
    ~200 chars, (C) pseudo-code calls like ``timer_set(duration='30m')``.
    All matched spans are removed from the content.
    """
    cleaned, calls = _extract_xml_calls(text)
    if calls:
        return cleaned, calls
    names = _tool_names(tools)
    if not names:
        return cleaned, calls
    objects = list(_iter_json_objects(cleaned))
    spans: list[tuple[int, int]] = []
    for start, end, obj in objects:
        name = obj.get("name")
        if isinstance(name, str) and name in names:
            spans.append((start, end))
            calls.append(_make_call(len(calls), name, obj.get("arguments")))
    if not spans:
        for start, end, obj in objects:
            if "name" in obj:
                continue
            name = _mentioned_tool(cleaned, start, names)
            if name:
                spans.append((start, end))
                calls.append(_make_call(len(calls), name, obj))
    if not spans:
        pseudo = _extract_pseudo_calls(cleaned, names)
        if pseudo:
            spans, extra = pseudo
            calls.extend(extra)
    if not spans:
        return cleaned, calls
    return _strip_spans(cleaned, spans), calls

def _merge_extracted_calls(calls: list[dict], extra: list[dict]) -> None:
    """Append text-extracted tool calls that are not already present."""
    seen = {(c.get("id"), c.get("name")) for c in calls}
    for call in extra:
        if (call["id"], call["name"]) not in seen:
            calls.append(call)


def _adapt_messages(messages: list[dict]) -> list[dict]:
    """Translate tool-result messages into user-role text that the chatml
    handler (which renders only system/user/assistant) keeps in the prompt."""
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id") or "?"
            out.append(
                {
                    "role": "user",
                    "content": (
                        f"<tool_result id={cid}>\n"
                        f"{m.get('content') or ''}\n"
                        "</tool_result>"
                    ),
                }
            )
        else:
            out.append(m)
    return out


def _inject_tools(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Render tool schemas into the system message (Qwen2.5 XML format).

    The caller's messages are never mutated: a shallow copy with a copied
    system message is returned, so repeated rounds do not accumulate
    duplicate tool blocks.
    """
    lines = [_TOOLS_BLOCK_HEADER]
    for tool in tools:
        lines.append(json.dumps(tool, ensure_ascii=False))
        lines.append("")
    lines.append(_TOOLS_BLOCK_FOOTER)
    block = "\n".join(lines)
    out = list(messages)
    for idx, m in enumerate(out):
        if m.get("role") == "system":
            sys_msg = dict(m)
            sys_msg["content"] = f"{m.get('content') or ''}\n\n{block}"
            out[idx] = sys_msg
            return out
    out.insert(0, {"role": "system", "content": block})
    return out


_JSON_OPEN_RE = re.compile(r'\{\s*"')


class _ToolCallStreamFilter:
    """Stream deltas while holding back tool-call fragments.

    Text is emitted only up to the last possible tool-call opener: the start
    of an unclosed ``<tool_call`` block, or (when valid tool names are
    known) a ``{"`` that may begin a bare-JSON tool call. Complete XML
    blocks are dropped; everything held back is reconciled at flush so the
    emitted text equals the final stripped content.
    """

    def __init__(
        self, cb: Callable[[str], None], tools: list[dict] | None = None
    ) -> None:
        self._cb = cb
        self._held = ""
        self._raw = ""
        self._emitted_len = 0
        self._tools = tools
        self._names = _tool_names(tools)

    def _emit(self, piece: str) -> None:
        if not piece:
            return
        self._emitted_len += len(piece)
        self._cb(piece)

    def feed(self, piece: str) -> None:
        if not piece:
            return
        self._raw += piece
        self._held += piece
        while self._held:
            start = self._held.find(_TOOL_CALL_OPEN)
            json_pos = -1
            if self._names:
                match = _JSON_OPEN_RE.search(self._held)
                if match is not None:
                    json_pos = match.start()
            pseudo_pos = -1
            if self._names:
                for name in self._names:
                    pm = re.compile(
                        r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\("
                    ).search(self._held)
                    if pm is not None and (pseudo_pos == -1 or pm.start() < pseudo_pos):
                        pseudo_pos = pm.start()
            positions = [p for p in (start, json_pos, pseudo_pos) if p != -1]
            if not positions:
                keep = min(len(self._held), len(_TOOL_CALL_OPEN) - 1)
                if keep:
                    emit, self._held = self._held[:-keep], self._held[-keep:]
                    self._emit(emit)
                else:
                    self._emit(self._held)
                    self._held = ""
                return
            cut = min(positions)
            if cut:
                self._emit(self._held[:cut])
                self._held = self._held[cut:]
            if cut == start:
                end = self._held.find(_TOOL_CALL_CLOSE, len(_TOOL_CALL_OPEN))
                if end == -1:
                    return
                self._held = self._held[end + len(_TOOL_CALL_CLOSE):]
                continue
            return

    def flush(self) -> None:
        if self._raw:
            stripped, _ = _extract_tool_calls(self._raw, self._tools)
            if self._emitted_len < len(stripped):
                self._emit(stripped[self._emitted_len:])
        self._raw = ""
        self._held = ""
        self._emitted_len = 0

class LLMEngine:
    """Lifecycle + inference wrapper around a single llama_cpp ``Llama``.

    ``console`` is optional and duck-typed: any object with ``.warn(msg)``
    and ``.info(msg)`` (e.g. a rich Console) works; when None, all notes are
    dropped silently.
    """

    def __init__(self, config: Config, console: Any = None) -> None:
        self._config = config
        self._console = console
        self._model: Any = None
        self._chat_format = ""
        self._loaded_at: float | None = None
        self._last_used: float | None = None

    # ------------------------------------------------------------- console

    def _warn(self, msg: str) -> None:
        if self._console is not None:
            self._console.warn(msg)

    def _info(self, msg: str) -> None:
        if self._console is not None:
            self._console.info(msg)

    # -------------------------------------------------------------- state

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> str:
        return str(self._config.model.model_path)

    # ----------------------------------------------------------- estimate

    def estimate_vram(self) -> dict:
        """Estimated VRAM budget ``{weights_mb, kv_mb, total_mb}``.

        See the module docstring for the heuristic. A ``note`` key is added
        when the model file cannot be stat-ed (weights counted as 0).
        """
        weights_mb = 0.0
        note = ""
        try:
            size_bytes = Path(self.model_path).stat().st_size
        except OSError as exc:
            note = f"model file not readable: {exc}"
        else:
            weights_mb = size_bytes / (1024 * 1024) * 1.15
        kv_quantized = bool(self._config.model.kv_quant)
        kv_mb = self._config.model.n_ctx * (0.035 if kv_quantized else 0.06)
        result = {
            "weights_mb": weights_mb,
            "kv_mb": kv_mb,
            "total_mb": weights_mb + kv_mb,
        }
        if note:
            result["note"] = note
        return result

    # -------------------------------------------------------------- load

    def load(self, force: bool = False) -> bool:
        """Load the model. Returns False when the resource check refuses.

        With ``force=True`` the check is downgraded to a warning and the
        load proceeds anyway.
        """
        if self.loaded:
            self.touch()
            return True
        cfg = self._config
        est = self.estimate_vram()
        ok, msg = check_model_load(cfg, int(est["weights_mb"]), int(est["kv_mb"]))
        if not ok:
            if not force:
                self._warn(msg)
                return False
            self._warn(f"{msg} — loading anyway (force=True)")
        elif msg:
            self._info(msg)

        from llama_cpp import Llama

        model_cfg = cfg.model
        chat_format = resolve_chat_format(model_cfg.chat_format)
        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": model_cfg.n_ctx,
            "n_gpu_layers": model_cfg.n_gpu_layers,
            "n_batch": model_cfg.n_batch,
            "n_threads": model_cfg.threads,
            "verbose": False,
            "chat_format": chat_format,
        }
        params = inspect.signature(Llama.__init__).parameters
        kv_quantized = False
        if model_cfg.kv_quant == "q8_0":
            if "type_k" in params and "type_v" in params:
                q8 = _kv_cache_q8_value()
                if q8 is not None and "flash_attn" in params:
                    # llama.cpp refuses a quantized V cache without flash
                    # attention ("quantized V cache requires flash_attn"),
                    # so the two are enabled together.
                    kwargs["type_k"] = q8
                    kwargs["type_v"] = q8
                    kwargs["flash_attn"] = True
                    kv_quantized = True
                    self._info(
                        f"KV cache quantized to {model_cfg.kv_quant} "
                        f"(type_k/type_v={q8}, flash_attn enabled — required "
                        "for a quantized V cache)"
                    )
                elif q8 is None:
                    self._info(
                        "KV cache quantization requested but unsupported by "
                        "this llama_cpp build; using F16"
                    )
                else:
                    self._info(
                        "KV cache quantization requested but this llama_cpp "
                        "build lacks flash_attn support (required for "
                        "quantized KV); using F16"
                    )
            else:
                self._info(
                    "KV cache quantization requested but Llama() has no "
                    "type_k/type_v parameters; using F16"
                )
        elif model_cfg.kv_quant:
            self._info(
                f"Unknown kv_quant {model_cfg.kv_quant!r}; "
                "using F16 KV cache"
            )
        if model_cfg.flash_attn and not kv_quantized:
            if "flash_attn" in params:
                kwargs["flash_attn"] = True
            else:
                self._info(
                    "flash_attn requested but unsupported by this llama_cpp "
                    "build; skipping"
                )

        started = time.perf_counter()
        try:
            model = Llama(**kwargs)
        except Exception as exc:
            raise EngineError(
                f"failed to load model {model_cfg.file!r}: {exc}. "
                "Check that the file exists at model.file/model_path, that "
                "VRAM/RAM are sufficient, and that this llama_cpp build "
                "supports GPU offload."
            ) from exc
        self._model = model
        self._chat_format = chat_format
        now = time.time()
        self._loaded_at = now
        self._last_used = now
        self._info(
            f"Loaded {model_cfg.file} in {time.perf_counter() - started:.1f}s "
            f"(chat_format={chat_format})"
        )
        return True

    def unload(self) -> None:
        """Free the model and let GC release its VRAM."""
        if not self.loaded:
            return
        before = gpu_info(force=True)
        self._model = None
        self._chat_format = ""
        self._loaded_at = None
        self._last_used = None
        gc.collect()
        after = gpu_info(force=True)
        if self._console is not None:
            b_free = before.get("vram_free_mb")
            a_free = after.get("vram_free_mb")
            if b_free is not None and a_free is not None:
                self._info(
                    f"Model unloaded: VRAM free {b_free} -> {a_free} MB "
                    f"({a_free - b_free:+d} MB)"
                )
            else:
                self._info("Model unloaded")

    def touch(self) -> None:
        """Mark the model as recently used (resets the idle timer)."""
        self._last_used = time.time()

    def idle_seconds(self) -> float:
        """Seconds since load() or the last touch()/chat()."""
        if not self.loaded or self._last_used is None:
            return 0.0
        return time.time() - self._last_used

    # ----------------------------------------------------------- tokens

    def token_count(self, text: str) -> int:
        """Exact token count when loaded, else a ``len//4`` estimate."""
        if self.loaded:
            try:
                return len(self._model.tokenize(text.encode()))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def prompt_token_estimate(self, messages: list[dict]) -> int:
        """Rough token budget for a message list (contents + overhead)."""
        total = 0
        for m in messages:
            total += self.token_count(str(m.get("content") or ""))
        return total + 4 * len(messages)  # role markers / template overhead

    # -------------------------------------------------------------- chat

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream_cb: Optional[Callable[[str], None]] = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Run a chat completion and normalize the result.

        Returns ``{"role", "content", "tool_calls", "finish_reason"}``;
        ``tool_calls`` entries are ``{"id", "name", "arguments"}`` with
        ``arguments`` always a dict. With ``stream_cb`` set, the completion
        is streamed and every content delta is passed to the callback.
        """
        if not self.loaded:
            raise EngineError("model not loaded")
        model_cfg = self._config.model
        # chat_format is a Llama *constructor* parameter in llama-cpp-python;
        # it is pinned at load() time, so it must not be passed per-request.
        request_messages = _adapt_messages(messages)
        if tools:
            request_messages = _inject_tools(request_messages, tools)
        kwargs: dict[str, Any] = {
            "messages": request_messages,
            "tools": tools or None,
            "temperature": model_cfg.temperature,
            "top_p": model_cfg.top_p,
            "max_tokens": max_tokens or model_cfg.max_tokens,
            "stream": stream_cb is not None,
        }
        if tools:
            kwargs["stop"] = ["</tool_call>"]
        try:
            if stream_cb is not None:
                content_parts: list[str] = []
                raw_tool_calls: list[dict] = []
                finish_reason = None
                display = _ToolCallStreamFilter(stream_cb, tools)
                for chunk in self._model.create_chat_completion(**kwargs):
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        content_parts.append(piece)
                        display.feed(piece)
                    _accumulate_stream_tool_calls(
                        raw_tool_calls, delta.get("tool_calls")
                    )
                    msg = choice.get("message") or {}
                    if msg.get("tool_calls"):
                        raw_tool_calls = list(msg["tool_calls"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                display.flush()
                content, text_calls = _extract_tool_calls("".join(content_parts), tools)
                calls = _normalize_tool_calls(raw_tool_calls)
                _merge_extracted_calls(calls, text_calls)
                result = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": calls,
                    "finish_reason": finish_reason,
                }
            else:
                response = self._model.create_chat_completion(**kwargs)
                choice = response["choices"][0]
                msg = choice["message"]
                content, text_calls = _extract_tool_calls(
                    msg.get("content") or "", tools
                )
                calls = _normalize_tool_calls(msg.get("tool_calls"))
                _merge_extracted_calls(calls, text_calls)
                result = {
                    "role": msg.get("role", "assistant"),
                    "content": content,
                    "tool_calls": calls,
                    "finish_reason": choice.get("finish_reason"),
                }
        except Exception as exc:
            raise EngineError(
                f"chat completion failed: {exc}. If the context was exceeded, "
                f"raise model.n_ctx (currently {model_cfg.n_ctx}); if VRAM was "
                "exhausted, reduce n_gpu_layers or enable kv_quant."
            ) from exc
        self.touch()
        return result
