"""Configuration for the assistant agent.

All settings live in dataclasses. Configs are read from / written to TOML
(``~/.config/assistant/config.toml`` by default, see :meth:`Config.default_path`).
Partial TOML files are deep-merged onto the defaults, so any key may be
omitted. ``~`` is expanded on all path-like string fields (and inside path
lists such as ``safety.write_approve_paths``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

import tomli_w


@dataclass
class PathsConfig:
    """Filesystem locations used by the agent."""

    models_dir: str = "/mnt/Vault/models"
    memory_db: str = "~/.local/share/assistant/memory.db"
    history_file: str = "~/.local/share/assistant/history.txt"
    log_dir: str = "~/.local/share/assistant/logs"
    state_file: str = "~/.local/share/assistant/state.json"
    projects_root: str = "/mnt/Data/app"


@dataclass
class ModelConfig:
    """Local LLM settings (llama-cpp-python)."""

    file: str = "qwen2.5-7b-instruct-q4_k_m.gguf"
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    n_batch: int = 512
    threads: int = 4
    keep_alive_seconds: int = 900
    kv_quant: str = "q8_0"
    chat_format: str = "auto"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    flash_attn: bool = False

    @property
    def model_path(self) -> Path:
        """Absolute, resolved path of the model file.

        If ``file`` is absolute it is used as-is; otherwise it is joined onto
        ``paths.models_dir``.
        """
        p = Path(self.file).expanduser()
        if not p.is_absolute():
            base = getattr(self, "_models_dir_override", None) or PathsConfig.models_dir
            p = Path(base).expanduser() / p
        return p.resolve()


@dataclass
class ResourceConfig:
    """Guardrails for RAM/VRAM usage."""

    min_free_ram_mb: int = 800
    warn_free_ram_mb: int = 1500
    max_model_vram_mb: int = 6600
    heavy_command_patterns: list[str] = field(
        default_factory=lambda: [
            "ffmpeg",
            "docker build",
            "make -j",
            "cargo build",
            "npm install",
            "pacman -S",
            "yay -S",
            "pip install",
        ]
    )


@dataclass
class SafetyConfig:
    """Approval and path protection rules."""

    auto_approve_categories: list[str] = field(
        default_factory=lambda: [
            "read-only",
            "app-launch",
            "workspace",
            "info",
            "safe-write",
            "network",
            "shell",
            "memory",
            "tasks",
        ]
    )
    write_approve_paths: list[str] = field(
        default_factory=lambda: ["~/.config", "~/.local/share"]
    )
    forbidden_paths: list[str] = field(
        default_factory=lambda: [
            "/etc",
            "/usr",
            "/boot",
            "/sys",
            "/proc",
            "/run",
            "/var/lib",
            "/var/log",
            "/opt",
            "~/.ssh",
            "~/.gnupg",
            "~/.config/hypr",
            "~/.config/caelestia",
        ]
    )


@dataclass
class WebConfig:
    """HTTP/web-search settings."""

    search_engine: str = "duckduckgo"
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:156.0) Gecko/20100101 Firefox/156.0"
    )
    request_timeout: int = 15
    youtube_api_key: str = ""


@dataclass
class PersonaConfig:
    """Persona/flavor settings."""

    name: str = "Aya"
    style: str = "tsundere"
    mood: float = 0.8
    extra_rules: list[str] = field(default_factory=list)


@dataclass
class ToolsConfig:
    """Default applications and workspaces for tool actions."""

    media_player: str = "browser"
    browser: str = "zen"
    editor: str = "codium"
    terminal: str = "foot"
    video_workspace: int = 3
    agent_workspace: int = 4
    ani_history: str = "~/.local/state/ani-cli/history.txt"


@dataclass
class Config:
    """Top-level configuration object."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    web: WebConfig = field(default_factory=WebConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    #: section name -> field names whose string values are filesystem paths
    #: (``~`` is expanded; list values have every element expanded).
    _PATH_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "paths": (
            "models_dir",
            "memory_db",
            "history_file",
            "log_dir",
            "state_file",
            "projects_root",
        ),
        "safety": ("write_approve_paths", "forbidden_paths"),
        "tools": ("ani_history",),
    }

    def __post_init__(self) -> None:
        self.model._models_dir_override = self.paths.models_dir
        self._expand_paths()

    # ------------------------------------------------------------------ load

    @classmethod
    def default_path(cls) -> Path:
        """Default config file location (``~/.config/assistant/config.toml``)."""
        return Path("~/.config/assistant/config.toml").expanduser()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load configuration from TOML.

        A missing file, or missing keys inside it, fall back to defaults.
        Partial TOML sections are deep-merged onto the defaults.
        """
        cfg = cls()
        p = Path(path).expanduser() if path is not None else cls.default_path()
        if p.exists():
            with p.open("rb") as fh:
                raw = tomllib.load(fh)
            if isinstance(raw, dict):
                cfg._apply_toml(raw)
        cfg._expand_paths()
        return cfg

    def _apply_toml(self, raw: dict[str, Any]) -> None:
        """Deep-merge a parsed TOML dict onto this config's defaults."""
        for section_name in (
            "paths",
            "model",
            "resources",
            "safety",
            "web",
            "persona",
            "tools",
        ):
            data = raw.get(section_name)
            if not isinstance(data, dict):
                continue
            section = getattr(self, section_name)
            for f in fields(section):
                if f.name in data:
                    setattr(section, f.name, _coerce(f.type, data[f.name]))
        self.model._models_dir_override = self.paths.models_dir

    # ------------------------------------------------------------------ save

    def save(self, path: str | Path | None = None) -> Path:
        """Write the configuration as TOML (creating parent dirs)."""
        p = Path(path).expanduser() if path is not None else self.default_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            tomli_w.dump(self.to_dict(), fh)
        return p

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict representation suitable for TOML/JSON serialization."""
        return _to_plain(self)

    # ----------------------------------------------------------------- paths

    def _expand_paths(self) -> None:
        """Expand ``~`` on every path-like field (and path-list elements)."""
        for section_name, field_names in self._PATH_FIELDS.items():
            section = getattr(self, section_name)
            for fname in field_names:
                value = getattr(section, fname)
                if isinstance(value, list):
                    setattr(section, fname, [str(Path(v).expanduser()) for v in value])
                elif isinstance(value, str):
                    setattr(section, fname, str(Path(value).expanduser()))
        self.model._models_dir_override = self.paths.models_dir

    def ensure_dirs(self) -> None:
        """Create models_dir, log_dir, and parents of the DB/history/state files."""
        paths = self.paths
        dirs: list[Path] = [
            Path(paths.models_dir).expanduser(),
            Path(paths.memory_db).expanduser().parent,
            Path(paths.history_file).expanduser().parent,
            Path(paths.log_dir).expanduser(),
            Path(paths.state_file).expanduser().parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def _coerce(typ: Any, value: Any) -> Any:
    """Best-effort coercion of a TOML value to the dataclass field's type."""
    if value is None:
        return value
    origin = getattr(typ, "__origin__", None)
    if origin is list or origin is set or origin is tuple:
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]
    if typ in (str, int, float, bool):
        if isinstance(value, typ):
            return value
        return typ(value)
    return value


def _to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses/Paths into plain dicts/scalars."""
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
