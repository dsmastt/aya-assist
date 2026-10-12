# assistant-agent

A fully-local AI desktop agent for Linux (Arch Linux + Hyprland), written in
Python. It lives in your terminal, chats with you naturally, and actually
operates your computer: it launches apps, arranges Hyprland workspaces,
searches the web, plays videos and anime, remembers your preferences, and runs
non-destructive shell commands — with approval gates for anything risky.

Everything that matters stays on your machine: LLM inference runs through
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) on your GPU
(NVIDIA, prebuilt CUDA wheels — no CUDA toolkit required), and memory lives in
a local SQLite database.

> Status: core agent loop, tool registry, memory, and CLI are in place; the
> project is under active development.

## Hardware context this project targets

- Arch Linux + Hyprland (+ Caelestia Shell)
- NVIDIA RTX 3060 Ti (8 GB VRAM), Ryzen 5 3500, 8 GB RAM
- Models stored on a slow HDD (`/mnt/Vault/models`)

## Quick start

```bash
# 1. Install (creates .venv, installs CUDA llama-cpp-python, links ~/.local/bin/assist)
bash scripts/install.sh

# 2. First-run setup (creates config dirs and writes a default config.toml)
assist setup

# 3. Download the default model (Qwen2.5-7B-Instruct Q4_K_M, ~4.7 GB)
bash scripts/download-model.sh

# 4. Run
assist
```

The model is loaded lazily on your first message (not at startup, to keep
RAM free); `assist status` shows hardware and model state without loading
anything, and `assist chat "hello"` runs a single non-interactive turn.

See [docs/install.md](docs/install.md) for details and troubleshooting,
[docs/usage.md](docs/usage.md) for the full usage guide, and
[docs/architecture.md](docs/architecture.md) for the design.

## Usage

```
assist                    interactive agent (default)
assist chat "message..."  one-shot, non-interactive turn
assist status             system + model status (no model load)
assist setup              create config dirs / default config
```

Slash commands inside the repl: `/help`, `/exit`, `/clear` (fresh session),
`/model load|unload|status`, `/memory list|search <text>|forget <id|text>`,
`/tasks`, `/reminders`, `/history`, `/status`.

### Tool overview

- **shell** — `run_shell`: non-destructive commands; sudo/destructive commands
  are refused, sensitive writes ask for approval
- **app-launch** — `open_app`, `open_path`, `open_project`, `open_url`,
  `play_video`, `ani_play` (never steal focus; dedicated workspaces)
- **workspace** — `hypr_workspaces`, `hypr_clients`, `move_window`
- **network** — `web_search` (DuckDuckGo), `fetch_page`, `find_video`
- **memory** — `memory_add`, `memory_recall`, `memory_list`, `memory_delete`
- **tasks** — `todo_add`, `todo_list`, `todo_toggle`, `reminder_add`,
  `reminder_list`, `reminder_cancel`
- **info** — `system_status`, `notify_user`, `ani_next`

### Hardware notes

- 8 GB VRAM (RTX 3060 Ti): the default Q4_K_M 7B model + Q8_0 KV cache fits;
  the resource check refuses a load that would not fit
- 8 GB RAM: keep free RAM above `resources.min_free_ram_mb`; the idle model
  unloads itself after `model.keep_alive_seconds`
- Models live on a mechanical HDD: loading takes a while — the CLI shows a
  spinner and hints on failure

## Project layout

```
assistant/          Python package
  config.py         TOML-backed configuration dataclasses
  persona.py        tsundere system-prompt builder
  llm/              message templates (chat loop lands here)
  tools/            tool registry + approval gating
  cli.py            console entry point (repl + chat/status/setup)
scripts/
  install.sh        venv + dependency installer (CUDA wheels first)
  download-model.sh model downloader with resume
docs/
  architecture.md   design overview
  install.md        install guide
```

## Configuration

Config lives at `~/.config/assistant/config.toml` (TOML, partial files are
deep-merged onto defaults). `assist setup` writes a default copy; the key
sections are `[model]` (file, n_ctx, n_gpu_layers, kv_quant, temperature),
`[resources]` (RAM/VRAM guardrails), `[safety]` (approval categories and
path protection), `[persona]`, and `[tools]` (apps and workspaces).
See `assistant/config.py` for every field.

## Development

```bash
bash scripts/install.sh          # creates .venv with [dev] extras
.venv/bin/python -m pytest       # test suite (coming with the rest of the agent)
```
