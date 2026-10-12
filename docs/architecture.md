# Architecture

The assistant agent is a local-first, single-user desktop agent. This document
describes the component layout and the design decisions behind it. It evolves
as modules land.

## Goals

- 100% local LLM inference and private memory; network only for explicitly
  requested web tasks.
- Terminal-first interaction (the agent lives where the user types).
- Real desktop control through Hyprland's IPC (reliable window/workspace
  interfaces) rather than fragile screen-coordinate automation.
- Respect for a constrained machine: 8 GB RAM, 8 GB VRAM, models on a slow HDD.

## Components

```
┌─────────────────────────────────────────────────────────────┐
│ cli.py (interactive terminal frontend, prompt-toolkit/rich) │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│ Agent loop (assistant/llm/ — chat loop, tool-call routing)  │
│   system prompt ← persona.build_system_prompt(config)       │
│   messages     ← llm/templates.py helpers                   │
└──────┬──────────────────────────┬───────────────────────────┘
       │                          │
┌──────▼───────┐          ┌───────▼──────────────┐
│ LLM backend  │          │ ToolRegistry         │
│ llama-cpp-   │          │ tools/registry.py    │
│ python, GPU  │          │  ToolEntry / Result  │
│ offload, HDD │          │  approval gating     │
│ model cache  │          │  OpenAI-style schema │
└──────────────┘          └───────┬──────────────┘
                                  │
                     ┌────────────┼───────────────┐
                     │            │               │
              desktop/ws    shell/runner    web/media/memory
              (Hyprland      (approval-     (search, yt-dlp,
               IPC, apps)     gated cmds)    ani-cli, sqlite)
```

### `assistant/config.py`

Typed configuration as dataclasses, loaded from TOML
(`~/.config/assistant/config.toml`) with deep-merge onto defaults.
Sections: `paths`, `model`, `resources`, `safety`, `web`, `persona`, `tools`.
Path-like fields get `~` expanded on load; `Config.ensure_dirs()` creates the
directories the agent writes into.

### `assistant/persona.py`

Builds the system prompt from `PersonaConfig`. The default persona ("Aya") is
a tsundere assistant: attitude is flavor, never a license to be wrong.
`mood` (0–1) scales the dramatics; `extra_rules` are appended verbatim.
`PERSONA_STYLES` is the extension point for other personas. The agent loop
prepends memory/context blocks before this prompt body.

### `assistant/llm/templates.py`

Tiny, dependency-free helpers for building chat messages
(`system/user/assistant/tool` roles), token estimation, and compact tool-result
formatting (truncated JSON for the model).

### `assistant/tools/registry.py`

A `ToolRegistry` singleton (`TOOLS`) mapping tool names to async functions,
with OpenAI-style function schemas for the model and a category-based approval
gate. A tool's category is auto-approved only if listed in
`safety.auto_approve_categories`; otherwise `ToolContext.approve()` asks the
user (or raises `NeedsApproval` when no interactive console is available).
Import-safe standalone: stdlib only.

### LLM backend

`llama-cpp-python` from prebuilt CUDA wheels (cu128 preferred, falling back to
cu126/cu124/CPU). The wheel bundles CUDA runtimes, so no nvcc or toolkit is
needed. GPU offload is controlled by `model.n_gpu_layers` (-1 = all).
Models live on the HDD (`paths.models_dir`), so load times are expected to be
slow; the backend keeps the model resident (see `keep_alive_seconds`) and
unloads it after inactivity.

## Key design decisions

1. **Python 3.12** (not the system 3.14): best prebuilt ML wheel coverage.
2. **Prebuilt CUDA wheels** for llama-cpp-python: no compiler/toolkit needed
   on this machine (no nvcc). CPU build is only a fallback.
3. **Hyprland IPC first**: workspace/window actions go through `hyprctl` and
   `hyprctl -j` rather than mouse/keyboard emulation. Dedicated agent
   workspaces (`tools.video_workspace`, `tools.agent_workspace`) keep agent
   activity off the user's active workspace.
4. **Approval, not sandboxing**: the agent may run commands, but anything
   outside `safety.auto_approve_categories` requires explicit user consent,
   and `safety.forbidden_paths` is never touched.
5. **Local memory**: durable facts in a local SQLite DB
   (`paths.memory_db`), so preferences survive restarts without leaving the
   machine.
6. **Resource guards**: `resources.*` thresholds drive pre-flight
   `system_status` checks and warnings before heavy commands.

## Data flow (interactive loop)

1. User prompt → context/memory blocks + persona prompt + history → model.
2. Model replies or emits tool calls → `TOOLS.call` → approval gate (if
   needed) → tool execution → `ToolResult`.
3. Results are fed back as `tool` messages; loop continues until the model
   produces a final text answer, which the CLI renders (markdown, concise).
