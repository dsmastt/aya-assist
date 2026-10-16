# Usage guide

The assistant is a fully-local AI desktop agent. You talk to it in a
terminal; it can launch apps, arrange Hyprland workspaces, search the web,
play videos and anime, run safe shell commands, and remember things about
you. Inference and memory stay on your machine.

## Commands

```
assist                    interactive agent (default)
assist repl               same, explicitly
assist chat "message..."  one-shot, non-interactive turn
assist status             system + model status (no model load)
assist setup              create config dirs / default config
assist selftest           self-test battery (launches real apps)
```

The model is **not** loaded at startup. It loads lazily on your first
message (with a spinner — models live on an HDD, so be patient), and unloads
itself after `model.keep_alive_seconds` of idleness. If a load fails you get
a hint about free RAM, picking a smaller model, or checking `/model status`.

### Slash commands (inside the repl)

| command | effect |
| --- | --- |
| `/help` | show the command list |
| `/exit`, `/quit` | leave the repl |
| `/clear` | start a fresh session (old one stays in the DB) |
| `/model load` | load the model now |
| `/model unload` | free the model (VRAM/RAM) |
| `/model status` | model file, loaded state, VRAM estimate |
| `/memory list` | all stored facts |
| `/memory search <text>` | full-text search over memories |
| `/memory forget <id\|text>` | delete a fact by id or text match |
| `/tasks` | open todos |
| `/reminders` | scheduled reminders |
| `/timers` | running countdown timers |
| `/history` | recent conversation of this session |
| `/status` | RAM/GPU/model summary |

### Tasks & timers

- **Timers**: there is no stopwatch app on this machine — `timer_set` *is*
  the stopwatch. Ask for e.g. "a 20 minute timer"; the agent notifies you
  when it ends, and you can ask for the remaining time anytime. `/timers`
  prints the readout of all running timers.

## Example conversations

```
❯ open my ear-training project in a separate workspace
Hmph. Fine. Opening it on workspace 6 so it doesn't bother you...
[opened /mnt/Data/app/ear-training in codium on workspace 6]

❯ find me an interesting video for lunch
Searching... honestly my taste would have been better, but here:
1. <title> (<channel>, 12:34)
...
Playing the first one on workspace 7.

❯ did a new episode of Re:ZERO come out?
According to your ani-cli history you watched episode 12. Next is 13.
Want me to play it, or are you just curious? 😼
```

- mpv needs yt-dlp on Hyprland's PATH to play YouTube streams when launched
  on a workspace; `scripts/install.sh` links `~/.local/bin/yt-dlp` to the
  venv copy for this.

## Self-test battery

`assist selftest` runs a repeatable verification battery: 12 scenarios, each
an independent `assist chat` process, checked against real desktop state —
tool lines extracted from the transcript (must appear in the expected
order) plus side-effect checks on Hyprland windows, processes, timers, the
memory DB and the tasks state file. Scenarios cover app/site launches
(codium, spotify, youtube/chatgpt), video search+play, `start_project`,
workspace organizing, timer set/cancel, memory add/recall, web search, a
15-second reminder (needs the background service; skipped with a note
otherwise), a safe shell command plus a refused `systemctl reboot`, and a
plain greeting that must not trigger the audit warning.

```
assist selftest                    run every scenario
assist selftest --scenario 1       run one scenario (repeatable)
assist selftest --scenario 1 --scenario 7
assist selftest --list             print the scenario ids
assist selftest --json out.json    also write machine-readable results
```

Exit code is 0 when every scenario passes, 1 otherwise. Each run gets a
240 s timeout; evidence (per-run transcripts, tool lines, answer tails,
state snapshots) lands in a fresh `/tmp/assist-selftest-*` directory.

**Warning:** the battery launches real applications and leaves them
running, and it needs the model and a Hyprland desktop session to verify
anything meaningful.

## Memory

The agent has durable memory (SQLite, FTS5 search). Facts are added with
the `memory_add` tool, recalled with `memory_recall`, and listed/searched
via `/memory list` and `/memory search`. Every conversation turn injects
relevant memories into the prompt automatically; the agent is instructed to
recall before answering questions about your preferences and to never
invent memories.

Delete requires approval: `/memory forget 12` or `/memory forget "coffee"`.

## Approval flow

Every tool is gated by its category. Categories listed in
`safety.auto_approve_categories` (read-only, app-launch, workspace, info,
safe-write, network, shell, memory by default) run without asking.
Everything else pops an approval prompt, e.g.:

```
approve: run_shell — {"command": "pacman -S foo"}? [y/N]
```

`run_shell` is double-gated: its internal classifier always refuses
privileged or destructive commands (sudo, mkfs, dd, shutdown, `rm -rf` on
protected paths), and asks approval for sensitive ones (writes outside the
home directory, package installs, process signals) — the approval callback
runs the same interactive flow. When you say no, the tool reports
`denied by user` and the agent moves on. In non-interactive mode
(`assist chat`, no TTY) approvals default to **no**.

## Safety model

- **No privilege escalation** — sudo/doas/pkexec are always denied.
- **Path protection** — `/etc`, `/usr`, `/boot`, `/sys`, `/proc`, `/run`,
  `/var`, `/opt`, `~/.ssh`, `~/.gnupg` and Hyprland/Caelestia configs are
  forbidden write targets; `~/.config` and `~/.local/share` writes need
  approval; everything else under `~`, `/tmp` and `/mnt` is allowed.
- **Destructive commands** — mkfs/partition tools, secure erase, dd, power
  control, `find -delete`, force pushes and similar are denied or
  approval-gated.
- **Workspace tools never steal focus** — apps launch with inline window
  rules (`movetoworkspacesilent`), media goes to `tools.video_workspace`
  (7) and agent work to `tools.agent_workspace` (6).
- **Resource guardrails** — the model load is refused when the VRAM/RAM
  estimate does not fit (`resources.max_model_vram_mb`,
  `resources.min_free_ram_mb`); heavy command patterns get a pre-flight
  RAM check.

## Configuration

`~/.config/assistant/config.toml` (created by `assist setup`). Notable
fields: `[model]` file/n_ctx/n_gpu_layers/kv_quant/temperature;
`[resources]` min_free_ram_mb / max_model_vram_mb / heavy_command_patterns;
`[safety]` auto_approve_categories / write_approve_paths / forbidden_paths;
`[persona]` name/style/mood/extra_rules; `[tools]` media_player/browser/
editor/video_workspace/agent_workspace. Logs go to
`~/.local/share/assistant/logs/assistant.log`, memory to
`~/.local/share/assistant/memory.db`, todos/reminders to
`~/.local/share/assistant/state.json`.
