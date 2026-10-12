# Install guide

Target machine: Arch Linux + Hyprland, NVIDIA GPU, 8 GB RAM.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (0.12+). Install with
  `curl -LsSf https://astral.sh/uv/install.sh | sh` if missing.
- CPython 3.12 (uv fetches it automatically on first `uv venv --python 3.12`;
  an already-installed 3.12 is reused).
- An NVIDIA GPU driver with CUDA runtime support. **No nvcc / CUDA toolkit
  needed**: llama-cpp-python is installed from prebuilt CUDA wheels that
  bundle their own CUDA runtimes.

## 1. Install

```bash
cd /path/to/assistant-agent
bash scripts/install.sh
```

What it does:

1. Locates `uv` (PATH or `~/.local/bin/uv`).
2. Creates `.venv` with CPython 3.12.
3. Installs `llama-cpp-python>=0.3.5`, trying prebuilt wheels in this order:
   `cu128` → `cu126` → `cu124` → CPU sdist build (last resort, ~10+ min).
   Each CUDA attempt is verified by importing `llama_cpp`.
4. Installs the project editable with dev extras.
5. Verifies all dependencies import.
6. Symlinks `~/.local/bin/assist` → `.venv/bin/assist`.

GPU offload status is printed during install (and recorded in
`.install-report.txt`).

## 2. Download a model

```bash
# default: Qwen2.5-7B-Instruct Q4_K_M (~4.7 GB) into /mnt/Vault/models
bash scripts/download-model.sh

# custom URL / filename (resumable)
bash scripts/download-model.sh "https://huggingface.co/.../model.gguf" my-model.gguf
```

Set `ASSISTANT_MODELS_DIR` to override the target directory. Downloads resume
from a `.part` file, so interrupting and re-running continues where it left
off.

## 3. Run

```bash
assist          # or: assist version
```

## Troubleshooting

- **All CUDA wheel attempts failed** → check the abetlen wheel index is
  reachable, or that the driver exposes CUDA 12.4+; the CPU fallback works but
  is much slower.
- **`assist: command not found`** → ensure `~/.local/bin` is on your `PATH`.
- **Out of memory when loading the model** → lower `model.n_ctx`, set
  `n_gpu_layers` to a partial offload, or pick a smaller quant in
  `~/.config/assistant/config.toml`.
