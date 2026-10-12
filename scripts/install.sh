#!/usr/bin/env bash
# Install the assistant agent into a local virtualenv.
# llama-cpp-python is installed from prebuilt CUDA wheels when possible
# (they bundle CUDA, so no nvcc/toolkit is required); CPU sdist build is
# the last resort.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    # Allow scripts/ to live one level deeper.
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "ERROR: could not locate project root (pyproject.toml) from $SCRIPT_DIR" >&2
    exit 1
fi
cd "$PROJECT_ROOT"

echo "==> Project root: $PROJECT_ROOT"

# --- 1. Locate uv ---------------------------------------------------------
UV=""
if command -v uv >/dev/null 2>&1; then
    UV="uv"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
fi
if [ -z "$UV" ]; then
    echo "ERROR: uv is not installed." >&2
    echo "Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
echo "==> Using uv: $($UV --version)"

# --- 2. Create virtualenv with CPython 3.12 --------------------------------
echo "==> Creating venv (python 3.12)..."
"$UV" venv --python 3.12 .venv

PY="$PROJECT_ROOT/.venv/bin/python"
"$PY" -c 'import sys; print("    venv python:", sys.version.split()[0])'

# --- 3. Install llama-cpp-python: CUDA wheels first -------------------------
# Prebuilt CUDA wheels bundle their CUDA runtime. We try, in order:
#   cu128 index -> cu126 index -> cu124 (direct 0.3.36 release wheel, then
#   the gh-pages index) -> cu125/cu132 direct wheels -> CPU sdist build.
# Wheel-only resolution (--only-binary) guarantees a missing index can never
# silently fall through to a slow source build mid-attempt.
FLAVOR=""
WHL_INDEX_BASE="https://abetlen.github.io/llama-cpp-python/whl"
WHL_RELEASE_BASE="https://github.com/abetlen/llama-cpp-python/releases/download"
WHL_036="llama_cpp_python-0.3.36-py3-none-manylinux_2_35_x86_64.whl"

imports_ok() {
    "$PY" -c "import llama_cpp; print(llama_cpp.__version__)" >/dev/null 2>&1
}

try_index() {
    local flavor="$1" index="$2" version="$3"
    echo "==> Trying llama-cpp-python [$flavor] via $index (==$version, wheel-only)"
    if "$UV" pip install --python "$PY" --only-binary llama-cpp-python \
        "llama-cpp-python==$version" --extra-index-url "$index" && imports_ok; then
        FLAVOR="$flavor"
        return 0
    fi
    echo "==> [$flavor] not available as a wheel."
    return 1
}

try_direct() {
    local flavor="$1" url="$2"
    echo "==> Trying llama-cpp-python [$flavor] prebuilt wheel: $url"
    if "$UV" pip install --python "$PY" --only-binary llama-cpp-python "$url" \
        && imports_ok; then
        FLAVOR="$flavor"
        return 0
    fi
    echo "==> [$flavor] wheel failed."
    return 1
}

try_index "cu128" "$WHL_INDEX_BASE/cu128" "0.3.19" \
    || try_index "cu126" "$WHL_INDEX_BASE/cu126" "0.3.19" \
    || try_direct "cu124" "$WHL_RELEASE_BASE/v0.3.36-cu124/$WHL_036" \
    || try_index "cu124" "$WHL_INDEX_BASE/cu124" "0.3.19" \
    || try_direct "cu125" "$WHL_RELEASE_BASE/v0.3.36-cu125/$WHL_036" \
    || try_direct "cu132" "$WHL_RELEASE_BASE/v0.3.36-cu132/$WHL_036" \
    || {
        echo "==> No CUDA wheel worked. Falling back to CPU sdist build (slow, 10+ min)..."
        "$UV" pip install --python "$PY" "llama-cpp-python>=0.3.5" \
            || { echo "ERROR: CPU build failed." >&2; exit 1; }
        if imports_ok; then
            FLAVOR="cpu"
        else
            echo "ERROR: llama_cpp import failed after CPU build." >&2
            exit 1
        fi
    }

echo "==> llama-cpp-python installed (flavor: $FLAVOR)"
"$PY" -c "import llama_cpp; print('    version:', llama_cpp.__version__)"

# --- 4. GPU offload check (informational) -----------------------------------
"$PY" - <<'PY' || true
import llama_cpp
v = getattr(llama_cpp, "llama_supports_gpu_offload", None)
if callable(v):
    try:
        v = v()
    except Exception as exc:
        v = f"error: {exc}"
print("    GPU offload supported:", v)
PY

# --- 5. Install project + dev extras ---------------------------------------
echo "==> Installing project (editable) + dev extras..."
"$UV" pip install --python "$PY" -e ".[dev]"

# --- 6. Verify dependencies -------------------------------------------------
echo "==> Verifying dependencies..."
"$PY" -c "import llama_cpp, psutil, rich, prompt_toolkit, httpx, yt_dlp, tomli_w; print('deps ok')"

# --- 7. Symlink ~/.local/bin/assist -----------------------------------------
mkdir -p "$HOME/.local/bin"
LINK="$HOME/.local/bin/assist"
TARGET="$PROJECT_ROOT/.venv/bin/assist"
if [ -L "$LINK" ]; then
    rm -f "$LINK"
elif [ -e "$LINK" ]; then
    echo "==> $LINK exists and is not a symlink; moving to $LINK.bak"
    mv "$LINK" "$LINK.bak"
fi
ln -s "$TARGET" "$LINK"

# --- 8. Symlink ~/.local/bin/yt-dlp -----------------------------------------
# mpv launched via Hyprland's exec (workspace-pinned playback) inherits
# Hyprland's PATH, which lacks the venv's yt-dlp.
if [ -x "$PROJECT_ROOT/.venv/bin/yt-dlp" ]; then
    mkdir -p "$HOME/.local/bin"
    rm -f "$HOME/.local/bin/yt-dlp"
    ln -sf "$PROJECT_ROOT/.venv/bin/yt-dlp" "$HOME/.local/bin/yt-dlp"
fi

echo
echo "==> Install complete."
echo "    venv:        $PROJECT_ROOT/.venv"
echo "    llama-cpp:   flavor=$FLAVOR"
echo "    entrypoint:  $LINK -> $TARGET"
echo "    Try: assist version"
