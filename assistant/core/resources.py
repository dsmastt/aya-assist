"""System resource monitoring for the local AI agent.

Provides RAM/swap, load average, GPU (via ``nvidia-smi``), and top-process
sampling. Everything returns plain JSON-serializable data so the status tool
can dump it directly.

The GPU query is cached for a few seconds: ``nvidia-smi`` takes ~100 ms per
call, and pre-load checks plus the status tool can hit it several times in
quick succession.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Any

import psutil

_GPU_CACHE_TTL = 3.0

_gpu_cache: dict[str, Any] = {"at": 0.0, "data": None}


def _mb(value: int) -> int:
    return int(value // (1024 * 1024))


def system_ram() -> dict:
    """RAM and swap totals in MB, plus usage percent (0-100)."""
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "total_mb": _mb(vm.total),
        "avail_mb": _mb(vm.available),
        "used_mb": _mb(vm.used),
        "percent": vm.percent,
        "swap_total_mb": _mb(sw.total),
        "swap_used_mb": _mb(sw.used),
    }


def load_avg() -> tuple[float, float, float]:
    """1-, 5- and 15-minute load averages (``os.getloadavg``)."""
    return os.getloadavg()


def top_processes(n: int = 5) -> list[dict]:
    """The ``n`` processes with the highest RSS, newest-first on ties.

    Returns ``{"pid", "name", "rss_mb", "cmd"}`` dicts. Processes that die
    or deny introspection while iterating are skipped.
    """
    procs: list[dict] = []
    for p in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            info = p.info
            mem = info.get("memory_info")
            rss_mb = _mb(mem.rss) if mem else 0
            cmdline = " ".join(info.get("cmdline") or []) or info.get("name") or ""
            procs.append(
                {
                    "pid": p.pid,
                    "name": info.get("name") or "",
                    "rss_mb": rss_mb,
                    "cmd": cmdline[:200],
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    procs.sort(key=lambda d: d["rss_mb"], reverse=True)
    return procs[:n]


def _query_gpu() -> dict:
    """Run nvidia-smi once and parse the first GPU's row."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "note": f"nvidia-smi failed: {exc}"}
    if out.returncode != 0:
        note = (out.stderr or "nonzero exit").strip().splitlines()[:1]
        return {"available": False, "note": f"nvidia-smi failed: {'; '.join(note)}"}
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"available": False, "note": "nvidia-smi returned no GPUs"}
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 6:
        return {"available": False, "note": f"unparseable nvidia-smi row: {lines[0]!r}"}
    try:
        name, total, used, free, util, temp = (
            parts[0],
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5]),
        )
    except ValueError:
        return {"available": False, "note": f"unparseable nvidia-smi row: {lines[0]!r}"}
    return {
        "available": True,
        "name": name,
        "vram_total_mb": int(total),
        "vram_used_mb": int(used),
        "vram_free_mb": int(free),
        "util_percent": int(util),
        "temp_c": int(temp),
    }


def gpu_info(force: bool = False) -> dict:
    """GPU state, cached for ~3 seconds (``force=True`` bypasses the cache).

    Returns ``{"available": True, "name", "vram_total_mb", "vram_used_mb",
    "vram_free_mb", "util_percent", "temp_c"}`` on success, or
    ``{"available": False, "note": ...}`` when nvidia-smi is missing,
    times out, or produces unparseable output. Never raises.
    """
    now = time.monotonic()
    cached = _gpu_cache["data"]
    if not force and cached is not None and now - _gpu_cache["at"] < _GPU_CACHE_TTL:
        return dict(cached)
    data = _query_gpu()
    _gpu_cache["at"] = now
    _gpu_cache["data"] = dict(data)
    return data


def vram_free_mb() -> int | None:
    """Free VRAM in MB, or None when GPU info is unavailable."""
    info = gpu_info()
    if info.get("available"):
        return info["vram_free_mb"]
    return None


def system_status() -> dict:
    """Everything at once: RAM, load, GPU, top processes (JSON-safe)."""
    return {
        "ram": system_ram(),
        "load_avg": list(load_avg()),
        "gpu": gpu_info(),
        "top_processes": top_processes(),
    }


def check_before_heavy(label: str, cfg) -> tuple[bool, str]:
    """Pre-flight check before a heavy command (build, ffmpeg, ...).

    Returns ``(False, warning)`` when free RAM is below
    ``cfg.resources.min_free_ram_mb`` (caller should refuse), ``(True, note)``
    when it is merely below ``cfg.resources.warn_free_ram_mb`` (caller may
    proceed with a warning), and ``(True, "")`` otherwise.
    """
    avail = system_ram()["avail_mb"]
    load = load_avg()
    load_note = f"load avg {load[0]:.2f} / {load[1]:.2f} / {load[2]:.2f}"
    if avail < cfg.resources.min_free_ram_mb:
        return (
            False,
            f"{label}: only {avail} MB RAM free (below "
            f"{cfg.resources.min_free_ram_mb} MB minimum); {load_note}",
        )
    if avail < cfg.resources.warn_free_ram_mb:
        return (
            True,
            f"{label}: low free RAM ({avail} MB < "
            f"{cfg.resources.warn_free_ram_mb} MB); {load_note}",
        )
    return True, ""


def check_model_load(cfg, weights_mb: int, kv_mb: int) -> tuple[bool, str]:
    """Check whether a model needing ``weights_mb + kv_mb`` VRAM can load.

    When GPU info is available, requires ``vram_free_mb + 300 MB margin >=
    budget``, where the budget is the estimate capped at
    ``cfg.resources.max_model_vram_mb`` (the host's configured model budget;
    the estimate is a conservative upper bound, so overshoot past the cap
    alone is not grounds for refusal). The 300 MB margin covers compute
    buffers and the CUDA context. Independently, free RAM must stay above
    ``cfg.resources.min_free_ram_mb`` (for mmap'd weights and CPU fallback).
    Returns ``(True, info_note)`` when fine.
    """
    need_mb = weights_mb + kv_mb
    ram = system_ram()
    avail_mb = ram["avail_mb"]
    gpu = gpu_info()
    notes: list[str] = []
    if gpu.get("available"):
        free = gpu["vram_free_mb"]
        budget_mb = min(need_mb, cfg.resources.max_model_vram_mb)
        if free + 300 < budget_mb:
            capped = (
                f" (capped from {need_mb} by max_model_vram_mb)"
                if budget_mb < need_mb
                else ""
            )
            return (
                False,
                f"not enough VRAM: model needs ~{budget_mb} MB{capped} "
                f"(weights {weights_mb} + KV cache {kv_mb}), "
                f"only {free} MB free on {gpu['name']}",
            )
        notes.append(
            f"VRAM ok: {free} MB free (budget ~{budget_mb} MB); RAM free {avail_mb} MB"
        )
    else:
        notes.append(f"GPU unavailable (assuming CPU inference); RAM free {avail_mb} MB")
    if avail_mb < cfg.resources.min_free_ram_mb:
        return (
            False,
            f"not enough free RAM: {avail_mb} MB < "
            f"{cfg.resources.min_free_ram_mb} MB minimum",
        )
    return True, "; ".join(notes)
