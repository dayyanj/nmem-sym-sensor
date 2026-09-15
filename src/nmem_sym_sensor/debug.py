"""Debug tracing for nmem-sym-sensor.

Toggle via NMEM_SENSOR_DEBUG=1 (env var) or config.DEBUG_ENABLED.
Traces are structured and filterable by subsystem.

Usage:
    from nmem_sym_sensor.debug import trace

    trace("fovea", "saccade", cx=100, cy=200, saliency=0.85)
    trace("attention", "phase_change", old="explore", new="inspect", reason="coverage 0.85")
    trace("binding", "sound_visual", sound_id=42, visual_ids=[1,2,3])

Subsystems:
    fovea       — saccade points, edge-following, hull, habituation
    attention   — phase transitions, budget allocation, coverage
    binding     — sound-visual co-occurrence, word-visual, onset gating
    promotion   — iconic buffer → short-term → long-term
    scene       — chronoception, scene changes, active set
    dreamstate  — decay, self-play, myelination, concept links
    prediction  — surprise, prediction accuracy, feedback

Output: structured log lines at DEBUG level, or JSON file if configured.
"""
import json
import logging
import os
import time
from collections import deque
from typing import Any

log = logging.getLogger("nmem_sensor.trace")

# Global toggle — check once at import, override via enable()/disable()
_enabled = os.environ.get("NMEM_SENSOR_DEBUG", "0") == "1"

# Filter to specific subsystems (empty = all)
_subsystems: set[str] = set()
_filter_str = os.environ.get("NMEM_SENSOR_DEBUG_FILTER", "")
if _filter_str:
    _subsystems = {s.strip() for s in _filter_str.split(",") if s.strip()}

# Recent trace buffer (ringbuffer for post-mortem inspection)
_trace_buffer: deque[dict] = deque(maxlen=500)

# Optional JSON file output
_trace_file = os.environ.get("NMEM_SENSOR_DEBUG_FILE", "")
_file_handle = None


def enable(subsystems: str | list[str] | None = None):
    """Enable debug tracing, optionally filtered to specific subsystems.

    Args:
        subsystems: Comma-separated string or list. None = all.
            e.g., enable("fovea,binding") or enable(["fovea", "binding"])
    """
    global _enabled, _subsystems
    _enabled = True
    if subsystems is None:
        _subsystems = set()
    elif isinstance(subsystems, str):
        _subsystems = {s.strip() for s in subsystems.split(",") if s.strip()}
    else:
        _subsystems = set(subsystems)
    log.info("Debug tracing enabled: %s", _subsystems or "ALL")


def disable():
    """Disable debug tracing."""
    global _enabled
    _enabled = False


def is_enabled(subsystem: str | None = None) -> bool:
    """Check if tracing is enabled for a subsystem."""
    if not _enabled:
        return False
    if not _subsystems:
        return True
    return subsystem in _subsystems if subsystem else True


def trace(subsystem: str, event: str, **kwargs: Any):
    """Emit a structured trace event.

    Args:
        subsystem: e.g., "fovea", "attention", "binding"
        event: e.g., "saccade", "phase_change", "sound_visual"
        **kwargs: event-specific data
    """
    if not _enabled:
        return
    if _subsystems and subsystem not in _subsystems:
        return

    entry = {
        "t": round(time.monotonic(), 4),
        "sub": subsystem,
        "evt": event,
        **{k: _serialize(v) for k, v in kwargs.items()},
    }

    _trace_buffer.append(entry)

    # Log at DEBUG level (visible when log level is DEBUG)
    log.debug("[%s:%s] %s", subsystem, event,
              " ".join(f"{k}={v}" for k, v in kwargs.items()))

    # File output if configured
    if _trace_file:
        _write_to_file(entry)


def get_recent(n: int = 50, subsystem: str | None = None) -> list[dict]:
    """Get recent trace entries from the ringbuffer.

    Useful for post-mortem debugging: "what happened in the last 50 events?"
    """
    if subsystem:
        return [e for e in _trace_buffer if e["sub"] == subsystem][-n:]
    return list(_trace_buffer)[-n:]


def dump(path: str | None = None, subsystem: str | None = None) -> str:
    """Dump trace buffer to JSON string or file."""
    entries = get_recent(500, subsystem)
    text = json.dumps(entries, indent=2, default=str)
    if path:
        with open(path, "w") as f:
            f.write(text)
    return text


def summary() -> dict:
    """Summary of trace activity by subsystem and event."""
    counts: dict[str, dict[str, int]] = {}
    for e in _trace_buffer:
        sub = e["sub"]
        evt = e["evt"]
        counts.setdefault(sub, {}).setdefault(evt, 0)
        counts[sub][evt] += 1
    return counts


def _serialize(v: Any) -> Any:
    """Make values JSON-safe."""
    import numpy as np
    if isinstance(v, np.ndarray):
        if v.size <= 8:
            return v.tolist()
        return f"array({v.shape}, mean={v.mean():.4f})"
    if isinstance(v, (set, frozenset)):
        return list(v)
    if isinstance(v, float):
        return round(v, 4)
    return v


def _write_to_file(entry: dict):
    """Append trace entry to JSON-lines file."""
    global _file_handle
    if _file_handle is None:
        try:
            _file_handle = open(_trace_file, "a")
        except Exception:
            return
    try:
        _file_handle.write(json.dumps(entry, default=str) + "\n")
        _file_handle.flush()
    except Exception:
        pass
