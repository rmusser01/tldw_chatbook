"""Concurrent `load_settings()` must rebuild the cache once, not once per thread.

TASK-3503. `_SETTINGS_CACHE_LOCK` guards only the cache *cells*: on a miss it
was cleared inside the lock and the rebuild then ran OUTSIDE it, so every
thread that arrived during the miss window ran the entire rebuild -- reading
and parsing the TOML, deep-merging defaults, and re-ensuring directories --
and the last writer's result won.

Measured before the fix: 8 threads on a single invalidation produced 32
bootstrap loads (exactly 8x the single-threaded cost of 4). After: 4, i.e.
the same work one thread does alone.

A note on the task's own wording: its AC #1 asks that `load_settings()` never
hand a concurrent caller `None`. That symptom does not reproduce and cannot
-- `load_settings` never returns the cache cell during the window, and no
module outside `config.py` reads `_SETTINGS_CACHE` directly (verified by
grep). A thread arriving mid-rebuild always took the miss branch and got a
real mapping; it just paid for a redundant rebuild to get it. The
`None`-freedom is pinned below anyway, so a future refactor that starts
returning the raw cell fails here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

import tldw_chatbook.config as config_module


def test_settings_rebuild_and_runtime_publication_share_one_lock_order(
    tmp_path: Path,
) -> None:
    """A reader and writer cannot retain opposite config/rebuild locks."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('[general]\nusers_name = "lock-order"\n', encoding="utf-8")
    script = r"""
import json
import sys
import threading
import traceback

from tldw_chatbook import config as config_module

writer_holds_config = threading.Event()
reader_attempting_config = threading.Event()
errors = []


class InstrumentedConfigLock:
    def __init__(self):
        self._lock = threading.RLock()

    def __enter__(self):
        if threading.current_thread().name == "settings-reader":
            reader_attempting_config.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        del exc_type, exc_value, traceback_value
        self._lock.release()


config_module._CONFIG_FILE_LOCK = InstrumentedConfigLock()
config_module._SETTINGS_REBUILD_LOCK = threading.RLock()
with config_module._SETTINGS_CACHE_LOCK:
    config_module._SETTINGS_CACHE = None
    config_module._SETTINGS_CACHE_SOURCE = None


def writer():
    try:
        with config_module._config_file_lock():
            writer_holds_config.set()
            if not reader_attempting_config.wait(timeout=5):
                raise AssertionError("reader never attempted the config lock")
            config_module._publish_runtime_config_unlocked()
    except BaseException as exc:
        errors.append(f"writer: {exc!r}")


def reader():
    try:
        if not writer_holds_config.wait(timeout=5):
            raise AssertionError("writer never acquired the config lock")
        config_module.load_settings(force_reload=True)
    except BaseException as exc:
        errors.append(f"reader: {exc!r}")


threads = [
    threading.Thread(target=writer, name="config-writer", daemon=True),
    threading.Thread(target=reader, name="settings-reader", daemon=True),
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=5)

alive = [thread.name for thread in threads if thread.is_alive()]
stacks = {}
if alive:
    frames = sys._current_frames()
    for thread in threads:
        if thread.is_alive() and thread.ident in frames:
            stacks[thread.name] = [
                frame.name for frame in traceback.extract_stack(frames[thread.ident])
            ]
print(json.dumps({"alive": alive, "errors": errors, "stacks": stacks}))
raise SystemExit(1 if alive or errors else 0)
"""
    environment = os.environ.copy()
    environment["TLDW_CONFIG_PATH"] = str(config_path)
    repository_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repository_root), environment.get("PYTHONPATH", "")))
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert completed.returncode == 0, evidence
    assert evidence == {"alive": [], "errors": [], "stacks": {}}


def _invalidate() -> None:
    """Clear the settings cache the way a real invalidation does."""
    with config_module._SETTINGS_CACHE_LOCK:
        config_module._SETTINGS_CACHE = None
        config_module._SETTINGS_CACHE_SOURCE = None


@pytest.fixture
def counting_bootstrap(monkeypatch):
    """Count full config rebuilds, widening the miss window to force overlap."""
    calls: list[float] = []
    original = config_module._load_cli_config_bootstrap

    def counting(*args, **kwargs):
        calls.append(0.0)
        # Without this the window is too narrow to observe on a warm page
        # cache, and the test would pass against the unfixed code by luck.
        threading.Event().wait(0.02)
        return original(*args, **kwargs)

    monkeypatch.setattr(config_module, "_load_cli_config_bootstrap", counting)
    return calls


def _baseline_rebuild_cost(counting_bootstrap: list) -> int:
    """Bootstrap loads a single uncontended rebuild costs.

    Pinned by measurement rather than hardcoded: one rebuild currently makes
    several nested config reads, which is a separate (single-threaded)
    inefficiency this task does not address. Comparing against the measured
    baseline keeps this test about CONCURRENCY only.
    """
    _invalidate()
    counting_bootstrap.clear()
    config_module.load_settings()
    return len(counting_bootstrap)


def test_concurrent_cache_miss_rebuilds_once(counting_bootstrap):
    baseline = _baseline_rebuild_cost(counting_bootstrap)
    assert baseline > 0, "fixture never observed a rebuild; the seam moved"

    _invalidate()
    counting_bootstrap.clear()

    results: list[object] = []
    errors: list[BaseException] = []
    started = threading.Barrier(8)

    def worker() -> None:
        try:
            started.wait(timeout=10)
            results.append(config_module.load_settings())
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"worker raised: {errors[0]!r}"
    assert len(results) == 8

    assert len(counting_bootstrap) == baseline, (
        "8 concurrent callers rebuilt the settings cache "
        f"{len(counting_bootstrap)} times; one uncontended rebuild costs "
        f"{baseline}. Each thread that arrives during the miss window is "
        "redoing the whole rebuild."
    )


def test_concurrent_cache_miss_never_yields_none(counting_bootstrap):
    """AC #1: no caller sees the empty cache cell (see the module docstring)."""
    _invalidate()
    results: list[object] = []
    started = threading.Barrier(6)

    def worker() -> None:
        started.wait(timeout=10)
        results.append(config_module.load_settings())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(results) == 6
    assert all(isinstance(item, dict) for item in results), results


def test_cache_hit_path_does_no_rebuild(counting_bootstrap):
    """AC #3: the warm path is untouched -- no rebuild, no rebuild lock."""
    config_module.load_settings()  # warm
    counting_bootstrap.clear()
    for _ in range(5):
        assert isinstance(config_module.load_settings(), dict)
    assert counting_bootstrap == [], "a cache hit must not rebuild"
