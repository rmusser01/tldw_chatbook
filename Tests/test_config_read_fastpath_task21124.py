"""TASK-21124: cache-hit config reads must not take the global file lock.

`get_cli_setting` (398 call sites, many on the Textual event loop) resolves
through `load_cli_config_and_ensure_existence` ->
`_load_cli_config_bootstrap`, which used to acquire `_config_file_lock()`
unconditionally BEFORE the cache short-circuit. A concurrent config write
holds that same lock through two fsyncs and multiple full TOML parses, so
one write stalled every loop-side read for the whole write.

These tests pin the repair:

* the warm-cache read path never calls `_config_file_lock()` at all
  (lock-acquisition counter, not timing -- CI-noise-proof);
* one write performs exactly one on-disk TOML parse (the read-modify-write
  read) plus one string parse (the TASK-13157 verify parse, whose output now
  feeds `_publish_runtime_config_unlocked` instead of two further re-reads);
* a reader racing a writer only ever observes complete configs -- two keys
  written atomically together are never seen torn, and the observed write
  counter never goes backwards;
* a read immediately after a completed write sees the new value (the fast
  path cannot serve a stale-forever cache);
* an informational timing probe prints solo vs concurrent reader latency
  percentiles (assertion deliberately generous; the lock counter above is
  the hard gate).
"""

from __future__ import annotations

import statistics
import threading
import time
from types import SimpleNamespace

import pytest

import tldw_chatbook.config as config_module


@pytest.fixture
def counting_file_lock(monkeypatch):
    """Count `_config_file_lock()` acquisitions without changing behavior."""
    counter = {"calls": 0}
    original = config_module._config_file_lock

    def counting():
        counter["calls"] += 1
        return original()

    monkeypatch.setattr(config_module, "_config_file_lock", counting)
    return counter


def _warm_cache() -> None:
    """Prime the bootstrap cache (creates the isolated config on first run)."""
    config = config_module.load_cli_config_and_ensure_existence(force_reload=True)
    assert isinstance(config, dict)


def test_cache_hit_read_takes_no_file_lock(counting_file_lock):
    """AC #1: cache-hit reads never take the file lock."""
    _warm_cache()
    counting_file_lock["calls"] = 0

    for _ in range(50):
        config_module.get_cli_setting("general", "users_name", "default")
        config_module.load_cli_config_and_ensure_existence()

    assert counting_file_lock["calls"] == 0, (
        f"{counting_file_lock['calls']} file-lock acquisitions on 100 "
        "warm-cache reads; the cache-hit path must not touch "
        "_config_file_lock at all"
    )


def test_cache_miss_still_takes_the_lock(counting_file_lock):
    """The slow path keeps its serialization: a miss goes through the lock."""
    _warm_cache()
    with config_module._config_file_lock():
        config_module._CONFIG_CACHE = None
        config_module._CONFIG_CACHE_SOURCE = None
    counting_file_lock["calls"] = 0

    config_module.load_cli_config_and_ensure_existence()

    assert counting_file_lock["calls"] >= 1, (
        "a cache miss must still serialize through _config_file_lock"
    )


def test_force_reload_still_takes_the_lock(counting_file_lock):
    """force_reload always goes through the locked path."""
    _warm_cache()
    counting_file_lock["calls"] = 0

    config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert counting_file_lock["calls"] >= 1


def test_read_after_write_sees_the_new_value():
    """The fast path must not serve a stale cache after a published write."""
    _warm_cache()
    for value in ("first", "second", "third"):
        assert config_module.save_setting_to_cli_config(
            "task21124_probe", "freshness", value
        )
        assert (
            config_module.get_cli_setting("task21124_probe", "freshness") == value
        )


def test_write_path_is_coalesced_to_one_disk_parse(monkeypatch):
    """AC #2: one write = one on-disk parse (RMW read) + one verify parse.

    Before TASK-21124 a single `save_settings_to_cli_config` parsed TOML four
    times while holding the write lock: the read-modify-write read, the
    TASK-13157 verify parse-back, and two further full re-reads of the file
    it had just written (`_publish_runtime_config_unlocked`'s bootstrap
    reload plus `load_settings(force_reload=True)`'s own bootstrap reload).
    The verify parse now feeds the publish step, so the re-reads are gone.
    """
    _warm_cache()

    real_tomllib = config_module.tomllib
    counts = {"load": 0, "loads": 0}

    def counting_load(stream):
        counts["load"] += 1
        return real_tomllib.load(stream)

    def counting_loads(text):
        counts["loads"] += 1
        return real_tomllib.loads(text)

    proxy = SimpleNamespace(
        load=counting_load,
        loads=counting_loads,
        TOMLDecodeError=real_tomllib.TOMLDecodeError,
    )
    monkeypatch.setattr(config_module, "tomllib", proxy)

    assert config_module.save_setting_to_cli_config(
        "task21124_probe", "parse_count", "value"
    )

    assert counts["load"] == 1, (
        f"{counts['load']} on-disk TOML parses for one write; only the "
        "read-modify-write read may touch the file"
    )
    assert counts["loads"] == 1, (
        f"{counts['loads']} string TOML parses for one write; only the "
        "TASK-13157 verify parse (which feeds the publish step) is allowed"
    )


def test_concurrent_writer_never_yields_torn_or_regressing_reads():
    """AC (correctness): readers racing a writer see only complete configs.

    The writer publishes two keys with the same value in one atomic
    mutation; a reader observing them unequal has seen a torn config, and a
    reader observing the counter decrease has seen a stale-beyond-current
    cache resurrected.
    """
    _warm_cache()
    assert config_module.save_settings_to_cli_config(
        {"task21124_pair": {"a": 0, "b": 0}}
    )

    stop = threading.Event()
    errors: list[str] = []

    def writer() -> None:
        for i in range(1, 9):
            if stop.is_set():
                break
            ok = config_module.save_settings_to_cli_config(
                {"task21124_pair": {"a": i, "b": i}}
            )
            if not ok:
                errors.append(f"write {i} failed")
                break
        stop.set()

    def reader() -> None:
        last_seen = -1
        while not stop.is_set():
            config = config_module.load_cli_config_and_ensure_existence()
            pair = config.get("task21124_pair", {})
            a, b = pair.get("a"), pair.get("b")
            if a is None or b is None:
                # A config from before the seeding write: complete, just old.
                continue
            if a != b:
                errors.append(f"torn read: a={a} b={b}")
                stop.set()
                return
            if a < last_seen:
                errors.append(f"regressing read: saw {a} after {last_seen}")
                stop.set()
                return
            last_seen = a
            # Yield the GIL so the spinning readers do not starve the writer
            # (three busy-loops stretched 8 writes to minutes on CI-class
            # hardware while proving nothing extra).
            time.sleep(0.0005)

    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader) for _ in range(3)]
    writer_thread.start()
    for thread in reader_threads:
        thread.start()
    writer_thread.join(timeout=120)
    stop.set()
    for thread in reader_threads:
        thread.join(timeout=30)

    assert not writer_thread.is_alive(), "writer never finished"
    assert not errors, errors[0]


def test_reader_latency_informational_probe(capsys):
    """AC #3 evidence: reader percentiles, solo vs concurrent writes.

    The hard no-lock gate is `test_cache_hit_read_takes_no_file_lock`; this
    probe exists to print the before/after numbers the task records. The
    assertion is deliberately generous (CI machines stall arbitrarily): the
    concurrent p50 must stay under 5 ms -- on the fixed fast path it is
    sub-microsecond dict work, while the pre-fix behavior queued every read
    behind whole multi-fsync writes.
    """
    _warm_cache()

    def timed_reads(count: int) -> list[float]:
        samples = []
        for _ in range(count):
            start = time.perf_counter()
            config_module.get_cli_setting("general", "users_name", "default")
            samples.append(time.perf_counter() - start)
        return samples

    solo = timed_reads(2000)

    # Concurrent phase: keep reading for as long as the writer is actually
    # writing (5 full write cycles), so every sample genuinely races a
    # write -- a fixed read count can finish before the writer thread even
    # reaches its first lock acquisition.
    writer_started = threading.Event()
    writer_finished = threading.Event()

    def writer() -> None:
        writer_started.set()
        for i in range(5):
            config_module.save_setting_to_cli_config(
                "task21124_probe", "latency_burn", i
            )
        writer_finished.set()

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    assert writer_started.wait(timeout=10)
    concurrent: list[float] = []
    deadline = time.monotonic() + 120
    try:
        while not writer_finished.is_set() and time.monotonic() < deadline:
            concurrent.extend(timed_reads(50))
    finally:
        writer_thread.join(timeout=120)
    assert not writer_thread.is_alive()
    assert len(concurrent) >= 100, "not enough overlapped samples collected"

    def pct(samples: list[float], q: float) -> float:
        return statistics.quantiles(samples, n=100)[int(q) - 1]

    stalled = [s for s in concurrent if s > 0.001]
    print(
        "task21124 reader latency (us): "
        f"solo p50={pct(solo, 50) * 1e6:.1f} p95={pct(solo, 95) * 1e6:.1f} "
        f"p99={pct(solo, 99) * 1e6:.1f} | concurrent-writes "
        f"p50={pct(concurrent, 50) * 1e6:.1f} "
        f"p95={pct(concurrent, 95) * 1e6:.1f} "
        f"p99={pct(concurrent, 99) * 1e6:.1f} "
        f"max={max(concurrent) * 1e6:.0f} "
        f"stalls>1ms={len(stalled)}/{len(concurrent)}"
        # An unpaced reader oversamples the uncontended gaps, so a whole-
        # write stall shows up as a FEW huge samples, invisible at p95 --
        # max and the >1ms count are the honest stall signal here.
    )

    assert pct(concurrent, 50) < 0.005, (
        "median read latency under concurrent writes exceeded 5 ms -- "
        "reads are queueing behind the config write lock again"
    )
