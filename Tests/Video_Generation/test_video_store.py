"""VideoStore: markers, slugs, capacity transactions, and retention."""

import io
import multiprocessing
import os
import subprocess
import threading
import time
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_chatbook.Video_Generation.video_store as video_store_module
from tldw_chatbook.Video_Generation.video_store import (
    VideoStore,
    parse_video_marker,
    slugify_prompt,
    video_content_marker,
)


def _spawn_hold_video_store_lease(root: str, ready, release, outcomes) -> None:
    """Hold the real root lease until the parent permits release."""
    store = VideoStore(root=Path(root), config=_config(max_store_mb=1))
    try:
        with store._root_lease():
            ready.set()
            if not release.wait(10):
                outcomes.put("release-timeout")
                return
        outcomes.put("released")
    except BaseException as exc:  # pragma: no cover - relayed to parent
        outcomes.put(type(exc).__name__)


def _spawn_capacity_save(
    root: str,
    message_id: str,
    payload_byte: bytes,
    start,
    outcomes,
) -> None:
    """Perform one spawn-safe capped save and relay its typed outcome."""
    store = VideoStore(root=Path(root), config=_config(retention="ttl", max_store_mb=1))
    if not start.wait(10):
        outcomes.put("start-timeout")
        return
    try:
        result = store.save(message_id, "clip", payload_byte * 700_000)
        outcomes.put("saved" if isinstance(result, Path) else type(result).__name__)
    except BaseException as exc:  # pragma: no cover - relayed to parent
        outcomes.put(type(exc).__name__)


@pytest.fixture
def store(tmp_path):
    return VideoStore(root=tmp_path / "generated_videos")


def _config(**overrides):
    base = {"retention": "session", "retention_ttl_hours": 24, "max_store_mb": 2048}
    base.update(overrides)
    return SimpleNamespace(**base)


# -- markers ---------------------------------------------------------------


def test_marker_round_trip():
    marker = video_content_marker("dusk-over-neon-tokyo")
    assert marker == "[video] dusk-over-neon-tokyo"
    assert parse_video_marker(marker) == "dusk-over-neon-tokyo"


def test_parse_marker_ignores_non_video_content():
    assert parse_video_marker("[image] a red dragon") is None
    assert parse_video_marker("plain text") is None
    assert parse_video_marker("[video] ") is None
    assert parse_video_marker("") is None


# -- slugify ---------------------------------------------------------------


def test_slugify_normalizes_prompt():
    assert slugify_prompt("A Red Dragon, soaring over Mt. Fuji!") == "a-red-dragon-soaring-over-mt"
    assert slugify_prompt("   ") == "clip"
    assert slugify_prompt("!!!") == "clip"
    assert len(slugify_prompt("word " * 40)) <= 48


def test_slugify_keeps_only_alnum_runs():
    assert slugify_prompt("dusk/over: neon—tokyo") == "dusk-over-neon-tokyo"


# -- save / resolve --------------------------------------------------------


def test_save_and_resolve_round_trip(store):
    path = store.save("msg-1", "a-red-dragon", b"video-bytes")
    assert path.parent.name == "msg-1"
    assert store.resolve("msg-1", "a-red-dragon") == path
    assert path.read_bytes() == b"video-bytes"


def test_resolve_missing_is_none_not_error(store):
    assert store.resolve("msg-1", "never-existed") is None
    # Unsafe components resolve to None on the read path (durable names may
    # be hand-edited) instead of raising.
    assert store.resolve("../escape", "x") is None
    assert store.resolve("msg-1", "../../etc/passwd") is None


def test_save_refuses_empty_and_unsafe(store):
    with pytest.raises(ValueError, match="empty"):
        store.save("msg-1", "clip", b"")
    with pytest.raises(ValueError, match="unsafe"):
        store.save("../escape", "clip", b"bytes")
    with pytest.raises(ValueError, match="unsafe"):
        store.save("msg-1", "has/slash", b"bytes")


def test_allocate_slug_suffixes_collisions(store):
    first = store.allocate_slug("msg-1", "a red dragon")
    store.save("msg-1", first, b"v1")
    second = store.allocate_slug("msg-1", "a red dragon")
    assert second == f"{first}-2"
    store.save("msg-1", second, b"v2")
    assert store.allocate_slug("msg-1", "a red dragon") == f"{first}-3"


def test_save_enforces_cap_oldest_first_without_startup_cleanup(tmp_path):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="session", max_store_mb=1),
    )
    oldest = store.save("old", "clip", b"a" * 600_000)
    os.utime(oldest, (1, 1))
    survivor = store.save("new", "clip", b"b" * 300_000)
    newest = store.save("latest", "clip", b"c" * 600_000)

    assert not oldest.exists()
    assert survivor.exists()
    assert newest.exists()
    assert store.resolve("old", "clip") is None
    assert store.resolve("new", "clip") == survivor
    assert store.resolve("latest", "clip") == newest
    assert sum(item.size_bytes for item in store.iter_stored()) <= 1024 * 1024


def test_save_uses_safe_path_to_break_equal_mtime_ties(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    later_path = store.save("z-message", "clip", b"z" * 500_000)
    earlier_path = store.save("a-message", "clip", b"a" * 500_000)
    os.utime(later_path, (10, 10))
    os.utime(earlier_path, (10, 10))

    newest = store.save("new-message", "clip", b"n" * 200_000)

    assert not earlier_path.exists()
    assert later_path.exists()
    assert newest.exists()


@pytest.mark.parametrize("retention", ["session", "ttl"])
def test_save_never_calls_startup_retention_or_evaluates_age(
    tmp_path, monkeypatch, retention
):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention=retention, retention_ttl_hours=1, max_store_mb=1),
    )
    stale = store.save("old", "clip", b"a" * 100_000)
    os.utime(stale, (1, 1))

    def fail_startup_cleanup(*args, **kwargs):
        raise AssertionError("save called startup retention")

    monkeypatch.setattr(store, "enforce_retention", fail_startup_cleanup)
    newest = store.save("new", "clip", b"b" * 100_000)

    assert stale.exists()
    assert newest.exists()


def test_oversized_save_returns_frozen_capacity_outcome_without_managed_write(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(max_store_mb=1))
    outcome_type = video_store_module.VideoCapacityExceeded

    result = store.save("msg", "too-large", b"x" * (1024 * 1024 + 1))

    assert result == outcome_type(size_bytes=1024 * 1024 + 1, max_bytes=1024 * 1024)
    assert list(store.iter_stored()) == []
    assert store.resolve("msg", "too-large") is None
    assert store.capacity_bytes == 1024 * 1024
    with pytest.raises(FrozenInstanceError):
        result.size_bytes = 0
    with pytest.raises(AttributeError):
        store.capacity_bytes = 2


def test_adopt_oversized_publishes_complete_candidate_before_removing_old_files(
    tmp_path, monkeypatch
):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    first = store.save("old-a", "clip", b"a" * 300_000)
    second = store.save("old-b", "clip", b"b" * 300_000)
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    original_commit = store._commit_sibling

    def observe_complete_candidate(sibling, target):
        assert sibling.stat().st_size == 1024 * 1024 + 1
        assert first.read_bytes() == b"a" * 300_000
        assert second.read_bytes() == b"b" * 300_000
        original_commit(sibling, target)

    monkeypatch.setattr(store, "_commit_sibling", observe_complete_candidate)
    result = store.adopt_oversized(
        "new", "large", stream, size_bytes=1024 * 1024 + 1
    )

    assert result.read_bytes() == b"z" * (1024 * 1024 + 1)
    assert [(item.message_id, item.slug) for item in store.iter_stored()] == [
        ("new", "large")
    ]
    assert stream.tell() == 0
    assert not stream.closed


def test_failed_oversized_adoption_keeps_source_open_and_rewound(
    tmp_path, monkeypatch
):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    old = store.save("old", "clip", b"a" * 300_000)
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))

    def fail_commit(sibling, target):
        raise OSError("PRIVATE-COMMIT-FAILURE")

    monkeypatch.setattr(store, "_commit_sibling", fail_commit)
    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.adopt_oversized(
            "new", "large", stream, size_bytes=1024 * 1024 + 1
        )

    assert old.read_bytes() == b"a" * 300_000
    assert stream.tell() == 0
    assert not stream.closed
    assert stream.read() == b"z" * (1024 * 1024 + 1)
    assert not list(store.root.rglob(".video-stage-*"))


def test_oversized_source_read_failure_is_typed_and_recoverable(tmp_path):
    class BrokenStream(io.BytesIO):
        def read(self, *args, **kwargs):
            raise RuntimeError("PRIVATE-SOURCE-FAILURE")

    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    stream = BrokenStream(b"z" * (1024 * 1024 + 1))

    with pytest.raises(video_store_module.VideoStoreSaveError) as raised:
        store.adopt_oversized(
            "new", "large", stream, size_bytes=1024 * 1024 + 1
        )

    assert "PRIVATE" not in str(raised.value)
    assert stream.tell() == 0
    assert not stream.closed
    assert not list(store.root.rglob(".video-stage-*"))


def test_fresh_ttl_startup_retains_one_sole_oversized_exception(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    adopted = store.adopt_oversized(
        "new", "large", stream, size_bytes=1024 * 1024 + 1
    )

    report = store.enforce_retention(now=time.time())

    assert report.removed_files == 0
    assert adopted.exists()


def test_ttl_startup_restores_cap_when_oversized_file_has_another_survivor(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    oversized = store.root / "old" / "large.mp4"
    companion = store.root / "new" / "small.mp4"
    oversized.parent.mkdir(parents=True)
    companion.parent.mkdir(parents=True)
    oversized.write_bytes(b"z" * (1024 * 1024 + 1))
    companion.write_bytes(b"n" * 100_000)
    os.utime(oversized, (1, 1))
    os.utime(companion, (2, 2))

    report = store.enforce_retention(now=3)

    assert report.evicted == (("old", "large"),)
    assert not oversized.exists()
    assert companion.exists()
    assert sum(item.size_bytes for item in store.iter_stored()) <= store.capacity_bytes


def test_session_startup_removes_sole_oversized_exception(tmp_path):
    ttl_store = VideoStore(
        root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1)
    )
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    adopted = ttl_store.adopt_oversized(
        "new", "large", stream, size_bytes=1024 * 1024 + 1
    )
    session_store = VideoStore(
        root=tmp_path / "gv", config=_config(retention="session", max_store_mb=1)
    )

    report = session_store.enforce_retention()

    assert report.removed_files == 1
    assert not adopted.exists()


def test_ordinary_save_evicts_sole_oversized_exception_before_success(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    adopted = store.adopt_oversized(
        "old",
        "large",
        io.BytesIO(b"z" * (1024 * 1024 + 1)),
        size_bytes=1024 * 1024 + 1,
    )

    saved = store.save("new", "small", b"n" * 100_000)

    assert not adopted.exists()
    assert saved.exists()
    assert sum(item.size_bytes for item in store.iter_stored()) <= store.capacity_bytes


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink containment case")
def test_capacity_operations_never_follow_symlinked_directories_or_files(
    tmp_path,
):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    external = tmp_path / "private"
    external.mkdir()
    sentinel = external / "PRIVATE-SENTINEL"
    sentinel.write_bytes(b"PRIVATE-SENTINEL")
    store.root.mkdir(parents=True)
    (store.root / "linked-message").symlink_to(external, target_is_directory=True)
    linked_file_dir = store.root / "real-message"
    linked_file_dir.mkdir()
    (linked_file_dir / "linked.mp4").symlink_to(sentinel)

    ordinary_old = store.save("ordinary-old", "clip", b"a" * 900_000)
    ordinary_new = store.save("ordinary-new", "clip", b"b" * 300_000)
    assert not ordinary_old.exists() and ordinary_new.exists()
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"

    ordinary_new.unlink()
    first = store.root / "startup-a" / "clip.mp4"
    second = store.root / "startup-b" / "clip.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"a" * 700_000)
    second.write_bytes(b"b" * 700_000)
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    store.enforce_retention(now=3)
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"

    adopted = store.adopt_oversized(
        "adopted",
        "large",
        io.BytesIO(b"z" * (1024 * 1024 + 1)),
        size_bytes=1024 * 1024 + 1,
    )
    assert adopted.exists()
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"
    assert (store.root / "linked-message").is_symlink()
    assert (linked_file_dir / "linked.mp4").is_symlink()


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse containment case")
def test_capacity_operations_never_follow_windows_junction(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    external = tmp_path / "private"
    external.mkdir()
    sentinel = external / "PRIVATE-SENTINEL"
    sentinel.write_bytes(b"PRIVATE-SENTINEL")
    store.root.mkdir(parents=True)
    junction = store.root / "linked-message"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        pytest.skip("host cannot construct a test junction")

    store.save("old", "clip", b"a" * 900_000)
    store.save("new", "clip", b"b" * 300_000)
    store.enforce_retention()
    store.adopt_oversized(
        "adopted",
        "large",
        io.BytesIO(b"z" * (1024 * 1024 + 1)),
        size_bytes=1024 * 1024 + 1,
    )
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"


def test_instance_rlock_prevents_thread_transaction_overlap(tmp_path, monkeypatch):
    store = VideoStore(root=tmp_path / "gv", config=_config(max_store_mb=1))
    monkeypatch.setattr(store, "_root_lease", lambda: nullcontext())
    original_publish = store._atomic_publish
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    calls = 0
    errors = []

    def blocking_publish(source, target, *, expected_size):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return original_publish(source, target, expected_size=expected_size)

    monkeypatch.setattr(store, "_atomic_publish", blocking_publish)

    def save(message_id):
        try:
            store.save(message_id, "clip", b"x" * 100_000)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=save, args=("first",), daemon=True)
    second = threading.Thread(target=save, args=("second",), daemon=True)
    first.start()
    assert first_entered.wait(5)
    second.start()
    assert not second_entered.wait(0.25)
    release.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert second_entered.is_set()
    assert errors == []


def test_two_store_instances_serialize_through_root_lease(tmp_path, monkeypatch):
    root = tmp_path / "gv"
    first_store = VideoStore(root=root, config=_config(max_store_mb=1))
    second_store = VideoStore(root=root, config=_config(max_store_mb=1))
    original_first = first_store._atomic_publish
    original_second = second_store._atomic_publish
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    errors = []

    def blocking_first(source, target, *, expected_size):
        first_entered.set()
        assert release.wait(5)
        return original_first(source, target, expected_size=expected_size)

    def observe_second(source, target, *, expected_size):
        second_entered.set()
        return original_second(source, target, expected_size=expected_size)

    monkeypatch.setattr(first_store, "_atomic_publish", blocking_first)
    monkeypatch.setattr(second_store, "_atomic_publish", observe_second)

    def save(store, message_id):
        try:
            store.save(message_id, "clip", b"x" * 100_000)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=save, args=(first_store, "first"), daemon=True)
    second = threading.Thread(target=save, args=(second_store, "second"), daemon=True)
    first.start()
    assert first_entered.wait(5)
    second.start()
    assert not second_entered.wait(0.25)
    release.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert second_entered.is_set()
    assert errors == []


def test_spawned_lease_holder_causes_bounded_busy_error(tmp_path, monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    outcomes = ctx.Queue()
    root = tmp_path / "gv"
    holder = ctx.Process(
        target=_spawn_hold_video_store_lease,
        args=(str(root), ready, release, outcomes),
    )
    holder.start()
    assert ready.wait(10)
    monkeypatch.setattr(video_store_module, "_ROOT_LEASE_TIMEOUT_SECONDS", 0.15)
    store = VideoStore(root=root, config=_config(max_store_mb=1))
    started = time.monotonic()
    try:
        with pytest.raises(video_store_module.VideoStoreBusyError):
            store.save("blocked", "clip", b"x")
        assert time.monotonic() - started < 2
    finally:
        release.set()
        holder.join(10)
        if holder.is_alive():
            holder.terminate()
            holder.join(5)
    assert holder.exitcode == 0
    assert outcomes.get(timeout=2) == "released"


def test_spawned_saves_leave_actual_store_within_capacity(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    outcomes = ctx.Queue()
    root = tmp_path / "gv"
    processes = [
        ctx.Process(
            target=_spawn_capacity_save,
            args=(str(root), message_id, payload, start, outcomes),
        )
        for message_id, payload in (("first", b"a"), ("second", b"b"))
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(outcomes.get(timeout=2) for _ in processes) == ["saved", "saved"]
    store = VideoStore(root=root, config=_config(retention="ttl", max_store_mb=1))
    assert sum(item.size_bytes for item in store.iter_stored()) <= store.capacity_bytes


def test_atomic_commit_failure_preserves_old_store_and_removes_sibling(
    tmp_path, monkeypatch
):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    first = store.save("old-a", "clip", b"a" * 300_000)
    second = store.save("old-b", "clip", b"b" * 300_000)
    original = {first: first.read_bytes(), second: second.read_bytes()}

    def fail_commit(sibling, target):
        assert sibling.exists()
        assert not target.exists()
        raise OSError("PRIVATE-COMMIT-FAILURE")

    monkeypatch.setattr(store, "_commit_sibling", fail_commit)
    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.save("new", "clip", b"n" * 500_000)

    assert {path: path.read_bytes() for path in original} == original
    assert store.resolve("new", "clip") is None
    assert not list(store.root.rglob(".video-stage-*"))


def test_first_required_victim_failure_withdraws_new_target(tmp_path, monkeypatch):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    oldest = store.save("old-a", "clip", b"a" * 600_000)
    survivor = store.save("old-b", "clip", b"b" * 300_000)
    os.utime(oldest, (1, 1))
    original_unlink = store._checked_unlink

    def fail_oldest(video):
        if video.path == oldest:
            raise OSError("PRIVATE-UNLINK-FAILURE")
        return original_unlink(video)

    monkeypatch.setattr(store, "_checked_unlink", fail_oldest)
    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.save("new", "clip", b"n" * 600_000)

    assert oldest.read_bytes() == b"a" * 600_000
    assert survivor.read_bytes() == b"b" * 300_000
    assert store.resolve("new", "clip") is None
    assert not list(store.root.rglob(".video-stage-*"))


def test_later_victim_failure_withdraws_new_target_and_leaves_bounded_store(
    tmp_path, monkeypatch
):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    first = store.save("old-a", "clip", b"a" * 350_000)
    second = store.save("old-b", "clip", b"b" * 350_000)
    third = store.save("old-c", "clip", b"c" * 100_000)
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    os.utime(third, (3, 3))
    original_unlink = store._checked_unlink

    def fail_second(video):
        if video.path == second:
            raise OSError("PRIVATE-LATER-UNLINK-FAILURE")
        return original_unlink(video)

    monkeypatch.setattr(store, "_checked_unlink", fail_second)
    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.save("new", "clip", b"n" * 700_000)

    assert not first.exists()
    assert second.read_bytes() == b"b" * 350_000
    assert third.read_bytes() == b"c" * 100_000
    assert store.resolve("new", "clip") is None
    assert sum(item.size_bytes for item in store.iter_stored()) <= store.capacity_bytes


# -- retention -------------------------------------------------------------


def _write(store, message_id, slug, payload=b"x" * 100, age_seconds=0):
    path = store.save(message_id, slug, payload)
    if age_seconds:
        old = path.stat().st_mtime - age_seconds
        os.utime(path, (old, old))
    return path


def test_session_retention_wipes_everything(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="session"))
    _write(store, "msg-1", "clip-a")
    _write(store, "msg-2", "clip-b")
    report = store.enforce_retention()
    assert report.removed_files == 2
    assert store.resolve("msg-1", "clip-a") is None
    # Empty per-message directories are pruned too.
    assert not (store.root / "msg-1").exists()


def test_ttl_retention_keeps_fresh_removes_stale(tmp_path):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="ttl", retention_ttl_hours=1),
    )
    fresh = _write(store, "msg-1", "fresh")
    stale = _write(store, "msg-2", "stale", age_seconds=3700)
    report = store.enforce_retention()
    assert report.removed_files == 1
    assert fresh.exists() and not stale.exists()


def test_size_cap_evicts_oldest_first_in_any_mode(tmp_path):
    # 1MB cap; three ~0.4MB clips (400_000 bytes each) cannot all fit.
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="ttl", retention_ttl_hours=999, max_store_mb=1),
    )
    oldest = _write(store, "msg-1", "oldest", payload=b"a" * 400_000, age_seconds=300)
    middle = _write(store, "msg-2", "middle", payload=b"b" * 400_000, age_seconds=200)
    newest = _write(store, "msg-3", "newest", payload=b"c" * 400_000)
    report = store.enforce_retention()
    assert not oldest.exists()  # evicted first
    assert report.removed_files == 0  # save already enforced the cap
    survivors = [p for p in (oldest, middle, newest) if p.exists()]
    assert newest in survivors
    total = sum(p.stat().st_size for p in survivors)
    assert total <= 1024 * 1024


def test_evicted_pairs_reported_for_tombstoning(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="session"))
    _write(store, "msg-9", "clip-x")
    report = store.enforce_retention()
    assert report.evicted == (("msg-9", "clip-x"),)
