"""VideoStore: markers, slugs, capacity transactions, and retention."""

import io
import inspect
import multiprocessing
import os
import subprocess
import threading
import time
from contextlib import contextmanager, nullcontext
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
        result = store.save(message_id, "clip", payload_byte * 700_000, extension="mp4")
        outcomes.put("saved" if isinstance(result, Path) else type(result).__name__)
    except BaseException as exc:  # pragma: no cover - relayed to parent
        outcomes.put(type(exc).__name__)


def _call_while_root_lease_is_held(root, monkeypatch, operation) -> None:
    """Assert one operation times out against a real spawned-process lease."""
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    outcomes = ctx.Queue()
    holder = ctx.Process(
        target=_spawn_hold_video_store_lease,
        args=(str(root), ready, release, outcomes),
    )
    holder.start()
    assert ready.wait(10)
    monkeypatch.setattr(video_store_module, "_ROOT_LEASE_TIMEOUT_SECONDS", 0.15)
    started = time.monotonic()
    try:
        with pytest.raises(video_store_module.VideoStoreBusyError):
            operation()
        assert time.monotonic() - started < 2
    finally:
        release.set()
        holder.join(10)
        if holder.is_alive():
            holder.terminate()
            holder.join(5)
    assert holder.exitcode == 0
    assert outcomes.get(timeout=2) == "released"


def _call_while_instance_rlock_is_held(store, operation, assert_still_blocked) -> None:
    """Assert one operation cannot cross a same-store held instance RLock."""
    held = threading.Event()
    attempted = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors = []
    real_lock = store._transaction_lock

    class SignalingRLock:
        def __enter__(self):
            attempted.set()
            real_lock.acquire()
            return self

        def __exit__(self, exc_type, exc, traceback):
            real_lock.release()

    def hold_lock():
        with real_lock:
            held.set()
            assert release.wait(5)

    def call_operation():
        try:
            operation()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            finished.set()

    holder = threading.Thread(target=hold_lock, daemon=True)
    caller = threading.Thread(target=call_operation, daemon=True)
    holder.start()
    assert held.wait(5)
    store._transaction_lock = SignalingRLock()
    caller.start()
    assert attempted.wait(5)
    assert not finished.is_set()
    assert assert_still_blocked() is not False
    release.set()
    holder.join(5)
    caller.join(5)
    assert not holder.is_alive() and not caller.is_alive()
    assert finished.is_set()
    assert errors == []


class _ScandirWrapper:
    """Wrap one scandir context while preserving its close semantics."""

    def __init__(self, wrapped, transform):
        self._wrapped = wrapped
        self._transform = transform

    def __enter__(self):
        self._wrapped.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._wrapped.__exit__(exc_type, exc, traceback)

    def __iter__(self):
        return self

    def __next__(self):
        return self._transform(next(self._wrapped))


class _DirEntryWrapper:
    """Delegate a DirEntry except for a test-controlled stat failure."""

    def __init__(self, wrapped, fail_stat):
        self._wrapped = wrapped
        self._fail_stat = fail_stat

    def stat(self, *, follow_symlinks=True):
        self._fail_stat(Path(self._wrapped.path))
        return self._wrapped.stat(follow_symlinks=follow_symlinks)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _inject_snapshot_failure(store, monkeypatch, victim, seam, error_type):
    """Inject a persistent private I/O error only while `_snapshot` runs."""
    state = {"active": True, "in_snapshot": False, "failures": 0}
    original_snapshot = store._snapshot

    def marked_snapshot():
        state["in_snapshot"] = True
        try:
            return original_snapshot()
        finally:
            state["in_snapshot"] = False

    def fail():
        state["failures"] += 1
        raise error_type("PRIVATE-INVENTORY-PATH")

    monkeypatch.setattr(store, "_snapshot", marked_snapshot)
    original_scandir = os.scandir

    def fail_stat(path):
        if not state["active"] or not state["in_snapshot"]:
            return
        if seam == "message_stat" and path == victim.parent:
            fail()
        if seam == "file_stat" and path == victim:
            fail()

    def patched_scandir(path):
        candidate = Path(path)
        if (
            state["active"]
            and state["in_snapshot"]
            and seam == "message_scan"
            and candidate == victim.parent
        ):
            fail()
        return _ScandirWrapper(
            original_scandir(path),
            lambda entry: _DirEntryWrapper(entry, fail_stat),
        )

    monkeypatch.setattr(os, "scandir", patched_scandir)
    if seam == "resolve":
        original_resolve = Path.resolve

        def patched_resolve(path, *args, **kwargs):
            if (
                state["active"]
                and state["in_snapshot"]
                and path == victim
            ):
                fail()
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", patched_resolve)
    return state


class _CloseRaisingHandle:
    """Delegate a lease file handle but raise after actually closing it."""

    def __init__(self, wrapped, close_calls):
        self._wrapped = wrapped
        self._close_calls = close_calls

    def close(self):
        self._close_calls.append("close")
        self._wrapped.close()
        raise OSError("PRIVATE-CLOSE-FAILURE")

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


@pytest.fixture
def store(tmp_path):
    return VideoStore(root=tmp_path / "generated_videos")


def _config(**overrides):
    base = {"retention": "session", "retention_ttl_hours": 24, "max_store_mb": 2048}
    base.update(overrides)
    return SimpleNamespace(**base)


def _gated_write(store, operation, gate):
    """Run either public managed-publication path through one gate."""
    if operation == "save":
        return store.save(
            "gated-message",
            "clip",
            b"gated-video",
            publication_gate=gate,
            extension="mp4",
        )
    return store.adopt_oversized(
        "gated-message",
        "clip",
        io.BytesIO(b"gated-video"),
        size_bytes=len(b"gated-video"),
        publication_gate=gate,
        extension="mp4",
    )


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


@pytest.mark.parametrize("extension", ["mp4", "webm"])
def test_save_and_resolve_round_trip(store, extension):
    path = store.save(
        "msg-1",
        "a-red-dragon",
        b"video-bytes",
        extension=extension,
    )
    assert path.parent.name == "msg-1"
    assert path.suffix == f".{extension}"
    assert store.resolve("msg-1", "a-red-dragon", extension=extension) == path
    assert path.read_bytes() == b"video-bytes"


@pytest.mark.parametrize("method", ["save", "adopt_oversized", "resolve"])
def test_video_store_public_methods_require_explicit_extension(store, method):
    parameter = inspect.signature(getattr(VideoStore, method)).parameters["extension"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty

    with pytest.raises(TypeError) as caught:
        if method == "save":
            store.save("message", "clip", b"video")
        elif method == "adopt_oversized":
            store.adopt_oversized(
                "message",
                "clip",
                io.BytesIO(b"video"),
                size_bytes=5,
            )
        else:
            store.resolve("message", "clip")

    assert "extension" in str(caught.value)


@pytest.mark.parametrize("operation", ["save", "adopt_oversized"])
def test_stale_slug_allocation_cannot_publish_a_second_canonical_extension(
    tmp_path, operation
):
    root = tmp_path / "generated-videos"
    first_store = VideoStore(root=root, config=_config(retention="ttl"))
    second_store = VideoStore(root=root, config=_config(retention="ttl"))
    allocated = threading.Barrier(2)
    first_published = threading.Event()
    slugs = {}
    paths = []
    errors = []
    second_stream = io.BytesIO(b"second-webm")
    second_stream.seek(4)

    def publish_first():
        try:
            slug = first_store.allocate_slug("message", "shared clip")
            slugs["first"] = slug
            allocated.wait(5)
            paths.append(
                first_store.save(
                    "message", slug, b"first-mp4", extension="mp4"
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            first_published.set()

    def publish_second():
        try:
            slug = second_store.allocate_slug("message", "shared clip")
            slugs["second"] = slug
            allocated.wait(5)
            assert first_published.wait(5)
            if operation == "save":
                second_store.save(
                    "message", slug, b"second-webm", extension="webm"
                )
            else:
                second_store.adopt_oversized(
                    "message",
                    slug,
                    second_stream,
                    size_bytes=len(second_stream.getvalue()),
                    extension="webm",
                )
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=publish_first, daemon=True)
    second = threading.Thread(target=publish_second, daemon=True)
    first.start()
    second.start()
    first.join(10)
    second.join(10)

    assert not first.is_alive() and not second.is_alive()
    assert slugs == {"first": "shared-clip", "second": "shared-clip"}
    assert len(paths) == 1
    assert paths[0].read_bytes() == b"first-mp4"
    assert len(errors) == 1
    assert isinstance(errors[0], video_store_module.VideoStoreSaveError)
    assert str(errors[0]) == "managed video target already exists"
    assert [video.path for video in first_store.iter_stored()] == [paths[0]]
    assert second_store.resolve(
        "message", "shared-clip", extension="webm"
    ) is None
    assert not list(root.rglob(".video-stage-*"))
    if operation == "adopt_oversized":
        assert second_stream.tell() == 0
        assert not second_stream.closed


@pytest.mark.parametrize("operation", ["save", "adopt_oversized"])
def test_existing_same_extension_target_still_fails_without_replacement(
    store, operation
):
    first = store.save("message", "clip", b"first-mp4", extension="mp4")
    stream = io.BytesIO(b"replacement-mp4")
    stream.seek(4)

    with pytest.raises(
        video_store_module.VideoStoreSaveError,
        match="^managed video target already exists$",
    ):
        if operation == "save":
            store.save("message", "clip", b"replacement-mp4", extension="mp4")
        else:
            store.adopt_oversized(
                "message",
                "clip",
                stream,
                size_bytes=len(stream.getvalue()),
                extension="mp4",
            )

    assert first.read_bytes() == b"first-mp4"
    assert [video.path for video in store.iter_stored()] == [first]
    assert not list(store.root.rglob(".video-stage-*"))
    if operation == "adopt_oversized":
        assert stream.tell() == 0
        assert not stream.closed


@pytest.mark.parametrize("extension", ["", ".mp4", "mov", "MP4", "mp4.exe"])
@pytest.mark.parametrize("method", ["save", "adopt_oversized"])
def test_invalid_explicit_extension_fails_before_root_creation(
    tmp_path, extension, method
):
    store = VideoStore(root=tmp_path / "generated-videos")
    stream = io.BytesIO(b"video")
    stream.seek(3)

    with pytest.raises(ValueError, match="unsupported video container"):
        if method == "save":
            store.save("message", "clip", b"video", extension=extension)
        elif method == "adopt_oversized":
            store.adopt_oversized(
                "message",
                "clip",
                stream,
                size_bytes=5,
                extension=extension,
            )
    assert not store.root.exists()
    if method == "adopt_oversized":
        assert stream.tell() == 0
        assert not stream.closed


@pytest.mark.parametrize("extension", ["", ".mp4", "mov", "MP4", "mp4.exe"])
def test_invalid_explicit_resolve_fails_closed_before_root_creation(
    tmp_path, extension
):
    store = VideoStore(root=tmp_path / "generated-videos")

    assert store.resolve("message", "clip", extension=extension) is None
    assert not store.root.exists()


def test_invalid_explicit_resolve_is_not_sanitized_to_mp4(store):
    stored = store.save("message", "clip", b"video", extension="mp4")

    assert stored.exists()
    assert store.resolve("message", "clip", extension=".mp4") is None


@pytest.mark.parametrize("operation", ["save", "adopt"])
def test_publication_gate_cancel_wins_before_save_or_adopt_commit(
    store, operation
):
    reached_precommit = threading.Event()
    release_precommit = threading.Event()
    errors = []
    gate_type = video_store_module.VideoPublicationGate

    class PausingGate(gate_type):
        @contextmanager
        def claim_publication(self):
            reached_precommit.set()
            assert release_precommit.wait(5)
            with super().claim_publication() as active:
                yield active

    gate = PausingGate()

    def write():
        try:
            _gated_write(store, operation, gate)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    worker = threading.Thread(target=write, daemon=True)
    worker.start()
    assert reached_precommit.wait(5)
    gate.cancel()
    release_precommit.set()
    worker.join(5)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], video_store_module.VideoStoreSaveError)
    assert "gated" not in str(errors[0]).lower()
    assert store.resolve("gated-message", "clip", extension="mp4") is None
    assert not list(store.root.rglob(".video-stage-*"))


@pytest.mark.parametrize("operation", ["save", "adopt"])
def test_publication_gate_commit_wins_before_later_cancel(
    store, operation, monkeypatch
):
    commit_started = threading.Event()
    release_commit = threading.Event()
    cancel_attempted = threading.Event()
    cancel_finished = threading.Event()
    errors = []

    class SignalingGate(video_store_module.VideoPublicationGate):
        def cancel(self):
            cancel_attempted.set()
            super().cancel()

    gate = SignalingGate()
    original_commit = store._commit_sibling

    def blocking_commit(sibling, target):
        commit_started.set()
        assert release_commit.wait(5)
        original_commit(sibling, target)

    def write():
        try:
            _gated_write(store, operation, gate)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def cancel():
        gate.cancel()
        cancel_finished.set()

    monkeypatch.setattr(store, "_commit_sibling", blocking_commit)
    worker = threading.Thread(target=write, daemon=True)
    worker.start()
    assert commit_started.wait(5)
    canceller = threading.Thread(target=cancel, daemon=True)
    canceller.start()
    assert cancel_attempted.wait(5)
    assert not cancel_finished.is_set()
    release_commit.set()
    worker.join(5)
    canceller.join(5)

    assert not worker.is_alive() and not canceller.is_alive()
    assert errors == []
    target = store.resolve("gated-message", "clip", extension="mp4")
    assert target is not None
    assert target.read_bytes() == b"gated-video"
    assert cancel_finished.is_set()
    assert not list(store.root.rglob(".video-stage-*"))


def test_resolve_missing_is_none_not_error(store):
    assert store.resolve("msg-1", "never-existed", extension="mp4") is None
    # Unsafe components resolve to None on the read path (durable names may
    # be hand-edited) instead of raising.
    assert store.resolve("../escape", "x", extension="mp4") is None
    assert store.resolve("msg-1", "../../etc/passwd", extension="mp4") is None


def _make_symlink_loop_root(tmp_path: Path) -> Path:
    root = tmp_path / "PRIVATE-MALFORMED-ROOT"
    try:
        root.symlink_to(root.name)
    except OSError:
        pytest.skip("symlink loops are unavailable on this platform")
    return root


@pytest.mark.parametrize("operation", ["save", "adopt"])
def test_malformed_root_public_write_translates_resolution_failure(
    tmp_path: Path, operation: str
) -> None:
    root = _make_symlink_loop_root(tmp_path)
    store = VideoStore(root=root)
    stream = io.BytesIO(b"exact adopted bytes")
    stream.seek(len(stream.getvalue()))
    logged: list[str] = []
    sink_id = video_store_module.logger.add(logged.append, format="{message}")
    try:
        with pytest.raises(video_store_module.VideoStoreSaveError) as raised:
            if operation == "save":
                store.save("message", "clip", b"exact saved bytes", extension="mp4")
            else:
                store.adopt_oversized(
                    "message",
                    "clip",
                    stream,
                    size_bytes=len(stream.getvalue()),
                    extension="mp4",
                )
    finally:
        video_store_module.logger.remove(sink_id)

    assert str(raised.value) == "managed video path resolution failed"
    if operation == "adopt":
        assert stream.tell() == 0
    assert all("PRIVATE-MALFORMED-ROOT" not in message for message in logged)
    assert all(str(root) not in message for message in logged)


def test_malformed_root_read_resolution_fails_closed(tmp_path: Path) -> None:
    root = _make_symlink_loop_root(tmp_path)
    store = VideoStore(root=root)

    assert store.resolve("message", "clip", extension="mp4") is None


def test_successful_save_debug_log_omits_size_and_prompt_derived_name(
    tmp_path: Path,
) -> None:
    store = VideoStore(root=tmp_path / "generated-videos")
    payload = b"x" * 123_457
    logged: list[str] = []
    sink_id = video_store_module.logger.add(
        logged.append, level="DEBUG", format="{message}"
    )
    try:
        saved = store.save(
            "PRIVATE-MESSAGE",
            "private-prompt-derived-name",
            payload,
            extension="mp4",
        )
    finally:
        video_store_module.logger.remove(sink_id)

    assert saved.read_bytes() == payload
    assert all("123457" not in message for message in logged)
    assert all("private-prompt-derived-name" not in message for message in logged)
    assert all("PRIVATE-MESSAGE" not in message for message in logged)


def test_save_refuses_empty_and_unsafe(store):
    with pytest.raises(ValueError, match="empty"):
        store.save("msg-1", "clip", b"", extension="mp4")
    with pytest.raises(ValueError, match="unsafe"):
        store.save("../escape", "clip", b"bytes", extension="mp4")
    with pytest.raises(ValueError, match="unsafe"):
        store.save("msg-1", "has/slash", b"bytes", extension="mp4")


@pytest.mark.parametrize("operation", ["save", "adopt"])
def test_public_writes_reject_internal_stage_namespace_without_mutation(
    store, operation
):
    existing = store.save("existing", "clip", b"existing-bytes", extension="mp4")
    stream = io.BytesIO(b"oversized-source")
    stream.seek(7)
    reserved_slug = ".video-stage-SECRET"

    with pytest.raises(ValueError, match="reserved internal stage namespace") as caught:
        if operation == "save":
            store.save("new", reserved_slug, b"new-bytes", extension="mp4")
        else:
            store.adopt_oversized(
                "new", reserved_slug, stream, size_bytes=len(stream.getvalue()),
                extension="mp4",
            )

    assert "SECRET" not in str(caught.value)
    assert existing.read_bytes() == b"existing-bytes"
    assert [(video.message_id, video.slug) for video in store.iter_stored()] == [
        ("existing", "clip")
    ]
    assert not list(store.root.rglob(".video-stage-*"))
    if operation == "adopt":
        assert not stream.closed
        assert stream.tell() == 0


@pytest.mark.parametrize("occupied_extension", ["mp4", "webm"])
def test_allocate_slug_suffixes_cross_extension_collisions(store, occupied_extension):
    first = store.allocate_slug("msg-1", "a red dragon")
    store.save("msg-1", first, b"v1", extension=occupied_extension)
    second = store.allocate_slug("msg-1", "a red dragon")
    assert second == f"{first}-2"
    store.save("msg-1", second, b"v2", extension="mp4")
    assert store.allocate_slug("msg-1", "a red dragon") == f"{first}-3"


def test_save_enforces_cap_oldest_first_without_startup_cleanup(tmp_path):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="session", max_store_mb=1),
    )
    oldest = store.save("old", "clip", b"a" * 600_000, extension="mp4")
    os.utime(oldest, (1, 1))
    survivor = store.save("new", "clip", b"b" * 300_000, extension="mp4")
    newest = store.save("latest", "clip", b"c" * 600_000, extension="mp4")

    assert not oldest.exists()
    assert survivor.exists()
    assert newest.exists()
    assert store.resolve("old", "clip", extension="mp4") is None
    assert store.resolve("new", "clip", extension="mp4") == survivor
    assert store.resolve("latest", "clip", extension="mp4") == newest
    assert sum(item.size_bytes for item in store.iter_stored()) <= 1024 * 1024


def test_save_uses_safe_path_to_break_equal_mtime_ties(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    later_path = store.save("z-message", "clip", b"z" * 500_000, extension="mp4")
    earlier_path = store.save("a-message", "clip", b"a" * 500_000, extension="mp4")
    os.utime(later_path, (10, 10))
    os.utime(earlier_path, (10, 10))

    newest = store.save("new-message", "clip", b"n" * 200_000, extension="mp4")

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
    stale = store.save("old", "clip", b"a" * 100_000, extension="mp4")
    os.utime(stale, (1, 1))

    def fail_startup_cleanup(*args, **kwargs):
        raise AssertionError("save called startup retention")

    monkeypatch.setattr(store, "enforce_retention", fail_startup_cleanup)
    newest = store.save("new", "clip", b"b" * 100_000, extension="mp4")

    assert stale.exists()
    assert newest.exists()


def test_oversized_save_returns_frozen_capacity_outcome_without_managed_write(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(max_store_mb=1))
    outcome_type = video_store_module.VideoCapacityExceeded

    result = store.save("msg", "too-large", b"x" * (1024 * 1024 + 1), extension="mp4")

    assert result == outcome_type(size_bytes=1024 * 1024 + 1, max_bytes=1024 * 1024)
    assert list(store.iter_stored()) == []
    assert store.resolve("msg", "too-large", extension="mp4") is None
    assert store.capacity_bytes == 1024 * 1024
    with pytest.raises(FrozenInstanceError):
        result.size_bytes = 0
    with pytest.raises(AttributeError):
        store.capacity_bytes = 2


def test_adopt_oversized_publishes_complete_candidate_before_removing_old_files(
    tmp_path, monkeypatch
):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    first = store.save("old-a", "clip", b"a" * 300_000, extension="mp4")
    second = store.save("old-b", "clip", b"b" * 300_000, extension="mp4")
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    original_commit = store._commit_sibling

    def observe_complete_candidate(sibling, target):
        assert sibling.stat().st_size == 1024 * 1024 + 1
        assert first.read_bytes() == b"a" * 300_000
        assert second.read_bytes() == b"b" * 300_000
        original_commit(sibling, target)

    monkeypatch.setattr(store, "_commit_sibling", observe_complete_candidate)
    result = store.adopt_oversized(
        "new", "large", stream, size_bytes=1024 * 1024 + 1,
        extension="mp4",
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
    old = store.save("old", "clip", b"a" * 300_000, extension="mp4")
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))

    def fail_commit(sibling, target):
        raise OSError("PRIVATE-COMMIT-FAILURE")

    monkeypatch.setattr(store, "_commit_sibling", fail_commit)
    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.adopt_oversized(
            "new", "large", stream, size_bytes=1024 * 1024 + 1,
            extension="mp4",
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
            "new", "large", stream, size_bytes=1024 * 1024 + 1,
            extension="mp4",
        )

    assert "PRIVATE" not in str(raised.value)
    assert stream.tell() == 0
    assert not stream.closed
    assert not list(store.root.rglob(".video-stage-*"))


def test_fresh_ttl_startup_retains_one_sole_oversized_exception(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    adopted = store.adopt_oversized(
        "new", "large", stream, size_bytes=1024 * 1024 + 1,
        extension="mp4",
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
        "new", "large", stream, size_bytes=1024 * 1024 + 1,
        extension="mp4",
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
        extension="mp4",
    )

    saved = store.save("new", "small", b"n" * 100_000, extension="mp4")

    assert not adopted.exists()
    assert saved.exists()
    assert sum(item.size_bytes for item in store.iter_stored()) <= store.capacity_bytes


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinked-root containment case")
def test_symlinked_store_root_blocks_save_before_external_publication(
    tmp_path, monkeypatch
):
    external = tmp_path / "private"
    external.mkdir()
    sentinel = external / "PRIVATE-SENTINEL"
    sentinel.write_bytes(b"PRIVATE-SENTINEL")
    linked_root = tmp_path / "gv"
    linked_root.symlink_to(external, target_is_directory=True)
    store = VideoStore(root=linked_root, config=_config(retention="ttl", max_store_mb=1))
    committed = False
    original_commit = store._commit_sibling

    def observe_commit(sibling, target):
        nonlocal committed
        committed = True
        original_commit(sibling, target)

    monkeypatch.setattr(store, "_commit_sibling", observe_commit)
    with pytest.raises(video_store_module.VideoStoreSaveError) as raised:
        store.save("new", "clip", b"new-bytes", extension="mp4")

    assert not committed
    assert "private" not in str(raised.value).lower()
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"
    assert not (external / "new").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinked-root containment case")
def test_symlinked_store_root_never_resolves_external_video(tmp_path):
    external = tmp_path / "private"
    external_target = external / "msg" / "clip.mp4"
    external_target.parent.mkdir(parents=True)
    external_target.write_bytes(b"PRIVATE-SENTINEL")
    linked_root = tmp_path / "gv"
    linked_root.symlink_to(external, target_is_directory=True)
    store = VideoStore(root=linked_root, config=_config(retention="ttl", max_store_mb=1))

    assert store.resolve("msg", "clip", extension="mp4") is None
    assert external_target.read_bytes() == b"PRIVATE-SENTINEL"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinked-root containment case")
def test_symlinked_store_root_blocks_startup_retention(tmp_path):
    external = tmp_path / "private"
    sentinel = external / "msg" / "clip.mp4"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"PRIVATE-SENTINEL")
    linked_root = tmp_path / "gv"
    linked_root.symlink_to(external, target_is_directory=True)
    store = VideoStore(root=linked_root, config=_config(retention="session", max_store_mb=1))

    with pytest.raises(video_store_module.VideoStoreSaveError) as raised:
        store.enforce_retention()

    assert "private" not in str(raised.value).lower()
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinked-root containment case")
def test_symlinked_store_root_blocks_oversized_adoption_before_external_publication(
    tmp_path, monkeypatch
):
    external = tmp_path / "private"
    external.mkdir()
    sentinel = external / "PRIVATE-SENTINEL"
    sentinel.write_bytes(b"PRIVATE-SENTINEL")
    linked_root = tmp_path / "gv"
    linked_root.symlink_to(external, target_is_directory=True)
    store = VideoStore(root=linked_root, config=_config(retention="ttl", max_store_mb=1))
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    committed = False
    original_commit = store._commit_sibling

    def observe_commit(sibling, target):
        nonlocal committed
        committed = True
        original_commit(sibling, target)

    monkeypatch.setattr(store, "_commit_sibling", observe_commit)
    with pytest.raises(video_store_module.VideoStoreSaveError) as raised:
        store.adopt_oversized(
            "new", "large", stream, size_bytes=1024 * 1024 + 1,
            extension="mp4",
        )

    assert not committed
    assert "private" not in str(raised.value).lower()
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"
    assert not (external / "new").exists()
    assert stream.tell() == 0
    assert not stream.closed


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinked-root containment case")
def test_symlinked_store_root_blocks_clear_all(tmp_path):
    external = tmp_path / "private"
    sentinel = external / "msg" / "clip.mp4"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"PRIVATE-SENTINEL")
    linked_root = tmp_path / "gv"
    linked_root.symlink_to(external, target_is_directory=True)
    store = VideoStore(root=linked_root, config=_config(retention="ttl", max_store_mb=1))

    with pytest.raises(video_store_module.VideoStoreSaveError) as raised:
        store.clear_all()

    assert "private" not in str(raised.value).lower()
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-root containment case")
def test_windows_reparse_store_root_blocks_startup_retention(tmp_path):
    external = tmp_path / "private"
    sentinel = external / "msg" / "clip.mp4"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"PRIVATE-SENTINEL")
    junction = tmp_path / "gv"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        pytest.skip("host cannot construct a test root junction")
    store = VideoStore(root=junction, config=_config(retention="session", max_store_mb=1))

    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.enforce_retention()

    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"


def test_save_refuses_existing_target_without_changing_old_bytes(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    existing = store.save("msg", "clip", b"old-bytes", extension="mp4")

    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.save("msg", "clip", b"new-bytes", extension="mp4")

    assert existing.read_bytes() == b"old-bytes"
    assert not list(store.root.rglob(".video-stage-*"))


def test_adoption_refuses_existing_target_without_changing_old_bytes(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    existing = store.save("msg", "clip", b"old-bytes", extension="mp4")
    unrelated = store.save("other", "clip", b"unrelated-bytes", extension="mp4")
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))

    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.adopt_oversized(
            "msg", "clip", stream, size_bytes=1024 * 1024 + 1,
            extension="mp4",
        )

    assert existing.read_bytes() == b"old-bytes"
    assert unrelated.read_bytes() == b"unrelated-bytes"
    assert stream.tell() == 0
    assert not stream.closed
    assert not list(store.root.rglob(".video-stage-*"))


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

    ordinary_old = store.save("ordinary-old", "clip", b"a" * 900_000, extension="mp4")
    ordinary_new = store.save("ordinary-new", "clip", b"b" * 300_000, extension="mp4")
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
        extension="mp4",
    )
    assert adopted.exists()
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"
    assert (store.root / "linked-message").is_symlink()
    assert (linked_file_dir / "linked.mp4").is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX internal directory-symlink case")
def test_snapshot_excludes_internal_message_directory_symlink_alias(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl"))
    real = store.save("real-message", "clip", b"real-video", extension="mp4")
    alias = store.root / "alias-message"
    alias.symlink_to(real.parent, target_is_directory=True)

    stored = list(store.iter_stored())

    assert [(item.message_id, item.slug) for item in stored] == [
        ("real-message", "clip")
    ]
    assert stored[0].path == real
    assert alias.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX internal file-symlink case")
def test_snapshot_excludes_internal_file_symlink_alias(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl"))
    real = store.save("real-message", "clip", b"real-video", extension="mp4")
    alias = real.parent / "alias.mp4"
    alias.symlink_to(real.name)

    stored = list(store.iter_stored())

    assert [(item.message_id, item.slug) for item in stored] == [
        ("real-message", "clip")
    ]
    assert stored[0].path == real
    assert alias.is_symlink()


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

    store.save("old", "clip", b"a" * 900_000, extension="mp4")
    store.save("new", "clip", b"b" * 300_000, extension="mp4")
    store.enforce_retention()
    store.adopt_oversized(
        "adopted",
        "large",
        io.BytesIO(b"z" * (1024 * 1024 + 1)),
        size_bytes=1024 * 1024 + 1,
        extension="mp4",
    )
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"


@pytest.mark.parametrize(
    ("seam", "error_type"),
    [
        pytest.param("message_stat", PermissionError, id="message-stat"),
        pytest.param("message_scan", OSError, id="message-scan"),
        pytest.param("file_stat", PermissionError, id="file-stat"),
        pytest.param("resolve", OSError, id="file-resolve"),
    ],
)
@pytest.mark.parametrize("operation", ["save", "adopt"])
def test_capacity_transactions_fail_closed_on_inventory_io_errors(
    tmp_path, monkeypatch, seam, error_type, operation
):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="ttl", max_store_mb=1),
    )
    existing = store.save("existing", "clip", b"o" * 700_000, extension="mp4")
    existing_bytes = existing.read_bytes()
    state = _inject_snapshot_failure(store, monkeypatch, existing, seam, error_type)
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    stream.seek(17)

    try:
        with pytest.raises(video_store_module.VideoStoreSaveError) as caught:
            if operation == "save":
                store.save("new", "clip", b"n" * 500_000, extension="mp4")
            else:
                store.adopt_oversized(
                    "new",
                    "clip",
                    stream,
                    size_bytes=1024 * 1024 + 1,
                    extension="mp4",
                )
    finally:
        state["active"] = False

    assert state["failures"] > 0
    assert "PRIVATE" not in str(caught.value)
    assert existing.read_bytes() == existing_bytes
    assert store.resolve("new", "clip", extension="mp4") is None
    assert not list(store.root.rglob(".video-stage-*"))
    if operation == "adopt":
        assert not stream.closed
        assert stream.tell() == 0
        assert stream.read(16) == b"z" * 16


def _plant_orphan_stage(store, *, size_bytes=900_000):
    message_dir = store.root / "orphaned"
    message_dir.mkdir(parents=True, exist_ok=True)
    stage = message_dir / ".video-stage-crash.tmp"
    stage.write_bytes(b"s" * size_bytes)
    return stage


@pytest.mark.parametrize("operation", ["startup", "save", "adopt"])
def test_transactions_remove_regular_orphan_stages_before_capacity_work(
    tmp_path, operation
):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="ttl", max_store_mb=1),
    )
    stage = _plant_orphan_stage(store)

    if operation == "startup":
        store.enforce_retention()
    elif operation == "save":
        saved = store.save("new", "clip", b"n" * 300_000, extension="mp4")
        assert saved.read_bytes() == b"n" * 300_000
    else:
        payload = b"z" * (1024 * 1024 + 1)
        adopted = store.adopt_oversized(
            "new", "clip", io.BytesIO(payload), size_bytes=len(payload),
            extension="mp4",
        )
        assert adopted.read_bytes() == payload

    assert not stage.exists()
    assert not list(store.root.rglob(".video-stage-*"))
    actual_bytes = sum(
        path.stat().st_size for path in store.root.rglob("*") if path.is_file()
    )
    if operation != "adopt":
        assert actual_bytes <= store.capacity_bytes


@pytest.mark.parametrize("operation", ["save", "adopt"])
def test_orphan_stage_cleanup_failure_aborts_capacity_transaction(
    tmp_path, monkeypatch, operation
):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="ttl", max_store_mb=1),
    )
    stage = _plant_orphan_stage(store)
    original_unlink = store._checked_unlink

    def fail_stage(video):
        if video.path == stage:
            raise OSError("PRIVATE-STAGE-CLEANUP")
        return original_unlink(video)

    monkeypatch.setattr(store, "_checked_unlink", fail_stage)
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    stream.seek(23)
    with pytest.raises(video_store_module.VideoStoreSaveError) as caught:
        if operation == "save":
            store.save("new", "clip", b"n" * 300_000, extension="mp4")
        else:
            store.adopt_oversized(
                "new", "clip", stream, size_bytes=1024 * 1024 + 1,
                extension="mp4",
            )

    assert "PRIVATE" not in str(caught.value)
    assert stage.read_bytes() == b"s" * 900_000
    assert store.resolve("new", "clip", extension="mp4") is None
    if operation == "adopt":
        assert not stream.closed
        assert stream.tell() == 0


def test_startup_orphan_stage_cleanup_failure_is_not_hidden(tmp_path, monkeypatch):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="ttl", max_store_mb=1),
    )
    stage = _plant_orphan_stage(store)

    def fail_stage(video):
        assert video.path == stage
        raise OSError("PRIVATE-STAGE-CLEANUP")

    monkeypatch.setattr(store, "_checked_unlink", fail_stage)
    with pytest.raises(video_store_module.VideoStoreSaveError) as caught:
        store.enforce_retention()

    assert "PRIVATE" not in str(caught.value)
    assert stage.read_bytes() == b"s" * 900_000


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinked orphan-stage case")
@pytest.mark.parametrize("operation", ["startup", "save", "adopt"])
def test_transactions_leave_suspicious_orphan_stage_link_and_external_target(
    tmp_path, operation
):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="ttl", max_store_mb=1),
    )
    external = tmp_path / "private"
    external.mkdir()
    sentinel = external / "PRIVATE-SENTINEL"
    sentinel.write_bytes(b"PRIVATE-SENTINEL")
    message_dir = store.root / "orphaned"
    message_dir.mkdir(parents=True)
    stage_link = message_dir / ".video-stage-linked.tmp"
    stage_link.symlink_to(sentinel)

    if operation == "startup":
        store.enforce_retention()
    elif operation == "save":
        store.save("new", "clip", b"n" * 300_000, extension="mp4")
    else:
        payload = b"z" * (1024 * 1024 + 1)
        store.adopt_oversized(
            "new", "clip", io.BytesIO(payload), size_bytes=len(payload),
            extension="mp4",
        )

    assert stage_link.is_symlink()
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"


def test_unlock_failure_after_success_does_not_reverse_committed_save(
    tmp_path, monkeypatch
):
    root = tmp_path / "gv"
    store = VideoStore(root=root, config=_config(retention="ttl", max_store_mb=1))
    warnings = []
    close_calls = []
    original_open = Path.open

    class CloseTrackingHandle:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def close(self):
            close_calls.append("close")
            self._wrapped.close()

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def tracked_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return CloseTrackingHandle(handle) if path == store._lease_path else handle

    with monkeypatch.context() as release_patch:
        release_patch.setattr(Path, "open", tracked_open)
        release_patch.setattr(
            video_store_module.portalocker,
            "unlock",
            lambda handle: (_ for _ in ()).throw(OSError("PRIVATE-UNLOCK")),
        )
        release_patch.setattr(
            video_store_module.logger,
            "warning",
            lambda message, *args: warnings.append((message, args)),
        )
        saved = store.save("new", "clip", b"committed", extension="mp4")

    assert saved.read_bytes() == b"committed"
    assert close_calls == ["close"]
    assert warnings == [("VideoStore: lease unlock failed ({})", ("OSError",))]
    assert VideoStore(root=root, config=_config(retention="ttl")).save(
        "later", "clip", b"later",
        extension="mp4",
    ).read_bytes() == b"later"


def test_unlock_failure_never_masks_primary_transaction_error(tmp_path, monkeypatch):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="ttl", max_store_mb=1),
    )
    warnings = []

    def fail_publish(*args, **kwargs):
        raise OSError("PRIVATE-PUBLISH")

    monkeypatch.setattr(store, "_atomic_publish", fail_publish)
    monkeypatch.setattr(
        video_store_module.portalocker,
        "unlock",
        lambda handle: (_ for _ in ()).throw(OSError("PRIVATE-UNLOCK")),
    )
    monkeypatch.setattr(
        video_store_module.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    with pytest.raises(
        video_store_module.VideoStoreSaveError,
        match="managed video publication failed",
    ) as caught:
        store.save("new", "clip", b"payload", extension="mp4")

    assert "PRIVATE" not in str(caught.value)
    assert warnings == [("VideoStore: lease unlock failed ({})", ("OSError",))]


def test_close_failure_after_success_does_not_reverse_committed_save(
    tmp_path, monkeypatch
):
    root = tmp_path / "gv"
    store = VideoStore(root=root, config=_config(retention="ttl", max_store_mb=1))
    original_open = Path.open
    close_calls = []
    warnings = []

    def close_failing_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == store._lease_path:
            return _CloseRaisingHandle(handle, close_calls)
        return handle

    with monkeypatch.context() as release_patch:
        release_patch.setattr(Path, "open", close_failing_open)
        release_patch.setattr(
            video_store_module.logger,
            "warning",
            lambda message, *args: warnings.append((message, args)),
        )
        saved = store.save("new", "clip", b"committed", extension="mp4")

    assert saved.read_bytes() == b"committed"
    assert close_calls == ["close"]
    assert warnings == [("VideoStore: lease close failed ({})", ("OSError",))]
    assert VideoStore(root=root, config=_config(retention="ttl")).save(
        "later", "clip", b"later",
        extension="mp4",
    ).read_bytes() == b"later"


def test_instance_rlock_prevents_thread_transaction_overlap(tmp_path, monkeypatch):
    store = VideoStore(root=tmp_path / "gv", config=_config(max_store_mb=1))
    monkeypatch.setattr(store, "_root_lease", lambda: nullcontext())
    real_lock = store._transaction_lock
    original_publish = store._atomic_publish
    first_entered = threading.Event()
    second_attempted_lock = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    calls = 0
    errors = []
    second_thread = None

    class SignalingRLock:
        def __enter__(self):
            if threading.current_thread() is second_thread:
                second_attempted_lock.set()
            real_lock.acquire()
            return self

        def __exit__(self, exc_type, exc, traceback):
            real_lock.release()

    store._transaction_lock = SignalingRLock()

    def blocking_publish(source, target, *, expected_size, publication_gate=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return original_publish(
            source,
            target,
            expected_size=expected_size,
            publication_gate=publication_gate,
        )

    monkeypatch.setattr(store, "_atomic_publish", blocking_publish)

    def save(message_id):
        try:
            store.save(message_id, "clip", b"x" * 100_000, extension="mp4")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=save, args=("first",), daemon=True)
    second = threading.Thread(target=save, args=("second",), daemon=True)
    second_thread = second
    first.start()
    assert first_entered.wait(5)
    second.start()
    assert second_attempted_lock.wait(5)
    assert not second_entered.is_set()
    release.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert second_entered.is_set()
    assert errors == []


def test_adopt_oversized_takes_instance_rlock(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    stream.seek(7)
    result = []

    def operation():
        result.append(
            store.adopt_oversized(
                "new", "large", stream, size_bytes=1024 * 1024 + 1,
                extension="mp4",
            )
        )

    def assert_blocked():
        assert result == []
        assert stream.tell() == 0
        assert store.resolve("new", "large", extension="mp4") is None

    _call_while_instance_rlock_is_held(store, operation, assert_blocked)

    assert result[0].read_bytes() == b"z" * (1024 * 1024 + 1)
    assert stream.tell() == 0
    assert not stream.closed


def test_enforce_retention_takes_instance_rlock(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="session", max_store_mb=1))
    existing = store.save("old", "clip", b"old-bytes", extension="mp4")

    _call_while_instance_rlock_is_held(
        store,
        store.enforce_retention,
        lambda: existing.read_bytes() == b"old-bytes",
    )

    assert not existing.exists()


def test_clear_all_takes_instance_rlock(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    existing = store.save("old", "clip", b"old-bytes", extension="mp4")

    _call_while_instance_rlock_is_held(
        store,
        store.clear_all,
        lambda: existing.read_bytes() == b"old-bytes",
    )

    assert not store.root.exists()


def test_two_store_instances_serialize_through_root_lease(tmp_path, monkeypatch):
    root = tmp_path / "gv"
    first_store = VideoStore(root=root, config=_config(max_store_mb=1))
    second_store = VideoStore(root=root, config=_config(max_store_mb=1))
    original_first = first_store._atomic_publish
    original_second = second_store._atomic_publish
    original_second_lease = second_store._root_lease
    first_entered = threading.Event()
    second_attempted_lease = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    errors = []

    def blocking_first(source, target, *, expected_size, publication_gate=None):
        first_entered.set()
        assert release.wait(5)
        return original_first(
            source,
            target,
            expected_size=expected_size,
            publication_gate=publication_gate,
        )

    def observe_second(source, target, *, expected_size, publication_gate=None):
        second_entered.set()
        return original_second(
            source,
            target,
            expected_size=expected_size,
            publication_gate=publication_gate,
        )

    @contextmanager
    def signaling_second_lease():
        second_attempted_lease.set()
        with original_second_lease():
            yield

    monkeypatch.setattr(first_store, "_atomic_publish", blocking_first)
    monkeypatch.setattr(second_store, "_atomic_publish", observe_second)
    monkeypatch.setattr(second_store, "_root_lease", signaling_second_lease)

    def save(store, message_id):
        try:
            store.save(message_id, "clip", b"x" * 100_000, extension="mp4")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=save, args=(first_store, "first"), daemon=True)
    second = threading.Thread(target=save, args=(second_store, "second"), daemon=True)
    first.start()
    assert first_entered.wait(5)
    second.start()
    assert second_attempted_lease.wait(5)
    assert not second_entered.is_set()
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
            store.save("blocked", "clip", b"x", extension="mp4")
        assert time.monotonic() - started < 2
    finally:
        release.set()
        holder.join(10)
        if holder.is_alive():
            holder.terminate()
            holder.join(5)
    assert holder.exitcode == 0
    assert outcomes.get(timeout=2) == "released"


def test_adopt_oversized_takes_root_lease(tmp_path, monkeypatch):
    root = tmp_path / "gv"
    store = VideoStore(root=root, config=_config(retention="ttl", max_store_mb=1))
    stream = io.BytesIO(b"z" * (1024 * 1024 + 1))
    stream.seek(9)

    _call_while_root_lease_is_held(
        root,
        monkeypatch,
        lambda: store.adopt_oversized(
            "new", "large", stream, size_bytes=1024 * 1024 + 1,
            extension="mp4",
        ),
    )

    assert store.resolve("new", "large", extension="mp4") is None
    assert stream.tell() == 0
    assert not stream.closed
    assert stream.read(16) == b"z" * 16


def test_enforce_retention_takes_root_lease(tmp_path, monkeypatch):
    root = tmp_path / "gv"
    store = VideoStore(root=root, config=_config(retention="session", max_store_mb=1))
    existing = store.save("old", "clip", b"old-bytes", extension="mp4")

    _call_while_root_lease_is_held(root, monkeypatch, store.enforce_retention)

    assert existing.read_bytes() == b"old-bytes"


def test_clear_all_takes_root_lease(tmp_path, monkeypatch):
    root = tmp_path / "gv"
    store = VideoStore(root=root, config=_config(retention="ttl", max_store_mb=1))
    existing = store.save("old", "clip", b"old-bytes", extension="mp4")

    _call_while_root_lease_is_held(root, monkeypatch, store.clear_all)

    assert existing.read_bytes() == b"old-bytes"


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
    first = store.save("old-a", "clip", b"a" * 300_000, extension="mp4")
    second = store.save("old-b", "clip", b"b" * 300_000, extension="mp4")
    original = {first: first.read_bytes(), second: second.read_bytes()}

    def fail_commit(sibling, target):
        assert sibling.exists()
        assert not target.exists()
        raise OSError("PRIVATE-COMMIT-FAILURE")

    monkeypatch.setattr(store, "_commit_sibling", fail_commit)
    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.save("new", "clip", b"n" * 500_000, extension="mp4")

    assert {path: path.read_bytes() for path in original} == original
    assert store.resolve("new", "clip", extension="mp4") is None
    assert not list(store.root.rglob(".video-stage-*"))


def test_first_required_victim_failure_withdraws_new_target(tmp_path, monkeypatch):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    oldest = store.save("old-a", "clip", b"a" * 600_000, extension="mp4")
    survivor = store.save("old-b", "clip", b"b" * 300_000, extension="mp4")
    os.utime(oldest, (1, 1))
    original_unlink = store._checked_unlink

    def fail_oldest(video):
        if video.path == oldest:
            raise OSError("PRIVATE-UNLINK-FAILURE")
        return original_unlink(video)

    monkeypatch.setattr(store, "_checked_unlink", fail_oldest)
    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.save("new", "clip", b"n" * 600_000, extension="mp4")

    assert oldest.read_bytes() == b"a" * 600_000
    assert survivor.read_bytes() == b"b" * 300_000
    assert store.resolve("new", "clip", extension="mp4") is None
    assert not list(store.root.rglob(".video-stage-*"))


def test_later_victim_failure_withdraws_new_target_and_leaves_bounded_store(
    tmp_path, monkeypatch
):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    first = store.save("old-a", "clip", b"a" * 350_000, extension="mp4")
    second = store.save("old-b", "clip", b"b" * 350_000, extension="mp4")
    third = store.save("old-c", "clip", b"c" * 100_000, extension="mp4")
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
        store.save("new", "clip", b"n" * 700_000, extension="mp4")

    assert not first.exists()
    assert second.read_bytes() == b"b" * 350_000
    assert third.read_bytes() == b"c" * 100_000
    assert store.resolve("new", "clip", extension="mp4") is None
    assert sum(item.size_bytes for item in store.iter_stored()) <= store.capacity_bytes


@pytest.mark.skipif(os.name == "nt", reason="POSIX deterministic symlink-swap case")
def test_capacity_victim_is_revalidated_after_snapshot_before_unlink(
    tmp_path, monkeypatch
):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl", max_store_mb=1))
    oldest = store.save("old", "clip", b"o" * 900_000, extension="mp4")
    os.utime(oldest, (1, 1))
    external = tmp_path / "private"
    external.mkdir()
    sentinel = external / "PRIVATE-SENTINEL"
    sentinel.write_bytes(b"PRIVATE-SENTINEL")
    original_sort = store._sorted_oldest
    swapped = False

    def swap_selected_victim(videos):
        nonlocal swapped
        selected = original_sort(videos)
        if not swapped and any(video.path == oldest for video in selected):
            oldest.unlink()
            oldest.symlink_to(sentinel)
            swapped = True
        return selected

    monkeypatch.setattr(store, "_sorted_oldest", swap_selected_victim)
    with pytest.raises(video_store_module.VideoStoreSaveError):
        store.save("new", "clip", b"n" * 300_000, extension="mp4")

    assert swapped
    assert sentinel.read_bytes() == b"PRIVATE-SENTINEL"
    assert oldest.is_symlink()
    assert oldest.resolve() == sentinel
    assert store.resolve("new", "clip", extension="mp4") is None
    assert not list(store.root.rglob(".video-stage-*"))


# -- retention -------------------------------------------------------------


def _write(store, message_id, slug, payload=b"x" * 100, age_seconds=0):
    path = store.save(message_id, slug, payload, extension="mp4")
    if age_seconds:
        old = path.stat().st_mtime - age_seconds
        os.utime(path, (old, old))
    return path


def _plant_unknown_video(store: VideoStore, *, size: int = 100) -> Path:
    path = store.root / "legacy-message" / "legacy.mov"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * size)
    return path


def test_snapshot_accounts_for_unknown_video_suffix_but_resolve_cannot_serve_it(
    tmp_path,
):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="ttl"))
    unknown = _plant_unknown_video(store)

    assert [video.path for video in store.iter_stored()] == [unknown]
    assert store.resolve("legacy-message", "legacy", extension="mp4") is None
    assert store.resolve("legacy-message", "legacy", extension="webm") is None


@pytest.mark.parametrize("retention", ["session", "ttl"])
def test_retention_removes_unknown_video_suffix(tmp_path, retention):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention=retention, retention_ttl_hours=1),
    )
    unknown = _plant_unknown_video(store)
    os.utime(unknown, (1, 1))

    report = store.enforce_retention(now=3700)

    assert report.removed_files == 1
    assert not unknown.exists()


def test_capacity_accounts_for_unknown_video_suffix(tmp_path):
    store = VideoStore(
        root=tmp_path / "gv",
        config=_config(retention="ttl", max_store_mb=1),
    )
    unknown = _plant_unknown_video(store, size=900_000)
    os.utime(unknown, (1, 1))

    saved = store.save(
        "new-message",
        "clip",
        b"n" * 300_000,
        extension="mp4",
    )

    assert not unknown.exists()
    assert saved.exists()
    assert sum(video.size_bytes for video in store.iter_stored()) <= store.capacity_bytes


def test_session_retention_wipes_everything(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="session"))
    _write(store, "msg-1", "clip-a")
    _write(store, "msg-2", "clip-b")
    report = store.enforce_retention()
    assert report.removed_files == 2
    assert store.resolve("msg-1", "clip-a", extension="mp4") is None
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
