"""VideoStore: markers, slugs, save/resolve, retention (task-3401.4)."""

import os
from types import SimpleNamespace

import pytest

from tldw_chatbook.Video_Generation.video_store import (
    VideoStore,
    parse_video_marker,
    slugify_prompt,
    video_content_marker,
)


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
    assert report.removed_files >= 1
    survivors = [p for p in (oldest, middle, newest) if p.exists()]
    assert newest in survivors
    total = sum(p.stat().st_size for p in survivors)
    assert total <= 1024 * 1024


def test_evicted_pairs_reported_for_tombstoning(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="session"))
    _write(store, "msg-9", "clip-x")
    report = store.enforce_retention()
    assert report.evicted == (("msg-9", "clip-x"),)
