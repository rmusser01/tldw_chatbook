"""B23: VideoStore save() takes one post-write snapshot; RecoveredMedia is reused.

Before the fix one ``save()`` walked the whole store 3-4 times (orphan-stage
cleanup walk + capacity snapshot + capacity re-verification walk + empty-dir
prune), and ``allocate_slug`` constructed one ``RecoveredMedia`` catalog handle
per resolve attempt per extension (growing linearly with slug collisions).

Golden equivalence: the scenario below was captured against the pre-change
implementation (paths, sizes, hashes, resolve outputs) and must stay identical.
RecoveredMedia remains per-operation scope: it is constructed at most once per
public call and never cached across calls (the catalog can change underneath
via other processes).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_chatbook.Backup_Recovery.recovered_media as recovered_media_module
from tldw_chatbook.Video_Generation.video_store import VideoStore

# -- fixtures -----------------------------------------------------------------


def _config(max_store_mb: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        retention="ttl", retention_ttl_hours=24, max_store_mb=max_store_mb
    )


@pytest.fixture
def store(tmp_path: Path) -> VideoStore:
    return VideoStore(
        root=tmp_path / "generated_videos",
        config=_config(),
        recovered_root=tmp_path / "recovered_media",
    )


def _dump_root(root: Path) -> dict:
    """Content-addressed on-disk layout of the store root."""
    layout: dict = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_dir():
            layout[rel + "/"] = "dir"
        else:
            data = path.read_bytes()
            layout[rel] = {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    return layout


class _WalkCounter:
    """Counts full-store walks: _snapshot passes + orphan-stage cleanup passes."""

    def __init__(self, store: VideoStore, monkeypatch: pytest.MonkeyPatch) -> None:
        self.snapshot_calls = 0
        self.cleanup_calls = 0
        real_snapshot = store._snapshot
        real_cleanup = store._cleanup_orphan_stages_unlocked

        def counting_snapshot():
            self.snapshot_calls += 1
            return real_snapshot()

        def counting_cleanup():
            self.cleanup_calls += 1
            return real_cleanup()

        monkeypatch.setattr(store, "_snapshot", counting_snapshot)
        monkeypatch.setattr(
            store, "_cleanup_orphan_stages_unlocked", counting_cleanup
        )

    @property
    def full_walks(self) -> int:
        return self.snapshot_calls + self.cleanup_calls


class _RecoveredMediaCounter:
    """Counts RecoveredMedia constructions via the call-time import seam."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.constructions = 0
        real = recovered_media_module.RecoveredMedia
        counter = self

        class Counting(real):
            def __init__(self, root):
                counter.constructions += 1
                super().__init__(root)

        Counting.__name__ = "CountingRecoveredMedia"
        monkeypatch.setattr(recovered_media_module, "RecoveredMedia", Counting)


# -- (a) one save = at most two full store walks -------------------------------


def test_save_on_populated_store_takes_at_most_two_full_walks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VideoStore(
        root=tmp_path / "generated_videos", config=_config(max_store_mb=2)
    )
    store.save("seed-1", "clip", b"A" * 1000, extension="mp4")
    store.save("seed-2", "clip", b"B" * 1000, extension="webm")

    counter = _WalkCounter(store, monkeypatch)
    result = store.save("seed-3", "clip", b"C" * 1000, extension="mp4")

    assert isinstance(result, Path)
    assert counter.full_walks <= 2, (
        f"save() walked the store {counter.full_walks} times "
        f"({counter.snapshot_calls} snapshots + {counter.cleanup_calls} cleanups)"
    )


def test_save_under_capacity_pressure_still_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VideoStore(
        root=tmp_path / "generated_videos", config=_config(max_store_mb=1)
    )
    store.save("seed-1", "clip", b"A" * 700_000, extension="mp4")

    counter = _WalkCounter(store, monkeypatch)
    result = store.save("seed-2", "clip", b"B" * 700_000, extension="mp4")

    assert isinstance(result, Path)
    assert counter.full_walks <= 2, (
        f"save() with eviction walked the store {counter.full_walks} times "
        f"({counter.snapshot_calls} snapshots + {counter.cleanup_calls} cleanups)"
    )
    # the older file was evicted exactly as before
    assert not (tmp_path / "generated_videos" / "seed-1" / "clip.mp4").exists()
    assert (tmp_path / "generated_videos" / "seed-2" / "clip.mp4").exists()


# -- (b) RecoveredMedia constructed at most once per operation -----------------


def test_allocate_slug_constructs_recovered_media_once(
    store: VideoStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovered_root = store._recovered_root
    recovered_root.mkdir(parents=True, exist_ok=True)
    (recovered_root / "catalog.sqlite3").write_bytes(b"")
    store.save("m1", "clip", b"A" * 10, extension="mp4")

    counter = _RecoveredMediaCounter(monkeypatch)
    slug = store.allocate_slug("m1", "clip")

    assert slug == "clip-2"
    assert counter.constructions <= 1, (
        f"allocate_slug constructed RecoveredMedia {counter.constructions}x "
        "for one allocation (must be reused across collision attempts)"
    )


def test_resolve_state_constructs_recovered_media_once(
    store: VideoStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovered_root = store._recovered_root
    recovered_root.mkdir(parents=True, exist_ok=True)
    (recovered_root / "catalog.sqlite3").write_bytes(b"")
    store.save("m1", "clip", b"A" * 10, extension="mp4")

    counter = _RecoveredMediaCounter(monkeypatch)
    status, path = store.resolve_state("m1", "clip", extension="mp4")

    assert status == "ready"
    assert path is not None
    assert counter.constructions <= 1


def test_resolve_state_without_catalog_constructs_nothing(
    store: VideoStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.save("m1", "clip", b"A" * 10, extension="mp4")

    counter = _RecoveredMediaCounter(monkeypatch)
    status, _path = store.resolve_state("m1", "clip", extension="mp4")

    assert status == "ready"
    assert counter.constructions == 0


# -- (c) golden end-state equivalence -----------------------------------------

GOLDEN = {
    "after_populate": {
        "msg-2/": "dir",
        "msg-2/intro.webm": {
            "sha256": "8b9a1a2e581354d25cffd60bec20dd8b9a024056a45e8cbfbab635402c2282a7",
            "size": 700000,
        },
    },
    "after_third_save": {
        "msg-3/": "dir",
        "msg-3/clip.mp4": {
            "sha256": "0ca052dcbbd61f462e68812bb461f33720d5c45133337c861df5aedf017f24f5",
            "size": 700000,
        },
    },
    "allocate_slug_collided": "clip-2",
    "allocate_slug_with_catalog": "clip-2",
    "resolve_missing": ["expired", None],
    "resolve_ready": [
        "ready",
        "<STORE>/msg-3/clip.mp4",
    ],
    "resolve_with_catalog_unknown": [
        "ready",
        "<STORE>/msg-4/clip.mp4",
    ],
    "save3_result": "<STORE>/msg-3/clip.mp4",
}


def test_save_and_resolve_end_state_matches_golden(store: VideoStore) -> None:
    root = store.root
    result = {}

    store.save("msg-1", "clip", b"A" * 700_000, extension="mp4")
    store.save("msg-2", "intro", b"B" * 700_000, extension="webm")
    result["after_populate"] = _dump_root(root)

    stale = root / "msg-2" / ".video-stage-stale.tmp"
    stale.write_bytes(b"stale")
    assert stale.exists()

    save3 = store.save("msg-3", "clip", b"C" * 700_000, extension="mp4")
    result["save3_result"] = str(save3).replace(str(store.root), "<STORE>")
    result["after_third_save"] = _dump_root(root)

    result["resolve_ready"] = _norm_state(
        store.resolve_state("msg-3", "clip", extension="mp4"), store
    )
    result["resolve_missing"] = _norm_state(
        store.resolve_state("msg-9", "clip", extension="mp4"), store
    )

    store.save("msg-4", "clip", b"D" * 10, extension="mp4")
    result["allocate_slug_collided"] = store.allocate_slug("msg-4", "clip")

    recovered_root = store._recovered_root
    recovered_root.mkdir(parents=True, exist_ok=True)
    (recovered_root / "catalog.sqlite3").write_bytes(b"")
    result["allocate_slug_with_catalog"] = store.allocate_slug("msg-4", "clip")
    result["resolve_with_catalog_unknown"] = _norm_state(
        store.resolve_state("msg-4", "clip", extension="mp4"), store
    )

    assert result == GOLDEN


def _norm_state(state: tuple, store: VideoStore) -> list:
    status, path = state
    if path is None:
        return [status, None]
    return [status, str(path).replace(str(store.root), "<STORE>")]
