"""File Notes replica retention bounds (task-34382, ADR-218).

Without retention the ADR-029 replica grows forever: one checkpoint per
editing session per protected file, and tombstones that outlive the deletion
by years. These tests pin the bounded policy — per-note checkpoint cap,
30-day tombstone/revision expiry, and the two things never evicted
(protected paths, the most-recent tombstone) — by exact counts before and
after.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Avoid importing the unrelated optional MLX stack during focused Notes tests.
sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

from tldw_chatbook.Notes.file_notes_replica import (  # noqa: E402
    MAX_REVISIONS_PER_NOTE,
    RECOVERY_EXPIRY_DAYS,
    FileNotesReplica,
)
from tldw_chatbook.Notes.file_notes_service import FileNotesService  # noqa: E402

NOW = datetime(2026, 10, 4, 12, 0, 0, tzinfo=timezone.utc)


def _digest(raw_bytes: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw_bytes).hexdigest()


def _upsert(
    replica: FileNotesReplica,
    root: str,
    relative_path: str,
    raw_bytes: bytes,
) -> None:
    replica.upsert_file(
        root,
        relative_path,
        raw_bytes,
        content_hash=_digest(raw_bytes),
        decoded_text=raw_bytes.decode("utf-8"),
        size=len(raw_bytes),
        mtime_ns=1,
    )


def _checkpoint(
    replica: FileNotesReplica,
    root: str,
    relative_path: str,
    payload: bytes,
    session_key: str,
    created_at: datetime,
) -> None:
    replica.checkpoint(
        root,
        relative_path,
        payload,
        content_hash=_digest(payload),
        session_key=session_key,
        created_at=created_at.isoformat(),
    )


def _tombstone(
    replica: FileNotesReplica,
    root: str,
    relative_path: str,
    deleted_at: datetime,
) -> None:
    payload = f"final bytes of {relative_path}".encode()
    _upsert(replica, root, relative_path, payload)
    replica.prepare_deletion(
        root,
        relative_path,
        payload,
        content_hash=_digest(payload),
        decoded_text=payload.decode("utf-8"),
        deleted_at=deleted_at.isoformat(),
        created_at=deleted_at.isoformat(),
    )


def _revision_count(
    replica: FileNotesReplica,
    root: str,
    relative_path: str,
    *,
    kind: str | None = None,
) -> int:
    if kind is None:
        query = "SELECT COUNT(*) FROM revisions WHERE root = ? AND relative_path = ?"
        parameters: tuple[object, ...] = (root, relative_path)
    else:
        query = (
            "SELECT COUNT(*) FROM revisions "
            "WHERE root = ? AND relative_path = ? AND kind = ?"
        )
        parameters = (root, relative_path, kind)
    return int(
        replica._connection.execute(query, parameters).fetchone()[0]
    )


@pytest.fixture
def replica() -> FileNotesReplica:
    value = FileNotesReplica(":memory:")
    yield value
    value.close()


def test_checkpoint_cap_evicts_the_oldest_beyond_the_bound(replica: FileNotesReplica) -> None:
    root = "/notes"
    total = MAX_REVISIONS_PER_NOTE + 7
    for index in range(total):
        stamp = NOW - timedelta(days=2) + timedelta(minutes=index)
        _checkpoint(
            replica,
            root,
            "plain.md",
            f"plain body {index}".encode(),
            f"session-{index:03d}",
            stamp,
        )
    for index in range(total):
        stamp = NOW - timedelta(days=2) + timedelta(minutes=index)
        _checkpoint(
            replica,
            root,
            "kept.md",
            f"kept body {index}".encode(),
            f"session-{index:03d}",
            stamp,
        )
    replica.protect(root, "kept.md")

    replica.enforce_retention(root, now=NOW)

    assert _revision_count(replica, root, "plain.md") == MAX_REVISIONS_PER_NOTE
    # Recency: the survivors are the NEWEST sessions, the oldest are gone.
    survivors = replica.list_revisions(root, "plain.md", limit=total)
    assert [entry.session_key for entry in survivors] == [
        f"session-{index:03d}"
        for index in range(total - MAX_REVISIONS_PER_NOTE, total)
    ][::-1]
    # Protected paths are exempt from every eviction (AC #3).
    assert _revision_count(replica, root, "kept.md") == total


def test_delete_revisions_do_not_consume_the_checkpoint_cap(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    for index in range(3):
        _tombstone_and_restore(replica, root, f"cycled-{index}.md")

    replica.enforce_retention(root, now=NOW)

    # Nothing expired (all fresh) and no path reached the cap; delete
    # revisions survive alongside the checkpoints they were taken with.
    assert (
        _revision_count(replica, root, "cycled-0.md", kind="delete") == 1
    )


def _tombstone_and_restore(
    replica: FileNotesReplica,
    root: str,
    relative_path: str,
) -> None:
    _tombstone(replica, root, relative_path, deleted_at=NOW - timedelta(days=1))
    replica.clear_tombstone(root, relative_path)


def test_old_tombstones_and_revisions_expire_but_fresh_ones_stay(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    # Active note with one stale and one fresh checkpoint.
    _upsert(replica, root, "active.md", b"active current bytes")
    _checkpoint(
        replica, root, "active.md", b"stale", "s-old", NOW - timedelta(days=40)
    )
    _checkpoint(
        replica, root, "active.md", b"fresh", "s-new", NOW - timedelta(days=1)
    )
    # Tombstones: one far past the cutoff, one well inside it.
    _tombstone(replica, root, "gone-old.md", NOW - timedelta(days=40))
    _tombstone(replica, root, "gone-fresh.md", NOW - timedelta(days=2))
    # A protected note's stale history is untouchable.
    _upsert(replica, root, "protected.md", b"protected current bytes")
    replica.protect(root, "protected.md")
    _checkpoint(
        replica,
        root,
        "protected.md",
        b"stale protected",
        "s-old",
        NOW - timedelta(days=40),
    )

    replica.enforce_retention(root, now=NOW)

    # Tombstone expiry: only the fresh deletion remains listed.
    assert replica.list_deleted(root) == ["gone-fresh.md"]
    assert replica.get_restore_bytes(root, "gone-old.md") is None
    assert (
        replica.get_restore_bytes(root, "gone-fresh.md")
        == b"final bytes of gone-fresh.md"
    )
    # Revision expiry: stale unprotected revisions gone, fresh and
    # protected ones kept.
    assert _revision_count(replica, root, "active.md") == 1
    assert replica.list_revisions(root, "active.md", limit=10)[0].session_key == (
        "s-new"
    )
    assert _revision_count(replica, root, "gone-fresh.md", kind="delete") == 1
    assert _revision_count(replica, root, "protected.md") == 1
    # Active rows are never touched by tombstone expiry.
    assert replica.get_bytes(root, "active.md") == b"active current bytes"


def test_the_most_recent_tombstone_is_never_evicted_even_when_old(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    _tombstone(replica, root, "older.md", NOW - timedelta(days=40))
    _tombstone(replica, root, "newer.md", NOW - timedelta(days=31))

    replica.enforce_retention(root, now=NOW)

    # Both are past the 30-day cutoff, but the most recent deletion always
    # survives -- the user's last delete stays restorable.
    assert replica.list_deleted(root) == ["newer.md"]
    assert (
        replica.get_restore_bytes(root, "newer.md") == b"final bytes of newer.md"
    )

    # A root whose ONLY tombstone is old keeps it, too.
    solo = "/solo"
    _tombstone(replica, solo, "lone.md", NOW - timedelta(days=60))
    replica.enforce_retention(solo, now=NOW)
    assert replica.list_deleted(solo) == ["lone.md"]


def test_retention_never_touches_other_roots(replica: FileNotesReplica) -> None:
    root_a = "/notes/a"
    root_b = "/notes/b"
    for index in range(MAX_REVISIONS_PER_NOTE + 3):
        _checkpoint(
            replica,
            root_a,
            "note.md",
            f"a {index}".encode(),
            f"a-{index:03d}",
            NOW - timedelta(days=1, minutes=-index),
        )
        _checkpoint(
            replica,
            root_b,
            "note.md",
            f"b {index}".encode(),
            f"b-{index:03d}",
            NOW - timedelta(days=1, minutes=-index),
        )

    replica.enforce_retention(root_a, now=NOW)

    assert _revision_count(replica, root_a, "note.md") == MAX_REVISIONS_PER_NOTE
    assert _revision_count(replica, root_b, "note.md") == MAX_REVISIONS_PER_NOTE + 3


def test_unparseable_timestamps_fail_safe_and_are_kept(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    _tombstone(replica, root, "weird.md", NOW - timedelta(days=90))
    replica._connection.execute(
        "UPDATE files SET deleted_at = ? WHERE root = ? AND relative_path = ?",
        ("not a timestamp", root, "weird.md"),
    )
    # Drop root_b's only tombstone competitor so weird.md is also the
    # most-recent tombstone; the unparseable value must not evict it.
    replica._connection.execute(
        "UPDATE revisions SET created_at = ? WHERE root = ?",
        ("also not a timestamp", root),
    )

    replica.enforce_retention(root, now=NOW)

    assert replica.list_deleted(root) == ["weird.md"]
    assert _revision_count(replica, root, "weird.md") == 1


def test_expiry_boundary_keeps_exactly_thirty_days_and_evicts_past_it(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    exactly_cutoff = NOW - timedelta(days=RECOVERY_EXPIRY_DAYS)
    _tombstone(replica, root, "edge.md", exactly_cutoff)

    replica.enforce_retention(root, now=NOW)

    # Exactly thirty days old is not OLDER than the cutoff: it survives
    # (here it is also the only, hence most-recent, tombstone).
    assert replica.list_deleted(root) == ["edge.md"]

    # With a fresh competitor, the exactly-cutoff tombstone still survives
    # on the boundary alone, and one minute past it is evicted.
    other = "/other"
    _tombstone(replica, other, "edge.md", exactly_cutoff)
    _tombstone(replica, other, "past.md", exactly_cutoff - timedelta(minutes=1))
    _tombstone(replica, other, "fresh.md", NOW - timedelta(days=1))
    replica.enforce_retention(other, now=NOW)
    # Newest deletion first is list_deleted's contract.
    assert replica.list_deleted(other) == ["fresh.md", "edge.md"]


def test_service_enforces_retention_for_its_own_root(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "retention-notes"
    root.mkdir()
    (root / "plain.md").write_text("body\n", encoding="utf-8")
    service = FileNotesService(root, replica)
    root_key = service.root_key
    for index in range(MAX_REVISIONS_PER_NOTE + 5):
        _checkpoint(
            replica,
            root_key,
            "plain.md",
            f"body {index}".encode(),
            f"s-{index:03d}",
            NOW - timedelta(days=1, minutes=-index),
        )
    before = _revision_count(replica, root_key, "plain.md")

    result = service.enforce_retention()

    assert result.status == "ok"
    assert before == MAX_REVISIONS_PER_NOTE + 5
    assert _revision_count(replica, root_key, "plain.md") == MAX_REVISIONS_PER_NOTE

    # Without a replica there is nothing to enforce, and that is reported
    # rather than silently succeeding.
    bare = FileNotesService(root, None)
    try:
        assert bare.enforce_retention().status == "replica-error"
    finally:
        bare.close()
