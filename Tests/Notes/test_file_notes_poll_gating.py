"""task-11: File Notes poll -- signature gate, bounds, cap, and backoff.

Covers the four performance pieces of the File Notes workspace poll:

1. ``reconcile`` short-circuits on an unchanged discovery signature, so an
   idle tab performs no replica reads and no second-pass loads after the
   first tick.
2. The discovery walk is bounded (files/entries/depth) and truncation is
   logged, not fatal.
3. The session-change log is capped (newest 500) and same-path repeat
   actions coalesce at append time.
4. The workspace poll backs off to 6 s after four quiet ticks and resets to
   the active cadence on the first real change.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Avoid importing the unrelated optional MLX stack during focused Notes tests.
sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

import tldw_chatbook.Notes.file_notes_service as service_module  # noqa: E402
import tldw_chatbook.Widgets.Library.library_file_notes_workspace as workspace_module  # noqa: E402
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica  # noqa: E402
from tldw_chatbook.Notes.file_notes_session_owner import (  # noqa: E402
    FileNotesSessionOwner,
    SessionChange,
    coalesce_session_changes,
)
from tldw_chatbook.Notes.file_notes_service import FileNotesService  # noqa: E402
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (  # noqa: E402
    LibraryFileNotesWorkspace,
)


@pytest.fixture
def replica() -> FileNotesReplica:
    value = FileNotesReplica(":memory:")
    yield value
    value.close()


def _service(root: Path, replica: FileNotesReplica) -> FileNotesService:
    return FileNotesService(root, replica)


def _seed_vault(root: Path, count: int) -> None:
    for index in range(count):
        (root / f"note-{index:04d}.md").write_text(
            f"body {index}\n", encoding="utf-8"
        )


class _ReplicaSpy:
    """Count replica reads and second-pass file loads during reconciles."""

    def __init__(self, replica: FileNotesReplica) -> None:
        self.replica = replica
        self.active_reads = 0
        self.deleted_reads = 0
        self.upserts = 0
        self._real_active = replica.list_active_files
        self._real_deleted = replica.list_deleted
        self._real_upsert = replica.upsert_file
        replica.list_active_files = self._list_active_files  # type: ignore[method-assign]
        replica.list_deleted = self._list_deleted  # type: ignore[method-assign]
        replica.upsert_file = self._upsert_file  # type: ignore[method-assign]

    def _list_active_files(self, root: str):
        self.active_reads += 1
        return self._real_active(root)

    def _list_deleted(self, root: str):
        self.deleted_reads += 1
        return self._real_deleted(root)

    def _upsert_file(self, *args, **kwargs):
        self.upserts += 1
        return self._real_upsert(*args, **kwargs)


# ---------------------------------------------------------------------------
# 1. Signature gate: idle ticks do nothing after the walk.
# ---------------------------------------------------------------------------


def test_unchanged_vault_skips_replica_reads_across_ticks(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    _seed_vault(root, 12)
    service = _service(root, replica)
    spy = _ReplicaSpy(replica)

    first = service.reconcile()
    assert first.status == "ok"
    assert first.vault_unchanged is False
    assert len(first.entries) == 12
    assert first.created == tuple(
        f"note-{index:04d}.md" for index in range(12)
    )
    assert spy.active_reads == 1

    for _ in range(3):
        repeat = service.reconcile()
        assert repeat.status == "ok"
        assert repeat.vault_unchanged is True
        assert repeat.entries == first.entries
        assert repeat.created == ()
        assert repeat.modified == ()
        assert repeat.deleted == ()

    # The walk still ran four times, but the replica read happened once and
    # the first tick's 12 upserts were the only second-pass work ever done.
    assert spy.active_reads == 1
    assert spy.upserts == 12


def test_touching_a_file_runs_exactly_one_full_reconcile(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    _seed_vault(root, 5)
    service = _service(root, replica)
    spy = _ReplicaSpy(replica)

    service.reconcile()
    service.reconcile()
    assert spy.active_reads == 1

    target = root / "note-0002.md"
    target.write_text("changed body\n", encoding="utf-8")
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000))

    changed = service.reconcile()
    assert changed.vault_unchanged is False
    assert changed.modified == ("note-0002.md",)
    assert spy.active_reads == 2
    assert spy.upserts == 6

    quiet = service.reconcile()
    assert quiet.vault_unchanged is True
    assert quiet.modified == ()
    assert spy.active_reads == 2


def test_reconcile_gate_returns_cached_warning_until_state_moves(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    _seed_vault(root, 2)
    service = _service(root, replica)

    first = service.reconcile()
    assert first.replica_warning is None
    repeat = service.reconcile()
    assert repeat.replica_warning is first.replica_warning


# ---------------------------------------------------------------------------
# 2. Walk bounds: truncation is bounded, deterministic, and non-fatal.
# ---------------------------------------------------------------------------


def test_walk_is_bounded_at_the_file_cap_without_crashing(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    _seed_vault(root, service_module.WALK_MAX_FILES + 500)

    service = _service(root, replica)
    result = service.reconcile()

    assert result.status == "ok"
    assert len(result.entries) == service_module.WALK_MAX_FILES
    # Deterministic lexicographic truncation: the sorted prefix survives.
    assert result.entries[0].relative_path == "note-0000.md"
    assert result.entries[-1].relative_path == (
        f"note-{service_module.WALK_MAX_FILES - 1:04d}.md"
    )
    # A second reconcile is signature-stable despite the invisible tail.
    repeat = service.reconcile()
    assert repeat.vault_unchanged is True


def test_walk_entries_cap_stops_the_walk_early(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    # Three directories of twenty files each: the entry cap lands mid-walk,
    # so the deterministic prefix (the first directory) survives and the
    # rest of the vault is never visited.
    for folder in ("d0", "d1", "d2"):
        directory = root / folder
        directory.mkdir()
        for index in range(20):
            (directory / f"note-{index:04d}.md").write_text(
                f"body {index}\n", encoding="utf-8"
            )
    monkeypatch.setattr(service_module, "WALK_MAX_ENTRIES", 25)

    service = _service(root, replica)
    result = service.reconcile()

    assert result.status == "ok"
    observed = [entry.relative_path for entry in result.entries]
    assert len(observed) == 20
    assert all(path.startswith("d0/") for path in observed)


def test_walk_depth_cap_excludes_deeper_files(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    deep = root
    for level in range(service_module.WALK_MAX_DEPTH + 2):
        deep = deep / f"d{level:02d}"
    deep.mkdir(parents=True)
    (deep / "too-deep.md").write_text("invisible", encoding="utf-8")
    shallow = root
    for level in range(service_module.WALK_MAX_DEPTH - 2):
        shallow = shallow / f"s{level:02d}"
    shallow.mkdir(parents=True)
    (shallow / "visible.md").write_text("visible", encoding="utf-8")

    service = _service(root, replica)
    result = service.reconcile()

    paths = [entry.relative_path for entry in result.entries]
    assert any(path.endswith("visible.md") for path in paths)
    assert not any(path.endswith("too-deep.md") for path in paths)


def test_truncated_walk_never_tombstones_the_invisible_tail(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    _seed_vault(root, 20)
    service = _service(root, replica)
    assert service.reconcile().status == "ok"
    assert sorted(
        item.relative_path for item in replica.list_active_files(service.root_key)
    ) == [f"note-{index:04d}.md" for index in range(20)]

    # Re-bound the walk below the replica's row count: every invisible file
    # must survive -- truncation must not read as deletion.
    monkeypatch.setattr(service_module, "WALK_MAX_FILES", 5)
    result = service.reconcile()

    assert len(result.entries) == 5
    assert result.deleted == ()
    assert len(replica.list_active_files(service.root_key)) == 20
    assert replica.list_deleted(service.root_key) == []


# ---------------------------------------------------------------------------
# 3. Session-change bound: cap + append-time coalescing.
# ---------------------------------------------------------------------------


def test_change_burst_is_capped_at_500_keeping_the_newest(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")

    for index in range(600):
        assert owner.record_change(
            binding,
            SessionChange("created", f"file-{index:04d}.md"),
        )

    changes = owner.snapshot(binding).changes
    assert len(changes) == 500
    paths = [item.change.relative_path for item in changes]
    assert "file-0599.md" in paths  # newest survives
    assert "file-0100.md" in paths  # boundary of the retained window
    assert "file-0099.md" not in paths  # oldest dropped
    groups = coalesce_session_changes(changes)
    assert len(groups) == 500


def test_same_path_same_action_records_append_below_the_bound(
    tmp_path: Path,
) -> None:
    """Below the bound the ledger is append-only (task-11 scope note).

    Commit-authority captures detect drift through lineage sequence ids,
    so same-path same-action records may not merge in normal operation;
    the bound is what protects memory, and compaction engages only past
    it.
    """
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")

    for _ in range(3):
        assert owner.record_change(binding, SessionChange("modified", "a.md"))

    changes = owner.snapshot(binding).changes
    assert [item.change for item in changes] == [
        SessionChange("modified", "a.md"),
    ] * 3
    assert [item.sequence for item in changes] == [1, 2, 3]
    (group,) = coalesce_session_changes(changes)
    assert group.sequence_ids == (1, 2, 3)
    assert group.latest_action == "modified"


def test_overflow_compacts_same_path_duplicates_before_dropping_oldest(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")

    # 300 distinct paths, each recorded twice: 600 raw records pass the
    # bound, but compaction can halve the log without losing a lineage.
    for round_index in range(2):
        for index in range(300):
            assert owner.record_change(
                binding,
                SessionChange("modified", f"file-{index:04d}.md"),
            )

    changes = owner.snapshot(binding).changes
    # The bound held: compaction fired when the log crossed the limit
    # (mid-burst), and the log never exceeded it afterwards.
    assert len(changes) <= 500
    by_path: dict[str, list[int]] = {}
    for item in changes:
        by_path.setdefault(item.change.relative_path, []).append(item.sequence)
    # Every lineage survived...
    assert len(by_path) == 300
    # ...each path kept its newest record (file-0000.md's pair compacted to
    # the second-round record; file-0299.md's first round was still a
    # singleton when compaction ran, so it holds both)...
    assert by_path["file-0000.md"] == [301]
    assert 600 in by_path["file-0299.md"]
    # ...and the coalesced view is one group per path with the latest
    # action intact.
    groups = coalesce_session_changes(changes)
    groups_by_path = {group.source_path: group for group in groups}
    assert len(groups_by_path) == 300
    assert groups_by_path["file-0000.md"].latest_sequence == 301
    assert groups_by_path["file-0299.md"].latest_sequence == 600
    assert groups_by_path["file-0000.md"].latest_action == "modified"


def test_change_log_version_moves_on_every_log_mutation(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    baseline = owner.change_log_version()

    owner.record_change(binding, SessionChange("modified", "a.md"))
    after_append = owner.change_log_version()
    assert after_append > baseline

    # A merge is still a log mutation: the snapshot tuple changes identity.
    owner.record_change(binding, SessionChange("deleted", "a.md"))
    assert owner.change_log_version() > after_append

    # Root selection clears the log.
    owner.select_root(tmp_path / "other")
    assert owner.change_log_version() > after_append


# ---------------------------------------------------------------------------
# 4. Workspace: poll backoff + per-tick session-change skip.
# ---------------------------------------------------------------------------


def _workspace(root: Path, replica: FileNotesReplica) -> LibraryFileNotesWorkspace:
    return LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=1.5,
    )


def test_poll_interval_backs_off_after_quiet_ticks_and_resets(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    workspace = _workspace(root, replica)

    assert workspace._poll_interval == 1.5
    for tick in range(1, workspace_module._POLL_BACKOFF_AFTER_TICKS):
        workspace._record_poll_activity(vault_unchanged=True)
        assert workspace._poll_interval == 1.5, tick

    workspace._record_poll_activity(vault_unchanged=True)
    assert workspace._poll_interval == workspace_module._POLL_BACKOFF_SECONDS

    # Quiet forever stays at the backoff ceiling...
    workspace._record_poll_activity(vault_unchanged=True)
    assert workspace._poll_interval == workspace_module._POLL_BACKOFF_SECONDS

    # ...and the first real change snaps back to the active cadence.
    workspace._record_poll_activity(vault_unchanged=False)
    assert workspace._poll_interval == 1.5
    workspace._record_poll_activity(vault_unchanged=False)
    assert workspace._poll_interval == 1.5


def test_poll_timer_is_recreated_only_when_the_interval_changes(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    workspace = _workspace(root, replica)

    stopped: list[object] = []
    created: list[float] = []

    def fake_set_interval(self, seconds, callback):
        created.append(seconds)
        return SimpleNamespace(stop=lambda: stopped.append(True))

    workspace._poll_timer = SimpleNamespace(stop=lambda: stopped.append(True))
    monkeypatch.setattr(
        workspace_module.LibraryFileNotesWorkspace,
        "set_interval",
        fake_set_interval,
    )

    workspace._set_poll_interval(6.0)
    assert workspace._poll_interval == 6.0
    assert len(stopped) == 1
    assert created == [6.0]

    # Same cadence is a no-op: no stop, no recreate, no phase reset.
    workspace._set_poll_interval(6.0)
    assert len(stopped) == 1
    assert created == [6.0]

    workspace._set_poll_interval(1.5)
    assert workspace._poll_interval == 1.5
    assert len(stopped) == 2
    assert created == [6.0, 1.5]


def test_refresh_session_changes_skips_when_no_new_changes_were_appended(
    tmp_path: Path,
    replica: FileNotesReplica,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("body", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=1.5,
    )
    workspace._session_binding = binding

    snapshots: list[object] = []
    real_snapshot = FileNotesSessionOwner.snapshot

    def counting_snapshot(self_owner, spy_binding):
        snapshots.append(spy_binding)
        return real_snapshot(self_owner, spy_binding)

    # The owner is slotted, so the spy patches the class, not the instance.
    monkeypatch.setattr(FileNotesSessionOwner, "snapshot", counting_snapshot)
    monkeypatch.setattr(
        workspace_module.LibraryFileNotesWorkspace,
        "is_mounted",
        property(lambda self: True),
    )
    monkeypatch.setattr(
        workspace, "_render_session_git_label", lambda *a, **k: None
    )
    monkeypatch.setattr(workspace, "_sync_git_last_action", lambda: False)
    monkeypatch.setattr(workspace, "_rehydrate_push_state", lambda *a, **k: False)
    monkeypatch.setattr(
        workspace._git_panel_widget, "mark_stale", lambda **kwargs: None
    )
    workspace._active = True

    workspace._refresh_session_changes()
    assert len(snapshots) == 1

    # Idle tick: no appends since the last refresh -- no snapshot, no
    # coalesce, no tuple compare.
    workspace._refresh_session_changes()
    workspace._refresh_session_changes()
    assert len(snapshots) == 1

    owner.record_change(binding, SessionChange("modified", "note.md"))
    workspace._refresh_session_changes()
    # The woken refresh snapshots again (plus one internal retain-rows
    # snapshot); the point is that it ran at all after the idle skips.
    assert len(snapshots) > 1


# ---------------------------------------------------------------------------
# 5. Spy evidence: five idle ticks cost walk stats only.
# ---------------------------------------------------------------------------


def test_five_idle_ticks_do_no_replica_work_after_the_first(
    tmp_path: Path,
    replica: FileNotesReplica,
    capsys: pytest.CaptureFixture[str],
) -> None:
    file_count = 40
    root = tmp_path / "notes"
    root.mkdir()
    _seed_vault(root, file_count)
    service = _service(root, replica)
    spy = _ReplicaSpy(replica)

    for _ in range(5):
        service.reconcile()

    print(
        f"\nspy-evidence: files={file_count} ticks=5 "
        f"active_reads={spy.active_reads} deleted_reads={spy.deleted_reads} "
        f"upserts={spy.upserts}"
    )
    assert spy.active_reads == 1
    # One deleted-list read: the one-shot hidden-tombstone sweep on the
    # first reconcile. Gated ticks never touch it again.
    assert spy.deleted_reads == 1
    # All 40 upserts belong to the first tick; the four gated ticks after
    # it did no replica work at all.
    assert spy.upserts == file_count
