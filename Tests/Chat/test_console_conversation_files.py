"""TASK-18060 Task 2 (review-rail spec §1): cross-turn conversation file
aggregation.

Two layers under test:

- The PURE assembly, ``conversation_file_summary`` -- dict rows, no I/O,
  exercised directly (mirrors ``Tests/Chat/test_console_turn_file_entries.py``'s
  ``_row`` convention).
- ``AgentRunsChangeReviewProvider.conversation_changed_files()`` against a
  REAL git-backed shadow repo, real ``ChangeTurnTracker`` turns, and a
  file-backed ``AgentRunsDB`` -- the fixture-invented-shapes trap has
  bitten this repo four separate times, so the provider is driven for
  real rather than faked (``Tests/UI/test_change_review_screen.py``'s
  ``review_fixture`` pattern).
"""
from __future__ import annotations

import shutil

import pytest

from tldw_chatbook.Chat.console_display_state import (
    ConversationFileEntry,
    conversation_file_summary,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.change_review_screen import (
    AgentRunsChangeReviewProvider,
)
from tldw_chatbook.Workspaces.change_tracking import ChangedFile, ShadowRepoService
from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker


def _row(row_id=1, root="/ws", run_id="run-1"):
    return {"id": row_id, "root": root, "run_id": run_id}


# -- pure assembly ------------------------------------------------------------


def test_latest_wins_same_file_two_rows():
    """Same (root, path) covered by two rows: the NEWER row's identity and
    counts win -- oldest-first input, last writer wins (spec §1)."""
    row1 = _row(row_id=1, run_id="run-1")
    row2 = _row(row_id=2, run_id="run-2")
    rows_with_files = [
        (row1, [ChangedFile(path="a.py", status="M", adds=1, dels=1)]),
        (row2, [ChangedFile(path="a.py", status="M", adds=5, dels=2)]),
    ]
    entries = conversation_file_summary(rows_with_files, {})
    assert entries == [
        ConversationFileEntry(
            root="/ws",
            path="a.py",
            label="a.py",
            status="M",
            adds=5,
            dels=2,
            run_id="run-2",
            snapshot_id=2,
            note_count=0,
        )
    ]


def test_rename_supersedes_old_path_entry():
    """A rename (status R) keys by its NEW path and deletes the old path's
    entry, even though the old path was written by an earlier row."""
    row1 = _row(row_id=1, run_id="run-1")
    row2 = _row(row_id=2, run_id="run-2")
    rows_with_files = [
        (row1, [ChangedFile(path="old.py", status="A", adds=3, dels=0)]),
        (
            row2,
            [
                ChangedFile(
                    path="new.py", status="R", adds=0, dels=0, old_path="old.py"
                )
            ],
        ),
    ]
    entries = conversation_file_summary(rows_with_files, {})
    assert entries == [
        ConversationFileEntry(
            root="/ws",
            path="new.py",
            label="new.py",
            status="R",
            adds=0,
            dels=0,
            run_id="run-2",
            snapshot_id=2,
            note_count=0,
        )
    ]


def test_delete_then_recreate_shows_added():
    """A path deleted in an earlier row and recreated in a later row shows
    the later row's "A", not "D" -- pure latest-wins, no special casing."""
    row1 = _row(row_id=1, run_id="run-1")
    row2 = _row(row_id=2, run_id="run-2")
    rows_with_files = [
        (row1, [ChangedFile(path="a.py", status="D", adds=0, dels=9)]),
        (row2, [ChangedFile(path="a.py", status="A", adds=4, dels=0)]),
    ]
    entries = conversation_file_summary(rows_with_files, {})
    assert len(entries) == 1
    assert entries[0].status == "A"
    assert entries[0].adds == 4
    assert entries[0].dels == 0
    assert entries[0].snapshot_id == 2


def test_multi_root_entries_prefix_the_root_name():
    """Same convention as ``turn_file_entries``: the label is bare when a
    single root contributed, root-name-prefixed when several did."""
    row_one = _row(row_id=1, root="/ws/one", run_id="run-1")
    row_two = _row(row_id=2, root="/ws/two", run_id="run-1")
    rows_with_files = [
        (row_one, [ChangedFile(path="x.md", status="A", adds=5, dels=0)]),
        (row_two, [ChangedFile(path="y.md", status="D", adds=0, dels=7)]),
    ]
    labels = {
        entry.path: entry.label
        for entry in conversation_file_summary(rows_with_files, {})
    }
    assert labels == {"x.md": "one/x.md", "y.md": "two/y.md"}


def test_single_root_entries_use_bare_relpaths():
    row1 = _row(row_id=1, root="/ws", run_id="run-1")
    rows_with_files = [
        (row1, [ChangedFile(path="a/b.py", status="M", adds=1, dels=1)]),
    ]
    entries = conversation_file_summary(rows_with_files, {})
    assert entries[0].label == "a/b.py"


def test_note_count_is_joined_from_the_mapping():
    row1 = _row(row_id=1, run_id="run-1")
    rows_with_files = [
        (row1, [ChangedFile(path="a.py", status="M", adds=1, dels=1)]),
    ]
    entries = conversation_file_summary(
        rows_with_files, {("/ws", "a.py"): 3, ("/ws", "unrelated.py"): 9}
    )
    assert entries[0].note_count == 3


def test_note_count_defaults_to_zero_when_absent():
    row1 = _row(row_id=1, run_id="run-1")
    rows_with_files = [
        (row1, [ChangedFile(path="a.py", status="M", adds=1, dels=1)]),
    ]
    entries = conversation_file_summary(rows_with_files, {})
    assert entries[0].note_count == 0


def test_ordering_is_newest_snapshot_first_then_path():
    row1 = _row(row_id=1, run_id="run-1")
    row2 = _row(row_id=2, run_id="run-2")
    rows_with_files = [
        (
            row1,
            [
                ChangedFile(path="b.py", status="A", adds=1, dels=0),
                ChangedFile(path="a.py", status="A", adds=1, dels=0),
            ],
        ),
        (row2, [ChangedFile(path="z.py", status="A", adds=1, dels=0)]),
    ]
    entries = conversation_file_summary(rows_with_files, {})
    assert [entry.path for entry in entries] == ["z.py", "a.py", "b.py"]


def test_empty_rows_yield_no_entries():
    assert conversation_file_summary([], {}) == []


# -- provider, real stack ------------------------------------------------------


def _record_turn(db, tracker, root, run_id: str, mutate) -> None:
    """One real tracked turn: baseline, mutate the tree, end, store rows."""
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    mutate()
    for rec in tracker.end_turn(handle):
        db.record_change_snapshot(
            run_id=run_id,
            root=rec.root,
            baseline_sha=rec.baseline_sha,
            end_sha=rec.end_sha,
            files_changed=rec.files_changed,
            adds=rec.adds,
            dels=rec.dels,
            tracking_error=rec.tracking_error,
            untracked_oversize=rec.untracked_oversize,
            nested_repos=rec.nested_repos,
        )


@pytest.fixture()
def provider_fixture(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "shared.txt").write_text("v1\n")

    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    conv = "conv-1"
    run1 = db.create_run(conversation_id=conv, agent_kind="primary")
    run2 = db.create_run(conversation_id=conv, agent_kind="primary")

    def turn_one():
        (root / "shared.txt").write_text("v2\n")

    def turn_two():
        (root / "shared.txt").write_text("v3\n")

    _record_turn(db, tracker, root, run1, turn_one)
    _record_turn(db, tracker, root, run2, turn_two)

    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=conv
    )
    return provider, db, service, root, run1, run2


def test_provider_two_turns_touching_one_file_use_latest_turn_identity(
    provider_fixture,
):
    provider, db, service, root, run1, run2 = provider_fixture
    entries, pruned_rows = provider.conversation_changed_files()
    assert pruned_rows == 0
    assert len(entries) == 1
    entry = entries[0]
    assert entry.path == "shared.txt"
    assert entry.status == "M"
    assert entry.run_id == run2


def test_provider_pruned_row_is_skipped_and_counted(tmp_path):
    """A row whose snapshots were pruned by retention (the shadow store
    directory reset, ``Tests/UI/test_change_review_screen.py``'s technique
    -- here scoped to ONE root so the OTHER root's row still lists)."""
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "a.txt").write_text("a1\n")
    (root_b / "b.txt").write_text("b1\n")

    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    conv = "conv-1"
    run1 = db.create_run(conversation_id=conv, agent_kind="primary")

    def mutate():
        (root_a / "a.txt").write_text("a2\n")
        (root_b / "b.txt").write_text("b2\n")

    handle = tracker.begin_turn([root_a, root_b])
    handle.await_baseline()
    mutate()
    for rec in tracker.end_turn(handle):
        db.record_change_snapshot(
            run_id=run1,
            root=rec.root,
            baseline_sha=rec.baseline_sha,
            end_sha=rec.end_sha,
            files_changed=rec.files_changed,
            adds=rec.adds,
            dels=rec.dels,
            tracking_error=rec.tracking_error,
            untracked_oversize=rec.untracked_oversize,
            nested_repos=rec.nested_repos,
        )

    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=conv
    )

    # Retention reset root_b's shadow store; root_a's remains untouched.
    pruned_repo = service.repo_for_root(root_b)
    shutil.rmtree(pruned_repo.git_dir.parent)

    entries, pruned_rows = provider.conversation_changed_files()
    assert pruned_rows == 1
    assert [entry.path for entry in entries] == ["a.txt"]
