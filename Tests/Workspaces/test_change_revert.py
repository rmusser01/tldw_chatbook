"""TASK-1974: revert engine — restore-to-B with guards, against real git."""
from __future__ import annotations

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Workspaces.change_revert import (
    RevertRefusedError,
    preflight_revert,
    revert_paths,
)
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker


@pytest.fixture()
def turn(tmp_path):
    """One real tracked turn covering create/modify/delete/rename."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "edit.txt").write_text("before\n")
    (root / "gone.txt").write_text("delete me\n")
    (root / "old_name.txt").write_text("stable rename content\n" * 5)
    (root / "sub").mkdir()
    (root / "sub" / "nested.txt").write_text("nested\n")

    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run = db.create_run(conversation_id="c", agent_kind="primary")

    handle = tracker.begin_turn([root])
    handle.await_baseline()
    (root / "new.txt").write_text("created\n")
    (root / "edit.txt").write_text("after\n")
    (root / "gone.txt").unlink()
    (root / "old_name.txt").rename(root / "new_name.txt")
    (root / "sub" / "nested.txt").unlink()
    records = tracker.end_turn(handle)
    assert len(records) == 1
    rec = records[0]
    db.record_change_snapshot(
        run_id=run,
        root=rec.root,
        baseline_sha=rec.baseline_sha,
        end_sha=rec.end_sha,
        files_changed=rec.files_changed,
        adds=rec.adds,
        dels=rec.dels,
    )
    row = db.change_snapshots_for_run(run)[0]
    return service, db, root, row


def test_each_change_kind_round_trips_back_to_baseline(turn):
    service, db, root, row = turn
    repo = service.repo_for_root(root)
    changed = repo.changed_files(row["baseline_sha"], row["end_sha"])

    outcomes = revert_paths(
        service, db, row, [c.path for c in changed], run_active=lambda: False
    )

    assert all(o.ok for o in outcomes), outcomes
    assert (root / "edit.txt").read_text() == "before\n"
    assert (root / "gone.txt").read_text() == "delete me\n"
    assert (root / "old_name.txt").exists()
    assert not (root / "new_name.txt").exists()
    assert not (root / "new.txt").exists(), "un-create did not remove the file"


def test_uncreate_is_a_guarded_delete_not_a_checkout(turn):
    """`checkout B -- path` errors on a B-absent path; un-create must be an
    explicit delete that only fires when the path is genuinely absent at B."""
    service, db, root, row = turn
    outcomes = revert_paths(
        service, db, row, ["new.txt"], run_active=lambda: False
    )
    assert outcomes[0].ok and not (root / "new.txt").exists()


def test_preflight_names_files_the_user_edited_after_the_turn(turn):
    service, db, root, row = turn
    (root / "edit.txt").write_text("USER EDIT after the turn\n")

    report = preflight_revert(service, row, ["edit.txt", "new.txt"])

    assert report.edited_since == ["edit.txt"], (
        "the confirm dialog cannot name what it does not know"
    )
    assert report.disk_state["edit.txt"].startswith("sha256:")
    assert report.disk_state["new.txt"].startswith("sha256:")


def test_preflight_names_the_old_path_of_a_rename(turn):
    """Reverting a rename ALSO restores old_path from B; if the user put a
    new file at the old name after the turn, the confirm dialog must name
    it — the requested (new) path alone is not the full overwrite set."""
    service, db, root, row = turn
    (root / "old_name.txt").write_text("USER'S NEW FILE at the old name\n")

    report = preflight_revert(service, row, ["new_name.txt"])

    assert "old_name.txt" in report.edited_since, (
        "rename revert will overwrite old_name.txt unwarned"
    )


def test_preflight_rename_old_path_untouched_is_not_flagged(turn):
    """Disk state at old_path matches E (absent) — nothing to warn about."""
    service, db, root, row = turn
    report = preflight_revert(service, row, ["new_name.txt"])
    assert report.edited_since == []


def test_uncreate_reports_failure_when_a_nonempty_dir_squats_the_path(turn):
    """A created-then-reverted path now holding a non-empty directory must
    surface as a per-path FAILURE — not silent success with the directory
    left in place (and never an rmtree of the user's data)."""
    service, db, root, row = turn
    (root / "new.txt").unlink()
    (root / "new.txt").mkdir()
    (root / "new.txt" / "keep.md").write_text("user data\n")

    outcomes = revert_paths(
        service, db, row, ["new.txt"], run_active=lambda: False
    )

    assert not outcomes[0].ok and outcomes[0].error, (
        "silent false success: dir left in place but ok=True"
    )
    assert (root / "new.txt" / "keep.md").exists(), "user data must survive"


def test_uncreate_removes_an_empty_dir_squatter(turn):
    service, db, root, row = turn
    (root / "new.txt").unlink()
    (root / "new.txt").mkdir()

    outcomes = revert_paths(
        service, db, row, ["new.txt"], run_active=lambda: False
    )

    assert outcomes[0].ok and not (root / "new.txt").exists()


def test_traversal_paths_are_refused_lexically(turn):
    """Defense-in-depth: git-relative paths never contain '..' or start
    absolute, so any such request is refused before ANY disk operation —
    in revert (per-path failure) and in preflight (skipped, since a path
    that will not be reverted cannot overwrite anything)."""
    service, db, root, row = turn
    outside = root.parent / "outside.txt"
    outside.write_text("do not touch\n")

    outcomes = revert_paths(
        service,
        db,
        row,
        ["../outside.txt", str(outside)],
        run_active=lambda: False,
    )
    assert all(not o.ok for o in outcomes)
    assert all("refused" in o.error for o in outcomes)
    assert outside.read_text() == "do not touch\n"

    report = preflight_revert(service, row, ["../outside.txt"])
    assert "../outside.txt" not in report.edited_since


def test_revert_refuses_while_a_run_is_active(turn):
    service, db, root, row = turn
    with pytest.raises(RevertRefusedError, match="run"):
        revert_paths(
            service, db, row, ["edit.txt"], run_active=lambda: True
        )
    assert (root / "edit.txt").read_text() == "after\n", (
        "the refusal must happen BEFORE any file is touched"
    )


def test_revert_takes_a_fresh_snapshot_and_updates_the_row(turn):
    service, db, root, row = turn
    repo = service.repo_for_root(root)
    tip_before = repo.tip()

    revert_paths(service, db, row, ["edit.txt"], run_active=lambda: False)

    assert repo.tip() != tip_before, "history must stay true after a revert"
    fresh = db.change_snapshots_for_run(row["run_id"])[0]
    assert "edit.txt" in fresh["reverted"]


def test_one_failing_path_reports_itself_and_the_rest_complete(turn):
    """BOTH failure shapes: the no-exception branch (unknown path) AND a
    real EXCEPTION mid-loop -- a directory squatting where the restore must
    write makes checkout raise. The first version only exercised the
    former, and a first-failure-aborts sabotage survived it.
    """
    service, db, root, row = turn
    # sub/nested.txt was deleted in the turn; a READ-ONLY parent dir makes
    # the restore raise. Structural squatters do NOT work: git force-
    # clobbers a non-empty directory at a file path AND tunnels through a
    # file at a directory path (both verified empirically while writing
    # this) -- permissions are the one wall it cannot pass as non-root.
    # Ordered FIRST so a raise-through would strand edit.txt.
    import os as _os

    if _os.name == "nt":
        pytest.skip("POSIX permission injection")
    _os.chmod(root / "sub", 0o500)
    request_cleanup = lambda: _os.chmod(root / "sub", 0o755)  # noqa: E731

    outcomes = revert_paths(
        service,
        db,
        row,
        ["sub/nested.txt", "not-in-this-turn.txt", "edit.txt"],
        run_active=lambda: False,
    )
    request_cleanup()
    by_path = {o.path: o for o in outcomes}
    assert not by_path["sub/nested.txt"].ok and by_path["sub/nested.txt"].error
    assert not by_path["not-in-this-turn.txt"].ok
    assert by_path["edit.txt"].ok, "an earlier exception stranded later paths"
    assert (root / "edit.txt").read_text() == "before\n"
