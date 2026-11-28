"""TASK-1974: revert engine — restore-to-B with guards, against real git."""

from __future__ import annotations

import pytest

from Tests.agent_runs_owner import owned_agent_runs_db
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
    db = owned_agent_runs_db(tmp_path / "runs.db", client_id="t")
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
    outcomes = revert_paths(service, db, row, ["new.txt"], run_active=lambda: False)
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

    outcomes = revert_paths(service, db, row, ["new.txt"], run_active=lambda: False)

    assert not outcomes[0].ok and outcomes[0].error, (
        "silent false success: dir left in place but ok=True"
    )
    assert (root / "new.txt" / "keep.md").exists(), "user data must survive"


def test_uncreate_removes_an_empty_dir_squatter(turn):
    service, db, root, row = turn
    (root / "new.txt").unlink()
    (root / "new.txt").mkdir()

    outcomes = revert_paths(service, db, row, ["new.txt"], run_active=lambda: False)

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
    with pytest.raises(RevertRefusedError) as raised:
        revert_paths(service, db, row, ["edit.txt"], run_active=lambda: True)
    # TASK-42137: the run-active check is per root folder, so it says so.
    assert str(raised.value) == (
        "a run is active in this folder — finish or stop the run first"
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


def _collect_previous_owner_during_admission(root):
    """Force the observed finalizer ordering in an isolated Python process."""
    import faulthandler
    import gc
    import os
    import sys
    import threading
    import weakref

    from Tests import real_profile_guard
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission

    real_profile_guard.install()

    def profile(name):
        home = root / name
        config = home / ".config" / "tldw_cli" / "config.toml"
        config.parent.mkdir(parents=True, mode=0o700)
        os.environ.update(
            HOME=str(home), USERPROFILE=str(home), TLDW_CONFIG_PATH=str(config)
        )
        return home / "runs.db"

    first = owned_agent_runs_db(profile("first"), client_id="first")
    first.close()
    first_owner = first._execution_owner
    first_ref = weakref.ref(first)
    gc.disable()  # Choose the collection thread and boundary in this child.
    cycle = [first]
    cycle.append(cycle)
    del cycle, first

    second_path = profile("second")
    second_bootstrap = str(bootstrap.default_bootstrap_root())
    holder_entry = storage_admission._Hold._run.__code__

    def collect_at_holder_entry(frame, event, _argument):
        if event != "call" or frame.f_code is not holder_entry:
            return
        sys.setprofile(None)
        assert frame.f_locals["self"].key[1] == second_bootstrap
        assert first_ref() is not None
        print("collecting previous owner on admission thread", flush=True)
        gc.collect()
        assert first_ref() is None
        assert first_owner._stream.closed
        print("previous owner released", flush=True)

    threading.setprofile(collect_at_holder_entry)
    faulthandler.dump_traceback_later(5)
    try:
        second = owned_agent_runs_db(second_path, client_id="second")
        second_owner = second._execution_owner
        second_owner.require_current(second_path)
        second.close()
        del second
        assert second_owner._stream.closed
    finally:
        faulthandler.cancel_dump_traceback_later()
        threading.setprofile(None)
        gc.enable()
    _check_reopen_during_retirement(second_path)


def _check_reopen_during_retirement(path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from unittest.mock import patch

    from Tests import agent_runs_owner

    handles = [owned_agent_runs_db(path)]
    old_owner = handles[0]._execution_owner
    handles[0].close()
    closing, allow_close, waiting = Event(), Event(), Event()
    close = old_owner.close
    wait = agent_runs_owner._changed.wait

    def observe_close():
        closing.set()
        assert allow_close.wait(5)
        close()

    def observe_wait(timeout=None):
        waiting.set()
        return wait(timeout)

    def open_peer():
        db = owned_agent_runs_db(path)
        db.close()
        return db

    with (
        patch.object(old_owner, "close", observe_close),
        patch.object(agent_runs_owner._changed, "wait", observe_wait),
        ThreadPoolExecutor(max_workers=2) as threads,
    ):
        retirement = threads.submit(handles.clear)
        try:
            assert closing.wait(5)
            reopening = threads.submit(open_peer)
            assert waiting.wait(5)
            assert not reopening.done() and not old_owner._stream.closed
        finally:
            allow_close.set()
        retirement.result(timeout=5)
        fresh = reopening.result(timeout=5)
        fresh_owner = fresh._execution_owner
        assert old_owner._stream.closed
        assert fresh_owner is not old_owner
        fresh_owner.require_current(path)
        del reopening, fresh
        assert fresh_owner._stream.closed
    print("same-path reopen waited for owner retirement", flush=True)


def test_owner_finalizer_cannot_block_new_native_admission(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    entry = (
        "from pathlib import Path; import sys; "
        "from Tests.Workspaces.test_change_revert import "
        "_collect_previous_owner_during_admission as probe; "
        "probe(Path(sys.argv[1]))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", entry, str(tmp_path)],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (error.stdout or b"").decode(errors="replace")
        stderr = (error.stderr or b"").decode(errors="replace")
        pytest.fail(stdout + stderr, pytrace=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        "collecting previous owner on admission thread",
        "previous owner released",
        "same-path reopen waited for owner retirement",
    ]
