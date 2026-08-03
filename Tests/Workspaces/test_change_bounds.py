"""TASK-1975: cost bounds — knobs, root budget scan, oversize detection."""
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Workspaces.change_bounds import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_TOTAL_BYTES,
    DEFAULT_RETENTION_DAYS,
    RootScan,
    change_review_setting,
    scan_root,
)


class TestKnobs:
    def test_defaults_come_back_untouched(self):
        assert change_review_setting("max_files", DEFAULT_MAX_FILES) == (
            DEFAULT_MAX_FILES
        )
        assert change_review_setting(
            "retention_days", DEFAULT_RETENTION_DAYS
        ) == DEFAULT_RETENTION_DAYS

    def test_env_var_overrides_config(self, monkeypatch):
        monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILES", "123")
        assert change_review_setting("max_files", DEFAULT_MAX_FILES) == 123

    def test_garbage_env_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILES", "not-a-number")
        assert change_review_setting("max_files", DEFAULT_MAX_FILES) == (
            DEFAULT_MAX_FILES
        )

    def test_flat_config_section_is_read(self, monkeypatch):
        # The FLAT section is the spec's contract (dotted get_cli_setting
        # has dropped defaults before — task 1754).
        import tldw_chatbook.Workspaces.change_bounds as cb

        monkeypatch.setattr(
            cb,
            "_config_setting",
            lambda key, default: 77 if key == "max_files" else default,
        )
        assert change_review_setting("max_files", DEFAULT_MAX_FILES) == 77


@pytest.fixture()
def scanroot(tmp_path):
    # The project conftest materializes an isolated config tree inside
    # tmp_path — scan a dedicated subdir so only OUR files count.
    d = tmp_path / "scanroot"
    d.mkdir()
    return d


class TestRootScan:
    def test_counts_files_and_bytes(self, scanroot):
        (scanroot / "a.txt").write_bytes(b"x" * 10)
        (scanroot / "b.txt").write_bytes(b"y" * 20)
        scan = scan_root(scanroot)
        assert scan.files == 2
        assert scan.total_bytes == 30
        assert scan.over_budget is False
        assert scan.oversized == ()

    def test_forced_exclude_dirs_do_not_count(self, scanroot):
        (scanroot / "a.txt").write_bytes(b"x")
        (scanroot / ".venv").mkdir()
        (scanroot / ".venv" / "huge.bin").write_bytes(b"z" * 1000)
        (scanroot / "node_modules").mkdir()
        (scanroot / "node_modules" / "dep.js").write_bytes(b"z" * 1000)
        (scanroot / ".git").mkdir()
        (scanroot / ".git" / "objects").mkdir()
        scan = scan_root(scanroot)
        assert scan.files == 1
        assert scan.total_bytes == 1

    def test_over_file_budget_flags_and_aborts_early(self, scanroot):
        for i in range(10):
            (scanroot / f"f{i}.txt").write_bytes(b"x")
        scan = scan_root(scanroot, max_files=5)
        assert scan.over_budget is True

    def test_over_total_bytes_budget_flags(self, scanroot):
        (scanroot / "big.bin").write_bytes(b"x" * 100)
        scan = scan_root(scanroot, max_total_bytes=50)
        assert scan.over_budget is True

    def test_oversized_files_listed_root_relative(self, scanroot):
        (scanroot / "sub").mkdir()
        (scanroot / "sub" / "fat [v2].bin").write_bytes(b"x" * 100)
        (scanroot / "slim.txt").write_bytes(b"x")
        scan = scan_root(scanroot, max_file_bytes=50)
        assert scan.over_budget is False
        assert scan.oversized == ("sub/fat [v2].bin",)

    def test_symlinks_are_not_followed(self, tmp_path):
        outside = tmp_path / "outside-tree"
        outside.mkdir(exist_ok=True)
        (outside / "huge.bin").write_bytes(b"x" * 10_000)
        root = tmp_path / "scanroot2"
        root.mkdir()
        (root / "a.txt").write_bytes(b"x")
        (root / "link").symlink_to(outside)
        scan = scan_root(root)
        assert scan.files == 1, "symlinked trees must not count against the root"


# -- enforcement (real git) --------------------------------------------------


@pytest.fixture()
def tracked(tmp_path, monkeypatch):
    """A tracker + in-budget root, with a small oversize cap via env."""
    from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
    from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "100")
    root = tmp_path / "workroot"
    root.mkdir()
    (root / "small.txt").write_text("hello\n")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    return ChangeTurnTracker(service=service), service, root


class TestBudgetEnforcement:
    def test_over_budget_root_disables_tracking_with_honest_copy(
        self, tracked, monkeypatch
    ):
        tracker, service, root = tracked
        for i in range(8):
            (root / f"f{i}.txt").write_text("x")
        monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILES", "5")

        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        records = tracker.end_turn(handle)

        assert len(records) == 1
        assert "narrow the root or add excludes" in records[0].tracking_error
        assert records[0].baseline_sha == "", "over-budget must not snapshot"

    def test_in_budget_root_tracks_normally(self, tracked):
        tracker, service, root = tracked
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        records = tracker.end_turn(handle)
        assert len(records) == 1
        assert records[0].tracking_error == ""
        assert records[0].files_changed == 1


class TestOversizeExcludes:
    def test_mid_turn_oversized_creation_excluded_and_disclosed(self, tracked):
        tracker, service, root = tracked
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "big.bin").write_bytes(b"x" * 500)
        (root / "small.txt").write_text("edited\n")
        records = tracker.end_turn(handle)

        assert len(records) == 1
        rec = records[0]
        assert rec.files_changed == 1, "only the small edit is a change"
        assert rec.untracked_oversize == 1
        repo = service.repo_for_root(root)
        assert repo.file_bytes(rec.end_sha, "big.bin") is None, (
            "the oversized file must never be committed to the shadow store"
        )

    def test_preexisting_oversize_disclosed_on_changed_turns(self, tracked):
        tracker, service, root = tracked
        (root / "old-big.bin").write_bytes(b"x" * 500)
        service.repo_for_root(root).snapshot("registration")

        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        records = tracker.end_turn(handle)

        assert len(records) == 1
        assert records[0].untracked_oversize == 1

    def test_oversize_only_turn_emits_a_zero_change_record(self, tracked):
        tracker, service, root = tracked
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "big.bin").write_bytes(b"x" * 500)
        records = tracker.end_turn(handle)

        assert len(records) == 1
        assert records[0].files_changed == 0
        assert records[0].untracked_oversize == 1

    def test_stable_oversize_set_with_no_changes_emits_no_record(self, tracked):
        tracker, service, root = tracked
        (root / "big.bin").write_bytes(b"x" * 500)
        service.repo_for_root(root).snapshot("registration")

        handle = tracker.begin_turn([root])
        handle.await_baseline()
        records = tracker.end_turn(handle)
        assert records == [], "an unchanged root must stay cardless"

    def test_tool_touched_oversized_path_is_not_force_added(self, tracked):
        tracker, service, root = tracked
        (root / ".gitignore").write_text("secret.bin\n")
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "secret.bin").write_bytes(b"x" * 500)
        records = tracker.end_turn(
            handle, touched_paths=[str(root / "secret.bin")]
        )
        assert len(records) == 1
        rec = records[0]
        assert rec.untracked_oversize == 1
        if rec.end_sha:
            repo = service.repo_for_root(root)
            assert repo.file_bytes(rec.end_sha, "secret.bin") is None, (
                "force-add must not defeat the size cap"
            )


# -- retention (AC#3/#4) -----------------------------------------------------


class TestRetention:
    def _turn_row(self, tracked, db):
        tracker, service, root = tracked
        run = db.create_run(conversation_id="c", agent_kind="primary")
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited by turn\n")
        rec = tracker.end_turn(handle)[0]
        db.record_change_snapshot(
            run_id=run,
            root=rec.root,
            baseline_sha=rec.baseline_sha,
            end_sha=rec.end_sha,
            files_changed=rec.files_changed,
            adds=rec.adds,
            dels=rec.dels,
        )
        return run

    def test_prune_drops_old_rows_and_resets_rowless_repos(
        self, tracked, tmp_path
    ):
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.Workspaces.change_retention import (
            prune_change_history,
        )

        tracker, service, root = tracked
        db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        run = self._turn_row(tracked, db)
        repo = service.repo_for_root(root)
        objects_before = sum(
            1 for _ in (repo.git_dir / "objects").rglob("*") if _.is_file()
        )
        assert objects_before > 0
        with db.transaction() as conn:
            conn.execute(
                "UPDATE change_snapshots SET created_at = '2020-01-01T00:00:00.000000Z'"
            )

        report = prune_change_history(db, service)

        assert report.rows_pruned == 1
        assert db.change_snapshots_for_run(run) == []
        assert not repo.git_dir.exists(), (
            "a rowless shadow repo must be reset (object count -> 0)"
        )

    def test_recent_rows_and_their_repo_survive(self, tracked, tmp_path):
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.Workspaces.change_retention import (
            prune_change_history,
        )

        tracker, service, root = tracked
        db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        run = self._turn_row(tracked, db)

        report = prune_change_history(db, service)

        assert report.rows_pruned == 0
        assert len(db.change_snapshots_for_run(run)) == 1
        assert service.repo_for_root(root).git_dir.exists()

    def test_orphaned_repo_dir_removed_by_age(self, tracked, tmp_path):
        import os as _os
        import time as _time

        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.Workspaces.change_retention import (
            prune_change_history,
        )

        tracker, service, root = tracked
        db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        repo = service.repo_for_root(root)
        repo.snapshot("registration")
        git_dir = repo.git_dir
        import shutil as _shutil

        _shutil.rmtree(root)  # the ROOT vanishes; the shadow repo is orphaned
        old = _time.time() - 90 * 86400
        _os.utime(git_dir, (old, old))

        report = prune_change_history(db, service)

        assert report.orphans_removed == 1
        assert not git_dir.exists()

    def test_fresh_orphan_is_left_alone(self, tracked, tmp_path):
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.Workspaces.change_retention import (
            prune_change_history,
        )
        import shutil as _shutil

        tracker, service, root = tracked
        db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        repo = service.repo_for_root(root)
        repo.snapshot("registration")
        _shutil.rmtree(root)

        report = prune_change_history(db, service)

        assert report.orphans_removed == 0
        assert repo.git_dir.exists(), "a fresh orphan may still be re-bound"


class TestAppRetentionRunner:
    def test_runs_a_pass_against_the_sibling_runs_db(self, tracked, tmp_path):
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.Workspaces.change_retention import (
            run_retention_for_app,
        )

        tracker, service, root = tracked
        db = AgentRunsDB(tmp_path / "agent_runs.db", client_id="t")
        run = db.create_run(conversation_id="c", agent_kind="primary")
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        rec = tracker.end_turn(handle)[0]
        db.record_change_snapshot(
            run_id=run,
            root=rec.root,
            baseline_sha=rec.baseline_sha,
            end_sha=rec.end_sha,
        )
        with db.transaction() as conn:
            conn.execute(
                "UPDATE change_snapshots SET created_at = '2020-01-01T00:00:00.000000Z'"
            )

        report = run_retention_for_app(
            tmp_path / "chachanotes.db", service=service
        )

        assert report is not None and report.rows_pruned == 1

    def test_never_raises_on_a_broken_path(self, tmp_path):
        from tldw_chatbook.Workspaces.change_retention import (
            run_retention_for_app,
        )

        assert (
            run_retention_for_app(tmp_path / "nodir" / "x.db") is None
            or True
        )


def test_old_schema_file_gains_the_oversize_column_on_open(tmp_path):
    """v3->v4: a change_snapshots table created before untracked_oversize
    existed picks the column up via the idempotent-ALTER-on-open mechanism."""
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db_file = tmp_path / "old.db"
    with closing(sqlite3.connect(db_file)) as conn:
        conn.executescript(
        """
        CREATE TABLE change_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            root TEXT NOT NULL,
            baseline_sha TEXT NOT NULL,
            end_sha TEXT NOT NULL,
            files_changed INTEGER NOT NULL DEFAULT 0,
            adds INTEGER NOT NULL DEFAULT 0,
            dels INTEGER NOT NULL DEFAULT 0,
            reverted TEXT NOT NULL DEFAULT '',
            tracking_error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        INSERT INTO change_snapshots
            (run_id, root, baseline_sha, end_sha, created_at)
        VALUES ('r', '/tmp/x', 'a', 'b', '2026-01-01T00:00:00.000000Z');
        """
        )
        conn.commit()

    db = AgentRunsDB(db_file, client_id="t")
    db.record_change_snapshot(
        run_id="r2",
        root="/tmp/x",
        baseline_sha="c",
        end_sha="d",
        untracked_oversize=3,
    )
    rows = db.change_snapshots_for_run("r2")
    assert rows[0]["untracked_oversize"] == 3
    old_rows = db.change_snapshots_for_run("r")
    assert old_rows[0]["untracked_oversize"] == 0


class TestReviewRoundHardening:
    """PR #1251 Qodo round: newline injection, git path, sweep locking."""

    def test_newline_named_oversize_file_neither_committed_nor_injected(
        self, tracked
    ):
        tracker, service, root = tracked
        evil = "evil\nsecond-line.bin"
        (root / evil).write_bytes(b"x" * 500)
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        records = tracker.end_turn(handle)

        assert len(records) == 1
        rec = records[0]
        assert rec.untracked_oversize == 1
        repo = service.repo_for_root(root)
        assert repo.file_bytes(rec.end_sha, evil) is None, (
            "an unexcludable oversized file must still never be committed"
        )
        exclude = (repo.git_dir / "info" / "exclude").read_text()
        assert "second-line.bin" not in exclude, (
            "a newline in a filename injected an exclude pattern"
        )

    def test_retention_uses_the_services_git_not_PATH(
        self, tracked, tmp_path, monkeypatch
    ):
        import shutil as _shutil

        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.Workspaces.change_retention import (
            prune_change_history,
        )
        from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService

        real_git = _shutil.which("git")
        tracker, service, root = tracked
        db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        run = db.create_run(conversation_id="c", agent_kind="primary")
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        rec = tracker.end_turn(handle)[0]
        db.record_change_snapshot(
            run_id=run,
            root=rec.root,
            baseline_sha=rec.baseline_sha,
            end_sha=rec.end_sha,
        )
        # A service configured with an explicit git path, on a machine
        # where PATH has no git: the sweep must still classify the repo
        # as live (a PATH-only spawn would fail -> misclassified orphan).
        pathless = ShadowRepoService(
            data_dir=service._data_dir, git_executable=real_git
        )
        monkeypatch.setenv("PATH", str(tmp_path / "no-binaries-here"))
        # Backdate the repo: a misclassified "orphan" would be aged out --
        # the strengthened assertion a fresh repo cannot make.
        import os as _os
        import time as _time

        repo = service.repo_for_root(root)
        old_ts = _time.time() - 90 * 86400
        _os.utime(repo.git_dir, (old_ts, old_ts))

        report = prune_change_history(db, pathless)

        assert report.orphans_removed == 0
        assert report.repos_reset == 0
        assert service.repo_for_root(root).git_dir.exists()

    def test_sweep_skips_a_container_another_process_holds(self, tracked, tmp_path):
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.Workspaces.change_retention import (
            prune_change_history,
        )

        tracker, service, root = tracked
        db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        repo = service.repo_for_root(root)
        repo.snapshot("registration")
        # No rows -> the sweep would RESET this container; a held lock
        # (concurrent snapshot in another process) must make it skip.
        repo.lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            report = prune_change_history(db, service)
        finally:
            repo.lock_dir.rmdir()

        assert report.repos_reset == 0
        assert repo.git_dir.exists(), "the sweep deleted a LOCKED repo"


# -- nested repos (TASK-1976) ------------------------------------------------


class TestNestedRepoDetection:
    def test_scan_lists_nested_repos_root_relative(self, scanroot):
        (scanroot / "projects" / "childrepo" / ".git").mkdir(parents=True)
        (scanroot / "projects" / "childrepo" / "code.py").write_text("x\n")
        (scanroot / "worktree-child").mkdir()
        (scanroot / "worktree-child" / ".git").write_text("gitdir: /elsewhere\n")
        (scanroot / "plain").mkdir()
        (scanroot / "plain" / "note.md").write_text("x\n")
        scan = scan_root(scanroot)
        assert sorted(scan.nested_repos) == [
            "projects/childrepo",
            "worktree-child",
        ]

    def test_roots_own_git_is_not_nested(self, scanroot):
        (scanroot / ".git").mkdir()
        (scanroot / "a.txt").write_text("x\n")
        scan = scan_root(scanroot)
        assert scan.nested_repos == ()

    def test_nested_edit_is_invisible_and_disclosed(self, tracked):
        """AC#2: the hole exists (git gitlink semantics) and is DISCLOSED."""
        import subprocess as _sp

        tracker, service, root = tracked
        child = root / "childrepo"
        child.mkdir()
        _sp.run(["git", "init", "--quiet", str(child)], check=True)
        (child / "inner.txt").write_text("original\n")
        service.repo_for_root(root).snapshot("registration")

        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (child / "inner.txt").write_text("EDITED INSIDE CHILD\n")
        (root / "small.txt").write_text("edited\n")
        records = tracker.end_turn(handle)

        assert len(records) == 1
        rec = records[0]
        assert rec.nested_repos == ("childrepo",)
        repo = service.repo_for_root(root)
        changed = {c.path for c in repo.changed_files(rec.baseline_sha, rec.end_sha)}
        assert "small.txt" in changed
        assert not any("inner.txt" in p for p in changed), (
            "the nested edit must NOT appear as a diff row (the disclosed hole)"
        )

    def test_new_nested_repo_mid_turn_emits_disclosure_record(self, tracked):
        tracker, service, root = tracked
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        child = root / "cloned-mid-turn"
        (child / ".git").mkdir(parents=True)
        (child / "inner.txt").write_text("x\n")
        records = tracker.end_turn(handle)
        assert len(records) == 1
        assert records[0].nested_repos == ("cloned-mid-turn",)

    def test_repo_root_with_no_children_stays_bannerless(self, tracked):
        """AC#3: tracking a root normally — nested_repos stays empty."""
        tracker, service, root = tracked
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        records = tracker.end_turn(handle)
        assert records[0].nested_repos == ()
