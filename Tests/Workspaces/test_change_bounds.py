"""TASK-1975: cost bounds — knobs, root budget scan, oversize detection."""
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Workspaces.change_bounds import (
    DEFAULT_MAX_FILES,
    DEFAULT_RETENTION_DAYS,
    change_review_setting,
    scan_root,
)


def test_missing_global_setting_keeps_capability_available(monkeypatch):
    """A missing master switch preserves capability, not workspace consent."""
    import tldw_chatbook.Workspaces.change_bounds as change_bounds

    monkeypatch.delenv("TLDW_CHANGE_REVIEW_ENABLED", raising=False)
    monkeypatch.setattr(
        change_bounds,
        "_change_review_enabled_setting",
        lambda: True,
        raising=False,
    )

    result = change_bounds.read_change_review_capability()

    assert result.state.value == "enabled"
    assert change_bounds.change_review_enabled_globally() is True


@pytest.mark.parametrize("raw", ["maybe", "", object()])
def test_invalid_global_setting_is_unavailable(monkeypatch, raw):
    """Unreadable/coercion-failed capability state must fail tracking off."""
    import tldw_chatbook.Workspaces.change_bounds as change_bounds

    monkeypatch.delenv("TLDW_CHANGE_REVIEW_ENABLED", raising=False)
    monkeypatch.setattr(
        change_bounds,
        "_change_review_enabled_setting",
        lambda: raw,
        raising=False,
    )

    result = change_bounds.read_change_review_capability()

    assert result.state.value == "unavailable"
    assert change_bounds.change_review_enabled_globally() is False


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

    def test_nested_edit_never_pollutes_the_parents_diff(self, tracked):
        """1976 AC#2's surviving core under TASK-1977: the PARENT's diff
        never carries nested content — the edit is tracked via the child's
        own sub-root instead of being disclosed as a hole."""
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

        by_root = {r.root: r for r in records}
        parent = by_root[str(root.resolve())]
        assert parent.nested_repos == (), "tracked sub-root wrongly disclosed"
        repo = service.repo_for_root(root)
        changed = {
            c.path
            for c in repo.changed_files(parent.baseline_sha, parent.end_sha)
        }
        assert "small.txt" in changed
        assert not any("inner.txt" in p for p in changed), (
            "nested content leaked into the PARENT's diff"
        )
        assert str(child.resolve()) in by_root, "the hole did not close"

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


class TestNestedRepoBudgetIsolation:
    """PR #1254 Qodo round: nested content must not distort budgets."""

    def test_nested_content_does_not_count_against_the_budget(self, scanroot):
        (scanroot / "a.txt").write_bytes(b"x")
        child = scanroot / "bigchild"
        (child / ".git").mkdir(parents=True)
        for i in range(50):
            (child / f"f{i}.bin").write_bytes(b"z" * 100)
        scan = scan_root(scanroot, max_files=10, max_total_bytes=1000)
        assert scan.over_budget is False, (
            "a large NESTED repo tripped the whole root's budget"
        )
        assert scan.files == 1
        assert scan.nested_repos == ("bigchild",)

    def test_oversized_inside_nested_is_not_disclosed(self, scanroot):
        (scanroot / "a.txt").write_bytes(b"x")
        child = scanroot / "child"
        (child / ".git").mkdir(parents=True)
        (child / "fat.bin").write_bytes(b"z" * 500)
        scan = scan_root(scanroot, max_file_bytes=100)
        assert scan.oversized == (), (
            "never-trackable nested files polluted the oversize disclosure"
        )

    def test_newline_named_commitless_child_does_not_kill_tracking(
        self, tracked
    ):
        import subprocess as _sp

        tracker, service, root = tracked
        evil = root / "evil\nrepo"
        evil.mkdir()
        _sp.run(["git", "init", "--quiet", str(evil)], check=True)
        (evil / "inner.txt").write_text("x\n")

        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        (evil / "inner.txt").write_text("EDITED\n")
        records = tracker.end_turn(handle)

        by_root = {r.root: r for r in records}
        parent = by_root[str(root.resolve())]
        assert parent.tracking_error == "", (
            "an unexcludable commitless child killed tracking for the root"
        )
        repo = service.repo_for_root(root)
        changed = {
            c.path
            for c in repo.changed_files(parent.baseline_sha, parent.end_sha)
        }
        assert "small.txt" in changed
        assert not any("inner.txt" in p for p in changed)
        # Under TASK-1977 the child is auto-registered (tracked), so it is
        # neither disclosed as a hole nor able to hurt the parent.
        assert str((root / "evil\nrepo").resolve()) in by_root


# -- auto sub-roots (TASK-1977) ----------------------------------------------


def _real_child(root, name: str):
    """A real nested git repo with one committed file."""
    import subprocess as _sp

    child = root / name
    child.mkdir()
    _sp.run(["git", "init", "--quiet", str(child)], check=True)
    (child / "inner.txt").write_text("original\n")
    _sp.run(
        ["git", "-C", str(child), "-c", "user.email=t@t", "-c", "user.name=t",
         "add", "-A"],
        check=True, capture_output=True,
    )
    _sp.run(
        ["git", "-C", str(child), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "--quiet", "-m", "init"],
        check=True, capture_output=True,
    )
    return child


class TestAutoSubRoots:
    def test_nested_edit_appears_attributed_to_the_sub_root(self, tracked):
        """AC#1: the 1976 hole closes — the child's edit is reviewable."""
        tracker, service, root = tracked
        child = _real_child(root, "childrepo")

        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (child / "inner.txt").write_text("EDITED\n")
        (root / "small.txt").write_text("edited\n")
        records = tracker.end_turn(handle)

        by_root = {r.root: r for r in records}
        child_key = str(child.resolve())
        assert child_key in by_root, f"no sub-root record: {list(by_root)}"
        child_rec = by_root[child_key]
        assert child_rec.files_changed == 1
        repo = service.repo_for_root(child)
        changed = {
            c.path
            for c in repo.changed_files(
                child_rec.baseline_sha, child_rec.end_sha
            )
        }
        assert changed == {"inner.txt"}

    def test_registered_child_leaves_the_disclosure_banner(self, tracked):
        tracker, service, root = tracked
        _real_child(root, "childrepo")

        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        records = tracker.end_turn(handle)

        parent = next(r for r in records if r.root == str(root.resolve()))
        assert parent.nested_repos == (), (
            "a TRACKED sub-root must not be disclosed as an untracked hole"
        )

    def test_sub_root_count_bound_truncates_with_disclosure(
        self, tracked, monkeypatch
    ):
        """AC#3: beyond max_sub_roots, children stay DISCLOSED untracked."""
        monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_SUB_ROOTS", "1")
        tracker, service, root = tracked
        _real_child(root, "alpha")
        _real_child(root, "beta")

        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (root / "small.txt").write_text("edited\n")
        (root / "alpha" / "inner.txt").write_text("EDITED\n")
        (root / "beta" / "inner.txt").write_text("EDITED\n")
        records = tracker.end_turn(handle)

        parent = next(r for r in records if r.root == str(root.resolve()))
        registered = {
            Path(r.root).name
            for r in records
            if r.root != str(root.resolve())
        }
        assert len(registered) == 1, (
            "exactly one child is within the bound; only ITS edit is "
            f"reviewable — got {registered}"
        )
        disclosed = set(parent.nested_repos)
        assert len(disclosed) == 1, "the beyond-bound child must stay disclosed"
        assert registered | disclosed == {"alpha", "beta"}
        assert registered.isdisjoint(disclosed), (
            "a child is both registered and disclosed"
        )

    def test_deleted_child_unregisters_via_orphan_gc(self, tracked, tmp_path):
        """AC#4: the existing orphan GC covers sub-root shadow repos."""
        import os as _os
        import shutil as _shutil
        import time as _time

        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.Workspaces.change_retention import (
            prune_change_history,
        )

        tracker, service, root = tracked
        child = _real_child(root, "childrepo")
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        (child / "inner.txt").write_text("EDITED\n")
        tracker.end_turn(handle)
        child_repo = service.repo_for_root(child)
        container = child_repo.git_dir.parent
        assert child_repo.git_dir.exists()

        _shutil.rmtree(child)
        old = _time.time() - 90 * 86400
        _os.utime(container, (old, old))
        _os.utime(child_repo.git_dir, (old, old))
        db = AgentRunsDB(tmp_path / "runs.db", client_id="t")

        report = prune_change_history(db, service)

        assert report.orphans_removed >= 1
        assert not child_repo.git_dir.exists()


def test_end_turn_survives_a_still_running_discovery_thread(tracked):
    """Qodo #1256: a timed-out baseline thread may still be appending
    sub-roots while end_turn runs — iteration must be over a snapshot,
    yielding one record per root, never churn-driven duplicates."""
    import threading
    import time

    tracker, service, root = tracked
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    (root / "small.txt").write_text("edited\n")

    stop = threading.Event()

    def churn() -> None:
        while not stop.is_set():
            handle.roots.append(root)
            time.sleep(0.0005)

    thread = threading.Thread(target=churn, daemon=True)
    thread.start()
    try:
        records = tracker.end_turn(handle)
    finally:
        stop.set()
        thread.join(timeout=2)

    mine = [r for r in records if r.root == str(root.resolve())]
    assert len(mine) == 1, (
        f"churned roots produced {len(mine)} records for one root"
    )


# -- settings + gating (TASK-1979) -------------------------------------------


class TestChangeReviewGating:
    def _registry(self, tmp_path):
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
        from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

        return LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.db", client_id="t")
        )

    def _bound_workspace(self, registry, tmp_path):
        root = tmp_path / "wsroot"
        root.mkdir()
        registry.create_workspace(workspace_id="ws-1", name="WS")
        registry.add_folder_binding("ws-1", root)
        return root

    def test_global_kill_knob_empties_the_root_list(
        self, tmp_path, monkeypatch
    ):
        import tldw_chatbook.Tools.workspace_file_roots as wfr

        registry = self._registry(tmp_path)
        root = self._bound_workspace(registry, tmp_path)
        registry.set_change_review_enabled("ws-1", True)
        monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)

        assert wfr.folder_binding_roots("ws-1") == (root.resolve(),)
        monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "0")
        assert wfr.folder_binding_roots("ws-1") == (), (
            "the global knob must stop tracking on the NEXT read"
        )

    def test_workspace_toggle_gates_only_that_workspace(
        self, tmp_path, monkeypatch
    ):
        import tldw_chatbook.Tools.workspace_file_roots as wfr

        registry = self._registry(tmp_path)
        self._bound_workspace(registry, tmp_path)
        other_root = tmp_path / "other"
        other_root.mkdir()
        registry.create_workspace(workspace_id="ws-2", name="Other")
        registry.add_folder_binding("ws-2", other_root)
        monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)

        registry.set_change_review_enabled("ws-1", False)
        registry.set_change_review_enabled("ws-2", True)

        assert registry.change_review_enabled("ws-1") is False
        assert registry.change_review_enabled("ws-2") is True
        assert wfr.folder_binding_roots("ws-1") == ()
        assert wfr.folder_binding_roots("ws-2") == (other_root.resolve(),)

    def test_reenabling_restores_tracking_without_restart(
        self, tmp_path, monkeypatch
    ):
        import tldw_chatbook.Tools.workspace_file_roots as wfr

        registry = self._registry(tmp_path)
        root = self._bound_workspace(registry, tmp_path)
        monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)

        registry.set_change_review_enabled("ws-1", False)
        assert wfr.folder_binding_roots("ws-1") == ()
        registry.set_change_review_enabled("ws-1", True)
        assert wfr.folder_binding_roots("ws-1") == (root.resolve(),)


class TestGatingCoversRegistrationHook:
    """Qodo #1264: the opt-out must gate the registration-time snapshot too."""

    def test_disabled_workspace_add_binding_takes_no_snapshot(
        self, tmp_path, monkeypatch
    ):
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
        from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces.db", client_id="t")
        )
        registry.create_workspace(workspace_id="ws-off", name="Off WS")
        registry.set_change_review_enabled("ws-off", False)
        root = tmp_path / "offroot"
        root.mkdir()
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

        registry.add_folder_binding("ws-off", root)

        from tldw_chatbook.Utils.paths import get_user_data_dir

        change_dir = get_user_data_dir() / "change_review"
        containers = (
            [d for d in change_dir.iterdir() if d.is_dir()]
            if change_dir.is_dir()
            else []
        )
        assert containers == [], (
            "a change-review-disabled workspace still snapshotted on "
            f"binding add: {containers}"
        )

    def test_global_kill_gates_the_registration_hook(
        self, tmp_path, monkeypatch
    ):
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
        from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

        monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "0")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg2"))
        registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "workspaces2.db", client_id="t")
        )
        registry.create_workspace(workspace_id="ws-g", name="G WS")
        root = tmp_path / "groot"
        root.mkdir()

        registry.add_folder_binding("ws-g", root)

        from tldw_chatbook.Utils.paths import get_user_data_dir

        change_dir = get_user_data_dir() / "change_review"
        containers = (
            [d for d in change_dir.iterdir() if d.is_dir()]
            if change_dir.is_dir()
            else []
        )
        assert containers == [], (
            f"globally-disabled change review still snapshotted: {containers}"
        )
