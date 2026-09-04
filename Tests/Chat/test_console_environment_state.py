"""Pure-state tests for the Console Environment panel (no I/O, no Textual app)."""
from datetime import datetime, timedelta, timezone

from tldw_chatbook.Workspaces.change_tracking import ChangedFile
from tldw_chatbook.Chat.console_environment_state import (
    ENV_ROW_BRANCH,
    ENV_ROW_CHANGES,
    ENV_ROW_CHECKS,
    ENV_ROW_CHECKS_FIX,
    ENV_ROW_COMMIT_PUSH,
    ENV_ROW_LOCAL,
    ENV_ROW_PR,
    ENV_ROW_PR_ADD,
    ENV_ROW_PR_OPEN,
    EnvSourceAvailability,
    ExecTargetKind,
    GitEnvState,
    PrCheck,
    PrEnvState,
    TasksEnvState,
    ExecTargetState,
    EnvironmentSnapshot,
    branch_task_id,
    compact_count,
    failing_checks_text,
    pr_summary_text,
    project_environment_section,
    relative_age,
    signed_change_counts,
)


def test_compact_count_small_numbers_keep_thousands_separators():
    assert compact_count(0) == "0"
    assert compact_count(1204) == "1,204"
    assert compact_count(99_999) == "99,999"


def test_compact_count_large_numbers_compress():
    assert compact_count(277_870) == "278k"
    assert compact_count(1_679_102) == "1.7M"


def test_signed_change_counts_pairs_plus_and_minus():
    assert signed_change_counts(1204, 86) == "+1,204 −86"
    assert signed_change_counts(1_679_102, 277_870) == "+1.7M −278k"


def test_branch_task_id_matches_plain_and_subtask_ids():
    assert branch_task_id("feat/task-3401-video-generation-foundation") == "3401"
    assert branch_task_id("fix/task-3401.6-comfyui-adapter") == "3401.6"
    assert branch_task_id("chore/no-task-reference-here") is None
    assert branch_task_id(None) is None


def test_relative_age_buckets():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    assert relative_age(None, now) == ""
    assert relative_age(now - timedelta(minutes=5), now) == "5m ago"
    assert relative_age(now - timedelta(hours=3), now) == "3h ago"
    assert relative_age(now - timedelta(days=6), now) == "6d ago"


def test_environment_snapshot_defaults_are_not_applicable():
    snapshot = EnvironmentSnapshot()
    assert snapshot.git.availability is EnvSourceAvailability.NOT_APPLICABLE
    assert snapshot.pr.availability is EnvSourceAvailability.NOT_APPLICABLE
    assert snapshot.tasks.availability is EnvSourceAvailability.NOT_APPLICABLE
    assert snapshot.target.kind is ExecTargetKind.LOCAL


_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def _git_state(**kw) -> GitEnvState:
    base = dict(
        availability=EnvSourceAvailability.OK,
        root="/w/repo",
        branch="feat/task-3401-video-generation",
        adds=1204,
        dels=86,
        files=(ChangedFile(path="a.py", status="M", adds=1200, dels=80),
               ChangedFile(path="b.py", status="A", adds=4, dels=6)),
    )
    base.update(kw)
    return GitEnvState(**base)


def test_no_git_workspace_projects_a_single_quiet_row():
    state = project_environment_section(EnvironmentSnapshot(), frozenset(), now=_NOW)
    assert [r.row_id for r in state.rows] == ["env-empty"]
    assert state.rows[0].primary_text == "No git workspace"
    assert not state.rows[0].clickable


def test_changes_row_shows_signed_totals_and_branch_row_shows_divergence():
    snapshot = EnvironmentSnapshot(git=_git_state(ahead=2, behind=1, upstream="origin/feat/x"))
    state = project_environment_section(snapshot, frozenset(), now=_NOW)
    by_id = {r.row_id: r for r in state.rows}
    assert by_id[ENV_ROW_CHANGES].secondary_text == "+1,204 −86"
    assert "↑2" in by_id[ENV_ROW_BRANCH].secondary_text
    assert "↓1" in by_id[ENV_ROW_BRANCH].secondary_text
    assert by_id[ENV_ROW_CHANGES].clickable and by_id[ENV_ROW_BRANCH].clickable


def test_commit_or_push_row_hidden_when_clean_and_synced_shown_when_dirty():
    clean = EnvironmentSnapshot(git=_git_state(adds=0, dels=0, files=()))
    dirty = EnvironmentSnapshot(git=_git_state())
    clean_ids = [r.row_id for r in project_environment_section(clean, frozenset(), now=_NOW).rows]
    dirty_ids = [r.row_id for r in project_environment_section(dirty, frozenset(), now=_NOW).rows]
    assert ENV_ROW_COMMIT_PUSH not in clean_ids
    assert ENV_ROW_COMMIT_PUSH in dirty_ids


def test_push_only_variant_when_tree_clean_but_ahead():
    snapshot = EnvironmentSnapshot(git=_git_state(adds=0, dels=0, files=(), ahead=2))
    by_id = {r.row_id: r for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows}
    assert by_id[ENV_ROW_COMMIT_PUSH].primary_text == "Push ↑2"


def test_changes_expansion_lists_files_with_per_file_counts():
    snapshot = EnvironmentSnapshot(git=_git_state())
    state = project_environment_section(snapshot, frozenset({ENV_ROW_CHANGES}), now=_NOW)
    ids = [r.row_id for r in state.rows]
    assert "env-file-0" in ids and "env-file-1" in ids
    file_row = next(r for r in state.rows if r.row_id == "env-file-0")
    assert file_row.primary_text == "M a.py"
    assert file_row.secondary_text == "+1,200 −80"


def test_pr_rows_absent_without_pr_and_present_with_actions_when_expanded():
    no_pr = EnvironmentSnapshot(git=_git_state())
    assert ENV_ROW_PR not in [r.row_id for r in project_environment_section(no_pr, frozenset(), now=_NOW).rows]
    pr = PrEnvState(
        availability=EnvSourceAvailability.OK, number=2281, title="Split boot CSS",
        state="OPEN", url="https://github.com/o/r/pull/2281", adds=36643, dels=2871,
        checks=(PrCheck("lint", "success"), PrCheck("ci", "failure", "https://ci/1"),
                PrCheck("docs", "pending")),
    )
    snapshot = EnvironmentSnapshot(git=_git_state(), pr=pr)
    collapsed = project_environment_section(snapshot, frozenset(), now=_NOW)
    by_id = {r.row_id: r for r in collapsed.rows}
    assert by_id[ENV_ROW_PR].primary_text == "PR #2281 · Open"
    assert by_id[ENV_ROW_CHECKS].primary_text == "1 failing check"
    expanded = project_environment_section(
        snapshot, frozenset({ENV_ROW_PR, ENV_ROW_CHECKS}), now=_NOW)
    expanded_ids = [r.row_id for r in expanded.rows]
    assert ENV_ROW_PR_OPEN in expanded_ids and ENV_ROW_PR_ADD in expanded_ids
    assert ENV_ROW_CHECKS_FIX in expanded_ids


def test_detached_head_labels_and_skipped_pr():
    snapshot = EnvironmentSnapshot(git=_git_state(branch=None, detached=True, head_short="abc1234"))
    by_id = {r.row_id: r for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows}
    assert by_id[ENV_ROW_BRANCH].primary_text == "detached @ abc1234"


def test_stale_marker_survives_on_error_with_prior_data():
    snapshot = EnvironmentSnapshot(git=_git_state(stale=True))
    by_id = {r.row_id: r for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows}
    assert by_id[ENV_ROW_CHANGES].status == "blocked"


def test_local_row_expansion_shows_remote_placeholder():
    snapshot = EnvironmentSnapshot(git=_git_state())
    state = project_environment_section(snapshot, frozenset({ENV_ROW_LOCAL}), now=_NOW)
    texts = [r.primary_text for r in state.rows]
    assert any("Remote tldw_server" in t for t in texts)


def test_composer_payload_builders():
    pr = PrEnvState(availability=EnvSourceAvailability.OK, number=7, title="T",
                    state="OPEN", url="https://x/pull/7",
                    checks=(PrCheck("ci", "failure", "https://ci/1"),))
    assert "PR #7" in pr_summary_text(pr) and "https://x/pull/7" in pr_summary_text(pr)
    fix = failing_checks_text(pr)
    assert "ci" in fix and "https://ci/1" in fix


from tldw_chatbook.Chat.console_environment_state import (
    BacklogTaskEntry,
    BranchTaskState,
    TASKS_ROW_ADD,
    TASKS_ROW_HEAD,
    project_tasks_section,
)


def _tasks_state(**kw) -> TasksEnvState:
    base = dict(
        availability=EnvSourceAvailability.OK,
        branch_task=BranchTaskState(task_id="3401", title="Video gen foundation",
                                    status="In Progress", ac_done=3, ac_total=6,
                                    path="backlog/tasks/task-3401 - Video.md"),
        in_progress=3, todo=12,
        entries=(BacklogTaskEntry("3401", "Video gen foundation", "In Progress"),
                 BacklogTaskEntry("25704", "Render-path sweep", "To Do")),
    )
    base.update(kw)
    return TasksEnvState(**base)


def test_tasks_card_absent_without_backlog_dir():
    state = project_tasks_section(EnvironmentSnapshot(), frozenset())
    assert state.rows == ()


def test_branch_task_headline_with_ac_progress():
    snapshot = EnvironmentSnapshot(tasks=_tasks_state())
    head = project_tasks_section(snapshot, frozenset()).rows[0]
    assert head.row_id == TASKS_ROW_HEAD
    assert head.primary_text == "task-3401 · In Progress"
    assert head.secondary_text == "3/6 ACs · Video gen foundation"
    assert head.clickable


def test_counts_headline_when_no_branch_task():
    snapshot = EnvironmentSnapshot(tasks=_tasks_state(branch_task=None))
    head = project_tasks_section(snapshot, frozenset()).rows[0]
    assert head.primary_text == "3 in progress · 12 to do"


def test_expansion_lists_entries_in_progress_first_and_add_action():
    snapshot = EnvironmentSnapshot(tasks=_tasks_state())
    rows = project_tasks_section(snapshot, frozenset({TASKS_ROW_HEAD})).rows
    ids = [r.row_id for r in rows]
    assert TASKS_ROW_ADD in ids
    entry_rows = [r for r in rows if r.row_id.startswith("task-entry-")]
    assert entry_rows[0].primary_text.startswith("task-3401")
    assert entry_rows[0].status == "running"


def test_scanning_placeholder():
    snapshot = EnvironmentSnapshot(
        tasks=TasksEnvState(availability=EnvSourceAvailability.OK, scanning=True))
    rows = project_tasks_section(snapshot, frozenset()).rows
    assert rows[0].primary_text == "Scanning backlog…"
