"""Pure-state tests for the Console Environment panel (no I/O, no Textual app)."""
from datetime import datetime, timedelta, timezone

from tldw_chatbook.Widgets.Console.console_inspector_section import (
    RAIL_CONTENT_WIDTH_MIN,
    SECTION_TOGGLE_WIDTH,
)
from tldw_chatbook.Workspaces.change_tracking import ChangedFile
from tldw_chatbook.Chat.console_environment_state import (
    ENV_PENDING_TEXT,
    ENV_ROW_BRANCH,
    ENV_ROW_CHANGES,
    ENV_ROW_CHECKS,
    ENV_ROW_CHECKS_FIX,
    ENV_ROW_COMMIT_PUSH,
    ENV_ROW_EMPTY,
    ENV_ROW_ERROR,
    ENV_ROW_LOCAL,
    ENV_ROW_PENDING,
    ENV_ROW_PR,
    ENV_ROW_PR_ADD,
    ENV_ROW_PR_OPEN,
    ENV_ROW_UNBOUND,
    ENV_ROW_UNBOUND_NOTE,
    ENV_UNBOUND_NOTE_TEXT,
    ENV_UNBOUND_TEXT,
    EnvSourceAvailability,
    unbound_snapshot,
    ExecTargetKind,
    GitEnvState,
    PrCheck,
    PrEnvState,
    TasksEnvState,
    ExecTargetState,
    EnvironmentSnapshot,
    ENV_SUMMARY_BUDGET,
    TASKS_SUMMARY_BUDGET,
    branch_task_id,
    compact_count,
    environment_summary,
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


def test_environment_snapshot_defaults_are_pending_not_a_negative_answer():
    """TASK-31660: the default snapshot means "nobody has looked yet".

    It used to default every tier to NOT_APPLICABLE, which the projection
    renders as the definitive "No git workspace" -- so a cold start asserted
    that a git worktree was not a git worktree for the ~20s until the first
    gatherer landed. NOT_APPLICABLE keeps its ONE meaning ("we looked; it is
    not a repository"); PENDING carries the other.
    """
    snapshot = EnvironmentSnapshot()
    assert snapshot.git.availability is EnvSourceAvailability.PENDING
    assert snapshot.pr.availability is EnvSourceAvailability.PENDING
    assert snapshot.tasks.availability is EnvSourceAvailability.PENDING
    assert snapshot.target.kind is ExecTargetKind.LOCAL


_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def _not_applicable_snapshot() -> EnvironmentSnapshot:
    """A snapshot whose git tier was CHECKED and is genuinely not a repo."""
    return EnvironmentSnapshot(
        git=GitEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE),
        pr=PrEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE),
        tasks=TasksEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE),
    )


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
    """NOT_APPLICABLE's rendering is UNCHANGED by TASK-31660.

    Updated pin: the state is now built explicitly rather than leaning on
    ``EnvironmentSnapshot()``'s default, because that default is PENDING now
    -- "checked: not a repo" and "never checked" stopped being the same
    value. The assertions about NOT_APPLICABLE itself are verbatim.
    """
    state = project_environment_section(
        _not_applicable_snapshot(), frozenset(), now=_NOW)
    assert [r.row_id for r in state.rows] == [ENV_ROW_EMPTY]
    assert state.rows[0].primary_text == "No git workspace"
    assert not state.rows[0].clickable


def test_errored_git_tier_gets_its_own_row_not_the_no_workspace_copy():
    """ERROR means "could not look", never "there is nothing here"."""
    errored = EnvironmentSnapshot(
        git=GitEnvState(availability=EnvSourceAvailability.ERROR))
    state = project_environment_section(errored, frozenset(), now=_NOW)
    assert [r.row_id for r in state.rows] == [ENV_ROW_ERROR]
    assert state.rows[0].primary_text == "Environment unavailable — Refresh to retry"
    assert state.rows[0].status == "blocked"
    # ... and the NOT_APPLICABLE copy is unchanged (negative control).
    not_applicable = project_environment_section(
        _not_applicable_snapshot(), frozenset(), now=_NOW)
    assert not_applicable.rows[0].primary_text == "No git workspace"
    assert not_applicable.rows[0].status == ""


# ---------------------------------------------------------------------------
# TASK-31660: PENDING (nobody has looked yet) and UNBOUND (no folder bound)
# ---------------------------------------------------------------------------


def test_pending_git_tier_renders_a_quiet_checking_row_never_a_negative():
    """AC #2: before the first landing the panel must not assert anything."""
    state = project_environment_section(EnvironmentSnapshot(), frozenset(), now=_NOW)
    assert [r.row_id for r in state.rows] == [ENV_ROW_PENDING]
    assert state.rows[0].primary_text == ENV_PENDING_TEXT == "Checking workspace…"
    assert not state.rows[0].clickable
    assert state.rows[0].status == ""
    assert state.summary == ""
    # The negative it replaced must be gone.
    assert "No git workspace" not in state.rows[0].primary_text


def test_pending_suppresses_counts_commit_push_and_pr_rows():
    """A PENDING tier never paints leftover counts or an action for them."""
    stale_looking = EnvironmentSnapshot(
        git=GitEnvState(
            availability=EnvSourceAvailability.PENDING,
            root="/w/previous", branch="feat/previous", adds=1204, dels=86,
            files=(ChangedFile(path="a.py", status="M", adds=1200, dels=80),),
        ),
        pr=PrEnvState(availability=EnvSourceAvailability.OK, number=7,
                      title="T", state="OPEN", url="https://x/pull/7"),
    )
    ids = [r.row_id for r in project_environment_section(
        stale_looking, frozenset(), now=_NOW).rows]
    assert ids == [ENV_ROW_PENDING]
    for suppressed in (ENV_ROW_CHANGES, ENV_ROW_BRANCH, ENV_ROW_COMMIT_PUSH,
                       ENV_ROW_PR, ENV_ROW_CHECKS):
        assert suppressed not in ids


def test_pending_tasks_tier_hides_the_tasks_card():
    assert project_tasks_section(EnvironmentSnapshot(), frozenset()).rows == ()


def test_unbound_git_tier_carries_change_reviews_copy():
    """AC #5 (TASK-31664): cause-agnostic -- ``workspace_roots == ()`` is
    NOT only "no folder is bound" (it also fires when Change Review
    consent is off for a bound folder, or the consent service is
    absent/raises), so the copy asserts only what stays true regardless of
    cause and names both remediation steps (bind AND enable)."""
    state = project_environment_section(unbound_snapshot(), frozenset(), now=_NOW)
    assert [r.row_id for r in state.rows] == [ENV_ROW_UNBOUND, ENV_ROW_UNBOUND_NOTE]
    assert state.rows[0].primary_text == ENV_UNBOUND_TEXT
    assert state.rows[1].primary_text == ENV_UNBOUND_NOTE_TEXT
    joined = " ".join(r.primary_text for r in state.rows)
    assert "No folder is bound" not in joined
    assert "aren't tracked" in joined
    assert "not a report that nothing changed" in joined
    assert "bind" in joined.lower() and "enable" in joined.lower()
    assert not any(r.clickable for r in state.rows)
    assert state.summary == ""


def test_unbound_suppresses_counts_commit_push_pr_and_checks():
    """AC #1/#3: the PREVIOUS root's data must not survive the switch."""
    previous_root_data = EnvironmentSnapshot(
        git=GitEnvState(
            availability=EnvSourceAvailability.UNBOUND,
            root="/w/previous", branch="feat/previous", adds=1204, dels=86,
            files=(ChangedFile(path="a.py", status="M", adds=1200, dels=80),
                   ChangedFile(path="b.py", status="A", adds=4, dels=6)),
            ahead=3,
        ),
        pr=PrEnvState(availability=EnvSourceAvailability.OK, number=7,
                      title="T", state="OPEN", url="https://x/pull/7",
                      checks=(PrCheck("ci", "failure", "https://ci/1"),)),
    )
    state = project_environment_section(previous_root_data, frozenset(), now=_NOW)
    ids = [r.row_id for r in state.rows]
    assert ids == [ENV_ROW_UNBOUND, ENV_ROW_UNBOUND_NOTE]
    for suppressed in (ENV_ROW_CHANGES, ENV_ROW_BRANCH, ENV_ROW_COMMIT_PUSH,
                       ENV_ROW_PR, ENV_ROW_CHECKS, ENV_ROW_LOCAL):
        assert suppressed not in ids
    # No count text leaks through any row, in either line.
    text = " ".join(r.primary_text + " " + r.secondary_text for r in state.rows)
    assert "1,204" not in text and "feat/previous" not in text


def test_unbound_tasks_tier_hides_the_tasks_card():
    assert project_tasks_section(unbound_snapshot(), frozenset()).rows == ()


def test_unbound_snapshot_marks_every_tier_unbound():
    """One factory, one consistent answer across the three tiers."""
    snapshot = unbound_snapshot()
    assert snapshot.git.availability is EnvSourceAvailability.UNBOUND
    assert snapshot.pr.availability is EnvSourceAvailability.UNBOUND
    assert snapshot.tasks.availability is EnvSourceAvailability.UNBOUND
    assert snapshot.git.files == () and snapshot.pr.number == 0


def test_missing_tool_git_tier_still_reads_as_no_git_workspace():
    missing = EnvironmentSnapshot(
        git=GitEnvState(availability=EnvSourceAvailability.MISSING_TOOL))
    state = project_environment_section(missing, frozenset(), now=_NOW)
    assert [r.row_id for r in state.rows] == [ENV_ROW_EMPTY]


def test_summary_budget_is_derived_from_the_measured_rail_width():
    """AC#5: the budget is not a taste call -- it is what the rail's real
    30-column content width leaves once the title, the chevron and one
    separating column are paid for. Probed on this branch 2026-09-05:
    `#console-environment-section` is 30 columns at 80x24 (36 at 200x50),
    and the old budget of 18 left the title painting "Environm…"."""
    assert (
        ENV_SUMMARY_BUDGET
        + len("Environment")
        + SECTION_TOGGLE_WIDTH
        + 1  # one column between title and summary
        == RAIL_CONTENT_WIDTH_MIN
    )


def test_summary_fits_the_budget_and_never_truncates_the_counts():
    """A long branch ellipsizes; the ± counts stay whole (F1)."""
    long_branch = _git_state(branch="feat/console-inspector-environment-redesign")
    summary = environment_summary(long_branch)
    assert len(summary) <= ENV_SUMMARY_BUDGET
    assert summary.endswith("+1,204 −86")  # counts intact
    # Pin narrowed with the budget (18 -> 15): 10 columns of counts leave
    # four for the branch fragment at the rail's real width.
    assert summary.startswith("fea") and "…" in summary
    # The projection uses the same budgeted summary, not the raw join.
    projected = project_environment_section(
        EnvironmentSnapshot(git=long_branch), frozenset(), now=_NOW)
    assert projected.summary == summary


def test_short_branch_summary_is_unchanged():
    short = _git_state(branch="dev", adds=10, dels=2)
    assert environment_summary(short) == "dev +10 −2"
    assert "…" not in environment_summary(short)


def test_summary_drops_the_branch_entirely_when_counts_fill_the_budget():
    huge = _git_state(branch="feat/whatever", adds=1_679_102, dels=277_870)
    # "+1.7M −278k" is 11 columns; a 12-column budget leaves 0 for a branch.
    assert environment_summary(huge, budget=12) == "+1.7M −278k"


def test_detached_head_summary_is_budgeted_too():
    detached = _git_state(branch=None, detached=True, head_short="abc1234")
    assert len(environment_summary(detached)) <= ENV_SUMMARY_BUDGET


def test_changes_row_shows_signed_totals_and_branch_row_shows_divergence():
    snapshot = EnvironmentSnapshot(git=_git_state(ahead=2, behind=1, upstream="origin/feat/x"))
    state = project_environment_section(snapshot, frozenset(), now=_NOW)
    by_id = {r.row_id: r for r in state.rows}
    assert by_id[ENV_ROW_CHANGES].secondary_text == "+1,204 −86"
    assert "↑2" in by_id[ENV_ROW_BRANCH].secondary_text
    assert "↓1" in by_id[ENV_ROW_BRANCH].secondary_text
    assert by_id[ENV_ROW_CHANGES].clickable and by_id[ENV_ROW_BRANCH].clickable


def test_branch_row_marker_survives_ellipsis_on_a_long_branch_name():
    """AC #1: the marker is appended AFTER truncation, never before.

    A branch name long enough to overflow the rail's real content region
    (`RAIL_CONTENT_WIDTH_MIN`) must still end in the chevron -- if the
    marker were appended before ellipsizing, the terminal's own CSS
    `text-overflow: ellipsis` could cut it off along with the tail of the
    label, silently losing the one thing that marks this row as
    expandable.
    """
    long_branch = _git_state(
        branch="feat/task-31450-console-inspector-environment-redesign-with-a-very-long-name"
    )
    snapshot = EnvironmentSnapshot(git=long_branch)
    by_id = {
        r.row_id: r
        for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows
    }
    branch_row = by_id[ENV_ROW_BRANCH]
    assert branch_row.primary_text.endswith(" ▸"), branch_row.primary_text
    assert "…" in branch_row.primary_text
    assert len(branch_row.primary_text) <= 30  # RAIL_CONTENT_WIDTH_MIN

    expanded_row = {
        r.row_id: r
        for r in project_environment_section(
            snapshot, frozenset({ENV_ROW_BRANCH}), now=_NOW
        ).rows
    }[ENV_ROW_BRANCH]
    assert expanded_row.primary_text.endswith(" ▾")


def test_commit_or_push_row_hidden_when_clean_and_synced_shown_when_dirty():
    clean = EnvironmentSnapshot(git=_git_state(adds=0, dels=0, files=()))
    dirty = EnvironmentSnapshot(git=_git_state())
    clean_ids = [r.row_id for r in project_environment_section(clean, frozenset(), now=_NOW).rows]
    dirty_ids = [r.row_id for r in project_environment_section(dirty, frozenset(), now=_NOW).rows]
    assert ENV_ROW_COMMIT_PUSH not in clean_ids
    assert ENV_ROW_COMMIT_PUSH in dirty_ids


def test_commit_or_push_label_pluralizes_the_file_count():
    """AC #2 (TASK-31664): renamed to name what it does, and to carry the
    same "…" Change Review's own destination uses (``Commit…``/``Push…``)."""
    one = EnvironmentSnapshot(git=_git_state(
        files=(ChangedFile(path="a.py", status="M", adds=1, dels=0),)))
    two = EnvironmentSnapshot(git=_git_state())
    one_by_id = {r.row_id: r for r in project_environment_section(one, frozenset(), now=_NOW).rows}
    two_by_id = {r.row_id: r for r in project_environment_section(two, frozenset(), now=_NOW).rows}
    assert one_by_id[ENV_ROW_COMMIT_PUSH].primary_text == "Review & commit… · 1 file"
    assert two_by_id[ENV_ROW_COMMIT_PUSH].primary_text == "Review & commit… · 2 files"
    assert "Commit or push" not in one_by_id[ENV_ROW_COMMIT_PUSH].primary_text


def test_push_only_variant_when_tree_clean_but_ahead():
    """AC #1: this row also navigates to Change Review, so it carries the
    same "opens another surface" marker as the dirty variant above."""
    snapshot = EnvironmentSnapshot(git=_git_state(adds=0, dels=0, files=(), ahead=2))
    by_id = {r.row_id: r for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows}
    assert by_id[ENV_ROW_COMMIT_PUSH].primary_text == "Push ↑2…"


def test_changes_expansion_lists_files_with_per_file_counts():
    snapshot = EnvironmentSnapshot(git=_git_state())
    state = project_environment_section(snapshot, frozenset({ENV_ROW_CHANGES}), now=_NOW)
    by_id = {r.row_id: r for r in state.rows}
    ids = list(by_id)
    assert "env-file-0" in ids and "env-file-1" in ids
    file_row = by_id["env-file-0"]
    assert file_row.primary_text == "M a.py"
    assert file_row.secondary_text == "+1,200 −80"
    # AC #1: file rows are inert (Enter does nothing) and carry no marker;
    # "Changes" flips to the open chevron; "Review in Change Review"
    # navigates elsewhere and carries the "…" marker.
    assert not file_row.clickable
    assert by_id[ENV_ROW_CHANGES].primary_text == "Changes ▾"
    assert by_id["env-changes-review"].primary_text == "Review in Change Review…"


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
    assert by_id[ENV_ROW_PR].primary_text == "PR #2281 · Open ▸"
    assert by_id[ENV_ROW_CHECKS].primary_text == "1 failing check ▸"
    expanded = project_environment_section(
        snapshot, frozenset({ENV_ROW_PR, ENV_ROW_CHECKS}), now=_NOW)
    by_id_expanded = {r.row_id: r for r in expanded.rows}
    expanded_ids = list(by_id_expanded)
    assert ENV_ROW_PR_OPEN in expanded_ids and ENV_ROW_PR_ADD in expanded_ids
    assert ENV_ROW_CHECKS_FIX in expanded_ids
    # AC #1: expanded rows flip the chevron, and the composer-insert rows
    # carry the "+ " marker, distinct from the "…" navigation marker.
    assert by_id_expanded[ENV_ROW_PR].primary_text == "PR #2281 · Open ▾"
    assert by_id_expanded[ENV_ROW_CHECKS].primary_text == "1 failing check ▾"
    assert by_id_expanded[ENV_ROW_PR_OPEN].primary_text == "Open in browser…"
    assert by_id_expanded[ENV_ROW_PR_ADD].primary_text == "+ Add to chat"
    assert by_id_expanded[ENV_ROW_CHECKS_FIX].primary_text == (
        "+ Fix — add failure summary to chat"
    )


def test_detached_head_labels_and_skipped_pr():
    snapshot = EnvironmentSnapshot(git=_git_state(branch=None, detached=True, head_short="abc1234"))
    by_id = {r.row_id: r for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows}
    assert by_id[ENV_ROW_BRANCH].primary_text == "detached @ abc1234 ▸"


def test_stale_marker_survives_on_error_with_prior_data():
    snapshot = EnvironmentSnapshot(git=_git_state(stale=True))
    by_id = {r.row_id: r for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows}
    assert by_id[ENV_ROW_CHANGES].status == "blocked"


def test_stale_carries_a_text_marker_alongside_color():
    """AC #4: color alone made "stale" indistinguishable from an error in
    the identical hue; the projection now also carries a text marker."""
    stale_with_secondary = EnvironmentSnapshot(git=_git_state(stale=True))
    by_id = {
        r.row_id: r
        for r in project_environment_section(stale_with_secondary, frozenset(), now=_NOW).rows
    }
    assert by_id[ENV_ROW_CHANGES].status == "blocked"
    assert "(stale)" in by_id[ENV_ROW_CHANGES].secondary_text
    assert "(stale)" in by_id[ENV_ROW_BRANCH].secondary_text

    fresh = EnvironmentSnapshot(git=_git_state(stale=False))
    fresh_by_id = {
        r.row_id: r
        for r in project_environment_section(fresh, frozenset(), now=_NOW).rows
    }
    assert "(stale)" not in fresh_by_id[ENV_ROW_CHANGES].secondary_text
    assert "(stale)" not in fresh_by_id[ENV_ROW_BRANCH].secondary_text


def test_stale_pr_row_also_carries_the_text_marker():
    pr = PrEnvState(
        availability=EnvSourceAvailability.OK, number=7, title="T",
        state="OPEN", url="https://x/pull/7", stale=True,
    )
    snapshot = EnvironmentSnapshot(git=_git_state(), pr=pr)
    by_id = {
        r.row_id: r
        for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows
    }
    assert by_id[ENV_ROW_PR].status == "blocked"
    assert "(stale)" in by_id[ENV_ROW_PR].secondary_text


def test_local_row_expansion_shows_remote_placeholder():
    snapshot = EnvironmentSnapshot(git=_git_state())
    state = project_environment_section(snapshot, frozenset({ENV_ROW_LOCAL}), now=_NOW)
    by_id = {r.row_id: r for r in state.rows}
    texts = [r.primary_text for r in state.rows]
    assert any("Remote tldw_server" in t for t in texts)
    assert by_id[ENV_ROW_LOCAL].primary_text == "Local ▾"
    # AC #1: "Local instance ✓" is inert (Enter does nothing on it).
    assert not by_id["env-local-current"].clickable


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
    """Updated pin: NOT_APPLICABLE built explicitly (the default is PENDING now)."""
    state = project_tasks_section(_not_applicable_snapshot(), frozenset())
    assert state.rows == ()


def test_branch_task_headline_with_ac_progress():
    snapshot = EnvironmentSnapshot(tasks=_tasks_state())
    head = project_tasks_section(snapshot, frozenset()).rows[0]
    assert head.row_id == TASKS_ROW_HEAD
    assert head.primary_text == "task-3401 · In Progress ▸"
    assert head.secondary_text == "3/6 ACs · Video gen foundation"
    assert head.clickable
    expanded_head = project_tasks_section(snapshot, frozenset({TASKS_ROW_HEAD})).rows[0]
    assert expanded_head.primary_text == "task-3401 · In Progress ▾"


def test_head_row_without_a_branch_task_names_the_list_not_the_counts():
    """AC#4: the head row used to read "3 in progress · 12 to do" while the
    section header read "3 doing · 12 todo" -- the same fact twice, in two
    vocabularies, one line apart. The counts stay in the header; the row is
    the handle onto the list and says what the list holds."""
    snapshot = EnvironmentSnapshot(tasks=_tasks_state(branch_task=None))
    state = project_tasks_section(snapshot, frozenset())
    head = state.rows[0]
    assert head.row_id == TASKS_ROW_HEAD  # still the expand/collapse handle
    assert head.clickable
    assert head.primary_text == "Backlog ▸"
    assert head.secondary_text == "2 tasks"
    assert state.summary == "3 doing · 12 todo"
    assert "in progress" not in head.primary_text


def test_head_row_secondary_is_singular_for_one_task():
    snapshot = EnvironmentSnapshot(tasks=_tasks_state(
        branch_task=None,
        entries=(BacklogTaskEntry("1", "Only one", "To Do"),),
    ))
    assert project_tasks_section(snapshot, frozenset()).rows[0].secondary_text == (
        "1 task"
    )


def test_tasks_summary_is_budgeted_for_the_shorter_tasks_title():
    """AC#5 / TASK-31629 #13: "task-31450 · In Progress" is 24 columns and
    left three for the 5-column "Tasks" title at 80x24."""
    snapshot = EnvironmentSnapshot(tasks=_tasks_state(
        branch_task=BranchTaskState(task_id="31450", title="Long", status="In Progress"),
    ))
    summary = project_tasks_section(snapshot, frozenset()).summary
    assert len(summary) <= TASKS_SUMMARY_BUDGET
    assert summary.startswith("task-31450")
    assert (
        TASKS_SUMMARY_BUDGET + len("Tasks") + SECTION_TOGGLE_WIDTH + 1
        == RAIL_CONTENT_WIDTH_MIN
    )


def test_expansion_lists_entries_in_progress_first_and_add_action():
    snapshot = EnvironmentSnapshot(tasks=_tasks_state())
    rows = project_tasks_section(snapshot, frozenset({TASKS_ROW_HEAD})).rows
    by_id = {r.row_id: r for r in rows}
    assert TASKS_ROW_ADD in by_id
    # AC #1: composer-insert marker, distinct from the expand chevron and
    # the navigation "…"; entry rows are inert and carry no marker at all.
    assert by_id[TASKS_ROW_ADD].primary_text == "+ Add task to chat"
    assert by_id[TASKS_ROW_HEAD].primary_text.endswith(" ▾")
    entry_rows = [r for r in rows if r.row_id.startswith("task-entry-")]
    assert entry_rows[0].primary_text.startswith("task-3401")
    assert entry_rows[0].status == "running"


def test_scanning_placeholder():
    snapshot = EnvironmentSnapshot(
        tasks=TasksEnvState(availability=EnvSourceAvailability.OK, scanning=True))
    rows = project_tasks_section(snapshot, frozenset()).rows
    assert rows[0].primary_text == "Scanning backlog…"
