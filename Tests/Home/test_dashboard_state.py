from tldw_chatbook.Home.dashboard_state import (
    HOME_FLASHCARDS_DUE_ROW_ID,
    RUNNING_RUN_STATUS,
    HomeActiveWorkItem,
    HomeDashboardInput,
    build_home_controls,
    build_home_triage_state,
    categorize_run_status,
    choose_home_selected_item,
    choose_next_best_action,
    summarize_home_dashboard,
)


def test_next_best_action_prioritizes_blockers():
    state = HomeDashboardInput(
        model_ready=False,
        pending_approval_count=2,
        active_run_count=1,
        has_library_content=True,
    )

    action = choose_next_best_action(state)

    assert action.action_id == "fix_model_setup"
    assert action.label == "Set up Console model"


def test_next_best_action_prioritizes_pending_approval_after_readiness():
    state = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=2,
        active_run_count=1,
        has_library_content=True,
    )

    action = choose_next_best_action(state)

    assert action.action_id == "review_approvals"
    assert action.target_route == "chat"


def test_next_best_action_surfaces_notifications_after_live_work_blockers():
    state = HomeDashboardInput(
        model_ready=True,
        notification_count=3,
        has_library_content=True,
    )

    action = choose_next_best_action(state)

    assert action.action_id == "review_notifications"
    assert action.label == "Review notifications"
    assert action.target_route == "subscriptions"


def test_pending_approval_still_outranks_unread_notifications():
    state = HomeDashboardInput(
        model_ready=True,
        pending_approval_count=1,
        notification_count=3,
        has_library_content=True,
    )

    action = choose_next_best_action(state)

    assert action.action_id == "review_approvals"


def test_failed_active_work_item_prioritizes_recovery_before_resume():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            has_library_content=True,
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="local:watchlist_run:5",
                    title="Daily security feed",
                    source="Watchlists",
                    status="failed",
                    detail_route="subscriptions",
                ),
            ),
        )
    )

    assert dashboard.next_action.action_id == "review_failed_work"
    assert dashboard.next_action.label == "Review failed work"
    assert dashboard.next_action.target_route == "subscriptions"
    controls_by_id = {control.control_id: control for control in dashboard.controls}
    assert controls_by_id["home-retry"].target_route == "subscriptions"
    assert controls_by_id["home-retry"].target_id == "local:watchlist_run:5"
    assert controls_by_id["home-open-details"].target_route == "subscriptions"


def test_failed_work_details_follow_failed_item_when_mixed_with_running_work():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            has_library_content=True,
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="local:watchlist_run:6",
                    title="Queued release feed",
                    source="Watchlists",
                    status="queued",
                    detail_route="subscriptions",
                ),
                HomeActiveWorkItem(
                    item_id="local:watchlist_run:5",
                    title="Daily security feed",
                    source="Watchlists",
                    status="failed",
                    detail_route="subscriptions",
                ),
            ),
        )
    )

    controls_by_id = {control.control_id: control for control in dashboard.controls}
    assert dashboard.next_action.action_id == "review_failed_work"
    assert controls_by_id["home-open-details"].target_id == "local:watchlist_run:5"
    assert controls_by_id["home-retry"].target_id == "local:watchlist_run:5"


def test_home_selected_item_uses_same_priority_as_default_details_control():
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:chatbook:77",
                title="Grounded Answer",
                source="Artifacts",
                status="ready",
                detail_route="artifacts",
                console_available=True,
            ),
            HomeActiveWorkItem(
                item_id="local:watchlist_run:5",
                title="Daily security feed",
                source="Watchlists",
                status="failed",
                detail_route="subscriptions",
                console_available=True,
            ),
        ),
    )

    dashboard = summarize_home_dashboard(state)
    selected_item = choose_home_selected_item(state)

    assert selected_item is not None
    assert selected_item.item_id == "local:watchlist_run:5"
    controls_by_id = {control.control_id: control for control in dashboard.controls}
    assert controls_by_id["home-open-details"].target_id == selected_item.item_id
    assert controls_by_id["home-open-details"].target_route == selected_item.detail_route
    assert controls_by_id["home-open-in-console"].target_id == selected_item.item_id


def test_dashboard_summary_exposes_required_sections():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            pending_approval_count=0,
            active_run_count=0,
            has_library_content=False,
        )
    )

    assert [section.section_id for section in dashboard.sections] == [
        "status",
        "attention",
        "active_work",
        "system_status",
        "next_best_action",
        "recent_work",
    ]


def test_dashboard_summary_exposes_notifications_without_live_work_controls():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            notification_count=3,
            has_library_content=True,
        )
    )

    attention_section = next(
        section for section in dashboard.sections if section.section_id == "attention"
    )
    assert "Unread notifications: 3" in attention_section.lines
    assert dashboard.next_action.action_id == "review_notifications"
    assert dashboard.controls == ()


def test_dashboard_summary_exposes_lightweight_control_ids_for_core_states():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            pending_approval_count=1,
            running_run_count=1,
            paused_run_count=1,
            failed_run_count=1,
            failed_schedule_count=1,
            active_run_count=3,
            has_library_content=True,
        )
    )

    control_ids = {control.control_id for control in dashboard.controls}
    assert {
        "home-approve",
        "home-reject",
        "home-pause",
        "home-resume",
        "home-retry",
        "home-open-details",
        "home-open-in-console",
    }.issubset(control_ids)


def test_dashboard_summary_exposes_active_work_item_context_and_targets():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            has_library_content=True,
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="run-1",
                    title="Daily digest",
                    source="workflows",
                    status="running",
                    detail_route="workflows",
                    console_available=True,
                ),
            ),
        )
    )

    active_work_section = next(section for section in dashboard.sections if section.section_id == "active_work")
    assert "Daily digest" in "\n".join(active_work_section.lines)
    assert "running" in "\n".join(active_work_section.lines)
    assert "workflows" in "\n".join(active_work_section.lines)

    controls_by_id = {control.control_id: control for control in dashboard.controls}
    assert controls_by_id["home-pause"].target_id == "run-1"
    assert controls_by_id["home-open-details"].target_id == "run-1"
    assert controls_by_id["home-open-details"].target_route == "workflows"
    assert controls_by_id["home-open-in-console"].target_id == "run-1"


def test_dashboard_summary_keeps_chatbook_artifact_reachable_when_mixed_with_watchlist_run():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            has_library_content=True,
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="local:watchlist_run:5",
                    title="Daily feed",
                    source="Watchlists",
                    status="running",
                    detail_route="watchlists",
                    console_available=True,
                ),
                HomeActiveWorkItem(
                    item_id="local:chatbook:77",
                    title="Grounded Answer",
                    source="Artifacts",
                    status="ready",
                    detail_route="artifacts",
                    console_available=True,
                ),
            ),
        )
    )

    controls_by_id = {control.control_id: control for control in dashboard.controls}
    assert controls_by_id["home-open-details"].target_id == "local:watchlist_run:5"
    assert controls_by_id["home-open-in-console"].target_id == "local:watchlist_run:5"
    assert controls_by_id["home-open-chatbook-details"].target_id == "local:chatbook:77"
    assert controls_by_id["home-open-chatbook-details"].target_route == "artifacts"
    assert controls_by_id["home-open-chatbook-in-console"].target_id == "local:chatbook:77"


def test_dashboard_item_statuses_gate_matching_controls():
    dashboard = summarize_home_dashboard(
        HomeDashboardInput(
            model_ready=True,
            has_library_content=True,
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="approval-1",
                    title="Tool request",
                    source="mcp",
                    status="approval_required",
                    detail_route="mcp",
                ),
            ),
        )
    )

    control_ids = {control.control_id for control in dashboard.controls}
    assert "home-approve" in control_ids
    assert "home-reject" in control_ids
    assert "home-open-details" in control_ids
    assert "home-pause" not in control_ids
    assert "home-open-in-console" not in control_ids


# --- T154: Home Pause has no wired action for Library ingest jobs -- suppress
# it when the selected item is one (item_id starts "local:ingest:", the same
# marker active_work_adapter._is_local_ingest_job_id uses), while a regular
# (non-ingest) running item keeps its Pause control. ----


def test_build_home_controls_suppresses_pause_for_selected_ingest_item():
    ingest_item = HomeActiveWorkItem(
        item_id="local:ingest:job-1",
        title="Importing report.pdf",
        source="Library",
        status="parsing",
        detail_route="library",
    )
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(ingest_item,),
    )

    controls = build_home_controls(state, selected_row_id=ingest_item.item_id, selected_item=ingest_item)

    control_ids = {control.control_id for control in controls}
    assert "home-pause" not in control_ids


def test_build_home_controls_keeps_pause_for_selected_non_ingest_running_item():
    running_item = HomeActiveWorkItem(
        item_id="run-1",
        title="Daily digest",
        source="workflows",
        status="running",
        detail_route="workflows",
    )
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(running_item,),
    )

    controls = build_home_controls(state, selected_row_id=running_item.item_id, selected_item=running_item)

    control_ids = {control.control_id for control in controls}
    assert "home-pause" in control_ids


# --- F3: Library ingest jobs' "parsing"/"writing" status literals map into
# the shared Running category, same as any other subsystem's "running". ----


def test_categorize_run_status_maps_parsing_and_writing_into_running():
    """(F3) The Library ingest job registry's PARSING/WRITING states
    (replacing the old single RUNNING state) report their own literal
    status strings through HomeActiveWorkItem.status -- this generic,
    subsystem-agnostic categorizer must still bucket them as "running" so
    they land in Home's Running rail/feed and count toward
    running_run_count, exactly like every other subsystem's "running"/
    "queued"/"active"/"scheduled" items already do."""
    assert categorize_run_status("parsing") == RUNNING_RUN_STATUS
    assert categorize_run_status("writing") == RUNNING_RUN_STATUS


def test_dashboard_counts_parsing_and_writing_ingest_items_as_running():
    dashboard_input = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:1",
                title="report.pdf",
                source="Library",
                status="parsing",
                detail_route="library",
            ),
            HomeActiveWorkItem(
                item_id="local:ingest:2",
                title="notes.txt",
                source="Library",
                status="writing",
                detail_route="library",
            ),
        ),
    )

    triage = build_home_triage_state(dashboard_input)

    running_section = next(s for s in triage.sections if s.section_id == "running")
    assert running_section.count == 2
    assert {row.title for row in running_section.rows} == {"report.pdf", "notes.txt"}


from datetime import datetime, timezone

from tldw_chatbook.Home.dashboard_state import build_home_triage_state

_NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


def _items_input(**overrides) -> HomeDashboardInput:
    defaults = dict(
        model_ready=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="wf:approve-1",
                title="Approval: publish chatbook",
                source="Workflows",
                status="pending_approval",
                detail_route="workflows",
                console_available=True,
                updated_at="2026-07-04T11:57:00+00:00",
            ),
            HomeActiveWorkItem(
                item_id="watch:run-1",
                title="Watchlist sweep",
                source="Watchlists",
                status="running",
                detail_route="watchlists",
                updated_at="2026-07-04T12:00:00+00:00",
            ),
            HomeActiveWorkItem(
                item_id="sched:fail-1",
                title="Retry: ingest failure",
                source="Schedules",
                status="failed",
                detail_route="schedules",
                updated_at="2026-07-04T11:00:00+00:00",
            ),
        ),
    )
    defaults.update(overrides)
    return HomeDashboardInput(**defaults)


def test_triage_sections_split_by_status_with_ages():
    triage = build_home_triage_state(_items_input(), now=_NOW)
    by_id = {section.section_id: section for section in triage.sections}
    attention = by_id["attention"]
    running = by_id["running"]
    assert attention.title == "Needs Attention"
    assert attention.count == 2  # approval + failed
    titles = [row.title for row in attention.rows]
    assert "Approval: publish chatbook" in titles
    assert "Retry: ingest failure" in titles
    approval_row = next(r for r in attention.rows if r.row_id == "wf:approve-1")
    assert approval_row.age_label == "3m"
    assert approval_row.glyph == "\u25cf"
    assert running.count == 1
    assert running.rows[0].age_label == "now"


def test_triage_header_line_formats():
    triage = build_home_triage_state(_items_input(), now=_NOW)
    assert triage.header_line == "Home | Ready \u00b7 Local"
    blocked = build_home_triage_state(
        _items_input(model_ready=False, runtime_source="server", server_label="lab"),
        now=_NOW,
    )
    assert blocked.header_line == "Home | Blocked \u00b7 Server: lab"


def test_triage_default_selection_prefers_attention_and_builds_canvas():
    triage = build_home_triage_state(_items_input(), now=_NOW)
    assert triage.selected_row_id == "wf:approve-1"
    assert triage.canvas.title == "Approval: publish chatbook"
    action_ids = [a.control_id for a in triage.canvas.actions]
    assert "home-approve" in action_ids and "home-reject" in action_ids
    assert triage.canvas.next_action_is_canvas is False


def test_triage_explicit_selection_and_missing_age_blank():
    triage = build_home_triage_state(
        _items_input(
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="x:1",
                    title="No timestamp item",
                    source="ACP",
                    status="running",
                ),
            )
        ),
        selected_row_id="x:1",
        now=_NOW,
    )
    assert triage.selected_row_id == "x:1"
    assert triage.sections[1].rows[0].age_label == ""


def test_triage_empty_input_makes_next_action_the_canvas():
    triage = build_home_triage_state(HomeDashboardInput(model_ready=True), now=_NOW)
    assert all(section.count == 0 for section in triage.sections[:2])
    assert triage.canvas.next_action_is_canvas is True
    assert triage.canvas.next_action.label
    by_id = {s.section_id: s for s in triage.sections}
    assert by_id["attention"].empty_copy == "No approvals or failures pending."
    assert triage.details_lines  # system status relocated here


def test_triage_flashcards_due_row_appears_in_attention_section_when_count_positive():
    triage = build_home_triage_state(
        HomeDashboardInput(model_ready=True, flashcards_due_count=12),
        now=_NOW,
    )
    by_id = {section.section_id: section for section in triage.sections}
    attention = by_id["attention"]
    row = next(r for r in attention.rows if r.row_id == HOME_FLASHCARDS_DUE_ROW_ID)

    assert attention.count == 1
    assert row.title == "Flashcards due: 12"
    assert row.source == "Library"
    assert row.detail_route == "study"


def test_triage_flashcards_due_row_absent_when_count_zero():
    triage = build_home_triage_state(
        HomeDashboardInput(model_ready=True, flashcards_due_count=0),
        now=_NOW,
    )
    by_id = {section.section_id: section for section in triage.sections}
    row_ids = [row.row_id for row in by_id["attention"].rows]

    assert HOME_FLASHCARDS_DUE_ROW_ID not in row_ids


def test_triage_selecting_flashcards_due_row_builds_canvas_without_stopiteration():
    triage = build_home_triage_state(
        HomeDashboardInput(model_ready=True, flashcards_due_count=12),
        selected_row_id=HOME_FLASHCARDS_DUE_ROW_ID,
        now=_NOW,
    )

    assert triage.selected_row_id == HOME_FLASHCARDS_DUE_ROW_ID
    assert triage.canvas.title == "Flashcards due: 12"
    assert triage.canvas.next_action_is_canvas is False


def test_build_home_controls_includes_review_flashcards_control_when_due():
    controls = build_home_controls(HomeDashboardInput(model_ready=True, flashcards_due_count=12))
    control = next(c for c in controls if c.control_id == "home-review-flashcards")

    assert control.label == "Review flashcards"
    assert control.target_route == "study"
    assert control.applies_to == "flashcards_due"
    assert control.target_id is None


def test_build_home_controls_omits_review_flashcards_control_when_not_due():
    controls = build_home_controls(HomeDashboardInput(model_ready=True, flashcards_due_count=0))

    assert all(c.control_id != "home-review-flashcards" for c in controls)


# --- C1: primary emphasis follows the selected row -------------------------


def test_canvas_primary_control_follows_selected_failed_item():
    triage = build_home_triage_state(_items_input(), selected_row_id="sched:fail-1", now=_NOW)

    assert triage.canvas.primary_control_id == "home-retry"
    assert any(c.control_id == "home-review-flashcards" for c in triage.canvas.actions) is False


def test_canvas_primary_control_follows_selected_approval_item():
    triage = build_home_triage_state(_items_input(), now=_NOW)

    assert triage.selected_row_id == "wf:approve-1"
    assert triage.canvas.primary_control_id == "home-approve"


def test_canvas_primary_control_follows_selected_flashcards_due_row():
    triage = build_home_triage_state(
        HomeDashboardInput(model_ready=True, flashcards_due_count=12),
        selected_row_id=HOME_FLASHCARDS_DUE_ROW_ID,
        now=_NOW,
    )

    assert triage.canvas.primary_control_id == "home-review-flashcards"


def test_canvas_primary_control_defers_to_open_details_for_running_item():
    triage = build_home_triage_state(_items_input(), selected_row_id="watch:run-1", now=_NOW)

    assert triage.selected_row_id == "watch:run-1"
    assert triage.canvas.primary_control_id == "home-open-details"


def test_canvas_primary_control_is_empty_when_nothing_selectable():
    triage = build_home_triage_state(HomeDashboardInput(model_ready=True), now=_NOW)

    assert triage.selected_row_id == ""
    assert triage.canvas.primary_control_id == ""


def test_canvas_omits_retry_control_when_selected_item_retry_unavailable():
    """(M4, fix batch F1b) A permanently-failed item (e.g. a Library ingest
    job with an unsupported file type) offers no Retry control at all --
    not even pointed at another failed item."""
    state = _items_input(
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:job-1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
                retry_available=False,
            ),
        )
    )
    triage = build_home_triage_state(state, selected_row_id="local:ingest:job-1", now=_NOW)

    control_ids = {c.control_id for c in triage.canvas.actions}
    assert "home-retry" not in control_ids
    assert triage.canvas.primary_control_id != "home-retry"


def test_canvas_keeps_retry_control_when_selected_item_retry_available():
    """(M4) An ordinary failure (retry_available defaults True) is
    unaffected -- Retry stays present and primary."""
    state = _items_input(
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:job-1",
                title="report.txt",
                source="Library",
                status="failed",
                detail_route="library",
            ),
        )
    )
    triage = build_home_triage_state(state, selected_row_id="local:ingest:job-1", now=_NOW)

    control_ids = {c.control_id for c in triage.canvas.actions}
    assert "home-retry" in control_ids
    assert triage.canvas.primary_control_id == "home-retry"


def test_build_home_controls_selected_item_retry_unavailable_omits_home_retry():
    """Unit-level check directly on build_home_controls (not just through
    the triage builder)."""
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:job-1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
                retry_available=False,
            ),
        ),
    )
    selected = state.active_work_items[0]

    controls = build_home_controls(state, selected_row_id=selected.item_id, selected_item=selected)

    assert all(c.control_id != "home-retry" for c in controls)


def test_build_home_controls_without_selected_item_keeps_default_retry_behavior():
    """No ``selected_item`` passed (every pre-M4 caller) -- Retry stays
    unconditionally offered for any failed run, unaffected by the new
    parameter's default."""
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:job-1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
                retry_available=False,
            ),
        ),
    )

    controls = build_home_controls(state)

    assert any(c.control_id == "home-retry" for c in controls)


def test_build_home_controls_selected_recent_only_item_gets_open_details():
    """T153: a selected item that lives ONLY in ``recent_work_items`` (a
    finished import, a chatbook artifact) bumps no active/failed/running/
    paused count, so the count-driven block that emits ``home-open-details``
    is skipped entirely -- before the fix, this leaves the selected item
    with no control to open it at all.
    """
    recent_item = HomeActiveWorkItem(
        item_id="local:ingest:done-1",
        title="report.pdf",
        source="Library",
        status="done",
        detail_route="library",
    )
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        recent_work_items=(recent_item,),
    )

    controls = build_home_controls(
        state, selected_row_id=recent_item.item_id, selected_item=recent_item
    )

    controls_by_id = {c.control_id: c for c in controls}
    assert "home-open-details" in controls_by_id
    assert controls_by_id["home-open-details"].target_id == recent_item.item_id
    assert controls_by_id["home-open-details"].target_route == recent_item.detail_route
    assert controls_by_id["home-open-details"].applies_to == "work_details"


def test_build_home_controls_open_details_targets_selected_failed_item_not_first():
    """PR #600 review (Gemini): the count-driven ``home-open-details`` must
    target the SELECTED item, not whichever ``choose_home_selected_item``
    ranks first. With 2+ failed items, selecting the second and opening its
    details must open the second's details -- the same selected-item scoping
    Retry/Pause already have.
    """
    first = HomeActiveWorkItem(
        item_id="fail-1", title="first", source="Runs", status="failed",
        detail_route="chat",
    )
    second = HomeActiveWorkItem(
        item_id="fail-2", title="second", source="Runs", status="failed",
        detail_route="chat",
    )
    state = HomeDashboardInput(
        model_ready=True,
        failed_run_count=2,
        active_work_items=(first, second),
    )

    controls = build_home_controls(
        state, selected_row_id=second.item_id, selected_item=second
    )

    controls_by_id = {c.control_id: c for c in controls}
    assert "home-open-details" in controls_by_id
    assert controls_by_id["home-open-details"].target_id == second.item_id


def test_triage_selecting_recent_only_row_builds_canvas_with_open_details():
    """Same defect as above, exercised through the full triage builder (the
    real production path: ``build_home_triage_state`` resolves the selected
    row's backing item from ``active_work_items + recent_work_items`` and
    threads it into ``build_home_controls`` as ``selected_item``).
    """
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        recent_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:done-1",
                title="report.pdf",
                source="Library",
                status="done",
                detail_route="library",
            ),
        ),
    )

    triage = build_home_triage_state(
        state, selected_row_id="local:ingest:done-1", now=_NOW
    )

    controls_by_id = {c.control_id: c for c in triage.canvas.actions}
    assert "home-open-details" in controls_by_id
    assert controls_by_id["home-open-details"].target_id == "local:ingest:done-1"
    assert controls_by_id["home-open-details"].target_route == "library"


def test_build_home_controls_does_not_duplicate_open_details_for_active_selection():
    """Guard against a double ``home-open-details``: when the selected item
    is already covered by the count-driven block (a real active/failed
    item), the recent-only fallback must not append a second control."""
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:watchlist_run:5",
                title="Daily security feed",
                source="W+C",
                status="failed",
                detail_route="subscriptions",
            ),
        ),
    )
    selected = state.active_work_items[0]

    controls = build_home_controls(
        state, selected_row_id=selected.item_id, selected_item=selected
    )

    open_details_controls = [c for c in controls if c.control_id == "home-open-details"]
    assert len(open_details_controls) == 1
    assert open_details_controls[0].target_id == selected.item_id


def test_canvas_primary_control_flips_between_failed_item_and_flashcards_row():
    """Selecting a different row moves primary emphasis with it (no button
    stays permanently accented) -- mirrors the live pilot: failed ingest
    selected -> Retry primary, Review flashcards not; select the
    flashcards row -> flips.

    H2 (fix batch F1b): when a real work item is selected, the global
    "Review flashcards" shortcut is scoped out of its canvas entirely (not
    merely non-primary) -- it has nothing to do with the selected item, so
    it no longer coexists with that item's own controls."""
    state = _items_input(flashcards_due_count=5)

    failed_triage = build_home_triage_state(state, selected_row_id="sched:fail-1", now=_NOW)
    assert failed_triage.canvas.primary_control_id == "home-retry"
    control_ids = {c.control_id for c in failed_triage.canvas.actions}
    assert "home-review-flashcards" not in control_ids  # scoped out of the item canvas

    flashcards_triage = build_home_triage_state(
        state, selected_row_id=HOME_FLASHCARDS_DUE_ROW_ID, now=_NOW
    )
    assert flashcards_triage.canvas.primary_control_id == "home-review-flashcards"
    flashcards_control_ids = {c.control_id for c in flashcards_triage.canvas.actions}
    assert "home-review-flashcards" in flashcards_control_ids


# --- C2: status is stated once on the Home canvas ---------------------------


def test_canvas_lines_merge_status_and_source_once_for_selected_item():
    triage = build_home_triage_state(_items_input(), selected_row_id="wf:approve-1", now=_NOW)

    assert triage.canvas.lines == (
        "● pending_approval · Workflows · since 3m",
        "Opens: Workflows",
    )


def test_canvas_lines_merge_status_and_source_for_running_item():
    triage = build_home_triage_state(_items_input(), selected_row_id="watch:run-1", now=_NOW)

    assert triage.canvas.lines == (
        "● running · Watchlists · since now",
        "Opens: Watchlists",
    )


def test_canvas_lines_omit_since_clause_when_age_is_blank():
    triage = build_home_triage_state(
        _items_input(
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="x:1",
                    title="No timestamp item",
                    source="ACP",
                    status="running",
                ),
            )
        ),
        selected_row_id="x:1",
        now=_NOW,
    )

    assert triage.canvas.lines == ("● running · ACP", "Opens: Console")


def test_canvas_lines_for_flashcards_due_row_are_sensible_not_duplicated():
    triage = build_home_triage_state(
        HomeDashboardInput(model_ready=True, flashcards_due_count=12),
        selected_row_id=HOME_FLASHCARDS_DUE_ROW_ID,
        now=_NOW,
    )

    # L7: "Library" alone read as a source/destination mismatch (flashcards
    # aren't literally filed in the Library tab) -- the copy now names the
    # actual feature (Study decks) while still crediting where the due
    # count comes from.
    assert triage.canvas.lines == (
        "● due for review · Study decks in Library",
        "Opens: Study",
    )


# --- H2: the global flashcards shortcut is scoped to "no real item selected" ---


def test_build_home_controls_omits_review_flashcards_when_a_row_id_is_selected():
    controls = build_home_controls(
        HomeDashboardInput(model_ready=True, flashcards_due_count=12),
        selected_row_id="local:ingest:1",
    )

    assert all(c.control_id != "home-review-flashcards" for c in controls)


def test_build_home_controls_keeps_review_flashcards_when_flashcards_row_selected():
    controls = build_home_controls(
        HomeDashboardInput(model_ready=True, flashcards_due_count=12),
        selected_row_id=HOME_FLASHCARDS_DUE_ROW_ID,
    )

    assert any(c.control_id == "home-review-flashcards" for c in controls)


def test_build_home_controls_keeps_review_flashcards_when_nothing_selected():
    """Default (no ``selected_row_id``) preserves today's count-only-path
    behavior -- the control stays reachable regardless of selection."""
    controls = build_home_controls(HomeDashboardInput(model_ready=True, flashcards_due_count=12))

    assert any(c.control_id == "home-review-flashcards" for c in controls)


def test_triage_count_only_canvas_still_offers_review_flashcards():
    """The count-only fallback canvas (no selectable rail row) keeps
    exposing Review flashcards -- H2 only scopes it out of a *selected real
    item's* canvas, not the no-selection path."""
    triage = build_home_triage_state(
        HomeDashboardInput(model_ready=True, flashcards_due_count=12, has_library_content=True),
        now=_NOW,
    )

    assert triage.selected_row_id == ""
    assert any(c.control_id == "home-review-flashcards" for c in triage.canvas.actions)


# --- H3: the Next hint must not duplicate the selected item's own recovery ---


def test_choose_next_best_action_exclude_suppresses_review_failed_work():
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
            ),
        ),
    )

    default_action = choose_next_best_action(state)
    assert default_action.action_id == "review_failed_work"

    suppressed_action = choose_next_best_action(state, exclude=frozenset({"review_failed_work"}))
    assert suppressed_action.action_id != "review_failed_work"


def test_triage_next_hint_suppresses_duplicate_recovery_for_selected_failed_item():
    """Selecting a failed ingest item whose own canvas already offers Retry
    must not also tell the user, via the Next hint, to do the very same
    thing -- the engine falls through to its next branch instead."""
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
            ),
        ),
    )

    triage = build_home_triage_state(state, selected_row_id="local:ingest:1", now=_NOW)

    assert triage.canvas.next_action.action_id != "review_failed_work"


def test_triage_next_hint_still_suggests_recovery_when_nothing_selected():
    """A genuine no-selection state keeps the unsuppressed suggestion.

    Count-only input (``failed_run_count`` without any ``active_work_items``)
    produces no selectable rail rows at all, so ``build_home_triage_state``
    takes its count-only fallback branch -- which is not any failed item's
    own canvas, so the H3 suppression must NOT fire there and the Next hint
    stays ``review_failed_work``.

    Note: an unknown ``selected_row_id`` with a failed item present is NOT a
    no-selection state -- ``choose_home_selected_item`` falls back to the
    failed item itself, which (correctly) triggers the suppression. Only the
    empty-rows path exercises the unselected behavior.
    """
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        failed_run_count=1,
    )

    triage = build_home_triage_state(state, now=_NOW)

    assert triage.selected_row_id == ""
    assert triage.canvas.next_action.action_id == "review_failed_work"


def test_triage_suppressed_next_hint_skips_false_resume_claim_when_nothing_running():
    """(F1b whole-wave review, live QA) With a failed item selected and
    NOTHING running, the H3 suppression must not fall through to
    ``resume_active_work`` -- its copy ("Live work is already running.") is
    false then, and the Running rail section says "Nothing running right
    now." directly beside it. ``_active_run_count`` counts failed/queued
    attention items too, which is why the branch would otherwise win. The
    recompute excludes it as well whenever ``running_run_count`` is 0."""
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
            ),
        ),
    )

    triage = build_home_triage_state(state, selected_row_id="local:ingest:1", now=_NOW)

    assert triage.canvas.next_action.action_id not in {
        "review_failed_work",
        "resume_active_work",
    }


def test_triage_suppressed_next_hint_allows_resume_when_work_is_running():
    """Counterpart to the false-claim guard above: when something genuinely
    IS running, ``resume_active_work`` is an honest fallthrough and stays
    allowed after ``review_failed_work`` is suppressed."""
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
            ),
            HomeActiveWorkItem(
                item_id="watch:run-1",
                title="Watchlist sweep",
                source="Watchlists",
                status="running",
                detail_route="watchlists",
            ),
        ),
    )

    triage = build_home_triage_state(state, selected_row_id="local:ingest:1", now=_NOW)

    assert triage.canvas.next_action.action_id == "resume_active_work"


def test_triage_next_hint_allows_resume_when_sibling_ingest_job_is_parsing():
    """(F3 Task-3 reviewer's guard note) The generic ``"running"`` sibling
    in ``test_triage_suppressed_next_hint_allows_resume_when_work_is_running``
    above proves the H3 fallthrough works for *some* running-category
    status -- this closes the gap for the Library ingest queue's own new
    F3 sub-state specifically: a sibling ingest job that is ``PARSING``
    (not yet ``WRITING``) must count as live work too, so
    ``resume_active_work`` still fires instead of the suppression falling
    all the way through to nothing. (The "nothing active" counterpart --
    a selected FAILED ingest item with no active sibling at all -- is
    already covered by
    ``test_triage_suppressed_next_hint_skips_false_resume_claim_when_nothing_running``
    above, which uses the same single-ingest-item shape.)"""
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="local:ingest:1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
            ),
            HomeActiveWorkItem(
                item_id="local:ingest:2",
                title="chapter-two.pdf",
                source="Library",
                status="parsing",
                detail_route="library",
            ),
        ),
    )

    triage = build_home_triage_state(state, selected_row_id="local:ingest:1", now=_NOW)

    assert triage.canvas.next_action.action_id == "resume_active_work"


def test_triage_next_hint_not_suppressed_when_routes_differ():
    """The suppression only fires when the Next hint's target route matches
    the selected failed item's own route -- a failed item routed elsewhere
    still gets the (non-duplicate) suggestion."""
    state = HomeDashboardInput(
        model_ready=True,
        has_library_content=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="sched:fail-1",
                title="Retry: ingest failure",
                source="Schedules",
                status="failed",
                detail_route="schedules",
            ),
            HomeActiveWorkItem(
                item_id="local:ingest:1",
                title="report.xyz",
                source="Library",
                status="failed",
                detail_route="library",
            ),
        ),
    )
    # The engine's failed-item resolution picks the *first* failed item
    # (sched:fail-1) for its route, regardless of which one is selected.
    triage = build_home_triage_state(state, selected_row_id="local:ingest:1", now=_NOW)

    assert triage.canvas.next_action.action_id == "review_failed_work"
    assert triage.canvas.next_action.target_route == "schedules"


# --- M1: failure reason on the Home canvas -----------------------------------


def test_canvas_lines_include_failure_reason_for_selected_failed_item_with_detail():
    triage = build_home_triage_state(
        _items_input(
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="local:ingest:1",
                    title="report.xyz",
                    source="Library",
                    status="failed",
                    detail_route="library",
                    status_detail="Unsupported extension",
                ),
            )
        ),
        selected_row_id="local:ingest:1",
        now=_NOW,
    )

    assert triage.canvas.lines == (
        "● failed · Library",
        "Unsupported extension",
        "Opens: Library",
    )


def test_canvas_lines_omit_status_detail_line_when_blank():
    triage = build_home_triage_state(_items_input(), selected_row_id="sched:fail-1", now=_NOW)

    assert triage.canvas.lines == (
        "● failed · Schedules · since 1h",
        "Opens: Schedules",
    )


def test_canvas_status_detail_truncated_to_140_chars_with_ellipsis():
    long_reason = "x" * 200
    triage = build_home_triage_state(
        _items_input(
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="local:ingest:1",
                    title="report.xyz",
                    source="Library",
                    status="failed",
                    detail_route="library",
                    status_detail=long_reason,
                ),
            )
        ),
        selected_row_id="local:ingest:1",
        now=_NOW,
    )

    detail_line = triage.canvas.lines[1]
    assert len(detail_line) == 140
    assert detail_line.endswith("…")
    assert detail_line[:-1] == "x" * 139
