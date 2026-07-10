from tldw_chatbook.Home.dashboard_state import (
    HOME_FLASHCARDS_DUE_ROW_ID,
    HomeActiveWorkItem,
    HomeDashboardInput,
    build_home_controls,
    build_home_triage_state,
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


def test_canvas_primary_control_flips_between_failed_item_and_flashcards_row():
    """Selecting a different row moves primary emphasis with it (no button
    stays permanently accented) -- mirrors the live pilot: failed ingest
    selected -> Retry primary, Review flashcards not; select the
    flashcards row -> flips."""
    state = _items_input(flashcards_due_count=5)

    failed_triage = build_home_triage_state(state, selected_row_id="sched:fail-1", now=_NOW)
    assert failed_triage.canvas.primary_control_id == "home-retry"
    control_ids = {c.control_id for c in failed_triage.canvas.actions}
    assert "home-review-flashcards" in control_ids  # both controls coexist
    assert failed_triage.canvas.primary_control_id != "home-review-flashcards"

    flashcards_triage = build_home_triage_state(
        state, selected_row_id=HOME_FLASHCARDS_DUE_ROW_ID, now=_NOW
    )
    assert flashcards_triage.canvas.primary_control_id == "home-review-flashcards"


# --- C2: status is stated once on the Home canvas ---------------------------


def test_canvas_lines_merge_status_and_source_once_for_selected_item():
    triage = build_home_triage_state(_items_input(), selected_row_id="wf:approve-1", now=_NOW)

    assert triage.canvas.lines == (
        "● pending_approval · Workflows · since 3m",
        "Route: workflows",
    )


def test_canvas_lines_merge_status_and_source_for_running_item():
    triage = build_home_triage_state(_items_input(), selected_row_id="watch:run-1", now=_NOW)

    assert triage.canvas.lines == (
        "● running · Watchlists · since now",
        "Route: watchlists",
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

    assert triage.canvas.lines == ("● running · ACP", "Route: chat")


def test_canvas_lines_for_flashcards_due_row_are_sensible_not_duplicated():
    triage = build_home_triage_state(
        HomeDashboardInput(model_ready=True, flashcards_due_count=12),
        selected_row_id=HOME_FLASHCARDS_DUE_ROW_ID,
        now=_NOW,
    )

    assert triage.canvas.lines == (
        "● due for review · Library",
        "Route: study",
    )
