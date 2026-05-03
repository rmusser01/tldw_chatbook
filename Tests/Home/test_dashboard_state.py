from tldw_chatbook.Home.dashboard_state import (
    HomeActiveWorkItem,
    HomeDashboardInput,
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
