from tldw_chatbook.Home.active_work_adapter import (
    HomeControlAction,
    HomeControlResultStatus,
    UnavailableHomeActiveWorkAdapter,
)


def test_unavailable_home_adapter_builds_dashboard_input_from_runtime_context():
    adapter = UnavailableHomeActiveWorkAdapter()

    dashboard_input = adapter.build_dashboard_input(
        providers_models={"OpenAI": ["gpt-4.1"]},
        has_recent_work=True,
    )

    assert dashboard_input.model_ready is True
    assert dashboard_input.has_recent_work is True
    assert dashboard_input.pending_approval_count == 0
    assert dashboard_input.active_run_count == 0
    assert dashboard_input.active_detail_route == "chat"


def test_unavailable_home_adapter_returns_honest_recovery_result():
    adapter = UnavailableHomeActiveWorkAdapter()

    result = adapter.handle_control(HomeControlAction.APPROVE)

    assert result.action is HomeControlAction.APPROVE
    assert result.status is HomeControlResultStatus.UNAVAILABLE
    assert result.severity == "warning"
    assert "Approve is not connected" in result.message
    assert result.recovery_route == "chat"
