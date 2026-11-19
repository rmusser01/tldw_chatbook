"""Agent extraction controls under the current owned bootstrap-profile contract."""

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize(
    "case", ["library-provider", "trace-request", "completes", "provider_raises", "cancelled"]
)
async def test_extracted_agent_reply_preserves_original_boundary_controls(
    monkeypatch, case
):
    from Tests.Chat import test_console_chat_controller as controls

    # Reuse the original behavior assertions under a real per-case admitted
    # profile; the legacy nodes themselves predate the raw-source boundary.
    if case == "library-provider":
        await controls.test_run_agent_reply_threads_library_provider_from_factory()
    elif case == "trace-request":
        await controls.test_run_agent_reply_threads_exact_admitted_trace_request_to_bridge()
    else:
        await controls.test_run_agent_reply_closes_the_runs_remote_sessions_at_run_end(
            monkeypatch, case
        )
