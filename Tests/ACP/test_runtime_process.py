"""ACP runtime process manager tests."""

from __future__ import annotations

import sys
from datetime import datetime

from tldw_chatbook.ACP_Interop.runtime_process import (
    ACPRuntimeProcessConfig,
    ACPRuntimeProcessManager,
    ACPRuntimeProcessStatus,
)


def test_acp_runtime_process_config_rejects_missing_command():
    config = ACPRuntimeProcessConfig.from_mapping({})

    assert config.is_configured is False
    assert config.disabled_reason == "Configure an ACP runtime command in ACP before launch."


def test_acp_runtime_process_manager_launches_and_stops_runtime():
    config = ACPRuntimeProcessConfig(
        command=sys.executable,
        args=("-c", "import time; time.sleep(30)"),
        runtime_id="test-runtime",
        runtime_label="Test ACP Runtime",
        startup_timeout_seconds=1,
    )
    manager = ACPRuntimeProcessManager(config=config)

    try:
        launch = manager.start_session(title="QA ACP session")

        assert launch.status == ACPRuntimeProcessStatus.RUNNING
        assert launch.session_state.runtime_id == "test-runtime"
        assert launch.session_state.runtime_label == "Test ACP Runtime"
        assert launch.session_state.session_title == "QA ACP session"
        assert launch.session_state.has_console_session_payload is True
        assert launch.session_state.session_payload["pid"] > 0
        assert launch.session_state.session_payload["command"] == sys.executable
        started_at = launch.session_state.session_payload["started_at"]
        assert isinstance(started_at, str)
        assert "T" in started_at
        datetime.fromisoformat(started_at)
    finally:
        stop = manager.stop()

    assert stop.status == ACPRuntimeProcessStatus.STOPPED


def test_acp_runtime_process_manager_reports_launch_failure():
    config = ACPRuntimeProcessConfig(
        command=sys.executable,
        args=("-c", "import sys; sys.exit(7)"),
        runtime_id="failing-runtime",
        runtime_label="Failing ACP Runtime",
        startup_timeout_seconds=0.2,
    )
    manager = ACPRuntimeProcessManager(config=config)

    result = manager.start_session(title="Broken session")

    assert result.status == ACPRuntimeProcessStatus.FAILED
    assert "exited before it became ready" in result.recovery
    assert result.session_state.has_console_session_payload is False
