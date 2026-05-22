"""ACP runtime process manager tests."""

from __future__ import annotations

import subprocess
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


def test_acp_runtime_process_config_parses_quoted_args_and_invalid_timeout_safely():
    config = ACPRuntimeProcessConfig.from_mapping(
        {
            "command": sys.executable,
            "args": '-m "agent runtime" --label "QA Agent"',
            "startup_timeout_seconds": "two",
        }
    )

    assert config.args == ("-m", "agent runtime", "--label", "QA Agent")
    assert config.startup_timeout_seconds == 2.0


def test_acp_runtime_process_config_clamps_tiny_timeout():
    config = ACPRuntimeProcessConfig.from_mapping(
        {
            "command": sys.executable,
            "startup_timeout_seconds": "-1",
        }
    )

    assert config.startup_timeout_seconds == 0.05


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


def test_acp_runtime_process_manager_rejects_invalid_cwd(tmp_path):
    config = ACPRuntimeProcessConfig(
        command=sys.executable,
        args=("-c", "import time; time.sleep(30)"),
        cwd=str(tmp_path / "missing"),
        runtime_id="test-runtime",
        runtime_label="Test ACP Runtime",
        startup_timeout_seconds=0.2,
    )
    manager = ACPRuntimeProcessManager(config=config)

    result = manager.start_session(title="QA ACP session")

    assert result.status == ACPRuntimeProcessStatus.FAILED
    assert "Invalid ACP runtime cwd" in result.recovery
    assert result.session_state.has_console_session_payload is False


def test_acp_runtime_process_manager_stop_handles_process_that_survives_kill():
    class StubbornProcess:
        pid = 1234

        def __init__(self):
            self.terminate_calls = 0
            self.kill_calls = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

        def wait(self, timeout):
            raise subprocess.TimeoutExpired(cmd="stubborn-acp", timeout=timeout)

    config = ACPRuntimeProcessConfig(command=sys.executable)
    manager = ACPRuntimeProcessManager(config=config)
    process = StubbornProcess()
    manager._process = process
    manager._status = ACPRuntimeProcessStatus.RUNNING

    result = manager.stop()

    assert result.status == ACPRuntimeProcessStatus.FAILED
    assert "did not stop after kill" in result.recovery
    assert process.terminate_calls == 1
    assert process.kill_calls == 1


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
