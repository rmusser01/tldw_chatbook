"""UI responsiveness diagnostics artifact tests."""

from pathlib import Path

from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor
from tldw_chatbook.Utils.ui_responsiveness_artifacts import (
    REQUIRED_RESPONSIVENESS_ARTIFACTS,
    write_responsiveness_artifacts,
)


def test_responsiveness_artifact_writer_creates_required_files(tmp_path: Path):
    monitor = UIResponsivenessMonitor(enabled=True, stall_threshold_ms=250)
    monitor.record_timer_created("ui-heartbeat")
    monitor.record_worker_started("console-sync")
    monitor.record_mounts("console-tabs", mounted=2, removed=1)
    monitor.record_heartbeat_delta(0.03)

    write_responsiveness_artifacts(
        tmp_path,
        monitor.snapshot(),
        route_switch_summary="route switches: 6, failures: 0",
    )

    for filename in REQUIRED_RESPONSIVENESS_ARTIFACTS:
        assert (tmp_path / filename).exists(), filename
    assert "route switches: 6" in (
        tmp_path / "route_switch_soak_result.txt"
    ).read_text(encoding="utf-8")
