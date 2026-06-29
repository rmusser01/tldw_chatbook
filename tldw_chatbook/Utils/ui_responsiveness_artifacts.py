"""Write Workbench responsiveness diagnostics artifacts."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessSnapshot


REQUIRED_RESPONSIVENESS_ARTIFACTS = (
    "ui_heartbeat.log",
    "worker_snapshot.log",
    "timer_registry.log",
    "mount_churn_summary.log",
    "route_switch_soak_result.txt",
)


def write_responsiveness_artifacts(
    output_dir: Path,
    snapshot: UIResponsivenessSnapshot,
    *,
    route_switch_summary: str,
) -> None:
    """Write the required freeze-diagnostics artifacts for a soak run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ui_heartbeat.log").write_text(
        (
            f"enabled={snapshot.enabled}\n"
            f"max_heartbeat_lag_ms={snapshot.max_heartbeat_lag_ms}\n"
            f"stalled={snapshot.stalled}\n"
        ),
        encoding="utf-8",
    )
    (output_dir / "worker_snapshot.log").write_text(
        f"active_workers={snapshot.active_workers}\n",
        encoding="utf-8",
    )
    (output_dir / "timer_registry.log").write_text(
        f"active_timers={snapshot.active_timers}\n",
        encoding="utf-8",
    )
    (output_dir / "mount_churn_summary.log").write_text(
        f"mounts={snapshot.mounts}\nremoves={snapshot.removes}\n",
        encoding="utf-8",
    )
    (output_dir / "route_switch_soak_result.txt").write_text(
        f"{route_switch_summary}\n",
        encoding="utf-8",
    )
