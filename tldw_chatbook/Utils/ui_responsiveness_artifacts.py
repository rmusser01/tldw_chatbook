"""Write Workbench responsiveness diagnostics artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path

from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessSnapshot


REQUIRED_RESPONSIVENESS_ARTIFACTS = (
    "ui_heartbeat.log",
    "worker_snapshot.log",
    "timer_registry.log",
    "mount_churn_summary.log",
    "route_switch_soak_result.txt",
)


def _responsiveness_artifact_roots() -> tuple[Path, ...]:
    """Return base directories allowed to receive diagnostics artifacts."""
    roots: list[Path] = []
    for root in (Path.cwd(), Path(tempfile.gettempdir())):
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _validated_output_dir(output_dir: Path) -> Path:
    """Resolve and validate an artifact output directory before writing."""
    candidate = validate_path_simple(output_dir)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (Path.cwd() / candidate).resolve()

    roots = _responsiveness_artifact_roots()
    for base_directory in roots:
        try:
            resolved.relative_to(base_directory)
        except ValueError:
            continue
        return resolved

    allowed_roots = ", ".join(str(root) for root in roots)
    raise ValueError(
        f"Responsiveness artifact output must be under one of: {allowed_roots}"
    )


def write_responsiveness_artifacts(
    output_dir: Path,
    snapshot: UIResponsivenessSnapshot,
    *,
    route_switch_summary: str,
) -> None:
    """Write the required freeze-diagnostics artifacts for a soak run.

    Args:
        output_dir: Directory that will receive the artifact files.
        snapshot: Responsiveness monitor snapshot to serialize.
        route_switch_summary: Human-readable route-switch soak result summary.

    Raises:
        ValueError: If ``output_dir`` fails path validation.
    """
    output_dir = _validated_output_dir(output_dir)
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
