"""Frozen before/after evidence for TASK-20937.2 conversation projections."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID


BASELINE_PATH = Path(__file__).with_name("fixtures") / (
    "console_workspace_tree_old_baseline.json"
)
FIXTURE_SEED = 20937
SOURCE_COMMIT = "5729439e5ad4fe0959b59a1fe699ef9ee3ebb2f8"
BASELINE_SHA256 = "140db572a9284b4cb6871483eab0ed720a2f2b417fb6a3d3ed08e1f26c909f34"
TERMINAL_SIZE = {"columns": 180, "rows": 52}
CASES = {
    "small": (3, 4, 4, 1, 4),
    "representative": (12, 12, 20, 3, 10),
    "stress": (50, 75, 75, 10, 10),
}
ITERATIONS = 20
WARMUPS = 3


def _rows(case: str) -> tuple[ConsoleConversationBrowserInputRow, ...]:
    workspace_count, per_workspace, flat_count, _, hit_modulus = CASES[case]
    rows: list[ConsoleConversationBrowserInputRow] = []
    ordinal = 0
    for workspace_index in range(workspace_count):
        workspace_id = f"ws-{workspace_index:03d}"
        for conversation_index in range(per_workspace):
            ordinal += 1
            rows.append(
                _row(
                    ordinal,
                    workspace_id=workspace_id,
                    workspace_label=f"Workspace [{workspace_index:03d}]",
                    title=(
                        f"Needle {ordinal:05d}"
                        if ordinal % hit_modulus == 0
                        else f"Conversation {ordinal:05d}"
                    ),
                )
            )
    for flat_index in range(flat_count):
        ordinal += 1
        rows.append(
            _row(
                ordinal,
                workspace_id=(DEFAULT_WORKSPACE_ID if flat_index % 2 else None),
                workspace_label=("Default" if flat_index % 2 else "Chats"),
                scope_type=("workspace" if flat_index % 2 else "global"),
                title=(
                    f"Needle {ordinal:05d}"
                    if ordinal % hit_modulus == 0
                    else f"Conversation {ordinal:05d}"
                ),
            )
        )
    return tuple(rows)


def _row(
    ordinal: int,
    *,
    workspace_id: str | None,
    workspace_label: str,
    title: str,
    scope_type: str = "workspace",
) -> ConsoleConversationBrowserInputRow:
    return ConsoleConversationBrowserInputRow(
        row_key=f"conversation:{ordinal:05d}",
        conversation_id=f"conversation-{ordinal:05d}",
        native_session_id=None,
        title=title,
        scope_type=scope_type,
        workspace_id=workspace_id,
        workspace_label=workspace_label,
        status="workspace-thread",
        selected=ordinal == 1,
        starred=ordinal % 7 == 0,
        starred_sort=f"2026-08-{1 + ordinal % 22:02d}T12:00:00+00:00",
        updated_sort=f"2026-08-{1 + ordinal % 22:02d}T11:00:00+00:00",
        run_marker="●" if ordinal % 19 == 0 else "",
    )


def _fixture_digest() -> str:
    fixture = {
        case: {
            "configuration": CASES[case],
            "rows": [asdict(row) for row in _rows(case)],
        }
        for case in CASES
    }
    encoded = json.dumps(
        fixture, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _percentile_95(samples: list[float]) -> float:
    return sorted(samples)[math.ceil(0.95 * len(samples)) - 1]


def test_old_projection_baseline_is_reproducible() -> None:
    """Load/checksum frozen evidence without invoking the changed projection."""
    baseline_bytes = BASELINE_PATH.read_bytes()
    assert hashlib.sha256(baseline_bytes).hexdigest() == BASELINE_SHA256
    baseline = json.loads(baseline_bytes)
    assert baseline["schema"] == "console-workspace-tree-old-projection-baseline-v1"
    metadata = baseline["metadata"]
    assert metadata["fixture_seed"] == FIXTURE_SEED
    assert metadata["fixture_sha256"] == _fixture_digest()
    assert metadata["source_commit"] == SOURCE_COMMIT
    assert metadata["terminal_size"] == TERMINAL_SIZE
    assert metadata["warmups"] == WARMUPS
    assert metadata["measured_iterations"] == ITERATIONS
    assert metadata["python_version"] == "3.12.11"
    assert metadata["textual_version"] == "8.2.8"
    assert metadata["machine"] == "macOS-15.6-arm64-arm-64bit"
    assert metadata["architecture"] == "arm64"
    for case, result in baseline["cases"].items():
        assert result["total_service_record_count"] == len(_rows(case))
        assert result["materialized_row_count"] >= 0
        assert result["reconcile_count"] == ITERATIONS * 5
        assert result["recompose_count"] == ITERATIONS * 5
        for operation, samples in result["raw_samples_ms"].items():
            assert len(samples) == ITERATIONS
            assert result["summary_ms"][operation] == {
                "median": round(statistics.median(samples), 6),
                "p95": round(_percentile_95(samples), 6),
            }
