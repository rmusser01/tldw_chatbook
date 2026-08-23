"""Frozen before/after evidence for TASK-20937.2 conversation projections."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import pytest
import textual
from textual.app import App, ComposeResult
from textual.containers import Vertical

from tldw_chatbook.Workspaces.conversation_browser_state import (
    CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
    ConsoleConversationBrowserInputRow,
)
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.workspace_tree_state import build_workspace_tree_state
from tldw_chatbook.Widgets.Console.console_workspace_tree import ConsoleWorkspaceTree


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


class _MeasuredWorkspaceTree(ConsoleWorkspaceTree):
    """Native Tree with counters around the real settled update seams."""

    def __init__(self) -> None:
        self.reconcile_count = 0
        self.node_refresh_count = 0
        self.recompose_count = 0
        super().__init__()

    def sync_projection(self, *args, **kwargs) -> None:
        self.reconcile_count += 1
        super().sync_projection(*args, **kwargs)

    def _refresh_node(self, node) -> None:
        self.node_refresh_count += 1
        super()._refresh_node(node)

    async def recompose(self) -> None:
        self.recompose_count += 1
        await super().recompose()


class _WorkspaceTreeBenchmarkHarness(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #benchmark-host { width: 100%; height: 100%; }
    ConsoleWorkspaceTree { width: 100%; height: 100%; }
    """

    def compose(self) -> ComposeResult:
        yield Vertical(id="benchmark-host")


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


def _ordinary_projection_rows(
    rows: tuple[ConsoleConversationBrowserInputRow, ...],
    expanded_workspace_ids: set[str],
) -> tuple[ConsoleConversationBrowserInputRow, ...]:
    """Return only the bounded ordinary children mounted for open branches."""

    counts: dict[str, int] = {}
    projected: list[ConsoleConversationBrowserInputRow] = []
    for row in rows:
        workspace_id = str(row.workspace_id or "")
        if workspace_id not in expanded_workspace_ids:
            continue
        count = counts.get(workspace_id, 0)
        if count >= CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT:
            continue
        counts[workspace_id] = count + 1
        projected.append(row)
    return tuple(projected)


def _ordinary_next_cursors(
    rows: tuple[ConsoleConversationBrowserInputRow, ...],
    expanded_workspace_ids: set[str],
) -> dict[str, int | None]:
    totals = {
        workspace_id: sum(row.workspace_id == workspace_id for row in rows)
        for workspace_id in expanded_workspace_ids
    }
    return {
        workspace_id: (
            CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT
            if total > CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT
            else None
        )
        for workspace_id, total in totals.items()
    }


async def _settle_native_tree(pilot) -> None:
    """Wait through deferred node refreshes, layout, paint, and compositor work."""

    await pilot.pause()
    pilot.app.screen._compositor.render_strips()


async def _measure_new_projection(case: str, pilot) -> dict[str, object]:
    rows = _rows(case)
    workspace_count, _, _, expanded_count, _ = CASES[case]
    workspaces = tuple(
        (f"ws-{index:03d}", f"Workspace [{index:03d}]")
        for index in range(workspace_count)
    )
    expanded = {f"ws-{index:03d}" for index in range(expanded_count)}
    ordinary_rows = _ordinary_projection_rows(rows, expanded)
    next_cursors = _ordinary_next_cursors(rows, expanded)
    marker_rows = tuple(
        replace(row, run_marker=("" if row.run_marker else "●"))
        if index % 20 == 0
        else row
        for index, row in enumerate(rows)
    )
    ordinary_marker_rows = _ordinary_projection_rows(marker_rows, expanded)
    selected_rows = tuple(
        replace(row, selected=row.conversation_id == "conversation-00002")
        for row in marker_rows
    )
    ordinary_selected_rows = _ordinary_projection_rows(selected_rows, expanded)
    raw_samples_ms: dict[str, list[float]] = {
        "initial_projection_mount": [],
        "marker_update_5_percent": [],
        "search_apply_clear": [],
        "active_row_selection": [],
    }
    node_refresh_counts = {operation: 0 for operation in raw_samples_ms}
    reconcile_count = 0
    recompose_count = 0
    materialized_node_count = 0
    host = pilot.app.query_one("#benchmark-host", Vertical)

    for iteration in range(WARMUPS + ITERATIONS):
        await host.remove_children()

        started = time.perf_counter()
        tree = _MeasuredWorkspaceTree()
        await host.mount(tree)
        projection = build_workspace_tree_state(
            workspaces=workspaces,
            rows=ordinary_rows,
            next_cursors=next_cursors,
            active_workspace_id="ws-000",
        )
        before_refreshes = tree.node_refresh_count
        tree.sync_projection(projection, expanded_workspace_ids=expanded)
        await _settle_native_tree(pilot)
        initial_ms = (time.perf_counter() - started) * 1_000
        initial_refreshes = tree.node_refresh_count - before_refreshes
        materialized_node_count = (
            len(tree.workspace_nodes)
            + len(tree.conversation_nodes)
            + len(tree.auxiliary_nodes)
        )

        started = time.perf_counter()
        marker_projection = build_workspace_tree_state(
            workspaces=workspaces,
            rows=ordinary_marker_rows,
            next_cursors=next_cursors,
            active_workspace_id="ws-000",
        )
        before_refreshes = tree.node_refresh_count
        tree.sync_projection(marker_projection, expanded_workspace_ids=expanded)
        await _settle_native_tree(pilot)
        marker_ms = (time.perf_counter() - started) * 1_000
        marker_refreshes = tree.node_refresh_count - before_refreshes

        started = time.perf_counter()
        search_projection = build_workspace_tree_state(
            workspaces=workspaces,
            rows=marker_rows,
            active_workspace_id="ws-000",
            query="needle",
        )
        tree.set_search_active(
            True,
            forced_workspace_ids={
                workspace.workspace_id for workspace in search_projection
            },
        )
        before_refreshes = tree.node_refresh_count
        tree.sync_projection(search_projection, expanded_workspace_ids=expanded)
        await _settle_native_tree(pilot)
        clear_projection = build_workspace_tree_state(
            workspaces=workspaces,
            rows=ordinary_marker_rows,
            next_cursors=next_cursors,
            active_workspace_id="ws-000",
        )
        tree.sync_projection(clear_projection, expanded_workspace_ids=expanded)
        tree.set_search_active(False)
        await _settle_native_tree(pilot)
        search_ms = (time.perf_counter() - started) * 1_000
        search_refreshes = tree.node_refresh_count - before_refreshes

        started = time.perf_counter()
        selection_projection = build_workspace_tree_state(
            workspaces=workspaces,
            rows=ordinary_selected_rows,
            next_cursors=next_cursors,
            active_workspace_id="ws-000",
        )
        before_refreshes = tree.node_refresh_count
        tree.sync_projection(selection_projection, expanded_workspace_ids=expanded)
        await _settle_native_tree(pilot)
        selection_ms = (time.perf_counter() - started) * 1_000
        selection_refreshes = tree.node_refresh_count - before_refreshes

        if iteration >= WARMUPS:
            raw_samples_ms["initial_projection_mount"].append(round(initial_ms, 6))
            raw_samples_ms["marker_update_5_percent"].append(round(marker_ms, 6))
            raw_samples_ms["search_apply_clear"].append(round(search_ms, 6))
            raw_samples_ms["active_row_selection"].append(round(selection_ms, 6))
            for operation, refreshes in (
                ("initial_projection_mount", initial_refreshes),
                ("marker_update_5_percent", marker_refreshes),
                ("search_apply_clear", search_refreshes),
                ("active_row_selection", selection_refreshes),
            ):
                node_refresh_counts[operation] += refreshes
            reconcile_count += tree.reconcile_count
            recompose_count += tree.recompose_count

    return {
        "total_service_record_count": len(rows),
        "materialized_node_count": materialized_node_count,
        "logical_reconciles_per_iteration": {
            "initial_projection_mount": 1,
            "marker_update_5_percent": 1,
            "search_apply_clear": 2,
            "active_row_selection": 1,
        },
        "reconcile_count": reconcile_count,
        "node_refresh_count": node_refresh_counts,
        "recompose_count": recompose_count,
        "raw_samples_ms": raw_samples_ms,
        "summary_ms": {
            operation: {
                "median": round(statistics.median(samples), 6),
                "p95": round(_percentile_95(samples), 6),
            }
            for operation, samples in raw_samples_ms.items()
        },
    }


@pytest.mark.asyncio
async def test_new_workspace_tree_benchmark_is_deterministic() -> None:
    app = _WorkspaceTreeBenchmarkHarness()
    async with app.run_test(
        size=(TERMINAL_SIZE["columns"], TERMINAL_SIZE["rows"])
    ) as pilot:
        results = {case: await _measure_new_projection(case, pilot) for case in CASES}

    assert {
        case: (
            result["total_service_record_count"],
            result["materialized_node_count"],
            result["reconcile_count"],
            result["recompose_count"],
        )
        for case, result in results.items()
    } == {
        "small": (16, 9, ITERATIONS * 5, 0),
        "representative": (164, 57, ITERATIONS * 5, 0),
        "stress": (3825, 840, ITERATIONS * 5, 0),
    }
    for result in results.values():
        assert result["logical_reconciles_per_iteration"] == {
            "initial_projection_mount": 1,
            "marker_update_5_percent": 1,
            "search_apply_clear": 2,
            "active_row_selection": 1,
        }
        assert result["node_refresh_count"]["initial_projection_mount"] == 0
        assert result["node_refresh_count"]["active_row_selection"] <= ITERATIONS * 2
        for operation, samples in result["raw_samples_ms"].items():
            assert len(samples) == ITERATIONS
            assert result["summary_ms"][operation] == {
                "median": round(statistics.median(samples), 6),
                "p95": round(_percentile_95(samples), 6),
            }

    evidence = {
        "schema": "console-workspace-tree-new-path-benchmark-v1",
        "metadata": {
            "fixture_seed": FIXTURE_SEED,
            "fixture_sha256": _fixture_digest(),
            "baseline_source_commit": SOURCE_COMMIT,
            "baseline_json_sha256": BASELINE_SHA256,
            "terminal_size": TERMINAL_SIZE,
            "warmups": WARMUPS,
            "measured_iterations": ITERATIONS,
            "python_version": platform.python_version(),
            "textual_version": textual.__version__,
            "machine": platform.platform(),
            "architecture": platform.machine(),
            "python_executable": sys.executable,
        },
        "cases": results,
    }
    print(
        "NEW_WORKSPACE_TREE_BENCHMARK="
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    )
