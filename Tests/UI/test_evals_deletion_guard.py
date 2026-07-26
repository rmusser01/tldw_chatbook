"""Regression guard: the unreachable Evals UI stays deleted.

PR 1 of the Evals Console rebuild retired an entire second-generation Evals
UI that no reachable code imported, plus the Widgets/Evals files only that
cluster used. The modules referenced each other, so a single stale import
anywhere would drag all ~10k lines back into the import graph without
anything being visibly wrong. This guard fails loudly if that happens.

See Docs/superpowers/specs/2026-07-25-evals-console-rebuild-design.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Repo-relative paths removed by PR 1. Task 3 extends this tuple.
REMOVED_MODULES: tuple[str, ...] = (
    "tldw_chatbook/UI/ResultsDashboardWindow.py",
    "tldw_chatbook/UI/ModelManagementWindow.py",
    "tldw_chatbook/UI/DatasetManagementWindow.py",
    "tldw_chatbook/UI/Views/evals_views.py",
    "tldw_chatbook/Event_Handlers/eval_events.py",
    "tldw_chatbook/Widgets/Evals/Evals_Sidebar.py",
    "tldw_chatbook/Widgets/Evals/ab_test_dialog.py",
    "tldw_chatbook/Widgets/Evals/ab_test_results_widget.py",
    "tldw_chatbook/Widgets/Evals/dataset_validation_dialog.py",
    "tldw_chatbook/Widgets/Evals/eval_cost_monitor.py",
    "tldw_chatbook/Widgets/Evals/eval_error_dialog.py",
    "tldw_chatbook/Widgets/Evals/eval_smart_suggestions.py",
    "tldw_chatbook/Widgets/Evals/metrics_display.py",
    "tldw_chatbook/Widgets/Evals/cost_estimation_widget.py",
    "tldw_chatbook/Widgets/Evals/eval_config_dialogs.py",
    "tldw_chatbook/Widgets/Evals/eval_results_widgets.py",
)

#: Module basenames that must not appear in any import statement.
REMOVED_STEMS: tuple[str, ...] = (
    "ResultsDashboardWindow",
    "ModelManagementWindow",
    "DatasetManagementWindow",
    "evals_views",
    "eval_events",
    "Evals_Sidebar",
    "ab_test_dialog",
    "ab_test_results_widget",
    "dataset_validation_dialog",
    "eval_cost_monitor",
    "eval_error_dialog",
    "eval_smart_suggestions",
    "metrics_display",
    "cost_estimation_widget",
    "eval_config_dialogs",
    "eval_results_widgets",
)


@pytest.mark.parametrize("rel_path", REMOVED_MODULES)
def test_removed_module_file_is_absent(rel_path: str) -> None:
    """Each retired module stays deleted."""
    assert not (ROOT / rel_path).exists(), (
        f"{rel_path} was retired in PR 1 of the Evals rebuild but exists again. "
        "If it was restored deliberately, update REMOVED_MODULES and say why."
    )


@pytest.mark.parametrize("stem", REMOVED_STEMS)
def test_no_source_imports_removed_module(stem: str) -> None:
    """No production or test source imports a retired module."""
    pattern = re.compile(rf"(?:^|\s)(?:from|import)\s+[\w.]*\b{re.escape(stem)}\b")
    offenders: list[str] = []
    for base in ("tldw_chatbook", "Tests"):
        for path in (ROOT / base).rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"'{stem}' was retired in PR 1 of the Evals rebuild but is still imported:\n"
        + "\n".join(offenders)
    )
