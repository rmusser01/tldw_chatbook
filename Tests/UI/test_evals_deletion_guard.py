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

#: Repo-relative paths removed by PR 1. Tasks 2 and 3 extend this tuple.
REMOVED_MODULES: tuple[str, ...] = (
    "tldw_chatbook/UI/ResultsDashboardWindow.py",
    "tldw_chatbook/UI/ModelManagementWindow.py",
    "tldw_chatbook/UI/DatasetManagementWindow.py",
    "tldw_chatbook/UI/Views/evals_views.py",
    "tldw_chatbook/Event_Handlers/eval_events.py",
)

#: Module basenames that must not appear in any import statement.
REMOVED_STEMS: tuple[str, ...] = (
    "ResultsDashboardWindow",
    "ModelManagementWindow",
    "DatasetManagementWindow",
    "evals_views",
    "eval_events",
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
