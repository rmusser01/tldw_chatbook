# Evals Rebuild PR 1 — Retire the Unreachable Evals UI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete 10,258 lines of Evals UI that no reachable code imports, and add a guard test that keeps it deleted.

**Architecture:** Pure deletion. Three groups, each gated on a reachability check that is verified before the files are removed: an orphaned second-generation Evals UI cluster (5 modules), the `Widgets/Evals/` files only that cluster imported (11 files), and 3 more widgets whose only remaining importers are tests. A parametrized guard test asserts the paths stay gone and no production import resurrects them. No behaviour changes and no new capability.

**Tech Stack:** Python 3.11+, pytest, Textual. No new dependencies.

## Global Constraints

- Base branch: `origin/dev` at `8242a5b58`. Work in a git worktree, not the primary checkout — many concurrent agents mutate branches there.
- **A git worktree has no `.venv`.** Use the primary checkout's interpreter by absolute path, always with cwd set to the worktree, so the worktree's copy of the package wins over the editable-install pointer:

  ```bash
  cd <worktree> && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ...
  ```

  Confirm resolution before the first test run — `python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"` must print a path inside the worktree. If it prints one in the primary checkout, **stop**: tests would be verifying the wrong tree and every deletion check would be meaningless.
- **`pytest Tests/UI` cannot be run in one call.** It collects 5,183 tests and exceeds the platform's hard 10-minute per-call cap. The per-task gate below preserves its intent — a deletion breaks imports, and broken imports surface as *collection* errors:

  ```bash
  python -m pytest Tests/UI --collect-only -q   # must report 0 collection errors (~2.4s)
  python -m pytest Tests/UI/test_evals_deletion_guard.py -q
  python -c "import tldw_chatbook.app"
  ```

  Full-suite runs are the controller's job, backgrounded, and belong to Task 5. An implementer that finds itself waiting on a backgrounded suite should stop and report instead.
- The `timeout` command is not available in this environment.
- Deletion is gated **per symbol**, never per file assumption. Every group below states the exact importer check that must return empty before deleting.
- No file outside the listed sets is modified, except `tldw_chatbook/app.py` in Task 4.

## Scope corrections to the spec

The spec's deletion table lists two items that **this PR must not touch**. Both were verified during planning and the spec is wrong about them:

**1. `css/features/_evaluation_unified.tcss` stays.** The spec calls it a "288-line legacy sheet" and PR 1 "behaviour neutral." It is not. Its rules are **completely unscoped** — bare `.action-bar`, `.action-button`, `.main-content`, `.status-bar`, `.error`, `.hidden`, `.loading`, `.help-text`, `.section-title`, `.button-row` in a globally-loaded bundle. **21 of its 33 selectors have surviving consumers across the app** (Chat, Logs, MCP, RAG search, Chatbooks, Notes, Coding, Voice Cloning and more). The file itself carries a comment warning that bare rules apply app-wide and outrank widget `DEFAULT_CSS`. Deleting it would silently restyle unrelated screens. Only these 12 selectors are genuinely Evals-only and could ever be dropped:

```
.advanced-config-form  .config-grid            .config-toggles
.cost-display          .dataset-management-form .empty-message
.model-management-form .quick-start-bar        .results-dashboard
.suggestion-text       .system-prompt-editor   .template-editor
```

That is a per-selector migration with its own risk profile. It belongs in PR 3 alongside the screen that replaces the consumers, not in a deletion PR.

**2. The card hub stays.** The spec's table lists `UI/Evals/navigation/`, `evals_window_v3.py`, `UI/Evals/screens/`, `widgets/progress_dashboard.py`, and `UI/evals_window_v2.py`. That is the **routed** Evals screen — `EvalsScreen.compose_content` mounts `EvalsWindowV3` — so it stays out of a deletion PR and is removed in PR 3, in the same change that provides its replacement. This also means `EvalsWindowV3` in the container list (`app.py:1520`) and the `"evals-window"` entry (`app.py:2847`) both **stay** in this PR.

**Correction, from live verification during Task 5:** "live" here means *routed*, not *working*. On the base commit the hub renders an **empty body** inside the app shell — `DestinationHeader` and `LabModeStrip` appear and no cards do. `EvalsWindowV3` mounts correctly in isolation (`EvalNavigationScreen` plus 8 buttons), so this is a shell-integration failure consistent with the `Screen`-inside-a-`Container` architecture the spec flags. Keeping it out of PR 1 is still right — a deletion PR should not be the thing that removes a routed screen — but **PR 3 must not treat the hub as a working surface to preserve parity with.** There is no working behaviour to preserve.

Revised PR 1 total: **19 files, 10,258 lines**, plus 12 dead lines in `app.py`.

## File Structure

**Deleted — Group A, orphan gen-2 UI cluster (5 files, 3,751 lines).** These import only each other; nothing else in the tree imports any of them.

| File | Lines |
|---|---|
| `tldw_chatbook/UI/ResultsDashboardWindow.py` | 477 |
| `tldw_chatbook/UI/ModelManagementWindow.py` | 421 |
| `tldw_chatbook/UI/DatasetManagementWindow.py` | 289 |
| `tldw_chatbook/UI/Views/evals_views.py` | 839 |
| `tldw_chatbook/Event_Handlers/eval_events.py` | 1,725 |

**Deleted — Group B, widgets orphaned by Group A (11 files, 4,473 lines).** Eight have zero importers today; three are imported only by Group A files.

| File | Lines | Only importer |
|---|---|---|
| `tldw_chatbook/Widgets/Evals/Evals_Sidebar.py` | 8 | none |
| `tldw_chatbook/Widgets/Evals/ab_test_dialog.py` | 788 | none |
| `tldw_chatbook/Widgets/Evals/ab_test_results_widget.py` | 401 | none |
| `tldw_chatbook/Widgets/Evals/dataset_validation_dialog.py` | 501 | none |
| `tldw_chatbook/Widgets/Evals/eval_cost_monitor.py` | 333 | none |
| `tldw_chatbook/Widgets/Evals/eval_error_dialog.py` | 368 | none |
| `tldw_chatbook/Widgets/Evals/eval_smart_suggestions.py` | 556 | none |
| `tldw_chatbook/Widgets/Evals/metrics_display.py` | 61 | none |
| `tldw_chatbook/Widgets/Evals/cost_estimation_widget.py` | 395 | `evals_views` (Group A) |
| `tldw_chatbook/Widgets/Evals/eval_config_dialogs.py` | 486 | `eval_events` (Group A) |
| `tldw_chatbook/Widgets/Evals/eval_results_widgets.py` | 576 | `eval_events`, `ResultsDashboardWindow`, `evals_views` (Group A) |

**Deleted — Group C, widgets whose only surviving importers are tests (3 files, 2,034 lines).**

| File | Lines | Test importers |
|---|---|---|
| `tldw_chatbook/Widgets/Evals/eval_additional_dialogs.py` | 551 | `test_bulk_selection_tooltips`, `test_file_picker_action_tooltips` |
| `tldw_chatbook/Widgets/Evals/eval_dialogs.py` | 672 | `test_file_picker_filters_callable` |
| `tldw_chatbook/Widgets/Evals/sample_browser_dialog.py` | 811 | `test_bulk_selection_tooltips`, `test_sample_browser_dialog_selection`, `test_non_obscuring_focus_contract` |

**Modified — test collateral.**

| File | Change |
|---|---|
| `Tests/UI/test_sample_browser_dialog_selection.py` | Deleted whole — all 4 tests are `SampleBrowserDialog` |
| `Tests/UI/test_bulk_selection_tooltips.py` | Remove 2 of 6 tests + 2 imports; 4 non-Evals tests remain |
| `Tests/UI/test_file_picker_action_tooltips.py` | Remove 1 of 5 tests + 1 import; 4 remain |
| `Tests/UI/test_file_picker_filters_callable.py` | Remove 1 test + 1 parametrize entry; 3 items remain |
| `Tests/UI/test_non_obscuring_focus_contract.py` | Remove 2 tests + `SAMPLE_BROWSER_DIALOG` constant. **Keep** `EVAL_NAV_SCREEN` and its test — the card hub is not deleted in this PR |

**Created.**

| File | Responsibility |
|---|---|
| `Tests/UI/test_evals_deletion_guard.py` | Assert the 19 paths stay absent and no production import references them |

**Modified — app wiring.**

| File | Change |
|---|---|
| `tldw_chatbook/app.py` | Remove dead `evals_sidebar_collapsed` reactive (`:2923`) and its no-op watcher `watch_evals_sidebar_collapsed` (`:8125`) |

---

### Task 1: Deletion guard test and the orphan gen-2 cluster

**Files:**
- Create: `Tests/UI/test_evals_deletion_guard.py`
- Delete: `tldw_chatbook/UI/ResultsDashboardWindow.py`, `tldw_chatbook/UI/ModelManagementWindow.py`, `tldw_chatbook/UI/DatasetManagementWindow.py`, `tldw_chatbook/UI/Views/evals_views.py`, `tldw_chatbook/Event_Handlers/eval_events.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Tests/UI/test_evals_deletion_guard.py` with module-level tuples `REMOVED_MODULES: tuple[str, ...]` (repo-relative POSIX path strings) and `REMOVED_STEMS: tuple[str, ...]` (module basenames without `.py`). Tasks 2 and 3 append their own paths and stems to these same two tuples.

- [ ] **Step 1: Confirm the reachability gate is still empty**

Nothing outside the group may import these five modules. Run:

```bash
cd "$(git rev-parse --show-toplevel)"
for m in ResultsDashboardWindow ModelManagementWindow DatasetManagementWindow evals_views eval_events; do
  echo "--- $m ---"
  grep -rn --include="*.py" -E "(from|import)[[:space:]]+[A-Za-z0-9_.]*\b${m}\b" tldw_chatbook/ Tests/ \
    | grep -v -E "tldw_chatbook/(UI/(ResultsDashboardWindow|ModelManagementWindow|DatasetManagementWindow)\.py|UI/Views/evals_views\.py|Event_Handlers/eval_events\.py)"
done
```

Expected: only `eval_events` prints hits, and every hit is inside `ResultsDashboardWindow.py`, `ModelManagementWindow.py`, or `DatasetManagementWindow.py` — all three of which are in this same group. Any hit from a file outside the group means **stop and re-plan**; the module is reachable.

- [ ] **Step 2: Write the failing guard test**

Create `Tests/UI/test_evals_deletion_guard.py`:

```python
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
```

- [ ] **Step 3: Run the guard to verify it fails**

```bash
python -m pytest Tests/UI/test_evals_deletion_guard.py -v
```

Expected: FAIL. All 5 `test_removed_module_file_is_absent` cases fail (files still exist), and `test_no_source_imports_removed_module[eval_events]` fails (the three Group A windows import it). The 4 other `stem` cases pass already, which is correct — those modules are imported by nothing.

- [ ] **Step 4: Delete the cluster**

```bash
git rm tldw_chatbook/UI/ResultsDashboardWindow.py \
       tldw_chatbook/UI/ModelManagementWindow.py \
       tldw_chatbook/UI/DatasetManagementWindow.py \
       tldw_chatbook/UI/Views/evals_views.py \
       tldw_chatbook/Event_Handlers/eval_events.py
```

- [ ] **Step 5: Run the guard to verify it passes**

```bash
pytest Tests/UI/test_evals_deletion_guard.py -v
```

Expected: PASS, 10 passed.

- [ ] **Step 6: Verify the app still imports and the suite is green**

```bash
python -c "import tldw_chatbook.app; print('app imports OK')"
python -m pytest Tests/UI --collect-only -q | tail -3
python -m pytest Tests/UI/test_evals_deletion_guard.py -q
```

Expected: `app imports OK`, and `Tests/UI` passes with **10 more tests than before this task** (the new guard's 5 + 5 parametrized cases) and no failures. `Tests/UI/test_non_obscuring_focus_contract.py` must still pass — it reads Evals files by path, but none of its subjects are in this group.

No `__init__.py` edits are needed: `Event_Handlers/`, `Widgets/Evals/`, `UI/Views/`, and `UI/` re-export none of the deleted modules (verified against `origin/dev`).

- [ ] **Step 7: Commit**

```bash
git add Tests/UI/test_evals_deletion_guard.py
git commit -m "refactor(evals): retire unreachable gen-2 Evals UI cluster

ResultsDashboardWindow, ModelManagementWindow, DatasetManagementWindow,
Views/evals_views, and Event_Handlers/eval_events imported only each
other; no reachable code imported any of them. 3,751 lines.

Adds a guard test so a stale import cannot silently resurrect them."
```

---

### Task 2: Widgets orphaned by the deleted cluster

**Files:**
- Modify: `Tests/UI/test_evals_deletion_guard.py`
- Delete: the 11 Group B files listed in File Structure

**Interfaces:**
- Consumes: `REMOVED_MODULES` and `REMOVED_STEMS` from Task 1.
- Produces: both tuples extended by 11 entries each.

- [ ] **Step 1: Confirm every Group B widget is now unimported**

Task 1 removed their only importers. Verify nothing else appeared:

```bash
cd "$(git rev-parse --show-toplevel)"
for w in Evals_Sidebar ab_test_dialog ab_test_results_widget dataset_validation_dialog \
         eval_cost_monitor eval_error_dialog eval_smart_suggestions metrics_display \
         cost_estimation_widget eval_config_dialogs eval_results_widgets; do
  hits=$(grep -rln --include="*.py" -E "(from|import)[[:space:]]+[A-Za-z0-9_.]*\b${w}\b" tldw_chatbook/ Tests/ \
         | grep -v "Widgets/Evals/${w}.py")
  echo "${w} :: ${hits:-NONE}"
done
```

Expected: every line reads `NONE`. Any other result means **stop** — that widget has a live consumer and must not be deleted here.

- [ ] **Step 2: Extend the guard test**

In `Tests/UI/test_evals_deletion_guard.py`, replace the `REMOVED_MODULES` and `REMOVED_STEMS` tuples with:

```python
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
```

- [ ] **Step 3: Run the guard to verify it fails**

```bash
pytest Tests/UI/test_evals_deletion_guard.py -v
```

Expected: FAIL — the 11 new `test_removed_module_file_is_absent` cases fail because the files still exist. All `test_no_source_imports_removed_module` cases pass, because Task 1 already removed the importers.

- [ ] **Step 4: Delete the widgets**

```bash
git rm tldw_chatbook/Widgets/Evals/Evals_Sidebar.py \
       tldw_chatbook/Widgets/Evals/ab_test_dialog.py \
       tldw_chatbook/Widgets/Evals/ab_test_results_widget.py \
       tldw_chatbook/Widgets/Evals/dataset_validation_dialog.py \
       tldw_chatbook/Widgets/Evals/eval_cost_monitor.py \
       tldw_chatbook/Widgets/Evals/eval_error_dialog.py \
       tldw_chatbook/Widgets/Evals/eval_smart_suggestions.py \
       tldw_chatbook/Widgets/Evals/metrics_display.py \
       tldw_chatbook/Widgets/Evals/cost_estimation_widget.py \
       tldw_chatbook/Widgets/Evals/eval_config_dialogs.py \
       tldw_chatbook/Widgets/Evals/eval_results_widgets.py
```

- [ ] **Step 5: Run the guard and the suite**

```bash
python -m pytest Tests/UI/test_evals_deletion_guard.py -v
python -c "import tldw_chatbook.app; print('app imports OK')"
python -m pytest Tests/UI --collect-only -q | tail -3
```

Expected: guard PASSES (32 passed), `app imports OK`, `Tests/UI` unchanged from Task 1.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(evals): retire widgets orphaned by the gen-2 cluster

Eight had no importers at all; cost_estimation_widget,
eval_config_dialogs, and eval_results_widgets were imported only by the
cluster deleted in the previous commit. 4,473 lines."
```

---

### Task 3: Test-blocked widgets and their test collateral

The last three widgets have no production importers — only tests that use them as convenient subjects for cross-cutting UI contracts. Each affected test file keeps its non-Evals coverage; only the Evals-subject tests go.

**Files:**
- Modify: `Tests/UI/test_bulk_selection_tooltips.py`, `Tests/UI/test_file_picker_action_tooltips.py`, `Tests/UI/test_file_picker_filters_callable.py`, `Tests/UI/test_non_obscuring_focus_contract.py`, `Tests/UI/test_evals_deletion_guard.py`
- Delete: `Tests/UI/test_sample_browser_dialog_selection.py`, `tldw_chatbook/Widgets/Evals/eval_additional_dialogs.py`, `tldw_chatbook/Widgets/Evals/eval_dialogs.py`, `tldw_chatbook/Widgets/Evals/sample_browser_dialog.py`

**Interfaces:**
- Consumes: `REMOVED_MODULES` and `REMOVED_STEMS` from Task 2.
- Produces: both tuples extended by 3 entries each. Final state: 19 paths, 19 stems.

- [ ] **Step 1: Confirm only tests import these three**

```bash
cd "$(git rev-parse --show-toplevel)"
for w in eval_additional_dialogs eval_dialogs sample_browser_dialog; do
  echo "--- $w ---"
  grep -rln --include="*.py" -E "(from|import)[[:space:]]+[A-Za-z0-9_.]*\b${w}\b" tldw_chatbook/ Tests/ \
    | grep -v "Widgets/Evals/${w}.py"
done
```

Expected: every hit is under `Tests/`. A hit under `tldw_chatbook/` means **stop** — the widget is live.

- [ ] **Step 2: Delete the all-Evals test file**

All 4 tests in `test_sample_browser_dialog_selection.py` target `SampleBrowserDialog`; nothing survives its subject.

```bash
git rm Tests/UI/test_sample_browser_dialog_selection.py
```

- [ ] **Step 3: Remove the 2 Evals tests from `test_bulk_selection_tooltips.py`**

Delete these two imports at lines 9-10:

```python
from tldw_chatbook.Widgets.Evals.eval_additional_dialogs import RunSelectionDialog
from tldw_chatbook.Widgets.Evals.sample_browser_dialog import SampleBrowserDialog
```

Then delete the whole body of both `async def test_eval_run_selection_bulk_controls_have_tooltips()` and `async def test_eval_sample_browser_bulk_controls_have_tooltips(monkeypatch)`. The four remaining tests — notes, tag management, multi-item review, mindmap — are untouched.

- [ ] **Step 4: Remove the 1 Evals test from `test_file_picker_action_tooltips.py`**

Delete the import at line 11:

```python
from tldw_chatbook.Widgets.Evals.eval_additional_dialogs import FileUploadDialog
```

Then delete the whole body of `async def test_eval_file_upload_actions_explain_browse_and_upload()`. The four remaining tests are untouched.

- [ ] **Step 5: Remove the Evals subject from `test_file_picker_filters_callable.py`**

Delete the whole body of `def test_eval_dialogs_dataset_filters_are_callable()` — it does `import tldw_chatbook.Widgets.Evals.eval_dialogs as ed`. Then remove this one entry from the `@pytest.mark.parametrize` list feeding `test_no_glob_string_filter_tester_in_source`:

```python
    "tldw_chatbook/Widgets/Evals/eval_dialogs.py",
```

**Keep** `test_eval_default_filters_are_callable` — despite the name it exercises `EvalFilePickerDialog` from `tldw_chatbook/Widgets/file_picker_dialog.py`, which is not being deleted.

- [ ] **Step 6: Remove the 2 sample-browser tests from `test_non_obscuring_focus_contract.py`**

Delete the constant at line 48:

```python
SAMPLE_BROWSER_DIALOG = ROOT / "tldw_chatbook/Widgets/Evals/sample_browser_dialog.py"
```

Then delete the whole body of `def test_evals_sample_browser_selected_row_uses_readable_inline_contract()` and `def test_evals_sample_browser_selected_row_children_show_inline_selected_cue()`.

**Keep** `EVAL_NAV_SCREEN` (line 36), `EVALUATION_UNIFIED` (line 35), and `def test_evals_navigation_card_focus_is_non_obscuring_and_ordered_after_type_borders()`. The card hub and the stylesheet are both out of scope for this PR, so those subjects still exist.

- [ ] **Step 7: Extend the guard test**

Append to `REMOVED_MODULES`:

```python
    "tldw_chatbook/Widgets/Evals/eval_additional_dialogs.py",
    "tldw_chatbook/Widgets/Evals/eval_dialogs.py",
    "tldw_chatbook/Widgets/Evals/sample_browser_dialog.py",
```

Append to `REMOVED_STEMS`:

```python
    "eval_additional_dialogs",
    "eval_dialogs",
    "sample_browser_dialog",
```

- [ ] **Step 8: Run the guard to verify it fails**

```bash
pytest Tests/UI/test_evals_deletion_guard.py -v
```

Expected: FAIL — the 3 new `test_removed_module_file_is_absent` cases fail because the widgets still exist. The 3 new `stem` cases now pass, because Steps 2-6 removed every importer.

- [ ] **Step 9: Delete the three widgets**

```bash
git rm tldw_chatbook/Widgets/Evals/eval_additional_dialogs.py \
       tldw_chatbook/Widgets/Evals/eval_dialogs.py \
       tldw_chatbook/Widgets/Evals/sample_browser_dialog.py
```

- [ ] **Step 10: Run the guard and the full UI suite**

```bash
python -m pytest Tests/UI/test_evals_deletion_guard.py -v
python -m pytest Tests/UI --collect-only -q | tail -3
```

Expected: guard PASSES (38 passed). `Tests/UI` shows a **net 5 fewer tests** than the Task 2 baseline: 11 removed (4 from the deleted file, 2 + 1 + 1 + 2 inline, plus 1 dropped `parametrize` case) minus the 6 new guard cases this task added. No failures, no errors, no collection errors.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "refactor(evals): retire test-only Evals widgets and their collateral

eval_additional_dialogs, eval_dialogs, and sample_browser_dialog had no
production importers left after the cluster deletion -- only tests using
them as subjects for cross-cutting UI contracts. 2,034 lines.

test_sample_browser_dialog_selection.py went whole (every test targeted
the deleted widget). Four other test files kept their non-Evals coverage
and lost only their Evals-subject cases."
```

---

### Task 4: Remove dead sidebar wiring from `app.py`

`evals_sidebar_collapsed` has exactly two references left: its declaration and a watcher whose body is `pass` with a comment saying it does nothing. The handler that used to set it is already gone from `dev`.

**Files:**
- Modify: `tldw_chatbook/app.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

- [ ] **Step 1: Confirm both references are dead**

```bash
cd "$(git rev-parse --show-toplevel)"
grep -rn --include="*.py" "evals_sidebar_collapsed" tldw_chatbook/ Tests/
```

Expected: exactly two hits, both in `tldw_chatbook/app.py` — the reactive declaration and `def watch_evals_sidebar_collapsed`. A third hit anywhere means something still drives it; **stop**.

- [ ] **Step 2: Delete the reactive declaration**

In `tldw_chatbook/app.py` (near line 2923), remove:

```python
    evals_sidebar_collapsed: reactive[bool] = reactive(False)  # Added for Evals tab
```

- [ ] **Step 3: Delete the no-op watcher**

In `tldw_chatbook/app.py` (near line 8125), remove the whole method:

```python
    def watch_evals_sidebar_collapsed(self, collapsed: bool) -> None:
        """EvalsLab uses unified dashboard - no sidebar to collapse."""
        # This method is kept for backwards compatibility but does nothing
        # The new EvalsLab UI doesn't have a collapsible sidebar
        pass
```

- [ ] **Step 4: Verify no references remain and the app imports**

```bash
grep -rn --include="*.py" "evals_sidebar_collapsed" tldw_chatbook/ Tests/ ; echo "exit=$?"
python -c "import tldw_chatbook.app; print('app imports OK')"
```

Expected: `grep` prints nothing and `exit=1`; then `app imports OK`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/app.py
git commit -m "refactor(evals): drop dead evals_sidebar_collapsed reactive

The toggle handler that set it is already gone from dev, leaving a
reactive nothing writes and a watcher whose body is 'pass'."
```

---

### Task 5: Full verification

Deletion PRs fail in ways unit tests miss — a module removed from the import graph can still be referenced by a string path, a lazy import, or a Textual CSS selector. This task proves the app actually runs.

**Files:** none modified.

**Interfaces:**
- Consumes: the completed state of Tasks 1-4.
- Produces: nothing.

- [ ] **Step 1: Confirm the deletion total**

```bash
cd "$(git rev-parse --show-toplevel)"
git diff --stat origin/dev...HEAD -- tldw_chatbook/ | tail -1
```

Expected: deletions of roughly 10,258 lines across `tldw_chatbook/`, plus the 12 removed from `app.py`. Insertions in `tldw_chatbook/` should be **zero** — this PR adds no production code.

- [ ] **Step 2: Search for stale string-path references**

Import-graph checks miss these. Run:

```bash
for n in ResultsDashboardWindow ModelManagementWindow DatasetManagementWindow evals_views \
         eval_events Evals_Sidebar ab_test_dialog ab_test_results_widget \
         dataset_validation_dialog eval_cost_monitor eval_error_dialog eval_smart_suggestions \
         metrics_display cost_estimation_widget eval_config_dialogs eval_results_widgets \
         eval_additional_dialogs eval_dialogs sample_browser_dialog; do
  hits=$(grep -rn --include="*.py" --include="*.tcss" --include="*.toml" --include="*.md" "$n" \
         tldw_chatbook/ Tests/ 2>/dev/null | grep -v "test_evals_deletion_guard.py")
  [ -n "$hits" ] && { echo "=== $n ==="; echo "$hits"; }
done
echo "(no output above means clean)"
```

Expected: no output. Any hit in `tldw_chatbook/` must be resolved before continuing. A hit in a `.md` doc is acceptable if it is describing the deletion; note it and move on.

- [ ] **Step 3: Run the whole test suite**

```bash
python -m pytest Tests/UI -q   # controller runs this backgrounded; see Global Constraints
```

Expected: no new failures relative to the `origin/dev` baseline. Record the pass/fail counts. Per repo convention, pre-existing failures on `dev` are not this PR's to fix — but you must confirm they are pre-existing by comparing against a clean `origin/dev` run, not by assuming.

- [ ] **Step 4: Launch the app and open Evals**

Use the `verify` skill to drive the TUI. Confirm all of:

1. The app boots and reaches Home without a traceback.
2. The Evals destination opens via the Lab tab, then the `Evals` mode chip. `Ctrl+1`..`Ctrl+0` **cannot** be verified through this harness — tmux `send-keys` has no ASCII encoding for ctrl+digit. Assert those bindings in a unit test instead; never conclude they are broken from a tmux probe.
3. **Capture the Evals screen on a second worktree checked out at the base commit, and diff the two captures.** They must be identical apart from the footer's live memory-telemetry readout.
4. No CSS warnings about missing selectors in the log.

Point 3 is the real gate, and the before/after diff is the only form of it that means anything — "it looks right" is not a check.

**Known pre-existing condition, confirmed on the base commit while executing this PR:** the Evals card hub renders an **empty body** inside the app shell — `DestinationHeader` and `LabModeStrip` appear, no cards. `EvalsWindowV3` mounts correctly in isolation (`EvalNavigationScreen` plus 8 buttons), so the failure is shell integration, consistent with the `Screen`-inside-a-`Container` architecture the spec flags. Do not chase it in PR 1; expect the empty body on both sides of the diff.

Use the command palette with care: several destinations match the query "evals", and a fast `Down`+`Enter` selects the wrong one. Clicking the tab with an SGR mouse sequence is more reliable.

- [ ] **Step 5: Confirm the CSS bundle is untouched**

`_evaluation_unified.tcss` is deliberately out of scope, so the generated bundle must be byte-identical.

```bash
git diff --stat origin/dev...HEAD -- tldw_chatbook/css/
```

Expected: no output. If the bundle changed, something rebuilt it — revert that; the bundle regenerates at boot and must not be hand-edited or committed from this PR.

- [ ] **Step 6: Push and open the PR**

```bash
git push -u origin HEAD
gh pr create --base dev --title "refactor(evals): retire ~10.3k lines of unreachable Evals UI" --body "$(cat <<'EOF'
PR 1 of 3 in the Evals Console rebuild. Pure deletion; no behaviour change.

Removes an entire second-generation Evals UI that no reachable code
imported -- ResultsDashboardWindow, ModelManagementWindow,
DatasetManagementWindow, Views/evals_views, and Event_Handlers/eval_events
referenced only each other -- plus the 14 Widgets/Evals files whose only
importers were that cluster or tests using them as contract subjects.

19 files, 10,258 lines, plus a dead reactive and its no-op watcher in
app.py. Adds Tests/UI/test_evals_deletion_guard.py so a stale import
cannot silently resurrect any of it.

Two items from the design spec's deletion table are deliberately NOT here:

- `_evaluation_unified.tcss` stays. Its rules are unscoped, and 21 of its
  33 selectors have surviving consumers across Chat, Logs, MCP, RAG
  search, Chatbooks and more. Deleting it would restyle unrelated screens.
  Only 12 selectors are genuinely Evals-only; that migration belongs with
  the screen rebuild.
- The card hub stays. It is the live Evals screen; it is replaced in PR 3,
  in the same change that provides its replacement.

Spec: Docs/superpowers/specs/2026-07-25-evals-console-rebuild-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes for the reviewer

- **Every group is gated on a verified importer check**, run as the first step of its task. If any check returns a hit from outside the group, the correct action is to stop and re-plan, not to delete and fix fallout.
- **The guard test grows across three tasks** rather than landing complete in Task 1, so each task has a genuine red-to-green cycle instead of a single test that stays red for the whole PR.
- **Line counts are from `origin/dev` at `8242a5b58`.** If the base moves, re-derive them; do not trust the numbers in this document after a rebase.
- **`Tests/UI` loses exactly 11 tests and gains 38.** Removed: four from the deleted file, two from `test_bulk_selection_tooltips`, one from `test_file_picker_action_tooltips`, one from `test_file_picker_filters_callable`, two from `test_non_obscuring_focus_contract`, and one dropped `parametrize` case. Added: the guard's 19 path cases + 19 stem cases. Net for the PR: **+27**. Any other delta needs explaining.
