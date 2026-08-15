# Console and Destination Sweep Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close TASK-15791 only after every failure cluster from the frozen Tests/UI sweep is attributed and every in-scope module passes whole on current `dev`.

**Architecture:** Treat the original inventory as an evidence ledger, not a request for speculative code changes. Reuse the completed atomic TASK-162xx follow-ups, rerun only the named affected modules, and change production only after a current-dev RED reproduction identifies a real root cause.

**Tech Stack:** Python 3.12, pytest, Textual 8, Backlog.md CLI, Ruff.

---

### Task 1: Reconcile the inventory with completed atomic tasks

**Files:**
- Modify: `backlog/tasks/task-15791 - Sweep-inventory-console-and-destination-contract-drift.md`

- [x] Map every original cluster in a ledger with: original module/row, causing commit or justified non-regression attribution, owning TASK/fix commit, current-dev command, and result.
- [x] Identify any row not covered by TASK-16220 or TASK-16244–TASK-16265.
- [x] Record ADR status as `N/A`; this is test-health closeout, not a new boundary.

### Task 2: Prove endpoint-probe and contention hypotheses

**Tests:**
- `Tests/UI/test_settings_provider_test_draft.py`
- `Tests/UI/test_product_maturity_phase1_empty_setup_states.py`
- `Tests/UI/test_product_maturity_phase1_first_run.py`
- `Tests/UI/test_product_maturity_phase1_harness.py`
- `Tests/UI/test_product_maturity_phase1_keyboard_focus.py`

- [x] Run `../../.venv/bin/python -m pytest -q Tests/UI/test_settings_provider_test_draft.py::test_test_provider_button_click_runs_the_check Tests/UI/test_settings_provider_test_draft.py::test_test_provider_button_runs_with_provider_input_focused --disable-warnings`; expect 2 passed with clean teardown and no network-guard error.
- [x] Run the four originally failing Phase 1 modules alone in one process.
- [x] Run `../../.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase1_empty_setup_states.py Tests/UI/test_product_maturity_phase1_first_run.py Tests/UI/test_product_maturity_phase1_harness.py Tests/UI/test_product_maturity_phase1_keyboard_focus.py --disable-warnings`; expect every collected test to pass without a 10-second timeout.
- [x] If a failure reproduces, trace its current root cause before editing and use a minimal RED-to-GREEN test.

### Task 3: Verify the original related module inventory

**Tests:**
- `Tests/UI/test_console_staged_evidence_strip.py`
- `Tests/UI/test_console_shell_regions.py`
- `Tests/UI/test_console_rail_width_budget.py`
- `Tests/UI/test_console_shell_chip_actions.py`
- `Tests/UI/test_console_dictionary_send_integration.py`
- `Tests/UI/test_console_world_info_send_integration.py`
- `Tests/UI/test_console_tab_scope.py`
- `Tests/UI/test_console_citation_sources.py`
- `Tests/UI/test_console_composer_collapse.py`
- `Tests/UI/test_console_live_work_handoffs.py`
- `Tests/UI/test_destination_visual_parity_correction.py`
- `Tests/UI/test_workbench_visual_snapshots.py`
- `Tests/UI/test_personas_generation_wiring.py`
- `Tests/UI/test_settings_rag_profile_region.py`
- `Tests/UI/test_settings_workspaces_category.py`
- `Tests/UI/test_settings_model_catalog_toggles.py`
- `Tests/UI/test_console_fleet_discoverability.py`
- `Tests/UI/test_console_internals_decomposition.py`
- `Tests/UI/test_console_rail_sections.py`
- `Tests/UI/test_destination_headers.py`
- `Tests/UI/test_settings_footer_hints.py`
- `Tests/UI/test_speech_rail_navigation.py`
- `Tests/UI/test_speech_tts_settings_ownership_closeout.py`
- `Tests/UI/test_stts_profile_library.py`
- `Tests/UI/test_schedules_ux_fixes.py`
- `Tests/UI/test_screen_navigation.py`
- `Tests/UI/test_ui_responsiveness.py`
- `Tests/UI/test_watchlists_check_now_failure.py`
- `Tests/UI/test_library_skills_canvas.py`
- `Tests/UI/test_product_maturity_phase6_first_time_release_replay.py`
- `Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py`
- `Tests/UI/test_product_maturity_phase6_packaging_data_safety.py`
- `Tests/UI/test_product_maturity_phase6_power_user_replay.py`
- `Tests/UI/test_product_maturity_phase6_recovery_docs.py`
- `Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py`

- [x] Run `../../.venv/bin/python -m pytest -q Tests/UI/test_console_staged_evidence_strip.py Tests/UI/test_console_shell_regions.py Tests/UI/test_console_rail_width_budget.py Tests/UI/test_console_shell_chip_actions.py Tests/UI/test_console_dictionary_send_integration.py Tests/UI/test_console_world_info_send_integration.py Tests/UI/test_console_tab_scope.py Tests/UI/test_console_citation_sources.py Tests/UI/test_console_composer_collapse.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_fleet_discoverability.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_rail_sections.py --disable-warnings`; expect all 13 Console modules to pass.
- [x] Run `../../.venv/bin/python -m pytest -q Tests/UI/test_destination_visual_parity_correction.py Tests/UI/test_workbench_visual_snapshots.py Tests/UI/test_destination_headers.py --disable-warnings`; expect all three destination modules to pass without updating a baseline.
- [x] Create temporary `Tests/UI/test_task15791_visual_capture_tmp.py` with the exact pytest helper below. Running it through the repository pytest entry point loads `Tests/conftest.py` before this module, which installs isolated XDG/config/data paths before app imports. Run `../../.venv/bin/python -m pytest -q Tests/UI/test_task15791_visual_capture_tmp.py --disable-warnings`; expect 1 passed and `/tmp/task15791-workbench.svg`.

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from Tests.UI.test_workbench_visual_snapshots import (
    _build_test_app,
    _mark_console_onboarding_complete,
    _open_console,
    _test_cli_setting,
)


@pytest.mark.asyncio
async def test_capture_task15791_workbench() -> None:
    app = _build_test_app()
    app.app_config = getattr(app, "app_config", {}) or {}
    app.app_config.setdefault("appearance", {})["ui_density"] = "normal"
    _mark_console_onboarding_complete(app)
    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(160, 42)) as pilot:
            await _open_console(app, pilot)
            Path("/tmp/task15791-workbench.svg").write_text(
                app.export_screenshot(title="TASK-15791 Workbench", simplify=True),
                encoding="utf-8",
            )
```

- [x] Run `/usr/bin/qlmanage -t -s 1600 -o /tmp /tmp/task15791-workbench.svg`; expect `/tmp/task15791-workbench.svg.png`. Inspect it with `view_image(path="/tmp/task15791-workbench.svg.png", detail="original")` and reject clipped/overlapping rails, unreadable controls, raw exceptions, or stale copy.
- [x] Delete `Tests/UI/test_task15791_visual_capture_tmp.py` with `apply_patch`, then run `rm -f /tmp/task15791-workbench.svg /tmp/task15791-workbench.svg.png`; confirm all three paths are absent before commit.

- [x] Run `../../.venv/bin/python -m pytest -q Tests/UI/test_personas_generation_wiring.py Tests/UI/test_settings_rag_profile_region.py Tests/UI/test_settings_workspaces_category.py Tests/UI/test_settings_model_catalog_toggles.py Tests/UI/test_settings_footer_hints.py --disable-warnings`; expect Personas plus all four Settings modules to pass.
- [x] Run `../../.venv/bin/python -m pytest -q Tests/UI/test_speech_rail_navigation.py Tests/UI/test_speech_tts_settings_ownership_closeout.py Tests/UI/test_stts_profile_library.py Tests/UI/test_schedules_ux_fixes.py Tests/UI/test_screen_navigation.py Tests/UI/test_ui_responsiveness.py Tests/UI/test_watchlists_check_now_failure.py Tests/UI/test_library_skills_canvas.py --disable-warnings`; expect all eight singleton modules to pass.
- [x] Run `../../.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase6_first_time_release_replay.py Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py Tests/UI/test_product_maturity_phase6_packaging_data_safety.py Tests/UI/test_product_maturity_phase6_power_user_replay.py Tests/UI/test_product_maturity_phase6_recovery_docs.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py --disable-warnings`; expect all five Phase-6 modules and recovery taxonomy to pass.

### Task 4: Close the task with fresh evidence

**Files:**
- Modify: `backlog/tasks/task-15791 - Sweep-inventory-console-and-destination-contract-drift.md`
- Modify only if an incident generalizes: `backlog/docs/lessons-testing-evidence.md`

- [x] Run `../../.venv/bin/ruff check` and `../../.venv/bin/ruff format --check` on every changed Python file (if any), plus `git diff --check`; expect exit 0 or an exact documented unchanged baseline finding.
- [x] Add concise implementation notes with exact pass/failure counts and ownership mapping.
- [x] Re-evaluate ADR need before any newly discovered production fix; keep `N/A` only if this remains an evidence/test-harness closeout.
- [x] Hard gate: if any in-scope module remains red or any inventory row lacks attribution, leave TASK-15791 In Progress and do not check AC #5.
- [x] Only when the hard gate passes, check every acceptance criterion and run `backlog task edit 15791 -s Done`; if the CLI renames the canonical task file, restore the canonical name before proceeding.
- [x] Review the final diff and commit only task-owned files.
