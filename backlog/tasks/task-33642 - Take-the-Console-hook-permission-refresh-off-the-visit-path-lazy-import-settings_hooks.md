---
id: TASK-33642
title: >-
  Take the Console hook-permission refresh off the visit path; lazy-import
  settings_hooks
status: Done
assignee:
  - '@claude'
created_date: '2026-09-30 15:00'
labels:
  - perf
  - console
  - hooks
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
#2922 (Console hooks) made every Console visit refresh hook-permission state from disk and made settings_screen import settings_hooks at module level. PERF-01's guards measured the cost when #2888 was rebased onto dev 75c06af39a: a visit went from 39 to 43 config admissions, 112 to 127-133 storage admissions and 35,351 to about 44,700 os.open calls, and the Settings pre-import pass from 556 to 557 modules. The owner approved re-pinning those ceilings on 2026-09-30 so the guards could land, on condition the cost is taken off the path afterwards. This task restores the previous ceilings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Console visit with unchanged hook configuration performs no hook-permission disk reads beyond the first visit; a changed hook file is still picked up on the next visit
- [x] #2 settings_hooks is not imported by the pre-import pass (Settings route back to 45 added modules)
- [x] #3 MAX_VISIT_STORAGE_UNITS and MAX_PASS_ADDED_MODULES are lowered back to 39/112/35,351 and 556 (or below), with the ADR-097 ledger updated
- [x] #4 Hook review behaviour is unchanged: pending hooks still surface in the Console control bar and review modal
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Make settings_hooks a lazy import in settings_screen (accessor, id selectors, name-routed message handler)
2. Give HookPermissions a visit snapshot that reuses the last read while nothing it read changed; route only the Console visit refresh through it
3. Tests: reuse until a store write, a config edit or in-memory sealing; negative controls
4. Measure the visit census and pre-import pass on dev and the branch; restore the ceilings; refresh the pre-import snapshot; ADR-097 ledger row
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Lazy import.** `settings_screen` loads `HooksSettingsPanel` through `_hooks_settings_panel_class()` when the Hooks category renders (the `_personal_context_settings_panel_class` pattern). Its queries use the panel's `#settings-hooks-panel` id, and `HooksSettingsPanel.Requested` is handled by `on_hooks_settings_panel_requested`, the name Textual routes it to, so the class is not needed to define the screen. The Settings pre-import pass is back to 556 modules; `preimport_payload.json` refreshed through `scripts/update_boot_budget_snapshots.py --only preimport`.

**Visit refresh.** `HookPermissions.visit_snapshot()` reuses the published snapshot when a stamp equals the one taken before and after the read that produced it. The stamp covers the config selection (the effective config path as well as the loaded source) and generation, `os.stat` identity/size/mtime/ctime of `config.toml` and `hook_permissions.json`, the lstat posture of every component of both parent directories (the full read refuses an unsafe one), and the in-memory sealed, refresh-pending and closed state. A write at any time (including during that read) forces a full read on the next visit; error ("recovery") snapshots are never reused. Only `ConsoleHooks.refresh()` (the visit path) uses it; Send, review and Settings keep `snapshot()`.

**Measured** (same-probe storage-unit census, macOS): warm visit 8 config / 56 storage / 9 helpers / ~2,160 opens, against dev `2612fc56b2`'s 8 / 59-60 / 8-9 / ~2,530. `MAX_VISIT_STORAGE_UNITS` restored to 39/112/35,351 and `MAX_PASS_ADDED_MODULES` to 556; ADR-097 ledger row 2026-10-03.

**Tests.** `test_a_warm_visit_reads_nothing_until_a_file_changes` (a warm visit makes no full read; an approval and a hook edit are picked up on the next visit) fails with reuse disabled; `test_a_sealed_hook_is_not_served_from_a_warm_visit` fails when the stamp ignores sealing; `test_a_retargeted_config_path_is_not_served_from_a_warm_visit` and `test_an_unsafe_store_directory_is_not_served_from_a_warm_visit` failed before the config path and directory posture joined the stamp. Hook UI/Chat/Agents suites: 184 passed, the same 2 `test_settings_search_index` failures as dev.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/Agents/hook_permissions.py`, `tldw_chatbook/UI/Console_Modules/hooks.py`, `Tests/Agents/test_hook_permissions.py`, `Tests/Performance/test_console_keystroke_work_census.py`, `Tests/Performance/test_screen_preimport_payload_budget.py`, `Tests/Performance/boot_budget_snapshots/preimport_payload.json`, `backlog/decisions/097-boot-budget-ratchets.md`.
<!-- SECTION:NOTES:END -->
