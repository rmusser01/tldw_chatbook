---
id: TASK-41512
title: "The UI-ready module census overruns at base about two runs in three"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 07:12
labels:
- follow-up
- tests
- flaky
priority: medium
references:
- Tests/UI
dependencies:
- TASK-41141
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41141's review (r15ti-rev2, round 15) and settled: base fails 2 of 3 and the branch fails 2 of 3, at 1041 against a limit of 1040 modules. The extra modules are `DB.character_conversation_search`, reached via `character_context._load_recent_sync` on an executor thread (timing-dependent), and `Provider_Contract.provider_setup_facts`, reached via `endpoint_refusal` then `_route_owning_presets` (a lazy import; TASK-40300).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census gives the same module count on 10 of 10 runs, by removing the timing-dependent import or by pinning it in the allowlist with its reason
- [x] #2 The cause of each extra module is named in the notes
<!-- AC:END -->

## Implementation Plan

1. Measure the census repeatedly at the base and diff each run's module list, to name every module that comes and goes.
2. Trace who imports each extra module, on which thread, and when relative to `_ui_ready`.
3. Pin the timing-dependent one in an allowlist in `Tests/Performance/test_ui_ready_module_census.py`, with its reason, so it counts on every run; leave real new residents to the budget.
4. Add a unit test for the allowlist (red with it emptied), then run the census 10 times.

## Implementation Notes

**Causes at this base (67583dfa72).** Before the fix, 18 warm boots through the census's own `_boot_and_census` (load 14-22) gave 1039 thirteen times and 1040 five times. Only one module came and went. Each module the snapshot lacks, traced with an import hook that records the importing thread and the stack:
- `DB.character_conversation_search` depends on timing. It is imported on an asyncio executor thread (`asyncio_1`) by `character_context._load_recent_sync`, whose `CharacterConversationNavigation.__init__` imports it lazily. In one traced boot it loaded 0.13 s after `_ui_ready`. It was resident in 5 of the 18 boots.
- `Provider_Contract.provider_setup_facts` is a real dependency and does not depend on timing. It is imported on MainThread during ChatScreen compose, about 1.0 s before ready, in all 18 boots: `_build_console_inspector_state` -> `_active_console_settings_readiness_uncached` -> `build_console_settings_readiness` -> `endpoint_refusal` -> `resolve_provider_endpoint` -> the lazy import in `_route_owning_presets` (TASK-40300).
- The other differences from the snapshot are also fixed on every boot. `DB.chachanotes_sql_steps` is new on the app-import path (`ChaChaNotes_DB` line 150, TASK-41310 A). `MCP.console_snapshot` is new on compose (TASK-41308). `Agents.execution_capacity`, `Agents.ordinary_tree` and `Chat.console_ordinary_tree` are gone (TASK-41319). The net is 1039, plus the timing-dependent module. CI's run on dev 97f8c58c0c reads the same: 1039/1040, drift +3/-3. The 1041 the round-16 lead saw at 45b1949ca3 fits that dev's 1040 plus the same module.

**Fix.** `TIMING_DEPENDENT_AT_READY` in `Tests/Performance/test_ui_ready_module_census.py` pins `DB.character_conversation_search` with its reason, and `with_timing_dependent_modules` counts each pinned module on every run. It runs inside `_boot_and_census`, so `scripts/update_boot_budget_snapshots.py` writes the same list. The module is counted rather than excluded because the 1037 -> 1038 raise (TASK-38951) already budgeted it, and excluding it would free a slot nobody shed. The budget stays at 1040, and the census now reads 1040/1040.

**Evidence.**
- New test `test_a_timing_dependent_module_counts_the_same_whether_or_not_it_loaded`. It is red with the union removed from `with_timing_dependent_modules` (`AssertionError: assert [...] == [...]`, a boot where the module loaded counts one more than a boot where it did not), and green with it.
- 20 boots through `_boot_and_census` after the fix (10 at 5 in parallel and 10 at 2 in parallel, load 15-28) all counted 1040 and gave identical module lists. The raw count at the flag was 1039 in 7 of them and 1040 in 13.
- The census node `test_ui_ready_module_census_stays_at_the_pinned_size` passed 10 of 10 serial pytest runs (load 13-16), each reading `ui-ready-census: 1040/1040 modules (headroom 0); snapshot drift +4/-3`.

**Not changed: the stale UI-ready snapshot (drift +4/-3).** Refreshing it makes `Tests/UI/test_console_first_send_preimport.py::test_the_send_guard_counts_a_module_resident_at_ui_ready_as_warm` fail. I checked this by adding only the `MCP.console_snapshot` line. TASK-41308 made that module resident at ready, but `LIVE_FIRST_SEND_IMPORTS` still lists it as a lazy send import. This is proposed as a follow-up.

**Files.** `Tests/Performance/test_ui_ready_module_census.py`. The census runs in Perf Guard's boot budget step, so no lane change was needed.
