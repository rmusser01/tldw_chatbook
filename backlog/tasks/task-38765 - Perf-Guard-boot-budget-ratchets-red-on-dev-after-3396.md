---
id: TASK-38765
title: "Perf Guard boot budget ratchets red on dev after #3396"
status: Done
assignee:
  - '@claude'
created_date: '2026-11-05 10:20'
updated_date: '2026-11-05 10:38'
labels:
- follow-up
- perf-guard
- ci
priority: high
references:
- Tests/Performance/test_app_import_weight.py
- Tests/Performance/test_ui_ready_module_census.py
- Tests/Performance/test_screen_preimport_payload_budget.py
- .github/workflows/perf-guard.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Perf Guard run for dev push 447628d08b (the #3396 merge, run 41994911006, 2026-11-05) fails its "Run boot budget ratchets" step. Every later step is skipped, including the Console gates, the keystroke census and the storage-unit ratchet. Three ratchets are over their limits:
- `test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size`: `import tldw_chatbook.app` loads 687 modules (limit 686).
- `test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size`: 1036 modules resident at `_ui_ready` (limit 1031).
- `test_screen_preimport_payload_budget.py::test_preimport_pass_payload_stays_within_budget`: the pre-import pass adds 426,512 LOC (limit 426,199).

The last dev push whose boot step passed was 335f984bd2 (2026-11-03 20:27Z). In the pushes between, the boot step was either cancelled or skipped because the latency step failed first. While this step is red, Perf Guard cannot report on any PR's later steps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the three over-limit readings is traced to the modules and commits that added them
- [x] #2 Each growth is deferred to first use at its source, or, where the import is legitimate, its ratchet is re-pinned at the measured value with a dated comment; the notes say which applies to each ratchet
- [x] #3 The three tests and the Perf Guard boot budget step command pass on this branch
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the CI assertion messages; run the three tests locally on this HEAD and on 335f984bd2 (last green boot step) and on 447628d08b^1, to find which side of #3396 each growth sits on.
2. For each new module, find the import edge that pulls it in (import tracing) and the commit that added the edge.
3. Defer imports that are only needed on first use; re-pin the rest at measured values with dated comments (owner size waiver 2026-10-31).
4. Run the three tests, the Perf Guard boot step command, Tests/CI and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**#3396 did not cause this; it exposed it.** Dev before #3396 (bb80905eb3) is red with the same three readings (687, 1036 and 426,512). On every dev push since 335f984bd2, the boot step was either cancelled or skipped: the destination tour failed first in "Run latency guardrails", and GitHub skips the steps after a failed one. #3396's idle-loop fix let the latency step pass, so the boot step ran again.

**Attribution (AC#1).** I measured each first-parent dev merge from 335f984bd2 to bb80905eb3 locally with the three tests, and diffed the module sets at each change. A temporary pytest probe (not committed) dumped each guard's set in a scratch worktree. The counts at c2cdb6b82e, the base before the arch/p2 batch, were 685, 1031 and 426,171. All of the growth is in the arch/p2 batch merged at 2026-11-04 02:24Z:

| merge | boot import | `_ui_ready` | pre-import LOC | new module (source) |
|---|---|---|---|---|
| #3359 db-errors | +1 | +1 | 0 | `DB.store_errors`: 19 lines of exception bases that ChaChaNotes, Media, Evals and Prompts DB errors derive from |
| #3362 approval-fmt | 0 | +1 | 0 | `Chat.approval_formatting`, moved out of `chat_approval_card` (already resident at `_ui_ready`) |
| #3363 inspector-state | 0 | +1 | 0 | `Chat.console_inspector_state`, moved out of `console_inspector_section` (already resident) |
| #3368 participant-names | 0 | 0 | +4 | lines in existing modules |
| #3369 widgets-screens | 0 | 0 | +21 | moves into `UI/Settings_Modules` (new package init) and `rag_profile_name_modal`; pre-import modules 554 -> 556 |
| #3375 capture-policy | +1 | +2 | +316 | `Chat.console_capture_policy` and `Canvas.config_policy`, moved out of `config.py` (resident at boot; it shrank by 787 lines) |

The other merges are renames that change no counts, such as `Provider_Contract`, `Utils.character_display_text` and `UI.Speech`. The CI failure message's NEW lists diff against stale snapshots, so they over-name; the table is the per-merge diff.

**Deferral or re-pin (AC#2): re-pin all three.** Each new module is an extraction from a module that was already on the guarded path, or exception bases the stores need at class definition. No new code reached boot or `_ui_ready`.
- Deferring them would not make any test pass on its own. `store_errors` cannot be lazy. The boot guard is `count < limit`, so deferring `console_capture_policy` alone would still read 686 against 686. Every `_ui_ready` arrival is used during the Console mount.
- The pre-import +312 is `config.py` code, counted as baseline before, now in a module `settings_screen` imports. In the running app it is already resident at `_ui_ready` (the Console mount loads it), so making the import lazy would only move the census.
- Re-pins, at the measured values under the owner size waiver 2026-10-31:
  - `MAX_TLDW_MODULE_COUNT` 686 -> 688 (measured 687; the guard is a strict `<`, as before)
  - `MAX_TLDW_MODULES_AT_UI_READY` 1031 -> 1036
  - `MAX_PASS_ADDED_LOC` 426,199 -> 426,512 (LOC-only; modules 557 of 558)
- Each constant has a dated comment, and ADR-097's exception ledger has a row for each.
- The three snapshots were refreshed with `scripts/update_boot_budget_snapshots.py --only import-weight --only ui-ready --only preimport` (687 / 1036 / 557 modules, 426,512 LOC).

**Red/green (AC#3).** Red on dev in CI (run 41994911006) and locally on this HEAD before the re-pin: `3 failed, 19 passed`, with 687/686, 1036/1031 and 426,512/426,199. After the re-pin, the three files pass. The Perf Guard boot step command passed `32 passed`, reporting 687/688, 1036/1036 and 557/558 modules with 426,512 LOC. `Tests/Performance/test_boot_budget_ratchet_messages.py` (the ADR/ledger pins) passed 11.

Files: `Tests/Performance/test_app_import_weight.py`, `Tests/Performance/test_ui_ready_module_census.py`, `Tests/Performance/test_screen_preimport_payload_budget.py`, `Tests/Performance/boot_budget_snapshots/{boot_import_modules.txt,ui_ready_modules.txt,preimport_payload.json}`, `backlog/decisions/097-boot-budget-ratchets.md`.
<!-- SECTION:NOTES:END -->
