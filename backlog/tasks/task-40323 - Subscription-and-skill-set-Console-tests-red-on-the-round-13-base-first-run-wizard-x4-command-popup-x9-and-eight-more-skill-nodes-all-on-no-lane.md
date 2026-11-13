---
id: TASK-40323
title: "Subscription- and skill-set Console tests red on the round-13 base: first-run wizard x4, command popup x9 and eight more skill nodes, all on no lane"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 04:57
labels:
- follow-up
- tests
- console
- dev-red
priority: low
references:
- Tests/Wizards/test_first_run_subscription_readiness.py
- Tests/UI/test_console_command_popup.py
- Tests/Performance/test_console_skill_trust_stock_owned.py
dependencies:
- TASK-38948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (send stream, TASK-38948's covering runs at `-n 4`, load 23-46). The notes record these at file level only; the nodes are not named.
- Subscription set: 4 in the first-run wizard file (most likely `Tests/Wizards/test_first_run_subscription_readiness.py`, 9 nodes) fail the same way on base ae9f9e8d21. (The 8 Personas nodes belong to TASK-40322.)
- Skill set: 17 fail on base too: `Tests/UI/test_console_command_popup.py` x9 (of 15 functions), visit dispatch x2, controller wiring x1, held pointer `[skill]`, managed received acceptance, and the stock-owned `[send_enter]` / `[send_button]` cases.

**Why.** None of these files is in the UI census or a PR job (`test.yml` runs Tests/UI only on main pushes), so they can stay red indefinitely, and nobody has established whether they are red on CI or only under the local RecoveryRequired trap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failing node is named in the notes with its failure, and red or green is established on CI as well as locally
- [ ] #2 Each red is fixed or re-pinned with the commit that changed what it pinned; a product defect gets a test that is red before its fix
- [x] #3 Every node of these files that is green runs on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Re-run every candidate file on base 26b2e677dd with the plain command at `-n 4`, and name each red node with its failure.
2. Clear RecoveryRequired where the file builds or reads config (a profile marker), and re-pin stale pins with the commit that moved what they pinned.
3. Diagnose what remains. Fix product defects that are in scope; report the rest.
4. Put every green node on a PR lane: Tests/UI nodes at the census tail, the others in the recovered-suites job.

## Implementation Notes

**Base 26b2e677dd, plain command, `-n 4`, load 55-87:** 29 failed and 65 passed over the seven files. Each node, with its failure and what happened to it:

- `Tests/Wizards/test_first_run_subscription_readiness.py`: 9 of 9 pass. Round 13's four reds (never named) are green here.
- `Tests/UI/test_console_command_popup.py`: 9 of 15 failed with `RecoveryRequired: raw_source_selection_changed`. `_build_test_app` -> `build_test_app_config` -> `load_settings` is refused by the per-test sandbox; CI's plain command takes the same path. The file is now `bootstrap_profile`, and 6 of the 9 then pass.
  - Re-pinned to 0520df15ac (TASK-32954 ships the built-in Character Creator skill): `test_slash_opens_popup_and_typing_filters` (an extra `/skills character-creator` row after the commands) and `test_enter_accepts_and_inserts_without_sending` (the arg-mode list is that skill, so the popup stays open).
  - Still red: `test_enter_with_popup_closed_sends_normally`, "Expected mock to have been awaited". The Send press never reaches `submit_draft`. It is red with the stand-in factory too.
- `Tests/UI/test_console_visit_dispatch_dedupe.py`: both nodes failed with RecoveryRequired (the same factory read). The file is now `bootstrap_profile`, and both are red on assertions:
  - `test_console_first_visit_refreshes_skills_and_task_resume_once`: "Console synced task-resume state 5x for one visit".
  - `test_console_later_resume_still_refreshes`: "Returning to Console must re-sync task-resume state".
- `Tests/UI/test_console_controller_wiring.py::test_character_controller_is_constructed_with_late_bound_screen_edges`: `AttributeError: ... '_character_console_session_defaults'`. Re-pinned to db515a71d3, which wires the character-chat defaults. Green; the file passes 48 of 48.
- `Tests/Performance/test_app_wiring_held_pointer_currentness.py[skill|collections]`: the observer's bare `assert frame.f_globals is vars(reads_source)`. Re-pinned to 064d5cb014 (TASK-39147 moved `run_preparation_read` into `Backup_Recovery.runtime_producer_lifetime`; `console_preparation_reads` re-exports it).
  - `[collections]` is green.
  - `[skill]` is still red at script :138: `source.runtime_reads` still holds a read right after the trust task returns; it is empty 0.5 s later.
- `Tests/Performance/test_console_managed_received_acceptance.py`: `KeyError` on `objects["hook_ready"]`. 077e5bfa09 (ADR-230 decision 3) routes the received snapshot through `HookPermissions.authority_read`. Re-pinned to that and to 064d5cb014.
  - It now reaches its end and fails its 10 s received-custody bound: 11.0 s measured at load 65-70. A trace-settlement thread was opening a ChaChaNotes connection through the private SQLite helper process.
- `Tests/Performance/test_console_skill_trust_stock_owned.py`: 17 of 17 failed. 14 failed at script :584 (the builder was never held): the 064d5cb014 globals pin swallowed in the monitor. Re-pinned. After that, four-wide:
  - 13 to 14 cases pass.
  - `[app_closing]`: its hang is TASK-40324's, now fixed.
  - `[send_enter]` and `[send_button]`: "Cold Send did not reach original configuration SQL". There is no received claim, no custody record and no notice, and the draft stays in the composer: the Send never started while the trust build was held.
  - `[mount]`: "Original stock skill trust setup blocked shared UI loop", the 0.35 s loop-progress gate, at load 64-77.
  - A product crash moves between cases from run to run (local_winner, app_closing, send_enter, cancel): `NoMatches: 'SelectOverlay' on PruneSafeSelect(id='recovery-scope')`. `ArchiveRecoveryConfirmationCard.set_confirmation` calls `set_options` before that Select has composed its overlay.

**CI.** No CI has run these files since 2026-10-03, when `test.yml`'s last main-push run was cancelled; they are on no PR lane. The lanes below make this PR's CI the first run. AC#1's CI half and AC#2 stay open until that run is read and the reds below are fixed.

**Lanes (AC#3).**
- UI census tail, 48 nodes: the popup's 14 green tests and the wiring file's 34 functions. `MINIMUM_FILES` is 2087; no re-deal. Serial lane times: popup 14 nodes 118.8 s at load 50-56; wiring 34 nodes 130.9 s at load 39-46.
- Recovered-suites job (`derived-artifacts.yml` plus `Tests/CI/test_ci_queue_pressure_contract.py`'s mirror): 16 targets. The wizard file goes in the sandboxed block; `[collections]` and the 14 green stock cases go last (bootstrap_profile; each a private subprocess).
  - Locally, serial, `--storage-lease-guard --timeout=180`: 24 passed in 8.0 min at load 38-44. Stock cases took 25-39 s each, `[collections]` 14.5 s.
  - That job took 5.7-7.3 min of its 20 in five recent PR runs, so it should land near 14-15 min.
  - The `recovery-scope` crash can fail a random stock case there until it is fixed.
- The Tests/CI contract files pass (193).

**Commits.** fde1a3a362 (the three Performance re-pins). This commit: the popup and visit markers, the popup and wiring re-pins, the lanes, and two `lessons-testing-evidence.md` entries.
