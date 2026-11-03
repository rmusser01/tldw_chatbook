---
id: TASK-38640
title: Eighteen module-size ratchet rows are over budget on dev; nine of them fail every PR's required check
status: To Do
created_date: 2026-11-02 21:52
labels:
- follow-up
- ci
- tests
priority: high
references:
- Tests/Architecture/test_module_size_ratchet.py
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- Docs/superpowers/qa/2026-10-31-pr3162-backlog-rebase.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while filing the round-3 follow-ups of the 2026-10-23 burn-down (dev 88af9f9379, 2026-11-02). `Tests/Architecture/test_module_size_ratchet.py`, run serially through the scratch-profile wrapper, gives 18 failed, 149 passed. Every failure is `test_module_does_not_grow_past_its_budget`, and no `test_budget_is_not_left_slack` row fails.

**Nine rows are gated, so every PR fails.** The PR Fast Lane (`derived-artifacts.yml`, "Run fast PR contract") runs 46 rows of this file by node id. The same list is pinned in `FAST_LANE_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`. Nine of those rows are over budget (measured/budget):

| Row | Measured / budget | Pushed over by |
|---|---|---|
| `Agents/agent_service.py` | 9719/9715 | #3330 |
| `app.py` | 5718/5704 | #3323 |
| `Chat/citation_trace_repository.py` | 4765/4719 | #3323 |
| `Chat/console_provider_gateway.py` | 8546/8528 | #3330 |
| `Chat/console_runtime.py` | 7231/7091 | #3318, #3323, #3331 |
| `Chat/console_trace_service.py` | 6787/6765 | #3304 |
| `DB/ChaChaNotes_DB.py` | 25880/25870 | #3341 |
| `UI/Console_Modules/session.py` | 7044/7005 | #3318, #3323 |
| `Widgets/Console/console_composer_bar.py` | 7403/7195 | #3319, #3330, #3335 |

PR #3344's run 41429394025 (its merge commit is dev 88af9f9379) failed exactly these nine in "PR Fast Lane". The required check "Derived artifacts reproduce from their sources" then failed at "Require successful PR fast lane". Any PR opened now inherits all nine. That run had 11 other PR Fast Lane failures, which are not ratchet rows and are not in scope here.

**Nine rows are not gated** (no PR lane runs them; Nightly Deep is disabled and `test.yml` runs only on main):

| Row | Measured / budget |
|---|---|
| `app_command_providers.py` | 1133/1123 (since #3186, 2026-10-21) |
| `app_service_wiring.py` | 3703/3701 |
| `Chat/console_chat_controller.py` | 29705/29323 |
| `Chat/console_chat_store.py` | 22400/22369 |
| `Chat/console_display_state.py` | 3065/2908 |
| `UI/Persona_Modules/roleplay_frame_state.py` | 1322/855 |
| `UI/Screens/personas_screen.py` | 17505/16859 |
| `Widgets/Console/console_transcript.py` | 8594/8584 |
| `Widgets/Persona_Widgets/personas_library_pane.py` | 1582/1297 |

**How they got there.** Each row went over on 2026-11-01/02 in a backlog integration merged with CI waived. The exception is `app_command_providers`, which went over on 2026-10-21. The owner's 2026-10-31 instruction was "Ignore size limits", recorded in the QA record's "Size waiver" section (`Docs/superpowers/qa/2026-10-31-pr3162-backlog-rebase.md`). Those integrations kept dev's budget rows unchanged, so the rows went red. The three Roleplay rows carry comments recording a raise to 17,505 / 1,318 / 1,578. #3334 kept dev's lower ceilings, and two of those files now measure 4 lines above the recorded figures.

**Overlap.** TASK-36161 lists the 8 rows that were red on 2026-10-22. Six of them are green now; its two still-red rows (`app_command_providers`, `console_transcript`) are in the table above. Meeting this task meets TASK-36161's AC#2, so that task does not need separate work on those rows. TASK-33910.21's AC#2-#3 already ask for the whole file to run in a required PR job. This task only makes the rows pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On dev, every row of test_module_size_ratchet.py passes both test_module_does_not_grow_past_its_budget and test_budget_is_not_left_slack
- [ ] #2 Each of the 18 rows is either re-pinned at its measured size with a dated comment citing the owner's 2026-10-31 size waiver, or brought down by moving code out and lowering the row
- [ ] #3 A PR on the fixed dev gets a PR Fast Lane run with no size-ratchet failure, and its run id is recorded here
<!-- AC:END -->
