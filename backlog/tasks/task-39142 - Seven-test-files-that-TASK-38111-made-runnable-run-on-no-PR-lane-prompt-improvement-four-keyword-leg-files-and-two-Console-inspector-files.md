---
id: TASK-39142
title: "Seven test files that TASK-38111 made runnable run on no PR lane: prompt improvement, four keyword-leg files and two Console inspector files"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- ci
- tests
- needs-owner-decision
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/CI/test_response_rules_critical_gate.py
- scripts/ui_pr_gate_census.txt
- Tests/Prompt_Management/test_prompt_improvement_service.py
- Tests/RAG_Search/test_keyword_leg_chacha.py
- Tests/RAG_Search/test_keyword_leg_prompts.py
- Tests/RAG_Search/test_keyword_leg_pushdown.py
- Tests/RAG_Search/test_keyword_leg_tiered_merge.py
- Tests/UI/test_console_inspector_focus_carriers.py
- Tests/UI/test_console_inspector_workflow.py
dependencies:
- TASK-38111
- TASK-39049
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, with TASK-38111's run logs (round-11 branch fu/r11-recovery).

**The seven files.**
- `Tests/Prompt_Management/test_prompt_improvement_service.py`;
- four `Tests/RAG_Search/test_keyword_leg_*` files: chacha, prompts, pushdown, tiered_merge;
- `Tests/UI/test_console_inspector_focus_carriers.py`;
- `Tests/UI/test_console_inspector_workflow.py`.

None is in the census or any PR workflow. The fifth keyword-leg file, `test_keyword_leg_db_resolution`, is TASK-39049's AC#3. 36e0f56c7c greened `test_console_inspector_workflow` on dev, not TASK-38111.

**Local times after TASK-38111.** These come from its scratch logs, not its notes, and the load was not recorded.

| File | `-n 4` | Serial, no guard |
|---|---|---|
| prompt improvement | 21.4 s | 22.3 s |
| keyword-leg chacha | 51.7 s | 31.5 s |
| keyword-leg prompts | 67.2 s | 14.2 s |
| keyword-leg pushdown | 67.3 s | 45.9 s |
| keyword-leg tiered_merge | 61.3 s | 68.6 s |
| inspector focus carriers | 115.5 s | 300.1 s (15 tests) |
| inspector workflow | 57.8 s | not run |

**Lane budgets.**
- Three PR runs on 2026-11-06 (42306963606, 42309351181, 42310788537): the Dreams boundary step took 6:56-7:34 and its job 11:03-12:05 of 20 min.
- TASK-39049's notes put the job at about 13.2-16.8 min once round 11 lands.
- No file's behaviour under `--storage-lease-guard` is known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides the lane for the five non-UI files (a new derived-artifacts job or an existing one); the decision is recorded
- [ ] #2 Each of the seven files runs on a PR lane, never the fast PR contract; the two Tests/UI files use UI census tail entries of about 30 s each, split by node where needed
- [ ] #3 Every step touched is timed 3 times with its exact command, with the load, and the CI job times of every touched lane are quoted before and after
- [ ] #4 Any target added to the Dreams boundary step passes under `--storage-lease-guard`, sits after the pinned storage-pausing targets, and leaves the job under its 20 min cap with the margin recorded; all lane pins are updated
- [ ] #5 The keyword-leg files end on one lane together with TASK-39049's `test_keyword_leg_db_resolution`, or the notes say why not
<!-- AC:END -->
