---
id: TASK-38976
title: "Prompt scope-service and prompt batch tests run on no PR lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 01:42
updated_date: 2026-11-08 00:45
labels:
- follow-up
- ci
- tests
- prompts
priority: low
references:
- Tests/Prompt_Management/test_prompt_scope_service.py
- Tests/Prompt_Management/test_prompt_batch_models.py
- Tests/Prompts_DB/test_prompts_db_batch_mutations.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38749
- TASK-38756
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b on 2026-11-05, and against the census and workflows as they will be once the round-8 branches merge.

No PR lane runs these three files. They are not in `scripts/ui_pr_gate_census.txt` and no workflow targets them. The only Prompts_DB lane target is `test_prompt_export_mixins.py` (`derived-artifacts.yml:371`). The notes of TASK-38749 and TASK-38756 both record this gap.

Runs on dev at `-n 4`, 3 runs each, load 11.9-12.6:

| File | Result | Wall time |
|---|---|---|
| `Tests/Prompt_Management/test_prompt_scope_service.py` | 187 passed | 5 s |
| `Tests/Prompt_Management/test_prompt_batch_models.py` | 51 passed, 1 failed (the lazy-export pin) | 5-7 s |
| `Tests/Prompts_DB/test_prompts_db_batch_mutations.py` | 64 passed, 2 failed (the privacy pins) | 11-12 s |

TASK-38756 fixes those three reds, and TASK-38749 adds two scope-service tests. Both are Done on branches that merge first.

Serial at load 5.5-5.8 (the dreams-boundary step runs serially at `--timeout=180`), the files take 1.4 s, 2.5 s and 11.5 s: about 16 s in all.

TASK-38760 (To Do) asks for a lane for the Prompts DB legacy and MCP execution-log tests. One PR can lane both sets in the same step.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The three files run in a PR lane step other than the PR Fast Lane, with a step comment that says why
- [x] #2 Before they are added, each file passes 3 of 3 runs in that step's configuration on dev, and the measured times are recorded (about 16 s serial in all, measured on dev)
- [x] #3 The job they join still ends at least 5 minutes under its cap in its slowest measured run
<!-- AC:END -->

## Implementation Plan

1. Put the three files in the new `recovered-suites` job (owner decision 2026-11-06), with a step comment that says why.
2. Run each file 3 times in that step's configuration (serial, `--storage-lease-guard --timeout=180`) and record the times.
3. Time the whole step with `jobstep.sh` and check it ends at least 5 min under its 20-min cap.

## Implementation Notes

**Owner decision (2026-11-06):** "New CI job: add one derived-artifacts job, 'PR lane (recovered suites)', holding all of these files serially under --storage-lease-guard; one more runner per PR; ADR-103's job and runner counts updated."

**Approach (AC#1).** The three files run in the new `recovered-suites` job ("PR lane (recovered suites)"), after TASK-39043's two Backup_Recovery files and before the bootstrap_profile suites. The step comment and `RECOVERED_SUITES_TARGETS` say why: TASK-38756's pins, red in no PR lane, in the owner's own job, so neither the Dreams job nor pr-fast-lane (35-min cap) grows.

**Guard findings, fixed in test code.** The step's `--storage-lease-guard` failed one node in each of two files on dev: `test_local_prompt_service_persists_prompt_collections` left its `prompts.db` handle open, and `test_begin_immediate_preflights_after_competing_recovery_metadata_commit` left the competing writer's schema-setup handle open on the test thread. Each now closes it. Red: the three files under the guard on the base worktree (dev 8b762b52fa): 307 passed, 2 errors (those two). Green here.

**AC#2, each file 3 times in the step's configuration** (serial, `--storage-lease-guard --timeout=180`), this branch:

| File | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| `test_prompt_scope_service.py` (189 passed) | 5.4 s | 6.9 s | 4.7 s |
| `test_prompt_batch_models.py` (52 passed) | 6.3 s | 10.8 s | 7.3 s |
| `test_prompts_db_batch_mutations.py` (66 passed) | 33.4 s | 26.0 s | 25.0 s |

Load 42-52 on 18 CPUs (the task's 16 s total was at load 5.5-5.8).

**The recovered-suites step**, `jobstep.sh fu-r12-lanes "Run the recovered suites"` (14 targets, serial, `--storage-lease-guard --timeout=180`), 3 runs on 2026-11-07: 534.8 s (load 40-63), 703.2 s (load 40-60), 641.7 s (load 50-60). Each run: 723 passed, 1 skipped, 2 failed. The 2 are `test_executor_failure_uses_stable_job_terminal[artifact_incompatible-failed]` and `[engine_crashed-failed]` in the ingest runner, TASK-39140's stale `error_detail` pin, red on dev too and fixed on round-12 branch fu/r12-ctlr (489e2338f5); the step is green once both land. No lease finding in any run. The slowest run plus about 1.5 min of setup is about 13.2 min, 6.8 min under the job's 20-min cap; CI ran comparable lanes at 0.5-1.0 times this machine's loaded time.

So AC#3 holds on the slowest measured run (about 13.2 of 20 min, locally).

**Files.** `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_derived_artifacts_workflow.py`, `Tests/CI/test_ui_lane_shard_count_statements.py`, `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`, `Tests/Prompt_Management/test_prompt_scope_service.py`, `Tests/Prompts_DB/test_prompts_db_batch_mutations.py`.

**Review r12l-rev1 (2026-11-08).** The two bootstrap_profile Tests/UI files left the recovered-suites step for the UI census (the census does hold bootstrap_profile files: 512 entries already, and TASK-36478 restores the shared bootstrap config after each test; my earlier "may not hold" was wrong). TASK-39124's built-in pixel buddy file joined the step last. The final step, 13 targets, `jobstep.sh fu-r12-lanes "Run the recovered suites"`, 3 runs: 363.3 s (load 38 falling to 21), 742.9 s (load 21-50), 467.1 s (load 50-43); each 715 passed, 1 skipped, 2 failed (TASK-39140's pin, fixed on fu/r12-ctlr), no lease finding. The slowest run plus about 1.5 min of setup is about 13.9 min, 6.1 under the cap.
