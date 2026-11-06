---
id: TASK-39055
title: "Four ChatScreen Architecture files fail on CI's depth-1 checkout because nine nodes read frozen revisions or origin/dev, so none of their 58 nodes runs on a PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 23:20
labels:
- follow-up
- tests
- architecture
- ci
- needs-owner-decision
priority: medium
references:
- Tests/Architecture/test_console_realtime_controller_boundary.py
- Tests/Architecture/test_console_review_selection_controller_boundary.py
- Tests/Architecture/test_console_wave6_inventory.py
- Tests/Architecture/test_console_wave6_closeout_inventory.py
- Tests/Architecture/test_screen_size_ratchet.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38988
- TASK-38989
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on Architecture PR #3436 (head b7c8e809b3), which lands TASK-38988 and TASK-38989. TASK-38989 keeps these four files off its new Architecture guards job pending an owner decision. Line numbers are dev's; #3436's are 11-12 higher (given in brackets).

**Which nodes read history.** Each file has a `git show` helper:
- `test_console_realtime_controller_boundary.py:358` (`_source_at_revision`, [#3436 :369]) reads `origin/dev`.
- `test_console_review_selection_controller_boundary.py:159` (`_source_at_revision`, [#3436 :171]) reads TASK_BASE c6218918d1 and `origin/dev`.
- `test_console_wave6_inventory.py:592` (`_source_at_revision`, [#3436 :603]) reads seven frozen revisions (bed39af6b0, 8d806b71d9, d4f3f97763, 02cd80b330, 0da426e1e4, ede2162143, a2c09daa81).
- `test_console_wave6_closeout_inventory.py:223` (`_source_file_at_revision`, [#3436 :235]) reads 8d806b71d9, 87791f8553 and d20dd733b7, plus seven (base, delivered) pairs.

That is 23 distinct revisions, all ancestors of origin/dev, so `fetch-depth: 0` would make them available.

**What fails on a depth-1 checkout.**
- `derived-artifacts.yml` sets no `fetch-depth` on any of its `actions/checkout@v4` steps, so every PR checkout is depth 1.
- A simulation pointed `GIT_DIR` at a depth-1 bare clone of HEAD:
  - with full history: 58 passed in 15.5 s;
  - at depth 1: **9 failed, 49 passed**.
- The nine failures:
  - realtime `test_origin_dev_…`;
  - review-selection `test_frozen_task_base_…` and `test_origin_dev_…`;
  - wave6 `…matches_the_implementation_base`, `…compatibility_inventory…`, `…first_chat_task_ratchet…` and `…fleet_task_ratchet…`;
  - two closeout nodes.
- No PR lane runs any of the four files.

**What the owner weighs.**
- Three of the nine read the moving `origin/dev`. On a PR job, even with full history, they check the base branch, not the PR head.
- After TASK-38988 the live ceilings in these files come from `test_screen_size_ratchet.py::_BUDGETS`, which already runs in the fast PR contract. What only these nine nodes add is facts about frozen revisions, which no product change can alter.
- Checkout cost on ubuntu:
  - depth 1: 24-42 s (run 42141155746, 2026-11-06);
  - `fetch-depth: 0`: 28-31 s (nightly-deep run 36306805067, 2026-09-27).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner decides among `fetch-depth: 0` for the job, laning only the 49 shallow-safe nodes, and retiring the frozen-history and `origin/dev` nodes; the decision is recorded
- [x] #2 Under a depth-1 checkout (the `GIT_DIR` simulation, or a CI run) no laned node fails for a missing revision; before the change 9 of 58 fail
- [x] #3 The laned nodes run on a PR lane (never the fast PR contract), and that lane's command is timed three times with them
<!-- AC:END -->

## Implementation Plan

Owner decision (2026-11-06): lane the 49 shallow-safe nodes, retire the 9 history nodes.

1. Reproduce the depth-1 failure with the `GIT_DIR` simulation (a depth-1 bare clone of HEAD): 9 of 58 fail; with full history 58 pass.
2. Retire the history reads. Seven of the nine nodes hold only frozen-revision or `origin/dev` facts plus checks other nodes already run, so they go whole. Two (`test_wave6_inventory_matches_the_implementation_base` and `test_wave6_compatibility_inventory_is_complete_and_phase_safe`) also hold live ownership and descriptor checks that no other node runs (the image and video families, every family's delegate contract, and every owner's compatibility descriptors). For those two, delete only the frozen-revision reads and keep the live checks, under names that say what they check.
3. Remove every helper, constant and import that only the retired code used (iterate an AST dead-name scan to a fixed point), including oracle lines that only exercised a removed helper.
4. Prove AC#2 with the same simulation: 0 laned nodes fail at depth 1.
5. Append the four files to the Architecture guards step and `ARCHITECTURE_GUARD_TARGETS`, update the job comment, and time the job command three times.

## Implementation Notes

**Status: Done.** Two of the nine nodes keep their live checks; the owner ruled on this (see "Deviation" below).

Owner decision (2026-11-06): "Lane 49, retire 9: put the 49 shallow-safe nodes on the Architecture guards job and delete the 9 history nodes. They only restate facts about frozen commits no product change can alter, and the 3 origin/dev ones would check the base branch, not the PR. Live ceilings already run in the fast contract." (AC#1)

**Before.** I pointed `GIT_DIR` at a depth-1 bare clone of HEAD (`git clone --bare --depth 1 file://<worktree>`), with the worktree as `GIT_WORK_TREE`, and ran the four files serially: 9 failed, 49 passed. These are the filed nine. With full history: 58 passed.

**Retired.**
- Seven nodes went whole, because their only live assertions are ceilings that the fast contract's `_BUDGETS` already runs, or checks other nodes in these files already make:
  - realtime `test_origin_dev_realtime_family_and_projection_still_match_review`;
  - review-selection `test_frozen_task_base_family_and_projection_match_review` and `test_origin_dev_review_selection_family_still_matches_review`;
  - wave6 `test_first_chat_task_ratchet_is_earned` and `test_fleet_task_ratchet_is_earned` (their "methods returned to ChatScreen" checks are in `test_first_chat_family_has_completed_controller_ownership` and `test_fleet_family_has_completed_controller_ownership`);
  - closeout `test_closeout_evidence_explains_the_remaining_absolute_deficit` and `test_closeout_candidates_have_exact_non_vacuous_margin`.
- Then, iterating an AST dead-name scan to a fixed point, I removed every helper, constant and import that only the retired code used. That is 97 top-level definitions: frozen SHAs, frozen line and method counts, `_source_at_revision`, `_optional_source_at_revision`, `_files_containing_at_revision`, `_assert_final_*ownership`, the closeout's review-selection sets, and so on. It also removed `subprocess`, `hashlib`, the closeout and wave6 `LIVE_SCREEN_*` ceilings, the now-empty `Wave6Group.source_revision` field, and the stale comments. Two names were already dead at base and are untouched: `TASK10_SCREEN_LINE_CEILING` and `FINAL_CLOSEOUT_COUNTS`. ruff F-checks are clean. No git subprocess is left in the four files.

**Deviation: two nodes keep their live checks.**
- `test_wave6_inventory_matches_the_implementation_base` also checked, on the working tree, that every moved method of every wave-6 family has exactly one owner, that delegates and stays exist, that target methods make no `query_one` or DOM access, and every family's residue and delegate contract. For the image and video families no other node checks any of this.
- `test_wave6_compatibility_inventory_is_complete_and_phase_safe` was the only node that checked each owner's compatibility descriptors: controller defaults, reads after build, wiring, and the runtime fail-loud contract.
- Deleting them whole would have dropped live guards the ruling did not mean to drop ("they only restate facts about frozen commits" is not true of these two). So their frozen-revision reads are gone, and their live assertions stay unchanged, as `test_wave6_families_have_one_owner_and_bounded_delegates` and `test_wave6_compatibility_descriptors_are_wired_and_fail_loud`.
- A temporary edit to `UI/Console_Modules/image.py` (renaming one moved method, changing one controller default) turned each of them red.
- Owner decision (2026-11-06) on this deviation: "Keep the two live checks. Only the frozen-history reads are retired, as decided; the live ownership and descriptor contracts keep guarding the code on the Architecture lane."

**After (AC#2).** I made the same depth-1 simulation of the committed HEAD 476cf26b2a. In that clone `git show 8d806b71d9:…` fails with `invalid object name`. The four files ran 51 passed and 0 failed (the 49, plus the 2 kept). With full history: 51 passed.

**Lane (AC#3).** All four files were appended to the Architecture guards step and to `ARCHITECTURE_GUARD_TARGETS`, and the job comment now says no target may read git history. Timings are below, under the shared job timing.

Files: the four `Tests/Architecture` files, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`. A lesson was added to `backlog/docs/lessons-testing-evidence.md`: a node filed as history-only can hold the only live check.

**Architecture guards job timings (shared by TASK-39054, TASK-39055, TASK-39060 and TASK-39061).**
- Before adding anything: CI ran the step in 402 s (job 7 m 38 s) on #3436's run 42218508176, 2026-11-06.
- The step's command, run with `bash $FU/jobstep.sh fu-r11-arch "Run Architecture guard contracts"` (serial, `--timeout=180 --tb=short`). Each run of the final 36 targets gave 383 passed:

| Command | Targets | Nodes | pytest time | Load |
|---|---|---|---|---|
| Base 2684e5453f | 29 | 160 | 153.9 s | 7-11 |
| Base 2684e5453f | 29 | 160 | 457.0 s | 27-45 |
| Base 2684e5453f | 29 | 160 | 283.9 s | 32-38 |
| This branch, run 1 | 36 | 383 | 563.4 s | 25-48 |
| This branch, run 2 | 36 | 383 | 451.2 s | 24-48 |
| This branch, run 3 | 36 | 383 | 440.6 s | 24-57 |
| This branch, right after the 283.9 s base run | 36 | 383 | 413.8 s | 32-48 |

- Six agents shared the machine, and the load swung from 24 to 57, so the totals mostly measure the load.
- One more run, with `--durations=0` (862.4 s at load 50-85), attributes setup, call and teardown time per file. The added files cost 146.9 s of its 858.3 s:
  - `test_no_undefined_module_globals.py`: 72.5 s for 3 nodes, with no periodic-GC teardown;
  - `test_log_sanitizer.py`: 40.6 s for 168 nodes, of which 31.8 s is the conftest's every-25th-test `gc.collect()`;
  - the four ChatScreen files: 33.5 s for 51 nodes, 8.9 s of it GC;
  - the two boundary nodes: 0.3 s.
- An extra job run overlapped my temporary source reverts for a boot-census bisect (12 failed). I discarded it.
