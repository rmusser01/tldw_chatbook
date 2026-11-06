---
id: TASK-39055
title: "Four ChatScreen Architecture files fail on CI's depth-1 checkout because nine nodes read frozen revisions or origin/dev, so none of their 58 nodes runs on a PR lane"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 15:01
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
- [ ] #1 The owner decides among `fetch-depth: 0` for the job, laning only the 49 shallow-safe nodes, and retiring the frozen-history and `origin/dev` nodes; the decision is recorded
- [ ] #2 Under a depth-1 checkout (the `GIT_DIR` simulation, or a CI run) no laned node fails for a missing revision; before the change 9 of 58 fail
- [ ] #3 The laned nodes run on a PR lane (never the fast PR contract), and that lane's command is timed three times with them
<!-- AC:END -->

## Implementation Plan

Owner decision (2026-11-06): lane the 49 shallow-safe nodes, retire the 9 history nodes.

1. Reproduce the depth-1 failure with the `GIT_DIR` simulation (a depth-1 bare clone of HEAD): 9 of 58 fail; with full history 58 pass.
2. Retire the history reads. Seven of the nine nodes hold only frozen-revision or `origin/dev` facts plus checks other nodes already run, so they go whole. Two (`test_wave6_inventory_matches_the_implementation_base` and `test_wave6_compatibility_inventory_is_complete_and_phase_safe`) also hold live ownership and descriptor checks that no other node runs (the image and video families, every family's delegate contract, and every owner's compatibility descriptors). For those two, delete only the frozen-revision reads and keep the live checks, under names that say what they check.
3. Remove every helper, constant and import that only the retired code used (iterate an AST dead-name scan to a fixed point), including oracle lines that only exercised a removed helper.
4. Prove AC#2 with the same simulation: 0 laned nodes fail at depth 1.
5. Append the four files to the Architecture guards step and `ARCHITECTURE_GUARD_TARGETS`, update the job comment, and time the job command three times.
