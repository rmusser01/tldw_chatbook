---
id: TASK-41060
title: "test_stock_local_declarations_keep_private_effects_bounded is red on dev: the declared maximum misses the skills gate"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
- ssh
priority: low
references:
- Tests/Chat/test_console_ssh_custody.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Chat/test_console_ssh_custody.py::test_stock_local_declarations_keep_private_effects_bounded[ssh_console0-False-False]` fails on dev ecad404bd4 (2026-11-15) (bootstrap profile, `-n 4`) with "maximum missed the actual skills/runtime gate", an assertion on the turn's `WorktreeFootprint`. The architecture-review fleet runs on dev beab5c1e4e (2026-11-13) saw it red too.

In the same runs, all six params of `test_unknown_peer_proof_keeps_original_model_worker_and_storage` timed out in their 40 s child pytest. TASK-40716 (parked workers that nothing completes) is the likely reason a child cannot exit, so those nodes are left to it. Their leaked helper processes are TASK-41030.

If the product is wrong, an SSH-backed turn can touch a source its declared maximum does not include. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether the product or the test is wrong, and name the commit
- [ ] #2 The node passes on dev; a fixed product defect is shown red first
<!-- AC:END -->
