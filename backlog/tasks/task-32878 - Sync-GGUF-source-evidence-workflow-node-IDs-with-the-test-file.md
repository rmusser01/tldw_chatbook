---
id: TASK-32878
title: Sync GGUF source-evidence workflow node IDs with the test file
status: Done
assignee: []
created_date: '2026-09-20 09:30'
labels:
  - ci
  - dev-blocker
dependencies: []
references:
  - .github/workflows/task-2062-2-gguf-source-evidence.yml
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dev-side CI breakage (found blocking PRs #2746/#2747/#2750, 2026-09-20): the GGUF source-evidence workflow's full-app loop references node IDs the test file no longer collects. Evidence: the workflow at origin/dev lines 119-121 lists test_llm_gguf_source_modes.py::test_lazy_mount_cached_inventory_preserves_handoff_awaiting_fresh_read and ::test_inventory_during_lazy_selector_mount_is_replayed_after_leaving_pane[False]/[True], but the test file at origin/dev defines both as plain unparametrized async defs (:518/:578) -- CI logs show "ERROR: not found ... (no match in any of [<Module test_llm_gguf_source_modes.py>])" on all three OSes (run 35491798980). Every open PR that runs this workflow against current dev fails it; the branch content is irrelevant (verified: identical branch content passed the job against an older dev). REVISED DIAGNOSIS (2026-09-20): not a live dev breakage. The three node IDs COLLECT AND PASS at current origin/dev with the workflow's exact command shape (verified locally, including plugin-autoload-disabled collection); merged PRs post-a2d89bd3fc never hit the job (path-filtered workflow). The failing runs used stale merge refs created before a2d89bd3fc landed (original run 2026-09-19T21:20Z pre-dates it; reruns reuse the snapshot). Resolution: rebase the gated PRs onto current dev so their candidates contain both the workflow nodes and the tests, then re-run. If any LLM-touching PR still fails GGUF source afterward with fresh refs, reopen this task with the new SHA.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The workflow's node list matches what Tests/UI/test_llm_gguf_source_modes.py collects at dev, and the GGUF source-evidence job is green on dev and on PRs #2746/#2747/#2750
<!-- AC:END -->

## Closeout audit 2026-10-11

Board audit (branch `chore/board-prune-prune-d`, dev at a3c47c6c72): this task's work is
landed on `origin/dev` but the status field was never flipped. Verdict: **Done**.

The node-ID desync was repaired in a2d89bd3fc (2026-09-19, "fix(models): defer GGUF
inventory until selectors mount"), which added the three `test_llm_gguf_source_modes.py` node IDs
to `task-2062-2-gguf-source-evidence.yml` in the same commit that added the tests; the three
blocked PRs (#2746, #2747, #2750) then merged on 2026-09-20. Verified today in this worktree:
all 7 node IDs listed by the workflow exist in `Tests/UI/test_llm_gguf_source_modes.py`
(ALL_NODES_PRESENT), and the later trigger narrowing (98f038494a, c3897ae846) kept the workflow
and the `test_task2062_*.py` contracts in sync.

AC checkboxes are left as found; this section is the audit's evidence record.
