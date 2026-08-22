---
id: TASK-13158
title: Duplicate backlog id task-3401 blocks the duplicate-ID CI gate on every PR
status: Done
assignee: []
created_date: '2026-08-09 20:28'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two unrelated task files share id task-3401 on origin/dev: 'Make-Console-rail-label-style-configurable' (In Progress) and 'Video-generation-ephemeral-storage-playback-and-streaming' (To Do). The 'No duplicate backlog task IDs' workflow therefore fails on every PR branched from dev, including ones that never touch backlog/tasks (observed on PR #1461). Needs an owner decision on which to renumber: the video-generation id is load-bearing for the active branch feat/task-3401-video-generation-foundation and its subtasks (e.g. task-3401.6), so renumbering that one has cross-branch blast radius; the Console-rail one is In Progress and may also be referenced by an open branch. Not renumbered unilaterally for that reason.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Owner decides which of the two task-3401 files is renumbered
- [x] #2 The chosen file is renumbered past the current max id with headroom, including its frontmatter id and any subtask/parent references
- [x] #3 Any branch or docs references to the renumbered id are updated
- [x] #4 The 'No duplicate backlog task IDs' check passes on dev
<!-- AC:END -->

## Implementation Notes

Resolved upstream, not by this task: the `task-3401` collision was cleared in
PR #1465, which renumbered the Console-rail task to `TASK-14650` (its file
carries the provenance line). The remaining six duplicate ids were then
cleared under TASK-19573, whose own notes record the 2026-08-21 owner rule
(TASK-19601): the older arrival keeps the id, the younger renumbers.

Verified by running the workflow's own check verbatim (`.github/workflows/
backlog-guard.yml`, "Fail on duplicate backlog task IDs") against this
branch's `backlog/tasks/`, both namespaces it tests:

    exact-CI-check fail=0  over 2325 task files

AC #1-#3 were satisfied by that renumbering; #4 is the check above.

**Left open for an owner call (not fixed here):** `task-13262` and
`task-14650` are the same task — identical title, identical created_date,
identical body except that only 14650 carries the renumber provenance — and
BOTH are still `In Progress`. They hold different ids, so the duplicate-ID
guard cannot see them, but a person picking up the work sees two live
copies. 14650 is the canonical one (referenced by task-3793, task-4000 and
`Docs/superpowers/plans/2026-08-08-console-rail-label-setting.md`); 13262 is
referenced only by TASK-19573. Deliberately not archived unilaterally.
