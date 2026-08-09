---
id: TASK-13158
title: Duplicate backlog id task-3401 blocks the duplicate-ID CI gate on every PR
status: To Do
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
- [ ] #1 Owner decides which of the two task-3401 files is renumbered
- [ ] #2 The chosen file is renumbered past the current max id with headroom, including its frontmatter id and any subtask/parent references
- [ ] #3 Any branch or docs references to the renumbered id are updated
- [ ] #4 The 'No duplicate backlog task IDs' check passes on dev
<!-- AC:END -->
