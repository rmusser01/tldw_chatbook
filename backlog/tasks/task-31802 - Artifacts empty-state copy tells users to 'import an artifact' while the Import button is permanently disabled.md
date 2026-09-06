---
id: TASK-31802
title: Artifacts empty-state copy tells users to 'import an artifact' while the Import button is permanently disabled
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ux
  - artifacts
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). The Artifacts empty state instructs importing an artifact, but the only Import button is disabled with no explanation of how to enable it. Either enable the path, explain the precondition, or change the copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Empty-state guidance matches an action the user can actually take.
<!-- AC:END -->
