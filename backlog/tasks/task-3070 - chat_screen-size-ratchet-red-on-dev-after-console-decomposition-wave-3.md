---
id: TASK-3070
title: chat_screen size ratchet red on dev after console decomposition wave 3
status: To Do
assignee: []
created_date: '2026-08-07 18:20'
updated_date: '2026-08-11 04:55'
labels:
  - console
  - architecture-gate
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/chat_screen.py]`
fails on dev: 18,930 lines against a budget of 18,909. Introduced by PR #1408 (console
decomposition wave 3) itself — confirmed byte-identical on a clean dev-tip worktree at
`15407a641` during the TASK-3035/3045 architecture-gate refresh (PR #1416), so it is not
an artifact of any other branch. The decomposition stream is actively shrinking this
screen; the 21-line overage is presumably transitional. Filed so the red gate is owned
rather than becoming the next "pre-existing noise" that hides something real (see
lessons-testing-evidence.md's TASK-2610 entry for how that ends). Resolve by shrinking
the screen below budget in the next wave — not by raising the budget, unless the
decomposition stream's owner explicitly decides the budget is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The chat_screen size-ratchet test passes on dev
- [ ] #2 The resolution shrinks the screen (or an explicit, documented owner decision adjusts the budget)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-10, supervisor-fleet PR 2b: still red. Verified failing at 48a54ed9c (pre-fleet dev) and at 762596846 (dev after fleet PR 2a), so this predates all fleet work. Recorded here because PR 2b adds +121 net lines to chat_screen.py (Task 4's fleet-section wiring and Task 5's coalescer/cancel handler/cost-ticker wiring), all DOM-adjacent and placed alongside the existing precedents they mirror — i.e. this PR grows an already-violated budget rather than breaching a green one. Also observed at the same commits: Tests/UI/test_console_control_bar_coalescing.py fails 2/3 on dev independently of any fleet change.
<!-- SECTION:NOTES:END -->
