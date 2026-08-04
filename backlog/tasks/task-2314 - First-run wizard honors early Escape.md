---
id: TASK-2314
title: First-run wizard honors early Escape
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - onboarding
  - ux
  - uat-2026-08-04
dependencies: []
priority: low
---

## Description (the why)

UAT: the first-run wizard advertises "Esc finish later" but silently ignores
Escape until it has fully settled — early keypresses during the opening
seconds are dropped, making the wizard feel frozen.

UAT finding F0a. (Positive to preserve: the explicit "Skip — explore on my
own" one-click path, and the Escape→confirm asymmetry.)

## Acceptance Criteria (the what)

- [ ] Escape pressed at any point after the wizard becomes visible reaches
      the finish-later flow (queued if mid-mount, not dropped).
- [ ] A regression test covers Escape during the wizard's first render.
