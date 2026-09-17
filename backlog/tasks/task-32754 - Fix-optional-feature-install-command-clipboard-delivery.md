---
id: TASK-32754
title: Fix optional feature install command clipboard delivery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 20:19'
updated_date: '2026-09-17 21:15'
labels:
  - bug
  - clipboard
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Missing-feature and import copy actions either rely only on unacknowledged OSC 52 delivery or use a native backend without checking delivery. Users need a usable command and truthful feedback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Copying install commands uses the native desktop clipboard where available without blocking the UI.
- [x] #2 Terminal or unavailable clipboard delivery never falsely claims confirmed success and leaves the command recoverable.
- [x] #3 Focused regression tests and native clipboard verification pass.
- [x] #4 Cancelling a copy kills and reaps its native helper before later copy attempts can write.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Repair existing clipboard behavior using the existing pyperclip dependency and Textual API; no new runtime or ownership boundary.
1. Reproduce copy false success with native unavailable and silent native failure tests.
2. Add one asynchronous checked install-command helper and connect both existing buttons; preserve visible literal commands.
3. Verify mounted handlers, native clipboard readback, focused tests and formatting; document fallback and open PR against dev.
4. Qodo follow-up: reproduce cancellation and overlapping writes, make child cleanup cancellation-aware, serialize copies per app, and add the handler docstring.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both install-command buttons use checked native copy/readback in an async child process, with truthful unconfirmed terminal fallback and visible literal commands. Cancellation and timeout kill/wait for the child process group, while per-app serialization prevents overlapping native writes. Added the handler docstring. Seven native-process and eight mounted cases pass, with independent macOS clipboard readback/restoration and Linux reruns. The final combined run has 692 passes with exactly 19 independently reproduced dev baseline exclusions; all seven preflight guards and scoped lint checks pass. No new ADR or dependency. See Docs/superpowers/reviews/2026-09-17-optional-install-stt-verification.md and backlog/docs/lessons-clipboard.md.
<!-- SECTION:NOTES:END -->
