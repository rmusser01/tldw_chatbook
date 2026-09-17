---
id: TASK-32754
title: Fix optional feature install command clipboard delivery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 20:19'
updated_date: '2026-09-17 20:45'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Repair existing clipboard behavior using the existing pyperclip dependency and Textual API; no new runtime or ownership boundary.
1. Reproduce copy false success with native unavailable and silent native failure tests.
2. Add one asynchronous checked install-command helper and connect both existing buttons; preserve visible literal commands.
3. Verify mounted handlers, native clipboard readback, focused tests and formatting; document fallback and open PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both existing install-command buttons now use checked native delivery off the UI thread with bounded subprocess cleanup and an honest Textual fallback. Commands preserve quoted extras and literal development instructions; Library details open for manual recovery. Added 8 mounted and 5 native-process regressions, verified independent macOS paste and clipboard restoration, and passed Linux checks. No new ADR or dependency. See Docs/superpowers/reviews/2026-09-17-optional-install-stt-verification.md for validation and existing baseline failures, and backlog/docs/lessons-clipboard.md for the delivery/timeout lesson.
<!-- SECTION:NOTES:END -->
