---
id: TASK-40035
title: Establish real admission before the Inspector first-poll assertion
status: In Progress
created_date: 2026-11-09 09:07
labels:
- ci
- console
- test-isolation
priority: high
references:
- Tests/UI/test_console_switcher_local_probe.py
- tldw_chatbook/UI/Screens/chat_screen.py
- backlog/docs/lessons-testing-evidence.md
updated_date: 2026-11-09 09:37
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI18 captured a stale header immediately after the first manual credential poll, even though a later poll installed the settled evidence. A controlled original case reproduced the stale header while a real foreign config lock correctly caused the first poll to defer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The available-control fixture waits within the existing setup bound before changing evidence, then proves the first real poll installs it and paints the header synchronously.
- [x] #2 The original held-lock deferral, readiness words, Inspector rows, and composer hint assertions pass at the original limits, including the controlled foreign-lock case.
- [ ] #3 The correction passes independent review of the published commit and its own hosted checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test setup only. Production credential polling and native config admission are unchanged.

1. Preserve the hosted UI18 failure and reproduce the first stale header under real foreign config custody.
2. Use the existing bounded warm setup to establish real config and display admission before settling evidence in the available phase. Keep settlement, the first poll, snapshot verification, and the header read synchronous. Preserve the existing foreign-held branch and post-release recovery checks.
3. Run the same custody control and original switcher cases with their existing limits. Require exact review and the published head’s own hosted checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
UI18 captured the first poll's old header, while a later poll published the real evidence version. The controlled original case reproduced that symptom with a foreign thread holding the real rebuild lock. The fixture now uses its existing warm-up bound to verify checked display admission and take the real config locks without blocking before it changes evidence. It settles once, calls the actual poll, and verifies the header and exact evidence version synchronously before any queued repaint. The held-lock and post-release assertions remain. The same foreign-lock control passed and the holder reaped. The exact hosted refusal branch was unobserved.

Both affected files passed all 58 original cases at 180 seconds per case. Scoped static comparison found only inherited findings. The rebased candidate passed all 15 original browser cases and the original mounted journey. Independent exact-commit review and the successor's own hosted checks remain due.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
