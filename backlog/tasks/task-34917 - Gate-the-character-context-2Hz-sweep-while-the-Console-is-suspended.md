---
id: TASK-34917
title: Gate-the-character-context-2Hz-sweep-while-the-Console-is-suspended
status: To Do
assignee: []
created_date: '2026-10-13 14:46'
labels:
  - performance
  - ui-perf
  - wave-1-timers
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleCharacterContext arms set_interval(0.5, _sync_progress_counts) in on_mount with no screen-active gate (console_character_context.py:131). The chat screen is reusable (suspended, not unmounted) on navigation away, so this sweep iterates every CharacterConversationButton (fingerprint and dict reads each) at 2Hz while the user is on a completely different screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Interval early-returns or pauses while the owning screen is not current,Sweep resumes within one tick of returning to the Console (test)
<!-- AC:END -->
