---
id: TASK-16242
title: Align agent rail provider fixture with bridge contract
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 10:14'
updated_date: '2026-08-14 10:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the agent-rail subagent summary regression test exercising the bridge after provider resolutions became typed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test supplies a valid `ConsoleProviderResolution`.
- [x] #2 The subagent markup regression remains covered.
- [x] #3 The full agent-rail module and affected sweep chunk are green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the stale opaque resolution fails before the scripted gateway records a spawn.
2. Replace only that fixture value with the minimal typed Groq resolution used by current bridge tests.
3. Run the focused regression, full agent-rail module, sweep chunk, and static checks.

ADR required: no
ADR path: N/A
Reason: This is a test-fixture compatibility update for an existing typed boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the opaque `object()` resolution with the minimal typed Groq provider resolution already used by bridge tests.
- Preserved the scripted gateway and bracket-rendering assertion, so the test once again reaches and records the subagent spawn.
- Verified the focused regression, all 33 agent-rail tests, and the exact sweep chunk (259 passed). Ruff check and diff hygiene pass; whole-file Ruff format remains pre-existing red outside the changed lines.
<!-- SECTION:NOTES:END -->
