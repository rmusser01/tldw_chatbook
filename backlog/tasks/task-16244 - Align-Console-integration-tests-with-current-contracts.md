---
id: TASK-16244
title: Align Console integration tests with current contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 10:52'
updated_date: '2026-08-14 11:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Console UI integration chunk after intentional runtime and layout contracts evolved, without changing production behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cost fingerprint streaming coverage advances payload revision without invalidating the active response.
- [x] #2 Dictionary send integration uses current typed provider and agent seams.
- [x] #3 Fleet help reachability assertions tolerate real wrapping while proving content is scroll-reachable.
- [x] #4 Left rail geometry assertions target the current split conversation section.
- [x] #5 Focused and chunk verification pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a test-only alignment with existing provider, transcript-tree, and Console layout contracts.

1. Preserve the strict RED evidence for the five chunk failures and trace each against current production behavior.
2. Update only the stale test fixtures and assertions, using typed provider seams and non-destructive store mutations.
3. Run focused, module-level, static, and exact chunk verification.
4. Document the evidence and close the task without changing production code.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the destructive mid-stream ancestor edit with a session-settings revision change, preserving the live assistant descendant while still exercising the fingerprint revision gate.
- Updated dictionary integration doubles to return `ConsoleProviderResolution`, honor the live agent-runtime config refresh, and accept the bridge's current keyword-only surface without making network calls.
- Made fleet-help assertions compositor-honest by checking wrapped marker copy across scroll checkpoints, and aligned the rail geometry assertion with the split Conversation section.
- Verification: 166 tests passed across the four affected modules; the exact 25-file chunk passed 429 tests. `git diff --check` passed. Ruff's two unused imports and whole-file formatting drift reproduce on HEAD and were left outside this test-only task.
- ADR check: no ADR required; production behavior and architecture were unchanged.
<!-- SECTION:NOTES:END -->
