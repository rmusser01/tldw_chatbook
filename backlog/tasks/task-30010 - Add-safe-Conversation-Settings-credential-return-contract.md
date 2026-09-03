---
id: TASK-30010
title: Add safe Conversation Settings credential return contract
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-02 18:04'
updated_date: '2026-09-02 18:04'
labels:
  - console
  - settings
  - ux
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user leave Conversation settings to configure a missing cloud credential in canonical Settings and return to the exact originating modal draft without leaking prompt content, losing unrelated Settings edits, or attaching the draft to another conversation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Missing-credential recovery opens Settings > Providers & Models on the exact provider credential control and provides an explicit return to the originating Conversation settings modal
- [ ] #2 The suspended modal draft survives in Console-owned process-memory screen state while the typed return handoff contains no API key, prompt, prefill, raw endpoint, transcript, or arbitrary text
- [ ] #3 Return claims and acknowledges the exact single-slot handoff revision and restores only when the originating session exists with the captured Console-settings revision
- [ ] #4 Superseded, consumed, abandoned, deleted-session, temporary-session, and revision-mismatched returns fail closed without applying the draft to another conversation
- [ ] #5 Same-provider unsaved Settings changes are preserved and disclosed, while different-provider changes require Review, explicit Discard, or Return
- [ ] #6 Provider navigation and Console return contexts are typed, allowlisted, and reject unknown keys and invalid enum/revision values
- [ ] #7 Focus/view restoration, credential-only/provider-wide/without-save result copy, and absent environment-variable recovery are covered by focused state and Textual tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR paths: backlog/decisions/012-provider-credential-settings-boundary.md; backlog/decisions/033-application-session-state-ownership.md
Reason: ADR-012 owns credential mutation and targeted recovery; ADR-033 already owns memory-only screen snapshots and typed single-slot destination handoffs.

Execute the red-green checklist in Docs/superpowers/plans/2026-09-02-task-30010-conversation-settings-return-contract.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Not implemented yet.
<!-- SECTION:NOTES:END -->
