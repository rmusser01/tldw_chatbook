---
id: TASK-30011
title: Separate Conversation Settings operability from verification evidence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-02 18:04'
updated_date: '2026-09-02 18:04'
labels:
  - console
  - settings
  - ux
dependencies:
  - TASK-30010
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give Conversation settings an honest, structured readiness model that distinguishes whether Chatbook can attempt a send from what configuration, connection, model-listing, credential, or generation evidence has actually been observed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Operability is exactly Ready to send or Not ready, while configuration, credential, endpoint, model, and generation evidence remain separate typed facets
- [ ] #2 Ready to send means no known local blocker and never implies provider acceptance or successful generation
- [ ] #3 One deterministic reason-code precedence selects the highest-priority blocker and its direct recovery action without composing contradictory prose
- [ ] #4 Missing credentials, invalid/unsaved endpoints, missing models, refused/timeout/auth failures, stale evidence, and unverified-but-configured states have distinct sanitized outcomes
- [ ] #5 Provider display names and credential source provenance are visible without exposing credentials, headers, raw responses, or credential-bearing URLs
- [ ] #6 The Console rail, setup card, Conversation settings modal, and Settings provider test consume compatible structured facts and cannot overclaim one another
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR paths: backlog/decisions/006-provider-aware-generation-settings.md; backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/012-provider-credential-settings-boundary.md; backlog/decisions/033-application-session-state-ownership.md
Reason: This refines the existing provider-readiness and evidence contracts without changing provider execution, credential ownership, or persistence boundaries.

Execute the red-green checklist in Docs/superpowers/plans/2026-09-02-task-30011-conversation-settings-readiness-evidence.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Not implemented yet.
<!-- SECTION:NOTES:END -->
