---
id: TASK-30014
title: Harden Conversation Settings verification accessibility and geometry
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-02 18:04'
updated_date: '2026-09-02 18:04'
labels:
  - console
  - ux
  - accessibility
  - testing
dependencies:
  - TASK-30013
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish Conversation settings with honest supported connection probes, an optional explicitly authorized paid generation check, deterministic keyboard/accessibility behavior, and verified rendering at compact, normal, and wide terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Test connection appears only when an existing provider service supports a meaningful bounded non-generating probe; other providers state that no non-billable live check is available
- [ ] #2 Any paid generation check is optional, provider-matrix-backed, cancellable, sanitized, and requires explicit usage confirmation before every request
- [ ] #3 Network workers reject stale results, use bounded timeouts, preserve the draft on failure, and never display or log secrets, raw response bodies, or credential-bearing URLs
- [ ] #4 Visible labels, Textual-supported accessible names/descriptions, selected-state semantics, non-color-only statuses, single announcements, and deterministic focus order are covered by focused tests
- [ ] #5 The approved save accelerator is discoverable and does not shadow terminal-convention or global bindings
- [ ] #6 At 80x24, 100x30, and 160x40 the Connection flow and completion actions remain visible/reachable without horizontal scrolling; compact actions stack or wrap with full labels
- [ ] #7 Isolated live UAT captures the cloud deep-link/return, local failure/recovery, provenance, disclosure, keyboard, disabled-action, and Ready to send evidence without touching the real profile
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR paths: backlog/decisions/006-provider-aware-generation-settings.md; backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/012-provider-credential-settings-boundary.md; backlog/decisions/033-application-session-state-ownership.md
Reason: This hardens existing UI, provider probe, and credential-safety contracts; paid generation remains explicit and optional and does not become a new runtime boundary.

Execute the red-green checklist in Docs/superpowers/plans/2026-09-02-task-30014-verification-accessibility-geometry.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Not implemented yet.
<!-- SECTION:NOTES:END -->
