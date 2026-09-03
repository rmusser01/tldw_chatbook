---
id: TASK-30013
title: Make local endpoint saves atomic and model provenance visible
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 18:04'
updated_date: '2026-09-03 20:09'
labels:
  - console
  - local-models
  - ux
dependencies:
  - TASK-30011
  - TASK-30012
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make local hosting setup complete and trustworthy by persisting required endpoints before applying them, identifying which models are currently served versus saved or custom, and preventing stale discovery from masquerading as verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A required unsaved endpoint exposes Save endpoint & use model with visible provider-wide/future-conversation impact
- [x] #2 Endpoint persistence fully succeeds before the active session changes or the modal dismisses; failure retains the draft and leaves the current session unchanged
- [x] #3 Conversation-only endpoint use is offered only for execution paths that actually support it
- [x] #4 Model choices visibly distinguish Served now, current cloud catalog, Saved fallback, and Custom/unverified provenance
- [x] #5 Discover models states that it lists models rather than testing generation, handles zero/one/many results accurately, and rejects stale results after provider or endpoint changes
- [x] #6 A model not reported by the current endpoint requires Keep unverified model confirmation scoped to the exact provider/endpoint/model draft generation
- [x] #7 Provider switching preserves bounded per-provider endpoint/model drafts without reusing evidence across identities
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR paths: backlog/decisions/006-provider-aware-generation-settings.md; backlog/decisions/020-automatic-model-catalog-refresh.md; backlog/decisions/033-application-session-state-ownership.md
Reason: Existing ADRs own endpoint/default persistence and cloud/local catalog authority; this task exposes provenance and orders existing mutations without adding a registry or cache.

Execute the red-green checklist in Docs/superpowers/plans/2026-09-02-task-30013-local-endpoint-model-provenance.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented source-aware model provenance, identity-fenced local discovery, exact unverified-model confirmation, and atomic endpoint persistence before session application. Conversation Settings now exposes one truthful save-and-use action for required endpoints, preserves custom-provider credentials through canonical ownership, reports partial or uncertain writes accurately, and reopens retained drafts only against the exact active conversation. Added targeted picker, discovery-race, CAS, mounted-modal, native Console lifecycle, focus, privacy, and recovery coverage. ADR required: no new ADR; existing ADR-006, ADR-020, and ADR-033 remain authoritative. Verification: 277 targeted tests passed with two documented unrelated baseline assertions deselected; Ruff, py_compile, and git diff --check passed. Added a testing-evidence lesson for strengthened postconditions exposing incomplete fixtures.
<!-- SECTION:NOTES:END -->
