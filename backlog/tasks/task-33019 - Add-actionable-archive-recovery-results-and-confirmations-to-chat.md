---
id: TASK-33019
title: Add actionable archive recovery results and confirmations to chat
status: Done
assignee:
  - '@codex'
created_date: '2026-09-27 15:26'
updated_date: '2026-09-27 19:22'
labels: []
dependencies:
  - TASK-33018
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users identify and restore exact conversations from accessible Console controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Structured results offer Preview and valid recovery actions with explicit workspace effects.
- [x] #2 Buttons and agent requests share confirmation and receipts, including stale and partial outcomes.
- [x] #3 Compact and wide keyboard flows preserve drafts and open the original chat only on explicit action.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md
Reason: direct implementation of the approved native recovery controls and shared coordinator boundary; no new ADR.
1. Read Task6 brief, design-language constitution, typed runtime presentation API and coordinator contract.
2. Add failing mounted Console tests for result-button and agent-request confirmation through actual production dispatch, real SQLite and Pilot controls.
3. Compose token-based cards, trusted view hooks, scope and Restore as controls, immutable source ownership, Preview and explicit original-ID Open.
4. Verify compact/wide keyboard flows, stale/imported/expiry/remount handling, draft/branch preservation and targeted existing UI regressions; rebuild CSS and run governance guards.
5. Run scoped static checks, self-review, commit only Task6 files and request independent review. No full sweep or remembered permission control.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added runtime-owned native result/confirmation cards, explicit scopes and receipts, original-ID Preview/Open and fresh Restore as consent under ADR-166. Mounted real-store flows qualify both entry points at compact/wide sizes, drafts/history/branches, expiry/stale/partial/remount/imported evidence boundaries. Independent review and scoped fix review clean; expected rename refusal now recoverable and literal target stays beside keyboard-accessible Confirm. 50 distinct final task nodes plus 10 fix nodes qualified; required regression selection 88 passed/8 failed with seven pre-task failures and one introduced bare-DOM issue fixed and revalidated. Scoped lint/format, generated CSS/token guards and diff checks passed. Existing baseline failures/architecture ceilings remain documented; complete-journey and real-model evidence are Task33020. Canonical stylesheet source correction is reflected in plan; no new boundary beyond ADR-166.
<!-- SECTION:NOTES:END -->

## Filing provenance

Created with Backlog CLI in a temporary Git workspace using the project configuration, after the production CLI repeatedly scanned local branch history. Copied into this project with a unique ID above the refreshed all-ref/worktree ceiling and current local task IDs; dependencies and contents were verified.
