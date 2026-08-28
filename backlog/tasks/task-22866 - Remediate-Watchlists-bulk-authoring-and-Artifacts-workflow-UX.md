---
id: TASK-22866
title: Remediate Watchlists bulk authoring and Artifacts workflow UX
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-27 04:14'
updated_date: '2026-08-28 15:43'
labels:
  - watchlists
  - ux
  - textual
  - briefings
dependencies:
  - TASK-22862
  - TASK-22863
  - TASK-22864
  - TASK-22865
references:
  - >-
    Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-08-27-watchlists-feed-and-interface-uat-remediation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make direct Watchlists inspection and recovery support bulk source entry, multi-source membership, legible operational states, and reliable briefing visibility at supported terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sources provides multiline/bulk entry with persistent labels, row-level validation/duplicate feedback, draft preservation, and an explicit partial-result decision.
- [ ] #2 Users can keyboard-select multiple sources, understand filtered/select-all/range semantics, and create a Watchlist from the selected set without repeated membership dialogs.
- [ ] #3 Implemented shortcuts are discoverable in valid footer/command-palette hints; primary meanings never depend on tooltips or color alone.
- [ ] #4 Artifacts foregrounds Generate/Schedule when empty and moves downstream actions into selected-briefing context or a labeled disclosure.
- [ ] #5 The collection automation receipt shows interval, app-open limitation, next eligibility, last attempt/success, reload state, and attention/recovery state.
- [ ] #6 Briefing refresh/generation retains the last good table/body, shows inline progress, preserves content on failure with Retry, and provides a recoverable reload diagnostic when durable storage and the pane disagree.
- [ ] #7 Production-shaped Textual tests cover first-time and power-user paths at the supported 160x42 pressure point and a normal size, including focus order, Escape, draft preservation, stale/error states, and receipt deep-links.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin canonical ID-based source selection semantics, including range, visible-only toggle, hidden selection reporting, pruning, and permitted keyboard bindings.\n2. Add a bulk-source modal over the TASK-22862 exact-batch service with ordered row outcomes, persistent drafts, and an explicit partial-result decision; keep single-source authoring.\n3. Route selected canonical source IDs through the existing atomic Watchlist bundle service, with discoverable focus-aware shortcuts and production-sized layout.\n4. Make Artifacts progressive and stale-while-refreshing, retain last-good content, foreground Generate/Every 24 hours when empty, and render honest automation/storage recovery receipts.\n5. Rebuild canonical CSS, run production-shaped 160x42 and normal-size keyboard/focus/recovery tests, targeted preservation suites, Ruff, diff checks, Impeccable review, and independent code review.\n\nADR required: no\nADR path: N/A\nReason: This is workflow/UI remediation over the durable operation, scheduling, permission, and application ownership boundaries already established by ADR-019 and ADR-032; no new storage, security, runtime-owner, or cross-module service contract is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented canonical-ID multi-selection, a bounded exact-batch source modal,
and atomic Watchlist creation from selected sources. Partial outcomes now pause
for an explicit decision; admitted writes cannot be dismissed; successful
continuation opens All Sources with its IDs selected and performs no implicit
membership write. Artifacts now uses stale-while-refreshing operational states,
retains readable briefing/provenance content through refresh and generation
failures, scopes completion receipts to their originating Watchlist, and keeps
downstream actions under a stable labeled disclosure. Updated canonical CSS,
the Watchlists user guide, and production-shaped Textual coverage at 160x42 and
normal size. No schema, provider, scheduler, permission, egress, or ownership
boundary changed; existing ADR-019 and ADR-032 remain the governing decisions.
