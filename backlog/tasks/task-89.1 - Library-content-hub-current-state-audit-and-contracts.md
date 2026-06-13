---
id: TASK-89.1
title: Library content hub current-state audit and contracts
status: Done
labels:
- library
- ux
- audit
- contracts
priority: high
parent_task_id: TASK-89
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit the current Library destination and its related routes to define the content-hub contract before implementation begins.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current Library modes, sub-screens, route handoffs, services, and QA evidence are inventoried from current dev.
- [x] #2 Actual CDP/Textual-web baseline screenshots are captured for the current Library destination and any Library-owned sub-routes before implementation begins.
- [x] #3 The audit identifies which areas are destination-native, which are intentional route handoffs, and which are placeholders or broken flows.
- [x] #4 The contract defines content-hub ownership, action availability, workspace eligibility, visual layout expectations, and handoff behavior for each Library mode.
- [x] #5 ADR check is documented with either an existing ADR link or a reason no ADR is required.
- [x] #6 A staged implementation plan is produced before code changes begin, with each slice explicitly bound to the contract produced here.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This slice audits current Library behavior and defines implementation contracts; it does not itself change storage/schema, sync policy, service boundaries, security posture, or long-lived runtime ownership.

1. Re-read current Library code paths, Library-owned sub-routes, related QA notes, and citation/workspace specs from the current worktree.
2. Inventory Library modes, route handoffs, source-of-truth owners, existing services, and visible placeholders or broken flows.
3. Capture CDP/Textual-web baseline screenshots for the current Library destination and Library-owned sub-routes where possible.
4. Draft the content-hub contract: module ownership, detail/inspector state, action availability, workspace eligibility, visual layout expectations, and handoff behavior by mode.
5. Produce a staged implementation plan for the child tasks, explicitly binding each slice to the contract.
6. Update TASK-89.1 notes with findings, evidence paths, ADR decision, staged plan, and residual risks before marking ready for review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the Library content-hub audit and contract at `Docs/superpowers/specs/2026-06-09-library-content-hub-contract-design.md`. The contract inventories `LibraryScreen`, the Library Conversations route, Library Search/RAG, Collections, workspace behavior, and route handoffs to Notes, Media, Ingest, and Study. Captured actual Textual-web/CDP PNG evidence for the main Library screen, focus sweep, command palette, and Library Conversations route under `Docs/superpowers/qa/product-maturity/screen-qa/library/source-workbench-audit/`.

Key finding: the main Library destination is already destination-native, but the corrected product purpose is a landing page and center hub for media/ingested content, not a primary source-selection or conversation-starting surface. Search/RAG readiness needs hardening, and `LibraryConversationsScreen` is screenshot-confirmed as a placeholder. The staged plan now binds TASK-89.2 through TASK-89.8 to the content-hub contract and preserves the rule that Library can browse/search all workspace content while active workspace gates staging/manipulation.

ADR required: no. This was an audit/contract slice only and did not change app behavior, storage, sync policy, security policy, runtime ownership, or service boundaries.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Library current-state audit and content-hub contract are complete. Follow-on implementation should start with TASK-89.2 and must use the contract as the gate for hub status, module ownership, inspector behavior, workspace eligibility, and handoff semantics.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
