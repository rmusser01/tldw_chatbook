---
id: TASK-21505
title: Design Local/Server Research Workspace and Research Runs navigation
status: Done
assignee: []
created_date: '2026-08-24 05:06'
updated_date: '2026-08-24 05:39'
labels:
  - research
  - workspace
  - ux
  - architecture
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-23-research-workspace-design.md
  - backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md
  - >-
    .impeccable/critique/2026-08-24T05-03-31Z__research-workspace-design-brief.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the product, authority, persistence, navigation, and terminal-UX contracts for a NotebookLM-like Research Workspace in Chatbook while keeping the existing durable ResearchScreen accessible as a separate Runs screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reviewed design spec defines one Research shell destination with separately routed Workspace and Runs screens while preserving the existing research route.
- [x] #2 The design defines fail-closed Local/Server data authority, explicit inference-egress disclosure, manual cross-authority Copy, and durable receipts without silent blending or fallback.
- [x] #3 The design maps the complete audited server control namespace into core, contextual, owner-link, capability-gated, or planned surfaces, with the primary five Studio outputs and More outputs hierarchy.
- [x] #4 The design defines canonical ownership for sources, conversations, notes, Studio outputs, device-only overlays, and Deep Research launch/return context without inventing duplicate content stores.
- [x] #5 The design specifies measurable Textual layout, keyboard, focus, accessibility, loading, conflict, failure, and recovery behavior.
- [x] #6 A canonical ADR records the shell, authority, ownership, overlay, and manual-transfer decisions and links the existing governing ADRs.
- [x] #7 Sources and Studio have independently testable ASCII collapse/reveal controls with deterministic focus, explicit-toggle feedback, and stored preferences that survive responsive overrides.
- [x] #8 Research ingestion creates or reuses an item in the selected authority's general Library/Media catalog, durably associates its stable identity to the captured workspace, reports partial failures by stage, and never treats a mutable tag as the relationship.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Incorporate the approved one-destination/two-screen IA and primary-five Studio hierarchy into the design.
2. Define the fail-closed authority, egress, async fencing, Copy receipt, ownership, overlay, and Deep Research return contracts.
3. Define complete control classification, responsive Textual behavior,
   two-sided ASCII collapse/focus semantics, accessibility, recovery, and
   verification requirements.
4. Write and link ADR-078 before implementation planning.
5. Amend the source contract so ingestion lands in the selected authority's
   general catalog before a stable, durable workspace association; define
   duplicate, partial-failure, retry, unlink, and no-blending behavior.
6. Self-review the spec and ADR for placeholders, contradictions, scope, and ambiguity, then request user review.

ADR required: yes
ADR path: backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md
Reason: This introduces a long-lived shell destination, new data-authority and cross-authority transfer boundaries, device-local overlay ownership, artifact association rules, and cross-screen Deep Research contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved an architectural design that adds one Research shell destination with
separately routed Workspace and Runs screens while preserving the existing
`research` route. The design establishes fail-closed Local/Server authority,
explicit processing-route consent, canonical content ownership, device-only
overlays, manual receipted Copy, and owner-routed Studio outputs.

The final review added independent Sources/Studio ASCII collapse contracts and
made Research ingestion create or reuse the selected authority's general
Library/Media item before a stable workspace association. Mutable workspace
tags remain optional projections, while Local `WorkspaceMembership` and the
server workspace-source row remain authoritative. Documentation verification
used scoped `git diff --check`, exact requirement searches, and Backlog plain
rendering; no production code or runtime schema changed in this design task.
<!-- SECTION:NOTES:END -->
