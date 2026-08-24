---
id: TASK-21505
title: Design Local/Server Research Workspace and Research Runs navigation
status: In Progress
assignee: []
created_date: '2026-08-24 05:06'
updated_date: '2026-08-24 05:14'
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
- [ ] #1 A reviewed design spec defines one Research shell destination with separately routed Workspace and Runs screens while preserving the existing research route.
- [ ] #2 The design defines fail-closed Local/Server data authority, explicit inference-egress disclosure, manual cross-authority Copy, and durable receipts without silent blending or fallback.
- [ ] #3 The design maps the complete audited server control namespace into core, contextual, owner-link, capability-gated, or planned surfaces, with the primary five Studio outputs and More outputs hierarchy.
- [ ] #4 The design defines canonical ownership for sources, conversations, notes, Studio outputs, device-only overlays, and Deep Research launch/return context without inventing duplicate content stores.
- [ ] #5 The design specifies measurable Textual layout, keyboard, focus, accessibility, loading, conflict, failure, and recovery behavior.
- [ ] #6 A canonical ADR records the shell, authority, ownership, overlay, and manual-transfer decisions and links the existing governing ADRs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Incorporate the approved one-destination/two-screen IA and primary-five Studio hierarchy into the design.
2. Define the fail-closed authority, egress, async fencing, Copy receipt, ownership, overlay, and Deep Research return contracts.
3. Define complete control classification, responsive Textual behavior, accessibility, recovery, and verification requirements.
4. Write and link ADR-078 before implementation planning.
5. Self-review the spec and ADR for placeholders, contradictions, scope, and ambiguity, then request user review.

ADR required: yes
ADR path: backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md
Reason: This introduces a long-lived shell destination, new data-authority and cross-authority transfer boundaries, device-local overlay ownership, artifact association rules, and cross-screen Deep Research contracts.
<!-- SECTION:PLAN:END -->
