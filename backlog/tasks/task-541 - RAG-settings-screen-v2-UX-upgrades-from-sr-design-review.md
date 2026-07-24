---
id: TASK-541
title: >-
  RAG settings screen v2 UX upgrades (from sr design review)
status: In Progress
assignee: []
created_date: '2026-07-24 03:30'
updated_date: '2026-07-24 03:30'
labels:
  - rag
  - settings
  - ux
  - followup
dependencies:
  - task-503
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
V2 items from the senior UX/HCI design review of the SP3 RAG settings screen (task-503, PR #829). The review's 9 quick wins (clone-flow guidance, decoupling caption, backfill nudge, terminology unification, ⚠ legend, provenance sub-line, Delete danger styling, RAG test action, inspector fit) shipped in the SP3 PR; these are the structural/deeper upgrades deferred as v2. Review context is in the SP3 PR discussion and Docs/superpowers/qa/rag-settings-sp3-2026-07/.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Split "manage profiles" from "edit config" structurally: a distinct profile picker/list region and an editor explicitly titled with the profile being edited (removes the dropdown/editor decoupling ambiguity that the v1 caption papers over; consider preview-on-select via a Select.Changed handler).
- [ ] #2 Pre-commit re-index confirmation: when an index-determining (⚠) field changed AND the current index is built, Save confirms with the real blast radius (e.g. "This empties the current index (N vectors). Search returns nothing until you Backfill. Save anyway?").
- [ ] #3 Context-sensitive Scope Inspector: guidance follows the expanded group / focused field (reranking guidance when in Reranking, etc.) instead of one static block.
- [ ] #4 Replace state-labeled toggle buttons ("Enabled") with checkboxes or "X: On/Off" + action labels for citations and reranking; hide or dim+annotate Reranker model / Rerank results while reranking is disabled.
- [ ] #5 First-run starter panel: instead of a wall of disabled fields, a brief "Search already works on Hybrid Basic. Clone to tune, or Backfill to enable semantic results" orientation with direct actions.
- [ ] #6 Keyboard accelerators for the profile workflow (Set active / Clone / Backfill) honoring the keyboard-first posture; document them in the footer or category help.
<!-- AC:END -->
