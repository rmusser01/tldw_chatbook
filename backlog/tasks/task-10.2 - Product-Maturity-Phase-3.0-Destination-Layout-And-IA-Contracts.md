---
id: TASK-10.2
title: 'Product Maturity Phase 3.0: Destination Layout And IA Contracts'
status: Done
assignee: []
created_date: '2026-05-06 01:58'
labels:
  - product-maturity
  - ux
  - phase-3
  - layout-contracts
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md
  - Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md
  - Docs/superpowers/trackers/product-maturity-roadmap.md
parent_task_id: TASK-10
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define destination-level layout and IA contracts before additional Phase 3 Knowledge and Study visual work continues so future screen changes are usable, consistent, and grounded in the approved product model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every top-level destination has an approved layout contract with binding regions, focus path, state handling, Console handoff behavior, and source/runtime authority rules.
- [x] #2 Major subflows under Library, Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, and Skills have owner placement and concise layout contracts.
- [x] #3 ASCII wireframes exist for every top-level destination and major subflow, with generated image references treated as non-binding inspiration only.
- [x] #4 The product-maturity roadmap and tracker record Phase 3.0 as a prerequisite before further Phase 3 visual rewrites.
- [x] #5 QA walkthrough criteria require compact, default, and large terminal verification that affected screens are usable, not merely rendered.
<!-- AC:END -->

## Implementation Plan

1. Align runtime route metadata so `study` is Library-owned.
2. Add focused contract regressions for Phase 3.0 docs, route inventory, tracker, QA evidence, prompt manifest, and backlog task in `Tests/UI/test_product_maturity_phase3_layout_contracts.py`.
3. Record Phase 3.0 QA evidence and tracker closeout evidence.
4. Add a non-binding destination image prompt manifest.
5. Run focused verification before checking ACs and closing the task against `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`.

## Implementation Notes

Route metadata treats `study` as Library-owned and keeps Study Dashboard, Flashcards, and Quizzes under Library instead of adding a separate top-level destination. Doc-focused regressions were added in `Tests/UI/test_product_maturity_phase3_layout_contracts.py`; QA evidence and the non-binding image prompt manifest were recorded; final focused verification passed.
