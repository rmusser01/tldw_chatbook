---
id: TASK-563
title: >-
  Image-gen P3 polish follow-ups
status: In Progress
assignee: []
created_date: '2026-07-24 23:34'
updated_date: '2026-07-25 01:30'
labels:
  - image-generation
  - personas
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred polish from the image-gen P3 whole-branch review (PR #859: ✨Generate for character avatar + expression slots; spec `Docs/superpowers/specs/2026-07-24-image-gen-p3-expression-generation-design.md`). None are defects in shipped behavior — the High/Medium findings were fixed pre-merge. Distinct from [[task-497]]/[[task-558]]/[[task-559]] (P1/P2a/P2b polish) and the pre-existing test failures ([[task-564]]).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The Expressions action row remains usable at narrow terminal widths (at 120 cols Import/Export currently overflow the row; at 80 cols Generate-all does too — pre-existing since the row shipped). Wrap, scroll, or fold the set-actions into a menu; pin with a width-parameterized test replacing the current 200-col-only claim.
- [ ] A generation in progress shows an in-slot "Generating…" affordance (spec §1 promised it; today the only feedback is the completion/failure notify or the "already generating" refusal on a second click).
- [ ] "✨ Generate all" asks for confirmation when it would overwrite existing images (staged avatar or populated expression slots) — the sweep's blast radius exceeds the per-slot regenerate-by-click contract.
- [ ] The Generate-all summary counts only genuinely persisted slots (today `_apply_expression_upload` swallows its own DB-write failure and the sweep counts that slot as a success — user sees both the per-slot error AND an inflated "k/4").
- [ ] Cosmetics: a one-line comment at `_after_character_save`'s record-reread-failure fallback noting the style-reset invariant holds via the closed editor gates; the generate-all narrow race (per-slot key freed mid-loop allowing a duplicate regeneration) either guarded or documented as accepted last-write-wins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
