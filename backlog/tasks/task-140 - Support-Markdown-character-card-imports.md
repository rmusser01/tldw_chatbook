---
id: TASK-140
title: Support Markdown character card imports
status: Done
assignee: []
created_date: '2026-06-28 02:43'
updated_date: '2026-06-28 02:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow the ds-native Personas character import flow to import Markdown files that embed existing supported character-card data, without introducing a new free-form Markdown schema.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Personas import picker offers Markdown files alongside existing JSON and PNG character-card imports
- [x] #2 Markdown files with YAML frontmatter import through the existing character-card parser
- [x] #3 Markdown files with fenced JSON card data import through the existing character-card parser
- [x] #4 Invalid Markdown fails without creating or selecting a character and shows the existing import failure path
- [x] #5 Existing JSON and PNG import behavior remains unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add parser regressions for Markdown YAML frontmatter, Markdown fenced JSON, and invalid Markdown.
2. Add Personas import-flow regressions for Markdown picker filters, `.md` path routing, and invalid Markdown failure.
3. Update the ds-native Personas import picker filters to include `.md` and `.markdown`.
4. Run focused parser/import tests plus `git diff --check`.
5. Check acceptance criteria, add implementation notes, and mark the task Done.

Plan: `Docs/superpowers/plans/2026-06-27-personas-markdown-character-import.md`

ADR required: no
ADR path: N/A
Reason: import picker/parser affordance and regression coverage using existing import and storage boundaries; no new long-lived Markdown schema or storage contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Markdown character-card import support to the ds-native Personas import picker by including `.md` and `.markdown` in the Character Cards filter and adding a Markdown-specific filter. Reused the existing character-card import helper and parser; no new Markdown schema was introduced. Added parser coverage for Markdown YAML frontmatter, Markdown fenced JSON, and invalid Markdown, plus Personas import-flow coverage for Markdown path routing, invalid Markdown failure behavior, and existing callable import filters.

ADR required: no
ADR path: N/A
Reason: import picker/parser affordance and regression coverage using existing import and storage boundaries; no new long-lived Markdown schema or storage contract.
<!-- SECTION:NOTES:END -->
