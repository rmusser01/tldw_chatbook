---
id: TASK-32235
title: >-
  Library: one state-glyph legend — `○` means disabled, unchecked and skipped;
  leading `▸` means selected row and disclosure
status: To Do
assignee: []
created_date: '2026-09-10 14:52'
labels:
  - library
  - ux
  - accessibility
  - design
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`○ Export selected` (disabled), `○ Media (0)` in the Search/RAG Sources panel (unchecked) and `○ skipped · weird.xyz` (a settled outcome) share one glyph; a leading `▸` marks the selected rail row and, one row below, an expandable node. The guide codifies `○` as the disabled marker and `▸` as the selected row; neither holds. The inline blocked-reason line already carries the meaning for gated actions. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A documented legend: `▸/▾` trailing for disclosure, `█` leading for the keyboard cursor, `☐/☑` for selection, `✓/✗/–` for settled outcomes; blocked actions keep their inline reason and drop the glyph
- [ ] #2 Every Library canvas uses the legend; the guide's glyph sentences match
- [ ] #3 Captures at 235x52 and 100x30 pin one meaning per glyph
<!-- AC:END -->
