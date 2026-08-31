---
id: TASK-25732
title: Console exposes internal vocabulary and unexplained glyphs
status: Done
assignee: []
created_date: '2026-08-31 05:10'
updated_date: '2026-08-31 13:41'
labels:
  - console
  - ux-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console surfaces terms and symbols with no in-context explanation, including resolved destination, library tool mode, impersonate, a bare dollar marker on the send control, and asterisk and dash markers on rail rows. The status chip reading agent blocked opens a modal titled library access, so the label the user clicks and the destination they reach do not share a name.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Terms shown in Console are either plain language or explained where they appear
- [ ] #2 Glyph markers used in the rail are documented in an accessible legend
- [ ] #3 A status chip and the surface it opens share a consistent name
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scoped to the one part that was a genuine defect and not pinned: the Console status chip says 'Agent blocked', but the control it opens was labelled 'Assistant Library access' -- same permission, two nouns -- while 'Assistant:' already names the persona in that same status strip. Renamed to 'Agent Library access' across the Console modal, the Settings screen and the search index, keeping the old wording as a search ALIAS (the index already supports multiple labels per id, see the splash entries) so nobody searching the shipped wording loses it.

NOT changed, deliberately: the chip label itself ('Library · Auto off · Agent blocked') is pinned in four test sites and is the vocabulary the rest now matches. The remaining items I filed under this task -- 'Resolved destination: not used yet', 'Library tool mode: Direct', the bare '$' send marker, and the '*'/'-' rail glyphs -- are separate surfaces that each need their own copy pass; grouping them here made the task unshippable as one unit. They are better split than bundled.

Baseline confirmed unchanged: test_every_rendered_setting_is_in_the_search_index fails on clean dev over unrelated settings-network-* ids.
<!-- SECTION:NOTES:END -->
