---
id: TASK-32618
title: >-
  Library Notes: Obsidian embeds become permanently dead text and the import
  review never says so
status: To Do
assignee: []
created_date: '2026-09-15 06:42'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D5, persona Riley / Jordan, Obsidian workflow.

What happened. An embed like an image wikilink survives Import once verbatim in the stored body and renders as literal text in Preview (B cap 27), while the PNG it points at was classified 'Unsupported · Image — not a note. Add it in Library ▸ Media' (B cap 23). The review tells you the image is not imported; it never tells you the notes that embed it will show broken syntax afterwards. The review is otherwise the strongest screen in the product -- it names a reason for every bucket and even says which frontmatter keys were dropped ('not imported: date, mood') -- which is exactly why this omission is worth closing: the surface has already earned the user's trust in its completeness.

Cause INFERRED (not traced). Distinct from the wikilink representation, which is recorded decision 32263.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When a source's embeds point at files the import will not bring in, the review says so on the affected rows or in the Unsupported group
- [ ] #2 The receipt repeats it, so a user who skipped the review still learns it
- [ ] #3 The wording names what the user will see in the note afterwards, not the internal classification
<!-- AC:END -->
