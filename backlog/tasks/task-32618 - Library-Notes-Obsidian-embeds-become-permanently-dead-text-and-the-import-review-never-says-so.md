---
id: TASK-32618
title: >-
  Library Notes: Obsidian embeds become permanently dead text and the import
  review never says so
status: Done
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
- [x] #1 When a source's embeds point at files the import will not bring in, the review says so on the affected rows or in the Unsupported group
- [x] #2 The receipt repeats it, so a user who skipped the review still learns it
- [x] #3 The wording names what the user will see in the note afterwards, not the internal classification
<!-- AC:END -->

## Implementation Plan

1. Find why an embed survives verbatim (it is excluded from the link grammar
   on purpose) and where a per-row fact is produced.
2. Count embeds at the producer, not at the renderer.
3. Say it on the review row and repeat it on the receipt.

## Implementation Notes

**Cause, now traced (the task recorded it as INFERRED).** `WIKILINK_SCAN`'s
`(?<!!)` deliberately refuses `![[...]]`, so an embed is never recorded as a
link and never rewritten: the executor leaves the span exactly as written and
Preview prints it as text. That is correct -- Chatbook cannot render another
file inside a note -- and is a different decision from the wikilink
representation (32263). What was missing was saying so.

**What changed.** `EMBED_SCAN` mirrors `WIKILINK_SCAN` in the same module
(code spans matched first, so a note DOCUMENTING embed syntax is not a note
with a dead embed), with `embedded_file_count` /
`item_embedded_file_count` / `embedded_file_plan_count` beside the existing
`resolved_wikilink_count`. Only payloads the item actually WRITES are counted:
a create writes all of them, an update only when it is replacing content, a
skip none.

The review row's effect summary gains "2 embeds show as `![[…]]` text, not the
files" (AC#1, AC#3 -- what the reader will SEE, not the classification), and
the receipt repeats the batch total as "3 embedded files left as text" (AC#2),
carried on the workflow snapshot the same way `latest_resolved_links` is,
because the approved plan is discarded when the next selection starts.

**Scope choice.** Every embed is counted, not only those pointing at files the
import will not bring in. AC#1 names the narrower set, but the wider one is
true of all of them -- an `![[Another note]]` whose target IS imported still
renders as literal text -- and computing "resolvable" embeds would state a
distinction the note does not have.

**Files.** `Notes/note_import_plan_models.py`,
`Library/library_note_import_state.py`,
`Tests/UI/test_library_notes_w5_import_preview.py`,
`Docs/User_Guide/library/notes.md`.
