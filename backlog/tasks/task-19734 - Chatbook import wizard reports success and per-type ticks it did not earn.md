---
id: TASK-19734
title: >-
  Chatbook import wizard reports success and per-type ticks it did not earn,
  and two of its options are inert
status: To Do
assignee: []
created_date: '2026-08-21 23:40'
labels:
  - chatbooks
  - honesty
  - ux
priority: high
dependencies: []
---

## Description

Source: surfaced during **TASK-19550** (the "Create backup" checkbox that wrote
a success row and took no backup) and confirmed at exact line numbers by that
task's reviewer. 19550 is Done and removed the fake backup control; these three
are its siblings in the same wizard, all still present at this branch base
(`839819e1a`). Same theme the holistic review named: *the app asserts outcomes
it did not produce.*

### (a) A fully-skipped import reports success, with per-type ✓ rows

`Chatbooks/chatbook_importer.py:397-399`:

```python
success = (
    status.successful_items + status.skipped_items
) > 0 or status.total_items == 0
```

Skipped items count as success. The wizard then paints its per-type completion
rows gated on **manifest counts, not results** —
`UI/Wizards/ChatbookImportWizard.py:929-961` ticks
`"✓ Imported conversations"`, `"✓ Imported notes"`, `"✓ Imported characters"`
and `"✓ Imported media"` on `manifest.total_conversations > 0`,
`manifest.total_notes > 0`, etc. — followed by `"✓ Updated indexes"`,
`"✓ Import completed"` and the `"✅ Import Completed Successfully!"` banner
(`:693`).

Re-importing the same chatbook under the default conflict strategy (skip
existing) therefore shows four green ✓ rows and "Import Completed
Successfully!" directly above a summary panel reading **Imported: 0 /
Skipped: N** (`:704-711`). The importer's own log line for that case is honest
("Skipped N/M items due to conflicts", `:414`) — the UI is not.

**A whole-import-granularity fix is not enough.** Making `success` mean "at
least one item landed" only relocates the overstatement to the partial case: an
import where notes landed and characters all failed would still tick
"✓ Imported characters", because that row never consults a result. The ✓ rows
have to be driven by **per-type result counts**, which means `ImportStatus`
must carry them.

### (b) Two collected options are consumed by nothing

`ChatbookImportWizard.py:630-633` collects `preserve_timestamps` (checkbox at
`:554-559`, default ON) and `import_tags` (checkbox at `:569-571`, default ON)
into the options dict returned by `get_step_data`. A whole-tree grep finds
**zero** other production consumers of either key —
`import_chatbook` is called at `:790-800` and `:913-923` with neither. The user
toggles them; nothing reads them. (`Media/local_media_reading_service.py`'s
`_normalize_import_tags` / `_merge_import_tags` are unrelated — different
feature, different key.)

### (c) "Merge with existing tags" is default-ON and does not touch tags

The checkbox at `ChatbookImportWizard.py:576-581` reads "Merge with existing
tags", `value=True`. Its value is passed as
`prefix_imported=options.get("merge_tags", False)` (`:796`, and again at
`:918-920` with the comment `# Use merge_tags as prefix flag`).

`prefix_imported` does exactly one thing, in four places — prepend
`"[Imported] "` to a **name**: `chatbook_importer.py:506-507` (conversation
name), `:1179-1180` (note title), `:1293-1294` (character name), `:1446-1447`
(prompt name). It never reads or writes a tag. So a default-ON control labelled
as tag behaviour silently renames every imported item instead, and the actual
tag-merge behaviour the label promises does not exist.

## Acceptance Criteria

- [ ] An import in which every item was skipped does not present itself as a
      successful import — neither the banner, nor the per-type rows, nor the
      caller-visible return value asserts that items were imported
- [ ] Each per-type completion row reflects that type's own **result count**,
      not its manifest count: a type with zero successful items never shows an
      "✓ Imported …" row, and a partially-failed type is distinguishable from a
      fully-successful one
- [ ] The wizard's headline and its Imported / Skipped / Failed summary can
      never contradict each other for any combination of imported, skipped and
      failed counts (covered by tests over those combinations, including
      all-skipped and mixed partial-failure)
- [ ] `preserve_timestamps` and `import_tags` are either implemented end-to-end
      or their controls are removed — no control remains that the user can
      toggle with no effect. Per the owner's standing ruling
      (durable/pragmatic over clever), removing an inert control is preferred
      over shipping a hurried implementation behind it; a disabled or
      greyed-out control does not satisfy this, since it still reads as
      "handled"
- [ ] The "Merge with existing tags" control either performs tag merging or is
      relabelled/removed to match what it actually does; the item-renaming
      behaviour currently hidden behind it is exposed under a label that names
      it, or dropped
- [ ] Tests pin each of the above and are mutation-checked (restoring the old
      wiring makes them red)

## Notes

Filed separately from TASK-19550 (Done) because that task's disposition was
narrowly "remove the lying backup checkbox"; these three are different
mechanisms in the same screen and (a) requires a change to `ImportStatus`'s
shape, not just a UI edit. Related but distinct: TASK-279 (wrap chatbook import
in a single transaction) — the import still has no rollback, which is part of
why an over-claiming success banner is expensive here.
