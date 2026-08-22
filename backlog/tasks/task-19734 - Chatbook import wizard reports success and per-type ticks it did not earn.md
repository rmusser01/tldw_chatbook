---
id: TASK-19734
title: >-
  Chatbook import wizard reports success and per-type ticks it did not earn,
  and two of its options are inert
status: In Progress
assignee:
  - '@claude'
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

- [x] An import in which every item was skipped does not present itself as a
      successful import — neither the banner, nor the per-type rows, nor the
      caller-visible return value asserts that items were imported
- [x] Each per-type completion row reflects that type's own **result count**,
      not its manifest count: a type with zero successful items never shows an
      "✓ Imported …" row, and a partially-failed type is distinguishable from a
      fully-successful one
- [x] The wizard's headline and its Imported / Skipped / Failed summary can
      never contradict each other for any combination of imported, skipped and
      failed counts (covered by tests over those combinations, including
      all-skipped and mixed partial-failure)
- [x] `preserve_timestamps` and `import_tags` are either implemented end-to-end
      or their controls are removed — no control remains that the user can
      toggle with no effect. Per the owner's standing ruling
      (durable/pragmatic over clever), removing an inert control is preferred
      over shipping a hurried implementation behind it; a disabled or
      greyed-out control does not satisfy this, since it still reads as
      "handled"
- [x] The "Merge with existing tags" control either performs tag merging or is
      relabelled/removed to match what it actually does; the item-renaming
      behaviour currently hidden behind it is exposed under a label that names
      it, or dropped
- [x] Tests pin each of the above and are mutation-checked (restoring the old
      wiring makes them red)

## Implementation Plan

1. Give `ImportStatus` a per-type ledger (`ImportTypeResult`) and record every
   success/skip/failure through it, so results exist at the granularity the
   rows need.
2. Name the outcomes once (`imported` / `partial` / `skipped` / `failed` /
   `empty` / `excluded` / `none`) and derive both the whole-import verdict and
   each type's verdict from that one vocabulary.
3. Replace the manifest-gated ✓ rows with a pure describer over one
   `ImportTypeResult`, and the pre-baked success banner with a describer over
   the whole `ImportStatus`.
4. Remove the two inert controls; relabel the mislabelled one to the behaviour
   it actually has, and rename its options key to match.
5. Pin it: a born-red end-to-end re-import test, an exhaustive
   headline-vs-summary consistency sweep, and a structural guard that no
   per-type row may hard-code an outcome state or see a manifest.

## Implementation Notes

**(a) Per-type results.** `ImportStatus` now carries `by_type:
{ContentType: ImportTypeResult}` with `attempted / excluded / successful /
skipped / failed`, populated through `plan()` / `record_success()` /
`record_skipped()` / `record_failure()` (all 42 former `status.X += 1` sites
in the six `_import_*` methods now go through these, so the aggregate and the
per-type ledger cannot drift; a test asserts they sum). `attempted` is
recorded up front, so a type that dies before recording anything reads as
`failed` rather than silently absent — silence must not read as success.

`ImportTypeResult.outcome` and `ImportStatus.outcome` share one vocabulary.
The wizard's four rows are painted by `_paint_type_result_rows`, whose text
comes from the pure `describe_type_result(result, noun)` — it takes a result
and a noun, so no manifest value can reach a row. The completion banner is
produced by `describe_import_outcome(status)` and carries the same numbers
the summary panel shows, which is what makes AC#3 checkable: a parametrised
sweep over every combination of imported/skipped/failed asserts the banner
never claims success with zero imports and never claims a clean import when
anything was skipped or failed.

`total_items` now counts what the run will actually attempt (media the user
opted out of, and content types this importer cannot write, are excluded and
reported as such) — otherwise "Total 5 / Imported 4 / Skipped 0 / Failed 0"
carries a permanent unexplained shortfall, which is the same contradiction in
another place.

**The return value.** `success` still means "the import ran without a fatal
error", because an all-skipped re-import is not an *error* and routing it to
"❌ Import Failed" would be a fresh lie. What changed is that it can no longer
be *read* as "items were imported": the message it is returned with now says
"No items were imported: N/M items were already present and were skipped",
and `status.outcome` names the case for callers that need to branch. The one
behavioural change: a run where nothing landed and something failed (skips
plus failures, zero successes) is now `success=False`, where it used to be
True.

**(b) Two dead controls — REMOVED.** `preserve_timestamps` and `import_tags`
had zero production consumers. Removing rather than implementing, and not
merely disabling (a greyed box still reads as "handled"): neither capability
exists anywhere below the checkbox. The importer writes its own `created_at`
handling per type and there is no timestamp-preservation seam to switch on;
tags are not a togglable path at all (note keywords are not even stored — see
the standing comment in `_import_notes`; media keywords and character tags are
imported unconditionally). Implementing either is a scoped piece of work, not
a call to make behind an existing checkbox.

**(c) The mislabelled control — RELABELLED.** "Merge with existing tags" was
passed as `prefix_imported`, whose only effect is prepending `"[Imported] "`
to a name in four places. The behaviour is real, reachable and useful, so it
is kept and the label now names it: *Prefix imported item names with
"[Imported]"*. Default stays ON — the other in-app import path
(`Tools_Settings_Window`) hard-codes `prefix_imported=True`, so OFF would have
made the two disagree; with a truthful label a default-ON is a disclosed
choice rather than a hidden rename. The options key was renamed `merge_tags`
→ `prefix_imported` so the wizard, the option dict and the importer parameter
all use one name.

**Guard extension.** task-19550's AST guard is extended, not weakened: its
TODO/"For now" scan now covers *every* function that paints a status row
(previously only those with a literal `"completed"`), and its allowlist of
hard-coded "completed" rows shrinks to prepare/indexes/finalize — the four
per-type rows left it because a literal outcome state is exactly the bug. In
their place, `test_chatbook_import_result_honesty.py` adds three structural
pins: no per-type row id may appear with a constant state other than
`"active"`; `describe_type_result` takes `(result, noun)` and its body
contains no manifest reference; and `_paint_type_result_rows` reads only
`self.import_status`. Mutation-checked: re-adding one
`_update_status("status-notes", "completed", ...)` reddens four tests.

**Born red.** The same probe run unchanged against the branch base
(`3193816e7`) prints
`{'status-conversations': '✓ Imported conversations', 'status-notes': '✓
Imported notes', 'status-characters': '✓ Imported characters', 'status-media':
'✓ Imported media'}`, headline `✅ Import Complete!`, banner `✅ Import
Completed Successfully!` — over `IMPORTED STAT: 0` — and lists the options
`['import_tags', 'merge_tags', 'preserve_timestamps', ...]`. On this branch
the same probe passes.

**Independent review fixes (2 commits on top).** Two claims in these notes
did not survive being probed:

1. *"a test asserts they sum"* was true only for the successful column.
   `test_per_type_results_sum_to_the_aggregate_counts` ran a clean import, so
   its skipped and failed assertions were `0 == 0`: deleting
   `record_failure`'s ledger write reddened **nothing** in the whole
   `Tests/Chatbooks/` suite, and deleting `record_skipped`'s reddened only
   the end-to-end re-import test. Added
   `test_the_ledger_also_sums_when_items_skip_and_fail`, which drives one
   import producing all three outcomes at once (a pre-seeded note skips,
   every character write raises, the rest lands). Both recorder mutations now
   redden it.

2. *"the headline and the summary can never contradict each other"* (AC#3)
   was swept only over statuses built as `total = imported + skipped +
   failed`, which excludes the one case the new pre-dispatch `plan()` call
   creates. A type that bails out before recording anything rendered
   `⚠️ Import finished — 1 of 5 item(s) imported.` over **Total 5 / Imported 1
   / Skipped 0 / Failed 0** — four items missing and "0 failed" on screen.
   `ImportStatus` now exposes `accounted_items` / `attempted_items` /
   `unaccounted_items`, the partial, skipped and failed banners name the
   shortfall ("… (4 unaccounted for)", the same phrase the per-type rows
   already used), and the AC#3 sweep gained an `unaccounted` axis plus an
   assertion that the banner must name any difference between Total and the
   three counters.

**Modified/added files.** `tldw_chatbook/Chatbooks/chatbook_importer.py`,
`tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`,
`tldw_chatbook/css/features/_wizards.tcss` (+ regenerated bundle),
`Tests/Chatbooks/test_chatbook_import_wizard_backup_honesty.py` (guard
extension), `Tests/Chatbooks/test_chatbook_import_result_honesty.py` (new).
No `Docs/User_Guide/` page documents this wizard (checked: `artifacts.md`
mentions Chatbooks but contains no import or wizard content), so no user-guide
update applies.

## Notes

Filed separately from TASK-19550 (Done) because that task's disposition was
narrowly "remove the lying backup checkbox"; these three are different
mechanisms in the same screen and (a) requires a change to `ImportStatus`'s
shape, not just a UI edit. Related but distinct: TASK-279 (wrap chatbook import
in a single transaction) — the import still has no rollback, which is part of
why an over-claiming success banner is expensive here.
