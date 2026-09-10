---
id: TASK-32057
title: >-
  Library Collections row: undocumented captures browser, legacy_read_only
  service, and rail side effects
status: Done
assignee: []
created_date: '2026-09-08 18:23'
updated_date: '2026-09-08 19:50'
labels:
  - library
  - collections
  - docs
  - ux
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 'Collections' row opens a 'Quick Capture' captures browser ('Sort: saved desc', 'Filter captures', '0–0 of 0') while collections.md still describes create/rename/delete records; LocalLibraryCollectionsService now raises LegacyCollectionsReadOnlyError on every write; the row shows no count until visited, then injects six sub-rows and collapses the Create section, and that collapse persists into the next launch. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 8.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A recorded decision states what the Collections row is today and the user guide matches it
- [x] #2 The row shows a count before it is visited
- [x] #3 Selecting the row never changes another rail section's disclosure state
- [x] #4 If writes are refused, the canvas says so with the recovery path the service names
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three of the four ACs are done in code and docs; AC#1 (what the Collections row IS) is a product decision and is deliberately left unticked -- both options are costed below.

## What shipped

**AC#2 -- count before visit.** The capture total rode `LibraryCollectionsCaptureController.state.exact_total`, which only exists after the canvas loads a page, so the rail painted a bare "Collections" beside "Media (11)". It is now read at the top of `_list_local_source_snapshot` -- it is one more local source count -- through the scope service's own `list_page`, and surfaced by `_build_library_shell_input` whenever the reader controller has no total of its own. Live: `Collections (0)` at first paint on both the seeded and the empty profile, at 235x52 and 100x30.

Two deliberate constraints, both learned the expensive way:
- It does NOT go through the reader controller. The first cut did, which pre-selected a capture and armed an applied scope for the Continue receipt -- a count read deciding part of a visit the user had not made.
- It does NOT schedule its own reconcile. The first cut ran as its own worker and bumped `_library_snapshot_state_generation`, adding 4 reds to `Tests/UI/test_library_entry_compose_once.py`. Folding it into the snapshot pass keeps the once-per-snapshot invariant.
- It is kept OUT of that method's shared gather: the gather's deadline and all-or-nothing failure branches would drop the Collections count whenever an unrelated source seam failed -- exactly the "counts disagree" complaint.

**AC#3 -- no rail side effects.** Not reproducible, and now pinned. At 235x52, selecting Collections mounts the six scope sub-rows and leaves the Create section open; no `[library.rail_state] sections` key is ever written (the only writer, `_set_library_rail_section`, is reachable only from the section toggle button). At 100x30 the rail is hidden entirely, which is the likely origin of the "Create collapsed" observation. `test_selecting_collections_leaves_every_other_section_disclosure_alone` pins it.

**AC#4 -- refused writes name their recovery.** There is NO user-facing write path to `LocalLibraryCollectionsService` -- `create/rename/delete/restore/add_item` have zero callers in the product, so a "refused write" cannot be triggered from the UI. Instead the reason and the recovery path are surfaced where a user actually meets legacy Collections: the legacy-recovery disclosure, which now leads with "Legacy Collections are read-only on this profile . Use the legacy Collections inspector or JSON recovery export." The second half is `LegacyCollectionsReadOnlyError.recovery` verbatim, so the copy cannot drift from the error. The reason half is scoped to LEGACY on purpose -- the captures browser around it accepts writes (Quick Capture), so an unqualified "Collections are read-only" would be false on that canvas. The two duplicated legacy-button blocks became one `_compose_legacy_recovery_disclosure` helper.

**Docs.** `Docs/User_Guide/library/collections.md` rewritten end to end: the old page described a create/rename/delete manager that has not shipped for some time.

## AC#1 -- the open decision

Option A: **retire "Collections", rename the row "Captures".** Cost: rail title + `short_title` in `library_shell_state.py`, the canvas heading, the guide page name and its inbound links (library.md, index), and a redirect note for anyone searching "Collections". No data migration. Buys honesty immediately -- the row would say what it opens. Costs the word "Collections" as a future feature name, and orphans the ~15 legacy DB rows' vocabulary (they would live under a "Captures" screen).

Option B: **restore Collections on the new storage** -- real named containers with membership, built over `Library_Collections_DB` rather than the superseded generic tables. Cost: a schema migration, a membership model, an "Add to collection" affordance on Media/Notes/Prompts/Conversations (nothing anywhere in the app has one today), a second browse mode inside this canvas or a seventh Browse row, plus the guide. Multi-PR. Buys the grouping feature users of the old page expected, and gives the legacy rows a real destination.

Both are viable; the choice is about whether grouping is a product commitment. Until it is made, the guide documents what ships and says the name may change.

## Files

- `tldw_chatbook/UI/Screens/library_screen.py` -- `_read_library_collections_count`, its call at the top of `_list_local_source_snapshot`, the `_library_collections_prefetched_total` field, the `collections_count` fallback in `_build_library_shell_input`, `CapturePageRequest` import.
- `tldw_chatbook/Widgets/Library/library_collections_capture_reader.py` -- `LEGACY_COLLECTIONS_READ_ONLY_NOTICE`, `_compose_legacy_recovery_disclosure` replacing two duplicated blocks.
- `Tests/UI/test_library_crit8_collections_row.py` (new, 3 tests).
- `Tests/UI/test_library_collections_characterization.py` -- its one-shot `list_page` failure fixture now arms after the rail count settles, since the row press is no longer the first call.
- `Docs/User_Guide/library/collections.md` -- rewritten; `Docs/User_Guide/library.md` -- child-page description.
- `Docs/security/production-diagnostic-inventory.json` -- +1 static debug call.
<!-- SECTION:NOTES:END -->

## Critique #9 evidence (2026-09-10)

Fresh eyes hit the same wall: rail row 'Collections', canvas header 'Quick Capture', empty state 'captures' with a filter blamed that is not set (captures 20/21); the six injected sub-rows are centre-aligned unlike every other rail row. The product decision in AC#1 is now the blocker for three critique findings (register rows 15 and 21 of critique #9).
