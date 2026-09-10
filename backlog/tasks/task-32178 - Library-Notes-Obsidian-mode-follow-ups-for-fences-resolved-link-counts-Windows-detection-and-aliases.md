---
id: TASK-32178
title: >-
  Library Notes: Obsidian mode follow-ups for fences, resolved-link counts,
  Windows detection, and aliases
status: Done
assignee: []
created_date: '2026-09-09 09:16'
updated_date: '2026-09-09 17:25'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - obsidian
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32129. Four follow-ups from the Obsidian-mode
review: an unterminated ``` fence lets the fence scanner keep treating the
rest of the file as code, so a `[[link]]` that follows it gets rewritten
when it should not; the import receipt has no "N links resolved" line even
though the durable-ledger schema already has room for it; there is no vault
detection through the Windows discovery adapter; and aliases are stored as
ordinary keywords, indistinguishable from real tags.

Update 2026-09-09 (PR #2549 review, integration): AC #1 is done — an
unterminated ``` or ~~~ opener now runs to end of file, so nothing after it
is recorded or rewritten as a link (`_CODE_SPAN` in
`note_import_plan_models.py`, pinned by
`test_an_unclosed_fence_keeps_the_rest_of_the_note_as_code` and
`test_a_closed_fence_still_ends_at_its_closer`). AC #3's "not supported on
Windows" half is now stated in the guide's "Obsidian vaults" section, so the
rider carries only the implementation choice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fence scanner handles an unterminated ``` fence without
  rewriting wikilinks past it
- [x] #2 The import receipt reports a "N links resolved" count
- [x] #3 Obsidian vault detection works through the Windows discovery
  adapter, or the review records an explicit "not supported on Windows"
  decision
- [x] #4 Aliases are distinguishable from tags at the storage or display
  layer, or the guide records the decision to keep them merged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC#2: share the wikilink key derivation in note_import_plan_models (one source for the executor's id map and a resolved-link count), project the count at settle time beside latest_skipped_items and print 'N links resolved' on the receipt line. No durable-ledger column: the receipt view is session state, following the latest_skipped_items precedent.
2. AC#3: implement vault detection and the vault-root skips in the Windows adapter, unit-tested through the injected filesystem seam; drop the guide's 'Not on Windows' caveat.
3. AC#4: store aliases as 'alias: <name>' keywords so they are distinguishable from tags; guide says so.
4. Docs stamp, live verify, backlog notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1 landed in the PR #2549 review round; the other three here.

AC#2 — "N links resolved" on the receipt. The link-key grammar moved into
note_import_plan_models as `creatable_wikilink_keys`, so the executor's
id map and the new `resolved_wikilink_count` cannot drift apart; the executor's
_wikilink_note_ids is now that map plus deterministic ids. The count is captured
at settle time into `latest_resolved_links` and projected as `resolved_links`,
exactly the way task-32130's `latest_skipped_items` captures the receipt's
skipped rows. Deviation from the brief, deliberately: no durable-ledger column.
The receipt view is session state (revisit_latest_receipt reads the in-memory
snapshot), the count is a pure function of the approved plan, and the private
receipt schema is census-validated against frozen canonical statements — a v3
migration to carry one line of copy would be the expensive way to get the same
sentence. The line only appears on a COMPLETED receipt, where failed == 0, so
it can never over-report links whose target note was not written.

AC#3 — implemented rather than documented as unavailable. The Windows adapter
now carries the same three state fields and reuses the shared
`_obsidian_skip_reason` / `_add_skip` / `OBSIDIAN_MARKER_DIRECTORY`, and the
dispatcher passes obsidian_mode through to it. Three tests drive it through the
injected FakeWindowsFilesystem seam, so they run on this host: detection plus
the three skips, toggle-off parity, and a plain folder that is not a vault. The
guide's "Not on Windows" paragraph is gone.

AC#4 — an `aliases:` entry is stored as `alias: <name>`, so Info shows which
keywords are alternate names and which are the author's tags, while the name
still matches a keyword search. An alias the prefix would push past the
keyword ceiling is stored un-prefixed rather than dropped — the display
prefix must never be the reason a name is lost (PR #2556 review). Vault
detection casefolds the `.obsidian` marker on both walkers, matching the skip
map that already did (same review).

Files: note_import_plan_models.py, note_import_executor.py,
note_import_parsers.py, note_import_windows_fs.py, note_import_discovery.py,
library_note_import_state.py, four test files, Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
