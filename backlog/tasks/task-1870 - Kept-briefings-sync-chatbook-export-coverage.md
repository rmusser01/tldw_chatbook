---
id: TASK-1870
title: 'Kept briefings: sync/chatbook-export coverage'
status: Done
assignee: []
created_date: '2026-08-02 00:16'
updated_date: '2026-08-02 13:34'
labels:
  - watchlists
  - briefings
  - chachanotes
  - persistence
  - sync
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up filed at task-1780's close-out, per that spec's "Non-goals (v1)" section
(`Docs/superpowers/specs/2026-08-01-kept-briefings-design.md`).

Task-1780 added `kept_briefings`/`kept_scripts` to ChaChaNotes (schema v29,
`tldw_chatbook/DB/migrations/chachanotes_v28_to_v29_kept_briefings.sql`) so that generated
briefings/scripts a user chooses to keep survive watchlist deletion and Subscriptions_DB
pruning. This was a deliberate, recorded v1 decision (spec, "Schema" section): the two tables
carry **no sync columns** (`client_id`/`version`/`deleted`) — they do not participate in
ChaChaNotes's existing bidirectional sync machinery, and deletion is a hard `DELETE`, not the
soft-delete-flag convention every synced entity in this DB uses.

The same gap exists on the chatbook-export side: chatbook export already knows how to walk
conversations, notes, characters, and other ChaChaNotes entities into a portable bundle, but has
no awareness of `kept_briefings`/`kept_scripts` at all — a user's kept briefings and cast scripts
are silently absent from any chatbook they export today.

Whether closing this gap means adding sync columns and wiring the tables into the existing sync
engine, adding a dedicated (non-sync) export path in the chatbook exporter, both, or neither with
a recorded rationale for staying out (e.g. "kept content is meant to be local-only for now") is
an open design question — not decided here. What matters is that the gap stops being silent:
either these artifacts become reachable through at least one of the two systems users already
rely on for taking their data with them, or the decision to leave them out is written down
somewhere a future reader will find it before assuming coverage that doesn't exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user's kept briefings and their kept scripts are included when the user exports a chatbook, OR a recorded decision explains why they are deliberately excluded — **arm taken: included.** New `ContentType.KEPT_BRIEFING`; `ChatbookCreator._collect_kept_briefings` walks selected kept briefings, nests each briefing's kept scripts inside its own JSON payload (scripts are not independently selectable), and also writes a companion human-readable Markdown file per briefing.
- [x] #2 A user's kept briefings and their kept scripts participate in ChaChaNotes sync between devices, OR a recorded decision explains why they are deliberately excluded — **arm taken: excluded, deliberately.** Extends task-1780's original "no sync columns" v1 ruling; recorded in the follow-up decision block appended to the design spec (see AC #3).
- [x] #3 Whatever is decided for #1 and #2 is written down (spec, ADR, or equivalent) so the next reader does not have to reverse-engineer it from the absence of code — dated "Follow-up decision (2026-08-02, task-1870)" block appended to `Docs/superpowers/specs/2026-08-01-kept-briefings-design.md` (history not rewritten); `Chatbooks/CHATBOOKS_GUIDE.md` updated with a "Kept Briefings" section (it already enumerated content types).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC #1 first arm: add kept briefings/scripts as a chatbook content type — creator walks `kept_briefings`+`kept_scripts` into the bundle (readable markdown + structured payload, following the existing per-type conventions in `Chatbooks/chatbook_creator.py`/`chatbook_models.py`); selection surfaced wherever other types are chosen.
2. Import: ride the house conflict machinery (`conflict_resolver.py`) if it fits kept rows' UNIQUE `source_briefing_id` (device-local id → cross-device collision is DIFFERENT content); otherwise import-when-free + honest per-item skip in the import summary. Re-import idempotent either way.
3. AC #2 second arm: record sync exclusion as deliberate (extends the owner's 1780 "no sync columns" v1 ruling) — spec delivery-notes update + decision note per AC #3.
4. Round-trip test (export → import into a fresh ChaChaNotes), collision test, and the recorded-decision docs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Export (AC #1, included).** `tldw_chatbook/Chatbooks/chatbook_models.py`: new
`ContentType.KEPT_BRIEFING`, `ChatbookContent.kept_briefings`, and
`ChatbookManifest.total_kept_briefings` (defaults to 0 in `from_dict` for bundles that predate
this type — backward compat). `tldw_chatbook/Chatbooks/chatbook_creator.py`:
`_collect_kept_briefings` walks selected kept briefings via the existing
`get_kept_briefing`/`list_kept_scripts` CRUD, nests each briefing's kept scripts inside its own
JSON payload (scripts are NOT independently selectable — mirrors how a conversation's messages
live inside the conversation's own JSON rather than as a separate content type), and writes a
companion `.md` (human-readable, never read back on import) alongside the `.json`
(machine-round-trippable), matching the JSON+report split conversations already use. All
provenance columns (source ids, coverage window, model/preset identifiers, origin,
original/kept timestamps) are carried through, so the export is self-interpreting per the 1780
spec's ethos. README/manifest stats updated. Selection UI wired in both live surfaces: the
`Chatbooks_Window_Improved` → `ChatbookCreationWizard` → `SmartContentTree` path (checkbox +
category node + `_load_content` entries) AND the standalone `ChatbookCreationWindow` (reachable
from Tools & Settings) — a full checkbox in both, not just the service/creator seam.

**Import (AC #1, the "how").** `ChatbookImporter._import_kept_briefings` +
`_import_kept_scripts`. Did NOT extend `conflict_resolver.py`'s ask/skip/rename/replace
machinery — that machinery is built for display-name-keyed conversations/notes/characters, and
doesn't have a sensible "rename" or "ask per item" meaning for a UNIQUE-source-id-keyed
idempotent artifact. Instead: try the insert, catch `ConflictError` (the same "raced keep"
pattern the keep service itself already uses per the 1780 delivery notes), then compare content
— byte-identical is an ordinary silent skip (already present; re-importing the same chatbook
must never duplicate it), differing content is reported as a named conflict in the import
summary and the existing local row is never overwritten (there is no update code path at all,
so "never overwrite" holds by construction; the code's actual contribution is telling the two
cases apart honestly). Added `CharactersRAGDB.get_kept_script_by_source` (mirrors
`get_kept_briefing_by_source`) to classify a script-level `ConflictError` the same way — needed
because `kept_scripts.source_script_id` is a table-wide UNIQUE column, so a script kept under a
*different* local briefing can collide with an incoming one. NULL-source scripts (cast directly
from a kept briefing) have no identity to key a `ConflictError` off of, so they're deduped by
content match against the parent's pre-existing scripts one-for-one (a matched candidate is
popped from the candidate pool so it can satisfy at most one incoming script — this preserves a
source export that legitimately contains two distinct byte-identical NULL-source scripts as two
rows, while still making re-import of the same chatbook idempotent).

**Sync (AC #2, excluded).** No sync columns were ever on the table for these two tables (1780's
original ruling); this task closes the follow-up by confirming that call still holds rather than
silently letting it drift, and writes down *why* it holds (schema-shape mismatch with the sync
engine's row-lineage model, not mere inertia).

**Docs (AC #3).** Dated decision block appended to
`Docs/superpowers/specs/2026-08-01-kept-briefings-design.md` (no existing history rewritten);
`Chatbooks/CHATBOOKS_GUIDE.md` structure diagram + a new "Kept Briefings" section.

**Tests.** `Tests/Chatbooks/test_chatbook_kept_briefings_round_trip.py` (new): export
JSON+Markdown+manifest entry, empty-selection export produces no kept section, byte-faithful
round trip of every provenance column, re-import idempotency (briefing + both a
subscriptions-sourced and a NULL-source script), briefing-level collision (differing content,
same `source_briefing_id`, existing row untouched + named warning), script-level collision
(table-wide `source_script_id` collision across different local parents), and backward
compatibility (importing a bundle with no `kept_briefings` section at all does not crash).
`Tests/DB/test_chachanotes_kept_briefings.py`: two new tests for
`get_kept_script_by_source`. `Tests/Chatbooks/test_chatbook_models.py`: `ContentType` value,
manifest statistics round-trip, and the backward-compat default. Mutation-tested three
behavioral guards by disabling each in turn (Edit, confirm RED, Edit-revert, confirm the diff
returned to exactly the pre-mutation file): the NULL-source script dedup guard (idempotency
test went RED), the briefing-level conflict-vs-identical classification (collision test's
warning assertion went RED), and scripts riding with their parent (4 of 7 round-trip tests went
RED when scripts were excluded from the export payload). Full `Tests/Chatbooks/` +
`Tests/DB/test_chachanotes_kept_briefings.py` + every UI test file touching the changed modules:
252 passed, 1 pre-existing skip (needs `--run-slow`). One unrelated pre-existing failure noted
during a broader `Tests/DB/` sweep (`test_chat_image_db_compatibility.py::test_image_data_integrity`,
`ModuleNotFoundError: No module named 'numpy'` — a missing optional dependency in this venv, not
touched by this task).

**Files modified:** `tldw_chatbook/Chatbooks/chatbook_models.py`,
`tldw_chatbook/Chatbooks/chatbook_creator.py`, `tldw_chatbook/Chatbooks/chatbook_importer.py`,
`tldw_chatbook/DB/ChaChaNotes_DB.py`, `tldw_chatbook/Chatbooks/CHATBOOKS_GUIDE.md`,
`tldw_chatbook/UI/ChatbookCreationWindow.py`, `tldw_chatbook/UI/Widgets/SmartContentTree.py`,
`tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py`,
`tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`,
`Docs/superpowers/specs/2026-08-01-kept-briefings-design.md`. **Files added:**
`Tests/Chatbooks/test_chatbook_kept_briefings_round_trip.py`. **Tests modified:**
`Tests/DB/test_chachanotes_kept_briefings.py`, `Tests/Chatbooks/test_chatbook_models.py`.

**Fix wave (2026-08-02, whole-branch review).** A whole-branch review (`.superpowers/sdd/briefings-residuals/task-1870-verdict.md`) found one high-severity defect plus four low/medium findings against the implementation above; all five fixed here.

F1 (HIGH, proven empirically): on a genuine cross-device conflict, `_import_kept_briefings` still called `_import_kept_scripts` unconditionally against `target_kept_id` -- the *unrelated local* briefing sharing the same `source_briefing_id` -- grafting the incoming bundle's scripts onto it while the warning claimed nothing was modified. Fixed: the conflict branch now skips `_import_kept_scripts` entirely (refuses the whole incoming item as a unit, parent AND children) and the warning names both; the byte-identical (non-conflict) branch is unchanged -- scripts still import additively there. Regression test extended with a local script under the conflicting briefing, asserted byte-unchanged; mutation-verified (removing the guard reproduces the reviewer's exact probe: 2 foreign scripts land on the local row).

F2: `ChatbookImportWizard._run_validation`'s statistics-mismatch check summed per-type manifest totals but omitted `total_kept_briefings` (this task's own type) and `total_prompts` (pre-existing) -- a kept-only chatbook always showed a false "Statistics mismatch" warning. Fixed by extracting the sum into `PreviewValidationStep._expected_content_total` (now including both terms), testable without mounting the widget; new `Tests/Chatbooks/test_chatbook_import_wizard_validation.py`, mutation-verified.

F3: `ChatbookCreationWizard`/`ChatbookCreationWindow` computed each kept briefing's script-count subtitle via `len(list_kept_scripts(kept_id, limit=1000))` per row on the UI thread -- up to 200 extra queries materializing full `turns_json`/`roster_snapshot_json` transcripts just to discard them. Added `CharactersRAGDB.kept_script_counts(ids) -> dict[int,int]`, one grouped `COUNT(*)` query, and switched both call sites to it. New CRUD tests in `Tests/DB/test_chachanotes_kept_briefings.py`.

F4: `kept_at` was exported but silently re-stamped with `CURRENT_TIMESTAMP` on import (no param existed to carry it), while the spec/guide/test docstring implied a full provenance round trip; the original round-trip test's `kept_at` assertion could false-pass because both writes landed in the same second. Fixed faithfully rather than just documenting the gap: `create_kept_briefing`/`create_kept_script` both gained an optional `kept_at` param (explicit column added to the INSERT only when provided; no schema change, the column already has `DEFAULT CURRENT_TIMESTAMP`), and the importer now passes the bundle's value through for both the briefing and every script. New test seeds `kept_at` years in the past on both rows so second-resolution coincidence cannot false-pass; mutation-verified.

F5: the briefing success/skip counters incremented only after `_import_kept_scripts` returned, so a script-level exception (e.g. a malformed non-list `scripts` payload) was caught by the outer per-item handler and counted the whole item -- including the already-durably-inserted briefing row -- as failed. Fixed by counting the briefing's outcome before touching its scripts and wrapping `_import_kept_scripts` in its own try/except that adds a warning naming the script failure instead of falsifying the briefing's own outcome. New test reproduces the reviewer's exact scenario (`scripts: "not-a-list"`) and asserts `successful_items == 1`, `failed_items == 0`, and the warning text; mutation-verified.

Verification: full `Tests/Chatbooks/` + `Tests/DB/test_chachanotes_kept_briefings.py` + `Tests/Watchlists/test_kept_briefings_modal.py` + the two other kept-briefings call sites (`Tests/Subscriptions/test_briefing_keep.py`, `test_briefing_cast.py`) -- 239 passed + 85 passed, 1 pre-existing `--run-slow` skip, 0 failed. All five fixes mutation-tested by reverting each guard/param in turn (Edit, confirm RED, Edit-revert, confirm `md5` byte-identical to the pre-mutation file).

**Files modified (fix wave):** `tldw_chatbook/Chatbooks/chatbook_importer.py`, `tldw_chatbook/DB/ChaChaNotes_DB.py`, `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`, `tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py`, `tldw_chatbook/UI/ChatbookCreationWindow.py`. **Tests modified:** `Tests/Chatbooks/test_chatbook_kept_briefings_round_trip.py`, `Tests/DB/test_chachanotes_kept_briefings.py`. **Tests added:** `Tests/Chatbooks/test_chatbook_import_wizard_validation.py`.
<!-- SECTION:NOTES:END -->
