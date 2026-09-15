---
id: TASK-32605
title: >-
  Library Notes: lasting sync plans a fresh create for every note Import once
  already made from the same vault
status: Done
assignee:
  - '@claude'
created_date: '2026-09-15 06:37'
updated_date: '2026-09-15 15:39'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D1, P1, persona Alex / researcher, Obsidian workflow. B's headline finding.

What happened. Import once on the power vault created 54 notes (B cap 25, rail count 10 -> 64, sqlite confirms). A second Import once of the same folder correctly reorganises into 'Unchanged repeat (54)', every row defaulted to Skip and annotated 'Existing note: Monday (version 1). Folder placement: no change' (B cap 28). Minutes later, Add from files -> Keep a folder synced on the SAME folder -> Check changes returns '54 safe · 0 need attention · 4 skipped · 0 folder moves' with every row reading 'Create a Library note' (B cap 32). Activating would leave 108 notes -- the whole vault duplicated. The dry run is the contract that screen exists to state, so the declared effect is the finding; B did not activate.

This is a seam between two individually-correct wave-4 fixes: 32541 (#2676) taught Import once repeat detection, 32535 (#2679) gave the sync review Import once's Obsidian pass -- but not its receipt ledger.

Cause, PROVEN at the planner. The two paths do not share a note identity. The sync runtime mints a binding for every discovered file with a synthetic note id, sha256('note\0<root_id>\0<relative_path>') (Notes/notes_sync_runtime.py:951-958), and observes only that id; a note Import once created carries a different id and is invisible, so _plan_unbound sees file_exists and not note_exists and returns CREATE_NOTE / file_discovered (Notes/notes_sync_reconciler.py:437-467). Import once's dedup lives in a different planner, pinned by Tests/Notes/test_note_import_planner.py::test_a_two_row_csv_imported_twice_classifies_as_unchanged_repeat. No test pins the observed sync behaviour. The reconciler already has the vocabulary for the join -- an unbound binding with both sides present raises duplicate_authority (notes_sync_reconciler.py:443-449) -- it is simply never reached because the note is never found.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keeping a folder synced on a folder Import once has already imported recognises the existing notes instead of planning a create for each
- [x] #2 The sync review states what it will do about each recognised file -- name the file, say it was already imported by Import once, and say it is left as it is -- and the guide gives the recourse. AMENDED from "offers the same choice Import once offers on a repeat -- link to the existing note, skip, or create new -- with the existing note named on the row", with evidence: (a) an attention row wedges the root permanently. `_blocked_plan_status` (`notes_sync_runtime.py:2154-2157`) returns `needs_attention` on ANY `plan.attention` at all -- conflict eligibility never enters -- and `activate_root` refuses on it (`:3283-3284`), as does `apply_reviewed` via `_reviewed_plan_has_non_content_blocker` (`:2692`). So 54 such rows would make the root unactivatable for good, which is worse than the reported defect. Re-using an ELIGIBLE reason instead is worse still, not better: `ELIGIBLE_CONFLICT_REASONS` is `{both_sides_changed, out_of_direction_change}` (`notes_sync_conflicts.py:29`), and `resolve_keep_note` maps to `UPDATE_FILE` (`notes_sync_executor.py:82`) -- the choice would offer to overwrite the user's vault file with the frontmatter-stripped imported body. (Review round 1 corrected this clause: the first draft cited `eligible_conflict_reason(reason, managed=True)`, but `managed=True` is passed at NO call site -- every one computes `managed=<binding_id> in <managed placement effect ids>`, and `_apply_blocker` itself passes `managed=False`. The conclusion held; the mechanism named did not.); (b) "link to the existing note" is not a no-op that sync can offer today: Import once strips the YAML frontmatter block from the note body in Obsidian mode (`note_import_parsers.py:505-514`) and rewrites `[[wikilinks]]` into `[[a|b]](note://uuid)`, while lasting sync keeps the block byte-exact because UPDATE_FILE writes the body back to disk (task-32535 controller ruling) -- so adopting an imported note into a bidirectional root REWRITES one side on the first pass, and which side is a data-loss decision; and (c) the executor's create path asserts the note is missing (`_require_note_missing`, `notes_sync_executor.py:3215`), so adoption needs a new journaled action with its own recovery semantics. Linking is a feature, filed as a rider, not part of repairing this seam
- [x] #3 Following the two documented Obsidian paths in the order the critique followed them -- Import once, then Keep a folder synced on the same folder -- cannot produce two notes per file. AMENDED from "Following both documented Obsidian paths on one vault cannot produce two notes per file": the OTHER order is a separate, un-repaired gap. Import once has no sync-awareness at all (`grep -n "notes_sync|lasting_sync" tldw_chatbook/UI/Library_Modules/library_note_import_controller.py` is empty) and its repeat detection reads only its own receipt ledger, so running Import once on a folder that is already kept synced still copies every file into a second note. Closing it needs a sync-root probe on the import planner, a new item classification and its own review copy -- filed as a rider; the guide now states the gap
- [x] #4 An end-to-end test covers import-then-sync on one folder and pins the recognition, on the production planner rather than a fake
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace both halves: Import once's receipt ledger (source_locator_digest -> note id) and the sync planner's synthetic note id; confirm they share a database file and that observe_root already calls the importer's own discover_import_sources on the root path.
2. Decide adopt vs skip vs report with evidence (executor CREATE_NOTE asserts the note is MISSING; a managed root's attention rows are never conflict-eligible, so an attention row would block activation forever).
3. RED pins: reconciler item skip for a recognised file; runtime end-to-end (real adapter + real receipt ledger written by the real import plan over the same folder) showing creates -> skips.
4. Fix: narrow read-only receipts query keyed on the discovered ImportSource; runtime marks never-bound candidates whose imported note still exists; reconciler skips them with reason already_imported; review copy names the effect.
5. GREEN, live walk (Import once the vault, then Keep a folder synced on the same vault, Check changes), note counts before/after, guide + stamps, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A folder Import once has already imported is now **skipped** by lasting sync,
with the reason on the row, instead of planned as a fresh create for every
file. AC#2 and AC#3 are amended to what ships, each with its evidence in the
AC itself.

**The join was already half-built.** `observe_root` walks the sync root with
`discover_import_sources` -- the importer's OWN walk -- so its
`ImportSource` values carry locator fields identical to the ones an Import
once of the same folder recorded, and the receipt ledger lives in the very
same device-state database (`get_notes_sync_state_db_path()` is both the
runtime's `database_path` and the receipt repository's path). Nothing about
the display-path convention had to be re-derived: the digest function moved
one argument down (`_private_source_locator_digest_for_source`), the ledger
read was factored out of `_prior_observations_for_plan` into
`_prior_observations`, and a read-only `prior_imported_notes_read_only`
queries it by source. No schema change; receipts written before this fix
count.

**Recognition tracks the live note, not the receipt.** The matched note ids
go through one bulk `observe_versions`, which omits deleted and missing
notes, so a note the user has since deleted stops counting and sync creates
the file again -- pinned as the test's negative control. The reconciler sees
one new boolean (`BindingObservation.prior_import_note`, part of the
observation token so a stale review cannot be applied) and `_item_skip`
returns `already_imported` for a never-bound file, exactly the per-file skip
machinery task-32535 shipped for `.trash/`, `Templates/` and empty files.

**Why skip and not adopt or report.** Both alternatives are in AC#2 with
their evidence. In short: an attention row on a managed root is never
conflict-eligible, so reporting would make the root permanently
unactivatable; and adopting is not a no-op, because Import once deliberately
transforms the note it creates (frontmatter stripped, wikilinks rewritten)
while sync keeps the file byte-exact -- binding one into a bidirectional root
rewrites one side on the first pass. The executor also asserts the note is
MISSING on its create path. Linking two notes that differ on purpose is a
feature with a data-loss decision inside it, not a seam repair.

Trade-off, stated in the guide: those files are then **not synced**. The
recourse is to delete the imported notes and run Check changes again. The
alternative -- silently rewriting the user's vault or their notes on the
first activation -- is worse.

**Live (235x52, scratch profile `notes-crit/wave5/vault-dup/power`, 65-file
`--git --archive 45` vault).** Notes before: 10. Import once on the vault:
"54 notes created * 11 files skipped", rail "Notes (64)", sqlite 64 -- the
critique's numbers exactly. Keep a folder synced on the SAME folder ->
Check changes: **"0 safe * 0 need attention * 58 skipped * 0 folder moves"**,
every previously-imported file reading "Already imported by Import once --
left as it is" and the 45-file Archive run collapsed to one disclosure row
(the critique read "54 safe ... every row Create a Library note"). Activate
reviewed root: "Sync root activated. 0 applied", **sqlite still 64** -- the
only repeated title in the database is the seeder's deliberate pair of
"Reading list" notes. Log grep: the one `app_stopping` line is my own Ctrl+Q;
no `unhandled_exception`. Captures under `scratchpad/wave5/caps/01..06`.

Tests vs a detached `origin/dev` worktree at 4e4558bff2: `Tests/Notes/`
3634 passed / 5 failed vs 3632 / 5 -- the SAME five names, all pre-existing
(git-commit signing and coordinator process races plus
`test_legacy_sync_config_is_read_only_and_only_the_migrator_reads_it`).
Library + UI pins (`test_library_notes_lasting_sync_state`,
`test_library_notes_wave_import_ux`, `test_library_notes_w3_layout`,
`test_library_crit10_notes_details`, `test_library_notes_w4_sync_review`):
137 / 0 vs 136 / 0. `./scripts/preflight.sh` green.

RED proven twice by removing each half in turn: without the planner's skip,
both new pins fail with the two files back as creates; without the runtime's
ledger join, the end-to-end pin fails the same way.

**Files.** `Notes/notes_sync_runtime.py` (`_prior_import_paths`, the
`prior_imports` seam and its production wiring),
`Notes/notes_sync_reconciler.py` (`prior_import_note`, `_item_skip`, the
observation token), `Notes/note_import_receipts.py`
(`prior_imported_notes_read_only`, `_prior_observations`),
`Notes/note_import_execution_models.py`
(`_private_source_locator_digest_for_source`),
`Library/library_notes_lasting_sync_state.py` (row copy);
pins in `Tests/Notes/test_notes_sync_runtime.py` (end-to-end, real ledger),
`Tests/Notes/test_notes_sync_reconciler.py`,
`Tests/Library/test_library_notes_lasting_sync_state.py`;
`Docs/User_Guide/library/notes.md`.

**Riders for the controller** (no ids minted): (1) sync adopts an imported
note -- the link/create-new choice AC#2 asked for, which needs a journaled
adopt action and a ruling on which side wins when the two bodies differ;
(2) Import once on a folder already kept synced still duplicates it -- the
reverse of this seam, needing a sync-root probe on the import planner.

**Fix round 1 (review).** Approved, 0 Critical. The reviewer verified all three
reasons for SKIP independently and found a STRONGER one than AC#2's first
draft gave -- see the AC, now corrected: the blocker is `_blocked_plan_status`
refusing on any attention at all, not conflict eligibility. One Important and
three Minors closed:

- The recourse this task printed was not executable at the scale it was
  offered for. The guide told a user whose 58 files were skipped to delete the
  imported notes, but Library ▸ Notes select mode yields Done / Select all /
  Clear / Export selected and nothing else
  (`Widgets/Library/library_notes_canvas.py:1636-1683`), with no folder
  cascade -- 54 notes is 54 individual confirmations. The paragraph now names
  that cost and points at the cheaper path; a bulk-delete rider is named for
  the controller.
- AC#2's citation of `eligible_conflict_reason(reason, managed=True)` was
  wrong (`managed=True` is passed nowhere) and `_require_note_missing` is at
  `:3215`, not `:3216`. Both corrected in the AC, which is the record a future
  rider reads.
- The ledger read, the digest helper and the runtime's lookup seam took
  `object` in modules that validate `type(x) is not Y` almost everywhere else,
  and the runtime paid for it with `getattr` defaults. All three now name
  `ImportSource` / `PriorImportObservation` (importable, no cycle) and the
  getattrs are gone. The end-to-end pin's duplicated import line (a new ruff
  I001) is merged.

Two ceilings the review named, both accepted as shipped rather than changed:
the steady-state-zero cost holds for a normally synced root, but NOT for the
root this feature exists for -- those files are never bound, so every check
pays one ledger read plus one `observe_versions` for them forever (bounded by
`max_files=1_000`, chunked); and after activation such a root publishes
`up_to_date` while those files will never sync, the skip count being visible
only inside the review. Both match the `Templates/` and `.trash/` behaviour
task-32535 shipped, so neither is new here.
<!-- SECTION:NOTES:END -->
