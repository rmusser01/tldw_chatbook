---
id: TASK-32604
title: >-
  Library Notes: a Both-ways sync root reports up to date while a Chatbook edit
  has never reached the file
status: Done
assignee:
  - '@claude'
created_date: '2026-09-15 06:36'
updated_date: '2026-09-15 15:45'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P0, persona Alex / solo-operator, Keep-a-folder-synced workflow. Heuristic 1 setter (2 -> 1).

What happened. Root PowerVault activated, direction Both ways. Edited the bound note 2026-09-07 in Chatbook; the editor said 'Saved 23:08' (A cap 51). md5 of vault/Daily/2026-09-07.md was 641d3a99... before the edit, 6 s after, 26 s after, and after an explicit Check changes. The root row read '✓ Up to date · Next: Check changes' throughout (A caps 52, 53); the manual check then set the status line to 'Manual check finished. Review exact effects.' with nothing to review and no Review control; no receipt was written. No pending state exists anywhere on the surface.

Attribution: INDEPENDENT of wave 4, PROVEN. A headless probe built on the production runtime/executor/controller stack (the shape Tests/UI/test_library_notes_files_sync_journey.py::_start_real_conflict_stack uses) was run twice -- against 77eb2601a6 and against a git archive extraction of 5fd502dbac, the critique-#3 tip. Both runs are identical: after a note-side edit and 3 s the row stays up_to_date/Check changes and the file is unchanged; sync_now sets phase=review and 'Manual check finished. Review exact effects.' while leaving the row untouched; request_sync_now returns [('update_file', 'note_changed')]; apply_reviewed on that plan writes the file correctly. Wave 4's own pre-fix probe recorded the same asymmetry in task-32534's description. Critique #3 missed it because its walk also edited the disk, which woke the watcher and carried the Chatbook edit across as a side effect.

Cause, PROVEN, three links. (1) Nothing on the note side ever triggers sync: _ProductionRuntimeAdapter.changed_root_ids (Notes/notes_sync_runtime.py:1063-1078) signs only filesystem metadata (display_path, device, inode, size, modified_ns, changed_ns) via _discovery_signature (:1048-1061), and PollingNotesSyncWatcher is the only producer of hints (:1919, :2979 -- no other schedule_hint caller exists). (2) The root row is projected from stored root state, never from a fresh plan. (3) The review the manual Check builds has no door: LibraryNotesSyncController.sync_now (UI/Library_Modules/library_notes_sync_controller.py:1378-1419) installs it and sets phase='review', but handle_library_notes_lasting_root_action (UI/Library_Modules/library_notes_controller.py:5152-5153) leaves the view on lasting_roots for the 'check' action, and only library_notes_add_from_files_canvas.py:482 renders phase=='review'; the roots canvas offers a Review button only when next_action == 'review_changes' (library_notes_sync_roots_canvas.py:174-176), which only the automatic pass can set.

Docs contradicted: notes.md promises Receipts shows 'Wrote note to file' for a note you edited in Chatbook. No user-reachable path produces that receipt today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a check, a root holding note-side changes that are not on disk never renders '✓ Up to date': the row renders an explicit pending state with the next action ('◌ Changes available · Next: Review changes') and the status line beside it names the count. SHIPPED SCOPE, narrowed in fix round 1: 'never' holds after a check, and for an editor save of an already-bound note while lasting sync is running (written within seconds, so the label is honest by the time it is read). A runtime that has stopped watching writes nothing, so the row wears '⚠ Sync stopped' rather than '✓ Up to date' (fix round 2). It does NOT hold for the note-write paths that still produce no signal -- see the ceiling list in the Implementation Notes; those keep showing '✓ Up to date' over a stale file until a check or a disk-side change, unchanged by this task and pre-existing
- [x] #2 A Chatbook-side save inside an active root either writes to disk on the same terms a disk-side edit is picked up, or the editor's own status line says the note is saved in Notes and not yet written to the named file. SHIPPED SCOPE, narrowed in fix round 2: the FIRST limb is what ships, for an editor save of an already-bound note while lasting sync is running. The second limb is not implemented -- when the runtime has stopped watching, the save succeeds, the file stays stale and the editor says nothing. The ROW no longer lies about that ('⚠ Sync stopped', fix round 2), but surfacing it in the editor needs a new field carried through DatabaseNotePortSaveReply -> NoteSaveOutcome -> the session snapshot, which is follow-up work, not a trailing minor. ANCHORED: that work is **task-32633** (filed in PR #2692, branch `docs/library-notes-crit4-riders`) -- it should carry the editor status line as its own AC#1. Its title says thirteen note-write paths; fix round 3 re-derived the enumeration from scratch and the number is **17 uncovered** (18 live seams, 1 covered), so the title wants retitling -- the coordinator owns that branch, not this one
- [x] #3 Check changes from Manage sync folders reaches the review it builds: when the plan has actions the user lands on them, and when it has none the status line says 'Nothing to review', not 'Review exact effects'
- [x] #4 Every note-to-file write leaves the Receipts row the guide already promises, and notes.md's Receipts paragraph matches what ships
- [x] #5 Regression test on the production runtime: note edit -> check -> the plan's update_file is reachable and applied; after a check, a row is never projected as up to date while an update_file action is pending. SHIPPED SCOPE, narrowed in fix round 1 and round 2: the 'never' is scoped to after a check, and the editor-save limb to while lasting sync is running, for the same reasons as AC#1
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run crit4 probe_p0.py on the branch tip to reproduce (done: row up_to_date, disk unwritten, plan holds update_file/note_changed).
2. Link 1 - note-side change signal: add NotesSyncRuntimeOwner.note_changed(note_id) that maps the note to its admitted roots (store.active_binding_note_ids) and calls the EXISTING schedule_hint - one hint mechanism, the watcher's. Route _LibraryDatabaseNoteSessionPort.save_note's success into it; library_screen injects the app-owned runtime accessor.
3. Link 2 - the row: LibraryNotesSyncController.sync_now never re-projects after the check, so the row keeps a stale up_to_date even though the runtime already published changes_available/review_changes. Add refresh_roots(publish=False) so the runtime stays the single publisher of row state.
4. Link 3 - the door: with next_action == review_changes the roots canvas already offers Review; additionally land the user on the review when the check found actions, and say 'Nothing to review.' when it did not.
5. Tests: RED->GREEN on the production runtime/executor/controller stack (Tests/Notes for the runtime hint, Tests/UI for the controller + handler route).
6. Extend probe_p0.py, live walk on a scratch profile with md5 before/after, guide + Verified against stamp, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three links, fixed at three seams; the machinery underneath was already correct.

**Link 1 - nothing on the note side ever signalled a change.** `changed_root_ids` signs filesystem metadata only, so `PollingNotesSyncWatcher` was the only hint producer and a Chatbook save left the root looking unchanged. Added `NotesSyncRuntimeOwner.note_changed(note_id)`: it maps the note to its admitted roots through the existing `store.active_binding_note_ids` and calls the EXISTING `schedule_hint` - the watcher's own mechanism, one consumer, one set of gates (admission, lease, blocked-root). The shipped editor save seam, `_LibraryDatabaseNoteSessionPort.save_note`, routes into it on a SAVED reply; `library_screen` injects the app-owned runtime accessor. The hint is best-effort and never fails a save (one metadata-only warning). No loop: the executor writes notes through `NotesScopeSyncAuthority`, not through the port.

**Link 2 - the row was never re-projected.** The runtime was ALREADY publishing `changes_available`/`review_changes` during a manual check (`_reconcile_locked`, the `elif selected` branch); `LibraryNotesSyncController.sync_now` simply never called `refresh_roots` afterwards, so the row kept whatever it said before the check. One `refresh_roots(publish=False)`. The runtime stays the single publisher of row state - the controller only re-reads it, which is why the status line's branch is now chosen FROM the row's `next_action` rather than from a second opinion about the plan.

**Link 3 - the review had no door.** The roots canvas offers Review only for `next_action == "review_changes"`, which link 2 now produces; additionally the `check` branch of `handle_library_notes_lasting_root_action` lands the user on the review when the check found effects (guarded on still being mounted and still on `lasting_roots`), and the status line says "Nothing to review." when it did not.

Status-line copy: "Manual check finished. N change(s) to review." (count from the plan, filtered by `NOTES_SYNC_MANUAL_APPLY_ACTION_KINDS` so a `no_change` action is not counted) / "Nothing to review." / for offline/unsupported/paused the existing `_restate_root` restates the row.

**Trade-off / ceiling. THE COUNT HAS MOVED THREE TIMES (12+1 -> 14/1/13 -> 18/1/17); the method below is the durable part, the number is not. Re-derive, do not copy.**

*Method (2026-09-15, re-derived from scratch in fix round 3).* An AST sweep of `tldw_chatbook/` in four passes, because each earlier undercount came from a grep that assumed a call shape:
1. direct calls `x.<verb>(` for verb in {add_note, save_note, update_note, replace_note, create_note, delete_note, restore_note, **save_note_with_organization**, create_note_with_metadata};
2. **bare attribute references** `x.<verb>` never called on the spot -- how a bound method reaches `asyncio.to_thread` (this pass, and only this pass, finds `MCP/server.py:717` and `review_selection.py:476`);
3. `getattr(x, "<verb>")` string lookups;
4. raw `INSERT INTO notes`.
That yields 151 candidates, then each is classified by hand. Two traps the candidate list is full of: **`BaseException.add_note()` (PEP 678)** accounts for dozens of hits in TTS/, Model_Artifacts/, Personal_Context/, app.py and DB/private_sqlite* and is not a note at all; and every `INSERT INTO notes...` outside `ChaChaNotes_DB` targets `notes_sync_*` / `notes_object_mirror` / `notes_organization_*`, not the `notes` table.

*Counting unit.* A **seam** is the one place a `note_changed(note_id)` call would go to cover that user action without covering unrelated ones. Where a service method serves ONLY uncovered callers it is the seam (`delete_note`, `restore_note`); where it is shared with the covered path it is not, and each caller is its own seam -- which is why `notes_scope_service.save_note` is not listed but its callers are.

**Result: 18 live seams -- 1 covered, 17 not (6 updates + 11 creates) -- plus 2 dead.**

*Covered (1):* `UI/Library_Modules/note_session_port.py:140` -- the Library editor's save of an already-bound note.

*Uncovered updates -- one `note_changed` call away each (6):*
- `Research_Workspace/local_adapter.py:389` (quick-note save of an existing note)
- `Notes/note_import_executor.py:302` `replace_note` (Import notes from files, over an existing note)
- `Notes/notes_scope_service.py:1638` `delete_note` / `:1672` `restore_note`
- `Tools/note_management_tools.py:377` -- the built-in `update_note` LLM tool
- `Notes/Notes_Library.py:758` `save_note_with_organization` -- the `library_save_note` agent tool (`Agents/library_tool_provider.py:227`, wired from `Chat/console_runtime.py:637`). Writes the `notes` table directly: `db._update_note_with_cursor` at `:1084` when given a note_id, `db._add_note_with_cursor` at `:1076` when not. One seam, both kinds; listed here because its update branch is the cheap half.

*Uncovered creates -- these need a FOLDER-MEMBERSHIP predicate, not a binding predicate, because `note_changed` keys on `if note_id in bound` (`notes_sync_runtime.py`) and a note that does not exist yet is in no root's bindings (11):*
- `UI/Screens/library_screen.py:30054` -- **the Library's own New note, on the same screen as this P0** (the `getattr` is two lines up at :30039); bypasses the port entirely with `note_id=None`
- `UI/Console_Modules/message.py:2227` (`_save_console_message_as_note`) and `:2294` (`_write_console_note`) -- Console Save-as-Note and the message-span action, both `note_id=None`. An EXISTING Chat-side write, not a future one
- `UI/Console_Modules/review_selection.py:476` -- `_create_console_selection_note`, "save this selection as a note"
- `Research_Workspace/local_adapter.py:419` · `Event_Handlers/note_ingest_events.py:557` · `MCP/local_runtime_delegate.py:602` · `MCP/server.py:717` (the MCP server's own `create_note` tool, a bound-method reference) · `Chatbooks/chatbook_importer.py:2393`
- `Tools/note_management_tools.py:191` -- the built-in `create_note` LLM tool, sibling of the `update_note` one above
- `Notes/note_import_executor.py:275` `create_note` -- the create half of the same import target

*Dead, counted in earlier rounds and dropped (2):* `Chat/document_generator.py:596` sits in `create_note_with_metadata` (def :571), which has **zero callers** in `tldw_chatbook/` or `Tests/`. `UI/MediaWindow_v2.py:2094` calls `self.app_instance.notes_db.create_note(...)`; the app exposes `chachanotes_db`, never `notes_db`, and no class in the tree defines `create_note` on a notes DB -- it raises before it writes.

*Where I differ from the round-2 re-review (17 live: 1/16, 5 updates + 11 creates).* One site: `Notes/Notes_Library.py:758` `save_note_with_organization`. Its verb is not in that review's pass list and it is reached as a bare attribute (`backend.save_note_with_organization`), so neither its call pass nor its bound-method pass could see it. Everything else matches, including both dead calls. **If task-32633's title carries 13, the number to retitle it to is 17 uncovered.**

None of the 17 is a regression -- all pre-existing, all unchanged by this task -- but "New note inside a synced folder never reaches disk until something touches the disk" is the same defect class as this P0, reachable from the same screen. Owned by **task-32633**.

The alternative that would have covered every path at once - signing note versions inside `changed_root_ids` - was rejected: it puts a ChaChaNotes read on the watcher's polling thread once per interval per root, and reaching the notes DB from there goes around `NotesScopeSyncAuthority` into `NotesScopeService.local_notes_service`, past the scope/policy gate the adapter otherwise always uses. `observe_versions` is async, and the watcher callback is deliberately sync and dependency-free.

**Evidence.** Five tests in `Tests/UI/test_library_notes_files_sync_journey.py` on the production runtime/executor/controller stack (and, for link 3, a mounted LibraryScreen over a REAL runtime owner). Substantive REDs before the fix: `assert 'baseline' == 'edited in Chatbook'` (link 1, with only the hint call disabled), `assert 'up_to_date' == 'changes_available'` (link 2), "Check changes never reached the review it built" (link 3), `'Manual check...xact effects.' == 'Nothing to review.'` (AC#3). Live walk on a scratch profile with a 65-file git vault: `md5 People/Sam.md` 1d209615... -> 4cd06840... within 4 s of pressing Save, with nothing touching the disk, and "2026-09-15 08:06 . Wrote note to file . People/Sam.md . Sam" at the top of Receipts. Profile log: zero `unhandled_exception`, zero ERROR.

**Files.** `Notes/notes_sync_runtime.py`, `UI/Library_Modules/note_session_port.py`, `UI/Library_Modules/library_notes_sync_controller.py`, `UI/Library_Modules/library_notes_controller.py`, `UI/Screens/library_screen.py`, `Tests/UI/test_library_notes_files_sync_journey.py`, `Docs/User_Guide/library/notes.md`, `Docs/security/production-diagnostic-inventory.json`.
<!-- SECTION:NOTES:END -->
