---
id: TASK-32633
title: >-
  Library Notes: seventeen note-write paths still give lasting sync no signal
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-15 10:15'
labels:
  - library
  - notes
  - critique-4
  - rider
dependencies: []
priority: medium
updated_date: '2026-10-04 13:20'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider off task-32604 (the critique-#4 P0). 32604 closed the reachable hole for
an editor save of an already-bound note, and made a root whose runtime has
stopped wear "⚠ Sync stopped" instead of "✓ Up to date". It did NOT cover the
other note-write paths. The enumeration in 32604's notes, settled at the end of
that task by **two independent derivations from opposite directions meeting at
the same number** — a four-pass AST sweep verb-inward, and a SQL-outward
reverse-reference index — counts **18 live write seams, 1 covered, 17 not — 6
updates and 11 creates — plus 2 dead**.

The two halves need different mechanisms, which is why 32604 left them here:

- The **6 update paths** write to an already-bound note, so they are one
  `note_changed(note_id)` call away from correct — the seam exists and is
  already proven by 32604's covered path.
- The **11 create paths** mint a note that no binding knows about yet, so
  `note_changed` cannot resolve it to a root. They need a folder-membership
  predicate, not a binding predicate.

Until then those paths leave the file on disk stale while the row reads
"✓ Up to date" on a live runtime, recoverable only by **Check changes** or a
disk-side change. This rider also carries Minor 7 from 32604's re-review
(the per-watchable-root read is a shape worth revisiting, acknowledged not
changed) and the deletion-group count nit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A save whose signal lasting sync refuses is visible to the user at the editor, not silent: the note session surfaces it rather than leaving the reassuring row to speak for the write. (Needs a new field through `DatabaseNotePortSaveReply` → `NoteSaveOutcome` → the session snapshot — the note-session state machine, which is why 32604 did not do it.)
- [ ] #2 Each of the 6 update paths signals lasting sync on success, verified per path against a live runtime with a real file on disk.
- [ ] #3 The 11 create paths are covered by a folder-membership predicate (or an explicit, user-visible decision that they are not), with the mechanism named and pinned.
- [ ] #4 The enumeration in the task notes and `Docs/User_Guide/library/notes.md` is re-derived at landing time, not copied — the count moved THREE times inside task-32604 alone (12+1 → 14/1/13 → 17 → 18/1/17), and was only settled by deriving it twice from opposite directions. The two traps that caused every undercount are recorded in 32604's notes — including one that shipped WRONG and was corrected: `Notes/note_import_executor.py` does write the real `notes` table.
- [ ] #5 Minor 7 adjudicated: either the per-watchable-root read is reshaped or the reason it stays is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
**Slice (TASK-34000 wave 1, Task 4; controller ruling 2026-10-03): fix review
finding N-03 only.** Deleting a synced note left the root row at "✓ Up to
date" while its file was still on disk. This slice lands:

(a) The `delete_note` and `restore_note` seams (two of the six uncovered
update seams in 32604's enumeration; see the Notes for the re-derivation)
signal lasting sync for a bound note, through the same `note_changed` hint
the editor save uses. What the hint does to the file follows the existing
reconciler design, not a new rule: a note-side deletion is a
`DELETION_REVIEW` (`note_missing`) -- the root is held for attention and
the file is NOT removed on its own (no automatic winner). A restore of that
note re-plans the root (the `_run_settled_pass` precedent) and, the note
being back unchanged, the root returns to healthy.
(b) Until every write path signals, a root's healthy status says when it
was last confirmed: "✓ Up to date as of HH:MM" (local clock), never an
unqualified "✓ Up to date".

1. RED: real-stack tests (real `CharactersRAGDB`, real temp vault, the
   production runtime owner) for delete -> held + file intact, and
   restore -> healthy; a projection test for the dated healthy copy.
2. Runtime: `NotesSyncRootRuntimeSnapshot.published_at`; `note_changed`
   releases a planner-only hold on a bound root before hinting.
3. UI: one shared signal helper in `library_notes_sync_attention.py`,
   called from the Library's delete and Undo/Restore seams; the healthy
   label reads the time; Manage sync folders rows follow publications
   through Task 2's listener seam (no second mechanism).
4. Live-verify in the isolated harness at 160x45; save captures under
   `qa/notes-library-ux-review-2026-10-02/fixes/task-32633-slice/`.
5. User Guide (`Docs/User_Guide/library/notes.md`) for the changed copy.

Everything else in this task (AC#1, the other four update seams, AC#3's
eleven create seams, AC#4, AC#5) stays open.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Slice landed (TASK-34000 wave 1, Task 4; review finding N-03, P0).** Status
stays **In Progress**: no AC is fully met -- see "What stays open".

**What this slice covers, exactly.** Re-derived from 32604's list, not
copied: the six uncovered *update* seams are (1) the Research quick-note save
`Research_Workspace/local_adapter.py`, (2) `note_import_executor.replace_note`,
(3) `notes_scope_service.delete_note`, (4) `notes_scope_service.restore_note`,
(5) the `update_note` LLM tool, (6) `Notes_Library.save_note_with_organization`
-- four bullets of one seam each plus one bullet naming two, so six. This
slice covers **(3) and (4), for the Library's own callers**: the visible
**Delete** (`library_screen._delete_library_note_claimed`) and the one
restore seam both the receipt's **Undo** and Recently deleted's **Restore**
commit through (`library_notes_controller._undo_library_note_delete`). The
signal is the same `note_changed` hint the editor save uses (task-32604),
raised through one new helper, `signal_library_note_lasting_sync` in
`UI/Library_Modules/library_notes_sync_attention.py`. It is raised at the UI
seam, not inside the service methods, on purpose: `delete_note` is also what
the runtime's own reviewed deletion calls (`delete_note_for_sync`), and a
hint from inside it would re-enter a pass mid-apply. **Not covered by this
slice, still open under AC#2:** the Research-workspace delete
(`local_adapter.py` `delete_note`), and the Library's two deletes of a new
note (blank-note GC and discard-new -- usually not yet bound, but a new note
in a synced folder is bound as soon as any automatic pass runs before it is
discarded); seams (1), (2), (5), (6).

**What delete does to the file (design decision).** Nothing -- and that is
the existing design, not a new rule. `notes_sync_reconciler._plan_bound`
classifies a bound note that is gone as `DELETION_REVIEW` / `note_missing`
(the same branch as a missing file): no action, attention only. The runtime
publishes `needs_attention` / `review_changes` and holds the root; the file
keeps its bytes; no operation is journaled. Pinned on the production stack
in `Tests/Notes/test_notes_sync_delete_restore_signal.py` (file bytes, store
rows, the plan's attention entry) and live (below). **What the hold means in
this release (review 1, Important 3):** Review changes shows the deletion but
cannot resolve it -- every choice is disabled (`can_apply=False`, blocker
`deletion_review`, "Deletion review is unavailable in this release"),
`apply_reviewed` raises `review_not_executable`, and nothing in the folder
syncs either way while held. TASK-34000.15 owns the review; until it lands
the only resolution is restoring the note. The delete prompt now says so for
a synced note (fix round 1, below) so the hold is never a surprise.

**Restore.** A hold the planner's own classification produced is released
by a change to one of the root's bound notes: `note_changed` now re-plans
such a root (`_release_planner_hold`: status `needs_attention /
review_changes`, no `action_id`, no incomplete operation, not paused;
durable startup holds included since fix round 1) instead of refusing the
hint, the `_run_settled_pass` precedent from TASK-34000.2. The restored
note matches its baseline, so the pass publishes `up_to_date`. Holds that
Recovery owns are never released this way (negative controls in the same
test file). Stale review plans are already handled by observation
tokens, which is why swapping `_reviews[root_id]` is safe.

**Healthy copy.** `NotesSyncRootRuntimeSnapshot.published_at` (epoch, set in
`_publish`) and `root_status_label` in
`Library/library_notes_lasting_sync_state.py` render "✓ Up to date as of
HH:MM" (local, tz-aware then localised). The status copy table moved there
too (`ROOT_STATUS_LABELS`; controller ratchet 2380 -> 2366, a move). The
Manage sync folders rows now follow publications through Task 2's listener
seam (`refresh_manage_sync_folders_rows`, publish-on-change so the
publication's own refresh cannot loop) -- no second mechanism.

**Found and filed, not fixed here: TASK-34000.49.** A restore bumps the
note's version without changing its content; the executor's `update_note`
precondition compares the binding's recorded `note_version`, so the next
file-to-note update is refused as `stale_observation` -- `needs_attention`
with nothing to review, for good. Pre-existing (the same happens with a
restore followed by Check changes), surfaced by this slice's own test, pinned
as a strict `xfail`.

**Evidence.** RED on base `4e955f40c0`: restore `note_changed` returned `()`;
`published_at` AttributeError/TypeError; helper AttributeErrors; the Pilot
timed out waiting for the delete to reach lasting sync. GREEN: 327 passed,
1 xfailed across the touched and guarding files; Pilot
`Tests/UI/test_library_notes_sync_delete_restore.py` 11-12 s locally (added to
the UI PR gate census). Inherited reds untouched by this branch: the
`library_skills_controller.py` and `chat_screen.py` size ratchets.
Live (isolated harness, 160x45, 2026-10-04, worktree task-34000-wave1):
`qa/notes-library-ux-review-2026-10-02/fixes/task-32633-slice/` -- delete:
file 74 bytes unchanged, tombstone v4, root `needs_attention`, 0 operations,
tree "⚠ Needs attention", list "⚠ A sync folder needs attention", Manage
"⚠ Needs attention · Next: Review changes"; Undo: file unchanged, note v5,
Manage "✓ Up to date as of 07:40 · Next: Check changes" at clock 07:40;
a disk edit while sitting on Manage repainted the row to "as of 07:41"
without a keypress -- **with a hand edit first**: the binding's recorded
`note_version` was set to the restored note's version by SQL before that disk
edit. Without it the pass is refused as `stale_observation` and the row reads
"⚠ Needs attention" (TASK-34000.49, pre-existing, below). 0 tracebacks,
isolation clean.

**What stays open.** AC#1 (editor surfacing of a refused signal), AC#2 for
the four other update seams and the non-Library delete callers (Research
quick-note delete; the Library's blank-note GC and discard-new deletes),
AC#3 (eleven create seams), AC#4 (full re-derivation at landing; the User
Guide list was updated for this slice only), AC#5 (Minor 7). Related, owned
elsewhere: resolving a deletion review (TASK-34000.15); the version-only
wedge after a restore (TASK-34000.49); Review on a startup-held root not
restarting the watcher (TASK-34000.50).

**Fix round 1 (review 1: no Critical, three Important, eight Minor).**
- Pilot flake (I1): `_open_manage_sync_folders` presses the node
  `_wait_for_selector` re-queried after the settle pause; 8/8 consecutive
  local runs (counts in the Task 4 report).
- Restore after a restart (I2, preferred fix): a root held since startup
  (`needs_attention / review_changes`, no `action_id`, no incomplete
  operation; durable and unleased) is released by a bound note's
  `note_changed`: `_release_planner_hold` clears both block sets,
  `_run_released_pass` leases the root, runs the settled pass directly and
  starts the watcher. `note_changed` iterates loaded roots (`_root_paths`),
  not only leased ones. Real-stack test with a restart between delete and
  restore; negative controls for `activation_recovery_required`
  (`review_settings`, refused) and a durable hold with an open operation
  (Recovery's, refused). Minor 1: `_closed_roots` refused and the status
  re-read after the await, so a Pause in flight keeps its block.
- Docs/notes (I3): User Guide says the review cannot resolve a deletion
  (TASK-34000.15), the folder waits until the note is restored, and the
  next disk edit after an Undo can hold the folder (TASK-34000.49); the
  "Sync stopped needs a redraw" caveat is back (Minor 5); capture 08's hand
  edit disclosed above.
- Delete prompt (ruled in): `delete_confirm_copy(synced=...)` in
  `Widgets/Library/library_notes_canvas.py`, chosen at show time from the
  same live location the row states; unsynced copy unchanged; unit tests
  for both variants plus the synced assertion in the Pilot.
- Minor 2: `Tests/Architecture/test_notes_sync_snapshot_construction.py`
  pins `_publish` as the only construction site and that it stamps
  `published_at`. Minor 3: another day's confirmation carries its date.
  Minor 4: `resolve_cleanup`'s completed branch re-plans
  (`_run_settled_pass`) instead of stamping `up_to_date` unchecked -- the
  smaller correct change, since the previous confirmed time does not exist
  for a root that was held. Minor 6: strict `>` after a 5 ms sleep.
  Minor 7: the Pilot asserts the Manage live repaint (a two-sided change
  turns the row to "⚠ Needs attention · Next: Review changes" without a
  keypress). Minor 8: TASK-34000.49 references (not depends on) this task
  and records the in-app-edit workaround and the durable-after-restart
  hold. TASK-34000.50 filed by the controller, committed here, not
  implemented.

**Fix round 2 (review 2: one Important open, five deferred Minors).**
- Open 1: a Review or Check changes on the held row in a later session
  leases the root and re-blocks it without starting a watcher, so the
  restore's release scheduled nothing. The released pass now runs whenever
  no hint can be scheduled -- unleased, OR leased with no watcher running --
  and the guide sentence "in this session or a later one ... on its own,
  with no Check changes" holds. Pinned on the real stack with a restart and
  a `check_root`, and a `request_sync_now`, between restart and restore.
- Minor 2: the released pass is admitted as a background task registered
  in `_hint_tasks` (`_schedule_released_pass`), so a save or restore
  returns at once while `settle()` -- and Task 2's post-save bounded wait
  through it -- joins it, maintenance drains it, and a hint arriving
  meanwhile is coalesced; a test asserts the signal returns under a second
  and `settle` sees the heal.
- Minor 1: `note_changed` returns early under `_maintenance_closed`, and
  the pass runs inside `_producer_lifetime.operation()`; a refusal puts the
  hold back. Test: closed admission neither leases nor runs, resume heals.
- Minor 3: `resolve_cleanup` re-plan pinned with a conflicting plan (held,
  one observe, no `up_to_date` published). Minor 4: the incomplete-
  operation gate pinned alone (planner-shaped status + open operation,
  refused). Minor 5: the stale "not durably blocked" sentence corrected
  above. Minor 7: `note_changed` docstring says what the paused-root gate
  does and does not cover.

**Fix round 3 (review 3: one Important in the round-2 diff, hardening Minors).**
- Important 1: a hint coalesced onto the released pass was discarded -- the
  dirty flag was cleared before it was tested. Now read first
  (`rerun = ran and root_id in _dirty_hints`), then discard, pop and
  `schedule_hint`. `settle()` repeats its join until no admitted work is
  left, so the re-run is joined too. Two-folder real-stack test (root-2
  healthy with the watcher running; root-1 held since startup; restore,
  pass held after its observation, edit + signal): RED on 335b00b3d1
  (1 observation, file without the save), GREEN (2+, file carries it).
- Minor 3: `note_changed` marks the root dirty whenever a `_hint_tasks`
  entry is live for it, so a single-folder save during the released pass
  (no watcher yet, `schedule_hint` would refuse) is carried; same test,
  one-folder variant.
- Minor 1: a refused `_schedule_released_pass` puts the root back in
  `_blocked_roots`; test closes admission inside the release's own await
  and checks the next signal after resume heals.
- Minor 2/6: the handler is `_run_hint`'s -- only the fence/admission
  refusals (`runtime_producer_paused`, `root_admission_closed`,
  `notes_sync_cutover_not_admitted`) put the hold back silently; any other
  `RuntimeError` or `Exception` blocks the root, publishes `failed`
  (`offline` for a lost lease) and logs the error type only
  (`_log_bounded_failure`). The redundant `RecoveryRequired` import and
  tuple are gone. Test: injected `sqlite3.OperationalError` from the pass's
  `get_root`.
- Minor 4: the fence lines are pinned -- `operation()` wrap and hold
  put-back by a test that closes admission between schedule and run; the
  gate in `_schedule_released_pass` by the Minor 1 test's `== ()`.
- Minor 7: wall-clock assertion removed. Minor 8: `updated_date` fixed.
- Minor 5: `_maintenance_resume` restores admission and the watcher but
  knows nothing about signals refused under the fence, and remembering them
  would be a new mechanism -- so the User Guide carries a one-line caveat
  (a backup capturing the profile at the moment of the restore: the folder
  waits for the next change to one of its notes or Check changes).

Files: `Notes/notes_sync_runtime.py`, `Library/library_notes_lasting_sync_state.py`,
`UI/Library_Modules/library_notes_sync_attention.py`,
`UI/Library_Modules/library_notes_sync_controller.py`,
`UI/Library_Modules/library_notes_controller.py` (module-alias import,
line-neutral), `UI/Screens/library_screen.py` (line-neutral),
`Tests/Notes/test_notes_sync_delete_restore_signal.py` (new),
`Tests/UI/test_library_notes_sync_delete_restore.py` (new),
`Tests/UI/Library_Modules/test_library_notes_sync_attention_listener.py`,
`Tests/UI/Library_Modules/test_library_notes_sync_controller.py`,
`Tests/UI/test_library_notes_files_sync_journey.py` (dated-label asserts),
`Tests/Architecture/test_library_modules_size_ratchet.py` (re-pin),
`scripts/ui_pr_gate_census.txt`, `Docs/security/production-diagnostic-inventory.json`,
`Docs/User_Guide/library/notes.md`, `backlog/tasks/task-34000.49 ...md` (new).
<!-- SECTION:NOTES:END -->
