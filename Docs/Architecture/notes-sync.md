# Notes and bidirectional sync

This document describes the lasting file↔DB notes sync: the runtime owner, the review-first reconciler (this is **not** last-write-wins), the recovery-first executor, root leases, the watcher, and the device-state store. Session-scoped file-notes editing and guarded git publishing are covered at the end.

## Authoritative files

| File | Role |
| --- | --- |
| `Notes/notes_sync_runtime.py` | `NotesSyncRuntimeOwner` — app-owned runtime: `start`, `_reconcile`, `review_setup`, `check_root`, `request_sync_now`, `apply_reviewed`, `pause_root`/`resume_root`/`activate_root`, `shutdown` |
| `Notes/notes_sync_reconciler.py` | `plan_reconciliation()` — classifies immutable observations **without selecting winners** |
| `Notes/notes_sync_executor.py` | `NotesSyncExecutor` — durable, recovery-first execution of reviewed actions |
| `Notes/notes_sync_watcher.py` | `PollingNotesSyncWatcher` — polling hint emitter (1–10 s, jittered backoff) |
| `Notes/notes_sync_conflicts.py` | `NotesSyncConflictChoice` (KEEP_FILE / KEEP_NOTE / KEEP_BOTH / SKIP), bounded comparisons |
| `Notes/notes_sync_coordinator.py` | `NotesSyncRootCoordinator`, `RootLease` — per-root process lease; two Chatbook processes cannot own one root |
| `Notes/notes_device_state_store.py` | Durable roots/bindings/operations/recovery records at `<user_data_dir>/tldw_chatbook_notes_sync_state.db` |
| `Notes/notes_sync_authority.py` + `notes_scope_service.py` | `NotesScopeSyncAuthority` (DB-side CRUD authority), global/workspace scopes |
| `Notes/sync_paths.py` + `notes_sync_filesystem.py` | Descriptor-anchored FS boundary (`O_NOFOLLOW`, dir-fd pinned ops, 10 MiB file cap, xattr bounds) |
| `Notes/notes_sync_legacy.py` | Reads canonical legacy evidence and persists device-private paused candidates; fixed receipt projection handles newly imported, unrepresented linked notes |
| `Notes/file_notes_service.py` | `FileNotesService` — session-scoped file-notes editing (disk authority) |
| `Notes/file_notes_git_*.py`, `git_process_containment.py` | Guarded session git commit/push (ADR-038/039) |

There is no single `Notes/sync_engine.py` — that name is legacy; the "lasting sync" engine is the module set above.

## The model: review-first, not LWW

Synced extensions are `.md`, `.markdown`, `.txt`. The reconciler classifies each binding from immutable observations:

| Observation | Plan |
| --- | --- |
| File changed, note unchanged | `UPDATE_NOTE` (disk → DB) |
| Note changed, file unchanged | `UPDATE_FILE` (DB → disk) |
| **Both changed** | **CONFLICT** (`both_sides_changed`) — surfaced for explicit resolution, never auto-picked |
| One side missing | `DELETION_REVIEW` |
| Direction violation | `out_of_direction_*` conflict |
| `duplicate_authority` | PAUSE the root |
| Move with identity mismatch | `ambiguous_identity` conflict |
| Root offline / overlapping / capability loss / stale observation | Plan-level skip with reason |

Last-write-wins (`newer_wins` and friends) exists only in the legacy policy vocabulary consumed by the read-only migrator.

## Startup and imported linked notes

`NotesSyncRuntimeOwner._start_once` retains the native cutover-admission, sole-process and closing gates. An absent marker uses the existing legacy migration path. With the exact current marker, `build_notes_sync_import_reviewer` can discover linked notes added by a completed mobile import without replaying old sync sessions or migrating represented note edits again.

The boot construction gate and its captured runtime start-evidence callback in `app_service_wiring.py` first retain the existing configured-directory/root checks. Their added path requires an already existing Notes store with the exact current marker and a constant receipt-presence probe on the selected App Core owner. No-receipt/no-linked fast paths do not create a Notes database. Before full receipt decoding or file/version probes, current linked IDs already owned by ANY-state binding are excluded. Complete native Chat owner receipts then identify imported mapped IDs under one ordinary 16 MiB materialization allowance.

Native root/file evidence and planning may add only genuinely new disjoint paused candidates. The private transaction rechecks all current root paths and binding ownership, including arbitrary root IDs and active roots. Ambiguous, overlapping, invalid or oversized evidence refuses the new enrollment; existing roots, leases and watchers keep their established behavior. These candidates and their pending reviews do not activate sync, grant file access or modify note/file contents.

Core snapshot reads use the actual selected native owner. Private schema initialization occurs outside its transaction; newly owned Core thread handles retire before private candidate persistence. Borrowed handles remain untouched, and borrowed active transactions refuse enrollment. These are native SELECT transaction and retirement guarantees, not physical read-only/no-checkpoint or cross-database atomicity claims.

The defaulted `import_review_required` boolean is projected separately through the runtime and Library snapshots. A markup-disabled status widget says: “At startup, some file links could not be added. Notes remain available in Notes; review folder setup before syncing.” It does not replace an action/failure message, change root controls or promise automatic enrollment through a new action. The runtime can remain `active` while individual paused candidate roots show `review_migration` and have no watcher.

Fresh absent- and present-marker App flows verify mapped note ID/version/content digest, retained canonical candidates, default character ID1 after canonical rename/import/process exit, repeated cold-start stability and native Notes handle closure. The present-marker flow retains handles from every actual private-store open, including the receipt callback's separate store. This qualifies startup and pending-review state; it does not claim a manual Notes-screen interaction or an active sync grant.

## Dataflow (edit → sync)

1. The polling watcher detects a changed root and emits a scheduling hint.
2. The runtime observes the root: file and note observations against device-state baselines, plus an observation token.
3. `plan_reconciliation` classifies bindings into safe actions and attention items (conflicts, deletion reviews, pauses) — no winners chosen.
4. The UI shows the reviewed plan; `apply_reviewed` accepts only the **current** token (a stale plan requires "Check again" — content may have moved).
5. `NotesSyncExecutor.execute` applies exactly the reviewed actions with a staged durable journal; a mid-apply crash leaves a resumable incomplete operation (`_resume_incomplete` classifies and finishes it on next start).
6. Conflicts resolve explicitly KEEP_FILE / KEEP_NOTE / KEEP_BOTH; `undo_resolution` reverses via linked operation ids; retained conflict copies live for 30 days.

## Root coordination

Each synced folder is a **root** with a process lease: admission validates reparse points and overlap; two Chatbook processes cannot own one root simultaneously. Roots can be paused/resumed/activated explicitly; duplicate-authority conditions pause rather than guess.

## Config keys

`notes.recovery_capacity_bytes` (default 256 MiB; env `TLDW_NOTES_SYNC_RECOVERY_CAPACITY_BYTES`), `notes.sync_watcher_interval_seconds` (1.0), `notes.sync_watcher_max_interval_seconds` (10.0, ceiling 3600).

## Session file-notes and git publishing

Separately from lasting sync, `FileNotesService` provides session-scoped editing of file-backed notes (open/save/move/delete/restore/reconcile) with **disk authority** (ADR-296): the file is the truth; the DB follows. Guarded git publishing (`file_notes_git_*`) performs session commits and pushes with process containment (ADR-035/038/039) — commit and push are separately gated user actions, never automatic.

Note templates (`Config_Files/create_custom_template.py`, stored in the user config dir) support `{date}`/`{time}`/`{datetime}` placeholders and feed the Library Notes UI.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Both sides changed | Bounded machine-readable CONFLICT attention; explicit resolution required |
| Mid-apply crash | Durable journal; resumable incomplete op on next start |
| Stale review plan | Token mismatch; re-check required before apply |
| Root not available / overlapping | PAUSE with reason code and next-action copy |
| Oversized file / xattr bounds | Refused at the FS boundary |
| Second process claims a root | Lease refusal |

## Governing decisions

ADR-021 (file-backed notes disk authority and recovery), ADR-296 (file-notes disk authority), ADR-035 (session git index controls), ADR-038/039 (guarded session commit/push), ADR-059 (notes folder import and device-local sync ownership), ADR-073 (round-trip and interoperability constraints), ADR-244 (shared Console/mobile ownership and receipt-backed startup candidates). Feature doc: `Docs/Features/notes_bidirectional_sync.md` (owner table matches code).

## Verified gotchas

1. "Bidirectional LWW sync" in old docs is wrong for the current engine — both-sides-changed is always a reviewed conflict.
2. The watcher is polling-only (no FS events dependency); hints drive reconciliation timing.
3. Executor worker coroutines run one loop per worker thread — an earlier per-call `asyncio.run` leaked SQLite connections.
4. `duplicate_authority` pauses the root rather than picking a winner, because two bindings claiming one note cannot be reconciled safely.


## Session file-notes implementation rationale

The original incident narratives below accompany the concise local contracts.
Only explanatory prose moved during PR3345 dev integration; executable code,
Args/Returns/Raises, ownership and deadlines remain unchanged.

### `library_file_notes_workspace._FolderRowStatus`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Folder-row status that fits itself again once layout changes its width.

TASK-38710: the fit is deferred with ``call_after_refresh``, which runs no
layout pass. When it ran before the layout that seats "Review…", it fitted
the old width and nothing fitted again. Layout sets this width (``1fr``,
one row, no wrap), so a refit cannot resize it. The prompt's
``-empty-root`` auto width is never fitted, but it is still recorded: a
relink back to the old width must refit.

### `library_file_notes_workspace.FileNotesEditorTextArea`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

The Folder files editor, with document-end and document-start keys.

task-32552 AC#1: the same defect task-32247 fixed in the Library editor
(``NoteEditorTextArea``): Textual 8's ``TextArea`` binds ``end``/``home``
to the LINE ends and defines no ``ctrl+end`` binding or
``cursor_document_end`` action at all, so Ctrl+End here was not
swallowed -- it did not exist, and the text typed after it landed at
the click (critique #3, B D9: ``NoteTAILEDITs`` on disk). The four lines
are copied rather than the class imported: that editor's priority Tab
pair is namespaced to the Library screen and is not this widget's
concern.

### `library_file_notes_workspace.FileNotesHistoryDialog`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Browse one protected file's bounded revision history and act on it.

task-34381: the dialog only PRESENTS the bounded listing and collects a
chosen action; every service call runs in the workspace under the same
transition guards as its other file actions, so a dialog opened against
one root can never publish into a root that changed beneath it.

TASK-36386 / ADR-097: the ``file-notes-history*`` rules in ``features/_notes.tcss``
leave the boot path: ``build_css.py`` splits them into the sheet this ``CSS_PATH``
loads on first open. Its ``App.-wide-viewport`` tier rule goes with them
(TASK-38719).

### `library_file_notes_workspace._read_replica`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Run one direct replica read that quit can close behind (TASK-38711).

These reads skip the service lock. ``_core_closing`` silently refuses
to close a file-backed replica while one is in flight, and a closed one
reopens on its next use. So an owned replica's reads share its owner
thread, where the close queues behind them, and a read that starts once
quit has begun returns None without touching the replica.

### `library_file_notes_workspace._structural_wait_line`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Return the busy row's line for one in-flight folder change.

task-32121: past the patience window the shared helper's line ends
in ``· Cancel``, which sat beside the actual Cancel button and read
as a second control with the same label. This surface reports how
far the scan has actually got instead, and the decisions are the
buttons in the same row -- measured, not stylistic: at 120 columns
the row leaves the status 46 cells, so a line naming both choices
as well elides to ``Changing folder… · sti...ting · Choose another``.

### `library_file_notes_workspace._configured_sync_folder`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Return the configured notes folder to offer by name, when usable.

task-32136: a user who already configured a notes folder should be
offered it by name instead of being sent to a file picker that
opens on their home directory.

task-32180: ``[file_notes] root`` -- the key this mode itself writes
on every successful folder change -- is read first; the legacy
``[notes] sync_directory`` (TASK-21112 calls it that) stays the
fallback for a profile that has never linked a folder here.

Both are config-derived input, so they go through
``path_validation`` (review round 2) rather than straight to
``is_dir()``: a relative spelling would otherwise resolve against
whatever directory the app was launched from.

### `library_file_notes_workspace._sync_body_panes`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Keep the Library rail; make only the file panes wait for a folder.

task-32173: Folder files is a mode of Notes (task-32136 user
decision), so its rail belongs on screen from the first frame, not
after a folder is linked. The body itself stays mounted --
display-gating it left the reader shell with zero width, so the
adaptive layout never resolved at all and the rail could not open
even at 235 columns.

Compact terminals are unchanged because the resolver already closes
the rail below the Library's compact breakpoint: at 60 and at 100
columns it reports ``library_open=False``, so the unlinked body
holds nothing to paint.

### `library_file_notes_workspace._fit_path_surfaces`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Keep both identity lines on one row instead of folding them.

task-32614: a path is one long token, and Textual folds a token that
does not fit at whatever column it runs out of -- at the 34 cells a
100x30 terminal gave this pane, the breadcrumb broke at
"2026"/"-09-14" and the absolute path took six rows broken at
"…m80n2j152t"/"9gw3w8qwk…". ``elide_path_middle`` rather than a
plain truncate for the reason it was written: the basename is the
fragment the reader recognizes the file by, and a tail-cut spends
the row on ``/private/var/folders/...`` and hides it.

Args:
    width: Settled work-pane width to fall back on when a widget has
        not been laid out yet (``0`` to use its own region).

Returns:
    None.

### `library_file_notes_workspace._sync_editor_visibility`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Show the editable body only when there is a file to edit.

task-32621 AC#1 (A cap 29): with nothing selected the right pane drew
a full, apparently editable box beside the words "No file selected".
Typing into it went nowhere, on the one surface whose promise is that
it edits the real file. Hidden rather than unmounted: every caller
queries ``#file-notes-editor`` unconditionally, and the editor is a
retained widget this workspace constructs once.

### `library_file_notes_workspace._scan_for_root`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Scan a candidate root on a worker thread, cancellably and bounded.

Two halves of the same task-32121 wedge. The scan itself is now
cancellable, so an abandoned one lets go of ``_service_lock``
instead of holding it to the end of a home directory. And the wait
for that lock is bounded, so a scan stuck inside a single
uninterruptible syscall (a dead network mount) makes the NEXT
change report a timeout instead of queueing behind it silently.

That bound is the CALLER's deadline, not a second one down here
(review round 2): abandoning an attempt sets its cancel flag, and
``Keep waiting`` extends the deadline that decides when to abandon.
A worker-side timeout of its own ended the change a moment after
the user asked for more time. A caller that passes no flag has
nobody to abandon it, so it keeps a bound of its own.

Args:
    service: Service bound to the candidate root.
    cancel_event: This attempt's own cancel flag, or None when the
        caller cannot abandon the scan.
    generation: The ``_root_generation`` this attempt owns, so its
        progress reports stop at the attempt that started them.

Returns:
    The candidate root's scan result.

Raises:
    ScanCancelled: If this attempt was abandoned, waiting for the
        lock or mid-scan.
    _ServiceLockBusy: If an earlier operation still holds the lock
        after ``ROOT_CHANGE_TIMEOUT_SECONDS``, for a caller that
        cannot abandon this attempt.

### `library_file_notes_workspace._record_root_scan_progress`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Publish the running entry count from the scan thread.

A plain attribute store, deliberately: the busy row reads it on its
own repaint tick, so no cross-thread call is needed and a stale
read is at worst one tick behind. It is not a cross-attempt store,
though (PR #2549 review, finding 2): an abandoned scan's thread can
still be inside its file loop when the next change resets the
counter, and its next report would otherwise land on the new
attempt's row.

Args:
    entries: Entries this scan has seen so far.
    generation: The ``_root_generation`` the reporting scan was
        started for. A report from any older attempt is dropped.

### `library_file_notes_workspace._report_root_change_reason`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Keep one folder-change outcome visible until the next attempt.

task-32121: the timeout reason used to be written only into the
action-status line at the bottom of the editor pane, while the root
row -- the line the user is actually watching -- silently reverted
to the previously linked folder. The reason now owns the row until
another folder change starts.

### `library_file_notes_workspace._set_poll_interval`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Re-arm the poll timer only when the cadence changes (task-11).

Stopping and recreating is safe even from inside a tick callback:
Textual delivers the cancellation to the timer task after the
callback returns. Re-arming on every call would reset the phase, so
unchanged cadences are a no-op. Before mount (``_poll_timer`` is
None) only the interval value moves; ``on_mount`` arms the timer
with whatever the value is then.

### `library_file_notes_workspace._change_root_with_deadline`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Run one folder change under a deadline the user can cut short.

task-32055: ``set_root`` itself is unbounded -- it scans a folder
that may live on an unreachable network volume -- and while it ran,
every leave guard that consults ``_root_transitioning`` refused,
which is how a folder change swallowed Escape, the back cue and the
palette. The wait now owns a cancellable task, publishes an honest
status line, and always leaves the previously linked folder in
place when it does not complete.

### `library_file_notes_workspace._abandon_root_change_task`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Cancel the wait's task and release the canvas in the same beat.

The transition flags are cleared HERE rather than in ``set_root``'s
own ``finally``: that only runs once the cancelled coroutine
unwinds, and every leave guard consulted in the meantime would
still refuse -- which is the swallowed-Escape bug. Bumping the
generation makes the abandoned run's late results stale, so it can
never commit the folder it was still scanning.

Invariant (task-32180): **every** way a folder change ends early
arrives here, and nothing else releases the previous scan's hold on
``_service_lock``. Three call sites reach it: the deadline and the
``CancelledError`` branch of ``_change_root_with_deadline``, and
the ``StructuralWait.cancel`` partial that Cancel, Escape, the back
cue, the navigation flush and ``Choose another`` all go through. A
re-entrant folder change arrives by the second of those: the worker
is ``exclusive=True`` in the ``file-notes-root-change`` group, so
picking a second folder cancels the first worker.

What it is NOT: synchronous. Textual's exclusive-worker cancel
delivers ``CancelledError`` on a later loop turn, so the new
attempt's ``set_root`` can reach ``_scan_for_root`` while the
abandoned scan still holds the lock. That overlap is covered, not
avoided: ``_scan_for_root`` polls the lock and gives up only on
THIS attempt's cancel flag, so the new attempt simply waits out
the microseconds the old one needs to notice its flag. Do not
write an assertion here claiming the previous attempt is already
finished -- it usually is not.

### `library_file_notes_workspace.cancel_structural_wait`

Source: [library_file_notes_workspace.py](../../tldw_chatbook/Widgets/Library/library_file_notes_workspace.py).

Abandon the in-flight structural wait, keeping the current folder.

The single seam for every exit: the Cancel button, Escape, the back
cue and the app's navigation flush all reach it, so a structural
wait can gate the WRITE (a second folder change) without ever
gating the way out.

Args:
    leaving: Whether this exit also leaves the surface, in which
        case the outcome is announced rather than written to the
        row the user is walking away from (task-32102).

Returns:
    True when a wait was actually abandoned.

### `library_file_notes_git_panel._entry_focus_is_still_ours`

Source: [library_file_notes_git_panel.py](../../tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py).

Return whether the armed entry focus may still be taken.

Two ways it is still ours, and no third (review F1):

* nothing moved -- the exact widget focus was on when the status
  landed still holds it; or
* the panel took that widget away AND focus is still in here.
  Trust landing hides the trust button, and Textual moves focus
  itself the moment the focused widget stops being focusable --
  onto this panel's own next control. The user's focus was yanked
  by US, so there is no deliberate choice left to protect. Asked
  HERE and not when the control was hidden, because Textual does
  not guarantee it moves focus inside the same call that flipped
  ``display`` -- and when it did not, AC#1 silently stopped firing
  (measured: 8 of 10 runs). Both halves are load-bearing: pressing
  Edit ALSO hides the anchor (the whole Manage surface goes), and
  that one is the user, which is why the landing place decides.

Anything else -- the user Tabbed away, pressed Edit, focused
another control in this panel -- is a focus move we must not undo.
