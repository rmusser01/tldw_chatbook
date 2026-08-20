# Library Notes, Folder Files, and Reviewed Sync Redesign

Date: 2026-08-19
Status: Independently reviewed; awaiting user approval of written specification

## Summary

Improve the existing Library Notes surfaces without merging their storage
models. Rename the two source modes to **Library notes** and **Folder files**,
make their authority visible in a pinned status row, add a mutation-free
**Check changes** phase before manual Notes Sync, retain truthful sync status
and receipts across navigation, reduce Files and Session Git action density,
and close contrast, recovery, keyboard, and compact-layout gaps.

The redesign preserves the current Library route, Database Notes storage,
File Notes disk authority, Session Git safety contracts, sync containment,
and existing wide/compact navigation. It introduces one new architectural
boundary: Notes Sync classifies a proposed plan separately from applying it.
The plan and apply paths share one classifier so preview and execution cannot
silently disagree.

## User Decisions

- Address all five critique priorities in one coordinated redesign tranche.
- Treat sync safety, the storage-authority mental model, and accessibility as
  co-equal goals.
- Keep Library notes and Folder files visibly separate.
- Preserve current modes and strengthen their labels, status, hierarchy, and
  recovery instead of building a unified Notes workbench.
- Use text-only design review; no browser visual companion.

## Job and Audience

The primary user is a local-first researcher or builder working in a terminal.
They need to know where note truth lives, edit without losing data, understand
what a sync will change before it changes anything, and distinguish ordinary
file editing from mirroring and Git publication.

The surface is **Operate** mode. Success is:

1. The current authority is understandable without reading documentation.
2. A manual sync never mutates before showing its proposed impact.
3. Auto-sync is visibly active wherever its consequences matter.
4. Every running, blocked, stale, failed, partial, and completed state names a
   next action.
5. Everyday Files editing remains fast while audit detail remains available.
6. All required controls and recovery paths remain keyboard-usable at the
   supported compact terminal sizes.

## Existing Decisions and Constraints

- ADR-021 and ADR-029 keep File Notes disk-authoritative. This redesign must
  not route Folder files through Database Notes or Notes Sync.
- ADR-027 keeps the active Database Note draft/save coordinator separate from
  Textual presentation and File Notes authority.
- ADR-031 governs screen keybindings and footer honesty. Do not add terminal-
  convention bindings or advertise actions that are not implemented.
- ADR-035, ADR-038, and ADR-039 govern Session Git staging, commit, and push.
  This redesign changes their presentation only, not their safety contracts.
- ADR-029 local-private-data containment continues to govern legacy Notes Sync
  traversal and file writes.
- The existing Library Notes adaptive design supports the Database Notes
  workflow down to 60x20. Sync must inherit that compact contract rather than
  invent another breakpoint.
- Textual `Select` and `Switch` were removed from this canvas after unreliable
  rendering. New choice controls must use proven Button/Static/Input and
  focusable-scroll patterns.

## Goals

- Rename `Database` to `Library notes` and `Files` to `Folder files` in the
  Notes source strip and related help/footer copy.
- Add a persistent authority/status row to each mode.
- Replace manual `Sync now` with Check, Review, Apply, and Receipt phases.
- Show exact predicted conflict winners before apply.
- Keep auto-sync opt-in and visibly active; invalidate its prior approval when
  user, root identity, direction, effective conflict policy, or allowed
  extensions change.
- Retain the latest durable sync outcome when leaving and returning to Sync.
- Persist validation and runtime failures in the panel with recovery actions.
- Reduce Files editor action competition without hiding state-relevant actions.
- Progressive-disclose Session Git audit facts while preserving decision facts.
- Meet the repository's measured contrast floor for error and disabled text.
- Prove compact containment, focus order, and recovery behavior in mounted UI.

## Non-goals

- Merge Library notes and Folder files into one authority or write path.
- Create a dedicated Notes destination outside Library.
- Change Database Notes or File Notes schema.
- Add a dependency or generic workflow/state-machine framework.
- Add per-conflict overrides or the existing `ASK` conflict strategy. TASK-97
  remains separate because this design previews the configured global winner;
  it does not prompt for a different choice on every conflict.
- Make disk and SQLite changes transactionally atomic; that is impossible with
  the current stores.
- Add rollback or history beyond the existing sync-session history and File
  Notes recovery contracts.
- Change Session Git trust, stage, commit, push, or uncertainty policy.
- Persist unapplied review plans across application restarts.
- Add new screen-specific keyboard shortcuts.

## Selected Interaction Direction

Keep the current Library shell and source strip. The two modes remain peers;
Sync remains a workflow opened from Library notes.

The source strip reads:

```text
Library notes (selected) | Folder files
```

or:

```text
Library notes | Folder files (selected)
```

Selection remains text-explicit and is reinforced by the existing semantic
selected treatment.

Immediately below the strip, the active mode owns one pinned authority row.
Examples:

```text
Stored in Library · Mirror ON · ~/Documents/Notes · Last: 2m ago · 2 changes
```

```text
Edits ~/vault directly · Saved · Session Git: 3 changes
```

The row uses product language, not implementation terminology. `Library DB`
may appear in diagnostic detail, never as the primary source label.

## Notes Sync Workflow

### Phases

Manual Sync is a five-phase in-canvas workflow:

```text
Configure -> Checking -> Review -> Applying -> Receipt
```

Back behavior:

- Configure -> Notes Navigator.
- Checking -> Notes Navigator; the non-mutating worker may finish, but its late
  result is discarded by generation token and performs no mutation.
- Review -> Configure, preserving the chosen folder/direction/policy.
- Applying -> Notes Navigator with `Back - sync continues`; apply is not
  cancelled or misreported as idle.
- Receipt -> Configure or Notes Navigator.

No modal owns the main review. A potentially long change list belongs in a
focusable in-canvas scroll region with the phase header and primary action
pinned above it.

### Configure

Configure contains:

- `Folder to mirror` Input and `Browse...`.
- `Direction: Bidirectional - change...`.
- `Conflicts: Newer wins - change...`.
- `Auto-sync every 5m: Off / Requested - Needs review / On`.
- Primary action `Check changes`.
- Latest receipt summary and `Show recent activity` disclosure.

Direction and conflict policy do not cycle hidden values. Pressing either
summary expands a three-row inline choice group. Choosing a row collapses the
group and restores focus to its summary. This uses the canvas's proven stacked
Button grammar and keeps all alternatives recognizable without `Select`.

Changing folder, direction, or conflict policy:

- discards any unapplied plan;
- increments the result-generation token;
- leaves any prior auto-sync request visible as `Needs review` but does not arm
  its timer;
- sets `Check changes, then explicitly enable auto-sync for these settings`;
- persists the new setting only at the existing explicit commit points.

In one-way directions, the direction itself decides the winner. The conflict
choice is disabled and reads `Conflicts: Direction decides`; it becomes
editable only for Bidirectional sync. This avoids asking for a policy that the
operation will ignore.

### Checking

`Check changes` runs filesystem scanning and Database-note reads off the UI
thread. Planning is strictly mutation-free:

- no note creates or updates;
- no disk writes;
- no sync-metadata updates;
- no `sync_sessions` or `sync_conflicts` rows;
- no config writes beyond configuration already committed by the user.

Progress reads `Checking - N/M`. Leaving the phase discards the eventual UI
result via generation token; it does not pretend to cancel a running thread.

### Review

The top summary is grouped into at most four decision categories:

```text
Library notes   3 create · 1 update
Folder files    1 create · 2 update
Conflicts       2 · Newer wins
Skipped         1 unsafe path
```

Each group expands to rows containing path/title, predicted action, and the
reason. Conflict rows name both changed sides and the configured winner.
Contents are not rendered into the review; hashes and bounded metadata are
enough for identity and avoid exposing entire notes unnecessarily.

Primary actions:

- `Apply sync` when at least one operation is applicable.
- `Record skipped result` when the plan has no applicable operation but has one
  or more skipped/conflict rows. This uses the guarded Apply path, performs a
  fresh replan, and writes history/conflict evidence without changing note
  content or disk files.
- `Done - no changes` only when the plan has no operations, conflicts, or
  skipped rows. It also performs a guarded fresh replan and records a durable
  `no_changes` receipt; Back remains the non-recording exit.
- `Back to settings`.

The review states:

`Apply is guarded per item, not atomic across Library and disk.`

This is a real storage limitation, not optional warning copy.

### Applying

Apply never trusts the earlier review blindly.

1. Acquire the ADR-021 `LegacyRootOwnershipGate` cooperative shared mutation
   lease. Hold it for the full legacy mutation pass, including fresh planning,
   writes, metadata, and terminal history. If that gate is unavailable, fail
   closed before mutation; `Check changes` may remain available, but `Apply`
   reads `Mutation ownership unavailable`.
2. Reopen the selected folder through `PinnedSyncRoot`.
3. Rebuild a fresh mutation-free plan before the first write.
4. Compare its fingerprint with the reviewed plan.
5. If they differ, perform no writes and return `Review stale - Check again`.
6. If they match, apply the operations in deterministic path order.
7. Immediately before every operation, revalidate its Database version/hash,
   disk hash/identity, and pinned containment facts supported by the existing
   primitives.
8. Use existing optimistic Database versions and pinned-root file operations.
9. If a detectable entry change occurs during apply, stop before mutating that
   entry and return a partial receipt. Already completed cross-store operations
   are not rolled back or described as atomic.
10. Release the shared lease in a `finally` boundary after terminal history is
    settled.

The first tranche has no user-facing apply cancellation. `Back - sync
continues` is the only Applying navigation action. Engine/shutdown cancellation
remains cooperative between operations, cannot undo completed operations, and
must settle a truthful `cancelled` receipt before releasing ownership.

### Receipt

Every apply result is grouped into:

- Applied.
- Conflicts resolved by configured policy.
- Skipped by containment or unsupported conditions.
- Not applied because stale/cancelled.
- Failed, with safe user-facing reasons.

The headline examples are:

```text
Done · 6 changes · 2 conflicts
Partial · 3 applied · 1 stale · Check again
Failed · Folder unavailable · Choose folder
```

The existing sync-session history remains the durable receipt owner. Planning
does not create history rows. No schema change is required: the existing
summary JSON adds backward-compatible `applied`, `reconciled`, `resolved`,
`skipped`, `stale`, `cancelled`, and `failed` counts while retaining every
current summary key and a new `outcome` discriminator. Current-session row
detail remains bounded screen state; durable history stores counts and safe
reason categories, not note contents.

The existing constrained `sync_sessions.status` column is mapped as follows:

| Receipt outcome | Existing durable status | Meaning |
| --- | --- | --- |
| `done` or `no_changes` | `completed` | Terminal and fully settled. |
| `partial` | `completed` | Terminal with some successful operations; `summary.outcome` prevents it being presented as full success. |
| `stale` | `completed` | Attempted Apply wrote nothing because the reviewed plan no longer matched. |
| `skipped` | `completed` | Terminal with no content/file operation applied; review evidence and safe skip reasons were recorded. |
| `cancelled` | `cancelled` | Internal/shutdown cancellation settled between operations. |
| `failed` | `failed` | No trustworthy successful completion claim can be made. |

The guarded terminal path—including `Apply sync`, `Record skipped result`, and
`Done - no changes`—creates the `running` session row only after mutation
ownership is acquired. A stale pre-write replan then completes that row with
`summary.outcome = "stale"`. A fresh empty plan completes with `no_changes`. A
fresh skipped-only plan completes with `summary.outcome = "skipped"`,
`processed_files = 0`, its exact skipped count in summary JSON, and
`errors_count` equal to containment/unsupported skips (not policy-skipped
conflicts). Conflict rows use the existing resolution values: `use_db` or
`use_disk` only after that winner is applied, `skip` for a conflict
intentionally left unapplied, and `NULL` only for unresolved legacy/ASK data.
`Record skipped result` writes each intentionally skipped conflict with
`resolution = "skip"` and `resolved_at`; skipped non-conflict operations do not
create conflict rows. The latest-status UI reads `summary.outcome`, falling
back to the legacy status/count fields for older rows.

### Classification truth table

One-way direction is authoritative and does not consult a conflict policy.
Bidirectional is the only mode that uses `disk_wins`, `db_wins`, or
`newer_wins`.

| Observed state | Disk -> Library | Library -> Disk | Bidirectional |
| --- | --- | --- | --- |
| Disk path exists; no linked Library note | Create Library note. | Leave disk path unchanged. | Create Library note. |
| Linked Library note exists; disk path is missing | Retain the Library note and unlink its mirror metadata. Never delete the note. | Recreate the disk file from Library. | `db_wins`: recreate disk; `disk_wins`: retain note and unlink; `newer_wins`: skip as a conflict because a deletion has no reliable comparable timestamp. |
| Both exist; contents and last-synced baseline match | No operation. | No operation. | No operation. |
| Both exist; contents match each other but baseline hash/mtime is stale or missing | Reconcile sync baseline metadata. | Reconcile sync baseline metadata. | Reconcile sync baseline metadata. |
| Both exist; only disk changed | Update Library from disk. | Restore disk from Library because the selected direction makes Library authoritative. | Update Library from disk. |
| Both exist; only Library changed | Restore Library from disk because the selected direction makes disk authoritative. | Update disk from Library. | Update disk from Library. |
| Both exist; both changed | Update Library from disk. | Update disk from Library. | `disk_wins`: update Library; `db_wins`: update disk; `newer_wins`: the strictly newer valid timestamp wins; equal/invalid/incomparable timestamps skip deterministically. |
| Neither destination can be safely identified/read | Skip with a bounded reason. | Skip with a bounded reason. | Skip with a bounded reason. |

The current data model has no durable tombstone that distinguishes a genuinely
new disk-only file from a disk file whose Library note was deleted. Therefore
Disk -> Library and Bidirectional continue to classify an unlinked disk-only
file as `Create Library note`; the review makes that potential resurrection
visible before Apply. Adding cross-store deletion tombstones is a schema and
conflict-policy change outside this tranche.

`newer_wins` compares only a Database content-modification timestamp and the
freshly descriptor-verified disk mtime. They are comparable only when the
Database timestamp is present, parseable, timezone-aware, and finite, and the
disk timestamp is present and finite. Both values are normalized to UTC at
their stored precision. A side wins only when its normalized value is strictly
greater; equal values or any failed precondition deterministically produce a
skipped conflict. Sync-metadata writes must not update the Database content
`last_modified` timestamp, because doing so would fabricate a newer content
edit.

## Sync Plan and Apply Contract

### Pure plan types

Add a small Notes-owned pure model, not a generic sync framework:

- `NotesSyncPlan`
  - lexical and canonical root identity;
  - user, direction, and conflict policy;
  - created time and deterministic fingerprint;
  - sorted `NotesSyncOperation` rows;
  - conflicts and skipped entries;
  - summary counts.
- `NotesSyncOperation`
  - operation kind;
  - relative path and optional note ID;
  - predicted destination/result;
  - Database expected version/content hash;
  - disk expected hash and descriptor-verified identity facts;
  - selected conflict winner when relevant.
  - `reconcile_baseline` is an applicable metadata-only operation when both
    contents match but `last_synced_disk_file_hash`/mtime is stale or missing.
    Its preconditions include the expected note version, equal current content
    hashes, prior baseline values, and verified disk identity/mtime.
- `NotesSyncApplyReceipt`
  - plan fingerprint;
  - applied, reconciled, resolved, skipped, stale, cancelled, and failed rows;
  - terminal status and safe recovery action.

Plans remain process-memory objects. They do not store full note contents and
are never accepted from an untrusted serialized source.

Applying `reconcile_baseline` updates only sync-link/baseline metadata through
an optimistic expected-version write. It increments the note version so a
concurrent mutation is detectable, but it preserves title, content, and the
user-visible/content `last_modified` timestamp. Review labels it `Refresh sync
baseline - contents already match`; Receipt counts it as `reconciled`, not as a
Library or disk content change. Its prior and proposed baseline values are part
of the plan fingerprint, so apply cannot silently repair different metadata
than the user reviewed.

### Ownership

- `NotesSyncEngine` owns classification and guarded execution.
- One shared classifier produces `NotesSyncPlan`; apply consumes that plan and
  reuses the same classifier for the stale-plan check. Do not maintain separate
  preview and mutation decision trees.
- `NotesSyncService` exposes typed `plan_folder(...)` and `apply_plan(...)`
  methods and retains history/conflict access.
- `plan_folder(...)` is read-only and does not acquire the mutation lease.
  `apply_plan(...)` must use the ADR-021 `LegacyRootOwnershipGate` shared lease
  for its full pass. The gate is a prerequisite: reviewed Apply and auto-sync
  remain fail-closed until the repository has a conforming cross-process gate.
- `LibraryScreen` owns the current in-memory plan, worker generation, visible
  phase, status, and screen-level auto-sync timer.
- `LibraryNotesCanvas` remains presentation-only.
- File Notes, its replica, recovery store, and Session Git do not consume these
  plan types.

### Fingerprint and race policy

The fingerprint covers all facts used to predict operations: root identity,
direction, policy, sorted paths, note IDs/versions/content hashes, disk hashes
and verified identities, and predicted operations. It does not claim to lock
either store.

The fresh-plan comparison prevents ordinary review staleness before writes.
Per-operation guards detect the races observable through optimistic Database
versions, hashes, descriptor identity, and cooperative in-app ownership.
Because SQLite and disk cannot share a transaction—and an external process can
write a file between the last supported precondition check and atomic
replacement—this design does not claim to prevent every external in-place-write
race. Observable drift yields Stale/Partial; an unobservable external race is a
documented residual risk. The interface and tests must never promise rollback,
all-or-nothing apply, or cross-process filesystem compare-and-swap.

## Auto-sync Contract

Turning Auto-sync On before review records only screen-level intent and renders
`Auto-sync requested - Needs review`; it does not persist an active timer. The
user first completes the reviewed terminal action (`Apply sync`, `Record
skipped result`, or `Done - no changes`) and reaches Receipt. Receipt then
offers `Enable auto-sync for these settings` only when the outcome is `done` or
`no_changes`; skipped, partial, stale, cancelled, and failed receipts show their
recovery action instead. Only the explicit Enable action binds the just-settled
receipt fingerprint, persists active auto-sync, and arms the timer. Applying or
finalizing a manual review alone is not implicit authorization for future
background writes.

Approval is invalidated whenever folder identity, user identity, direction, or
effective conflict policy changes. In one-way modes the effective policy is
the normalized value `direction_decides`, not a hidden prior selection.

Once enabled, each timer tick performs plan then apply using the approved
configuration and the same stale/per-operation guards. It does not open an
interactive review. The persistent authority row always shows that auto-sync
is On, its folder, and the latest outcome.

Auto-sync:

- skips quietly only when another sync is already running;
- records a durable receipt for every Apply attempt after ownership is
  acquired; screen state may retain additional bounded row detail;
- pauses and shows a recovery action when the folder is invalid or the service
  is unavailable;
- pauses after any `partial`, `stale`, `skipped`, `cancelled`, or `failed`
  receipt instead of repeating an unresolved operation every interval;
- honors the normalized effective conflict policy (`direction_decides` for a
  one-way sync, or the configured global policy for Bidirectional);
- does not enable `ASK` or per-conflict interaction.

The persisted `auto_sync` boolean alone represents a request, not proof of
approval. A new optional `[notes] auto_sync_approval_fingerprint` binds:

- the authenticated/configured user identifier used for Notes operations;
- lexical and canonical root paths;
- the root descriptor identity (`st_dev`/`st_ino` where meaningful), so a root
  replaced at the same path does not inherit approval;
- direction, normalized effective conflict policy, and allowed extensions;
- the reviewed plan fingerprint and approval contract version.

`Enable auto-sync for these settings` writes the fingerprint from a settled
`done` or `no_changes` Receipt. A bound-setting change clears approval and
leaves the request in `Needs review`. On startup, a missing, unresolvable,
replaced-root, user-mismatched, or otherwise mismatched fingerprint does not arm
the timer. The next tick always replans; the receipt plan fingerprint records
the exact approval event but is not expected to equal every future
changing-content plan.

## Persistent Status and Recovery

`_reset_library_notes_sync_transient_state()` no longer erases a running or
completed operation merely because the user leaves the panel. Screen-owned
state distinguishes presentation reset from operation state.

The authority row derives:

- configured auto-sync state from config;
- running/current state from the screen owner;
- latest durable outcome from existing sync history when available;
- detailed current-session activity from bounded screen memory.

Required statuses:

- Off / Needs review / Ready.
- Checking / Review ready / Applying.
- Done / No changes / Partial.
- Conflict / Stale / Skipped.
- Failed / Auto-sync paused.

Every error maps to safe, plain-language presentation plus a literal recovery
action. Raw exception strings remain in logs. Toasts may reinforce failures but
are never their sole carrier.

## Folder Files Refinement

Folder files retains the current navigator/editor topology and disk authority.
The purpose line becomes the pinned authority row rather than a clipped prose
sentence.

The path field gets a persistent one-row label:

```text
Target path · New / Move / Save copy
```

Default editor actions are:

```text
New · Move · Delete · More file actions
```

Contextual actions outrank the default:

- `Restore` appears immediately after a deletion.
- `Reload from disk` becomes primary on external-change conflict.
- `Save copy` appears when Dirty, Conflict, or Error makes it applicable.

`More file actions` toggles one inline secondary row containing only currently
applicable Protect/Unprotect, Reload, Save copy, and Refresh actions. The toggle
label and disclosure glyph expose its state. No modal, generic menu framework,
or new shortcut is added.

The action-status line remains visible and names outcomes/recovery.

## Session Git Refinement

Session Git behavior and safety contracts remain unchanged. Commit and push
reviews lead with four decision-fact groups:

1. **What:** exact commit or push action.
2. **Where:** local branch or configured remote/ref.
3. **Impact:** included session notes and unrelated-change promise.
4. **Recovery:** cancel/back boundary and uncertain-result action.

The Commit review keeps these facts visible without disclosure:

- canonical repository path, local branch, and author identity;
- exact commit subject/body and candidate commit ID when available;
- included session-note paths/count and the unrelated-change promise;
- whether trusted repository hooks or filters can run;
- local-only/no-network effect and the Back/cancel boundary.

The Push authorization/final review keeps these facts visible without
disclosure:

- exact commit subject/ID and expected parent-to-candidate transition;
- local branch, full destination ref, sanitized endpoint identity, and secure
  transport/authentication method;
- included session-note provenance and the fact that required Git objects—not
  independently selected note rows—are published;
- credential-helper or SSH-agent contact, local pre-push-hook bypass, and the
  possibility of remote hooks, CI, mirrors, reflogs, or policy effects;
- exact-lease protection in user language, later edits remaining local, and
  the uncertain-result recovery action.

`Show technical details` contains only lower-frequency implementation evidence:
repository/worktree filesystem identity tuples, raw HEAD/parent object IDs
already summarized above, index and ownership signatures, policy fingerprints,
provenance sequence numbers, and internal lease/status identifiers. Endpoint
details remain in the existing selectable Endpoint Details surface; transport,
authentication, hook effects, destination ref, commit ID, and author identity
must not be hidden because they can change an authorization decision.

## Visual and Accessibility Contract

- Preserve the established Neon Workbench visual system; this is refinement,
  not a replacement visual world.
- Use semantic tokens only. Bright color is earned by focus, running, success,
  warning, conflict, blocked, and error state.
- Use `$ds-status-error-readable` for error text. Decorative error tokens may
  support borders/backgrounds.
- Scoped app-level disabled styling must neutralize compounded dimming and meet
  the repository's measured 3:1 minimum while remaining visibly disabled.
- Every selected/running/conflict/disabled/auto-sync state uses literal text in
  addition to color or glyph.
- Focus changes never alter geometry.
- Pinned authority and phase/status rows do not scroll away.
- Scrollable review/activity surfaces show the existing visible overflow/fold
  hint when content continues below the viewport.
- User-controlled paths/titles render with markup disabled or escaped.

## Responsive and Keyboard Contract

- Database Notes Sync inherits the existing compact Library Notes stage and
  Back hierarchy at measured widths; no new breakpoint is introduced.
- At 60x20, Configure, Review, Receipt, and every recovery action are reachable
  without horizontal scrolling.
- Expanded direction/conflict choices and More file actions may reduce content
  height temporarily but must remain inside a keyboard-scrollable owner.
- At Folder files' existing narrow breakpoint, navigator and editor remain
  mutually exclusive and focus always moves to a mounted visible target.
- Session Git retains its 40x20 scroll and phase-safe Escape behavior.
- Footer hints advertise only actions implemented in the current phase.
- No bindings use terminal-convention keys or shadow global Ctrl+P, Ctrl+Q,
  F1, or F6.

## States and Ranges

Test at minimum:

- No folder, missing folder, file selected instead of folder.
- Empty folder and no Library notes.
- No changes.
- One Library create, one disk create, and mixed updates.
- Both-changed and deleted-on-disk conflicts under every exposed global policy.
- Every row in the classification truth table, including one-way restoration
  and the disk-only resurrection limitation.
- Containment skip, unreadable file, optimistic DB conflict, and disk race.
- Stale plan before apply and race during apply.
- Partial receipt after one or more successful operations.
- Ownership-gate unavailable, shared-lease acquisition, full-pass retention,
  shutdown cancellation, and release on every terminal path.
- Apply failure and service unavailable.
- Auto-sync Off, requested/Needs review, explicit no-change approval, On,
  running, paused, resumed, user mismatch, and same-path root replacement.
- Activity from 0 through the current 20-entry cap.
- Long Unicode paths/titles and paths wider than the viewport.
- Large roots: summary rendering must remain bounded and row detail scrollable;
  the UI never renders full file contents.

## Error Handling

- Boundary validation failures are persistent status with `Choose folder`.
- Plan staleness is not an error toast; it is a review state with `Check again`.
- Per-item containment rejection becomes Skipped, never generic Failed.
- Optimistic DB or disk identity drift stops before that item and yields Partial
  or Stale depending on whether prior operations applied.
- Service unavailability yields `Open settings` or the nearest existing
  configuration action.
- Unknown exceptions log full context without note contents or credentials and
  render a safe generic reason with `Retry`.
- Late worker results are ignored by exact generation/phase identity.

## Testing and Evidence

### Pure/service tests

- Plan classification is mutation-free for every direction and conflict policy.
- The complete direction/state/policy truth table is table-driven in one test
  corpus consumed by planner and apply-replan tests.
- Equal-content/stale-baseline plans produce one guarded metadata
  reconciliation, preserve content `last_modified`, and cannot poison later
  one-sided change classification.
- `newer_wins` strictly orders valid timestamps and skips equal, missing,
  non-finite, naive, or unparseable values.
- Preview and apply share the same classifier.
- Deterministic plan fingerprints change for every relevant precondition.
- Fresh-plan mismatch performs zero writes.
- Per-operation DB/disk guards stop every supported detectable stale write.
- Detectable mid-apply races produce truthful partial receipts; tests do not
  claim an external filesystem compare-and-swap that the primitives lack.
- Plan objects contain no full note content.
- Auto-sync approval binds user/root identity/direction/effective
  policy/extensions and invalidates on every mismatch.
- Durable receipt tests cover every `summary.outcome` to constrained status
  mapping—including skipped-only/conflict-only plans—and the legacy-history
  fallback.
- Skipped-only Review offers `Record skipped result`, persists conflict
  resolution `skip`, and never offers Auto-sync enablement on its Receipt.
- Apply cannot mutate without the ADR-021 shared legacy lease and holds it
  through terminal history settlement.
- Existing containment, permissions, sync history, and File Notes authority
  suites remain green.

### Mounted Textual tests

- Source labels and text-explicit selection.
- Authority rows in Library notes and Folder files.
- Configure -> Check -> Review -> Apply -> Receipt focus flow.
- Back behavior and late-result rejection in every phase.
- Persistent status while leaving/re-entering Sync.
- Inline direction/conflict choices expose all options.
- Files contextual/secondary action visibility and focus restoration.
- Session Git decision facts plus technical-detail disclosure.
- Error/disabled classes, recovery actions, and footer honesty.
- Geometry containment and keyboard reachability at 60x20, 80x24, 120x35,
  and 160x45.

### Visual/live evidence

- Current-HEAD captures for wide and compact Configure, Review, Receipt,
  Folder files, commit review, and push review.
- Focused, disabled, running, conflict, stale, partial, and failed states.
- Representative theme contrast measurements for ordinary, error, disabled,
  selected, and focused text.
- Scratch-profile live run with a temporary folder and Database proving Check
  performs no mutation and Apply produces the reviewed receipt.

## Implementation Decomposition

This design is one coordinated tranche but should not become one indivisible
task or one enormous commit.

1. **Reviewed Sync contract:** ADR, pure plan/receipt types, shared classifier,
   conflict truth table, mutation-free plan, guarded apply, service tests, and
   history compatibility.
2. **Legacy ownership prerequisite:** locate or implement the ADR-021
   `LegacyRootOwnershipGate`, route every legacy mutation entry point through
   its full-pass shared lease, and keep reviewed Apply/auto-sync fail-closed
   until its cross-process contract is proven.
3. **Sync presentation and durable status:** Canvas/screen phases, authority
   row, explicit choices, auto-sync approval, recovery, compact UI tests.
4. **Folder Files and Session Git refinement:** authority row, target-path
   label, progressive actions/details, semantic state styling, responsive and
   accessibility verification.
5. **Polish and evidence:** generated CSS, live captures, contrast checks,
   documentation, critique rerun, and regression closeout.

The implementation plan may combine adjacent steps into one PR only when each
step remains independently testable and the Backlog acceptance criteria remain
atomic.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/068-reviewed-notes-sync-plan-and-apply.md`

Reason: this design introduces a long-lived Notes Sync service contract,
separates read-only planning from mutation, defines stale/partial apply policy,
defines a direction/conflict truth table and durable status mapping, applies the
ADR-021 ownership lease to the reviewed path, and changes auto-sync approval
semantics. Existing ADRs govern File Notes authority and legacy Sync
containment but do not decide this combined boundary.

The ADR must be created before implementation and linked from the Backlog task,
implementation plan, and closeout notes.

## Known Risks and Resolutions

### Preview/execution drift

Risk: a separately implemented preview will disagree with mutation logic.

Resolution: one classifier produces the plan; apply consumes and rechecks that
same plan. No duplicate preview decision tree.

### False atomicity

Risk: users assume reviewed apply is one transaction across SQLite and disk.

Resolution: fresh-plan comparison before writes, per-item guards during apply,
deterministic ordering, and explicit partial receipts. Never promise rollback.

### TOCTOU after replan

Risk: disk or Database changes after fresh-plan comparison.

Resolution: revalidate every operation immediately before mutation and stop
before supported detectable stale writes. State explicitly that external
writers outside cooperative ownership can still win an unobservable filesystem
race between the final check and replacement.

### Missing legacy ownership gate

Risk: reviewed Sync adds a safer UI while continuing to violate ADR-021's
cross-process mutation-ownership contract.

Resolution: make the gate an explicit prerequisite, acquire its cooperative
shared lease for the entire apply/history pass, and fail closed when ownership
cannot be proven. Do not substitute path-prefix checks or the File Notes
process-memory session gate.

### Auto-sync approval surviving changed settings

Risk: the user reviews one configuration but auto-sync later runs another.

Resolution: explicit authorization after Review binds user, canonical and
descriptor root identity, direction, normalized policy, extensions, and
contract version. A mismatch keeps the request visible as Needs review but
does not arm background writes.

### History schema creep

Risk: detailed receipts trigger an unrelated Database migration.

Resolution: map every new receipt outcome onto the existing constrained status
values and store the precise outcome/counts in backward-compatible summary
JSON. Keep detailed current-session rows in screen memory. No schema change in
this tranche.

### Equal content with stale baseline

Risk: treating equal current hashes as a no-op leaves an old baseline, so the
next one-sided edit appears to be a both-changed conflict.

Resolution: model and review a guarded metadata-only baseline reconciliation,
fingerprint its old/new values, preserve content `last_modified`, and count it
separately from user-content changes.

### Deletion ambiguity without tombstones

Risk: a disk file belonging to a deleted Library note is indistinguishable from
a genuinely new disk-only note and can be proposed for recreation in Library.

Resolution: disclose the exact `Create Library note` operation in Review and
document the limitation. Durable tombstones require a separate schema/conflict
ADR and are not smuggled into this UI tranche.

### Large roots and terminal overload

Risk: review lists overwhelm the UI or retain note contents.

Resolution: bounded summary first, expandable metadata rows, no contents in
plan/presentation, focusable scroll owners, and compact geometry tests.

### TASK-97 overlap

Risk: the existing conflict-dialog task is mistaken as delivered.

Resolution: explicitly keep per-conflict overrides out of scope. The reviewed
global winner improves prevention but does not complete TASK-97.

### Scope collision with File Notes authority

Risk: shared visual language becomes shared storage behavior.

Resolution: authority rows share presentation grammar only. ADR-021/029 File
Notes ownership and all Session Git contracts remain separate and unchanged.
