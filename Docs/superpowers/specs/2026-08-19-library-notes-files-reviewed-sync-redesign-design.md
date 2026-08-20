# Library Notes, Folder Files, and Lasting Sync Surface Redesign

Date: 2026-08-20
Status: User-approved corrected direction; pending independent specification review
Governing decisions:
[ADR-059](../../../backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md),
[ADR-060](../../../backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md)

## Summary

Improve the Library Notes experience without merging its two storage models.
Rename the source modes to **Library notes** and **Folder files**, keep their
authority visible in a pinned status row, replace the legacy single-root Sync
panel with the accepted **Add from files…** and lasting-sync-root model, reduce
Folder Files and Session Git action density, and close contrast, recovery,
keyboard, and compact-layout gaps.

This specification is a surface and delivery addendum to ADR-059/060 and the
approved [Notes Folder Import and Lasting Sync design](2026-08-12-notes-folder-import-sync-design.md).
It does not define a second sync engine. In particular, it removes the prior
draft's global `newer_wins`/`disk_wins`/`db_wins` policies, single timer,
legacy `sync_sessions` receipt mapping, and legacy ownership-gate extension.
Those concepts conflict with the accepted multi-root, explicit-attention,
device-private journal architecture.

## User Decisions

- Address all five original critique priorities in one coordinated programme:
  mutation preview, persistent status, storage-authority clarity,
  accessibility, and action-density reduction.
- Treat sync safety, the storage-authority mental model, and accessibility as
  co-equal goals.
- Keep **Library notes** and **Folder files** visibly separate.
- Preserve the current two-mode Library structure and strengthen labels,
  hierarchy, status, and recovery.
- Align Sync with ADR-059/060 rather than extending the legacy engine.
- Use text-only design review; no browser visual companion.

## Job and Audience

The primary user is a local-first researcher or builder working in a terminal.
They need to know where note truth lives, edit without losing data, distinguish
a one-time import from a lasting relationship, understand what a manual sync
will change, and separate ordinary file editing from Git publication.

The surface is **Operate** mode. Success means:

1. Current storage authority is understandable without documentation.
2. The user chooses one-time import or lasting sync before a source is read.
3. Root activation and manual reconciliation show a mutation-free review first.
4. Automatic reconciliation never resolves conflicts, deletions, or changes on
   a one-way root's non-authoritative side without explicit review.
5. Root state and the latest durable outcome survive screen navigation.
6. Everyday Folder Files editing stays fast while audit detail remains
   available.
7. Required controls and recovery paths remain keyboard-usable at supported
   compact terminal sizes.

## Existing Decisions and Implemented Foundations

- ADR-059 owns multi-root lasting-sync authority, managed folder memberships,
  the device-private root/binding/journal store, explicit conflict/deletion
  review, root coordinator leases, migration, and privacy.
- ADR-060 owns one-way direction semantics, representation/metadata
  preservation, binding uniqueness, guarded replacement, round-trip behavior,
  server claim fencing, and portable backup exclusions.
- ADR-021 and ADR-029 keep Folder Files disk-authoritative. Lasting Database
  Notes sync may reuse low-level containment and replacement primitives, but
  never File Notes tables, editor authority, recovery store, or orchestration.
- ADR-027 keeps the active Database Note draft/save coordinator separate from
  Textual presentation and File Notes authority.
- ADR-031 governs keybindings and footer honesty.
- ADR-035, ADR-038, and ADR-039 govern Session Git staging, commit, and push.
- TASK-15705/15706 already deliver local note folders, ownership-aware
  memberships, and the Library Notes folder navigator.
- TASK-16230 already delivers the immutable, mutation-free one-time import
  planner.
- TASK-16309 already delivers approved one-time import execution and durable
  receipts in the private `notes.sync_state` owner.
- The lasting-sync root registry, bindings, journal, coordinator, migration,
  and root UI are not yet implemented on current `dev`.
- The Library Notes adaptive design supports Database Notes down to 60x20.
- Textual `Select` and `Switch` were removed from this canvas after unreliable
  rendering. Choices use proven Button/Static/Input and focusable-scroll
  patterns.
- New Library collaborators belong in focused `UI/Library_Modules/` regions or
  controllers instead of adding another behavior cluster to the already-large
  `library_screen.py`.

## Scope

### In scope

- Rename the source strip to `Library notes | Folder files`.
- Add one pinned authority/status row per mode.
- Replace the legacy Notes `Sync` and `Import` entries with `Add from files…`.
- Present `Import once` and `Keep a folder synced` as distinct choices before
  reading a path.
- Wire the existing one-time import planner/executor into a production Textual
  review and receipt flow.
- Shape lasting-root setup, dry-run activation, manual reviewed reconciliation,
  root status, attention review, pause/resume, and disconnect surfaces around
  ADR-059/060.
- Define the safe transition from legacy sync metadata to paused candidates.
- Refine Folder Files action hierarchy and target-path labeling.
- Progressive-disclose Session Git implementation evidence while keeping all
  authorization-changing facts visible.
- Fix disabled/error contrast, text-explicit state, focus, overflow, and
  compact-layout defects.

### Out of scope

- Merge Library notes and Folder files into one authority or write path.
- Keep the legacy engine running beside the lasting-root registry.
- Add global automatic conflict winners, timestamp-based conflict resolution,
  or deletion propagation.
- Reuse legacy `sync_sessions`/`sync_conflicts` as lasting-sync history.
- Store device-local paths, hashes, recovery bytes, or root state in ordinary
  ChaChaNotes, server payloads, logs, backups, or portable exports.
- Add new schema or service decisions beyond ADR-059/060.
- Change Session Git trust, stage, commit, push, or uncertainty policy.
- Add a generic workflow/state-machine framework or new dependency.
- Add new screen-specific keyboard shortcuts.
- Implement every lasting-sync backend and UI slice in one PR.

## Information Architecture

The Library source strip remains the top-level authority switch:

```text
Library notes (selected) | Folder files
```

or:

```text
Library notes | Folder files (selected)
```

Immediately below it, the selected mode owns a pinned authority/status row.
Examples:

```text
Stored in Library · 2 synced folders · 1 needs attention · Last: 2m ago
```

```text
Edits ~/vault directly · Saved · Session Git: 3 changes
```

The row uses product language, not implementation vocabulary. `ChaChaNotes`,
private SQLite owner IDs, hashes, leases, and device claim tokens appear only
in bounded diagnostics where they are genuinely useful.

### Library Notes toolbar and tree

Replace the legacy adjacent `Sync` and `Import` actions with:

```text
Add from files…
```

When lasting roots exist, add `Manage sync folders` as a lower-frequency action.
Each root remains visible as a decorated top-level folder node with plain text
state:

```text
▾ ⇄ Work Notes  Up to date
▸ ⇄ Research Archive  2 need attention
▸ ⇄ Old Vault  Paused · Review migration
```

`Sync now`, `Pause/Resume`, `Review attention`, and root settings are contextual
actions for the selected root, not global toolbar competition.

## Add from Files Flow

### 1. Choose relationship

The first in-canvas step appears before any picker or scan:

- **Import once** — “Copy files or a folder into Library notes. Later changes
  to the originals are not tracked.”
- **Keep a folder synced** — “Create a lasting connection. Changes can continue
  between this folder and Library notes.”

The choice explains authority and persistence. It is not a cosmetic fork after
the user has already selected a path.

### 2A. Import once

Use the existing immutable planner and executor contracts from TASK-16230 and
TASK-16309. The UI sequence is:

```text
Choose -> Select source -> Checking -> Review -> Importing -> Receipt
```

Review groups New, Unchanged repeat, Changed repeat, Uncertain match,
Unsupported, and Failed items. It exposes the existing approved per-item
choices and separates content replacement from folder-membership addition.
No note, folder, receipt, or configuration mutation occurs before the user
approves the resolved plan.

The receipt is durable through the existing device-private import ledger.
Leaving the screen never turns a running import into an idle-looking state.

### 2B. Keep a folder synced

The setup sequence is:

```text
Explain -> Select folder -> Configure -> Checking -> Review -> Activating -> Receipt
```

Configure contains:

- display name;
- directory;
- local or server-backed Notes destination/profile;
- direction: `Bidirectional`, `Folder -> Library`, or `Library -> Folder`;
- capability-dependent advanced settings;
- primary action `Check folder`.

There is no conflict-policy selector and no global `Auto-sync every 5m` toggle.
Activation creates one root whose normal active state watches while Chatbook is
running and performs complete startup/manual reconciliation. `Pause` is the
explicit way to stop automatic work for that root.

Checking is strictly mutation-free. It performs path-safety, overlap,
capability, representation, metadata-preservation, binding-uniqueness, and
destination preflight plus a complete dry-run. Planning creates no root,
binding, membership, journal, recovery, config, file, or note mutation.

Review groups:

```text
Will import/update safely     12
Will publish safely            3
Needs attention                2
Skipped                        1
Managed folder placements     15
```

Every row names the predicted effect and reason. Full note contents are not
rendered in list rows; a user-requested comparison view may show the bounded
file/note diff for attention decisions.

Activation is available only after every setup ambiguity is resolved or
explicitly skipped. It admits required recovery capacity before destructive
work, creates the root/bindings/journal through the device-private owner, and
returns an honest durable Receipt. Partial cross-authority outcomes are never
described as atomic.

## Lasting Root Operation

### Root states

The authority row and root nodes derive state from the application-owned
lasting-sync coordinator and private ledger, not mounted widgets:

- Checking / Activating / Reconciling.
- Up to date / Changes applied.
- Needs attention.
- Paused / Offline.
- Partial / Recovery required.
- Failed / Unsupported / Capability lost.
- Passive in this process / Owned by another process.

Every non-ready state names a next action. Navigation preserves running state,
latest outcome, and bounded current activity.

### Direction matrix

Direction controls automatic one-sided propagation. It never grants permission
to erase an unexpected change on the non-authoritative side.

| Observation | Folder -> Library | Library -> Folder | Bidirectional |
| --- | --- | --- | --- |
| New/changed file only | Apply to Library. | Needs attention. | Apply to Library. |
| New/changed note only | Needs attention. | Apply to folder. | Apply to folder. |
| Both sides changed | Needs attention. | Needs attention. | Needs attention. |
| File or note missing | Needs attention; never infer deletion from an offline root. | Needs attention; never infer deletion from an offline root. | Needs attention; never propagate automatically. |
| Unsafe representation, metadata, identity, overlap, or capability | Skip or block with an actionable reason. | Skip or block with an actionable reason. | Skip or block with an actionable reason. |

The old `newer_wins`, `disk_wins`, and `db_wins` values have no counterpart in
the new model. Timestamp recency is evidence, not intent.

### Manual Sync now

Manual reconciliation is a reviewed flow:

```text
Checking -> Review -> Applying -> Receipt
```

`Check changes` performs a complete mutation-free reconciliation. Review shows
safe one-sided operations, attention items, skips, and managed-placement
effects. The user may apply the safe set, open attention review, or leave with
no mutation. Applying performs a fresh guarded comparison before admission and
uses the root's durable journal. If reviewed observations changed, it returns
`Review stale - Check again` before applying the stale operation.

This reviewed manual path is intentionally more inspectable than event-driven
automatic reconciliation. It does not change the root's future direction or
attention policy.

### Automatic reconciliation

Filesystem notifications are scheduling hints. Active roots perform debounced
authoritative reconciliation through versions, canonical paths, identities,
and hashes. Only direction-authorized one-sided operations may proceed.
Conflicts, deletions, unexpected non-authoritative changes, representation
drift, capability loss, root replacement, and large deletion bursts pause into
Needs attention.

Automatic work records a durable journal/receipt, remains visible across
navigation, and never repeats an unresolved operation every interval. A root in
Needs attention, Offline, Partial, or Failed state does not silently resume
destructive work.

### Attention review

Conflict rows offer the accepted explicit actions:

- `Keep file`.
- `Keep note`.
- `Keep both`.

Deletion rows offer bounded choices appropriate to the missing side:

- restore the missing side;
- explicitly delete/archive the counterpart; or
- disconnect that item.

The review shows which root owns the binding, the affected relative path and
note title, both changed sides, the effect of each choice, and whether extra
manual folder placements remain. Choosing one side resolves that occurrence
only; it does not change root direction.

### Pause, resume, retarget, and disconnect

- Pause closes new automatic admission but lets admitted work reach a durable
  terminal or recovery state.
- Resume performs a complete Check and requires review when observations or
  capabilities changed.
- Retarget pauses the root, dry-runs the new directory, and never treats absence
  in the new root as deletion.
- Disconnect never deletes notes or files. It converts managed organization to
  manual placement by default or removes only that root's managed memberships.

## Durable Ownership, Privacy, and Recovery

Lasting roots extend the existing private `notes.sync_state` owner established
for import receipts. They do not reuse legacy note-row sync metadata or
`sync_sessions`/`sync_conflicts` as the source of truth.

The private owner stores the root registry, bindings, representation profiles,
hashes, versions, cursor state, durable operation journal, recovery admission,
receipts, and bounded recovery content required by ADR-059/060. Public status
models expose opaque IDs and sanitized reason codes only.

One cross-process coordinator lease owns watcher and mutation authority for a
root. Passive processes may display durable state but cannot reconcile or
write. Root overlap with another lasting root, Folder Files root, or
application-private path fails closed.

No transaction spans disk, local SQLite, and a server. Each operation records
intent, verifies preconditions, performs guarded authority mutations, verifies
outcomes, updates bindings, and completes last. Interruption resumes only
against matching observations or produces Needs attention with explicit
recovery. The interface never promises rollback or all-or-nothing apply.

External filesystem writers remain outside Chatbook's cross-process lease. The
implementation must minimize and detect supported races through descriptor
identity, hashes, representation profiles, and postflight verification, but it
must not claim a filesystem compare-and-swap the platform cannot provide.

## Legacy Transition

The transition is fail-closed and one-way:

1. Do not add preview/apply, auto-sync approval, conflict winners, or new
   persistence to the legacy engine.
2. Implement the lasting-root registry/coordinator/journal before enabling any
   new root mutation.
3. Migrate legacy per-note metadata and configured root evidence into one or
   more **paused candidate roots** without mutating files or notes.
4. Show `Review migration` with the proposed root, bindings, direction, skips,
   and unsupported conditions.
5. Require a complete current dry-run and explicit activation.
6. Retire legacy mutation entry points in the same release that activates the
   replacement. Never permit both owners to run.
7. Preserve legacy history as read-only historical evidence; do not present it
   as lasting-root journal state.

If the replacement backend is unavailable, the product does not ship a
half-wired `Keep a folder synced` action. The choice remains visibly
`Unavailable - lasting sync setup is not installed` with the nearest valid next
step; it never falls back to legacy mutation.

## Folder Files Refinement

Folder Files retains its navigator/editor topology and disk authority. The
purpose line becomes the pinned authority row rather than clipped prose.

The path field gets a persistent one-row label:

```text
Target path · New / Move / Save copy
```

Default editor actions are:

```text
New · Move · Delete · More file actions
```

Contextual actions outrank defaults:

- `Restore` appears immediately after deletion.
- `Reload from disk` becomes primary on external-change conflict.
- `Save copy` appears when Dirty, Conflict, or Error makes it applicable.

`More file actions` toggles one inline secondary row containing only currently
applicable Protect/Unprotect, Reload, Save copy, and Refresh actions. The toggle
label and disclosure glyph expose its state. No modal, generic menu framework,
or new shortcut is added.

The action-status line remains visible and names outcomes and recovery.

## Session Git Refinement

Session Git behavior and safety contracts remain unchanged. Commit and push
reviews lead with four decision-fact groups: What, Where, Impact, and Recovery.

The Commit review keeps visible:

- canonical repository path, local branch, and author identity;
- exact commit subject/body and candidate commit ID when available;
- included session-note paths/count and unrelated-change promise;
- whether trusted repository hooks or filters can run;
- local-only/no-network effect and Back/cancel boundary.

The Push authorization/final review keeps visible:

- exact commit subject/ID and expected parent-to-candidate transition;
- local branch, full destination ref, sanitized endpoint identity, and secure
  transport/authentication method;
- included session-note provenance and the fact that required Git objects—not
  independently selected note rows—are published;
- credential-helper or SSH-agent contact, local pre-push-hook bypass, and the
  possibility of remote hooks, CI, mirrors, reflogs, or policy effects;
- exact-lease protection in user language, later edits remaining local, and
  uncertain-result recovery.

`Show technical details` contains only lower-frequency implementation evidence:
repository/worktree filesystem identity tuples, duplicate raw object IDs,
index/ownership signatures, policy fingerprints, provenance sequence numbers,
and internal lease/status identifiers. Endpoint Details remains its existing
selectable surface. Transport, authentication, hook effects, destination ref,
commit ID, and author identity must not be hidden because they can change an
authorization decision.

## Visual and Accessibility Contract

- Preserve the Neon Workbench system; this is a refinement, not a replacement
  visual world.
- Use semantic tokens only. Bright color is earned by focus, running, success,
  warning, attention, blocked, and error state.
- Use `$ds-status-error-readable` for error text.
- App-level disabled styling for the touched Notes/File/Git surfaces must
  neutralize compounded dimming and meet the measured 3:1 minimum while
  remaining visibly disabled.
- Every selected, running, paused, offline, attention, passive, disabled, and
  authority state uses literal text in addition to color or glyph.
- Focus changes never alter geometry.
- Pinned authority, phase, and status rows do not scroll away.
- Scrollable review/activity surfaces show the existing overflow/fold hint.
- User-controlled paths and titles render with markup disabled or escaped.
- Raw exception strings, absolute paths, hashes, note contents, recovery bytes,
  and credentials do not enter ordinary logs or public diagnostics.

## Responsive and Keyboard Contract

- Notes flows inherit the existing compact Library stage and Back hierarchy;
  no new breakpoint is introduced.
- At 60x20, chooser, Configure, Review, Attention, Receipt, and every recovery
  action are reachable without horizontal scrolling.
- Large review lists live in keyboard-scrollable owners with pinned phase and
  primary-action rows.
- At Folder Files' existing narrow breakpoint, navigator and editor remain
  mutually exclusive and focus always moves to a mounted visible target.
- Session Git retains its 40x20 scroll and phase-safe Escape behavior.
- Footer hints advertise only actions implemented in the current phase.
- No bindings use terminal-convention keys or shadow Ctrl+P, Ctrl+Q, F1, or F6.

## State and Range Coverage

Test at minimum:

- No folders/notes, one file, nested root, and current import bounds.
- Local and server-backed destination capability states.
- Empty, missing, replaced, overlapping, symlinked, hard-linked, unreadable,
  unsupported-encoding, mixed-newline, and metadata-unsafe roots/files.
- No changes, safe one-sided changes, both-side changes, non-authoritative-side
  changes, missing sides, and large deletion bursts in every direction.
- Root setup, activation, stale review, partial journal, interruption/restart,
  recovery-capacity block, passive owner, pause/resume, retarget, disconnect,
  and capability/claim loss.
- Legacy metadata absent, valid candidate, ambiguous roots, unsafe paths,
  unsupported bindings, and migration cancellation.
- One-time import new/repeat/uncertain/unsupported/failed classifications,
  partial execution, retry, and receipt reopen.
- Long Unicode paths/titles, viewport-wide paths, and large roots with bounded
  rendering and no retained contents in presentation models.
- Folder Files Dirty/Saving/Saved/Conflict/Error/Deleted states.
- Session Git commit and push authorization/recovery states.

## Error and Recovery Copy

- Boundary validation: `Folder unavailable - Choose another folder`.
- Review drift: `Review stale - Check again`.
- Coordinator held elsewhere: `Passive here - Another Chatbook process owns this folder`.
- Missing root: `Offline - Reconnect or retarget`; never infer mass deletion.
- Non-authoritative change: `Needs attention - Review both sides`.
- Representation/metadata issue: `Write blocked - Keep read-only or choose another file`.
- Recovery capacity: `Sync paused - Free recovery space or disconnect items`.
- Server claim/capability loss: `Sync paused - Reauthorize or review takeover`.
- Unknown failure: safe generic reason plus `Retry check`; raw context remains
  in private/sanitized diagnostics according to ADR-059/060.
- Toasts may reinforce failures but are never their sole carrier.

## Component and Ownership Boundaries

- Existing `note_import_*` planner/executor/receipt modules remain the one-time
  import contract. The UI adapts their immutable states; it does not duplicate
  parsing, planning, approval, or retry logic.
- New lasting-sync domain, owner, coordinator, journal, and service modules live
  under `tldw_chatbook/Notes/` and expose typed immutable public states.
- New screen behavior lives in focused `tldw_chatbook/UI/Library_Modules/`
  controllers with named late-binding dependencies. `LibraryScreen` coordinates
  navigation and services but does not absorb the root state machine.
- Focused Library region widgets own their DOM and consume complete snapshots.
  They do not own persistence or retain removable child instances.
- Folder Files, its replica/recovery store, and Session Git never consume
  Database Notes sync-root models.
- Shared low-level containment/atomic-write adapters expose primitive
  capabilities only; neither Notes feature calls the other's high-level
  service.

## Testing and Evidence

### Pure/service evidence

- Mutation-free setup/manual plans for every direction and attention class.
- Binding/root/file identity uniqueness and overlap rejection.
- Representation and metadata profile round trips.
- Durable journal intent/apply/verify/complete ordering with crash injection.
- Root coordinator single-owner/passive-process behavior and bounded shutdown.
- Missing-root and deletion-burst fail-closed behavior.
- Legacy migration creates paused candidates with zero note/file mutation.
- Legacy and lasting mutation owners cannot both become active.
- Public models/logs remain free of content, absolute paths, hashes, raw
  exceptions, and recovery bytes.
- Existing one-time import, folder, File Notes, and Session Git suites remain
  green.

### Mounted Textual evidence

- Text-explicit source labels and pinned authority rows.
- Add from files chooser before source access.
- Import once and lasting setup phase/focus flows.
- Root tree status/context actions and persistent state across navigation.
- Manual Check/Review/Apply/Receipt and stale-plan recovery.
- Attention resolution, pause/resume, retarget, disconnect, and passive states.
- Folder Files contextual/secondary action visibility and focus restoration.
- Session Git decision facts plus technical-detail disclosure.
- Error/disabled classes, recovery actions, fold hints, and footer honesty.
- Production-shaped geometry at 60x20, 80x24, 120x35, and 160x45 with the
  exact application stylesheet stack.

### Live evidence

- Scratch-profile local root activation and restart recovery.
- A real temporary folder proving Check performs no mutation and Apply matches
  the reviewed safe set.
- External edit, missing root, conflict, deletion, and passive second-process
  demonstrations.
- Current-HEAD wide/compact captures of chooser, setup Review, root status,
  Attention, Receipt, Folder Files, commit review, and push review.
- Measured contrast for ordinary, error, disabled, selected, and focused text.

## Delivery Decomposition and Concurrency

This programme needs separate atomic plans. It must not become one enormous PR.

1. **Mode labels, authority rows, and accessibility:** behavior-preserving
   source labels/status grammar, semantic state styling, and compact evidence.
2. **Folder Files and Session Git refinement:** target-path label, contextual
   action hierarchy, exact decision/detail disclosure, and focused regressions.
3. **One-time import UI:** Add from files chooser plus production UI over the
   already-implemented planner/executor/receipt contracts.
4. **Lasting-sync foundation:** private root registry, bindings,
   representation profiles, coordinator, durable journal/recovery, and service
   contracts from the 2026-08-12 design and ADR-059/060.
5. **Lasting-sync UI and legacy cutover:** setup dry-run, root states, manual
   review, attention, pause/resume/retarget/disconnect, paused legacy migration,
   and atomic retirement of legacy mutation entry points.
6. **Polish and live evidence:** production-shaped captures, contrast checks,
   critique rerun, docs, and regression closeout.

Tracks 1, 2, 3, and 4 can be developed concurrently with separate file
ownership. Track 5 depends on the lasting-sync service contract from track 4;
it may build against fakes earlier but cannot ship an active root or legacy
cutover first. Track 6 closes all tracks together.

## ADR Check

ADR required: no new ADR.

ADR paths:
`backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`,
`backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: ADR-059/060 already decide storage ownership, sync/conflict/deletion
policy, privacy, recovery, coordinator boundaries, migration, interoperability,
and lasting application structure. This revision conforms the UI programme to
those accepted decisions. Folder Files and Session Git changes are
presentation-only refinements governed by their existing ADRs.

## Known Risks and Resolutions

### Competing legacy and lasting writers

Risk: a polished legacy engine remains active while the new registry activates.

Resolution: no legacy feature extension; paused candidate migration and legacy
mutation retirement ship in the same cutover tranche. Fail closed if the new
owner is unavailable.

### UI work outruns the lasting-sync contract

Risk: the screen invents root, plan, journal, or attention semantics before the
service owns them.

Resolution: immutable service models first, UI adapters second. Pre-foundation
UI may use fakes for layout tests but cannot ship enabled mutation.

### Duplicate implementation of one-time import

Risk: Add from files rebuilds planning/execution logic already delivered.

Resolution: reuse TASK-16230/16309 contracts directly and keep UI code limited
to phase state, rendering, focus, and service invocation.

### False atomicity

Risk: users infer one transaction across disk, local Notes, and server Notes.

Resolution: journaled intent/verification, explicit Partial/Needs attention,
and no rollback language.

### External filesystem race

Risk: a non-cooperative writer changes a file outside Chatbook's root lease.

Resolution: descriptor identity, hashes, representation profiles, pre/postflight
verification, stale review, and honest residual-risk documentation. Never claim
cross-process filesystem compare-and-swap.

### Dense terminal review

Risk: detailed safety evidence overwhelms the next action.

Resolution: decision summary first, grouped expandable rows, bounded metadata,
no contents in list rows, pinned actions, and visible fold hints.

### Scope collision with Folder Files

Risk: shared visual language becomes shared storage behavior.

Resolution: authority rows share presentation grammar only. ADR-021/029 Folder
Files ownership and all Session Git contracts stay separate and unchanged.
