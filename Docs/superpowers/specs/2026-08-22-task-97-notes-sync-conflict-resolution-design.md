# TASK-97 — Inline Database Notes Sync Conflict Resolution

**Status:** Approved conceptual design; implementation not started

**Date:** 2026-08-22

**Task:** `TASK-97`

**Applies to:** lasting local Database Notes sync on the Library screen

## Decision summary

TASK-97 enables content-conflict resolution inside the existing mutation-free
review canvas. It does not revive the retired ASK engine, add an interrupting
modal, or create a second sync service.

A user may inspect a bounded Note-to-File comparison, stage **Keep file**,
**Keep note**, **Keep both**, or **Skip for now**, and then apply the reviewed
subset. Staging never mutates. Apply re-observes the root and requires the exact
review token before it sends any selected item through the existing durable
runtime and executor.

Unresolved conflicts remain **Needs attention**. Deletion review, pause,
managed-placement, capability, and activation blockers stay unavailable in this
slice.

## Context

The original TASK-97 described a per-conflict modal driven by the legacy sync
engine's ASK strategy and reachable from a legacy activity log. That engine is no
longer the active owner. The current lasting-sync architecture already provides:

- pure reconciliation that classifies both-sides-changed bindings as attention;
- a paged inline Library review canvas;
- an observation token and fresh-authority check before apply;
- device-local operation and recovery journals;
- guarded note and filesystem mutations;
- typed one-occurrence direction overrides; and
- root status, recovery, and receipt projections.

The current UI deliberately renders conflict and deletion choices disabled
because no reviewed execution seam existed at cutover. TASK-97 fills only the
content-conflict seam.

## Goals

- Resolve a content conflict without selecting an automatic winner.
- Keep selection mutation-free until the user presses **Apply reviewed**.
- Allow safe actions and explicitly resolved conflicts to proceed while skipped
  conflicts remain paused.
- Preserve both versions durably for **Keep both**.
- Retain stale-review, direction, lease, recovery-capacity, crash-recovery, and
  privacy protections.
- Record the explicit choice in durable resolution history.
- Offer exact per-item Undo while recovery authority remains valid.
- Keep the experience keyboard-usable and contained at the supported 60x20
  terminal size.

## Non-goals

- No modal conflict prompt.
- No automatic newest-wins or global conflict preference.
- No bulk-resolution control.
- No deletion resolution, grouped deletion execution, pause resolution,
  managed-placement execution, retarget, or disconnect work.
- No change to future root direction after a one-occurrence choice.
- No server-note, File Notes, watcher, scheduler, or legacy sync ownership change.
- No all-or-nothing claim across a batch of independently journaled items.
- No new dependency or parallel persistence layer.

## Existing decisions

No new ADR is required.

- `backlog/decisions/055-library-destructive-action-reversibility-rule.md`
  governs retained recovery and refusal to overwrite later edits.
- `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
  defines Keep file, Keep note, Keep both, and the manual Conflict copies
  placement.
- `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
  permits an explicit one-occurrence choice without changing root direction.

## Architecture

The existing owner chain remains intact:

```text
pure reconciliation
  -> inline Library review/controller
  -> reviewed runtime admission
  -> fresh authority and token equality
  -> existing durable executor/journal
  -> sanitized receipt/history projection
```

The reconciler continues to emit `ReconciliationAttention(CONFLICT, ...)` and
never emits a winning action. The controller stores staged choices privately.
The runtime validates and translates those choices only during manual reviewed
apply. The executor remains the sole mutation and recovery owner.

Automatic reconciliation retains its current global attention blocker. Manual
reviewed apply gets one narrow admission rule: unresolved content-conflict rows
may coexist with selected safe work, but every other attention or capability
class remains blocking.

The runtime owns one in-process asynchronous mutation lock per root. Automatic
execution, manual reviewed apply, startup recovery for that root, and Undo all
use the same lock. After acquiring it, a caller must reacquire root ownership
and repeat every freshness, token, plan, authority, and recovery check before it
builds a request or mutates anything. The existing planning lease remains a
lifecycle/cancellation admission mechanism; it is not serialization. A lock
entry may be discarded only after the root has no admitted or waiting mutation
task. Executor operation-ID locks remain durable replay guards, not substitutes
for root serialization.

Only bound content rows with reason `both_sides_changed` or
`out_of_direction_change` are eligible for choices, and only when both note and
file content exist and the row has no managed-placement effect. All four choices
are available for either eligible reason; the one-way case uses an occurrence-
only direction override when its selected write opposes the configured
direction. Every other `CONFLICT` reason—including duplicate authority,
out-of-direction create/move/representation, ambiguous identity, and implied
filesystem moves—retains its existing blocking copy and exposes no resolution
choice.

## Typed contracts

### Conflict choice

A string enum has exactly four values:

- `keep_file`
- `keep_note`
- `keep_both`
- `skip`

A staged selection contains only a validated binding ID and the typed choice.
The enclosing apply call already carries the root ID and observation token.
Duplicate binding selections and unknown choices are rejected before mutation.

The controller keys its private selection map by the current observation token
and binding ID. A new token replaces the map; selections never migrate between
reviews by binding ID alone.

### Conflict comparison

The runtime returns a frozen, private-representation comparison containing:

- binding ID;
- bounded note title;
- normalized root-relative path;
- note version;
- optional validated note update time, rendered as unavailable when absent;
- exact file modified time from the reviewed filesystem state;
- bounded line and character counts;
- a Note-to-File unified diff; and
- explicit input/output elision flags.

It does not expose an absolute path, content digest, file identity, root path,
recovery bytes, or raw exception text. Full raw note/file snapshots never enter
controller state.

The Database Notes comparison helper is independent of File Notes authority but
uses the same established limits:

- 200,000 input characters per side;
- 10,000 input lines per side;
- 120,000 output characters; and
- 2,000 output lines.

When an input exceeds a limit, the UI reports the exact size and that the diff
was omitted. When output reaches a limit, it appends one elision marker. Diff
generation uses the Python standard library only.

### Apply result

Manual apply returns a typed bounded result instead of making the controller
infer state from tuple length. It reports:

- executor results for attempted items;
- safe-action count completed;
- conflict count resolved;
- skipped/unresolved conflict count;
- whether attention remains;
- whether an operation is partial or needs recovery; and
- the fresh post-apply plan only when every attempted operation reached a
  durable terminal state.

No public result contains content, paths, hashes, or recovery data.

### Resolution history

Completed conflict choices use durable operation kinds:

- `resolve_keep_file`
- `resolve_keep_note`
- `resolve_keep_both`

The underlying executor action remains validated recovery metadata. Durable
history therefore retains the user's choice after temporary recovery payloads
expire.

The durable history projection contains only operation ID, typed choice,
bounded state, completion/update time, and Undo availability/reason. The
interactive adapter may decorate a row from fresh authority with the current
bounded note title and normalized root-relative path; those labels are never
stored in the operation journal, logged, or returned in representations. When
fresh authority cannot identify the item, the row uses the first eight
characters of the opaque operation ID. No history form exposes binding IDs,
content, hashes, absolute paths, or exception text. History pages contain at
most 100 rows.

## Review and comparison flow

1. **Check** produces the existing immutable reconciliation plan and token.
2. The controller projects conflict rows but does not copy private authority.
3. **View comparison** calls the runtime with root ID, token, and binding ID.
4. Under the root's planning lease, the runtime performs a fresh observation and
   recomputes the plan. Comparison is read-only and does not acquire the root
   mutation lock.
5. The root, token, complete plan, and exact conflict binding must equal the
   reviewed values.
6. The adapter builds the bounded comparison while its private observation
   bundle is alive, then releases that bundle before returning.
7. The controller publishes the comparison only if its review generation,
   token, expanded binding, and mounted canvas still match.

If freshness fails, the controller clears choices/comparison, marks the review
stale, and presents **Check again**. Viewing a comparison never creates an
operation or changes a note, file, binding, folder, membership, or status row.

Only one comparison payload is retained in controller memory. Collapsing it,
changing pages, replacing the review, navigating Back, or remounting releases
it.

## Selection and manual apply

Staging a choice changes only controller memory and the current presentation.
It updates the existing status line to `Choice staged. No changes yet.` and does
not issue a notification.

**Apply reviewed** sends all existing safe-action IDs plus all staged conflict
choices. Skip is transmitted as reviewed intent so the result can count remaining
attention, but it creates no operation and no history row.

The runtime first acquires the root mutation lock. While holding it and before
mutation, it:

1. reacquires authoritative root ownership;
2. requires an active root and exact reviewed token;
3. performs a fresh observation and exact plan equality check;
4. rejects duplicate, unknown, non-conflict, or cross-root binding choices;
5. rejects deletion groups, deletion attention, pause attention, managed
   placement, skips caused by root/capability conditions, or activation review;
6. builds every selected execution request while private observation authority
   is still retained; and
7. admits required recovery separately for each item before that item's first
   mutation.

Automatic execution and recovery use the same lock and repeat their own fresh
authority checks after acquisition. This prevents reviewed apply, automatic
work, recovery, and Undo from deriving contradictory requests concurrently even
when their durable operation IDs differ.

Unselected and skipped content conflicts do not block safe actions or selected
content resolutions. Automatic reconciliation is unchanged and still blocks on
any attention.

Execution order is deterministic: existing safe actions retain their stable
plan order, followed by selected conflicts ordered by binding ID. Each item is
independently durable. A non-terminal result stops later work and produces an
honest partial receipt; already completed items are not described as rolled
back.

When all attempted items are terminal, the runtime performs a fresh
reconciliation. Remaining conflicts keep the root in **Needs attention**.

## Choice semantics

### Keep file

- Translate to the existing guarded update-note execution.
- The reviewed file supplies desired logical text.
- The original note title/body/version and binding baseline are admitted as
  recovery authority.
- Add a typed direction override only when the configured direction would
  otherwise disallow this one occurrence.
- Record `resolve_keep_file` durably.

### Keep note

- Translate to the existing guarded update-file execution.
- Preserve the reviewed file's byte representation and supported metadata in
  recovery.
- Use the note body as desired logical text.
- Add a typed direction override only when needed for this occurrence.
- Unsupported Windows or other non-write-capable observations remain blocked.
- Record `resolve_keep_note` durably.

### Keep both

Keep both leaves the reviewed file unchanged, preserves the original Database
Note content in a new unbound manual note, then makes the reviewed file the
content of the original bound note. The bound note retains its identity.

Stable identities are domain-separated SHA-256 values:

- the top-level Conflict copies folder is stable per local note scope;
- its child folder is stable per sync root; and
- the conflict-copy note is stable per root, binding, and observation token.

The folder identities do not depend on the observation token. The child display
name comes from the current logical root folder name, not from a private
filesystem path. Existing active manual folders at the normalized path are
reused. If absent, the caller-owned deterministic ID is used to create them. An
ID/path mismatch fails closed. Existing manual folders are never automatically
renamed.

The sync authority exposes three narrow idempotent seams over the existing
local folder, note, and membership repositories:

- `create_or_verify_manual_folder(request)` performs at most one folder create
  and handles the parent and child in separate invocations;
- `create_or_verify_conflict_note(request)` performs at most one note create;
  and
- `create_or_verify_manual_placement(request)` performs at most one manual
  placement create.

The requests carry only the caller-derived object ID, expected parent/owner,
normalized manual name or bounded title/body, and the prior step's verified
actual ID. Each call returns the actual verified identity and optimistic
version. They do not accept filesystem paths and cannot create sync-managed
placements. Read-only repository observations jointly verify the copy after the
three mutation seams complete.

For each folder level its invocation resolves the normalized active manual path
first. If that path exists, it reuses the existing folder and returns its actual
ID. If absent, it creates the folder with the caller-owned deterministic ID. A
deterministic ID already attached to a different active path, a non-manual
folder, a different owner, or a deleted object fails closed. Concurrent create
losers reread and verify the winning row rather than choosing another identity.
For the copy, existing deterministic note and manual placement calls reuse rows
only when owner, parent, title, body, deletion state, and placement all match;
any mismatch fails closed. The authority uses repository optimistic versions
and returns fresh versions after each create-or-verify step. These are
idempotent repository operations, not a cross-database transaction. Because
each invocation performs at most one external effect, the executor persists the
corresponding substage before invoking the next seam.

After recovery admission the executor performs an idempotent, journaled
sequence:

1. locate or create the active manual Conflict copies parent;
2. locate or create the root child;
3. create the deterministic unbound note from the original note title/body;
4. create its manual placement in the child;
5. verify the exact note and placement;
6. replace the bound note from the reviewed file;
7. update the binding baseline; and
8. verify the conflict copy, bound note, file, and binding before completion.

The folder/note preparation is not described as one cross-repository
transaction. Every step is idempotent and re-observed. A crash may leave an
empty folder or an additional preserved copy, but never removes either user's
version. An existing deterministic note with different identity, content, or
placement is Needs attention and is never overwritten.

The generic operation table is not widened. While the operation state remains
`recovery_admitted`, private recovery metadata carries an exact
`conflict_substage` enum and is advanced by compare-and-set with the operation
and recovery IDs:

| Completed durable boundary | Operation state | `conflict_substage` |
| --- | --- | --- |
| recovery admitted | `recovery_admitted` | `recovery_admitted` |
| both folders verified | `recovery_admitted` | `folders_established` |
| conflict-copy note verified | `recovery_admitted` | `copy_created` |
| manual placement verified | `recovery_admitted` | `placement_created` |
| copy note and placement jointly reverified | `recovery_admitted` | `copy_verified` |
| bound note updated and reobserved | `first_authority_applied` | `bound_note_updated` |
| reviewed file reobserved unchanged | `second_authority_applied` | `file_reverified` |
| binding baseline committed | `binding_updated` | `binding_updated` |
| every authority reverified | `verified` | `verified` |

Each external side effect is followed by its checkpoint. A crash between them
replays the same create-or-verify step, then advances the checkpoint; it never
guesses that an effect occurred. Startup reconstruction rejects unknown,
skipped, or regressing substages and any collision with the checkpointed
identities. Completion follows `verified` as it does for existing operations.

The durable operation kind is `resolve_keep_both`; validated recovery metadata
records the underlying update-note action and the exact conflict-copy identities.

### Skip for now

Skip performs no mutation, creates no operation, and creates no resolution
history entry. The binding remains attention. If other reviewed work succeeds,
the fresh review presents it again under a new token.

## Recovery and Undo

Recovery capacity is admitted before each selected item's first mutation. Every
admitted mutation is cancellation-shielded through its current durable
sub-step, then cancellation is propagated after the journal is coherent.

Conflict-resolution and linked Undo recovery expires 30 days after admission,
matching ADR-059's normal recovery-retention contract. One named duration
constant serves these operation kinds; this task does not change any shorter
retention for unrelated operation kinds. History remains after recovery expiry,
but Undo then reports `Undo expired` and cannot mutate.

Startup recovery reconstructs conflict requests from the durable operation kind
and validated private metadata. It never treats a resolution operation as an
automatic action.

Per-item Undo is available only for a completed conflict resolution with
unexpired exact recovery. Undo acquires the same root mutation lock as apply,
then reacquires the root lease and validates the current note, file, binding,
and conflict copy before mutation.

Before its first mutation, Undo admits a separate durable `undo_resolution`
operation. Its ID is the domain-separated SHA-256 of the canonical tuple
`("undo_resolution_v1", root_id, source_operation_id)`; private recovery
metadata repeats and validates the source operation ID. The linked operation
follows the existing generic state machine:

Undo admission is self-contained and capacity-accounted. While the source
recovery is still exact and unexpired, the linked recovery copies every byte and
typed fact needed to finish or safely refuse the Undo: the pre-resolution
note/file authority, the post-resolution authority used for optimistic checks
and rollback, original binding identity/path/serialization/digests, source
choice and operation ID, and any conflict-copy identity/content/placement
checks. Once admission commits, startup reconstruction reads only the linked
Undo recovery; it never depends on the source recovery payload, which may expire
or be evicted independently.

| Undo boundary | Durable state/checkpoint |
| --- | --- |
| exact pre-Undo authority admitted | `recovery_admitted` |
| changed note or file restored and verified | `first_authority_applied` |
| unchanged opposite authority reverified | `second_authority_applied` |
| original binding identity/path/serialization/digests restored with fresh note version | `binding_updated` |
| unchanged Keep-both copy soft-deleted, or no copy required | private `undo_substage=copy_cleanup_complete` while `binding_updated` |
| all resulting authority verified | `verified` |
| source operation marked Undone | linked Undo `completed` |

Startup recovery resumes the linked Undo from these checkpoints. Each step is
idempotent and authority-checked; a crash can therefore leave a recoverable
partial Undo but never an unjournaled restore. Only after verification does one
device-state transaction complete the Undo operation and compare-and-set the
source completed operation's empty reason to `undo_completed`. A zero-row source
CAS makes the Undo operation Needs attention rather than rewriting history.

Undo restores the pre-resolution note/file authority and the original binding
identity, relative path, serialization, and content digests. Restoring a note is
an optimistic write and produces a fresh note version, so the active binding is
atomically updated to that fresh version rather than its historical version.
The binding remains `active`; **Needs attention** is projected from the freshly
divergent reconciliation plan, so the same binding can be reviewed and resolved
again. For Keep both, Undo restores and verifies the original bound note before
updating the binding, then soft-deletes only the unchanged operation-owned
conflict-copy note.

If final cleanup fails, both visible versions remain and the linked Undo is
Needs attention. Empty manual folders may remain. Undo never deletes a
shared/manual folder. A changed bound authority or edited conflict copy refuses
Undo with bounded `changed_since_resolution` and makes no mutation.

## Inline Library behavior

Conflict rows are collapsed by default and show bounded note title, wrapped
root-relative path, current selection, and the four choices. Reason copy is
exactly `Both file and note changed` for `both_sides_changed` and `This change
is outside the root direction` for `out_of_direction_change`.

Choice effect copy is explicit:

- **Keep file** — update the Library note.
- **Keep note** — replace the folder file.
- **Keep both** — preserve an unbound note copy, then update the bound note.
- **Skip for now** — make no changes.

Expanded comparison appears inside the same row in a read-only, horizontally
scrollable TextArea with markup disabled. A **Return to choices** control
collapses it and restores focus to the originating View button.

Async comparison completion moves focus to the diff only when the originating
View button is still focused and all request provenance remains current. It
never steals focus after the user moves elsewhere.

The selected button gains a checkmark and a separate `Selected: ...` label.
Color is not the only signal. Enter and Space use standard Button activation.
Staging updates the current widgets in place rather than recomposing the review,
preserving scroll and focus.

Paging preserves selections. New checks, stale results, Back, root changes, and
controller remounts clear selections and expanded content.

The pinned Apply button is enabled when there is at least one safe action or one
mutating conflict selection and no non-conflict blocker. Skip alone does not
enable it. Disabled tooltips name the exact blocker.

After successful subset apply:

- each completed destructive resolution adds an in-place retained receipt at
  the action point with bounded item label, choice, **Undo**, and **Dismiss**;
- the receipt remains until that item is undone, explicitly dismissed, or
  superseded by a newer resolution of the same item;
- within the current app runtime, navigation or controller remount reconstructs
  undismissed receipts from runtime-owned private per-root sets of completed and
  dismissed operation IDs;
- Dismiss changes only those in-memory opaque-ID sets and never deletes
  operation history or recovery authority; dismissed receipts cannot reappear
  during that runtime;
- a process restart starts with no at-action receipts, dismissed or otherwise;
  the durable Resolution history remains the restart-spanning recovery surface;
- if conflicts remain, receipts stay above page 1 of the fresh review, focus
  moves to its first conflict, and the existing status line reports applied and
  remaining counts once;
- if no attention remains, the normal receipt phase is shown; and
- if an admitted operation is non-terminal, the existing root recovery path is
  shown instead of a fresh review.

Deletion and other unsupported attention rows retain disabled controls and the
existing unavailable explanation. Activation reviews continue to reject all
attention.

## Resolution history UI

Each lasting-sync root gains a bounded **Resolution history** action in the
existing retained canvas. It supplements rather than replaces the at-action
Undo/Dismiss receipts and is not a modal. The action is enabled when the runtime
reports at least one durable conflict-resolution operation.

History shows newest first, 100 rows per page. Each row shows its fresh bounded
item label (or short operation-ID fallback), choice, timestamp,
terminal/recovery status, and one of:

- **Undo** — exact recovery is currently valid;
- `Undo expired` — recovery elapsed;
- `Changed since resolution` — current authority no longer matches; or
- `Undone` — the one-shot restore completed.

Undo is per item. There is no batch Undo. Skip never appears.

## Error and status behavior

- Stale review: no mutation; clear ephemeral state; show **Check again**.
- Unknown/duplicate/cross-root selection: no mutation; bounded invalid-review
  status.
- Recovery-capacity refusal: no mutation; root remains Needs attention.
- Unsupported filesystem write: no mutation; retain capability reason.
- Deterministic copy collision: no overwrite; operation Needs attention.
- Failure after admission: persist the last coherent stage and expose recovery.
- Partial cleanup: preserve both versions and expose recovery.
- History/Undo read failure: keep the root usable and show bounded unavailable
  copy; never leak raw backend text.

Status lines and durable public projections contain only bounded counts, enum
labels, opaque IDs, and reason codes. Note title and relative path are displayed
only in active review, receipt, or interactive-history adapters derived from
fresh authority; they are excluded from persistence, status, logs, exceptions,
and object representations.

## Verification

### Pure contracts

- exact choice enum and selection validation;
- no duplicate/cross-root selection admission;
- comparison input/output bounds and elision;
- Note-to-File diff orientation;
- missing note timestamp reports unavailable;
- no hashes/absolute paths in comparison projections or representations;
- review projection and typed blocker accounting;
- only `both_sides_changed` and placement-free `out_of_direction_change` rows
  expose choices; every other conflict reason remains blocked.

### Runtime and executor

- Keep file and Keep note in bidirectional and both one-way directions;
- direction override is one occurrence and does not mutate root direction;
- Skip plus safe actions applies only the reviewed subset;
- unresolved conflicts remain Needs attention;
- deletion, pause, managed placement, activation, capability, and root-level
  skips remain blocking;
- stale token, changed plan, unknown binding, and duplicate choice reject before
  mutation;
- unsupported Windows write remains blocked;
- deterministic conflict-copy folder/note replay and mismatch refusal;
- each single-effect create-or-verify seam returns actual reused IDs and rejects
  owner, path, kind, version, content, placement, and concurrent-create
  collisions before the next durable substage;
- durable operation kinds survive recovery expiry;
- automatic execution, reviewed apply, recovery, and Undo serialize on one root
  lock and revalidate after acquisition;
- conflict resolution and Undo recovery retain exact authority for 30 days;
- per-item failure stops later items and reports completed work honestly;
- fresh post-apply review occurs only after terminal attempted work.

### Keep-both crash matrix

Restart after each boundary:

- recovery admitted;
- folders established;
- conflict copy created;
- placement created;
- copy verified;
- bound note updated;
- binding updated; and
- operation verified before completion.

Every case resumes idempotently or becomes Needs attention without losing either
version.

### Undo

- Keep file, Keep note, and Keep both restore the pre-resolution conflict and
  baseline fields with the freshly restored note version while the binding
  remains active;
- Keep both restores before soft-deleting the exact copy;
- edited copy, changed bound note/file, expired recovery, wrong root, duplicate
  delivery, and stale lease all refuse safely;
- every Undo crash boundary resumes from its linked durable operation;
- linked Undo resumes after source recovery expiry using only its own
  capacity-accounted recovery;
- Undo completion is one-shot and marks the source only after verification;
- a second resolution after Undo succeeds without stale binding authority;
- concurrent apply/apply and apply/Undo attempts serialize, and the loser
  revalidates against the winner's state before mutation;
- empty/manual folders are retained.

### Controller and mounted Textual behavior

- choices stage without runtime mutation;
- selection survives paging and clears on every invalidating lifecycle;
- delayed comparison results cannot publish into a newer review or steal moved
  focus;
- selection updates preserve scroll/focus without full review recomposition;
- comparison and Return controls are keyboard reachable;
- checkmark plus text communicates selection without color;
- Apply enablement uses typed blocker facts;
- partial subset status uses the existing status line once;
- each completed resolution leaves a retained item receipt with working Undo
  and Dismiss even when other conflicts remain;
- navigation/remount preserves current-runtime receipt dismissal, while process
  restart shows durable history and no reconstructed at-action receipt;
- interactive history rows use fresh bounded labels or the short opaque-ID
  fallback without persisting the labels;
- deletion choices remain disabled;
- review, comparison, receipt, and history remain contained at 60x20 and wide
  production CSS sizes.

### Governance and privacy

- no active legacy ASK or sync-engine path returns;
- no new direct private SQLite owner or filesystem authority is introduced;
- comparison content, paths, hashes, raw exceptions, and recovery bytes are
  absent from logs and persistent public diagnostics;
- focused related tests, Ruff, formatter, MyPy, compileall, inventory/privacy
  checks, and diff checks cover touched files only.

Mutation checks must individually prove observation-token enforcement,
recovery-before-write, deletion blocking, deterministic copy collision refusal,
durable choice recording, stale async comparison rejection, per-root mutation
serialization, durable Undo admission, fresh post-Undo binding version, and
30-day recovery expiry.

## Documentation and task hygiene

TASK-97 is renamed **Resolve Notes sync conflicts inline** and its acceptance
criteria describe the lasting-sync outcomes rather than the retired modal.

Before Done:

- add a reviewed implementation plan after this written spec is approved;
- link ADR-055, ADR-059, and ADR-073 in the task plan and notes;
- record focused verification and review evidence;
- update user-facing Notes sync documentation for inline choices, Skip, history,
  and Undo;
- add a lesson only if implementation exposes a genuinely reusable incident;
- check every acceptance criterion; and
- transition TASK-97 to Done through the Backlog CLI only after all Definition of
  Done requirements are satisfied.
