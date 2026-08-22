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

History projections contain only operation ID, typed choice, bounded state,
completion/update time, and Undo availability/reason. They never expose the
binding ID, path, title, content, hash, or exception text. History pages contain
at most 100 rows.

## Review and comparison flow

1. **Check** produces the existing immutable reconciliation plan and token.
2. The controller projects conflict rows but does not copy private authority.
3. **View comparison** calls the runtime with root ID, token, and binding ID.
4. Under the root's planning lease, the runtime performs a fresh observation and
   recomputes the plan.
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

Before mutation the runtime:

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
Note as an unbound manual note, then makes the reviewed file the bound version.

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

Startup recovery reconstructs conflict requests from the durable operation kind
and validated private metadata. It never treats a resolution operation as an
automatic action.

Per-item Undo is available only for a completed conflict resolution with
unexpired exact recovery. Undo reacquires the root lease and validates the
current note, file, binding, and conflict copy before mutation.

Undo restores the pre-resolution authority and the original binding baseline,
leaving the binding **Needs attention** so the original conflict can be reviewed
again. For Keep both it:

1. restores and verifies the original bound note first;
2. restores the original binding baseline as Needs attention; and
3. soft-deletes only the unchanged operation-owned conflict-copy note.

If the final cleanup fails, both visible versions remain and the operation is
Partial. Empty manual folders may remain. Undo never deletes a shared/manual
folder.

Undo is one-shot. A compare-and-set changes the completed operation's bounded
reason code to `undo_completed` only when its reason is still empty. The durable
operation state remains `completed`; history projects the reason as **Undone**.
A changed bound authority or edited conflict copy disables/refuses Undo with a
bounded `changed_since_resolution` reason and makes no mutation.

## Inline Library behavior

Conflict rows are collapsed by default and show bounded note title, wrapped
root-relative path, `Both file and note changed`, current selection, and the
four choices.

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

- if conflicts remain, page 1 of the fresh review stays visible, focus moves to
  its first conflict, and the existing status line reports applied and remaining
  counts once;
- if no attention remains, the normal receipt phase is shown; and
- if an admitted operation is non-terminal, the existing root recovery path is
  shown instead of a fresh review.

Deletion and other unsupported attention rows retain disabled controls and the
existing unavailable explanation. Activation reviews continue to reject all
attention.

## Resolution history UI

Each lasting-sync root gains a bounded **Resolution history** action in the
existing retained canvas. It is not a modal. The action is enabled when the
runtime reports at least one durable conflict-resolution operation.

History shows newest first, 100 rows per page. Each row shows choice, timestamp,
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
only in the active comparison/review surface and are excluded from status,
history, logs, exceptions, and object representations.

## Verification

### Pure contracts

- exact choice enum and selection validation;
- no duplicate/cross-root selection admission;
- comparison input/output bounds and elision;
- Note-to-File diff orientation;
- missing note timestamp reports unavailable;
- no hashes/absolute paths in comparison projections or representations;
- review projection and typed blocker accounting.

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
- durable operation kinds survive recovery expiry;
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
  baseline;
- Keep both restores before soft-deleting the exact copy;
- edited copy, changed bound note/file, expired recovery, wrong root, duplicate
  delivery, and stale lease all refuse safely;
- Undo completion is one-shot and durable;
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
durable choice recording, and stale async comparison rejection.

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
