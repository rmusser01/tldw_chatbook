# Library Notes Adaptive 60×20 Design

Date: 2026-07-30
Status: User-selected pre-implementation revisions applied; independent review approved

## Summary

Adapt the existing Database Notes canvas inside Library into a lossless,
keyboard-usable interface down to a supported 60×20 terminal. At compact
widths, Library becomes a single-stage shell and Notes drills through
Navigator, Editor, and Context regions. At wider widths, the current Library
rail and canvas remain side by side.

The change preserves the current Database Note storage, sync, export, template,
autosave, optimistic-lock, and Console-handoff boundaries. It also closes a
session-safety gap required by responsive presentation: note text becomes one
canonical in-memory draft with revisioned, serialized saves, so resizing,
Preview, Context, or continued typing during an in-flight save cannot mislabel
newer text as saved or reconstruct the editor from stale database state.

The canonical draft and persistence state machine live in the portable
Database Note session coordinator defined by ADR-027. `LibraryScreen` remains
the Adapt UI host but does not become the permanent owner of save/conflict
orchestration.

This is the first command in the approved phased sequence:

1. **Adapt** — responsive 60×20 presentation and lossless session state.
2. **Harden** — existing defects, recovery clarity, and keyboard throughput.
3. **Clarify** — information hierarchy and terminology.
4. **Distill** — simplify action density and visual noise.
5. **Shape** — migrate into a dedicated Notes workbench.
6. **Polish** — interaction and visual refinement.

Only Adapt is in scope here.

Backlog task:
[`TASK-1333`](../../../backlog/tasks/task-1333%20-%20Adapt-Library-Notes-for-lossless-60x20-workflow.md)

## Motivation and Evidence

Senior UX/HCI UAT found that the current Notes workflow is functionally capable
but not usable at 60×20:

- the Library rail and Notes canvas expand beyond the viewport;
- the Notes canvas has a 40-cell minimum in addition to the rail minimum;
- list and editor actions are single horizontal rows that clip at compact
  widths;
- the editor body has a 12-row minimum before accounting for navigation,
  footer, title, status, and actions;
- a long Markdown Preview is not keyboard-scrollable because its current
  `Markdown` widget is not focusable;
- the standalone fallback CSS names older Library shell selectors, so an
  isolated harness can disagree with the production bundle.

The same UAT verified useful behavior that must survive:

- blank and template creation;
- long Unicode and Rich-markup-like title/body persistence;
- explicit Save and debounced autosave;
- optimistic-lock conflict handling;
- Preview;
- sort, multi-select, import, export, and sync entry;
- delete cancel/confirm;
- Console handoff;
- stale-result guards and navigation flushes.

The archived critique is
[`.impeccable/critique/2026-07-30T21-12-35Z__w-chatbook-widgets-library-library-notes-canvas-py.md`](../../../.impeccable/critique/2026-07-30T21-12-35Z__w-chatbook-widgets-library-library-notes-canvas-py.md).

The independently assessed pre-implementation critique is
[`.impeccable/critique/2026-07-31T00-57-03Z__2026-07-30-library-notes-adaptive-60x20-design-md.md`](../../../.impeccable/critique/2026-07-31T00-57-03Z__2026-07-30-library-notes-adaptive-60x20-design-md.md).

The current UAT score was 22/40: a strong local-authority and recovery
foundation inside a flat CRUD shell, not yet a capable knowledge workbench.
Adapt does not claim Obsidian parity by itself; it makes the current surface
safe and usable enough to support the later knowledge-loop work.

## Goals

- Make the Notes workflow fully usable at 60×20.
- Preserve visible access to every current Notes capability.
- Use stateful drill-in:
  - Notes opens in Navigator when no unsafe active session exists;
  - selecting a note opens Editor;
  - Back returns to Navigator;
  - Context is an explicit toggle.
- Prevent terminal resizing or presentation changes from discarding or
  reverting unsaved note text.
- Correctly save edits made while an earlier save is still in flight.
- Keep primary actions, recovery, destructive confirmation, and workflow
  navigation visible rather than hiding them in the command palette.
- Preserve the current wide Library layout and existing Notes service/storage
  contracts, including direct access to the current wide editor utilities.
- Keep the implementation compatible with the future dedicated Notes
  workbench without building that workbench in this task.

## Non-goals

- Backlinks, outgoing links, linked mentions, graph navigation, or graph view.
- Full-text search upgrades or query-language parity.
- A dedicated Notes destination/workbench.
- File Notes integration or changes to ADR-021's File Notes controller.
- A side-by-side Context inspector.
- New global shortcuts, command-palette systems, or Vim-style navigation.
- Storage, schema, sync-policy, file-authority, or sync-conflict-policy changes.
  Adapt does harden the in-editor optimistic-conflict interaction so a delayed
  Reload cannot discard a newer draft.
- Changes to filter-clearing behavior after successful create/delete.
- Crash-proof draft recovery or new application-shutdown orchestration beyond
  the existing persistence posture.
- Broad visual polish.
- A claim of terminal screen-reader parity.

## Responsive Contract

### Breakpoint

Compact mode is driven by the invariant outer allocation width reported by
`#library-shell-grid.region.width`, not by its mode-dependent content box or a
device label. Compact CSS may change borders and inner padding but must not
change that outer allocation. The breakpoint is 120 cells:

- `< 120`: active Database Notes uses one Library stage at a time;
- `>= 120`: wide, existing Library rail and canvas side by side.

The boundary is conservative enough to fit the rail plus an unclipped Notes
canvas. Boundary tests use whatever host terminal dimensions produce measured
`#library-shell-grid.region.width` values of 119 and 120 cells; they do not
assume that terminal width and workbench width are equal. Because the measured
outer allocation is unchanged by compact presentation, crossing the boundary
cannot make the measurement reverse and oscillate. The 120-cell boundary is an
acceptance contract for this task; implementation may not silently tune it.
Evidence that requires a different boundary returns the design and task to
review before code proceeds.

The screen updates responsive state only when it crosses the boundary. Resize
events that remain on the same side perform no presentation-state work.

### Compact Library shell

While the Database Notes Navigator, Editor, Create, or Sync workflow is active,
the Library rail and canvas remain mounted but become mutually exclusive
visible stages. Adapt does not change compact behavior for unrelated Library
canvases; making the single-stage shell generic belongs in a separately
specified and tested Library-wide task.

Activating Browse Notes or New Note from the compact Library rail explicitly
switches the visible stage to the Notes canvas; focus-derived rules apply only
when crossing the breakpoint without a new activation intent.

Browse Notes opens Navigator when there is no active Database Note session.
While a dirty, saving, failed, or conflicted session exists, Browse Notes
resumes its exact Editor, Preview, or Context region instead of silently
returning to Navigator. New Note first crosses the same flush barrier; a veto
keeps the active draft onscreen.

When entering compact mode:

- an active dirty, saving, failed, or conflicted Database Note session wins and
  keeps the Notes canvas visible;
- an explicit navigation or deep-link context that has crossed the current
  session's flush barrier wins next;
- focus in the Library rail keeps the rail visible;
- focus in the active canvas keeps the canvas visible;
- an active non-list Notes workflow or selected note activates the canvas;
- only when none of those signals exists does the Library rail become the safe
  fallback.

Crossing back to wide mode reveals both panes without changing selection,
workflow, or note draft state.

Compact Back hierarchy is:

```text
Library → Notes Navigator → Editor → Context
```

- Context Back returns to the same Edit or Preview presentation.
- Editor Back flushes pending work and returns to Navigator.
- Navigator Back returns to the Library rail.
- Create and Sync Back return to Navigator.

The compact shell position is not independently persisted. It is derived from
restored route, selection, workflow, and focus intent.

## Notes Regions

### Navigator

Navigator is the default for an explicit user selection of Notes from the
Library rail when no unsafe active session exists.

Presentation:

- `‹ Library` and `Notes (count)`;
- a persistently labeled filter field;
- a compact **Browse** action group:
  - New note, visually primary;
  - Sort, opening an explicit Newest/Oldest/Title choice rather than cycling
    hidden values;
  - Select;
- a compact **Transfer** action group:
  - Sync;
  - Import;
  - Export;
- persistent filter/status or truthful empty-state copy;
- note list occupying the remaining height.

Browse and Transfer are separate one-row groups, so no decision group exceeds
three actions and no generic More menu is introduced. Only the note list
scrolls. Header, filter, actions, and status remain visible. The list has no
page-level horizontal scrolling.

Empty-state semantics are distinct:

- zero total notes → `No notes yet. Create your first note.` while New note
  remains visible;
- non-empty source with zero filter matches →
  `No notes match “<query>”. Clear the filter.` with an activatable Clear
  filter action.

Filtered-empty copy is markup-disabled, fixed to one row, and cell-ellipsizes
the displayed query so Clear remains in-budget. The full unmodified query stays
in the Filter input.

Tab order follows the visible order, then enters the list. Row identity and
focus restoration use `note_id`, never the current index-based widget id.

Normal row activation opens the note. In multi-select mode, activation toggles
the row instead.

Compact multi-select replaces the normal action group with:

- Done;
- `Select all N shown`;
- Clear;
- Export selected.

It also shows a persistent `N selected` status. New, Sync, and Import are not
mixed into the selection task. Returning to Library clears transient
selection.

### Editor

Presentation:

- `‹ Notes · Edit note`;
- persistently labeled editable title;
- persistently labeled editable body;
- persistent textual save status;
- Save, Preview/Edit, and Context actions.

At `>= 120` workbench cells, Editor and Preview also retain direct access to
the incumbent wide utilities:

- persistently labeled editable keywords;
- metadata;
- Use in Console;
- Copy;
- Export Markdown;
- Export text;
- Delete.

These controls stay inline on wide screens even though Context can also group
them. The deliberate duplication protects the current efficient wide workflow
until Shape introduces a true inspector. No existing utility becomes
Context-only at wide width.

At compact widths, keywords, metadata, Console handoff, Copy, per-note exports,
and Delete move to Context. Editor keeps only Save, Preview/Edit, and Context so
the writing task remains dominant.

The body is the Editor's only scroll owner. At 60×20 its content box receives
10 rows normally, 9 during validation/delete states, and 8 during conflict. On
taller terminals it grows into all remaining rows. It uses soft wrapping, so no
page-level horizontal scroll is needed. Long Input/TextArea values may pan
internally.

Save status uses explicit text and never relies only on color:

- `Unsaved changes`;
- `Saving…`;
- `Saved HH:MM`;
- `Save failed — edits kept. Press Save to retry.`;
- `Conflict — review the choices below`.

Saving never steals focus or moves the caret.

### Preview

Preview shares Editor's header, save status, and actions. Edit and Preview
surfaces are mounted once within the current canvas instance and visibility is
toggled.

The Preview header displays the current title as one markup-disabled row using
cell-width-aware ellipsis. The full raw title remains available in the Editor
title field and is never truncated in session or persistence state.

The Preview body is a focusable scrolling surface:

- Tab enters and leaves it;
- arrows and Page Up/Down scroll it;
- returning to Edit restores the prior body caret, selection, and scroll
  offset.

Preview renders the canonical session draft, never a separate content
snapshot.

### Context

Context is titled `‹ Note · <current title>` and is an explicit alternate
region at every width during Adapt. The title is markup-disabled, fixed to one
row, and cell-width-ellipsized. The neutral Back label returns to whichever
Edit or Preview presentation opened Context.

At compact width, Context is the only access point for keywords, metadata,
Console handoff, Copy, per-note exports, and Delete. At wide width those
controls intentionally remain available inline in Editor/Preview as well.
A simultaneous side-by-side inspector belongs to Shape.

Context is one vertically scrollable region with:

1. **Properties** — persistently labeled editable keywords.
2. **Metadata** — created, modified, version, and word count.
3. **Chatbook** — Use in Console.
4. **Utilities** — Copy, Export Markdown, Export text.
5. **Danger zone** — Delete.

The same save status shown in Editor is visible in Context because changing
keywords mutates and autosaves the note draft.

Context is truthful about current capability. It does not show empty Backlinks
or Linked mentions placeholders; those are added only when the knowledge
relationship model exists.

### Create

New note opens the existing Blank/template chooser as a temporary Navigator
task. It includes Back to Notes.

Selecting Blank or a template retains the existing immediate-create behavior:
the service creates the note first, then the newly created note opens in
Editor with title focus. A failed create stays in Create with a visible
warning.

Until the first genuine edit, the newly created session exposes
`Discard new note` beside the normal Editor actions. It removes only the note
created by that still-current operation token and returns to Navigator. It is
disabled while running, stale completions are ignored, and failure retains the
new note in Editor with actionable status. The action disappears after the
first genuine edit or explicit Save acknowledgement.

Create actions expose a running state and cannot be activated twice while the
request is in flight.

Successful create retains the existing behavior of clearing the active filter
to avoid a stale filtered snapshot.

### Sync

Sync retains the existing folder, direction, conflict, automatic-sync, run,
status, and activity controls. It is one vertically scrollable temporary
Navigator task with Back to Notes.

Direction and conflict policy expose their available values in explicit
selectable controls; they do not require cycling through hidden alternatives.

This task does not change sync ownership or policy.

### Transfer and handoff feedback

Import, whole-source export, selected export, per-note export, Copy, and Use in
Console publish visible running/success/failure text in the active region's
status channel. An action that can duplicate external side effects is disabled
while its request is active. Failure copy names the failed action, retains the
current selection/draft, and gives the next safe action when one exists.

## Canonical Note Session State

### State owners

The note session distinguishes:

1. **Persisted baseline**
   - last confirmed database detail;
   - current optimistic-lock version;
   - created/modified metadata.
2. **Canonical session draft**
   - note id;
   - raw title;
   - raw body;
   - raw keywords text;
   - monotonically increasing draft revision;
   - most recently saved revision.
3. **Presentation state**
   - Edit or Preview;
   - Editor or Context;
   - conflict;
   - conflict-resolution running flag and operation generation/token;
   - untouched-new-note eligibility and create operation token;
   - delete confirmation;
   - loading and mutation-running flags.

Preview and conflict no longer own competing title/body/keyword snapshots.
They render the canonical draft. The persisted baseline remains a comparison
and metadata source; it does not overwrite a mounted draft during presentation
changes.

Pure immutable state snapshots and transition helpers live in
`tldw_chatbook/Library/library_notes_state.py`.

The ADR-027 `DatabaseNoteSessionCoordinator` lives in
`tldw_chatbook/Library/library_notes_session.py`. It owns canonical session
state, revisioned save serialization, editor-conflict operation gating, and
typed flush outcomes. It depends only on an injected async Database Note
session port and imports no Textual or File Notes type.

The port's normalized detail result contains note id, exact title, exact body,
semantic keyword tokens, optimistic-lock version, and created/modified
metadata. The Library adapter may combine existing detail and keyword service
calls internally, but the coordinator receives one coherent normalized result.

`LibraryScreen` owns the coordinator instance plus Textual workers, autosave
debounce timing, route and region transitions, focus, service-port adaptation,
and visible handling of coordinator outcomes. A later Database Notes workbench
can host the same coordinator without moving or rewriting its save/conflict
state machine.

### Draft mutation

The existing editor-arming guard continues to ignore Textual's mount-time
change events.

Every genuine user mutation is sent to the coordinator and:

- updates the canonical draft;
- increments the draft revision;
- marks the session dirty;
- rearms the autosave debounce unless the session is in conflict.

Programmatic presentation synchronization runs under a dedicated host guard.
`apply_session_state()` and rehydration assign title/body/keyword widget values
only when those values differ, and all related change handlers ignore guarded
assignments. Mount-time arming remains a separate lifecycle guard. Preview,
Context, status, conflict, and rehydration synchronization therefore cannot
advance `draft_revision`, mark the session dirty, or arm autosave.

Title/body widgets, the wide keyword field, and the Context keyword field update
the same coordinator draft. Save, Preview, Context, export, copy, and Console
handoff all read its immutable snapshot.

### Lossless persistence validation

The coordinator never silently truncates, strips, or rewrites a user draft to
make it persistable.

Before a service call, typed payload construction validates:

- title: at most 300 characters and accepted unchanged by the existing
  non-HTML title validator;
- body: at most `LIBRARY_NOTE_CONTENT_MAX_CHARS` (currently 2,000,000
  characters), with exact codepoint preservation;
- keywords: comma-delimited semantic tokens, each at most 100 characters and
  accepted without token-content transformation. Delimiter-adjacent whitespace
  and empty delimiters are presentation syntax, not keyword content.

If the current sanitization/validation helpers would truncate, remove control
characters, strip markup-like text, drop a keyword, or otherwise alter title,
body, or keyword token content, payload construction returns a typed
validation-veto outcome and does not call the service. The raw draft remains
dirty and visible. Status names the field and remedy, for example
`Title is 312/300 characters — shorten it to save.`

`Saved HH:MM` is permitted only when the successful persisted payload exactly
represents the canonical title/body and semantically represents the canonical
keyword tokens for that revision. Long Unicode and markup-like persistence
claims apply within these existing storage/validation limits; out-of-contract
input is rejected visibly rather than rewritten.

### Revisioned, serialized save

Only one coordinator save request may be active for a note.

A save request captures:

- note id;
- expected optimistic-lock version;
- draft revision;
- losslessly validated persistence payload.

Save/autosave requests made while one is running are coalesced into one pending
follow-up instead of cancelling the in-flight request. This matters because
cancelling an asyncio worker cannot stop a service call already executing in a
thread.

The coordinator save driver owns a single `pending_save_requested` flag. Every
genuine edit that occurs during an in-flight attempt raises it immediately,
including during a follow-up attempt. A Save/autosave request also raises it
when it targets a revision newer than the active attempt; a request for the
already-active revision is satisfied by that attempt. Before starting an
attempt, the driver clears the flag and captures the latest revision, version,
and payload. After a successful attempt, it rechecks the flag and current
revision, clears a request already satisfied by the saved revision, and starts
another attempt only when the current revision is newer. This loop continues
until the current revision equals the saved revision and the pending flag is
clear; there is no two-attempt limit.

On success:

- always accept the returned/new optimistic-lock version for the same active
  note;
- patch the persisted detail/list baseline with the payload that actually
  reached storage;
- if the current draft revision equals the saved revision and the validated
  payload losslessly represents that canonical revision, clear dirty and show
  `Saved HH:MM`;
- if the current draft revision is newer, retain dirty, retain the newer raw
  draft, and continue the save driver against the updated version.

On validation veto:

- make no persistence call;
- retain the raw draft, dirty state, caret, and focus;
- clear automatic chaining until the user changes the invalid field or invokes
  Save again;
- show field-specific actionable status;
- for title/body vetoes, return Preview/Context to Editor and focus the
  corresponding field;
- for keyword vetoes at compact width, remain/return to Context and focus its
  keyword control; at wide width, focus the inline Editor/Preview keyword
  control;
- veto Back and navigation exactly like a save failure.

On ordinary failure:

- retain the current draft and dirty state;
- show `Save failed — edits kept. Press Save to retry.`;
- clear the pending flag and stop automatic chaining so a failing service
  cannot enter a retry loop;
- let the next genuine edit/autosave or explicit Save retry the latest
  revision;
- do not navigate away.

On optimistic-lock conflict:

- retain the latest canonical draft, including edits made after the conflicted
  request began;
- stop automatic retry;
- switch from Context or Preview to Editor;
- show a focusable explanatory callout plus Overwrite and Reload.

Conflict resolution is a single gated operation. The first Overwrite or Reload
sets `conflict_resolution_running`, increments/captures an operation token, and
disables both actions. Duplicate activation or activation of the other action
while the fetch is running is a no-op. Every completion verifies the active
note id, conflict generation, and operation token before it may update state.
The gate clears only at the terminal success, ordinary failure, missing-note,
or renewed-conflict outcome.

Overwrite fetches the fresh persisted version without changing the draft,
rebases the expected version, and enters the same serialized save driver with
the latest revision and payload. Edits made during either the fetch or save are
therefore included in that attempt or a later coalesced attempt. A second
optimistic-lock conflict returns to the same conflict state.

Reload captures the draft revision when the user activates it, then fetches the
fresh persisted detail. The result may replace the canonical draft only when
the active note and conflict are unchanged and the current revision still
equals the captured revision. If the user typed during the fetch, Reload is not
applied; the conflict and draft remain, the UI reports
`Draft changed — Reload not applied. Choose again.`, and the user must choose
again.

If the note was deleted elsewhere, Overwrite retains the draft in Editor and
reports that the target no longer exists. An unchanged-revision Reload is the
explicit discard decision and may return to Navigator with a warning; a
changed-revision Reload remains vetoed.

### Flush and lifecycle

Back, row switching, rail switching, route navigation, and controlled screen
replacement continue to use the pending-work flush barrier before permitting
Library to unmount.

Flush:

1. cancels the debounce timer;
2. asks the coordinator to wait for the current serialized save;
3. lets the coordinator run the coalesced/latest save if a newer draft exists;
4. permits navigation only from a successful typed flush outcome with no
   dirty/error/conflict state.

Resize, Preview, and Context toggles never force a save.

`on_unmount` is cleanup-only: it cancels timers and workers after the
navigation barrier has succeeded and does not attempt asynchronous persistence.
The app must not voluntarily replace a dirty Library screen by bypassing the
barrier. Abrupt process termination and application-wide shutdown
orchestration are outside Adapt.

Transient Context, Preview, confirmation, loading, and compact-stage state are
not restored on a new screen instance. Selected note, list/editor workflow,
filter, and sort retain their existing persisted behavior. A restored editor
session opens Editor, not Context or Preview.

### Destructive mutation admission

Discard-new-note and general Delete share a coordinator-backed destructive
admission gate even though the host continues to call the existing delete
service.

Entering delete confirmation:

1. cancels debounce and crosses the coordinator flush barrier;
2. atomically enters a destructive-pending state for the active note/session
   generation and expected version;
3. disables title/body/keyword mutation and new save/autosave admission;
4. renders the existing content read-only with Cancel focused.

Delete invoked from compact Context switches to the Editor
delete-confirmation surface for this gated state. Cancel or failure restores
the originating Context region and its focus/scroll; success returns to
Navigator.

Cancel exits the gate and restores editing. Confirm revalidates note id,
session generation, expected version, and—when applicable—the untouched-create
token immediately before the host invokes the delete service. The running state
keeps all draft fields and destructive actions disabled; queued key events,
`Ctrl+S`, duplicate activation, and Escape cannot mutate, save, cancel, or
start another operation after the service mutation begins.

Discard new note enters the same gate atomically before its first service call.
It is admitted only while the coordinator still marks the active note as the
untouched result of the matching create token. An explicit no-op Save means
“keep this note”: after that Save is acknowledged, discard eligibility clears
even though no versioned write was necessary.

Failure or optimistic conflict exits the running gate, restores editable
widgets from the unchanged coordinator draft, and presents actionable status.
Success ends the session and returns to Navigator. Invalidating a token only
after a delete call has started is never treated as sufficient protection.

## Stable Composition

The Library rail and canvas remain mounted across Notes compact shell changes.

After a note detail loads, the current `LibraryNotesCanvas` instance mounts
these surfaces once:

- Edit;
- Preview;
- Context;
- conflict callout;
- delete confirmation.

Presentation changes toggle display/classes and synchronize visible text. They
do not remove/remount the note session subtree.

`LibraryNotesCanvas` remains presentation-only and gains an idempotent
presentation seam, conceptually `apply_session_state(state)`, which:

- updates Editor/Preview/Context visibility;
- updates the Preview source from the canonical draft;
- updates Context title/properties/metadata;
- updates save status in Editor and Context;
- updates conflict and delete-confirmation visibility;
- applies compact/wide action-group classes.

The seam performs value-difference checks and uses the screen-owned
presentation-sync guard for any Input/TextArea assignment.

The method does not query a database, call services, perform navigation, own
timers, or mutate global application state.

Navigator → Editor and Editor → Navigator may recompose because they are
workflow transitions. Resize, Preview, Context, status, and confirmation
changes do not.

An unrelated whole-`LibraryScreen` recompose may replace the canvas under the
current architecture. Adapt introduces one central screen-level interception
seam for every `refresh(recompose=True)` call while a Database Note session is
active. Before delegating to the base refresh, it captures caret, selection,
body scroll, active presentation, and focus identity. The coordinator and its
canonical draft remain alive on the still-mounted screen object. After the new
canvas mounts, the seam rehydrates presentation values by `note_id` from the
coordinator snapshot without consulting stale persisted text.

No individual recompose caller is responsible for this safety behavior. Initial
composition and a screen with no active note make the seam a no-op. Stable
widget identity is guaranteed across Notes compact/wide, Preview, Context,
status, conflict, and confirmation transitions within one canvas instance, not
across an intentional whole-screen recompose.

## Focus and Safety

### Entry and return focus

- Explicit Notes selection with no active session → Navigator.
- Explicit Notes selection with an unsafe active session → resume its exact
  Editor, Preview, or Context region.
- Direct note link or resumed saved editor → Editor.
- Existing note → focus body after detail load.
- New note → focus title.
- Editor Back → focus the originating note row by `note_id`.
- If the row vanished or is filtered out → focus the filter.
- Context Back → focus the Editor/Preview Context button that opened Context.
- Preview Edit → restore body caret, selection, and scroll.
- Navigator Back → restore the selected Library rail row.

Programmatic focus changes explicitly scroll the target into view without
animation.

The local Escape binding follows the same visible Back hierarchy:

- delete confirmation → Cancel;
- Context → its originating Edit/Preview presentation;
- Editor/Preview → attempt the same flush-guarded Back to Navigator;
- Create/Sync → Navigator;
- Navigator → Library rail.

Escape cannot bypass a save failure or conflict veto. It is a Notes-local
accelerator, not a new global shortcut.

### Local accelerators and contextual help

Notes adds only region-scoped accelerators:

- `Ctrl+N` → New note, after the current session flush barrier;
- `/` in Navigator → focus Filter;
- `Ctrl+S` in Editor, Preview, or Context → request immediate save through the
  coordinator;
- `Escape` → the Back hierarchy above.

`Ctrl+S` is ignored with visible status while conflict resolution or a
destructive pending/running state owns mutation admission.

The existing one-row application footer exposes help for the active region
without consuming the 15-row Notes canvas:

- Navigator: `Ctrl+N New · / Find · Esc Library`;
- Editor: `Ctrl+S Save · Esc Notes`;
- Preview: `Pg Scroll · Esc Notes`;
- Context: `Enter Act · Esc Note`;
- conflict: `Enter Choose · Esc Locked`.

Create, Sync, selection, and confirmation states replace the footer text with
their own Back/Cancel and activation guidance. These are local Notes bindings
and do not expand the global destination shortcut layer.

At compact Notes widths, the footer gives shortcut guidance priority and
suppresses its ancillary word-count, token-count, and database-size children.
The essential abbreviated hint must remain fully visible at 60 columns.
Crossing back to wide restores those indicators from their existing state.

### Safe interruption

- Loading always retains a visible, focusable Back control.
- A stale detail result is discarded after Back or note switching.
- Save errors keep the current focus and draft.
- Save failure text says `Press Save to retry`.
- Conflict interruption focuses the explanatory callout, not Overwrite or
  Reload.
- Delete confirmation focuses Cancel by default.
- Delete confirmation is two-step and names the consequence.
- Delete remains disabled while the request runs.
- Delete failure returns to Context with the draft intact.
- Delete success returns to Navigator.
- A surface hidden for stable composition is removed from layout, focus order,
  and action queries; visibility alone is not considered sufficient.

The app's supported Textual 8 `TextArea` defaults to `tab_behavior="focus"`,
so Tab and Shift+Tab can leave the body editor. Buttons use Enter. Adapt does
not claim Space activation unless a control defines and tests it explicitly.

## Scrolling and Geometry

Each active Notes region has one clear scroll owner:

| Region | Scroll owner |
| --- | --- |
| Navigator | Note list |
| Editor | TextArea body |
| Preview | Focusable preview body |
| Context | Whole Context region |
| Create | Whole Create region |
| Sync | Whole Sync region |

Geometry-critical compact rules exist in both:

- `LibraryScreen.DEFAULT_CSS`, for an isolated screen/harness;
- `tldw_chatbook/css/components/_agentic_terminal.tcss`, for the app bundle.

The stale fallback selectors are aligned with the live
`#library-shell-grid`, `#library-rail`, and `#library-canvas` selectors.

The source TCSS is authoritative.
`tldw_chatbook/css/tldw_cli_modular.tcss` is regenerated with
`tldw_chatbook/css/build_css.py` and never hand-edited. The generated diff must
contain only expected source-derived rules plus its timestamp.

### 60×20 vertical budget

The terminal-level budget is fixed:

| Owner | Rows |
| --- | ---: |
| Main navigation | 3 |
| Library status/header | 1 |
| Notes canvas content box | 15 |
| Footer | 1 |
| **Total** | **20** |

At compact width, `#library-shell-grid` and `#library-canvas` remove their
decorative borders, vertical padding, and vertical margins so the Notes canvas
receives the full 15-row content box. The wide border treatment remains
unchanged.

Within the 15-row canvas:

| Region/state | Exact allocation | Total |
| --- | --- | ---: |
| Navigator normal | header 1 + labeled filter 1 + Browse actions 1 + Transfer actions 1 + status 1 + note list 10 | 15 |
| Navigator filtered-empty | header 1 + labeled filter 1 + Browse actions 1 + Transfer actions 1 + ellipsized status/Clear 1 + empty viewport 10 | 15 |
| Navigator sort choice | header 1 + labeled filter 1 + explicit sort chooser replacing Browse row 1 + Transfer actions 1 + status 1 + note list 10 | 15 |
| Navigator selection | header 1 + labeled filter 1 + selection actions 1 + selection status 1 + note list 11 | 15 |
| Navigator/loading | header/Back 1 + status 1 + loading viewport 13 | 15 |
| Editor normal or untouched-new | header 1 + labeled title 1 + Body label 1 + body 10 + status 1 + actions 1 | 15 |
| Editor validation veto | header 1 + labeled title 1 + Body label 1 + body 9 + actionable validation status 2 + actions 1 | 15 |
| Editor conflict | header 1 + labeled title 1 + Body label 1 + body 8 + status 1 + conflict explanation 2 + resolution actions 1 | 15 |
| Editor delete confirmation | header 1 + labeled title 1 + Body label 1 + read-only body 9 + status 1 + confirmation copy 1 + Cancel/Delete actions 1 | 15 |
| Preview | header/title 1 + preview body 12 + status 1 + actions 1 | 15 |
| Context | header 1 + status 1 + Context scroll viewport 13 | 15 |
| Create | header/Back 1 + Create scroll viewport 14 | 15 |
| Sync | header/Back 1 + Sync scroll viewport 14 | 15 |

Multi-select replaces both normal Navigator action rows with its one action row
and one selection-status row. Sort choice replaces the Browse row rather than
adding a row. Compact labels may share a row with their single-line Input, but
they remain persistent text rather than placeholder-only guidance. Conflict
mode hides Preview/Context actions until the conflict resolves so recovery
keeps the row budget and focus priority. On terminals taller than 20 rows,
surplus rows flow only to the active scroll/content owner named in the table.

General-delete running reuses the Editor delete-confirmation allocation;
untouched-note discard running reuses the Editor untouched-new allocation.
Transfer/handoff running, success, and failure replace text within the existing
one-row status allocation and never add geometry.

All dynamic headers render with `markup=False`, a fixed one-row height, and a
pure cell-width ellipsis helper. Raw user titles are never interpolated as Rich
markup and never truncated in the draft or persisted value.

At 60×20:

- every required visible control has positive width and height;
- every required visible control lies inside the terminal viewport;
- no page-level horizontal scrollbar is required;
- focused controls are visibly identifiable and scrolled onscreen;
- action labels remain readable;
- the active scroll/content owner meets the numeric floor in the table above.

Hidden stable-composition surfaces are intentionally excluded from visible
geometry assertions.

## Accessibility Scope

Adapt guarantees:

- keyboard-only traversal with Tab/Shift+Tab;
- Enter activation for every control and Space only where a control explicitly
  defines and tests it;
- the local Escape/Back hierarchy defined above;
- keyboard scrolling of list, body, Preview, Context, Create, and Sync;
- visible, non-obscuring focus;
- persistent visible text labels for filter, title, body, and keywords;
- text labels for actions;
- deterministic linear focus order in normal, conflict, confirmation, and
  rehydrated states;
- persistent textual save and error status;
- meaning that does not rely only on color;
- safe focus defaults for conflict and deletion.

Terminal screen-reader behavior depends on the terminal emulator and host
application and is not claimed as parity in this phase.

## Component and File Boundaries

### `tldw_chatbook/UI/Screens/library_screen.py`

Owns:

- compact/wide shell state;
- active Library stage;
- Notes region transitions;
- the `DatabaseNoteSessionCoordinator` instance;
- Textual autosave debounce scheduling;
- the Database Note service-port adapter;
- the central whole-screen recompose capture/rehydration seam;
- focus restoration;
- navigation and visible handling of typed coordinator outcomes.

`LibraryScreen` does not own canonical draft, save-queue, or editor-conflict
transition logic. No dedicated Notes workbench or route migration is introduced
in Adapt.

### `tldw_chatbook/Library/library_notes_session.py`

Owns the ADR-027 `DatabaseNoteSessionCoordinator` and its minimal async
Database Note session port:

- canonical active-note session state;
- normalized, complete note-detail loading through the session port;
- draft mutations and revision tracking;
- lossless save-payload validation and typed validation vetoes;
- serialized/coalesced saves;
- editor-conflict operation generation and gating;
- untouched-new-note eligibility and create-token gating;
- destructive-operation admission, mutation locking, and typed outcomes;
- revision-safe Overwrite/Reload;
- typed save, conflict, missing-note, validation, destructive, and flush
  outcomes.

It imports no Textual type, widget, route, File Notes controller, database
handle, or global application state. It does not schedule the autosave debounce
or decide navigation/focus.

### `tldw_chatbook/Library/library_notes_state.py`

Owns:

- immutable display/draft/session snapshots;
- revision comparisons and pure transition helpers;
- existing list/editor display builders.

It owns no asyncio, Textual workers, services, database access, or global state.

### `tldw_chatbook/Widgets/Library/library_notes_canvas.py`

Owns:

- Notes presentation;
- stable Edit/Preview/Context child surfaces;
- the focusable Preview scroll wrapper;
- compact action groups and compatible wide inline utilities;
- idempotent presentation synchronization.

### CSS

- Update `LibraryScreen.DEFAULT_CSS` for harness geometry.
- Update `tldw_chatbook/css/components/_agentic_terminal.tcss`.
- Regenerate `tldw_chatbook/css/tldw_cli_modular.tcss`.

### Tests

- Extend `Tests/Library/test_library_notes_state.py` for pure state.
- Add `Tests/Library/test_library_notes_session.py` for coordinator concurrency,
  conflict, and flush behavior without mounting Textual.
- Extend `Tests/UI/test_library_shell.py` for Pilot, CSS, geometry, and
  concurrency, reusing its real-bundle `LibraryHarness` and gated service
  fakes.
- Do not create another Library harness or extract shared test infrastructure
  during Adapt.

## Acceptance References

### Existing capability parity

“Every current Notes capability” in TASK-1333 means:

| Capability | Compact access | Wide access |
| --- | --- | --- |
| Filter and sort | Navigator | Navigator |
| Blank/template create and untouched-note discard | Navigator → Create/Editor | Existing rail/Navigator → Create/Editor |
| Sync | Navigator → Sync | Navigator → Sync |
| Import and whole-source export | Navigator Transfer group | Navigator Transfer group |
| Multi-select, select shown, clear, selected export | Navigator selection mode | Navigator selection mode |
| Edit title/body | Editor | Editor |
| Edit keywords | Context | Inline Editor/Preview and Context |
| Explicit Save and autosave | Editor | Editor |
| Preview/Edit | Editor/Preview | Editor/Preview |
| Metadata | Context | Inline Editor/Preview and Context |
| Use in Console | Context | Inline Editor/Preview and Context |
| Copy | Context | Inline Editor/Preview and Context |
| Export Markdown/text | Context | Inline Editor/Preview and Context |
| Delete cancel/confirm | Context | Inline Editor/Preview and Context |
| Overwrite/Reload conflict recovery | Editor conflict state | Editor conflict state |

### Focus identity

“Focus intent” means the stable tuple:

- Library stage;
- Notes region;
- active `note_id`, when any;
- semantic control role (`filter`, `note-row:<note_id>`, `title`, `body`,
  `preview-body`, `context`, `context-action`, `save`, `conflict-callout`,
  `delete-cancel`, `library-row:<row_id>`,
  `create-template:<template_key>`, `sync-folder`, `sync-direction`,
  `sync-conflict-policy`, `sync-auto`, `sync-run`, or region Back);
- body caret and selection, when applicable;
- active scroll-owner offset.

“Relevant widget identity” means the mounted title, body, Preview body, Context,
conflict callout, delete confirmation, and status surfaces named in Stable
Composition. Identity is stable across compact/wide, Preview, Context, status,
conflict, and confirmation toggles within one canvas instance. A deliberate
whole-screen recompose instead proves coordinator continuity plus exact
presentation rehydration.

## Verification

### Pure state tests

- genuine user edits increment draft revision;
- mount events do not;
- raw title/body/keywords snapshots remain unsanitized;
- header title formatting is markup-disabled, one row, cell-width ellipsized,
  and leaves the raw title unchanged;
- Preview and Context derive from the canonical draft;
- transient presentation state is excluded from persistence.

### Coordinator tests

- coordinator construction and save/conflict tests use an injected fake async
  port and import no Textual module;
- the normalized detail port returns title, body, keywords, version, and
  metadata as one coherent result;
- title/body/keyword overflow or transforming validation returns a typed veto,
  makes no service call, retains raw dirty text, and never reports Saved;
- a current-revision success clears dirty;
- stale-revision successes retain dirty and continue through three or more
  successive edit revisions until the latest revision is saved;
- ordinary failure stops automatic chaining and the next edit or explicit Save
  retries the latest revision;
- conflict preserves the newest draft;
- Overwrite rebases through the serialized save driver;
- Reload applies only when its captured draft revision is still current;
- untouched-new-note discard eligibility clears on the first genuine edit and
  rejects a stale create token;
- explicit no-op Save clears untouched-new-note discard eligibility;
- destructive admission rejects draft mutation, save, autosave, duplicate
  delete/discard, and stale note/session/create tokens;
- guarded programmatic synchronization never advances the revision or arms
  autosave;
- typed flush outcomes permit or veto navigation correctly.

### Pilot interaction tests

Run with the real generated stylesheet:

- 60×20 Library rail → Notes Navigator;
- compact Browse Notes with an unsafe session resumes its exact region;
- Navigator → Editor → Context → Editor → Navigator;
- local Escape follows the same hierarchy and cannot bypass a flush veto;
- local `Ctrl+N`, `/`, and `Ctrl+S` act only in their documented Notes regions
  and footer help changes with region/state;
- create Blank/template → Editor;
- discard an untouched newly created note; edit first and prove Discard is no
  longer available;
- type/Save/Escape attempts during discard/delete pending and running states
  cannot mutate, save, cancel a started service call, or race the delete;
- zero total notes and zero filter matches render distinct actionable copy;
- Sort, Sync direction, and Sync conflict policy expose direct choices rather
  than hidden cycles;
- compact Sync entry and Back;
- compact multi-select and Done;
- compact keyboard activation of Import, whole-source Export, Copy,
  per-note Markdown/text export, and Use in Console with temporary paths plus
  stub clipboard/navigation services;
- transfer/handoff running, success, failure, and duplicate-activation states;
- compact footer hides ancillary indicators and keeps the essential abbreviated
  hint fully visible at 60 columns;
- loading Back plus late-result rejection;
- Preview keyboard focus and scroll;
- Context keyword edit and autosave;
- three successive edits while earlier saves are in flight;
- explicit Save while save is in flight;
- edit during Overwrite fetch/save;
- edit during Reload fetch, proving the fetched detail is not applied;
- Overwrite then Reload, Reload then Overwrite, and duplicate conflict-action
  activation while the first fetch runs, proving only one token can apply;
- Preview, Context, save-status, conflict, and rehydration synchronization
  proving no draft-revision or autosave change;
- Back waiting for the complete save chain;
- save failure vetoing Back;
- conflict from Editor, Preview, and Context;
- Delete cancel, failure, and success;
- duplicate Create/Delete activation prevention;
- direct-note deep link;
- saved editor restore without Context/Preview restore;
- normal/conflict/confirmation/rehydrated focus-order assertions;
- long Unicode/Rich-like titles remain raw in the editor while headers stay one
  markup-disabled ellipsized row;
- Enter activates every Notes control; no generic Space claim is asserted;
- at `>= 120`, keywords, metadata, Console, Copy, exports, and Delete remain
  directly reachable without entering Context.

### Responsive tests

- 60×20 fully usable;
- representative compact 80×24 and 100×30;
- host sizes that produce measured `#library-shell-grid.region.width` values of
  119/120 without breakpoint oscillation;
- existing 170×48 wide layout;
- dirty 170 → 60 → 170 round trip preserving:
  - draft text;
  - caret;
  - selection;
  - body scroll;
  - Preview/Edit choice;
  - selected note;
- 170 → 60 → 170 crossings preserve the complete focus tuple for:
  - Library rail row;
  - Navigator Filter and a selected/scrolled note row;
  - Editor and Preview;
  - Context plus its scroll offset;
  - Create plus its focused/scrolled template row;
  - Sync plus folder, direction, conflict policy, automatic-sync value,
    running/status/activity state, focused control, and scroll offset;
- crossing without changing breakpoint mode performs no state work.

### Geometry tests

- required visible regions stay inside terminal bounds;
- required visible controls have non-zero dimensions;
- no required action is clipped;
- each active scroll/content owner meets its numeric 60×20 row floor;
- normal, filtered-empty, sort-choice, selection, loading, untouched-new,
  validation, conflict, delete-confirmation, Preview, Context, Create, and Sync
  states match their exact 15-row allocations;
- no page-level horizontal overflow;
- programmatically focused controls are visible.

### Lifecycle and responsiveness tests

- repeatedly cross the breakpoint and toggle Context/Preview;
- assert Editor widget identities remain stable;
- await a quiet baseline, then assert the relevant Notes save/autosave worker
  groups and timers do not grow;
- assert no unbounded mount/remove churn;
- land an unrelated source snapshot during a dirty edit and verify the new
  canvas rehydrates the draft, caret, selection, scroll, presentation, and
  focus;
- invoke representative non-Notes `refresh(recompose=True)` origins while a
  draft is active and prove the central seam, not each caller, rehydrates it;
- navigate away with a pending and an in-flight save;
- unmount/remount and verify only intended persistent state returns.

### ADR-011 responsiveness evidence

- record a 100 ms event-loop heartbeat during a 30-second compact/wide,
  Preview/Context, and route-switch soak with no gap above 250 ms;
- after a quiet baseline, verify Notes worker backlog and timer registry return
  to baseline;
- perform at least 50 breakpoint crossings and 50 Library route switches with
  no unbounded worker, timer, or mount/remove growth;
- retain before/after evidence in the task implementation notes.

### Regression and completion checks

- focused Library/Notes state and UI tests;
- TASK-400 complete and the supported dependency metadata constrained to
  Textual `>=8.0.0,<9` before TASK-1333 can complete;
- navigation, persistence, sync, import/export, and Console-handoff regressions;
- CSS generation, parsing, and source/bundle parity;
- targeted parity for geometry-critical selectors/properties shared by
  `LibraryScreen.DEFAULT_CSS` and `_agentic_terminal.tcss`;
- repository static checks;
- full project tests required by the repository Definition of Done;
- interface detector on the changed UI sources as supplemental evidence;
- final keyboard UAT with synthetic data and saved 60×20/wide SVG evidence.

All runtime UAT uses a temporary profile, temporary databases/directories,
synthetic notes, and stub services. It must not touch real user notes, sync
configuration, clipboard, provider settings, or API credentials.

## Rollout and Failure Posture

Responsive behavior is always enabled; there is no feature flag or data
migration.

Wide behavior is the compatibility baseline. A compact-mode failure must fail
visibly rather than discard content:

- save or conflict errors retain the draft and veto navigation;
- a missing note with no unsafe draft returns to Navigator; an unsafe
  missing-note conflict retains the draft until explicit discard;
- missing services disable/report the relevant action;
- stale async results are discarded;
- responsive presentation never decides storage authority.

## ADR Check

ADR required: yes

ADR path:
`backlog/decisions/027-portable-database-note-session-coordinator.md`

Reason: the portable Database Note session coordinator is a new long-lived
cross-module interface for draft, save, editor-conflict, and flush
orchestration. ADR-027 records its port, ownership, and separation from Textual
presentation and ADR-021 File Notes authority. Adapt still changes no schema,
file authority, sync policy, route ownership, security boundary, or provider
boundary.

Related decisions:

- `backlog/decisions/011-chatbook-workbench-ui-system.md` — explicit state,
  stable composition, visible workflow, and responsiveness gates.
- `backlog/decisions/015-shell-destination-ia.md` — Notes remains owned by
  Library.
- `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`
  — File Notes keeps a separate future workbench/controller boundary and is
  not folded into this Database Notes session.
- `backlog/decisions/022-textual-8-runtime-floor.md` — supported Textual runtime
  for focus, layout, and widget APIs. TASK-1333 depends on TASK-400 completing
  this accepted runtime-floor decision.

The later Shape migration to a dedicated Notes workbench must review whether
ADR-027 remains sufficient for the Database session host move and whether
ADR-021 needs amendment for combined Database/File Notes presentation.
