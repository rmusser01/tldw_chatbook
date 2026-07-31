# Library Notes Adaptive 60×20 Design

Date: 2026-07-30
Status: User-approved

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

This is the first command in the approved phased sequence:

1. **Adapt** — responsive 60×20 presentation and lossless session state.
2. **Harden** — existing defects, recovery clarity, and keyboard throughput.
3. **Clarify** — information hierarchy and terminology.
4. **Distill** — simplify action density and visual noise.
5. **Shape** — migrate into a dedicated Notes workbench.
6. **Polish** — interaction and visual refinement.

Only Adapt is in scope here.

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

The archived critique is:

`/.impeccable/critique/2026-07-30T21-12-35Z__w-chatbook-widgets-library-library-notes-canvas-py.md`

The current UAT score was 22/40: a strong local-authority and recovery
foundation inside a flat CRUD shell, not yet a capable knowledge workbench.
Adapt does not claim Obsidian parity by itself; it makes the current surface
safe and usable enough to support the later knowledge-loop work.

## Goals

- Make the Notes workflow fully usable at 60×20.
- Preserve visible access to every current Notes capability.
- Use stateful drill-in:
  - Notes opens in Navigator;
  - selecting a note opens Editor;
  - Back returns to Navigator;
  - Context is an explicit toggle.
- Prevent terminal resizing or presentation changes from discarding or
  reverting unsaved note text.
- Correctly save edits made while an earlier save is still in flight.
- Keep primary actions, recovery, destructive confirmation, and workflow
  navigation visible rather than hiding them in the command palette.
- Preserve the current wide Library layout and existing Notes service/storage
  contracts.
- Keep the implementation compatible with the future dedicated Notes
  workbench without building that workbench in this task.

## Non-goals

- Backlinks, outgoing links, linked mentions, graph navigation, or graph view.
- Full-text search upgrades or query-language parity.
- A dedicated Notes destination/workbench.
- File Notes integration or changes to ADR-021's File Notes controller.
- A side-by-side Context inspector.
- New global shortcuts, command-palette systems, or Vim-style navigation.
- Storage, schema, sync-policy, file-authority, or conflict-policy changes.
- Changes to filter-clearing behavior after successful create/delete.
- The existing zero-match copy defect; Harden will distinguish “no notes” from
  “no notes match this filter.”
- Broad visual polish.
- A claim of terminal screen-reader parity.

## Responsive Contract

### Breakpoint

Compact mode is driven by the available Library workbench width, not by a
device label. The initial breakpoint is 120 cells:

- `< 120`: compact, one Library stage visible at a time;
- `>= 120`: wide, existing Library rail and canvas side by side.

The boundary is conservative enough to fit the rail plus an unclipped Notes
canvas and is verified at 119/120 in the real generated stylesheet. If Pilot
geometry proves the actual content budget differs, the constant may be tuned
before implementation completion, with the measured reason documented in the
task notes and tests moved to the final boundary.

The screen updates responsive state only when it crosses the boundary. Resize
events that remain on the same side perform no presentation-state work.

### Compact Library shell

The Library rail and canvas remain mounted but are mutually exclusive visible
stages. This shell behavior is generic so any Library canvas selected at
60×20 has a way back to the Library rail; the full interaction verification in
this task remains Notes-focused.

When entering compact mode:

- focus in the Library rail keeps the rail visible;
- focus in the active canvas keeps the canvas visible;
- a Notes or item deep link explicitly activates the canvas;
- when no meaningful focus exists, an explicit navigation context wins,
  otherwise the Library rail is the safe default.

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
- Other compact Library canvases receive an immediate `Back to Library`
  escape without adopting Notes-specific state.

The compact shell position is not independently persisted. It is derived from
restored route, selection, workflow, and focus intent.

## Notes Regions

### Navigator

Navigator is the default for an explicit user selection of Notes from the
Library rail.

Presentation:

- `‹ Library` and `Notes (count)`;
- filter field;
- a two-column compact action group:
  - New note;
  - Sort;
  - Sync;
  - Import;
  - Export;
  - Select;
- persistent filter/status or actionable empty state;
- note list occupying the remaining height.

Only the note list scrolls. Header, filter, actions, and status remain visible.
The list has no page-level horizontal scrolling.

Tab order follows the visible order, then enters the list. Row identity and
focus restoration use `note_id`, never the current index-based widget id.

Normal row activation opens the note. In multi-select mode, activation toggles
the row instead.

Compact multi-select replaces the normal action group with:

- Done;
- Select all;
- Clear;
- Export selected.

New, Sync, and Import are not mixed into the selection task. Returning to
Library clears transient selection.

### Editor

Presentation:

- `‹ Notes · Edit note`;
- editable title;
- editable body;
- persistent textual save status;
- Save, Preview/Edit, and Context actions.

The body is the Editor's only scroll owner. At 60×20 it has a 5–6-row target
floor, may shrink further for a visible conflict callout, and grows with
available height. It uses soft wrapping, so no page-level horizontal scroll is
needed. Long Input/TextArea values may pan internally.

Save status uses explicit text and never relies only on color:

- `Unsaved changes`;
- `Saving…`;
- `Saved`;
- `Save failed — edits kept`;
- `Conflict — review the choices below`.

Saving never steals focus or moves the caret.

### Preview

Preview shares Editor's header, save status, and actions. Edit and Preview
surfaces are mounted once and visibility is toggled.

The Preview body is a focusable scrolling surface:

- Tab enters and leaves it;
- arrows and Page Up/Down scroll it;
- returning to Edit restores the prior body caret, selection, and scroll
  offset.

Preview renders the canonical session draft, never a separate content
snapshot.

### Context

Context is titled `‹ Editor · <current title>` and replaces Editor at every
width during Adapt. A simultaneous wide inspector belongs to Shape.

Context is one vertically scrollable region with:

1. **Properties** — editable keywords.
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

Create actions expose a running state and cannot be activated twice while the
request is in flight.

Successful create retains the existing behavior of clearing the active filter
to avoid a stale filtered snapshot.

### Sync

Sync retains the existing folder, direction, conflict, automatic-sync, run,
status, and activity controls. It is one vertically scrollable temporary
Navigator task with Back to Notes.

This task does not change sync ownership or policy.

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
   - delete confirmation;
   - loading and mutation-running flags.

Preview and conflict no longer own competing title/body/keyword snapshots.
They render the canonical draft. The persisted baseline remains a comparison
and metadata source; it does not overwrite a mounted draft during presentation
changes.

The pure, immutable state snapshots and transition helpers live in
`Library/library_notes_state.py`. Workers, timers, locks, services, and
navigation stay owned by `LibraryScreen`.

### Draft mutation

The existing editor-arming guard continues to ignore Textual's mount-time
change events.

Every genuine user mutation:

- updates the canonical draft;
- increments the draft revision;
- marks the session dirty;
- rearms the autosave debounce unless the session is in conflict.

Title/body widgets and the Context keyword field update the same draft. Save,
Preview, Context, export, copy, and Console handoff all read it.

### Revisioned, serialized save

Only one save request may be active for a note.

A save request captures:

- note id;
- expected optimistic-lock version;
- draft revision;
- sanitized persistence payload.

Save/autosave requests made while one is running are coalesced into one pending
follow-up instead of cancelling the in-flight request. This matters because
cancelling an asyncio worker cannot stop a service call already executing in a
thread.

On success:

- always accept the returned/new optimistic-lock version for the same active
  note;
- patch the persisted detail/list baseline with the payload that actually
  reached storage;
- if the current draft revision equals the saved revision, clear dirty and
  show `Saved`;
- if the current draft revision is newer, retain dirty, retain the newer raw
  draft, and run one follow-up save against the updated version.

On ordinary failure:

- retain the current draft and dirty state;
- show `Save failed — edits kept`;
- do not navigate away.

On optimistic-lock conflict:

- retain the latest canonical draft, including edits made after the conflicted
  request began;
- stop automatic retry;
- switch from Context or Preview to Editor;
- show a focusable explanatory callout plus Overwrite and Reload.

Overwrite fetches the fresh server version and attempts to save the current
canonical draft. Reload explicitly replaces the draft from the fresh persisted
detail. If the note was deleted elsewhere, the workflow returns to Navigator
with a warning.

### Flush and lifecycle

Back, row switching, rail switching, route navigation, and unmount continue to
use the pending-work flush barrier.

Flush:

1. cancels the debounce timer;
2. waits for the current serialized save;
3. runs the coalesced/latest save if a newer draft exists;
4. permits navigation only when no dirty/error/conflict state remains.

Resize, Preview, and Context toggles never force a save.

Transient Context, Preview, confirmation, loading, and compact-stage state are
not restored on a new screen instance. Selected note, list/editor workflow,
filter, and sort retain their existing persisted behavior. A restored editor
session opens Editor, not Context or Preview.

## Stable Composition

The Library rail and canvas remain mounted across compact shell changes.

After a note detail loads, the note session mounts these surfaces once:

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

The method does not query a database, call services, perform navigation, own
timers, or mutate global application state.

Navigator → Editor and Editor → Navigator may recompose because they are
workflow transitions. Resize, Preview, Context, status, and confirmation
changes do not.

## Focus and Safety

### Entry and return focus

- Explicit Notes selection → Navigator.
- Direct note link or resumed saved editor → Editor.
- Existing note → focus body after detail load.
- New note → focus title.
- Editor Back → focus the originating note row by `note_id`.
- If the row vanished or is filtered out → focus the filter.
- Context Back → focus the Context action that opened it.
- Preview Edit → restore body caret, selection, and scroll.
- Navigator Back → restore the selected Library rail row.

Programmatic focus changes explicitly scroll the target into view without
animation.

### Safe interruption

- Loading always retains a visible, focusable Back control.
- A stale detail result is discarded after Back or note switching.
- Save errors keep the current focus and draft.
- Conflict interruption focuses the explanatory callout, not Overwrite or
  Reload.
- Delete confirmation focuses Cancel by default.
- Delete confirmation is two-step and names the consequence.
- Delete remains disabled while the request runs.
- Delete failure returns to Context with the draft intact.
- Delete success returns to Navigator.

The app's supported Textual 8 `TextArea` defaults to `tab_behavior="focus"`,
so Tab and Shift+Tab can leave the body editor. Buttons use Enter/Space.

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
- `css/components/_agentic_terminal.tcss`, for the app bundle.

The stale fallback selectors are aligned with the live
`#library-shell-grid`, `#library-rail`, and `#library-canvas` selectors.

The source TCSS is authoritative. `css/tldw_cli_modular.tcss` is regenerated
with `tldw_chatbook/css/build_css.py` and never hand-edited. The generated diff
must contain only expected source-derived rules plus its timestamp.

At 60×20:

- every required visible control has positive width and height;
- every required visible control lies inside the terminal viewport;
- no page-level horizontal scrollbar is required;
- focused controls are visibly identifiable and scrolled onscreen;
- action labels remain readable;
- the note body or Preview retains a useful editing/reading area.

Hidden stable-composition surfaces are intentionally excluded from visible
geometry assertions.

## Accessibility Scope

Adapt guarantees:

- keyboard-only traversal with Tab/Shift+Tab;
- Enter/Space activation for controls;
- keyboard scrolling of list, body, Preview, Context, Create, and Sync;
- visible, non-obscuring focus;
- text labels for actions;
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
- draft mutation orchestration;
- serialized saves and autosave timer;
- optimistic-lock conflict resolution;
- focus restoration;
- service calls and navigation.

No general-purpose Notes controller or dedicated workbench is introduced in
Adapt. The pure state seam makes later extraction possible without prematurely
moving route or service ownership.

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
- responsive action groups;
- idempotent presentation synchronization.

### CSS

- Update `LibraryScreen.DEFAULT_CSS` for harness geometry.
- Update `css/components/_agentic_terminal.tcss`.
- Regenerate `css/tldw_cli_modular.tcss`.

### Tests

- Extend `Tests/Library/test_library_notes_state.py` for pure state.
- Extend `Tests/UI/test_library_shell.py` for Pilot, CSS, geometry, and
  concurrency, reusing its real-bundle `LibraryHarness` and gated service
  fakes.
- Do not create another Library harness or extract shared test infrastructure
  during Adapt.

## Verification

### Pure state tests

- genuine user edits increment draft revision;
- mount events do not;
- a current-revision success clears dirty;
- a stale-revision success retains dirty and requests one follow-up;
- conflict preserves the newest draft;
- Preview and Context derive from the canonical draft;
- transient presentation state is excluded from persistence.

### Pilot interaction tests

Run with the real generated stylesheet:

- 60×20 Library rail → Notes Navigator;
- Navigator → Editor → Context → Editor → Navigator;
- create Blank/template → Editor;
- compact Sync entry and Back;
- compact multi-select and Done;
- loading Back plus late-result rejection;
- Preview keyboard focus and scroll;
- Context keyword edit and autosave;
- edit while an earlier autosave is in flight;
- explicit Save while save is in flight;
- Back waiting for the complete save chain;
- save failure vetoing Back;
- conflict from Editor, Preview, and Context;
- Delete cancel, failure, and success;
- duplicate Create/Delete activation prevention;
- direct-note deep link;
- saved editor restore without Context/Preview restore.

### Responsive tests

- 60×20 fully usable;
- representative compact 80×24 and 100×30;
- breakpoint boundary 119/120, or the final measured boundary;
- existing 170×48 wide layout;
- dirty 170 → 60 → 170 round trip preserving:
  - draft text;
  - caret;
  - selection;
  - body scroll;
  - Preview/Edit choice;
  - selected note;
- crossing without changing breakpoint mode performs no state work.

### Geometry tests

- required visible regions stay inside terminal bounds;
- required visible controls have non-zero dimensions;
- no required action is clipped;
- active scroll owners have a usable viewport;
- no page-level horizontal overflow;
- programmatically focused controls are visible.

### Lifecycle and responsiveness tests

- repeatedly cross the breakpoint and toggle Context/Preview;
- assert Editor widget identities remain stable;
- assert worker and timer counts do not grow;
- assert no unbounded mount/remove churn;
- navigate away with a pending and an in-flight save;
- unmount/remount and verify only intended persistent state returns.

### Regression and completion checks

- focused Library/Notes state and UI tests;
- navigation, persistence, sync, import/export, and Console-handoff regressions;
- CSS generation, parsing, and source/bundle parity;
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
- missing note returns to Navigator;
- missing services disable/report the relevant action;
- stale async results are discarded;
- responsive presentation never decides storage authority.

## ADR Check

ADR required: no new ADR

ADR path: N/A

Reason: Adapt changes presentation and in-session safety inside accepted
Library/Database Note ownership. It changes no schema, file authority, sync
policy, route ownership, security boundary, provider/runtime boundary, or
long-lived dedicated-workbench structure.

Existing decisions:

- `backlog/decisions/011-chatbook-workbench-ui-system.md` — stable composition,
  explicit state snapshots, visible workflows, and responsiveness gates.
- `backlog/decisions/015-shell-destination-ia.md` — Notes remains owned by
  Library.
- `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery-replica.md`
  — File Notes keeps a separate future workbench/controller boundary and is
  not folded into this Database Notes session.
- `backlog/decisions/022-textual-8-runtime-floor.md` — supported Textual runtime
  for focus, layout, and widget APIs.

The later Shape migration to a dedicated Notes workbench must review whether
an ADR-021 amendment or a new Database/File Notes workbench ADR is required.
