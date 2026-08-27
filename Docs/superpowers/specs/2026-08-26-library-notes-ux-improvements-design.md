# Library Notes Work-First UX Improvements

**Status:** Approved design

**Date:** 2026-08-26

**Area:** Library → Notes → Database Notes / Folder Files

## Goal

Make both Library Notes workspaces calmer and more efficient for sustained writing. The selected note should receive most of the available canvas, navigation should remain available without becoming dominant, and focusing a note body should communicate focus through its boundary instead of illuminating the entire editor.

The two note authorities retain their existing capabilities and storage behavior:

- **Database Notes** edits records owned by the application database.
- **Folder Files** edits files owned by a user-selected directory on disk.

This design changes presentation, pane behavior, and control hierarchy only. It does not merge those authorities or alter persistence, autosave, conflict, sync, recovery, or Git semantics.

## Problem Statement

The current Notes experience has three related problems:

1. The global Library rail and local Database Notes list can consume too much width after a note is opened, leaving the content canvas visually subordinate to navigation.
2. Folder Files uses a persistent tree/editor split but does not offer the same manual navigator control as Database Notes.
3. The shared focused `TextArea` styling changes the full editor background. On the two large note-body editors this produces excessive luminance and visual distraction while typing.

Both workspaces also expose many low-frequency actions alongside frequent writing actions. Every capability remains useful, but displaying all of them at once weakens hierarchy and reduces room for the note itself.

## Design Principles

The design applies the following Nielsen Norman Group usability principles:

- **Visibility of system status:** ownership, save state, conflicts, read-only state, and consequential Git activity remain visible as concise text.
- **Match between system and the real world:** Database Notes and Folder Files use authority-specific language rather than implying they are the same storage system.
- **User control and freedom:** automatic space-making is reversible, manual pane choices win, and destructive/recovery flows keep their existing guards.
- **Consistency and standards:** both authorities use the same workbench grammar—navigator, grip, primary work canvas—without inventing capabilities that one authority does not have.
- **Error prevention and recovery:** conflicts and uncertain operations replace ordinary actions with contextual, safe next steps.
- **Recognition rather than recall:** frequent actions stay visible; lower-frequency actions move into named, stable groups rather than disappearing.
- **Aesthetic and minimalist design:** chrome and metadata are reduced in the writing view while information with current decision value remains present.
- **Progressive disclosure:** metadata, export, destructive actions, Git, and maintenance appear in secondary sections or task-specific flows.

References:

- [10 Usability Heuristics for User Interface Design](https://www.nngroup.com/articles/ten-usability-heuristics/)
- [Progressive Disclosure](https://www.nngroup.com/articles/progressive-disclosure/)
- [Aesthetic and Minimalist Design](https://www.nngroup.com/articles/aesthetic-minimalist-design/)

## Scope

### In scope

- Work-first wide-screen behavior after opening a Database Note or Folder File.
- A manually collapsible Folder Files tree with a grip matching the Database Notes list behavior.
- Independent, profile-local persistence for the Database Notes list and Folder Files tree preferences.
- A boundary-only focused state for the two note-body editors.
- Reorganization of existing controls into frequent actions and clearly labeled secondary groups.
- More explicit text status for authority, saves, conflicts, Git activity, and recovery.
- Preservation of editor and navigator state across collapse and responsive transitions.
- Targeted automated and live Textual verification.

### Out of scope

- Database, schema, migration, or data-model changes.
- Changes to note or file storage authority.
- New autosave, sync, conflict-resolution, recovery, or Git behavior.
- A Markdown preview for Folder Files.
- A new generic pane framework or third-party dependency.
- New global or screen-level keyboard shortcuts.
- Redesign of compact Library navigation outside the existing staged flow.

## Shared Workbench Grammar

Both Notes authorities use the same spatial model without pretending their features are identical:

```text
wide browse
┌──────────────┬──────────────────┬────────────────────────────────┐
│ Library      │ Local navigator  │ Select or create a note/file   │
│ destinations │ list or tree     │                                │
└──────────────┴──────────────────┴────────────────────────────────┘

wide editing, effective work-first layout
┌─┬──────────────────┬─────────────────────────────────────────────┐
│›│ Local navigator  │ Note/file work canvas                       │
│ │ list or tree     │                                             │
└─┴──────────────────┴─────────────────────────────────────────────┘
  Library grip

wide editing with both navigators collapsed
┌─┬─┬──────────────────────────────────────────────────────────────┐
│›│›│ Note/file work canvas                                        │
└─┴─┴──────────────────────────────────────────────────────────────┘
  Library and local navigator grips
```

The global Library rail yields width once when a user enters an editing task on a wide layout. The local Database Notes list or Folder Files tree remains available according to its own preference and can be collapsed independently.

## Responsive Behavior

| Context | Library rail | Local navigator | Work canvas |
| --- | --- | --- | --- |
| Wide browse | Saved preference | Saved preference | Empty/select prompt |
| Wide editing entry | May close through a transient work-first override | Saved preference | Receives remaining width |
| Wide after manual Library expansion | Open; manual choice wins | Unchanged | Reflows without losing state |
| Compact Library | Existing one-stage navigation | Existing list-to-editor flow | Existing compact stage |
| Folder Files below its existing internal narrow threshold (approximately 80 allocated columns) | Existing Library behavior | Existing navigator-to-editor stage | Existing editor stage |
| Width restored | Transient responsive overrides clear | Saved preferences return | Same selection and editor state |

Responsive and work-first overrides are effective layout state only. They never write user preferences.

### Requested versus effective visibility

Pane state is modeled as:

- **Requested visibility:** the user's saved preference.
- **Effective visibility:** requested visibility after transient work-first and responsive rules are applied.

Rules:

1. If the saved Library preference is open, opening a note/file on a wide layout may temporarily close it.
2. If the user expands the Library rail during that editing task, the manual action cancels the transient override. Because the saved request was already open, no preference write is necessary.
3. If the saved Library preference is closed and the user expands it, the existing manual persistence semantics apply.
4. Creating, importing, choosing a root, browsing an empty authority, or entering sync/setup state does not trigger work-first collapse.
5. When the reason for a transient override ends, the saved request becomes effective again.

The Database Notes list and Folder Files tree preferences are independent and profile-local. The existing Notes reader configuration owner, coercion, and configuration-failure behavior remain authoritative; the Folder Files addition is a destination-specific requested state alongside the existing Database Notes list state.

## Database Notes

### Primary work view

`Edit` remains the first and default work mode. `Preview` and `Info` remain available modes. The work header prioritizes:

- selected note title;
- database authority and save/draft/conflict status;
- `Edit`, `Preview`, and `Info` mode controls;
- `Save` when applicable;
- `Use in Console`.

The body receives the dominant vertical and horizontal space. Metadata and low-frequency actions do not occupy permanent side-by-side canvas space.

### Secondary information architecture

The existing capabilities are preserved in stable groups within `Info`:

1. **Properties** — keywords and existing note metadata.
2. **Reuse & Export** — existing copy/export/reuse actions.
3. **Danger** — existing deletion or other destructive actions, retaining confirmations and guards.

An empty, clean note does not display a false dirty or save-warning state. Status copy must distinguish persisted, unsaved, saving, conflict, and failed states rather than relying on color alone.

## Folder Files

### Primary work view

Folder Files exposes `Edit` and `Info`; it does not gain a Markdown `Preview` mode. The existing large-file preview/guard behavior is a safety mechanism, not a new authoring mode.

The work header prioritizes:

- selected file name or path;
- folder/file authority and save/read-only/conflict state;
- `Edit` and `Info` controls;
- the applicable save action;
- concise consequential Git status when it has current decision value.

The tree header retains frequent creation/navigation affordances. The new tree grip provides the same visible open/closed affordance as the Database Notes list grip.

### Secondary information architecture

Existing capabilities are preserved in stable `Info` groups:

1. **File details & path** — authority, selected path, file properties, and current disk state.
2. **File actions** — existing move, copy, reload, refresh, protect, compare, or resolution actions when applicable.
3. **Session Git** — existing session Git controls and details.
4. **Danger** — destructive actions and their existing guards.

New, Move, and Save Copy use named task flows that reveal target-path inputs only when invoked. Restore remains its existing contextual recovery action and does not invent a target-path requirement. Compare, resolve, reload, and other conditional actions appear only in states where they are meaningful.

Git controls may be secondary, but consequential Git state remains visible in the primary work header as text—for example `Git · 3 changes`, `Checking…`, `Pushing…`, `Push failed`, or an explicit uncertain state. Ordinary success does not demand persistent attention.

## Note-Body Focus Treatment

The boundary-only focus treatment applies exactly to:

- `#library-note-body`
- `#file-notes-editor`

When either body editor receives focus:

- its resting background remains unchanged;
- its boundary changes to the existing semantic focus border color;
- focus does not change geometry or cause layout shift;
- text selection and cursor rendering remain unchanged;
- focus remains distinguishable in every supported theme without color being the only cue.

Title, path, search, filter, and other compact inputs retain their existing focused-fill treatment. Other `TextArea` widgets elsewhere in the application are unchanged.

## State Preservation and Focus

Collapsing either navigation pane or crossing a responsive threshold retains mounted widget state rather than reconstructing the workspace. The following state must survive:

- selected note or file identity;
- filter, search, and sort state;
- list scroll position and tree expansion;
- draft content, cursor, selection, undo history, and editor scroll position;
- dirty, conflict, read-only, or recovery state;
- running Git state and its result;
- semantic focus where the target remains visible.

If the currently focused control becomes hidden, focus moves to that pane's visible grip. Reopening the pane restores access without changing the current note/file. `Back` and `Esc` continue through the existing navigation and unsaved-change guards.

No new shortcut is introduced. In particular, the UI must not advertise `Ctrl+S` or another terminal-convention binding that the screen does not implement. Existing global focus-navigation behavior, including `F6`, remains intact.

## Status, Errors, and Recovery

Text labels accompany semantic color for all consequential states:

- database versus folder authority;
- saved, unsaved, saving, failed, or conflict;
- read-only or protected;
- offline/unavailable storage;
- Git changes, progress, failure, or uncertain completion.

Recovery callouts state:

1. what happened;
2. the impact on the current draft/file;
3. the safest available next action.

When a conflict or failed operation requires attention, contextual recovery actions replace ordinary controls in that region. Existing confirmation, overwrite, and navigation guards remain unchanged.

Empty Database Notes and missing/unselected Folder Files roots keep the Library rail available and present explicit next steps. They do not enter work-first mode.

## Architecture and Ownership

The change reuses the current Notes architecture:

- `LibraryScreen` continues to own global Library requested preferences, transient work-first state, and destination orchestration.
- `LibraryAdaptiveReaderShell` continues to own the Database Notes Library/list/work geometry and retained-widget behavior.
- `LibraryFileNotesWorkspace` gains retained requested/effective tree visibility and a matching tree grip while preserving its existing internal narrow staging.
- `LibraryNotesCanvas` and `LibraryFileNotesWorkspace` reorganize their existing actions into the approved primary and secondary hierarchy.
- Component CSS applies the body-editor focus exception by exact widget ID and uses existing semantic `ds-*` tokens.

No second generic pane framework is introduced. No service, route, draft-registry, sync-engine, database, file-authority, or Git boundary changes.

## Expected Code Surface

Implementation is expected to stay within the existing Library Notes surface and its focused tests, primarily:

- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py` only if the existing API needs a small extension
- `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- the existing Library/forms component TCSS files
- focused Library Notes unit, integration, and Textual pilot tests
- Library Notes user-guide pages where interaction guidance changes

The implementation plan must inspect the current dirty worktree and adjust to concurrent changes rather than assuming these files are pristine.

## Acceptance Criteria

- [ ] Opening a selected note/file on a wide layout gives the work canvas priority by applying the approved transient Library-rail behavior.
- [ ] Manual Library expansion wins over a transient work-first collapse and does not accidentally change an already-open saved preference.
- [ ] The Folder Files tree has a visible, operable collapse/expand grip on wide layouts.
- [ ] Database Notes list and Folder Files tree preferences are independent, profile-local, and unaffected by responsive-only changes.
- [ ] Collapse/expand and width transitions preserve the current identity, editor draft, cursor/selection/undo/scroll, navigator state, and contextual operation state.
- [ ] Database Notes retains Edit, Preview, and Info; Folder Files retains Edit and Info and does not advertise a Markdown Preview mode.
- [ ] Every existing Notes capability remains reachable through the approved primary controls, stable secondary sections, or contextual recovery flows.
- [ ] Git controls are secondary while consequential Git status remains visible as text.
- [ ] Only the two note-body editors keep their resting background when focused and communicate focus through a stable, theme-appropriate boundary.
- [ ] Compact and internally narrow flows retain their existing staged navigation behavior.
- [ ] Empty/setup states remain navigable and do not trigger work-first collapse.
- [ ] No unsupported keyboard shortcut or invented capability appears in the UI or documentation.
- [ ] Targeted automated checks and live Textual walkthroughs cover the approved wide, compact, resize, focus, status, and recovery behaviors.

## Verification Strategy

Targeted verification is sufficient unless a full test sweep is separately requested:

1. Unit-test requested versus effective visibility, including manual override and preference-write boundaries.
2. Pilot the Database Notes and Folder Files layouts at wide, compact, threshold-crossing, and restored widths.
3. Verify identity and draft preservation while collapsing panes and resizing.
4. Verify focus evacuation when a pane hides its focused child.
5. Assert the two body editors retain their resting background on focus while compact inputs and unrelated text areas keep existing styles.
6. Check focus-border contrast and text status in supported light and dark themes.
7. Exercise clean, dirty, saving, conflict, failed, read-only, Git-progress, Git-failure, and uncertain Git states.
8. Confirm all existing actions remain reachable and retain their current guards.
9. Confirm Folder Files has no invented Markdown Preview and footers contain no unsupported key hints.
10. Perform a live Textual walkthrough of both authorities at representative wide and compact sizes.

## Risks and Mitigations

- **Automatic collapse feels surprising.** Limit it to wide editing entry, keep the grip visible, and let any manual expansion cancel the override.
- **Preference corruption during responsive changes.** Keep requested and effective state separate and test that automatic transitions do not write configuration.
- **Hidden operations reduce discoverability.** Use stable, named `Info` groups and keep consequential status in the primary header.
- **State loss during layout changes.** Retain mounted widgets and test draft, cursor, selection, undo, scroll, selection identity, and active operations.
- **Focus becomes too subtle after removing the fill.** Use the semantic focus boundary, preserve non-color structure, and verify contrast across themes.
- **The two authorities drift into false parity.** Share only the workbench grammar; retain authority-specific modes, wording, actions, and status.

## ADR Check

**ADR required:** No

**ADR path:** `backlog/decisions/086-library-adaptive-reader-shell.md`; `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`

**Reason:** This is a bounded refinement of the adaptive reader shell and progressive-disclosure model already recorded by ADR-086 and ADR-076. It does not change storage, synchronization, data ownership, service contracts, security boundaries, or long-lived application architecture.

## Resolved Decisions

- The work-first wide layout is the approved direction.
- Edit is the first Database Notes view.
- Boundary-only focus applies only to the two note-body editors.
- Existing capabilities are reorganized, not removed.
- Folder Files gains a collapsible tree but not a Markdown Preview mode.
- Automatic layout state is transient; manual intent and saved preferences remain authoritative.
- No new ADR, dependency, pane framework, or keybinding is required.
