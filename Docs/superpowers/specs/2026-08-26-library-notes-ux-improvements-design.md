# Library Notes Work-First UX Improvements

**Status:** Approved design; amended after independent critique

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
- One shared adaptive shell for Database Notes and Folder Files geometry.
- A manually collapsible Folder Files tree with the same five-column grip grammar as the Database Notes list.
- Independent, profile-local persistence for the Database Notes list and Folder Files tree preferences.
- A boundary-only focused state for the two note-body editors.
- Reorganization of existing controls into frequent actions and clearly labeled secondary groups.
- More explicit text status for authority, saves, conflicts, Git activity, and recovery.
- Preservation of editor and navigator state across collapse and responsive transitions.
- Removal of the Notes-specific `Ctrl+S` binding and its footer/F1 hints while preserving the visible Save operation.
- Targeted automated and live Textual verification.

### Out of scope

- Database, schema, migration, or data-model changes.
- Changes to note or file storage authority.
- New autosave, sync, conflict-resolution, recovery, or Git behavior.
- A Markdown preview for Folder Files.
- A new generic pane framework or third-party dependency.
- New global or screen-level keyboard shortcuts; this change removes one prohibited Notes binding but introduces no replacement binding.
- Redesign of compact Library navigation outside the existing staged flow.

## Shared Workbench Grammar

Both Notes authorities use the same spatial model without pretending their features are identical:

```text
wide browse
┌──────────────┬─────┬──────────────────┬─────┬────────────────────┐
│ Library      │grip │ Local navigator  │grip │ Select/create      │
│ destinations │     │ list or tree     │     │                    │
└──────────────┴─────┴──────────────────┴─────┴────────────────────┘

wide editing, effective work-first layout
┌─────┬──────────────────┬─────┬──────────────────────────────────┐
│ --->│ Local navigator  │grip │ Note/file work canvas            │
│     │ list or tree     │     │                                  │
└─────┴──────────────────┴─────┴──────────────────────────────────┘

wide editing with both navigators collapsed
┌─────┬─────┬──────────────────────────────────────────────────────┐
│ --->│ --->│ Note/file work canvas                                │
└─────┴─────┴──────────────────────────────────────────────────────┘
```

The diagrams are structural, not to scale. Both shell grips occupy the existing `PANE_GRIP_WIDTH` of five terminal columns and remain visible for pointer, keyboard, tooltip, and focus recovery.

`LibraryAdaptiveReaderShell` owns this geometry for both authorities. Its local navigator slot contains the Database Notes list or Folder Files tree; its work slot contains the corresponding retained editor surface. Authority-owned widgets keep their services, state, modes, and actions. The global Library rail yields width once when a user enters an editing task on a wide layout, while the local navigator remains available according to its own independent preference.

## Responsive Behavior

| Context | Library rail | Local navigator | Work canvas |
| --- | --- | --- | --- |
| Wide browse | Saved preference | Saved preference | Empty/select prompt |
| Wide editing entry at `width >= LIBRARY_NOTES_COMPACT_BREAKPOINT` (`120`) | Closes once when the work-session transition activates and the saved request is open | Saved authority-specific preference | Receives remaining width |
| Wide after manual Library expansion | Open; manual choice wins | Unchanged | Reflows without losing state |
| Compact Library | Existing one-stage navigation | Existing list-to-editor flow | Existing compact stage |
| Folder Files with allocated width `< FILE_NOTES_NARROW_BREAKPOINT` (`80`) | Shell policy applies | Navigator stage requests shell `items` priority; editor stage clears that priority | Work remains the permanent shell region |
| Width restored | Responsive geometry is recomputed from saved requests; work-session cancellation is unchanged | Saved authority-specific preference returns | Same selection and editor state |

Responsive and work-first overrides are effective layout state only. They never write user preferences.

### Preference and persistence contract

Pane state is modeled as:

- **Requested visibility:** the user's saved preference.
- **Work-session visibility:** the requested Library value after the transient Notes work-session override.
- **Effective visibility:** work-session visibility after explicit pane priority, responsive constraints, and hysteresis are applied.

The exact profile-local keys are:

| Pane | Configuration | `LibraryScreen` runtime authority |
| --- | --- | --- |
| Shared Library rail | `[library.reader].library_open` and `library_width` | existing `library` authority |
| Database Notes list | `[library.notes_reader].items_open` and `items_width` | existing `notes_items` authority |
| Folder Files tree | `[library.notes_reader].files_tree_open` and `files_tree_width` | new `notes_file_items` authority |

At runtime `LibraryScreen` owns the requested mirrors for all three authorities. For grip-mutated open keys it also owns generation counters, per-authority locks, optimistic updates, durable writes, rollback, and configuration-failure notification. It consumes normalized widths but does not provide an in-Library width editor. The Folder Files keys use the existing adaptive-reader boolean and width bounds. Database and Folder grip writes cannot supersede or roll back each other.

The Settings Appearance surface remains the custom-width mutation owner required by ADR-086. It gains staged fields named `library_notes_files_tree_open` and `library_notes_files_tree_width`, rendered as **Folder Files tree pane** and **Folder Files tree width** beside the existing Notes Items fields. Settings writes those values to `[library.notes_reader].files_tree_open/files_tree_width`; Library grips may write `files_tree_open` but never width. Configuration normalization supports `TLDW_LIBRARY_NOTES_READER_FILES_TREE_OPEN` and `TLDW_LIBRARY_NOTES_READER_FILES_TREE_WIDTH`, matching the existing destination-reader environment convention. Reset-to-default and validation include both fields.

Manual grips persist requested open/closed state only. Work-first, responsive, narrow-stage, focus-priority, and hysteresis changes never persist.

### Work-first session reducer

A small pure Notes-specific reducer owns the transient work-first lifecycle. It has three states:

- `inactive` — no work-first override;
- `active` — force the effective Library request closed for the current Notes work session;
- `manually_cancelled` — do not auto-collapse again during the current Notes work session.

Transitions are deterministic:

| Current state | Event | Next state | Preference write |
| --- | --- | --- | --- |
| `inactive` | A selected Database Note editor or successfully opened Folder File becomes editable while allocated width is at least `120` | `active` | none |
| `inactive` | Create/import/root selection/loading/empty/setup, or an item opens below `120` | `inactive` | none |
| `active` | User expands Library while saved request is already open | `manually_cancelled` | none |
| `active` | User expands Library while saved request is closed | `manually_cancelled` | persist the explicit open request normally |
| `active` or `manually_cancelled` | Selection changes, Edit/Preview/Info/Manage changes, save/conflict/recovery changes, or any resize | unchanged | none |
| any | Reach the authority-specific cleared-selection predicate, change Database/Folder authority, change Folder root, or leave Notes | `inactive` | none |

Opening another item during the same Notes work session does not rearm collapse after manual cancellation. The cleared-selection predicates are exact:

- **Database Notes:** selected/loaded note identity becomes `None` and the work pane shows the select/create prompt.
- **Folder Files:** the opened-file identity is explicitly closed/cleared. Narrow `Back to navigator` does not clear the retained opened file and therefore does not reset the work session.

Reaching one of those cleared-selection states establishes a new session boundary, so a later editable selection may activate work-first behavior again.

Layout precedence is:

1. load persisted requested visibility and widths;
2. apply `active` work-first override to the Library request only;
3. apply an explicit one-shot pane-priority request from navigation or a grip;
4. resolve responsive constraints and hysteresis in the shared adaptive layout policy.

An explicit manual request always cancels or outranks the work-first override, but hard width constraints may still show one pane at a time. A resize never arms, cancels, or persists work-session state.

The existing Folder `<80` behavior migrates into shell priority requests rather than remaining a second display/geometry controller inside `LibraryFileNotesWorkspace`. In its navigator stage, Folder Files requests `items` priority; opening a file clears that priority and lets the permanent work pane win. At `80` or wider, normal saved visibility applies.

## Database Notes

### Primary work view

`Edit` remains the first and default work mode. `Preview` and `Info` remain available modes. The work header prioritizes:

- selected note title;
- database authority and save/draft/conflict status;
- `Edit`, `Preview`, and `Info` mode controls;
- `Save` when applicable;
- `Use in Console`.

On a wide layout the header uses at most two logical rows:

1. identity and authority;
2. content status plus the grouped mode controls and primary task actions.

The mode controls form one group and Save/Use in Console form a separate task group. A blocking content state replaces ordinary task actions with its safe next action. The selected title may ellipsize after its useful prefix; authority, status, and recovery copy may not. Compact presentation may stack the same groups vertically without changing their order.

The body receives the dominant vertical and horizontal space. Metadata and low-frequency actions do not occupy permanent side-by-side canvas space.

### Secondary information architecture

The existing capabilities are preserved in stable groups within `Info`:

1. **Properties** — keywords and existing note metadata.
2. **Reuse & Export** — existing copy/export/reuse actions.
3. **Danger** — existing deletion or other destructive actions, retaining confirmations and guards.

An empty, clean note does not display a false dirty or save-warning state. Status copy must distinguish persisted, unsaved, saving, conflict, and failed states rather than relying on color alone.

### Database capability placement

| Existing capability | Approved location | Visibility and retained behavior |
| --- | --- | --- |
| Database / Folder authority switch | Existing Notes authority control | Remains outside editor modes; changing authority resets work-session state but preserves each authority's retained selection and navigator state. |
| Filter, clear filter, sort, note selection | Shared-shell Database local navigator | Existing controls, values, sort choices, selection identity, result paging, and focus behavior remain unchanged. |
| Folder tree and note placement management | Database local navigator contextual rows/dialogs | New/rename/move/remove/restore folder and add/move/remove placement keep their existing services, confirmations, receipts, and disabled reasons. |
| New note | Existing Database navigator `New` action | Existing creation, draft, discard, validation, and selection behavior remains unchanged. |
| Add from files… | Existing Database navigator transfer actions | Existing Import once and lasting-sync setup/review workflows remain reachable. Setup/import/sync does not activate work-first collapse until an editable note is selected. |
| Manage sync folders | Existing conditional Database navigator transfer action | Remains visible under its current configured-root predicate and opens the existing sync-roots workflow. |
| Last import | Existing conditional Database navigator transfer action | Remains visible while an import receipt is retained and opens the existing receipt workflow. |
| Whole-source Export | Existing Database navigator `Export` action | Existing export scope, service, progress, and error behavior remains unchanged. |
| Bulk selection and Export selected | Existing Database local-navigator selection presentation | Existing selected set, Select all shown, Clear, Done, disabled reason, export service, progress, partial failure, and recovery behavior remain unchanged. No bulk-delete capability is invented. |
| Undo note/folder deletion | Existing Database navigator receipt/action | Existing receipt lifetime, generation fencing, and recovery behavior remain unchanged. |
| Title/body editing | `Edit` | Primary work surface; existing draft, autosave, validation, and conflict guards remain mounted. |
| Save | Primary task group | Visible when applicable; existing save service and disabled reasons remain authoritative. |
| Markdown rendering | `Preview` | Retained presentation; no hidden render backlog is introduced. |
| Keywords and metadata | `Info` → **Properties** | Available without consuming the Edit canvas. |
| Use in Console | Primary task group | Existing handoff behavior and status remain unchanged. |
| Copy/export/reuse | `Info` → **Reuse & Export** | Existing services, filenames, and validation remain unchanged. |
| Delete | `Info` → **Danger** | Existing confirmation, running, and cancellation guards remain unchanged. |
| Conflict/recovery | Contextual recovery region | Replaces ordinary task actions until resolved; draft and selected identity remain mounted. |

## Folder Files

### Primary work view

Folder Files exposes `Edit` and `Manage`; it does not gain a Markdown `Preview` mode. The existing large-file preview/guard behavior is a safety mechanism, not a new authoring mode.

The work header prioritizes:

- selected file name or path;
- folder/file authority and save/read-only/conflict state;
- `Edit` and `Manage` controls;
- autosave state and any contextual recovery action; Folder Files gains no ordinary manual Save control;
- concise consequential Git status when it has current decision value.

The wide Folder header follows the same two-row priority as Database Notes. The friendly folder/file identity may ellipsize after a useful prefix; the exact selected path remains available in `Manage`. Content/recovery status and its safe next action never ellipsize. The tree header retains frequent creation/navigation affordances. Its grip is the shared shell's five-column items grip rather than a second Folder-specific grip implementation.

### Secondary information architecture

Existing capabilities are preserved in stable `Manage` groups:

1. **File details & path** — authority, selected path, file properties, and current disk state.
2. **File actions** — existing move, copy, reload, refresh, protect, compare, or resolution actions when applicable.
3. **Session Git** — existing session Git controls and details.
4. **Danger** — destructive actions and their existing guards.

New, Move, and Save Copy use named task flows that reveal target-path inputs only when invoked. Restore remains its existing contextual recovery action and does not invent a target-path requirement. Compare, resolve, reload, and other conditional actions appear only in states where they are meaningful.

Git controls may be secondary, but consequential Git state remains visible in the primary work header as text—for example `Git · 3 changes`, `Checking…`, `Pushing…`, `Push failed`, or an explicit uncertain state. Ordinary success does not demand persistent attention.

### Folder Files capability placement

Only one named path task is open at a time: `none | new | move | save_copy`. Opening another path task replaces the current task only after the existing unsaved/confirmation guard permits it. `Cancel` or guarded `Esc` closes the task and returns focus to the control that opened it.

| Existing capability | Approved location | Visibility predicate and retained contract |
| --- | --- | --- |
| Choose folder / root details | Browse/setup header | Visible when no usable root exists or root details are requested; root validation and transition guards remain unchanged. |
| Search, tree, selection | Shared shell local navigator | Retained Folder navigator region; search value, expansion, selection, and scroll survive collapse and authority switches. |
| Editing autosave | `Edit` status channel | Existing debounce, flush-before-leave, conflict, error, and draft-preservation behavior remains authoritative; no manual Save button is added. |
| New | Tree header → named `new` task | Reveals target path only after invocation; existing path validation and create guard remain. |
| Move | `Manage` → **File actions** → named `move` task | Available only for an applicable selected file; existing path validation and move guard remain. |
| Save Copy | Contextual recovery and `Manage` → **File actions** → named `save_copy` task | Remains immediately reachable during conflict/reload recovery where it is a safe action; never overwrites an existing target. |
| Restore | Contextual file action | Appears only for the retained deleted-file selection; no invented target-path requirement. |
| Compare / Resolve conflict | Contextual recovery region | Appears only for disk divergence/conflict; ordinary actions are replaced until the user chooses a safe path. |
| Reload | `Manage` → **File actions**, then contextual confirmation | Existing dirty-draft confirmation, Cancel, and Discard/load-disk behavior remain mounted. |
| Protect / Refresh | `Manage` → **File actions** | Existing maintenance semantics and disabled reasons remain unchanged. |
| Delete | `Manage` → **Danger** | Existing deletion guard, deleted-path retention, and restore affordance remain unchanged. |
| Review session changes / Git panel | `Manage` → **Session Git** | The existing Git widget remains mounted while hidden; running and uncertain operations retain state. Consequential summary remains in the primary header. |
| Keep editing / Save draft as new note / Discard and load disk | Contextual recovery region | Retains current conflict and reload services; Save Copy stays reachable and focus returns to the editor or originating action after cancellation. |

`Manage`, the editor, the retained Git panel, and contextual recovery regions toggle presentation without remounting active editor, Git, or recovery state.

## Note-Body Focus Treatment

The boundary-only focus treatment applies exactly to:

- `#library-note-body`
- `#file-notes-editor`

When either body editor receives focus:

- its computed background is identical to its unfocused computed background;
- a same-geometry `heavy` outline using the existing semantic `ds-*` focus token supplies the non-color boundary cue;
- the focused boundary has at least `3:1` contrast against every adjacent surface in each shipped theme;
- focus does not change geometry or cause layout shift;
- text selection and cursor rendering remain unchanged;
- focus remains distinguishable through boundary weight as well as color.

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

If the currently focused control becomes hidden, focus moves to that pane's visible grip and the shell retains the last valid focused descendant for that pane. Activating the grip to reopen the pane restores that target when it still exists and is enabled; otherwise it uses the pane's first valid focus target. A responsive-only restoration never steals focus. `Back` and `Esc` continue through the existing navigation and unsaved-change guards.

No new shortcut is introduced. The Notes-specific `Ctrl+S` binding and its Notes footer/F1 hints are removed because screen-level terminal-convention bindings are prohibited. The visible Save control remains reachable through normal focus navigation, including `F6`. The separately gated Skill-editor binding is outside this Notes-only scope.

## Status, Errors, and Recovery

Text labels accompany semantic color for all consequential states:

- database versus folder authority;
- saved, unsaved, saving, failed, or conflict;
- read-only or protected;
- offline/unavailable storage;
- Git changes, progress, failure, or uncertain completion.

The header has two independent status channels:

1. **Content and recovery:** `conflict` → unavailable/read-only blocker → save failure → saving → dirty → saved/clean.
2. **Authority and Git:** authority/root state is always named; Git uses `failure or uncertain` → running → changes, while ordinary clean success is omitted.

The first applicable state in each order is rendered. Git never replaces a content conflict, failed save, or safe recovery action. When space is constrained, the content/recovery channel remains in the header and the complete authority/Git details remain in `Info` or `Manage`.

Recovery callouts state:

1. what happened;
2. the impact on the current draft/file;
3. the safest available next action.

When a conflict or failed operation requires attention, contextual recovery actions replace ordinary controls in that region. Existing confirmation, overwrite, and navigation guards remain unchanged.

Empty Database Notes and missing/unselected Folder Files roots keep the Library rail available and present explicit next steps. They do not enter work-first mode.

## Architecture and Ownership

The change reuses the current Notes architecture:

- `LibraryScreen` continues to own destination orchestration, the shared Library request, both authority-specific local-navigator requests, and their independent persistence authorities.
- A small pure Notes work-session reducer owns only `inactive | active | manually_cancelled` and accepts named events. It performs no UI queries, data access, or preference writes.
- One `LibraryAdaptiveReaderShell` implementation owns Library/list/work geometry, both five-column grips, responsive resolution, width application, and focus evacuation/restoration for both Notes authorities.
- The shell's retained items/work hosts present the Database Notes list/canvas or Folder Files tree/editor regions according to the active authority without rebuilding the underlying editors.
- `LibraryFileNotesWorkspace` remains the Folder domain owner and supplies distinct retained navigator and work regions to the shell. It no longer sets competing navigator/editor widths or display geometry; its exact `<80` stage becomes shell priority input.
- `LibraryNotesCanvas` and `LibraryFileNotesWorkspace` reorganize their existing actions into the approved primary and secondary hierarchy without changing their services or guards.
- Component CSS applies the body-editor focus exception by exact widget ID and uses existing semantic `ds-*` tokens.

No second shell, generic editor framework, or schema-driven action framework is introduced. No service, route, draft-registry, sync-engine, database, file-authority, or Git boundary changes.

## Expected Code Surface

Implementation is expected to stay within the existing Library Notes surface and its focused tests, primarily:

- `tldw_chatbook/UI/Screens/library_screen.py`
- a Notes-specific pure work-session reducer adjacent to the Library screen under `tldw_chatbook/UI/Library_Modules/`
- `tldw_chatbook/Utils/adaptive_reader_state.py` for the Folder profile/priority contract only if the existing profile API cannot express it
- `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py` for authority-neutral focus restoration or retained slot support
- `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- `tldw_chatbook/config.py` for `files_tree_open` / `files_tree_width` defaults and normalization
- `tldw_chatbook/UI/Screens/settings_appearance_defaults.py` and `settings_screen.py` for Folder-tree staged open/width fields, validation, reset, and durable Settings writes
- the existing Library/forms component TCSS files
- focused reducer, configuration, Settings Appearance, Library Notes, shell, Folder Files, accessibility, and Textual pilot tests
- Library Notes user-guide pages where interaction guidance changes

The implementation plan must inspect the current dirty worktree and adjust to concurrent changes rather than assuming these files are pristine.

## Acceptance Criteria

- [ ] Database Notes and Folder Files use `LibraryAdaptiveReaderShell` as their only geometry, grip, responsive, and focus-evacuation owner.
- [ ] Opening an editable note/file at width `>=120` activates work-first collapse exactly once per approved Notes work session.
- [ ] Manual Library expansion moves the session to `manually_cancelled`, survives selection/mode/save/conflict/resize changes, and does not accidentally write an already-open saved preference.
- [ ] Database selected/loaded identity clearing, Folder opened-file identity explicitly clearing, authority/root change, or leaving Notes resets the work session; narrow Folder Back does not reset it, and no other event rearms collapse.
- [ ] The Folder Files tree uses the shared, five-column collapse/expand grip on wide layouts.
- [ ] `[library.notes_reader].items_open/items_width` and `files_tree_open/files_tree_width` are independent, profile-local, normalized, and written through distinct persistence authorities.
- [ ] Settings Appearance owns the Folder-tree width editor and exposes `library_notes_files_tree_open/width`; environment normalization supports the two declared `TLDW_LIBRARY_NOTES_READER_FILES_TREE_*` overrides.
- [ ] Work-first, responsive, `<80` Folder stage, focus-priority, and hysteresis transitions never write preferences.
- [ ] Collapse/expand and width transitions preserve the current identity, editor draft, cursor/selection/undo/scroll, navigator state, and contextual operation state.
- [ ] Database Notes retains Edit, Preview, and Info; Folder Files exposes Edit and Manage and does not advertise a Markdown Preview mode.
- [ ] Every existing Notes capability maps to the approved primary controls, secondary groups, named path task, or contextual recovery region with its existing service and guard intact.
- [ ] Existing Database filter/sort, folder/placement, New, Add from files, Manage sync folders, Last import, whole-source Export, bulk selection/Export selected, and undo capabilities remain in their declared navigator/workflow locations; no bulk delete is invented.
- [ ] Folder Files retains autosave and gains no ordinary manual Save button.
- [ ] Only one Folder path task is active at a time; Cancel/guarded Esc returns focus to its origin; Save Copy remains reachable in applicable conflict/reload recovery.
- [ ] Git controls are secondary while consequential Git status remains visible as text.
- [ ] Content/recovery and authority/Git status channels follow the declared precedence; blocking status and the safe next action are never ellipsized.
- [ ] Only the two note-body editors have identical focused/unfocused computed backgrounds, a geometry-stable heavy focus outline, and at least 3:1 boundary contrast in every shipped theme.
- [ ] Compact and exact `<80` Folder flows retain their existing staged navigation behavior through shared-shell priority rather than a second geometry controller.
- [ ] Manual grip expansion restores the last valid focus target; responsive restoration never steals focus.
- [ ] Empty/setup states remain navigable and do not trigger work-first collapse.
- [ ] Notes no longer binds or advertises `Ctrl+S`; Save remains visibly and keyboard-accessibly reachable, and no replacement shortcut or invented capability is introduced.
- [ ] Targeted automated checks and live Textual walkthroughs cover the approved wide, compact, resize, focus, status, and recovery behaviors.

## Verification Strategy

Targeted verification is sufficient unless a full test sweep is separately requested:

1. Unit-test the pure `inactive | active | manually_cancelled` reducer across activation, manual expansion, item/mode/save/conflict/resize preservation, exact Database/Folder cleared-identity resets, and the non-resetting narrow Folder Back event.
2. Prove requested versus work-session versus effective visibility and verify that only explicit requested-state changes enter persistence.
3. Test distinct `notes_items` and `notes_file_items` grip writes, generation races, rollback, normalization, configuration-failure copy, Settings staged open/width save/reset/validation, and both Folder-tree environment overrides.
4. Pilot the shared-shell Database Notes and Folder Files layouts at `160`, `120`, `119`, `80`, `79`, and restored widths.
5. Verify retained identity, draft, cursor, selection, undo, scroll, tree expansion, Git, and recovery state across pane collapse, authority changes, and resize.
6. Verify focus evacuation, manual restoration to the last valid target, fallback focus, and no focus theft during responsive restoration.
7. Assert focused/unfocused computed-background equality for the two body IDs, a heavy outline, and at least `3:1` boundary contrast across shipped light and dark themes.
8. Assert compact inputs and unrelated `TextArea` widgets retain their existing focused fill.
9. Exercise the declared content and Git status precedence across clean, dirty, saving, conflict, failed, read-only/offline, Git-progress, Git-failure, and uncertain combinations.
10. Inventory every capability against its incumbent stable control/event and approved location, including Database New/Add from files/Manage sync folders/Last import/whole-source Export/Export selected with an explicit no-bulk-delete assertion, Folder autosave with no ordinary Save button, named path-task exclusivity, cancellation focus, and Save Copy recovery reachability.
11. Confirm Folder Files has no invented Markdown Preview and Notes footer/F1 help no longer contains `Ctrl+S`.
12. Perform a live Textual walkthrough of both authorities at representative wide and compact sizes.

## Risks and Mitigations

- **Automatic collapse feels surprising.** Activate it once per explicit work session, keep the five-column grip visible, and preserve `manually_cancelled` until a named reset event.
- **Preference corruption during transient changes.** Keep requested, work-session, and effective state separate; use distinct Database/Folder persistence authorities; test that automatic transitions do not write configuration.
- **Shared-shell migration remounts Folder state.** Supply retained Folder navigator/work regions to the shell and inventory identity, draft, Git, recovery, and focus preservation before changing geometry.
- **Hidden operations reduce discoverability.** Use stable Database `Info` and Folder `Manage` groups, truthful disabled reasons, and consequential status in the primary header.
- **State loss during layout changes.** Retain mounted widgets and test draft, cursor, selection, undo, scroll, selection identity, and active operations.
- **Focus becomes too subtle after removing the fill.** Use a heavy, geometry-stable semantic outline and verify at least 3:1 boundary contrast across shipped themes.
- **The two authorities drift into false parity.** Share only the workbench grammar; retain authority-specific modes, wording, actions, and status.

## ADR Check

**ADR required:** No

**ADR path:** `backlog/decisions/086-library-adaptive-reader-shell.md`; `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`

**Reason:** Folder Files now adopts ADR-086's required single Library-local shell instead of introducing a parallel geometry engine. The new Folder tree preference is an authority-specific requested-visibility/width pair under the existing `[library.notes_reader]` owner, matching ADR-086's persistence contract. The work remains a bounded refinement of ADR-086 and ADR-076 and does not change storage, synchronization, data ownership, service contracts, security boundaries, or long-lived application architecture.

## Resolved Decisions

- The work-first wide layout is the approved direction.
- Edit is the first Database Notes view.
- Boundary-only focus applies only to the two note-body editors.
- Existing capabilities are reorganized, not removed.
- Folder Files uses `Edit` and `Manage`, gains a collapsible tree through the shared adaptive shell, and does not gain a Markdown Preview mode.
- Automatic collapse runs once per Notes work session; manual expansion suppresses it until browse/no selection, authority/root change, or Notes exit.
- Database Notes list and Folder Files tree preferences use independent keys and persistence authorities.
- The two body editors use background-stable, heavy-outline focus with a 3:1 boundary-contrast minimum.
- The Notes `Ctrl+S` binding and its hints are removed without adding a replacement shortcut.
- No new ADR, dependency, or pane framework is required.
