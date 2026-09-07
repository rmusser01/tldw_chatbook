---
target: Notes screen/workflow toward Obsidian-capable UX
total_score: 22
max_score: 40
na_heuristics:
p0_count: 1
p1_count: 4
timestamp: 2026-07-30T21-12-35Z
slug: w-chatbook-widgets-library-library-notes-canvas-py
---
Method: dual-agent (A: `/root/notes_ux_assessment_a` · B: `/root/notes_evidence_assessment_b`)

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of system status | 2/4 | Local authority and final save states are visible, but dirty/saving state is not surfaced at the moment the user needs reassurance. |
| 2 | Match between system and real world | 3/4 | Notes, preview, templates, and sync are understandable; “keywords,” raw versions, and cycling controls expose the implementation model. |
| 3 | User control and freedom | 2/4 | Back, selection clearing, delete confirmation, overwrite, and reload exist; undo, trash, diff/merge, and revision restore do not. |
| 4 | Consistency and standards | 3/4 | The surface follows Chatbook’s terminal system, but Notes/New note are split across rail sections and autosave plus manual Save is unexplained. |
| 5 | Error prevention | 3/4 | Navigation flushes, optimistic locking, delete confirmation, and preview preservation are strong. |
| 6 | Recognition rather than recall | 2/4 | Actions are visible, but title/body are unlabeled and sort/sync policies must be discovered by cycling them. |
| 7 | Flexibility and efficiency | 2/4 | Keyboard traversal, templates, batch export, and Console handoff help; there is no note quick switcher, note command surface, pane model, or common note hotkeys. |
| 8 | Aesthetic and minimalist design | 2/4 | The terminal language is coherent, but five list actions and seven equal-weight editor actions compete at once. |
| 9 | Error recognition and recovery | 2/4 | Sync validation and conflict copy are useful, but zero-result filtering uses a false empty-vault message and conflict recovery has no diff, merge, save-both, history, or trash. |
| 10 | Help and documentation | 1/4 | Placeholders/tooltips do most of the teaching; there is no contextual help for autosave, sync consequences, shortcuts, or the Notes/New note split. |
| **Total** |  | **22/40** | **Acceptable — significant improvements needed** |

## Design Specificity Verdict

**Authored shell, generic notes core.**

The local authority label, explicit folder-sync direction/conflict policy, templates, and `Use in Console` handoff are genuinely Chatbook-specific. The main loop, however, is still a flat record list feeding a form-like editor. It does not yet feel like a connected, high-throughput knowledge workspace.

The unanchored design assessment and runtime evidence agree on the main problem: useful breadth is present, but navigation, retrieval, recovery, and knowledge relationships do not form a coherent expert workflow.

### Deterministic scan

`detect.mjs --json tldw_chatbook/Widgets/Library/library_notes_canvas.py` exited 0 with `[]`. That means the bundled markup detector found no applicable rule violations. It is not evidence that the Textual UI is accessible or responsive: the same runtime pass found a blocking narrow-terminal layout failure that a web/markup detector cannot model. No detector false positives were present.

### Visual evidence

Browser DOM overlays are not applicable to a terminal-rendered Textual app. Assessment B instead used `App.run_test()`/Pilot with the production stylesheet and captured SVGs for empty, template, editor, preview, multiselect, sync, delete-confirmation, and narrow states.

## Overall Impression

This is a credible CRUD, import/export, autosave, and sync surface. It is not yet an Obsidian-class knowledge environment. The biggest opportunity is not adding more toolbar buttons; it is promoting Notes from a Library sub-canvas into an adaptive knowledge workbench with a navigator, editor, context inspector, and command layer.

## What’s Working

1. **Data-loss prevention has a strong technical foundation.** Explicit save, debounced autosave, navigation flush, optimistic conflicts, preview preservation, and delete confirmation all exist.
2. **Local-first operations are meaningful.** Import, Markdown/text export, selected-note export, folder sync, direction/conflict policy, and validation cover real workflows.
3. **Chatbook has a differentiator Obsidian does not own.** `Use in Console` can turn notes into grounded agentic work; it should become a first-class contextual action rather than one of seven equal toolbar buttons.

## UAT Results

| Journey | Result | Evidence |
|---|---|---|
| Enter Library → Notes | Pass | Keyboard focus + Enter opened Notes at 170×48. |
| Empty state | Partial | Correctly shows `Notes (0)`, but tells the user to create a note without placing a Create action next to the instruction. |
| Blank/template creation | Pass | Blank path and eight configured templates rendered; template creation opened the populated editor. |
| Explicit save | Pass | A 260-character Unicode/markup-like title and 7,936-character body persisted intact at version 2. |
| Autosave | Pass for persistence; weak feedback | Final DB contained the sentinel at version 3, but dirty/saving communication is not immediate or persistent enough for user trust. |
| Preview/Edit | Pass | Preview preserved all 7,936 source characters and returned an `Edit` action. |
| Sort | Pass | Cycled Newest → Oldest → Title → Newest. |
| Filter | **Fail on zero results** | `filter: … · 0 results` is followed by `No notes yet. Create one to see it here.` even though notes exist. |
| Multiselect/export | Pass with interaction warning | Select all/clear/export scope worked by keyboard; a direct Pilot click on an off-viewport Export-selected control raised `OutOfBounds`. |
| Import | Pass | Picker/cancel and callback-seam import worked; Markdown, brackets, Arabic text, and emoji persisted. |
| Sync | Pass | Missing folder and file-as-folder gave actionable warnings; a valid local run ended `done · no changes`. |
| Delete | Pass | Cancel preserved the note; confirm removed it. Copy correctly warns it cannot be undone from Library. |
| Narrow terminal 60×20 | **Blocking fail** | Rail width became 168 cells; Notes began near x=172. The target screen was entirely off-screen and unusable. |
| Focused pytest suite | Not run | Runtime harness passed after stubbing two optional ML imports that aborted on unavailable Metal. |

## Priority Issues

### [P0] Notes becomes unreachable at narrow terminal widths

**Why it matters:** At 60×20, the rail and every Notes control are beyond the viewport. This is complete task failure in a terminal-native product.

**Fix:** Define and enforce supported breakpoints. Below the two-pane threshold, render one region at a time (`Navigator`, `Editor`, `Context`) with an explicit mode switch; collapse the global Library rail instead of preserving desktop widths. Add automated 60×20, 80×24, 100×30, and wide-terminal UAT.

**Suggested command:** `$impeccable adapt`

### [P1] The information architecture models records, not knowledge

**Why it matters:** Rows expose only title and age. There are no folders, structured properties, tags, internal links, backlinks, outline, related sources, or visible note provenance. A large collection becomes a flat database rather than a system for thinking.

**Fix:** Build an adaptive three-region workbench: note navigator/saved views, editor, and optional context inspector. Make internal links, backlinks, tags/properties, outline, related media/citations, and file/workspace authority first-class. On narrow screens these become explicit modes, not squeezed panes.

**Suggested command:** `$impeccable shape`

### [P1] Expert throughput is far below the Obsidian benchmark

**Why it matters:** The only Library binding is `u`. Common note loops require traversing global navigation and equal-weight toolbars. Obsidian’s quick switcher and command palette make note opening and command discovery keyboard-first.

**Fix:** Add a note-scoped command registry, searchable command palette, and quick switcher. Minimum shortcuts: new, save-now, switch/open recent, full-vault search, toggle/split preview, insert link, open backlinks, sync, and Console handoff. Show the relevant subset in the footer.

**Suggested command:** `$impeccable shape`

### [P1] Save confidence and recovery are incomplete

**Why it matters:** Persistence works, but users do not get a continuously truthful `Unsaved → Saving → Saved` story. A conflict offers only Overwrite or destructive Reload; deletion has no recovery in Library.

**Fix:** Put `Local · Unsaved`, `Saving…`, `Saved at 14:32`, `Conflict`, and `Offline` near the title. Add undo/redo, recoverable trash, revision history, conflict diff, `Keep mine`, `Keep theirs`, and `Save both`. Rename destructive conflict choices in user language.

**Suggested command:** `$impeccable harden`

### [P1] Empty, result, and scale states misrepresent reality

**Why it matters:** A zero-match search tells users there are no notes. The list and search are capped at 100 while the canvas header reports only rendered rows, so larger libraries can look complete when they are not.

**Fix:** Separate states: `No notes yet` with inline Create, `No matches for “…”` with Clear filter, and `Showing 100 of N` with pagination/load-more. Preserve the exact total in every filtered and capped state.

**Suggested command:** `$impeccable clarify`

## Cognitive Load

Cognitive load is high: roughly six of eight checks fail.

- The list toolbar exposes five actions; the editor exposes seven; conflict mode can expose nine.
- The create view presents Blank plus eight templates without search or grouping.
- Sort, sync direction, and conflict policy are cyclic controls; users must remember unseen choices.
- Title and body rely on position instead of durable labels.
- Notes browsing and note creation live in different rail sections.
- Select and Sync modes provide useful progressive disclosure, but the primary editor does not.

Keep Save/Preview visible, make Console handoff contextual, move export/copy/delete into a labeled More menu, add persistent field labels, and convert comma-separated Keywords into discoverable tag/property editing with autocomplete.

## Obsidian Capability Gap

The benchmark is capability, not visual imitation. Obsidian’s current core model includes keyboard quick switching and commands, rich search operators, internal links/backlinks, properties and tags, tabs/splits/workspaces, and recoverable history.

| Capability | Current Chatbook Notes | Parity gap |
|---|---|---|
| Local ownership/authority | Local DB plus explicit folder sync | Good foundation; per-note path/sync authority is not visible |
| Quick switcher/command palette | No note-scoped equivalent | Major |
| Markdown editing | TextArea + separate full preview | Partial; no live/split preview, link completion, or outline |
| Search | Submitted full-text filter, capped results | Major; no operators, snippets, facets, saved/recent queries |
| Links/backlinks/graph | No visible semantic note-link system | Critical foundation gap |
| Tags/properties | Comma-separated keywords | Major |
| Tabs/panes/workspaces | One canvas replaces list/editor/create/sync | Major |
| History/trash/recovery | Autosave + overwrite/reload conflict | Major |
| Templates | Eight bundled/user-configurable templates | Strong foundation; lacks management/search |
| Import/export/sync | Broad and functional | Relative strength |
| Extensibility | No visible Notes plugin/command API | Major, but later than core interaction parity |
| Canvas/database views | None | Later-stage gap after links/properties are sound |
| Agent workflow | `Use in Console` | Chatbook differentiator to elevate |

Do not begin with a graph or plugin marketplace. The shortest credible parity sequence is: adaptive workbench → command/quick-switch layer → links/backlinks/properties → recovery/history → tabs/splits and advanced search → graph/canvas/extensibility.

## Persona Red Flags

**Alex — keyboard-first power user:** No quick switcher, no note hotkeys, long focus traversal, cyclic controls, no tabs/panes, and limited batch operations. Alex becomes slower as the collection grows.

**Jordan — first-time user:** The empty state has no adjacent Create action; Browse → Notes and Create → New note are separated; title/body are unlabeled; autosave is unexplained; a zero-match search falsely implies the library is empty.

**Sam — accessibility-dependent user:** Real Buttons/Inputs, keyboard navigation, and text conflict copy are positives. Dirty/saving changes are not reliably announced, the conflict warning uses muted styling, the focus chain is long, and the narrow layout makes the entire surface unreachable.

**Rina — evidence researcher/builder:** Import, sync, local authority, templates, and Console handoff are valuable. Missing provenance, citations, internal links, backlinks, related media, version history, and side-by-side context prevent durable research synthesis.

## Minor Observations

- `Filter notes… (Enter)` is commendably explicit; live filtering or a global search command would still be faster.
- `sort: Newest ▸` signals cycling but hides the option set.
- Raw `v#` metadata has little user value without accessible history.
- `Overwrite` and `Reload` should become `Keep my version` and `Discard mine and reload`, with consequences visible.
- The empty-state Select button is correctly disabled.
- Long Unicode and markup-like content survived save/import, which is a meaningful robustness win.
- The product copy says Library is a hub and Notes should own deeper work, yet Notes currently has no first-class destination or shortcut context. That structural contradiction should be resolved before adding advanced features.

## Questions to Consider

1. Should Notes become a first-class destination/workbench, or must it remain a Library sub-surface?
2. Is the parity target Obsidian’s core knowledge loop (switch, link, retrieve, recover), or full ecosystem breadth including graph, Canvas/Bases, themes, and plugins?
3. Should Chatbook’s center of gravity be “Obsidian parity plus agents,” with provenance, citations, related media, and Console handoff more prominent than generic export controls?
