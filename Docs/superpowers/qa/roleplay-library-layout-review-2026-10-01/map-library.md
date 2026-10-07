# Library layout system: structural map (as a template for Roleplay)

Worktree `origin/dev @ 84247cb843`. Paths are relative to the repo root. This was a read-only study. The geometry numbers come from running the pure resolvers, not from reading the code alone (see §2). The one screen-level test I ran errored in fixture setup; that is noted where it applies.

## 0. Why it reads as space-efficient (summary)

1. **Little chrome above the workbench.** Library uses one 1-row `#library-header-line` (`Library | Local`, `UI/Screens/library_screen.py:15017-15021`). It does not use the multi-row `DestinationHeader`, a purpose line, or a mode strip. Personas uses all three (`personas_screen.py:1547-1585`).
2. **The rail is dense.** Every rail row is one line holding a title, a count and an optional gloss, inside collapsible sections. The rail width is an exact cell count from a bounded policy, not a fraction (`Utils/library_rail_width.py:46-62`).
3. **Collapsed panes cost very little.** The ordinary "Nav" handle is 3 cells (`Widgets/Library/library_rail.py:538-576`) and the adaptive grips are 5 cells each. Personas' collapsed handles cost 13 + 11 cells (`personas_screen.py:608-609`).
4. **Reader routes use three panes with a fixed-width list.** The list gets about 50 cells, the work pane takes the rest, and the side panes collapse in a fixed order (Library first, then Items). The work pane never collapses.
5. **Empty panes donate their width.** A work pane with nothing open gives its columns to the list (`Utils/adaptive_reader_state.py:519-528`). The landing hub is capped at 96 cells.

## 1. Shell anatomy

### 1.1 DOM tree (ordinary, non-reader routes)

`LibraryScreen(BaseAppScreen)` is at `library_screen.py:996`. Its CSS is lazy: `CSS_PATH` points at `css/screen_agentic_library.tcss` (`:1002-1008`). `compose_content` is at `:14943`.

```
#screen-content (BaseAppScreen)
├─ #library-header-line            Static .destination-status-row      :15017
├─ #library-lifecycle-status       (only while onboarding)              :15023
├─ artifacts share strip / install progress / Notes source strip (conditional) :15031-15055
└─ #library-shell-grid  Horizontal .ds-panel.destination-workbench      :15057
   ├─ #library-rail-handle  LibraryNavigationRailHandle (3 cells, "N/a/v") :15454
   ├─ #library-rail         LibraryRail .destination-workbench-pane     :15461
   └─ #library-canvas       Vertical .w-13fr min-width:40               :15477
      ├─ #library-emergency-return  LibraryEmergencyReturn ("‹ Library", hidden ≥64) :15487
      └─ #library-canvas-route-content  (landing | list | viewer | editor | import) :15489
```

### 1.2 Rail (`Widgets/Library/library_rail.py`)

`LibraryRail(PostRecomposeCallback, RecomposeCaptureGuard, Vertical)` is at `:579`. Its `compose` is at `:1065` and builds, in order:

- **Heading row.** `#library-rail-heading` holds the `#library-rail-heading-label` "Navigation" and the compact `#library-rail-collapse` "Collapse" button (`:1072-1092`). On adaptive routes the screen hides Collapse and lets the grip take over (`library_screen.py:6494-6498`).
- **Starter mode.** A new profile in the STARTER or UNKNOWN lifecycle gets only Import…, New note and `#library-rail-explore-all` (`:1093-1097`).
- **Top action.** `top_action_factory` supplies `#library-ingest-top-button` "Import…", with `variant=primary` (`library_screen.py:14864-14881`).
- **Search row.** `#library-rail-search-row` holds `LibraryRailSearchInput#library-search-input` and `#library-search-clear` "x" (`:1099-1126`).
  - `/` focuses the box from anywhere outside a text field (`library_screen.py:8749-8860`, F-012).
  - A second `/` inside the box selects all instead of typing a slash (`library_rail.py:455-509`).
  - Submitting the box opens the Search/RAG canvas.
- **Sections.** Each section is a `DestinationRailSectionHeader` (shared, ▾/▸) plus `#library-rail-section-body-<id>` (`:1337-1357`). The section titles are Browse, Artifacts, Create, Study and Import / Export (`Library/library_shell_state.py:756-763`). Open/closed state persists to `library.rail_state.sections` (`library_screen.py:21785-21790`; model in `Library/library_rail_state.py:38-71`).
- **Rows.** Each row is a `LibraryRailRowButton#library-row-<row_id>` (`:512-535`, `:1313-1335`) built from the frozen `LibraryRailRow` (`library_shell_state.py:366-420`). Its fields are title, count/count_known/count_display/count_emphasis, count_loading "(…)", count_pending, subtitle (the gloss) and short_title.
  - Label fitting happens per button in `on_resize` (`:526-535`), using `_row_label` (`:1000-1063`).
  - Fitting order: the gloss drops first and is shown whole or not at all. Then the title switches to its `short_title` (Conversations→"Chats" `:521`, Collections→"Captures" `:596`, Flashcards→"Cards" `:705`). The title is ellipsized only as a last resort. The count is never clipped.
  - Glosses: Media "your files" `:504`, Prompts "reuse" `:557`, Skills "AI add-ons" `:572`, Collections "saved captures" `:584`, Search/RAG "find all" `:606`.
  - The selected row is marked ▸ and gets `.library-rail-row-selected`.
- **Details.** A disclosure that is collapsed by default and contains Status, counts, runtime, DB sizes/Diagnostics, Workspaces, Chunking Lab and similar (`:1127-1160`, `:1190-1290`).
- **Fold cue.** A docked-bottom `#library-rail-fold-cue` "▾ scroll for more" (`:1162-1180`).

### 1.3 Collapse

There are two different collapse grammars (see §7):

- **Ordinary routes.** "Collapse" swaps the rail for `#library-rail-handle`, whose label is drawn vertically as "N\na\nv" (`library_rail.py:538-576`). `#library-rail-open` restores the rail and focuses `#library-search-input` (`library_screen.py:21793-21823`). The state lives in `_library_rail_collapsed` (`:3977`), is **session-only** and is never persisted.
- **Adaptive routes.** `LibraryAdaptiveReaderPaneGrip` 5-cell grips, which paint "Nav" through `LIBRARY_PANE_GRIP_NAMES` (`Widgets/Library/library_adaptive_reader_shell.py:36`). The state is **persisted** in `[library.reader].library_open`.
- **Notes editor exception.** At ≥120 columns the Notes editor deliberately collapses the rail to Nav (`library_screen.py:4970-4976`; `Docs/User_Guide/library.md:313-318`).

### 1.4 Canvas kinds and footer

- **Landing.** `LibraryLandingCanvas#library-landing-canvas` (`Widgets/Library/library_entry_canvases.py:110`) has the sections Continue, Needs attention, From your Library and Quick actions (`#library-hub-*`, `:181-375`).
- **Other canvases.** List, viewer/editor and import canvases are chosen by `shell.canvas_kind` (`library_screen.py:15489-15560+`).
- **Footer.** Footer chips come from `_library_footer_shortcuts_for_current_state` (`:4704-4870`). They are registered on every recompose through the shared `BaseAppScreen.register_footer_shortcuts` (`UI/Navigation/base_app_screen.py:465`), called from `library_screen.py:8724-8737`.
  - While a text field has focus, the footer leads with "typing in field" and parks printable verbs under "after esc:".
  - Below 64 columns, "esc back to Library" or "esc rail" replaces the `/` and F6 chips.
  - `LibraryPaneVisibilityChanged` re-registers the footer whenever a pane's applied visibility flips (`:7332-7350`).

## 2. Width model (constants and where each is enforced)

| Rule | Value | Implementation |
|---|---|---|
| Default rail | `floor((3W+8)/16)+5`, clamped 29–39 | `Utils/library_rail_width.py:46-62` (`LIBRARY_DEFAULT_*` `:13-20`) |
| Custom rail | 24–48, opt-in | `_validate_saved_width` `:81-88`; ordinary routes compress it to `max(24, min(saved, W-40))` `:138-144` |
| Canvas floor | 40 | `LIBRARY_CANVAS_MIN_WIDTH` `:19`; canvas `min_width=40`, `.w-13fr` (`library_screen.py:15480-15481`) |
| Emergency single stage | `W < 64` (24+40) | `ordinary_emergency_required` `:65-78` |
| Ordinary apply | equality-guarded inline width/min/max | `_sync_library_ordinary_rail_width_contract` (`library_screen.py:6516-6541`) → `LibraryRail.apply_ordinary_width_contract` (`library_rail.py:707-735`) |
| Items (list) default | 50 (fit test uses 40, then grows by +10 spare) | `ITEMS_TARGET_WIDTH`/`ITEMS_DEFAULT_EXTRA_WIDTH` (`Utils/adaptive_reader_state.py:31-34`), fit logic `:327-343` |
| List bounds | min 32 / comfort 56 / max 72 | `AdaptiveReaderLayoutProfile` `:43-98` |
| Work floor | 44 default; 46 Media; 48 Prompts/Skills/Notes/Collections | profiles at `UI/Library_Modules/screen_constants.py:48-77`, `Library/library_media_reader_state.py:33-43` |
| Grips | 2×5, **always reserved** | `grip_width = 2*profile.grip_width` `:327` |
| Hysteresis | 4 cells, re-open only | `LAYOUT_HYSTERESIS_WIDTH` `:37`, `:408-428` |
| Empty pane donates width | `reader_has_item=False` → list absorbs width down to the work floor | `:519-528` (task-31979); rationale in `css/features/_library.tcss:1293-1312` |
| List-first below 64 | Notes/Media/Artifacts keep the list rather than an empty reader | `:431-472` (task-32065) |
| `list_grows` | Media/Notes split Reader surplus into the list up to comfort | `:494-517` |
| Landing measure | `max-width: $ds-size-96` (96) | `css/features/_library.tcss:1310-1312`; test `Tests/UI/test_library_crit9_shell.py:129-145` |
| Persistence | `[library.reader]` (shared Library pane + custom flag + width); `[library.<dest>_reader]` holds each destination's Items; effective geometry is never written | `config.py:2362-2430`; Settings model `UI/Screens/settings_appearance_defaults.py:66-82` |

**Resolved widths.** I computed these by running `resolve_adaptive_reader_layout` and `resolve_ordinary_rail_contract` (pure functions). W is the shell content width, which is the terminal width minus 4 at ≥120 columns. All cells.

| W | Ordinary rail / canvas | Prompts (item open) Lib / Items / Work | Prompts (empty) | Notes (item open) |
|---|---|---|---|---|
| 231 (235 col) | 39 / 192 | 39 / 50 / 132 | 39 / 134 / 48 | 39 / 64 / 118 |
| 166 (170 col) | 36 / 130 | 36 / 50 / 70 | 36 / 72 / 48 | 36 / 61 / 59 |
| 116 (120 col) | 29 / 87 | **–** / 56 / 50 | – / 58 / 48 | – / 58 / 48 |
| 100 | 29 / 71 | – / 42 / 48 | – / 42 / 48 | – / 42 / 48 |
| 80 | 29 / 51 | – / – / 70 | – / – / 70 | – / – / 70 |
| 60 | single stage | – / – / 50 | – / – / 50 | – / 50 / 0 (list-first) |

Two consequences:

- On reader routes the Library pane is closed by default below roughly 126 terminal columns. It needs 10 + 24 + 40 + 48 = 122 content cells.
- At 80 columns only the work pane and its two grips are visible.

## 3. Adaptive reader shell (ADR-086) and the narrow stage

- **ADR.** `backlog/decisions/086-library-adaptive-reader-shell.md` sets up one *Library-local* shell. It owns three regions, two full-height grips, effective geometry, responsive collapse, focus integration and late-bound slots. It owns no records, drafts or actions. The rail and list are independently collapsible and the work pane is permanent. Library visibility is shared; list visibility and width are per destination. The ADR explicitly "does not authorize sharing an application-wide implementation with Watchlists" and requires a new ADR for wider sharing. ADR-084 (`084-library-media-reader-ia.md`) is the Media precursor; ADR-172 later folded Artifacts in.
- **Widget.** `LibraryAdaptiveReaderShell(Horizontal)` is at `Widgets/Library/library_adaptive_reader_shell.py:232-455`.
  - Constructor `(library, items, work, layout, *, id_prefix, library_label, items_label)`; ids come out as `<prefix>-library-grip` and `<prefix>-items-grip` (`:235-298`).
  - `compose` yields library | grip | items | grip | work (`:300-306`).
  - `sync_layout` (`:342-455`) patches display, disabled and exact width/min/max in place. It evacuates focus to the grip when a focused pane hides. On a manual reopen it restores the pane's last-focused descendant (`_last_focused_descendant`, `:317-326`). It posts `LibraryPaneVisibilityChanged` (`:52-70`).
  - Messages: `PaneToggleRequested` (`:40`), `AdaptiveReaderShellResized` (`:48`, from `on_resize`). The grips respond to Enter and Space (`:73-230`).
- **Instances.**
  - `LibraryBrowseReaderShell#library-browse-reader-shell` is resident across Media and Notes and marks the active route with marker classes (`Widgets/Library/library_browse_reader_shell.py:60-200`; `library_screen.py:15111`, `:15441`).
  - Plain shells exist for Collections, Prompts, Skills and Conversations (`library_screen.py:15221`, `:15269`, `:15316`, `:15396`).
  - `LibraryArtifactsReaderShell` is at `Widgets/Library/library_artifacts_reader_shell.py:27`. File Notes builds its own shells (`library_file_notes_workspace.py:1699`, `:1733`).
- **Single stage below 64.**
  - **Ordinary routes.** `_apply_library_emergency_geometry` (`library_screen.py:5790-5870`) shows only the rail *or* the canvas, chosen from focus. It shows `LibraryEmergencyReturn` ("‹ Library"/"< Library", `Widgets/Library/library_emergency_return.py`). Eligibility is computed in one place (`:5634-5672`) and drives the button, Escape `check_action` and the footer "esc rail" chip.
  - **Adaptive routes.** The resolver itself closes the Library pane. `action_library_narrow_stage_return` re-opens it through `PaneToggleRequested("library")` (`:7352-7360`). The gate `_library_narrow_stage_return_active` (`:7260-7330`) stands down when an earlier Escape owner is live. Footer: "esc back to Library".
- **Restoring on widen.**
  - Capture: `_capture_library_emergency_restore_receipt` (`:5728-5788`) records the focused id, a source id probed from hard-coded attribute names, and the scroll owner's offset, tagged with a generation.
  - Restore: `_restore_library_emergency_receipt` (`:5674`) runs when width reaches 64 or more.
  - Any newer user interaction bumps the generation (`:5616-5632`, `:8242-8258`), so a stale restore cannot steal focus.
  - Width-only changes are equality-guarded and do no data work (test `test_library_63_64_width_only_transition_does_no_non_layout_work`, `Tests/UI/test_library_shell.py:5264`).
- **F6 pane cycle.**
  - Binding: app-level F6 (`app.py:929`) and screen-level Shift+F6 (`library_screen.py:1022-1028`).
  - Actions `:8960-8972` and `:9154-9185` call the shared `focus_relative_workbench_pane` (`Widgets/workbench_focus.py:20-53`).
  - Each destination has its own `WorkbenchPaneTarget` tuple, which names a preferred child per pane (`library_screen.py:1402-1515`).
  - Artifacts computes its targets dynamically and substitutes a pane's grip when that pane is closed (`library_artifacts_reader_shell.py:43-72`). That is the better pattern.

## 4. Reusable primitives versus Library-specific code

| Tier | Piece | Reuse status |
|---|---|---|
| **Already app-shared** | `Widgets/destination_rail.py`: `DestinationRailHandle` `:48`, `DestinationRailSectionHeader` `:163`, ▾/▸ glyphs `:44-45` (ADR-034) | Used by Personas, Home, Console, Lab, Evals |
| | `Widgets/workbench_focus.py` `WorkbenchPaneTarget`/`focus_relative_workbench_pane`; app F6 | Library, Personas, Settings, Workflows |
| | `BaseAppScreen.register_footer_shortcuts` (`base_app_screen.py:465`) | any screen (Personas drives `set_shortcut_context` directly) |
| | `UI/Workbench/workbench_widgets.py` (DestinationHeader `:164`, ModeStrip `:366`, RecoveryCallout `:457`, WorkbenchFrame `:623`); `css/layout/_destination.tcss` `.destination-workbench(-pane)` | ADR-011 primitives. **Library does not use DestinationHeader.** |
| **Library-named but domain-free** (low-effort lift) | `Utils/adaptive_reader_state.py` (pure resolver and profiles; already in `Utils`, imported by `config.py:79`) | Directly reusable: one new `AdaptiveReaderLayoutProfile` per Roleplay destination |
| | `Utils/library_rail_width.py` (pure) | Directly reusable |
| | `LibraryAdaptiveReaderShell` (imports only `Utils`) | Python is reusable. **CSS is not**: rules are `library-`-prefixed in `css/features/_library.tcss:109-176`, and `build_css.py:440-455` (`prefixes={"library": ("library",)}`) routes them into the lazily loaded `screen_agentic_library.tcss`, which only LibraryScreen loads. Pin them or rename to neutral tokens. |
| | `LibraryEmergencyReturn` (37 lines), `library_choice_strip.py`, `library_canvas_sync.PostRecomposeCallback`, `SelectAllOnFocusingClickInput`/`LibraryRailSearchInput` (`library_rail.py:380-509`) | Reusable as-is |
| **Library-specific** | `LibraryRail`: depends on `LibraryShellState`, `LibraryLifecycle` starter mode, the Details/Workspaces body and `library-*` ids (`library_rail.py:579-640`) | Needs extraction into a generic "destination nav rail" (the row/section model `LibraryRailRow`/`LibraryRailSectionState` is already generic) or a Roleplay twin |
| | Screen glue in `library_screen.py` (35,902 lines): emergency/narrow stage, footer projection, a 15-entry Escape chain, a per-destination pane-toggle if-ladder (`:7615-7700+`), the preference snapshot (`:6543+`) | **Not reusable.** Copy the *contracts*, not the code. |
| | `LibraryArtifactsController` (`UI/Library_Modules/library_artifacts_controller.py:140-225`) | **The cleanest adopter template**: `build_shell`, `resize()` → resolver → `shell.sync_layout`, `toggle_pane()` with off-thread persistence. About 90 lines. |

**Has any other screen adopted it?** None outside Library imports `LibraryAdaptiveReaderShell`, the resolver or `library_rail_width` (grep). The exceptions are config and settings, which read only the width preferences. Related screens copy the grammar instead:

- **Watchlists.** An independent copy of the same grammar: `UI/Watchlists_Modules/pane_grip.py` (`WatchlistsPaneGrip`, 5 cells) and `region_layout.py` (`PANE_GRIP_WIDTH = 5`), kept separate on purpose under ADR-084 and ADR-086.
- **MCP hub.** Its own rail | canvas | inspector triad, which stacks the inspector below 120 columns (ADR-148, `UI/MCP_Modules/mcp_workbench.py`).
- **Evals.** Its own unrelated `LibraryRail` class with the same name (`UI/Evals/library_rail.py:310`).
- **Home.** Its own rail built on `DestinationRailSectionHeader` (`Widgets/Home/home_rail.py`).
- **Artifacts.** Folded *into* Library (ADR-172) rather than adopting the shell on its own screen.
- **Personas.** Already uses `DestinationRailHandle`, `DestinationHeader`, `DestinationModeStrip` and `WorkbenchPaneTarget` (`personas_screen.py:171`, `:178`, `:361`, `:1123-1160`, `:1540-1735`). It has a rail | work | inspector workbench, not rail | list | work.

**Effort implications for Roleplay:**

1. Write an ADR amendment to ADR-086 that promotes the shell out of "Library-local".
2. Neutralize or pin the shell CSS tokens.
3. Write Roleplay profiles and an Artifacts-style controller per destination (Characters, Dictionaries, Lore, Personas).
4. Generalize `LibraryRail`, or build a rail from `DestinationRailSectionHeader` plus the row model.
5. Re-implement the narrow-stage, footer and Escape contracts cleanly.

## 5. Authoring in Library (the closest analogs to the Roleplay editors)

- **Placement.** The editor *is* the work pane, and the list stays mounted beside it (ADR-086 "keep list-to-work takeovers" was rejected).
  - The work-pane classes subclass the list canvases and switch by mode: `LibraryNoteWorkPane(LibraryNotesCanvas)` (`Widgets/Library/library_note_work_pane.py:19-63`) and `LibraryPromptWorkPane(LibraryPromptsListCanvas)` (`library_prompt_work_pane.py:13-58`).
  - Empty-state copy: "Select a note/prompt to edit it here."
  - At ≥120 columns the Notes editor collapses the rail to make room (§1.3).
- **Modes inside the work pane** (a `ds-toolbar` button strip):
  - Notes: Edit / Preview / Info (`library_notes_canvas.py:2830-2850`).
  - Prompts: Basic / Advanced / Info (`library_prompts_canvas.py:1319-1345`).
  - Skills: Overview / Edit / Trust / Files (`library_skill_work_pane.py:178-190`).
  - Secondary metadata (keywords, backlinks, provenance, collections) lives in the **Info** mode, not in a separate inspector column.
- **Header.**
  - Notes: "‹ Notes" / "‹ Back to list" plus a title row and authority status (`library_notes_canvas.py:634-645`, `:2752-2800`).
  - Prompts: "‹ Back to list", a Name field and "Use in Console" in the header (`library_prompts_canvas.py:1296-1375`).
- **Per-item actions follow a state-machine action bar.** Only the valid verbs are visible at any moment.
  - Prompts `#library-prompt-editor-actions` (`:1628-1720`):
    - new item: Save prompt + Cancel
    - dirty: Save changes + Discard changes
    - clean: **More actions** disclosure with Export…, Copy Markdown, Duplicate, Collections, History and Delete (danger)
    - conflict: Save as new + Reload
  - Skills uses the same machine (`library_skills_canvas.py:1122-1150`).
  - Notes puts Save, Use in Console and Discard new note in the header toolbar (`library_notes_canvas.py:2828-2870`). Info holds "Reuse & Export" and "Danger ▸ Delete", and there is an inline conflict region and an inline delete confirmation (`:2950-3125`).
- **Unsaved changes. There are two models:**
  - Notes **autosaves**, with a status channel such as "Saved 15:23" (`UI/Library_Modules/library_notes_controller.py:1334-1480`).
  - Prompts and Skills require an **explicit Save**. Back and Escape are **vetoed** while the editor is dirty, with a warning toast: "Unsaved Prompt changes — Save or Discard changes first." (`library_prompts_controller.py:3489-3640`; copy at `screen_constants.py:209-222`; Skills twin at `library_skills_controller.py:761`).
  - The backup subsystem's refusal probe lists these editors (`Backup_Recovery/unsaved_editors.py:520-663`).
- **Version history. Prompts only:**
  - `LibraryPromptHistoryRegion` is a paged `Collapsible` (`UI/Library_Modules/prompt_history_region.py:45`). It is mounted in the Advanced extras (`library_prompts_canvas.py:1616-1626`) and is reachable from More actions ▸ History.
  - Restore is gated on a clean editor (`Library/library_prompts_state.py:1325`, `history_restore_gate`).
  - Notes have no history in Library (File Notes has a Session Git panel). Skills show only a version field (`library_skill_work_pane.py:101`).

## 6. Tests that pin the layout (templates)

- **Pure policy.**
  - `Tests/Library/test_library_rail_width.py`: the bounds constant tuple `(36, 24, 39, 48, 40, 64)` `:23-32`, a projection table `:35-54`, compression `:92` and the emergency floor `:69`.
  - `Tests/Library/test_library_adaptive_reader_state.py` (860 lines).
  - `test_library_rail_state.py`, `test_library_shell_state.py`.
- **Shell in isolation (best template).** `Tests/UI/test_library_adaptive_reader_shell.py` uses a `_ProbeApp(ConsolidatedCSSApp)` with three `Static` panes. It covers:
  - grip widths `:150`
  - toggles from Enter, Space and click `:259`
  - focus evacuation and restore `:312-430`
  - no geometry writes for an unchanged layout `:436`
  - regions staying inside bounds at representative widths `:571`
  - CSS ownership `:594-700`
  - list growth `:790`
- **Screen-level.**
  - `Tests/UI/test_library_crit9_shell.py`:
    - empty pane donates columns `:91`
    - landing at 96 `:129`
    - Escape below 64 `:180`
    - footer chip appearing and standing down `:221-354`
    - the `/` chip in a closed pane `:440`
  - `test_library_crit10_layout.py`:
    - grips paint their names `:193`
    - no overlap `:215`
    - one chip at 60 columns `:615`
    - narrow-stage return `:469`
  - `test_library_layout_repair.py:46` (five-cell click targets).
  - `Tests/UI/test_library_shell.py`:
    - emergency-stage matrix `:4463-5500`
    - ordinary/adaptive width matrices `:5609-5840`
  - Also: `test_library_footer_focus.py`, `test_library_rail_focus_visibility.py`, `Tests/UI/test_destination_rail.py`, `test_destination_shells.py`, `test_workbench_focus_help.py`.
- **Caveat.** The production ordinary width matrix (`test_library_shell.py:5654-5664`, which expects rail widths 24/24/31/34) was written on 2026-08-26 (`47c4ee83ec`), *before* `7fc19997e3` (2026-09-09) raised the default to 29–39. It still pins the old numbers; the pure resolver gives 29/29/36/39. I tried to run it and it errored in setup (`RecoveryRequired: raw_source_selection_changed`, a Backup_Recovery harness problem), so whether it is red on dev is unconfirmed. Do not copy its expected values.

## 7. Weaknesses not to copy (observations)

1. **Two collapse grammars.** Ordinary routes use a text "Collapse" button with a 3-cell handle and do not persist the state. Adaptive routes use 5-cell grips and persist it. The guide says the Nav handle is "five cells wide" (`library.md:224-227`), but the ordinary handle is `WIDTH = 3` (`library_rail.py:541`).
2. **The grips always cost 10 columns.** They are reserved even when both side panes are open (`adaptive_reader_state.py:327`), which is 12.5% of an 80-column terminal.
3. **The rail disappears at common sizes.** On reader routes at 120×40 the Library pane is closed by default (§2 table). Navigation context is one grip away and you are not shown where you are, which works against NN/g's "visibility of system status".
4. **Escape is overloaded.** 15 Escape bindings are resolved by declaration order (`library_screen.py:1043-1129`), and the code comments record the roster going stale repeatedly. Use a single back-stack owner instead.
5. **The glue is copy-pasted per destination.** This includes the pane-toggle if-ladder, five static F6 target tuples, and emergency geometry that hard-codes shell ids and focused-attribute names (`:5803-5809`, `:5760-5770`). The shell is shared; the orchestration is not.
6. **Three border layers.** `#library-shell-grid`, `#library-rail` and `#library-canvas` each draw a border, and the grid adds padding (`css/features/_library.tcss:49-84`). That costs about 8 columns at ≥120; compact mode removes the outer one.
7. **Unsaved-work handling is inconsistent.** Notes autosave, while Prompts and Skills hard-veto Back with a toast instead of offering Save / Discard / Stay. Delete confirmation is inline for Notes and Skills but a modal for Prompts (`library_prompts_controller.py:4121`).
8. **The rail depends on fitting heuristics.** For example, the Collections gloss is hidden at the default width (`library.md:246-251`). The Details disclosure is a catch-all bucket (Status, Workspaces, Diagnostics, Chunking Lab).
9. **Scale and governance cost.** The screen is 35.9k lines. The shell CSS is screen-scoped by naming convention. Sharing beyond Library needs a new ADR under ADR-086.
