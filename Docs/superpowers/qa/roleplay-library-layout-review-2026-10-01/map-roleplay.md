# Roleplay screen (PersonasScreen): structural map

Worktree `origin/dev @ 84247cb843`. Paths are relative to the worktree root. `PS` = `tldw_chatbook/UI/Screens/personas_screen.py` (16,533 lines). `PW` = `tldw_chatbook/Widgets/Persona_Widgets/`. `AT` = `tldw_chatbook/css/components/_agentic_terminal.tcss`. `WB` = `tldw_chatbook/css/components/_workbench.tcss`.

**Evidence base.** I read the source. I also measured live geometry with a read-only probe (`probe_rp_geometry.py` in this directory) that mounts the real screen in `StyledPersonasTestApp`, which loads the real app CSS bundle. The probe ran at 52x20, 60x20, 80x24, 100x30, 120x40 and 170x50. Screenshots: `rp-*.svg`. Text grids of the same frames: `rp-text-renders.txt`.

Harness caveats:
- The stub characters have no description, so their rows render on one line. Real character rows with a description or a date are **2 lines** (`h-2`).
- No provider is configured, so the header badge reads **Blocked**.

## 1. Compose tree (top → bottom, left → right)

`BaseAppScreen.compose` (`UI/Navigation/base_app_screen.py:371-422`) wraps the content:

| # | Region | Widget / id | Source | Rows |
|---|---|---|---|---|
| 0 | Nav bar | `MainNavigationBar` | `UI/Navigation/main_navigation.py:235` | 3 |
| 1 | Header | `DestinationHeader#personas-header` holds three children: title "Roleplay", a subtitle (dynamic), and a status chip (Ready/Blocked) | `PS:1547-1554`; widget `UI/Workbench/workbench_widgets.py:164-245`; CSS `WB:22-52` (border solid + 3 lines) | 5 (4 at terminal height ≤24: the subtitle hides, `base_app_screen.py:62-65`, `WB:310-317`) |
| 2 | Purpose line | `Static#personas-purpose.destination-purpose`, e.g. "Characters — who the AI plays · 2" | `PS:1558-1562`, text `PS:4598+` | 1 |
| 3 | Mode strip | `DestinationModeStrip#personas-mode-strip` holds the label `Static#personas-mode-label` "Modes:" (8 cols) and chips `Button#personas-mode-{characters,personas,dictionaries,lore}.personas-mode-chip[.is-active]`. Each chip's tooltip is its descriptor plus its hotkey. | `PS:1563-1586`; CSS `PS:1171-1197`, `AT:381-395,857-870` | 1 (`overflow: hidden`) |
| 4 | Workbench | `Horizontal#personas-workbench.ds-panel.destination-workbench` | `PS:1587`; CSS `AT:639-660` (solid border + padding 1) | rest |
| 4a | Library handle | `DestinationRailHandle#personas-library-rail-handle`, button `#personas-library-rail-open` | `PS:1590-1603`; `Widgets/destination_rail.py:48-125` | 13 cols, shown only when the rail is collapsed |
| 4b | Library rail | `PersonasLibraryPane#personas-library-pane` | `PS:1605-1611`; `PW/personas_library_pane.py` | see §3 |
| 4c | Centre | `Vertical#personas-work-area` | `PS:1613-1715` | |
| 4c.i | Preview pane | `PersonasPreviewPane#personas-preview-pane`: toggle "▸ Try a test chat (nothing saved)" over a collapsed body | `PS:1620`; `PW/personas_preview_pane.py:112-170` | 1 collapsed (`max-height: 60%`) |
| 4c.ii | Detail stack | `VerticalScroll#personas-detail-stack`: one exclusive centre view, plus the character-attachments wrapper | `PS:1634-1709` | fill |
| 4c.iii | Try-it panels | `PersonasDictionaryTryItWidget#personas-dict-tryit` and `PersonasLoreTryItWidget#personas-lore-tryit`, siblings below the stack | `PS:1710-1715` | about 12 |
| 4d | Inspector | `PersonasInspectorPane#personas-inspector-pane.ds-inspector` (a `VerticalScroll`) | `PS:1717-1723`; `PW/personas_inspector_pane.py` | |
| 4e | Inspector handle | `DestinationRailHandle#personas-inspector-rail-handle`, button `#personas-inspector-rail-open` | `PS:1725-1738` | 11 cols × 3 rows |
| 5 | Footer | `AppFooterStatus#screen-footer-status`, contextual hints | `base_app_screen.py:403-422`; hints `PS:16455-16505` | 1 |

**Children of the detail stack, in document order** (`PS:1635-1709`). The exclusive set is `_CENTER_VIEW_IDS` (`PS:643-654`), toggled by `_show_center` (`PS:15909-15975`):
- `PersonasCharacterCardWidget` (`#ccp-character-card-view`)
- `#personas-character-editor-slot`, which demand-mounts `#ccp-character-editor-view`
- `#personas-character-attachments`, holding `PersonasCharacterDictionariesWidget` and `PersonasCharacterWorldBooksWidget`. This is not exclusive: it shows only with the character card or editor and a local runtime (`PS:15952-15964`).
- `PersonaProfileCardWidget` (`#ccp-persona-card-view`)
- `#personas-persona-editor-slot`
- `#personas-conversation-actions`, shown only alongside the transcript. It holds:
  - Resume chat
  - Send transcript to Console draft
  - Back to conversations
  - Open in Library
  - The block is 9 rows tall, made of 3-row buttons.
- `#personas-dictionary-detail-slot`
- `#personas-lore-detail-slot`
- `PersonasConversationTranscriptWidget` (`#personas-conversation-transcript-view`)
- `#personas-mode-placeholder` (dead: the "prompts" copy)
- `#personas-characters-empty`
- `#personas-character-link-recovery`, a deep-link retry state

Four heavy bodies are demand-mounted (`_DEMAND_CENTER_VIEW_ROOTS`, `PS:659-670`; ADR-115): the character editor, the persona editor, the dictionary detail and the lore detail.

**What changes per mode** (`_apply_mode`, `PS:4459-4549`):
- Chip `.is-active`
- The purpose line
- Library toolbar visibility (`set_mode`)
- The preview pane is hidden in Dictionaries and Lore. The dictionary or lore Try-it panel is shown in its own mode.
- The centre goes to `None`. In Characters that means empty or picker guidance; in every other mode the centre is blank.
- The inspector is cleared.

## 2. Geometry

**Widths.** The app CSS outranks `BUNDLED_CSS`:

| Pane | Wide | Compact (≤90 cols) | Narrow (≤60 cols) |
|---|---|---|---|
| Library | 2fr, min 24 (`AT:726-729`) | fill, min 16 | single pane |
| Work area | 4fr, min 40 (`AT:731-734`) | 3fr, min 34 | single pane |
| Inspector | 2fr, min 30, solid border, padding 1 (`AT:709-714`) | fill, min 22 | single pane |

- The compact classes are at `AT:736-756` and `PS:1232-1252`. In compact mode the workbench padding drops to 0.
- Narrow (`PS:2317-2318`, `2399-2414`) shows exactly one pane, selected by `_compact_active_pane`. Rail handles shrink to 3 cols (`PS:2383-2392`).
- `_sync_responsive_workbench` (`PS:2316-2393`) runs on resize (`PS:2308`). Constants are at `PS:607-609`.
- The Library and Work panes get `border: round` plus `padding 0 1` from the shared `.destination-workbench-pane`. Its source is `css/layout/_destination.tcss`; in the bundle it is `tldw_cli_modular.tcss:1009-1017`. The inspector overrides that with a square border and padding 1.

**Measured regions**, as `x,y w×h`:

| Size | Library | Work | Inspector | First list row | List rows visible | Centre content starts |
|---|---|---|---|---|---|---|
| 170x50 | 2,12 41×35 | 43,12 83×35 | 126,12 42×35 | y22 | 23 lines (about 11 two-line characters) | y14 |
| 120x40 | 2,12 28×25 | 30,12 58×25 | 88,12 30×25 | y22 | 13 lines (about 6) | y14 |
| 100x30 | 2,12 24×15 | 26,12 42×15 | 68,12 30×15 | y22 | 3 lines (about 1) | y14 |
| 80x24 (compact) | 1,10 16×12 | 17,10 40×12 | 57,10 22×12 | y20 | **1 line** (the bottom border is at y21) | y12 |
| 52x20 and 60x20 (narrow) | hidden (3-col handle) | 4,10 44/52×8 | hidden | – | 0 | y12 (5 rows of card visible) |

**Chrome above the first list row is 22 rows** at sizes of 100x30 and larger. In Characters mode it breaks down as:

| Rows | Item |
|---|---|
| 3 | nav |
| 5 | header |
| 1 | purpose line |
| 1 | mode strip |
| 1 | workbench border |
| 1 | workbench padding |
| 1 | pane border |
| 1 | "Library <" title |
| 1 | search (collapsed to 1 row while the toolbar is stacked) |
| 5 | **stacked toolbar** |
| 2 | **stacked filter bar** |

- Below the list sit another 5 chrome rows: pane border, count line, padding, border and footer.
- Personas mode needs 19 rows of chrome (toolbar 3 + filter bar 1), and Dictionaries also 19 (toolbar 3 + an **empty** filter-bar row).
- The centre pane has 13 chrome rows plus the 1-row preview toggle above the card.

**Rail collapse** (`PS:4427-4457`). Collapsing both rails at 120x40 makes the work area 92 wide (x15). The Library handle is 13×25 and the Inspector handle is 11×3. Collapse state is held in `_library_rail_collapsed` and `_inspector_rail_collapsed`. Those are instance attributes, **not** part of `save_state`, so leaving the screen resets them.

**Toolbar stacking** (`PW/personas_library_pane.py:200-242`). The two bars stack vertically, and the search field drops its border, whenever the label-derived single-row width exceeds the pane:
- Characters needs 64 cols: New, New Actor Pack, Import, Import Actor Pack, Duplicate, plus 3 cols of chrome each. That means it is stacked at **every** size measured; the widest pane was 37 content cols at 170.
- Personas needs 43 cols.
- Dictionaries needs 27 cols, so it fits on one row at 170.
- At 80x24 the labels truncate to "New / New / Import / Import / Duplic / Sort: / Tag:".

**Narrow-width traps:**
- At 52x20 the "Lore" chip is clipped off the strip.
- The inspector handle reads "In".
- The Library handle button is `h-full` and overflows the frame (region h20 starting at y10).

## 3. Per mode

**Library rail, common to all modes** (`PW/personas_library_pane.py:244-330`):
- `.console-rail-header`: the title "Library" plus a `<` collapse button (`#personas-library-rail-collapse`).
- `Input#personas-library-search`, with a 0.2 s debounce (`PS:483`).
- `#personas-library-toolbar`: New, New Actor Pack, Import, Import Actor Pack, Duplicate.
- `#personas-library-filterbar`: "Sort: Name", "Tag: All".
- `ListView#personas-library-rows`.
- `#personas-library-pagebar`: `<` / "a-b of N" / `>`. It shows only when the total exceeds the page size of 50 (`PS:486`).
- `#personas-library-count`. It stays empty unless the list is filtered or rows are marked.

Row and key behaviour:
- A row is a name line, with a "● " prefix when marked, plus an optional muted meta line that makes it a 2-line `h-2` row (`:458-475`).
- Highlighting a row does not select it. Enter or a click posts `PersonaEntitySelected` (`:648-659`).
- Keys:
  - `m` marks a row for bulk delete/export.
  - `s` cycles the sort.
  - `space` toggles a dictionary on or off (`:80-86`).
- Empty copy: "No {noun} yet - use New or Import to add one." (`:431-442`).

| Mode | Rows | Toolbar (`set_mode` `:332-379`) | Sort/Tag/Page |
|---|---|---|---|
| Characters | name + description snippet (≤72 chars) or date, 2 lines (`PS:613-638`) | New, New Actor Pack, Import, Import Actor Pack, Duplicate | Sort cycle: relevance (search only) / Name / Recent edit / Recent add (`PS:491-496`). Tag uses the `TagFilterPicker` modal. Pages of 50. |
| Personas | name only, 1 line | New, New Actor Pack, Import Actor Pack | Sort; no Tag; paged |
| Dictionaries | name + "N entries · on/off", 2 lines (`PS:4273`) | New, Import, Duplicate | none; the filter bar is still 1 empty row |
| Lore | name + "N entries · state" (`PS:4350`) | New, Import, Duplicate | none |

**Centre states.**

Characters:
- **No selection:** guidance at `PS:474-479`. First paint auto-selects the first row (`PS:1878-1924`). A mode switch never auto-selects.
- **Card** (`PW/personas_character_card_widget.py:94-140`), from top to bottom:
  - "Character" heading
  - Name, Description, First message and Version rows
  - Tags
  - Alternate greetings: a count plus a preview
  - Voice & Speech (`PersonasCharacterTTSWidget`)
  - Avatar status
  - A toolbar with Edit and "Conversations (N)". The second button only focuses the inspector list (`PS:7062-7065` → `7084`).
  - Below the fold: the collapsible headers "▸ Dictionaries (N)" and "▸ World Books (N)".
- **Editor** (`PW/personas_character_editor_widget.py:400-711`):
  - Generate toolbar: context plus "Generate whole character…", and a concept row
  - Name
  - First message, Description, Personality and System prompt, each a TextArea with a "Generate" button
  - Voice & Speech
  - "Advanced ▸", which reveals Scenario, Post-history instructions, Creator notes, an alternate-greetings table with Add/Update/Delete/Move, Creator, Version and Tags
  - Avatar row: upload, generate and remove
  - Pack status
  - Expression slots, each with Upload, ✨ Generate and Clear
  - Visual-identity host
  - Validation and Save/Cancel, anchored outside the scroll
  - There are no tabs and no collapsibles beyond "Advanced".
- **Transcript:** a read-only view plus the 4-button action block (`PS:1652-1678`).
- **Preview** (`PW/personas_preview_pane.py:112-170`). The toggle reveals:
  - provider readout
  - transcript (≤10 rows)
  - status
  - greeting select
  - "Test message..." input
  - buttons: Test Reply, Reset, Send to Console draft, Configure

Personas:
- **No selection:** a blank centre.
- **Card** (`PW/persona_profile_card_widget.py:38-80`): Name, Description, System prompt, Edit.
- **Editor** (`PW/persona_profile_editor_widget.py:103-175`):
  - Required portrait character (actor pack)
  - Name, Description, System prompt, Personality traits
  - Mode, Enabled
  - Shared visual identity
  - Persona visual states (`PersonasPersonaVisualPackWidget`)
  - Policy rules editor, shown only for a local, saved persona
  - Validation, Save, Cancel

Dictionaries:
- **No selection:** a blank stack, with the Try-it panel always shown below it ("Select a dictionary to preview substitutions.").
- **Detail:** a `TabbedContent` (`PW/personas_dictionary_detail.py:201-324`) with these tabs:
  - **Entries:** a table, an entry form (pattern, regex, probability, group, max replacements, enabled, case, priority, replacement), Add/Update/Delete/Move up/Move down, and a validation OptionList.
  - **Settings:** name, description, strategy, max tokens, enabled, "Save settings", Export JSON/Markdown.
  - **Stats.**
  - **Versions:** a table with View and "Revert…".
  - **Attachments.**
- **Try it:** a sample TextArea, a Run button, then original/processed text, fired rules and near misses (`PW/personas_dictionary_tryit.py:105-121`).

Lore:
- **Tabs:** Entries, Settings and Attachments; Attachments has "Attach to conversation…" and Detach (`PW/personas_lore_detail.py:135-242`).
- **Try it:** an injection preview with an "Include recent turns (soon)" switch (`PW/personas_lore_tryit.py:72-96`).

**There is no separate User Profiles mode.** Personas are "who you play" (ADR-037).

**Inspector** (`PW/personas_inspector_pane.py:212-363`; visibility logic in `_apply_action_state` at `:964-1159`):

| Selection | Shows |
|---|---|
| none | header "Inspector >", "Selected: none", "Type: -", guidance "Pick a character or persona to start chatting." (shown in **every** mode, including Dictionaries and Lore), plus **4 Buddy buttons** |
| character | avatar thumb (≤24×10), "Validation: OK", Conversations (search + list capped at 10 rows + an older/retry tail), Readiness line, Chat now (primary), Send to Console draft, [Include voice profile], Export JSON, Export PNG, Export Actor Pack, Delete, plus 4 Buddy buttons |
| persona | Validation, "Tool policy: …", Readiness, Chat now, Send to Console draft, Export JSON, Export Actor Pack, Delete, plus Buddy |
| dictionary / lore | Validation, Readiness "Console chat is for characters and personas.", Delete, plus Buddy |

- The Buddy buttons (Manage, Show, Close, Disable) are forced `display = True` in every state (`:1145-1159`).
- With rows marked, Delete and Export JSON act on the whole marked set (`:1095-1143`).
- At 100x30 the action stack starts at y28. The pane ends at y26, so **Chat now sits below the fold**.

## 4. Navigation and keyboard

**Bindings** (`PS:1083-1121`):

| Key | Action |
|---|---|
| F6 / Shift+F6 | Focus the next / previous pane (priority binding) |
| Ctrl+N | New |
| Ctrl+F | Focus search |
| Ctrl+Enter | Send to Console draft |
| Ctrl+S | Save (hidden from the footer) |
| Esc | Back (hidden from the footer) |
| c / p / d / l | Switch mode (ADR-031 rule 3, ADR-152). Printable, so a focused text field swallows them. |
| [ / ] | Previous / next mode |

The library pane adds `space`, `m` and `s`.

**F6 order** (`PS:1122-1155`): Library handle → Library (search) → Work (resume / preview input / preview toggle) → Inspector (conversations list) → Inspector handle. Pinned by `Tests/UI/test_workbench_pane_focus.py:106-126`.

**Escape** (`PS:16318-16351`). It never leaves the screen. It is tried in this order:
1. Cancel a pending resume.
2. In an editor, cancel through the unsaved-changes guard.
3. In the transcript, go back to the card and the conversations list.
4. In the search field, move focus to the list.

**Footer hints** (`PS:16455-16505`) are computed from live state and truncated by width. For example, at 80 cols only "ctrl+n new | ctrl+f search" plus the globals survive.

**Command palette** (`tldw_chatbook/app_command_providers.py:709-781`). "Create New Character" and "Show All Characters" only *navigate* to the screen; they do not open the editor or the list.

**Mode switch** (`PS:4459`, guarded by `_run_guarded` at `PS:16179`). `state.switch_mode` (`PW/personas_state.py:54-65`) clears:
- the selection
- sort, tag and page
- unsaved changes

`_apply_mode` also clears the search box, the marks, the preview and the inspector. Nothing is remembered per mode.

**Leaving and returning.** The route is not reusable (`UI/Navigation/screen_registry.py:116-121`), so every visit builds a fresh instance:
- `save_state` (`PS:1748`) stores the `PersonasWorkbenchState` dataclass and the preview greeting and turns. `app_navigation.py:799-806` calls it.
- `restore_state` (`PS:1768`) seeds that state, and `_apply_pending_restore` (`PS:1819`) re-applies the mode and selection.
- Not persisted:
  - edit mode and editor contents (the unsaved-changes guard handles these)
  - rail collapse
  - scroll positions
  - the marked rows

## 5. Module ownership and seams

| File | Owns | Layout or behaviour |
|---|---|---|
| `PS` | Compose, CSS, responsive and rail logic, mode switching, selection, all CRUD, import/export, delete, save/cancel, Console handoff, expressions, actor packs, visual identity, TTS, footer and header sync. Sections at `1538/1740/2428/4417/4634/5997/6576/7024/7132/7481/12573/13442/13648/14744/15214/15818/15883/16272/16369/16451`. There are 488 methods and 102 `@on` handlers. | mixed (God object) |
| `UI/Persona_Modules/personas_conversations_controller.py` | Inspector conversations paging and search, the transcript centre view, Resume, continue in Console, Open in Library | behaviour |
| `…/personas_preview_controller.py`, `…/personas_preview_coordinator.py` | The ephemeral test-chat gateway and its app-lifetime serialization | behaviour |
| `…/buddy_conversion.py` | First-use Buddy conversion | behaviour |
| `UI/CCP_Modules/ccp_{character,persona}_handler.py`, `ccp_enhanced_handlers.py` | Data access and loading decorators | behaviour |
| `PW/personas_library_pane.py` | Rail chrome, toolbar, list, paging, marks | **layout + thin intents** |
| `PW/personas_inspector_pane.py` | Rail chrome, summary, conversations list, actions, Buddy | **layout + gating** |
| `PW/personas_preview_pane.py` | Test-chat UI | layout |
| `PW/personas_character_card_widget.py`, `persona_profile_card_widget.py`, `personas_conversation_transcript_widget.py` | Read-only views | layout |
| `PW/personas_character_editor_widget.py`, `persona_profile_editor_widget.py`, `personas_{dictionary,lore}_detail.py`, `*_tryit.py`, `personas_character_{dictionaries,world_books}.py`, `personas_policy_rules_editor.py`, `*visual*_pack_widget.py`, `personas_character_tts_widget.py` | Forms and editors that are self-contained and post typed messages | re-host, don't rewrite |
| `PW/personas_messages.py`, `personas_pane_messages.py` | Typed message contracts between widgets and the screen | **the seam: keep** |
| `PW/personas_state.py` | `PersonasWorkbenchState`, `MODE_LABELS` | keep |
| Pickers and modals (`tag_filter_picker`, `dictionary_picker`, `world_book_picker`, `*_attach_picker`, buddy, actor pack, petdex review) | Modals | untouched |

**A layout rewrite would touch:**
- In `PS`: compose `1540-1738`, `BUNDLED_CSS` `1160-1325`, constants `607-609`, focus targets `1122-1155`, the responsive and rail code `2308-2425` and `4427-4457`, the chip and visibility parts of `_apply_mode` `4459-4549`, `_show_center` `15909-15975`, the header/purpose/footer text `4551-4620` and `16455-16526`.
- `PersonasLibraryPane` and `PersonasInspectorPane` compose, CSS and `set_mode`.
- CSS: `AT:342-410`, `639-760`, `818-912`; `WB:55-110`.
- The generated `tldw_cli_modular.tcss` and `widget_defaults_*.tcss`, rebuilt through the build and checked by `preflight.sh`.

**Leave alone:**
- Behaviour sections `PS:4634-15880`.
- The controllers.
- The editors' internals.
- The shared chrome: `DestinationHeader`, `DestinationModeStrip`, `DestinationRailHandle`, `.destination-workbench-pane` (13 screens use it), `BaseAppScreen`, `workbench_focus`.

**Coupling hazards for a rewrite:**
- The screen queries panes by id everywhere.
- Tests pin **325 distinct `#personas-*`/`#ccp-*` ids across 51 files**.
- Moving widgets is safe; renaming ids is not.

## 6. Tests that constrain a redesign

- `Tests/UI/test_personas_workbench.py` (358 tests):
  - `:510`: at 80x40 the compact class is set, Library ≥12, Work ≥34, Inspector ≥18, and the inspector stays inside the workbench.
  - `:532`: 52x20 shows exactly one pane.
  - `:548`: the purpose text is exactly "Characters — who the AI plays · 2", **the workbench starts at y == 10** at 170x50, and the count line is empty.
  - `:586/608/630`: rail collapse and reopen work, and Shift+F6 reaches the handle.
  - `:652`: a resize is a no-op when nothing changed.
  - `:698/743/8291/8439/11967/12688`: footer hint contents.
  - `:761/838`: chip tooltip copy; no "soon".
  - `:871/8131/8154`: the title is "Roleplay", and the Blocked badge is red.
  - `:10183`: the preview pane is mounted inside the work area.
  - `:12315-12471`: first-paint auto-select and guidance copy (left-aligned).
  - `:12536-12603`: Escape semantics.
  - `:12729`: mode keys.
  - `:6400-6905`: compact transcript behaviour and its F6 order.
- `test_personas_library_toolbar_layout.py`: stacked at 100x30 and 80x24, one row at 170x50, and every button inside the pane.
- `test_personas_center_canvas_layout.py`:
  - The card fills the stack viewport.
  - The attachments are 1-row headers below the fold at 170x50.
  - Sections expand in place.
  - Card rows sit inside the card at 100x30.
- `test_personas_library_pane.py` (24 tests) and `_paging.py` (8): toolbar classes, empty and count copy, `m`/`s`, visibility per mode, the page bar.
- `test_personas_inspector_pane.py` (60): single guidance line, sections revealed on selection, conversations list capped at 10, copy at 24x20, CTA labels, Buddy labels.
- `test_personas_library_rail_focus_outline.py`: the `#personas-library-rows:focus{outline:none}` rule must exist in both `AT` and the bundle.
- `test_personas_deferred_center_views.py`: the four heavy views are absent at first paint (ADR-115).
- `test_personas_workbench_state.py`: save/restore of mode, selection and preview.
- `test_workbench_pane_focus.py`: the F6 cycle.
- `test_destination_visual_parity_correction.py:997-1007,1590`: at 140x42:
  - the workbench y ≤12
  - three horizontal panes on the same y, each ≥20 rows tall
  - the strip ≤2 rows
  - pane titles "Library" and "Inspector"
- `test_destination_shells.py:1306`: the header contains "who the ai plays".
- `test_screen_footer_hints.py:599`: no `recompose=True` anywhere in `PS`.
- `test_roleplay_recovery_layout.py`: the recovery modal is centred at 52x20, 80x24 and 170x48.
- Performance: `test_ui_latency_guardrails.py:66` and `test_screen_leaks.py:39` (Ctrl+4), and the `boot_css_bytes.json:145` budget of 2,996 bytes for the screen's scoped CSS.
- **`Tests/Architecture/test_module_size_ratchet.py:97` budgets `PS` at 16,436 lines. It is currently RED at 16,533 (+97), verified by running it on this commit.** Any rewrite has to move code out of `PS`.
- Governing decisions: ADR-004, 007, 011, 031, 037, 115, 120 (`backlog/decisions/`).

## 7. Observations (unverified smells, for the heuristic review)

1. **Vertical chrome dominates.** It takes 22 of 40 rows before the first list row at 120x40, leaving 1 visible list line at 80x24 and 3 at 100x30. The header alone spends 5 rows on a title, a static subtitle and a one-word badge. Purpose, "Modes:" and the strip are each a separate row.
2. **The toolbar is permanently stacked in Characters (7 rows), with four creation verbs.** At 80 cols the truncated labels can't be told apart ("New"/"New", "Import"/"Import").
3. **Box-in-box framing.** The workbench border and padding sit around three bordered panes. The borders are mixed: round for Library and Work, square plus padding 1 for the Inspector. As a result the rail titles are misaligned by a row (Library y13, Inspector y14).
4. **The inspector carries noise.** It always shows 4 Buddy buttons, "Type: character", and "Validation: OK". Its guidance is about characters/personas even in Dictionaries and Lore. The primary CTA is below the fold at ≤100x30.
5. **Mode switches drop context.** The selection is not remembered per mode, the centre goes blank in three modes, the rail collapse is forgotten across visits, and the search is cleared.
6. **Duplicate paths.** "Send to Console draft" appears in 3 places. The card's "Conversations (N)" only focuses the inspector. "Try a test chat" shows in Personas mode with nothing selected.
7. **Dictionaries mode.** An empty filter-bar row remains, and the Try-it panel (12 rows) shows before anything is selected.
8. **Narrow widths.** At 52 cols the "Lore" chip is clipped, the handles are 3 cols wide and unlabelled, and the Library handle overflows.
9. **Row density is inconsistent.** Persona rows are 1 line; the other modes use 2.
10. **Drift and dead code:**
    - `Docs/User_Guide/roleplay-chat-dictionaries.md:36-68` still describes the "Roleplay & Chat Dictionaries" title, a "Status row", the old Personas descriptor, and "Ctrl+Enter Attach".
    - `shell_destinations.py:85` still says "user profiles… behavior profiles".
    - The prompts placeholder and `_COMING_SOON_MODES` (`PS:462-468,1690-1693`) are dead.
    - The `AT:357,408-409,664-665,689-690,804-805` rules target `#personas-title/list-pane/detail-pane/*-divider`, ids that are never composed.
    - The lore Try-it has an "Include recent turns (soon)" switch.
11. **Palette entries promise actions they don't perform** ("Create New Character" only navigates).
12. **`PS` is a 16.5k-line God object over its ratchet.** Layout is interleaved with ~9 behaviour domains.

---
**Run notes.**
- The probe was isolated with `Tests.conftest`, imported first, plus the `bootstrap_profile` marker. I confirmed that nothing under `~/.config/tldw_cli` or `~/.local/share/tldw_cli` changed.
- The Python runs left git-ignored `__pycache__/*.pyc` files in the worktree.
- To reproduce:

  ```
  cd <worktree> && PYTHONPATH=$PWD <venv>/python -m pytest <dir>/probe_rp_geometry.py --rootdir=$PWD -c pyproject.toml -p no:cacheprovider -o "markers=bootstrap_profile: probe" -s
  ```

  The probe file must first be renamed to `test_*.py`, or passed explicitly. Set `RP_PROBE_OUT` to write SVGs.
