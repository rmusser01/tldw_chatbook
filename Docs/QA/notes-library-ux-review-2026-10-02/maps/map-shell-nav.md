# UX map: Library shell, entry and navigation

Scope: how a user arrives at Library, the first-run (Get started / STARTER)
vs expanded lifecycle, the left rail, the adaptive reader shells and their
width tiers, footer + F1, key bindings and their `check_action` gates, the
Escape ladder, focus order / F6, status conventions, selection grammar, and
what Library remembers between visits.

- Code base: worktree `.worktrees/notes-library-ux-review` at `origin/dev`
  `2d34cbf80d` (2026-10-02). All paths below are relative to
  `tldw_chatbook/` unless they start with `Docs/`.
- `LS` = `UI/Screens/library_screen.py` (35,902 lines). `RAIL` =
  `Widgets/Library/library_rail.py`. `SHELL` = `Library/library_shell_state.py`.
  `SC` = `UI/Library_Modules/screen_constants.py`. `ENTRY` =
  `Widgets/Library/library_entry_canvases.py`. `ARS` =
  `Utils/adaptive_reader_state.py`. `RW` = `Utils/library_rail_width.py`.
- Method: code reading only, plus one pure-function run of
  `resolve_adaptive_reader_layout` / `project_default_library_width` with
  default preferences (script in the session scratchpad) to get the width
  tables in section 2.4. Nothing here has been verified live. Every item in
  section 7 is a SUSPECTED issue with a probe.

---

## 1. Inventory

### 1.1 Route and registration

| Item | Evidence |
|---|---|
| Library is one `Screen` route, `"library"` -> `LibraryScreen`, **reusable** (one installed instance per app run, suspended rather than unmounted between visits) | `UI/Navigation/screen_registry.py:101-115`; reuse path `app_navigation.py:852-900` ("warm visits skip construction, mount, and snapshot-restore entirely") |
| Aliases resolving to Library: `artifacts`, `notes`, `prompts`, `skills`, `ingest`, `search`, `media` | `screen_registry.py:216-282` |
| `research` is a real screen again (NOT a Library alias) | `screen_registry.py:188-198, 251-252` |
| Shell destination "Library" also folds `conversation`, `study`, `chunking_lab`, `writing`, `chatbooks` for nav-bar highlight purposes | `UI/Navigation/shell_destinations.py:56-75` |
| Legacy-route nav contexts applied when a route id arrives without its own context: `artifacts`->`artifacts-all`, `prompts`, `skills`, `search`, `media` (no entry for `notes` or `ingest`) | `app.py:2442-2449`, consumed `app_navigation.py:919-924` |
| Hotkeys: `ctrl+3` Library (hidden binding, `show=False` because not an F-key); `ctrl+6` -> Library ▸ All artifacts (ADR-172) | `shell_destinations.py:200-215`; `app.py:931-958`; `app_lifecycle.py:1705-1722` |
| Cross-visit state: in-memory `ScreenStateStore` ("Memory-only ownership") -- nothing about the current route survives an app restart | `UI/Navigation/screen_state_store.py:1` |

### 1.2 Screen chrome (top to bottom), `LS:14943-15100`

1. Optional Character return bar `#library-character-return` with button
   **Back to Console** (only on Conversations after a Console character deep
   link) -- `Widgets/Library/library_character_return.py:18-23`,
   `UI/Library_Modules/library_unavailable_navigation.py:150-160`.
2. Header `#library-header-line`: **Library | Local** or
   **Library | Server: <label>** (`unknown` when no label) -- `SHELL:766-772`.
3. `#library-lifecycle-status` (hidden unless the lifecycle choice could not
   be persisted: "Library view is updated for this session, but the choice
   may not be remembered.") -- `LS:15022-15030`, `LS:21611, 21617`.
4. Artifacts share strip (conditional) -- `LS:15031`.
5. Model-install progress pair (conditional) -- `LS:15038-15046`.
6. Notes-only source strip above the whole shell grid: **‹ Library / Notes**
   (task return), **Library notes** | **Folder files** --
   `UI/Library_Modules/library_browse_route_swap.py:109-160`, mounted at
   `LS:15047-15056` for canvas kinds `notes` and `notes-create`.
7. `#library-shell-grid` (`ds-panel destination-workbench`): either the
   ordinary two-pane shell (rail handle + rail + canvas) or one of the
   adaptive three-pane reader shells (rail + grip + Items + grip + Work).
8. App footer (`Widgets/AppFooterStatus.py`) with the Library-registered
   context chips plus the global cluster (F1 help, F6, Ctrl+P, Ctrl+Q).

### 1.3 Rail (`RAIL:1065-1160`)

Expanded/Graduated rail, in order:

| # | Control | Label / tooltip as rendered | Evidence |
|---|---|---|---|
| 1 | Heading | **Navigation** | `RAIL:1076` |
| 2 | Collapse | **Collapse** / "Collapse Library navigation" (hidden inside adaptive reader shells) | `RAIL:1080-1091`; hidden by `LS:6494-6498` |
| 3 | Primary top button `#library-ingest-top-button` | **Import…** / "Add files, links, and transcripts to your Library." | `LS:14864-14880` (via `top_action_factory`, `RAIL:1097-1098`) |
| 4 | Search box `#library-search-input` | placeholder **Search Library…**; seeded with the Search/RAG query only while Search/RAG is selected | `RAIL:1099-1106`; placeholder `UI/Library_Modules/library_rag_search_controller.py:821-832`; value `LS:34063-34076` |
| 5 | Clear `#library-search-clear` | **x** / "Clear the Library search box" | `RAIL:1110-1124` |
| 6 | Section **Browse ▾** | Media (N) — your files; Conversations (N) [short: Chats]; Notes (N); Prompts (N) — reuse; Skills (N) — AI add-ons; Collections (N) — saved captures [short: Captures; "(—)" on failed count]; Search / RAG — find all | `SHELL:493-608` |
| 7 | Section **Artifacts ▾** | All artifacts; Chatbooks; Reports (no counts) | `SHELL:610-631` |
| 8 | Section **Create ▾** | New note; New prompt; New skill | `SHELL:633-662` |
| 9 | Section **Study ▾** | Study decks (N); Flashcards due: N [short: Cards]; Quizzes (N) -- target_kind `handoff` | `SHELL:664-713` |
| 10 | Section **Import / Export ▾** | Import…; Export (disabled in server mode, tooltip "Export packages local content only.") | `SHELL:715-752` |
| 11 | Section **Details ▸** (closed by default) | Status (Source · Local, counts line), Diagnostics ▸ (DB sizes; rail-local, never persisted), Workspace (Active · name, Handoff · …), Actions (Create local workspace; "Everything here is stored on this machine · syncing to a server isn't available yet."; Import sources [only when no eligible sources]; Use in Console; "Chunking Lab — compare how text is split for search"; Chunking Lab; Try selected text) | `RAIL:1128-1153, 1190-1238`; `LS:14700-14838` |
| 12 | **Back to Get started** (only EXPANDED + authoritatively all-empty) | compact button, below Details | `RAIL:1154-1159` |
| 13 | Fold cue `#library-rail-fold-cue` | "▾ scroll for more" / tooltip "The rail has more below. Scroll it, or press F6 to move focus into it." (docked bottom, only while the rail overflows) | `RAIL:54-57, 1162-1188, 1283-1304` |

Row label grammar (`RAIL:1001-1063`): leading `▸ ` marks the selected row;
counts `(N)`, `(N+)` when sampled, dim `(…)` while loading, nothing when the
source is "off" (Search / RAG, Create rows); gloss `— text` renders whole or
not at all; title falls back to `short_title` before an ellipsis. Row tooltip
is just the row title (`RAIL:1329-1331`) except a disabled row.

STARTER / UNKNOWN rail (`RAIL:1092-1096`): heading + **Collapse**, rail rows
**Import…** and **New note**, button **Explore all tools**. No search box, no
sections, no Details.

Collapsed rail handle (ordinary routes only): `LibraryNavigationRailHandle`,
`WIDTH = 3`, paints `N/a/v` down the column, no arrow, tooltip "Expand
Library navigation" (`RAIL:538-576`).

### 1.4 Canvas kinds (selected rail row -> canvas), `SHELL:774-805`

- `""` -> landing (`LibraryLandingCanvas`, `ENTRY:110-523`).
- browse rows -> adaptive reader shells: Media and Notes share
  `#library-browse-reader-shell` (`Widgets/Library/library_browse_reader_shell.py`),
  Conversations/Collections/Prompts/Skills/Artifacts each own one
  (`SC:281-290` selector list). Create prompt / Create skill reuse the Prompts
  / Skills shells; Create note uses the Notes route (`notes-create`).
- `browse-search`, `ingest-import-media`, `ingest-export`, Study handoff rows
  -> ordinary two-pane shell (`LS:5602-5613` `_library_ordinary_route_active`).
- Study rows -> `LibraryStudyHandoffCanvas` (`ENTRY:526-577`): header,
  purpose, "Carries forward …", owner line "This page shows what carries over;
  generation and review run in Study." (`SC:359-361`), recovery, button
  **Continue in Study** (`SC:325-357`).

### 1.5 Pane grips in adaptive shells (`Widgets/Library/library_adaptive_reader_shell.py`)

- Two grips per shell, each `PANE_GRIP_WIDTH = 5` cells (`ARS:36`), label
  `<---` (open) / `--->` (closed), pane name painted down the column, tooltip
  "Collapse/Expand <label> pane" (`:142-221`). Library pane name is painted
  as **Nav** (`:37`).
- Items-pane names differ per destination: Media "Items" (Notes route renames
  it "Notes", `library_browse_reader_shell.py:196-198`), Prompts "Prompts",
  Skills "Skills", Collections "Items", Conversations "Items", Artifacts
  "Items", File notes "Folder files" (`LS:15228, 15276, 15323, 15403`;
  `library_artifacts_reader_shell.py:39`; `library_file_notes_workspace.py:1706`).
- A narrow-only named return **‹ Library** (`< Library` in ASCII mode) exists
  for **Media only** (`LS:7125-7210`); ordinary routes have a separate
  full-width **‹ Library** emergency bar (`Widgets/Library/library_emergency_return.py`,
  composed `LS:15487`).

---

## 2. States

### 2.1 Lifecycle (progressive disclosure)

States: `UNKNOWN`, `STARTER`, `EXPANDED`, `GRADUATED`
(`Library/library_rail_state.py:31-37`). Persisted at
`[library.rail_state] lifecycle` (`LS:21597-21622`).

| Transition | Trigger | Evidence |
|---|---|---|
| absent + profile created this run -> UNKNOWN | construction | `library_rail_state.py:80-104`; `app.py:1143`; `LS:2350-2362`; first mount persists UNKNOWN `LS:9481-9485` |
| absent + existing profile -> EXPANDED (default, "so a returning profile does not flash the starter rail") | construction | `library_rail_state.py:98-101` |
| any -> GRADUATED | any of 7 sources reports user content | `library_rail_state.py:118-128`; `LS:21505-21508` |
| UNKNOWN -> STARTER | all 7 sources authoritatively EMPTY | `library_rail_state.py:124-127` |
| **unstored** EXPANDED -> UNKNOWN -> STARTER | all-EMPTY evidence and config value is not literally "expanded" (task-32349) | `LS:21509-21543` |
| STARTER/UNKNOWN -> EXPANDED | **Explore all tools** (rail or landing) | `LS:21843-21866`; `library_rail_state.py:131-135` |
| EXPANDED + all-empty -> STARTER | **Back to Get started** | `LS:21941-21960` |
| Evidence re-read | every source-snapshot refresh (every visit/resume, every ingest completion poke) | `LS:11411-11413` |
| Evidence deadline | 5 s (`LIBRARY_ONBOARDING_EVIDENCE_TIMEOUT_SECONDS`) | `SC:137-138`; `LS:21421-21479` |

Lifecycle status copy on the Get started landing: "Checking existing Library
content…" (loading), "Some Library sources are unavailable." (partial
failure, plus **Retry source check**) -- `LS:21253-21262`, `ENTRY:374-375`,
`LS:14082-14086`.

### 2.2 Landing canvas states (`ENTRY:262-375`, builder `LS:14042-14092`)

| State | What renders |
|---|---|
| Get started (UNKNOWN/STARTER) | optional load-failure callout; heading **Get started**; purpose "Add something useful, then use it in Console or Study."; lifecycle status; three step buttons **Import a file** / **Find it** / **Use it in Console** with one hint line (first blocked reason: "Find it needs something to search — Import a file first."); toolbar **Import…**, **New note**, (**Explore all tools** only while the rail is collapsed); optional **Retry source check** (`ENTRY:382-428`, `LS:14087`) |
| Expanded, loading | purpose "Search everything, pick a section, or add something new." (`SHELL:13-15`); counts line "Notes (…) · Media (…) · Conversations (…)" (`LS:13990-14020`) |
| Expanded, populated | counts line; **Continue** heading + one primary button whose label is a scope description (e.g. "Media · type: video · recent first · page 2"), tooltip "Resume this Library view.", optional "Item views resume at the source list." (`LS:14196-14270`, `ENTRY:178-187, 299-312`); **Needs attention** callout with **Review**/**Retry** (`LS:14098-14194`); **From your Library** recent rows "Notes · <title>", "Media · <title>", "Conversations · <title>" (`LS:13961-13988`, `ENTRY:162-175`); **Quick actions**: **Import…**, **New note**, **Search** (`ENTRY:342-371`) |
| Source snapshot failed | one bordered callout (amber "Library sources did not answer · waited 5 s" / red "Library source services unavailable; retry Library later. · <reason>") with its own **Retry** (`#library-source-retry`); counts line empty; Continue withheld (`ENTRY:201-214, 293-298`; `SC:119-138`) |
| Width < 120 shell cells | **Needs attention** is not built (`not self._notes_state.compact`, `LS:14056-14061`; compact = shell `< 120`, `UI/Library_Modules/library_notes_controller.py:2867`) |

### 2.3 Global shell states

- **Loading sources**: rail counts dim `(…)`; Notes canvas "Loading local
  Library sources…" (`LS:15087-15093`).
- **Lookup error**: canvas error widget (`LS:15094-15095`); browse-row
  callouts with **Retry** (guide; `UI/destination_recovery.py`).
- **Server mode**: header "Library | Server: X"; Export row disabled
  (`SHELL:723-752`).
- **Prompt write in flight**: every rail press and every navigation-context
  deep link is silently ignored (`LS:22132-22133, 22210-22211, 22228-22229`;
  `UI/Library_Modules/library_navigation_controller.py:68-69`).
- **Narrow single stage (< 64 shell cells, ordinary routes)**:
  `_library_emergency_stage` is `"rail-only"` (landing / focus in rail) or
  `"canvas-only"` (a row is selected); canvas-only shows the **‹ Library**
  bar, disabled with tooltip "Finish or cancel the current Library action
  first." while an editor/strip/confirm/export/consent is open
  (`LS:5634-5671, 5790-5870`).

### 2.4 Width tiers

Ordinary rail width (landing, Search/RAG, Import, Export, Study):
`(3*w+8)//16 + 5`, clamped 29–39, never more than `w - 40`; single-stage
below 64 (`RW:13-20, 47-80`). Import auto-collapses the rail below 100
(`SC:276`; `UI/Library_Modules/library_ingest_controller.py:1053-1074`).
Notes "compact" (and the landing's Needs-attention cutoff) below 120 shell
cells (`SC:275`).

Adaptive reader layout, default preferences, **no priority** (pure resolver
output; W is the reader-shell width, which is the terminal width minus the
shell border/padding -- about 4 cells at >= 120, ~0 below 120 per the compact
box model `LS:6500-6514`). `L`=Library pane, `I`=Items pane, `R`=Reader;
`–` = collapsed to its grip.

| W | Media (no item) | Notes/Prompts/Skills/Coll. (no item) | Conversations (no item) |
|---|---|---|---|
| 235 | L39 I140 R46 | L39 I138 R48 | L39 I142 R44 |
| 200 | L39 I105 R46 | L39 I103 R48 | L39 I107 R44 |
| 160 | L35 I69 R46 | L35 I67 R48 | L35 I71 R44 |
| 120 | L24 I40 R46 | **L– I62 R48** | L26 I40 R44 |
| 100 | **L–** I44 R46 | **L–** I42 R48 | **L–** I46 R44 |
| 80 | **L– I– R70** | **L– I– R70** | **L– I– R70** |
| 64 | **L– I– R54** | **L– I– R54** | **L– I– R54** |
| 60 | L– I50 R0 (list_first_when_empty) | Notes: L– I50; Prompts/Skills/Coll.: **L– I– R50** | **L– I– R50** |

Library pane opens at W >= 120 (Media), 118 (Conversations), 122
(Notes/Prompts/Skills/Collections); Items opens at W >= 94–98 unless a
priority is requested (`ARS:395-424`). Routes that request an automatic
`items` priority on their LIST view: Notes (`LS:6656-6671`), Prompts
(`library_prompts_controller.py:801`), Skills (only above a floor,
`library_skills_controller.py:812-818`), Media Trash (`LS:7377-7393`).
**Media list, Conversations and Collections request none** -- so at 64–~95
cells they open on an empty Reader with both panes collapsed (see 7.2).

### 2.5 Destructive / confirm / receipt states at shell level

- Media bulk-delete confirm (Escape cancels; footer "esc cancel delete")
  `LS:1093-1104, 1344-1348`.
- Notes delete confirmation traps Tab between Cancel/Delete `LS:8830-8850`.
- Prompt / skill dirty vetoes surface as toasts: "Unsaved Prompt changes —
  Save or Discard changes first." / "Unsaved skill changes — Save or Discard
  changes first." (`SC:197-199, 227`); footer chip swaps to
  "esc save or discard first" / "esc busy, try again" (`SC:208-217`,
  `LS:4466-4495`).
- Data profile mismatch on character repair: toast "The active Data Profile
  changed. Repair was not applied." (`library_navigation_controller.py:150-158`).

---

## 3. Controls & copy (shell-level vocabulary)

| Concept | Names used in the UI | Evidence |
|---|---|---|
| The left pane | "Navigation" (heading), "Nav" (handle/grip), "Library pane" (grip tooltip), "Library navigation" (Collapse/handle tooltips), "rail" (footer: "esc focus rail", "esc rail", F6 fold tooltip) | `RAIL:1076, 549, 1085`; adaptive grip `:156-170`; `LS:1352-1356, 4834-4837` |
| The landing | unnamed on screen; footer/F1 call it "hub" ("esc back to hub") and "Landing" (F1 title "Library Shortcuts — Landing") | `LS:4257, 4267, 25780-25783` |
| Import | "Import…" (top button, row, quick action), "Import a file" (Get started), "Quick Actions: Import Media File", toast "Opened Import/Export for media import", "Library: Import…" (palette) | `LS:14876`; `SHELL:735`; `ENTRY:351, 383`; `app_command_providers.py:517-527, 592-595, 888-890` |
| Search | "Search Library…" (box), "Search / RAG — find all" (row), "Search" (quick action), "Find it" (Get started), "/ focus search" / "/ search" / "/ find note" / "/ find" (footer variants) | `RAIL:1102`; `SHELL:601-606`; `ENTRY:365-370, 384`; `LS:1247, 1389, 1549`; `LS:4711-4714` |
| New note | "New note" (rail/quick action), "ctrl+n new note" (landing footer), "n new note" (Notes list footer) | `LS:1258-1263, 1547-1552` |
| Disabled state | leading "○" + inline reason line (canvases); tooltip-only reason (Export row, ‹ Library bar); "pressable blocked" class that explains on press (Get started steps, Use in Console) | `SHELL:111, 169-245`; `RAIL:1329-1331`; `LS:5658-5671`; `ENTRY:418-422`; `LS:14820-14838` |
| Back | "‹ <destination>" grammar (`back_cue_label`), "‹ Library", "‹ Library / Notes", but "Back to Console", "Back to Get started" | `SHELL:249-270`; `library_character_return.py:20`; `RAIL:1156` |
| Choosers | "name: value" opener -> one-row strip with "✓ " active option; toggles "mode: ✓ Search ⇄ RAG Answer" | `SHELL:272-363`; `Widgets/Library/library_choice_strip.py:94-134` |
| Selection mode | "Select"/"Done", "Select all N shown", "Clear", "Export selected" (Notes compact "Export"), "Delete selected" (Prompts), Media "Delete" + confirm "Delete N selected items? …", reason line "Select items to enable." | `library_media_canvas.py:815-832, 1248-1267`; `library_notes_canvas.py:1834-1892`; `library_prompts_canvas.py:820-885`; `library_conversations_canvas.py:163-252`; `SHELL:101` |

Toast / status conventions:
- `app.notify` toasts for vetoes, refusals and palette navigation
  ("Switched to Library", "Opened Library Media"), severity
  information/warning/error (`app_command_providers.py:221-229, 470-482`).
- Inline status lines for saves ("Saved.", "Couldn't save this prompt. Try
  again.") (`SC:181-189, 232-238`).
- Recovery callouts (`.ds-recovery-callout`) with their own **Retry**; a
  repeated identical failure appends "· attempt N" (guide; `ENTRY:242-260`).
- Inline receipts with Undo/Dismiss (Notes delete, Media bulk delete), and
  export last-path receipt persisted across visits (`LS:9876-9882`).
- Graduation is silent (no toast) -- task-32555 (guide) / `LS:14298-14307`.

---

## 4. Key bindings

### 4.1 Global (app-level)

| Key | Action | Evidence |
|---|---|---|
| ctrl+q | quit (priority) | `app.py:932` |
| ctrl+p | command palette | `app.py:933` |
| f1 | `show_workbench_help` -> Library override | `app.py:934`; `LS:25683-25770` |
| f6 | next workbench pane | `app.py:935`; `LS:8960-8973` |
| ctrl+3 / ctrl+6 | Library / Library ▸ All artifacts | `shell_destinations.py:200-215`; `app.py:936-942` |

### 4.2 LibraryScreen `BINDINGS` (`LS:1010-1222`) and gates (`LS:25253-25681`)

| Key | Action | Active when (check_action) |
|---|---|---|
| tab / shift+tab | focus next/prev **within `#screen-content`** (never the nav bar) | always, except the narrow canvas-only emergency stage where `on_key` owns Tab (`LS:25289-25303, 8785-8829, 8978-9008`) |
| shift+f6 (priority) | previous workbench pane | **only when a row is selected** -- inert on the landing (`LS:25679-25680`) |
| u | Use Library context in Console | Search/RAG row (`LS:25531-25537`) |
| ctrl+n | New note | landing (no row) OR Notes visible in navigator/editor/preview/context (`LS:25316-25333`) |
| / (binding) | Find notes | Notes navigator, non-text focus (`LS:25334-25339`) |
| g | Go to folder | Notes navigator, non-text focus, folder rows exist (`LS:25343-25349`) |
| e | Export selected | Notes select mode with a selection (`LS:25350-25357`) |
| escape ×14 | see Escape ladder 4.4 | first passing gate in declaration order |
| ctrl+s | Save skill | skill editor, save available (`LS:25365-25366`) -- no equivalent in the Prompt editor |
| enter / o | select / open evidence card | a focused RAG result card (`LS:25538-25542`) |
| r | Retry this batch / Restore (media trash) / Restore note | Ingest with retry available / Media Trash actions live / Notes Trash with rows (`LS:25396-25414, 25473-25491`) |
| x | Delete forever | Media Trash actions live |
| ] / [ | next / previous media item | plain media Reader with a neighbour or an active review set (`LS:25601-25617`) |
| s / space(priority) | media select mode / toggle row | media list surface; space only on a media row or media grip (`LS:25543-25600`) |
| l, c, t, ctrl+f | read later / use in Console / move to trash / find | plain settled media Reader (`LS:25618-25667`) |
| c | resume conversation | Conversations with an open conversation (`LS:25453-25465`) |
| R / m | exit review / toggle reviewed | media Reader with active review set |

### 4.3 Screen `on_key` accelerators (not Bindings) -- `LS:8749-8958`

- `up`/`down`: move focus between list rows of classes `library-media-row`,
  `library-notes-row`, `library-prompt-row`, `library-skill-row`,
  `library-notes-create-row`, `library-conversation-row` (`SC:425-450`).
- `/`: Conversations -> its filter (if Items pane open); Media/Prompts -> canvas
  filter (`SC:457-460`); Notes -> notes binding; otherwise -> rail
  `#library-search-input`; **does nothing in STARTER/UNKNOWN** (`LS:8859-8921`).
  Footer drops the `/` chip when the target is not focusable
  (`LS:7211-7251`).
- `i`: open Import from **any** canvas with non-text focus (`LS:8922-8939`).
- `n`: same gate as ctrl+n (`LS:8940-8958`).
- Any key disarms pending list entry focus (`LS:8823-8825`).

### 4.4 Escape ladder (declaration order `LS:1043-1129`; first passing gate wins)

1. `library_notes_escape` -- Notes workflow visible: cancel delete confirm ->
   refuse during conflict -> close Info/context -> **exit select mode** ->
   leave Trash -> leave editor (guarded) -> … (`library_notes_controller.py:3009-3080`).
2. `library_skill_back` -- skill editor (dirty veto).
3. `library_notes_files_back` -- Folder files mode -> Library notes.
4. `library_media_viewer_back` -- media viewer (sub-states step back one level).
5. `library_media_trash_back` -- Media Trash -> list.
6. `library_note_editor_back`, 7. `library_prompt_editor_back`.
8. `library_ingest_back` -- Import -> landing (first Esc only disarms a
   pending start-consent) (`library_ingest_controller.py:1547-1568`).
9. `library_export_back` -- Export -> origin canvas or landing.
10. `library_handoff_back` -- Study staging -> landing (`LS:25940-25951`).
11. `library_media_bulk_delete_cancel`.
12. `library_emergency_return` -- narrow canvas-only -> rail-only.
13. `library_narrow_stage_return` -- reader shell with Library pane closed
    below 64 -> reopen Library pane (posts `PaneToggleRequested("library")`,
    which also persists the shared preference) (`LS:7260-7360`).
14. `library_blur_text_field` -- any Input/TextArea -> first non-text control
    in `#library-canvas` (`LS:27190-27212`).
15. `library_list_focus_rail` -- plain list (Media/Notes/Prompts/Skills/
    Conversations) or Collections -> rail search box, or Conversations filter
    (`LS:27214-27270`).

Escape never leaves Library and has no binding on the landing itself.
Artifacts canvases handle Escape inside their own shell (focus Items; clear a
non-empty search first) and never reach the rail
(`library_artifacts_reader_shell.py:98-112`, `library_artifacts_widgets.py:23-29`).

### 4.5 Footer sets and F1

- Sets: `LS:1243-1600`; selector `LS:4193-4525`; artifacts override
  `LS:4704-4714`; typing-in-field transform ("typing in field", "after esc:
  …", "esc leave field") `LS:4736-4831`; narrow overrides ("esc rail",
  "esc back to Library") `LS:4832-4875`; STARTER drops "/" `LS:4876-4885`.
- Footer Enter chip follows focus (`LS:4564-4631, 4715-4735`).
- F1 = footer set + `check_action`-active Binding extras, deduped by key,
  titled "Library Shortcuts — <surface>" from `_LIBRARY_HELP_SURFACE_LABELS`
  (`SC:488-503`) -- **no label for Artifacts rows or Study rows** (generic
  "Library Shortcuts"), "Landing" for no row (`LS:25772-25783`).

### 4.6 F6 / Shift+F6 pane cycle

- Targets per route: `LS:1402-1517`; chooser `LS:9162-9185`; resolver
  `Widgets/workbench_focus.py:20-82` (preferred ids only; **no fallback to a
  pane's first focusable child**; a non-focusable container pane is skipped).
- Default target list (used by landing, Search/RAG, Import, Export, Study,
  **New note**, **New prompt**) only knows canvas ids `library-hub-*` and
  `library-ingest-path`, plus the note work pane's save/title/body
  (`LS:1420-1443`).

---

## 5. Flows

### 5.1 First-timer: brand-new profile (config created this run)

1. First-run wizard exit "Add your first document" -> Library with
   `{ingest_media: True}` (`app.py:4055-4065`), or "Write your first note" ->
   `{notes_create: True}` (`app.py:4067-4086`). Without the wizard: click
   **⌃3 Library** / Ctrl+3.
2. Lifecycle UNKNOWN -> Get started rail (Import…, New note, Explore all
   tools) + Get started canvas; status "Checking existing Library content…"
   until the 7-source evidence settles (≤ 5 s).
3. Press **Import a file** (or rail **Import…**, or toolbar **Import…**, or
   `i`) -> Import canvas; Esc returns to Get started.
4. After the first item lands, the next snapshot refresh re-reads evidence ->
   GRADUATED -> full rail; Get started is replaced by the expanded landing
   silently.
5. Optional: **Explore all tools** -> EXPANDED (persisted), focus to the
   search box; **Back to Get started** at the rail bottom while still empty.

### 5.2 First-timer: existing but empty profile

1. Lifecycle absent -> EXPANDED default -> full rail paints, every count `(…)`.
2. Evidence settles all-EMPTY -> task-32349 demotes the unstored EXPANDED to
   STARTER -> rail recomposes to the 3-control Get started rail
   (`LS:21509-21556`). (The User Guide, `Docs/User_Guide/library.md`
   "Get started on a new profile", still says this profile keeps the full
   rail.)

### 5.3 Returning user, first Library visit in this app run

1. Ctrl+3 -> landing (fresh instance, no snapshot) with counts, From your
   Library, Quick actions; Continue only if this run already left Library
   once from a browse scope (`LS:9780-9814, 10169-10178`).
2. Click a rail row / recent row / quick action -> canvas. Rail-row presses
   reset that destination to a fresh entry (media viewer closed, Notes filter
   cleared) unless it is a retained reader hop (`LS:22223-22410`).

### 5.4 Power user: keyboard loop

1. Ctrl+3 from anywhere -> **the last Library canvas** (reused instance),
   not the landing.
2. `/` -> canvas filter (Media/Prompts/Notes/Conversations) or rail search
   (others); type, Enter on rail search -> Search/RAG canvas runs the query in
   **Search** mode, focus returns to the box (`library_rag_search_controller.py:776-819`).
3. Up/Down/Enter in lists; Esc -> rail search; F6 cycles rail -> list ->
   work pane; Shift+F6 reverse (not on landing).
4. `i` -> Import from anywhere; `n`/ctrl+n -> New note from landing/Notes.
5. Media power keys: `s` select, Space toggle, `]`/`[` traverse, `l` `c` `t`,
   Ctrl+F find, `R`/`m` review sets; Trash `r`/`x`.
6. F1 for the current surface's key list.

### 5.5 Power user: deep links in from other destinations

| From | Context | Lands |
|---|---|---|
| Home recent note / media | `note_id` / `open_source_type=media`+id | Notes editor / Media item (`UI/Screens/home_screen.py:1044-1078`) |
| Home ingest job rows | `ingest_media` | Import (`app_destinations.py:515-529`) |
| Console message "note" | `note_id` | Notes editor (`UI/Console_Modules/message.py:2218-2227`) |
| Console notes create | `notes_create` | New note (`UI/Screens/chat_screen.py:4840`) |
| Console prompt/recipe save | `mode=prompts`, `open_source_type=prompt`, id | Prompts item (`UI/Console_Modules/prompts.py:1555-1570`) -- dropped if Library is already on Prompts (7.7) |
| Console character context | character inspection/browse | Conversations + **Back to Console** bar (`UI/Console_Modules/wiring.py:1236-1254`) |
| Roleplay conversations | `mode=conversations`, `conversation_id` | Conversations reader (`UI/Persona_Modules/personas_conversations_controller.py:1280-1292`) |
| Study Escape | `mode=study` | Study decks staging (`UI/Screens/study_screen.py:1355`) |
| Workflows | `note_id` | Notes (`UI/Screens/workflows_screen.py:298`) |
| Meetings | `ingest_media` | Import (`UI/Screens/meetings_screen.py:1606`) |
| Artifacts screen "Open Library" / Console "Library home" | none | last Library canvas (`UI/Screens/artifacts_screen.py:1397`; `wiring.py:1253`) |
| `open_notes_workspace` (Study back-to-workspace etc.) | `mode=notes` | Notes list (`app_destinations.py:125-139`) |
| Palette | "Tab Navigation: Switch to Library" (no context), "Tab Navigation: Library — Skills", "Tab Navigation: Library — Artifacts", "Quick Actions: New Note" (`notes_create`), "Quick Actions: Search All Content" (`search`), "Quick Actions: Import Media File" (`ingest`, no context), "Media & Content: Open Media Library" (`media`), "Library: Import…" (`ingest_media`), "Library: Chunking Lab" | `app_command_providers.py:322-338, 445-462, 517-595, 801-863, 878-935` |

### 5.6 Narrow terminal (< 64 cells)

- Landing: rail only (rail fills width). Select a row -> canvas only with a
  full-width **‹ Library** bar; Esc (footer "esc rail") or the bar returns to
  the rail; the bar is disabled with a tooltip-only reason while an
  editor/strip/confirm is open (`LS:5634-5671, 27272-27281`).
- Reader routes: Library pane closed; Media shows a "‹ Library" control; all
  readers honour Esc "back to Library" when no earlier Escape owner is active
  (`LS:7260-7327`).

### 5.7 What is remembered

| State | Scope | Evidence |
|---|---|---|
| Current route, filters, editor/viewer, select mode, rail Collapse | whole app run (reused instance) | `screen_registry.py:101-115` |
| Snapshot (`save_state`): row, selections, views, RAG query/results/mode, scopes, export receipt, Continue receipt | memory only; used only when a fresh instance is built (e.g. runtime identity change) | `LS:9780-9883`; `screen_state_store.py:1` |
| Rail section open/closed | config `library.rail_state.sections` | `LS:21736-21791` |
| Lifecycle | config `library.rail_state.lifecycle` | `LS:21597-21622` |
| Reader Library-pane open, custom widths | config `library.reader` -- **shared by all reader destinations** | `LS:6543-6620, 6767-6797` |
| Reader Items-pane open | config per destination (`*_reader.items_open`) | `LS:6588-6599` |
| Search history | config `library.search.history` (≤10) | `LS:21660-21734` |
| Details ▸ Diagnostics open | never (rail-local) | `RAIL:625-630` |

---

## 6. Cross-links out of Library (shell level)

- **Use in Console** (Details ▸ Actions) stages "Local Library Sources"
  (`LS:14820-14838`); `u` on Search/RAG; media `c`; conversation `c`.
- **Continue in Study** from Study staging (`ENTRY:553-562`); Study's Escape
  returns to Study decks staging.
- **Chunking Lab** / **Try selected text** (Details ▸ Actions) open the
  full-screen `chunking_lab` route.
- Media full viewer "Open in Library ▸ Media" posts `NavigateToScreen("media")`
  from inside Library (`UI/Library_Modules/library_media_controller.py:4725-4733`).
- **Back to Console** on character-context Conversations.

---

## 7. Suspected issues with live probes

All are code-derived and UNVERIFIED. Probe recipe assumes an isolated
profile (`TLDW_CONFIG_PATH` + both path families per
`backlog/docs/lessons-live-verification.md`) and the `verify` skill's tmux
launch. Anchor ids are for Pilot/inspection; visible labels are given for
manual driving.

### 7.1 F6 never enters the canvas on Search/RAG, Export, Study staging, New note, New prompt (High)
- Evidence: `_library_workbench_focus_targets` returns the default list for
  these rows (`LS:9162-9185`); its canvas candidates are only `library-hub-*`
  and `library-ingest-path` (`LS:1420-1435`), and `_resolve_focus_target` has
  no first-focusable fallback (`Widgets/workbench_focus.py:70-82`). Footer
  still advertises "F6 next pane" there (`LS:1243-1249, 4258-4268`).
- Probe: 160x45. Ctrl+3 -> click **Search / RAG** row -> press F6 five times
  and watch focus (expect it to bounce only to `#library-search-input`).
  Repeat on **Export**, **Study decks**, **New note**, **New prompt**.

### 7.2 Media / Conversations / Collections open to an empty Reader with both panes collapsed at 64–~95 columns (High)
- Evidence: resolver table 2.4 (W=80/64 -> `L– I– R70/54`); no automatic
  `items` priority for these list views (`LS:7377-7393` Media only for Trash;
  `library_conversation_reader_controller.py` ~848; `library_collections_controller.py`
  ~247); `list_first_when_empty` only applies below 64 (`ARS:426-467`). Empty
  copy "Select a media item to read it here." / "Select a conversation to
  read it here." / "Select a capture to read it here." points at a list that
  is not visible.
- Probe: resize to 80x24 (and 90x30). Ctrl+3 -> click **Media**, then
  **Conversations**, then **Collections**. Expect two 5-cell grips and an
  empty Reader; count the presses needed to see the list (Items grip).

### 7.3 The landing (Continue / Needs attention / From your Library) is effectively unreachable once you leave it, and Continue can be stale (Medium-High)
- Evidence: only first visit, Escape from Import/Export/Study, and Back to Get
  started select row `""` (`library_ingest_controller.py:1566`,
  `library_export_controller.py:1258`, `LS:25949, 21956`); Library is reused so
  Ctrl+3 resumes the last canvas (`app_navigation.py:852-900`); the nav bar
  swallows a click on the already-active destination (`app_navigation.py:733-746`);
  the Continue receipt is only written in `save_state` when leaving Library
  (`LS:9808-9814`), not on in-Library hops.
- Probe: Ctrl+3 -> Media -> set type chooser -> Ctrl+1 Home -> Ctrl+3 (note:
  lands on Media, not landing) -> click **Notes** -> click **Import…** -> Esc.
  Read the Continue label: does it say Media (stale) rather than Notes? Then
  try to reach the landing from Notes by any visible control.

### 7.4 Get started steps 2 and 3 can never be seen unlocked; three Import controls on one view (Medium)
- Evidence: graduation happens on the first snapshot refresh after content
  appears (`LS:11411-11413, 21505-21508`), replacing Get started, so
  `has_any_content=True` never paints in Get started (`ENTRY:388-399`;
  `LS:34091-34143`). Get started shows rail **Import…**, step **Import a
  file**, toolbar **Import…** simultaneously (`RAIL:1092-1096`; `ENTRY:349-363, 382-428`).
- Probe: brand-new profile (delete scratch config so it is created this run).
  Ctrl+3 -> **Import a file** -> import a small .txt -> Esc. Observe whether
  "Find it"/"Use it in Console" ever appear unlocked or the view jumps to the
  full rail.

### 7.5 Empty pre-existing profile: full rail flashes then collapses to Get started (doc says it never does) (Medium)
- Evidence: `LS:21509-21543` (task-32349) vs `Docs/User_Guide/library.md`
  "You will probably never see this…". Recompose on lifecycle change
  `LS:21287-21310`; selected row is not cleared.
- Probe: scratch profile with a pre-written config and empty DBs. Launch,
  Ctrl+3, immediately click **Media** within ~1 s; watch for 5 s: does the
  rail shrink to Import…/New note/Explore while the Media canvas stays?

### 7.6 Palette and legacy routes do not land on the promised row (Medium)
- Evidence: `_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` lacks `notes` and `ingest`
  (`app.py:2442-2449`); "Quick Actions: Import Media File" navigates to
  `ingest` with no context and toasts "Opened Import/Export for media import"
  (`app_command_providers.py:592-595`); palette alias terms (notes, prompts,
  media, search…) only match the generic "Tab Navigation: Switch to Library"
  (`app_command_providers.py:322-338, 380-397, 445-462`).
- Probe: Library on Media list. Ctrl+P -> "Import Media File" -> Enter: does
  Import open? Ctrl+P -> type "notes" / "prompts" -> pick the Library hit:
  which canvas opens?

### 7.7 Deep link into a specific prompt is dropped when Library is already on Prompts; any deep link is dropped during a prompt write (Medium)
- Evidence: `library_navigation_controller.py:68-69, 113-118`.
- Probe: Ctrl+3 -> **Prompts** -> Ctrl+2 Console -> save a recipe/prompt via
  the Console prompts flow that offers "open in Library" -> observe Library
  shows the list without the saved prompt opened.

### 7.8 Rail appears/disappears when hopping rows at ~120–126 columns (Medium)
- Evidence: table 2.4 thresholds (Library pane needs W ≥ 118–122) vs ordinary
  routes keeping a 29-cell rail at ≥ 64 (`RW:47-62`); reader shell width ≈
  terminal − 4 at ≥ 120 (`LS:6500-6514`).
- Probe: 120x40, Ctrl+3 landing (rail visible) -> click **Media**,
  **Conversations**, **Notes**, **Search / RAG** in turn and note when the
  rail becomes a "Nav" grip. Repeat at 126.

### 7.9 Two collapse mechanisms with different persistence and visuals (Medium)
- Evidence: ordinary **Collapse** -> session-only `_library_rail_collapsed`,
  3-cell "N a v" handle without arrows (`LS:21793-21823`; `RAIL:538-576`);
  reader grip `<---` -> persisted, shared `library_open` across all seven
  readers (`LS:7615-7744, 6767-6797`); the Collapse button is hidden inside
  readers (`LS:6494-6498`); the narrow-stage Esc return also writes the
  shared preference (`LS:7352-7360`).
- Probe: 200x50. On **Search / RAG** press **Collapse** -> click **Media**
  (rail back?). On **Media** press the left grip `<---` -> click **Notes**,
  **Prompts**, **Search / RAG** -> quit and relaunch -> open **Media**
  (rail still collapsed?).

### 7.10 Shift+F6 is inert on the landing (Low-Medium)
- Evidence: `LS:25679-25680` gates `focus_previous_workbench_pane` on a
  selected row; no app-level shift+f6.
- Probe: Ctrl+3 to the landing (Esc from Import), press F6 (moves), then
  Shift+F6 (expect nothing).

### 7.11 "Needs attention" disappears below 120 shell cells (Low-Medium)
- Evidence: `LS:14056-14061`; compact rule `library_notes_controller.py:2867`.
- Probe: cause a failed import (bad path in a 2-file batch), Esc to the
  landing at 140 columns (callout + **Review**), resize to 110 (callout gone?).

### 7.12 Rail search submit silently resets Search/RAG to "Search" mode (Low-Medium)
- Evidence: `library_rag_search_controller.py:791-794`.
- Probe: **Search / RAG** -> toggle mode to RAG Answer -> type in the rail
  **Search Library…** box, Enter -> mode label.

### 7.13 `/` on Skills / Collections focuses the global search, whose Enter leaves the canvas (Low-Medium)
- Evidence: `SC:457-460` (only Media/Prompts mapped); `LS:8859-8921`; footer
  "/ focus search" (`LS:1352-1356`). Canvas filters exist
  (`LS:1456-1466, 1507-1517`).
- Probe: **Skills** row -> press `/`, type "x", Enter -> which canvas?

### 7.14 `i` from any canvas navigates to Import without being advertised there (Low-Medium)
- Evidence: `LS:8922-8939`; only the landing footer lists `i` (`LS:1258-1263`).
- Probe: **Media** list with a row focused (or a Search/RAG result card) ->
  press `i`.

### 7.15 Rail clicks and `i` are silently ignored during a prompt write (Low)
- Evidence: `LS:22132-22133, 22210-22211`.
- Probe: **Prompts** -> Import… a large prompt file -> immediately click
  **Media** and press `i`; look for any feedback.

### 7.16 Selection-mode Escape is inconsistent across canvases (Low)
- Evidence: Notes Esc exits select mode (`library_notes_controller.py:3043-3053`);
  Media select footer says "esc focus rail" (`LS:1367-1373`) and select mode
  survives rail switches (`LS:25578-25581`); keyboard selection keys exist
  only on Media (`s`/Space).
- Probe: **Media** -> `s` -> Space on two rows -> Esc -> click **Notes** ->
  back to **Media** (still selecting?). Then Notes -> **Select** -> Esc.

### 7.17 Ctrl+S saves skills but not prompts (Low)
- Evidence: only `library_skill_save` is bound (`LS:1047`); no prompt save
  binding anywhere in `Widgets/Library` / `UI/Library_Modules`.
- Probe: **Prompts** -> open a prompt -> edit name -> Ctrl+S -> status line.

### 7.18 Disabled controls whose only reason is a tooltip (Low)
- Evidence: Export row in server mode (`SHELL:723-752`; `RAIL:1323-1331`);
  narrow **‹ Library** bar when guarded (`LS:5658-5671`). Contrast the
  pressable-blocked pattern (`ENTRY:418-422`; `LS:14820-14838`).
- Probe: at 60x24 open **New prompt**, type a name, then try **‹ Library**
  (keyboard Enter and click) and read what explains the refusal.

### 7.19 Internal jargon and five names for the rail/landing (Low)
- Evidence: section 3 table ("rail", "hub", "Landing", "Items", "Nav",
  "Navigation", "Library pane").
- Probe: read the footer on Media list ("esc focus rail"), Import
  ("esc back"), Export from rail ("esc back to hub"), Study staging
  ("esc back to hub"), and F1 on the landing (title "… — Landing").

### 7.20 Artifacts rows use a different footer/Escape grammar and a generic F1 title (Low)
- Evidence: `LS:4708-4714` (no F6 chip, "↑↓ select", "esc items", "/ find");
  `library_artifacts_reader_shell.py:98-112`; `SC:488-503` has no artifacts or
  Study entries.
- Probe: Ctrl+6 -> F1 (title?), Esc from the list (rail reached?).

### 7.21 Named "‹ Library" return exists only on Media at < 64; Prompts paints a 32/18 split with an empty reader at 60 (Low)
- Evidence: `LS:7125-7210` (Media only); table 2.4 row W=60.
- Probe: 60x24 -> **Prompts**, **Skills**, **Conversations**, **Collections**:
  look for a visible way back to the rail besides Esc and the grip.

### 7.22 Details copy claims local storage even in server mode (Low)
- Evidence: unconditional "Everything here is stored on this machine ·
  syncing to a server isn't available yet." (`LS:14806-14808`) vs header
  "Library | Server: …" (`SHELL:766-770`).
- Probe: server runtime (if available) -> open **Details**.

### 7.23 User Guide drift on shell facts (Low, doc)
- Evidence: guide says four rail sections (code has five incl. Artifacts,
  `SHELL:755-764`); "Nav handle is five cells" (ordinary handle is 3,
  `RAIL:541`); research listed as retired into Library (it is a real screen,
  `screen_registry.py:196-198`); "their names now route to the matching
  Library row" (7.6); existing empty profile keeps full rail (7.5).
- Probe: read `Docs/User_Guide/library.md` sections "Getting there",
  "Get started on a new profile", "Layout tour" against a live run.

### 7.24 Minor copy inconsistencies (Low)
- `ctrl+n new note` on the landing vs `n new note` on Notes (`LS:1258-1263, 1547-1552`).
- Recent rows "Notes · <title>" and footer "enter open notes" for one note
  (`ENTRY:164`; `LS:4606-4607`).
- Ingest footer "esc back" / "/ search" vs every other "esc back to …" /
  "/ focus search" (`LS:1386-1391`).
- Character return **Back to Console** vs the "‹ <destination>" back-cue
  grammar (`library_character_return.py:20`; `SHELL:252-270`).
- Rail row tooltips repeat the visible title (`RAIL:1329-1331`).
