# J5: keyboard-only, low-vision and design-system consistency audit (Library + Library › Notes)

- **Build:** origin/dev `2d34cbf80d`, Textual 8.2.8, golden profile, isolated harness (`nlrev`).
- **Sessions:** `nl-j5-1` (120x36, later resized to 80x24 and 60x24) and `nl-j5-2` (160x45 reference). Both were killed at the end, `pgrep -fl "runs/nl-j5"` came back empty, and `snap.sh` against `isolation-before.txt` showed the profile untouched. Neither run log has a Traceback, and neither log mentions the real `~/.config` or `~/.local`.
- **Evidence:** `evidence/j5-a11y-consistency/NN-<what>-<cols>x<rows>.{txt,ansi}`, about 420 captures. PNGs exist for six key moments: 59, 151, 166, 174, 177 and 185.
- **Method:** keyboard only. I used a mouse just once: in step 37, a click to recover from a dead keyboard state. For focus visibility, each pair of consecutive `.ansi` captures went through `a11y/ansidiff.py`; a key press that changed no cells counts as an **invisible focus stop**. Colour claims come from `rowstyles.py`, and contrast was computed with the WCAG formula.

## Persona and goals

**Sam** uses the keyboard only and has low vision. A large terminal font means small grids, so 80x24 and 120x36 are Sam's real sizes. Sam depends on a focus indicator Sam can see, on text labels and on predictable patterns. Sam also acts as the design-system consistency auditor. Sam's goals:

1. Reach every Library destination and every Notes mode without a mouse, always knowing where focus is.
2. Tell active, focused, selected and disabled apart without relying on colour.
3. Trust the footer and F1 to describe the keys that actually work.
4. Find the same grammar everywhere: toolbars, pagers, selection, confirm/undo, back and save.
5. Still complete the main tasks at 80x24 and at 60x24.

## Step log

| # | Intent | Keys | Expected | Observed | Friction | Capture |
|---|---|---|---|---|---|---|
| 1 | Open Library by keyboard | Ctrl+P, "Switch to Library", Down, Enter | Library opens with focus in Library content | Library opens, but focus sits on the nav bar's **⌃1 Home** tab | minor | 02, 03 |
| 2 | Reach the first Library control | Tab ×14 | Tab stays inside Library (as the User Guide says) | 13 stops walk the whole nav bar, which scrolls sideways so that **⌃3 Library** goes off-screen; the 14th lands on rail **Collapse** | major | 04-landing-tab01…14 |
| 3 | Walk the landing | Tab ×34 | Visible focus at every stop | Every stop is visible. Rail rows show a `█` bar plus an underline; quick actions show an underline plus a tint; the **x** (clear search) stop shows only a one-cell underline; section toggles underline only the 3-cell `▾` | minor | 04-landing-tab14…47 |
| 4 | F6 / Shift+F6 on the landing | F6 ×4, Shift+F6 ×2 | Moves between panes | F6 alternates rail search ↔ first recent row. Shift+F6 does nothing | minor | 05-landing-* |
| 5 | F1 on the landing, then close it | F1, Esc | Help lists the keys; focus goes back to the row I was on | Help lists `/`, `i`, `ctrl+n`, `F6`, `enter: open notes`. After Esc, focus is on the nav bar's **⌃1 Home**; the row I was on has lost focus | major | 07, 08, 09 |
| 6 | Open Notes from the rail | F6, Tab ×5, Enter | List with focus on its first row | Nav collapses to a grip and focus parks on the **N a v** grip. Footer shows the generic set | minor | 12, 13 |
| 7 | Tab into the tree past its bottom | Tab ×24 | The list scrolls to show the focused row | Stops 14–24 change **no cells**: focus moves onto folder and note rows below the pane, and the pane never scrolls. Mouse wheel does nothing either | **blocker** (visual) | 14-notes-tab14…24, 15, 16, 17 |
| 8 | Find out what has focus | Enter | — | Opens the hidden row **Daily log 2026-09-20** | major | 18 |
| 9 | Tab through the editor at 120x36 | Tab ×14 | Every control visible | Body → ‹ Notes → Edit → Preview → Info → **[invisible: Save]** → **[invisible: Use in Console]** → Title → Keywords. Only the footer names the two invisible stops ("enter save note", "enter use in Console"). The status line is squeezed into one column and shows only "S" | major | 19-editor-tab01…14 |
| 10 | Tell which mode is active | Enter on Preview | Active tab is marked | Edit, Preview and Info all render `bold #e1e1e1 on #1e1e1e`, before and after the switch | major | 21 |
| 11 | Create a note | Ctrl+N, type a title | Visible status | The status reads "E / n / —" stacked down one column. After the save, the header says "Saved 20:19 · Next: Start typing." | minor | 27, 29 |
| 12 | Delete the note from Info | Shift+Tab ×3, Enter, Tab ×5, Enter | A visible confirm with visible focus | The confirm text is cut off ("Delete this note? Undo will be available"). **Cancel and Delete are below the visible area.** Tab toggles between them invisibly; only the footer ("enter cancel" / "enter delete") shows which one has focus | major | 33, 34, 35, 36 |
| 13 | Confirm the delete | Tab, Enter | Receipt plus Undo | "✓ deleted · J5 temp note" appears, and **Undo** has visible focus (`┃ Undo ┃`) | none | 37 |
| 14 | F1 from Undo, then close | F1, Esc | Focus back on Undo | Focus jumps to the Filter field. The "enter undo delete" chip is gone | major | 38, 39, 40 |
| 15 | Esc on the Notes list | Esc | Focus moves to the rail and nothing else changes (User Guide) | At 120x36 the Nav pane **expands and the Notes list collapses** | minor | 41 |
| 16 | F6 after that | F6 ×5 | Back to the list | Rail search, then **two invisible stops**, then rail search again. Focus never returns to the list | major | 42-rail-f6* |
| 17 | Media at 120x36 | grip, Tab… | — | Tab onto a row loads it in the Reader (focus equals select). The selected row reads "▸ … · loaded"; the Reader mode reads "Read (selected)" | none | 61, 62 |
| 18 | Media select mode | s, Space, Esc | Esc leaves select mode (it does in Notes) | `s` and Space work. **Esc does nothing**; the footer offers "s done selecting" instead | minor | 65, 66, 67, 68 |
| 19 | Esc on a Media list with an item loaded | Esc ×3 | "esc focus rail" | No focus move and no visual change (an advertised key that does nothing) | minor | 70-*, 71 |
| 20 | F6 in Media | F6 ×3 | Rail, then list, then reader | Reader (amber `#fea62b` heavy border) ↔ filter. The collapsed rail is skipped, so the only way back to another destination is Shift+Tab ×2 to the **N a v** grip, then Enter | minor | 72, 73 |
| 21 | Conversations | grip, Tab, Enter | — | Focus parks on the grip. Read and Info have no active marker. F1 lists no Esc. After F1 closes, focus jumps from the focused row to the grip | major | 76, 77, 78, 79 |
| 22 | Prompts | — | — | Focus lands in the filter ("typing in field"). Only 3 of 10 prompts are visible, in an inner scroller, with about 10 empty rows below it | minor | 81 |
| 23 | Edit a prompt name, then try Ctrl+S | End, type, Ctrl+S | Saves, or says why not | Nothing happens. The dirty cues are the footer "esc save or discard first" and a Save changes / Discard changes pair at the bottom. The Esc veto toast covers those two buttons | minor | 85, 86, 87 |
| 24 | Discard | Tab…, Enter | Back on the edited row | The editor closes; focus lands on **sort: Newest**, not on the prompt row | minor | 89, 90 |
| 25 | Esc on the Prompts list (Nav collapsed) | Esc | Rail | Nothing happens | minor | 91, 92 |
| 26 | Skills | — | — | Rows have no `█` focus bar (only underline plus a 1.34:1 tint). The selected marker is "›" here, not "▸". The active mode tab looks **disabled**. After F1 closes, focus is in the nav bar | major | 95, 97, 99, 101 |
| 27 | Collections | — | — | "✓ Read" marks the active reader mode. Rows are centre-aligned and begin with "▸ Loaded in Reader". The sort label reads "Sort: saved desc" | minor | 108 |
| 28 | Search / RAG F6 | F6 ×4 | Reaches the query box | Stuck on the rail search box. Footer advertises "o open evidence" when there is no evidence | major | 112, 113 |
| 29 | Filter Notes at 160x45 | `/`, "Index", Enter, Tab… | Results | "filter: Index · 4 results". The disabled sort reads "○ Sort: Newest" with the line "Sort unavailable — clear the filter" | none | 56, 57, 58 |
| 30 | Esc out of the editor (160) | Esc | Focus back on the note's row | Focus is on "█ Index — start here" | none | 120 |
| 31 | Notes select mode (160) | Select, Enter, Space, Esc | — | Enter toggles a row; Space does nothing; Esc leaves select mode. The label says "Select all **100** shown" though about 13 rows are visible, and "○ Export selected" is cut to "○" | minor | 124, 125, 126, 127 |
| 32 | Sort chooser | Enter on Sort | Focus on "✓ Newest" | **Focus goes to the Filter field**, while the footer says "enter choose sort". Down and Right do nothing | minor | 128, 129 |
| 33 | New-note view, Add from files | Enter, Esc | Focus back on the opener | New-note view: Esc puts focus in the Filter field. Sort strip: Esc returns to its opener (inconsistent) | minor | 131–137 |
| 34 | Reach Folder files | Shift+Tab from rail Import… | — | The source strip, the topmost control on screen, is the **last** stop in Tab order. Its active option has no marker | minor | 132, 140, 141 |
| 35 | 80x24: Search to Notes | Shift+Tab ×27, Enter | — | 27 presses to get from the Search/RAG canvas to the Notes row | major | 151 |
| 36 | 80x24: Tab, F6, F1, Ctrl+P | Tab ×50, F6 ×3, F1, Ctrl+P | Something visible | **Nothing changed for any key, including F1 and Ctrl+P.** One mouse click on a row brought the keyboard back. Two attempts to reproduce this failed | **blocker** (once) | 152–156 |
| 37 | Recover | mouse click, F1 | — | F1 works again | — | 157 |
| 38 | 80x24 editor | Enter on a note | Everything reachable | The compact layout shows the status, Save, Use in Console and inline labels in full | none | 166 (PNG) |
| 39 | Cancel Export text | Info, Export text, Esc | "Export cancelled", focus on the opener | Status reads "**Export failed** — choose another destination and try again. · Next: Review the error". Toast says "Note export cancelled.". Focus lands on **Delete** | major | 168, 169 |
| 40 | Delete confirm at 80x24 | Enter on Delete | Visible buttons | Not visible until I scrolled the wheel by hand | major | 171, 172 |
| 41 | 60x24 | Esc, Tab…, Enter | Operable | The Notes list, editor, Info and the Prompts editor all work. The footer drops "esc …" chips | minor | 174–178 |
| 42 | Same cancel at 160x45 | Info, Export Markdown, Esc | — | Same "Export failed" in both the work-pane status and inside Info, the "cancelled" toast, and focus on Delete | major | 185 (PNG) |

## Task outcomes

| Task | Outcome | Steps / keystrokes | Note |
|---|---|---|---|
| 1. Keyboard traversal of every destination and Notes mode | partial | about 600 key presses across both sessions | Every surface was reached. Invisible stops appeared in the Notes tree, editor Save/Use in Console, the delete confirm and collapsed panes. Focus was lost after F1 and other dismissals on 5 of 6 surfaces tested |
| 2. Colour-only meaning and contrast | partial | — | Disabled controls carry the "○" glyph plus a reason line, at 7.25:1. Active state is **not shown at all** for Notes Edit/Preview/Info, the Notes source strip, and Conversations Read/Info, and it reads as disabled in Skills. Focus tints run 1.34–1.64:1 and rely on an underline |
| 3. Footer and F1 per route | success | 8 routes | Mismatches are listed under findings 6, 10 and 14, and in the matrix |
| 4. Grammar consistency matrix | success | — | See below |
| 5. Docs vs live | success | 12 claims checked | 8 claims disagree (finding 14) |
| 6. Operability at 80x24 and 60x24 | partial | — | The compact layout is usable, and arguably better than 120–160. The delete-confirm buttons are off-screen. Once, the keyboard died at 80x24 |
| Open a specific note by keyboard (120x36) | success | 9 keys (`/`, title, Enter, Tab ×n, Enter) | The tree is clipped, so the filter is the only reliable way in |
| Delete a note and undo it | success | 14 keys | The confirm buttons were never visible at 120x36 |
| Export a note, then cancel | fail (misreported) | 9 keys | Cancel is reported as a failure and focus lands on Delete |
| Switch destination with Nav collapsed (120x36) | partial | 13 keys (Shift+Tab to the grip, Enter, Tab ×n) | 27 Shift+Tabs at 80x24 |

## Emotional journey

- **Start: confident.** The palette finds "Switch to Library" at once.
- **First valley: irritation.** Fourteen Tabs through a nav bar that scrolls the active destination off-screen.
- **Peak: reassurance.** The rail's `█` bar plus underline is easy to follow, and the "enter save note" / "enter undo delete" footer chips feel like a guide holding Sam's hand.
- **Deep valley: disorientation.** In the Notes tree, eleven Tab presses with nothing moving on screen; then Enter opens a note Sam never saw. The editor hides Save. The delete confirm hides its own buttons.
- **Alarm.** Cancelling an export is called a failure, and the next Enter would arm Delete.
- **End: wary.** The compact 80x24 layout is calmer and complete. But one dead-keyboard episode at 80x24 means Sam would keep a mouse nearby, which Sam cannot use well.

## Strengths

1. **Disabled actions can be read and explain themselves.** "○ Add to folder" / "○ Move note" come with "Note actions unavailable — select a note in the list", and "○ Sort: Newest" with "Sort unavailable — clear the filter"; Media select mode has "Select items to enable.". The labels measure `#9e9e9e` on `#0d0d0d`, which is **7.25:1**, so the meaning rests on a glyph and a sentence, not colour. This matches DESIGN.md's Legible Disabled Rule (captures 13, 57, 65).
2. **The footer names the focused action.** Chips such as "enter save note", "enter use in Console", "enter cancel" / "enter delete" and "enter undo delete" follow focus. They were the only reason Sam could operate controls that were clipped or off-screen (steps 9, 12). F1 shows the same set for each surface, plus whatever gated bindings are active.
3. **Recovery after a delete is solid.** Delete returns to the list with "✓ deleted · <title>" and focus already on **Undo**. Esc from the editor returns focus to that note's own row (capture 120). The compact 80x24 editor puts every control on screen with inline labels (capture 166).

## Findings

Ranked most severe first. Each has evidence, cause (traced to code where I could), and a fix.

### F1 (P1): the Notes tree does not scroll at ≥ 120 columns; rows below the pane cannot be seen, and Tab moves focus onto them invisibly

- **Evidence:**
  - At 120x36, Tab stops 14–24 changed no cells (`14-notes-tab14…24-120x36`).
  - Wheel over the tree did nothing (16, 17). Enter then opened the hidden "Daily log 2026-09-20" (18).
  - At 160x45 the same happens after "A very long note title…" (`53-notes-g-tab16…19`). The wheel scrolls the rail (54, the control case) but not the tree (55).
  - The unfiltered tree's last rows, "Load more notes" and "Recently deleted (2)", are never visible, so the Trash only shows up while a filter is active.
  - In select mode the label says "Select all 100 shown" (124) when about 13 rows are visible.
- **Who it hurts:** everyone at a desktop width, and keyboard users most, because focus disappears.
- **Cause (traced):**
  - `LibraryNotesCanvas(… Vertical)` (`Widgets/Library/library_notes_canvas.py:909`) and `Vertical(id="library-notes-list")` (`:2345`) inherit Textual 8.2.8's `Vertical { overflow: hidden hidden }`.
  - Only the compact rule `#library-shell-grid.library-notes-compact #library-notes-list { overflow-y: auto }` makes it scroll (`css/features/_library.tcss:1004-1013`).
  - Compact applies only when the shell is narrower than 120 (`UI/Library_Modules/library_notes_controller.py:2867`). So 80x24 scrolls, and 120x36 and 160x45 do not.
- **Fix:**
  - Give the non-compact `#library-notes-list` `height: 1fr; overflow-y: auto` (in `css/features/_library_panels.tcss:543`, the source of the generated `screen_agentic_library.tcss`), or make it a `VerticalScroll`.
  - Add a focus-in-view test: Tab onto the 30th tree row at 120x36 and assert the row's region lies inside the list's visible region.
  - Make "Select all N shown" count rows that are actually visible, or rename it "Select all N loaded".

### F2 (P1): closing F1, or any modal, drops keyboard focus on most Library surfaces

- **Evidence:**
  - Landing: focus moves from the focused recent row to the nav bar's "⌃1 Home" (06, then 08, then 09: Tab walks the nav bar).
  - Notes: focus moves from Undo to the Filter field (37 → 39 → 40).
  - Conversations: from the focused row to the Nav grip (77-tab12 → 79).
  - Skills: into the nav bar (97-tab1…6).
  - Media keeps focus (64), which shows it can be done.
- **Who it hurts:** keyboard users. Each trip to help costs up to 14 Tabs to get back.
- **Cause (traced):**
  - Textual posts `ScreenResume` when a modal pops.
  - `LibraryScreen.on_screen_resume` (`UI/Screens/library_screen.py:9262-9299`) runs `_refresh_library_visit_surfaces` on **every** resume. That re-requests the Notes tree (`:9418-9422`), the Prompts and Skills lists with `focus_identity=None` (`:9423-9439`) and the Conversations page (`:9320-9330`), so the focused widget is recomposed away.
- **Fix:**
  - Skip the visit refresh when the resume follows a modal pop. For example, set a flag in `on_screen_suspend` only when the active screen changes to another *route*, not when a `ModalScreen` is pushed.
  - Otherwise, capture `self.focused` identity before `push_screen` and restore it after the refresh, the way Media already does.

### F3 (P1): cancelling a note export is reported as "Export failed", and focus then lands on Delete

- **Evidence:**
  - 80x24 Export text → Esc (169).
  - 160x45 Export Markdown → Esc (185, PNG). The work-pane status and an inline line in Info both read "Export failed — choose another destination and try again. · Next: Review the error, then keep editing.", while the toast says "Note export cancelled.".
  - The footer becomes "enter delete note", with `┃ Delete ┃` focused.
- **Who it hurts:**
  - Anyone who changes their mind gets told something went wrong.
  - Keyboard users are one Enter away from opening the delete confirm (the inline confirm still guards it).
- **Cause (traced):** `library_screen.py:21136-21144` handles `selected_path is None` with `_finish_library_notes_operation(..., success=False, failure_next_action="choose another destination and try again")`. The focus move to Delete is a hypothesis: the export operation disables the Info actions, so focus can't come back to "Export Markdown" and falls through to the next enabled control.
- **Fix:**
  - Add a `cancelled` outcome that writes no failure copy (status returns to "Saved", or reads "Export cancelled.").
  - When FileSave dismisses, re-focus the button that opened it (`#library-note-context-export-md` / `-txt`).

### F4 (P1): from 120 to about 160 columns the editor clips Save and "Use in Console" and squeezes its status into one column; Tab and F6 land on the hidden Save

- **Evidence:**
  - At 120x36, Tab stops 5–6 change no cells while the footer reads "enter save note" and "enter use in Console" (19-editor-tab05, 06).
  - F6's first work-pane target is Save.
  - The status renders as "S" (19), or as "E / n / —" stacked vertically for "Empty note — type to keep it" (27).
  - At 160x45 with Nav open, Save is gone too (184). With Nav collapsed, the label reads "Use in" (59).
  - The same editor at 80x24 (compact) shows everything (166).
- **Who it hurts:** low-vision users at mid widths. The User Guide tells them to "use the visible **Save** button".
- **Cause (traced):** `#library-note-status`, `#library-note-mode-controls` and `#library-note-task-actions` share one horizontal row (`library_notes_canvas.py:2820-2860`). F6 prefers `library-note-save` (`library_screen.py:1484-1494`). Compact mode starts only below 120.
- **Fix:**
  - Put `#library-note-status` on its own full-width line.
  - Let `#library-note-task-actions` wrap under the mode controls whenever the work pane is narrower than the sum of their widths. Or use the compact editor layout whenever the **work pane** (not the shell) is narrower than about 70 cells.
  - Make F6 target Title when Save has no visible region.

### F5 (P1): the inline delete confirmation's Cancel and Delete buttons are off-screen, and focus on them is invisible

- **Evidence:** 120x36 (33–36) and 80x24 (171). Tab toggles between them and only the footer changes. A manual wheel scroll at 80x24 shows them ("┃ Cancel ┃   Delete", 172). At 120x36 even the wheel did not reveal them (36).
- **Who it hurts:** keyboard and low-vision users confirming something destructive. They cannot see what they are about to press.
- **Cause (hypothesis):** focus moves to Cancel while the confirm is still mounting, so `scroll_visible` has no region yet. The Info region then doesn't scroll at 120.
- **Fix:**
  - After the confirm mounts, `call_after_refresh(lambda: cancel.scroll_visible(animate=False))`.
  - Better still, render the confirm directly under the "Danger" heading, replacing the Delete button in place, so its position never depends on scrolling (`library_notes_canvas.py:3099-3123`).

### F6 (P1, low confidence): once at 80x24, the keyboard stopped working entirely (Tab, F6, F1, Ctrl+P) until a mouse click

- **Evidence:**
  - Sequence: resize 120 to 80 while on Search/RAG, Shift+Tab ×27 to rail **Notes**, Enter (151).
  - Then 50 Tabs, 3 F6 and F1 changed no cells (152–156), and Ctrl+P opened no palette.
  - The app process was idle (0.1% CPU) and alive: one click on "▸ Journal" expanded it, and F1 worked straight after (157).
  - Two exact replays did not reproduce it.
- **Who it hurts:** keyboard-only users. To them this looks like a hang.
- **Cause (hypothesis):** focus was left on a rail widget that the 120 to 80 recompose removed, which breaks binding dispatch until something re-focuses.
- **Fix:** whenever the reader recompose collapses the Library pane while it holds focus (`_apply_library_notes_stage_visibility_for_resize`), move focus to the list's first row. Add a Pilot test: resize across 120 with focus in the rail, then assert `app.focused` is attached and visible.

### F7 (P2): active state is invisible or misleading on mode and segment controls

- **Evidence:**
  - Notes **Edit / Preview / Info** render identically in Edit and in Preview (21). `library_notes_canvas.py:3536-3541` sets an `is-active` class that no stylesheet styles.
  - The Notes source strip **Library notes | Folder files** renders identically whichever is active (131, 141): `widget_defaults_scoped.tcss:874` sets `.-selected { text-style: bold }`, and Button text is already bold.
  - Conversations **Read / Info** are identical (76).
  - The active Skills mode is dimmed to `#a2a2a2 on #1a1a1a` so it looks disabled; when focused, its brackets measure 1.98:1 (101-tab5).
  - Other surfaces do it right: Media "Read (selected)", Collections "✓ Read", Sort "✓ Newest", Conversations "✓ Active".
- **Fix:** use the house "✓ " prefix (already used by choosers) for every mode and segment control, applied in the canvas builders. Delete the dimmed-active style in Skills.

### F8 (P2): the Escape and F6 contracts differ by destination, and some advertised keys do nothing

- **Evidence:**
  - "esc focus rail":
    - On Notes at 120x36 it expands Nav and **collapses the list** (41). The User Guide says it "never… changes what's shown".
    - On Media (70) and Prompts (91, 92) with Nav collapsed it does nothing.
    - On Skills, Collections and Study the chip is the same, but the result was not verified.
  - Media select mode ignores Esc (67), while Notes select mode exits on Esc (127).
  - F6 on Search/RAG never leaves the rail search box (113). After Notes' Esc, F6 cycles through two invisible targets (42).
  - The footer advertises "o open evidence" with no evidence present (112).
- **Cause (traced):** `action_library_list_focus_rail` (`library_screen.py:27214-27240`) focuses `#library-search-input` even when the rail is collapsed. The default F6 target list has no Search/RAG canvas ids and no first-focusable fallback (`library_screen.py:1402-1443`; `Widgets/workbench_focus.py:70-82`).
- **Fix:**
  - When the Library pane is collapsed, either open it consistently in every reader (and say so: "esc open rail") or drop the chip.
  - Make Esc in Media select mode leave select mode.
  - Add the Search/RAG query and mode controls to the default F6 targets, and give the resolver a first-visible-focusable fallback.
  - Show "o" only when an evidence card exists.

### F9 (P2): on arrival, focus is not where the User Guide says, and the order of the top controls breaks reading order

- **Evidence:**
  - After the palette opens Library, focus is on the nav bar's "⌃1 Home" (03), and Tab walks 13 nav items (04-tab01…13).
  - Entering Notes, Conversations or Collections at 120x36 parks focus on the Nav grip (13, 76, 108). Prompts lands in its filter (81).
  - The guide promises that "Entering a … list … focuses the list's first row".
  - The **Library notes | Folder files** strip at row 5 is the **last** Tab stop: Shift+Tab from rail "Import…" wraps to it (140), and Tab from the New-note view reaches it only after the work pane (132).
- **Fix:**
  - After palette or nav-bar navigation into Library, focus the destination's first content control.
  - On list entry, focus the first row.
  - Move `#library-notes-source-strip` ahead of `#library-shell-grid` in the focus chain, or give it `can_focus` order by mounting it first (`library_browse_route_swap.py:109-160`).

### F10 (P2): the Notes sort chooser opens with focus in the Filter field

- **Evidence:** Enter on "Sort: Newest" opens "✓ Newest  Oldest  Title" but focuses the Filter input. Down and Right do nothing, and one Tab is needed to reach "✓ Newest", while the footer already says "enter choose sort" (128, 129).
- **Fix:** focus the current "✓" option when the strip opens (handler `library_notes_controller.py:4916-4952`). That matches Esc, which already returns focus to the opener.

### F11 (P2): Notes text fields do not switch the footer to "typing in field", so bare-letter chips collide

- **Evidence:**
  - With the Notes filter, Title or Keywords focused, the footer never shows "typing in field" (14-tab01, 29, 56).
  - The rail search box, the Conversations filter (77-tab06) and Prompts (81) all do.
  - At 80x24, pressing `g` (go to folder) after closing the palette typed "g" into the Notes filter (164).
- **Fix:** include `#library-notes-filter`, `#library-note-title`, `#library-note-keywords` and `#library-note-context-keywords` in the typing-in-field transform (`library_screen.py:4736-4831`).

### F12 (P2): focus indicators are weak and differ from surface to surface

- **Evidence:**
  - At least six focus treatments are in use:
    - `█` bar plus underline (rail and list rows);
    - `┃ label ┃` brackets (Notes and Media toolbars);
    - underline plus tint only (quick actions, chips, source checkboxes);
    - box border (rail search, Conversations filter);
    - an amber `#fea62b` heavy border (Media reader, 72);
    - a lone 1-column blue `│` or `┐` for container stops (Prompts list 82-tab6, Skills work pane 98, Add-from-files 136-tab3).
  - Skills rows drop the `█` bar (99).
  - The rail **x** clear button's focus is a one-cell underline (04-tab17).
  - Tint contrasts measure 1.34–1.64:1 against the resting fill.
- **Fix:**
  - Adopt the `┃ ┃` bracket (or DESIGN.md's `outline: heavy $accent`) for every Button and toggle.
  - Use the `█` bar for every row type, Skills included.
  - Make container scrollers non-focusable unless they are the only target in their pane, or give them a full heavy border.

### F13 (P3): copy and layout polish observed along the way

- **Truncation at 120x36:** the ordinary rail header reads "Navigat…" (02, 112); 80x24 shows "▾ scroll for" (158).
- **Prompts at 120x36:** only 3 of 10 prompts are visible in an inner scroller with empty space below (81).
- **Toast placement:** "Unsaved Prompt changes — Save or Discard changes first." covers the Save changes / Discard changes buttons it names (87).
- **Stale guidance:** "Saved 20:19 · Next: Start typing." after the user has already typed (29).
- **Long-lived receipt:** "✓ deleted · J5 temp note" was still on the Notes list about 40 minutes and many destination hops later (175).
- **Jargon:** the Study handoff footer says "esc back to hub", but the landing is never called "hub" on screen; its F1 title is a generic "Library Shortcuts" (182).

### F14 (P3, docs): User Guide claims that disagree with the live app

| Claim | Where | Live |
|---|---|---|
| "Tab stays on this screen; the top navigation bar is reached with its own keys…, never by tabbing" | `library.md` Keyboard table | Focus starts in the nav bar after the palette and after F1 closes on the landing or Skills, and Tab walks it (04, 97) |
| "Entering a Media, Notes, Prompts, or Skills list… focuses the list's first row" | `library.md` | Notes and Conversations park on the Nav grip; Prompts lands in its filter (13, 81) |
| "On the plain list — Escape moves focus to the rail's Search Library… box…; it never… changes what's shown" | `library.md` | Notes 120x36 collapses the list (41); Media and Prompts do nothing (70, 91) |
| "↑ / ↓ inside a … Notes … list — move to the previous/next row" | `library.md` | Arrows do nothing on folder rows (24-list-g-down1…3) |
| "Escape … in any search or filter box … the footer switches from 'typing in field'" | `library.md` | The Notes filter never shows "typing in field" (F11) |
| "Use the visible **Save** button" | `notes.md` ≈ l.493 | Save is off-screen at 120x36, and at 160x45 with Nav open (F4) |
| "Info's footer is fixed instead — 'enter run action'" | `notes.md` ≈ l.512 | Info names the focused action: "enter copy note", "enter export Markdown", "enter export text", "enter delete note" (32) |
| "Escape — Returns to the list — one press, from Edit, Preview, or Info. From Info it goes back to the editor first." | `notes.md` editor-keys table | The two sentences contradict each other; live, Info goes to the editor first ("esc back to note") |
| "Delete … the prompt renders … on the row directly under the Delete button" | `notes.md` ≈ l.470 | It renders there but out of view (F5) |
| "Export…" | `notes.md:391` | The toolbar label is "Export" (12) |
| Select mode keeps "all four actions … on the pane" | `notes.md:392` | True in compact; at 160x45 "○ Export selected" is cut to "○" (124) |

Fix: correct these sentences, and add a live-verified screenshot reference for the Keyboard tables.

## Grammar consistency matrix (live, golden profile)

| Element | Notes | Media | Conversations | Prompts | Skills | Collections | Search / RAG |
|---|---|---|---|---|---|---|---|
| **Primary toolbar** | New · Sort: Newest · Select / Add from files… · **Export** / Folders & placement | type: All types · sort: Newest / **Export…** · Trash · Select / Review these; header "Sets" | ✓ Active · Archived · All / **Export…** · Select | collection: All prompts / sort: Newest · Select / Import… · **Export…** | sort: Name · Import skill… (+ "Set up skill trust") | Quick Capture · Filters / Sort: saved desc | mode: ✓ Search ⇄ RAG Answer · Run |
| **Sort label case** | "Sort:" | "sort:" | — | "sort:" | "sort:" | "Sort:" + raw enum, press-to-cycle | — |
| **Pager** | "Notes 1–20 of N · Load more notes" (tree; hidden by F1) | "1-20 of 23 · Page 1 of 2" + "Already on the first page." + ○ Previous / Next | "1-12 of 12" | "1-10 of 10" | "1-4 of 4" | "1–6 of 6" (en dash) | "Evidence · top 15 per source" |
| **Selection mode** | Select/Done; **Enter** toggles; no s/Space; Esc exits | **s** / **Space**; Esc **inert** | Select/Done (Archive/Restore selected) | Select (Export/Delete selected) | none | none | single evidence selection |
| **Confirm / undo / receipt** | Inline confirm in Info (off-screen, F5) → "✓ deleted · t" + **Undo**/Dismiss | Bulk "Delete N selected items?" inline; Trash | Modal "Archive N conversation(s)?" + Undo | Modal "Delete Prompt?" + Undo receipt | inline "Reset skill trust…" | "Move to Archive" (no confirm) | — |
| **Back control** | "‹ Notes" (wide) / "‹ Back to list" (compact): same control, two labels | "‹ Back" | none (reader inline); "Back to Console" bar from Console | "‹ Back to list" | "Back to list" (no ‹) | none | — |
| **Filter** | "Filter notes… (Enter)", Enter-applied, status "filter: q · N results" + Clear filter | "Title/keyword…", as you type | "Filter conversations… (Enter)" | "Filter prompts… (Enter)", debounced | "Filter skills… (Enter)" | "Filter captures" | two mirrored boxes (rail + canvas) |
| **"typing in field" footer** | **no** | yes | yes | yes | — | — | yes |
| **Empty work pane** | "Select a note to edit it here." | "Select a media item to read it here." | auto-loads newest | "Select a prompt to edit it here." | "Select a skill to inspect it here." | auto-loads first | "No evidence yet. Run Search/RAG to populate results." |
| **Status line** | Authority "Library notes · Ready · Next: …" + editor "Saved · Next: …" | scope line "Media · 23 of 23 · all types · sort: Newest" | "Loaded <t> · 40 of 40 messages · complete." | none until dirty (footer only) | "Trust: trust uninitialized" | "Local Collections · example.org" | "Enter a question or search query." |
| **Save model** | Autosave 2 s + Save button; **no Ctrl+S** | n/a (metadata edits explicit) | n/a | Explicit Save changes / Discard; **no Ctrl+S**; Esc vetoed | Explicit Save skill; **Ctrl+S** (hint "ctrl+s Save · esc Back to list") | n/a | n/a |
| **Active mode marker** | **none** (Edit/Preview/Info) | "Read (selected)" | **none** (Read/Info) | not verified | **dimmed** (looks disabled) | "✓ Read" | "✓ Search" |
| **Selected-row marker** | none visible in tree | "▸ … · loaded" | "▸" | none | "›" | "▸ Loaded in Reader" prefix | ☑ checkboxes |
| **Row focus marker** | █ + underline | █ + underline | █ + underline | █ + underline | **underline + tint only** | not checked | underline + tint |
| **Entry focus (120x36)** | Nav grip | toolbar chooser | Nav grip | filter | — | Nav grip | rail row |
| **Esc on list (Nav collapsed)** | expands Nav, collapses list | no-op | focus filter / Library | no-op | (after F1: nav bar) | grip | rail search |
| **F1 title** | "Library Shortcuts — Notes" (also used in Folder files) | "— Media" | "— Conversations" (no Esc listed) | not captured | "— Skills" (F6/esc/shift+f6 only) | "— Collections" | not captured; Study handoff: generic "Library Shortcuts" |

## Improvement opportunities

1. **A focus-in-view invariant for the whole Library.** A shared Pilot check run at 80, 120 and 160 columns: Tab through each surface, and for each focused widget assert two things. Its region must intersect the visible region of every scroll ancestor. Its style must differ from the resting style by a glyph or weight change, not only a tint. That one test would have caught F1, F4, F5 and F12.
2. **A single `restore_focus_after_modal` helper** used by every Library `push_screen` call: F1, FileSave, folder dialogs and confirmations. Pair it with skipping the visit refresh on a modal pop (F2, F3).
3. **A "Prefer compact layout" setting, and a compact threshold based on work-pane width.** Low-vision users with large fonts get the calmer 80x24 layout, which has everything on screen, at any width. The layout switch would follow the work pane's real width instead of the shell's.
4. **Mode keys for the Notes editor.** For example Alt+1/2/3 for Edit/Preview/Info and Alt+S for Save, advertised in the footer, so the Tab loop (9 stops) isn't the only way between modes. Notes' explicit Save also needs a reachable key now that Skills owns Ctrl+S.
5. **One segment-control component** (label plus "✓ " active prefix plus `┃ ┃` focus) for every mode, scope, sort and source switch. That replaces six visual dialects with one.

## Nielsen heuristics (0–4)

| Surface | # | Heuristic | Score | Key issue |
|---|---|---|---|---|
| Library | 1 | Visibility of system status | 2 | Invisible focus stops; active modes unmarked in Conversations; collapsed panes hide where focus went |
| Library | 2 | Match with the real world | 2 | "hub", "rail", "Nav", "Library pane" all name the same two things |
| Library | 3 | User control and freedom | 2 | Esc means a different thing per destination; focus lost after F1 |
| Library | 4 | Consistency and standards | 1 | See the matrix: six focus styles, three sort-label cases, four back labels, three active-state idioms |
| Library | 5 | Error prevention | 3 | Disabled controls carry reasons; dirty vetoes guard Prompts and Skills |
| Library | 6 | Recognition rather than recall | 2 | Footer chips help, but footers truncate at 120 and below, and F1 becomes required |
| Library | 7 | Flexibility and efficiency | 2 | Media has accelerators; other destinations need 13–27 Tab hops to switch |
| Library | 8 | Aesthetic and minimalist design | 2 | Duplicate search boxes on Search/RAG; "Navigat…" truncation; Prompts list wastes half the pane |
| Library | 9 | Help users recover from errors | 2 | Undo receipts are good; the toast covers the Save/Discard buttons it names |
| Library | 10 | Help and documentation | 2 | F1 per surface is good; the User Guide's keyboard tables drift (F14) |
| Notes | 1 | Visibility of system status | 1 | Clipped tree; status squeezed to "S"; hidden Save; cancel reported as "Export failed" |
| Notes | 2 | Match with the real world | 2 | "Folders & placement", "Remove placement" jargon |
| Notes | 3 | User control and freedom | 3 | Autosave, Undo receipt, Esc returns to the note's row; focus loss after modals |
| Notes | 4 | Consistency and standards | 2 | Same back control has two labels; Enter-only selection where Media uses s/Space; "Export" vs "Export…" |
| Notes | 5 | Error prevention | 3 | Inline delete confirm and disabled reasons; but the confirm can't be seen |
| Notes | 6 | Recognition rather than recall | 2 | Active mode and active source unmarked; arrows dead on folder rows |
| Notes | 7 | Flexibility and efficiency | 2 | n, /, g, Ctrl+N, Ctrl+End exist; no mode/save/select keys |
| Notes | 8 | Aesthetic and minimalist design | 2 | 4–5 rows of always-on prose push the tree below the fold at 120x36 |
| Notes | 9 | Help users recover from errors | 2 | The export-cancel "failure" sends users to fix something that didn't break |
| Notes | 10 | Help and documentation | 2 | F1 is accurate to the footer; the guide contradicts itself on Esc and on Info's footer |

## Harness caveats

- **Ctrl+digit can't be sent** through tmux, so Ctrl+3 / Ctrl+6 (the direct Library hotkeys) were not tested. Keyboard entry went through the Ctrl+P palette.
- **The 80x24 dead-keyboard episode (F6) happened once.** Two exact replays didn't reproduce it, and tmux key loss can't be fully ruled out. The app, though, was alive and answered a mouse click.
- **The skill-edit save was not verified.** My typed text didn't land in Description (cause unknown; probably my Tab count), so Ctrl+S on Skills is checked only through the on-screen hint and the map.
- **Media select mode showed "No analysis provider is configured"** although the shared mock OpenAI provider was set up. I did not investigate it, and it is not reported.
- **Null keyring**, so the Skills trust banner reads "trust uninitialized". Not reported.
- **PNG renders are approximate**, with no box borders. Every claim rests on the `.txt` / `.ansi` captures.
