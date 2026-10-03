# J1: First-time user ("Jordan") in Library ▸ Notes

- **Code:** worktree `notes-library-ux-review` @ origin/dev `2d34cbf80d`, Textual 8.2.8. Driven live through the nlrev harness.
- **Sockets:** `nl-j1-1` was the main run: empty profile at 120x36, then relaunched with `REUSE=1` at 160x45. `nl-j1-2` was used for data-loss and validation repros on a fresh empty profile at 120x36. Both were killed at the end, `pgrep` came back empty, and `snap.sh` showed the real profile untouched.
- **Captures:** `../evidence/j1-firsttimer-notes/NN-<what>-<cols>x<rows>.{txt,ansi}`. Six key moments also have a `.png`: 08, 16, 20, 25, 49 and 66.
- **Date:** 2026-10-02, 20:05–20:35 PDT.
- **Blind marker:** `[B]` marks a finding from the blind pass (Phase 1: app screens and F1 only). `[P2]` marks one from the map-informed probe (Phase 2).

## Persona and goals

Jordan is a grad student. He is comfortable opening a terminal app but has never used Chatbook. He reads labels literally, hesitates before unfamiliar controls, and gives up rather than guessing. His profile is empty. His goals:

1. Find where to write a note.
2. Write a first note with a heading, a bulleted list and a checkbox, and know that it is saved.
3. Leave and come back, and find it again.
4. Rename it and tag it.
5. Link a second note to the first and follow the link.
6. Delete a note and get it back.
7. Bring in an existing folder of Markdown notes.
8. Learn shortcuts from help.
9. Ask Chatbook about a note.

## Step log

| # | Intent | Keys / clicks | Expected | Observed | Friction | Capture |
|---|---|---|---|---|---|---|
| 1 | Find "Notes" | read nav bar | a Notes tab | Tabs are Home, Console, Library, Roleplay, Watchlists… There is no "Notes". | minor | 01 |
| 2 | Look under More | click `More ▾` | Notes listed | The "All destinations" list has no Notes (it does have F7 Meetings). | minor | 02 |
| 3 | Try Library | `Esc`, click `⌃3 Library` | somewhere to keep things | "Get started — Import a file / Find it / Use it in Console" with buttons [Import…] [New note]. Jordan is relieved. | none | 03 |
| 4 | Start a note | click body `New note` | an editor | New note view: "Ready · Next: Press Blank note, or choose a template.", with focus on **Blank note**. | none | 04 |
| 5 | Create | `Enter` | editor | Editor opens with Title focused. Status reads "Empty note — type to keep it · Next: Start typing." A stray vertical **E / n / —** is painted down the pane edge. | minor | 05 |
| 6 | Title | type "Thesis meeting prep" | — | "Unsaved changes"; the vertical stack now reads **U / c**. | minor | 06 |
| 7 | Body | `Tab` `Tab`, type `# Questions…`, `- Chapter 2 scope`, `- Which datasets…`, `- [ ] Email draft by Friday` | text in body | Body shows only 4 lines at 120x36, then "Saved 20:11 · Next: Keep editing; changes save automatically." | none | 08 |
| 8 | Check rendering | click `Preview` | formatted note | Heading and bullets render. The checkbox prints literally as "• [ ] Email draft by Friday". | minor | 09 |
| 9 | Leave and return | click `⌃2 Console`, then `⌃3 Library` | find the note | Returns straight to the same note. A new folder "▸ Agent_Lessons" has appeared in the tree, unexplained. | none | 10 |
| 10 | Rename | click `Edit`, click Title, `End`, 4×`BkSp`, type "agenda" | rename | List row becomes "Thesis meeting agenda · now"; status "Saved 20:12". | none | 11 |
| 11 | Tag | `Tab`, type "thesis, advisor" | keyword | Saved. Info shows the keywords, re-sorted as "advisor, thesis". | none | 12 |
| 12 | Info | click `Info` | properties | "Linked from — checking…" was still showing after 15 s. It resolved only after reopening the note from the list. | minor | 12, 14 |
| 13 | Second note | `Esc`, `Esc`, `New`, `Enter`, type title, `Tab` `Tab`, type "See " | — | — | none | — |
| 14 | How do I link? | `F1` | help on linking | "Library Shortcuts — Notes" lists "- : typing in field", "esc: back to list", "F6", "shift+f6", "ctrl+n". There is nothing on linking. | major | 15 |
| 15 | Close help | `Esc`, wait 25 s | autosave | Still "Unsaved changes · Next: Keep editing; changes save automatically." The DB still holds "Untitled" with an empty body (v1). Autosave is dead. | **blocker** (data) | 16 |
| 16 | Try palette | `Ctrl+P`, "link" | a link command | Only Chunking Lab, Artifacts and Skills come up. | major | 17 |
| 17 | Guess Obsidian syntax | type `[[Thesis meeting agenda]] for the questions.` | link | Saved, but Preview shows the raw `[[Thesis meeting agenda]]` as plain text (same colour, no underline). Clicking it does nothing. | major | 18 |
| 18 | Check backlink | open first note ▸ Info | "Linked from (1)" | "Linked from (0) — no notes link here yet". **Task 5 failed; Jordan gives up.** | blocker | 19 |
| 19 | Delete | open 2nd note ▸ `Info` ▸ `Delete` | confirm prompt | Only the footer changes ("enter cancel \| tab switch button"). Scrolling Info reveals "Delete this note? Undo will be available" and no buttons at all. | major | 20, 21 |
| 20 | Confirm blind | `Tab` (footer now "enter delete"), `Enter` | deleted | "✓ deleted · Advisor meeting follow-up" with ┃Undo┃ Dismiss, plus "Recently deleted (1)". | none | 22 |
| 21 | Recover later | `Dismiss`, click `Recently deleted (1)`, `Restore` | note back | Clear copy: "Deleted notes stay here until you restore them…". Restored; Notes (2). | none | 23, 24 |
| 22 | Import folder | `Esc`, `Add from files…` | options | "Import once — Copy files into Notes… Later changes to the originals are not tracked." / "Keep a folder synced — …" / "Neither? … Folder files". The copy is literal and clear. | none | 25 |
| 23 | Pick folder | `Import once`, click `📁 fixtures`, `Enter` | open fixtures | A single click had already entered fixtures, so `Enter` opened **import-docs** instead. Back via `📁 ..`, then `📁 md-folder`. "1 of 2 entries shown" and "0 notes" on fixtures were puzzling. | minor | 26–28 |
| 24 | Import | `Select folder`, `Check selection`, `▶ md-folder`, `Import selected items` | notes created | The review lists each file with its resulting title, e.g. "Create 1 new note: Ideas · Create in md-folder ☐ Skip ☑ Create new". Then "Import completed. 10 notes created". | none | 29–32 |
| 25 | See them | `View 10 imported notes` | the 10 notes | List shows "▸ md-folder" **collapsed**, so the notes are not visible until it is expanded. | minor | 33, 34 |
| 26 | Help again | `F1` in list | useful help | 6 key rows. It lists "ctrl+n" while the footer says "n new note", and "/: focus search" where the footer says "/ find note". | major | 35 |
| 27 | Find first note | wheel and arrow keys over the tree | scroll to Unfiled | The tree does not scroll (0-cell diff after 10 wheel events). Arrow focus moves to rows off-screen, and `Enter` opened "Ideas", which had never been visible. | major | 36–39, 56–58b |
| 28 | Filter instead | type "thesis", `Enter` | find it | "filter: thesis · 2 results". | none | 40 |
| 29 | Ask Chatbook | open note, look for an action | "Use in Console" | Nothing visible at 120x36: the Save / Use in Console / Discard buttons are off the right edge. | major | 41 |
| 30 | Palette | `Ctrl+P`, "console" | note action | Only "Switch to Console". | minor | — |
| 31 | Console route | `⌃2 Console`, `Search Library`, type query | search | The query field was not focused, so the typed text was lost. After clicking the field and searching: "Staged for next send · 1 source". | major | 42, 43 |
| 32 | Send | type a question, `Enter` | answer | "Console send blocked: Library search has no available evidence. Review source authority before sending." The search was still loading an embeddings model (15.7 s). | major | 44 |
| 33 | Retry | `Un-stage`, then the late results stage themselves anyway, `Send` | answer | Answer grounded in both notes ("[S1] NOTES — Advisor…"). | minor | 45, 46 |
| 34 | 160x45 relaunch | `⌃3 Library`, `Notes (12)`, open note | persisted | All 12 notes present. Toolbar shows `Save` and a truncated **`Use in`**. Status is still a 1-column "S". | minor | 47–49 |
| 35 | Ask (160) | click `Use in` | staged | Console opens with "Staged for next send · 1 source — Thesis meeting agenda — note" and the prompt "Use this note as context and help me work with it." pre-filled. `Enter` gives a grounded reply. | none | 50, 51 |
| 36 | Delete (160) | Info ▸ Delete | prompt | "Delete this note? Undo will be available in the Notes list." with ┃Cancel┃ Delete visible. | none | 52 |

Phase 2 probes (map-informed): steps P1–P8 are written up under the findings. They are the F1/quit data-loss repro (53–55), the tree scroll check (56–58b), filter partial and keyword (59, 60, 67), landing Import (61), the trailing-space veto (62, 63), the canonical link and backlinks (64, 65) and stale backlinks on a new note (66).

## Task outcomes

| Task | Outcome | Steps | Actions (keys+clicks; typed text counted as 1) | Note |
|---|---|---|---|---|
| 1 Find where to write | success | 3 | 3 | One wrong turn (More ▾). Notes is not named anywhere outside Library. |
| 2 Create note, confirm saved | success | 6 | ~14 | "Saved 20:11" is clear. The checkbox does not render in Preview. |
| 3 Leave and find again | success | 2 | 2 | Same note reopens. After restart at 160: 3 clicks. With 12 notes at 120x36 the note was below an unscrollable fold, so the filter was needed. |
| 4 Rename + keyword | success | 6 | ~10 | Works. Phase 2 showed the keyword cannot be searched ("zebra · 0 results"). |
| 5 Link + follow | **fail** | 9 | ~16 | No affordance. Typed `[[Title]]` is inert, and the target says "Linked from (0)". |
| 6 Delete + get back | success | 8 | 8 | At 120x36 the confirm buttons are invisible: Tab+Enter blind, guided only by the footer. Recovery is excellent. |
| 7 Import md-folder | success | 11 | ~13 | The choice is understandable and the result matched expectation (copies plus a Library folder "md-folder"). The receipt's "View" lands on a collapsed folder. |
| 8 Help | partial | 2 | 2 | A key list only, with a malformed row. Nothing on autosave, links, Delete or Use in Console. |
| 9 Ask Chatbook about a note | partial (120) / success (160) | 12 / 3 | ~25 / 3 | At 120x36 "Use in Console" is off-pane; the Console search route needed a retry and showed a jargon error. At 160x45, one click plus Enter. |

## Emotional journey

- **Start: unsure.** "Notes" is not in the nav bar or in More ▾. Jordan guesses Library.
- **First peak.** The empty Library says "Get started" with a plain **New note** button, and "Saved 20:11" appears on its own. Jordan trusts autosave.
- **Wobble.** Odd glyphs appear: a single letter stacked down the pane ("E / n / —", then "U / c", then "S"), and `--->`/`<---` arrows with vertical "N a v" and "N o t e s" labels. Jordan ignores them but feels the app is "glitchy".
- **Valley 1: linking.** F1 shows five keys and an empty box. The palette has nothing for "link". Guessed `[[…]]` does nothing, and the first note says "no notes link here yet". Jordan abandons the task.
- **Valley 2 (felt later): the save.** Jordan closed help and kept reading. The note never saved while the status kept promising "changes save automatically". In Phase 2 a quit at that moment lost the title and body.
- **Anxiety: delete.** After pressing Delete nothing visible happened except the footer. Jordan read "tab switch button", tabbed, and pressed Enter on a button he could not see.
- **Relief.** The "✓ deleted · …" receipt with Undo, and "Recently deleted" with "Restore puts a note back where it was", are the best moments of the session.
- **Second peak: import.** The three-option copy answered his exact question ("copy, not linked"). The review showed every file's resulting title, and the receipt said "10 notes created".
- **Valley 3: asking Chatbook at 120 columns.** There was no button, the search swallowed his typing, and the send was "blocked … Review source authority". He got there on a retry without understanding why.
- **End at 160 columns: satisfied.** "Use in" staged the note with a ready prompt and a grounded answer came back. The net feeling is "powerful but fragile; I'd keep a copy of anything important elsewhere".

## Strengths

1. **Delete recovery is complete and plainly worded.** A named receipt ("✓ deleted · Advisor meeting follow-up" with Undo / Dismiss) persists until dismissed, and "Recently deleted (1)" explains itself ("Deleted notes stay here until you restore them. Restore puts a note back where it was; nothing is removed for good from here."). Restore took one click, and the note came back to its folder. This is exactly the recovery story a hesitant first-timer needs (captures 22–24).
2. **Import once is honest about consequences before acting.** The chooser's copy states the one thing a user with an existing folder worries about ("Copy files into Notes… Later changes to the originals are not tracked"). It also offers a third path ("Neither? … Folder files — it opens the folder directly and imports nothing"). The review then shows each file → resulting note title, with per-row Skip / Create new, before anything is written, and the receipt counts what happened (captures 25, 30–32).
3. **Save state and location are text-labelled and timestamped.** "Saved 20:11", "Unsaved changes" and "In the Library database only — no file on disk" sit at the top of the editor. A Console round-trip returns to the same open note. At 160x45, **Use in Console** stages the note and pre-fills a sensible prompt, which is one click to a grounded answer (captures 08, 10, 50).

## Findings

Severity follows the brief: P0 = data loss or a lie about saved state; P1 = would make a user give up or contact support; P2 = annoyance with a workaround; P3 = polish.

### F1 [B] P0: Opening help mid-edit silently kills autosave, and Ctrl+Q then throws the edit away
- **Evidence.**
  - Typed a title and body, pressed `F1`, then `Esc`. 25 s later the status still read "Unsaved changes · Next: Keep editing; changes save automatically." (16). The DB row was `Untitled||1` (title, empty body, version).
  - Fresh repro on `nl-j1-2` (53): `Ctrl+Q`, relaunch, and the landing shows "Notes · Untitled". The title "F1 autosave probe" and the body line are gone (54).
  - Control: typing then `Ctrl+Q` within 0.4 s (no F1) also loses the text (55). Quit never flushes, and the F1 path turns a 2-second window into an unbounded one.
- **Cause (traced).**
  - `LibraryScreen.on_screen_suspend` stops `_notes_state.autosave_timer` (`UI/Screens/library_screen.py:9256-9260`). Textual posts ScreenSuspend when a modal (F1 help, palette) is pushed, not only on navigation.
  - `on_screen_resume` (`:9262+`) never re-arms the timer for a dirty draft.
  - `LibraryScreen` implements neither `confirm_quit` nor `prepare_for_quit`, the hooks `_confirm_and_quit` calls (`app_lifecycle.py:1829+`, `Widgets/confirmation_dialog.py:174-240`). So `_flush_library_note_save` (`library_screen.py:20956`), which navigation does call, never runs on quit.
- **Who it hurts.** Anyone who opens help or the palette while writing, then quits. They lose work while the UI promises it is being saved.
- **Fix.**
  1. In `on_screen_resume`, if the note session is dirty, call `_schedule_library_note_autosave()`.
  2. Better: on suspend, flush instead of stopping (`await _flush_library_note_save()`).
  3. Add `LibraryScreen.prepare_for_quit` that awaits `_flush_library_note_save()`. If the flush is vetoed or fails, have `confirm_quit` ask "Quit and lose unsaved changes to "<title>"?".

### F2 [B] P1: The editor's action row overflows: Save and Use in Console are off-pane at 120 cols, truncated at 160, and the status becomes a vertical letter stack
- **Evidence.**
  - At 120x36 the row shows only "Edit  Preview  Info". `where.py` finds no "Save" or "Use in" anywhere (41).
  - At 160x45 it reads " Save " · " Use in " (ANSI row 13, cols 151-158), and the label is cut from "Use in Console" (49).
  - The save-status widget is squeezed to 1 column, so "Empty note…", "Unsaved changes" and "Saved" paint as "E/n/—", "U/c" and "S" down col 71 (05, 06, 08) and col 77 (49).
- **Cause (traced).**
  - `#library-note-header-second-row` holds the status Static plus mode and task actions in one Horizontal (`Widgets/Library/library_notes_canvas.py:2814-2870`).
  - `#library-note-task-actions { min-width: 61 }` (`css/features/_library.tcss:951-953`, `$ds-library-note-task-actions-min-width: 61`) applies whenever the **terminal** is ≥120 cols (`LIBRARY_NOTES_COMPACT_BREAKPOINT = 120`, `UI/Library_Modules/screen_constants.py:275`).
  - The editor pane is about 50 cols wide at 120 and about 86 at 160.
- **Who it hurts.** Every user at ≤160 cols with the list open. Task 9 failed at 120x36 because of it.
- **Fix.** Decide the compact toolbar from the editor pane width (`_effective_pane_width()` already exists for the location row). Give `#library-note-status` its own full-width row above the actions. Let task actions wrap to a second row, or drop the 61-cell min-width below a pane width of 110.

### F3 [B] P1: At 120x36 the notes tree cannot scroll; keyboard focus walks off-screen and Enter opens an unseen note
- **Evidence.**
  - With md-folder expanded (10 notes) plus Unfiled (2), only 6–7 rows are visible, under about 20 rows of chrome: a 4-line purpose paragraph, filter, two toolbars and "Folders & placement".
  - 10 wheel-down events over the tree produce a 0-line `ansidiff` (58a/58b).
  - After `Tab` + 8×`Down` no visible row is highlighted (57), and `Enter` opened "Ideas", the 9th row, never on screen.
  - "Thesis meeting agenda" was reachable only via the filter (36–40).
- **Cause (hypothesis).** `#library-notes-list` is a plain `Vertical` (`library_notes_canvas.py:2345`) inside a plain `Vertical` canvas (`:909`). The only `overflow-y: auto` for it is the compact rule (`_library.tcss:1004-1012`), so at ≥120 cols there is no scroll owner and rows are clipped.
- **Who it hurts.** Anyone at ≤120x36 with more than about 7 notes. Notes "disappear", and keyboard users open the wrong note.
- **Fix.** Give `#library-notes-list` `overflow-y: auto; height: 1fr` at all widths and call `scroll_visible()` on row focus. At heights under 40 rows, shrink the always-on purpose paragraph to one line ("Stored in your Library · on-disk notes: Folder files").

### F4 [B] P1: A user cannot create a working link between notes
- **Evidence.**
  - No link control, hint, `[[` autocomplete, F1 entry or palette command ("link" → Chunking Lab / Artifacts / Skills) (15, 17).
  - Hand-typed `[[Thesis meeting agenda]]` renders as plain `#e0e0e0` text in Preview (18), and the target's Info says "Linked from (0) — no notes link here yet" (19).
  - Only `[[Title]](note://<uuid>)` counts (User Guide `notes.md:463`). The uuid is shown nowhere.
  - [P2] Once that form is typed using the DB id, Preview underlines it (64). However, Preview's `Markdown(...)` keeps Textual's default `open_links` (`library_notes_canvas.py:2909-2913`), and Textual's handler calls `self.app.open_url(href)`. With no `LinkClicked` handler or `open_url` override anywhere (grep), a click hands `note://…` to the OS URL handler. This was traced statically and not clicked, to avoid invoking the host OS.
- **Who it hurts.** Every note-taker coming from Obsidian, Notion or Logseq. Task 5 failed.
- **Fix.**
  - On save, resolve bare `[[Title]]` and `[[Title|alias]]` by exact or unique title, using the resolver Import once already uses, and store the canonical form.
  - Add `[[` title autocomplete in the body, and a "Copy link to this note" action in Info.
  - In Preview pass `open_links=False` and handle `Markdown.LinkClicked` for `note://` by opening that note in-app.

### F5 [P2] P1: A trailing space in the title makes autosave yank focus mid-sentence, so the user's next words go into the title
- **Evidence.**
  - Typed "Groceries " `Tab` `Tab` "milk", waited 3.5 s, typed " eggs". Focus had jumped to Title (blue border, ANSI row 18), and the note saved as **"Groceries  eggs"** with body "milk" (62, 63, DB row).
  - The status during the veto contradicts itself: "Title begins or ends with whitespace — remove it to save. · Next: Keep editing; changes save automatically."
  - The offending character is invisible.
- **Cause.** The veto at `Library/library_notes_session.py:582-597` is routed with focus on every autosave (`UI/Library_Modules/library_notes_controller.py:3945-3978`).
- **Fix.** Strip leading and trailing whitespace from the title at the save boundary instead of vetoing. Never move focus on an *autosave* veto; show the message inline under the field, and route focus only on explicit Save or leave. Drop the "Next: … save automatically" suffix whenever the status is a veto.

### F6 [P2] P1: The filter matches only whole words in title or body; the keywords users add are unsearchable
- **Evidence.**
  - "meet" gives "filter: meet · 0 results" while "meeting" gives "3 results" (59, 60).
  - Keyword "zebra", saved per the DB `note_keywords` row, gives "filter: zebra · 0 results" (67).
- **Cause.** `notes_fts` indexes `title,content` only (`DB/ChaChaNotes_DB.py:1096-1101`), and the filter builds one quoted phrase with no prefix (`Utils/fts5_match_forms.py:368-391`, `Notes/note_folder_repository.py:808-826`).
- **Who it hurts.** Jordan's task-4 tag is decorative, and type-ahead habits ("thes…") return nothing.
- **Fix.** Prefix-match the last token (`meet*`), AND the tokens instead of a phrase, and OR-in notes whose keywords match. In the empty state, say "No notes match "meet" — the filter matches whole words; keywords aren't searched yet".

### F7 [B] P1: The delete confirmation's buttons are invisible at 120x36
- **Evidence.**
  - After Info ▸ Delete, the only change is the footer ("enter cancel | tab switch button | esc cancel") (20).
  - Wheel-scrolling Info stops at "Delete this note? Undo will be available". The rest of the sentence and **Cancel / Delete** are never shown (21).
  - Jordan confirmed by `Tab` + `Enter` on a button he could not see. At 160x45 it renders correctly ("┃ Cancel ┃ Delete") (52).
- **Cause (hypothesis).** The confirmation mounts inside the Info `VerticalScroll` (`library_notes_canvas.py:2990-3006, 3099-3123`). Its scroll range is not extended or scrolled into view when it is revealed at a 46-col pane.
- **Fix.** On reveal, call `query_one("#library-note-delete-actions").scroll_visible()` and focus Cancel. Or render a one-row pinned confirm ("Delete note? [Cancel] [Delete]") outside the scroll region.

### F8 [P2] P2: "Import…" on the empty landing goes to Media with no pointer to notes
- **Evidence.**
  - The first-run landing's main call is "Import a file" / [Import…].
  - It opens "Import media — Import a file, a whole folder, or a URL. Supported: PDF documents, … plain text files, web pages." Scrolled to the end, it never mentions notes or "Add from files…" (61).
  - A user with a Markdown folder ends up with Media items, not editable notes.
- **Cause.** Map S-14 (`library_screen.py:8929-8940`, `library_entry_canvases.py:349-356`).
- **Fix.** On the Import media canvas add: "Bringing in Markdown notes? Notes ▸ Add from files… keeps them as editable notes" with a button. On the empty landing, offer "Import documents…" and "Import notes…" as two actions.

### F9 [B+P2] P2: Info "Linked from" lies on newly created notes
- **Evidence.**
  - [B] The first Blank note's Info read "Linked from — checking…" indefinitely. It resolved only after reopening the note from the list (12, 14).
  - [P2] After opening "Thesis meeting agenda" ("Linked from (1) · Advisor meeting follow-up", 65), pressing `n` for a new note showed that note's Info as "Linked from (1) · Advisor meeting follow-up" (66).
- **Cause (traced).** The create path (`library_screen.py:30847-30882`) sets `view="editor"` and `selected_note_id` but never resets `backlinks` / `backlinks_status` or runs `_load_library_note_backlinks`. Only `_begin_library_note_open` does that (`library_notes_controller.py:3452-3468`), and the state default is `"loading"` (`library_notes_state.py:524`).
- **Fix.** In the create path set `backlinks=()` and `backlinks_status="ready"` (a new note has no inbound links), or run the same worker.

### F10 [B] P2: F1 help is a bare key list with a malformed row and nothing about how Notes works
- **Evidence.**
  - Editor F1 lists "- : typing in field", "esc: back to list", "F6: next pane", "shift+f6: Previous pane" and "ctrl+n: New note", then about 20 empty rows (15).
  - List F1 lists "/: focus search", F6, "esc: focus rail", shift+f6, "ctrl+n: New note" and "g: Go to folder" (35), while the footer says "n new note" and "/ find note".
  - There is nothing about autosave, Save, Edit/Preview/Info, where Delete lives, link syntax or Use in Console.
- **Cause (traced).** The panel is footer chips plus active bindings (`library_screen.py:25730-25768`). The key-less status chip `("", "typing in field")` (`:4827`) leaks in as a row.
- **Fix.** Drop key-less chips from the panel, and use the footer's spelling ("n", "find note"). Add a 4–6 line "How Notes works" block per surface: "Autosaves 2 s after you stop typing · Save saves now · Preview renders Markdown · Delete is in Info ▸ Danger · Link with [[Title]] · Use in Console asks about this note".

### F11 [B] P2: Console's Library search fails a first-timer: lost typing, a premature "staged" chip, a jargon block, and Un-stage being undone
- **Evidence.**
  - Typed text did not reach the query field; the modal opens without focus (42).
  - "Staged for next send · 1 source — Library Search/RAG retrieval" appeared while the search was still loading an embeddings model. The log shows `_build: Loaded model default in 15.69s` at 20:22:55 (43).
  - Send at 20:22:43 gave "Console send blocked: Library search has no available evidence. Review source authority before sending." (44).
  - The 2 results then staged themselves *after* Un-stage was pressed (45).
- **Who it hurts.** Anyone asking about notes at a width where "Use in Console" is hidden (F2).
- **Confidence.** Medium; this is Console scope, observed once.
- **Fix.** Focus the query input on open. Label the chip "Searching… (first run loads a model)" and keep Send disabled until it settles. Ignore results that arrive after Un-stage. Replace the block copy with "No matching notes for "<q>" yet — wait for the search to finish or Un-stage".

### F12 [B] P2: "View 10 imported notes" lands on the list with the import folder collapsed
- **Evidence.** The receipt button leads to the list showing "▸ md-folder" collapsed beside Unfiled. None of the 10 notes are visible until the folder is clicked (33, 34).
- **Fix.** When arriving from an import receipt, expand and select the destination folder and scroll it into view, or open the list filtered to the imported notes.

### F13 [B] P3: Unexplained chrome glyphs and a system folder in a brand-new profile
- **Evidence.**
  - ASCII `--->` / `<---` grips with vertical "N a v" / "N o t e s" labels: three arrows in one 120x36 list view (13). Their tooltips need a mouse hover.
  - Nav expands and collapses between views at 120 cols (24 vs 25).
  - "▸ Agent_Lessons" appears in Jordan's tree after his first Console visit with no gloss (10). The map says the gloss shows only when there are zero notes.
  - The task checkbox prints as "• [ ] Email draft by Friday" (09).
- **Fix.**
  - Replace the grip glyphs with labelled toggles ("‹ Hide list" / "Show list ›") and keep one per pane.
  - Always show the Agent_Lessons gloss ("— where Console agents file lessons") or hide the folder while it is empty.
  - Render task items as ☐/☑ in Preview.

## Improvement opportunities

1. **Empty-list starter card.** On "Notes (0)" or the first note, show a dismissible 5-line card: autosave behaviour, Preview, "[[Title]] links another note", keywords, and "Use in Console asks Chatbook about this note". This covers tasks 2, 5, 8 and 9 at the moment of need.
2. **Name Notes where people look for it.** Add "Notes" to More ▾ and a palette command "Open Notes" (map S-19), both landing on the Notes list. Jordan's first two actions were a search for that word.
3. **A "Link to note…" command.** Add it in the editor (button plus palette) to pick a note by title and insert the canonical link. Pair it with an outgoing "Links (N)" list in Info, so following a link never depends on Preview's link handling.
4. **One key to ask about a note.** Make "Use in Console" a list-row and editor accelerator (e.g. `u`), shown in the footer at every width. This removes the dependence on the toolbar fitting (F2).
5. **Height-aware list chrome.** At under 40 rows, collapse the always-on purpose paragraph and the "Folders & placement" group behind a "More actions" toggle. That gives the tree 10+ rows at 120x36 instead of 6.

## Nielsen scores (0 = fails, 4 = no problems)

### Library shell (landing, rail, nav, import entry)

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of system status | 3 | "Library \| Local" and the rail counts are clear. Panes auto-collapse and expand between views at 120 cols without saying so. |
| 2 | Match with real world | 2 | "Import…" means *media* import; "Nav" grips, "placement" and "Library notes \| Folder files" need prior knowledge. |
| 3 | User control & freedom | 3 | Returning to Library resumes the last note; Escape ladders work. The landing is hard to get back to. |
| 4 | Consistency & standards | 2 | Import… (Media) vs Add from files… (Notes); "New note" opens a chooser while `n` creates directly. |
| 5 | Error prevention | 3 | Nothing destructive at shell level; the risks are in Notes. |
| 6 | Recognition over recall | 2 | "Notes" is absent from the nav bar, More ▾ and the palette. You must know it lives in Library. |
| 7 | Flexibility & efficiency | 3 | Palette, Ctrl+3, F6 and `i` exist. |
| 8 | Aesthetic & minimalist | 2 | `--->`/`<---` grips and vertical labels; rail header truncated "Navigati…" at 120 cols. |
| 9 | Error recognition & recovery | 3 | No shell errors met. |
| 10 | Help & documentation | 1 | F1 is a key list ("Library Shortcuts — Landing") with no orientation. |

### Notes (list, editor, Preview, Info, New note, Trash, Import once)

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of system status | 1 | "Changes save automatically" stays on screen while autosave is dead (F1). Status squeezed to one letter (F2). "Linked from" stuck or stale (F9). |
| 2 | Match with real world | 2 | `[[Title]]`, the convention every note app uses, is inert (F4). Keywords look like tags but are not searchable (F6). |
| 3 | User control & freedom | 3 | Undo receipt plus Recently deleted is excellent. Focus theft on a veto takes control away (F5). |
| 4 | Consistency & standards | 2 | Typed links vs imported links behave differently. Help spells keys differently from the footer. `n` vs New. |
| 5 | Error prevention | 1 | Quit discards unsaved edits with no prompt (F1). Typing is silently redirected into the title (F5). Delete confirmed blind at 120 cols (F7). |
| 6 | Recognition over recall | 2 | No link affordance. Save and Use in Console are hidden at 120 cols. Delete is only in Info ▸ Danger. |
| 7 | Flexibility & efficiency | 2 | Whole-word-only filter, no keyword facet, no Ctrl+S, no key for Use in Console. |
| 8 | Aesthetic & minimalist | 2 | At 120x36 list chrome takes about 20 rows and leaves 6 for notes (F3). Body shows 4 lines. |
| 9 | Error recognition & recovery | 2 | The veto copy contradicts itself ("remove it to save · … changes save automatically"). Import review and receipts are good. |
| 10 | Help & documentation | 1 | F1 has 5–6 keys and a malformed row; nothing on links, autosave or Delete (F10). |

## Harness caveats

- The harness doc (HARNESS.md, required reading) already describes Library ▸ Notes anchors and recipes. Phase 1 was therefore "blind" only in decisions: navigation choices were made from on-screen copy, not from the doc.
- Null keyring and the per-run isolation profile are harness artifacts, not findings.
- PNG renders are approximate (no box borders). All claims rest on `.txt`/`.ansi`, and colour and focus claims use `rowstyles.py`/`ansidiff.py`.
- One wasted click (step 10's first "Edit" click) hit the word "Edit" inside the status copy "Press Edit to change this note". That was a `where.py` text-match artifact, not a product defect.
- The Preview `note://` click (F4) was deliberately **not** clicked live, because it would invoke the host macOS URL handler. The behaviour is traced statically.
- The Console search delay (F11) includes a first-run HuggingFace embeddings model load (15.7 s, unauthenticated HF Hub request in the log). A machine with a warm model cache may not reproduce the timing.
- Ages and times ("Saved 20:11") are local PDT. DB timestamps are UTC.
- The mock LLM (shared, :18777) answered both Console sends. The reply text is mock output, but the request carried the note evidence ("Evidence: [S1] NOTES — …").
