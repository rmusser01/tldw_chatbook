# Journey J3: power user "Alex" in Library ▸ Notes

- **Build:** worktree `notes-library-ux-review` at origin/dev `2d34cbf80d`, Textual 8.2.8, golden profile (122 notes).
- **Runs:** `nl-j3-1` (the main journey, then a relaunch with REUSE=1), `nl-j3-2` (repro of the sync failure and of the missing folders), `nl-j3-3` (both-sides conflict, delete-while-synced, 160x45 and 100x30 checks). All three were killed at the end. `pgrep -fl runs/nl-j3` returns nothing. The isolation snapshot is unchanged, and none of the three run logs references the real profile.
- **Captures:** `evidence/j3-poweruser-notes/NN-<what>-<cols>x<rows>.{txt,ansi}`, with PNGs for six key moments (41, 58, 62, 65, 66, 68). The `.txt` and `.ansi` files are authoritative.
- **Sizes:** 200x50 is the primary size. I also used 160x45 and 100x30 (a crosscheck at 160 and the compact layout at 100x30).

## Persona and goals

Alex is an expert PKM user. They have used Obsidian, vim and zk for years and keep 2,000+ notes in a vault backed by git. They work keyboard-only, count keystrokes and have no patience for hand-holding. They know the product: they read `map-notes.md`, `map-shell-nav.md` and the `library/notes.md` User Guide first. Alex judges each task against what an expert tool needs:

| Reference task | Obsidian / vim+zk |
|---|---|
| Capture a note from anywhere | 1 key (Ctrl+N, or a global capture) |
| Open a note by partial title | 2 keys + query (Ctrl+O, Enter) |
| Back to the previous note | 1 key (Alt+Left / Ctrl+O) |
| Link while typing | `[[` + 2–3 chars + Enter (autocomplete) |
| Bulk move or tag | select a range, then 1 command |

Alex wants:
1. capture without losing context;
2. find by prefix, tag, body or folder;
3. drive the tree with the keyboard;
4. links and backlinks that work;
5. bulk organisation;
6. templates;
7. two-way vault sync they can trust;
8. edit files in place with git;
9. send notes to Console;
10. resume after a restart.

## Step log

Keystroke counts include only keys. Mouse clicks, used where keyboard focus could not be found quickly, are marked "(click)".

| # | Intent | Keys / clicks | Expected | Observed | Friction | Capture |
|---|---|---|---|---|---|---|
| 1 | Capture from Console | `C-p` `note` `Enter` | Blank note | Opens the **New note chooser** ("Ready · Next: Press Blank note, or choose a template."). | minor | 02, 03 |
| 2 | Create a blank note | `Enter` | Editor with the title focused | Editor opens on "Untitled", status "Empty note — type to keep it". The duplicate status cell is crushed to "Empty / note — / type". | minor | 04 |
| 3 | Type title, keywords, body | text `Tab` text `Tab` text | Autosave | "Saved 20:11 · Next: Keep editing…" | none | 05 |
| 4 | Look at the tree after the capture | `Esc` | Folders + Unfiled | **The 7 folders are missing**: two blank rows, then "▾ Unfiled". `g` lands on Unfiled. They only appear after a rail **Notes** press. Reproduced on a fresh run. | major | 06, 07, 53, 54 |
| 5 | Capture from Media | `n`, then `C-n` (with list and reader focused) | New note | Both do nothing, with no feedback. | major | 08 |
| 6 | Capture from Media via the palette | `C-p` `note` `Enter` `Enter` + text | New note | Works, but it leaves Media. | minor | 09 |
| 7 | Back to the Media item I was reading | `Esc` `Esc` (to the rail search) `Tab`×3 `Enter` | Same item | The rail press resets Media: "Select a media item to read it here." | major | 10 |
| 8 | Switch Library destination by keyboard | `Esc` + `Tab`×5 + `Enter` = 7 | ≤2 keys | `Esc` goes to the rail **search box**, not the rows. Ctrl+3 is not sendable here (see caveats). | major | — |
| 9 | Filter by partial word | `/` `Lisb` `Enter` | Trip planning: Lisbon | "filter: Lisb · 0 results" | blocker | 11 |
| 10 | Filter by keyword | `homelab` `Enter` (a keyword on *Home network setup*) | 1 result | "filter: homelab · 0 results". `keyword:homelab` also gives 0. | blocker | 12 |
| 11 | Filter by body phrase | `chunk size` | Matches | 4 results, grouped by folder ✓. The reordered `size chunk` gives 0. | minor | 13 |
| 12 | Filter by folder | `Japanese` | Study/Japanese | 16 results (folder path + content) ✓ | none | — |
| 13 | Leave the filter with `Esc`, then `Esc` again | `Esc` `Esc` | Back to the results | The first `Esc` jumps to the rail search. The second goes to the rail **Notes** row. Typing then lands on the row, and `Enter` re-selects Notes and **clears the filter**. | major | 14 |
| 14 | Filter → open the first result | `/` `start here` `Enter` `Tab`×8 `Enter` | ≤2 keys after the query | **8 Tabs** (New, Select, Add from files…, Export, New folder, Clear filter, ▾ Unfiled, row). | major | — |
| 15 | Keyboard tree: down a folder | `g` `Down` `Right` | Next folder / expand | `g` → Agent_Lessons. **`Down` and `Right` do nothing on folder rows.** `Tab` moves between folders and `Enter` toggles. | major | — |
| 16 | Open Journal ▸ Daily ▸ 3rd note | `g` `Tab` `Enter` `Tab` `Enter` `Tab` `Down` `Down` `Enter` = 9 | ~4 | Works. Focus lands in the body. | minor | 15 |
| 17 | Next or previous note | `Esc` `Down` `Enter` = 3 | 1 | `Esc` returns focus to the **exact row**, which is good. There is no `]`/`[` for notes (bound for Media only). | minor | — |
| 18 | Back to a non-adjacent previous note | — | Alt+Left | **No history.** You re-filter (≥11 keys). | major | — |
| 19 | Scroll the tree with Journal ▸ Daily open | wheel; `Tab`×26; `PageDown`; `End` | The list scrolls | **The list never scrolls at ≥120 cols.** Meetings…Unfiled, the pager and Recently deleted are unreachable, and Tab focus moves off-screen. The same happens at 160x45. At 100x30 it scrolls (scrollbar visible). | blocker | 25, 26, 27, 56, 67, 68 |
| 20 | Preview links | `S-Tab`×6 `Tab` `Enter` | Links rendered and followable | Links render as text. No key follows them, and the click handler goes to the OS (code-traced, not clicked). | major | 17 |
| 21 | Backlinks of Index | Info | List | "Linked from (0) — no notes link here yet" (correct) | none | 18 |
| 22 | Create a link while typing | in a new note: `See [[Trip planning: Lisbon]] for details.` | Autocomplete, then a real link | No completion. Saved as plain text. | blocker | 19 |
| 23 | Does Lisbon see the backlink? | Lisbon ▸ Info | Linked from (2) | "Linked from (1) · Index — start here". **The typed link is ignored.** | blocker | 20 |
| 24 | Follow a backlink | click "Index — start here" in Linked from | Opens | Opens ✓. `Esc` returns to that row. | none | — |
| 25 | Move an Unfiled note to a folder | row → `S-Tab`×3 → **Move note** `Enter` | Dialog naming the note | The dialog is titled just "Move note". The picker lists only loaded folders (no Projects/Thesis or Study/Japanese). Choose → **"That folder changed elsewhere — refresh and try aga…"** | major | 21, 22, 23 |
| 26 | Same note, Add to folder | `S-Tab`×4 `Enter` `Enter` `Down`×7 `Enter` `Tab` `Tab` `Enter` ≈ 17 | Placed | "Folder organization updated." The DB row is in Research, but the filtered row still says "· Unfiled". | minor | 24 |
| 27 | Bulk-select 10 daily logs | `/` `Daily log` `Enter` `Tab` `Tab` `Enter` (Select), then `Enter` `Down` ×10 | Range select | 10 selected after 20 keys. `Space` does nothing, and there is no Shift-range. | major | 28 |
| 28 | Bulk move / tag / delete | — | Actions | The strip offers only **Done · Select all · Clear · Export selected**. | blocker | 28 |
| 29 | Bulk export | `e` → Choose destination (click) → Save → Export bundle (click) | Files | A `.zip` with `content/notes/*.md` + manifest ✓. "Bundle: 10 items · size known once it runs" stays after it has run. | minor | 29, 30 |
| 30 | Select all after returning from Export | "Select all 100 shown" (click) | 100 | **26 selected.** The label and the action disagree. | major | 31 |
| 31 | Note from a template | `C-p` `note` `Enter` `Down` `Enter` `Down`×4 `Enter` = 13 | Pre-filled | "Daily Journal - 2026-10-02" with sections ✓. The note lands in Unfiled. There is no user or vault template. | minor | 32, 33 |
| 32 | Add from files… (keyboard) | `/` `Tab`×4 `Enter` | Chooser | ✓ Three options with consequences | none | 34 |
| 33 | Keep a folder synced | `Tab` `Enter` | Focus on Display name | Focus lands on the collapsed **Notes grip** while the footer says "enter run action". | minor | 35 |
| 34 | Choose the vault, Check | Choose folder… → path → Select folder → Check changes | Review | "That folder is inside Chatbook's own data directory." This is a harness artifact (see caveats). I copied the vault outside the run. | n/a | 36, 37 |
| 35 | Check + Activate | Check changes → Activate reviewed root | Review | "64 safe · 0 need attention · 4 skipped". Then "Sync root activated. 64 applied". The vault is **flattened** into one "PowerVault ⇄ Sync managed" folder. | minor | 38, 39 |
| 36 | App edit → disk | Method: `C-End` `Enter` + "Edited in Chatbook…" | File updated | md5 changed in ≤3 s, and `git diff` shows exactly +2 lines. Frontmatter is intact ✓. | none | — |
| 37 | Disk append → app | `printf >> Vocabulary.md` | Note updated | **Never arrives**, even after 25 s, a reopen and a restart. | blocker | 40 |
| 38 | Manage sync folders | click | Status | "⚠ Needs attention · Next: Review changes". The row is titled "Sync folder (name unavailable before cutover)". There is no receipt for my app write. | major | 41 |
| 39 | Review / Check again | click | Act on it | "A recovery is still open for that folder. Resolve it, then Check again." with "0 safe · 0 need attention". Check again does the same. | blocker | 42 |
| 40 | Recovery | click | Resolve | "Recovery failed — RuntimeError. Next: Check changes", which loops back to step 39. **It survives a relaunch.** | blocker | 43, 52 |
| 41 | Repro on a clean root (run 2) | edit ending **with** `\n` | — | `update_file completed` ✓ | — | 55 |
| 42 | Repro: edit **without** a trailing `\n` | `C-End` + text | — | `update_file needs_attention postcondition_failed`. The root is stuck again. The note list still shows "Vault3 ⇄ Sync managed" and "Library notes · Ready". | blocker | 57, 58 |
| 43 | Both-sides conflict (run 3) | Pause → edit in app → append on disk → Resume (click) → Review | Conflict review | "Both file and note changed (1)". View comparison shows a unified diff. **Keep both** → "1 applied · no conflicts remain" + Undo, and the copy goes to a "Conflict copies" folder ✓. The metadata line reads "File modified 1790998978887732948 ns". | minor | 61, 62, 63 |
| 44 | Delete a synced note in the app | Info ▸ Delete ▸ Delete | Status changes | `second.md` stays on disk, and the root reads **"✓ Up to date · Next: Check changes"**. | blocker | 64, 65 |
| 45 | Check changes after the delete | click | Choice | "One side was deleted (1) · Resolution unavailable for this item. No changes can be staged." All three actions are "○" (disabled). | blocker | 66 |
| 46 | Folder files on a vault copy | strip ▸ Folder files ▸ Choose folder… | Tree | Linked, with the vault tree. "Selected notes root changed; the commit draft was cleared." appears on a first link. | minor | 44, 45 |
| 47 | Edit a file in place | click body, `C-End` `Enter` text | Autosave | "Saved". `git diff --stat` shows +2 ✓. The frontmatter is hidden but preserved. | none | — |
| 48 | Git status and diff | Manage ▸ Review session changes ▸ Trust and check status | Diff | "EDITED · READY TO STAGE · Git: unstaged". **No diff view anywhere** in the panel. | major | 46, 47 |
| 49 | Send one note to Console | from body `S-Tab`×3 `Enter` = 4 | Staged | "Staged for next send · 1 source · Thesis outline…" and the prompt is prefilled ✓ | none | 48 |
| 50 | Send a second note | Use in Console on Lisbon | 2 sources | "Staged for next send · **1 source** · Trip planning: Lisbon". The first note is silently dropped. | major | 49 |
| 51 | Type, then Ctrl+Q at once | `C-End` " UNSAVED-TAIL…" `C-q` | Flush or confirm | The app exits immediately. The DB note stays at version 1 and the **typed text is lost**. | blocker | — (DB query) |
| 52 | Relaunch (REUSE=1) | — | Resume context | Lands on Console with "Sources: 0" (the staged note is gone). The Library landing has no Continue, and "From your Library: Notes · Method". The last note, filter and expanded folders are not restored. | major | 50, 51 |
| 53 | 100x30 key moments | resize | Usable | The list scrolls with a scrollbar. The editor beside it works. The title is cut to "Ideas in…". | minor | 67, 69 |

## Task outcomes

The "expert" column gives Obsidian, vim or zk equivalents.

| Task | Outcome | Steps | Keys (Alex) | Expert | Note |
|---|---|---|---|---|---|
| 1 Quick capture (Console, Media) | partial | 1–8 | Console 7 + 2 Tabs; Media 7 (n/Ctrl+N inert), then 6+ to get back, and the reading position is lost | 1 | A capture through the palette also hides the folder tree. |
| 2 Find (prefix / tag / body / folder), jump, return | partial | 9–18 | `/`+q+`Enter`+8 Tab+`Enter` = 11+q; adjacent jump 3; back: no history | 2+q; 1 | Prefix and keyword searches fail. Body and folder searches work. |
| 3 Keyboard tree | partial | 15–19 | 9 to the 3rd daily log | ~4 | Arrows do nothing on folders. The tree does not scroll at ≥120 cols. |
| 4 Links + backlinks | fail | 20–24 | backlinks 6 | `[[`+3+Enter | Typed `[[links]]` are inert. Backlinks work for imported links only. |
| 5 Organise + bulk | partial | 25–30 | Add to folder ≈17; select 10 = 20; bulk move/tag/delete impossible (≈120 keys one by one) | ~5 | Only Export exists for bulk. Move note on an Unfiled note fails. |
| 6 Templates | success (limited) | 31 | 13 | ~4 | 8 fixed templates. No user or vault templates. |
| 7 Vault sync | **fail** | 32–45 | ~10 clicks to set up | — | One ordinary edit wedges the root for good. After a delete the root lies with "✓ Up to date". |
| 8 Folder files + git | partial | 46–48 | — | — | In-place editing works well. There is no diff. |
| 9 Notes → Console | partial | 49–50 | 4 per note | — | A second note replaces the first. There is no multi-send. |
| 10 Relaunch | partial / fail | 51–52 | — | — | Ctrl+Q loses unsaved text. Context is not restored. |

## Emotional journey

- **Start: sceptical but willing.** The palette capture works, but it takes 7 keys and a detour through a chooser. "Fine, I'll learn it."
- **Valley 1: search.** "Lisb" gives 0 results and the keyword "homelab" gives 0. Alex gives up on the filter as a quick switcher.
- **Valley 2: tree.** `Down` does nothing on a folder. The list stops at "Daily log 2026-08-14" and nothing scrolls it. Folders vanish after a quick capture. "Is this thing even rendering?"
- **Low point: links.** A typed `[[Trip planning: Lisbon]]` is dead text, and Lisbon's "Linked from (1)" ignores it. For a Zettelkasten user this is disqualifying.
- **Small lift:** Esc puts focus back on the exact row; backlinks open; the template works; Add from files explains its three options clearly.
- **Peak:** the sync review ("64 safe · 0 need attention · 4 skipped") and a clean 2-line `git diff` 3 seconds after editing in the app.
- **Crash:** 30 seconds later the root is wedged. Review says to "Resolve" a recovery. Recovery says "RuntimeError. Next: Check changes". The loop survives a restart, and the note list still says "⇄ Sync managed · Ready". Ctrl+Q then eats a sentence.
- **Partial redemption:** the both-sides conflict flow shows a real diff, has a clear Keep both, and leaves an Undo receipt.
- **End: distrust.** "I would not point this at my real vault." The deleted-note "✓ Up to date" confirms it.

## Strengths

1. **The both-sides conflict review is honest and recoverable** (captures 61–63). After Pause, edits on both sides and Resume, the root says "⚠ Needs attention". Review lists "conflict.md · Both file and note changed". **View comparison** shows a scrollable unified diff (`--- Note / +++ File`, `-App side edit.` / `+Disk side edit.`). There are four explicit choices, and staging a choice ("Choice staged. No changes yet.") is separate from applying it. **Keep both** ends with "1 applied · no conflicts remain", an **Undo**, and the losing text kept in a "Conflict copies" folder. Nothing is settled silently by last-write-wins, which is exactly what a vault owner needs.
2. **Escape keeps your place in the list** (step 17). Escape from the editor returns focus to the very row you opened ("Daily log 2026-09-03"). That makes Down + Enter a 3-key walk through adjacent notes, and the same holds for notes opened from a backlink. Together with the "✓ deleted · …" receipt that parks focus on **Undo**, keyboard recovery is cheap where it exists.
3. **On a healthy root, file round-trips are fast and byte-careful** (steps 36, 41, 47). An app edit reached disk in ≤3 s as a clean +2-line `git diff` with the YAML frontmatter untouched. A disk append reached the note in about 6 s. Folder files hides the frontmatter ("5 lines of YAML frontmatter above this body are hidden here and kept exactly as they are on disk."), and Session Git scopes itself to "notes changed during this Chatbook session".

## Findings

Ranked most severe first.

### P0-1: One ordinary edit to a synced note permanently wedges lasting sync, and the Notes list keeps saying "⇄ Sync managed · Ready"
- **Surface / kind:** Notes-sync · defect
- **Evidence:** Captures 40, 41, 42, 43, 52 (run 1) and 55, 57, 58 (clean repro in run 2).
  - The sync-state DB records `update_file | needs_attention | postcondition_failed` for an edit whose body does not end in `\n`, and `update_file | completed` for one that does.
  - The note content ends `…here.` (`20686572652E`). The file ends `…here.\n`.
  - Root row: "⚠ Needs attention · Next: Review changes". Review: "A recovery is still open for that folder. Resolve it, then Check again." with "0 safe · 0 need attention". Recovery: "Recovery failed — RuntimeError. Next: Check changes".
  - Log: `reason=sync_recovery_unresolved` / `reason=unclassified error_type=RuntimeError`.
  - After that, a disk append to `Vocabulary.md` never reaches the note, even after a relaunch.
  - Meanwhile the tree reads "▸ Vault3 ⇄ Sync managed", the list reads "Library notes · Ready", and the editor reads "Saved".
- **Cause (traced):**
  - `Notes/notes_sync_filesystem.py:259-260`: `serialize()` appends `\n` when the captured profile has `final_newline`.
  - `Notes/notes_sync_executor.py:5384`: the "desired" classification requires `file.text == request.note.content`, which fails when the note lacks the newline. `:5398-5400` then raises `postcondition_failed`.
  - The UI falls back to the exception class name and to Check changes as the next action (`Library/library_notes_lasting_sync_state.py:1199-1204`), which creates the loop.
- **Repro:**
  1. Keep a folder synced on any folder whose files end with a newline.
  2. Open a synced note, press Ctrl+End and type a word without pressing Enter.
  3. Wait 5 s.
  4. Open Manage sync folders.
- **Why it matters:** This is the normal way to edit. After one edit both directions stop, the only offered actions loop, and every surface outside Manage sync folders says all is well. A vault user would keep writing on both sides and diverge silently.
- **Fix:**
  - Compare like with like in the postcondition: `serialize(note.content, profile) == file.raw_bytes`, or normalise trailing newlines before comparing.
  - Let Recovery accept the on-disk state when the bytes equal the serialised note.
  - Replace the `type(error).__name__` fallback with a plain sentence and a next action that does not loop.
  - Propagate root attention to the tree folder row ("⇄ Sync needs attention") and to the editor's location row.

### P0-2: Ctrl+Q quits with no flush and no confirmation while the editor shows "Unsaved changes", and the typed text is lost
- **Surface / kind:** Notes · defect
- **Evidence:**
  - Step 51 (run 1, Trip planning: Lisbon). The header read "Unsaved changes · …" with " UNSAVED-TAIL-1790998421" visible in the body.
  - `C-q` → the process exited within 3 s.
  - The DB afterwards: `version=1`, `last_modified=2026-09-17…`, and the content ends `…/40cf9035-…)`, without the tail.
- **Cause (traced):**
  - The quit flow consults only `confirm_quit` / `prepare_for_quit` on the active screen (`app_lifecycle.py:1829-1870`; `Widgets/confirmation_dialog.py:174-240`).
  - `LibraryScreen` defines neither. Only `chunking_lab`, `profile_interview`, `chat`, `personas` and `settings` do (grep of `UI/Screens`).
- **Repro:**
  1. Open any note.
  2. Type a word.
  3. Press Ctrl+Q within 2 s, before autosave fires.
- **Why it matters:** Quick quit is part of a keyboard user's muscle memory, and the 2 s autosave debounce guarantees a window for losing data.
- **Fix:**
  - Add `LibraryScreen.prepare_for_quit` that awaits the notes session flush (and the prompt and skill editor saves).
  - Add `confirm_quit` that returns False with a "Save failed — stay?" dialog when the flush is vetoed or fails.

### P0-3: Deleting a synced note leaves "✓ Up to date", and the deletion it hides cannot be resolved in-app
- **Surface / kind:** Notes-sync · defect
- **Evidence:**
  - Captures 64, 65, 66 (run 3). After Info ▸ Delete on `second`, `ls` still shows `second.md`.
  - Manage sync folders reads "✓ Up to date · Next: Check changes".
  - Check changes → "One side was deleted (1) · second.md · Resolution unavailable for this item. No changes can be staged.", with "○ Restore missing side", "○ Delete/archive counterpart" and "○ Disconnect item" all disabled.
  - The User Guide (`notes.md` "What this covers, exactly") documents that deleting, restoring and creating notes do not notify sync while the row "goes on reading ✓ Up to date".
- **Repro:**
  1. Activate a root.
  2. Delete one synced note in the app.
  3. Open Manage sync folders.
  4. Press Check changes.
- **Why it matters:** A false "up to date" is the one status a sync user must never see. The follow-up dead end leaves Undo or Recently deleted as the only exit, and nothing points there.
- **Fix:**
  - Emit a sync intent from note delete, restore and create in managed folders, so the row flips to "◌ Changes available".
  - Enable "Restore missing side" (recreate the note from the file) and "Delete counterpart (move the file to the vault's .trash)" for this item. If they stay disabled, give the reason and point to "Recently deleted → Restore".

### P0-4: At ≥120 columns the Notes list does not scroll, so folders, pagers and results below the fold are unreachable
- **Surface / kind:** Notes · defect
- **Evidence:**
  - Captures 25, 26, 27, 56, 68 (wide) against 67 (100x30 shows a ▇ scrollbar and scrolls).
  - With Journal ▸ Daily open at 200x50 the pane ends at "Notes 1–20 of 56 Load more notes". Meetings, Projects, Research, Study and Unfiled sit below the border. Wheel, PageDown and End do nothing, and 26 Tabs move focus off-screen.
  - Filter "Method" shows "20 results", but the title match under **Vault3** is never visible.
  - Reproduced at 160x45 in a fresh run, where the list clips after 3 Unfiled notes.
- **Cause (traced):**
  - Compact mode gives `#library-notes-list` `height: 1fr; overflow-y: auto` (`css/features/_library.tcss:1004-1013`).
  - The wide rule sets only widths (`css/features/_library_panels.tcss:543-548`).
  - The host is a plain `Vertical` (`UI/Screens/library_screen.py:15100-15104`).
- **Repro:**
  1. Open Notes at 200x50.
  2. Expand Journal and then Daily.
  3. Try to reach Unfiled.
- **Why it matters:** With 2,000 notes nearly every list overflows. "Load more notes", "Recently deleted" and whole folders become unreachable unless you collapse other folders. A long single branch can never reach its own pager.
- **Fix:**
  - Give the wide `#library-notes-list` (or the canvas) `height: 1fr; overflow-y: auto`, or make it a `VerticalScroll`.
  - Call `scroll_visible()` on the focused row in `_move_library_list_row_focus` and on Tab focus.

### P1-5: Typed `[[wikilinks]]` are dead text: no autocomplete, no backlink, nothing to follow
- **Surface / kind:** Notes · missing-capability
- **Evidence:** Captures 19, 20, 17.
  - Typing `See [[Trip planning: Lisbon]] for details.` saves as plain text with no completion popup.
  - Lisbon ▸ Info shows "Linked from (1) · Index — start here" and excludes the new note.
  - Edit mode shows imported links raw as `[[Thesis outline — retrieval-augmented tutoring]](note://758cf130-…)`, wrapped across lines.
  - Preview renders links, but no key follows them. A click calls `app.open_url("note://…")`: `Widgets/Library/library_notes_canvas.py:2909-2913` builds `Markdown(...)` with no `open_links=False` and no `LinkClicked` handler, and Textual's driver uses `webbrowser.open`. **I did not click it**, so this part is code-traced only.
- **Why it matters:** Linking is the core PKM verb. Alex cannot create a link without knowing a UUID that the UI never shows.
- **Fix:**
  - On save, resolve `[[Title]]` and `[[Title|alias]]` against note titles with the importer's resolver, and record the link edges.
  - Add a `[[` completion popup in the body TextArea.
  - Handle `Markdown.LinkClicked` for `note://` by opening the note in-app (and bind Enter on a focused link).
  - Render stored links as `[[Title]]` in Edit and keep the id hidden.

### P1-6: The notes filter cannot match prefixes, keywords or reordered words
- **Surface / kind:** Notes · usability
- **Evidence:** Captures 11, 12, 13.
  - "filter: Lisb · 0 results".
  - "filter: homelab · 0 results" (homelab is a keyword on Home network setup).
  - `keyword:homelab` gives 0, `size chunk` gives 0, and `chunk size` gives 4.
  - The map traces this to a phrase-only FTS query on title and content (`DB/ChaChaNotes_DB.py:1096-1101`, `Utils/fts5_match_forms.py:368-391`).
- **Why it matters:** Alex searches by prefix and tag dozens of times a day. With exact phrase matching only, the filter is useless as a quick switcher.
- **Fix:**
  - Build an AND-of-tokens query with a `*` prefix on the last token.
  - Index keywords, or support `#tag` / `tag:` syntax that resolves through `note_keywords`.
  - Keep phrase matching for quoted input.

### P1-7: There is no fast path from query to note, and no note history
- **Surface / kind:** Notes · usability
- **Evidence:** Step 14:
  - From the filter, the first result is 8 Tabs away: New, Select, Add from files…, Export, New folder, Clear filter, ▾ Unfiled, then the row.
  - `Esc` from the filter jumps to the rail search instead.
  - There is no back/forward. `]`/`[` are bound only to Media (`library_screen.py` BINDINGS).
- **Why it matters:** Finding a note costs about 11 keys plus the query, against 2 in an expert tool. Getting back after following a backlink means searching again.
- **Fix:**
  - Make `Down` or `Enter` in the filter focus the first result row.
  - Add a "Go to note…" fuzzy-title palette command (and a single Notes key, for example `o`).
  - Add Alt+Left/Alt+Right note history, and reuse `]`/`[` for the next and previous note in the list.

### P1-8: Bulk work is export-only, Select-all lies about its count, and selecting takes 2 keys per row
- **Surface / kind:** Notes · missing-capability
- **Evidence:** Captures 28, 31.
  - The select strip reads "10 selected · Done · Select all 20 shown · Clear · Export selected".
  - Selecting 10 took `Enter` + `Down` ×10. `Space` does nothing.
  - After returning from Export the strip read "Select all 100 shown", and pressing it gave "26 selected".
- **Cause (traced):** The label falls back to `list_state.rows` when the canvas has no `tree_projection` (`library_notes_canvas.py:1786-1795`), while the handler builds a fresh projection (`library_screen.py:30125-30138`).
- **Why it matters:** Re-filing or re-tagging 10 daily logs means about 120 keys, one note at a time. Deleting an Import-once vault before syncing it is, per the guide, "one note at a time".
- **Fix:**
  - Add **Move to folder…**, **Add/remove keyword…** and **Delete (undoable)** to the select strip.
  - Make `Space` toggle a row and Shift+Up/Down extend the selection.
  - Compute the "N shown" label from the same projection the handler uses.

### P1-9: A second "Use in Console" silently replaces the first staged note
- **Surface / kind:** Cross-destination · defect
- **Evidence:** Captures 48, 49.
  - The first send shows "Staged for next send · 1 source · Thesis outline — retrieval-augmented tutoring — note".
  - After Use in Console on Lisbon it shows "Staged for next send · **1 source** · Trip planning: Lisbon — note", and the inspector reads "Sources: 1 staged".
  - Select mode has no Use in Console.
  - Handoffs go through the single `HandoffChannel.CHAT` slot (`app_destinations.py:175-197`).
- **Why it matters:** Alex builds context from several notes. The first is dropped without notice and the send goes out under-grounded.
- **Fix:** Append to the staged sources and dedupe by note id, or ask "Replace or add?". Add "Use in Console (N)" to the select strip.

### P1-10: Arriving through the palette "New Note" renders the notes tree without its folders
- **Surface / kind:** Notes · defect
- **Evidence:** Captures 04, 06, 53, 54 (reproduced in two fresh runs).
  - After `C-p` "New Note" → Blank note → `Esc`, the tree shows two blank rows, then "▾ Unfiled", with no Agent_Lessons, Journal, Meetings, Projects, Recipes, Research or Study.
  - `g go to folder` lands on Unfiled.
  - The folders appear after a rail **Notes** press (capture 07) or after another create.
- **Cause:** Untraced. My hypothesis is that the `notes_create` entry composes the tree before the root-folder branch is requested.
- **Why it matters:** Right after a capture is when Alex files the note, and the folders appear to be gone.
- **Fix:** Request the root folder branch on the `notes_create` and `note_id` entries, the same way the rail-row entry does, and show "Loading folders…" instead of blank rows.

### P2-11: "Move note" on an Unfiled note fails with a false, truncated cause, and the dialog hides which note and which folders
- **Surface / kind:** Notes · defect
- **Evidence:** Captures 21, 22, 23.
  - The dialog is titled only "Move note".
  - The picker lists Agent_Lessons, Journal, Journal / Daily, Meetings, Projects, Recipes, Research and Study, but not Projects / Thesis, Projects / App launch or Study / Japanese.
  - Choose → "That folder changed elsewhere — refresh and try aga…". Nothing had changed. The same flow with Add to folder worked.
  - The error string comes from the `FolderConflictError` mapping at `library_screen.py:18797-18801`.
- **Fix:**
  - Disable Move note for Unfiled rows with the reason "Unfiled has no folder to move from — use Add to folder".
  - Title the dialog "Move “<title>” to…".
  - Load every folder in the picker and make it type-to-filter.
  - Wrap the error text instead of ellipsising it.

### P2-12: Arrow keys skip folder rows and there is no expand or collapse key
- **Surface / kind:** Notes · usability
- **Evidence:** Step 15: `g` → "▸ Agent_Lessons", then `Down` and `Right` produced no change (ansidiff empty). `Tab` walks the folders and `Enter` both selects and toggles. The cause is `_LIBRARY_LIST_ROW_CLASSES`, which excludes `library-notes-folder-row` (`UI/Library_Modules/screen_constants.py:434-450`).
- **Fix:** Include folder and pager rows in Up/Down traversal. Map Right/`l` to expand or step into a folder, and Left/`h` to collapse or go to the parent. Separate selection from toggling.

### P2-13: Quick capture from Media is impossible by key, and coming back loses your place
- **Surface / kind:** Library-shell · usability
- **Evidence:** Captures 08, 09, 10.
  - `n` and `C-n` in the Media list and reader do nothing silently.
  - The palette path leaves Media.
  - Coming back via the rail shows "Select a media item to read it here.".
  - The More menu has only Edit metadata, Open original, Open manager and Move to trash.
- **Fix:**
  - Let `n` / Ctrl+N work from any Library canvas.
  - Add "Note on this item" to Media's More menu, creating a note pre-linked to the media and placed in an Inbox folder.
  - Return to the originating reader on Esc.

### P2-14: Sync and comparison copy leaks implementation detail
- **Surface / kind:** Notes-sync · copy
- **Evidence:** Captures 62, 41, 43.
  - "File modified 1790998978887732948 ns · 8 lines/82 chars" and "Note v2, updated 2026-10-03T03:43:24.542000+00:00" (`Widgets/Library/library_notes_add_from_files_canvas.py:855`).
  - "Recovery failed — RuntimeError".
  - Every root is titled "Sync folder (name unavailable before cutover)", even after naming one "PowerVault" (`library_notes_sync_controller.py:819`).
  - "Configure a local lasting sync root." and "Resolution history unavailable — it starts after this root is activated" still show after activation.
- **Fix:** Format both times as local "YYYY-MM-DD HH:MM". Title roots with their display name and path. Replace exception names with the reason table's plain sentence. Drop the stale lines once the root is active.

### P2-15: Folder files' Session Git shows status but never a diff
- **Surface / kind:** Notes-sync/import/folder-files · missing-capability
- **Evidence:** Capture 47 (from 46). The panel shows "EDITED Projects/Thesis/Method.md · READY TO STAGE · Git: unstaged · Status: CURRENT · READY — 1 can be staged", with "Stage" and "Show bulk · 1 stage". There is no diff control; a grep of `library_file_notes_git_panel.py` finds no diff.
- **Fix:** Add a "View diff" action for the selected row that reuses the conflict-compare unified-diff box, and show the staged diff in Review commit.

### P2-16: A relaunch drops the working context
- **Surface / kind:** Library-shell · usability
- **Evidence:** Captures 50, 51.
  - After REUSE=1 the app lands on Console with "Sources: 0", so the staged note is gone.
  - The Library landing has no Continue, and "From your Library" shows only "Notes · Method".
  - The open note, the filter and the expanded folders are not restored. `ScreenStateStore` is memory-only, per the map.
- **Fix:** Persist the last Notes route, note id, filter and expanded branches beside `library.reader`. Offer "Continue: <note>" on the landing and restore it on the first Notes visit.

### P2-17: After a root is activated, "Add from files…" keeps reopening the old receipt
- **Surface / kind:** Notes-sync · defect
- **Evidence:** Captures 59, 60 (run 2).
  - "Add from files…" shows "Sync root activated. 64 applied · listed under Receipts" again, with only "○ Resolution history" and "‹ Notes".
  - Escape and re-entry show the same receipt, so a second root cannot be set up in the same session.
- **Fix:** Reset the `lasting_add` phase to `choose` when the receipt is left, or add "Set up another folder" to the receipt bar.

### P3-18: At 200x50 with Nav open, the editor's mode row clips and its duplicate status cell collapses
- **Surface / kind:** Notes · defect
- **Evidence:** Captures 16, 04. The row reads "Edit · Preview · Info · Save · Use in C" and is cut at the pane edge. The left status cell shows "S" or "Empty / note — / type", while the header above repeats the full status.
- **Fix:** Remove the duplicate status cell from the mode row, and let the row wrap like the list toolbar.

### P3-19: After visiting Notes, Media's Items grip is painted "Notes"
- **Surface / kind:** Library-shell · consistency
- **Evidence:** Capture 08: the grip column reads N/o/t/e/s on the Media list. It repaints to "Items" only after the next refresh (opening More).
- **Cause (traced):**
  - `library_browse_reader_shell.py:194-198` assigns `pane_label` without a refresh.
  - `LibraryAdaptiveReaderPaneGrip.sync_open` repaints only when the arrow changes (`library_adaptive_reader_shell.py:142-170`).
- **Fix:** Call `self.items_grip.refresh()` after changing `pane_label`.

## Improvement opportunities

These go beyond fixing the defects above.

1. **A quick switcher and capture overlay for notes.** Ctrl+N from any destination opens a 3-line capture overlay that files to an Inbox folder and returns you where you were. A "Go to note…" fuzzy title switcher, with recent notes first, gets its own key and a palette entry "Library — Notes". This brings capture and find to 1–2 keys plus the query, the bar Alex brings from Obsidian.
2. **Link-native editing.** Add `[[` autocomplete, link edges recorded on save, an always-visible "Links / Linked from / Unresolved" panel on wide layouts, and Enter-to-follow inside Preview. This turns Notes from a database of documents into a graph Alex can navigate without the mouse.
3. **Keyword facets.** Add a "# Keywords" pseudo-branch in the tree (counts per keyword) and a `#tag` filter syntax. The 76 note-keyword links in this profile are currently invisible to navigation.
4. **Sync that keeps vault structure, with health on the row.** Sync managed subfolders instead of one flat "PowerVault" folder. Show the root's health on its tree row and in the editor's location line ("⇄ in sync · 3 s ago" / "⚠ sync paused — Review").
5. **Templates from the vault.** Offer the vault's `Templates/` folder (currently skipped) as user templates, and let each template carry a default destination folder, so the Daily journal lands in Journal ▸ Daily.
6. **Multi-note Console context.** Add "Use selected in Console" from select mode and from a filter result set, with sources appended rather than replaced.

## Nielsen heuristic scores

Scale: 0 = fails the heuristic, 4 = excellent.

| Surface | Heuristic | Score | Key issue |
|---|---|---|---|
| Notes | H1 Visibility of system status | 1 | The tree shows "⇄ Sync managed" and the list "Ready" while sync is wedged; "✓ Up to date" after a delete. Conflict review status is good. |
| Notes | H2 Match between system and real world | 2 | "placement", "lasting sync root", "cutover", "File modified 1790998978887732948 ns", "RuntimeError". |
| Notes | H3 User control and freedom | 1 | Ctrl+Q discards unsaved text; no note history; the sync recovery loop. The delete Undo is good. |
| Notes | H4 Consistency and standards | 2 | Down works on note rows but not folders; `]`/`[` for Media only; "New" opens a chooser but `n` does not; Use in Console replaces. |
| Notes | H5 Error prevention | 1 | Move note enabled for Unfiled notes; an edit without a trailing newline breaks sync; quit does not flush. |
| Notes | H6 Recognition rather than recall | 2 | Footer chips name keys, but Tab counts must be memorised; no link autocomplete; no keyword browser. |
| Notes | H7 Flexibility and efficiency of use | 1 | No prefix search or quick switcher; 8 Tabs from filter to result; no bulk move, tag or delete; 2 keys per selected row. |
| Notes | H8 Aesthetic and minimalist design | 2 | Dense and readable, but a duplicate status cell crushed to "S", always-visible prose, and "Use in C" clipped. |
| Notes | H9 Help users recognize, diagnose, recover from errors | 1 | "That folder changed elsewhere — refresh and try aga…" (false and clipped); the "Recovery failed — RuntimeError" loop; the unresolvable deletion item. |
| Notes | H10 Help and documentation | 2 | F1 lists only 3 keys. The 2,353-line guide even documents the false "✓ Up to date". |
| Library | H1 Visibility of system status | 2 | Live counts, but the Items grip reads "Notes" on Media and the footer is stale in some focus states. |
| Library | H2 Match between system and real world | 3 | Mostly plain language; "rail"/"hub" in the footer is minor. |
| Library | H3 User control and freedom | 2 | Esc goes to the rail search, not the previous place; a rail press resets the Media reader. |
| Library | H4 Consistency and standards | 2 | `n`/Ctrl+N are dead in Media; `]`/`[` for Media only; two collapse mechanisms. |
| Library | H5 Error prevention | 3 | Few destructive shell actions. |
| Library | H6 Recognition rather than recall | 3 | The rail with counts and per-surface footers. |
| Library | H7 Flexibility and efficiency of use | 1 | Switching destination by keyboard takes about 7 keys (Esc, Tab×5, Enter); no "Library — Notes" palette command. |
| Library | H8 Aesthetic and minimalist design | 3 | Coherent, dense, calm. |
| Library | H9 Help users recognize, diagnose, recover from errors | 2 | Reasonable callouts, rarely exercised here. |
| Library | H10 Help and documentation | 2 | F1 is per-surface but thin. |

## Harness caveats

- The vault at `$RUN/home/fixtures/vault` was refused with "That folder is inside Chatbook's own data directory." The harness puts `config.toml` at the run root (`TLDW_CONFIG_PATH=$RUN/config.toml`), so the effective config directory contains HOME. The refusal comes from `find_root_binding_conflict` (`Utils/sensitive_paths.py`). I copied the vault to `scratchpad/j3/vault*` (outside the run), so this is not a product finding. Untested: the copy says "data directory" even when the conflicting path is the config directory.
- Ctrl+digit cannot be sent through tmux (xterm sends ESC for Ctrl+3). I used the palette, the nav bar and rail clicks instead, so the keystroke counts for switching destination assume no working Ctrl+3. Many real terminals have the same limit (an untested hypothesis).
- I did not click Preview `note://` links, so as not to launch the real macOS URL handler. That part of P1-5 is code-traced.
- The null keyring backend applies. The Git "Trust this repository" modal is real app behaviour.
- PNGs are approximations; the `.txt` and `.ansi` captures are authoritative.
- In one run, the palette "note" + `Enter` landed on Home instead of New Note (a result-ordering race). I did not investigate it and did not report it.
- Some steps used mouse clicks to save time (marked "(click)"). Keystroke counts exclude them.
