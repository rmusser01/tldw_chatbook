# J6: Stress test, "Riley" (Library + Library › Notes)

- **Build:** worktree `notes-library-ux-review` @ origin/dev `2d34cbf80d`, Textual 8.2.8.
- **Harness:** nlrev live harness, socket `nl-j6-1` (one run, relaunched 4× with `REUSE=1`), GOLDEN profile, mock LLM on :18777.
- **Sizes:** 160x45 (primary), 235x52, 120x36, 100x30, 60x24.
- **Evidence:** `../evidence/j6-stress/NN-<what>-<cols>x<rows>.{txt,ansi[,png]}`. A capture is cited below as `NN`.
- **Disk/DB truth:** checked with read-only `sqlite3` on the run's `tldw_chatbook_ChaChaNotes.db`, `tldw_chatbook_notes_sync_state.db` and `tldw_chatbook_media_v2.db`, and with `ls`/`tail` on the synced folders.
- **Isolation:** `snap.sh` diff "untouched"; log references to the real profile: 0; `pgrep -fl runs/nl-j6` empty at the end.

## Persona and goals

**Riley** is a methodical stress tester. Riley does not trust status lines. After every action Riley checks what actually landed on disk, in the DB and in the log, and pushes past the happy path: quitting mid-keystroke, malformed input, destructive operations, filesystem edits behind the app's back, resizing with a modal open, and key-mashing.

Riley's goals:

1. Never lose typed text.
2. Never be told "Saved" or "synced" when it is not true.
3. Always have a way out of an error state.

## Step log

| # | Intent | Keys / clicks | Expected | Observed | Friction | Capture |
|---|---|---|---|---|---|---|
| 1 | Open Library › Notes | nav `⌃3 Library`, rail `Notes (122)` | List | List, folders collapsed, Unfiled expanded | none | 01 |
| 2 | Body edit, then Esc within 0.3 s | click note, `C-End`, type, `Escape` | Flush, then close | Closed; DB v1→v2 contains the text | none | 02 |
| 3 | Body edit, then nav-bar away within 0.3 s | type, click `⌃2 Console` | Flush | DB v3 contains the text | none | 03 |
| 4 | Body edit, then **Ctrl+Q** within 0.3 s | type `RILEY-CTRLQ-BODY`, `C-q` | Flush or confirm | Quits at once. Status before quit was "Unsaved changes"; DB still v3; text absent after relaunch | **blocker** | 04-before-ctrlq-body, 05 |
| 5 | Title edit + Esc; keywords edit + nav away | type, `Escape` / click `⌃1 Home` | Flush | Both saved (v4, v5) | none | 06, 07 |
| 6 | New note + Esc | `C-n`, title, Tab Tab, body, `Escape` | Saved | Saved, v2 | none | 08 |
| 7 | New note + **Ctrl+Q** | `C-n`, title, body, `C-q` | Flush or confirm | Title and body lost. An empty **"Untitled"** row persists and shows in the list after relaunch | **blocker** | 09, 10 |
| 8 | Trailing-space title, then keep typing in body | title `Groceries␠`, Tab Tab, `milk`, wait 3 s, `eggs` | Inline error; typing stays in Body | Autosave veto moved focus to Title; `eggs` landed in the title → "Groceries eggs" | major | 11, 12 |
| 9 | Esc with trailing-space title | `Escape` | Explain the block | Toast "Can't leave yet — fix the title or press Discard new note." There is no Discard button on screen | minor | 13 |
| 10 | Nav bar while title is vetoed | click `⌃2 Console` | Refusal with a reason | Silent refusal (log only). Nav bar now highlights **Console** while Library shows | major | 14, 15 |
| 11 | Duplicate keyword, then type in body | Keywords `ok, x, X`, Tab, ` body1`, wait 3 s, ` body2` | Inline error | Pane flipped from Edit to **Info**; ` body2` went into keywords → "ok, x, X body2" | major | 16, 17 |
| 12 | Empty title | clear Title, wait | Validation or "Untitled" | Saved as literal "Untitled"; editor heading blank | minor | 18 |
| 13 | 300-char title | type 300 chars | Saved, truncated display | Saved (len 300); heading cut at the pane edge without an ellipsis | none | 19 |
| 14 | CJK / emoji / RTL in title and body | type mixed scripts | Round-trip | Round-trips in DB and list (ZWJ loss came from tmux send-keys, not the app; see step 15) | none | 20 |
| 15 | ~20k paste | bracketed `paste-buffer -p` | Fast save | Rendered in 0.33 s; saved 19,980 chars; ZWJ preserved | none | 21 |
| 16 | ~20k typed (`send-keys -l`) | 14 chunks | Keeps up | 71.7 s for 19.3k keystrokes (≈3.6 ms/key); saved | none | 22 |
| 17 | Latency on the 6,575-word note | 20 keystrokes, 0.25 s apart; 8 more landing during autosaves | <100 ms | median 42 ms, p90 45, max 72 (includes ~10–15 ms capture overhead) | none | — |
| 18 | Use in Console, then delete the note | `Use in`, back, Info › Delete › Tab › Enter | Console source invalidated | Console still lists "Trip planning: Lisbon (note) · Ready" (Sources: 1) and sent the turn | major | 24–30 |
| 19 | After delete, press Move note | click `Move note` | Disabled | Enabled. Picking Recipes gave "That folder changed elsewhere — refresh and try aga…" | minor | 26–28 |
| 20 | Undo after navigating away | Console → Library, `Undo` | Restored | Restored; the stale "That folder changed elsewhere" stays in the authority line | minor | 31, 32 |
| 21 | Arrow past the visible rows | `Down` ×12 from the first Unfiled row | List scrolls | Focus marker leaves the screen and nothing scrolls; `Enter` opened an invisible note ("Daily log 2026-09-21"); the wheel does nothing | **major** | 33–35 |
| 22 | Remove folder with notes, restore | ▸ Recipes, Remove, Remove folder, Restore folder | Notes kept | Notes went to Unfiled; restore worked (DB `deleted` 1→0) | none | 36–39 |
| 23 | Recently deleted | filter, `Recently deleted (2)`, Tab, `r` | Restore | Restore works; `r` without a focused row silently does nothing; no permanent delete exists | minor | 40–42 |
| 24 | Move folder into a collapsed folder | Recipes › Move › Research | Target shows all its children | Research expanded showing only Recipes; its 5 notes appear only after collapse + re-expand | minor | 43–46 |
| 25 | Keep vault synced | Add from files › Keep a folder synced › choose › Check › Activate | 64 created | 64 safe / 4 skipped matches disk; flattened into one managed folder | none | 47–56 |
| 26 | Edit both sides of a synced note | app: add `riley-app` (pushed to disk ✓); then disk: append `riley-disk`; app: type ` app2` | Conflict surfaced | Editor says **"Saved"** + "In a synced folder"; DB has `app2`, disk has `riley-disk`; root shows "⚠ Needs attention" | **blocker** | 57, 58, 63 |
| 27 | Resolve it | Review / Check again / Recovery / Check changes | A path out | Loop: "A recovery is still open… Resolve it" ↔ "Recovery failed — RuntimeError. Next: Check changes" ↔ "Check failed — recovery still open. Next: Resolve recovery". Persists after restart | **blocker** | 59–62, 93 |
| 28 | Add from files after the failed Recovery | `Add from files…` | Chooser | Blank page: heading + "Recovery failed — RuntimeError", no choices, no back button; reproduced twice; only a restart clears it | major | 92, 94 |
| 29 | Second root, disk-side edge cases | rm `ideas.md`, rename `quotes.md`, add emoji file, latin-1 file, CRLF+LF file, empty file, bad YAML | Per-file rows | One latin-1 file refused the **whole root**: "Check failed — NotesSyncRootRefused"; then the same for mixed newlines | major | 96, 97 |
| 30 | After removing the two bad files | Check changes | Resolve delete/rename | "One side was deleted" and "Preview explicit filesystem move" rows: "Resolution unavailable for this item"; all choices disabled; Apply reviewed disabled with no reason shown | **major** | 98, 99 |
| 31 | Media import: nonexistent / unsupported | path + Enter | Clear errors | "Path not found…" / "Unsupported file type: .xyz." | none | 65, 66 |
| 32 | Enter on an ingest path | footer says "enter check this path" | Check | Imports immediately (1 file; earlier a whole 10-file folder) | minor | 67, 67b |
| 33 | Cancel a 60-file import midway | look for Cancel; Tab; Esc | Cancel | No cancel control; Esc leaves while it continues to 60/60 | major | 68, 69 |
| 34 | Export to an unwritable dir | Info › Export Markdown › `ro/…` | Error | "Export failed — check the destination…" ×3 + toast "Error exporting note: PermissionError" | minor | 71 |
| 35 | Export onto an existing file | Export to `precious.md` | Confirm replace | Silently overwritten: "Note exported successfully to precious.md" | **major** | 72 |
| 36 | Zero-result filter; rapid filter typing | `zzqqxx` ↵; then `Lisbon`↵ + immediate `xyzi` | Results / filter keeps focus | Zero state good. During the re-render, `i` escaped the input and opened **Import media** | major | 73, 75 |
| 37 | Page beyond the end; open a trashed item | Next ×6; Media Trash › click item | Bounded; preview | Next disables at 5/5 ✓; the trashed item cannot be opened and the reader keeps showing an unrelated active item with Use in Console | minor | 76–79 |
| 38 | Resize 235→60→160 with editor + Move-note modal | type, open modal, resize, Cancel | Layout recovers; autosave fires | Layout recovers; **autosave never fired**: "Unsaved changes · …changes save automatically" for 15+ s and DB unchanged. Isolated: the modal causes it; a resize alone does not | major | 82–86 |
| 39 | Key-mash on lists | 80 random Up/Down/Enter/Esc at 20 ms; Enter/Esc ×10 | No crash | No crash, no double-open, no stuck loading (ended in Search/RAG via rail focus; expected) | none | 88, 89 |
| 40 | List at other widths | 120x36, 100x30 | Scrollable | 120x36: "Unfiled" itself is below the clip, wheel does nothing. 100x30 (compact): scrolls, thumb shown | major | 90, 91 |

## Task outcomes

| Task | Outcome | Steps | Keystrokes | Note |
|---|---|---|---|---|
| 1. Unsaved work survives Esc / nav / Ctrl+Q | fail | 7 variants | ~120 | Esc and nav flush every field ✓. Ctrl+Q loses existing and new-note edits and leaves an orphan "Untitled" |
| 2. Validation and big/odd input | partial | 10 | ~40k (typing tests) | Vetoes steal focus and redirect typing into the wrong field. Perf excellent |
| 3. Destructive: delete, undo, trash, folders | success | 18 | ~45 | Works, with a stale selection after delete, a hidden-children render after a move, and no permanent delete |
| 4. Sync / import edge cases vs disk truth | fail | 30 | ~90 | Both-sides edit wedges the root permanently; delete/rename unresolvable; one bad file blocks a root |
| 5. Library edge cases | partial | 20 | ~60 | Good error copy, but silent overwrite, no import cancel, Enter imports, shortcut race |
| 6. Resize with editor + modal | partial | 6 | ~20 | Layout fine at all sizes; opening the modal kills the pending autosave |
| 7. Key-mashing | success | 2 | 90 | No crash, double-open or stuck state |

Delete-a-note keystroke cost (keyboard + mouse): Info (1) → Delete (1) → Tab (1) → Enter (1) = 4 actions. Undo = 1.

## Emotional journey

- **Start: confident.** Escape and nav-bar flushes saved everything inside 0.3 s. The design is clearly careful here.
- **First valley: betrayal.** Ctrl+Q quits on the spot. The status line said "Unsaved changes · changes save automatically" and the work is gone after relaunch. The new-note variant also leaves an empty "Untitled" behind.
- **Irritation.** The validation veto yanks the caret: Riley types "eggs" in the body and finds it in the title. With a duplicate keyword the whole pane flips to Info.
- **Relief.** The delete receipt, Undo, Restore folder and the Media "Delete permanently" confirm are well built.
- **Deep valley: trapped.** A both-sides edit leaves the editor saying "Saved" while disk and DB disagree. Every recovery button points to another button that fails ("RuntimeError"), the state survives a restart, and Add from files goes blank.
- **Frustration.** Most of the list is physically unreachable at ≥120 columns.
- **End: distrust.** Riley now checks the DB after every save and would not point a real vault at Keep-synced.

## Strengths

1. **The guarded leave-flush is solid.** Escape, nav-bar switches and `C-n` all `await flush_pending_work()` (`app_navigation.py:513`). Body, title and keywords typed ≤0.3 s before leaving all reached the DB (02, 03, 06, 07, 08), including on a brand-new note. A focused user who leaves normally never loses text.
2. **Large content is fast and faithful.**
   - A 20k-character bracketed paste rendered and saved in 0.33 s with ZWJ emoji intact.
   - The 6,575-word note answers a keystroke in a median 42 ms (max 72 ms), even while autosaves land.
   - 300-character titles and CJK/emoji/RTL titles and bodies round-trip exactly.
3. **Destructive actions are recoverable and say so.**
   - The inline "Delete this note? Undo will be available in the Notes list." defaults to Cancel.
   - "✓ deleted · <title> [Undo]" survives navigation and Undo restores the row (32).
   - Remove folder says "Notes are not deleted…" and offers Restore folder (36–39).
   - Media's permanent delete needs "Delete permanently" with Cancel focused (80).
   - Ingest errors name the cause plainly ("Path not found: …", "Unsupported file type: .xyz.").

## Findings

Ranked most severe first. "Hypothesis" marks a cause I could not trace end to end.

### P0-1: Ctrl+Q discards unsaved note edits, and a new note becomes an empty orphan

- **Where:** Notes editor (Edit tab, wide reader), any field.
- **Repro:**
  1. Open "Gardening log" and press `C-End`, `Enter`.
  2. Type `RILEY-CTRLQ-BODY`. The status reads "Unsaved changes · Next: Keep editing; changes save automatically." (04-before-ctrlq-body).
  3. Press `C-q` within 2 s. The app quits with no prompt.
  4. Relaunch. The DB is still v3 and the text is gone (05).
  5. Same for a new note: `C-n`, type "Riley new ctrlq", Tab Tab, "lost?", `C-q`. The DB keeps `'Untitled'` with an empty body, and "Untitled · now" is in the list after relaunch (09, 10).
- **Who it hurts:** Anyone who quits within the 2 s autosave debounce, which is the normal "type the last line, quit" pattern. Their text is lost, and the status line had promised it would save.
- **Cause:**
  - `LibraryScreen` implements neither `confirm_quit` nor `prepare_for_quit`. The quit flow only consults those hooks (`Widgets/confirmation_dialog.py:174-238`; `app_lifecycle.py:1853-1923`).
  - `flush_pending_work` (`UI/Screens/library_screen.py:11109`) is awaited only by navigation (`app_navigation.py:513`).
  - The untouched-blank GC lives in `_flush_library_note_save` (`library_screen.py:20956`), so it never runs on quit.
- **Fix:**
  - Add `LibraryScreen.prepare_for_quit()` that awaits `flush_pending_work()`.
  - Add `confirm_quit()` that, on a non-PERMITTED flush (validation, conflict or failed save), shows "Quit and discard unsaved changes to '<title>'?" with Stay focused.
  - Run the blank-note GC in the same hook.

### P0-2: A both-sides edit on a synced note reports "Saved" while file and DB diverge, then wedges the whole sync root in an unrecoverable loop

- **Where:** Notes editor status line + Library notes › Manage sync folders.
- **Repro:**
  1. Keep `vault` synced (64 notes).
  2. Open "Budget 2026" and append `| riley-app | 1 |`. This is pushed to disk ✓.
  3. On disk, append `| riley-disk | 2 |`.
  4. In the app, type ` app2`.
- **Observed:**
  - The editor shows "Saved 20:33" and "In a synced folder · …Budget 202…" (57, 63). The DB has `app2` and not `riley-disk`; the file has `riley-disk` and not `app2`.
  - `notes_sync_operations` holds `update_file | needs_attention | postcondition_failed`.
  - Manage sync folders reads "⚠ Needs attention · Next: Review changes". The buttons then loop:
    - **Review:** "A recovery is still open for that folder. Resolve it, then Check again." with "0 safe · 0 need attention…" (59).
    - **Recovery:** "Recovery failed — RuntimeError. Next: Check changes." (61).
    - **Check changes:** "Check failed — recovery still open. Next: Resolve recovery." (62).
  - The loop survives a restart (93). The log says only `reason=unclassified error_type=RuntimeError`.
- **Who it hurts:** Any vault user who edits a file in another editor while it is open in Chatbook. Their edits silently stop syncing for all 64 notes, the UI claims "Saved", and there is no exit except removing the root, which is not offered ("Disconnect unavailable — not in this release").
- **Cause:**
  - `notes_sync_runtime.py:2468-2484` refuses every check while any operation is incomplete.
  - `resolve_cleanup` (`:3415-3446`) raises for this operation kind (exact raise untraced).
  - The controller renders the exception class (`library_notes_sync_controller.py:1517-1530` → `check_failure_row`, `Library/library_notes_lasting_sync_state.py:1189-1204`).
  - The editor's status/location line never consults the binding's sync state.
- **Fix:**
  1. Turn a `postcondition_failed` `update_file` into an ordinary reviewable conflict row: Keep file / Keep note / Keep both, using the existing conflict UI.
  2. Make the editor location line say "Not synced — the file changed on disk · Review" whenever the binding has an incomplete operation.
  3. Never render a class name. Map `unclassified` to "Chatbook couldn't finish recovery for <file>. Pause this folder and keep both copies?" with a working action.
  4. Ship Disconnect as the escape hatch.

### P1-3: The Notes list cannot scroll at ≥120 columns, so most notes are unreachable

- **Where:** Library notes list, wide adaptive reader (160x45, 120x36, 235x52).
- **Repro:**
  - 160x45: expand Projects › Thesis. Unfiled shows 2 of 40+ rows. The mouse wheel over the list changes nothing (capture diff empty).
  - Focus "Trip planning: Lisbon" and press `Down` ×12. The focus marker leaves the screen and nothing scrolls (34). `Enter` opens "Daily log 2026-09-21", a row the user never saw (35).
  - 120x36: even the "▸ Study" and "▾ Unfiled" rows are below the clip (90).
  - 100x30 (compact) scrolls correctly with a thumb (91).
- **Who it hurts:** Every user with more than a screenful of notes, especially mouse users. They cannot reach Unfiled notes, paging rows or "Recently deleted (N)" except through Filter.
- **Cause (hypothesis):**
  - In the wide path the canvas sits in a plain `Vertical` (`#library-canvas`, `library_screen.py:15097-15104`), and `#library-notes-list` is a `Vertical` (`Widgets/Library/library_notes_canvas.py:2345`).
  - `overflow-y: auto` is applied only under `.library-notes-compact` (`css/screen_agentic_library.tcss:1242-1249`).
- **Fix:**
  - Make the tree region the scroll owner in the wide reader: `#library-notes-list { height: 1fr; overflow-y: auto; }` outside the compact rule.
  - Call `scroll_visible()` when `_move_library_list_row_focus` moves focus (`UI/Library_Modules/canvas_sync.py:61-107`).

### P1-4: An autosave validation veto steals focus mid-typing and puts the user's text into the wrong field

- **Where:** Notes editor, Title / Keywords / Body.
- **Repro A:**
  1. Set the title to `Groceries␠`.
  2. Press Tab Tab and type `milk` in Body.
  3. Wait 3 s. Status: "Title begins or ends with whitespace — remove it to save."
  4. Type `eggs`. It lands in the title, which becomes "Groceries eggs" (11, 12).
- **Repro B:**
  1. Type `ok, x, X` in Keywords and press Tab.
  2. Type ` body1` in Body and wait 3 s.
  3. The pane switches from Edit to **Info**.
  4. Type ` body2`. Keywords become "ok, x, X body2" (16, 17).
- **Who it hurts:** Fast typists and screen-reader/keyboard users. Text silently changes fields, and the title is corrupted without them noticing.
- **Cause:** `_apply_library_note_save_outcome` always routes and focuses the validation field, even for `explicit=False` autosaves (`UI/Library_Modules/library_notes_controller.py:3945-3978`). For keywords it also forces Info (`:3865-3874`). That forcing is stale: keywords moved into Edit in task-32642.
- **Fix:**
  - On an autosave veto, keep focus where it is and mark the offending field inline (red border + "Remove the trailing space" under Title).
  - Move focus only on an explicit Save or a leave attempt.
  - Better still, auto-trim whitespace and auto-dedupe keywords, with a one-line notice.

### P1-5: Opening a folder dialog cancels the pending autosave, while the status keeps promising "changes save automatically"

- **Where:** Notes editor + list toolbar "Move note" / "Add to folder" dialog.
- **Repro:**
  1. Type ` A2` in a note body.
  2. Within 1 s, click `Move note`, then `Cancel`.
  3. Wait 6–15 s. Status: "Unsaved changes · Next: Keep editing; changes save automatically." The DB is unchanged (85, 86).
  4. Control: the same edit with a 160→120→160 resize and no modal saved normally.
- **Who it hurts:** Users who organise a note right after editing it. Combined with P0-1, a quit loses the edit, and the status line lies meanwhile.
- **Cause (hypothesis):** Something on that path calls `_invalidate_library_note_autosave()` (`library_notes_controller.py:3697`), which bumps the generation without a flush, and nothing re-arms the debounce on dismiss. I did not trace the exact caller.
- **Fix:**
  - Flush (`_flush_library_note_save`) before pushing any Notes modal, or re-arm `_schedule_library_note_autosave()` on dismiss.
  - Make the status text derive from "a save is scheduled" rather than from dirty alone.

### P1-6: Single-letter shortcuts fire while the user is still typing a filter, opening Import

- **Where:** Library notes filter → global `i` (Import media).
- **Repro:**
  1. Click the filter, type `Lisbon`, press Enter.
  2. Immediately type `xyzi`. "Import media" opens with focus in its path field, and `xyz` is dropped (75).
  3. Earlier in the session, a stray `i` from typing on an unfocused surface led to an Enter in the ingest path and imported a 10-file folder into Media with no confirmation (64).
- **Who it hurts:** Fast keyboard users who refine a search. They are thrown into a different destination, and one more Enter imports files.
- **Cause (hypothesis):**
  - The filter apply recomposes the canvas, so focus is briefly `None`.
  - `on_key` treats that as "no text field focused" and routes `i` to Import (`library_screen.py:8929-8940`).
- **Fix:**
  - Keep the filter `Input` mounted across apply (update the tree only), or restore focus to it synchronously.
  - Ignore single-letter shortcuts while `screen.focused is None` and for ~300 ms after an Input submit.

### P1-7: Single-note export silently overwrites an existing file

- **Where:** Info › Reuse & Export › Export Markdown (FileSave "Export Note as Markdown").
- **Repro:** Create `exp/precious.md` with "PRECIOUS USER FILE". Export any note to that exact path. The toast says "Note exported successfully to precious.md" and the file is replaced (72).
- **Who it hurts:** Anyone who picks an existing file name. A file outside Chatbook is destroyed with no undo.
- **Cause:** `FileSave(location=str(Path.home()) …)` at `library_notes_controller.py:4318-4323` uses `textual_fspicker`'s default `can_overwrite=True` (`Third_Party/textual_fspicker/file_save.py:37`). The write is a plain `write_text` (`library_screen.py:21160`).
- **Fix:** Pass `can_overwrite=False`, or show a "Replace precious.md?" confirm with Cancel focused. Also remember the last export directory.

### P1-8: One non-UTF-8 or mixed-newline file blocks the entire sync folder, with opaque copy

- **Where:** Manage sync folders › Check changes.
- **Repro:**
  1. In a synced folder, add `latin1.md` (byte `0xE9`).
  2. Check changes. Root row: "⚠ Needs attention · Check failed — NotesSyncRootRefused · Next: Check changes" (97). The log has `reason=unsupported_encoding`.
  3. Remove that file and add a CRLF+LF file. Same UI, with `reason=mixed_newlines`.
  4. Deletions and renames in the folder are not processed until both files are gone.
- **Who it hurts:** Anyone whose folder contains a single legacy-encoded or Windows-edited file. They cannot tell which file is wrong, and the suggested next action repeats the failure.
- **Cause:**
  - Per-file parse errors raise a root-level refusal (`Notes/notes_sync_filesystem.py:136-150`).
  - `_CHECK_REFUSAL_COPY` (`Library/library_notes_lasting_sync_state.py:1007+`) has no entries for these reasons, so the class name is shown (`:1185`, `:1201-1204`).
- **Fix:**
  - Treat unreadable files as review rows: "Skipped — not UTF-8 (latin1.md)" / "Skipped — mixed line endings (…)". Let the rest of the root sync.
  - Add copy for every reason code.

### P1-9: Files deleted or renamed on disk can never be resolved, which also blocks applying safe changes

- **Where:** Keep-synced review.
- **Repro:**
  1. In a synced folder, `rm ideas.md` and `mv quotes.md "quotes renamed.md"`.
  2. Check changes shows "One side was deleted · ideas.md" and "Preview explicit filesystem move · quotes renamed.md". Each reads "Resolution unavailable for this item. No changes can be staged." with `○ Restore missing side` / `○ Delete/archive counterpart` / `○ Disconnect item` / `○ Apply once` / `○ Leave unchanged`, all disabled.
  3. "Apply reviewed" is dim and disabled. It has no "○" prefix and no reason on screen, so the two safe creates (emoji file, bad-frontmatter file) never apply either (98, 99).
- **Who it hurts:** Anyone who deletes or renames a file, which are routine filesystem operations. Their folder stops syncing for good.
- **Cause:** `Widgets/Library/library_notes_add_from_files_canvas.py:780-800` renders non-conflict attention choices with `disabled=True` unconditionally.
- **Fix:**
  - Wire these choices to the runtime's operations.
  - Let Apply reviewed apply the safe set while attention rows stay pending.
  - Show the blocker reason as a visible line, not tooltip-only.

### P1-10: After a failed sync Recovery, "Add from files…" opens a blank page for the rest of the session

- **Where:** Notes list › Add from files…
- **Repro:** After P0-2, go to Manage sync folders, press Recovery (it fails), press ‹ Notes, then Add from files…. The page shows only "Add files to Library notes." / "Recovery failed — RuntimeError. Next: Check changes." There are no Import once or Keep synced choices, no back button, and the footer offers only "F6 next pane" (92). Escape, then reopening, gives the same page (94). Only a restart clears it.
- **Who it hurts:** Anyone after any sync failure. Import once and new sync setups are blocked too.
- **Cause:**
  - The sync controller leaves `phase="roots"` after a failure (`UI/Library_Modules/library_notes_sync_controller.py:1527-1529`).
  - The Add-from-files canvas has no `roots` branch (`library_notes_add_from_files_canvas.py:356-595`, `1026-1156`), so it renders an empty body.
- **Fix:** Reset the lasting phase to `choose` whenever Add from files is opened, and render the chooser as a fallback for any unknown phase.

### P2-11: A vetoed nav-bar switch is silent, and the nav bar highlights the wrong destination

- **Where:** Top nav bar while a note has a validation error.
- **Repro:** With the trailing-space title from P1-4 in place, click `⌃2 Console`. Nothing happens on screen; the log says "Navigation to chat vetoed by the outgoing screen's pending-work flush". The nav bar now frames **⌃2 Console** while the Library screen shows (14, 15).
- **Who it hurts:** Users who think the app is frozen, or that they are on Console.
- **Cause:** `app_navigation.py:529-535` returns False with only a `logger.info`.
- **Fix:** Notify with `_library_note_editor_exit_veto_message(kind)` and re-select the current tab in the nav bar.

### P2-12: The leave-veto toast names a control that isn't there and blames the title for any veto

- **Where:** Notes editor, Escape / ‹ Notes.
- **Repro:** On an existing note, set a trailing-space title and press Escape. Toast: "Can't leave yet — fix the title or press Discard new note." There is no Discard button on screen (13). A keyword veto produces the same sentence.
- **Cause:** `library_screen.py:909-921` returns a fixed string for every `VALIDATION_VETO`.
- **Fix:** Name the field ("Remove the trailing space from the title" / "Remove the duplicate keyword 'x'"). For existing notes, offer "Revert changes" instead of the absent Discard.

### P2-13: After a delete, note actions stay armed on the deleted note and give a wrong, sticky error

- **Where:** Notes list › Folders & placement.
- **Repro:**
  1. Delete "Trip planning: Lisbon".
  2. "Add to folder" and "Move note" stay enabled (bold, 26). Move note › Recipes › Choose gives "That folder changed elsewhere — refresh and try aga…" (28, truncated).
  3. The notice persists in the authority line after a successful Undo and later operations (32). There is no refresh control.
- **Cause:**
  - The placement selection is not cleared on delete (hypothesis).
  - `FolderConflictError` maps to generic copy (`library_screen.py:18797-18801`).
- **Fix:** Clear `tree_selected_placement_id` when the selected note is deleted. Say "That note was deleted — Undo to restore it". Clear the notice on the next successful operation.

### P2-14: A deleted note stays staged in Console as "Ready" and the turn sends anyway

- **Where:** Console › Sources — next send.
- **Repro:** Use in Console on "Trip planning: Lisbon", go back to Library and delete it, then return to Console. "Sources: 1 staged · Trip planning: Lisbon (note) · Ready". Send works with no notice (29, 30).
- **Unverified:** whether the deleted note's text was transmitted (the mock log has no bodies).
- **Fix:** Revalidate staged sources at send time. Show "Deleted — remove from this send", and block the send until the user acknowledges it.

### P2-15: Moving a folder into a collapsed folder hides the target's own notes

- **Repro:** Select Recipes › Move › Research › Choose. "▾ Research" shows only "▾ Recipes" and its 2 notes. Research's own 5 notes (in the DB) appear only after collapse + re-expand (44–46).
- **Fix:** Reload the destination branch fully after `move_folder`, in `_reconcile_library_notes_tree_mutation`.

### P2-16: Folder target pickers list only folders that happen to be expanded

- **Repro:** With Projects collapsed, Move Recipes offers Top level, Agent_Lessons, Journal, Meetings, Projects, Research, Study (43). With Projects expanded, the Move note picker also listed "Projects / App launch" and "Projects / Thesis" (27). Journal/Daily and Study/Japanese never appear while their parents are collapsed.
- **Cause:** Options are built from loaded `tree_branches` (`library_notes_controller.py:3250-3273`).
- **Fix:** Query all active folders for the picker.

### P2-17: Media import has no cancel, and Enter imports while the footer says "check"

- **Repro:**
  - Type a folder path with 60 `.md` files and press Enter (footer: "enter check this path"). The import starts at once.
  - There is no Cancel on queued or parsing rows. Escape leaves while it runs to 60/60 (68, 69).
- **Cause:** `can_cancel` is true only for active local STT jobs (`Library/library_ingest_state.py:2240-2250`; `Widgets/Library/library_ingest_canvas.py:924-933`).
- **Fix:** Relabel the chip "enter import". Add a batch "Cancel remaining (N)" that marks queued jobs cancelled. Confirm before importing a folder with more than 10 files.

### P2-18: Notes Trash cannot delete anything permanently

- **Repro:** Recently deleted: "Deleted notes stay here until you restore them… nothing is removed for good from here." Restore is the only action (40). Media's trash has "Delete forever" with a confirm (80).
- **Who it hurts:** Users who deleted a note with sensitive content. It stays in the DB and the trash forever.
- **Fix:** Add "Delete forever" (key `x`, as in Media) with the same "This cannot be undone" confirm.

### P2-19: Selecting an item in Media Trash leaves an unrelated live item in the reader

- **Repro:** Media › Trash › click "Old draft — delete candidate". The reader keeps showing active item "bulk-59" with Find / Read later / **Use in Console** (79). Enter does not change it.
- **Who it hurts:** Users who may act on, or permanently delete, the wrong item, believing the reader matches the selection.
- **Fix:** Show a read-only preview of the trashed item, or clear the reader to "Trashed · Restore to read".

### P3-20: The editor status is crushed into a one-column vertical strip, and "Use in Console" is clipped

- **Where:** Editor header row 2 at 160x45 (Nav collapsed).
- **Observed:** A single letter per row at column 74: "S" for Saved, "E n —" for "Empty note — …", "U c" for Unsaved changes. "Use in" is cut at the pane edge (08, 11, 84).
- **Cause:** `#library-note-status` shares `#library-note-header-second-row` with the action toolbar (`library_notes_canvas.py:2818-2823`) and gets no width.
- **Fix:** Give it `width: 1fr; min-width: 12`, or drop it here, since the authority line already shows the status.

### P3-21: Sync roots are indistinguishable and receipts are unlabeled

- **Repro:** With two roots named "Riley Vault" and "Riley MD", both read "Sync folder (name unavailable before cutover)". The Receipts list mixes both roots with no root label (96). The cause is hard-coded copy at `library_notes_sync_controller.py:819`.
- **Fix:** Use the display name the user typed, and prefix receipts with it.

### P3-22: Error toasts expose exception class names and repeat themselves

- **Repro:** Export to a read-only dir. "Export failed — check the destination and try again." appears 3× (status line, Info, bottom bar), plus the toast "Error exporting note: PermissionError" (71).
- **Fix:** One message: "Can't write to <dir> — you don't have permission. Choose another folder."

## Improvement opportunities (beyond the defects)

1. **One "your text is safe" contract.**
   - Write a crash-safe draft journal (per-note draft on disk, flushed on every debounce tick), so quit, crash or power loss never loses more than a few hundred ms.
   - Restore it on next open: "Recovered unsaved changes from 20:11 — Keep / Discard".
2. **A per-root sync health panel.**
   - Name each root.
   - List each blocked file with a plain reason and one working action.
   - Always offer "Pause and keep both copies" plus Disconnect, so no state is a dead end.
3. **Forgiving input instead of vetoes.** Trim title whitespace and dedupe keywords on save, and show what changed ("Removed trailing space"). Vetoing and stealing focus is the most hostile option.
4. **Shortcut safety.** Single-letter globals (`i`, `n`, `g`, `/`) should require a focused list row, not "no input focused". After a jump to another destination, show a toast like "Opened Import (i) — Esc to go back".
5. **A list that behaves like a list at every width.** Scroll the focused row into view, add a scrollbar, and support Home/End/PageUp/PageDown plus ←/→ to collapse and expand folders.

## Nielsen heuristic scores

4 = fully meets, 0 = fails, −1 = not assessed.

| Surface | H | Score | Key issue |
|---|---|---|---|
| Notes | 1 Visibility of system status | 1 | "Saved"/"changes save automatically" shown when nothing will save (P0-2, P1-5); a vetoed nav highlights the wrong tab |
| Notes | 2 Match with real world | 2 | "placement", "cutover", "NotesSyncRootRefused", "RuntimeError", "Preview explicit filesystem move" |
| Notes | 3 User control & freedom | 2 | Undo delete and Restore folder are good; no Revert on a veto; Ctrl+Q has no guard; sync loop has no exit |
| Notes | 4 Consistency & standards | 2 | Disabled "Apply reviewed" lacks the "○" used elsewhere; `r` advertised with nothing to restore; compact scrolls but wide doesn't |
| Notes | 5 Error prevention | 1 | Silent overwrite on export; quit without flush; shortcuts fire during re-render; modal kills autosave |
| Notes | 6 Recognition rather than recall | 2 | Unreachable list rows; folder picker missing collapsed folders; identical root names |
| Notes | 7 Flexibility & efficiency | 3 | Fast keyboard create/edit, 42 ms latency, instant paste; no Ctrl+S |
| Notes | 8 Aesthetic & minimalist | 2 | Crushed vertical status strip; triple-repeated export errors; stale notices persist |
| Notes | 9 Recover from errors | 1 | "Recovery failed — RuntimeError" ↔ "Resolve recovery" loop; "That folder changed elsewhere" for a deleted note |
| Notes | 10 Help & documentation | −1 | Not assessed in this journey |
| Library | 1 Visibility of system status | 2 | Import queue counts are good; Trash reader shows an unrelated item; background import invisible after Esc |
| Library | 2 Match with real world | 3 | Clear ingest copy ("Path not found", "Unsupported file type: .xyz") |
| Library | 3 User control & freedom | 2 | No batch import cancel; Media delete-forever confirm is good |
| Library | 4 Consistency & standards | 2 | Footer "enter check this path" but Enter imports; Media has Delete forever, Notes does not |
| Library | 5 Error prevention | 2 | A whole folder imports on Enter with no confirm; global `i` reachable by stray typing |
| Library | 6 Recognition rather than recall | 3 | Rail counts and paging ("1-20 of 94 · Page 1 of 5") are clear |
| Library | 7 Flexibility & efficiency | 3 | Paging, type filter and page reset on filter all work |
| Library | 8 Aesthetic & minimalist | 3 | Dense but legible queue |
| Library | 9 Recover from errors | 3 | Specific, actionable ingest errors |
| Library | 10 Help & documentation | −1 | Not assessed |

## Harness caveats

- **Vault location:** Keep-synced refused the per-run fixture vault with "That folder is inside Chatbook's own data directory". `config.toml` lives at the run root, which is an ancestor of `home/fixtures`. I copied `vault/` and `md-folder/` to my scratchpad (outside the run dir) for all sync tests.
- **tmux typing fidelity:** `tmux send-keys -l` drops U+200D (ZWJ) and turns `\n` into a non-Enter key. Bracketed `paste-buffer -p` preserved both, so the app is not at fault for those.
- **Mock LLM log:** it records metadata only (no message bodies), so whether the deleted note's text was sent in P2-14 is unverified.
- **Accidental imports:** the 10-file Media import (capture 64) came from my own keystrokes landing on an unfocused surface. It is reported only for the product behaviour it exposed (P1-6, P2-17). Media counts in later captures include it plus my 60-file and 1-file test imports.
- **Latency figures** include ~10–15 ms of `tmux capture-pane` polling overhead.
- **Not reported (unreproduced or timing-dependent):**
  - Rapid Escape+Up mashing once left a stray "^" in the rail Search field. This is likely terminal escape-sequence ambiguity under tmux.
  - One mouse click on "Activate reviewed root" focused the button without pressing it. This did not reproduce on the second root.
- **Log and processes:** the run log accumulates across `REUSE=1` relaunches. Its 2 Tracebacks are handled exceptions logged at WARNING (rejected ingest path, export PermissionError), not crashes. The null keyring backend applies as described in HARNESS.md.
