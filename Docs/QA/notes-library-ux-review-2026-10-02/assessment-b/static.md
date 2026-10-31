# Assessment B: static and deterministic sweeps of Library and Library › Notes

- **Code:** `.worktrees/notes-library-ux-review` @ origin/dev `2d34cbf80d` (Textual 8.2.8). Unless a path says otherwise, it is relative to `tldw_chatbook/`.
- **Scope:** `UI/Screens/library_screen.py` (35,902 lines), `UI/Library_Modules/*`, `Widgets/Library/*`, and the UI-facing parts of `Notes/*`. The copy sweep also reads `Library/*` state modules, because they build most of the Library copy.
- **Isolation:** this sweep did not read `journeys/**`, `maps/known-context.md`, `.impeccable/critique/*` or the backlog. It did read the `map-*.md` files.
- **Detector:** not run here. This is the static-sweep agent; the mechanical agent runs the detector.
- **How the evidence was gathered:** each sweep is a script under `assessment-b/static/` with its raw output saved next to it. A hit counts only after I read the code around it. One finding, F1, was also confirmed live on socket `nl-sb-1` (160x45, golden profile). Its captures are in `assessment-b/static/live/`. After the run, the isolation check reported the real profile `untouched`, the run log contained 0 real-profile paths, and no `nl-sb-1` processes were left running.

## Sweeps run (rerun with `<venv>/bin/python <script> <worktree-root>`)

| # | Script → output | What it checks | Result after reading the hits |
|---|---|---|---|
| 1 | `s1_unresolved_attrs.py` → `s1_out.txt` | AST check of every `self.X`, `self._screen.X` (controllers) and `self.app.X` reference in scope. Each name is resolved against the class and its MRO, including Textual bases (imported and their source parsed), `self.X =` assignments, and the module-level `for f in dataclasses.fields(State): setattr(Ctrl, …)` shim loops. Imports nothing from `tldw_chatbook`. | **No unresolved method or attribute on any Library UI path.** A negative control proves the check works: a copy of the tree with three injected bogus calls is flagged at all three sites. 38 raw hits were all false positives: `__slots__`/`object.__setattr__`, a mixin, and a `staticmethod` assigned from outside the class. The one real leftover is dead code: `NotesInteropService.add_character_card`/`update_character_card` read `self.unified_db`, which is never set (only `unified_db_template` is) (`Notes/Notes_Library.py:2277-2303`). Nothing calls them through the service, so no user can reach it. |
| 1b | `s1b_bindings.py` → `s1b_out.txt` | `BINDINGS` → `action_*` methods; same-key chains; `check_action` coverage; `action_*` methods that no binding reaches | All 41 screen bindings resolve. Every bound action has an explicit `check_action` branch. `escape` is bound 15 times, `r` 3 times, `c` twice. **`action_library_notes_save` is unreachable** (F6). |
| 1c | `s1c_dead_buttons.py` → `s1c_out.txt` | Buttons with a literal id that nothing handles; handler selectors that match no composed control | No dead buttons. Artifacts and Collections route presses by type or id prefix. The one unhandled button is `#notes-sync-destination-server` "○ Server notes", which is disabled and always has a reason line. The "selector never composed" list is all helper-composed ids (spot-checked). |
| 2 | `s2_vocab.py` → `s2_out.tsv`, `s2_ui.tsv` | Prose string literals in UI sinks (notify, Static, Button, tooltip, status/notice assignments, `return` from `*_copy`/`*_label`) matching a 40-term internal-vocabulary list. f-string placeholders and SQL are stripped first. | 204 UI-sink hits, triaged in F8. Snake_case leaks are limited to config hints (`[analysis_defaults]`, `allowed_hosts under web_security`), which are defensible pointers. |
| 2b | `s2b_raw_errors.py` → `s2b_out.tsv` | f-strings or `str()` that interpolate a caught exception, or an enum `.value`/`.name`/`reason`, into UI copy | 38 exception interpolations, 23 of them in Folder files (`library_file_notes_workspace.py`). See F8. |
| 3 | `s3_swallowed.py` → `s3_out.tsv` | `except` bodies that only pass, return or log, inside user-action handlers | 92 raw hits. Most are false positives once read: a failure copy is set after the `try`, or the code is a teardown marshal. Real ones: the sync receipts (F3) and the ingest-backend toggle, which logs, then silently reverts the switch to the config value (`library_screen.py:27921-27939` → `library_ingest_controller.py:1751-1768`; rare, not written up). |
| 4 | `s4_workers.py` → `s4_out.txt` | `push_screen_wait` outside a worker; `call_from_thread` from a non-thread; `run_worker`/`@work` without `exclusive=True` on repeatable actions | **Clean.** All 7 awaited modal pushes run inside `run_worker`. All 7 flagged `call_from_thread` sites run on threads, guarded for teardown. The non-exclusive user-action workers (conflict Overwrite/Reload, recovery approve, pairing review) each have a token or `_busy` guard against double presses (`ConflictOutcomeKind.ALREADY_RUNNING`, `notes_recovery_dialog.py:144`, `library_file_notes_workspace.py:6655`). |
| 5 | manual trace: every Escape, back and close path, plus the quit and suspend lifecycle, guided by sweeps 1b and 3 | Edits discarded without confirmation or autosave; destructive actions without confirm or undo | F1 (P0, live-confirmed), F4, F5 |
| 6 | `s6_layout.py` → `s6_out.txt` | A `Static` (width unset, so it fills the row) inside a `Horizontal` that also holds buttons, with no width rule anywhere; fixed `width`/`min-width` ≥ 40 cells | **No defect.** The 2 candidates are cleared: `#library-note-heading` is `layout: vertical`, and `#library-collections-archive-receipt > Static` is `width: $ds-width-fill`. No Library dialog is wider than 80 cells without `max-width` (`#file-notes-conflict-dialog` is 110 but capped at 95%). Limitation: this only sees `with Horizontal(...)` compositions, not containers made horizontal in CSS. |
| 7 | manual table built from the `BINDINGS`, `on_key` and footer constants (`library_screen.py:1243-1600`) | The same key meaning different things on sibling canvases; save models per editor | F6, F7 |

---

## Findings (most severe first)

### F1 · P0 · Ctrl+Q throws away unsaved Library edits without saving or asking (live-confirmed for notes)

**What happens.** When you type in a note and press Ctrl+Q within 2 s of the last keystroke, Chatbook quits and the text is gone. The screen said "Unsaved changes · Next: Keep editing; changes save automatically." right up to the keypress. The 2 s window covers *any* continuous typing burst: autosave is a pure debounce that re-arms on every change, so it waits for a 2 s pause. The same quit path also drops, without a prompt:
- dirty Prompt drafts (explicit Save);
- dirty Skill drafts (explicit Save / Ctrl+S);
- a pending Folder files autosave.

**Who it hurts.** Every writer, and keyboard-first users most of all. Ctrl+Q is on the footer of every screen ("Ctrl+Q quit") and needs no confirmation.

**Trace.**
- The quit flow consults only `confirm_quit` / `prepare_for_quit` on the active screens: `app_lifecycle.py:1829-1923`, which uses `quit_confirmation_screens`, `confirm_quit_screens` and `prepare_quit_screens` from `Widgets/confirmation_dialog.py:174-238`.
- LibraryScreen defines neither. A grep of `library_screen.py`, `Library_Modules/`, `Widgets/Library/` and `UI/Navigation/base_app_screen.py` finds 0 hits. Chat, Settings, Personas and Chunking Lab all implement them.
- The Library's own `flush_pending_work` (`library_screen.py:11109-11141`) is wired only to tab navigation (`app_navigation.py:513`), never to quit.
- `on_unmount` actively *cancels* the pending save: `self._invalidate_library_note_autosave()` at `library_screen.py:9738`, which stops the timer at `library_notes_controller.py:3697-3702`. It never calls `_flush_library_note_save`.
- The debounce: `library_screen.py:20738-20749`, using `LIBRARY_NOTES_AUTOSAVE_SECONDS = 2.0` (`UI/Library_Modules/screen_constants.py:177`).
- Folder files: `LibraryFileNotesWorkspace.shutdown` stops `_autosave_timer` and only awaits a save that has *already started* (`library_file_notes_workspace.py:2428-2455`).

**Live evidence (socket `nl-sb-1`, golden, 160x45).** Library › Notes ▸ "Ideas inbox" ▸ Ctrl+End:
1. Typed ` ZZPROBE1` and waited 4 s. The status read "Saved 20:26 · Next: Keep editing; changes save automatically." (`static/live/03-after-autosave-160x45.txt`).
2. Typed ` ZZPROBE2`. The status read "Unsaved changes · Next: Keep editing; changes save automatically." (`static/live/04-dirty-160x45.txt`).
3. Pressed Ctrl+Q. The app exited with no prompt (log: `Application quit initiated`, 0 tracebacks).
4. `sqlite3 "file:…/runs/nl-sb-1/data/nl_review/tldw_chatbook_ChaChaNotes.db?mode=ro" "select version, instr(content,'ZZPROBE1')>0, instr(content,'ZZPROBE2')>0 from notes where title='Ideas inbox'"` returned `2|1|0`. The first edit was saved; the second was lost.

**Probe for the other editors:**
- Prompts: open "Rewrite for clarity", change the name, press Ctrl+Q, relaunch with `REUSE=1`, reopen.
- Skills: the same, after "Set up skill trust".
- Folder files: link the fixture vault, edit a file, press Ctrl+Q at once, then `cat` the file.

**Fix.** Give `LibraryScreen` the two quit hooks the quit flow already calls:
- `async def prepare_for_quit(self) -> None`: `await self.flush_pending_work()`.
- `async def confirm_quit(self) -> bool`: run the same flush, and return `False` (stay) when it refuses. The flush already raises the right toasts for a validation veto, a conflict, or a dirty prompt or skill.
- For explicit-save editors, show a `ConfirmationDialog`: "Save changes to ‹name› before quitting? · Save · Discard · Stay".
- Make `on_unmount` flush before it invalidates.
- Add a regression test that types into `#library-note-body` and then calls `app.action_quit()`.

**Confidence:** high. The notes path is live-confirmed; the prompt, skill and Folder files paths are confirmed by reading the code.

### F2 · P1 · Every lasting-sync folder is titled "Sync folder (name unavailable before cutover)"

**What happens.** In Notes ▸ Add from files ▸ Keep a folder synced, the user types a "Display name" (`#notes-sync-display-name`, placeholder "Folder label in Notes"). Manage sync folders never shows it back. Every root row's heading is the same hard-coded literal, so with two synced folders the rows look identical. Check changes, Pause, Resume and Recovery then act on a folder the user cannot identify. The text also leaks the internal word "cutover".

**Trace.**
- `UI/Library_Modules/library_notes_sync_controller.py:817-819`: `LastingSyncRootRow(root.root_id, "Sync folder (name unavailable before cutover)", …)`.
- It is rendered as the row's `destination-section` heading by `Static(root.display_name)` at `Widgets/Library/library_notes_sync_roots_canvas.py:72-76`.

**Probe.**
1. Golden, 160x45 ▸ Notes ▸ Add from files… ▸ Keep a folder synced.
2. Display name "Vault", folder `home/fixtures/vault` ▸ Check changes ▸ Activate.
3. Repeat with "Plain", folder `home/fixtures/md-folder`.
4. Manage sync folders: both headings read the literal.

**Fix.** Carry the root's display name, or the managed Library folder's name, in the runtime root projection (`runtime.roots`) and pass it as the second argument. Fall back to "Synced folder 1/2/…" in creation order, never to the literal. The same projection should feed the receipt rows, so a receipt names its folder.

**Confidence:** high (hard-coded literal). Not run live: activating two roots is a long flow.

### F3 · P1 · Sync "Receipts" says "No writes yet." when the history could not be read

**What happens.** Manage sync folders ▸ Receipts is, in the code's own words, "the only trace of the writes lasting sync performs on its own" (`library_screen.py:27661-27664`). If reading a root's history fails, that root's receipts are silently skipped (`continue`). If every read fails, the section renders "No writes yet. Sync writes are listed here when you open this list." That is a false statement about what sync has done to the user's files and notes. A partial failure is worse to spot: the remaining roots' receipts look complete.

**Trace.**
- `library_notes_sync_controller.py:875-913` (`except Exception: logger.warning(...); continue`) publishes `write_receipts=()`.
- `library_notes_sync_roots_canvas.py:122-127` renders the empty copy.
- The screen-level `try/except` at `library_screen.py:27665-27671` also only logs.
- A realistic trigger: `NotesSyncRuntime.write_receipts` calls `_require_cutover`, which raises `RuntimeError("notes_sync_cutover_not_admitted")` whenever admission is closed during maintenance, startup or shutdown (`Notes/notes_sync_runtime.py:2687, 3711-3714`).

**Probe.**
1. With one active root that has applied writes, quit and relaunch.
2. Open Manage sync folders as soon as Notes is reachable, while the runtime status still reads "◌ Starting".
3. Compare against reopening the list after the runtime is active.
4. Alternative: `chmod 000` the notes device-state DB under `runs/<s>/data` and reopen the list.

**Fix.**
- Collect the failed root ids in `refresh_receipts`.
- Render "Couldn't read sync history for N folder(s) · [Retry]" (and per row when the roots are named, see F2).
- Show "No writes yet." only when every read succeeded and returned nothing.
- Change the empty copy to "No sync writes recorded." Drop "when you open this list", which describes the implementation.

**Confidence:** high on the code path; medium on how often it triggers.

### F4 · P1 · Exporting a note or prompt silently overwrites existing files, always starting at ~ with a title-derived name

**What happens.** Info ▸ "Export Markdown" or "Export text" (and the prompt "Export…") opens a FileSave rooted at `Path.home()` and prefilled with `<title>.md`. FileSave's default is `can_overwrite=True`, and the write is a plain `write_text`. So:
- Exporting the seeded "Reading list" (Unfiled) and then "Reading list" (Study) replaces the first file without a word.
- A note titled "TODO" replaces the user's own `~/TODO.md`.

The success toast names only the file ("Note exported successfully to Reading list.md"), not the directory. A failure toast shows a class name: "Error exporting note: PermissionError".

**Trace.**
- `library_notes_controller.py:4300-4323`: `safe_title`, then `FileSave(location=str(Path.home()), …)` with no `can_overwrite`.
- `Third_Party/textual_fspicker/file_save.py:37` (`can_overwrite: bool = True`) and `:90` (the only overwrite check).
- `library_screen.py:21158-21180`: `validated_path.write_text(...)`; toast `f"Error exporting note: {type(exc).__name__}"`; toast `f"... to {validated_path.name}"`.
- Prompts: `library_prompts_controller.py:3709-3712`, the same pattern.
- The Export-bundle canvas *does* check `destination_exists` (`library_export_controller.py:1408`), so the Library is inconsistent with itself.

**Probe.**
1. Notes ▸ open "Reading list" (Unfiled) ▸ Info ▸ Export Markdown ▸ Enter.
2. Open "Reading list" (Study) ▸ Info ▸ Export Markdown ▸ Enter.
3. `cat runs/<s>/home/Reading\ list.md`: it holds the Study copy only, and no prompt appeared.

**Fix.**
- Pass `can_overwrite=False`, or better, intercept an existing path with "Replace ‘Reading list.md’ in ~/? · Replace · Choose another name".
- Remember the last export directory, as the import and sync pickers already do (`remember_browse_directory`).
- Toast the full path.
- Map `PermissionError` and `OSError` to "Couldn't write to ~/x — the folder is read-only. Choose another folder."

**Confidence:** high.

### F5 · P2 · Escape means three different things in the Add-from-files flows, and two of them throw work away

**What happens.**
- **Import once, while importing.** Escape *cancels the import*, and finished items are not rolled back ("Cancelled. Finished items were not rolled back."). The visible "‹ Notes" button does the opposite: you leave, and the import keeps running. The footer does say "esc cancel" here, but everywhere else in Library, Escape means back.
- **Keep a folder synced ▸ review** (setup review, or a review opened from Manage sync folders). Escape *abandons the whole review*. That discards the reviewed plan and every conflict choice staged across pages ("Choice staged. No changes yet."), with no confirmation, and returns to the Notes list. This includes the case where a "View comparison" panel is expanded, where Escape is the natural "close this" key.
- **Keep a folder synced ▸ checking / activating.** Escape is refused ("wait current step").

**Who it hurts.** Users partway through a long import, or through a conflict review of a real vault. The review has to be redone with Check again and every choice made again.

**Trace.**
- `library_notes_controller.py:3095-3107` (Esc branch) → `:5394-5406` (`_exit_library_notes_lasting_sync`) → `library_notes_sync_controller.py:1166-1178` (`abandon_setup`, `_review_plan = None`).
- Import cancel: `library_note_import_controller.py:754-769`; cancel copy at `Library/library_note_import_state.py:1386`.
- The Back button that keeps importing: `library_notes_controller.py:5407-5412`.
- The comparison is inline (`library_notes_add_from_files_canvas.py:700-724`), so it has no Escape of its own.

**Probe.**
- Import once on `home/fixtures/vault` ▸ Import selected items ▸ Escape right away. Read the receipt, and compare with pressing "‹ Notes".
- Keep a folder synced on the vault, after editing a file in `home/fixtures/vault` and the matching note so that a conflict row appears ▸ Keep both ▸ View comparison on another row ▸ Escape.

**Fix.**
- Make Escape equal the visible "‹ Notes" in Import (the import continues). Cancel stays on its own "Cancel import" button.
- In a lasting review, Escape first collapses an open comparison.
- With ≥1 staged choice, either keep the review resumable from Manage sync folders, or confirm: "Leave review? N staged choices will be discarded · Leave · Stay".
- The footer label should then say exactly that.

**Confidence:** high.

### F6 · P2 · Ctrl+S saves only Skills; the Notes save action exists but no key reaches it

**What happens.** One screen holds four editors with four save models:
- **Library notes:** 2 s autosave plus a "Save" button.
- **Folder files:** autosave, with no Save button.
- **Prompts:** an explicit "Save prompt" / "Save changes" button and a dirty-veto when you leave.
- **Skills:** explicit Save plus Ctrl+S, advertised as "ctrl+s save skill".

Ctrl+S does nothing in Notes, Prompts or Folder files. A user trained by Skills presses Ctrl+S in a prompt, sees nothing happen, and then hits the dirty-veto toast on leaving.

**Trace.**
- `library_screen.py:1047`: `("ctrl+s", "library_skill_save", "Save skill")` is the only ctrl+s binding (sweep 1b).
- `action_library_notes_save` (`library_screen.py:8688-8719`, with refusal copy already written) and its `check_action` gate (`:25358-25363`) have **no binding and no caller**. Sweep 1b reports "unbound action method: action_library_notes_save (other refs by name: 0)".
- The editor footer advertises no save key (`LIBRARY_NOTES_EDITOR_SHORTCUTS`, `:1575-1578`).

**Probe.**
1. Open a note, type, press Ctrl+S at once. The status stays "Unsaved changes" until the 2 s debounce fires.
2. Open the prompt "Rewrite for clarity", edit the name, press Ctrl+S: nothing happens. Press Escape: the dirty-veto toast appears.

**Fix.**
- Add `Binding("ctrl+s", "library_notes_save", "Save note", show=False)` and a `library_prompt_save` binding next to the skill one. The gates are already disjoint by editor.
- Add `("ctrl+s", "save")` to the notes editor and prompt editor footer sets.
- Folder files: bind Ctrl+S to "save now" (flush the autosave).

**Confidence:** high.

### F7 · P3 · Select-mode and hand-off keys differ between sibling list canvases

**What happens.**

| Key | Media | Notes | Prompts and others |
|---|---|---|---|
| Enter or exit select mode | `s` | Select button only (no key) | — |
| Toggle a row | Space | Enter ("enter select note") | — |
| Leave select mode | `s` ("done selecting") | Esc ("esc done") | — |
| Export selected | Export… button | `e` ("e export selected") | — |
| Use in Console | `c` in the Reader | "Use in" button, no key | — |

In Conversations, `c` means "Resume conversation". A user who learns one canvas gets a no-op or a different action on the next.

**Trace.**
- `library_screen.py:1158-1196` (s, space, c, c) and `:1039-1042` (g, e).
- Footer sets: `:1361-1372` (Media) vs `:1547-1562` (Notes).
- Media's Space binding is gated to media rows (`:25563-25600`).

**Probe.** At 160x45 in Media, press `s`, then Space on a row, then `s`. In Notes, press `s` (it types nothing, since focus is not in a field; check the footer first). Press the Select button, then Space on a row, then Enter.

**Fix.** One grammar on every list canvas:
- `s` toggles select mode;
- Space toggles the focused row;
- `e` exports the selection;
- Esc leaves select mode.

Bind `c` to "Use in Console" in the note editor when focus is not in a text field. Gate all of these through `check_action`, as the Media keys already are.

**Confidence:** high (static).

### F8 · P3 · Internal vocabulary and raw exception text reach the screen (traced)

Each row below gives the string, the code that puts it on screen, and a replacement.

| On screen | Where it renders | Replace with |
|---|---|---|
| Footer "esc focus rail" / compact "esc rail" | `LIBRARY_LIST_SHORTCUTS` / `LIBRARY_NOTES_NAVIGATOR_SHORTCUTS*` (`library_screen.py:1352-1356, 1547-1562`). The pane they point at is titled **"Navigation"** with the grip "N a v" (live capture `01`). The same word appears in "use Create ▸ New skill in the rail" (`library_skills_canvas.py:78`), "choose All Captures in the rail" (`library_collections_capture_reader.py:338`), "Choose Archived in the Library rail" (`library_collections_controller.py:1573`), and the tooltip "Return to the Library rail" (`library_screen.py:5671`). | "esc navigation" / "esc nav"; "…in Navigation" |
| "Folders & placement" heading; "○ Remove placement"; import row "Add folder placement" | `library_notes_canvas.py:118` (heading) and `:2629-2650` (buttons); `library_note_import_canvas.py:1206`. Visible on every wide Notes list (live capture `01`, rows 22-24). | "Folders"; "Remove from folder"; "Also add to folder" |
| "! Needs owner review" (tree row status) | `Library/library_notes_tree_state.py:718, 1004` | "! Sync folder missing — review in Manage sync folders" |
| "Stored System lane" / "Stored User lane" (Prompt history) | `UI/Library_Modules/prompt_history_region.py:412, 422`, as `Static` labels above read-only TextAreas | "System prompt (saved)" / "User prompt (saved)" |
| "Persisted source: Local · Prompt · …" (Prompt Info) | `library_prompts_canvas.py:497` | "Saved in: Local library · Prompt · …" |
| "The selected conversation does not match the retained transcript." (tooltip on the disabled hand-off) | `library_conversation_reader.py:45` | "Still loading this conversation — wait a moment, then try again." |
| "Undo finished, but its fresh projection is unavailable." (sync status) | `library_notes_sync_controller.py:1977` | "Undo finished. Couldn't refresh the list — press Check changes." |
| "Git mutation in progress…" / "save authority changed" / "Status: STALE · BLOCKED — Repository or session authority changed before the action." (Folder files) | `library_file_notes_workspace.py:4908, 6333, 8034` | "Git is updating…" / "Save stopped: the file changed hands — review it before saving." / "Out of date: the repository changed. Refresh, then try again." |
| Raw exceptions: `f"Open failed: {error}"`, `f"Refresh failed: {error}"`, `f"reload failed: {error}"`, `f"Pairing review unavailable: {error}"`, `f"Pairing was not approved: {error}…"` | `library_file_notes_workspace.py:6055, 6186, 6228, 6497, 6725`; `notes_recovery_dialog.py:184`. Example text: `FileNotFoundError(f"File Notes root is offline: {self.root}")` (`Notes/file_notes_service.py:465`) becomes "Open failed: File Notes root is offline: /abs/path". The full list is in `s2b_out.tsv`. | Map known types to sentences: "Can't open — the folder is offline. Reconnect the drive, then Refresh." Never interpolate `str(error)`. |
| "Re-chunk items persisted before the current chunking engine through the template-aware path…" (tooltip) | `library_search_rag_panel.py:227` | "Re-split older items with the current chunking settings, then re-index them." |

**Probe.** Read each surface at 160x45:
- Notes list ("Folders & placement", footer);
- Prompts ▸ open a prompt ▸ History ▸ open a version;
- Folder files ▸ Choose folder ▸ point at the vault ▸ `mv` the vault away ▸ open a file.

**Fix.** As in the last column. Add the terms to a copy-lint test: `s2_vocab.py`, run over the UI sinks with an allowlist.

**Confidence:** high for the sinks traced above.

---

## Not reported, and why

- The `n` and toolbar "New" mismatch, the trash cap and missing purge, keyword-blind filtering, and Use-in-Console auto-linking: these are behaviour or IA issues, already mapped (map-notes S-10/S-15/S-02/S-11) and outside the seven static checks.
- Manage sync folders: for an offline root, "Next: Reconnect folder" maps its primary action to the permanently disabled "○ Retarget" (`library_notes_sync_roots_canvas.py:189-201`). This is a real dead end, but map-notes S-12 already covers it with a probe.
- The ingest-backend toggle reverts silently when the config write fails (sweep 3). It needs a failing config write to trigger, and it is not in Notes.
