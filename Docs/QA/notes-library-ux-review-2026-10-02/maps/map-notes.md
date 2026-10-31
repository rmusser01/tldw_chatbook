# UX map: Library > Notes (all modes)

- **Code base:** `.worktrees/notes-library-ux-review` @ origin/dev `2d34cbf80d` (2026-10-02), Textual 8.2.8.
- **Method:** static read of source, with every claim cited `path:line`. Paths are relative to `tldw_chatbook/` unless they start with `Docs/` or `Helper_Scripts/`. I read the User Guide pages (`Docs/User_Guide/library/notes.md`, `file-notes.md`) only to learn the intended behaviour; where the guide and the code disagree, the code is treated as the truth and the gap is flagged.
- **Status of findings:** nothing here was run live. Section 7 lists **suspected** issues, each with a probe for live verification. This document makes no design recommendations.
- **Live-probe hygiene:** never touch the real profile. Use `Helper_Scripts/verify_notes_files_sync_tui.py`, or a disposable `HOME`/XDG with `TLDW_CONFIG_PATH`. Test at both 235x52 (wide) and 100x30 (compact, under the 120-column breakpoint). Spot-check 60x20 as well.
- **Task-brief correction:** `Notes/sync_engine.py` does not exist. Lasting sync lives in `Notes/notes_sync_runtime.py`, `notes_sync_reconciler.py`, `notes_sync_coordinator.py`, `notes_sync_executor.py`, `notes_sync_watcher.py` and the related modules. `UI/Screens/notes_scope_models.py` is a vestige of the retired standalone Notes screen (scopes LOCAL_NOTE/SERVER_NOTE/WORKSPACE). It is imported only by `UI/Screens/study_screen.py:41`. Library notes exposes the local scope only.

---

## 1. Inventory

### 1.1 Three "notes worlds" that meet in one screen

| World | What it is | Entry | Primary code |
|---|---|---|---|
| **Library notes** (database) | Notes stored in ChaChaNotes DB. Folder tree, keywords, templates, autosave, Console hand-off, soft delete. | Rail Browse ▸ **Notes (N)**, Create ▸ **New note**, hub **New note**, `n` / `Ctrl+N` | `Widgets/Library/library_notes_canvas.py`, `UI/Library_Modules/library_notes_controller.py`, `UI/Screens/library_screen.py` |
| **Folder files** | Plain `.md/.markdown/.txt/.text` files edited in place in a linked folder, plus a Session Git panel. | Source strip **Folder files** | `Widgets/Library/library_file_notes_workspace.py` (8,856 lines), `library_file_notes_git_panel.py` (4,309 lines) |
| **Keep a folder synced** (lasting sync) | A reviewed, two-way relationship between a local folder and one managed Library notes folder. | Notes list **Add from files…** ▸ **Keep a folder synced**; **Manage sync folders** | `Widgets/Library/library_notes_add_from_files_canvas.py`, `library_notes_sync_roots_canvas.py`, `UI/Library_Modules/library_notes_sync_controller.py` |
| **Import once** (a one-shot flow, not a world) | Copies files into Library notes, reproducing the source's folder hierarchy. | **Add from files…** ▸ **Import once** | `Widgets/Library/library_note_import_canvas.py`, `UI/Library_Modules/library_note_import_controller.py` |

### 1.2 Surfaces, by mode (`LibraryNotesCanvas.mode`, `library_notes_canvas.py:1154-1200`)

| # | Surface | Mode / owner | Key anchors (DOM ids) | Code |
|---|---|---|---|---|
| S1 | Rail rows: Browse ▸ "Notes (N)", Create ▸ "New note"; starter rail "Import…", "New note", "Explore all tools" | Library rail | `#library-rail`, `#library-rail-explore-all` | `Library/library_shell_state.py:523-532, 634-643, 734`; `Widgets/Library/library_rail.py:1093-1095` |
| S2 | Library hub quick actions "Import…", "New note", "Search"; Get-started steps | Landing | `#library-hub-action-new-note`, `#library-hub-action-import` | `Widgets/Library/library_entry_canvases.py:349-397` |
| S3 | Source strip "Library notes \| Folder files" (compact: "‹ Library / Notes") | Notes route | `#library-notes-source-database`, `#library-notes-source-files`, `#library-notes-task-return` | `UI/Library_Modules/library_browse_route_swap.py:110-174` |
| S4 | Notes list (navigator): authority line, header, purpose line, filter, toolbars, tree, receipts | `mode="list"` | `#library-notes-header`, `#library-notes-filter`, `#library-notes-new`, `#library-notes-sort`, `#library-notes-select-toggle`, `#library-notes-add-from-files`, `#library-notes-export`, `#library-notes-manage-sync-folders`, `#library-notes-import-receipt`, `#library-notes-list` | `library_notes_canvas.py:1740-2202` |
| S5 | Sort chooser strip (Newest / Oldest / Title) | inside list | `#library-notes-sort-choices`, `.library-notes-sort-choice` | `:2014-2027`; handler `library_notes_controller.py:4916-4952` |
| S6 | Select mode strip | list with `select_mode` | `#library-notes-selection-actions`, `#library-notes-select-all`, `#library-notes-select-clear`, `#library-notes-export-selected`, `#library-notes-selection-status` | `:1796-1902` |
| S7 | Folder tree rows (folder / Unfiled / note / pager) | list | `.library-notes-folder-row`, `.library-notes-tree-note-row`, `.library-notes-tree-pager` | `:2295-2450`; `Library/library_notes_tree_state.py:690-1220` |
| S8 | "Folders & placement" action group | list (wide), reduced (compact) | `#library-notes-folder-new`, `-folder-rename`, `-folder-move`, `-folder-remove`, `-placement-add`, `-placement-move`, `-placement-remove`, `-folder-restore` | `:2452-2686` |
| S9 | Delete receipt "✓ deleted · <title>" with Undo / Dismiss | list | `#library-notes-delete-undo`, `#library-notes-delete-receipt-dismiss` | `:2121-2160` |
| S10 | "Recently deleted (N)" opener + Trash view | `mode="trash"` | `#library-notes-trash-open`, `#library-notes-trash-back`, `.library-notes-trash-restore` | `:2204-2293` |
| S11 | Work pane, empty ("Select a note to edit it here.") | wide reader, list mode | `#library-note-work-empty` | `Widgets/Library/library_note_work_pane.py:37-52` |
| S12 | Note loading / load failed + Retry | `mode="loading"` | `#library-note-loading`, `#library-note-load-retry` | `:1705-1738`; timeout `UI/Library_Modules/screen_constants.py:290-297` |
| S13 | Editor: heading, location row, status + mode/task actions, Title / Keywords / Body, chrome strip, conflict callout | `mode="editor"`, Edit | `#library-note-back`, `#library-note-edit`, `#library-note-preview`, `#library-note-context`, `#library-note-save`, `#library-note-use-in-console`, `#library-note-discard-new`, `#library-note-title`, `#library-note-keywords`, `#library-note-body`, `#library-note-chrome-facts`, `#library-note-conflict-overwrite`, `#library-note-conflict-reload` | `:2730-3097`, `apply_session_state` `:3296-3633` |
| S14 | Preview (rendered Markdown) | editor, `presentation="preview"` | `#library-note-preview-region`, `#library-note-preview-body` | `:2896-2913`; `render_preview_source` `:440-467` |
| S15 | Info: Properties / Keywords / Linked from / Reuse & Export / Danger + inline delete confirm | editor, `region="context"` | `#library-note-context-region`, `#library-note-context-keywords`, `.library-note-backlink`, `#library-note-context-copy`, `#library-note-context-export-md`, `#library-note-context-export-txt`, `#library-note-context-delete`, `#library-note-delete-cancel`, `#library-note-delete-confirm` | `:2914-3006, 3099-3123` |
| S16 | New note view (Blank note, "From a template…" disclosure, 8 template rows) | `mode="create"` | `#library-notes-create-back`, `#library-notes-create-blank`, `#library-note-from-template`, `.library-notes-template-row` | `:3750-3835`; templates `Event_Handlers/notes_events.py:54-132` |
| S17 | Folder name dialog (New folder / Rename folder) and folder target dialog (Add note to folder / Move note / Move <folder>) | modal | `#library-note-folder-name`, `#library-note-folder-target` | `Widgets/Library/library_note_folder_dialog.py:17-148`; callers `library_screen.py:29903-30090` |
| S18 | Remove-folder confirmation "Remove folder organization?" | modal | ConfirmationDialog | `library_screen.py:29997-30028` |
| S19 | Add from files chooser (Import once / Keep a folder synced / Folder files pointer) | `mode="lasting_add"`, phase `choose` | `#notes-add-import-once`, `#notes-add-keep-synced`, `#notes-sync-back` | `library_notes_add_from_files_canvas.py:354-427` |
| S20 | Keep-synced configure / checking / review / activating / receipt / history / comparison | `lasting_add` phases | `#notes-sync-display-name`, `#notes-sync-folder-choose`, `#notes-sync-obsidian`, `#notes-sync-direction-*`, `#notes-sync-check`, `#notes-sync-activate`, `#notes-sync-apply`, `#notes-sync-check-again`, `#notes-sync-history-open` | `:428-1157` |
| S21 | Manage sync folders (roots, receipts) | `mode="lasting_roots"` | `#notes-sync-root-check-N`, `-review-N`, `-pause-N`, `-resume-N`, `-recover-N`, `-retarget-N`, `-disconnect-N`, `#notes-sync-roots-back` | `Widgets/Library/library_notes_sync_roots_canvas.py:47-205` |
| S22 | Import once: selection, review (grouped, paged), importing, receipt | `mode="import"` | `.note-import-primary`, "Check selection", "Import selected items", "Cancel import" | `Widgets/Library/library_note_import_canvas.py:600-1290` |
| S23 | Export bundle (.zip) canvas scoped to notes / selected notes | Export canvas | (shared Export canvas) | `library_notes_controller.py:5610-5621`; `library_screen.py:30146-30158` |
| S24 | Folder files workspace: authority line, link row, navigator (New, "File contents…", Files tree, Search results), work area (Edit / Manage), path tasks, conflict Compare/Resolve, reload confirm, details dialog | Files source | `#file-notes-root-status`, `#file-notes-choose-root`, `#file-notes-root-details`, `#file-notes-recovery-review`, `#file-notes-search`, `#file-notes-new`, `#file-notes-edit`, `#file-notes-manage` | `library_file_notes_workspace.py:2116-2180, 1432-1490, 314-318` |
| S25 | Session Git panel (trust, stage, commit review, guarded push) | Folder files ▸ Manage | "Review session changes (N)" | `library_file_notes_git_panel.py:655-656, 1203-1350, 1810-1830` |
| S26 | Notes recovery pairing dialog | modal from Folder files | `#notes-recovery-approve`, `#notes-recovery-close` | `Widgets/Library/notes_recovery_dialog.py:18-195` |

### 1.3 Non-visual seams that shape behaviour

- **Editor session coordinator:** `Library/library_notes_session.py` (save serialization, validation veto `:582-597`, conflict tokens). The DB adapter is `UI/Library_Modules/note_session_port.py:32+`.
- **Work session (auto-collapse Library nav once on wide screens):** `UI/Library_Modules/library_notes_work_session.py:47-70`. It activates only at a reader width of 120 or more.
- **Refresh-while-typing guard:** while a note field has focus, the canvas skips its recompose (`library_notes_canvas.py:1661-1684`).
- **Responsive breakpoints:** compact below 120 columns (`screen_constants.py:275`). The editor chrome strip and location row need 80 columns or more (`library_notes_canvas.py:201`). The authority prefix is dropped below 64 columns (`:189`). The toolbar merges at a pane width of 109 or more (`:110`) and stacks below 48 (`:154`).
- **List action budget:** a deliberate ceiling of 13 visible actions (`library_notes_canvas.py:125-139`).

---

## 2. States

### 2.1 Notes list (navigator)

| State | What renders | Cite |
|---|---|---|
| Brand-new profile (all Library sources empty) | Rail is in the "starter" shape (Import…, New note, Explore all tools). There is no Browse ▸ Notes row until any source has content. | Guide `notes.md:21-29`; rail `library_rail.py:1093-1095` |
| Zero notes, list open | Header "Notes (0)"; empty copy "No notes yet. Create your first note." shown above the tree even if the seeded Agent_Lessons folder exists. That folder row reads "▸ Agent_Lessons — where Console agents file reusable lessons (empty)". | `Library/library_notes_state.py:20, 649-654, 661`; canvas `:2320-2327, 2380-2384`; gloss `Notes/agent_lessons.py:23` |
| Loading branch | Pager row "Loading notes…" / "Notes 1–20 of N  Loading…". The empty banner is suppressed while any row is loading. | `library_notes_tree_state.py:936-945`; canvas `:2320-2322` |
| Populated | Rows "Title · age". Duplicate siblings add "· Folder", then a time-of-day or "#id" tie-break. Sync-managed folders read "⇄ Sync managed"; an orphaned managed folder or placement reads "! Needs owner review". | canvas `:2405-2449`; tree state `:703-721, 998-1007` |
| Paged branch | "Notes 1–20 of 45  Load more notes", "Load earlier", "Folders …". Each page holds 20. | tree state `:810-930` |
| Branch error | "Couldn't load notes/folders/contents · Retry", "Couldn't load more · Retry" | tree state `:874-925` |
| Stale branch | "N placements loaded · May be out of date · Retry"; "Tree changed · Refreshing…". Mutation actions are disabled with "This branch may be out of date; retry it before changing it." | tree state `:825-864`; canvas `:2554` |
| Filtered | Status "filter: <q> · N results" plus a "Clear filter" button. Sort is blocked ("○ Sort: Newest") with the reason line "Sort unavailable — clear the filter". | `library_notes_state.py:639-644`; canvas `:2080-2120`; reason `Library/library_shell_state.py:154-156, 169` |
| Filter with no matches | "No notes match “<q>”. Clear the filter." | `library_notes_state.py:655-658` |
| Operation running | "Updating notes…" or an operation line ("Export…", "Copy…"). New, Sort, Select, transfer actions and note rows are disabled with the tooltip "Wait for the running notes operation to finish." | canvas `:1325-1328, 1906-1907, 2046` |
| Select mode | Rows show ☐/☑. Strip reads "N selected · Done · Select all N shown · Clear · ○ Export selected"; compact reads "Done · All N · Clear · Export". Count line "0 selected — Export selected unavailable". An open note becomes a read-only preview. | canvas `:1796-1902`; work pane `:3524-3532` |
| After delete | Receipt "✓ deleted · <title>" with Undo / Dismiss. Focus parks on Undo. | canvas `:2121-2160`; controller `:5874-5906` |
| Trash present | "Recently deleted (N)" is the last row of the list. It is absent at 0. | canvas `:2204-2226` |
| Import running in background | "Add from files…" relabels to "View import" | canvas `:2042-2051` |
| Lasting roots exist | "Manage sync folders" appears | canvas `:2057-2067` |
| Same-session import receipt | "Last import" appears | canvas `:2068-2075` |

### 2.2 Editor / Preview / Info

| State | Status-line copy (header `#library-note-status`) | Work-pane authority line (`_authority_copy`) | Cite |
|---|---|---|---|
| Loading | n/a | "Loading note…" | canvas `:1241-1248` |
| Load failed / timeout (3 s) | "Unable to load note — timed out after 3 s. Press Retry." with a Retry button | "… · Next: Retry loading." | `screen_constants.py:292-297`; `note_session_port.py:133-135` |
| Fresh blank note | "Empty note — type to keep it" | "Next: Start typing." | controller `:1441-1442`; canvas `:1276-1280` |
| Clean | "Saved", or "Saved HH:MM" after a save in this session | "Saved · Next: Keep editing; changes save automatically." | canvas `:829-830, 1281-1282` |
| Dirty | "Unsaved changes" | same | canvas `:827-828` |
| Saving | "Saving…" | no Next | canvas `:825-826, 1261-1262` |
| Explicit Save with nothing dirty | "Saved — no changes." | n/a | `library_notes_session.py:543` |
| Validation veto | The veto message, e.g. "Title begins or ends with whitespace — remove it to save." + " Next: Retry Save." | "… · Next: Review the error, then keep editing." (only if the text contains "failed") | `library_notes_state.py:182-269`; canvas `:822-824, 3449-3451, 1268-1269` |
| Save failed | "Save failed — your draft remains in the editor. Next: Retry Save." | "Next: Review the error, then keep editing." | canvas `:822-824` |
| Conflict | "Conflict — review the choices below. Next: Review recovery." + callout "This note changed elsewhere — Overwrite saves your text; Reload discards it." + [Overwrite] [Reload] | "… · Next: Resolve the conflict or reload the note." | canvas `:811-813, 1259-1260, 3073-3097`; controller `:1390-1395` |
| Preview | as above | "Next: Press Edit to change this note." | canvas `:1270-1275` |
| Select mode (bulk read-only) | "Read-only — this note cannot be changed; your draft is preserved. Next: Keep the draft." + "Read-only preview · Included/Not included in bulk selection" | — | canvas `:819-821, 3524-3532`; controller `:1474-1480` |
| Delete confirm (Info) | Inline "Delete this note? Undo will be available in the Notes list." [Cancel] [Delete]. Other Info actions are disabled and Tab is trapped. | — | canvas `:3099-3123, 3576-3617`; screen `on_key` `:8830-8850` |
| Use in Console result | "Use in Console complete — Linked to <ws> · staged in Console." / "Already linked …" / "Staged in Console.", or a failure such as "Can't use this note in Console — … Next: …" | — | controller `:4560-4656`; `library_notes_state.py:380-393` |
| Leave refused | Toast "Can't leave yet — fix the title or press Discard new note." (any validation veto) / "… the save failed; press Save to retry or Discard." / "… changed elsewhere; choose Overwrite or Reload." | — | `library_screen.py:909-931, 30193-30205` |
| Backlinks | "Linked from — checking…" / "— couldn't check" / "(0) — no notes link here yet" / "(N)" / "(50+)" | — | canvas `:605-631` |
| Location row (≥80 cols) | "In the Library database only — no file on disk", or "In a synced folder · <path> · <written>" | — | canvas `:236-273, 3702-3733` |

### 2.3 New note view, Trash, dialogs

| Surface | States / copy | Cite |
|---|---|---|
| New note view | Heading "New note"; rows "Blank note", "From a template…" (folded); template rows show "<Template>\n<resolved title>". Authority line "Ready · Next: Press Blank note, or choose a template." While creating: "Creating note…" with the rows disabled. On failure: "Create failed — …" | canvas `:1287-1303, 3750-3835`; screen `:30733-30782` |
| Trash | Header "Recently deleted", purpose "Deleted notes stay here until you restore them. Restore puts a note back where it was; nothing is removed for good from here." Rows "Title · age  [Restore]". Empty: "Nothing deleted recently. …". Over the cap: "Showing the 20 most recently deleted of N. Restore one to see the rest." | canvas `:2228-2293`; `library_notes_state.py:422` |
| Undo / Restore failures | Toasts "This deleted note changed elsewhere — refresh and try again.", "Could not restore this note.", "Note restore is unavailable." | controller `:6069-6118` |
| Folder name dialog | Title "New folder" / "Rename folder"; Input placeholder "Folder name"; [Cancel] [Save]. Save with an empty name does nothing. | folder dialog `:37-61` |
| Folder target dialog | Title "Add note to folder" / "Move note" / "Move <name>"; Select of loaded folders only (+ "Top level" for folder moves); [Cancel] [Choose]. Choose with no selection does nothing. | folder dialog `:77-148`; options `library_notes_controller.py:3250-3273` |
| Remove folder | Modal "Remove folder organization?" / "Remove <name> and its nested folder organization? Notes are not deleted; they remain in other folders or Unfiled." [Remove folder] [Cancel]. Afterwards "Restore folder" appears. | `library_screen.py:29997-30032` |

### 2.4 Add from files / lasting sync / Import once

| Phase | Copy / controls | Cite |
|---|---|---|
| Choose | "Add files to Library notes." + status. Text "Import once — Copy files into Notes, reproducing your folder structure…" [Import once]. Text "Keep a folder synced — Create a lasting connection…" [Keep a folder synced] or "○ Keep a folder synced" + "Keeping a folder synced isn't ready on this profile yet. Nearest valid action: Import once." Pointer "Neither? … switch the strip above to Folder files…". Bar: [‹ Notes] | add-from-files `:283-427, 1026-1032` |
| Configure | "Keep a folder synced"; "Chatbook watches only while running…"; Display name ("Folder label in Notes"); Folder ("No folder selected") [Choose folder…]; ☐ Obsidian vault (vault only; "Turning it off lasts until you quit Chatbook…"); Notes destination [Local Library notes (selected)] [○ Server notes] + reason "Unavailable — server sync-folder capability not installed"; Direction [⇄ Both ways] [→ Folder to Notes] [← Notes to folder] (✓ on the active one); validation line; bar [Check changes / ○ Check changes] [‹ Notes] | `:428-520, 1033-1051`; `library_notes_lasting_sync_state.py:166` |
| Checking / Activating | "◌ Checking. No unreviewed action is being hidden."; bar "Wait for the current step to finish."; footer "wait current step". **No cancel control.** | `:521-528, 1052-1058`; screen `:8434-8437` |
| Review | Summary "N safe · N need attention · N skipped · N folder moves"; "Syncs .md, .markdown and .txt only; other files are left alone."; stale "⚠ This review is stale. Check again before applying."; grouped rows; pager. Bar [Activate reviewed root] (disabled while attention > 0) or [Apply reviewed] (tooltip-only blocker reasons) / [Check again]; [○ Resolution history] + reason; [‹ Notes] | `:529-585, 1059-1182` |
| Conflict rows | [View comparison]; choices Keep file / Keep note / Keep both / Skip for now; "Choice staged. No changes yet." | `:700-830`; sync controller `:1591` |
| Receipt | e.g. "N applied · listed under Receipts" | `:586-594`; sync controller `:1358-1414` |
| Manage sync folders | Heading "Library notes · Manage sync folders"; per-root title **"Sync folder (name unavailable before cutover)"** (hard-coded); status "✓ Up to date · Next: Check changes", "◌ Changes available", "Ⅱ Paused", "⚠ Offline", "Ⅱ Open in another process", "⚠ Needs attention", "⚠ Sync stopped", "⚠ Partial", "✕ Failed", "✕ Blocked", "◌ Starting"; actions Check changes / Review / Review migration / Resume / Pause / Recovery / "○ Retarget unavailable — not in this release" / "○ Disconnect unavailable — not in this release"; Receipts list ("No writes yet. Sync writes are listed here when you open this list."); empty "No lasting sync folders. Nearest valid action: ‹ Notes." | roots canvas `:47-205`; labels `library_notes_sync_controller.py:249-280, 819` |
| Import once | Selection: "Choose a file or folder" / "Add another file", "Change selection", "Clear", "Notes destination" ("Existing or new folder path"), Obsidian vault line. Review groups: New, Unchanged repeat, Changed repeat, Uncertain match, Unsupported, Skipped, Empty, Failed. Collision: "Folder collision: <name>" [Use existing folder] [Create a unique sibling] [Use another name]. Per-row Skip / Create new / Update existing / Confirm this match / Replace note content / Add folder placement. Paging "Previous page / Page X of Y / Next page". Execute [Import selected items], [Cancel import] → "Stopping…". Receipt: "View N imported notes", "Retry N failures", "Skipped (N)". Toast on failure: "Notes import needs attention. Review the import panel and try again." | import canvas `:38-44, 600-1290`; controller `library_notes_controller.py:5056-5063` |

### 2.5 Folder files

| State | Copy | Cite |
|---|---|---|
| Unlinked | Link row "Choose a notes folder." [Details] [Choose folder…] **[Review recovered pairing…]**; purpose "Folder files edits Markdown files in a folder on disk, in place. Nothing is copied into the Library."; optional [Use <folder>] | workspace `:2116-2180, 314-318, 2964` |
| Linked | "Linked · Local folder: <folder>" / "Checking · …" / "Offline · …"; button relabels to "Change…" | `:3005-3051` |
| Folder change slow | "Changing folder… · N entries so far" [Cancel] [Keep waiting] [Choose another]; timeout copy | `:300-312, 2146-2165` |
| No file open | "No file open." / breadcrumb "No file selected" | `:224, 3062-3070` |
| Save states | Saved / Unsaved changes / Saving… / "Conflict — the disk file changed; your draft is preserved." (Next: Save Copy) / "Save failed — …" / "Read-only — the file cannot be edited…" / "Unavailable — the folder cannot be reached…" | `:200-230` |
| Delete | Two-press: "Activate Delete again to confirm." then "Deleted. Restore remains available."; breadcrumb "Recently deleted: <path>" | `:8421-8454, 3069` |
| Pairing review | Dialog "Review recovered File Notes pairing" …; errors surface as "Pairing review unavailable: {error}" / "Pairing was not approved: {error}. Close and review again." | `:6653-6727`; dialog `:47-185` |
| Session Git | "Review session changes"; "Repository: not checked"; "Up/Down select · Tab actions · Enter run · Esc back"; "No current-session Git changes."; [Trust and check status]; "Stage all (N)"; "Commit staged (N)"; "Review push (1 commit)…"; "Push 1 commit"; "Check remote again — no push"; "Back to Files — push continues" | git panel `:1203-1350, 1810-1830, 2293-2295` |

---

## 3. Controls and copy (as rendered)

### 3.1 Notes list toolbar (wide, nothing open)

Rendered order: **New** · **Sort: Newest** · **Select** / **Add from files…** · **Export** · [Manage sync folders] · [Last import] / heading **"Folders & placement"** / **New folder** · **○ Add to folder** · **○ Move note** · **○ Remove placement** · [Restore folder] + reason "Note actions unavailable — select a note in the list" (wide only) / status row / tree.

| Label | Behaviour | Notes | Cite |
|---|---|---|---|
| "Filter notes… (Enter)" | Filters on Enter. FTS5 **phrase** match on title + body, plus substring match on folder path. | Keywords are not indexed; there is no prefix match. See S-02. | canvas `:1776-1781`; screen `:24113-24275`; `DB/ChaChaNotes_DB.py:1096-1101`; `Notes/note_folder_repository.py:803-826`; `Utils/fts5_match_forms.py:368-391` |
| "Clear filter" | Clears the filter and restores the pre-filter browse receipt | Appears only while a filter is active | canvas `:2114-2120`; screen `:24088-24111` |
| "New" | Opens the **New note view** (chooser) | `n`/`Ctrl+N` instead create a blank note directly. Same verb, different result. | canvas `:1941, 1981-1988`; controller `:4953-4959` vs `:2966-2992` |
| "Sort: Newest/Oldest/Title" | Replaces the toolbar with a choice strip; re-pages the tree | Blocked while filtered | canvas `:1942-2027`; controller `:4916-4952` |
| "Select" / "Done" | Toggles select mode (flushes the open note first) | "○ Select" with 0 rendered notes | canvas `:1945-1979`; controller `:5648-5674` |
| "Select all N shown" / "All N" | Selects **rendered** note rows only | Notes in collapsed or unloaded folders are excluded | canvas `:1783-1795, 1839-1848` |
| "Export selected" / "Export" | Opens the Export bundle canvas for the selected ids | The only bulk action | screen `:30146-30158` |
| "Add from files…" | Opens the relationship chooser | Becomes "View import" while importing | canvas `:2038-2056`; controller `:5064-5090` |
| "Export" | Opens "Export bundle (.zip)" scoped to notes | No ellipsis; the guide says "Export…" | canvas `:2040`; controller `:5610-5621`; guide `notes.md:391` |
| "Manage sync folders" | Opens the roots canvas | Only when roots exist | canvas `:2057-2067` |
| "Last import" | Reopens the same-session import receipt | Session-only | canvas `:2068-2075` |
| "New folder" | Name dialog. Creates a **child of the selected folder**; with a note or nothing selected it creates at **top level**. | The dialog does not say where the folder will go | screen `:29903-29931` |
| "Rename" / "Move" / "Remove" (folder selected) | Name dialog / target dialog (loaded folders + "Top level") / confirm modal | Disabled with a tooltip on sync-managed or stale folders | canvas `:2575-2600`; screen `:29933-30028` |
| "Add to folder" / "Move note" / "Remove placement" (note selected) | Target dialog / target dialog / immediate detach (no confirm) | "Remove placement" is blocked for Unfiled ("Unfiled is shown automatically; move the note into a folder.") | canvas `:2601-2675`; screen `:30044-30115` |
| "Restore folder" | Restores the last removed folder | — | canvas `:2676-2685` |
| Folder row "▸/▾ Name" | Click/Enter **selects and toggles expansion** in one action | You cannot select a folder (for Rename) without toggling it | controller `:5622-5647` |
| Note row | Opens the editor (flushes first); in select mode toggles its check | Clicking also sets the placement selection | screen `:29750-29801` |
| "Undo" / "Dismiss" (receipt) | Restore via `restore_note` / drop the receipt | Focus parks on Undo | controller `:5874-5932, 6061-6159` |
| "Recently deleted (N)" | Trash view | Last row of the list | canvas `:2204-2226` |

Always-visible prose (wide only): "These notes live in the Library's own database — for notes that live in a folder on disk, switch to Folder files. To copy or keep a folder synced, choose Add from files." (canvas `:1754-1762`). List authority line: "Library notes · Ready · Next: Create a note or add from files." (canvas `:1325-1345`).

### 3.2 Editor controls

| Label | Behaviour | Cite |
|---|---|---|
| "‹ Notes" (wide) / "‹ Back to list" (compact) | Guarded exit to the list: flush, then veto toast if the save is blocked | canvas `:634-645, 2764-2775`; screen `:30164-30210` |
| "Edit" / "Preview" / "Info" | Region switch. Preview and Info take focus. | controller `:4185-4270` |
| "Save" | Explicit save; clears blank-GC protection so even an empty note is kept | controller `:3799-3815` |
| "Use in Console" | Stages the note in Console; **auto-links the note to the active workspace** if needed | controller `:4560-4656` |
| "Discard new note" / "Discard" | Only for an untouched new note; deletes it | canvas `:2862-2870`; controller `:6251-6331` |
| Title ("Untitled" placeholder for a blank seed), Keywords ("Comma-separated keywords"), Body | Autosave 2.0 s after the last change. Validation vetoes move focus (and for keywords, switch to **Info**). | canvas `:2871-2894`; `screen_constants.py:177`; controller `:3703-3783, 3848-3874, 3945-3978` |
| Chrome strip "N words · L:C" | Edit only, ≥80 cols | canvas `:204, 3637-3700` |
| Conflict "Overwrite" / "Reload" | Token-gated resolution | canvas `:3073-3097`; controller `:4076-4173` |
| Info ▸ Keywords (second field, same value) | Edits the same keyword draft | canvas `:2916-2924`; controller `:3764-3783` |
| Info ▸ "Linked from" rows | Open the linking note | canvas `:2689-2714`; screen `:29803-29810` |
| Info ▸ "Copy" | Clipboard as markdown; toast "Note copied to clipboard as markdown!" | controller `:4355-4414` |
| Info ▸ "Export Markdown" / "Export text" | FileSave "Export Note as Markdown/Text", **always opening at `Path.home()`**; FileSave default `can_overwrite=True` | controller `:4271-4334`; `Third_Party/textual_fspicker/file_save.py:37` |
| Info ▸ Danger "Delete" | Inline confirm in Info (the only visible delete path) | canvas `:2990-3006`; controller `:5713-5804` |
| Hidden or dead controls | `#library-note-wide-utilities` (Export/Copy/Delete duplicates) is always `display=False`; `#library-note-context-use-in-console` is hidden | canvas `:2955-2964, 3031-3071, 3546` |

### 3.3 Copy that uses internal or implementation jargon (inventory only)

The following strings are rendered in the UI, with their source:

- "placement", "Remove placement", "Folders & placement", "Add folder placement" (canvas `:118, 2629-2650`; import canvas `:1206`)
- "Unfiled is shown automatically; move the note into a folder." (canvas `:2647-2648`)
- "! Needs owner review" (tree state `:718, 1004`)
- "N placements loaded · May be out of date", "Tree changed · Refreshing…" (tree state `:829, 858`)
- "Remove folder organization?" / "nested folder organization" (screen `:30009-30013`)
- "Lasting sync" prefix and "Next: Review the current lasting-sync step." (canvas `:1308-1324`)
- "Configure a local lasting sync root.", "Activate reviewed root", "Activating the reviewed sync root…", "Sync root activated." (sync controller `:964, 2203, 2243`; add-from-files `:1073`)
- "Sync folder (name unavailable before cutover)" (sync controller `:819`)
- "Nearest valid action: …" (add-from-files `:410`; roots canvas `:63`)
- "◌ Checking. No unreviewed action is being hidden." (add-from-files `:524`)
- "Recovery reviewed. Check changes before the next mutation." (sync controller `:1542`)
- "Ⅱ Open in another process", "Next: Open active process", "Finish upgrade", "Review settings" (sync controller `:249-280`)
- "Create a unique sibling", "Changed repeat", "Unchanged repeat", "Uncertain match" (import canvas `:38-44, 651`)
- "Review recovered File Notes pairing", "Historical managed memberships remain inactive; approval does not transfer them or replay old filesystem intents." (recovery dialog `:47-90`)
- "Read-only — this note cannot be changed; your draft is preserved." shown during **select mode** (canvas `:819-821`)
- "Next: Review recovery" during a conflict, while the buttons say Overwrite/Reload (canvas `:811-813`)

---

## 4. Key bindings

### 4.1 Library notes (screen `BINDINGS`, `library_screen.py:1010-1222`; gates in `check_action` `:25303-25365`)

| Key | Action | Active when | Advertised | Cite |
|---|---|---|---|---|
| `n` | Create a blank note and open it (no chooser) | Notes navigator, editor, preview or Info with no text field focused; also the Library landing | Footer "n new note" (list) | on_key `:8940-8955`; gate `:25313-25327`; controller `:2966-2992` |
| `Ctrl+N` | Same | Same gate, and it also works from inside the filter Input. **Inert in Folder files mode** (`visible_notes` is False). | Not on the list footer (`n` replaced it) | `:1030, 25303-25327` |
| `/` | Focus and select the notes filter | Navigator, no text field focused | "/ find note" | `:1031`; on_key `:8860-8905`; gate `:25328-25333` |
| `g` | Focus the selected folder row, else the first folder row | Navigator, folder rows rendered | "g go to folder" (dropped when there are no folder rows) | `:1039, 8651-8668, 25337-25343` |
| `e` | Press "Export selected" | Select mode with ≥1 checked | "e export selected" (select tier) | `:1040-1042, 8670-8683, 25344-25351` |
| `Esc` | Chain: delete-confirm → cancel; conflict → refuse with toast; Info → editor; select mode → exit; trash → list; editor/preview → guarded exit to list; create → list; sync/import → back (or cancel a running import); sort strip → close; list → focus rail search | Notes workflow | Tier-specific chips | controller `:3009-3173` |
| `r` | Restore the focused Trash row | Trash view with rows | "r restore note" | `:1221`; screen `:30566-30584`; gate `:25384-25395` |
| `Tab` / `Shift+Tab` | Cycle inside `#screen-content`. Note fields use priority bindings. The delete confirm traps Tab between Cancel and Delete. | Always | Hidden | `:1020-1021`; canvas `:677-680`; on_key `:8830-8850` |
| `F6` / `Shift+F6` | Next/previous workbench pane | Always | "F6 next pane" (Folder files tier) | `:1022-1028, 8956-8969` |
| `Ctrl+End` / `Ctrl+Home` | Caret to end/start of the body | Body focused | "ctrl+end end of note" (only while the body is focused) | canvas `:684-688, 769-777`; screen `:8403-8417` |
| `↑` / `↓` | Move between **`.library-notes-row`** siblings and New-note view rows only | A note row or create row focused | Hidden | `UI/Library_Modules/canvas_sync.py:61-107`; `screen_constants.py:434-450` |
| `Enter` | Press the focused button (row open, toggle, action) | Buttons | Footer names the focused control ("enter save note", "enter delete note", "enter undo delete"…) | screen `:8400-8470` |
| `i` | Open Library Import (Media ingest) from any canvas | No text field focused | "i" on hub tiers | on_key `:8929-8940` |
| **`Ctrl+S`** | **Unbound in Notes.** Bound to "Save skill", gated to the Skills editor. `action_library_notes_save` exists with a gate but **no binding**. | — | Guide states "Notes does not use Ctrl+S" | `:1047, 8688-8719, 25352-25357`; guide `notes.md:493-496` |
| (absent) | No key for: Save, Delete, Use in Console, Edit/Preview/Info switch, enter select mode (Media uses `s`), toggle a row (Media uses Space), new folder, rename, move, collapse/expand (←/→), next/previous note | — | — | `BINDINGS` `:1010-1222` |

Footer tiers (`library_screen.py:1547-1600`, logic `:8279-8469`):

- **List:** "n new note \| / find note \| g go to folder \| esc focus rail". Compact: "n new \| / find \| g folder \| esc rail".
- **Select mode:** "enter select note \| e export selected \| esc done".
- **Sort strip:** "enter choose sort \| esc cancel".
- **Editor:** "esc back to notes \| ctrl+end end of note" plus a focus chip.
- **Preview:** "pgup/pgdn scroll \| esc back to notes".
- **Info:** "enter <focused action> \| esc back to note".
- **Conflict:** "enter choose version".
- **Delete confirm:** "enter cancel/delete \| tab switch button \| esc cancel".
- **Create:** "enter create note \| esc back to notes".
- **Lasting:** "enter <focused> \| esc back to notes"; during checking or activating, "wait current step".
- **Trash:** "r restore note \| esc back to notes".
- **Import:** "esc back to notes"; while running, "esc cancel".

### 4.2 Folder files / Session Git

| Key | Action | Cite |
|---|---|---|
| `Esc` (file editor) | To the Files tree ("esc files"); a second Esc leaves for Library notes ("esc notes") | workspace `:344-345, 6606`; screen `:1058, 22106` |
| `Ctrl+End` / `Ctrl+Home` | End/start of file | workspace `:335-337` |
| `/`, `F6` | Focus search, next pane | screen `:1281-1284` |
| Session Git: `↑`/`↓`, `Tab`, `Enter`, `Esc` | Select row / step into actions / run / step back | git panel `:655-656` |
| Search | **Live, as you type** (`Input.Changed`). Contrast with Library notes' Enter-to-filter. | workspace `:6519-6520` |
| `Ctrl+N` / `n` | Inert (gated to the Database source) | screen `:25303-25327` |

---

## 5. Flows

Anchors are DOM ids, so a live tester can drive each step.

### 5.1 First-timer flows

**F1. Arrive with zero notes and create the first one**

1. `Ctrl+3` (or the nav "Library"). The rail is in the starter shape (Import…, New note, Explore all tools); there is no Notes row (guide `notes.md:21-29`). The hub shows Get started (Import a file → Find it → Use it in Console) and the quick actions "Import…", "New note" (`library_entry_canvases.py:349-397`).
2. Any of three routes:
   - **(a)** press `n` on the landing. A blank note is created and the editor opens (`library_screen.py:8948-8955`; controller `:2966-2992`).
   - **(b)** click hub/rail "New note". The New note view opens with focus on "Blank note"; Enter creates (canvas `:3750-3835`).
   - **(c)** "Explore all tools" → Browse ▸ "Notes (0)" → empty list "No notes yet. Create your first note." + Agent_Lessons gloss → "New" → New note view → "Blank note".
3. The editor opens. The title is empty with an "Untitled" placeholder; status "Empty note — type to keep it"; authority "Next: Start typing." (controller `:1441-1442`).
4. Type the title, Tab to Keywords, Tab to Body. Autosave fires 2 s after the last change → "Saving…" → "Saved HH:MM". A rename patches the list row live (controller `:3891-3914`).
5. Leave with "‹ Notes", `Esc`, or the rail. A note left completely untouched is silently garbage-collected. A whitespace-only one is discarded with "Empty note discarded" (controller `:3979-4075`; guide `notes.md:594-611`).

Risk points: a trailing-space title vetoes the save (S-01); the veto toast names "Discard new note" (S-05).

**F2. Create from a template**

New note view → "From a template…" (expands in place; focus is preserved, canvas `:1514-1521`) → ↑/↓ to e.g. "Meeting notes / Meeting Notes - 2026-10-02" → Enter. The note is created pre-filled (title, body, keywords) and opens in the editor (controller `:6201-6232`). There is no way to author a template in-app; templates come from `note_templates.json` and are read once at import time (`Event_Handlers/notes_events.py:54-132`).

**F3. Bring existing notes in: Import once (e.g. an Obsidian vault)**

1. In the Notes list, "Add from files…" (`#library-notes-add-from-files`). The chooser opens with focus on the first safe control (canvas `:3125-3137`).
2. Press "Import once" (`#notes-add-import-once`). The file-or-folder picker opens immediately (controller `:5091-5107`), at the last Import-once directory or home.
3. Pick a folder (or "Add another file" for individual files, plus a destination path). The confirmation shows the full path, and a vault line ("Obsidian vault · N notes · skips …") if `.obsidian/` is present.
4. "Check selection" opens a read-only review grouped by outcome, paged, with collapsed runs and per-row and per-group actions. If the root name collides, the default is a unique sibling.
5. "Import selected items" runs with progress; "Cancel import" stops cooperatively. The receipt follows ("N notes created · … links rewritten"). "View N imported notes" opens the list.
6. Back in the list, "Last import" reopens the receipt within this session only.

Discoverability gap: the rail/hub "Import…" and `i` go to **Media** ingest, which has no pointer to Notes (S-14).

**F4. Keep a folder synced (lasting)**

"Add from files…" → "Keep a folder synced" → Configure: Display name, "Choose folder…" (folder-only picker, `library_notes_controller.py:5115-5150`), Obsidian toggle (vault only, not persisted), Direction → "Check changes" (read-only; refusals name a rule) → Review → "Activate reviewed root" (disabled while attention > 0, with no reason on screen) → receipt → "‹ Notes". The list then shows a "⇄ Sync managed" folder and a "Manage sync folders" toolbar action (guide `notes.md:1125-1157`).

**F5. Edit a folder in place (Folder files)**

Source strip "Folder files" (`#library-notes-source-files`) → "Opening File Notes…" → "Choose a notes folder." [Choose folder…] → SelectDirectory "Choose File Notes Folder" (the path field is focused and pre-filled) → "Linked · Local folder: …" → Files tree → open a file → edit (autosave; there is no Save button) → `Esc` → tree → `Esc` → Library notes (workspace `:2116-2180, 6732-6750`).

**F6. Delete and recover**

Open the note → "Info" → Tab to "Delete" → Enter → inline "Delete this note? …" (focus on Cancel) → Tab → "Delete" → Enter. You return to the list with "✓ deleted · <title>" and focus on "Undo"; Enter restores it, and the row returns in its folder (controller `:5713-5906, 6061-6159`). Later recovery: "Recently deleted (N)" → Restore / `r` (20 most recent only; no permanent delete).

**F7. Use a note in Console**

Editor → "Use in Console" (`#library-note-use-in-console`). If the note is not in the active workspace it is linked first. The hand-off stages the note with the prompt "Use this note as context and help me work with it." and navigates to Console. The status line records "Use in Console complete — Linked to <ws> · staged in Console." (controller `:4415-4656`). Return leg: in Console, an assistant message → More… → "Capture as note" → "Saved to Notes" receipt → "Open note" (`Chat/console_message_actions.py:398`; `UI/Console_Modules/message.py:2233`).

### 5.2 Power-user flows

**P1. Keyboard-only rapid capture**

`Ctrl+N` (from anywhere in Notes, including the filter) → type the title → `Tab` → keywords → `Tab` → body → `Esc` (guarded exit; autosave has flushed) → `Ctrl+N` again. There is no Ctrl+S (S-09). A trailing-space title blocks `Esc` with a toast (S-01, S-05). `Ctrl+N` while a note is open flushes it, then creates (a flush refusal keeps you in place; controller `:2989-2992`).

**P2. Find and open**

`/` (from the list) → type → `Enter` (filter; FTS phrase, no prefix, no keywords, S-02) → `Tab`… to the first note row, or click → `↑`/`↓` walk note rows only (folder rows skipped, S-08) → `Enter` opens. Quick-open by title from elsewhere: the rail "Search Library…" runs Search/RAG, which is a different surface. The command palette has no "Library — Notes" command (S-19).

**P3. Link notes (wikilinks / backlinks)**

- **Authoring:** no in-editor link insertion or autocomplete. Hand-typed `[[Title]]` is plain text. Only `[[…]](note://<id>)` counts, and note ids are not displayed anywhere: Info shows Created, Modified, Version and Words only (`library_notes_state.py:846-866`).
- **Import once** rewrites `[[wikilinks]]` within the same batch.
- **Reading:** Preview renders links as `[label](note://id)` (canvas `:440-467`), using Textual's default `open_links=True` with no `LinkClicked` handler, so a click goes to `app.open_url("note://…")`, the OS handler (S-03).
- **Backlinks:** Info ▸ "Linked from (N)" rows open the linking note (canvas `:605-631, 2689-2714`).

**P4. Organize at scale**

Folders are created, renamed and moved through modal dialogs. Note placements are added, moved and removed with "Add to folder", "Move note" and "Remove placement" (a note can sit in multiple folders). The target picker lists **only already-loaded folders** (S-18). Keywords are editable per note but there is no keyword browser, facet or keyword filter (S-02). There is no drag and drop, and no keyboard shortcuts for any of these operations.

**P5. Bulk operations**

"Select" (Tab to it; no `s` key) → Enter on rows to toggle (no Space) → "Select all N shown" (rendered rows only) → `e` / "Export selected" → Export bundle canvas (.zip). There is **no bulk delete, move or keyword edit** (guide `notes.md:1187-1191` and canvas `:1796-1881` agree).

**P6. Vault sync at scale**

"Manage sync folders" → per-root "Check changes" → "Manual check finished. N changes to review." → Review (paged, grouped, collapsed runs) → per-conflict "View comparison" and Keep file / Keep note / Keep both / Skip for now → "Apply reviewed" → receipts and "Resolution history" (undo for up to 30 days). Constraints:

- every root is titled identically, "Sync folder (name unavailable before cutover)" (sync controller `:819`);
- "Retarget" and "Disconnect" are unavailable in this release;
- only edits made in the Library editor push note→file automatically, not creates or agent writes (guide `notes.md:802-822`);
- the row status does not repaint live (guide `notes.md:794-796`).

**P7. Folder files + Session Git**

Folder files → Manage → "Review session changes (N)" → "Trust and check status" (modal; Cancel focused) → Stage / "Stage all (N)" → "Commit staged (N)" → Subject/Body → "Review commit" → "Confirm commit" → "Review push (1 commit)…" → "Authorize configured destination" dialog → "Authorize and check" → "Push 1 commit". Keyboard: Up/Down, Tab, Enter, Esc (git panel `:1203-1350, 1810-1830`; guide `file-notes.md:269-420`).

**P8. Export**

- **Single note:** Info → "Export Markdown" / "Export text" → FileSave dialog (always opens at home; silent overwrite possible, S-17).
- **Many notes:** list "Export" or select mode → Export bundle (.zip, chatbook format).
- There is no "export notes as Markdown files into a folder" path (`library_notes_controller.py:4271-4334, 5610-5621`).

---

## 6. Cross-links

| From Notes | To | Mechanism | Cite |
|---|---|---|---|
| Editor "Use in Console" | Console (staged context) + Workspaces (membership write) | `open_chat_with_handoff`; `registry.link_membership` | controller `:4453-4656` |
| Console message "Capture as note" | Library notes ("Saved to Notes" → Open note) | Console message actions | `Chat/console_message_actions.py:398`; `UI/Console_Modules/message.py:2233` |
| Console / MCP agents | `library_search_notes`, `library_save_note`, Agent_Lessons folder | Agent tools | guide `notes.md:221-311`; `Notes/agent_lessons.py:23` |
| Notes list "Export" / "Export selected" | Library Export canvas (bundle .zip) | `_open_library_export_canvas(ExportScope("notes"…))` | controller `:5610-5621`; screen `:30146-30158` |
| Rail "Search Library…" | Search/RAG canvas (all sources) | shared query state | screen `:34037-34076` |
| Rail/hub "Import…", key `i` | **Media** Ingest canvas (not Notes import) | row switch | screen `:8929-8940`; `Widgets/Library/library_ingest_canvas.py` (no Notes pointer) |
| Settings ▸ Manual Sync | Sync-v2 organization (folders, keywords) adoption review | Settings | guide `notes.md:195-219` |
| Legacy route "notes", default tab "notes" | Library (generic; no `mode: notes` context) | `_SCREEN_ALIASES`, `_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` | `UI/Navigation/screen_registry.py:223-227`; `app.py:2442-2449` |
| `NavigateToScreen` with `{mode:"notes"}` / `note_id` | Notes row / a specific note | nav context | screen `:11188-11235` |
| Study screen | `notes_scope_models.WorkspaceSubview` (vestigial model reuse) | import | `UI/Screens/study_screen.py:41` |
| Research workspace | separate "Quick Notes" section (not Library notes UI) | — | `UI/Research_Workspace_Modules/quick_notes_section.py` |

---

## 7. Suspected issues with live probes

All of these are **suspected**; none has been run live. Severity is a guess. Each probe assumes an isolated profile, at 235x52 and at 100x30, unless stated.

**S-01: An autosave validation veto steals focus, and for keywords switches the pane to Info (High)**
- **Evidence:** Autosave fires 2.0 s after the last keystroke (`screen_constants.py:177`). A veto (title leading or trailing whitespace; a case-insensitive duplicate keyword) returns `VALIDATION_VETO` even on autosave (`Library/library_notes_session.py:582-597`; `library_notes_state.py:194-198, 263-269`). `_apply_library_note_save_outcome` then always calls `_route_library_note_validation_field` plus `_focus_library_note_validation_field` (controller `:3945-3978`). For `"keywords"` that sets `_library_note_context=True` and focuses `#library-note-context-keywords` in **Info**. The comment there says keywords "belong to Info at every width", which task-32642 superseded by moving keywords into Edit (controller `:3865-3874`; canvas `:2879-2892`).
- **Probe A:** `Ctrl+N` → type `Groceries ` (trailing space) → `Tab` `Tab` → type `milk` → wait 3 s → type `eggs`. Check where "eggs" lands, and whether focus jumped to Title.
- **Probe B:** open a note → in the Edit pane's Keywords field type `ai, AI` → wait 3 s. Does the pane flip to Info with focus in Info's keyword box?

**S-02: The notes filter cannot match keywords, partial words or reordered words (High)**
- **Evidence:** `notes_fts` indexes only `title,content` (`DB/ChaChaNotes_DB.py:1096-1101`). The filter uses `build_phrase_match_query`, which produces one quoted phrase with no `*` prefix (`Utils/fts5_match_forms.py:368-391`; `Notes/note_folder_repository.py:808-826`). No keyword facet exists anywhere in the canvas. The guide claims "filtering the list on `conversation:` finds every answer kept from one chat" (`notes.md:1266-1268`), but those are keywords.
- **Probe:** create note "Meeting notes" with keyword `zebra` and body "budget review". Filter `meet` + Enter; then `zebra`; then `review budget`; then `Meeting`. Expected if confirmed: only `Meeting` returns a result. Also capture a Console answer and filter `conversation:`.

**S-03: Wikilinks are not navigable in-app, and Preview hands `note://` to the OS (High)**
- **Evidence:** Preview's `Markdown(...)` is built without `open_links=False` (canvas `:2909-2913`). Textual 8.2.8's `Markdown.on_markdown_link_clicked` calls `self.app.open_url(event.href)`, and there is no `open_url` override and no `LinkClicked` handler in Library code (grep confirmed). Hand-typed `[[Title]]` is not a link, and note ids are never displayed (Info properties at `library_notes_state.py:846-866`).
- **Probe:** Import once a two-file vault where `A.md` contains `[[B]]`. Open A → Preview → click "B". Observe whether a browser or OS dialog opens, or nothing happens, and whether B opens in-app. Then in a new note type `[[B]]`, wait for the save, open B → Info → read "Linked from (N)".

**S-04: The filter result count disagrees with the tree's total (Medium)**
- **Evidence:** The status line counts `len(rows)` of `filter_records`, which is the loaded placement window of up to 20 per page (`library_notes_state.py:639-644`; screen `:24270-24272`). The filtered tree pager shows "Notes 1–20 of <total>" (tree state `:810-816, 1200-1219`). The guide says the result count is the exact total (`notes.md:370, 381`). In the window before results arrive, `filter_records` is `None`, so the count falls back to all local notes (screen `:17101-17105`).
- **Probe:** create 25+ notes containing "alpha" (e.g. via Import once of a small folder). Filter `alpha`. Read the status line against the pager row, then press "Load more notes" and re-read.

**S-05: The "Can't leave yet" toast names the wrong field or a control that is not on screen (Medium)**
- **Evidence:** Every `VALIDATION_VETO` produces "Can't leave yet — fix the title or press Discard new note.", whatever the field (body control characters, keyword duplicates) and whether or not Discard is displayed (`library_screen.py:909-931`). Discard is only shown for an untouched new note (canvas `:2862-2870`; controller `:1513-1515`). FAILED says "press Save to retry or Discard".
- **Probe:** open an existing note → Keywords `x, X` → `Esc`. Read the toast; check whether a "Discard new note" button exists.

**S-06: Conflicting or ill-matched "Next:" guidance and status copy in the editor (Medium)**
- **Evidence:** The work-pane authority line and the header status both carry status and "Next:" (canvas `:1249-1286` versus `:2819-2824, 3445-3454`).
  - Conflict: header "… Next: Review recovery." (no such control) versus authority "Next: Resolve the conflict or reload the note." (canvas `:811-813, 1259-1260`).
  - Validation: "… remove it to save. Next: Retry Save." (canvas `:822-824`).
  - Select mode: "Read-only — this note cannot be changed; your draft is preserved. Next: Keep the draft." (canvas `:819-821`; controller `:1476`).
- **Probe:** create a conflict (S-07 probe), read both lines; set a trailing-space title, read both; open a note, press "Select" in the list, read the editor header.

**S-07: The "changed elsewhere" conflict banner is deferred while you type (Medium)**
- **Evidence:** `sync_state` skips the recompose while an editor field has focus; the comment records it as a known ceiling: "a banner … waits for the next refresh that arrives with the reader's hands off the field" (canvas `:1661-1684`).
- **Probe:** put a note under lasting sync. Open it and keep typing in the body while editing the backing `.md` on disk, or have a Console agent `library_save_note` the same note. Observe whether the conflict callout appears before you leave the field, and what happens to autosave.

**S-08: Arrow keys skip folder rows, pagers and "Recently deleted"; `g` lands where arrows do nothing (Medium)**
- **Evidence:** `_move_library_list_row_focus` only claims rows whose class is in `_LIBRARY_LIST_ROW_CLASSES` (`screen_constants.py:434-450`), which has `library-notes-row` but not `library-notes-folder-row`, `library-notes-tree-pager` or the trash opener (`canvas_sync.py:87-107`). `g` focuses a folder row (screen `:8651-8668`). There is no ←/→ expand or collapse.
- **Probe:** with two folders each holding notes, press `g` then `↓`. Then focus a note row and press `↓` past a folder boundary to check whether the folder row is skipped. Try `→` on a folder row.

**S-09: No Ctrl+S in the Notes editor, while the same screen binds it elsewhere (Medium)**
- **Evidence:** `ctrl+s` is bound only to "Save skill" (`library_screen.py:1047`). `action_library_notes_save` has a gate (`:25352-25357`) but no binding (`:8688-8719`), so it is unreachable. The Import picker uses Ctrl+S for "Select this folder" (guide `notes.md:1117`).
- **Probe:** type in a note body, press `Ctrl+S` immediately, and watch the status ("Unsaved changes" persists until autosave?). Repeat in a Skill editor for contrast.

**S-10: "New" and `n` share a verb but do different things (Medium)**
- **Evidence:** Toolbar "New" opens the New note chooser (controller `:4953-4959`). The footer chip "n new note" creates a blank note directly (`:2966-2992`). The rail and hub "New note" also open the chooser.
- **Probe:** from the list press `n` and note the result; return; Tab to "New" and press Enter; compare.

**S-11: "Use in Console" silently writes a workspace membership that nothing in the UI can remove (Medium)**
- **Evidence:** The hand-off links the note into the active workspace when the gate is closed (controller `:4594-4616`). The docstring states "No surface in the app removes an item_type='note' membership — Notes has no unlink affordance" (`:4515-4523`).
- **Probe:** new note → "Use in Console" → read the status line ("Linked to <ws>") → open the Workspaces surface and look for the note and any unlink control.

**S-12: Manage sync folders shows identical root names and dead-end "Next" actions (Medium)**
- **Evidence:** The root title is hard-coded "Sync folder (name unavailable before cutover)" (sync controller `:819`; guide `notes.md:770-777`). For an offline or unreadable root the next action is `reconnect_folder` → label "Reconnect folder", but the primary control is `retarget`, rendered "○ Retarget unavailable — not in this release" (roots canvas `:189-201, 206-216`; `library_notes_lasting_sync_state.py:1126-1128`). "Next: Open active process", "Finish upgrade" and "Review settings" have no matching control. "Next: Resolve recovery" sits beside a button labelled "Recovery" (sync controller `:265-280`; roots canvas `:183-187`).
- **Probe:** activate two roots with different display names → "Manage sync folders" → compare the titles. Then rename or move one root folder on disk → "Check changes" → read the row and its first (primary) control.

**S-13: Keep-synced setup has no cancel during Check, and Activate's blocked reason is not on screen (Medium)**
- **Evidence:** During `checking`/`activating` the bar only says "Wait for the current step to finish." (add-from-files `:1052-1058`). "Activate reviewed root" is `disabled=attention_count > 0` with no tooltip or reason line (`:1071-1080`). Apply's blockers are tooltip-only (`:1083-1093, 1159-1182`).
- **Probe:** choose a large folder (thousands of `.md` files) → "Check changes" → look for a cancel. Use a folder whose review yields ≥1 "need attention" row → read why Activate is "○" and how to proceed.

**S-14: A first-timer importing Markdown via "Import…" lands in Media, not Notes (Medium)**
- **Evidence:** Rail and hub "Import…" and `i` open `LIBRARY_ROW_INGEST_MEDIA` (screen `:8929-8940`; `library_entry_canvases.py:349-356`). The Ingest canvas never mentions Notes or "Add from files…" (grep of `library_ingest_canvas.py`). Notes import is only behind the Notes list's "Add from files…".
- **Probe:** fresh profile → press "Import…" → look for any notes or Obsidian path → ingest a `.md` → check whether it appears under Media or Notes.

**S-15: The Notes Trash is capped at 20 with no paging, and notes can never be permanently deleted (Medium)**
- **Evidence:** Page size 20 (`library_notes_state.py:422`); "Showing the 20 most recently deleted of N. Restore one to see the rest." (canvas `:2287-2293`). Restore is the only action, by design (canvas `:2228-2237`); Media's trash has "Delete forever" (`x`) (screen `:1215-1216`). No purge path in the Notes controller or service (grep).
- **Probe:** delete 21 notes → open "Recently deleted (21)" → try to reach the oldest; look for a permanent delete or empty-trash action.

**S-16: Bulk work is export-only, and "Select all N shown" silently excludes collapsed folders (Medium)**
- **Evidence:** The select strip offers Done, Select all, Clear and Export selected only (canvas `:1796-1881`). "Select all" counts rendered tree rows only (`:1783-1795`). There is no notes key for select mode or row toggle; Media's `s`/Space are media-gated (screen `:1158-1178`).
- **Probe:** with three folders (two collapsed) → "Select" → "Select all N shown" → compare N with the "Notes (N)" header → `e` → check what the export includes.

**S-17: Single-note export always opens at home and may overwrite without asking (Low-Medium)**
- **Evidence:** `FileSave(location=str(Path.home()) …)` (controller `:4318-4323`). The FileSave default is `can_overwrite=True` (`Third_Party/textual_fspicker/file_save.py:37`). The write is a plain `write_text` (screen `:21160-21166`). Contrast the remembered directories for Import and Sync (guide `notes.md:1335-1342`).
- **Probe:** Info → "Export Markdown" → save to `~/x/note.md`. Repeat with the same name: is there an overwrite prompt? Does the dialog reopen at `~/x`?

**S-18: Folder pickers list only loaded folders; New-folder placement is implicit; selecting toggles expansion (Low-Medium)**
- **Evidence:** Target options come from loaded `tree_branches` only (controller `:3250-3273`; dialog docstring `library_note_folder_dialog.py:78`). "New folder" uses the selected folder as parent, otherwise top level, and the dialog title is just "New folder" (screen `:29903-29931`). A folder row press both selects and toggles (controller `:5622-5647`). "Choose" with nothing picked does nothing (folder dialog `:133-138`).
- **Probe:** create more than 20 root folders, or nested folders inside a collapsed parent → select a note → "Add to folder" → check whether every folder is listed. Select a note in folder X → "New folder" → see where it is created. Click a folder to select it for "Rename" and watch it expand or collapse.

**S-19: No direct route to Notes from the command palette or the legacy "notes" route (Low)**
- **Evidence:** `_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` has artifacts, prompts, skills, search and media but **not notes** (`app.py:2442-2449`). The palette's Library deep links are Artifacts and Skills only (`app_command_providers.py:322-338`).
- **Probe:** `Ctrl+P`, type "notes": is there a command landing on the Notes list? In the isolated config set the default tab to `notes`, launch, and check whether you land on the hub or the Notes list.

**S-20: Folder files: "Review recovered pairing…" is always visible, its dialog is heavy with jargon, and raw errors are shown (Low)**
- **Evidence:** The button is yielded unconditionally on the link row and never display-toggled (workspace `:2143-2145`; no other references). The User Guide's link-row description omits it (`file-notes.md:70-82`). Errors are interpolated raw: "Pairing review unavailable: {error}" (`:6725`), "Pairing was not approved: {error}" (dialog `:184`).
- **Probe:** Folder files with no folder linked → read the link row at 100x30 and 235x52 → press "Review recovered pairing…".

**S-21: The active source switch is shown by bold text alone (Low)**
- **Evidence:** `#library-notes-source-strip Button.-selected { text-style: bold; }`, with transparent backgrounds and no glyph (`css/widget_defaults_scoped.tcss:864-876`). Contrast the "✓" markers on the Sort and Direction strips.
- **Probe:** at 235x52 and 100x30 in a low-contrast theme, can you tell whether "Library notes" or "Folder files" is active?

**S-22: Two search idioms side by side (Low)**
- **Evidence:** The Library notes filter applies on Enter ("Filter notes… (Enter)", screen `:24113`). Folder files search is live (`workspace:6519-6520`). The rail "Search Library…" (Search/RAG) is visible beside the notes filter.
- **Probe:** type in each without pressing Enter; note which one updates. Press `/` from the list and see which box gets focus; press `Esc` from the list and see where the caret goes (rail search).

**S-23: Delete is deep and keyless; the Edit-pane Delete is a hidden dead control (Low)**
- **Evidence:** Delete exists only in Info ▸ Danger; the Edit-pane duplicate container is always `display=False` (canvas `:3031-3071, 3546`). No delete key exists.
- **Probe:** count the keystrokes to delete an open note by keyboard only, from Edit.

**S-24: The new-note load timeout and "Edit note" loading heading (Low)**
- **Evidence:** A 3.0 s hard deadline produces "Unable to load note — timed out after 3 s. Press Retry." (`screen_constants.py:290-297`). The loading heading says "Edit note" instead of the title (canvas `:1717-1721`).
- **Probe:** open a ~2M-character note on a slow disk, or under CPU load; watch for a spurious timeout.

**S-25: Keep-synced setup contains dead or single-option controls (Low)**
- **Evidence:** "Local Library notes (selected)" is a button with no alternative, and "○ Server notes" is permanently disabled (add-from-files `:477-498`). The Obsidian toggle is not persisted across restart and is not editable after activation (`:463-476`; guide `notes.md:1060-1071`).
- **Probe:** Tab through Configure and count the stops that do nothing; restart and re-open setup for a vault.

**S-26: Escape semantics differ across sibling surfaces (Low)**
- **Evidence:**
  - Library notes editor: Esc goes to the list in one press; Info → editor first (controller `:3038-3083`).
  - Folder files: Esc goes to the Files tree, then to Library notes (workspace `:344-345`).
  - Notes list: Esc focuses the rail search box, not the previous location (controller `:3116-3166`).
  - Delete confirm: Esc cancels.
- **Probe:** walk Esc from each surface and note where focus lands each time.

**S-27: Templates are fixed at startup and cannot be created in the UI (Low)**
- **Evidence:** `NOTE_TEMPLATES = load_note_templates()` runs at module import, reading `note_templates.json` from the config directory or app defaults (`Event_Handlers/notes_events.py:54-132`). There is no "save as template" in the canvas.
- **Probe:** edit `note_templates.json` in the isolated config while the app runs, then reopen "From a template…" and check whether the change shows. Look for a template-authoring path.

**S-28: The compact layout hides placement actions until a note has been opened (Low)**
- **Evidence:** In compact mode with nothing selected only "New folder" is composed, with no heading and no reason line (canvas `:2455-2466, 2614-2622, 2510-2511`). Selecting a note in compact mode opens it on the full stage.
- **Probe:** at 100x30, look for "Add to folder" on the list → open a note → "‹ Back to list" → check whether the actions appear.

**S-29: The User Guide has drifted from the code and is hard to scan (Low; docs)**
- **Evidence:**
  - "Export…" in the guide versus "Export" in code (`notes.md:391` vs canvas `:2040`).
  - A duplicated sentence (`notes.md:873-874`).
  - Two different "3." steps, one stale (`notes.md:1131-1143`).
  - The filter-count and keyword-filter claims (S-02, S-04).
  - The Folder files link row omits "Review recovered pairing…".
  - `notes.md` is 2,353 lines, of which ~957 (`:1396-2353`) are "Verified against" stamps, despite CLAUDE.md now forbidding them.
- **Probe:** read the guide sections against the live screens during the walk.

**S-30: Copy uses implementation terms throughout (Low)**
- **Evidence:** The inventory in §3.3: "placement", "root", "cutover", "lasting", "Nearest valid action", "No unreviewed action is being hidden", "mutation", "Create a unique sibling", "Needs owner review", "pairing", "Ⅱ Open in another process".
- **Probe:** a first-timer reads each surface aloud and states what each term means and what they would press next.
