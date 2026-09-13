# Library ▸ Notes — Wave 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Each Task below is one branch, one worktree, one implementer, one task review, then a landing pass. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every open finding from the second Library ▸ Notes critique (20/40 at dev `e6cb464239`, snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md`), the test-health riders that wave 2 filed, the two idea tasks the user approved, and the backlinks persistence follow-up — then re-run the critique for a third like-for-like score.

**Architecture:** Eleven independent branches off `dev`, grouped by surface so no two branches edit the same function. Each group's backlog task files ARE the requirements: every task carries a `## Description` with the PROVEN or INFERRED cause and outcome-shaped `## Acceptance Criteria`; the implementer reads the task files first and the plan section second. Groups land one at a time under dev's strict protection; the docs sweep (Task 11) lands last and re-verifies every guide claim the wave touched.

**Tech Stack:** Python 3.12, Textual 8.x, SQLite/FTS5, pytest (foreground, chunked), tmux + SGR mouse injection for live verification, Backlog.md task files.

**Spec:** the critique snapshot above plus each group's task files under `backlog/tasks/` on `dev` (ids listed per Task). Where a task's Description and the snapshot disagree, the task file wins — it was written after the reconciliation.

## Global Constraints

- **Product decisions stand:** Folder files is a mode of Notes; Import once adopts (never copies) an Obsidian vault; Sort is composed unconditionally on the folder tree with a Newest default (task 32172) — never re-pin Sort's absence.
- **Reviewed diagnostics are metadata-only.** `REVIEWED_METADATA_ONLY_DIAGNOSTICS` inside `Tests/Architecture/test_persistent_diagnostic_inventory.py` forbids `logger.opt(exception=True)`, `exc_info=`, `stack_info=` on listed diagnostics because the log-file sink runs with `diagnose=True`. Add a reason as `error_type={}` metadata, never a traceback. Any new or reworded log/notify call requires `scripts/check_persistent_diagnostic_inventory.py --write` in the same PR, re-run AFTER every dev merge (the JSON auto-merges to impossible totals silently).
- **Never search a path-bearing string for a word a path can contain** (the `"private"` canary incident): leak canaries are tokens like `zqleakcanary`.
- **Harness vaults live under `$HOME` outside the profile directory.** A vault inside the profile config dir is rejected as `private_path_overlap`; anything under `/private/tmp` fails the `owner_group == os.getegid()` check. Use `$HOME/.cache/tldw-crit/<name>/vault`.
- **Task ids:** sweep with `git rev-list --objects --all | grep -oE 'task-[0-9]+' | grep -oE '[0-9]+' | sort -rn | head -1` plus every worktree's `backlog/tasks` immediately before minting and again before pushing; older `created_date` keeps a colliding id. The peer session tldw-chatbook-da holds 32301–32310 and 32346–32393 (all on dev); mint above the swept global max (32450 at dispatch time, so 32451+).
- **Tests:** foreground, ≤40 node ids per chunk; `Tests/UI/test_library_shell.py` is 19k lines — never wholesale. Every fix ships a test proven RED without the change and GREEN with it, on the REAL route (no `SimpleNamespace` receivers for behaviour that lives in `LibraryScreen`). A repair that loosens an assertion is a defect.
- **Live verification** of every user-visible change at 235x52 and one compact size (100x30 or 60x24) via tmux with `TLDW_CONFIG_PATH` pointing at a scratch profile — never the user's real database — with captures saved and cited.
- **Guide:** `Docs/User_Guide/library/notes.md` and `file-notes.md` get body-text updates plus a "Verified against" stamp for every behaviour change; stale sentences are rewritten as "was … — superseded by task-N below", never deleted. Stamps keep a blank line before each.
- **Never** `git stash`, `git clean -fdx`, force-push, broad `pkill`, or merge a PR; the controller merges. The `timeout` command does not exist.
- **Peer session** tldw-chatbook-da's Notes work is ALREADY on dev: #2590 (32233 Escape-in-filter, 32215 Sort/folder verbs, 32218 vocabulary) and #2605 (ctrl+n creates directly, Details panel copy, the compact notes sheet at 60 cols, a `DiagnosticsOpened` rail message). Its Escape/footer pins live in `Tests/UI/test_library_crit10_notes_details.py` and layout pins in `test_library_crit10_layout.py` — respect them; every group merges dev before its first push and again before landing.

---

### Task 1: Lasting-sync Check — honest refusal, no leak, real reasons (P0)

**Tasks:** 32243 (P0), 32244, 32269 (docs half, blocked on the fix)
**Files:** `tldw_chatbook/Notes/notes_sync_runtime.py` (`_ensure_lease` ~:1598, `review_setup` ~:1769-1774, `_publish` ~:2889), `tldw_chatbook/Notes/notes_sync_filesystem.py` (~:191 group check), `tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py` (~:806-814 bare except), `Tests/Notes/test_notes_sync_runtime.py`, `Tests/Notes/test_notes_sync_filesystem.py`, `Tests/UI/test_library_notes_lasting_sync_flow.py`, `Docs/User_Guide/library/notes.md` (lasting-sync chapter).
**Proven mechanism (from the bisect, in the task):** on any non-OWNER admission `_ensure_lease` calls `_publish(..., persist=True)` for a root setup never persisted → `NotesDeviceStateError` masks the intended `RuntimeError("root_lease_unavailable")`; `_root_paths[root_id]` is only popped when `_ensure_lease` returns falsy, so a raise leaks it and the next attempt rejects the same folder as `lasting_root_overlap`; the controller's bare except substitutes "Check failed. Review the folder and settings, then try again." and logs nothing. Separately `notes_sync_filesystem.py` requires `owner_group == os.getegid()` (the user's PRIMARY group), which rejects every file whose group is any other group the user belongs to.

- [ ] **Step 1: reproduce with the headless script** — `review_setup()` against a vault inside a scratch profile dir (rejection `private_path_overlap`) and one under `/private/tmp` (`unsupported_metadata`); confirm the `NotesDeviceStateError` traceback and the leaked `_root_paths` entry after one failure.
- [ ] **Step 2: failing tests first** — (a) a rejected admission on the setup path raises a reason-carrying error, not `NotesDeviceStateError`; (b) a failed Check leaves `_root_paths` empty and the same folder is admissible on retry once the cause is fixed; (c) the controller maps `private_path_overlap`, `lasting_root_overlap`, `unsupported_metadata`, `root_discovery_incomplete`, `notes_sync_cutover_not_admitted` to distinct user copy and logs a metadata-only warning with `reason_code`; (d) group check: a file whose group is any group in `os.getgroups()` passes.
- [ ] **Step 3: fix** — `_ensure_lease(..., persist=False)` on the setup path (or `_publish` skipping the store write for unpersisted roots); `try/finally` around the `_root_paths` pop; carry `admission.reason_code` in the raised error; controller copy per reason ("That folder is inside Chatbook's own data directory", "Another Chatbook window is using that folder", "That folder is already connected", "Some files there use a permission model this sync can't track", "Sync isn't available while another Chatbook instance owns this profile"); keep the bare except but log `logger.warning("notes sync check_setup failed; reason={}", reason)`; relax the group check to `os.getgroups()` membership.
- [ ] **Step 4: inventory re-pin**, tests green, live: Keep a folder synced → Check changes on (i) a vault under `$HOME` → review plan; (ii) a vault inside the profile dir → the specific copy, then pick the `$HOME` vault in the same session → succeeds (the leak is gone). Captures at 235x52.
- [ ] **Step 5: docs** — rewrite the lasting-sync chapter's claims that cannot run today; list the refusal reasons and their copy; stamp. Commit per step.

### Task 2: Editor keys — Tab focus order, Ctrl+End, Shift+Tab, slash, Escape, delete prompt

**Tasks:** 32246, 32247, 32253, 32252, 32267, 32268
**Files:** `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (NoteEditorInput / NoteEditorTextArea, `_NOTE_FIELD_TAB_BINDINGS` from #2571, Info pane, delete prompt), `tldw_chatbook/UI/Library_Modules/library_notes_controller.py` (body cursor restore ~:2016, slash handling from 32131), `tldw_chatbook/UI/Screens/library_screen.py` (tab trap for the delete prompt ~:7943, Escape ladder), `Tests/UI/test_library_notes_wave_editor_keys.py`, `Tests/UI/test_library_canvas_sync_defects.py`.
**Facts:** 32246's burst half is fixed by #2571 (`_NOTE_FIELD_TAB_BINDINGS`); what remains is focus ORDER — Tab from the body lands on `#library-notes-source-database` (a Button that swallows typed keys) and the reconciler saw "no focused control" once. 32247: `Ctrl+End` landed insertion at char 23–125 of 35,187 in three encodings (`\x1b[1;5F` included). 32253: Shift+Tab into Title selects all, autosave commits the loss; select-on-focus is ON in Title and OFF in path fields. 32252: `/` still types itself from an unfocused canvas with an empty field — the green test `test_slash_focuses_the_notes_filter_without_inserting_itself` constructs a focused state. 32267: Escape from Info needs two presses. 32268: the delete prompt paints six rows below Delete, outside the Info border.

- [ ] **Step 1:** RED tests on the real editor route for each: Tab from body lands on a text-accepting or shape-focused control and typed keys land there; `Ctrl+End` (all three encodings) moves to the document end; Shift+Tab into Title keeps the caret at end with no selection; `/` from an unfocused canvas focuses the filter with an empty value; one Escape from Info returns to the editor; the delete prompt renders inside the Info border adjacent to Delete.
- [ ] **Step 2:** fixes at the shared seams (focus chain order in the canvas; a TextArea `end`/`document_end` action bound to the three sequences; Title select-on-focus off with caret-to-end; the slash binding at the canvas/screen level guarded by "no text-accepting focus"; Escape ladder one step; prompt placement). Grep every caller.
- [ ] **Step 3:** GREEN; live at 235x52 and 100x30 with captures; guide updates for the key table; stamps; inventory check.

### Task 3: Preview and layout — 20-row cap, primary actions far below content, return cue, low-vision polish

**Tasks:** 32249, 32259, 32270, 32261, plus peer riders 32389 (60-col notes reader keeps an empty work pane — `_sync_library_notes_reader_layout_from_shell` priority="items" at library_screen.py ~6248) and 32390 (dead `#library-notes-template-section` rule in _agentic_terminal.tcss)
**Files:** `tldw_chatbook/css/components/_agentic_terminal.tcss` (~:2898 `max-height: 20`, the compact `1fr` rule next to it), the four surfaces named in 32259 (Session Git panel, Folder files empty state, import review, Info), `library_notes_canvas.py` (return cue display flag), `library_file_notes_workspace.py`, `Tests/UI/test_library_notes_wave_list.py`, `Tests/UI/test_library_crit8_waits.py`.
**Facts:** the preview box closes at row 33 of 52 with 14 blank rows below and PageDown is inert until clicked, while the COMPACT layout gets `height: 1fr` — the inversion is a CSS bug. 32259: four surfaces put the primary action 20–38 rows below its content; 32270: the `‹ Library / Notes` cue's display flag is always false in wide Database Notes; 32261: compact select strip, stray `○`, unnamed `--->` grips.

- [ ] Tests that measure region heights at 235x52 and 100x30 (wide preview fills the pane; each surface's primary action within N rows of its content or pinned to the pane bottom; the cue displays when the guide says it does; grips carry names).
- [ ] CSS + compose fixes; CSS bundle sync (`tldw_chatbook/css/check_bundle_sync.py`) and the boot-CSS byte budget (`Tests/Performance/test_boot_css_byte_budget.py`, ratchet 768,000 — do not grow the boot bundle; Library rules belong in the screen-owned sheet).
- [ ] Live captures; guide; stamps.

### Task 4: Import-once review — paging, elision, reasons, receipts, fidelity, wikilink rendering

**Tasks:** 32250, 32256, 32257, 32258, 32262, 32263
**Files:** `tldw_chatbook/Widgets/Library/library_note_import_canvas.py`, `library_notes_add_from_files_canvas.py`, `tldw_chatbook/Notes/note_import_{planner,parsers,executor,receipts}.py`, `tldw_chatbook/UI/Library_Modules/library_note_import_controller.py`, `Tests/UI/test_library_notes_wave_import_ux.py`, `Tests/Notes/test_note_import_*.py`.
**Facts:** page 1 of 3 is 23 identical `Archive/Archived note NNN.md` rows, groups split across pages, payloads elide to `· ke…`; the two relationship explanations cut mid-word at 235 columns; "Import selected items unavailable" gives no reason at the control; the receipt repeats three times with three denominators (66/67/59+8); fidelity: dropped non-tag frontmatter keys, "Content: no change" above a changed diff, a pre-armed collision panel, generic `.canvas` copy; wikilinks are rewritten to `note://<uuid>` inside the user's prose (32263 is a design decision to revisit — implement display-text links that keep the target title visible, e.g. `[[Title]](note://uuid)` rendering as the title, and record the decision).

- [ ] RED tests per item (grouping keeps a group on one page or repeats its header; row payload shows the decision-bearing text at 235 cols; the unavailable control names its reason; one receipt with one denominator; each fidelity case; wikilink rendering).
- [ ] Fixes; inventory check; live captures of the review on the 71-file vault (vault under `$HOME`); guide; stamps.

### Task 5: Pickers, export destination, ingest browser, Session Git keyboard, Folder files frontmatter

**Tasks:** 32251, 32242, 32248, 32265, 32264
**Files:** `tldw_chatbook/Widgets/enhanced_file_picker.py` (~:2547, :2563-2565 pre-fill), `tldw_chatbook/Third_Party/textual_fspicker/*` if the path field lives there, `tldw_chatbook/UI/Screens/library_screen.py` (`_library_ingest_browse_location`, `_persist_library_ingest_location`), `tldw_chatbook/Library/library_browse_location.py`, `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py` + the Session Git panel widget, `Tests/UI/test_library_notes_wave_file_notes.py`, `Tests/Library/test_library_browse_location.py`, `Tests/UI/test_library_file_notes_git.py`.
**Facts:** path fields pre-fill with the current directory and do not select on focus, so click+type yields `/Users/…/private/tmp/…`; the error paints INSIDE the dialog's bottom border; the export destination ACCEPTS `…zip/private/…zip` and fails at write time with a raw errno; the older ingest browser reads `[library.ingest] last_directory` unvalidated and writes unconditionally. Session Git prints `Up/Down select · Tab actions · Enter run · Esc back` but Tab never reaches the actions (mouse-only, 22 rows below the file row); copy says "1 session notes" and `refs/heads/main`. Folder files silently preserves YAML frontmatter it hides.

- [ ] RED tests: select-on-focus + typed absolute path replaces the field; export destination validated before accepting (parent must be a directory, name must not sit under a file); ingest browser through `validate_existing_absolute_directory` with the same generation-claim write; Tab from the Session Git file list reaches Stage/Commit; plural copy and branch name; a visible frontmatter affordance in Folder files.
- [ ] Fixes at the shared picker seam (one field behaviour for all three pickers); live captures (Folder files picker, export, Session Git keyboard journey to a real commit, verified with `git log`); guide; stamps.

### Task 6: List and tree — duplicate titles, Undo into a collapsed folder, selection counters, seeded-open latency

**Tasks:** 32254, 32255, 32272, 32260
**Files:** `library_notes_canvas.py` (row label composition from 32137, select-mode counters), `library_notes_controller.py` (Undo restore path from 32124, tree expansion), `Notes/note_folder_repository.py`, `Tests/Widgets/Library/test_library_notes_canvas.py`, `Tests/UI/test_library_notes_wave_list.py`, `Tests/Performance/` for the latency probe.
**Facts:** two `Reading list · Unfiled · 2m` rows are byte-identical because folder+age cannot disambiguate two unfiled notes of the same age — 32254 is a design decision: add a third key (a stable 4-char id suffix, or the modified time-of-day) only when folder+age tie; update the pinning test `test_duplicate_titles_render_folder_and_age_suffixes` to the new truth. Undo restores the row into a collapsed folder so it stays invisible — expand the folder and reveal the row. Two selection counters disagree ("1 selected" above "0 selected") — one source of truth. Library opens in 12.6 s on a seeded 27-item profile vs 2.7 s fresh — profile before optimising: find the slow path with a timing probe, fix the cause, pin a budget.

- [ ] RED tests; fixes; live captures (duplicate rows, delete→Undo reveal, select mode counts, timed open on the seeded profile); guide; stamps.

### Task 7: First-run wizard completion toast

**Tasks:** 32266
**Files:** `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`, `tldw_chatbook/css/features/_wizards.tcss`, `Tests/UI/test_library_notes_wave_onboarding.py` or the wizard tests.
**Facts:** the completion toast overlays the Summary action bar (new with #2538). Move or delay the toast so the action bar stays clickable; RED/GREEN; live capture; keep `_wizards.tcss` within the boot-CSS budget.

### Task 8: Test health — the riders wave 2 filed

**Tasks:** 32184, 32185, 32201, 32202, 32204, 32294, 32295, 32298
**Files:** the test files each task names; production changes only where a task's AC says the product is wrong (32201: the filter never reaches the search service — that is a product defect; 32185 lists several "newly reachable" behaviours, some product).
**Facts:** each task records the exact assertion text of every red. Repair each to the current truth at equal or better strength; where the product is wrong (32201), fix the product with a RED/GREEN test. 32204: sweep every harness pinning `CSS_PATH` to the bundle alone and load `APP_STYLESHEETS`; add the guard test its AC#2 describes. 32298: the Textual `text-area--gutter` COMPONENT_CLASSES race — isolate with a fixture-level pause, not a retry. 32295: decide repair vs retire for the two notes_create deep-link tests (repair: wait on `.library-notes-row`, not `#library-notes-row-0`).

- [ ] Per task: run the named node ids RED, fix, GREEN; compare FAILED name sets against a detached `origin/dev`; every red left must be filed or explained in the task notes.

### Task 9: Idea — editor chrome strip

**Tasks:** 32143
**Files:** `library_notes_canvas.py` (a one-row strip under the editor: word count, cursor line:col, save state, the main accelerator), `library_notes_controller.py` (state feed), `_agentic_terminal.tcss` (screen-owned rules), `Docs/User_Guide/library/notes.md`.
**Design:** one row, right-aligned facts, left-aligned save state (`Draft — not saved yet` / `Saved 16:24`), hidden below 80 columns. Read the task's ACs; brainstorm nothing new. Tests on the real editor route; live captures at 235x52 and 100x30; guide; stamps; CSS budget.

### Task 10: Idea — capture a Console answer into a Library note

**Tasks:** 32146
**Files (corrected after #2591 / TASK-32312):** `tldw_chatbook/Chat/console_message_actions.py` (the action registry — a new row), `tldw_chatbook/UI/Console_Modules/message.py` (`ConsoleMessageController` handler, injected like `save_console_video_copy`), `tldw_chatbook/Widgets/Console/console_transcript.py` + `console_message_more_menu.py` (menu and route), the Save-as flow (`console_save_as_modal.py`, `_console_save_as_destinations`) as the shape for receipt + Open-note hand-off, `tldw_chatbook/Notes/` create path with provenance keywords, `Docs/User_Guide/console/*.md` and `library/notes.md`. NOT `chat_message_enhanced.py` (legacy Chat tab) and NOT `console_chat_controller.py` (the send/turn controller).
**Design:** a message action "Save as note" creating a Database note whose title is the first line, body the answer, keywords `console`, `conversation:<id>`; a receipt with "Open note"; no LLM call. Read the task's ACs. Tests on the real action route; live capture from Console → Library; guides; stamps; inventory check for any new log call.

### Task 11: Backlinks persistence

**Tasks:** 32186
**Files:** `tldw_chatbook/DB/ChaChaNotes_DB.py` (schema bump: `_CURRENT_SCHEMA_VERSION` + migration adding a `note_links(source_id, target_id)` table maintained on save), `tldw_chatbook/Notes/` link extraction (reuse the wikilink parser from 32145/32129), `get_notes_linking_to` → table lookup, `Tests/DB/`, `Tests/Notes/test_note_backlink_query.py`.
**Facts:** today every note open scans every note body. Migration must backfill from existing bodies; the save path updates the relation; the query reads the table. Schema migration tests; performance probe before/after; guide note.

### Task 12: Docs sweep (lands last)

**Tasks:** 32271 plus every guide claim touched by Tasks 1–11
**Files:** `Docs/User_Guide/library/notes.md`, `file-notes.md`, `import-and-export.md`, `library.md`, the Console guide page for Task 10.
- [ ] After all code groups are on dev: re-read both guides against the shipped behaviour with a live walk; fix contradictions with the "superseded" treatment; consolidate stamps; run `Tests/Docs/`.

---

## Landing order
1 (P0) → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12. Each group: merge dev, sweep for duplicate defs/ids, inventory check after the merge, push, required check, merge. Alternate slots with the peer session one-for-one.

## After landing
Critique #3 at the new dev tip with two isolated assessors, vaults under `$HOME`, like-for-like against 20/40.
