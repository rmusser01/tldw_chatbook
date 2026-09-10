# Library ▸ Notes riders wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the nine riders filed from the Notes critique fix wave (tasks 32172–32180) and build the two idea tasks the controller approved on the user's delegation (32144 Trash view, 32145 backlinks), in eight branches that each ship as their own PR stacked on `fix/library-notes-docs`.

**Architecture:** Each task below is one branch/worktree/PR grouped by file overlap. Work stays inside the Notes surface; behaviour changes are test-first against the existing Library harnesses; every task ends with a live tmux check on its own scratch profile.

**Tech Stack:** Python 3.12, Textual 8.x, pytest, Backlog.md CLI, tmux.

**Spec:** the task files `backlog/tasks/task-32172 … task-32180`, `task-32144`, `task-32145` (acceptance criteria) and the wave-1 reports under `.superpowers/sdd/2026-09-09-library-notes-critique-wave/` in the `notes-crit-docs` checkout (mechanisms). Base for every branch: `fix/library-notes-docs` @ d0ff40842f (all seven wave-1 branches merged and comment-fixed).

## Global Constraints

- `SCRATCH` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/ebe96959-a4f7-4839-8522-01876feccafb/scratchpad`. Work only inside your assigned worktree `SCRATCH/wave2/<group>` (branch `fix/library-notes-<group>`, stacked on `fix/library-notes-docs`). Every shell command starts with `cd SCRATCH/wave2/<group> &&`. Never touch the main checkout `/Users/macbook-dev/Documents/GitHub/tldw_chatbook` or another group's worktree.
- Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (3.12). Run pytest FROM the worktree cwd with `-q -p no:cacheprovider`; whole files or explicit node ids, never `-k`, never the whole suite, never `Tests/UI/test_library_shell.py` whole (explicit node ids only; compare failing NAME sets against `d0ff40842f` in a throwaway detached worktree you remove afterwards).
- TDD: failing test first, shown failing, then green. New tests go in the most specific existing `Tests/UI/test_library_notes_*.py` / `Tests/Notes/test_note_import_*.py` file or a new `Tests/UI/test_library_notes_riders_<group>.py`.
- Smallest diff that satisfies the acceptance criteria; no new abstractions; fix at the shared seam (grep every caller first).
- CSS: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, run `python -m tldw_chatbook.css.build_css`, commit the regenerated bundles alongside. No ancestor-scoped bare-type subject rules in `BUNDLED_CSS`.
- Git: explicit paths only; one commit per task id; do NOT push, open or merge PRs (the controller does after review). NEVER a bare `git stash` (shared stash stack). Trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Backlog hygiene from the worktree root: `backlog task edit <id> -s "In Progress" --plan "…"` before the first code change; at the end tick the ACs in the task file, then `backlog task edit <id> -s Done --notes "…"` (replaces the notes section; write once). For the two idea tasks, AC#1 ("Design agreed with the user before implementation") is satisfied by the controller's recorded ruling on the user's delegation; tick it with that note.
- Docs: every user-visible change updates `Docs/User_Guide/library/notes.md` or `file-notes.md` with a `*Verified against fix/library-notes-<group> — 2026-09-09 (task-NNNNN: …)*` stamp appended to the page's TRAILING stamp block (one chronological block per page; never an inline mid-page stamp).
- Live verification (required before reporting): `SCRATCH/notes-crit/launch_wave2.sh <group> <fresh|power> [cols rows]` (socket `nw2-<group>`; profiles `SCRATCH/notes-crit/wave2/<group>/{fresh,power}`, seeded power: 10 notes, `vault/` an Obsidian-style vault with a git baseline). `sleep 15`; drive with `tmux -L nw2-<group> send-keys`; observe with `capture-pane -p`. Reach Library via `C-p`, type `Switch to Library`, `Down`, `Enter`. Click a rail row by anchoring on `^│ │    Notes (` in a capture. In every folder picker: type the path, press **Enter**, THEN Select. Quit `C-q`, then `tmux -L nw2-<group> kill-server`. Save captures under `SCRATCH/notes-crit/wave2/<group>/caps/`. Foreground test runs only; never a background Monitor.
- Copy rules: blocked or disabled states carry a text reason and a next step on the same line; never colour-only meaning; no raw errno, UUID or ISO timestamp reaches the user.
- Scope: the acceptance criteria of your tasks and nothing else; an AC needing a product decision you cannot make stays unticked with a note.

---

### Task 1: Date ordering back in Database Notes (group `r-list`, task 32172)

**Files:** `tldw_chatbook/Notes/notes_organization_repository.py` (`page_note_placements`, the deep-link locator — both `ORDER BY title COLLATE NOCASE`), `tldw_chatbook/Library/library_notes_tree_state.py` (paging), `tldw_chatbook/UI/Library_Modules/library_notes_controller.py` (`_library_notes_sort`, the `{"newest","oldest","title"}` guard), `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (`sort_available = self.tree_projection is None`, task-32128). Tests: `Tests/UI/test_library_notes_wave_list.py`, `Tests/UI/test_library_notes_folder_navigator.py`, `Tests/UI/test_library_notes_rename_propagation_t31796.py:254` (the title-order pin: reconcile with a recorded reason), repository tests under `Tests/Notes/`.
- [ ] Failing repository test: `page_note_placements(..., order="newest")` returns placements by `last_modified DESC` with a stable tiebreaker, page 2 continues correctly; the locator finds the same page for a note under that order.
- [ ] Failing UI test: with a tree projection, `Sort: Newest` reorders rows inside a folder across a page boundary; the flat list keeps working.
- [ ] Implement the ORDER BY parameter through paging AND the locator; re-enable the Sort control for the tree only once both hold; keep the flat-list control.
- [ ] Live-verify on the power profile; docs; backlog; commit.

### Task 2: Folder files rail before linking, and wait polish (group `r-file-notes`, tasks 32173, 32180)

**Files:** `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py` (`body.display = self._root is not None` ~:2650; `_use_configured_sync_folder`; `_abandon_root_change_task`; the slow-wait row), `tldw_chatbook/UI/Screens/library_screen.py` (the Folder files swap keeps `#library-file-notes-rail`). Tests: `Tests/UI/test_library_notes_wave_file_notes.py`, `Tests/UI/test_library_crit8_waits.py`, `Tests/UI/test_library_file_notes_workspace.py` (name-set vs base).
- [ ] 32173: failing test — Folder files with `root=None` at (235, 52) still shows the Library rail; implement by keeping the rail outside the root-gated body (or un-gating only the rail); compact unchanged; docs: `file-notes.md` pre-link sentence and 32136's `[~]` AC note updated; re-tick 32136 AC#1 only if the qualifier is no longer needed.
- [ ] 32180: pin the slow-wait busy row at 60 columns (every control on-pane, no two Cancels); the `Use <folder>` button reads the modern config key as well as the legacy `notes.sync_directory` (grep `get_cli_setting("notes"` and the modern `[file_notes] root` / notes sync settings to find the right one) or the guide names the legacy key explicitly; an assertion or comment on `_abandon_root_change_task` records the invariant that every re-entrant root change abandons the previous one synchronously.
- [ ] Live-verify on the fresh profile (pre-link rail) and the power profile; docs; backlog; one commit per task id.

### Task 3: Last-used start directory for the vendored pickers (group `r-pickers`, task 32174)

**Files:** `tldw_chatbook/UI/Screens/library_screen.py` (`_push_library_note_import_picker`, and `_library_ingest_browse_location` ~25670 as the precedent), `tldw_chatbook/UI/Library_Modules/library_notes_controller.py:~4433` (keep-synced picker), `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py` `_choose_root` (Folder files) — caller-side only, via `get_cli_setting`/`save_setting_to_cli_config` the way ingest does. Tests: `Tests/UI/test_library_notes_wave_import_ux.py`, `Tests/UI/test_library_notes_wave_file_notes.py`, `Tests/UI/test_file_picker_start_dir.py` (extend).
- [ ] Failing tests per picker: after a successful selection the next open starts at that directory's parent (or the directory itself for Folder files), else home; the setting key per context (`library.notes_import`, `library.notes_sync`, `file_notes.browse`) is documented in the guide.
- [ ] Implement; live-verify; docs; backlog; commit.

### Task 4: Retire or repair the flat-row Notes tests (group `r-tests`, task 32175)

**Files:** `Tests/UI/test_library_notes_reader.py` (13 reds), `Tests/UI/test_library_shell.py` (the notes tests that wait for `#library-notes-row-0/-1`, incl. the `filter_sort` capability node ~:33944 and `test_library_note_60x20_editor_state_allocation`), `Tests/UI/test_library_file_notes_workspace.py` (`test_wide_files_task_return_restores_database_browse_receipt` ~:3671), `Tests/UI/test_library_canvas_scoped_sync.py`, `Tests/UI/test_library_honesty_accessibility.py` (the compact notes footer test). Method: `grep -rn "library-notes-row-[0-9]" Tests/` for the census.
- [ ] For every hit: drive the tree row (`.library-notes-tree-note-row` buttons, `screen.query(...).first(Button)`) if the assertion is still meaningful, or retire the test with a one-line reason in its docstring and a `pytest.skip`-free deletion (no skips). Record the census (kept/repaired/retired) in the task notes.
- [ ] Run each touched file whole and report base-vs-head failing-name sets; the notes-related reds must go to zero or be named as unrelated with proof.
- [ ] Backlog; one commit per file.

### Task 5: Import copy edges and Obsidian follow-ups (group `r-import`, tasks 32176, 32178)

**Files:** `tldw_chatbook/Notes/note_import_parsers.py` (`_payload_from_mapping`, the whole-document `not_a_note` verdict), `note_import_plan_models.py` (`_NON_IMPORTABLE_CLASSIFICATIONS`, `WIKILINK_SCAN`, `rewrite_wikilinks`), `note_import_executor.py` (unchanged-repeat → Update existing abort: `ImportReceiptTransitionError("Membership receipt authority does not match the approved plan.")`, pre-existing on the wave base), `note_import_receipts.py` (resolved-link count), `note_import_discovery.py` (Windows vault detection), `tldw_chatbook/Widgets/Library/library_note_import_canvas.py` (`_NON_IMPORTABLE`, `_CLASSIFICATION_LABELS`, `Skip all` copy), `tldw_chatbook/Library/library_note_import_state.py`. Tests: `Tests/Notes/test_note_import_*.py`, `Tests/UI/test_library_notes_wave_import_ux.py`, `Tests/UI/test_library_note_import_flow.py`.
- [ ] 32176: failing tests — a JSON array with one content-less record reports which record failed (others import); `Skip all` / `Create all` copy says "on this page"; ONE source of truth for the non-importable set (the canvas derives from the enum); an unchanged repeat switched to Update existing either updates cleanly or refuses with a plain reason before execution (fix the transition authority mismatch at the executor/receipt seam, with a test on the wave base's behaviour first).
- [ ] 32178: failing tests — an unterminated ``` fence protects everything after it from wikilink rewriting; the receipt shows `N links resolved` (extend the receipt model with a defaulted field; migrate any durable-ledger schema the way the codebase does, with a test); Windows discovery reports `vault_detected` (or the review shows "Obsidian detection is unavailable on Windows" on that platform, unit-tested through the adapter seam); aliases are stored distinguishably (a `alias:` keyword prefix, or a decision recorded in the guide).
- [ ] Live-verify on the fresh profile (vault import); docs; backlog; one commit per task id.

### Task 6: Editor leftovers and the duplicated screen method (group `r-editor`, tasks 32177, 32179)

**Files:** `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (New-note and load-retry views' back cues; `#library-note-context-status` composed-then-hidden), `tldw_chatbook/Library/library_notes_state.py` (`_parse_browser_timestamp` private import → a public helper), `tldw_chatbook/UI/Screens/library_screen.py` (`_seed_local_source_snapshot_from_cache` defined twice ~:9353 and ~:11729; the first is dead). Tests: `Tests/UI/test_library_notes_wave_editor_keys.py`, `Tests/UI/test_library_notes_reader.py` (only the node ids you touch), `Tests/UI/test_library_shell.py:24538,32097` (geometry pins listing the dead widget — reconcile).
- [ ] 32177: failing tests — one back-cue rule in the New-note view and the load-retry view (wide `‹ Notes`, compact `‹ Back to list`); the dead widget removed and its two geometry pins updated; the timestamp helper public and imported cleanly.
- [ ] 32179: delete the dead first definition; a test that the surviving definition is the one callers reach (`grep -n _seed_local_source_snapshot_from_cache` for callers).
- [ ] Live-verify on the power profile; docs; backlog; one commit per task id.

### Task 7: Trash view for Database Notes (group `i-trash`, task 32144)

**Files:** `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (a `Recently deleted (N)` row under the folder tree; a trash list view listing soft-deleted notes with title, age, `Restore`), `tldw_chatbook/UI/Library_Modules/library_notes_controller.py` (`_undo_library_note_delete` is the restore seam: `NotesScopeService.restore_note(note_id, version)`; reuse it), `tldw_chatbook/Notes/notes_scope_service.py` / `tldw_chatbook/DB/ChaChaNotes_DB.py` (a bounded query for `deleted = 1` notes newest first, paged 20), `tldw_chatbook/UI/Screens/library_screen.py` (bindings: `r` restore on the focused trash row, `escape` back; footer context), `_agentic_terminal.tcss`. Reference grammar: the Media Trash view (`grep -n "Trash" tldw_chatbook/Widgets/Library/library_media_*.py`, `library_media_trash_browse_controller.py`). Tests: new `Tests/UI/test_library_notes_riders_trash.py`, `Tests/Notes/` for the query.
- [ ] Failing tests: the row shows the soft-deleted count and hides at zero; the view lists deleted notes newest first with `Restore`; Restore returns the row to its folder/Unfiled and the rail count exactly as Undo does (same seam); `r` restores the focused row; Escape returns to the list; no permanent delete anywhere (Danger stays out of the Trash view).
- [ ] Implement; live-verify on the power profile (delete two notes, open the Trash view, restore one); guide section "Recently deleted"; backlog (AC#1 ticked with the controller-ruling note); commit.

### Task 8: Backlinks in Info (group `i-backlinks`, task 32145)

**Files:** `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (Info → Properties gains `Linked from (N)` listing note titles; each entry opens that note), `tldw_chatbook/UI/Library_Modules/library_notes_controller.py` (load backlinks when a note opens, off the UI thread, bounded to 50), `tldw_chatbook/Notes/notes_scope_service.py` / `tldw_chatbook/DB/ChaChaNotes_DB.py` (a bounded query: notes whose body contains the exact link token for the open note — the Obsidian executor writes `[label](note://<id>)`; confirm the exact form with `grep -n "note://" tldw_chatbook/Notes/note_import_executor.py note_import_plan_models.py`; use an FTS5 or `LIKE` query with a parameter, never string formatting). Tests: new `Tests/UI/test_library_notes_riders_backlinks.py`, `Tests/Notes/` for the query.
- [ ] Failing tests: a note linked from two others shows `Linked from (2)` with both titles; a note with no inbound links shows `Linked from (0) — no notes link here yet`; activating an entry opens that note; the query is bounded and parameterised; deleted notes are excluded.
- [ ] Implement; live-verify on the fresh profile after an Obsidian import (the vault's `Projects/Library review` links to `Daily/2026-09-07` and `Reading/Zettelkasten — overview`); guide; backlog (AC#1 ticked with the controller-ruling note); commit.
