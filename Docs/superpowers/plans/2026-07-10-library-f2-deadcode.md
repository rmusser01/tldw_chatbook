# F2 — Dead-code sweep (notes screen retirement + zero-importer orphans) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. All deadness verdicts below were established by a full-repo import/route inventory at branch point 673dda26; every deletion must re-prove itself with the gate greps before the commit.

**Goal:** Remove the retired standalone Notes screen and every zero-importer legacy module, with zero-survivor grep discipline. No behavior changes: the only live call path touched (`refresh_notes_tab_after_ingest`) is already an always-False no-op.

**Architecture:** Two tasks. Task 1 retires notes_screen and its orphan widget/handler graph (needs co-edits in app.py, the screens registry, and shared tests). Task 2 deletes the proven zero-importer orphans (pure deletion).

## Global Constraints

- Stage only changed files by explicit path; NEVER `git add -A`. Never touch `.claude/settings.local.json`. Bare `git stash` FORBIDDEN.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Test command: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`.
- Zero-survivor rule: after each task, its gate grep block (below) must print NOTHING before committing. A single survivor = stop and fix, never delete around it.
- Deletion must preserve current behavior exactly. Anything that would IMPROVE behavior (e.g. making post-ingest refresh target the Library notes canvas) is out of scope — log it in the report as a follow-up instead.
- KEEP list (live, verified): `Widgets/Note_Widgets/note_selection_dialog.py` (STTS_Window.py:3557), `Widgets/Note_Widgets/note_creation_modal.py` (document_generation_modal.py:240 via live chat_events chain), `UI/Screens/notes_scope_models.py` (study_screen.py:35), `Chat_Window_Enhanced.py`, `splash_screen.py`, `SearchRAGWindow.py`, `Tests/UI/test_bulk_selection_tooltips.py`.

### Task 1: Retire notes_screen + orphan notes widget/handler graph

**Files:**
- Delete: `tldw_chatbook/UI/Screens/notes_screen.py` (3,323 lines)
- Delete: `tldw_chatbook/Widgets/Note_Widgets/notes_workbench_panes.py`, `workspace_context_panel.py`, `workspace_source_picker.py`, `notes_toolbar.py`, `notes_editor_widget.py`, `notes_status_bar.py`, `notes_sync_widget_improved.py`, `notes_sync_widget.py`
- Delete: `tldw_chatbook/Event_Handlers/notes_sync_events.py` (closed island with notes_sync_widget: each is the other's only importer; verify the island greps first)
- Modify: `tldw_chatbook/app.py` — delete `save_current_note` (~:6009, no external caller) and `refresh_notes_tab_after_ingest` (~:4444; its isinstance check is always False since NotesScreen is unroutable — screen_registry aliases "notes"→"library")
- Modify: `tldw_chatbook/Event_Handlers/note_ingest_events.py:443` — remove the `app.call_later(app.refresh_notes_tab_after_ingest)` call (preserves current no-op behavior exactly); log "post-ingest Library notes refresh" as a follow-up in the report
- Modify: `tldw_chatbook/UI/Screens/__init__.py` — remove the `"NotesScreen": ".notes_screen"` entry (:17) and `'NotesScreen'` in `__all__` (:32)
- Modify (docstring mentions only, reword to drop the dead reference): `tldw_chatbook/UI/Screens/library_screen.py:3526,3535,3590,3665,3703,4198,4231,5242`, `tldw_chatbook/Library/library_notes_state.py:267`
- Delete tests: `Tests/UI/test_notes_screen.py` (FIRST relocate its `TestNotesScreenState` class — it exercises `NotesScreenState` from the SURVIVING `notes_scope_models.py` — into a new `Tests/UI/test_notes_scope_models.py`), `Tests/UI/test_notes_workbench_layout.py`, `Tests/Widgets/test_notes_widgets.py`
- Edit shared tests: `Tests/UI/test_workbench_pane_focus.py` (drop NotesScreen cases), `Tests/UI/test_file_picker_action_tooltips.py` (drop the 3 dead notes imports at :14-16 + their cases), `Tests/UI/test_ux_audit_smoke.py` (drop NotesScreen usage; keep notes_scope_models if used), `Tests/UI/test_non_obscuring_focus_contract.py:1691` (drop notes_editor_widget), `Tests/UI/conftest.py:280-284` (`notes_screen_state` fixture: repoint the import to `notes_scope_models` if any surviving test uses the fixture, else delete the fixture)
- Keep untouched: `Tests/UI/test_screen_navigation.py:299` (the TAB_NOTES-stays-dead guard)

**Gate grep (must print nothing before commit; run from worktree root):**
```bash
grep -rn "notes_screen\|NotesScreen" --include=*.py tldw_chatbook Tests | grep -v test_screen_navigation.py
for w in notes_workbench_panes workspace_context_panel workspace_source_picker notes_sync_widget notes_toolbar notes_editor_widget notes_status_bar notes_sync_widget_improved notes_sync_events; do grep -rn "$w" --include=*.py tldw_chatbook Tests; done
```
(`notes_sync_widget` pattern also matches `notes_sync_widget_improved` — both must be gone. `NotesSyncWidget` class refs count as survivors.)

**Suites:** the edited/new test files + `Tests/UI/test_screen_navigation.py` + `Tests/UI/test_ux_audit_smoke.py` + `Tests/Notes/` (if present) + `Tests/UI/test_library_shell.py` + `python -c "import tldw_chatbook.app"`.

**Commit:** `refactor(notes): retire the standalone Notes screen and its orphaned widget graph`

### Task 2: Delete zero-importer legacy orphans

**Files (all proven zero external importers — re-prove with the gate before deleting):**
- Delete: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_backup.py` (4,586), `chat_events_fixed.py` (898), `chat_events_refactored.py` (397), `chat_streaming_refactored.py` (263), `chat_streaming_events_fixed.py` (333)
- Delete: `tldw_chatbook/UI/Chat_Window_Enhanced_Fixed.py` (452), `tldw_chatbook/UI/Chat_Window_Enhanced_Refactored.py` (468)
- Delete: `tldw_chatbook/Widgets/settings_sidebar_backup.py` (351), `tldw_chatbook/Widgets/splash_screen_old.py` (2,051)
- Delete: `tldw_chatbook/UI/SearchRAGWindow.py.bak` (2,309), `Tests/UI/test_media_window_v88.py.bak`

**Gate grep (must print nothing before commit):**
```bash
for f in chat_events_backup chat_events_fixed chat_events_refactored chat_streaming_refactored chat_streaming_events_fixed Chat_Window_Enhanced_Fixed Chat_Window_Enhanced_Refactored settings_sidebar_backup splash_screen_old; do grep -rn "$f" --include=*.py tldw_chatbook Tests; done
grep -rn "SearchRAGWindow.py.bak\|test_media_window_v88" --include=*.py tldw_chatbook Tests
```

**Suites:** `Tests/Chat/` (if present) + `Tests/UI/test_chat_window*.py` (whatever exists; `ls` first) + `python -c "import tldw_chatbook.app"` + `python -m compileall tldw_chatbook -q`.

**Commit:** `chore: delete zero-importer legacy modules (backup/fixed/refactored twins, .bak files)`

## Verification & gate

No pixels change (pure dead code) — no visual QA. Final: combined suite (Tests/UI/test_screen_navigation.py, test_ux_audit_smoke.py, test_workbench_pane_focus.py, test_file_picker_action_tooltips.py, test_non_obscuring_focus_contract.py, test_library_shell.py, Tests/Notes/ if present, Tests/Widgets/, the new notes_scope_models test) + app import + compileall. PR to dev; merge only on explicit user authorization.
