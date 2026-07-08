# Library L2b.2 — Notes Sync, Import, Count Seam, and Tab Deprecation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Notes absorption — in-Library Sync and Import Note, a real notes count, the flush-serialization fix — then deprecate the standalone Notes tab.

**Architecture:** Extends the shipped L2b.1 notes canvas (`LibraryNotesCanvas` gains `"sync"` mode; list header gains Sync/Import actions). Sync executes through the existing `NotesSyncService`/`NotesSyncEngine` seams, offloaded, with progress marshaled to targeted UI updates. Deprecation re-points every `TAB_NOTES` consumer to Library navigation contexts and removes the tab registration; file deletion stays with the dead-code sweep.

**Tech Stack:** Python ≥3.11, Textual, pytest pilots, real ChaChaNotes DB + real tmp-dir sync for integration tests.

**USER DECISION (2026-07-08): Templates are parity-complete.** The standalone screen never had template create/edit/delete — only browse + create-from-template, which L2b.1 already ships. NO template-management canvas in this phase. Template CRUD is a logged optional future feature.

## Global Constraints (binding on every task)

- Spec: `Docs/superpowers/specs/2026-07-07-library-l2b-l3-design.md` (Phase L2b.2 + Global constraints), amended by the templates decision above. Branch `claude/library-l2b2` in worktree `.claude/worktrees/library-l2`.
- Test command: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q <target> --tb=short`. No `timeout` command. Full affected set: `Tests/Library/ Tests/UI/test_library_shell.py Tests/UI/test_destination_shells.py Tests/Notes/`.
- CSS: edit `css/components/_agentic_terminal.tcss` → `./build_css.sh` → commit both.
- RENDER RULES: no `Select` (the standalone sync pane's Direction/Conflict Selects become **cycling buttons**; its auto-sync `Switch` becomes a **toggle button**); stacked full-width widgets; no 1fr+fixed Horizontal; canvas child width `1fr`.
- EDITOR AMENDMENTS (from L2b.1, binding): autosave never recomposes; disarm/rearm `_library_note_editor_armed` around EVERY notes-canvas recompose; flush on every exit path; post-await freshness guards on every worker result.
- Service seams (verified): `NotesSyncService(notes_service=<app.notes_service>, db=<per-user CharactersRAGDB>)`; `async sync_folder(root_folder: Path, user_id: str, direction: SyncDirection = BIDIRECTIONAL, conflict_resolution: ConflictResolution = ASK, extensions: List[str] = None, progress_callback: Optional[Callable[[SyncProgress], None]] = None) -> Tuple[str, SyncProgress]` (`Notes/sync_service.py:190`); enums in `Notes/sync_engine.py:30-50` (`SyncDirection.{DISK_TO_DB,DB_TO_DISK,BIDIRECTIONAL}`, `ConflictResolution.{ASK,DISK_WINS,DB_WINS,NEWER_WINS}`); config keys `("notes","sync_directory")` default `"~/Documents/Notes"`, `("notes","auto_sync")` default False (existing), plus NEW `("notes","sync_direction")`/`("notes","sync_conflict_resolution")` (the standalone pane never persisted these — persist all four, via `save_setting_to_cli_config`/`get_cli_setting`).
- Import parsing: reuse `_parse_note_from_file_content(file_path, file_content_str) -> (title|None, content|None)` from `Event_Handlers/notes_events.py:45`; caps 2_000_000 content / 300 title (mirror `notes_screen.py:2869-2870`); file dialog `FileOpen` from `Third_Party.textual_fspicker` (real constructor signature — the notes screen's import usage at `notes_screen.py:2856-2862` is the working reference); imported path validated with `validate_path_simple(..., require_exists=True)`.
- All service calls through `_run_library_service_call(..., isolate_in_worker=True)`; sync progress callbacks fire OFF the UI thread — marshal every UI update with `self.app_instance.call_from_thread(...)` (or equivalent thread-safe path) doing TARGETED `update()`s only; recompose only on sync start/finish transitions.
- Real-backend tests for every mutation/integration (real ChaChaNotes DB; real tmp-dir for sync round-trips). Fakes mirror real seam shapes.
- Stage only changed files by path; NEVER `git add -A`; never touch `.claude/settings.local.json`. Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Live screenshot QA at 2050×1240 + explicit user approval before merge (Task 6). The deprecation makes this gate NON-NEGOTIABLE: the user must see the app without the Notes tab.

## File Structure

- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (add `count_notes`), `tldw_chatbook/Notes/Notes_Library.py` (interop passthrough), `tldw_chatbook/Notes/notes_scope_service.py` (scope passthrough)
- Create: `tldw_chatbook/Library/library_notes_sync_state.py` — pure sync-panel display state
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (list-header actions; `"sync"` mode), `tldw_chatbook/UI/Screens/library_screen.py` (sync/import wiring, flush serialization, count), `tldw_chatbook/Library/library_shell_state.py` (only if row copy changes)
- Modify (deprecation): `tldw_chatbook/Constants.py`, `tldw_chatbook/app.py`, `tldw_chatbook/Event_Handlers/event_dispatcher.py`, `tldw_chatbook/UI/Screens/notes_tab_initializer.py` (delete), `tldw_chatbook/Config_Files/note_templates.json` (title polish rider)
- Tests: extend `Tests/Library/test_library_notes_state.py` (or new sync-state file), `Tests/UI/test_library_shell.py`, `Tests/UI/test_destination_shells.py` (fake), `Tests/Notes/test_notes_scope_service_library_canvas.py`, new `Tests/Notes/test_library_notes_sync_integration.py`

---

### Task 1: Real notes count seam (rail badge honesty)

**Files:** Modify `tldw_chatbook/DB/ChaChaNotes_DB.py`, `tldw_chatbook/Notes/Notes_Library.py`, `tldw_chatbook/Notes/notes_scope_service.py`, `tldw_chatbook/UI/Screens/library_screen.py` (snapshot fetch + counts). Tests: `Tests/Notes/test_notes_scope_service_library_canvas.py`, `Tests/UI/test_library_shell.py`, fake in `Tests/UI/test_destination_shells.py`.

**Interfaces produced:** `CharactersRAGDB.count_notes() -> int` (`SELECT COUNT(*) FROM notes WHERE deleted = 0`, parameterized style mirroring `count_messages_for_conversation` at `ChaChaNotes_DB.py:4613`); `NotesInteropService.count_notes(user_id) -> int`; `NotesScopeService.count_notes(*, scope, user_id=None) -> int` (local branch only; server scope raises/returns the server path consistent with siblings — read `list_notes` for the routing shape). Library snapshot: notes fetch also calls `count_notes` (same gathered offload), sets `notes_count=<true count>`, `notes_known=True`; rail badge renders `Notes (42)` not `(42+)`.

- [ ] Steps: TDD — real-DB test (seed 3 notes + 1 deleted → count 3); scope-service passthrough test; pilot re-anchors (rail badge asserts exact `(N)`; update any `(N+)` assertions); fake gains `count_notes` mirroring the real signature; implement; full affected suites; commit `feat(notes): real notes count seam; Library rail badge shows exact count`.

### Task 2: Flush-vs-autosave serialization (carried L2b.1 follow-up)

**Files:** Modify `tldw_chatbook/UI/Screens/library_screen.py` (`_flush_library_note_save`). Test: `Tests/UI/test_library_shell.py`.

**The bug:** the flush issues an inline `await _save_library_note(...)` that bypasses the `library_note_save` exclusive worker group, so a just-fired debounce autosave and the flush can save the same note with the same version → spurious self-`ConflictError`.

- [ ] Steps: TDD — regression pilot (monkeypatch debounce 0.05; type; poll until the fake's `save_calls` grows by one [autosave in flight/complete]; immediately press Back; assert NO conflict banner ever appears and the note saved exactly the expected number of times) proven RED if the race is reproducible deterministically with a delayed-save fake (delay the AUTOSAVE's seam response, then Back mid-flight — the flush's second save must WAIT). Implementation: in `_flush_library_note_save`, before saving, find running workers in the save group (`[w for w in self.workers if w.group == "library_note_save" and w.is_running]` — verify the real WorkerManager API and use its idiom) and `await w.wait()` on them; re-check `_library_note_dirty` after (the completed save may have cleared it); only then save inline. Commit `fix(library): flush waits for in-flight autosave (no spurious self-conflict)`.

### Task 3: In-canvas Notes Sync

**Files:** Create `tldw_chatbook/Library/library_notes_sync_state.py`; modify `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `tldw_chatbook/UI/Screens/library_screen.py`, CSS. Tests: pure units + pilots + new `Tests/Notes/test_library_notes_sync_integration.py`.

**Pure state (complete contract):**

```python
# tldw_chatbook/Library/library_notes_sync_state.py
"""Pure display-state for the Library notes sync panel."""
from __future__ import annotations
from dataclasses import dataclass

SYNC_DIRECTIONS = ("bidirectional", "disk_to_db", "db_to_disk")
SYNC_CONFLICTS = ("newer_wins", "disk_wins", "db_wins", "ask")
_DIRECTION_LABELS = {"bidirectional": "Bidirectional", "disk_to_db": "Disk → DB", "db_to_disk": "DB → Disk"}
_CONFLICT_LABELS = {"newer_wins": "Newer wins", "disk_wins": "Disk wins", "db_wins": "DB wins", "ask": "Ask"}

@dataclass(frozen=True)
class LibraryNotesSyncState:
    """Display state for the sync panel (folder, options, status, activity)."""
    folder: str
    direction: str
    conflict: str
    auto_sync: bool
    status_line: str        # "idle" | "syncing · 3/12" | "done · 12 files · 2 conflicts" | "failed · <reason>"
    activity_lines: tuple[str, ...]  # most-recent-first, capped at 20

def next_sync_direction(value: str) -> str: ...   # cycle SYNC_DIRECTIONS, unknown -> first
def next_sync_conflict(value: str) -> str: ...    # cycle SYNC_CONFLICTS, unknown -> first
def sync_direction_label(value: str) -> str: ...  # _DIRECTION_LABELS with raw fallback
def sync_conflict_label(value: str) -> str: ...
def sync_status_line(status: str, *, processed: int = 0, total: int = 0, conflicts: int = 0, error: str = "") -> str: ...
def append_activity(lines: tuple[str, ...], entry: str, *, cap: int = 20) -> tuple[str, ...]: ...
```

(Implement the `...` bodies per the docstring semantics; full unit coverage: cycling+wrap+unknown, labels, status-line variants, activity cap/ordering.)

**Widget:** list mode's header area gains two compact `library-canvas-action` buttons on their own toolbar row under the sort button: `#library-notes-sync-open` ("Sync") and `#library-notes-import` ("Import note"). New `mode == "sync"` compose (stacked): `‹ Back to notes` (`#library-notes-sync-back`) → section header `Notes sync` → folder `Input` (`#library-notes-sync-folder`, value from state) + `[Browse…]` (`#library-notes-sync-browse`) → `direction: Bidirectional ▸` (`#library-notes-sync-direction`) → `conflicts: Newer wins ▸` (`#library-notes-sync-conflict`) → `auto-sync: ○` / `auto-sync: ✓` toggle (`#library-notes-sync-auto`) → `[Sync now]` (`#library-notes-sync-run`) → status `Static` (`#library-notes-sync-status`) → activity `Static` (`#library-notes-sync-activity`, joined lines, markup=False).

**Screen wiring:** `_library_notes_view` gains `"sync"`. State fields `_library_notes_sync_direction/_conflict/_auto` seeded from config at first entry (persist on every change via `save_setting_to_cli_config`); folder read from `("notes","sync_directory")`. Sync run: exclusive worker `group="library_notes_sync"`; builds `NotesSyncService(notes_service=app.notes_service, db=<the per-user db the standalone pane uses — read notes_workbench_panes.py:794-804 and mirror>)`; validates the folder (`validate_path_simple`, must exist, expanduser); calls `sync_folder(Path(folder), user_id, direction=SyncDirection(direction), conflict_resolution=ConflictResolution(conflict), progress_callback=<thread-safe marshal>)` via `_run_library_service_call(..., isolate_in_worker=True)`. Progress callback: marshal to the UI thread and do TARGETED updates of `#library-notes-sync-status`/`-activity` only (never recompose mid-run); completion/failure recomposes once (freshness-guarded: still in sync view?). Auto-sync toggle arms/cancels a 300s repeating timer (only while the Library screen lives; document that scope honestly — the standalone pane's timer had the same lifetime). Entering/leaving sync mode flushes any pending editor save first and resets on rail re-entry (extend the existing reset).

- [ ] Steps: pure units → widget/pilots (mode entry/exit/reset; cycling buttons persist config; sync-now passes the chosen enums to a recording fake; status updates without recompose [same-widget-instance assertion]; folder validation failure notifies quietly) → real integration (`Tests/Notes/test_library_notes_sync_integration.py`: REAL `NotesSyncEngine` + real temp ChaChaNotes DB + `tmp_path` folder — db_to_disk: seeded note lands as file with content; disk_to_db: dropped `.md` file becomes a note; conflict newer_wins: newer disk edit wins) → CSS + build → full suites → commit `feat(library): in-canvas notes sync (folder, direction, conflicts, activity)`.

### Task 4: Import Note + template-JSON polish rider

**Files:** Modify `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (button landed in T3), `tldw_chatbook/Config_Files/note_templates.json`. Tests: pilots + parse-wrapper units.

- [ ] `#library-notes-import` → `FileOpen` dialog (mirror `notes_screen.py:2856-2862` usage exactly — that constructor call is the WORKING reference, unlike the broken FileSave one was) → callback validates path (`validate_path_simple`, `require_exists=True`), reads ≤ `LIBRARY_NOTE_CONTENT_MAX_CHARS`, parses via `_parse_note_from_file_content`, caps title at 300, creates through the existing `_create_library_note(title=..., content=...)` (lands in the editor; snapshot+count refresh comes free). Failure paths (unreadable, oversize, unparsable) → quiet warning notify, no crash. Pilots: import-lands-in-editor (fake dialog result → created note), oversize rejected, `.md` title-from-stem. RIDER: in `note_templates.json`, retitle `project`: `"Project: "` → `"Project Plan - {date}"` and `research`: `"Research Notes - "` → `"Research Notes - {date}"` (kills the dangling separators AND gives both real secondaries); update any pilot that asserted the old resolved titles. Commit `feat(library): import note into the canvas; polish template titles`.

### Task 5: Deprecate the standalone Notes tab

**Files:** `tldw_chatbook/Constants.py`, `tldw_chatbook/app.py`, `tldw_chatbook/Event_Handlers/event_dispatcher.py`, delete `tldw_chatbook/UI/Screens/notes_tab_initializer.py`, `tldw_chatbook/UI/Screens/library_screen.py` (nav context). Tests: re-anchor routing tests; grep-audit.

**Consumer table (verified — every site must be handled):**

| Site | Action |
|---|---|
| `Constants.py:18,52,59,68` | Delete `TAB_NOTES` + `ALL_TABS`/group/label entries |
| `app.py:359` (registration), `:6028` + `event_dispatcher.py:207` (screen-id maps), `event_dispatcher.py:199` | Remove |
| `app.py:505` (palette "Switch to notes management") | Replace with a Library-routing entry ("Notes (Library)") or remove — match how other palette entries route to Library |
| `app.py:760` (new-note deep link) | Re-point: navigate to Library with a navigation context selecting the `create-note` row (extend `_library_navigation_context` handling — mirror the existing `conversation_id` deep-link mechanism; add `create_note` flag or `notes_row` param) |
| `app.py:1674` (chat-sidebar NavigateToScreen) | Re-point to Library `browse-notes` (with `note_id` context opening that note's editor — add `note_id` to the Library navigation context, mirroring `conversation_id`) |
| `app.py:3409,5145,5289` (tab-transition branches) | Remove the dead branches |
| `notes_tab_initializer.py` | Delete file + its registration |

`notes_screen.py` + `Widgets/Note_Widgets/*` + their green tests stay in-tree (unreachable) for the dead-code sweep — only ROUTING tests re-anchor. The standalone screen's latent FileSave TypeError dies with reachability.

- [ ] Steps: TDD the two new nav contexts first (pilot: app-level deep link lands on Library create view / opens note editor); re-point + remove sites; grep-audit step `grep -rn "TAB_NOTES" tldw_chatbook/ Tests/` → zero hits in `tldw_chatbook/` (test files referencing the retired screen may keep local literals only if they don't route); full suites + re-anchors; commit `feat(library)!: retire the standalone Notes tab; Notes lives in Library`.

### Task 6: Final review + live QA + approval gate

- [ ] Full affected suites green; whole-branch review (most capable model, `merge-base(origin/dev, HEAD)..HEAD`, with the accumulated-minors ledger); live QA at 2050×1240 (seeded HOME): nav bar WITHOUT the Notes tab, notes list with exact count badge, sync panel idle + after a real tmp-dir sync run (status + activity populated), import flow landing in the editor, deep-link (palette/new-note) landing in Library. Commit evidence + README. **Present to the user for the explicit approval gate — the deprecation ships only with their sign-off. Then PR to dev.**
