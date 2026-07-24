# F4 — Library chatbook export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Spec: `Docs/superpowers/specs/2026-07-11-library-f4-chatbook-export-design.md` — the authority on behavior; read the sections named in your task brief. Anchors exact at branch point 98835549; grep symbols, lines drift.

**Goal:** Export the Library (everything, or a browse section's current filter scope) as a chatbook zip through the existing `ChatbookCreator`/`local_chatbook_service`, registered as a first-class chatbook artifact.

**Architecture:** Three tasks: (1) a pure scope-resolver module + the missing full-id DB queries (the truncation-proof foundation); (2) the rail row, in-canvas export form, counts worker, and destination picker; (3) the execution worker wired through `local_chatbook_service` with registry, notifications, and the round-trip integration test.

**Tech Stack:** Python ≥3.11, Textual, SQLite (parameterized only), existing `Chatbooks/` module.

## Global Constraints

- Stage only changed files by explicit path; NEVER `git add -A`. Never touch `.claude/settings.local.json`. Bare `git stash` FORBIDDEN (WIP commit instead).
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- RED-first for behavior changes; bounded polls in pilots (`range(150)`/`pause(0.02)`); `escape_markup` for user text reaching Button/Static markup; user paths through `validate_path_simple`; parameterized SQL only; workers for operations >100ms.
- Test command: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`.
- Spec-critical invariants (verbatim from spec): `include_media=True` is ALWAYS passed when media is in scope; scope resolution NEVER reads rendered snapshots (`LIBRARY_SOURCE_PAGE_SIZES`: conversations 50, media 50, notes 100); zip first, registry record only on success; destination normalized to `.zip` BEFORE overwrite confirmation.
- CSS via source modules + `./build_css.sh` when touched (commit both).

### Task 1: Scope resolver + full-id DB queries

**Files:**
- Create: `tldw_chatbook/Library/library_export_scope.py`
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py` (near `get_paginated_files` :4538)
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (near `list_notes` :6642)
- Test: `Tests/Library/test_library_export_scope.py`, additions to the existing media/ChaChaNotes DB test files (`ls Tests/MediaDB* Tests/ChaChaNotesDB* Tests/Media` — put DB-method tests where that class's tests live)

**Interfaces produced (Tasks 2/3 rely on these exact names):**
- `ExportScope` frozen dataclass in `library_export_scope.py`: `kind: str` ("everything" | "media" | "conversations" | "notes"), `media_type: str | None = None` (only meaningful for kind="media"; "All" and None both mean unfiltered).
- `export_scope_label(scope: ExportScope, counts: dict[str, int]) -> str` — the form's summary copy: `Everything: 128 media · 542 conversations · 87 notes` (omit zero-count sources only when everything is zero → see empty copy in Task 2) or `Media (type: video) · 12 items` / `Media · 12 items` for unfiltered.
- `count_export_scope(scope, media_db, chachanotes_db) -> dict[str, int]` — keys "media"/"conversations"/"notes", zero for sources outside the scope.
- `resolve_export_selections(scope, media_db, chachanotes_db) -> dict[ContentType, list[str]]` — ids as `str(int)` for media (the creator's `_collect_media` does `int(media_id)`), native id strings for conversations/notes; sources outside the scope omitted from the dict entirely.
- DB methods: `Client_Media_DB_v2.get_all_active_media_ids(media_type: str | None = None) -> list[int]` (WHERE `deleted = 0 AND is_trash = 0`, optional `AND type = ?`, parameterized); `ChaChaNotes_DB.get_all_conversation_ids() -> list[str]` and `get_all_note_ids() -> list[str]` (soft-delete aware — mirror the WHERE clauses the existing list/paginated queries use; read them first).

Requirements:
- `library_export_scope.py` is a pure module (stdlib + `Chatbooks.chatbook_models.ContentType` + type hints only; DB handles are passed in, never constructed).
- RED tests: (a) THE TRUNCATION LOCK — seed 60+ conversations and 60+ media (beyond the 50-row snapshot caps) into real in-memory DBs, assert `resolve_export_selections(ExportScope("everything"), ...)` returns ALL ids; (b) media type filter resolves only matching, non-deleted rows (seed a deleted and a trashed row — excluded); (c) empty scope → counts all zero, selections `{}`; (d) label copy exact-match for both forms; (e) DB-method unit tests in the DB suites (parameterized filter, soft-delete exclusion).
- The ChaChaNotes id queries must scope to the same visibility the Library uses (single local user scope — read how `list_notes`/the conversations snapshot fetch filter, e.g. deleted flags, and mirror exactly; document the WHERE clause choice in the docstring).

Commit: `feat(library): export scope resolver with truncation-proof full-id queries`

### Task 2: Rail row, export form canvas, counts worker

**Files:**
- Modify: `tldw_chatbook/Library/library_rail_state.py` (sections tuple :8 region — the "ingest" section gains the export row; header copy "Import / Export"), `tldw_chatbook/Library/library_shell_state.py` (row → `canvas_kind` mapping ~:223-248)
- Create: `tldw_chatbook/Library/library_export_state.py` (form-state builder), `tldw_chatbook/Widgets/Library/library_export_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (row/canvas wiring; section "Export…" actions on the media/conversations/notes canvases; counts worker; FileSave destination flow — mirror the existing `FileSave` usage at ~:3694 and its default-filename sanitization note at :3664)
- Modify: CSS source module for the new canvas classes if needed (+ `./build_css.sh`, commit both)
- Test: `Tests/Library/test_library_export_state.py`, pilots in `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes Task 1's `ExportScope`, `count_export_scope`, `export_scope_label`.
- Produces: `LibraryExportFormState` frozen dataclass (fields: `scope: ExportScope`, `scope_line: str`, `counts_loading: bool`, `name: str`, `description: str`, `media_quality: str` ("thumbnail"|"compressed"|"original", default "thumbnail"), `destination: str` ("" until picked), `running: bool`, `status_line: str`, `error_line: str`, `export_enabled: bool`) + `build_library_export_form_state(...)` — Task 3 reads `scope`/`name`/`description`/`media_quality`/`destination` and drives `running`/`status_line`/`error_line`.
- Screen attrs produced for Task 3: `_library_export_scope: ExportScope`, `_library_export_counts: dict[str,int] | None`, `_library_export_form: dict` (name/description/quality/destination), `_library_export_running: bool`, `_library_export_error: str`.

Requirements:
- Rail: new row (id follows the existing `library-row-*` convention, e.g. `library-row-ingest-export`, label "Export", subtitle "in Library") under the ingest section; section header copy becomes "Import / Export". Selecting it → `canvas_kind` "export" with scope Everything.
- Section "Export…" actions: compact button on the media/conversations/notes canvases (id per section) that switches to the export canvas with `ExportScope(kind=<section>, media_type=self._library_media_type_filter if media else None)`.
- Counts worker: on entering the export canvas, a thread worker (`group="library_export_counts"`, exclusive in group) runs `count_export_scope` and marshals results via `call_from_thread`; until it lands, `counts_loading=True` renders the scope line as `Counting…`; zero-total scope → Export disabled + helper `Nothing to export in this scope.`
- Form: name prefilled `Library export YYYY-MM-DD` (stamp at form-open, local date); quality select with helper line `original copies full media files into the zip`; destination row: `Choose destination…` button → `FileSave` dialog; returned path → `validate_path_simple` → normalize suffix to `.zip` → if the NORMALIZED path exists, an in-form confirm line (`Overwrites <name>.zip` + the Export press proceeds) — read whether the FileSave dialog itself confirms overwrite (:3664-3670 note) and adjudicate: the .zip-normalized path is what must be confirmed, not the raw picked path.
- Export button disabled until: counts loaded, total > 0, destination set, not running. Quality/media rows hidden when the scope contains no media (conversations/notes-only scope).
- Pilots (RED where assertable): rail row opens the form with Everything scope and counts land (bounded poll); media section Export… opens with the type filter carried; empty-scope disable; destination normalization (`foo` → `foo.zip` shown in the form). Server-mode gating: trace how the Library shell currently surfaces the runtime source (the header shows "Library | Local"); if a server source can host the Library shell, render the Export row disabled with tooltip `Export packages local content only.`; if the Library shell is provably local-only today, document that in the code comment and skip the dead gating (adjudicate with evidence in the report).

Commit: `feat(library): export rail row, in-canvas chatbook export form with truthful counts`

### Task 3: Execution worker, service integration, round-trip test

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (the export worker + button handler + completion marshaling)
- Modify: `tldw_chatbook/Library/library_export_state.py` (running/error state transitions if builder-level)
- Test: pilots in `Tests/UI/test_library_shell.py`; integration `Tests/Library/test_library_export_roundtrip.py`

**Interfaces consumed:** Task 1 resolver; Task 2 form state/screen attrs; `self.app_instance.local_chatbook_service` (app.py wires it at :2889; `export_chatbook(request_data) -> dict` with keys success/message/path/dependency_info/name — see `Chatbooks/local_chatbook_service.py:240`; `create_chatbook(name=..., description=..., file_path=..., tags=...)` registry record at :173).

Requirements:
- Button handler (UI thread): re-validate (destination set, not running), set `_library_export_running=True`, recompose, start `@work(thread=True, exclusive=True, group="library_export")` worker.
- Worker (thread): resolve selections via Task 1 resolver (fresh DB reads — the worker may open its own connections per the DB classes' threading.local pattern); build the request payload: name/description/content_selections/output_path/media_quality, **`include_media=True` whenever `ContentType.MEDIA` is in the selections** (spec-critical: the creator silently skips all media otherwise); call `asyncio.run(service.export_chatbook(payload))` (async-signature, sync-body — never touches the app loop); on success call `asyncio.run(service.create_chatbook(name=..., description=..., file_path=<output path>, tags=["library-export"]))`; marshal completion via `call_from_thread`.
- Completion (UI thread): success → `notify(f"Exported chatbook to {path}", severity="information")` + when `dependency_info.get("auto_included")` non-empty append `f" ({n} characters auto-included)"`; clear running, reset error. Failure → `_library_export_error = <message>` rendered in the form (escape_markup), running cleared, Export re-enabled.
- Single-flight pilot: second press while running is a no-op (exclusive group + disabled button both asserted).
- Round-trip integration test (marked like the existing integration tests): seed real in-memory/tmp DBs (reuse `Tests/Chatbooks/` factories if importable — `Tests/Chatbooks/factories.py`), export Everything to a tmp path through the REAL service + creator, then (a) `zipfile` inspection: manifest counts match seeded counts AND media content files/entries present (pins include_media semantics — assert actual media textual content, not just counts), (b) `ChatbookImporter` imports into fresh DBs and item counts match. Plus a failure integration: unwritable destination (a directory path) → success=False surfaced, no registry record created (assert via `service.list_chatbooks`).
- Suites: `Tests/Library/`, `Tests/UI/test_library_shell.py`, `Tests/Chatbooks/`, `python -c "import tldw_chatbook.app"`.

Commit: `feat(library): chatbook export execution through local_chatbook_service with artifact registration`

## Verification & gate

- Combined gate: `Tests/Library/ Tests/Chatbooks/ Tests/Home/ Tests/UI/test_library_shell.py Tests/UI/test_home_screen.py Tests/UI/test_destination_shells.py Tests/UI/test_library_content_hub.py` + gate16 + app import.
- Visual QA (served TUI, seeded HOME, 2050×1240 recipe; seed beyond-cap data is unnecessary for pixels): export form via rail (Everything counts), media-section-scoped form (type filter in the scope line), running state, done notification, and the Artifacts/Home surfaces showing the registered chatbook. Present captures for user approval; PR to dev; merge only on explicit user authorization.
