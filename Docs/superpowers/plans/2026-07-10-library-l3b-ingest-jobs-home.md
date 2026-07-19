# Library L3b — Ingest canvas + job registry + Home Running feed + legacy cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild media ingest as an in-Library canvas backed by an app-level job registry with a serial queue-runner; feed running/failed jobs into Home; remove the placeholder Import/Export row; retire the legacy `LIBRARY_MODES` machinery.

**Architecture:** Pure job model + registry (`Library/library_ingest_jobs.py`, Textual-free) owned by the app; ONE long-lived thread queue-runner (exclusive in its own group) pulls jobs FIFO and calls the proven local seam `ingest_local_file`; ALL registry mutations happen on the UI thread via `call_from_thread`; the Library canvas and Home both render from registry snapshots. Canvas follows the shipped rail+canvas grammar (pure state → render-from-state widget → screen orchestration).

**Tech Stack:** Python 3.11+, Textual, pytest, SQLite MediaDatabase, textual-serve + playwright QA.

**Spec:** `Docs/superpowers/specs/2026-07-07-library-l2b-l3-design.md` (Phase L3b + Global constraints). Branch `claude/library-l3b` off `origin/dev` (2618c67a), worktree `.claude/worktrees/library-l3b`.

## Inventory-resolved decisions (binding; supersede the spec's sketches where they conflict — the spec itself delegates these to the inventory)

1. **Parity bar = TAB_INGEST's Local Files tab only.** The three server tabs are server-mode-gated scaffolding and the Server Jobs tab calls scope-service methods that do not exist on `MediaReadingScopeService` (`submit_media_ingest_jobs` etc. — only `submit_ingest_jobs` etc. exist; masked by mocked tests). Server ingest is NOT demonstrably shipped ⇒ the canvas is **local-only** with a quiet `ingest runs on Local` line under `Library | Server`. The broken server-jobs wiring is logged as a pre-existing follow-up bug, NOT fixed here.
2. **Backend seam = `ingest_local_file(file_path, media_db, *, title=None, author=None, keywords=None, ..., perform_analysis=False, chunk_options=None) -> Dict`** (`Local_Ingestion/local_file_ingestion.py:110`), writing via `MediaDatabase.add_media_with_keywords` (`Client_Media_DB_v2.py:2584`). Plaintext (.txt/.md/.rst/.log/.csv) needs zero optional deps — the guaranteed baseline. `Widgets/NewIngest/` is a **model-shape donor only** (`ProcessingJob`/`ProcessingState` at `ProcessingDashboard.py:50-142`); its `UnifiedProcessor`/`BackendIntegration` are mocked test shims and MUST NOT be adopted.
3. **`ingest-import-export` row is REMOVED** (spec's pre-authorized rule: a missing row beats an apologizing canvas). Inventory: the row renders 13 placeholder Statics, two deep-link buttons, and a permanently-disabled export button with NO seam; every real import/export action already lives on a better surface. Bulk Library export = tracked follow-up.
4. **No `type: auto ▸` cycling button and no URL input** (deviations from the spec's form sketch, justified by inventory): `ingest_local_file` has NO type-override parameter (auto-detects by extension — a type control would be fake), and TAB_INGEST ships no local URL ingest (web ingest is server-mode). The form takes a local file path; detected type renders on the job row. One file per submission — the queue provides batching (UX-equivalent to TAB_INGEST's batch select; flag at the gate).
5. **File picker = the existing `FileOpen` modal** already proven inside the Library screen (L2b.2 notes import, `handle_library_notes_import`).
6. **Stage labels:** the local seam emits NO intra-file stage/progress to the caller (processor callbacks are unwired even in TAB_INGEST). Real signals only: `queued → running → done/failed` + per-file completion within a job + elapsed time. No fabricated percentages.
7. **TAB_INGEST deprecation does NOT ship in L3b.** The rail row re-points to the canvas, but the nav tab stays routed (server-mode Sources/Web Clipper panels are real UI; the deprecation decision is presented at the user gate with options).
8. **Cleanup split:** Tasks 8–9 (legacy retirement) are pre-authorized by the spec to split into a trivial L3c if L3b runs hot. Task boundaries are drawn so Tasks 0–7 are independently mergeable.

## Global Constraints

- Canvas grammar: stacked full-width render-verified widgets; no `Select`; cycling/toggle Buttons; never a `Horizontal` mixing 1fr + fixed-width children. Canvas children `1fr`/width:100%; `overflow-x: hidden`. The ingest canvas root is a `VerticalScroll` (L3a clipping lesson — plain `Vertical` canvases clip past the fold).
- **Registry mutation discipline (spec, binding):** registry mutations ONLY on the UI thread via `self.app.call_from_thread(...)` from the runner thread (`self.app`, not `app_instance` — L2b.2 lesson); progress ticks = targeted row `update()`; recompose only on job add/remove/state-change. Submissions append and NEVER spawn their own runner.
- Accepted v1 limits (stated in code docstrings + QA README): in-memory registry — history dies with the app; a running job dies on quit (same as TAB_INGEST today); serial queue, parallelism is a follow-up.
- Services via `getattr(app, ..., None)` quiet degrade; user paths through `validate_path_simple`/`path_validation`; `_safe_text` on persisted text fields; `escape_markup` on any user text in Button/label markup (L3a lesson).
- Pilots: poll-after-recompose; gated fakes bound waits at 30.0s; geometry checks (`scroll_visible()` + `export_screenshot()`) for anything that could clip.
- Real-backend tests: real `MediaDatabase` (temp file-backed for thread tests — in-memory SQLite is thread-local; the runner is a THREAD worker, so in-memory DBs cannot cross into it. Where an in-memory DB is unavoidable, run inline via the established `is_memory_db` guard).
- CSS in `css/components/_agentic_terminal.tcss` → `./build_css.sh` → commit BOTH files.
- Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread` with `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share`.
- Git: stage ONLY changed files by explicit path; never touch `.claude/settings.local.json`; commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Exact copy values in this plan are binding: `Import media`, `Browse…`, `Start ingest`, `ingest runs on Local`, `Open in Library`, `Retry`, `Queue`, `queued`, `running`, `done`, `failed`, glyphs `●`/`✓`/`✗` as written.

### Key anchors (verified 2026-07-10 by inventory)

| Anchor | Location |
|---|---|
| Local seam | `Local_Ingestion/local_file_ingestion.py:110` (`ingest_local_file`), `:35` (`detect_file_type`), `get_supported_extensions` |
| DB write | `DB/Client_Media_DB_v2.py:2584` (`add_media_with_keywords`) |
| Donor model shapes | `Widgets/NewIngest/ProcessingDashboard.py:50` (`ProcessingState`), `:59` (`ProcessingJob`) |
| Ingest rows | `Library/library_shell_state.py:192-211`; canvas resolution `:236-250` |
| Rail dispatch / row select | `UI/Screens/library_screen.py:4634-4655` / `:4657-4696` |
| Import-export placeholder | `library_screen.py:1751-1806` (+ inspector `:1808`, action-panel branch `:2759`, handlers `:7979-7986`) |
| Shared open-by-id route | `library_screen.py` `_open_library_item_by_id` (L3a; media branch runs `_refresh_library_media_detail`) |
| Snapshot refresh chain | `library_screen.py:789-838` (`_refresh_local_source_snapshot` → `_apply_local_source_snapshot`) |
| FileOpen precedent | `library_screen.py` `handle_library_notes_import` (~`:4874` pre-drift) |
| Home categorization | `Home/dashboard_state.py:33-55` (`RUNNING_STATUSES = {running, queued, active, scheduled}`; failed → attention), sections `:654-702` |
| Home adapter templates | `Home/active_work_adapter.py:441-467` (watchlist items), `:508-525` (chatbook), build `:249-272`, handle_control `:274-343` |
| Home refresh cadence | `UI/Screens/home_screen.py:116-146` (mount thread worker + in-memory guard precedent) |
| Home→screen routing | `app.py:1832-1848` (`open_active_home_item_details`; does NOT pass screen_context — extend), nav-context apply `app.py:3462-3469` |
| Library nav-context | `library_screen.py:658-787`; constants `Constants.py:47-51` |
| `_active_mode` LIVE guards to remap | search guards `library_screen.py:7607,7686,7695`; collections lazy-load `:638,782,4691`; study copy `:2149,2535`; nav-context `:729,734` |
| Dead legacy chrome | `library_screen.py:6599-6970` (guarded by `_legacy_workbench_present` `:6614`), dead builders `:1860,1960,2236-2249,2703`, mode-chip CSS (4 files) |
| Mode-anchored tests | `Tests/UI/test_destination_shells.py:1594-1718`; `test_master_shell_design_system_contract.py:166-190`; `test_non_obscuring_focus_contract.py:680-699`; `Tests/Library/test_library_shell_state.py:43,69-71,114-120` |

---

### Task 0: Headless ingest smoke (real seam → real MediaDatabase row)

Spec-mandated gate: prove the chosen seam end-to-end BEFORE any UI work.

**Files:**
- Test: `Tests/Library/test_library_ingest_seam.py` (new)

**Interfaces:**
- Consumes: `ingest_local_file`, `MediaDatabase`.
- Produces: the executable proof + the reference fixture pattern every later task's tests reuse.

- [ ] **Step 1: Write the test** (no implementation exists to fail against — this is a seam-proof, not TDD):

```python
"""Task 0 smoke: the local ingest seam writes a real MediaDatabase row headlessly."""

from pathlib import Path

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Local_Ingestion import ingest_local_file


def test_ingest_local_text_file_creates_media_row(tmp_path: Path) -> None:
    source = tmp_path / "smoke-note.txt"
    source.write_text("Tides are driven by the moon's gravity.", encoding="utf-8")
    db = MediaDatabase(tmp_path / "smoke_media.db", client_id="l3b-smoke")

    result = ingest_local_file(
        file_path=source,
        media_db=db,
        title="Smoke note",
        author="tester",
        keywords=["smoke"],
        perform_analysis=False,
        chunk_options=None,
    )

    media_id = result["media_id"]
    assert isinstance(media_id, int)
    row = db.get_media_by_id(media_id)
    assert row is not None
    assert row["title"] == "Smoke note"
    assert "moon's gravity" in row["content"]
    assert row["type"] == "plaintext"


def test_ingest_failure_surfaces_as_exception(tmp_path: Path) -> None:
    db = MediaDatabase(tmp_path / "smoke_media.db", client_id="l3b-smoke")
    missing = tmp_path / "does-not-exist.txt"
    import pytest
    from tldw_chatbook.Local_Ingestion import FileIngestionError

    with pytest.raises((FileIngestionError, FileNotFoundError)):
        ingest_local_file(file_path=missing, media_db=db, perform_analysis=False)
```

(Verify `get_media_by_id`'s exact name/kwargs and the exception type in-file first; adapt asserts to the real row shape — the CONTRACT is: real row, real content, detected type.)
- [ ] **Step 2: Run it.** Both must pass with zero optional deps. If anything fails, STOP and escalate — the seam decision is wrong.
- [ ] **Step 3: Commit** — `test(library): headless local-ingest seam smoke for L3b`.

---

### Task 1: Pure job model + registry (`Library/library_ingest_jobs.py`)

**Files:**
- Create: `tldw_chatbook/Library/library_ingest_jobs.py` (Textual-free; stdlib + loguru only)
- Test: `Tests/Library/test_library_ingest_jobs.py`

**Interfaces (produced — later tasks consume verbatim):**

```python
class IngestJobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

@dataclass
class LibraryIngestJob:
    job_id: str                      # "ingest-job-{n}", registry-assigned
    source_path: str                 # validated absolute path (display uses basename)
    title: str = ""
    author: str = ""
    keywords: tuple[str, ...] = ()
    perform_analysis: bool = False
    chunk_enabled: bool = False
    chunk_size: int = 500
    state: IngestJobState = IngestJobState.QUEUED
    detected_type: str = ""          # filled when running/done
    media_id: int | None = None      # filled on success
    error: str = ""                  # filled on failure (sanitized single line)
    submitted_at: float = 0.0        # time.monotonic() reference for elapsed
    started_at: float | None = None
    finished_at: float | None = None

class LibraryIngestJobRegistry:
    def submit(self, *, source_path, title="", author="", keywords=(), perform_analysis=False,
               chunk_enabled=False, chunk_size=500) -> LibraryIngestJob   # appends QUEUED, returns the job
    def next_queued(self) -> LibraryIngestJob | None
    def mark_running(self, job_id, *, detected_type="") -> LibraryIngestJob | None
    def mark_done(self, job_id, *, media_id) -> LibraryIngestJob | None
    def mark_failed(self, job_id, *, error) -> LibraryIngestJob | None
    def requeue(self, job_id) -> LibraryIngestJob | None    # failed → fresh QUEUED copy appended (new job_id); returns new job
    def jobs(self) -> tuple[LibraryIngestJob, ...]           # newest-first snapshot
    def counts(self) -> dict[str, int]                       # per-state counts
    runner_active: bool                                      # set/cleared by the runner owner (UI thread only)
    def add_listener(self, callback: Callable[[], None]) -> None   # fired after every mutation (UI thread)
    def remove_listener(self, callback) -> None
```

Model shape follows the NewIngest donor (`ProcessingJob`/`ProcessingState`) simplified to the single-file-per-job design: no per-file dicts, no pause/cancel states (v1 limits documented in the module docstring: in-memory, serial, dies with the app). `mark_*` methods return the updated job or None for unknown ids (idempotent-safe). Listener errors are swallowed with a debug log (a broken listener must not corrupt the registry).

- [ ] **Step 1: Failing tests:** submit assigns sequential ids + QUEUED; `next_queued` returns FIFO order and skips non-queued; `mark_running/done/failed` transition + stamp times + fill fields; unknown id → None, no raise; `requeue` on a failed job appends a fresh QUEUED copy preserving form fields (and only works on FAILED jobs — others return None); `jobs()` newest-first immutable snapshot; `counts()`; listener fires once per mutation, listener exception swallowed; keywords stored as tuple.
- [ ] **Step 2: Run to fail. Step 3: Implement. Step 4: Run green** (whole file).
- [ ] **Step 5: Commit** — `feat(library): pure ingest job model and registry`.

---

### Task 2: App-level queue-runner + submission seam

**Files:**
- Modify: `tldw_chatbook/app.py` (registry construction + `submit_library_ingest_job` + runner)
- Test: `Tests/Library/test_library_ingest_runner.py`

**Interfaces:**
- Consumes: Task 1 registry; `ingest_local_file`; `detect_file_type`; `app.media_db` (may be None → submission fails the job immediately with `error="Media database is unavailable."`).
- Produces: `app.library_ingest_jobs: LibraryIngestJobRegistry` (constructed in `__init__` near the study wiring); `app.submit_library_ingest_job(**form_fields) -> LibraryIngestJob` (UI-thread only: appends via registry, then starts the runner IFF `not registry.runner_active`); `app.retry_library_ingest_job(job_id)` (requeue + same conditional start); the runner worker `_run_library_ingest_queue` — `@work(exclusive=True, thread=True, group="library_ingest_queue")`.

Runner contract (spec architecture, binding):
- Started ONLY by the submission/retry seams when `runner_active` is False; sets `runner_active=True` synchronously (UI thread) BEFORE the worker call so double-submission cannot double-start; the worker clears it via `call_from_thread` on exit (try/finally).
- Loop: `job = call_from_thread(registry.next_queued)` → if None, exit. Else `call_from_thread(registry.mark_running, job.job_id, detected_type=detect_file_type(path) or "")` → run `ingest_local_file(file_path=..., media_db=self.media_db, title=job.title or None, author=job.author or None, keywords=list(job.keywords) or None, perform_analysis=job.perform_analysis, chunk_options=({"method": "sentences", "size": job.chunk_size, "overlap": 100} if job.chunk_enabled else None))` on the runner thread → success: `call_from_thread(registry.mark_done, job.job_id, media_id=result["media_id"])`; any exception: `call_from_thread(registry.mark_failed, job.job_id, error=<sanitized first line, ≤200 chars>)`. Continue looping.
- EVERY registry touch from the worker goes through `self.call_from_thread` (the runner lives on the App, so `self` IS the app). One job's failure never kills the loop.

- [ ] **Step 1: Failing tests.** Drive a REAL file-backed `MediaDatabase` (tmp_path) and real small .txt files through the real app seams — but do NOT boot the full TldwCli. Instead build a minimal Textual `App` test-harness class that mixes in / hosts the same runner method and registry (mirror how `Tests/UI/test_library_shell.py` builds harness apps), OR test through the real `TldwCli` if a lightweight boot fixture already exists (check `Tests/` for a precedent first; prefer the harness). Cover: (a) submit → job reaches DONE with a real `media_id` row in the DB; (b) two submissions while the first runs → serial FIFO, ONE runner (assert `runner_active` never double-sets; second job also completes); (c) a failing job (missing file) → FAILED with sanitized error, next queued job still runs; (d) retry of the failed job → new QUEUED job that then succeeds (point it at a now-existing file); (e) `media_db=None` → job fails immediately with the exact unavailable copy, no runner crash; (f) listener fired on state changes (count the calls). Use `pilot.pause()` polling with bounded loops to await runner completion (never unbounded waits).
- [ ] **Step 2: Run to fail. Step 3: Implement. Step 4: Run green.**
- [ ] **Step 5: Commit** — `feat(library): app-level ingest job queue-runner`.

---

### Task 3: Rail flip + Import/Export row removal

**Files:**
- Modify: `tldw_chatbook/Library/library_shell_state.py` (rows `:192-211`)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (remove placeholder bodies/handlers; canvas dispatch stub)
- Test: `Tests/Library/test_library_shell_state.py`, `Tests/UI/test_library_shell.py`, `Tests/UI/test_destination_shells.py`

**Interfaces:**
- Produces: constant `LIBRARY_ROW_INGEST_MEDIA = "ingest-import-media"`; the row flips to `target_kind="canvas"`, `target_id="ingest-media"` (NOT `"ingest"` — avoids the TAB_INGEST route-name collision), title stays `Import media`; `ingest-import-export` row DELETED; `compose_content` gains an `elif shell.canvas_kind == "ingest-media":` branch yielding a placeholder Static (`id="library-ingest-canvas-placeholder"`, copy `Ingest canvas arrives in the next task.`) that Task 4 replaces.
- Removals: `_import_export_workflow_rows` (`:1751`), `_import_export_inspector_rows` (`:1808`), the `_active_mode == "import-export"` action-panel branch (`:2759`), handlers `open_ingest_from_import_export`/`open_media_from_import_export` (`:7979-7986`) and the import-export `#library-open-*` dead handler; the `"import-export"` entries stay in `LIBRARY_MODES`/`LIBRARY_MODE_TO_ROW_ID` until Task 9 ONLY if removing them breaks nav-context validation tests — otherwise remove now (grep `import-export` in Tests/ first; re-anchor `test_library_import_export_dedicated_import_action_emits_ingest_route` in `test_destination_shells.py:1679` — the row now selects a canvas instead of emitting the route; preserve equivalent coverage: pressing the row mounts the placeholder).
- [ ] **Step 1: Failing pure tests:** `ingest-import-media` → `canvas_kind == "ingest-media"`; the ingest section has exactly ONE row; no row with id `ingest-import-export` exists. Re-anchor `test_library_shell_state.py:43` and `:114-120`.
- [ ] **Step 2: Failing pilot:** pressing `#library-row-ingest-import-media` mounts `#library-ingest-canvas-placeholder` (poll).
- [ ] **Step 3: Run to fail. Step 4: Implement (deletions + flip + stub). Step 5: Run the three files green.**
- [ ] **Step 6: Commit** — `feat(library): ingest rail row becomes a canvas; remove placeholder Import/Export row`.

---

### Task 4: Ingest canvas — pure state + widget + form handlers

**Files:**
- Create: `tldw_chatbook/Library/library_ingest_state.py` (pure), `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (render-from-state)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (compose branch replaces the Task 3 stub; handlers), `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ build_css, commit both)
- Test: `Tests/Library/test_library_ingest_state.py`, `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: registry snapshots (`jobs()`, `counts()`); `validate_path_simple` (same import the notes-export path uses); `FileOpen` modal (notes-import precedent); `escape_markup` for any user text in labels.
- Produces (pure): `build_library_ingest_state(jobs, *, form, runtime_source, media_db_available, now) -> LibraryIngestCanvasState` with: header `Import media`; when `runtime_source == "server"` a muted line `ingest runs on Local`; when media_db unavailable a blocked line `Media database is unavailable.` + Start disabled; form echo state; `queue_rows: tuple[IngestQueueRow, ...]` where each row carries `glyph` (`●` running/queued, `✓` done, `✗` failed), `line` (binding formats: running `● running · {basename}` [+ ` · {detected_type}` when known]; queued `● queued · {basename}`; done `✓ done · {basename} · {elapsed}` where elapsed is `Ns`/`Nm Ss` from started→finished; failed `✗ failed · {basename} · {error}`), `can_open` (done + media_id), `can_retry` (failed), `job_id`. Queue section heading `Queue` + per-state counts line (reuse `count_noun` from `library_notes_sync_state` if importable without cycle, else a local twin).
- Produces (widget): stacked in a `VerticalScroll`: header → server/db quiet lines → `Input#library-ingest-path` (placeholder `Path to a local file…`) → `Button#library-ingest-browse` (`Browse…`) → `Input#library-ingest-title` / `#library-ingest-author` / `#library-ingest-keywords` (comma-separated) → `Collapsible("Advanced options", id="library-ingest-advanced", collapsed=True)` containing toggle Buttons `#library-ingest-analyze-toggle` (`○ Analyze after ingest` / `✓ …`) and `#library-ingest-chunk-toggle` (`○ Chunk content` / `✓ …`) + `Input#library-ingest-chunk-size` (default "500", shown state only) → `Button#library-ingest-start` (`Start ingest`, class `console-action-primary`-mirrored, disabled when path empty or db unavailable) → `Queue` section: per-row Static (`id=library-ingest-row-{i}`, user text escaped) + action Buttons `library-ingest-open-{i}` (`Open in Library`, only when can_open) / `library-ingest-retry-{i}` (`Retry`, only when can_retry).
- Produces (screen): `_library_ingest_form` dict-or-dataclass field + reset in `_select_library_rail_row`'s entry discipline; handlers — Browse pushes `FileOpen` (mirror the notes-import call incl. its worker/callback shape) writing the picked path into the form + recompose; toggles flip form fields + recompose; Start: `validate_path_simple(raw_path)` → invalid → `notify(..., severity="warning")` + return; valid → `self.app_instance.submit_library_ingest_job(source_path=..., title=self._safe_text(...), author=self._safe_text(...), keywords=<split/strip/comma>, perform_analysis=..., chunk_enabled=..., chunk_size=<int, clamp 100..5000, default 500>)` → clear the path field (keep metadata fields) → recompose; `Open in Library` → `await self._open_library_item_by_id("media", str(job.media_id))`; `Retry` → `app.retry_library_ingest_job(job_id)` + recompose. Index parsing via the existing `_trailing_index` helper.
- [ ] **Step 1: Failing pure tests** (row line formats verbatim incl. glyphs and elapsed; can_open/can_retry gating; server + db-unavailable lines; escape safety happens at widget layer — state carries raw).
- [ ] **Step 2: Failing pilots:** (a) full happy path with a REAL tmp .txt + real file-backed MediaDatabase on the harness app: type path → Start → poll until the row reaches `✓ done` → `Open in Library` lands in the media viewer with the ingested title; (b) invalid path → notify, no job; (c) failed job (missing file, submit programmatically) renders `✗ failed` + Retry; Retry appends a queued job; (d) db-unavailable state disables Start with the exact copy; (e) markup-hostile filename (`weird [/bracket].txt`) renders without MarkupError.
- [ ] **Step 3: Run to fail. Step 4: Implement (+ CSS + build_css). Step 5: Run green.**
- [ ] **Step 6: Commit** — `feat(library): in-Library ingest canvas over the job queue`.

---

### Task 5: Live updates — registry listener → canvas refresh + media count poke

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: Task 1's `add_listener`/`remove_listener` (listeners already fire on the UI thread because ALL mutations are marshaled there).
- Produces: LibraryScreen registers a listener `on_mount` and removes it `on_unmount`; the listener (a) recomposes when the ingest canvas is the active canvas (state-change granularity is acceptable v1 — jobs are single-file; document that targeted-row updates become worthwhile only when per-file progress exists), (b) on any transition INTO `done`, kicks `_refresh_local_source_snapshot()` (rail media count updates) — dedupe: track last-seen done-count, only poke when it grows; (c) is a no-op when the screen isn't mounted (guard `self.is_mounted`).
- [ ] **Step 1: Failing pilots:** (a) with the ingest canvas open, a programmatic submit → row appears without user interaction (poll); job completes → row flips to done AND the rail `Media (N)` count increments (poll both); (b) listener removed on screen unmount — submit after unmount does not raise (drive via harness: unmount screen, mutate registry, assert no exception and no stray recompose calls); (c) completing a job while a DIFFERENT canvas is open does not recompose it away from the user's canvas (assert selected row unchanged, no ingest widgets mounted) but still pokes the snapshot refresh.
- [ ] **Step 2: Run to fail. Step 3: Implement. Step 4: Run green.**
- [ ] **Step 5: Commit** — `feat(library): live ingest queue updates and post-ingest count refresh`.

---

### Task 6: Home integration — Running feed, failure mirror, one-hop routing

**Files:**
- Modify: `tldw_chatbook/Home/active_work_adapter.py`, `tldw_chatbook/UI/Screens/home_screen.py`, `tldw_chatbook/app.py`, `tldw_chatbook/UI/Screens/library_screen.py` + `tldw_chatbook/Constants.py` (nav-context key)
- Test: `Tests/Home/test_active_work_adapter.py`, `Tests/Home/test_dashboard_state.py` (only if a pure change lands), `Tests/UI/test_home_screen.py`, `Tests/UI/test_library_shell.py` (nav-context)

**Interfaces:**
- Consumes: registry snapshots; Home categorization is AUTOMATIC — `running`/`queued` land in Running, `failed` in Needs Attention (`dashboard_state.py:33-55`) — NO dashboard_state changes needed for the basic feed.
- Produces:
  - Adapter: ctor param `ingest_jobs_provider: Callable[[], tuple] | None = None` (appended last); `_local_ingest_job_items()` mapping registry jobs → `HomeActiveWorkItem(item_id=f"local:ingest:{job_id}", title=<basename, markup-safe plain text>, source="Library", status=job.state.value, detail_route="library", console_available=False, updated_at=<iso from started/finished>)`, INCLUDING only running/queued/failed jobs (done jobs stay out of active work — v1; note in docstring); spliced into `build_dashboard_input`'s `active_work_items`.
  - `handle_control`: new branch for `target_id.startswith("local:ingest:")` returning `HomeControlResult(HANDLED, target_route="library", target_id=...)`.
  - `app.open_active_home_item_details` (`app.py:1832-1848`): when the HANDLED result's `target_route == "library"` and target is an ingest item, post `NavigateToScreen("library", {LIBRARY_NAV_CONTEXT_INGEST: True})` (mirror the subscriptions staging special-case shape already in that method).
  - Nav-context: `LIBRARY_NAV_CONTEXT_INGEST = "ingest_media"` in `Constants.py`; `_apply_navigation_context_state` branch setting `_library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA` + recompose (mirror the notes_create branch; flush discipline comes free from the existing dispatcher).
  - Home refresh: registry reads are in-memory and synchronous — call the provider directly in `build_dashboard_input` (no caching/thread hop needed; unlike the DB-backed due-count there is NO in-memory-SQLite hazard). Wire `ingest_jobs_provider=lambda: self.library_ingest_jobs.jobs()` at adapter construction in app.py.
- [ ] **Step 1: Failing adapter tests:** provider with one running + one queued + one failed + one done job → items for the first three only, correct statuses/routes/titles; no provider → no items; `handle_control` routes ingest ids to library.
- [ ] **Step 2: Failing pilots:** (a) Home with a running ingest job → Running section shows `● {basename}` sourced `Library`; (b) failed job → Needs Attention row; selecting it + `Open details` control → navigation posted to library with the ingest context (assert via the recorded NavigateToScreen or the staged context, mirroring how the L3a flashcards pilot asserted `pending_study_initial_section`); (c) Library screen receiving `{LIBRARY_NAV_CONTEXT_INGEST: True}` lands on the ingest canvas (poll for `#library-ingest-path`).
- [ ] **Step 3: Run to fail. Step 4: Implement. Step 5: Run green.**
- [ ] **Step 6: Commit** — `feat(home): library ingest jobs feed Running and Needs Attention`.

---

### Task 7: Whole-track verification gate (LEAD-EXECUTED)

- [ ] Full local gate: `Tests/Library/ Tests/Home/ Tests/UI/test_library_shell.py Tests/UI/test_home_screen.py Tests/UI/test_destination_shells.py Tests/UI/test_product_maturity_gate16_library_search_rag.py` + `python -c "import tldw_chatbook.app"`.
- [ ] Whole-branch review (most capable model) over Tasks 0–6 BEFORE starting the cleanup tasks — the cleanup diff is noisy and must not obscure ingest-track defects.

---

### Task 8: Legacy remap — live `_active_mode` guards + handoff canvas kind

**Files:**
- Modify: `tldw_chatbook/Library/library_shell_state.py`, `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/Library/test_library_shell_state.py`, `Tests/UI/test_library_shell.py`, `Tests/UI/test_destination_shells.py`

The inventory's live-consumer map (binding):
- Search guards `:7607, 7686, 7695`: `_active_mode == "search"` → re-express as `self._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH` (equivalent post-L3a; verify the row id is set before any await on every path that can reach these guards — it is, `_select_library_rail_row` sets it synchronously).
- Collections lazy-load `:638, 782, 4691` → `_library_selected_row_id == LIBRARY_ROW_BROWSE_COLLECTIONS`.
- Study handoff: flip `create-study/flashcards/quizzes` rows to `target_kind="handoff"` (new kind; resolution `canvas_kind="handoff"`, `canvas_target=target_id` mirroring the mode branch at `:241-245`); `compose_content` gains `elif shell.canvas_kind == "handoff":` yielding the title/description/next-action trio sourced from `LIBRARY_STUDY_HANDOFF_MODES` (NOT `LIBRARY_MODES`) + `_study_handoff_detail_widget(shell.canvas_target)`; parameterize `_study_handoff_detail_widget` and `_study_handoff_copy` (kind arg replaces `_active_mode` reads at `:2149, 2535`). Open handlers (`:8051-8061`) unchanged.
- Nav-context (`:729, 734`): replace the `LIBRARY_MODES` validation + `LIBRARY_MODE_TO_ROW_ID` with a dedicated `LIBRARY_NAV_MODE_TO_ROW_ID` table containing exactly the modes nav-context supports (notes, conversations, media, search, collections — grep callers of `open_notes_workspace`/nav emitters to confirm the live set) — same behavior, no LIBRARY_MODES dependency.
- `_active_mode` writes: after the above, remove the reads first, then drop now-redundant writes EXCEPT keep the field itself if `_compose_mode_canvas` still exists (it dies in Task 9; sequence: Task 8 removes every LIVE read, Task 9 deletes the dead machinery + the field).
- [ ] **Step 1: Re-anchor + failing tests:** `test_destination_shells.py:1594-1718` `_active_mode` assertions → `_library_selected_row_id`/`canvas_kind`; new pure tests for the `handoff` kind; pilot: create-flashcards row renders the handoff canvas with the flashcards open button; search stale-outcome guard still drops (existing mid-flight pilots re-run green).
- [ ] **Step 2: Run to fail. Step 3: Implement. Step 4: Run green** (the three files + gate16).
- [ ] **Step 5: Commit** — `refactor(library): remap live mode guards; study rows become handoff canvases`.

---

### Task 9: Legacy deletion — LIBRARY_MODES, chips, dead chrome, CSS

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss` + generated, `tldw_chatbook/css/core/_variables.tcss` (via its source module), `Tests/UI/test_master_shell_design_system_contract.py`, `Tests/UI/test_non_obscuring_focus_contract.py`, any test grepping mode machinery

Deletions (inventory-mapped; verify each is dead by grep before deleting):
- `LIBRARY_MODES` (`:176-243`), `LIBRARY_MODE_BY_BUTTON_ID`, `LIBRARY_MODE_TO_ROW_ID`, `LIBRARY_COLUMN_TITLES`, `LIBRARY_LOCAL_SNAPSHOT_MODES`; `_active_mode` field + `_set_active_mode` + `switch_library_mode` + `_legacy_workbench_present` + `_refresh_active_mode_widgets` + the `_sync_*` legacy helpers (`:6599-6970` EXCEPT `_sync_collections_panel`'s live branch — keep that method, strip only its legacy branch) + dead builders (`_library_action_widgets` `:2703`, `_source_module_action_widgets` `:1860`, `_hub_inspector_rows` `:1960`, `_active_mode_contract`/`_active_column_titles`/`_active_source_action_id`/`_should_show_local_snapshot_region` `:2236-2249`) + dead `#library-open-*` handlers (`:7881, 8025-8038` — NOT the study trio) + `_compose_mode_canvas` + the `canvas_kind == "mode"` dispatch branch + the legacy DEFAULT_CSS region ids (`:433-459`) and mode-chip CSS (`:400-412`).
- CSS: `.library-mode-chip` rules + `$ds-library-mode-chip-*` vars from the source modules; `./build_css.sh`; commit both. Update the two contract tests (`test_master_shell_design_system_contract.py:166-190`, `test_non_obscuring_focus_contract.py:680-699`) — the retired selector must now be asserted ABSENT (mirror how those tests treat other retired selectors; do not delete the tests).
- `LIBRARY_STUDY_HANDOFF_MODES` STAYS (Task 8 made it the handoff copy table) — rename to `LIBRARY_STUDY_HANDOFF_COPY` if trivial, else leave.
- [ ] **Step 1:** grep-audit every symbol above across tldw_chatbook/ AND Tests/ — list remaining references, migrate/delete tests first (RED where behavior contracts move).
- [ ] **Step 2: Delete. Step 3: Full run:** the whole L3b gate set + `Tests/UI/test_library_content_hub.py` + contract tests + import check.
- [ ] **Step 4: Commit** — `refactor(library): retire LIBRARY_MODES and legacy workbench chrome`.

---

### Task 10 (LEAD-EXECUTED): Live QA + evidence + gate

- [ ] Full gate (all suites from Tasks 7/9 + Home + gate16).
- [ ] Seeded textual-serve at 2050x1240 dsf1 (`/private/tmp/tldw-l3b-qa` HOME; seed notes/media/conversations + study data as L3a did). Required captures: ingest canvas idle (form + empty queue); Browse modal open; a running job row; done row with `Open in Library`; failed row with `Retry`; Open-in-Library landed in the media viewer on the ingested item; rail `Media (N)` count grown after ingest; Home Running with an active ingest job; Home Needs Attention with a failed job + routing landed on the ingest canvas; handoff canvases (study/flashcards/quizzes) post-cleanup; server-scope quiet line (flip runtime to server in config if feasible, else document why not captured).
- [ ] QA README at `Docs/superpowers/qa/library-l3b-2026-07/`: captures, v1 limits, follow-ups (server ingest wiring bug in TAB_INGEST's Server Jobs tab; bulk Library export; ingest parallelism + persistent history; URL/web local ingest).
- [ ] STOP at the user gate with the **TAB_INGEST deprecation decision** explicitly posed: (a) keep the Ingest nav tab (server Sources/Web Clipper remain reachable; Library canvas is the local path) — recommended; (b) retire the tab now and accept losing the server-mode scaffolding until a server-ingest phase. NO merge without explicit approval.

---

## Task order & dependencies

0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 (ingest track, strictly ordered) → 8 → 9 (cleanup; splittable to L3c per spec pre-authorization) → 10. Single implementer at a time.

## Self-review notes

- Spec coverage: rail flip (T3), inventory-decided seam + Task-0 smoke (T0/T2), form (T4, with two inventory-justified deviations: no type button, no URL — documented at the gate), queue rows with real signals only (T4), job registry architecture exactly per spec (T1/T2: UI-thread mutations, one exclusive-group runner, FIFO, in-memory v1 limits stated), Home Running + failures→Needs Attention with detail_route to the ingest canvas (T6), import-export row removal per the spec's own rule (T3), LIBRARY_MODES retirement + handoff kinds (T8/T9), server scope = local-only + quiet line (T4), deprecation gated on the user (T10).
- Deviations flagged for the gate: TAB_INGEST stays routed pending the user's deprecation decision; one-file-per-job (queue = batching); no type/URL controls.
- Type consistency: `LibraryIngestJob`/`IngestJobState`/registry method names used identically across T1/T2/T4/T5/T6; canvas id `ingest-media`; row id constant `LIBRARY_ROW_INGEST_MEDIA`; nav key `LIBRARY_NAV_CONTEXT_INGEST`.
