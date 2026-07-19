# F3 — Parallel-parse ingest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Spec: `Docs/superpowers/specs/2026-07-10-library-f3-parallel-parse-design.md` (read it per task brief; it is the authority on architecture decisions). Anchors exact at branch point 5760b7c8; grep symbols, lines drift.

**Goal:** Library ingest parses files in a small spawn-context process pool and persists parsed payloads through the existing single writer thread — SQLite never sees a concurrent ingest writer.

**Architecture:** Four tasks: (1) import-chain slimming so parse workers and app boot stop paying for torch/docling/nltk they don't use; (2) parse/persist split + light worker entry module; (3) registry states `PARSING`/`WRITING` + full `"running"`-literal sweep; (4) coordinator + writer rewire in the app mixin with pool lifecycle. UI copy changes in task 3/4 require visual QA before merge.

**Tech Stack:** Python ≥3.11, `multiprocessing.get_context("spawn").Pool` (NOT ProcessPoolExecutor — spec explains why), Textual workers, pytest.

## Global Constraints

- Stage only changed files by explicit path; NEVER `git add -A`. Never touch `.claude/settings.local.json`. Bare `git stash` FORBIDDEN (shared stack — WIP commit instead).
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- RED-first for behavior changes; bounded polls in pilots (`range(150)`/`pause(0.02)`); gated fakes bound waits at 30.0s.
- Registry mutations UI-thread-only; worker threads/pool callbacks marshal via `self.call_from_thread`; frozen dataclasses mutate via `dataclasses.replace`.
- Behavior preservation everywhere the spec doesn't explicitly change it: `batch_ingest_files`, `quick_ingest`, the server ingest path, retry/supersede/dismiss semantics, F1b permanent-failure gating, quit contract (in-flight DB write completes).
- Test command: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`.
- Config reads use the 1-arg dotted `get_cli_setting("library.ingest_parse_workers")` fallback pattern (`load_settings()` does not carry CLI `[library.*]` tables — see `_library_rail_preferences` for the template).

### Task 1: Import-chain slimming (OCR + analyze deferral) with regression guard

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/OCR_Backends.py` (`_register_backends` ~:975 — registration currently imports docling eagerly)
- Modify: `tldw_chatbook/Local_Ingestion/PDF_Processing_Lib.py:41` (`from .OCR_Backends import ocr_manager`), `:39` (`from ..LLM_Calls.Summarization_General_Lib import analyze`)
- Modify: `tldw_chatbook/Local_Ingestion/Image_Processing_Lib.py:53` (`from .OCR_Backends import ocr_manager, OCRResult`)
- Modify: `tldw_chatbook/Local_Ingestion/Document_Processing_Lib.py:67` (`from ..LLM_Calls.Summarization_General_Lib import analyze`)
- Test: new `Tests/Local_Ingestion/test_ingest_import_weight.py`

**Interfaces produced:** none new — same public functions, lighter imports.

Requirements:
- `OCR_Backends`: backend classes register as lazy descriptors — the docling (and any other heavy) import happens on the first actual OCR invocation of that backend, not at module import or registration. Keep the registry API (`ocr_manager`) identical for callers. If a backend's deps are missing, the existing unavailable-backend behavior must be preserved (probe cheaply via `importlib.util.find_spec`, not by importing).
- `PDF_Processing_Lib`/`Image_Processing_Lib`: move `ocr_manager`/`OCRResult` imports into the functions that use OCR. `Document_Processing_Lib`/`PDF_Processing_Lib`: move the `analyze` import into the call sites that run when analysis is enabled.
- Regression test (subprocess-based, mirrors `Tests/…` conventions; check whether `Tests/Local_Ingestion/` exists, else place in the closest existing suite dir): spawn `sys.executable -c "import sys, time; t0=time.time(); import tldw_chatbook.Local_Ingestion.local_file_ingestion; assert time.time()-t0 < 5.0; assert not any(m.split('.')[0] in ('torch','docling','nltk','torchvision','transformers') for m in sys.modules), [m for m in sys.modules if m.split('.')[0] in ('torch','docling','nltk')][:5]"` — the module-absence assertions are the real guard; the generous 5.0s bound only catches catastrophic regressions without flaking slow CI. Also assert the measured import locally in the report (expect ≤1.5s).
- Suites: `Tests/Media/` (if it covers ingestion), whatever `ls Tests/ | grep -i ingest` finds, `Tests/UI/test_library_shell.py` ingest pilots, plus `python -c "import tldw_chatbook.app"`.

Commit: `perf(ingestion): defer OCR and analysis imports — parse chain drops torch/docling/nltk at import time`

### Task 2: Parse/persist split + light worker entry module

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py` (`ingest_local_file` :105-473 — split at the `add_media_with_keywords` call :442)
- Create: `tldw_chatbook/Local_Ingestion/ingest_parse_worker.py`
- Test: `Tests/Local_Ingestion/test_ingest_parse_worker.py` (same dir as Task 1's test)

**Interfaces produced (Tasks 3/4 rely on these exact names):**
- `parse_local_file_for_ingest(file_path: str, options: dict) -> dict` in `local_file_ingestion.py` — the pre-DB half; returns the payload dict consumed by `persist_parsed_media`.
- `persist_parsed_media(payload: dict, media_db) -> tuple[int | None, str, str]` in `local_file_ingestion.py` — the `add_media_with_keywords` half, returning (media_id, media_uuid, message) exactly as today's write does.
- `run_parse_job(file_path: str, options: dict) -> dict` in `ingest_parse_worker.py` — the pool entry point: top-level, spawn-safe, module scope imports ONLY stdlib. Returns `{"ok": True, "payload": {...}}` or `{"ok": False, "error": str, "permanent": bool}`; catches ALL exceptions (never raises across the process boundary). Permanent classification mirrors `app.py`'s `_classify_library_ingest_failure` (unsupported type / missing file) — move that logic INTO `ingest_parse_worker.py` as `classify_parse_failure(exc) -> bool` and have `app.py` import it from there (single source of truth; app keeps classifying WRITE-stage failures with it too).

Requirements:
- `ingest_local_file` becomes `persist_parsed_media(parse_local_file_for_ingest(...))` composed, preserving its exact return shape and error behavior — `batch_ingest_files`/`quick_ingest`/server path untouched (their tests must stay green unmodified).
- The `options` dict carries what today's `ingest_local_file` params carry (title/author/keywords/perform_analysis/chunk options/api params) — plain picklable data only.
- Tests: unit — parse returns payload for a txt file (no DB touched: assert no media_db needed); persist writes the payload via a real in-memory MediaDatabase and returns media_id; run_parse_job returns ok=True payload for txt, ok=False+permanent=True for a missing file and for an unsupported extension, ok=False+permanent=False for a corrupt PDF; `ingest_local_file` end-to-end unchanged (compose test). Integration — one real `multiprocessing.get_context("spawn").Pool(1)` test running `run_parse_job` on a small txt file through an actual subprocess (bounded join, marked with the repo's integration marker if one exists).
- Import-weight guard: `ingest_parse_worker` module import must not pull `local_file_ingestion` (assert in test via subprocess `sys.modules` check).

Commit: `refactor(ingestion): split parse from persist; spawn-safe worker entry with structured results`

### Task 3: Registry states PARSING/WRITING + "running" literal sweep

**Files:**
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py` (`IngestJobState` :69; `mark_running` :296 → split into `mark_parsing`/`mark_writing`; docstrings describing the serial contract)
- Modify: `tldw_chatbook/Library/library_ingest_state.py` (`_build_queue_row` state branches; counts line builder)
- Modify: `tldw_chatbook/Home/active_work_adapter.py` (`_HOME_INGEST_JOB_ACTIVE_STATES` ~:50)
- Modify: `tldw_chatbook/Home/dashboard_state.py` (status-category mapping if it special-cases "running")
- Test: `Tests/Library/test_library_ingest_jobs.py`, `test_library_ingest_state.py`, `Tests/Home/` re-anchors
- Grep-sweep obligation: every consumer of the `"running"` literal tied to ingest jobs (`grep -rn '"running"\|RUNNING' tldw_chatbook/Library tldw_chatbook/Home tldw_chatbook/app.py Tests/Library Tests/Home Tests/UI/test_library_shell.py Tests/UI/test_home_screen.py` — adjudicate each hit: ingest-job related → migrate; other subsystems (watchlist runs, server jobs) → leave).

**Interfaces produced:** `IngestJobState.PARSING = "parsing"`, `IngestJobState.WRITING = "writing"` (RUNNING removed); `registry.mark_parsing(job_id) -> LibraryIngestJob | None` (QUEUED→PARSING), `registry.mark_writing(job_id) -> LibraryIngestJob | None` (PARSING→WRITING). `started_at` stamps at PARSING. Invalid transitions return None (existing pattern).

Requirements:
- Queue rows: "parsing" and "writing" state labels; counts line covers all states present (e.g. `2 parsing · 1 writing · 3 queued · 1 done · 1 failed` — same `·` join and ordering convention as the existing counts builder, states listed in pipeline order).
- Home: PARSING and WRITING both map into the Running feed; DONE/FAILED behavior unchanged; `retry_available`/`status_detail` untouched.
- The registry's quit-contract docstring (~:14-20) must be rewritten for the new pipeline (writer joins; parses terminated).
- RED-first on the new transitions; re-anchor existing "running" asserts without weakening.
- NOTE: at this task's commit the app still calls `mark_running` — keep a thin `mark_running = mark_writing`-style alias ONLY if needed to keep the branch green mid-stack, and Task 4 MUST delete it (the plan's final gate greps for `mark_running` and expects zero hits).

Commit: `refactor(library): ingest job states PARSING/WRITING replace RUNNING across registry, queue, and Home`

### Task 4: Coordinator + writer rewire, pool lifecycle, quit path

**Files:**
- Modify: `tldw_chatbook/app.py` — `LibraryIngestQueueMixin` (~:1150-1480: `_run_library_ingest_queue` :1372, `_claim_next_ingest_job_or_release` :1307, `_release_ingest_runner_after_crash`, `submit_library_ingest_job`, `retry_library_ingest_job` :1256, quit/unmount hooks)
- Test: `Tests/UI/test_library_shell.py` (ingest pilots), new coordinator unit file `Tests/Library/test_library_ingest_coordinator.py` if the mixin logic is extractable — otherwise pilots + app-level tests
- Visual QA prep: none in-code; captures happen at the gate

**Interfaces consumed:** Task 2's `run_parse_job`/`persist_parsed_media`/`classify_parse_failure`; Task 3's states/transitions.

Requirements:
- Coordinator state on the mixin: `_ingest_parse_pool` (lazy spawn-context Pool), `_ingest_parsed_payloads: dict[str, dict]`, `_ingest_shutdown: bool`. Pool size: `get_cli_setting("library.ingest_parse_workers")` int-coerced, invalid/missing → `min(3, max(1, os.cpu_count() - 1))`.
- Submission: after `submit_library_ingest_job`/`retry_library_ingest_job` enqueue, and after every parse completion, top up: while jobs in QUEUED and count(PARSING) < N → `mark_parsing`, `pool.apply_async(run_parse_job, (path, options), callback=..., error_callback=...)`. Callbacks marshal via `call_from_thread`; if `_ingest_shutdown`, they return without touching the app.
- Parse completion (UI thread): ok → store payload, wake writer (start the exclusive writer worker if not running — same `group="library_ingest_queue"` worker as today); not ok → `mark_failed(error, permanent)`.
- error_callback (unexpected pool-level exception, e.g. terminated pool): mark ALL PARSING jobs failed retryable with a broken-pool message, drop `_ingest_parse_pool` to None (lazy rebuild on next submission).
- Writer loop: claims the OLDEST payload-ready job in submission order (`_claim_next_ingest_job_or_release` reworked: claim = pop payload + `mark_writing`, atomically on the UI thread via `call_from_thread`; release when no payload-ready job). Write via `persist_parsed_media` incl. the existing `get_media_by_url` re-ingest fallback; `mark_done`/`mark_failed` as today.
- Quit path (find the app's unmount/exit hook where workers are joined): set `_ingest_shutdown = True`, `pool.terminate()` + `pool.join()`; writer thread joins as today.
- Delete Task 3's `mark_running` alias if it exists; delete the old in-runner parse code path.
- Fake-pool seam: a `_create_ingest_parse_pool()` factory method tests monkeypatch to an inline-synchronous fake (runs `run_parse_job` in-process, invokes callback immediately). Pilots: two files queued → both reach DONE; permanent parse failure → no Retry (F1b M4 pilot re-anchored); broken-pool simulation → PARSING jobs → FAILED retryable + pool rebuilt on next submit; quit-path unit (shutdown flag stops callbacks).
- One real-pool end-to-end test (marked): queue one txt file through a REAL spawn pool inside the app harness OR reuse Task 2's integration test + a pilot with the fake pool — implementer's choice, justified in the report.

Commit: `feat(library): parallel parse pool + single-writer persistence for the ingest queue`

## Verification & gate

- Final greps: `grep -rn "mark_running\|IngestJobState.RUNNING" tldw_chatbook Tests` → nothing; import-weight regression test green.
- Combined gate: `Tests/Library/ Tests/Home/ Tests/Local_Ingestion/ Tests/Media/ Tests/UI/test_library_shell.py Tests/UI/test_home_screen.py Tests/UI/test_destination_shells.py` + gate16 + content hub + `import tldw_chatbook.app`.
- Visual QA (served TUI, seeded HOME, 2050×1240 recipe): queue canvas showing parsing+writing+queued simultaneously (bulk-submit 4+ files incl. one slow), Home Running feed with parsing/writing rows, permanent-failure row unchanged. Present captures for user approval; PR to dev; merge only on explicit user authorization.
- App-boot side benefit: measure `import tldw_chatbook.app` wall time before/after Task 1 and report it in the PR.
