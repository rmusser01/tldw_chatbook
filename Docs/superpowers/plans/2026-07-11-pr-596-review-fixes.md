# PR 596 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct every verified PR 596 review finding while preserving the approved Library ingest, navigation, and UI architecture.

**Architecture:** Keep the spawn `multiprocessing.Pool` and single writer, adding pool generations and sentinel monitoring for real worker-death containment. Preserve all user edits through existing fresh-screen state seams, sanitize plain-text FTS input at raw FTS boundaries, and restore automatic parser fallback behavior. Test-only changes isolate config writes and retire obsolete CSS assertions.

**Tech Stack:** Python 3.11+, Textual, `multiprocessing`, SQLite FTS5, pytest.

**Spec:** `Docs/superpowers/specs/2026-07-11-pr-596-review-fixes-design.md`

**Backlog:** `backlog/tasks/task-152 - Address-PR-596-verified-review-findings.md`

**ADR required:** no  
**ADR path:** N/A  
**Reason:** Regression fixes implement existing pool/single-writer and fresh-screen state contracts; no new boundary, dependency, schema, or policy is introduced.

---

### Task 1: Contain ingest pool death, stale callbacks, and shutdown

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`
- Modify or create: `Tests/Library/test_library_ingest_pool_lifecycle.py`

- [ ] **Step 1: Write failing generation and shutdown tests**

Add tests proving:

```python
def test_stale_pool_generation_error_does_not_fail_replacement_job(): ...
def test_stale_pool_generation_success_does_not_enqueue_replacement_payload(): ...
def test_shutdown_refuses_to_claim_another_ready_payload(): ...
```

Use existing fake-pool and registry helpers. The first two tests create generation A, fail it, retry into generation B, then fire A's delayed error and success callbacks; generation B must remain `PARSING` with no stale payload. The shutdown test sets `_ingest_shutdown = True`, calls `_claim_next_ingest_job_or_release`, and asserts `None` plus `runner_active is False`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_ingest_runner.py -q --tb=short
```

Expected: stale callback fails the replacement job and shutdown still claims a payload.

- [ ] **Step 3: Bind pool generations to submissions and callbacks**

Add coordinator state for a monotonic generation and `generation -> set[job_id]`. Bind generation and job ID through `functools.partial` for both success and error callbacks. Both success handling and `_handle_broken_ingest_parse_pool` validate the current generation and membership. Broken-pool handling filters recorded members to jobs still in `PARSING`, clears the generation membership, detaches and terminates only the affected pool, and leaves `_ingest_parse_pool = None` for lazy rebuild.

- [ ] **Step 4: Add a worker-sentinel monitor**

After creating a real Pool, snapshot its worker processes and start one daemon monitor thread using `multiprocessing.connection.wait` on their sentinels. If a sentinel becomes ready before the generation stop event or shutdown flag, marshal a retryable broken-pool error for that generation. Stop the generation monitor before intentional terminate/join. Keep the private Pool worker-list access in one small helper and fail pool creation closed if no worker sentinels can be obtained.

- [ ] **Step 5: Add a real abrupt-exit regression**

Use a module-level picklable target calling `os._exit(17)` in a real spawn Pool. Bound the wait and assert the monitor reports the failed generation, only generation-owned `PARSING` jobs become retryable failures, the affected pool is terminated/cleared, and retry lazily constructs a replacement generation. Always terminate/join the pool in cleanup.

- [ ] **Step 6: Guard shutdown claims**

At the start of `_claim_next_ingest_job_or_release`, when `_ingest_shutdown` is true, clear the runner-active flag and return `None`. This lets an already-claimed write finish but prevents another claim.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_ingest_runner.py Tests/Library/test_library_ingest_pool_lifecycle.py Tests/Library/test_library_ingest_jobs.py -q --tb=short
```

Expected: all pass, including a real `os._exit` worker.

### Task 2: Make supported launchers spawn-lightweight

**Files:**
- Create: `tldw_chatbook/cli.py`
- Create: `tldw_chatbook/__main__.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify or create: `Tests/Local_Ingestion/test_ingest_spawn_bootstrap.py`

- [ ] **Step 1: Write a failing spawn-bootstrap probe**

The test launches the supported CLI bootstrap as the parent main module, creates a spawn child, and has the child report whether `tldw_chatbook.app`, `torch`, or `transformers` were imported before the target runs.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Local_Ingestion/test_ingest_spawn_bootstrap.py -q --tb=short
```

Expected: current app entry point preloads the application/heavy modules.

- [ ] **Step 3: Add lightweight launch modules**

`tldw_chatbook/cli.py` exposes `main_cli_runner()` and imports `tldw_chatbook.app.main_cli_runner` only inside that function. `tldw_chatbook/__main__.py` imports the lightweight CLI function and calls it only under `if __name__ == "__main__"`. Point `tldw-cli` at `tldw_chatbook.cli:main_cli_runner`. Update the primary README development command to `python3 -m tldw_chatbook`; keep direct app-module execution documented only as a legacy/debug path if retained.

- [ ] **Step 4: Verify GREEN and CLI behavior**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Local_Ingestion/test_ingest_spawn_bootstrap.py Tests/unit/test_core_imports_unit.py -q --tb=short
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.cli; assert 'tldw_chatbook.app' not in __import__('sys').modules"
```

Expected: lightweight import passes and spawned probe reports no app/Torch/Transformers preload.

### Task 3: Preserve edits across Library, app, and Settings navigation

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Add failing navigation regressions**

Add tests proving:

```python
async def test_navigation_flush_exception_vetoes_switch(): ...
async def test_failed_note_save_keeps_editor_open_and_dirty(): ...
def test_settings_state_round_trips_search_category_and_drafts(): ...
```

Exercise the complete destructive-transition set that currently flushes and then checks only `conflict`: `_apply_navigation_context_after_flush`, `_select_library_rail_row`, `handle_library_notes_sync_open`, `handle_library_note_row`, `handle_library_note_back`, `handle_library_note_delete`, and the media/notes branches of `_open_library_item_by_id`. Use a save seam that raises and assert each path leaves the dirty editor state and selection unchanged.

- [ ] **Step 2: Verify RED**

Run the three exact new tests. Expected: app switches after flush error, Library leaves editor after error, and Settings loses drafts.

- [ ] **Step 3: Fail app navigation closed**

After logging/notifying a raised `flush_pending_work`, return before `save_state`, screen creation, or `switch_screen`.

- [ ] **Step 4: Gate every destructive Library transition on the dirty flag**

Replace post-flush checks that special-case only `conflict` in `_apply_navigation_context_after_flush`, `_select_library_rail_row`, `handle_library_notes_sync_open`, `handle_library_note_row`, `handle_library_note_back`, `handle_library_note_delete`, and both destructive `_open_library_item_by_id` branches with the invariant: if `_library_note_dirty` remains true, leave the editor and selection unchanged. Reuse one small helper if and only if it reduces duplicated conditions across all identified handlers.

- [ ] **Step 5: Persist Settings drafts through screen state**

Override `save_state`/`restore_state`, calling `super()`, and round-trip validated `active_category`, sanitized `category_search_query`, and `copy.deepcopy(_settings_drafts)`. Ignore malformed external state rather than throwing. State remains process-local and is not written to config.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_screen_navigation.py Tests/UI/test_library_shell.py Tests/UI/test_settings_configuration_hub.py -q --tb=short
```

Expected: all pass and no edit-loss path remains.

### Task 4: Harden plain-text FTS queries and Docling fallback

**Files:**
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py`
- Modify: `Tests/Library/test_library_local_rag_search_service.py`
- Modify: `tldw_chatbook/Local_Ingestion/Document_Processing_Lib.py`
- Modify: `Tests/Local_Ingestion/test_ocr_backend_self_heal.py` or closest document-processing test

- [ ] **Step 1: Write failing public-behavior tests**

Seed real local records and assert public keyword search matches queries containing `?`, `foo-bar`, unmatched/embedded quotes, and a multiword query whose terms are not adjacent. The multiword assertion preserves the existing implicit-AND behavior. Add a document-processing test where Docling is discoverable but its deferred import raises; automatic mode must call the native parser, explicit Docling must return the Docling failure.

- [ ] **Step 2: Verify RED**

Run the exact tests. Expected: punctuation returns no result and automatic broken Docling does not call native.

- [ ] **Step 3: Quote only raw FTS boundaries**

Create a small pure helper that splits plain input on whitespace and emits double-quoted FTS phrases with embedded quotes doubled. Apply it only to seams that pass directly to SQLite FTS `MATCH`; do not quote service seams that already accept plain text.

- [ ] **Step 4: Restore automatic native fallback**

Remember whether the requested method was `auto`. If deferred Docling loading raises `ImportError`/dependency-load failure in automatic mode, log the availability failure and run the existing native branch. Explicit `docling` keeps its current error result.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_local_rag_search_service.py Tests/Local_Ingestion -q --tb=short
```

Expected: all pass.

### Task 5: Repair test isolation and retired CSS contract

**Files:**
- Modify: `Tests/conftest.py`
- Modify: `Tests/UI/test_non_obscuring_focus_contract.py`

- [ ] **Step 1: Lock in isolated config path**

Extend `isolate_test_environment` to set `TLDW_CONFIG_PATH` to `test_data_dir / "config" / "config.toml"`. Assert a save resolves under `tmp_path`, never `DEFAULT_CONFIG_PATH`.

- [ ] **Step 2: Remove obsolete Notes mode-strip assertions**

Update the focus-contract test to assert both retired `.library-mode-chip` and `.notes-mode-chip` selectors are absent. Leave active Personas mode-strip coverage to its existing screen-specific tests.

- [ ] **Step 3: Verify the previously failing slice**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py::test_library_shell_notes_sync_direction_cycles_and_persists Tests/UI/test_library_shell.py::test_library_shell_notes_sync_conflict_cycles_and_persists Tests/UI/test_library_shell.py::test_library_shell_notes_sync_auto_toggle_flips_and_persists Tests/UI/test_library_shell.py::test_library_shell_notes_sync_folder_typing_does_not_write_config Tests/UI/test_library_shell.py::test_library_shell_notes_sync_run_persists_validated_folder Tests/UI/test_non_obscuring_focus_contract.py::test_library_mode_chip_selector_is_retired_from_focus_contracts -q --tb=short
```

Expected: 6 passed without writing outside pytest temp directories.

### Task 6: Final verification and PR closeout

**Files:**
- Modify: `backlog/tasks/task-152 - Address-PR-596-verified-review-findings.md`

- [ ] Run focused combined suites for Library, Home, Local Ingestion, navigation, Settings, Prompts DB, and affected UI contracts.
- [ ] Run `python -m compileall -q tldw_chatbook` and `git diff --check`.
- [ ] Review the complete diff for regressions and unnecessary scope.
- [ ] Add concise Implementation Notes, check all TASK-152 acceptance criteria, and mark it Done only after every gate passes.
- [ ] Commit the complete repair, rebase/fetch-check against `origin/dev`, push the commit to `dev`, and recheck PR #596 review threads and GitHub Actions.
