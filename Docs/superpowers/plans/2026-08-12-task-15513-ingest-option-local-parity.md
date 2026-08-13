# TASK-15513 Ingest Option Local Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose honest high-value ingestion controls across Local and Server while keeping Keep original file Server-only.

**Architecture:** Extend the existing capability-driven generic options panel and keep `LibraryIngestJob.ingest_options` as the snapshot boundary. Project shared values explicitly into the Local writer and Server request builder. Add a context-local RAG-hook suppression scope so a Local Generate embeddings opt-out affects only that ingest and never changes authoritative media persistence or concurrent jobs.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest, SQLite, contextvars, existing Library ingest capability/state/job seams.

## Global Constraints

- Use the capability schema as the single source of defaults and backend visibility.
- Keep original file is rendered only for Server ingestion.
- Visible controls must never be silently inert for the selected backend.
- Prompt controls are disabled with readable reasons while Analyze after import is off.
- Generate embeddings defaults on to preserve ADR-005 ingestion-time indexing behavior.
- Local source persistence remains authoritative and indexing remains best-effort per ADR-030.
- No schema migration or new dependency.
- Preserve retry snapshots, config persistence, keyboard reachability, and compact viewport containment.

---

### Task 1: Capability schema and mode-aware canvas

**Files:**

- Modify: `tldw_chatbook/Library/ingest_capabilities.py`
- Modify: `tldw_chatbook/Widgets/Library/library_ingest_canvas.py`
- Modify: `Tests/UI/test_library_ingest_canvas.py`
- Modify: `Tests/Library/test_library_ingest_state.py`

**Interfaces:**

- Consumes: `OptionField`, `TypeGroupCapabilities`, `LibraryIngestCanvasState.ingest_backend`.
- Produces: `OptionField.backends`, `field_available_for_backend(field, backend)`, and mode-correct option widgets under `#type-group-generic`.

- [ ] Write failing schema and Pilot tests asserting four shared fields in both modes, Keep original file only in Server, prompt disabled reasons, multiline prompt widgets, and compact-viewport containment.
- [ ] Run the selected tests and confirm failures name missing fields/widgets rather than harness setup.
- [ ] Add backend metadata and filter fields before title/body composition; rename the generic panel to Import behavior and add the five option declarations.
- [ ] Add TextArea composition and changed-message forwarding without disturbing existing Checkbox, Select, and Input behavior.
- [ ] Run the focused UI/state tests and confirm they pass.

### Task 2: Snapshot, persistence, and Server request projection

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Library/server_ingest_request.py`
- Modify: `Tests/App/test_submit_library_ingest_job.py`
- Modify: `Tests/Library/test_server_ingest_request.py`
- Modify: `Tests/Library/test_server_ingest_field_contract.py`

**Interfaces:**

- Consumes: `ingest_options["generic"]` from the form snapshot.
- Produces: Local parser options containing analysis prompts only when analysis is on; Server kwargs containing `overwrite_existing`, `custom_prompt`, `system_prompt`, `generate_embeddings`, and `keep_original_file` under their declared names.

- [ ] Write failing tests for snapshot round-trip, analysis prompt gating, retry preservation, Server kwargs, and the captured declared-field fixture.
- [ ] Run the selected tests and confirm the missing projections fail.
- [ ] Explicitly project generic shared fields in `_ingest_job_options` and `build_server_ingest_kwargs`; do not rely on the detected type-group loop.
- [ ] Preserve prompt text in form state while omitting it from backend kwargs when analysis is off.
- [ ] Run the focused request/snapshot tests and confirm they pass.

### Task 3: Local overwrite behavior

**Files:**

- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Local_Ingestion/test_ingest_option_wiring.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`

**Interfaces:**

- Consumes: `generic.overwrite_existing` from the claimed `LibraryIngestJob`.
- Produces: `persist_parsed_media(payload, media_db, *, overwrite_existing=False, generate_embeddings=True)` and a DB call with `overwrite=overwrite_existing`.

- [ ] Write a failing real-SQLite test that imports matching content twice with changed metadata and proves Off skips while On updates the existing row.
- [ ] Run the test and confirm On still produces the pre-feature duplicate outcome.
- [ ] Add keyword-only persistence options and pass overwrite from the app writer.
- [ ] Run the focused Local ingestion and runner tests and confirm they pass.

### Task 4: Per-ingest Local embedding control

**Files:**

- Modify: `tldw_chatbook/RAG_Search/ingestion_indexing.py`
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/RAG/test_ingestion_indexing.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`

**Interfaces:**

- Consumes: `generic.generate_embeddings` from the claimed job.
- Produces: `suppress_ingestion_indexing()` context manager whose state is consulted by `_media_post_ingest_hook` and always reset after the write.

- [ ] Write failing hook tests proving suppression affects one write, does not block SQLite persistence, resets after an exception, and does not suppress another thread.
- [ ] Write a failing runner test proving the claimed job's option reaches persistence.
- [ ] Run the tests and confirm the hook currently fires while suppression is requested.
- [ ] Implement the context-local guard and wrap only Local persistence when Generate embeddings is off.
- [ ] Run the focused RAG/runner tests and confirm they pass.

### Task 5: Verification and task closeout

**Files:**

- Modify: `backlog/tasks/task-15513 - Surface-the-high-value-server-only-ingest-options-in-server-mode.md`
- Verify: all production and test files above.

**Interfaces:**

- Consumes: completed implementation and test evidence.
- Produces: completed acceptance criteria, implementation notes, ADR links, and a review-ready branch.

- [ ] Run focused UI, Library, App, Local Ingestion, RAG, and server field-contract tests with a sandbox-writable `--basetemp`.
- [ ] Run Ruff on modified Python files, `git diff --check`, the Backlog duplicate-ID guard, and an import smoke test.
- [ ] Export a Textual rendered frame at normal and compact widths and assert the shared labels render, Keep original file is absent in Local, and Start/metadata controls remain inside the viewport.
- [ ] Mutation-check the backend filter, prompt analysis gate, overwrite kwarg, and indexing suppression guard by temporarily removing each and confirming its owning test fails, then restore the implementation.
- [ ] Update every acceptance criterion, add concise implementation notes and verification evidence, and set task-15513 Done only after all Definition of Done conditions are satisfied.

## Plan self-review

- Spec coverage: all seven acceptance criteria map to Tasks 1 through 5.
- Placeholder scan: no deferred implementation or unnamed error/test steps remain.
- Type consistency: all cross-task values remain nested under `ingest_options["generic"]`; persistence adds two keyword-only booleans; the RAG suppression API is a zero-argument context manager.
