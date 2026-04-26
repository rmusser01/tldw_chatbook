# Media Ingest Recent Job Watch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Let Chatbook watch recent server media ingest jobs via the existing unscoped SSE stream, without claiming historical batch browsing that the server does not expose.

**Architecture:** Reuse the existing media reading API, server service, and scope-service stream method because `batch_id` is already optional. Add a single remote-ingestion UI control that calls `stream_media_ingest_job_events(mode="server", batch_id=None, after_id=0)` and renders the returned snapshot/events into the existing job list/status surface. Keep known-batch lookup separate and keep historical batch discovery documented as blocked on a server endpoint.

**Tech Stack:** Python 3.12, Textual widgets, pytest/pytest-asyncio.

---

### Task 1: Lock The No-Batch Stream Contract

**Files:**
- Modify: `Tests/Media/test_server_media_reading_service.py`
- Modify: `Tests/Media/test_media_reading_scope_service.py`

- [x] **Step 1: Add service/scope tests for unscoped recent stream**

Add tests that call `stream_media_ingest_job_events(batch_id=None, after_id=0)` and assert the fake client/server receives `batch_id=None`.

- [x] **Step 2: Run service/scope tests**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/Media/test_server_media_reading_service.py Tests/Media/test_media_reading_scope_service.py -q
```

Expected: PASS if the existing seam already supports the contract, otherwise FAIL before implementation.

### Task 2: Add Recent Watch UI Control

**Files:**
- Modify: `Tests/UI/test_media_ingest_window_rebuilt.py`
- Modify: `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`

- [x] **Step 1: Add a failing UI test**

Cover a `Watch Recent Server Jobs` button or handler that calls `scope_service.stream_media_ingest_job_events(mode="server", batch_id=None, after_id=0)` and renders the snapshot into the existing status/job controls.

- [x] **Step 2: Run the UI test and verify failure**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/UI/test_media_ingest_window_rebuilt.py -q
```

Expected: FAIL on missing button or handler.

- [x] **Step 3: Implement minimal UI**

Add:
- `Watch Recent Server Jobs` button.
- Handler that starts a worker.
- `watch_recent_job_events(after_id=0)` method that streams with `batch_id=None`.
- Status text that clearly says it is watching recent visible server jobs, not browsing historical batches.

- [x] **Step 4: Run UI tests and verify pass**

Run the same command. Expected: PASS.

### Task 3: Update Docs And Verify

**Files:**
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`

- [x] **Step 1: Update parity docs**

Record recent live server-job watching as landed and keep true historical batch discovery blocked on a server list endpoint.

- [x] **Step 2: Run focused verification**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/Media/test_server_media_reading_service.py Tests/Media/test_media_reading_scope_service.py Tests/UI/test_media_ingest_window_rebuilt.py Tests/tldw_api/test_media_reading_client.py -q
git diff --check
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall tldw_chatbook/Media tldw_chatbook/UI/MediaIngestWindowRebuilt.py tldw_chatbook/tldw_api Tests/Media Tests/UI/test_media_ingest_window_rebuilt.py -q
```

Expected: PASS.
