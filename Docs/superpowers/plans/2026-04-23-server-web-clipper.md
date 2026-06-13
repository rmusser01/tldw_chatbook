# Server Web Clipper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the server-owned Web Clipper save/status/enrichment contract in Chatbook as a remote-only ingestion surface.

**Architecture:** Add typed API models and `TLDWAPIClient` methods for `/api/v1/web-clipper`, then wrap them in a remote-only service plus a policy-aware scope seam. Mount a lightweight `Web Clipper` tab in the existing media ingest window that can save a clip, inspect a known clip, and persist OCR/VLM enrichment without pretending that local clipper or server list APIs exist.

**Tech Stack:** Python 3, Pydantic v2, Textual, pytest/pytest-asyncio, existing `runtime_policy` action IDs.

---

### Task 1: Typed API Contract

**Files:**
- Create: `tldw_chatbook/tldw_api/web_clipper_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_web_clipper_client.py`

- [x] **Step 1: Write failing client tests**

Cover clip save, status lookup, and enrichment persistence.

- [x] **Step 2: Run the new API test**

Run: `/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_web_clipper_client.py -q`

Expected: fail because schemas and methods do not exist.

- [x] **Step 3: Add schemas and client methods**

Mirror the server schema from `tldw_server2/tldw_Server_API/app/api/v1/schemas/web_clipper_schemas.py`. Use `/api/v1/web-clipper/save`, `/api/v1/web-clipper/{clip_id}`, and `/api/v1/web-clipper/{clip_id}/enrichments`.

- [x] **Step 4: Re-run the API test**

Expected: pass.

### Task 2: Remote-Only Service And Scope Seam

**Files:**
- Create: `tldw_chatbook/WebClipper/server_web_clipper_service.py`
- Create: `tldw_chatbook/WebClipper/server_web_clipper_scope_service.py`
- Create: `tldw_chatbook/WebClipper/__init__.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/WebClipper/test_server_web_clipper_service.py`

- [x] **Step 1: Write failing service/scope tests**

Cover normalized clip IDs, policy action IDs, save/status/enrichment dispatch, and local-mode rejection.

- [x] **Step 2: Run the new service test**

Run: `/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/WebClipper/test_server_web_clipper_service.py -q`

Expected: fail because the Web Clipper service package does not exist.

- [x] **Step 3: Implement service and scope seam**

Use `web_clipper.capture.server` for save/enrichment and `web_clipper.status.server` for status lookup. Do not implement local behavior; local mode must fail explicitly.

- [x] **Step 4: Wire app bootstrap**

Attach `server_web_clipper_service` and `server_web_clipper_scope_service` beside existing media-reading ingestion services.

- [x] **Step 5: Re-run the service test**

Expected: pass.

### Task 3: Media Ingest Web Clipper Tab

**Files:**
- Modify: `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
- Test: `Tests/UI/test_media_ingest_window_rebuilt.py`

- [x] **Step 1: Write failing UI tests**

Cover server mode enabling the panel, local mode showing unavailable guidance, save payload dispatch, status lookup rendering, and enrichment payload dispatch.

- [x] **Step 2: Run the UI test**

Run: `/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/UI/test_media_ingest_window_rebuilt.py -q`

Expected: fail because the panel does not exist.

- [x] **Step 3: Add lightweight panel**

Use explicit fields for common clip data and JSON text inputs for metadata, attachments, and enrichment payloads. Do not add a list view because the server does not expose one.

- [x] **Step 4: Re-run the UI test**

Expected: pass.

### Task 4: Docs And Verification

**Files:**
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`

- [x] **Step 1: Update parity status**

Record Web Clipper as remote-only control surfaced in Chatbook, with remaining gaps limited to browser-extension UX, local mirroring, and any future server list/history endpoint.

- [x] **Step 2: Run focused verification**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_web_clipper_client.py Tests/WebClipper/test_server_web_clipper_service.py Tests/UI/test_media_ingest_window_rebuilt.py Tests/UI/test_screen_navigation.py Tests/RuntimePolicy/test_runtime_policy_core.py -q
git diff --check
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall tldw_chatbook/WebClipper tldw_chatbook/UI/MediaIngestWindowRebuilt.py tldw_chatbook/tldw_api Tests/WebClipper Tests/UI/test_media_ingest_window_rebuilt.py Tests/tldw_api/test_web_clipper_client.py -q
```

Expected: all commands pass before committing.
