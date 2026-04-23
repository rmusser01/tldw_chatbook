# Research Live Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add source-aware Chatbook support for the server deep-research run SSE event stream.

**Architecture:** Extend the existing research-runs client/service/scope seam rather than adding a parallel stream client. `TLDWAPIClient` parses SSE into a typed `ResearchRunStreamEvent`, `ServerResearchService` normalizes events to plain dictionaries, `ResearchScopeService` gates server streaming with `research.runs.observe.server`, and `ResearchWindow` exposes a `Watch Events` action for selected server runs. Local mode stays explicit: local research sessions do not have a live autonomous event stream in this slice.

**Tech Stack:** Python 3.12, Pydantic v2 schemas, Textual widgets, pytest/pytest-asyncio.

---

### Task 1: Pin The Research SSE Client Contract

**Files:**
- Modify: `tldw_chatbook/tldw_api/research_runs_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_research_runs_client.py`

- [x] **Step 1: Write failing client tests**

Add a fake SSE stream test for `stream_research_run_events("rs_1", after_id=3)` and assert it calls `/api/v1/research/runs/rs_1/events/stream` with `after_id=3`.

- [x] **Step 2: Run the client test and verify failure**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_research_runs_client.py -q
```

Expected: FAIL on missing schema/export/client method.

- [x] **Step 3: Implement typed stream schema and client method**

Add `ResearchRunStreamEvent(event: str, id: str | None, data: dict[str, Any] | str | None)` and use the existing SSE parser with a model override or generic dict conversion.

- [x] **Step 4: Run client tests and verify pass**

Run the same command. Expected: PASS.

### Task 2: Route Stream Events Through Services

**Files:**
- Modify: `tldw_chatbook/Research_Interop/server_research_service.py`
- Modify: `tldw_chatbook/Research_Interop/research_scope_service.py`
- Test: `Tests/Research_Interop/test_research_scope_service.py`

- [x] **Step 1: Write failing service/scope tests**

Cover server stream delegation, event normalization to plain dictionaries, policy action `research.runs.observe.server`, and local-mode explicit unavailable error.

- [x] **Step 2: Run service/scope tests and verify failure**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/Research_Interop/test_research_scope_service.py -q
```

Expected: FAIL on missing stream methods.

- [x] **Step 3: Implement service/scope methods**

Add `stream_run_events(run_id, after_id=0)` to server and scope services. Raise `ValueError("Local research live events are not available in this slice.")` for local mode.

- [x] **Step 4: Run service/scope tests and verify pass**

Run the same command. Expected: PASS.

### Task 3: Expose Watch Events In Research UI

**Files:**
- Modify: `tldw_chatbook/UI/Research_Modules/research_controller.py`
- Modify: `tldw_chatbook/UI/Research_Window.py`
- Test: `Tests/UI/test_research_screen.py`

- [x] **Step 1: Write failing UI tests**

Cover selecting a server run, invoking `watch_selected_run_events()`, routing through the controller/scope with `mode="server"`, and rendering snapshot/progress text into the detail panel.

- [x] **Step 2: Run UI tests and verify failure**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/UI/test_research_screen.py -q
```

Expected: FAIL on missing controller/window method or button.

- [x] **Step 3: Implement minimal UI**

Add a `Watch Events` button and a `watch_selected_run_events(after_id=0)` method that consumes the stream and appends concise event lines. Do not implement background persistence or local autonomous execution in this slice.

- [x] **Step 4: Run UI tests and verify pass**

Run the same command. Expected: PASS.

### Task 4: Docs And Verification

**Files:**
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`

- [x] **Step 1: Update docs**

Record server live event streaming as landed. Keep local autonomous research execution and richer checkpoint/artifact/bundle UX as remaining gaps.

- [x] **Step 2: Run focused verification**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_research_runs_client.py Tests/Research_Interop/test_research_scope_service.py Tests/UI/test_research_screen.py Tests/UI/test_screen_navigation.py -q
git diff --check
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall tldw_chatbook/Research_Interop tldw_chatbook/UI/Research_Window.py tldw_chatbook/UI/Research_Modules tldw_chatbook/tldw_api Tests/Research_Interop Tests/UI/test_research_screen.py Tests/tldw_api/test_research_runs_client.py -q
```

Expected: PASS.
