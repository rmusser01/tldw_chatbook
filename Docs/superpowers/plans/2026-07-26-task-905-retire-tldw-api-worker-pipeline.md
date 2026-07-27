# Retired TLDW API Worker Pipeline Removal Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the unreachable pre-Library TLDW API worker pipeline, its shared application request context, and its payload-bearing completion handlers without disturbing the live Library ingest implementation.

**Architecture:** Latest `dev` contains no producer for the `api_calls` worker group, no `tldw_api_events.py`, no retired MediaIngestScreen, and no matching `#tldw-api-*` widgets. The only remaining pieces are an application field, a worker-registry branch, compatibility exports, and completion handlers. Delete that orphan graph instead of manufacturing a replacement producer or result-envelope abstraction. Library remains the only ingest owner through its queue, request mapping, and server batch cancellation seams.

**Tech Stack:** Python 3.11+, Textual production app, Python AST, pytest/pytest-asyncio, Ruff.

**Backlog:** [TASK-905](../../../backlog/tasks/task-905%20-%20Retire-unreachable-TLDW-API-worker-context-and-handlers.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-647

**ADR required:** yes

**ADR path:** `backlog/decisions/029-local-private-data-boundary.md`; `backlog/decisions/033-application-session-state-ownership.md`

**Reason:** The accepted ADRs prohibit application-root request ownership and payload-bearing diagnostics. With the producer and destination already retired, deletion is the only current interface consistent with those decisions.

---

## Execution and Test Boundary

Static ownership tests prove the orphan graph is absent. Live request mapping
is tested directly in `Tests/Library/test_server_ingest_request.py`. Route and
owner behavior use a normal mounted `TldwCli` in
`Tests/ProductionApp/test_retired_tldw_api_worker_pipeline.py`; no app/screen
substitute or unbound app method is allowed.

## File Structure

- Delete `tldw_chatbook/Event_Handlers/media_ingest_workers.py`: unreachable payload-bearing success/failure consumers.
- Modify `tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py`: remove the unproducible `api_calls` group and routing branch while retaining live miscellaneous worker groups.
- Modify `tldw_chatbook/Event_Handlers/ingest_events.py`: remove only retired TLDW API compatibility exports; retain live Notes compatibility exports until separately retired.
- Modify `tldw_chatbook/Event_Handlers/note_ingest_events.py`: correct the
  stale worker-registry group comment after `api_calls` is deleted; do not
  change Notes behavior.
- Modify `tldw_chatbook/app.py`: remove `_last_tldw_api_request_context`.
- Create `Tests/ProductionApp/test_retired_tldw_api_worker_pipeline.py`.
- Modify `Tests/test_application_state_ownership.py`.

## Task 1: Start TASK-905 and Freeze the Verified Removal Boundary

- [ ] Move the task In Progress and add its task-local plan:

```bash
backlog task edit 905 -s "In Progress"
backlog task edit 905 --plan $'ADR required: yes\nADR path: backlog/decisions/029-local-private-data-boundary.md; backlog/decisions/033-application-session-state-ownership.md\nReason: Existing ADRs require payload-free diagnostics and forbid application-root request ownership; latest dev has no producer for the retired pipeline.\n\n1. Prove the api_calls producer and MediaIngestScreen are absent.\n2. Remove the orphan field, handlers, routing, and exports.\n3. Exercise the live Library ingest owner in the production app.\n4. Verify live Library request mapping/cancellation and static absence.'
```

- [ ] Add failing source sentinels for:
  `_last_tldw_api_request_context`, the `api_calls` group/branch,
  `media_ingest_workers` imports/exports, and `#tldw-api-*` selectors.
- [ ] Scope the AST checks to production application/worker ownership so live
  Library request fields and unrelated provider API calls remain legal.
- [ ] Assert the route registry maps `ingest` to `library` and that no
  `MediaIngestScreen` or `tldw_api_events.py` production path exists.
- [ ] Run:

```bash
pytest Tests/test_application_state_ownership.py -q
```

Expected: FAIL on the orphaned field/routing/handlers before production edits.

## Task 2: Delete the Orphaned Worker Graph

- [ ] Remove `_last_tldw_api_request_context` from `TldwCli`.
- [ ] Delete `media_ingest_workers.py`; do not move its payload rendering,
  result normalization, UI queries, or database writes into another module.
- [ ] Remove the `api_calls` worker group, dynamic ingest-events import, and
  `_handle_api_calls()` from `MiscWorkerHandler`. Preserve `ollama_api` and
  `model_download` behavior.
- [ ] Remove only the two retired TLDW API imports/exports from
  `ingest_events.py`.
- [ ] Update `note_ingest_events.py`'s stale comment that lists `api_calls` as
  a live registry group; do not change its code path.
- [ ] Confirm no producer, selector, handler, compatibility property, or
  dynamic reference remains.

## Task 3: Prove the Live Library Owner Remains Intact

- [ ] Mount the normal production `TldwCli`, navigate through the `ingest`
  alias, and assert the registered owner is `LibraryScreen`.
- [ ] Exercise one safe Library ingest form-state transition and leave/return
  without any retired import, selector query, or root attribute access.
- [ ] Cover `build_server_ingest_kwargs()` directly for local files and URLs;
  preserve its detached keyword/options result and accepted server types.
- [ ] In that same mounted production app, seed one real server-backed Library
  ingest job, inject only a narrow recording server-service collaborator,
  press the rendered row's cancel action through the pilot, and assert the
  public cancellation seam receives the exact batch id. Do not use the
  existing minimal App harness, call an unbound `TldwCli` method, or fabricate
  a `Worker.StateChanged` event for the deleted group.

## Task 4: Verify and Close TASK-905

- [ ] Run:

```bash
pytest Tests/Library/test_server_ingest_request.py Tests/ProductionApp/test_retired_tldw_api_worker_pipeline.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

- [ ] Run:

```bash
python -m compileall -q tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/Event_Handlers/note_ingest_events.py tldw_chatbook/Library/server_ingest_request.py tldw_chatbook/app.py
python -m ruff check tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/Event_Handlers/note_ingest_events.py tldw_chatbook/Library/server_ingest_request.py tldw_chatbook/app.py Tests/Library/test_server_ingest_request.py Tests/ProductionApp/test_retired_tldw_api_worker_pipeline.py Tests/test_application_state_ownership.py
python -m ruff format --check tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/Event_Handlers/note_ingest_events.py tldw_chatbook/Library/server_ingest_request.py Tests/Library/test_server_ingest_request.py Tests/ProductionApp/test_retired_tldw_api_worker_pipeline.py Tests/test_application_state_ownership.py
git diff --check
```

- Do not mass-format the verified pre-task `app.py` baseline exception.

- [ ] Commit:

```bash
git add tldw_chatbook/Event_Handlers/media_ingest_workers.py tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/Event_Handlers/note_ingest_events.py tldw_chatbook/app.py Tests/ProductionApp/test_retired_tldw_api_worker_pipeline.py Tests/test_application_state_ownership.py
git commit -m "refactor(ingest): retire dead TLDW API worker pipeline (task-905)"
```

- [ ] Re-read TASK-905, add Implementation Notes containing the latest-dev
  producer/selector census, actual commands, counts, durations, Library route
  and cancellation evidence, deleted/modified files, ADRs, and deviations;
  check all acceptance
  criteria, then mark Done and commit its task file:

```bash
backlog task 905 --plain
backlog task edit 905 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
git add 'backlog/tasks/task-905 - Retire-unreachable-TLDW-API-worker-context-and-handlers.md'
git commit -m "docs(backlog): close retired TLDW API worker pipeline (task-905)"
```
