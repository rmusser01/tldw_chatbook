# TLDW API Worker Result Envelope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate cross-request ingestion-context contamination and payload-bearing diagnostics by returning each TLDW API worker result with its own frozen detached fallback context.

**Architecture:** A dependency-light contract module defines a frozen fallback context and frozen worker envelope. The submit path detaches immutable fallback values before launch; the worker returns response plus that exact context. The centralized worker success handler validates/unpacks the envelope and never consults application-global request state.

**Tech Stack:** Python 3.11+, frozen dataclasses, Textual workers, tldw API response models, pytest/pytest-asyncio, Loguru capture, AST privacy checks.

**Backlog:** [TASK-905](../../../backlog/tasks/task-905%20-%20Replace-shared-TLDW-API-request-context-with-a-frozen-result-envelope.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-647

**ADR required:** yes

**ADR path:** `backlog/decisions/029-local-private-data-boundary.md`; `backlog/decisions/033-application-session-state-ownership.md`

**Reason:** The accepted ADRs require payload-free diagnostics and prohibit application-root request ownership; the envelope implements those existing decisions.

---

## Execution and Test Boundary

Value objects, detachment, validation, and result normalization are tested
directly in `Tests/Event_Handlers/test_tldw_api_worker_envelope.py`. End-to-end
worker dispatch uses a normal mounted `TldwCli` and registered rebuilt Media
Ingest screen in
`Tests/ProductionApp/test_tldw_api_worker_envelope.py`. A narrow API client or
media DB collaborator may be injected; no app/screen substitute or unbound app
method is allowed.

## File Structure

- Create `tldw_chatbook/Event_Handlers/tldw_api_worker_contracts.py`: frozen context/envelope and detachment helper.
- Modify `tldw_chatbook/Event_Handlers/tldw_api_events.py`: expose one app-independent async request executor with explicit inputs, capture context before launch, return envelopes, remove duplicate unused local success/failure callbacks and shared-field write.
- Modify `tldw_chatbook/Event_Handlers/media_ingest_workers.py`: validate/unpack envelope and use exact fallback fields with payload-free diagnostics.
- Modify `tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py`: preserve public worker-state routing while keeping success/failure cleanup deterministic.
- Modify `tldw_chatbook/Event_Handlers/ingest_events.py`: export only live contracts/handlers.
- Modify `tldw_chatbook/app.py`: remove `_last_tldw_api_request_context`.
- Create `Tests/Event_Handlers/test_tldw_api_worker_envelope.py`.
- Create `Tests/ProductionApp/test_tldw_api_worker_envelope.py`.
- Modify `Tests/test_application_state_ownership.py`.

## Task 1: Start TASK-905 and Specify the Frozen Contract

- [ ] Move the task In Progress and add its task-local plan:

```bash
backlog task edit 654 -s "In Progress"
backlog task edit 654 --plan $'ADR required: yes\nADR path: backlog/decisions/029-local-private-data-boundary.md; backlog/decisions/033-application-session-state-ownership.md\nReason: Existing ADRs require payload-free diagnostics and worker-owned completion context.\n\n1. Add frozen detached value objects.\n2. Return an envelope from every success path.\n3. Validate/unpack centrally and remove shared state.\n4. Prove interleaving, cancellation, cleanup, and privacy.'
```

- [ ] Define:

```python
@dataclass(frozen=True)
class TldwApiIngestionFallback:
    keywords: tuple[str, ...] = field(repr=False)
    author: str | None = field(repr=False)
    custom_prompt: str | None = field(repr=False)
    overwrite_db: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class TldwApiWorkerResult:
    response: Any = field(repr=False)
    context: TldwApiIngestionFallback = field(repr=False)
```

- [ ] Add a pure detachment helper that normalizes keywords to an immutable
  tuple and copies scalar author/custom-prompt/overwrite values. It must reject
  or safely normalize malformed mutable input without retaining the request
  model.
- [ ] Direct tests mutate the original request/list after detachment, assert
  the context is unchanged, and assert `repr(context)`/`repr(envelope)` omit
  unique response, keyword, author, prompt, and overwrite sentinels.
- [ ] Run:

```bash
pytest Tests/Event_Handlers/test_tldw_api_worker_envelope.py -q
```

Expected: FAIL before the contract exists, then PASS.

## Task 2: Return the Exact Envelope from the Worker

- [ ] Build the detached fallback once after request validation and before
  `run_worker()`.
- [ ] Extract the nested worker body into a module-level
  `execute_tldw_api_request(*, media_type, api_client, request_model,
  local_file_paths, fallback)` coroutine. It accepts no app or widget and is
  the exact callable submitted by the production event path.
- [ ] Make every successful branch of `execute_tldw_api_request()` return
  `TldwApiWorkerResult(response=<api result>, context=fallback)`.
- [ ] Keep API client close in `finally`; cancellation must re-raise after
  cleanup and must not create shared context.
- [ ] Delete the nested `on_worker_success`/`on_worker_failure` callbacks in
  `tldw_api_events.py`, because `MiscWorkerHandler` is the live completion
  route. Remove their payload-bearing response/error logging rather than
  maintaining two completion implementations.
- [ ] Remove `app._last_tldw_api_request_context` assignment.

## Task 3: Validate and Consume Centrally

- [ ] In `handle_tldw_api_worker_success()`, validate and unpack a
  `TldwApiWorkerResult` before any UI query; malformed results show bounded
  internal-error copy when the owner remains mounted and perform no ingestion.
- [ ] Use only `context.keywords`, `context.custom_prompt`,
  `context.author`, and `context.overwrite_db` as response fallbacks.
- [ ] Remove request-model reads and `_last_tldw_api_request_context` from the
  app and source guard.
- [ ] Rewrite success/failure diagnostics to include only operation, media
  type, outcome/count, HTTP status when safe, and exception category. Never
  interpolate the envelope, response, input reference/path/URL, custom prompt,
  author, keywords, API response detail, or traceback carrying payloads.
- [ ] Separate durable result consumption from best-effort presentation.
  A valid result must normalize and persist even when its originating screen
  has been unmounted; only after that work settles may the handler update
  loading/button/status state when the matching owner still exists.
- [ ] Ensure best-effort UI loading/button cleanup occurs for success,
  malformed result, error, and cancellation/teardown where the mounted owner
  still exists. A `QueryError` must never return before valid ingestion.
- [ ] Add an explicit `WorkerState.CANCELLED` route in `MiscWorkerHandler`
  that performs bounded UI settlement without calling success ingestion or
  reusing a prior result. Do not infer cancellation from exception text.

## Task 4: Prove Interleaving and Public Worker Routing

- [ ] Directly interleave two calls to the production
  `execute_tldw_api_request()` function with distinct
  keyword/author/prompt/overwrite sentinels and reverse their completion order.
  Assert each envelope retains its own fallback and neither retains either
  mutable request model.
- [ ] In the normal production app, navigate to rebuilt Media Ingest, invoke
  the mounted submit path with a narrow injected API-client collaborator so it
  schedules the real `api_calls` worker through `app.run_worker()`, and let
  the public `MiscWorkerHandler` route the `Worker.StateChanged` event. Inject
  only that client and a recording media DB collaborator.
- [ ] Prove one success ingests with its exact context; malformed success,
  worker error, and cancellation do not consume a previous context; all
  loading/button cleanup settles.
- [ ] Start a valid request, navigate away before completion, and prove the
  durable ingestion still commits exactly once without querying a replacement
  or unmounted screen. Navigate back separately and prove no stale status or
  fallback context is replayed.
- [ ] Retrieve the worker scheduled by the mounted submit path from Textual's
  public worker manager/group API, call its public `cancel()` method, await its
  terminal state, and observe the normal public state-change route. Do not
  instantiate or call a handler with a fabricated `Worker.StateChanged` event.
- [ ] Capture Loguru, pytest logs, notifications, status text, and exception
  text. Assert unique response/prompt/author/keyword/input-ref sentinels are
  absent.
- [ ] Run:

```bash
pytest Tests/Event_Handlers/test_tldw_api_worker_envelope.py Tests/ProductionApp/test_tldw_api_worker_envelope.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

## Task 5: Verify and Close TASK-905

- [ ] Run:

```bash
python -m compileall -q tldw_chatbook/Event_Handlers/tldw_api_worker_contracts.py tldw_chatbook/Event_Handlers/tldw_api_events.py tldw_chatbook/Event_Handlers/media_ingest_workers.py tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/app.py
python -m ruff check tldw_chatbook/Event_Handlers/tldw_api_worker_contracts.py tldw_chatbook/Event_Handlers/tldw_api_events.py tldw_chatbook/Event_Handlers/media_ingest_workers.py tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/app.py Tests/Event_Handlers/test_tldw_api_worker_envelope.py Tests/ProductionApp/test_tldw_api_worker_envelope.py Tests/test_application_state_ownership.py
python -m ruff format --check tldw_chatbook/Event_Handlers/tldw_api_worker_contracts.py tldw_chatbook/Event_Handlers/tldw_api_events.py tldw_chatbook/Event_Handlers/media_ingest_workers.py tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py tldw_chatbook/Event_Handlers/ingest_events.py Tests/Event_Handlers/test_tldw_api_worker_envelope.py Tests/ProductionApp/test_tldw_api_worker_envelope.py Tests/test_application_state_ownership.py
git diff --check
```

- Do not mass-format the verified pre-task `app.py` baseline exception.

- [ ] Commit:

```bash
git add tldw_chatbook/Event_Handlers/tldw_api_worker_contracts.py tldw_chatbook/Event_Handlers/tldw_api_events.py tldw_chatbook/Event_Handlers/media_ingest_workers.py tldw_chatbook/Event_Handlers/worker_handlers/misc_worker_handler.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/app.py Tests/Event_Handlers/test_tldw_api_worker_envelope.py Tests/ProductionApp/test_tldw_api_worker_envelope.py Tests/test_application_state_ownership.py
git commit -m "fix(ingest): bind TLDW API context to worker results (task-905)"
```

- [ ] Re-read TASK-905, add Implementation Notes containing actual commands,
  counts, durations, interleaving/cancellation/navigation-away/privacy
  evidence, modified files, ADRs, and deviations, check all acceptance
  criteria, then mark Done and commit its task file:

```bash
backlog task 654 --plain
backlog task edit 654 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
git add 'backlog/tasks/task-905 - Replace-shared-TLDW-API-request-context-with-a-frozen-result-envelope.md'
git commit -m "docs(backlog): close TLDW API worker context (task-905)"
```
