# Session-bound file-to-note workflow implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved saved `.txt` -> prompt -> llama.cpp -> editable
human review -> Local Note workflow through the existing Workflows destination.

**Architecture:** One lazy app-owned in-memory session runs a closed set of five
sequential operation subsets. Reuse document/expression, permission and Notes
owners; add one opt-in bounded request function in the existing LLM domain.
Screen navigation subscribes to the session, never owns its physical work.

**Tech Stack:** Python >=3.12, installed Textual 8.2.8, existing httpx, asyncio,
existing SQLite/Notes services, pytest. No new dependencies.

**Spec:** [Approved first-run design](../specs/2026-09-16-workflows-first-run-design.md).
**Backlog:** TASK-32691; approved design task TASK-32690.
**Status:** Approved; execution in progress using subagents with per-task spec
and quality review. Checkboxes record completed, verified steps only.

ADR required: no new ADR; direct implementation of the approved amendment.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: the session lifetime and narrow integration boundaries are already
approved. ADR-125 SQLite, ADR-036 composition, ADR-029 privacy, ADR-031 bindings
and ADR-150 design language remain in force. Stop for design review if satisfying
this plan appears to require changing those boundaries.

## Global constraints

- One application-owned sequential run at a time in each app instance.
- Restart-resumable execution is not part of this delivery.
- Branching remains v2 and parallelism v3.
- This first slice admits keyless loopback requests only.
- No automatic retry, fallback provider, PDF/RAG expansion, server publication,
  workflow synchronization, Console execution engine, background schedule, or
  21-adapter rollout belongs to this first delivery.
- No new run schema, database owner, lock file, PID record, helper protocol,
  recovery scanner or persistent execution service. Do not modify migrations,
  historical run/ownership/capacity rows, or shared SQLite infrastructure.
- Saved revisions are immutable. Drafts, definitions and committed Notes keep
  their existing persistence; run state, review text and counters remain in memory.
- No new settings surface or framework. Run setup carries this slice's choices;
  existing settings remain canonical. Imported JSON never grants authority.
- Admission defaults: 100 steps, 2 MiB canonical definition, 10 MiB serialized
  inputs, 1 MiB serialized output per step, 100 MiB aggregate outputs, 60 minutes
  active execution, 100,000 input-plus-maximum-output token admission units.
- Source file and HTTP body: 1 MiB each, bounded before decoding. Strict UTF-8
  file reads; no silent truncation, artifact spill or limit-increase UI.
- Explicit positive whole-second attempt deadline; example 300 s. Captured model
  request deadline defaults to 120 s and cannot extend the attempt deadline.
  Review uses its own config response deadline; example 3600 s, omitted or
  non-positive means unlimited. Navigation does not reset it.
- One Note-create ID per attempt; no update, blind retry, or Sync v2 dispatch.
  Uncertain readback must match destination, ID, title and accepted content.
- Cancel/timeout stops advancement; keep the run slot until physical work settles.
  A confirmed late Note commit is saved, not rolled back. No forced shutdown.
- Use `$ds-*` tokens and existing component states; edit CSS sources and rebuild
  the bundle, never hand-edit it. No terminal-convention or global-key shadowing.
- Targeted verification only. Never start a live app or import profile-owning
  code from a bare probe before establishing disposable config/data isolation.

Task5 review clarification (existing ADR-138, no new ownership boundary): prepare
authoring quit reversibly on the same owners; retain/drain accepted work without
closing their DB before final Console consent. Renewed Stay/cancelled confirmation
restores authoring and successfully drained session admission for a new run only.
Persistence/physical-drain failure still blocks exit; final resource-close errors
follow existing committed-exit teardown policy. Narrow methods may be added to
Workflows/authoring.py, draft_session.py and session.py, with real-owner lifecycle
regressions; no owner replacement, generalized lifecycle framework or storage change.

## Baseline and working rules

Plan source: authoring merged dev `657f70ffe7fbb92b637a89cc2961e8bf95ec6005`,
design commit `2a35efcebb`. Existing isolated worktree:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev`.
Do not substitute the dirty root checkout. Preserve unrelated TASK-32601 notes
and `.uat-workflows-9NUT5t/`; stage only task-owned paths.

At execution start, check branch/status and upstream movement. If the baseline
changes, recheck the named symbols before editing; do not blindly replay code
from the preserved runtime branch. The code blocks here are implementation/test
instructions, not claims that the code already runs. Execute RED before GREEN.

Commands below assume this worktree and the existing root venv:

```sh
export WORKFLOW_PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
export WORKFLOW_RUFF=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff
export PYTHONPATH=.
```

Read the approved spec, ADRs above, `backlog/docs/design-language.md`, and the
relevant entries in the testing/live/backlog lessons before touching the area.
In particular: physical cancellation is not await cancellation; a saved endpoint
must be pinned through the real transport; use real DB commit barriers; and a
mounted harness without app-tier CSS is not layout evidence.

## File responsibilities

| File | Responsibility |
| --- | --- |
| `LLM_Calls/llamacpp_bounded.py` (new) | Closed, keyless, numeric-loopback async completion; no config discovery or Console ownership. |
| `MCP/permission_store.py`, `Agents/builtin_tool_gate.py` (existing) | Opt-in fresh strict authority through the existing resolver; ordinary callers unchanged. |
| `Workflows/session_permissions.py` (new) | Three fixed workflow effects and run/step/effect-scoped Ask handoff; no second policy engine. |
| `Workflows/local_steps.py` (new) | Non-mutating bounded file read and local Notes policy/transaction bridge with captured destination. |
| `Workflows/session.py` (new) | Admission, detached run values, closed dispatcher, review/cancel/drain and view subscription. Split only if the implementation cannot stay readable. |
| `app.py` (existing) | Lazy composition, app-wide quit confirmation/fence/drain and normal dependent-service shutdown. |
| `UI/Workflows_Modules/run_controls.py` (new) | Setup and session result/review widgets; no execution ownership. |
| `UI/Screens/workflows_screen.py` (existing) | Enable the existing Run control, attach/detach session view, Open Note route. |
| `css/features/_workflows.tcss` (existing) | Token-backed setup/review states without changing the three-pane authoring layout. |
| `Docs/User_Guide/workflows.md` (existing) | Supported execution subset and session-loss/uncertain-save disclosure after verification. |

Paths in this table are beneath `tldw_chatbook/` unless prefixed `Docs/`.
The tasks below define public interfaces; do not invent a generalized executor,
adapter registry, durable receipt store or alternate permission namespace.

## Task 1: Bounded llama.cpp completion

**Files:** Create `tldw_chatbook/LLM_Calls/llamacpp_bounded.py` and
`Tests/LLM_Calls/test_llamacpp_bounded.py`. Reuse, without broad refactoring,
`Chat/provider_endpoint_contract.py`, `Chat/local_server_discovery.py`,
`Chat/sampling_params.py`, `Chat/llamacpp_think_filter.py` and
`Utils/sensitive_llm_logging.py`.

**Consumes:** `resolve_provider_endpoint("llama_cpp", value)`,
`read_bounded_model_response(response) -> bytes | None`,
`validate_sampling_params(params)`, `split_start_anchored_thinking(text)` and
`sensitive_llm_request()`.

**Produces:** Frozen `BoundedLlamaRequest` with string fields `provider_id`,
`selected_url`, `dispatch_url`, `model`, `prompt`; integer `max_tokens`; float
`request_timeout_seconds`; and `sampling: tuple[tuple[str, int | float], ...]`.
Mark URLs/prompt `repr=False`. `LlamaResult` is a TypedDict with `text: str` and
`usage: LlamaUsage | None`; `LlamaUsage` has nonnegative integer `input_tokens`
and `output_tokens`. Expose these functions:

- `resolve_llama_loopback_url(selected_url: str) -> str`: synchronous setup-only
  resolution to one full numeric-loopback chat URL, before approval.
- `estimate_llama_reservation(prompt: str, *, max_tokens: int) -> int`.
- `async complete_llama_bounded(request: BoundedLlamaRequest, *, deadline_at: float) -> LlamaResult`.
- `BoundedLlamaError(code: str)`: payload-free exception for validation,
  transport/status, deadline, oversized body, malformed/incomplete answer.

- [x] **1. Write a failing reservation and captured-request test.** Use an owned
  loopback listener, not a fake that copies the expected endpoint. Start with:

```python
def test_reservation_counts_utf8_and_full_output_allowance():
    from tldw_chatbook.LLM_Calls.llamacpp_bounded import estimate_llama_reservation

    assert estimate_llama_reservation("é", max_tokens=512) == 578
```

  Add a `@pytest.mark.loopback_network` async test whose handler reads headers
  and the declared body, records exactly one request, and returns a bounded
  OpenAI chat-completion document. The listener's essential body is:

```python
body = json.dumps({
    "choices": [{"message": {"content": "review me"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}).encode()
writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
             + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
             + body)
await writer.drain()
writer.close()
await writer.wait_closed()
```

  Use `asyncio.start_server(handler, "127.0.0.1", 0)` in an async context; retain
  and join accepted handler tasks in fixture cleanup. Assert the actual request
  URL/model, no Authorization/tools, and no change after mutating live config.
- [x] **2. Run RED:**
  `"$WORKFLOW_PY" -m pytest -o addopts= -q Tests/LLM_Calls/test_llamacpp_bounded.py`.
  Confirm failure is the missing implementation/behavior, not a profile import,
  unavailable dependency, or blocked socket fixture.
- [x] **3. Implement the closed request function.** Validate before any I/O;
  reject booleans/nonfinite budgets, userinfo/query/fragment and non-loopback
  dispatch. Resolve literal localhost only in the retained setup worker; reject
  non-loopback DNS answers, display/pin one address, and do not retry another.
  Preserve provider identity independently of the transport family. Sampling
  subset is `temperature`, `top_p`, `top_k`, `min_p`, `seed`; other explicit
  executable settings block admission. Use these concrete request settings:

```python
reservation = len(prompt.encode("utf-8")) + 64 + max_tokens
transport = httpx.AsyncHTTPTransport(
    retries=0, trust_env=False,
    limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
)
client = httpx.AsyncClient(
    transport=transport,
    trust_env=False,
    follow_redirects=False,
)
payload = {
    "model": request.model,
    "messages": [{"role": "user", "content": request.prompt}],
    "max_tokens": request.max_tokens,
    "stream": False,
    **dict(request.sampling),
}
deadline = min(deadline_at, asyncio.get_running_loop().time()
               + request.request_timeout_seconds)
```

  Wrap send/headers/raw body in `asyncio.timeout_at(deadline)` and stream with
  `Accept-Encoding: identity`. Read via the unconsumed-response branch of the
  existing 1 MiB raw reader; `None` means reject, never parse. Refuse non-2xx
  without logging error bodies. Close response/client in a retained cleanup
  task outside the expired deadline, including cancellation; the caller cannot
  release capacity before cleanup settles. Use TLS verification for HTTPS;
  never disable it to accommodate a localhost certificate mismatch.

  Extract only chat message content, reject tools/function calls, malformed JSON,
  empty visible answers and `finish_reason="length"`. Strip only the existing
  start-anchored reasoning envelope; reject incomplete envelopes. Missing or
  invalid usage is `None`, not zero. Keep reservation after a dispatched request
  even when usage is absent. Use existing sensitive-request context; if DEBUG
  transport-header canaries leak, add a context-local filter for this path to
  `httpx`/`httpcore` loggers, never global level changes or a logging framework.
- [x] **4. Add/execute focused qualification cases:** 429/500/disconnect causes
  one POST; poisoned HTTP(S)_PROXY is ignored; redirects are not followed;
  1 MiB/overflow, chunked bodies and compressed bodies; continuous trickle exceeds
  absolute deadline; cancelled body closes the real connection; held cleanup
  retains the operation. Test UTF-8, reasoning, usage, boolean/nonfinite settings
  and payload/header log canaries with unrelated concurrent logging preserved.
  Run Task 1 tests plus `Tests/Chat/test_provider_endpoint_contract.py`,
  `Tests/Chat/test_local_server_discovery.py` and
  `Tests/Chat/test_sensitive_llm_logging.py`; run scoped Ruff/format checks.
- [x] **5. Review and commit only the new module/tests:**

```sh
git add -- tldw_chatbook/LLM_Calls/llamacpp_bounded.py Tests/LLM_Calls/test_llamacpp_bounded.py
git commit -m "feat(workflows): qualify bounded local llama.cpp requests"
```

Do not route ordinary LLM calls through this opt-in function in this task.

## Task 2: Fresh, fail-closed effect permission checks

**Files:** Modify `tldw_chatbook/Agents/builtin_tool_gate.py`; create
`tldw_chatbook/Workflows/session_permissions.py` and
`Tests/Workflows/test_session_permissions.py`; extend
`Tests/Agents/test_builtin_tool_gate.py`. Reuse existing
`MCP/permission_store.py::read_snapshot_strict()` unchanged unless a regression
test exposes a required compatibility correction. It already exists: do not
implement another strict reader or permission store.

**Consumes:** Existing `MCPPermissionStore`, `resolve_builtin_state`, `Tool`,
`CreateNoteTool`, `BuiltinToolGate.begin_turn/stamp/check_detailed`.
**Produces:** Add keyword-only `strict: bool = False` to `resolve` and
`check_detailed`, and `allow_session_approvals: bool = True` to `check_detailed`.
Add optional `refusal_code` to `BuiltinGateDecision`, default `None`, with strict
values `approval_required`, `denied`, `kill_switch`, `unavailable`. Preserve the
existing `approval_decision` fact and ordinary caller behavior; never parse
human-readable refusal strings to decide whether Ask can be approved.

`session_permissions.py` defines frozen `EffectRequest(run_id: str, step_id: str,
kind: Literal["file", "model", "note"], payload_json: str)` with private payload
repr, and `WorkflowPermissions(gate: BuiltinToolGate)` exposing
`check(effect: EffectRequest, *, approved_once: bool = False) -> BuiltinGateDecision`.
Only the coordinator may supply `approved_once`, after matching its current
pending effect identity; no imported JSON field maps to it. This wrapper is not
an alternative resolver.

- [x] **1. Add missing-file strict refusal RED test** using the actual store:

```python
def test_missing_strict_authority_is_not_a_default_grant(tmp_path):
    from types import SimpleNamespace
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    from tldw_chatbook.Tools.note_management_tools import CreateNoteTool

    store = MCPPermissionStore(tmp_path / "permissions.json")
    gate = BuiltinToolGate(SimpleNamespace(permission_store=store))
    decision = gate.check_detailed(
        CreateNoteTool(), "workflow-test", strict=True,
        allow_session_approvals=False,
    )
    assert decision.refusal_code == "unavailable"
    assert not store.path.exists()
```

  These gate/store constructor signatures were checked against the pinned source;
  recheck them if the execution baseline changes.
- [x] **2. Run RED:**
  `"$WORKFLOW_PY" -m pytest -o addopts= -q Tests/Workflows/test_session_permissions.py Tests/Agents/test_builtin_tool_gate.py`.
- [x] **3. Implement one-snapshot strict verdicts.** Strict checks call
  `service.permission_store.read_snapshot_strict()` freshly; require
  `file_exists=True` and the captured permission profile to be available. Derive
  kill-switch and effective state from that same immutable payload through the
  existing resolver. Missing service/store, read/schema failure or invalid
  profile returns unavailable without repairing, normalizing or saving anything.
  `resolve(strict=True)` raises the existing strict snapshot error on unavailable
  authority; `check_detailed` converts it into the structured refusal. Do not
  feed strict reads into the ordinary turn cache.

  Use fixed metadata-only Tool descriptors `workflow_read_file` (`reads`) and
  `workflow_local_model` (`network`), preserving the historical workflow names,
  and existing `CreateNoteTool()` (`create_note`). Descriptor execution raises;
  do not register these as arbitrary executable LLM tools or add a new hub.
  Wrap each check with a unique run/step/payload-digest gate key:

```python
effect_key = f"{effect.run_id}:{effect.step_id}:{hashlib.sha256(effect.payload_json.encode()).hexdigest()}"
gate.begin_turn(effect_key)
try:
    if approved_once:
        gate.stamp(effect_key, tool.name, "approve_once")
    decision = gate.check_detailed(
        tool, effect_key, strict=True, allow_session_approvals=False,
    )
finally:
    gate.begin_turn(effect_key)
```

  Revoked Off/kill always outranks the stamp. No session/always approval UI is
  added. The coordinator consumes a pending approval exactly once to dispatch
  that effect, while immediate pre-write rechecks may reuse that same approved
  effect only during its still-owned physical operation.
- [x] **4. Cover corrupt/unreadable/missing snapshots, stale cached allow,
  absent captured profile, deny/kill after approval, changed effect arguments,
  duplicate approval, and ordinary nonstrict behavior.** Assert strict reads
  never call `.load()`/`.save()` or create repair backups. Run the task tests,
  `Tests/MCP/test_permission_store.py`, `Tests/MCP/test_permission_resolution.py`;
  scoped Ruff/format checks must pass.
- [x] **5. Review and commit:**

```sh
git add -- tldw_chatbook/Agents/builtin_tool_gate.py tldw_chatbook/Workflows/session_permissions.py Tests/Agents/test_builtin_tool_gate.py Tests/Workflows/test_session_permissions.py
git commit -m "feat(workflows): add strict effect-time permission checks"
```

## Task 3: Bounded file and captured Local Note effects

**Files:** Create `tldw_chatbook/Workflows/local_steps.py` and
`Tests/Workflows/test_local_steps.py`; narrowly extend
`tldw_chatbook/Notes/Notes_Library.py` with the bound-cache context manager below;
add its tests to `Tests/Workflows/test_local_steps.py`. Reuse `NotesScopeService`
without duplicating Note business logic. Qualification also covers existing
title-bearing error diagnostics: remove the private title from the null-ID
message in `Notes_Library.py` and the `CharactersRAGDBError` message in
`DB/ChaChaNotes_DB.py`, without changing storage behavior. A resolved Note title
can contain private workflow output; test the actual bridge error paths.
Update the existing title-bearing log assertion in
`Tests/Notes/test_notes_library_unit.py` to match the payload-free diagnostic.

**Consumes:** `validate_path_simple(..., probe_existing=False)`, `lexical_path`,
`verify_trusted_directory`, `NotesInteropService.notes_db`,
`NotesScopeService.save_note/get_note_detail`, current-thread
`CharactersRAGDB.close_connection()`, and Task 2's exact-effect recheck callback.

**Produces:** Blocking worker functions in `local_steps.py`:

- `read_local_text(source: Path, *, protected_paths: tuple[Path, ...], before_read: Callable[[], None]) -> str`.
- Frozen, private-repr `LocalNoteDestination(scope, owner, db, user_id: str,
  db_path: str, client_id: str)` using the concrete existing Notes/DB types.
- `capture_local_note_destination(scope: NotesScopeService, *, user_id: str) -> LocalNoteDestination`.
- `create_local_note(destination: LocalNoteDestination, *, create_note_id: str,
  title: str, content: str, before_write: Callable[[], None]) -> str`.
- `read_local_note(destination: LocalNoteDestination, *, note_id: str) -> dict[str, Any] | None`.
- `LocalNoteCleanupError`: fixed payload-free `note_cleanup_failed` signal if the existing DB owner's current-thread close call raises. Do not change the owner or infer failures it swallows internally.
- `NotesInteropService.bound_notes_db(user_id: str, expected_db: CharactersRAGDB) -> Iterator[CharactersRAGDB]`, a context manager over its existing `_db_lock`.

- [x] **1. Write the non-mutating source test first:**

```python
def test_source_read_does_not_change_permissions(tmp_path):
    import stat
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "source.txt"
    source.write_text("hello é", encoding="utf-8")
    source.chmod(0o644)
    assert read_local_text(source, protected_paths=(), before_read=lambda: None) == "hello é"
    assert stat.S_IMODE(source.stat().st_mode) == 0o644
```

  Also use a real temporary file-backed `CharactersRAGDB` and actual Notes
  services for create/readback; in-memory mocks cannot prove worker-thread
  routing or commit outcomes. Seed the existing runtime policy/permission
  fixtures explicitly rather than substituting an allow-all service.
- [x] **2. Run RED:**
  `"$WORKFLOW_PY" -m pytest -o addopts= -q Tests/Workflows/test_local_steps.py`.
- [x] **3. Implement file reading without mutation.** Do not use
  `open_private_binary()` here: it chmods selected files. Validate the lexical
  path and trusted directory through current utilities; reject unverified
  platform safety, wrong-owner/symlink/nonregular/multiple-link files and visible identities
  matching known live workflow/Notes DBs or sidecars before any raw file open.
  The app supplies exact known DB paths; never scan/open databases to prove
  identity. Keep the approved stable-file/path assumption, not a new lease.
  Call `before_read()` immediately before opening. Use standard read-only
  `os.open` with available required no-follow/nonblocking flags, `fstat` regular
  file check and `os.fdopen`; close on every failure. The bounded payload core is:

```python
raw = stream.read(1024 * 1024 + 1)
if len(raw) > 1024 * 1024:
    raise ValueError("source_limit")
text = raw.decode("utf-8", errors="strict")
```

  Do not add line numbers, UTF-8 replacement, chmod, ingestion records or fallback.
  Check resulting serialized output against the session's result budget as well.
- [x] **4. Implement the Notes bridge.** Capture the actual cached Notes DB
  off-loop, then close only that worker thread's connection. Bound operations
  refuse a changed service, user/client, DB object/path or missing cached owner.
  `bound_notes_db` checks the existing cache while holding the existing lock;
  it never calls `_get_db` to manufacture a replacement on mismatch. Its local
  operation uses the cached fast path, so do not reacquire the same nonreentrant
  lock inside it. Production `ServicePolicyEnforcer` reads immutable policy
  state and can run here; no widget or approval callbacks execute in the worker.

```python
with destination.owner.bound_notes_db(destination.user_id, destination.db):
    try:
        before_write()
        note_id = asyncio.run(destination.scope.save_note(
            scope=ScopeType.LOCAL_NOTE,
            user_id=destination.user_id,
            create_note_id=create_note_id,
            title=title.strip(),
            content=content,
            sync_v2_profile=None,
        ))
    finally:
        destination.db.close_connection()
```

  Run the local-only coroutine in the retained worker, never `asyncio.run` on the
  Textual loop. Leave keywords/organization/Research arguments unset. Use the
  same bound route and current-thread cleanup for `get_note_detail`. Do not call
  cache-evicting `close_user_connection`. Require nonblank normalized title and
  exact accepted content; the coordinator retains the attempted ID even on error.
- [x] **5. Prove refusal/commit outcomes and commit.** Test source overflow,
  invalid UTF-8, links/FIFO/DB aliases before raw-open, policy denial, destination
  replacement, transaction rollback, lost response after commit, duplicate ID,
  deleted/mismatched readback, and cleanup on the actual worker thread. Hold
  commit barriers before/after SQLite commit; assert actual rows and effect
  counts, not just returned status. Run task tests and
  `Tests/Notes/test_notes_scope_service.py`, plus scoped Ruff/format checks.

```sh
git add -- tldw_chatbook/Workflows/local_steps.py tldw_chatbook/Notes/Notes_Library.py tldw_chatbook/DB/ChaChaNotes_DB.py Tests/Workflows/test_local_steps.py Tests/Notes/test_notes_library_unit.py
git commit -m "feat(workflows): add bounded local file and Note effects"
```

## Task 4: One in-memory sequential session

**Files:** Create `tldw_chatbook/Workflows/session.py`,
`Tests/Workflows/test_session_admission.py`,
`Tests/Workflows/test_session.py` and `Tests/Workflows/test_session_lifecycle.py`.
Reuse `models.Revision`, `DocumentService.project`, `expressions.json_copy` and
`resolve_value`; do not change document persistence to accommodate execution.

**Consumes:** Tasks 1–3. App-supplied current Notes scope/user and permission
profile, captured known DB paths, and immutable saved `Revision`.
**Produces:** `SessionError(code: str)` with payload-free messages;
`admit_definition(revision: Revision) -> dict[str, Any]`; frozen `ModelSelection`
with Task 1 request fields except prompt/max_tokens/dispatch_url; frozen `RunSetup`
with `source: Path`, `model: ModelSelection`, `review_actor: str` and
`protected_paths: tuple[Path, ...]`; frozen `RunBindings` with
`source: Path`, `model: ModelSelection`, `notes: LocalNoteDestination`,
`dispatch_url: str`, `review_actor: str`, `protected_paths: tuple[Path, ...]`.

Frozen `RunView` contains `run_id`, `workflow_id`, `revision_id`, `step_id`,
`state`, `message_code`, `review_text`, `note_id`, `generation` and
`pending_effect: EffectRequest | None`. Optional IDs/text are `None` when absent;
`review_instructions: str | None` exposes the resolved human-step instructions
with private repr so the remounted UI need not inspect session internals.
generation is a monotonic integer for stale-control/quit checks. States are
`ready`, `running`, `approval`, `review`, `stopping`, `cancelled`, `rejected`,
`failed`, `uncertain`, `completed`. Payload fields have private repr.

Construct `WorkflowSession(permissions: WorkflowPermissions, *,
notes_scope: Callable[[], NotesScopeService], notes_user: Callable[[], str])`
on the app loop. It exposes:

- `async prepare(revision: Revision, inputs: dict[str, Any], setup: RunSetup) -> int`:
  detach/validate, retain worker capture of the concrete model URL and Notes
  destination, then issue a monotonic launch ticket. Only one pending ticket;
  a newer setup invalidates an older unused one. A cancelled waiter does not
  abandon the capture worker or accidentally start the run.
- `bindings(ticket: int) -> RunBindings`: expose only the current prepared
  binding for the final displayed launch confirmation; stale tickets fail.
  `discard_setup() -> None`: invalidate the pending ticket/capture result without
  abandoning an already-started capture worker.
- `start(ticket: int) -> str`: synchronously consume that ticket, reserve the
  session slot and retain a run task before returning its UUID run ID. Repeated
  delivery of the current consumed ticket returns that ID, never dispatches.
  Older tickets remain invalid even after a newer run; no growing run-history set.
- `view() -> RunView | None`; `subscribe(callback: Callable[[], None]) -> Callable[[], None]`.
- `run_bindings(run_id: str) -> RunBindings`: read-only binding of the matching
  current singleton run, including its terminal result for Open Note validation;
  stale/wrong IDs fail. This does not expose a run history or reopen setup tickets.
- `update_review(run_id: str, step_id: str, text: str) -> bool` and
  `answer_review(run_id: str, step_id: str, *, accept: bool) -> bool`.
- `answer_effect(run_id: str, step_id: str, payload_json: str, *, approve: bool) -> bool`.
- `cancel(run_id: str) -> None`; `begin_close() -> None`; `abort_close() -> None`;
  `async close() -> None` for retained, idempotent physical settlement.

`begin_close` fences external controls and subsequent effect dispatch without
discarding current work. `abort_close` reopens admission only before actual
session cancellation/terminal close; it never resurrects a cancelled run.

- [x] **1. Start with whole-definition admission RED:**

```python
def test_retry_is_refused_not_silently_removed():
    import json
    import pytest
    from Tests.Workflows.helpers import prompt_definition
    from tldw_chatbook.Workflows.models import Revision
    from tldw_chatbook.Workflows.session import admit_definition, SessionError

    document = prompt_definition()
    document["steps"][1]["retry"] = 1
    identity = document["metadata"]["tldw_workflow"]
    revision = Revision(identity["workflow_id"], identity["revision_id"], (), json.dumps(document))
    with pytest.raises(SessionError, match="retry_unsupported"):
        admit_definition(revision)
    assert document["steps"][1]["retry"] == 1
```

  Run the three task test files with `-o addopts= -q`; confirm RED before code.
- [x] **2. Implement bounded admission, not another schema/expression engine.**
  Check raw/canonical size before expensive copies. Project with the existing
  parser, then call `json_copy(..., byte_limit=2 * 1024 * 1024)` to reject opaque
  numbers rather than interpreting display projection as execution data. Validate
  every step and reference before file/model/Note effects. Use the fixture's
  envelope and metadata namespace; reject unknown potentially executable fields,
  hooks, forward/self references, duplicate IDs, conditions and parallel groups.
  Preserve the stored bytes regardless of refusal.

  Closed config subsets are:

  | Type | Admitted configuration |
  | --- | --- |
  | `media_ingest` | Exactly one `sources[].uri`, captured local `.txt`; `extraction.extract_text=true`; no other ingestion effects. |
  | `prompt` | `template` string with existing dotted references. |
  | `llm` | `provider`, `model`, `prompt`, positive `max_tokens`; optional captured request timeout and the five Task 1 sampling fields. Provider/model must match the launch binding. |
  | `wait_for_human` | `instructions`, bound `assigned_to_user_id`, response `timeout_seconds`; editable text comes from the immediately preceding text result. |
  | `notes` | `action="create"`, nonblank `title`, `content`; Local Note destination only. |

  Requirements/input-schema admission supports the fixture's file/model/actor/
  notes bindings and object/string/required/minLength/additionalProperties
  declarations. Reject unsupported schema keywords, external references or
  capability declarations rather than evaluating or ignoring them. Merge saved
  ordinary input defaults and reviewed run inputs, then materialize binding-owned
  values; conflicting imported provider/path/actor values do not override bindings.
  Validate concrete types again after expression resolution. `step.retry` must be
  integer zero, excluding bool; timeout and all budgets reject bool/nonfinite
  values. Error copy names the field/code, not private payloads.
- [x] **3. Implement detached state and closed sequential dispatch.** Keep only
  current session state/results. Use a normal tuple/list of validated steps and
  an explicit five-way dispatch; no plugin/runtime base classes. For each step:

```python
resolved = resolve_value(step["config"], context, byte_limit=1024 * 1024)
result = json_copy(result, byte_limit=1024 * 1024)
context[step["id"]] = result
```

  The two lines involving `result` run only after a successful admitted operation.
  Charge aggregate output bytes and active time, excluding only waits without
  live work. Reserve Task 1 model admission units before POST. Build the final
  `BoundedLlamaRequest` from resolved prompt/max_tokens and the captured selection;
  never reload provider config. Recheck Task 2 permissions off-loop immediately
  before effects. Ask pauses with the exact resolved effect; approval consumes
  only that pending identity and is rechecked for revocation at dispatch.

  Persist review edits only in session memory, on every TextArea change, even
  while the screen later unmounts. Oversized/invalid edits make Accept unavailable
  with an error; never accept a silently retained older value. Check the finite
  review deadline and captured current actor atomically on Accept. A response
  already consumed, expired, cancelled, or from another run/step cannot advance.
  Accepted `review.text` is exactly the displayed edited text. The fixture's Note
  content resolves from that value; no automatic save after rejection.
- [x] **4. Implement physical retention and commit reconciliation.** Retain
  `asyncio.create_task(asyncio.to_thread(...))` for file/Notes work and shield its
  wait. Cancellation sets stop intent; it does not cancel a thread's asyncio
  wrapper or mark the slot free. Async model work may receive one cancellation
  request, followed by retained cleanup settlement from Task 1. Test cancellation
  before worker entry and cancellation of a close waiter as well as live workers.

  Generate the Note ID once, before dispatch. If creation raises after possible
  commit, perform same-owner authorized readback after the worker settles. Match
  active row ID, normalized title and exact content; record confirmed save or
  explicit uncertain outcome. Never create a replacement ID. Revoked readback
  authority or changed owner produces uncertain, not a guessed failure/success.
  A confirmed save after cancel carries `note_id` and saved-after-cancel copy,
  but never forwards an output to another step. Domain data is never deleted to
  simulate rollback. `close()` returns only after all setup/effect/cleanup tasks
  settle; failures preserve a nonaccepting state and remain observable.
  Catch Task 3's `LocalNoteCleanupError` separately from ordinary uncertain-write
  errors: even confirmed readback cannot erase failed physical-cleanup status.
- [x] **5. Prove lifecycle cases, then commit.** Use releasable `threading.Event`
  gates and real file-backed Notes fixtures, not sleeps. For a blocked writer,
  the essential assertions before releasing it are:

```python
session.cancel(run_id)
assert session.view().state == "stopping"
assert session.view().run_id == run_id
with pytest.raises(SessionError):
    session.start(other_ticket)
release_writer.set()
await session.close()
assert physical_worker_exited.is_set()
```

  Define `other_ticket` by a superseded setup ticket in the fixture, so this also
  exercises stale delivery rather than allocating an unapproved second run.
  Add full saved-fixture flow with a transport seam, duplicate Start/Accept/Ask,
  changed saved draft, late response, finite/unlimited review wait, budget
  exhaustion, current actor/destination changes, independently created app
  sessions, and zero writes to historical run tables. Fresh sessions have no
  resumed run/review even when the same authoring/Notes DBs reopen.
  Run all `Tests/Workflows/` tests and scoped lint/format checks.

```sh
git add -- tldw_chatbook/Workflows/session.py Tests/Workflows/test_session_admission.py Tests/Workflows/test_session.py Tests/Workflows/test_session_lifecycle.py
git commit -m "feat(workflows): execute one session-owned sequential run"
```

## Task 5: App lifecycle and existing Workflows controls

**Files:** Modify `tldw_chatbook/app.py`,
`tldw_chatbook/UI/Screens/workflows_screen.py`,
`tldw_chatbook/css/features/_workflows.tcss`; create
`tldw_chatbook/UI/Workflows_Modules/run_controls.py`,
`Tests/UI/test_workflows_run.py` and
`Tests/ProductionApp/test_workflows_session_lifecycle.py`; extend
`Tests/UI/test_app_quit_guard.py`. At the app/session integration boundary,
also update `Workflows/session.py` and `Tests/Workflows/test_session.py` so the
captured actual Notes DB path always joins app-supplied protected DB paths in
`RunBindings` (no second capture/owner or raw DB inspection).
The reviewed shutdown correction also touches existing `Workflows/authoring.py`,
`Workflows/draft_session.py` and `Tests/Workflows/test_authoring.py` for reversible
preparation and real same-owner usability after renewed confirmation is aborted.
Regenerate the committed CSS bundles through
`tldw_chatbook/css/build_css.py`.

**Consumes:** Task 4 session, actual app `notes_scope_service`, `notes_user_id`,
`unified_mcp_service`, existing runtime config snapshot and selected provider
catalog data, existing file picker and Library navigation message.
**Produces:** `TldwCli.ensure_workflow_session()` (lazy single owner),
`_confirm_workflow_session_quit()` (app-level loss confirmation) and
`_shutdown_workflow_session()` (drain); `WorkflowRunSetup` modal and
`WorkflowRunPanel` widgets in `run_controls.py`. Setup returns reviewed ordinary
inputs plus `RunSetup` or None. The session's `prepare()` captures the concrete
bindings in retained workers, then the UI displays `bindings(ticket)` for final
confirmation before `start(ticket)`; cancelling discards that setup ticket.

- [x] **1. Add actual-control RED tests.** Extend the existing production-CSS
  `WorkflowEditorHarness` pattern with real services and the session owner. Do
  not replace the screen's Run handler or quit implementation in the test.
  Test a real Button press and actual navigation, including different selected
  revision while the original run waits. Add an app lifecycle test where the
  Workflows screen is not mounted at quit. The review interaction must include:

```python
review = app.screen.query_one("#workflow-review-text", TextArea)
review.load_text("Human-edited summary")
await pilot.pause()
await pilot.click("#workflow-review-accept")
await pilot.pause()
```

  Assert the actual saved Note content and the Library landing, not merely that
  `NavigateToScreen` was posted. Run the new UI/lifecycle tests to capture RED.
- [x] **2. Compose the owner and safe quit path.** Add one `_workflow_session`
  field beside the authoring owner; constructing/visiting the editor must not
  start effects or initialize a second storage owner. Capture Notes identity,
  user, permission profile and provider facts once for setup. Retain resolver/
  Notes capture workers even if the setup modal closes. Missing prerequisites
  leave authoring usable and explain why Run is unavailable.

  Integrate with `_confirm_and_quit`, not only `_shutdown_app_owned_lifecycles`:
  approved cleanup eventually exits, so a late close-hook exception alone cannot
  keep the app open. Confirm Stay/Cancel run and quit at app level, pin the view
  generation, recheck after other awaited confirmations, fence controls/effects,
  flush drafts, drain the workflow, then allow existing service-close/exit work.
  Stay, failed draft flush or stale confirmation must not silently cancel a run.
  This applies before run cancellation is accepted. A later Console Stay after
  accepted cancellation restores future admission only after successful physical
  settlement; the cancelled attempt is not resumed. Pre-quit authoring uses its
  reversible preparation barrier; permanent authoring close is final teardown.
  If physical drain fails, remain open with a stopping/error indication and do
  not enter unconditional cleanup. Normal teardown calls the same idempotent
  drain before closing Notes; unrelated lifecycle owners retain their behavior.

```python
owner.begin_close()
try:
    if workflow_authoring is not None:
        await workflow_authoring.flush()
except (Exception, asyncio.CancelledError):
    owner.abort_close()  # No cancellation was accepted by close yet.
    raise
await owner.close()  # Failure stays fenced; caller must not enter exit cleanup.
```

- [x] **3. Wire the existing Run button and minimal UI.** Keep library,
  navigator and continuous collapsed authoring form unchanged. Dirty drafts
  require Save revision or an explicit Run saved revision choice; show the exact
  workflow/revision that will run. History inspection can run only its selected
  immutable saved revision, never the current draft by accident.

  Setup uses existing `EnhancedFileOpen(filters=["*.txt"], context="workflow_source")`,
  a captured provider/model choice (existing values plus explicit model text),
  editable Note title and fixed Local Note destination. Resolve localhost before
  approval; show selected and concrete numeric endpoint, actual model, file and
  Notes destination, and explicitly show keyless execution. A credentialed or
  unsupported provider is unavailable for this slice, not silently converted
  into a different provider. Do not auto-discover/fallback during model dispatch. Render
  the exact disclosure: “Session only: leaving this screen keeps the run;
  quitting loses pending review and intermediate results. Saved Notes remain.”

  Add one collapsible session panel, using existing Button/Input/TextArea/Static
  states and `$ds-*` spacing/status tokens. Show original run/revision, current
  step, Ask approval, editable review, Cancel and confirmed Open Note. Rest,
  hover, focus and disabled states must remain visible at 160x48, 110x36 and
  60x20. No new diagram/carousel or competing authoring modes. Subscribe on mount,
  unsubscribe on unmount; unmount must never cancel the session. Keep private
  run payload out of saved screen state, durable draft JSON and Console metadata.

```python
self.app_instance.post_message(NavigateToScreen(
    TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_NOTE_ID: confirmed_note_id}
))
```

  Open only the confirmed captured-destination result. Refuse unavailable/changed
  destination instead of routing an old ID into whichever library is now selected.
  Note-step readback and open routing are distinct: later legitimate edits are
  not a reason to overwrite or recreate the saved Note.
- [x] **4. Test interactions and shutdown order.** Cover stale/double button
  delivery, blocked setup, invalid edited review, expiry while off-screen,
  rejection, cancellation while Note commit is held, navigation away/back,
  quit from another screen, Stay, failing draft flush, cancelled quit waiter,
  failed drain and successful normal close. Inspect painted text and hit regions;
  test keyboard Tab/Enter/F6 without adding printable shortcuts that hijack fields.
  Run targeted UI, lifecycle and existing editor/paging tests, then CSS governance:

```sh
"$WORKFLOW_PY" tldw_chatbook/css/build_css.py
"$WORKFLOW_PY" -m pytest -o addopts= -q Tests/UI/test_workflows_run.py Tests/UI/test_workflows_editor.py Tests/UI/test_workflows_paging.py Tests/UI/test_app_quit_guard.py Tests/ProductionApp/test_workflows_session_lifecycle.py Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py Tests/UI/test_workflows_stylesheet_loading.py
```

- [x] **5. Self-review, scoped lint/format and commit exact paths.** Stage the
  source/test files above and only generated CSS outputs changed by the builder;
  inspect the staged diff before `git commit -m "feat(workflows): wire session execution into the app"`.
  Do not stage unrelated task files, fixtures or UAT scratch.

## Task 6: End-to-end qualification and user documentation

Qualification amendment: the merged Open Note test exposed a deterministic
nested-mount race in `Widgets/Library/library_notes_canvas.py`. The old readiness
guard sees authority/title before deeper mode buttons exist. A canvas-local
readiness flag may defer presentation updates until the existing completed-mount
hook applies the latest retained state. Cover it with a permanent held-mount
regression and existing Library recompose/Workflow Open Note tests; no framework,
storage, owner, or design change. This routine fix needs no new ADR. Separately,
the existing sensitive-logging test capture may snapshot/set/restore downstream
logger levels to remove proven test-order dependence without weakening checks.

A second held-callback reproduction found stale `WorkflowEditor.restore_view`
focus/scroll restoration overriding a newer field selection. Permit a narrow
focus-identity guard in `UI/Workflows_Modules/editor.py` and a permanent regression
in `Tests/UI/test_workflows_editor.py`; preserve invalid-field visibility and
ordinary restoration without timing delays or a new focus framework. No new ADR.

**Files:** Add `Tests/Workflows/test_file_to_note_integration.py` and
`Docs/Developer/Workflows/2026-09-16-first-run-uat.md`; update
`Docs/User_Guide/workflows.md` and the TASK-32691 record. Use the existing
`Tests/fixtures/workflows/file_to_note.json` unchanged as the common example.

**Consumes/produces:** No new production interfaces. Produce recorded targeted
test counts, real Notes readback, UI captures and explicit limitations, not a
schema-parity or exactly-once claim.

- [x] **1. Add a failing joined-boundary test.** Load the saved fixture through
  real document services, mount the actual Run/setup/review controls, dispatch
  the Task 1 real HTTP path to an owned loopback fixture, accept modified text,
  and read the real Note through captured NotesScopeService. Test with tldw_server
  unavailable and with conflicting current provider configuration after launch.
  Do not stub both sides of any integration being claimed. Capture RED on the
  preceding unintegrated boundary, or a focused removal of the exact new wiring;
  restore only task-owned edits afterward. Do not weaken an assertion to get RED.

```python
assert saved_note["content"] == "Human-edited summary"
assert session.view().note_id == saved_note["id"]
assert request_count == 1
assert inserted_note_count == 1
```

- [x] **2. Run the targeted merged set and static checks.** Include Tasks 1–5
  test paths, all `Tests/Workflows/`, touched Notes/gate/MCP regressions and
  `Tests/UI/test_workflows_projection_performance.py`. Run Ruff and formatter
  on new/changed scope. New files must be clean; for touched large legacy files,
  identify unchanged baseline diagnostics by source span and correct introduced
  findings. Do not assume the prior authoring task's exception waives this
  task's no-new-debt criterion. Save complete outputs and actual counts, not a
  truncated failure list. No full repository sweep without explicit permission.
- [x] **3. Run isolated live UAT with the user's llama.cpp at localhost:9099.**
  Allocate a private scratch directory, create/parse a complete disposable TOML
  before app import, and verify config, data, cache, permission and all DB paths
  resolve into it. Disable unrelated model-catalog/provider networking. Check
  OS-keyring behavior explicitly; a config override alone is not isolation.
  Do not copy real secrets or append duplicate TOML tables. Use the root venv
  with `PYTHONPATH` bound to this worktree; no package installs or endpoint switch.

  Use a non-sensitive known `.txt`, inspect the endpoint's actual model ID,
  launch the real app and complete each supported-width walkthrough. Exercise
  Ask, edit/Accept, reject, cancel, off-screen review, Stay and Cancel run and quit.
  Verify exact approved content in a Local Note and its Open Note landing. Restart
  the same scratch profile: definitions/Note survive, active run/review does not
  return and no POST/write is replayed. If the endpoint is unavailable, report
  live UAT blocked; do not replace it with Ollama, a mock or a health-check claim.
- [x] **4. Record evidence and update the guide.** Record commit, Python/Textual/
  HTTPX versions, effective isolated paths, actual provider/model/endpoint,
  request/effect counts, reviewed Note identity/content comparison, terminal
  sizes/captures and log scan for `unhandled_exception|app_stopping`. Explain
  session data loss, independent app instances, uncertain commits and the
  stable-file assumption. Change “Run disabled” documentation only after the
  real integration works; list unsupported operations without implying full
  server parity. Keep source/prompt/response canaries out of ordinary logs.
- [ ] **5. Review and close the implementation task only on evidence.** Perform
  spec and code-quality reviews; a reviewer may reject any task independently.
  Recheck Backlog ID/path uniqueness, exact staged files and `git diff --check`.
  Add implementation notes linking ADR-138 and UAT, check only proven ACs, and
  use `backlog task edit 32691 -s Done` only once all outcomes pass. Preserve
  failed/blocked evidence and leave In Progress otherwise. Creating/pushing a PR
  or merging is a separate integration action, not implied by this plan.

## Spec coverage and execution handoff

| Approved requirement | Implementation / evidence |
| --- | --- |
| Selected localhost llama.cpp, fixed request limits, no fallback | Task 1 + Task 6 live model |
| Fresh authority, exact-effect Ask, Off/kill precedence | Task 2 + Tasks 3–5 revocation barriers |
| Read-only bounded `.txt`, existing Notes policy/transaction | Task 3 + real DB tests |
| Saved snapshot, typed references, five-step flow and budgets | Task 4 + joined fixture |
| Duplicate delivery, review/deadline/data-loss and physical settlement | Task 4 + Task 5 app quit |
| Continuous existing authoring UI, navigation and Open Note | Task 5 painted actual-app path |
| No new storage ownership or durable run recovery | Task 4 historical-row assertions + Task 6 restart |
| Targeted lint/performance/UI/live evidence and truthful docs | Task 6 |

Execute Tasks 1–3 as independently reviewable capabilities with disjoint writes;
Task 4 depends on all three; Task 5 depends on Task 4; Task 6 qualifies the joined
result. Never parallelize app-booting suites against one profile. Prefer fresh
subagents per implementation task with spec and quality review between tasks, or
inline execution with the same checkpoints. This planning document does not
report any implementation tests or live UAT as already performed.
