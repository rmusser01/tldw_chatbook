# Shared Console Mobile Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inspect and correct existing served Console agents from a phone over VPN, share ordinary chats and storage, and preserve accepted work across browser disconnection.

**Architecture:** Keep execution in the existing App-owned Console runtime. Retain one served child per admitted canonical storage identity, make browsers revocable attachments, and route authenticated phone reads and commands to native owners. Consolidate legacy mobile history through the existing recovery subsystem before replacing its separate engine.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, SQLite/FTS5, aiohttp/textual-serve, Pydantic, existing plain HTML/CSS/JavaScript, pytest and the existing Playwright served harness. No new runtime dependency.

**Spec:** [Shared Console ownership and mobile agent supervision](../specs/2026-10-25-shared-console-mobile-supervision-design.md)

**Decision:** [ADR-244](../../../backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md)

**Planning task:** [TASK-36729](../../../backlog/tasks/task-36729%20-%20Plan-shared-served-Console-ownership-and-mobile-supervision-implementation.md)

**Baseline:** `8dbf79077b10d5671dabd4f9cbe06ea9c2ee8c15`. This commit contains the approved specification and overflow correction; product implementation has not started. Re-check symbols and schema ceilings after rebasing.

**Status:** Implementation in progress. Tasks 1–7 passed independent task review; targeted evidence and inherited-check limits are recorded in their Backlog records and execution ledger. Tasks 8–11 remain pending. Final shared-mobile qualification and phone/VPN receipt remain open.

```text
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Direct implementation of the accepted ownership, transport, migration
        and mobile contracts. Update ADR-244 before any architectural deviation.
```

## Global Constraints

These requirements apply to every task. Quoted requirements are copied from the spec.

- "Start with `--serve`; native terminal attachment follows in a later phase."
- "Remove frontend-specific database selection and the mobile-only controller/store/bridge construction."
- "The executing process owns the fence: the served child for served execution, the native App process for native execution."
- "Projection congestion must not close that channel or trigger this shutdown path."
- "Draining and publication never wait on a browser subscriber."
- "Phone transcript subscriptions do not resize the terminal."
- "First-release phone Send has no attachments, one-shot prefill or staged RAG launch."
- "No socket reconnect automatically resends a mutation."
- "Do not persist drafts in browser localStorage or queue offline commands."
- "Preserve original run statuses, results, accounting and truthful delivery stamps; never forge `wake_delivered_at` to suppress that work."
- "No automatic deletion is part of this release."
- "Upgrade/consolidation requires older app processes to quiesce before the new ownership contract applies."
- "Run the full test suite only if the user opts in."
- Preserve native hooks, permission stores, scratch/binding/exclusion authority, project-instruction consent, thinking/capture policy, and dispatch recovery. A projection, ticket or VPN connection grants no execution permission.
- Use existing session admission/preparation, queues, interrupt-round resolvers and subagent mailboxes. Do not switch the desktop tab to perform phone work.
- Keep the mobile-surface exposure flag, session/CSRF/origin protections, device revocation, TLS and trusted-proxy policy. Canvas disablement revokes Canvas capabilities without disabling Console control.
- Read `backlog/docs/design-language.md` and `backlog/docs/component-patterns.md` before UI edits. Browser CSS uses documented semantic equivalents; do not paste TCSS tokens into browser CSS or edit the generated TCSS bundle.
- Every task starts by reading its task file, setting it In Progress and adding its own Implementation Plan through Backlog. The To Do records intentionally contain outcomes and links, not premature Implementation Plan sections.
- At execution time use an isolated checkout through the worktree skill; preserve existing dirty work. Commit only the task's named changes. Do not push, merge, migrate real user data or run an unrequested full sweep as part of this plan.

## Delivery order and review boundaries

This is one connected product change. Each row is a separate PR-sized deliverable and review gate. Tasks 1, 2 and 7 are independent foundations; the numbered order is the safe default for one executor.

| Step | Backlog task | Earlier dependencies | Independently testable result |
| --- | --- | --- | --- |
| 1 | [TASK-36759](../../../backlog/tasks/task-36759%20-%20Fence-Console-execution-by-the-physical-agent-run-store.md) | Planning task | Native/run-store execution fencing and non-mutating inspection |
| 2 | [TASK-36760](../../../backlog/tasks/task-36760%20-%20Keep-served-control-and-projection-draining-independent-of-phone-clients.md) | Planning task | Authenticated control and event draining without phone use |
| 3 | [TASK-36761](../../../backlog/tasks/task-36761%20-%20Retain-served-Console-hosts-and-safely-reattach-terminal-browsers.md) | 1, 2 | Retained process and terminal reattachment |
| 4 | [TASK-36762](../../../backlog/tasks/task-36762%20-%20Admit-phone-owned-Console-input-and-explicit-agent-commands.md) | 1, 2 | Command-owned Send and native exact-target commands |
| 5 | [TASK-36763](../../../backlog/tasks/task-36763%20-%20Project-Console-chats-and-retained-decisions-from-runtime-owners.md) | 2, 4 | Runtime projections and background decisions |
| 6 | [TASK-36764](../../../backlog/tasks/task-36764%20-%20Reconcile-mobile-command-tickets-and-durable-outcomes.md) | 2, 4, 5 | Expiring tickets and durable outcome reconciliation |
| 7 | [TASK-36765](../../../backlog/tasks/task-36765%20-%20Preview-and-coherently-capture-legacy-mobile-history.md) | Planning task | Read-only migration preview and coherent capture coverage |
| 8 | [TASK-36766](../../../backlog/tasks/task-36766%20-%20Consolidate-legacy-mobile-history-with-recoverable-ownership-and-quarantine.md) | 1, 6, 7 | Recoverable history import and inert imported authority |
| 9 | [TASK-36767](../../../backlog/tasks/task-36767%20-%20Route-the-mobile-API-to-canonical-retained-Console-owners.md) | 3, 4, 5, 6, 8 | Canonical mobile API and retired mobile engine |
| 10 | [TASK-36768](../../../backlog/tasks/task-36768%20-%20Build-the-Chats-first-shared-Console-phone-workflow.md) | 9 | Usable phone workflow |
| 11 | [TASK-36769](../../../backlog/tasks/task-36769%20-%20Qualify-shared-served-Console-control-and-document-release-limits.md) | 10 | Real served/browser qualification and release documentation |

No partial stage may be described as shared mobile Console while it still creates `mobile/chat.db`, `mobile/workspaces.db` or `mobile/runs.db`. Keep existing behavior behind the current API until Task 9's verified cutover gate is ready; do not introduce another user-facing mode or permanent compatibility engine.

## Source map and verification discipline

Read the living references in `Docs/Architecture/{console,agent-runtime,web-server,backup-recovery,database-layer,console-file-authority}.md` plus ADR-244. ADR-094 governs accepted turns, ADR-126 recovery, ADR-233 authentication, ADR-230 preparation, and ADR-150/161 visual language. ADR-234/236 apply only where ADR-244 has not replaced them.

| Unit | Existing owner to extend | New focused source, where needed |
| --- | --- | --- |
| Physical execution fence | `ConsoleRuntime`, `AgentRunsDB`, existing recovery/writer callers | `Chat/console_execution_owner.py` |
| Wire identity and projection pressure | `Canvas/control_protocol.py`, `app_served.py`, relay publisher | `Chat/console_remote_models.py`, `Web_Server/served_console_events.py` |
| Retained host/terminal | `Web_Server/serve.py`, its AppService subclass, App composition | `Web_Server/served_console_hosts.py`, `Web_Server/console_web_driver.py` |
| Native commands | Received-intent/dispatch mixins, controller and agent bridge | `Chat/console_remote_commands.py` |
| Native reads/decisions | Store, runtime, `InterruptRoundHost`, relay publisher | `Chat/console_remote_projection.py` |
| Command receipts | Existing `AgentRunsDB` transactions and recovery adapter | `DB/console_command_receipts.py`, `Chat/console_remote_receipts.py` |
| Consolidation | Existing recovery inventory, service, CLI, journal and owner adapters | `Backup_Recovery/mobile_consolidation.py`, `mobile_consolidation_apply.py` |
| Phone API/UI | Existing mobile service/API, Library read services and `/m` assets | Extend existing files; no second frontend or engine |

Use `.venv/bin/python`, not the macOS system Python. Use real file-backed SQLite when worker threads must share a store; memory SQLite is appropriate for single-connection owner tests. Reuse `Tests/console_provider_doubles.py` and its typed provider destinations. Never make a provider test pass by omitting native persistence or permission checks.

For each task: record a failing new behavior test, implement, run its listed targeted tests, run scoped syntax/lint/format checks and self-review, then commit. Re-run only failed/changed connected tests when new evidence warrants it. Do not count a skipped browser test or static source assertion as runtime qualification. A baseline failure must be reproduced against the same pre-change source before attributing it to this work.

Use the existing Ruff qualification toolchain for changed Python: `python -m ruff check` and `python -m ruff format --check` on explicit changed paths. Do not reformat an entire legacy module to hide inherited debt. For Task 1's inherited runtime path, the existing ratchet commands are shown below; use the current task's exact paths and its own evidence file for subsequent tasks. Missing tooling or inherited failures are evidence gaps to report, not green checks. `python -m compileall -q` on touched Python, `node --check` on touched JS and `git diff --check` provide additional scoped checks.

```bash
.venv/bin/python scripts/terminal_qualification/format_ratchet.py snapshot --base HEAD --output Docs/superpowers/qa/2026-10-25-shared-console-mobile-supervision/task-36759-format.json --path tldw_chatbook/Chat/console_runtime.py
.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --baseline Docs/superpowers/qa/2026-10-25-shared-console-mobile-supervision/task-36759-format.json
```

### Task 1: Fence Console execution by the physical run store — TASK-36759

**Files**

- Create: `tldw_chatbook/Chat/console_execution_owner.py`.
- Modify: `Chat/console_runtime.py::ensure_activity_receipt_service`, `DB/AgentRuns_DB.py::__init__/reconcile_orphaned_runs`, `DB/private_sqlite.py` owner policy; paths are under `tldw_chatbook/`.
- Modify every production auxiliary opener: `Chat/console_worktree_recovery.py`, `Workspaces/change_retention.py`, `Widgets/settings_agents_panel.py`, and the temporary legacy `Web_Server/mobile_service.py` until its retirement.
- Test: create `Tests/Chat/test_console_execution_owner.py`; extend `Tests/DB/test_agent_runs_db.py`, `Tests/Agents/test_agent_runs_db_connection_reuse.py`, `Tests/Chat/test_console_runtime_lifetime.py`, `Tests/Chat/test_console_runtime_shutdown.py` and affected opener fixtures.
- Docs: `Docs/Architecture/console.md`, `database-layer.md`, `web-server.md`; link ADR-244 in implementation notes.

**Interfaces**

- Produces `ConsoleExecutionOwner.acquire(run_store: Path) -> ConsoleExecutionOwner`, `require_current(run_store: Path) -> None`, `close() -> None`, and context-manager support. Errors `ConsoleExecutionBusy` and `ConsoleExecutionUnavailable` distinguish a contender from failed/unsafe lock admission; both refuse execution.
- Extend file-backed `AgentRunsDB` construction with `execution_owner: ConsoleExecutionOwner | None` and add `AgentRunsDB.open_read_only(db_path: Path) -> AgentRunsDB`. In-memory isolated DBs need no OS lease. Non-owning file reads skip schema initialization and reject every mutation/recovery entry.
- Consumes existing `Utils.private_paths.open_private_lock_stream`, `verify_trusted_directory` and `Utils.platform_files.fcntl/os`. Keep execution ownership separate from maintenance admission and the warning-only `Utils/instance_lock.py` API.

- [x] **Step 1: Add a real competing-process regression.** Place the complete test below in the new test file; import `Path`, `subprocess`, `sys` and the two owner symbols. The child imports only the dependency-light owner module and inherits the test's isolated environment.

```python
def test_other_process_cannot_execute_the_same_store(tmp_path):
    path = tmp_path / 'agent_runs.db'
    program = (
        'import sys\n'
        'from pathlib import Path\n'
        'from tldw_chatbook.Chat.console_execution_owner import '
        'ConsoleExecutionOwner, ConsoleExecutionBusy\n'
        'try:\n'
        '    ConsoleExecutionOwner.acquire(Path(sys.argv[1]))\n'
        'except ConsoleExecutionBusy:\n'
        '    raise SystemExit(19)\n'
        'raise SystemExit(0)\n'
    )
    with ConsoleExecutionOwner.acquire(path):
        result = subprocess.run(
            [sys.executable, '-c', program, str(path)],
            capture_output=True, text=True, timeout=10,
        )
    assert result.returncode == 19, result.stderr
    with ConsoleExecutionOwner.acquire(path):
        pass
```

- [x] **Step 2: Run the regression before adding the owner.** Run `.venv/bin/python -m pytest Tests/Chat/test_console_execution_owner.py -q`; the new owner import must fail. Then add variants for two chat filenames in one parent, safe creation before the DB exists, a surviving child after parent death, actual holder death, and alias/unsupported-lock refusal. Use IPC entry/exit barriers, not sleeps, to establish contention.
- [x] **Step 3: Implement the retained OS lease and guard all disk execution opens.** Key by the secured physical parent plus fixed run-store leaf, verify the existing store when present, and reject aliases that do not preserve one authority. Use a stable private lock leaf beside that resource and never unlink it. Keep the acquired stream strongly referenced through physical teardown; lock-file/PID existence is informational only.

```python
from tldw_chatbook.Utils.platform_files import fcntl
from tldw_chatbook.Utils.private_paths import open_private_lock_stream

# Inside acquire, after native path/identity admission:
stream = open_private_lock_stream(lock_path)
try:
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BaseException:
    stream.close()
    raise
```

`lock_path` is derived only from that admitted resource. Translate genuine contention to Busy and all other lock/path failures to Unavailable. Do not copy the advisory helper's fail-open exception handling. Acquire before an execution-capable constructor, migration, orphan sweep or automatic recovery. `require_current` checks the original process and admitted identity; it is not a serializable boolean permission. Propagate that same owner into worktree recovery and retention writers. Do not release it while a retained worker/DB callback can still write.
- [x] **Step 4: Add and use the sweep-free read path.** Add the private SQLite read policy for the actual AgentRuns owner, use read-only connection mode without initialization, and reject writes rather than silently upgrading to an execution owner.

```python
def _require_execution_owner(self):
    if self.is_memory_db:
        return
    if self._read_only or self._execution_owner is None:
        raise ConsoleExecutionUnavailable('execution_owner_required')
    self._execution_owner.require_current(self.db_path)
```

Call this before constructor-side schema/reconciliation and explicit recovery/mutation paths. Audit `rg -n 'AgentRunsDB\(' tldw_chatbook Tests`; assign readers versus execution writers explicitly and update connected file-backed fixtures. Do not bypass the disk guard globally in conftest. The native App's empty runtime holder and ordinary saved-history reads do not acquire execution authority.
- [x] **Step 5: Verify native compatibility and commit.** Run the new test plus the five listed connected modules, `Tests/Chat/test_console_worktree_recovery_lifetime.py`, `Tests/DB/test_private_sqlite.py` and `Tests/Backup_Recovery/test_agent_runs_recovery_schema.py`. Assert read-only opens leave file/schema/run rows unchanged, failed contenders perform no sweep, and owner release follows physical cleanup. Apply scoped static checks, update docs/task notes, and commit `fix: fence Console execution by physical run store` using explicit file paths.

### Task 2: Independent control lifetime and bounded projection recovery — TASK-36760

**Files**

- Create: `tldw_chatbook/Chat/console_remote_models.py`, `tldw_chatbook/Web_Server/served_console_events.py`.
- Modify: `Canvas/control_protocol.py`, `app_served.py`, App served-control construction in `app.py`, `Web_Server/serve.py::on_startup/_disable_canvas_runtime/_attach_mobile_relay`, `Web_Server/served_relay_publisher.py`, and the existing relay-consumer wiring in `Web_Server/mobile_service.py`.
- Test: extend `Tests/Canvas/test_control_protocol.py`, `Tests/Web_Server/test_served_relay_publisher.py`; create `Tests/Web_Server/test_served_console_events.py`.
- Docs: `Docs/Architecture/web-server.md`; preserve ADR-233 and amended ADR-236 scope rules.

**Interfaces**

- Produces frozen, extra-forbidden Pydantic `ConsoleHostIdentity(server_epoch: str, console_id: str)` and `ConsoleTarget` below. The Console ID is stable for one retained child; a replacement child receives a new ID even under the same server epoch. Apply existing `validate_opaque_identifier` to all non-null identity fields; reject raw paths and labels as identities.
- Produces `ConsoleEventBuffer(limit: int = 64)`, `publish(kind: str, payload: dict) -> int`, `read(after_sequence: int) -> ConsoleEventBatch`, async `wait_changed(after_sequence: int) -> None`, `note_changed(sequence: int) -> None`, and `acknowledge_snapshot(sequence: int) -> None`. `ConsoleEventBatch` has `sequence: int`, `reset_required: bool`, `events: tuple[dict, ...]`.
- The private protocol adds bounded `console.snapshot`, `console.events`, `console.ticket`, `console.command`, `console.receipt` request/response pairs and `console.changed` events. Each pair uses the existing `.request`/`.response` suffix convention. Later tasks supply their native handlers; unsupported handlers refuse rather than constructing an engine. Bump the private protocol from version 2 to 3 for both parent and child; reject old versions.

```python
from pydantic import BaseModel, ConfigDict

class ConsoleHostIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    server_epoch: str
    console_id: str

class ConsoleTarget(ConsoleHostIdentity):
    session_id: str
    incarnation_id: str
    turn_id: str | None = None
    run_id: str | None = None
    handle_id: str | None = None
    round_id: str | None = None
```

- [x] **Step 1: Add a bounded-overflow test with no subscriber.** The new buffer's reset marker must survive trimming and clear only on a current snapshot acknowledgement.

```python
def test_unread_projection_overflow_requires_snapshot_not_disconnect():
    from tldw_chatbook.Web_Server.served_console_events import ConsoleEventBuffer

    buffer = ConsoleEventBuffer(limit=2)
    for revision in range(7):
        buffer.publish('chat_changed', {'session_id': 'chat-a', 'revision': revision})
    batch = buffer.read(after_sequence=0)
    assert batch.reset_required
    assert len(batch.events) <= 2
    buffer.acknowledge_snapshot(batch.sequence)
    current = buffer.read(after_sequence=batch.sequence)
    assert not current.reset_required
```

- [x] **Step 2: Run RED and add the transport integration cases.** Run `.venv/bin/python -m pytest Tests/Web_Server/test_served_console_events.py -q`. Extend the existing broker/client loopback fixture with >64 child projection events before any mobile service exists, Canvas-disabled startup, mobile disabled, a paused subscriber and forged/old protocol envelopes. Assert the same authenticated connection survives projection overflow and command responses are still correlated.
- [x] **Step 3: Implement the bounded projection branch.** Retain a sequence counter and sticky reset marker in `ConsoleEventBuffer`; the buffer never owns native outcomes. In the broker, route only replaceable Console projection events into this bounded path. On overflow, coalesce/drop projection history and mark reset; do not raise `event_backpressure` for that projection loss. Continue resolving response frames directly through the existing pending map. Preserve authenticated connection failure on malformed messages or revoked child authority.

```python
# In the dedicated Console projection-event branch:
sequence = host_events.publish(message.message_type, dict(message.payload))
host_events.note_changed(sequence)
# Responses continue through the existing request-id resolver, not this buffer.
```

Add `note_changed(sequence: int) -> None` as the buffer's bounded wake signal: one set Event, no per-event task or unbounded queue. Existing Canvas events retain their feature semantics; if sharing a bounded carrier with Console projections, its overflow must also use a reset rather than destroying runtime control.
- [x] **Step 4: Move drain lifetime to child registration.** Register the lifecycle drainer before starting child publication. It drains without waiting on phone sockets and outlives every browser subscription. Replace `MobileChatService.attach_relay`'s exclusive lazy broker drain with observation of the lifecycle-owned projection state; keep its existing API behavior until Task 9 retires the engine. Consumers of a lost projection show reset/unavailable state until native refresh, never stale approval authority. Keep the runtime broker alive when Canvas is disabled; revoke only Canvas capabilities/routes. Keep parent/child launch secrets out of representations and wire projections. Actual control-channel loss remains observable for Task 3's shutdown path, rather than being hidden by a fake reconnect success.
- [x] **Step 5: Verify and commit.** Run `.venv/bin/python -m pytest Tests/Web_Server/test_served_console_events.py Tests/Canvas/test_control_protocol.py Tests/Web_Server/test_served_relay_publisher.py Tests/Web_Server/test_web_auth_integration.py -q`. Verify bounded memory/reset state under overflow and current-child-only response routing. Apply scoped static checks and commit `fix: decouple Console control from projection subscribers` with explicit paths.

### Task 3: Retained child hosts and terminal reattachment — TASK-36761

**Files**

- Create: `tldw_chatbook/Web_Server/served_console_hosts.py`, `tldw_chatbook/Web_Server/console_web_driver.py`.
- Modify: `Web_Server/serve.py::handle_websocket/on_shutdown/build_chatbook_app_service_class`, `app_served.py`, the served driver selection in `app.py`, and attachment readiness handling in `Web_Server/static/served_shell.js`.
- Test: create `Tests/Web_Server/test_served_console_hosts.py`, `Tests/Web_Server/test_console_web_driver.py`; extend `Tests/Canvas/browser/canvas_live_harness.py`, `test_canvas_served_flow.py`, `Tests/Web_Server/test_served_shell_browser.py`.
- Docs: `Docs/Architecture/web-server.md`, `console.md`.

**Interfaces**

- Consumes Task 1's `ConsoleExecutionOwner` in the child and Task 2's identity/event contracts.
- Produces `ServedConsoleHosts.get_or_start(storage_identity: str) -> ServedConsoleHost` (async, serialized per admitted identity) and `aclose() -> None` (async). Identity is derived from admitted server resources, never a browser path.
- `ServedConsoleHost.attach_terminal(browser_session_id: str, width: int, height: int) -> TerminalAttachment` is async. Attachment exposes `generation: str`, async `send_bytes(data: bytes)`, `resize(width: int, height: int)`, `detach()`. Every method verifies the current attachment identity.
- `ConsoleWebDriver` extends the installed `WebDriver.on_meta` seam and adds `reattach(width: int, height: int, generation: str) -> None` on the App thread. It restores modes and schedules a full repaint without starting another input thread.

- [x] **Step 1: Add the stale-attachment regression.** In the new host test module, provide a local `RecordingAppService` with async `start/stop/send_bytes/send_meta/set_terminal_size`, an output callback and lists of received operations. Construct the host through a fixture `recording_host` using that service; this is a unit seam, not real-process qualification.

```python
async def test_old_attachment_cannot_resize_or_detach_successor(recording_host):
    host, service = recording_host
    first = await host.attach_terminal('browser-a', 80, 24)
    second = await host.attach_terminal('browser-b', 100, 30)
    with pytest.raises(StaleTerminalAttachment):
        await first.resize(10, 10)
    await first.detach()
    await second.send_bytes(b'x')
    assert service.input_bytes == [b'x']
    assert service.stop_calls == 0
    assert host.current_attachment is second
```

Declare `StaleTerminalAttachment` in `served_console_hosts.py`, and `current_attachment` as the host's read-only current-attachment property. Add concurrent `get_or_start` and slow/dead-socket tests. Run `.venv/bin/python -m pytest Tests/Web_Server/test_served_console_hosts.py -q` and observe RED.
- [x] **Step 2: Retain AppService and continuously drain output.** `handle_websocket` attaches to a host and its `finally` detaches only that exact attachment. Remove per-socket `AppService.stop`. AppService callbacks publish to a bounded attachment queue without awaiting a browser; queue pressure detaches/resets the terminal subscriber while the AppService continues draining. Reuse the existing subclass/environment seam and keep one child start task per admitted identity.

```python
# The WebSocket handler's ownership boundary:
attachment = await host.attach_terminal(browser_session.id, width, height)
try:
    await self._process_attached_terminal(request, websocket, attachment)
finally:
    await attachment.detach()
```

Add `_process_attached_terminal(request, websocket, attachment) -> None` to the mixin as the existing authenticated input loop with generation-checked sends/resizes. Session revocation closes that browser's attachments/subscriptions and Canvas scopes, not the child. Reserve saved-conversation hydration under the host and register the persistent ID on save.
- [x] **Step 3: Implement an explicit render boundary.** Send a `console_reattach` meta packet carrying generation and dimensions. The driver posts work to the App thread; emit `console_attachment_begin` before restoring alternate-screen/mouse/bracketed-paste/cursor/sync state, apply dimensions, force full compositor invalidation, then emit matching `console_attachment_ready` after paint. The parent drops stale/detached output until the matching begin boundary; input stays disabled until matching ready. Do not re-emit the startup `__GANGLION__` framing sentinel or call `start_application_mode` again.

```python
# On the owner thread, after validating the attachment generation:
self.write_meta({'type': 'console_attachment_begin', 'generation': generation})
self.write('\x1b[?1049h')
self._enable_mouse_support()
self.write('\x1b[?25l')
self._enable_bracketed_paste()
self._request_terminal_sync_mode_support()
self.flush()
```

Route ordinary meta kinds through `super().on_meta`. Add a repo-owned driver subclass rather than editing `.venv` or vendor source. Test that input-thread start count stays one and that resize/repaint completion precedes readiness.
- [x] **Step 4: Join the actual host lifecycle to native shutdown.** On genuine parent/control loss close new admission and automatic wakes, invoke the existing bounded runtime shutdown and retain the execution lease through physical retirement. A replacement must remain blocked if the child survives or hangs. Server shutdown iterates retained hosts; explicit session close and native Stop retain their separate scopes. Do not convert a timer into proof of owner death.
- [x] **Step 5: Verify real retention and update changed old expectations.** Run the two new unit files plus `Tests/Chat/test_console_runtime_shutdown.py` and focused real-child cases in `Tests/Canvas/browser/test_canvas_served_flow.py`. Update the old tests named `test_terminal_websocket_echo_survives_private_control_channel_loss` and `test_hot_reload_branch_switch_exact_reopen_and_control_loss_keep_terminal`: genuine runtime-control loss now invokes bounded shutdown; disabling Canvas alone preserves runtime control. Replace two-browser/two-child assumptions with one terminal attachment plus independent phone clients while preserving Canvas cross-session refusal. Use the existing `start_live_served_stack` rather than a new server harness. Commit `feat: retain served Console hosts across browser disconnects` after scoped static checks.

### Task 4: Command-owned phone input and native exact-target controls — TASK-36762

**Files**

- Create: `tldw_chatbook/Chat/console_remote_commands.py`.
- Modify: `Chat/console_received_intent.py`, `console_received_dispatch.py`, `console_configuration_preparation.py`, `console_runtime.py`, `console_chat_controller.py`; extend `console_agent_bridge.py` only where its existing handle/run checks need an expected-run witness.
- Test: extend `Tests/Chat/test_console_received_intent_inputs.py`, `test_console_received_intent_custody.py`, `test_console_received_intent_transitions.py`; create `Tests/Chat/test_console_remote_commands.py`.
- Docs: `Docs/Architecture/console.md`, `agent-runtime.md`.

**Interfaces**

- Consumes Task 2's `ConsoleHostIdentity` and `ConsoleTarget` and Task 1's executing runtime owner.
- Extend the existing frozen `ConsoleSessionInputSnapshot` with `input_origin: Literal['composer', 'command'] = 'composer'` and `command_id: str | None = None`. Do not replace `ConsoleReceivedTurnIntent` or add another admission slot.
- Add `ConsoleChatStore.command_input_snapshot(session_id: str, *, text: str, command_id: str) -> ConsoleSessionInputSnapshot`; only the runtime command path constructs this origin. It captures session/configuration/binding witnesses, submitted text, empty attachments/prefill and no desktop staged evidence.
- Produce `ConsoleRemoteCommands(runtime, host_identity)` with async `send(target: ConsoleTarget, text: str, command_id: str, *, expected_revisions: dict[str, int]) -> str` returning the existing custody turn ID, and async `execute(action: str, target: ConsoleHostIdentity | ConsoleTarget, payload: dict, command_id: str) -> dict`. Send payload has exactly `text` and `expected_revisions`; the revision map has `settings_revision`, `generation_settings_revision`, `context_policy_revision`, `identity_revision` and `conversation_binding_revision` from the last confirmed target snapshot. Require non-negative integers, excluding bool, and revalidate them before capture and native promotion. Initially support Send, main controls and subagent guidance/Stop; unsupported actions refuse. Task 5 adds decisions and Task 9 adds create/open through these same owners.
- Add controller `steer_run(session_id: str, text: str, *, expected_run_id: str) -> str | None`, `redirect_run` with the same signature, and `stop_run(session_id: str, *, expected_run_id: str | None, expected_turn_id: str | None, record_user_stop: bool = True) -> bool`. Stop requires an exact existing turn or run; do not allow an untargeted remote Stop of whatever happens to be next. Existing active-tab methods derive the current native target and delegate.

- [x] **Step 1: Add the independent-composition regression to the existing input fixture.** This complete test reuses that module's `input_store` and `_attachment` helpers.

```python
def test_command_inputs_ignore_desktop_composition_but_keep_native_scope(input_store):
    store, session = input_store
    store.set_session_draft(session.id, 'desktop first')
    store.add_pending_attachment(session.id, _attachment('desktop-only'))
    store.set_session_one_shot_prefill(session.id, 'desktop prefill')
    snapshot = store.command_input_snapshot(
        session.id, text='phone correction', command_id='command-a',
    )
    store.set_session_draft(session.id, 'desktop newer')
    assert store.session_inputs_are_current(snapshot)
    assert snapshot.draft == 'phone correction'
    assert snapshot.attachments == ()
    assert snapshot.one_shot_prefill is None
    assert store.session_draft(session.id) == 'desktop newer'
    assert session.pending_attachments[0].attachment_id == 'desktop-only'
    assert session.one_shot_prefill == 'desktop prefill'
```

- [x] **Step 2: Run RED and add promotion/race cases.** Run `.venv/bin/python -m pytest Tests/Chat/test_console_received_intent_inputs.py -q`. Extend the real custody fixture so phone input reaches the existing prepared/queued send while desktop typing and active-tab switches occur. Gate actual preparation entry and mutate native settings/binding separately to prove those changes still refuse. Assert one exact phone echo, no desktop composition transfer, no global staged-RAG consumption and no early accepted/applied claim.
- [x] **Step 3: Extend the existing input origin instead of editing the desktop draft.** Factor shared core witness checks within `session_inputs_are_current`; retain the original full composer comparison for composer snapshots. A command snapshot compares session/incarnation/settings/context/binding but has no authority over desktop input revisions. Keep `repr=False` on text-bearing fields.

```python
# In received admission, after session_inputs_are_current has checked its origin:
claim = store.claim_received_turn(
    intent.session_id,
    intent.turn_id,
    draft_revision=(
        intent.inputs.draft_revision
        if intent.inputs.input_origin == 'composer'
        else None
    ),
)
```

The native claim already supports `draft_revision=None`; reuse it rather than adding a parallel claim API. In phone preparation skip only composer-owned draft transfer, attachment/prefill capture, view-generation coupling and global staged-evidence checks. Keep native source/configuration/currentness, permissions, hook policy, physical callback custody and atomic promotion. `commit_session_input_draft` remains a composer operation and is never called for phone input. This is manual origin for native hook/budget semantics, not automatic-work authority.
- [x] **Step 4: Add target-based native controls.** Resolve the exact live session/incarnation before touching per-session maps, then check expected turn/run/handle under the native ownership fence. Extract the existing active control bodies into those explicit-session methods; active wrappers delegate without temporarily switching `store.active_session_id`. Include response-rule preparation, queued-continuation Stop and physical cancellation paths from the existing `stop_active_run`, not only streaming cancellation. Return native refusals unchanged. Subagents use `steer_subagent`/`cancel_subagent` and their conversation-owned mailbox; add no subagent Redirect.

```python
def steer_active_run(self, text: str) -> str | None:
    session_id = self.store.active_session_id
    if session_id is None:
        return 'no active run to steer — plain messages queue for the next turn'
    run_id = self._current_primary_run_id(session_id)
    if run_id is None:
        return 'no active run to steer — plain messages queue for the next turn'
    return self.steer_run(session_id, text, expected_run_id=run_id)
```

Implement `_current_primary_run_id(session_id: str) -> str | None` from the bridge's existing live-primary lookup and session's durable conversation identity; do not select the newest historical row as proof of live execution. Runtime `send` captures checked target configuration via `capture_received_configuration_owned`, constructs the existing received intent with command origin and submits it under native custody. Its return means Preparing; acceptance is observed later by Task 6.
- [x] **Step 5: Verify and commit.** Run the three existing input/custody files, `Tests/Chat/test_console_received_intent_saved_acceptance.py`, `Tests/Chat/test_console_remote_commands.py`, `Tests/Chat/test_console_runtime_shutdown.py`, and connected steering/subagent tests located by `rg -l 'steer_active_run|redirect_active_run|steer_subagent' Tests/Chat Tests/Agents`. Cover two concurrent chats, stale incarnation/run, worker completion during submission, queued-versus-consumed guidance and untouched desktop state. Commit `feat: add exact-target Console command input` after scoped static checks.

### Task 5: Runtime projections and answerable decisions — TASK-36763

**Files**

- Create: `tldw_chatbook/Chat/console_remote_projection.py`.
- Modify: `Chat/console_runtime.py`, `console_interrupt_rounds.py`, `console_remote_commands.py`, `Web_Server/served_relay_publisher.py`, child control-handler wiring in `app_served.py`.
- Test: create `Tests/Chat/test_console_remote_projection.py`; extend `Tests/Chat/test_console_interrupt_rounds.py`, `Tests/Web_Server/test_served_relay_publisher.py`, `Tests/Chat/test_console_runtime_lifetime.py`.
- Docs: `Docs/Architecture/console.md`, `web-server.md`.

**Interfaces**

- Consumes Task 2's event buffer/identities and Task 4's exact command dispatcher.
- Produces `ConsoleRemoteProjection(runtime, host_identity, events)` with `snapshot(target: ConsoleTarget | None, *, cursor: str | None = None, limit: int = 50) -> dict`, `subscribe(after_sequence: int) -> ConsoleEventBatch`, and `set_answerable(client_id: str, target: ConsoleTarget, *, answerable: bool) -> None`.
- Produce `InterruptRoundHost.remote_rounds(session_id: str) -> tuple[dict, ...]` and `resolve_remote_round(target: ConsoleTarget, *, kind: str, answer: dict) -> dict`, implemented through existing retained kind-specific resolvers. Remote snapshot rows include `supported`, a reason when unsupported, kind/round/run identities and current answerability, never widget references or automatic instruction bodies.
- Snapshot DTO keys: `server_epoch`, `console_id`, `sequence`, `reset_required`, `chats`, `messages`, `next_cursor`, `fleet`, `decisions`, `recoveries`. Live chat rows include the five-field `expected_revisions` map consumed by Task 4, with no draft/attachment revisions. Scope every non-null target before reading. Existing per-page maximum remains 100; bound serialized projection pages to 64 KiB with cursor continuation within a large text message. Include content revision in cursors so replaced/swiped text forces a reset rather than corrupting append offsets.

- [x] **Step 1: Add background-decision and privacy regressions.** Reuse `ConsoleChatStore` ephemeral sessions and native interrupt-round test setup. Add fixture `remote_round` returning `(projection, host, target)` for one real retained approval round in a background session; its finalizer stops that round through the existing native test cleanup. Attach no ChatScreen.

```python
def test_background_round_remains_answerable_for_the_other_phone(remote_round):
    projection, host, target = remote_round
    projection.set_answerable('phone-a', target, answerable=True)
    projection.set_answerable('phone-b', target, answerable=True)
    projection.set_answerable('phone-a', target, answerable=False)
    rows = host.remote_rounds(target.session_id)
    row = next(row for row in rows if row['round_id'] == target.round_id)
    assert row['answerable']
    assert row['supported']
```

In the same module, seed a literal automatic-instruction body and hidden thinking, take list/fleet/decision metadata snapshots, and assert those sentinel bodies never appear. Test allowed transcript content separately under current capture policy. Run the new module RED before adding projection hooks.
- [x] **Step 2: Read native state and publish invalidations from runtime lifetime.** Move the relay source from mounted approval-card callbacks to retained runtime/round state. Reuse the existing publisher's bounded loop/thread handoff and cadence; a small `console.changed` event can name session/revision and cause a paginated authoritative read rather than retaining a full transcript per subscriber. Keep every published metadata field allowlisted.

```python
def set_answerable(self, client_id, target, *, answerable):
    self._require_target(target)
    key = (target.session_id, target.round_id)
    clients = self._answerable_clients.setdefault(key, set())
    if answerable:
        clients.add(client_id)
    else:
        clients.discard(client_id)
    self._publish_round_answerability(key, bool(clients))
```

Implement `_require_target` through the same runtime/current-incarnation checks as commands, and `_publish_round_answerability(key: tuple[str, str | None], answerable: bool) -> None` through the native round host. Bound client registrations by authenticated live subscriptions and delete them on disconnect/resolution. Do not pause or resume a native deadline based solely on a stale device row.
- [x] **Step 3: Establish snapshot/event boundaries and byte limits.** Register a subscriber cursor before capture, obtain native snapshots under the owners' synchronization and return the matching sequence boundary. If any required event has been dropped or a cursor/message revision is stale, return `reset_required` and a current snapshot. Preserve the sticky reset until the current boundary is acknowledged. Chunk individual large text values as well as row pages; reject oversized control envelopes before attempting serialization. Read saved chat history through the canonical conversation service without hydration or starting a provider.
- [x] **Step 4: Route supported decision answers through retained owners.** Map approvals, user questions and supported confirmations to their existing `resolve_pending_*` methods with exact kind/session/run/round validation. Include runtime-owned project-instruction dispatch confirmations when their admitted projection can express the native review; otherwise expose an unsupported pending decision with an Open terminal action. Never auto-approve, deny or fabricate a completed resolution to simplify the phone UI. One winner resolves the original round; subsequent answers read its current outcome.
- [x] **Step 5: Verify and commit.** Run the new projection file, existing interrupt-round/publisher/runtime-lifetime modules and `Tests/Canvas/test_control_protocol.py`. Cover unmounted/background rounds, competing answers, two phone clients with one disconnect, Canvas disabled, sequence gaps, text replacement, byte-budget continuation and inert content. Commit `feat: project Console state from retained runtime owners` after scoped static checks.

### Task 6: Expiring command tickets and durable outcome reconciliation — TASK-36764

**Files**

- Create: `tldw_chatbook/DB/console_command_receipts.py`, `tldw_chatbook/Chat/console_remote_receipts.py`, `tldw_chatbook/DB/migrations/agent_runs_v24_to_v25_console_command_receipts.sql`.
- Modify: `DB/AgentRuns_DB.py` migration registry/property, `DB/recovery_operations.py`, `Chat/console_runtime.py` custody/acceptance observers, child control-handler wiring.
- Test: create `Tests/Chat/test_console_remote_receipts.py`, `Tests/DB/test_console_command_receipts.py`; extend `Tests/Backup_Recovery/test_agent_runs_recovery_schema.py`.
- Docs: `Docs/Architecture/database-layer.md`, `web-server.md`.

**Interfaces**

- Consumes `ConsoleRemoteCommands.execute`, native turn/queue/round outcome identities, Task 1's owner and Task 2's private request/response envelopes.
- Produces `ConsoleCommandReceipt` with `command_id`, `server_epoch`, `console_id`, `action`, `target`, keyed `payload_fingerprint`, `state`, optional `native_id`/`reason`, issuance/expiry facts and timestamps. States are `pending`, `accepted`, `queued`, `applied`, `refused`, `outcome_unknown`; no text body is stored here.
- `ConsoleCommandReceiptRepository(runs_db)` exposes `get(command_id: str) -> ConsoleCommandReceipt | None`, `claim(receipt: ConsoleCommandReceipt) -> ConsoleCommandClaim`, `transition(command_id: str, *, expected_state: str, state: str, native_id: str | None = None, reason: str | None = None) -> ConsoleCommandReceipt`, and `prune_invalid(owner_epoch: str, now_ns: int) -> int`. Frozen `ConsoleCommandClaim` contains `receipt: ConsoleCommandReceipt` and `inserted: bool`; only the inserter may issue the effect.
- `ConsoleRemoteReceipts(commands, repository, host_identity, *, monotonic_ns=time.monotonic_ns)` exposes `issue(action: str, target: ConsoleHostIdentity | ConsoleTarget, payload: dict) -> ConsoleCommandTicket`, async `submit(ticket: str, payload: dict) -> ConsoleCommandReceipt`, and `query(command_id: str) -> ConsoleCommandReceipt | None`. Frozen `ConsoleCommandTicket` contains `token: str` and `command_id: str`; the latter lets the client retain a query identity before requesting the effect. Issue is a submission ticket, not permission to skip native admission.
- Add repository `list_unresolved(*, cursor: str | None = None, limit: int = 50) -> ConsoleReceiptPage`; this frozen page contains `items: tuple[ConsoleCommandReceipt, ...]` and `next_cursor: str | None`. Add bounded `command_receipts` and `next_receipt_cursor` snapshot fields so a fresh page can discover pending/unknown outcomes after losing its in-memory command IDs. Scope reads to the admitted canonical profile; prior-epoch receipts remain reviewable but cannot be replayed.

- [x] **Step 1: Write the duplicate and expiry regression.** Add a test-only `RecordingCommands` whose async `execute` increments `calls` and returns `{'state': 'applied', 'native_id': 'native-a'}`. In fixture `receipt_owner`, create a real tmp-path AgentRunsDB under Task 1's lease, its repository, a fixed host identity, a mutable monotonic clock and `ConsoleRemoteReceipts`; return `(owner, commands, advance_seconds)` and close DB/lease after the test. The callback `advance_seconds(value)` increments that clock by `int(value * 1_000_000_000)`.

```python
async def test_duplicate_submission_is_not_reissued_after_expiry(receipt_owner):
    owner, commands, advance = receipt_owner
    target = ConsoleTarget(
        server_epoch='epoch-a', console_id='console-a',
        session_id='chat-a', incarnation_id='incarnation-a',
    )
    payload = {'text': 'one exact correction'}
    ticket = owner.issue('send', target, payload)
    first = await owner.submit(ticket.token, payload)
    assert (await owner.submit(ticket.token, payload)).command_id == first.command_id
    advance(301)
    assert owner.query(first.command_id) is not None
    with pytest.raises(ExpiredConsoleCommand):
        await owner.submit(ticket.token, payload)
    assert commands.calls == 1
```

The recording dispatcher isolates receipt behavior from native turn execution; native target currentness remains covered by Task 4. Add `ExpiredConsoleCommand` to the receipt module. Test payload mismatch, HMAC/deadline tampering, prior epoch, full budget and native Stop with a full receipt table. Run both new files RED.
- [x] **Step 2: Add the executed migration and strict repository.** At this baseline AgentRuns is v24; bump to v25 and execute the named SQL file through its installed migration registry. Use the existing AgentRuns transaction owner; do not create another DB. Include exact recovery schema catalog/read-authorizer changes and backup coverage. Core table shape:

```sql
CREATE TABLE console_command_receipts (
    command_id TEXT PRIMARY KEY,
    owner_epoch TEXT NOT NULL,
    console_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target_json TEXT NOT NULL,
    payload_fingerprint TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN (
        'pending', 'accepted', 'queued', 'applied', 'refused', 'outcome_unknown'
    )),
    issued_ns INTEGER NOT NULL,
    expires_ns INTEGER NOT NULL CHECK (expires_ns > issued_ns),
    native_id TEXT,
    reason TEXT,
    updated_at TEXT NOT NULL
);
```

Every query is parameterized; cap retained live/unknown records at a fixed documented internal bound and refuse new non-idempotent effects before dispatch when capacity is exhausted. Never prune pending/unknown evidence because its TTL elapsed. Reuse referenced native recovery facts instead of copying bodies or creating a second recovery state machine.
- [x] **Step 3: Issue authenticated immutable tickets and claim before effects.** Generate a random command ID and per-host secret, canonically encode action/target/payload fingerprint/host epochs and the monotonic deadline, and authenticate with stdlib HMAC and constant-time comparison. Store `owner_epoch` as the server-epoch/Console-ID pair, not the server epoch alone. The child verifies it even when no receipt exists; the browser cannot extend expiry. Do not reuse the browser CSRF token or expose a child launch secret as the ticket key. Return the token and command ID together; the API does not need to decode an opaque token to invent a query ID.

```python
expires_ns = issued_ns + 300 * 1_000_000_000
if ticket_epoch != self.host_identity.server_epoch or now_ns >= expires_ns:
    raise ExpiredConsoleCommand('command_expired')
claim = self.repository.claim(pending_receipt)
if not claim.inserted:
    return claim.receipt
```

Consumers must use the explicit insertion result, never infer ownership from `state == pending`. Matching retries return/reconcile an existing receipt, and mismatched payload reuse refuses without effects. Expiry blocks new submission but permits query and does not cancel already accepted work.
- [x] **Step 4: Couple state transitions to native custody.** `accept_received_intent` returns Preparing: keep the command pending until its existing custody acceptance callback reports actual accepted/queued status and native ID. Steering queued into a mailbox is accepted/queued, not consumed; Stop accepted is not proof the worker exited. Atomically store a receipt with native acceptance where the existing transaction allows it. Otherwise record an interrupted handoff as `outcome_unknown` and never replay it automatically. On owner loss invalidate old tickets and retain/query durable outcomes by canonical profile identity even if a live session no longer exists. Native exact-target Stop has an idempotent path under receipt capacity pressure; do not turn repeated Stop into unbounded rows or a new run.
- [x] **Step 5: Verify and commit.** Run both new files, `Tests/Chat/test_console_received_intent_saved_acceptance.py`, `Tests/Chat/test_console_dispatch_recovery.py`, `Tests/Canvas/test_control_protocol.py`, `Tests/Backup_Recovery/test_agent_runs_recovery_schema.py`. Inject interruption before claim, after claim, after native acceptance and before acknowledgement; assert no duplicate effect and honest unknown outcomes. Test valid-ticket retention, querying/pruning after expiry, original-versus-tampered target/action, unresolved outcomes and Stop at capacity. Commit `feat: reconcile Console remote command outcomes` after exact-schema and scoped static checks.

### Task 7: Read-only legacy preview and coherent capture — TASK-36765

**Files**

- Create: `tldw_chatbook/Backup_Recovery/mobile_consolidation.py`, `Tests/Backup_Recovery/mobile_consolidation_fixture.py`, `Tests/Backup_Recovery/test_mobile_consolidation_preview.py`.
- Modify: `Backup_Recovery/inventory.py`, `participants.py`, `profile_paths.py`, `recovery_service.py`, `launcher.py`, and `DB/recovery_operations.py` for explicit legacy-store declarations.
- Test: extend `Tests/Backup_Recovery/test_inventory.py`, `test_capture.py`, `test_agent_runs_recovery_schema.py`.
- Docs: `Docs/Architecture/backup-recovery.md`, `Docs/Backup-and-Recovery.md`.

**Interfaces**

- Produces frozen `CanonicalConsoleStores(chat_path: Path, workspace_path: Path, run_path: Path)` with `identity: str` as a derived read-only property from the physical/admitted resources and existing profile/native path boundaries; it is metadata, not execution permission.
- Produces `MobileConsolidationPreview` containing `source_dir`, `source_fingerprint`, `destination`, `has_data`, `needs_destination_review`, `issues`, schema identities, counts, mappings/dependency inventory and capture/rollback scope. It carries no live DB handles or captured bodies in metadata summaries.
- `preview_mobile_consolidation(source_dir: Path, destination: CanonicalConsoleStores) -> MobileConsolidationPreview` performs read-only observation; `RecoveryService.preview_mobile_consolidation(source_config: Path, target_config: Path) -> MobileConsolidationPreview` resolves those paths through existing admitted config selection, never browser input.
- Add recovery CLI `consolidate-mobile --source-config PATH --target-config PATH --preview`. It prints that reviewed inventory; mutation flags are added only with Task 8's apply/recovery support.

- [x] **Step 1: Add the absent-source and no-constructor-write regressions.** The following test is complete without opening a user database.

```python
def test_absent_mobile_source_preview_does_not_create_databases(tmp_path):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores, preview_mobile_consolidation,
    )

    target = tmp_path / 'canonical'
    target.mkdir(mode=0o700)
    stores = CanonicalConsoleStores(
        chat_path=target / 'chat.db',
        workspace_path=target / 'workspaces.db',
        run_path=target / 'agent_runs.db',
    )
    source = tmp_path / 'mobile'
    preview = preview_mobile_consolidation(source, stores)
    assert not preview.has_data
    assert not source.exists()
    assert not stores.chat_path.exists()
    assert not stores.workspace_path.exists()
    assert not stores.run_path.exists()
```

Run `.venv/bin/python -m pytest Tests/Backup_Recovery/test_mobile_consolidation_preview.py -q` RED. Add constructor traps around source inspection: patch legacy writer constructors to raise and prove preview never calls them.
- [x] **Step 2: Build reusable isolated source fixtures with installed schemas.** In `mobile_consolidation_fixture.py`, provide `seed_legacy_mobile(source_dir: Path) -> dict` and `seed_canonical_console(destination: CanonicalConsoleStores) -> dict`. Use real `CharactersRAGDB`, `WorkspaceDB` and owned `AgentRunsDB` only to create isolated fixtures, then close every creator before read-only inspection. Return expected source IDs, branch/reference sets and byte fingerprints, not mocks. Seed a branched chat, Default and conflicting named workspaces, excluded paths, a completed parent/child with an undelivered result, an active but fully settled automatic chain, an unfinished claim, a Goal and a referenced artifact. Use existing public DB test builders for each native record shape. This fixture is consumed by Tasks 8 and 11.
- [x] **Step 3: Admit exact physical sources and schema inventory.** Resolve legacy directories using current `profile_paths.user_data_dir`, independently resolve canonical DB overrides, group shared physical sources across candidate configs and set `needs_destination_review` when records cannot prove attribution. Follow existing private path/alias refusal rules. Inventory every populated source table against installed exact schema/owner policy; classify derived indexes separately and refuse unknown populated content. List message-parent, workspace, chain/run, definition, step, trace and artifact dependencies explicitly. Do not infer equivalence from titles or path strings.

```python
# Destination is explicit even when several configurations name one source.
if len(candidate_destinations) > 1:
    needs_destination_review = True
    issues.add('mobile_destination_review_required')
```

This review issue blocks apply until a fresh explicit destination choice is bound; it is not a heuristic that assigns mixed history to the launching profile.
- [x] **Step 4: Add explicit capture participants and preview-only CLI.** Extend existing inventory/owner declarations so `mobile/chat.db`, `mobile/workspaces.db`, `mobile/runs.db` and required referenced files are inside the normal capture scope. Reuse the installed exact schema adapters; introduce explicit legacy logical identities, not guessed filename globbing. Capture through `MaintenanceSession` and existing WAL-consistent SQLite capture, with affected canonical destination rollback coverage. Honor encrypted credential/capture policy. The preview CLI prints source, selected destination, preserved scope, unresolved issues and source identity without invoking the apply path.
- [x] **Step 5: Verify and commit.** Run the new preview file and exact inventory/capture/schema modules touched by the declarations. Test WAL-visible source writes, missing dependencies, unsupported schema, unsafe aliases, two configs sharing `user_data_dir()/mobile` but overriding canonical DB paths, and empty source. Assert preview/capture do not reconcile source runs or mutate original bytes. Commit `feat: preview and capture legacy mobile Console data` after scoped static and schema checks.

### Task 8: Journaled consolidation and imported-authority quarantine — TASK-36766

Execution clarification (R122, before accounting policy changes): issue one fixed
512 MiB public mobile-worker aggregate with the ORIGINAL shared 16 MiB native
T/S/input/proof child, atomically charging both for child work. Ordinary defaults,
status and cutover observer remain unchanged. Store the aggregate in a distinct
native admission ledger slot; preserve slot 2 and all original child identities.
New coordinator/frame/terminal/rollback work uses the aggregate explicitly; keep
all pre-admission, native validation/custody and both-bound before-effect checks.
Record actual phase deltas and qualify full populated/no-data/recovery paths plus
original/default exhaustion and atomic refusal. The larger fixed envelope is not
measured success; no reset, refund, hidden retry, caller limit or authority change.
ADR244 and the immutable R122 snapshot/current checkpoint bind this correction.

Execution clarification (R124, before capacity edit): supersede ONLY R122's
public512MiB literal with fixed2GiB after actual measured prefix/capture exhaustion.
Original native16MiB/ordinary16MiB, one genuine coupled pair and all current
validation/independent limits remain. Preserve failed epochs; qualify five
coupling controls then one fresh guarded populated start with unchanged-charge
phase/child evidence and corrected isolated credential setup. No retained-operation
continuation is authorized. ADR244 and the immutable R124 snapshot/checkpoint bind
this change; engineering margin is not complete-path success or heap measurement.

**Task8 execution clarification — R125:** implement fixed SourceClaim-owned
ancestor physical custody after reciprocal reservation and before child EX,
with full fresh raw/native validation and no nested ancestor locks/cache/third
budget. Mobile retained-copy no-archive calls authenticate source evidence only;
actual acquired archive/plan/staging/publication binding, Complete current review
and new safety capture remain mandatory before effects. ADR244 R125 and its
immutable snapshot/checkpoint precede code. Current successor and later-copy
qualification are open;63574 is a private-fixture materialization refusal, while
57976 is only the first-operation frame identity/pair proof.

**Task8 execution clarification — R126:** after the actual successor budget
refusal and unchanged-source diagnostic, use the existing native journal frame
only for genuine public initial captured-input planning and reader retirement.
Close it before history/owner/target effects; private native16 remains unframed.
Keep every full verification and the same coupled2GiB/16MiB pair. ADR244 R126 and
the supplemental checkpoint precede this sequencing edit; no budget increase or
stopped verification optimization is authorized. Actual successor/private/default
and post-retirement consumer qualification remains required.


**Task8 execution clarification — R127:** use the existing public journal frame
only inside the three fixed pure receipt/row/recovery-history verification
bodies, closing before subsequent writes, commit/rollback or owner retirement.
Preserve private unframed16MiB, the public coupled pair and every full native
check. ADR244 R127 and the current checkpoint precede sequencing code. The
source-bounded before-effect coordinator tail and genuine refusal test remain
open independently of framed success; no cap or authority change is approved.

Task 8 execution clarification R119: qualify the SAME completed target across an
actual native theme save and a fresh ordinary process. Only the postterminal
observer may accept current persisted native config binding with unchanged
ancestors, routes, physical stores, reviewed cohort and per-row candidate
membership. Recheck current observations and native registry/bindings after
owner/journal reads under the same allowance. Pending/finish/rollback/successor
proofs remain strict. Record direct-edit, redirect, new/joined candidate, unbound,
ancestor, concurrent-publication and pending refusals. Integrated children install
both normal guards; earlier narrower scripts remain historical evidence. ADR-244
and the pre-edit native SHA/AST/status checkpoint govern this implementation.

Execution clarification R120: fixed pre-cutover rollback continuation authenticates
the original canonical archive with the supplied password and validates its exact
fixed owner/member/schema/stage/move/metadata closure under actual native custody.
Retain original claim/flock, strict selectors/pending and physical source-path
witnesses; remove only the newly introduced live semantic source preview. The old
fingerprint remains an immutable operation reference, not a fresh content claim.
Import/finish/successor checks remain strict. Preserve all source/current/capture
bytes and same-call budget/full verification/before-effect cost checks. ADR244 and
the separate R120 pre-edit checkpoint govern this clarification; qualify the
actual retained intended-only rename with both guards and honest epoch reporting.

Execution clarification (R121, before reader-policy changes): price initial generic
preview/discovery and both held re-reviews on the same early-issued allowance.
Use actual existing PreviewScope/DiscoveryScope issuers, fixed prefetch/schema
readers and bounded custody. Preserve the entire installed registry/inventory,
Complete semantics, register-before-maintenance ordering and strict post-binding
review. No unpriced adapter fallback, new role, caller records or allowance reset.
Qualify actual unbound populated data, not only no-data or parameter presence;
ADR244 and the R121 immutable snapshot/current checkpoint bind this clarification.

Execution clarification (R118, before output-policy changes): allow the exact
native live partition's default-service encrypted output through only the
authenticated service-container overlap. Keep current journal intention, derived
filename, original budget/custody, profile-root disjointness and physical identity
checks at capture preflight and both writer checks. All other overlap guards stay
in force. Qualify the default no-data operation and finite forged/retired/review/
custom/unknown-child/profile-overlap/parent-change refusals; ADR244 and the R118
snapshot/checkpoint bind this correction. Acceptance criteria remain unchanged.

Execution clarification (R117, before enrollment behavior changes): explicit
selected start may compose first native enrollment for missing source/target
profiles only. Prove per-selector complete configured roots and fixed source claim
parent; observe default config controls through installed exact exclusions without
enrolling their enclosing parent. Register before native maintenance, then recheck
and publish under unbound/all-overlap exclusion retained through import. Preserve
existing bindings and the single early-issued budget/custody. Post-binding native
review must be equivalent for all reviewed user scope; new sibling data yields
fresh-review-required before preparation. Qualify ordinary unbound no-data and
populated profiles plus refusal/interruption cases. ADR244 R117 and its source
snapshot/checkpoint bind this clarification; acceptance criteria remain unchanged.

Execution clarification (R116, before dependent no-data changes): preserve native
capture completeness. For an uninitialized required canonical target, qualify an
explicit ordinary app `--config PATH` startup and physical retirement, followed by
a fresh preview/capture; do not synthesize original absence or initialize inside
the importer. Refuse incomplete inventory before preparation/pending/claim. Prove
full no-data cutover on that same initialized target, with any supported optional
Runs creation journaled, no legacy constructors and ordinary post-retirement reads.
The rollback image covers the actual post-initialization pre-import state. The
R115 terminal/rollback/lineage requirements and ADR-244 remain otherwise unchanged.

**Execution dependency notes (R98/R99; original Task8 AC/status/BASE retained).** Complete R/I/C/T/S pure planning is independently approved through `01f42ea55592296224f69b5964449b3fd3bc91b4`; this is not owning-write or cutover approval. Before the pure A-runs foundation, support the actual pre-feature `81/8/24` source alongside existing `81/8/25`, keeping canonical AgentRuns26 unchanged, captured24 bytes/schema intact and ONLY authenticated native24 `console_command_receipts` absence. Amend ADR244 and preserve contemporary before-code evidence for this fixed compatibility prerequisite; independently review it. Then derive complete PURE Run/Goal transformations and ALL-owner supported-content/reference/artifact preflight before native Chat/Workspace writer activation. The foundation has no imported DB/file/hold/receipt/dispatch/cutover effects and does not make pending `require_complete()` succeed. Subsequent real same-operation owner writes must obtain complete actual all-owner preflight before the first file publication or data batch, preserve native legal lifecycle and atomic rows/maps/holds/receipts, and resume from actual receipts. Actual runtime quarantine/history consumers, all-owner closure, public B recovery/cutover/full original Task8 review, Tasks9–11 and the root whole-branch gate remain mandatory.

**Files**

- Create: `tldw_chatbook/Backup_Recovery/mobile_consolidation_apply.py`, `Tests/Backup_Recovery/test_mobile_consolidation_apply.py`, `test_mobile_consolidation_recovery.py`.
- Create migrations: `DB/migrations/chachanotes_v81_to_v82_mobile_consolidation.sql`, `workspaces_v8_to_v9_mobile_consolidation.sql`, `agent_runs_v25_to_v26_mobile_consolidation.sql`, under `tldw_chatbook/`. Task 6 owns v25; rebase/re-number if another accepted migration advances any owner first.
- Modify: `Backup_Recovery/mobile_consolidation.py`, `journal.py`, `recovery_service.py`, `launcher.py`, `storage_admission.py`, relevant owner migrations in `DB/{ChaChaNotes_DB,Workspace_DB,AgentRuns_DB}.py`, `DB/{recovery_core_schema,recovery_operations,sql_validation}.py`.
- Modify authority consumers: `DB/automatic_work.py`, `Chat/console_fleet_wake.py`, `console_launch_wake.py`, `console_goal_runs.py`, and `Workspaces/registry_service.py`.
- Test: extend `Tests/Chat/test_automatic_wake_recovery.py`, `test_automatic_launch_recovery.py`, `test_console_goal_recovery.py`, `Tests/DB/test_automatic_work_migration.py`, relevant Workspace authority tests and exact recovery schema tests.

**Interfaces**

- Consumes Task 7's immutable preview/physical destination and capture scopes, Task 1's run-store authority and Task 6's schema ceiling.
- Produces `RecoveryService.start_mobile_consolidation(preview: MobileConsolidationPreview, *, destination_confirmed: bool, rollback_password: bytes | None) -> str` returning an operation ID; no `False` confirmation fallback. Resume/rollback use existing `start_recovery(operation_id, action=...)` and typed recovery CLI.
- `require_mobile_cutover(source_dir: Path, destination: CanonicalConsoleStores) -> None` in `mobile_consolidation.py` verifies the completed source/destination receipt or refuses writes. It does not create/import data merely because an API request arrives.
- Native AgentRuns methods `hold_imported_authority(cursor, operation_id: str, authorities: tuple[tuple[str, str], ...]) -> None` and `imported_authority_allowed(kind: str, authority_id: str) -> bool` gate exact imported chain/run/Goal/wake/chat-start identities. They do not gate unrelated new native work by a broad conversation flag.
- Workspace imported-binding holds use the same owner transaction and existing registry resolution. Imported roots remain unavailable to tools until exact native review; historical binding/exclusion records retain their original meaning. Unsupported release remains paused. Sending a manual turn never clears these holds.

- [ ] **Step 1: Add a crash-after-commit regression through real owner transactions.** Define fixture `consolidation_case` in `test_mobile_consolidation_recovery.py` using Task 7's seed helpers. It returns a case with `preview`, `service`, `destination`, `fail_after_batch: str | None`, `expected_counts: dict[str, int]`, and a `read_counts() -> dict[str, int]` helper reading real stores without reconciliation. The adapter exposes a test-only fault callback after each committed owner batch, passed as a constructor test seam rather than a production configuration option. The callback raises local `InjectedImportCrash` when the committed batch equals `fail_after_batch`.

```python
def test_resume_after_chat_commit_does_not_duplicate_history(consolidation_case):
    case = consolidation_case
    case.fail_after_batch = 'chat'
    operation = case.service.start_mobile_consolidation(
        case.preview, destination_confirmed=True, rollback_password=b'test-only',
    )
    failed = case.service.wait(operation, timeout=10)
    assert failed['state'] == 'failed'
    first = case.read_counts()
    assert first['chat'] == case.expected_counts['chat']
    case.fail_after_batch = None
    resumed = case.service.start_recovery(operation, action='finish')
    case.service.wait(resumed, timeout=10)
    assert case.read_counts() == case.expected_counts
    require_mobile_cutover(case.preview.source_dir, case.destination)
```

Use the production dependency order: committed prefix counts remain complete while later batches are finished, not duplicated. Add a repeat-after-completed case and verify full mapped reference/identity closure separately; counts alone are insufficient. Run the new recovery file RED and add SQL-level hold/delivery-stamp assertions.
- [ ] **Step 2: Claim the source once under source-addressed coordination.** Use a stable private lock/claim beside the secured legacy source, visible regardless of destination config/control root. Never unlink the lock. Require affected legacy/older app writers to quiesce and physically retire; if that cannot be established, refuse with a close-old-instance instruction rather than treating the advisory instance lock or absent heartbeat as proof. Under maintenance admission, revalidate source fingerprint, destination resources and explicit review, then journal the operation and persist its source claim before the first import batch. Store operation/control locator and reviewed destination identities in the bounded claim, validate that locator through existing recovery control-root rules, and let another config only resume/review that same operation, never redirect it. An absent-source cutover uses safe creation identity for its admitted source location and does not create legacy DBs.

```python
if existing_claim is not None:
    if existing_claim.destination_identity != preview.destination.identity:
        raise MobileConsolidationRefused('mobile_source_already_claimed')
    return resume_or_verify(existing_claim.operation_id)
```

Declare `MobileConsolidationRefused` in `mobile_consolidation.py`. `resume_or_verify(operation_id: str) -> str` is the service's existing-operation branch: inspect the stored journal and either return completed verification or enter explicit recovery; it never starts a second import. A changed fingerprint requires a new reviewed operation, not mutation of the old claim.
- [ ] **Step 3: Execute per-owner migrations and batch receipts.** Add `mobile_consolidation_batches(operation_id, source_fingerprint, batch_key, mapping_json, committed_at)` to each participating owner with primary key `(operation_id, batch_key)`. Add run authority holds and Workspace binding holds in their respective owners. Increment actual installed versions and execute the SQL files; update exact recovery catalogs and the ChaChaNotes table allowlist. Deterministic UUID mappings are suitable for UUID identities; allocate numeric identities and persist their mapping in the same owner transaction. Never guess that a UUID string can be inserted into an integer key.

```python
with destination_db.transaction() as cursor:
    existing = read_batch_receipt(cursor, operation_id, batch_key)
    if existing is None:
        apply_mapped_rows(cursor, mapping)
        insert_batch_receipt(cursor, operation_id, batch_key, mapping)
```

Implement those three adapter-local functions with fixed table/column policy and parameterized values. No cursor carries a transaction across DB owners. Re-read committed batch receipts before journal resume; journal progress alone is not proof a DB batch committed. Choose dependency order from the inventoried references, allowing only registered rewrite/deferral rules for cross-owner references. Reconcile Default through native invariant; preserve conflicting named workspaces and restrictions as distinct records.
- [ ] **Step 4: Quarantine authority atomically with imported records.** Record exact holds in the same transaction as the corresponding authority rows; mark imported automatic chains review-required without falsifying run status, accounting or `wake_delivered_at`. Hold terminal undelivered child run IDs even when no active unfinished reservation exists or a chain is absent. Gate native automatic admission, recovery seeding, wake result eligibility, Goal/chat-start dispatch and imported binding advertisement through these holds.

```sql
CREATE TABLE mobile_import_authority_holds (
    operation_id TEXT NOT NULL,
    authority_kind TEXT NOT NULL,
    authority_id TEXT NOT NULL,
    reviewed_at TEXT,
    PRIMARY KEY (authority_kind, authority_id)
);
```

Do not copy permission-store grants or activate imported folders. Exact reviewed release must use native current scope/consent; if no supported release exists, surface the hold and leave it in place. A new manual turn may obtain its own fresh native authority without releasing any old imported chain.
- [ ] **Step 5: Verify closure, publish cutover and support reviewed recovery.** Verify mapped identities/message trees/settings/workspace restrictions/run parentage/steps/artifact bytes and unknown-data coverage under the same admission fence. Publish completed cutover only after all owner receipts and verification are durable. Retain original stores/captures. Before cutover, rollback restores affected originals through existing recovery ownership; after canonical writes begin, route later rollback through the existing reviewed recovery flow, never automatic pre-import overwrite. Extend the CLI to preview, type-confirm destination and start the operation; errors expose the same operation ID and explicit recovery actions.
- [ ] **Step 6: Verify and commit.** Run the three consolidation test modules, exact affected core/Workspace/AgentRuns schema tests, automatic wake/launch/Goal recovery tests and connected binding/exclusion tests. Inject faults at source claim, capture, mapping, before/after each DB commit, verification and cutover. Cover populated destinations, duplicate IDs, branches, Default conflict, unsupported/unknown content, empty sources, second-config claim, repeated cutover and stale-backup rollback. Require zero automatic provider/tool calls for imported completed undelivered results and settled active chains through recovery/resume/manual-turn controls. Commit `feat: consolidate legacy mobile history with recovery fences` after scoped static and catalog checks.

R128 execution clarification (ADR244): exactly two fixed public owner transaction
frames begin after construction and close before retirement/publication. Three
pure verifiers require the exact current frame; private16MiB and public2GiB stay.
Before-effect capacity and restored-generation observer decisions remain open.
Precode SHA/AST/status supplement saved before wrapper/helper/tests.

R129 execution clarification (ADR244): ordinary observer only supports the exact
direct native restoration of this mobile canonical copy, with positive current
activation, receipt-bound native installed chain and same-generation physical/schema
proof. Original source-once/pending/execution restrictions remain; no fake census.
Distinct precode checkpoint saved; actual data/config-write and tamper proof required.

R130 execution clarification (ADR244): exact early terminal pricing, per-allocation
late refusal/recovery, stronger complete rollback preflight; no generalized future
SQL estimator. Fixed public4GiB replaces2GiB after actual8599 frontier, native16
and ordinary16 unchanged. Precode checkpoint saved; qualification remains required.
Execution status: automatic review rejected the R129 observer edit and R130
capacity/early-pricing edit before execution. Both await explicit human
authorization; current production remains2GiB and the original restored-target
refusal remains. Precode decisions above are not implementation-success claims.

Task 8 cold-import correction: preserve the Subscription owner's public getter
name through a lazy exact forwarder, with guarded cold class/schema proof and
existing threshold precedence tests. Actual settings use keeps native admission.
No fixture startup-shutdown bypass or rejected R129/R130 edit is part of this fix.

R131 execution clarification (ADR244): preserve only exact unchanged installed
builtin members under native mobile-copy/current-target provenance. New accounting
is optional on existing native copy/digest loops, one genuine-or-ordinary16 lexical
allowance, with every read/view/slice charged before allocation. Package files are
never restored or modified; native dependency validation and public pending gates
remain mandatory. Changed selector/package/history shapes still refuse.

R133 execution correction (ADR244) supersedes R132's false capture-entry
assumption: Core's dependency method is not called by mobile capture/import.
Restore R115 actual linked-note capture/import/marker qualification; first a
genuine nonconflicting capture control, then concrete guarded App bodies for
controller inspection before runtime. No new generic capture guard, source owner,
source stripping, fabricated marker or automatic file authority. Preserve the
R132 snapshots as chronology; actual support/refusal remains runtime-unqualified.

R131 execution update: explicit human approval now covers the reviewed bundled
artwork code and isolated tests. Resume its native accounting and exact ordinary
builtin composition through normal tool review; original rejection retained,
qualification still open, R129/R130 unaffected and still stopped.

R134 execution clarification (ADR244): acquire actual complete reviewed native capture names before maintenance for existing profiles too, preserving their bindings unchanged. Fixed authenticated mobile-copy package acquisition uses native source namespaces; generic replacement checks remain. Same allowance/custody and all alias/current-review checks apply. Precode SHA/AST/status saved before production edits; full existing-bound/SafetyCopy replacement and stage16 fit are pending. R129/R130 remain stopped.

R134 authorization update: the user explicitly approved the exact native backup-source namespace registration/holding code and isolated tests for already-bound profiles. The earlier automatic-review rejection remains recorded. Implementation proceeds through normal review with unchanged profile records, agent permissions, public2GiB/native16MiB limits and existing physical admission checks. R129/R130 remain awaiting separate authorization.

R135 execution clarification (ADR244, ADR082, ADR126): compose only actual native Console scratch writer/manager leases with unchanged persistent profile/source admission. Authenticate every affected writer callback and fixed worker handoff; independent workers own real leases through retirement, no copied-context grant or persistent temporary registry. Three stock writer acquire sites use the same exact current private association or original ordinary storage fallback. Save precode identities/caller supplement; targeted identity/refusal/cancellation/default controls precede one fresh mounted manual proof. R129/R130 remain stopped; no user-data action.

R135 qualification update: explicit human approval followed the preserved rejection. Actual native bound/default, callback identity, independent worker/cancellation/fork and default controls pass. Fresh mounted45390 completes a new manual turn on its public-produced target, drains and reopens, retaining imported identity-scoped authority/accounting; exactly one scripted provider generation and no tool dispatches. Character-estimate fallback only; no real-provider/hardware/main-entrypoint claim.

R136 execution clarification (ADR126/244): validate the complete fixed native Notes lock representation as excluded device state, with bounded no-follow inventory, exact private empty-lock shape and current custody/accounting. Preserve malformed-content refusal; qualify actual lock release/closed handle and Complete/refusal controls before fresh PRESENT startup. No marker or imported execution authority changes. Precode identities saved before implementation.



R134 held-preview accounting clarification: the fixed authenticated mobile canonical-copy path carries one original ordinary16MiB allowance explicitly through its new source-prefix, selector/mapping, bootstrap/journal and framed-observation representations, and its already budgeted package read/copy/view/slice work. The actual native `_capture_bound_sources` context remains the sole source-read authority; generic capture admission is unchanged. No PreviewScope may be issued during maintenance, no scope fields or generic role are added, and nested fixed callbacks reuse this exact allowance without reset. Original generic verify_sealed/plaintext and encrypted hash passes, Zip/config/native planner internals retain their existing ArchiveLimits and behavior; this counter is not a whole-engine or RSS bound. Factor the existing authenticated candidate prefix and preserve positive current nonpackage profile bindings, complete target/member/selector/store checks, cancellation and physical retirement. Optional private forwarding defaults preserve existing callers; component identity/refusal/cancel/exhaustion/default checks precede another full replacement. R129/R130 remain blocked separately.

R137 execution clarification (ADR059/073/244): current-marker startup reviews only receipt-backed imported canonical note IDs unrepresented by any existing binding. Native evidence adds only new disjoint paused roots with transactional complete-root/binding rechecks; invalid/overlapping evidence surfaces a separate body-free warning while current roots retain normal behavior. Both start gates preserve zero-profile/no-receipt behavior; native SELECT-only Core lifetime and original16 receipt bound remain. No absent migration/schema/activation change. Saved precode identities precede code; finite controls precede another PRESENT App flow. R129/R130 remain independently pending.

R138 execution clarification (ADR244): one native-owned current journal read only in the existing accounted fixed-mobile branch; preserve no-current/None generic/mobile reads and budget defaults. At the adjacent planner boundary remove exactly one duplicate fixed-mobile source/archive verification, dispatching on actual native proof; generic double verification/hash counts and fresh entry/plan/exit predicates remain. No new authority/cache/issuer/cap or fit claim. Saved source/AST/full-status checkpoint precedes implementation; focused native call-count/refusal/default/custody controls precede another held-prefix attempt.

R139 execution clarification (ADR244): reuse one prepaid local <=64KiB buffer only for existing explicit-budget digest/noncollect reads lacking a native scope budget. Retain every verification, per-loop price, actual scope/lease/identity/cancel/EOF/growth/retirement check and default/native-buffer/copy/collect/fallback behavior. No new API/field/helper/authority/cache/cap/reset; ordinary/native16 and public2GiB unchanged. Precode+finite native controls precede another held-preview release; source arithmetic is not a fit claim. R129/R130 stay pending.

R140 execution clarification (ADR244): only the fixed canonical preview entry performs source-only native verification then the immediately adjacent full acquired-archive verification using its returned actual proof. Planner/exit remain full checks; canonical archive count4→3 and verify_sealed6→5, source reads/original matches3 unchanged. The fixed caller changes for explicit and outer-None budgets; all outside generic/default source dispatch stays unchanged. Preserve every predicate, original limits/custody/cancel and separate temporal-sampling caveat; no cache/API/cap/reset/fit claim. Save precode, focused unit RED/GREEN and static report before a fresh sentinel. Bounded finally metadata labels source_context_exit_before_staging; R129/R130 stay pending.

R141 execution clarification (ADR126/244): add only one private byte workspace to the existing native CaptureScope, with same-budget full66,560 prepayment before first eligible allocation and clearing in existing retirement finally before unchanged cleanup. Every file/pass retains fresh hash/admission/identity/cancel/EOF/custody checks. Ordinary adaptive/default/native Preview/Discovery behavior remains; first tiny-file earlier refusal is explicit. No budget/API/helper/evidence cache/role/cap change. Actual two-file multipass, scope isolation, defaults and fault/FD controls plus static report precede any new sentinel; no predicted fit. R129/R130 stay pending.

R142 execution clarification (existing ADR126/244; no new ADR): apply the reviewed fixed SafetyCopy alternative only after original profile containment fails for an included preserved installed builtin. Reprove committed original source/archive/new prepared receipt and exact unchanged members at all three native boundaries; allow current held exact or broader native namespace geometry with pinned identity, never a profile/permission grant. Issue one distinct original16MiB phase allowance only after actual stage retirement; retain original generic limits/accounting and tracked physical custody/quarantine. Save precode/callers/AST/status before code, then finite real source, geometry, default, cancellation, budget and custody tests plus component review before a fresh full integration. No retained98188 continuation; R129/R130 remain pending.

R143 execution clarification (existing ADR126/244): retain R142 failed57145 and original same16MiB/three-check contract. Apply only reviewed local parent grouping with genuine registered custody; every member leaf/stat/frame remains, current parent reopens while original FD stays held, and cancellation precedes new reopen pricing/IO. All original source/archive/overlap predicates remain. Precode/ADR before code; finite native local-mechanics RED/GREEN then one fresh exact component, one combined genuine geometry variant only if exact passes, and root report before any full replacement. No retained fixture continuation or R129/R130 scope.

#### R142 remaining refusal controls after R143 geometry refusal

ADR required: no new ADR; test-only qualification under ADR126/244 and R142/R143. Save source/precode before new tests. Add cheap local guard controls for source/receipt/self-lock, current namespace and native leaf predicates; clearly label mocked upstream prefix as unit mechanisms, never positive committed authority. Use actual temporary handles/session retirement for physical identities and existing native generic prepared fixture for nonpackage refusal. Preserve earlier lower-level mapping/cipher/default evidence rather than repeating producers. Record any unisolated predicates honestly. Prepare only an UNAPPLIED source alternative for duplicate write-path overlap work, with precise observation-count and timing changes; no production edit, cap/reset, path shortening, retained continuation or full integration is released.

#### R144 — distinct lexical write-path validation

ADR required: no new ADR; clarify ADR244 before code. Apply only the reviewed eight-line local set/lookup/growth change in `_mobile_safety_members`, preserving all logical mappings, native units and every distinct-path overlap predicate. Record reduced repeated observation timing explicitly. First finite RED/GREEN controls forwarding actual native overlap (ordering, equal-repeat vs alias, immediate refusal, fresh second call, capacity/cancel/retirement), then scoped static. Only after those pass, one fresh unchanged broader/unprefixed/multi-root three-check component; stop on actual failure. No retained continuation or full replacement before root gate.

R144 full-run oracle correction (ADR required: no; test-only under existing ADR244): preserve 97088 actual replacement/SafetyCopy success and final expected-refusal exception-family failure. Change only the existing postrestore assertion to exact `bootstrap.RecoveryRequired` matching `^mobile_object_changed$`, then one genuinely fresh full integration to execute its skipped final source/package/guard footer. No R129 behavior or production change; native/public limits unchanged. Save exact precode/diff/AST and closed outcome, stop at any new failure.

### Task 9: Canonical mobile API and legacy engine retirement — TASK-36767

**Files**

- Modify: `tldw_chatbook/Web_Server/mobile_service.py`, `mobile_api.py`, `serve.py`, `Chat/console_remote_commands.py`, child command/read wiring in `app_served.py`.
- Modify native create/open integration in `Chat/console_runtime.py` using existing `console_conversation_hydration.py` and canonical conversation services.
- Test: extend `Tests/Web_Server/test_mobile_service.py`, `test_mobile_api.py`, `test_mobile_page.py`, `test_web_auth_integration.py`; create `Tests/Web_Server/test_mobile_shared_console.py`.
- Docs: `Docs/Architecture/web-server.md`, `console.md`, `tldw_chatbook/Web_Server/README.md`.

**Interfaces**

- Consumes `ServedConsoleHosts`, `ConsoleRemoteProjection`, `ConsoleRemoteReceipts` and `require_mobile_cutover` from Tasks 3/5/6/8.
- Keep `MobileChatService` as the existing API-facing adapter, now initialized with the admitted host registry/canonical read services/cutover gate. Remove `data_dir`/provider-gateway engine construction, `HeadlessAppContext`, `_Engine` and the private controller/store/bridge graph. Parent Library reads stay on their existing canonical read services.
- Add native `ConsoleRemoteCommands.create_chat(workspace_id: str, command_id: str) -> ConsoleTarget` and async `open_chat(conversation_id: str, command_id: str) -> ConsoleTarget`. Host-level create/open tickets use `ConsoleHostIdentity`; per-session actions require `ConsoleTarget`. Reserve the canonical saved identity before hydration; all concurrent opens return the same live incarnation. Register the persisted identity when an unsaved session commits.
- Retain `/api/v1/mobile/stream` with its authenticated WebSocket and extend the v1 route family with `GET /console/snapshot`, `POST /commands/tickets`, `POST /commands`, `GET /commands/{command_id}` beneath `/api/v1/mobile`. Existing Library routes remain. Reject obsolete mutation envelopes rather than invoking the retired engine.
- Ticket request JSON is `{protocol_version: 1, action, target, payload}`; response is `{ticket, command_id}`. Command JSON is `{protocol_version: 1, ticket, payload}`; require version 1 and return the native receipt and HTTP 202 for pending/accepted/queued. Query uses canonical profile receipt lookup, not only a live registry. Snapshot query scopes the host/session/cursor; parse with strict Pydantic models and existing body bounds. Keep the HTTP v1 contract distinct from private control version 3; stale clients receive a reload-required refusal before effects.

- [ ] **Step 1: Add a constructor-retirement trap.** In the new shared-service test, instantiate the adapter with a recording host registry/canonical read service and a verified cutover fixture. Monkeypatch the old engine factories to raise while they still exist, then list/open/create/send/read through the API. After deleting those symbols, retain a fixture spy on the DB constructors to prove no `mobile/` DB open occurs. The recording registry exposes the exact Task 3 `get_or_start/aclose` interface and counts starts; its command/read responses use Task 5/6 DTOs.

```python
async def test_two_clients_open_one_saved_console_owner(shared_mobile_case):
    case = shared_mobile_case
    first, second = await asyncio.gather(
        case.open_from_client('phone-a', case.conversation_id),
        case.open_from_client('phone-b', case.conversation_id),
    )
    assert first['session_id'] == second['session_id']
    assert first['incarnation_id'] == second['incarnation_id']
    assert case.host_start_count == 1
    assert case.mobile_db_opens == []
```

Define test helper `open_from_client(client_id: str, conversation_id: str) -> dict` to issue an explicit `open` ticket through that authenticated client, submit it once, query its outcome, refresh the host snapshot and return the row matching the receipt's native session ID. Do not add a hidden service path that creates fresh tickets during retry. `shared_mobile_case` uses isolated canonical SQLite, Task 8's verified cutover and the recording host factory; include a real retained-child variant for final qualification. Run `.venv/bin/python -m pytest Tests/Web_Server/test_mobile_shared_console.py -q` RED.
- [ ] **Step 2: Replace the engine graph with owner routing.** Derive storage identity from admitted canonical paths and wait for child ownership admission. Phone-only first use starts the same retained App at default dimensions without a terminal subscriber. Saved-history GETs do not hydrate, reconcile or start an agent; explicit open/create uses the native runtime/hydration services and ownership reservation. Do not restore saved history from legacy stores after completed cutover.
- [ ] **Step 3: Gate new writes on verified consolidation.** Run `require_mobile_cutover` before enabling new canonical mobile mutation routes. A verified empty source is valid; an ambiguous/incomplete/conflicting migration yields a recoverable status with the operation/review path and no new engine. Never silently run migration from an HTTP read or start a competing native process. Delete mobile-only model/hook/project-instruction exceptions; use the same canonical configuration capture and permission owners as terminal Console.
- [ ] **Step 4: Preserve authentication and bounded streams.** Reuse middleware session/CSRF/same-origin checks, WebSocket subprotocol proof, signed-in device management, expiry and TLS/proxy policy. Scope every command and query to the admitted host/profile. Revocation unregisters that client's answerability/subscriptions/terminal scope only. Subscriber overflow returns reset/current snapshot; it cannot propagate socket pressure into the private control channel. Stream reconnection sends an outcome query and snapshot request, never a queued mutation.

```python
# Existing authenticated route, after strict JSON parsing:
receipt = await service.submit_command(body.ticket, body.payload)
status = 202 if receipt.state in {'pending', 'accepted', 'queued'} else 200
return web.json_response(receipt.model_dump(mode='json'), status=status)
```

Add async adapter `submit_command(ticket: str, payload: dict) -> ConsoleCommandReceipt` and `issue_command_ticket(action: str, target: ConsoleHostIdentity | ConsoleTarget, payload: dict) -> dict`, forwarding to the owning child and serializing `ConsoleCommandTicket` as `{ticket: value.token, command_id: value.command_id}`. A timed-out response is unknown/queryable rather than successful. Owner/epoch mismatch, expiry, unknown target and unavailable execution use stable existing error status conventions, preserving distinct expired-auth versus network behavior.
- [ ] **Step 5: Update callers/fixtures, verify and commit.** Replace tests that asserted mobile-only engine behavior with shared-owner tests while preserving their agent-tool/approval evidence through the real runtime. Update all `MobileChatService(data_dir=...)` callers/fixtures found by `rg`, API fakes and page wire contracts. Run the new shared-service file and the four named existing API/service/page/auth modules; include Library notes/media/conversations regression node IDs in those modules. Commit `refactor: route mobile Console through canonical owners` after scoped checks. This is the public cutover gate; no earlier task claims engine retirement.

### Task 10: Chats-first phone interaction — TASK-36768

**Files**

- Modify: `tldw_chatbook/Web_Server/static/mobile/{index.html,app.js,app.css}`, `static/served_shell.html`, `static/served_shell.css` only for the existing phone discovery/terminal link, and `Web_Server/README.md` for semantic browser equivalents.
- Test: extend `Tests/Web_Server/test_mobile_page.py`; create `Tests/Web_Server/test_mobile_console_ui.py` with actual Chromium execution of the served mobile assets.
- Read first: `backlog/docs/design-language.md`, `component-patterns.md`, especially action, status, navigation and message patterns. Keep the existing manifest/icons and no-service-worker posture.

**Interfaces**

- Consumes Task 9's snapshot/ticket/command/receipt routes and authenticated stream. `console.changed` invalidates the affected view; refresh a bounded projection rather than rendering an event as execution authority.
- Keep page state inside the existing JS closure: `selectedTarget`, `drafts: Map`, `outstandingCommands: Map`, `connectionState` and last-confirmed update. No localStorage and no offline work queue.
- Define local JS `draftKey(target)`, `captureSubmission(target, draft, expectedRevisions)`, `clearAcceptedDraft(submission)` and `applySnapshot(snapshot)`. Existing `showView` handles Chats/chat/agent panels and local Back; it never asks the terminal to switch sessions.
- DOM contracts for browser checks: `#view-chats`, `#chat-list`, `#view-chat`, `#chat-transcript`, `#agent-list`, `#target-label`, `#chat-draft`, `#send-command`, `#stop-command`, `#connection-state`. Use real labels, semantic buttons and `aria-live` for connection/decision status. A selected worker always has visible chat and worker labels.

- [ ] **Step 1: Add the late-acknowledgement regression against real assets.** Create an aiohttp browser-test fixture by reusing the existing page server/auth scaffolding in `test_mobile_page.py` and injecting an API recording service with a delayed command receipt. Expose fixture `mobile_browser_case` with `page`, async `open_chat(session_id)`, async `release_receipt(state)` and recorded `commands`. Its constructor loads the actual `/m` HTML/JS/CSS in an authenticated Playwright Chromium context at a phone viewport; only API responses are controlled here. This is UI behavior evidence, separate from Task 11's actual child.

```python
async def test_late_receipt_cannot_clear_a_newer_phone_draft(mobile_browser_case):
    case = mobile_browser_case
    await case.open_chat('chat-a')
    await case.page.locator('#chat-draft').fill('first correction')
    await case.page.locator('#send-command').click()
    await case.page.locator('#chat-draft').fill('newer unsent correction')
    await case.release_receipt('accepted')
    await expect(case.page.locator('#chat-draft')).to_have_value(
        'newer unsent correction'
    )
    assert case.commands[0]['payload']['text'] == 'first correction'
```

Import `expect` from `playwright.async_api`. The fixture signals actual recorded submission before returning from the click path; do not release a receipt after an arbitrary sleep. Add navigation between two chats and a worker, connection-loss disabled state, unsupported pending decision, escaped tool/provider HTML and no-duplicate reconnect cases. Run the new UI file RED; an absent browser is reported as an unmet check, not PASS.
- [ ] **Step 2: Keep each submitted input immutable and clear by exact revision.** Use chat/incarnation for main drafts and add handle/run for worker drafts. Retain unsent text while disconnected; when an owner/incarnation changes, keep the old draft visibly associated with its unavailable original target until the user explicitly chooses a current target. Do not silently retarget text.

```javascript
function draftKey(target) {
  return JSON.stringify([
    target.server_epoch, target.console_id, target.session_id,
    target.incarnation_id, target.handle_id || null,
    target.handle_id ? target.run_id : null
  ]);
}

function captureSubmission(target, draft, expectedRevisions) {
  return Object.freeze({
    key: draftKey(target), revision: draft.revision, text: draft.text,
    target: Object.freeze({...target}),
    expected_revisions: Object.freeze({...expectedRevisions})
  });
}

function clearAcceptedDraft(submission) {
  const current = state.drafts.get(submission.key);
  if (current && current.revision === submission.revision &&
      current.text === submission.text) {
    state.drafts.set(submission.key, {text: '', revision: current.revision + 1});
  }
}
```

Issue a fresh ticket only for an explicit user submit, retain its command ID before the effect request, and suppress duplicate submits while that same submission is pending. A Preparing receipt does not clear text; clear only the exact accepted/queued input after native acceptance. A lost response queries its ID; reconnect never submits it again. A fresh page also reads bounded unresolved receipts from the snapshot. Apply explicit typed target/run/round data to each action and replace controls only after the new target is confirmed.
- [ ] **Step 3: Render shared chat/agent/decision views and honest outcomes.** Default to Chats; merge saved/live identity rows from the API, show workspace/activity plus text statuses/attention markers, and distinguish unavailable from finished. Reuse transcript disclosure for tool activity and a compact Agents roster/detail. Main controls are Send/Steer/Redirect/Stop; workers expose guidance/Stop. Show queued guidance versus consumed state and requested Stop versus physical completion. Supported pending decisions use native choices; unsupported decisions stay visible with the existing Open terminal link. Use `textContent` or the admitted renderer for bodies, never raw `innerHTML` from tool/provider data.
- [ ] **Step 4: Compose phone layout and accessibility from existing semantics.** Keep the existing single-column safe-area layout and semantic CSS variables; document status/focus/disabled/error equivalents in the existing Web Server README. Preserve separate rest/hover/focus/disabled states, visible Back/close controls, reachable touch targets and keyboard action names. The transcript owns scrolling; the soft keyboard/visual viewport must leave the composer and its action reachable. Contain long output in disclosure/scroll regions and preserve wrapping. Do not redesign unrelated Library pages or add a framework.
- [ ] **Step 5: Verify and commit.** Run `.venv/bin/python -m pytest Tests/Web_Server/test_mobile_console_ui.py Tests/Web_Server/test_mobile_page.py Tests/Web_Server/test_mobile_api.py -q`, `node --check tldw_chatbook/Web_Server/static/mobile/app.js`, and scoped stylesheet/HTML checks exercised by those browser tests. Inspect 320px and 390px portrait, landscape, keyboard navigation and reduced-height soft-keyboard emulation with actual rendered controls. Record screenshots only as presentation evidence; assert command targets and draft/receipt behavior independently. Commit `feat: supervise shared Console chats from the phone` after scoped checks.

### Task 11: Real served/browser qualification and release documentation — TASK-36769

**Files**

- Create: `Tests/Web_Server/test_mobile_served_console_browser.py`, `Docs/superpowers/qa/2026-10-25-shared-console-mobile-supervision/README.md`.
- Extend: `Tests/Canvas/browser/canvas_live_harness.py`, `canvas_live_chatbook_child.py`, `test_canvas_served_flow.py` for reusable actual-App tooling and revised lifecycle assumptions. Keep existing Canvas assertions intact.
- Docs: `Docs/Architecture/{console,agent-runtime,web-server,backup-recovery,database-layer}.md`, `Docs/Backup-and-Recovery.md`, `tldw_chatbook/Web_Server/README.md`, `Docs/User_Guide/console.md`, ADR-244's implementation/evidence status and the spec's review/delivery state.
- Keep unrelated user changes in architecture/lessons files outside the task's commit. Record a lesson only if execution produces a concrete reusable trap.

**Interfaces and fixture ownership**

- Reuse existing async `start_live_served_stack(tmp_path, monkeypatch, *, access_token, child_module, trusted_proxy=False) -> LiveServedStack`. Its `aclose` reaps exact owned children and cleans temporary resources. Extend it for retained hosts; do not replace it with a mocked terminal server for this qualification.
- Promote the existing login/readiness/terminal-input helpers from `test_canvas_served_flow.py` into `canvas_live_harness.py` when the second test module consumes them; update both callers. These remain test helpers, not product APIs.
- Add an actual-Console script to `canvas_live_chatbook_child.py` using the existing provider-double seam: per-chat exact prompt/result markers, a barrier held after native acceptance, one real admitted local tool, a supported approval/question and two bounded subagents. Explicit entry/consumption/release signals establish the races; no real external provider is called.
- New fixture `live_mobile_console` owns the TLS stack and two authenticated browser contexts plus helper methods `start_gated_chat`, `phone_send`, `release_agent`, `snapshot`, `close_terminal` and `reattach_terminal`. It returns observed canonical chat/run IDs, actual process PID, exact gateway call count and persisted transcript markers. Helpers must act through production routes; assertions may inspect isolated DB evidence after controlled close.

- [ ] **Step 1: Prove retention with an actual child before adding more journeys.** Use the existing isolated profile and controlled gateway, not a subprocess that merely echoes WebSocket packets. Capture the PID/turn/run and provider entry barrier before closing the terminal.

```python
async def test_phone_reconnect_keeps_the_same_console_run(live_mobile_console):
    case = live_mobile_console
    before = await case.start_gated_chat('EXACT_SHARED_PROMPT')
    await case.close_terminal()
    during = await case.snapshot()
    assert during['pid'] == before['pid']
    assert during['run_id'] == before['run_id']
    await case.phone_send('EXACT_PHONE_FOLLOWUP')
    await case.release_agent()
    after = await case.reattach_terminal()
    assert after['pid'] == before['pid']
    assert after['session_id'] == before['session_id']
    assert after['prompt_occurrences']['EXACT_SHARED_PROMPT'] == 1
    assert after['prompt_occurrences']['EXACT_PHONE_FOLLOWUP'] == 1
```

The phone follow-up above is an explicit Send queued through native policy, not guidance assumed consumed. Read exact persisted prompt/reply rows and native acceptance markers before declaring this test qualified. Run `.venv/bin/python -m pytest Tests/Web_Server/test_mobile_served_console_browser.py -q` and require actual browser execution with no skipped core journey.
- [ ] **Step 2: Exercise terminal interaction and independent phone targets.** In the reattached terminal, click a real Console control, paste a multiline draft and resize the viewport; prove the input thread was not restarted. Run two chats and multiple workers; phone navigation/steer/redirect/Stop and subagent guidance must leave desktop selection/drafts and unrelated runs untouched. Resolve background and viewless decisions from one phone while another disconnects. Test an unsupported confirmation remains pending. Include browser-session takeover/revocation and stale old attachment input/cleanup.
- [ ] **Step 3: Force transport failures at witnessed boundaries.** Drop the phone connection before admission, after native acceptance and after effect before response; query the original ticket/receipt without replay. Fill the private projection buffer before first phone use, with the mobile surface disabled and with a slow subscriber. Require the same child/run/fence and current snapshot convergence. Separately kill the actual parent process and confirm a surviving child retains the fence through bounded shutdown; a replacement cannot execute/reconcile during that interval. Verify genuine control loss, actual child crash and clean server shutdown separately. Never use a browser close or `tmux kill-server` as proof a process died.
- [ ] **Step 4: Re-run connected migration/security checks on final code.** Run the new browser file, targeted control/host/input/receipt/API files and consolidation/automatic-recovery schema cases from Tasks 1-9. Use direct TLS and the existing trusted-proxy control; test unauthenticated, wrong-origin/CSRF, revoked and stale-epoch commands. Preserve no-provider/no-tool evidence before explicit imported-authority review. Verify Library read routes still use canonical stores. Record exact commit, commands, pass/skip/failure counts, child exit evidence and retained temporary artifacts for failures. Do not broaden to all repository tests without opt-in.
- [ ] **Step 5: Document actual behavior and perform the separate phone/VPN check.** Update the named living/user docs after verified implementation, including migration preview/apply/recovery, one-terminal attachment policy, receipt uncertainty, the no-localStorage draft limit and the first-release exclusive native-or-served boundary. Publish a packet with separate automated, real served-browser, and actual phone/VPN sections. For the latter, connect the physical phone through its VPN, open authenticated `/m`, inspect the already-running chat, submit one exact correction, drop/rejoin VPN and reconnect to the same run, then check worker targeting and keyboard reachability. Record phone/browser/network details and observed identities; if hardware/network access is unavailable, mark that section unverified and do not call emulation physical qualification.
- [ ] **Step 6: Review completion evidence and commit.** Self-review the connected implementation, check every task AC against current evidence, and resolve concrete failures without changing unrelated policies. Require executed migrations and real child identity evidence, not a version-label assertion or screenshot alone. Set each implementation task Done only when its own AC/docs/static/targeted checks are satisfied; unavailable mandatory checks stay open. Commit `test: qualify shared served Console mobile supervision` with the actual evidence/docs/tests; do not claim a push, merge or physical phone run unless it occurred.

## Targeted command index

These are the minimum connected GREEN runs for the named deliverables, after their RED tests and implementation. Add exact connected caller node IDs when that task changes another caller; do not substitute a full sweep. All paths below either exist at the baseline or are explicitly created by the preceding task section.

| Task | Run from the executing checkout root |
| --- | --- |
| 1 | `.venv/bin/python -m pytest Tests/Chat/test_console_execution_owner.py Tests/DB/test_agent_runs_db.py Tests/Agents/test_agent_runs_db_connection_reuse.py Tests/Chat/test_console_runtime_lifetime.py Tests/Chat/test_console_runtime_shutdown.py Tests/Chat/test_console_worktree_recovery_lifetime.py Tests/DB/test_private_sqlite.py Tests/Backup_Recovery/test_agent_runs_recovery_schema.py -q` |
| 2 | `.venv/bin/python -m pytest Tests/Web_Server/test_served_console_events.py Tests/Canvas/test_control_protocol.py Tests/Web_Server/test_served_relay_publisher.py Tests/Web_Server/test_web_auth_integration.py Tests/Web_Server/test_mobile_service.py -q` |
| 3 | `.venv/bin/python -m pytest Tests/Web_Server/test_served_console_hosts.py Tests/Web_Server/test_console_web_driver.py Tests/Web_Server/test_served_shell_browser.py Tests/Chat/test_console_runtime_shutdown.py Tests/Canvas/browser/test_canvas_served_flow.py -q` |
| 4 | `.venv/bin/python -m pytest Tests/Chat/test_console_received_intent_inputs.py Tests/Chat/test_console_received_intent_custody.py Tests/Chat/test_console_received_intent_transitions.py Tests/Chat/test_console_received_intent_saved_acceptance.py Tests/Chat/test_console_remote_commands.py Tests/Chat/test_console_runtime_shutdown.py -q` |
| 5 | `.venv/bin/python -m pytest Tests/Chat/test_console_remote_projection.py Tests/Chat/test_console_interrupt_rounds.py Tests/Web_Server/test_served_relay_publisher.py Tests/Chat/test_console_runtime_lifetime.py Tests/Canvas/test_control_protocol.py -q` |
| 6 | `.venv/bin/python -m pytest Tests/Chat/test_console_remote_receipts.py Tests/DB/test_console_command_receipts.py Tests/Chat/test_console_received_intent_saved_acceptance.py Tests/Chat/test_console_dispatch_recovery.py Tests/Canvas/test_control_protocol.py Tests/Backup_Recovery/test_agent_runs_recovery_schema.py -q` |
| 7 | `.venv/bin/python -m pytest Tests/Backup_Recovery/test_mobile_consolidation_preview.py Tests/Backup_Recovery/test_inventory.py Tests/Backup_Recovery/test_capture.py Tests/Backup_Recovery/test_capture_service.py Tests/Backup_Recovery/test_agent_runs_recovery_schema.py -q` |
| 8 | `.venv/bin/python -m pytest Tests/Backup_Recovery/test_mobile_consolidation_apply.py Tests/Backup_Recovery/test_mobile_consolidation_recovery.py Tests/Backup_Recovery/test_mobile_consolidation_preview.py Tests/Backup_Recovery/test_agent_runs_recovery_schema.py Tests/Chat/test_automatic_wake_recovery.py Tests/Chat/test_automatic_launch_recovery.py Tests/Chat/test_console_goal_recovery.py Tests/DB/test_automatic_work_migration.py -q` |
| 9 | `.venv/bin/python -m pytest Tests/Web_Server/test_mobile_shared_console.py Tests/Web_Server/test_mobile_service.py Tests/Web_Server/test_mobile_api.py Tests/Web_Server/test_mobile_page.py Tests/Web_Server/test_web_auth_integration.py -q` |
| 10 | `.venv/bin/python -m pytest Tests/Web_Server/test_mobile_console_ui.py Tests/Web_Server/test_mobile_page.py Tests/Web_Server/test_mobile_api.py -q` |
| 11 | `.venv/bin/python -m pytest Tests/Web_Server/test_mobile_served_console_browser.py Tests/Canvas/browser/test_canvas_served_flow.py Tests/Web_Server/test_mobile_shared_console.py Tests/Backup_Recovery/test_mobile_consolidation_recovery.py -q` plus the exact final-code caller/schema/automatic-authority cases recorded in Task 11 Step 4. |

## Spec coverage map

| Approved contract | Implementation tasks | Evidence gate |
| --- | --- | --- |
| Spec 1/2: same Console/runtime, served first | 1-6, 9 | Actual child and canonical DB identity; no mobile engine constructors |
| Spec 3: physical execution ownership, canonical identities, native limit | 1, 3, 8, 9 | Real contender/survivor/crash tests and read-only no-sweep checks |
| Spec 4: retained lifetime, terminal modes, one current attachment | 2, 3, 11 | Same PID/run after disconnect plus actual click/paste/resize |
| Spec 5: authenticated reads, exact commands, input ownership | 2, 4-6, 9 | Native custody and current-target/permission/CSRF race controls |
| Spec 5: five-minute tickets, bounded receipts, unknown outcomes | 6, 9-11 | Expiry/tampering/capacity/lost-ack/query tests with no repeated effects |
| Spec 6: Chats, agents, decisions, independent drafts and accessible phone UI | 5, 9-11 | Actual assets/browser behavior and separate phone/VPN receipt |
| Spec 7: physical source review, capture, mapping, quarantine, recoverable cutover | 7-9, 11 | Per-owner batch/crash/reference closure and zero imported automatic calls |
| Spec 8/9: domain boundaries and failure behavior | All | No second engine/state machine; failure-specific ownership evidence |
| Spec 10/11: qualification, governing decisions, truthful delivery state | All, 11 | Targeted evidence, exact schema catalogs, updated ADR/living docs |

All seven review corrections have explicit owners: independent phone text (4), child-held physical fence (1/3), completed-result automatic quarantine (8), shared-source attribution (7/8), receipt expiry/dedupe (6), terminal modes (3/11), and projection overflow independent of host loss (2/3/11).

## Handoff and stop conditions

The plan/task records are the deliverable of planning. They do not mean the mobile storage split or retained-host behavior is implemented. At execution start read the spec and this plan together, then begin Task 1; Task 2 or 7 can be worked independently only in their own checkout with their declared interfaces respected.

Stop dependent work when an admitted resource cannot be fenced, capture scope is incomplete, schema/authority mapping is unknown, or a mandatory qualification fails. Retain evidence and resolve the concrete issue through the native owner; never relax authentication, import quarantine or recovery fencing to pass a demonstration. Routine implementation details within these contracts do not require repeated user approval.

Implementation can proceed using reviewed subagents per task or sequential execution in this session. Neither option authorizes real user-data migration, full-suite runs, publication or merging beyond the user's existing instructions.

## Planning validation

Self-review mapped all specification sections and seven review corrections to implementation tasks, reconciled ticket/query identity and native input-revision contracts, and checked existing source/test paths against this checkout. The 11 task records reference only already-created earlier dependencies. New implementation/test files are explicitly declared above; none has been created by planning.

Document checks validate task frontmatter/owned sections, local links, task order/dependencies, absence of unfinished markers, Python example syntax, JavaScript syntax and the two SQL examples against isolated in-memory SQLite. These are documentation checks: no product test suite, real agent/provider call, user database, migration or phone/VPN journey has been run.

The repository-wide task-ID guard reports pre-existing duplicates `34991` and `35028` from unrelated tracked/untracked files present before this work. None of the 12 new task IDs collides. Those files are preserved; this plan does not claim the entire checkout's Backlog guard is green.


### R129/R130 explicit human approval and current-source rebase (2026-10-31)

The human explicitly approved code and isolated tests for both existing decisions; the prior missing-authorization block is resolved (SDD `task-8-r129-r130-explicit-human-approval.md`). The old rejected drafts and failing epochs remain historical evidence. Implementation is pending root drift review, not yet applied or qualified. No new ADR/ruling is required.

R129 remains an ordinary16-only direct restore of this mobile operation’s canonical copy: native committed replacement/receipt-bound plan/manifest/activation and current private-copy schema plus main-generation/ancestor continuity; original claim, retained encrypted captures/history, current selectors and pending fences remain checked. Ordinary imported-state census stays unchanged outside that authenticated transition. Ephemeral WAL/content timestamps are not permanent generation identity; every native copy and physical safety check still runs. No fabricated imported receipts, reimport, owner grant or activation bypass.

R130 changes only the public cumulative envelope to4GiB and constructs/discards fixed maximum-width terminal representations before no-data owner creation or populated history/owner effects. Every actual late reservation remains; no escrow, refund, future-SQL estimator or universal-fit/RSS claim. Existing JSON sizing covers non-BMP escaping for every new serialization/hash. Private no-public calls still use their original16 and must be checked proportionally. Ordinary/native16, atomic pair coupling, independent archive/SQL limits and full rollback preflight remain unchanged. Current production stays2GiB until the reviewed delta is applied.

Finite sequence: new local identity/manifest/Unicode/dimension controls plus existing five pair/default controls and affected capture-names debit; early refusal before first owner for populated/no-data; one proportional original private/preterminal native16 control; then one fresh direct-restore positive with native data/theme-save and tamper/refusal checks, and one fresh successor. Existing late-budget real-home two-process evidence is retained; any rerun is a new isolated fixture with unchanged reviewed harness, only if needed to qualify changed prepricing. No old-fixture continuation or accepted App/Notes/CLI repeats.


R129/R130 component checkpoint: approved delta applied; local15P, genuine populated/no-data early-refusal2P and original-private16 preterminal3-receipt1P pass. Retain the earlier1F empty-tuple oracle epoch separately. `task-8-r129-r130-component-report.md` records exact scope/static/evidence; root integration gate precedes actual restored-generation/write/theme/tamper and4GiB successor runs. No wholeTask8 completion or new authority follows from this checkpoint.


### R130 cold-resume implementation correction — original invocation reuse (2026-11-01)

Existing ADR244/R130; no new ADR or accounting policy. The first changed-code real-home Workspace cold run failed its original16 allowance before stale-owner settlement. Its new early-pricing helper repeated native journal/prepared/pending work. The reviewed v2 correction uses pricing-only local references from the actual PRE-R130 entry preopen for no-data or post-reader preopen for populated, after original validation and retirement. Default callers still return None and retain their original call shape/checks. The new two-reference collector and shallow sizing views are prepaid and discarded before effects; no readiness, cache, native scope or authority is issued. All future original-record Ref encoding costs, fixed maximum terminal representations, actual final checks/hashes/charges and independent native16/public4 bounds remain. No new unbudgeted invocation replaces the deleted duplicate work.

SDD `task-8-r130-cold-pricing-v2-reviewed-release.md` governs the exact two-file correction, affected local/default/codec controls, genuine populated/no-data refusal and private16 preterminal proof, then one fresh observed Workspace73→0. Actual fit remains unqualified until that run; the prior failed fixture and rejected v1 accounting rationale remain preserved. If green, the already released B–E integrations continue serially. No full Task8 completion follows from these components.

### R130 Event tree accounting correction (2026-10-31)

The cold3 native16 resume exhausted at final JSON scanning (requested256, remaining58). Before another runtime, the reviewed helper-only schedule replaces two coarse eight-text-copy Event-stage reservations with separate complete-tree reservations: the native terminal model dump, conservative Event evidence validation, and each of the two Event dumps. Constructor/model/field containers and a shallow six-field sizing view are prepaid separately. The first eight-copy `_MobileCommitted`/Path validation, existing pre-R130 input reuse, future Ref coverage, non-BMP serialization, record bound, JSON roundtrip, all final charges and native16/public4 limits remain unchanged. No authority or readiness value escapes. Pure retained-shape arithmetic is not a fit claim. Source contract and finite verification: `.superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-8-r130-event-tree-reviewed-release.md`; prior failed epochs remain preserved.

### R145 - production recovery qualification scope (2026-10-31)

Both production captured-consolidate entry points use a genuine public pair. Keep
the current early prototype and every late check on all callers; no private
exception or accounting change is made. The larger private Workspace cold4 fixture
now legitimately exhausts original16 after early pricing, and remains a recorded
failure rather than a parity or complete-path claim. R130 promises bounded
refusal, not universal fit. Replace only the added private-success gate with the
same committed Workspace/unmatched-owner graph through actual public start and
fresh-process public finish, each genuine4GiB/16MiB pair. The smaller private
three-receipt component retains its actual passing scope. Genuine restored-gate
B and C-E may proceed independently; complete public crash/receipt/no-replay and
all original Task8 qualification still precede Task9. Source contract: SDD
`task-8-r145-public-recovery-gate.md`.


R146 execution (ADR required: yes; existing ADR244 amendment): replace only ordinary cutover-call16 with fixed64MiB-minus512 issuance for BOTH branches; generic/private16 and public4/native16 remain. Apply exact two-source issuer/import-call delta after precode. Add finite noargument/identity/default/atomic exhaustion and before-first-read controls; preserve all native validators and default call shapes. B's finally records actual issued counter remaining on success and failure. Run one fresh B, conditional C/D/E only after green; first material failure stops. R145 public Workspace remains separately review-gated. Historical failures/limits are retained.

R146 B3 fixture correction: B2 passed both restored/data-edit gates (13,610,208 remaining each), then config import refused actual data/shared intermediate directories at0755 (config_participants369), not a stale alias or failed profile association. For one fresh B fixture only, use saved/restored umask077 during genuine construction and assert owner/private mode before public start. No old fixture repair or native policy change; C–E remain conditional on B green.

R146 B4 test-only cohort correction: B3 native theme save/binding/inode assertions passed, but later selecting the target changed the original effective-first selector cohort. Keep target selection stable after fixture close and obtain a genuinely fresh public target-target preview before start; no prepared witness or native guard changes. Preserve B3 failure and all scoped positives; one fresh affected B only, C–E conditional.


### Original Task8 FIX1 — T8FULL-I1 / M1

ADR required: no new ADR. ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md. Reason: correct accepted R146 same-coordinator propagation, without changing limits, roles or validation. Before code, review identified that ordinary original-generation owner verification entered an unbudgeted preview. Change only that call to forward the existing budget. Extend the genuine public-produced B target with real native scope/copy identity and same-counter source/copy scarcity checks; preserve all restored controls. Record RED/GREEN, physical native cleanup/source identity, focused static and one scoped local FIX1 commit. Carry M1's zero introduced/1,841 inherited Ruff, unassigned caller-scan warning origins and historical/runtime/hardware limits explicitly. Full task and Task9 stay held for fresh scoped re-review.


### Task9 R147 explicit first-host bootstrap

An authenticated, CSRF-protected `POST /api/v1/mobile/console/host` accepts only
`{protocol_version: 1}`. After verified cutover it reuses the canonical registry,
awaits connection and positively checks native run-store execution admission,
then returns the exact `ConsoleHostIdentity`. It creates no native chat, provider
request or ticket. An explicit Create/Open may bootstrap then issue one exact-host
ticket; races refuse. Reads and reconnects never bootstrap, reticket or replay.


### Task9 connected native project callback correction

The first native remote Send reached durable persistence, then the unchanged
project-provider composition invoked `project_authority_guard` on the controller
loop. Its local `on_owning_loop` unconditionally used Textual `call_from_thread`,
which refuses on that loop. Preserve the four original activation/authority/
dispatch callers and every project/source/permission check: invoke directly only
on a non-null matching controller owner loop; otherwise retain worker marshalling
and the existing no-App fallback. Catch only absence of a running loop, never a
callback exception. Original first-Send exception evidence is in Task9 agent10;
qualify actual native Send plus owner-loop/worker/error/default routing controls.
This is a routine shared bug fix under ADR244, with no new authority or API.


### Task9 cold viewless source metadata dependency

The un-warmed Library-mounted native first Send refused before execution because
`_resident_skill_wiring()` had no resident `Widgets.compact_model_bar` source
metadata. The existing cold stock catalog proof deliberately requires that
module's original source capsule; a mounted Console normally imports it. Explicit
remote Send loads this same dependency through `run_preparation_read`, observed by
controller/runtime physical teardown, before unchanged received-configuration
capture. Its import reads/compiles Textual parent source metadata, so it runs off
the owner loop. Exact native target and all five revisions are rechecked after
that await. No widget is mounted, trust/plugin getter invoked or cold/custom/
memory/source/permission qualification relaxed. Preserve cold refusal evidence
and qualify the actual un-warmed first Send plus no-retention negative control.


### Task9 fix1 — shared saved-session readiness and cleanup custody

ADR required: yes; this clarifies existing ADR244, with no new execution or file
authority. F02–F07 use ONE store-owned saved-conversation operation, reserved
before I/O/native creation's first durable commit. Every observer shields the
owned task/future. A provisional row is not a usable published owner: readiness
includes progress, native Library policy and membership reconciliation. Actual
failure/owner shutdown retains physical retirement and rolls back exact
provisional state once; observer loss alone cannot cancel another opener.

Presentation callers receive explicit exact cleanup custody. Terminal workspace,
Buddy and the outer Character activation retain it until their final successful
commit. Another successful opener adopts/publishes the ready owner and revokes
that cleanup capability. Failed solitary unpublished restores still roll back.
No blanket rollback suppression, hidden duplicate session, or extra registry.
Cold saved publication defaults to the durable ID to retain Goal's existing
sealed attempt/session contract; never renumber a live first-persisted owner,
and refuse conflicting explicit IDs. Remote scope comes from the durable row,
including explicit global, independently of terminal selection.

Native new_chat/fork reserve their preallocated durable ID in this same operation
before commit/copy. Their intended settings, assistant identity, opening draft
and existing new-session project controls must be durable before a phone can
observe publication; direct native completion joins the same readiness. Do not
overwrite an accepted phone turn or widen project/file/permission authority.
Source-close/orphan cleanup remains exact. Native shutdown drains owned physical
work and detached exceptions are consumed. Focused regressions cover mixed
native/phone readiness, presentation/cancellation, Goal iteration in both orders,
native create/fork publication races and close/reopen. F01 separately bounds the
saved discovery DTO (including scope/activity); F08 refreshes only current
nonstarting registry discovery and retains stale target refusal.

Fix1 implementation qualification: the common operation also joins the existing
first-identity reservations for manual Save and durable first Send; these already
live owners stay controllable while saved-open observers await publication. Native
new_chat controls commit through the existing project-context writer in the native
atomic creation transaction; fork's first commit and copy share the reservation.
Shutdown carries saved-read cancellation through existing physical native cleanup.
Host discovery refresh does not move a running event subscription: register and
disconnect use the exact child captured with its parent buffer.
