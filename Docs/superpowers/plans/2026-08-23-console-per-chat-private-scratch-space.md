# Console Per-Chat Private Scratch Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every live Console chat a private temporary filesystem, keep user-folder access exclusive to explicit Workspace bindings, and verify the complete flow with DeepSeek.

**Architecture:** `ConsoleRuntime` owns a thread-safe `ConsoleScratchSpaceManager`; each turn captures an immutable, generation-fenced scratch snapshot and every Chatbook-managed filesystem consumer receives that authority explicitly. Built-in tools use a run-scoped sandbox `ContextVar`, local `fs_*`/Git tools use scratch unless ADR-069 selected a project binding, and retained skill output plus fallback run logs lease the same scratch root. Session close tombstones authority synchronously and a single daemon cleanup worker removes directories only after active leases drain.

**Tech Stack:** Python 3.11+, Textual 8.x, `pathlib`, `tempfile`, `threading`, pytest, real temporary directories, existing DeepSeek provider gateway.

**Spec:** `Docs/superpowers/specs/2026-08-23-console-per-chat-private-scratch-space-design.md`

## Global Constraints

- Ordinary Console Chats never require a folder and receive only private scratch authority.
- Named Workspaces may add only active explicit folder bindings; a folder remains optional.
- A selected ADR-069 project-instruction binding remains the sole local `fs_*`/Git root and keeps locator-identity and read-only enforcement.
- Existing Ask/Allow/Off permissions, kill switches, sensitive-path checks, and temporary-conversation write restrictions remain unchanged.
- Scratch paths contain no session, conversation, title, provider, or user identifier and are never persisted.
- Scratch directories are real owner-only directories, not symlinks, and stale filesystem identity fails closed.
- Closing a session revokes new work synchronously; recursive deletion never runs on Textual's event loop and waits for leases.
- Third-party MCP servers, provider-hosted tools, attachments, Library/RAG data, and generated media retain separate authority contracts.
- Non-Console callers retain the existing configured global file-tool sandbox behavior.
- Cleanup is ordinary best-effort deletion, not secure erase; no cumulative per-chat disk quota is added in this task.
- Targeted tests only; do not run the full repository suite without explicit user approval.
- ADR required: yes.
- ADR path: `backlog/decisions/082-console-per-chat-private-scratch-space.md`.
- Reason: filesystem authority, temporary-data ownership, provider composition, and cross-thread teardown are security and runtime-boundary decisions.

---

### Task 1: Scratch authority manager

**Files:**
- Create: `tldw_chatbook/Chat/console_scratch_space.py`
- Create: `Tests/Chat/test_console_scratch_space.py`

**Interfaces:**
- Consumes: Python `tempfile.mkdtemp`, `os.chmod`, `Path.lstat`, `threading.Condition`, and `shutil.rmtree`.
- Produces: `ConsoleScratchSnapshot(root: Path, token: str, identity: tuple[int, int])`; `ConsoleScratchSpaceUnavailable`; `ConsoleScratchSpaceManager.snapshot(session_id) -> ConsoleScratchSnapshot`; `lease(snapshot) -> ContextManager[Path]`; `is_live(snapshot) -> bool`; `close(session_id) -> None`; `tombstone_all() -> None`; `wait_for_cleanup(timeout_seconds) -> bool`; `dispose(timeout_seconds=2.0) -> bool`.

- [ ] **Step 1: Write allocation and isolation tests**

```python
def test_snapshots_are_distinct_owner_only_and_identifier_free(tmp_path: Path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    first = manager.snapshot("session-visible-id-a")
    second = manager.snapshot("session-visible-id-b")
    assert first.root != second.root
    assert "session-visible-id" not in first.root.name
    assert stat.S_IMODE(first.root.stat().st_mode) == 0o700
    assert not first.root.is_symlink()


def test_chat_cannot_lease_another_chat_snapshot(tmp_path: Path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    first = manager.snapshot("a")
    second = manager.snapshot("b")
    with manager.lease(first) as root:
        (root / "marker.txt").write_text("a", encoding="utf-8")
    assert not (second.root / "marker.txt").exists()
```

- [ ] **Step 2: Run the allocation tests and confirm they fail**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_scratch_space.py -k 'snapshots_are_distinct or cannot_lease' -v`

Expected: collection fails because `tldw_chatbook.Chat.console_scratch_space` does not exist.

- [ ] **Step 3: Add the immutable snapshot, record, allocation, and lease skeleton**

```python
@dataclass(frozen=True, slots=True)
class ConsoleScratchSnapshot:
    root: Path
    token: str
    identity: tuple[int, int]


class ConsoleScratchSpaceUnavailable(RuntimeError):
    pass


class ConsoleScratchSpaceManager:
    def snapshot(self, session_id: str) -> ConsoleScratchSnapshot:
        # Under the condition lock, reuse a live record or create a random
        # mkdtemp directory, chmod 0700, lstat it, and index it by opaque token.

    @contextmanager
    def lease(self, snapshot: ConsoleScratchSnapshot) -> Iterator[Path]:
        # Match token/root/device/inode, reject tombstones, increment the
        # lease count, yield the root, and decrement in finally.
```

The concrete implementation must compare `(st_dev, st_ino)` at every lease acquisition. A missing, symlinked, or replaced root is tombstoned and rejected; cleanup must never recursively delete a replacement whose identity does not match the created directory.

- [ ] **Step 4: Run the allocation tests and confirm they pass**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_scratch_space.py -k 'snapshots_are_distinct or cannot_lease' -v`

Expected: 2 passed.

- [ ] **Step 5: Write lifecycle, stale-generation, and cleanup-thread tests**

```python
def test_close_rejects_new_lease_and_waits_for_last_active_lease(tmp_path: Path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = manager.snapshot("a")
    lease = manager.lease(snapshot)
    lease.__enter__()
    manager.close("a")
    with pytest.raises(ConsoleScratchSpaceUnavailable):
        with manager.lease(snapshot):
            pass
    assert snapshot.root.exists()
    lease.__exit__(None, None, None)
    assert manager.wait_for_cleanup(timeout_seconds=2.0)
    assert not snapshot.root.exists()


def test_reopen_gets_new_generation_and_empty_root(tmp_path: Path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    old = manager.snapshot("session")
    (old.root / "marker").write_text("old", encoding="utf-8")
    manager.close("session")
    assert manager.wait_for_cleanup(timeout_seconds=2.0)
    fresh = manager.snapshot("session")
    assert fresh.token != old.token
    assert fresh.root != old.root
    assert not (fresh.root / "marker").exists()
```

Also test idempotent `close`/`dispose`, cleanup failure remaining tombstoned, filesystem identity replacement refusing a lease without deleting the replacement, and `dispose(timeout_seconds=0.01)` returning `False` while an active lease remains.

- [ ] **Step 6: Implement the single daemon cleanup worker and bounded disposal**

```python
def close(self, session_id: str) -> None:
    with self._condition:
        record = self._by_session.pop(str(session_id), None)
        if record is None:
            return
        record.tombstoned = True
        if record.leases == 0:
            self._schedule_cleanup_locked(record)


def dispose(self, timeout_seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    self.tombstone_all()
    with self._condition:
        while self._records and time.monotonic() < deadline:
            self._condition.wait(deadline - time.monotonic())
        return not self._records
```

`tombstone_all` sets `_disposed`, revokes every live record, and schedules zero-lease records while holding the condition; it performs no recursive I/O. The worker is capped at one daemon thread, logs only bounded failure categories and opaque tokens, removes a record after successful deletion or a confirmed already-missing root, and leaves a failed record tombstoned for a later `dispose` retry.

- [ ] **Step 7: Run the complete manager suite**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_scratch_space.py -v`

Expected: all tests pass and no scratch directory owned by a completed test remains under `tmp_path`.

- [ ] **Step 8: Commit the manager**

```bash
git add tldw_chatbook/Chat/console_scratch_space.py Tests/Chat/test_console_scratch_space.py
git commit -m "feat(console): add private scratch authority manager"
```

---

### Task 2: Runtime ownership, turn snapshots, and session lifecycle

**Files:**
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_turn_context.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`
- Modify: `Tests/Chat/test_console_turn_execution_context.py`
- Modify: `Tests/Chat/test_console_agent_swap.py`

**Interfaces:**
- Consumes: Task 1's `ConsoleScratchSpaceManager` and `ConsoleScratchSnapshot`.
- Produces: `ConsoleRuntime.scratch_spaces`; `ConsoleTurnExecutionContext.scratch_space`; a `scratch_spaces: ConsoleScratchSpaceManager | None = None` keyword on `ConsoleChatController.__init__`; a `scratch_snapshot_provider: Callable[[str], ConsoleScratchSnapshot]` dependency on `ConsoleSessionController`; close-before-store-removal and app-disposal cleanup.

- [ ] **Step 1: Write failing runtime and turn-context tests**

```python
def test_runtime_reuses_one_scratch_manager_across_console_visits(app) -> None:
    runtime = ConsoleRuntime(app)
    first = runtime.scratch_spaces
    runtime.detach_view(None)
    assert runtime.scratch_spaces is first


def test_turn_context_captures_frozen_scratch_snapshot(controller, scratch_manager) -> None:
    context = controller.resolve_turn_execution_context("session-a")
    assert context.scratch_space == scratch_manager.snapshot("session-a")
    assert context.scratch_space.root.is_dir()
```

Extend `test_session_builder_captures_roots_rag_tools_and_generation` with a fake `scratch_snapshot_provider` and assert the mounted-session builder captured that exact immutable snapshot. Add an async runtime-disposal test proving cleanup is invoked through `asyncio.to_thread`, a navigation test proving `leave_console` preserves the root, a same-saved-conversation/two-live-sessions test proving their roots differ, and a controller close test proving `scratch_spaces.close(session_id)` occurs before `store.close_session(session_id)`.

- [ ] **Step 2: Run the lifecycle tests and confirm they fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_runtime_ownership.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_agent_swap.py -k 'scratch or close_session_tombstones' -v`

Expected: failures report missing `scratch_spaces`/`scratch_space` interfaces.

- [ ] **Step 3: Wire the manager through runtime and controller**

```python
class ConsoleRuntime:
    def __init__(self, app: Any) -> None:
        self._scratch_spaces = ConsoleScratchSpaceManager()

    @property
    def scratch_spaces(self) -> ConsoleScratchSpaceManager:
        return self._scratch_spaces

    def ensure_chat_controller(self, **kwargs: Any) -> ConsoleChatController:
        if self._chat_controller is not None or self._disposed:
            return self._chat_controller
        from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

        kwargs.setdefault("scratch_spaces", self._scratch_spaces)
        self._chat_controller = ConsoleChatController(**kwargs)
        self._bind_view_hooks()
        return self._chat_controller
```

`ConsoleChatController` accepts an optional manager for compatibility with direct tests, records whether it owns that fallback manager, and never stores scratch paths in `ConsoleChatStore` or session metadata.

- [ ] **Step 4: Add the scratch snapshot to detached turn context**

```python
@dataclass(frozen=True, slots=True)
class ConsoleTurnExecutionContext:
    session_id: str
    provider_selection: ConsoleProviderSelection
    scratch_space: ConsoleScratchSnapshot | None = None
    session_settings: ConsoleSessionSettings | None = None
    workspace_roots: tuple[str, ...] = ()
    capabilities: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    rag_defaults: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    tool_configuration: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    provider_payload_settings: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
```

Add the exact keyword `scratch_space=self._scratch_spaces.snapshot(session_id)` to the fallback `ConsoleTurnExecutionContext.capture` call in `ConsoleChatController.resolve_turn_execution_context`. Set that fallback's `workspace_roots=()` and remove its `[console] workspace_root` value from `tool_configuration`; it has no Workspace registry authority and must not feed the change-review root scanner. The snapshot is already immutable and must not be deep-copied into a different identity token.

Add `scratch_snapshot_provider` to `ConsoleSessionController.__init__`, store it as `_scratch_snapshot_provider`, and add this keyword to the mounted builder's existing capture call:

```python
scratch_space=self._scratch_snapshot_provider(session_id),
```

`ChatScreen` wires the dependency as `lambda session_id: self._console_runtime().scratch_spaces.snapshot(session_id)`. Its existing `folder_binding_roots(workspace_id)` result remains the mounted turn's `workspace_roots`: Default yields no external roots, while named Workspaces yield only explicit bindings.

- [ ] **Step 5: Fence close and disposal in the approved order**

```python
def close_session(self, session_id: str) -> ConsoleChatSession | None:
    self._scratch_spaces.close(session_id)
    self.prompt_queue_coordinator.mark_closing(session_id)
```

In `ConsoleRuntime.dispose`, call `self._scratch_spaces.tombstone_all()` immediately after setting `_disposed` and before controller shutdown, then insert `await asyncio.to_thread(self._scratch_spaces.dispose)` after controller shutdown and before provider-gateway close. Keep the existing exception containment around app-exit teardown.

Ordinary `ConsoleRuntime.leave_console` remains unchanged and must not close scratch. A controller that created its own compatibility manager disposes it from `shutdown`; a runtime-injected controller leaves final manager disposal to `ConsoleRuntime`.

- [ ] **Step 6: Run lifecycle and context tests**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_scratch_space.py Tests/UI/test_console_runtime_ownership.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_agent_swap.py -k 'scratch or close_session or runtime' -v`

Expected: all selected tests pass.

- [ ] **Step 7: Commit runtime ownership**

```bash
git add tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_turn_context.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_runtime_ownership.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_agent_swap.py
git commit -m "feat(console): bind scratch authority to live sessions"
```

---

### Task 3: Built-in and local filesystem confinement

**Files:**
- Modify: `tldw_chatbook/Tools/workspace_file_roots.py`
- Modify: `tldw_chatbook/Tools/file_operation_tools.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/Tools/test_workspace_file_roots.py`
- Modify: `Tests/Tools/test_file_tools_workspace_roots.py`
- Modify: `Tests/Agents/test_builtin_provider_workspace_binding.py`
- Modify: `Tests/Chat/test_console_agent_bridge_local.py`
- Modify: `Tests/Chat/test_console_local_review_hook.py`

**Interfaces:**
- Consumes: Task 2's `ConsoleTurnExecutionContext.scratch_space` and controller manager.
- Produces: `run_file_sandbox(root)` / `current_run_sandbox_root()`; explicit `sandbox_root` and `sandbox_lease` on `BuiltinToolProvider`; optional `authority_scope` on `LocalToolProvider`; Default Chat local root selection with no config/cwd fallback.

- [ ] **Step 1: Write failing run-scoped built-in isolation tests**

```python
def test_run_file_sandbox_overrides_global_only_inside_scope(tmp_path, monkeypatch) -> None:
    global_root = tmp_path / "global"
    scratch = tmp_path / "chat"
    global_root.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(file_tools, "_resolve_sandbox_config", lambda: str(global_root))
    with run_file_sandbox(scratch):
        assert file_tools._tool_sandbox_root() == scratch.resolve()
    assert file_tools._tool_sandbox_root() == global_root.resolve()


def test_builtin_provider_cannot_read_another_chat_scratch(tmp_path) -> None:
    # Construct providers for A and B with separate roots and allowing gates;
    # A writes a marker and B's absolute read of A returns an outside-root error.
```

Also assert a named Workspace provider resolves `(scratch, *live_bindings)` and registry failure resolves `(scratch,)`.

- [ ] **Step 2: Run the built-in tests and confirm they fail**

Run: `.venv/bin/python -m pytest Tests/Tools/test_workspace_file_roots.py Tests/Tools/test_file_tools_workspace_roots.py Tests/Agents/test_builtin_provider_workspace_binding.py -k 'run_file_sandbox or scratch or another_chat' -v`

Expected: failures report missing run-scoped sandbox and provider constructor parameters.

- [ ] **Step 3: Add the scoped sandbox and explicit built-in provider authority**

```python
_RUN_FILE_SANDBOX_ROOT: ContextVar[Path | None] = ContextVar(
    "run_file_sandbox_root", default=None
)


@contextmanager
def run_file_sandbox(root: Path | None) -> Iterator[None]:
    token = _RUN_FILE_SANDBOX_ROOT.set(root.resolve() if root is not None else None)
    try:
        yield
    finally:
        _RUN_FILE_SANDBOX_ROOT.reset(token)
```

`_tool_sandbox_root()` returns the scoped root without creating or widening it; outside a scope it retains the existing configured global behavior. `BuiltinToolProvider.path_targets` and file-tool `invoke` enter the injected lease and both `run_file_sandbox(self._sandbox_root)` and `run_workspace(self._workspace_id)`. `path_precheck_failed` accepts the same explicit root and lease so review and dispatch consult identical roots.

- [ ] **Step 4: Write failing Default/local provider tests**

```python
def test_default_chat_local_provider_uses_scratch_not_config_or_cwd(controller, turn_context, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path / "cwd")
    turn_context = replace_tool_config(turn_context, workspace_root=str(tmp_path / "configured"))
    provider, review = controller._compose_local_provider(
        session_id=turn_context.session_id,
        turn_context=turn_context,
    )
    assert provider.workspace_root == turn_context.scratch_space.root
    assert review is not None


def test_selected_project_binding_still_wins_and_read_only_omits_writes(
    controller, turn_context, tmp_path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    provider, review = controller._compose_local_provider(
        session_id=turn_context.session_id,
        turn_context=turn_context,
        project_root=project,
        allow_write=False,
        project_root_guard=lambda: True,
    )
    assert provider.workspace_root == project.resolve()
    assert "fs_write" not in {entry.name for entry in provider.list_catalog()}
    assert review is not None
```

Use the existing project-root helpers and assertions in `Tests/Chat/test_console_local_review_hook.py`; repair `test_selected_root_swap_fails_closed_before_local_invoke` to call `review(calls, "run-root-swap")` so it exercises the production guard.

- [ ] **Step 5: Run the local tests and confirm the Default root assertion fails**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge_local.py Tests/Chat/test_console_local_review_hook.py -k 'default_chat_local_provider or selected_root_swap or read_only' -v`

Expected: Default composition resolves configured root/cwd before the implementation; the repaired root-swap regression reaches its guard rather than raising `TypeError`.

- [ ] **Step 6: Add lease-aware local invocation and remove the Console fallback**

```python
if project_root is None:
    snapshot = turn_context.scratch_space if turn_context is not None else None
    if snapshot is None:
        return None, None
    root = snapshot.root
    authority_scope = functools.partial(self._scratch_spaces.lease, snapshot)
else:
    root = project_root
    authority_scope = None
```

`LocalToolProvider` exposes a read-only `workspace_root` property for verification and accepts `authority_scope: Callable[[], ContextManager[Path]] | None`. It enters that scope around `fs_*` and Git path preflight/invocation only; lease acquisition failure returns `LOCAL_ROOT_CHANGED_REFUSAL`. Web, Watchlists, and todo calls do not hold a scratch lease. Existing project-root `root_guard`, fingerprint, and `allow_write` behavior remains byte-for-byte at the controller seam.

- [ ] **Step 7: Run the complete authority regression set**

Run: `.venv/bin/python -m pytest Tests/Tools/test_workspace_file_roots.py Tests/Tools/test_file_tools_workspace_roots.py Tests/Agents/test_builtin_provider_workspace_binding.py Tests/Chat/test_console_agent_bridge_local.py Tests/Chat/test_console_local_review_hook.py -v`

Expected: all tests pass, including the selected-root swap fail-closed regression.

- [ ] **Step 8: Commit provider confinement**

```bash
git add tldw_chatbook/Tools/workspace_file_roots.py tldw_chatbook/Tools/file_operation_tools.py tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Chat/console_chat_controller.py Tests/Tools/test_workspace_file_roots.py Tests/Tools/test_file_tools_workspace_roots.py Tests/Agents/test_builtin_provider_workspace_binding.py Tests/Chat/test_console_agent_bridge_local.py Tests/Chat/test_console_local_review_hook.py
git commit -m "fix(console): confine file tools to chat scratch authority"
```

---

### Task 4: Agent bridge propagation and cross-chat dispatch tests

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_agent_swap.py`
- Modify: `Tests/Agents/test_project_instruction_path_targets.py`

**Interfaces:**
- Consumes: Task 3's `BuiltinToolProvider` keywords `sandbox_root: Path | None` and `sandbox_lease: Callable[[], ContextManager[Path]] | None`, plus local provider composition.
- Produces: `scratch_root`/`scratch_lease` propagation through `_compose_run_registry_and_allowed`, `build_console_first_request_plan`, and `ConsoleAgentBridge.run_reply`; identical scratch authority for review precheck and actual dispatch.

- [ ] **Step 1: Write failing bridge propagation tests**

```python
def test_run_registry_binds_builtin_provider_to_captured_scratch(tmp_path) -> None:
    registry, _, _, _ = _compose_run_registry_and_allowed(
        {},
        builtin_gate=_AllowGate(),
        workspace_id=DEFAULT_WORKSPACE_ID,
        scratch_root=tmp_path / "chat-a",
        scratch_lease=nullcontext,
    )
    resolved = registry.resolve_owner_for_name("read_file")
    assert resolved is not None
    _, provider = resolved
    assert provider.sandbox_root == (tmp_path / "chat-a").resolve()


def test_two_console_runs_cannot_dispatch_across_scratch_roots(tmp_path) -> None:
    root_a = tmp_path / "chat-a"
    root_b = tmp_path / "chat-b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "marker.txt").write_text("chat-a", encoding="utf-8")
    registry_b, _, _, _ = _compose_run_registry_and_allowed(
        {},
        builtin_gate=_AllowGate(),
        workspace_id=DEFAULT_WORKSPACE_ID,
        scratch_root=root_b,
        scratch_lease=nullcontext,
    )
    result = registry_b.invoke_by_name(
        "read_file", {"file_path": str(root_a / "marker.txt")}
    )
    assert result.ok is False
    assert "outside" in str(result.error).lower()
```

- [ ] **Step 2: Run the bridge tests and confirm they fail**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_swap.py Tests/Agents/test_project_instruction_path_targets.py -k 'captured_scratch or across_scratch or project_instruction_path' -v`

Expected: missing `scratch_root`/`scratch_lease` parameters or the shared global sandbox remains visible.

- [ ] **Step 3: Thread the immutable authority through the live run**

```python
snapshot = turn_context.scratch_space
if snapshot is None:
    return self._block(session_id, "Private scratch space is unavailable.")
scratch_lease = functools.partial(self._scratch_spaces.lease, snapshot)

scratch_run_kwargs = {
    "scratch_root": snapshot.root,
    "scratch_lease": scratch_lease,
}
```

Add the two entries in `scratch_run_kwargs` to the existing `asyncio.to_thread` call that invokes `self._agent_bridge.run_reply`. Add matching parameters to `ConsoleAgentBridge.run_reply`, `build_console_first_request_plan`, and `_compose_run_registry_and_allowed`; pass them to every fresh `BuiltinToolProvider`. The disposable Context/preview composition must receive the same snapshot but must not create a second directory.

- [ ] **Step 4: Make review precheck and dispatch authority identical**

```python
builtin_review_provider = BuiltinToolProvider(
    gate=builtin_gate,
    workspace_id=review_workspace_id,
    sandbox_root=snapshot.root,
    sandbox_lease=scratch_lease,
)
```

`build_tool_review_hook` reads path-precheck roots through this provider authority. Add a spy test asserting precheck and invocation observed the exact same canonical scratch path and Workspace ID.

- [ ] **Step 5: Run bridge and provider suites**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_swap.py Tests/Agents/test_project_instruction_path_targets.py Tests/Chat/test_console_local_review_hook.py Tests/Agents/test_builtin_provider_workspace_binding.py -v`

Expected: all selected tests pass.

- [ ] **Step 6: Commit bridge propagation**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_swap.py Tests/Agents/test_project_instruction_path_targets.py
git commit -m "feat(console): propagate scratch authority through agent runs"
```

---

### Task 5: Retained skill output and run-log fallback

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py`
- Modify: `tldw_chatbook/Skills_Interop/skills_scope_service.py`
- Modify: `tldw_chatbook/Agents/run_log.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/Skills/test_skill_script_service.py`
- Modify: `Tests/Agents/test_run_log_sandbox_isolation.py`
- Modify: `Tests/Agents/test_run_log_workspace_isolation.py`
- Modify: `Tests/Agents/test_run_log_service_wiring.py`
- Modify: `Tests/Chat/test_console_agent_tool_result_cap.py`

**Interfaces:**
- Consumes: Task 4's run `scratch_root` and lease factory.
- Produces: optional explicit `output_root` for local skill execution; explicit root/access scope for `RunLogWriter`; live in-memory run-log authority lookup scoped to the owning Console session.

- [ ] **Step 1: Write failing retained-output tests**

```python
@pytest.mark.asyncio
async def test_explicit_console_output_root_overrides_shared_skill_config(service, tmp_path) -> None:
    chat_root = tmp_path / "chat"
    result = await service.run_skill_script(
        "producer", "scripts/write.py", [], output_root=chat_root / "skill_script_output"
    )
    assert Path(result.output_dir).is_relative_to(chat_root)


@pytest.mark.asyncio
async def test_two_console_skill_runs_retain_under_their_own_chat_roots(
    service, tmp_path
) -> None:
    first = tmp_path / "chat-a" / "skill_script_output"
    second = tmp_path / "chat-b" / "skill_script_output"
    result_a = await service.run_skill_script(
        "producer", "scripts/write.py", [], output_root=first
    )
    result_b = await service.run_skill_script(
        "producer", "scripts/write.py", [], output_root=second
    )
    assert Path(result_a.output_dir).is_relative_to(first)
    assert Path(result_b.output_dir).is_relative_to(second)
    assert not Path(result_a.output_dir).is_relative_to(second)
```

Also assert the existing non-Console no-argument path still honors `[skills] script_scratch_root` or the global file-tool sandbox.

- [ ] **Step 2: Run skill-output tests and confirm the new calls fail**

Run: `.venv/bin/python -m pytest Tests/Skills/test_skill_script_service.py -k 'explicit_console_output_root or own_chat_roots or script_output_root' -v`

Expected: `run_skill_script` rejects the unknown `output_root` keyword.

- [ ] **Step 3: Add explicit output-root forwarding and lease it in the bridge**

```python
async def run_skill_script(
    self,
    skill_name: str,
    script_path: str,
    args: Sequence[str],
    *,
    limits: ScriptRunLimits | None = None,
    output_root: Path | None = None,
) -> ScriptRunResult:
    effective_output_root = (
        output_root.resolve() if output_root is not None else self._script_output_root()
    )
    return await asyncio.to_thread(_run_in_scratch_dir, effective_output_root)
```

`SkillsScopeService.run_skill_script` forwards `output_root`. The local service resolves and creates the explicit root without reading the configured shared root. `ConsoleAgentBridge.run_skill_script_tool` holds `scratch_lease()` around the whole blocking script coroutine call and passes `scratch_root / "skill_script_output"`; the existing offloaded subprocess callable still owns its child scratch lifecycle.

- [ ] **Step 4: Write failing run-log authority tests**

```python
def test_writer_explicit_fallback_root_never_uses_global_sandbox(tmp_path, monkeypatch) -> None:
    chat_root = tmp_path / "chat"
    chat_root.mkdir()
    writer = RunLogWriter(root=chat_root, access_scope=nullcontext)
    writer.bind("run-a")
    assert writer.log_dir.is_relative_to(chat_root)


def test_console_bridge_does_not_find_another_sessions_scratch_log(
    console_bridge, tmp_path
) -> None:
    root_a = tmp_path / "chat-a"
    root_b = tmp_path / "chat-b"
    root_a.mkdir()
    root_b.mkdir()
    writer = RunLogWriter(root=root_a)
    writer.bind("run-a")
    writer.append(run_id="run-a", kind="primary", type="model", content="secret")
    console_bridge._remember_run_log_authority(
        run_id="run-a", session_id="session-a", root=root_a, access_scope=nullcontext
    )
    console_bridge._remember_run_log_authority(
        run_id="run-b", session_id="session-b", root=root_b, access_scope=nullcontext
    )
    assert console_bridge.run_log_available("run-b") is False
```

Add a lease test: close/tombstone while a writer access scope is active leaves the directory until access exits; a later append fails closed and does not recreate it. Preserve the Workspace test proving a validated writable binding remains preferred over scratch.

- [ ] **Step 5: Run run-log tests and confirm they fail**

Run: `.venv/bin/python -m pytest Tests/Agents/test_run_log_sandbox_isolation.py Tests/Agents/test_run_log_workspace_isolation.py Tests/Agents/test_run_log_service_wiring.py Tests/Chat/test_console_agent_tool_result_cap.py -k 'explicit_fallback_root or another_sessions or workspace_root or lease' -v`

Expected: missing explicit writer root/access-scope interfaces or current global-root lookup finds the wrong container.

- [ ] **Step 6: Add explicit run-log roots and live authority lookup**

```python
class RunLogWriter:
    def __init__(
        self,
        *,
        dir_name: str | None = None,
        segment_bytes: int | None = None,
        max_record_bytes: int | None = None,
        root: Path | None = None,
        access_scope: Callable[[], ContextManager[Path]] | None = None,
        on_bound: Callable[[str, Path], None] | None = None,
    ) -> None:
        self._explicit_root = root.resolve() if root is not None else None
        self._access_scope = access_scope or nullcontext
        self._on_bound = on_bound
```

Wrap `bind`, `append`, `write_manifest`, and filesystem-touching `close` work in `access_scope()`. An unavailable scope deactivates logging without widening. `resolve_existing_log_dir(run_id, *, root=None)` uses an explicit root when supplied and retains the existing global resolver only for non-Console callers.

`ConsoleAgentBridge` records an in-memory map from owning primary run ID to `(session_id, resolved_log_root, access_scope)` via `RunLogWriter.on_bound`. Its `run_log_available` and `load_run_log_text` require that explicit live authority; they never search the global sandbox for a Console run. Add `forget_session_file_authority(session_id)` and call it during controller close after tombstoning scratch. Do not persist the root or token in `AgentRunsDB`.

Before constructing `AgentService`, the bridge resolves the writer root with the run's captured authority:

```python
with scratch_lease():
    run_log_root = resolve_log_root(
        sandbox_root=scratch_root,
        workspace_id=run_workspace_id,
    )
run_log_writer = RunLogWriter(
    root=run_log_root,
    access_scope=scratch_lease,
    on_bound=functools.partial(
        self._remember_run_log_authority,
        session_id=session_id,
        access_scope=scratch_lease,
    ),
)
```

Add explicit `sandbox_root` and `workspace_id` keywords to `resolve_log_root`; it enters `run_workspace(workspace_id)` and calls the existing `allowed_file_roots(write=True, sandbox_root=sandbox_root)`, preserving the first writable Workspace binding preference and falling back only to this chat's scratch. Pass `run_log_writer=run_log_writer` into `AgentService`.

- [ ] **Step 7: Run skill, run-log, bridge, and lifecycle suites**

Run: `.venv/bin/python -m pytest Tests/Skills/test_skill_script_service.py Tests/Agents/test_run_log_sandbox_isolation.py Tests/Agents/test_run_log_workspace_isolation.py Tests/Agents/test_run_log_service_wiring.py Tests/Chat/test_console_agent_tool_result_cap.py Tests/Chat/test_console_scratch_space.py Tests/Chat/test_console_agent_bridge.py -v`

Expected: all selected tests pass.

- [ ] **Step 8: Commit scratch-adjacent artifacts**

```bash
git add tldw_chatbook/Skills_Interop/local_skills_service.py tldw_chatbook/Skills_Interop/skills_scope_service.py tldw_chatbook/Agents/run_log.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py Tests/Skills/test_skill_script_service.py Tests/Agents/test_run_log_sandbox_isolation.py Tests/Agents/test_run_log_workspace_isolation.py Tests/Agents/test_run_log_service_wiring.py Tests/Chat/test_console_agent_tool_result_cap.py
git commit -m "fix(console): isolate retained artifacts by chat"
```

---

### Task 6: Folderless UX, recovery copy, and documentation

**Files:**
- Modify: `tldw_chatbook/Workspaces/display_state.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_details.py`
- Modify: `tldw_chatbook/Utils/path_validation.py`
- Modify: `Tests/Workspaces/test_workspace_display_state.py`
- Modify: `Tests/UI/test_console_workspace_details_tray.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify: `Tests/Utils/test_path_validation_multi.py`
- Modify: `Docs/User_Guide/console/sessions-tabs-workspaces.md`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/settings.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: the implemented authority semantics from Tasks 1-5.
- Produces: truthful `Local file tools` status copy, optional-folder recovery guidance, and user/developer documentation.

- [ ] **Step 1: Write failing display and new-chat tests**

```python
def test_default_workspace_reports_private_scratch(state) -> None:
    assert state.runtime_label == "Local file tools: Private scratch"


def test_details_tray_reports_private_scratch_plus_bound_folders() -> None:
    friendly = ConsoleWorkspaceDetailsTray._friendly_status_label(
        "Local file tools: Private scratch + 2 folders"
    )
    label, value = ConsoleWorkspaceDetailsTray._split_status_row(
        friendly, "Local file tools"
    )
    assert label == "Local file tools"
    assert value == "Private scratch + 2 folders"
```

Extend `test_console_workspace_rail_new_conversation_creates_default_workspace_session` to assert no modal/folder picker appears and the new session can resolve a scratch snapshot.

- [ ] **Step 2: Run UI/display tests and confirm the old `Off in Default` copy fails**

Run: `.venv/bin/python -m pytest Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_details_tray.py Tests/UI/test_console_native_chat_flow.py -k 'private_scratch or new_conversation_creates_default' -v`

Expected: Default renders `Runtime: none, file tools disabled` / `File tools: Off in Default` before the copy change.

- [ ] **Step 3: Implement status copy from actual folder bindings**

```python
runtime_label = (
    "Local file tools: Private scratch"
    if not runtime_bindings
    else f"Local file tools: Private scratch + {len(runtime_bindings)} {_plural('folder', len(runtime_bindings))}"
)
```

Keep missing-binding detail available in recovery text or a secondary diagnostic; do not count non-filesystem runtime bindings as folders. `_friendly_status_label` passes `Local file tools:` through and removes the `Off in Default` branch.

- [ ] **Step 4: Write and implement optional-folder recovery copy**

```python
ROOT_DENIAL_RECOVERY_POINTER = "Need that folder? Add it to a named Workspace."
ROOT_DENIAL_RECOVERY_HINT = (
    "Chats do not need a folder. To give local file tools access outside "
    "private scratch, bind that folder in Settings > Workspaces and use a "
    "chat in that Workspace."
)
```

Update `Tests/Utils/test_path_validation_multi.py` to assert the short pointer survives the real 160-character transcript truncation and the full hint says folders are optional for Chats.

- [ ] **Step 5: Run UI and recovery tests**

Run: `.venv/bin/python -m pytest Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_details_tray.py Tests/UI/test_console_native_chat_flow.py Tests/Utils/test_path_validation_multi.py -v`

Expected: all selected tests pass.

- [ ] **Step 6: Update user and developer documentation**

Document these exact rules:

```text
Chats: private temporary scratch only; no folder setup; scratch is new after close/reopen.
Workspaces: private scratch plus explicit bound folders; folders are optional.
Local fs/Git: scratch unless project instructions explicitly select one binding.
[console] workspace_root: retained for compatibility outside this Console authority path; never grants a Console Chat access.
```

The agent-tools guide must distinguish local Chatbook tools from MCP/provider-hosted tools, describe normal approval prompts, and state that normal deletion is not secure erase.

- [ ] **Step 7: Commit UX and documentation**

```bash
git add tldw_chatbook/Workspaces/display_state.py tldw_chatbook/Widgets/Console/console_workspace_details.py tldw_chatbook/Utils/path_validation.py Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_details_tray.py Tests/UI/test_console_native_chat_flow.py Tests/Utils/test_path_validation_multi.py Docs/User_Guide/console/sessions-tabs-workspaces.md Docs/User_Guide/console/agent-runs-and-tools.md Docs/User_Guide/settings.md AGENTS.md
git commit -m "docs(console): clarify private scratch and optional folders"
```

---

### Task 7: Targeted verification and live DeepSeek UAT

**Files:**
- Create: `Docs/UAT/2026-08-23-console-deepseek-private-scratch.md`
- Modify: `backlog/tasks/task-21161 - Console-per-chat-private-scratch-space-and-workspace-file-authority.md`
- Modify only if the implementation surfaced a reusable incident: `backlog/docs/lessons-live-verification.md` or `backlog/docs/lessons-testing-evidence.md`

**Interfaces:**
- Consumes: all prior tasks, the existing DeepSeek config through `TLDW_CONFIG_PATH`, and the targeted-test requirement in `AGENTS.md`.
- Produces: reproducible automated evidence, sanitized live-UAT evidence, completed task acceptance criteria, implementation notes, and Done status only when every Definition-of-Done condition is satisfied.

- [ ] **Step 1: Run static and focused authority verification**

Run:

```bash
.venv/bin/python -m pytest Tests/Chat/test_console_scratch_space.py Tests/Chat/test_console_turn_execution_context.py Tests/Tools/test_workspace_file_roots.py Tests/Tools/test_file_tools_workspace_roots.py Tests/Agents/test_builtin_provider_workspace_binding.py Tests/Agents/test_project_instruction_path_targets.py Tests/Chat/test_console_agent_bridge_local.py Tests/Chat/test_console_local_review_hook.py -q
```

Expected: all tests pass; record exact counts.

- [ ] **Step 2: Run artifact, lifecycle, and UI verification**

Run:

```bash
.venv/bin/python -m pytest Tests/Skills/test_skill_script_service.py Tests/Agents/test_run_log_sandbox_isolation.py Tests/Agents/test_run_log_workspace_isolation.py Tests/Agents/test_run_log_service_wiring.py Tests/Chat/test_console_agent_tool_result_cap.py Tests/UI/test_console_runtime_ownership.py Tests/Chat/test_console_agent_swap.py Tests/Workspaces/test_workspace_display_state.py Tests/UI/test_console_workspace_details_tray.py Tests/UI/test_console_native_chat_flow.py Tests/Utils/test_path_validation_multi.py -q
```

Expected: all tests pass; if a pre-existing unrelated failure appears, reproduce it against `7363592020076c9508fe4a0eee0c1a1679ec7851` before classifying it as baseline.

- [ ] **Step 3: Run formatting/static checks scoped to changed Python files**

Run:

```bash
git diff --check "$(git merge-base origin/dev HEAD)" HEAD
.venv/bin/python -m compileall -q tldw_chatbook/Chat/console_scratch_space.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_turn_context.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Agents/run_log.py tldw_chatbook/Skills_Interop/local_skills_service.py tldw_chatbook/Skills_Interop/skills_scope_service.py
```

Expected: both commands exit 0.

- [ ] **Step 4: Prepare a credential-safe live-UAT profile**

Create a private temporary directory with `mktemp -d`, copy the user's config without printing it, set file mode `0600`, and compute and save the original config's SHA-256 before launch. In the private copy only, set `[paths].data_dir` to a directory under that temporary profile and enable the existing Console local/file-tool gates required by the scenario; do not alter the DeepSeek credential value. Launch with both `HOME` pointing at a temporary home and `TLDW_CONFIG_PATH` pointing at the private copy, so databases, permission decisions, logs, caches, and config writes cannot reach the user's live profile. Never pass or echo the DeepSeek key in a command argument, screenshot, log, or repository file.

- [ ] **Step 5: Exercise the mounted Console with DeepSeek**

Record pass/fail and sanitized observations for:

```text
1. Select DeepSeek and create a Chat under Chats without any folder prompt.
2. Send a plain prompt and observe a successful streamed response.
3. Approve a local file-tool request that writes and reads a unique relative marker.
4. Create a second Chat and verify listing/reading the first marker is refused or absent.
5. Bind a disposable folder to a named Workspace and verify an allowed read plus a read-write write.
6. Return to a Chat and verify that Workspace file is outside its authority.
7. Close the first Chat, reopen its saved conversation, and verify the old marker is absent.
8. Verify status copy says Private scratch and no Chat folder prompt appeared.
```

If DeepSeek declines to call a tool, make the prompt explicit once and record that model tool selection is observational; deterministic tests remain the authority gate.

- [ ] **Step 6: Verify configuration integrity and clean UAT artifacts**

Compute the original config's SHA-256 again and require an exact match. Delete the temporary profile and disposable Workspace folder after recording only path-free outcomes and counts. Confirm no UAT marker or credential-like string is staged by `git status --short` and a focused secret-pattern scan.

- [ ] **Step 7: Write the UAT report and perform final self-review**

`Docs/UAT/2026-08-23-console-deepseek-private-scratch.md` records the branch/base/commit, DeepSeek provider and model identifier without credentials, terminal size/profile isolation method, each scenario outcome, targeted command counts, known hard-crash residue/no-quota limitations, and any deviations. Review the complete diff for authority widening, raw path logging, unbounded UI-thread cleanup, persisted scratch locators, and accidental changes to non-Console callers.

- [ ] **Step 8: Complete Backlog task hygiene only after every gate passes**

Check all nine acceptance criteria, add concise `## Implementation Notes`, include the ADR-082 link and exact verification evidence, then run:

```bash
backlog task edit 21161 -s Done
backlog task 21161 --plain
```

Expected: TASK-21161 reports Done, all acceptance criteria checked, implementation plan and notes present.

- [ ] **Step 9: Commit verification evidence**

```bash
git add Docs/UAT/2026-08-23-console-deepseek-private-scratch.md 'backlog/tasks/task-21161 - Console-per-chat-private-scratch-space-and-workspace-file-authority.md' backlog/docs/lessons-live-verification.md backlog/docs/lessons-testing-evidence.md
git commit -m "test(console): verify DeepSeek private scratch UAT"
```

Stage a lessons file only when a real reusable incident was added; otherwise omit it from `git add`.
