"""SSH bindings in run composition + the availability guarantee (Phase 4a).

Spec: ``Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md``,
"Run composition & tool parity", "Degraded semantics", "Automatic recovery",
and Testing #6's three NAMED availability regression tests:

(a) workspace with remote BLOCKED + local READY -> send composes, local
    tools advertised, remote excluded, the context note mentions it;
(b) a healthy remote whose op times out returns a typed error and stays
    admitted on the next send (the cache never flips on OP_TIMEOUT);
(c) a BLOCKED remote whose debounced recovery probe succeeds is
    re-admitted on the next send with no user action.

Plus the hot-path rule (composition spawns no subprocess and touches no
network), cold-start-optimistic admission, the degraded-selection
warning, the any_write preflight for read-only remotes, git_*'s
call-time typed error, the URI-in-path teaching message, and the
wire-dict-to-str adapter the composition seam formalizes.

No real ssh anywhere: fake registry rows, the real status cache, and
stub executors (the loopback/real transports are covered by
``Tests/Tools/test_remote_executor_ssh.py``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import (
    capture_run_admitted_workspace_roots,
)
from tldw_chatbook.Tools.remote_root_types import RemoteRoot, is_remote


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Binding:
    """One registry row (fake shape the controller admission reads)."""

    def __init__(
        self,
        workspace_id: str,
        binding_id: str,
        *,
        kind: str = "local-filesystem",
        locator: str = "",
        access: str = "rw",
        status: str = "ready",
    ):
        self.workspace_id = workspace_id
        self.binding_id = binding_id
        self.binding_kind = kind
        self.status = status
        self.locator = locator
        self.metadata = {"access": access}


class _Registry:
    def __init__(self, *bindings: _Binding):
        self._bindings = {b.binding_id: b for b in bindings}

    def list_runtime_bindings(self, workspace_id):
        return tuple(
            b for b in self._bindings.values() if b.workspace_id == workspace_id
        )

    def get_runtime_binding(self, binding_id):
        return self._bindings.get(binding_id)

    def get_workspace(self, workspace_id):
        return SimpleNamespace(name=f"workspace {workspace_id}")

    def list_folder_bindings(self, workspace_id):
        return tuple(
            b
            for b in self.list_runtime_bindings(workspace_id)
            if b.binding_kind == "local-filesystem"
        )

    def list_ssh_bindings(self, workspace_id):
        return tuple(
            b
            for b in self.list_runtime_bindings(workspace_id)
            if b.binding_kind == "ssh-filesystem"
        )


class _Session:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id


class _StubRemoteExecutor:
    """Composition-slot stand-in honoring the adapter's dispatch contract.

    ``execute`` mirrors the real ssh executor's wire-dict result and its
    cache taxonomy for failures (OP_TIMEOUT records the transient, never
    a state flip); ``maybe_schedule_recovery_probe`` debounces through
    the REAL cache (``should_schedule_probe``) and runs the probe
    synchronously so the recovery test drives it deterministically.
    """

    def __init__(
        self,
        binding_id: str,
        host_key: str,
        cache,
        *,
        failure=None,
        probe=None,
        result_text: str = "stub remote content\n",
    ):
        self.binding_id = binding_id
        self.host_key = host_key
        self.cache = cache
        self.failure = failure
        self.probe = probe
        self.result_text = result_text
        self.execute_calls: list[tuple[str, dict]] = []
        self.probe_runs = 0

    def execute(self, tool: str, args: dict, *, intent: str):
        self.execute_calls.append((tool, dict(args)))
        if self.failure is not None:
            kind, reason, admitted = self.failure
            if kind is not None:
                from tldw_chatbook.Tools.remote_binding_status import (
                    RemoteBindingStatusCache,
                )

                assert isinstance(self.cache, RemoteBindingStatusCache)
                self.cache.record_transport_failure(
                    self.binding_id, kind, reason
                )
            from tldw_chatbook.Tools.remote_workspace_executor import (
                RemoteWorkspaceExecutionError,
            )

            raise RemoteWorkspaceExecutionError(reason, reason, admitted=admitted)
        return {
            "version": 1,
            "operation_id": "op-stub",
            "outcome": "success",
            "code": "ok",
            "result": self.result_text,
            "error": None,
            "elapsed_ms": 1,
            "truncated": False,
            "cleanup_proven": True,
        }

    def ping(self):
        if self.probe is not None:
            self.probe()
        return {
            "identity_chain": [["/srv/www", 99, 7, 16877]],
            "canonical_path": "/srv/www",
            "python_version": "3.12.0",
            "bundle_sha256": "0" * 64,
        }

    def maybe_schedule_recovery_probe(self) -> None:
        if not self.cache.should_schedule_probe(self.host_key):
            return
        self.probe_runs += 1
        self.ping()


def _stub_factory(executors: dict[str, _StubRemoteExecutor]):
    """``remote_executor_factory`` seam: stubs behind the REAL adapter.

    Wrapping the stubs in the production wire-dict-to-str adapter keeps
    the stub-based tests exercising the actual composition surface
    (git refusal, URI teaching, result extraction) instead of a
    test-only pass-through.
    """

    from tldw_chatbook.Chat.console_chat_controller import (
        _RemoteExecutorDispatchAdapter,
    )

    def factory(binding, binding_id, *, status_cache, sensitive_exclusions):
        return _RemoteExecutorDispatchAdapter(
            executors[str(binding_id)], alias=binding_id
        )

    return factory


def _blocked_cache(cache, binding_id: str, *, reason: str = "host unreachable"):
    from tldw_chatbook.Tools.remote_workspace_transport import TransportFailureKind

    cache.record_transport_failure(
        binding_id, TransportFailureKind.UNREACHABLE, reason
    )
    return cache


def _allow_provider(tmp_path: Path, roots, monkeypatch=None):
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    if monkeypatch is not None:
        # This machine's config bootstrap is environmentally broken for the
        # provider's construction-time catalog gates (249 pre-existing
        # failures in Tests/Agents/test_local_tool_provider.py on HEAD);
        # route the two gate reads at their defaults so the provider builds.
        import tldw_chatbook.Agents.local_tool_provider as provider_module

        monkeypatch.setattr(
            provider_module,
            "get_cli_setting",
            lambda section, key=None, default=None: default,
        )
    return LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
        admitted_roots=roots,
    )


def _local_binding(tmp_path: Path, binding_id: str = "folder-1") -> _Binding:
    root = (tmp_path / binding_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return _Binding("ws", binding_id, locator=str(root))


def _ssh_binding(binding_id: str = "ssh-b1", *, access: str = "rw") -> _Binding:
    return _Binding(
        "ws", binding_id, kind="ssh-filesystem", locator="ssh://devbox/srv/www", access=access
    )


def _host_key(binding_id: str = "ssh-b1") -> str:
    return "ssh-b1@devbox:22" if binding_id == "ssh-b1" else f"{binding_id}@devbox:22"


def _compose(
    registry: _Registry,
    cache,
    *,
    executors: dict[str, _StubRemoteExecutor] | None = None,
    project_selection=None,
    project_authority_guard=None,
):
    return capture_run_admitted_workspace_roots(
        session=_Session("ws"),
        registry=registry,
        project_selection=project_selection,
        project_authority_guard=project_authority_guard,
        status_cache=cache,
        remote_executor_factory=(
            _stub_factory(executors) if executors is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# App-level status cache singleton
# ---------------------------------------------------------------------------


def test_singleton_cache_accessor_is_stable_and_injectable():
    from tldw_chatbook.Tools.remote_binding_status import (
        RemoteBindingStatusCache,
        get_remote_binding_status_cache,
        set_remote_binding_status_cache,
    )

    first = get_remote_binding_status_cache()
    assert isinstance(first, RemoteBindingStatusCache)
    assert get_remote_binding_status_cache() is first

    injected = RemoteBindingStatusCache()
    set_remote_binding_status_cache(injected)
    try:
        assert get_remote_binding_status_cache() is injected
    finally:
        set_remote_binding_status_cache(None)
    assert get_remote_binding_status_cache() is not injected


# ---------------------------------------------------------------------------
# Named availability regression (a): BLOCKED remote + READY local
# ---------------------------------------------------------------------------


def test_blocked_remote_and_local_ready_composes_local_only(tmp_path, monkeypatch):
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache
    from tldw_chatbook.Tools.workspace_file_roots import workspace_context_note

    cache = RemoteBindingStatusCache()
    _blocked_cache(cache, "ssh-b1")
    stub = _StubRemoteExecutor("ssh-b1", _host_key(), cache)
    registry = _Registry(_local_binding(tmp_path), _ssh_binding())

    roots = _compose(registry, cache, executors={"ssh-b1": stub})

    # The remote root is excluded; only the local binding is admitted.
    assert [root.alias for root in roots] == ["folder-1"]
    assert all(not is_remote(root.root) for root in roots)
    # The debounced recovery probe was scheduled exactly once.
    assert stub.probe_runs == 1

    provider = _allow_provider(tmp_path, roots, monkeypatch)
    advertised = sorted(provider._specs)
    assert "fs_list" in advertised and "fs_read" in advertised
    assert "fs_write" in advertised  # local binding is rw
    fs_list = provider._specs["fs_list"].parameters
    assert fs_list["properties"]["root_alias"]["enum"] == ["folder-1"]

    # The context note tells the model the remote binding was excluded.
    note = workspace_context_note(
        "ws", registry=registry, launch_cwd=tmp_path, status_cache=cache
    )
    assert "remote binding unreachable — excluded this run" in note
    assert "ssh-b1" in note


# ---------------------------------------------------------------------------
# Named availability regression (b): operation timeout is a typed error,
# the cache never flips, the next send still admits the root
# ---------------------------------------------------------------------------


def test_operation_timeout_typed_error_and_next_send_admits(tmp_path, monkeypatch):
    from loguru import logger as loguru_logger

    from tldw_chatbook.Tools.remote_binding_status import BindingState
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache
    from tldw_chatbook.Tools.remote_workspace_transport import TransportFailureKind

    cache = RemoteBindingStatusCache()
    stub = _StubRemoteExecutor(
        "ssh-b1",
        _host_key(),
        cache,
        failure=(TransportFailureKind.OP_TIMEOUT, "operation timed out", True),
    )
    registry = _Registry(_ssh_binding())

    roots = _compose(registry, cache, executors={"ssh-b1": stub})
    assert len(roots) == 1 and is_remote(roots[0].root)

    provider = _allow_provider(tmp_path, roots, monkeypatch)
    captured_errors: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: captured_errors.append(message), level="DEBUG"
    )
    try:
        result = provider._invoke_detailed(
            "local:fs_read",
            {"path": "app/main.py", "root_alias": "ssh-b1"},
        ).result
    finally:
        loguru_logger.remove(sink_id)
    assert result.ok is False
    assert "operation timed out" in (result.error or "")

    # OP_TIMEOUT is a no-flip taxonomy row: the state stays READY (only
    # the transient counter moves), so the next send admits the root.
    assert cache.status("ssh-b1").state is BindingState.READY
    assert cache.transient_failure_count("ssh-b1") == 1

    again = _compose(registry, cache, executors={"ssh-b1": stub})
    assert len(again) == 1 and is_remote(again[0].root)


# ---------------------------------------------------------------------------
# Named availability regression (c): the debounced probe flips the cache,
# the next send re-admits the remote root, no user action
# ---------------------------------------------------------------------------


def test_blocked_remote_probe_success_readmits_next_send(tmp_path: Path):
    from tldw_chatbook.Tools.remote_binding_status import (
        BindingState,
        RemoteBindingStatusCache,
    )

    cache = RemoteBindingStatusCache()
    _blocked_cache(cache, "ssh-b1")

    def successful_probe():
        cache.record_success(
            "ssh-b1", [["/srv/www", 99, 7, 16877]]
        )

    stub = _StubRemoteExecutor(
        "ssh-b1", _host_key(), cache, probe=successful_probe
    )
    registry = _Registry(_ssh_binding())

    first = _compose(registry, cache, executors={"ssh-b1": stub})
    assert first == ()  # excluded while BLOCKED
    assert stub.probe_runs == 1
    assert cache.status("ssh-b1").state is BindingState.READY

    second = _compose(registry, cache, executors={"ssh-b1": stub})
    assert len(second) == 1
    assert is_remote(second[0].root)
    assert second[0].workspace_executor._inner is stub
    # The debounce window held: this pass did not schedule a second probe.
    assert stub.probe_runs == 1


# ---------------------------------------------------------------------------
# Hot-path rule + cold-start optimism
# ---------------------------------------------------------------------------


def test_composition_spawns_no_subprocess_and_no_network(monkeypatch, tmp_path: Path):
    """Composition reads cached status only: with a colliding laptop dir,
    a cold remote, and a BLOCKED remote (stub probe), no subprocess is
    spawned and no network is touched."""
    import subprocess as subprocess_module

    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    def boom(*_args, **_kwargs):
        raise AssertionError("composition must not spawn processes or connect")

    monkeypatch.setattr(subprocess_module, "Popen", boom)
    monkeypatch.setattr(subprocess_module, "run", boom)

    cache = RemoteBindingStatusCache()
    _blocked_cache(cache, "ssh-b2")
    stub_blocked = _StubRemoteExecutor("ssh-b2", _host_key("ssh-b2"), cache)
    registry = _Registry(
        _local_binding(tmp_path), _ssh_binding(), _ssh_binding("ssh-b2")
    )

    roots = _compose(
        registry,
        cache,
        executors={"ssh-b1": _StubRemoteExecutor("ssh-b1", _host_key(), cache), "ssh-b2": stub_blocked},
    )
    # Cold (unseen) ssh-b1 admitted optimistically; BLOCKED ssh-b2 excluded.
    assert [root.alias for root in roots] == ["folder-1", "ssh-b1"]
    assert is_remote(roots[1].root)
    assert roots[1].workspace_executor is not None
    assert stub_blocked.probe_runs == 1


def test_cold_remote_admitted_with_real_factory_no_subprocess(
    monkeypatch, tmp_path: Path
):
    """The DEFAULT executor factory (real ``for_ssh`` construction) is
    side-effect free: a cold binding admits without any spawn."""
    import subprocess as subprocess_module

    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    def boom(*_args, **_kwargs):
        raise AssertionError("composition must not spawn processes or connect")

    monkeypatch.setattr(subprocess_module, "Popen", boom)
    monkeypatch.setattr(subprocess_module, "run", boom)

    from tldw_chatbook.config import ConsoleSshSettings
    from tldw_chatbook.Tools.remote_workspace_transport import (
        SshMasterManager,
        get_master_manager,
    )

    # This machine's config bootstrap is environmentally broken (see
    # _allow_provider); hand the factory a real manager + shipped-default
    # settings so the REAL for_ssh construction runs to completion.
    monkeypatch.setattr(
        "tldw_chatbook.config.get_console_ssh_settings",
        lambda: ConsoleSshSettings(
            control_persist="10m",
            enable_multiplexing=True,
            connect_timeout_s=3,
            max_concurrent_calls=8,
        ),
    )
    monkeypatch.setattr(
        get_master_manager.__module__ + ".get_master_manager",
        lambda: SshMasterManager(state_dir=tmp_path),
    )

    cache = RemoteBindingStatusCache()
    registry = _Registry(_ssh_binding())
    roots = _compose(registry, cache)  # default (real) factory
    assert len(roots) == 1
    root = roots[0]
    assert isinstance(root.root, RemoteRoot)
    assert root.alias == "ssh-b1"
    assert root.allow_write is True  # from metadata access
    assert root.workspace_executor is not None


# ---------------------------------------------------------------------------
# Degraded selection warning: BLOCKED selected working folder
# ---------------------------------------------------------------------------


def test_degraded_selected_remote_warns_and_proceeds(tmp_path, monkeypatch):
    from loguru import logger as loguru_logger

    from tldw_chatbook.Chat.console_chat_controller import (
        _validate_project_instruction_binding,
    )
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = RemoteBindingStatusCache()
    _blocked_cache(cache, "ssh-b1")
    binding = _ssh_binding()
    selection = _validate_project_instruction_binding(
        _Session("ws"), binding, status_cache=cache
    )
    assert selection is not None and selection.degraded

    registry = _Registry(binding)
    captured: list[str] = []
    sink_id = loguru_logger.add(captured.append, format="{message}", level="WARNING")
    try:
        roots = _compose(registry, cache, project_selection=selection)
    finally:
        loguru_logger.remove(sink_id)
    # No forced re-selection (no raise): the run proceeds without the
    # blocked remote root, and a warning was logged.
    assert roots == ()
    assert any("BLOCKED" in message for message in captured)
    # The send then composes the local provider over the run's scratch
    # (the compose helpers' remote posture): zero path tools advertised,
    # no crash from the RemoteRoot selection.
    provider = _allow_provider(tmp_path, roots, monkeypatch)
    advertised = set(provider._specs)
    assert not any(name.startswith(("fs_", "git_")) for name in advertised)


# ---------------------------------------------------------------------------
# any_write preflight: read-only remote binding
# ---------------------------------------------------------------------------


def test_read_only_remote_drops_mutating_specs(tmp_path, monkeypatch):
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = RemoteBindingStatusCache()
    registry = _Registry(_ssh_binding(access="ro"))
    roots = _compose(
        registry,
        cache,
        executors={"ssh-b1": _StubRemoteExecutor("ssh-b1", _host_key(), cache)},
    )
    assert len(roots) == 1
    assert roots[0].allow_write is False

    provider = _allow_provider(tmp_path, roots, monkeypatch)
    advertised = set(provider._specs)
    assert {"fs_list", "fs_read", "fs_glob", "fs_grep"} <= advertised
    assert not {"fs_write", "fs_edit", "fs_patch"} & advertised


# ---------------------------------------------------------------------------
# git_* on a remote alias: call-time typed error, local aliases unaffected
# ---------------------------------------------------------------------------


def test_git_on_remote_alias_typed_error_local_alias_advertised(tmp_path, monkeypatch):
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = RemoteBindingStatusCache()
    stub = _StubRemoteExecutor("ssh-b1", _host_key(), cache)
    registry = _Registry(_local_binding(tmp_path), _ssh_binding())
    roots = _compose(registry, cache, executors={"ssh-b1": stub})

    provider = _allow_provider(tmp_path, roots, monkeypatch)
    assert "git_status" in provider._specs
    # git tools stay advertised for local aliases on the same mixed run.
    assert set(provider._specs["git_status"].parameters["properties"]["root_alias"]["enum"]) == {
        "folder-1",
        "ssh-b1",
    }

    remote_result = provider._invoke_detailed(
        "local:git_status", {"root_alias": "ssh-b1"}
    ).result
    assert remote_result.ok is False
    assert "git tools not yet supported on SSH bindings" in (remote_result.error or "")
    # The stub transport never saw a git operation.
    assert all(not tool.startswith("git_") for tool, _args in stub.execute_calls)


# ---------------------------------------------------------------------------
# URI-in-path rejection teaches the root_alias convention
# ---------------------------------------------------------------------------


def test_uri_in_path_rejection_teaches_root_alias(tmp_path, monkeypatch):
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = RemoteBindingStatusCache()
    stub = _StubRemoteExecutor("ssh-b1", _host_key(), cache)
    registry = _Registry(_ssh_binding())
    roots = _compose(registry, cache, executors={"ssh-b1": stub})
    provider = _allow_provider(tmp_path, roots, monkeypatch)

    result = provider._invoke_detailed(
        "local:fs_read",
        {"path": "ssh://devbox/srv/www/app/main.py", "root_alias": "ssh-b1"},
    ).result
    assert result.ok is False
    error = result.error or ""
    assert 'root_alias "ssh-b1"' in error
    assert "relative path" in error
    # The URI never reached the transport.
    assert stub.execute_calls == []


# ---------------------------------------------------------------------------
# The wire-dict-to-str composition adapter (its own unit test)
# ---------------------------------------------------------------------------


def test_remote_executor_dispatch_adapter_unit():
    from tldw_chatbook.Chat.console_chat_controller import (
        _RemoteExecutorDispatchAdapter,
    )
    from tldw_chatbook.Tools.remote_workspace_executor import (
        RemoteWorkspaceExecutionError,
    )

    class _Inner:
        def __init__(self):
            self.pinged = 0
            self.probed = 0

        def execute(self, tool, args, *, intent):
            if tool == "bad":
                return {
                    "version": 1,
                    "operation_id": "x",
                    "outcome": "success",
                    "code": "ok",
                    "result": 42,  # non-str result: contract violation
                    "error": None,
                    "elapsed_ms": 1,
                    "truncated": False,
                    "cleanup_proven": True,
                }
            return {
                "version": 1,
                "operation_id": "x",
                "outcome": "success",
                "code": "ok",
                "result": "line1\nline2",
                "error": None,
                "elapsed_ms": 1,
                "truncated": False,
                "cleanup_proven": True,
            }

        def ping(self):
            self.pinged += 1
            return {"identity_chain": []}

        def maybe_schedule_recovery_probe(self):
            self.probed += 1

    inner = _Inner()
    adapter = _RemoteExecutorDispatchAdapter(inner, alias="ssh-b1")

    # str contract: the wire dict's result text is what handlers return.
    assert adapter.execute("fs_read", {"path": "x"}, intent="read") == "line1\nline2"
    # git refusal is call-time and typed.
    with pytest.raises(RemoteWorkspaceExecutionError, match="git tools"):
        adapter.execute("git_status", {}, intent="read")
    # A non-str result refuses loudly rather than stringifying a dict.
    with pytest.raises(RemoteWorkspaceExecutionError):
        adapter.execute("bad", {}, intent="read")
    # passthroughs used by probe scheduling and status entry points.
    assert adapter.ping() == {"identity_chain": []}
    assert inner.pinged == 1
    adapter.maybe_schedule_recovery_probe()
    assert inner.probed == 1


# ---------------------------------------------------------------------------
# Missing remote bindings are excluded with the recovery probe scheduled
# ---------------------------------------------------------------------------


def test_local_root_still_requires_identity_remote_root_may_be_cold(tmp_path):
    """The empty-identity relaxation is REMOTE-only (cold optimistic
    admission, worker pin as the per-call guard): a LOCAL root keeps
    refusing an empty identity chain."""
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot

    with pytest.raises(ValueError, match="root_identity must be non-empty"):
        RunAdmittedWorkspaceRoot(
            workspace_id="ws",
            binding_id="folder-1",
            alias="folder-1",
            root=tmp_path,
            locator_fingerprint="f" * 64,
            root_identity=(),
            allow_write=True,
            guard=lambda _write: True,
        )


def test_missing_remote_excluded_with_probe(tmp_path: Path):
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = RemoteBindingStatusCache()
    cache.record_missing("ssh-b1", "root path absent on host")
    stub = _StubRemoteExecutor("ssh-b1", _host_key(), cache)
    registry = _Registry(_local_binding(tmp_path), _ssh_binding())

    roots = _compose(registry, cache, executors={"ssh-b1": stub})
    assert [root.alias for root in roots] == ["folder-1"]
    assert stub.probe_runs == 1


# ---------------------------------------------------------------------------
# App-cache consistency at the remaining admission surfaces: the chooser
# commit, the authority revalidation, and the snapshot capture must all see
# the SAME process-wide cache the dispatch/preview sites now use (fragmented
# caches revoke chain-bearing selections).
# ---------------------------------------------------------------------------


def _warm_singleton_cache():
    """A cache holding one READY identity for ssh-b1, installed as the
    process singleton (callers restore with
    ``set_remote_binding_status_cache(None)`` in a finally)."""
    from tldw_chatbook.Tools.remote_binding_status import (
        RemoteBindingStatusCache,
        set_remote_binding_status_cache,
    )

    cache = RemoteBindingStatusCache()
    cache.record_success("ssh-b1", [["/srv/www", 99, 7, 16877]])
    set_remote_binding_status_cache(cache)
    return cache


def test_commit_setup_revalidates_against_the_app_cache():
    """The chooser commit re-lists the chosen binding to revalidate it;
    with the warm app cache that re-list must see the SAME identity, or
    a chain-bearing expected option is silently cancelled."""
    import copy

    from tldw_chatbook.Chat.console_chat_controller import (
        commit_project_instruction_setup_decision,
        list_project_instruction_bindings,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Tools.remote_binding_status import (
        set_remote_binding_status_cache,
    )

    cache = _warm_singleton_cache()
    try:
        store = ConsoleChatStore()
        session = store.create_session(
            workspace_id="ws",
            project_instruction_state=ProjectInstructionControlState.new_session(),
        )
        expected_state = copy.deepcopy(session.project_instruction_state)
        registry = _Registry(_ssh_binding())
        # The dispatch recovery path lists options WITH the app cache:
        # the sole option carries the captured identity chain.
        options = list_project_instruction_bindings(
            session, registry, status_cache=cache
        )
        assert len(options) == 1 and options[0].root_identity

        action, selection = commit_project_instruction_setup_decision(
            store=store,
            session_id=session.id,
            registry=registry,
            expected_state=expected_state,
            expected_options=options,
            action="select",
            binding_id="ssh-b1",
        )

        assert action == "select"
        assert selection is not None and selection.root_identity
    finally:
        set_remote_binding_status_cache(None)


def test_authority_snapshot_is_current_with_app_cache_identity():
    """Authority revalidation re-resolves the selection; against the warm
    app cache the identity chain matches instead of reading cold-empty
    and wrongly revoking the selection."""
    from tldw_chatbook.Chat.console_chat_controller import (
        _validate_project_instruction_binding,
        project_instruction_authority_snapshot_is_current,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Tools.remote_binding_status import (
        set_remote_binding_status_cache,
    )

    cache = _warm_singleton_cache()
    try:
        selection = _validate_project_instruction_binding(
            _Session("ws"), _ssh_binding(), status_cache=cache
        )
        assert selection is not None and selection.root_identity

        store = ConsoleChatStore()
        session = store.create_session(
            workspace_id="ws",
            project_instruction_state=ProjectInstructionControlState(
                project_instructions_enabled=True,
                working_folder_binding_id="ssh-b1",
                working_folder_locator_fingerprint=selection.locator_fingerprint,
                project_instruction_notice_key="n" * 64,
            ),
        )

        assert project_instruction_authority_snapshot_is_current(
            session_snapshot=session,
            registry=_Registry(_ssh_binding()),
            expected_selection=selection,
        )
    finally:
        set_remote_binding_status_cache(None)


def test_capture_authority_snapshot_carries_app_cache_identity():
    """The frozen authority snapshot (turn context / UI state) lists
    remote bindings through the app cache: the captured snapshot's
    root_identity is the cached chain, not a cold-empty tuple."""
    from tldw_chatbook.Chat.console_chat_controller import (
        capture_project_instruction_authority,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Tools.remote_binding_status import (
        set_remote_binding_status_cache,
    )

    _cache = _warm_singleton_cache()
    try:
        store = ConsoleChatStore()
        session = store.create_session(
            workspace_id="ws",
            project_instruction_state=ProjectInstructionControlState.new_session(),
        )
        snapshot = capture_project_instruction_authority(session, _Registry(_ssh_binding()))
        assert snapshot.options, "remote binding must appear in the options"
        assert snapshot.options[0].root_identity, (
            "snapshot must carry the app-cache identity chain"
        )
    finally:
        set_remote_binding_status_cache(None)


# ---------------------------------------------------------------------------
# Invoke-level remote CAS through composition (progress carry from Task 17):
# a read through a composition-admitted remote root stamps the ledger from
# the WORKER-REPORTED result text the dispatch adapter returns.
# ---------------------------------------------------------------------------


def test_invoke_level_remote_read_cas_through_composition(tmp_path, monkeypatch):
    import hashlib

    from tldw_chatbook.Agents.local_tool_provider import _remote_ledger_key
    from tldw_chatbook.Agents.run_context import use_run_id
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    body = "REMOTE server bytes\n"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    stub = _StubRemoteExecutor(
        "ssh-b1",
        _host_key(),
        RemoteBindingStatusCache(),
        result_text=f"{body}sha256: {digest}\nsize: {len(body.encode('utf-8'))}",
    )
    cache = RemoteBindingStatusCache()
    registry = _Registry(_ssh_binding())

    roots = _compose(registry, cache, executors={"ssh-b1": stub})
    assert len(roots) == 1 and is_remote(roots[0].root)

    provider = _allow_provider(tmp_path, roots, monkeypatch)
    run = "compose-cas-run"
    with use_run_id(run):
        result = provider.invoke(
            "local:fs_read", {"path": "app/main.py", "root_alias": "ssh-b1"}
        )
    assert result.ok, result.error
    assert "REMOTE server bytes" in result.content

    key = _remote_ledger_key(roots[0].root, "app/main.py", intent="read")
    assert key is not None
    with use_run_id(run):
        stamp = provider._read_ledger.stamp_for(run, key)
    assert stamp is not None, "worker-reported stamps must reach the ledger"
    assert stamp.sha256 == digest


def test_remote_admission_fingerprints_recorded_destination_not_alias_text():
    """The consented destination (``ssh -G`` at add time) is the authority
    identity: a renamed alias for the same destination keeps it, and the same
    alias recorded against another destination does not share it."""
    from tldw_chatbook.Chat.console_chat_controller import (
        _validate_remote_project_instruction_binding,
    )

    def selection(locator: str, recorded: str):
        binding = SimpleNamespace(
            binding_id="ssh-b1",
            locator=locator,
            metadata={"access": "ro", "canonical_fingerprint": recorded},
        )
        return _validate_remote_project_instruction_binding(binding)

    same_dest = selection("ssh://devbox/srv/www", "a" * 64)
    renamed = selection("ssh://devbox-alias/srv/www", "a" * 64)
    moved = selection("ssh://devbox/srv/www", "b" * 64)
    assert same_dest.locator_fingerprint == renamed.locator_fingerprint
    assert same_dest.locator_fingerprint != moved.locator_fingerprint
