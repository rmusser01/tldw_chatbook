"""Original Console storage admission survives real MCP context and child models."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_native_v2_mcp_events import (
    event_case as _event_case,
)
from Tests.Chat.test_native_v2_mcp_events import (
    hook_file as _hook_file,
)
from Tests.Chat.test_native_v2_mcp_events import (
    mcp_case as _mcp_case,
)
from Tests.Chat.test_native_v2_mcp_events import (
    mcp_handler,
)
from Tests.Chat.test_native_v2_mcp_events import (
    native_console as _native_console,
)
from Tests.Chat.test_native_v2_mcp_events import (
    native_package as _native_package,
)

event_case, hook_file, mcp_case = _event_case, _hook_file, _mcp_case
native_console, native_package = _native_console, _native_package

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


@pytest.mark.parametrize(
    "mutation,workers",
    [
        ("none", 1),
        ("none", 2),
        ("logging_disabled", 1),
        ("workspace", 1),
        ("workspace_distinct_registry", 1),
        ("admission_refused", 1),
        ("workspace_disable", 1),
        ("workspace_switch", 1),
        ("workspace_retarget", 1),
        ("workspace_custom_reader", 1),
        ("unrelated", 2),
        ("mcp_deny", 1),
        ("database", 1),
        # Replaced writer methods (writer, writer_segment_path,
        # writer_write_bytes, writer_manifest, writer_close) were retired:
        # ADR-356 decision 4 (06f8b0bf94) dropped implementation-identity
        # checks; owner decision 2026-11-17 (TASK-41314, TASK-41315).
        ("writer_source", 1),
        ("root", 1),
        ("log_dir", 1),
        ("bound_directory", 1),
        ("late_permission", 1),
    ],
)
async def test_original_console_mcp_context_reaches_child_model(
    mcp_case, hook_file, monkeypatch, tmp_path, mutation, workers
):
    from Tests.Agents.test_agent_service import fence
    from Tests.Agents.test_hook_permissions import _approve
    from Tests.Chat.test_console_hook_preparation_demand import _save_hooks
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookExecutor
    from tldw_chatbook.Agents.mcp_tool_provider import MCPResultPermission
    from tldw_chatbook.Agents.run_context import current_run_id
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_agent_sources import current_agent_sources
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Tools import workspace_file_roots

    case = mcp_case
    rig = case.rig
    workspace_root = tmp_path / "workspace-folder"
    if mutation.startswith("workspace"):
        workspace_root.mkdir()
        rig.registry.add_folder_binding(
            rig.session.workspace_id, workspace_root, allow_write=True
        )
    rig.store.persistence.db.close()
    rig.store.persistence = ChatPersistenceService(
        CharactersRAGDB(tmp_path / "console-chat.sqlite", "source-tree"),
        workspace_registry=rig.registry,
    )
    project_notices = []
    if mutation.startswith("workspace"):

        def approve_project_notice(notice):
            project_notices.append(notice)
            return "proceed"

        rig.controller._confirm_project_instruction_dispatch = approve_project_notice
        if mutation == "workspace_custom_reader":
            authority_reader = rig.controller._hook_authority_values
            monkeypatch.setattr(
                rig.controller,
                "_hook_authority_values",
                lambda sid: authority_reader(sid),
            )
    value = mcp_handler().model_dump()
    value.update(effects=["context"], require_context=True)
    value["input"]["payload"]["structuredContent"]["context"] = [
        {"text": "Original Console child instructions", "lifetime": "runtime"}
    ]
    _save_hooks(
        hook_file,
        {
            "enabled": True,
            "handler": [
                {k: v for k, v in value.items() if v is not None and v not in ({}, ())}
            ],
        },
    )
    permissions = case.runtime.ensure_hook_permissions()
    rig.controller._hook_permissions_accessor = lambda: permissions
    assert _approve(permissions).ready
    monkeypatch.setenv("TLDW_AGENTS_MAX_LIVE_SUBAGENTS", str(workers))
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    global_registry = rig.registry
    if mutation == "workspace_distinct_registry":
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
        from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

        global_registry = LocalWorkspaceRegistryService(
            WorkspaceDB(tmp_path / "global-workspaces.sqlite", client_id="source-tree")
        )
        global_registry.ensure_default_workspace()
    monkeypatch.setattr(
        workspace_file_roots, "_default_registry_instance", global_registry
    )

    admission_errors = []
    if mutation == "admission_refused":
        from tldw_chatbook.Agents.mcp_tool_provider import (
            current_mcp_invocation_policies,
        )
        from tldw_chatbook.Backup_Recovery.storage_admission import _begin_local_pause

        submit_coroutine = asyncio.run_coroutine_threadsafe

        def pause_original_admission(coroutine, loop):
            if (
                not current_mcp_invocation_policies()
                or coroutine.cr_code.co_qualname
                not in {
                    "async_worker_guard.<locals>.decorate.<locals>.call",
                    "MCPToolProvider._execute.<locals>.execute_completed",
                }
            ):
                return submit_coroutine(coroutine, loop)
            pause = _begin_local_pause()
            try:
                future = submit_coroutine(coroutine, loop)
                admission_errors.append(future.exception(timeout=8))
                return future
            finally:
                pause.resume()

        monkeypatch.setattr(
            asyncio, "run_coroutine_threadsafe", pause_original_admission
        )

    if mutation == "logging_disabled":
        monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_ENABLED", "false")
    original_sources, model_sources, model_runs, child_threads = [], [], [], []
    thread_start = threading.Thread.start

    def observed_thread(thread):
        if thread.name.startswith("fleet-"):
            child_threads.append(thread)
        return thread_start(thread)

    monkeypatch.setattr(threading.Thread, "start", observed_thread)

    permission_reads = []
    permission_check = MCPResultPermission.check

    def checked_permission(grant):
        if sources:
            assert not sources[0].lifecycle.checkpoints._condition._is_owned()
        permission_reads.append(grant)
        return permission_check(grant)

    monkeypatch.setattr(MCPResultPermission, "check", checked_permission)
    edited_at = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            model_sources.append(_guard.captured_sources(None))
            model_runs.append(current_run_id())
            if len(self.payloads) == 1:
                source = original_sources[0]
                assert (source.writer.log_dir is None) is (
                    mutation == "logging_disabled"
                )
                if mutation in {"unrelated", "mcp_deny"}:
                    edited_at.append(len(permission_reads))
                    case.service.set_tool_state(
                        "local:fixture",
                        "other" if mutation == "unrelated" else "fixture",
                        "deny",
                    )
                elif mutation == "database":
                    bridge._db = replacement_database
                elif mutation == "writer_source":
                    from types import SimpleNamespace

                    source.writer._console_storage_source = SimpleNamespace(
                        writer_root=lambda _writer: pytest.fail(
                            "custom source callback was used"
                        )
                    )
                elif mutation == "root":
                    source.writer._explicit_root = tmp_path
                elif mutation == "log_dir":
                    source.writer.log_dir = source.root / "different-run"
                elif mutation == "bound_directory":
                    source.writer.log_dir.rename(
                        source.writer.log_dir.with_name("replaced-run")
                    )
                    source.writer.log_dir.mkdir()
                elif mutation in {
                    "workspace_disable",
                    "workspace_switch",
                    "workspace_retarget",
                }:
                    from dataclasses import replace

                    from tldw_chatbook.Chat.console_chat_controller import (
                        list_project_instruction_bindings,
                    )
                    from tldw_chatbook.Chat.console_project_instructions import (
                        ProjectInstructionControlState,
                    )

                    if mutation == "workspace_disable":
                        rig.store.set_session_project_instruction_state(
                            rig.session.id,
                            ProjectInstructionControlState.legacy_disabled(),
                        )
                    else:
                        other = tmp_path / "other-workspace-folder"
                        other.mkdir()
                        if mutation == "workspace_retarget":
                            binding = rig.registry.list_folder_bindings(
                                rig.session.workspace_id
                            )[0]
                            rig.registry.save_runtime_binding(
                                replace(binding, locator=str(other))
                            )
                        else:
                            binding = rig.registry.add_folder_binding(
                                rig.session.workspace_id, other, allow_write=True
                            )
                            selected = next(
                                item
                                for item in list_project_instruction_bindings(
                                    rig.session, rig.registry
                                )
                                if item.binding.binding_id == binding.binding_id
                            )
                            rig.store.set_session_project_instruction_state(
                                rig.session.id,
                                ProjectInstructionControlState(
                                    project_instructions_enabled=True,
                                    working_folder_binding_id=binding.binding_id,
                                    working_folder_locator_fingerprint=selected.locator_fingerprint,
                                ),
                            )
                elif mutation == "late_permission":
                    case.service.permission_store.path = (
                        tmp_path / "late-permissions.json"
                    )
                yield fence(
                    SPAWN_TOOL_NAME, {"task": "Read the original child instructions"}
                )
            elif len(self.payloads) == 2:
                yield "child completed"
            else:
                yield "parent completed"

    gateway = Gateway()
    database = AgentRunsDB(tmp_path / "console-runs.sqlite", client_id="source-tree")
    bridge = ConsoleAgentBridge(
        agent_runs_db=database,
        store=rig.store,
        provider_gateway=gateway,
        skills_service=rig.skills,
        runtime_capacity=case.runtime.execution_capacity,
        get_hooks_v2=case.runtime.get_hooks_v2,
    )
    case.runtime.set_agent_bridge(bridge)
    rig.controller._agent_bridge = bridge
    rig.controller._agent_runtime_enabled = True
    rig.controller.provider_gateway = gateway
    replacement_database = AgentRunsDB(
        tmp_path / "replacement-runs.sqlite", client_id="source-tree"
    )
    sources, jobs = [], []
    start = MCPHookExecutor.start

    def observed(self, *args, **kwargs):
        job = start(self, *args, **kwargs)
        jobs.append(job)
        sources.append(job.native_target.event.source)
        original_sources.append(current_agent_sources())
        return job

    monkeypatch.setattr(MCPHookExecutor, "start", observed)
    try:
        result = await rig.controller.submit_draft(
            "spawn a child", session_id=rig.session.id
        )
        if mutation == "admission_refused":
            from Tests.Chat.test_native_v2_command_events import opaque_busy, until

            assert len(admission_errors) == 1
            assert str(admission_errors[0]) == "agent_activation_required"
            assert not case.service.execution_log.read_recent()
            assert not gateway.payloads
            job = jobs[0]
            assert job.bridges[0].dispatch.state == "not_started"
            assert job.bridges[0].completed.is_set(), (
                "returned guard refusal lost completion"
            )
            await until(lambda: sources[0].custody.retired)
            assert not job.executor.cleanup_pending
            assert not opaque_busy(case.runtime)
            assert case.runtime.execution_capacity.snapshot().tool_workers == 0
            return
        assert result.accepted, result
        if child_threads:
            from Tests.Chat.test_native_v2_command_events import until

            # The ordinary parent may return while its real child survives.
            # Read terminal rows only after the observed worker actually exits.
            await until(lambda: not any(thread.is_alive() for thread in child_threads))
        conversation = rig.session.persisted_conversation_id
        rows = [database.get_run(row["id"]) for row in database.list_runs(conversation)]
        if mutation == "workspace_custom_reader":
            assert project_notices and not gateway.payloads
            assert not any(row["agent_kind"] == "subagent" for row in rows)
            return
        if mutation not in {
            "none",
            "unrelated",
            "logging_disabled",
            "workspace",
            "workspace_distinct_registry",
        }:
            child_ids = {row["id"] for row in rows if row["agent_kind"] == "subagent"}
            assert not child_ids.intersection(model_runs), (rows, model_runs)
            assert 1 <= len(gateway.payloads) <= 2, rows
            assert not any(
                row["agent_kind"] == "subagent" and row["status"] == "done"
                for row in rows
            ), rows
            assert len(case.service.execution_log.read_recent()) == 1
            return
        assert len(gateway.payloads) == 3, rows
        # Admitted sources are a set; nesting order follows the entry path.
        assert [set(seen) for seen in model_sources] == [set(sources[0].sources)] * 3
        assert bool(child_threads) is (workers > 1)
        assert any(
            row["id"] in model_runs for row in rows if row["agent_kind"] == "subagent"
        )
        source = original_sources[0]
        assert ("db.agent_runs", database.db_path) in source.sources
        assert (
            "db.chachanotes.primary",
            rig.store.persistence.db.db_path,
        ) in source.sources
        assert ("agents.history", source.root) in source.sources
        if mutation in {"workspace", "workspace_distinct_registry"}:
            assert len(project_notices) == 1
        assert source.root == (
            workspace_root if mutation.startswith("workspace") else source.scratch.root
        )
        if mutation == "workspace_distinct_registry":
            assert global_registry is not rig.registry
            assert ("db.workspaces", rig.registry.db.db_path) in source.sources
            assert ("db.workspaces", global_registry.db.db_path) in source.sources
        assert ("agents.history", source.writer.log_dir) not in source.sources
        assert all(
            "Original Console child instructions" in str(payload)
            for payload in gateway.payloads
        )
        assert any(
            row["agent_kind"] == "subagent" and row["status"] == "done" for row in rows
        ), rows
        assert any(
            row["agent_kind"] == "primary" and row["status"] == "done" for row in rows
        ), rows
        assert sources and len(case.service.execution_log.read_recent()) == 1
        if mutation == "unrelated":
            assert len(permission_reads) > edited_at[0]
        with pytest.raises((RuntimeError, PermissionError)):
            sources[0].admit()
        assert len(case.service.execution_log.read_recent()) == 1
    finally:
        bridge._db = database
        database.close()
        replacement_database.close()
        if global_registry is not rig.registry:
            global_registry.db.close()


async def test_original_mcp_capture_defers_a_non_stock_catalog(mcp_case):
    """A managed runtime defers a non-stock MCP catalog (TASK-41315).

    The fixture's catalog is not the stock manifest, so the synchronous
    capture pins no definition hash or structured name it could not prove
    (11d71e5db5); the send treats the pending capture as opaque instead.
    """
    case = mcp_case
    configuration = await case.rig.controller.capture_turn_configuration_snapshot(
        case.rig.session.id
    )
    assert configuration.mcp_capture_pending is True
    assert dict(configuration.mcp_definition_maximum) == {}
    assert tuple(configuration.mcp_definition_sources) == ()
    assert configuration.mcp_definition_complete is False


async def test_reordered_admission_is_not_a_source_change(tmp_path):
    """A worker guard nests the same sources in another order (TASK-41315)."""
    from tldw_chatbook.Agents.activation import (
        captured_sources,
        execution,
        selected_sources,
        sources_changed,
    )

    with execution(sources=tuple(reversed(selected_sources(None)))):
        reordered = captured_sources(None)
    with execution():
        assert captured_sources(None) != reordered
        assert not sources_changed(reordered)
        assert sources_changed(reordered[1:])
        assert sources_changed((*reordered, ("agents.history", tmp_path)))


def _refused_source_turn(owner):
    """An ordinary turn whose original sources changed (TASK-41315)."""
    import threading
    from concurrent.futures import Future
    from contextlib import contextmanager
    from types import SimpleNamespace

    from tldw_chatbook.Agents.ordinary_tree import OrdinaryTurn, _Member

    from tldw_chatbook.Agents.activation import OriginalSourceChanged

    class Source:
        @contextmanager
        def activate(self):
            raise OriginalSourceChanged("original agent storage source changed")
            yield

    tree = object.__new__(OrdinaryTurn)
    tree.lock = threading.Lock()
    tree.sealed = tree.primary_finishing = tree.settled = False
    tree.primary_hosts, tree.future_events, tree.source_scopes = set(), {}, 0
    tree.source, tree.stop_notification = Source(), None
    tree.scratch_lease = None
    tree.configuration = SimpleNamespace(scratch_space=None)
    member = _Member(settlement=None, run_id="run", row_started=True)
    member.terminal = ("done", "", None)
    tree.members = {owner: member}
    tree.retired = Future()
    tree.runtime = SimpleNamespace(_wake_first_send=lambda: None)
    return tree, member


class _Owner:
    finished = False
    cleanup_unproven = False

    def finish_root(self):
        self.finished = True

    def mark_cleanup_unproven(self):
        self.cleanup_unproven = True

    def drain_state(self):
        if not self.finished:
            return "live"
        return "unproven" if self.cleanup_unproven else "proven"


async def test_changed_source_at_finish_leaves_the_primary_returned_unproven():
    """A refused original source must not strand the turn (TASK-41315).

    The primary returned, so its host settles, but with nothing proven
    durable: the claim stays held (TASK-40716) and disposal can leave it.
    """
    owner = _Owner()
    tree, member = _refused_source_turn(owner)
    tree.owner = owner

    tree.finish()

    assert member.host_returned and not member.durable
    assert owner.finished
    assert tree.unproven_only() and not tree.retired.done()


async def test_changed_source_at_child_return_leaves_the_child_unproven():
    """A surviving child's return refuses a changed source without a live owner."""
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_native_execution import _finish_execution

    child = _Owner()
    tree, member = _refused_source_turn(child)
    tree.owner = _Owner()
    tree.sealed = True

    with pytest.raises(RuntimeError, match="original agent storage source changed"):
        _finish_execution(SimpleNamespace(_ordinary_tree=tree), child)

    assert member.host_returned and not member.durable
    assert child.finished
    assert tree.unproven_only() and not tree.retired.done()


async def test_refused_owner_whose_row_never_started_stays_unproven():
    """Skipped cleanup must not drain proven, even with no row to read (TASK-41315)."""
    owner = _Owner()
    tree, member = _refused_source_turn(owner)
    member.row_started = False
    tree.sealed = True

    refused = tree.return_host(owner, lambda: pytest.fail("release ran unscoped"))

    assert refused is not None and owner.cleanup_unproven
    assert tree.unproven_only() and not tree.retired.done()


async def test_unrelated_runtime_error_in_host_return_propagates_once():
    """Only the source-activation refusal is absorbed (TASK-41315)."""
    from contextlib import contextmanager

    owner = _Owner()
    tree, member = _refused_source_turn(owner)
    returns = []
    tree.host_returned = returns.append

    @contextmanager
    def lease_gone():
        raise RuntimeError("original ordinary scratch lease retired")
        yield

    tree.source = SimpleNamespace(activate=lease_gone)
    with pytest.raises(RuntimeError, match="scratch lease retired"):
        tree.return_host(owner, lambda: None)

    @contextmanager
    def live():
        yield

    tree.source = SimpleNamespace(activate=live)

    def broken():
        raise RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        tree.return_host(owner, broken)
    assert returns == [] and not owner.finished


async def test_first_send_wait_wakes_when_a_parked_turn_becomes_unproven_only():
    """The give-up needs a wake: a turn can turn unproven after the waiter parks."""
    from tldw_chatbook.Chat.console_received_dispatch import (
        _persist_first_send_when_quiescent,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    owner = _Owner()
    tree, member = _refused_source_turn(owner)
    tree.sealed = True
    member.row_started = False
    member.host_returned = False
    states = {"owner": "live"}
    owner.drain_state = lambda: states["owner"]
    runtime = SimpleNamespace(
        _first_send_wakeup=asyncio.Event(),
        _first_send_loop=asyncio.get_running_loop(),
        _execution_capacity_lock=threading.Lock(),
        _first_send_sources=0,
        _turn_custody={},
        _ordinary_turns={tree},
    )
    runtime._wake_first_send = lambda: ConsoleRuntime._wake_first_send(runtime)
    tree.runtime = runtime
    waiter = asyncio.ensure_future(
        _persist_first_send_when_quiescent(
            runtime, None, None, None, SimpleNamespace(), None
        )
    )
    await asyncio.sleep(0.05)
    assert not waiter.done()  # parked on a live owner

    states["owner"] = "unproven"
    tree.host_returned(owner)

    await asyncio.wait_for(waiter, 5)
