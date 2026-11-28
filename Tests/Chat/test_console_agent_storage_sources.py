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
    mcp_case,
    hook_file,
    monkeypatch,
    tmp_path,
    mutation,
    workers,
    owned_console_databases,
):
    from Tests.Agents.test_agent_service import fence
    from Tests.Agents.test_hook_permissions import _approve
    from Tests.Chat.test_console_hook_preparation_demand import _save_hooks
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Agents import scratch_admission
    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Backup_Recovery.participants import (
        native_prepared_scratch,
        native_scratch_writer,
    )
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookExecutor
    from tldw_chatbook.Agents.mcp_tool_provider import MCPResultPermission
    from tldw_chatbook.Agents.run_context import current_run_id
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_agent_sources import current_agent_sources
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.console_execution_owner import ConsoleExecutionOwner
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
    scratch_frames, scratch_cleanup = [], []
    positive_source_tree = False
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
            active_source = original_sources[0]
            if active_source.root == active_source.scratch.root:
                manager, snapshot = active_source.scratch_manager, active_source.scratch
                native = native_scratch_writer(active_source.writer)
                assert native is not None
                assert native[0] is manager and native[1] is snapshot
                frame = scratch_admission._current.get()
                assert frame is not None
                assert frame.writer is active_source.writer
                assert frame.manager is manager and frame.snapshot is snapshot
                frame.check()
                scratch_admission.check_prepared_scratch(
                    manager, snapshot, active_source.session.id
                )
                scratch_frames.append(frame)
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

    retirement_started = retirement_finished = False
    retirement_disposition = None

    async def retire_runtime():
        nonlocal retirement_started, retirement_finished, retirement_disposition
        if retirement_finished:
            return retirement_disposition
        assert not retirement_started, "runtime retirement was already uncertain"
        retirement_started = True
        await case.runtime.close_hooks_v2()
        await case.runtime.dispose()
        assert not case.runtime.hooks_v2_cleanup_pending
        for frame in scratch_frames:
            assert not frame.live
        retained_trees = set()
        for source, manager, snapshot, receipt in scratch_cleanup:
            tree = source.ordinary_tree
            assert tree.source is source
            assert tree.controller is rig.controller and tree.runtime is case.runtime
            assert source.scratch_manager is manager and source.scratch is snapshot
            assert tree.configuration.scratch_space is snapshot
            if positive_source_tree:
                assert tree.retired.done()
            if tree.retired.done():
                tree.retired.result()
                assert tree.settled and tree.scratch_lease is None
                assert receipt.outcome == "deleted"
                assert not manager.is_live(snapshot)
                assert not snapshot.root.exists()
            else:
                assert tree in case.runtime._ordinary_turns
                assert tree.sealed and tree.unproven_only() and not tree.settled
                assert tree.scratch_lease is not None
                with tree.activity._lock:
                    claim = tree.activity._claims.get(tree.claim.id)
                    assert claim is not None and claim.custody is tree.claim
                    assert tree.footprint._conflicts(claim.footprint)
                    assert any(
                        member.owner is tree.owner for member in claim.members.values()
                    )
                with manager._condition:
                    record = manager._matching_record_locked(snapshot)
                    assert record is not None and record.snapshot is snapshot
                    assert record.tombstoned and record.leases > 0
                    assert manager._identity_matches(snapshot)
                assert receipt.outcome is None and snapshot.root.exists()
                assert not manager.is_live(snapshot)
                retained_trees.add(tree)
        with case.runtime._execution_capacity_lock:
            assert set(case.runtime._ordinary_turns) == retained_trees
        if retained_trees:
            execution_owner.require_current(runs_path)
            replacement_owner.require_current(replacement_path)
            retirement_disposition = False
        retirement_finished = True
        return retirement_disposition

    gateway = Gateway()
    runs_path = tmp_path / "console-runs.sqlite"
    execution_owner = ConsoleExecutionOwner.acquire(runs_path)
    # A failed constructor retains its native lease until cleanup is proved.
    database = AgentRunsDB(
        runs_path, client_id="source-tree", execution_owner=execution_owner
    )
    owned_console_databases(
        database, rig.controller, execution_owner=execution_owner, retire=retire_runtime
    )
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
    replacement_path = tmp_path / "replacement-runs.sqlite"
    replacement_owner = ConsoleExecutionOwner.acquire(replacement_path)
    replacement_database = AgentRunsDB(
        replacement_path, client_id="source-tree", execution_owner=replacement_owner
    )
    owned_console_databases(
        replacement_database,
        rig.controller,
        execution_owner=replacement_owner,
        retire=retire_runtime,
    )
    sources, jobs = [], []
    start = MCPHookExecutor.start

    def observed(self, *args, **kwargs):
        job = start(self, *args, **kwargs)
        jobs.append(job)
        sources.append(job.native_target.event.source)
        original_sources.append(current_agent_sources())
        source = original_sources[-1]
        manager, snapshot = source.scratch_manager, source.scratch
        assert native_prepared_scratch(manager, snapshot, source.session.id)
        scratch_cleanup.append(
            (source, manager, snapshot, manager.observe_cleanup(snapshot))
        )
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
        positive_source_tree = True
        if source.root != source.scratch.root:
            assert ("agents.history", source.root) in source.sources
        else:
            assert ("agents.history", source.root) not in source.sources
            assert len(scratch_frames) == 3
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
        await retire_runtime()
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


def _reordered_capture():
    from tldw_chatbook.Agents.activation import (
        captured_sources,
        execution,
        selected_sources,
    )

    with execution(sources=tuple(reversed(selected_sources(None)))):
        return captured_sources(None)


def _legacy_hook_event(sources):
    from tldw_chatbook.Agents.legacy_hook_ownership import _EventSource

    event = object.__new__(_EventSource)
    event.sources, event.current_memory = sources, lambda: True
    return event.check


def _plugin_hook_preparation(sources):
    from tldw_chatbook.Chat.console_plugin_hook_preparation import _Preparation

    preparation = object.__new__(_Preparation)
    preparation.sources, preparation.check_memory = sources, lambda _service: None
    return lambda: preparation.check(None)


def _reference_read(sources):
    from tldw_chatbook.Chat.console_reference_preparation import _ReferenceRead

    read = object.__new__(_ReferenceRead)
    read.sources, read.authority, read.check_memory = sources, None, lambda: None
    return read.check


@pytest.mark.parametrize(
    "site", [_legacy_hook_event, _plugin_hook_preparation, _reference_read]
)
async def test_source_check_accepts_reordered_admission(site):
    """The same sources captured in another order are no change (TASK-41639)."""
    from tldw_chatbook.Agents.activation import captured_sources, execution

    reordered = _reordered_capture()
    with execution():
        assert captured_sources(None) != reordered
        site(reordered)()
        with pytest.raises(RuntimeError, match="source changed"):
            site(reordered[1:])()


async def test_sources_changed_observes_the_named_service(tmp_path):
    """Workflow checks compare against their bridge's sources (TASK-41639)."""
    from tldw_chatbook.Agents.activation import selected_sources, sources_changed

    bridge = SimpleNamespace(run_log_writer=SimpleNamespace(log_dir=tmp_path))
    captured = tuple(reversed(selected_sources(bridge)))
    assert not sources_changed(captured, bridge)
    assert sources_changed(captured)


def _ordered_source_checks(text):
    """Line numbers where ``==``/``!=`` compares a ``captured_sources`` call.

    ``set(...)``/``frozenset(...)`` operands are the order-free form that
    ``sources_changed`` itself uses. A value assigned first and compared on a
    later line is not followed.
    """
    import ast

    tree = ast.parse(text)
    names = {"captured_sources"} | {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name == "captured_sources" and alias.asname
    }

    def ordered(node):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", getattr(func, "attr", None))
            if isinstance(func, ast.Name) and name in {"set", "frozenset"}:
                return False
            if name in names:
                return True
        return any(ordered(child) for child in ast.iter_child_nodes(node))

    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
        and any(ordered(side) for side in (node.left, *node.comparators))
    ]


@pytest.mark.parametrize(
    "text,line",
    [
        ("flag = (\n    captured_sources(bridge)\n    != sources\n)\n", 2),
        ("flag = captured_sources(self.bridge()) != sources\n", 1),
        ("flag = sources != activation.captured_sources(None)\n", 1),
        (
            "from tldw_chatbook.Agents.activation import captured_sources as observe\n"
            "flag = observe(None) != sources\n",
            2,
        ),
        ("flag = tuple(captured_sources(bridge)) != sources\n", 1),
        ("flag = _guard.captured_sources(None) == self.sources\n", 1),
    ],
)
async def test_ordered_source_census_flags_each_form(text, line):
    assert _ordered_source_checks(text) == [line]


async def test_ordered_source_census_passes_the_set_forms():
    text = (
        "flag = sources_changed(sources, bridge)\n"
        "flag = set(captured_sources(None)) != set(sources)\n"
        "sources = captured_sources(bridge)\n"
    )
    assert _ordered_source_checks(text) == []


async def test_no_source_check_compares_captured_sources_in_order():
    """Admitted sources are a set; an ordered check refuses a reordered nest."""
    from pathlib import Path

    import tldw_chatbook

    root = Path(tldw_chatbook.__file__).parent
    hits = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        # Every form the census follows, aliases included, names the function.
        if "captured_sources" in text:
            hits += [
                f"{path.relative_to(root)}:{line}"
                for line in _ordered_source_checks(text)
            ]
    assert not hits, f"use activation.sources_changed(): {hits}"


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


async def test_unrelated_activation_error_settles_unproven_and_propagates():
    """Any entry failure returns the owner unproven once, then propagates (TASK-41315)."""
    from contextlib import contextmanager

    owner = _Owner()
    tree, member = _refused_source_turn(owner)
    tree.sealed = True
    member.row_started = False
    wakes = []
    tree.runtime = SimpleNamespace(_wake_first_send=lambda: wakes.append(1))

    @contextmanager
    def lease_gone():
        raise RuntimeError("original ordinary scratch lease retired")
        yield

    tree.source = SimpleNamespace(activate=lease_gone)
    with pytest.raises(RuntimeError, match="scratch lease retired"):
        tree.return_host(owner, lambda: pytest.fail("release ran unscoped"))

    assert owner.finished and owner.cleanup_unproven and member.host_returned
    assert tree.unproven_only() and not tree.retired.done() and wakes


async def test_body_error_in_host_return_propagates_without_a_second_return():
    from contextlib import contextmanager

    owner = _Owner()
    tree, member = _refused_source_turn(owner)
    returns = []
    tree.host_returned = returns.append

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
