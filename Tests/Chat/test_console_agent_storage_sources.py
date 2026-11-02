"""Original Console storage admission survives real MCP context and child models."""

import asyncio
import threading

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
        ("writer", 1),
        ("writer_segment_path", 1),
        ("writer_write_bytes", 1),
        ("writer_manifest", 1),
        ("writer_close", 1),
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
    edited_at, writer_callbacks = [], []
    sentinel = None

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            nonlocal sentinel
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
                elif mutation == "writer":
                    source.writer.append = lambda **kwargs: None
                elif mutation in {
                    "writer_segment_path",
                    "writer_write_bytes",
                    "writer_manifest",
                    "writer_close",
                }:
                    sentinel = source.root / "visible-user-file.txt"
                    sentinel.write_text("original user file")

                    def changed_callback(*args, **kwargs):
                        writer_callbacks.append(mutation)
                        if mutation == "writer_segment_path":
                            return sentinel
                        sentinel.write_text("changed writer invoked")

                    setattr(
                        source.writer,
                        {
                            "writer_segment_path": "_segment_path",
                            "writer_write_bytes": "_write_bytes",
                            "writer_manifest": "write_manifest",
                            "writer_close": "close",
                        }[mutation],
                        changed_callback,
                    )
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
        if sentinel is not None:
            assert sentinel.read_text() == "original user file"
            assert not writer_callbacks, "changed writer callback was invoked"
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
        assert model_sources == [sources[0].sources] * 3
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
