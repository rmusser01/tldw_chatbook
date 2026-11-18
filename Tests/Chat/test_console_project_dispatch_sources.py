"""Original registry, notice and consent checks on local and selected owners."""

import asyncio
import sys
import threading
import time
from dataclasses import replace

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [False, True])
async def test_project_dispatch_keeps_original_registry_and_consent_checks(
    request, tmp_path, selected
):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Agents.project_instruction_resolver import (
        InstructionChainDelivery,
        InstructionSnapshot,
        ProjectInstructionResolver,
    )
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleChatController,
        commit_project_instruction_setup_decision,
        list_project_instruction_bindings,
    )
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_project_dispatch import (
        build_project_dispatch_callbacks,
        project_dispatch_registry,
    )
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="project-registry"
    )
    pending = None
    local_app_context = None
    monitoring_slot = None
    write_code = None
    try:
        backend.start()
        registry = profile.workspace_registry
        workspace = registry.create_workspace(
            workspace_id="project-registry", name="Original project registry"
        )
        root = tmp_path / "project"
        root.mkdir()
        (root / "AGENTS.md").write_text("Consult README.md.\n", encoding="utf-8")
        binding = registry.add_folder_binding(workspace.workspace_id, root)
        session = backend.store.create_session(workspace_id=workspace.workspace_id)
        controller = backend.controller
        if not selected:
            from textual.app import App

            # The local control keeps actual Textual worker-to-App marshaling.
            app = App()
            app.workspace_registry_service = registry
            local_app_context = app.run_test()
            await local_app_context.__aenter__()
            controller = ConsoleChatController(
                store=backend.store,
                provider_gateway=backend.controller.provider_gateway,
            )
            controller.app = app
            controller._confirm_project_instruction_dispatch = (
                backend.runtime._confirm_project_instruction_dispatch
            )
        else:
            assert controller.app is None and backend.runtime._app is None
        assert project_dispatch_registry(controller) is registry
        options = tuple(list_project_instruction_bindings(session, registry))
        action, selection = commit_project_instruction_setup_decision(
            store=backend.store,
            session_id=session.id,
            registry=registry,
            expected_state=session.project_instruction_state,
            expected_options=options,
            action="select",
            binding_id=binding.binding_id,
        )
        assert action == "select" and selection is not None
        assert not selection.allow_write
        candidate = ProjectInstructionResolver().resolve_startup(
            binding_id=binding.binding_id,
            binding_root=root,
            locator_fingerprint=selection.locator_fingerprint,
            max_bytes=8192,
            dispatch_started_wall_ns=time.time_ns(),
        )
        assert candidate.source is not None
        snapshot = InstructionSnapshot(
            candidate.binding_id,
            candidate.binding_root,
            candidate.locator_fingerprint,
            candidate.dispatch_started_wall_ns,
            candidate.source,
            candidate.outcomes,
            InstructionChainDelivery((candidate.source.digest,), candidate.outcomes),
            (),
        )
        _activate, guard, confirm = build_project_dispatch_callbacks(
            controller,
            session_id=session.id,
            registry=project_dispatch_registry(controller),
            project_selection=selection,
            state=session.project_instruction_state,
            resolution=ConsoleProviderResolution(
                "llama_cpp",
                "http://127.0.0.1:1",
                "test",
                True,
                execution_key="llama_cpp",
            ),
        )
        assert guard()
        owner_thread = threading.get_ident()
        write_threads = []
        monitoring = sys.monitoring
        monitoring_slot = next(
            value for value in range(6) if monitoring.get_tool(value) is None
        )
        monitoring.use_tool_id(monitoring_slot, "project-consent-owner")
        write_code = type(backend.store).set_session_project_instruction_state.__code__

        def observe_write(_code, *_args):
            if sys._getframe(1).f_locals.get("self") is backend.store:
                write_threads.append(threading.get_ident())

        monitoring.register_callback(
            monitoring_slot, monitoring.events.PY_START, observe_write
        )
        monitoring.set_local_events(
            monitoring_slot, write_code, monitoring.events.PY_START
        )
        pending = asyncio.create_task(asyncio.to_thread(confirm, snapshot))

        async def wait_notice():
            while not backend.runtime._project_dispatch_decisions:
                if pending.done():
                    pytest.fail(f"Dispatch ended before disclosure: {pending.result()}")
                await asyncio.sleep(0.001)
            return next(iter(backend.runtime._project_dispatch_decisions.values()))

        decision = await asyncio.wait_for(wait_notice(), 10)
        assert decision.notice.relative_source == "AGENTS.md"
        assert session.project_instruction_state.project_instruction_notice_key is None
        backend.runtime.resolve_project_instruction_dispatch(
            decision.decision_id, "proceed"
        )
        assert await asyncio.wait_for(pending, 10) == "proceed"
        assert write_threads == [owner_thread]
        assert session.project_instruction_state.project_instruction_notice_key
        assert guard()

        if selected:
            sources = backend.domain.sources
            backend.domain.sources = replace(sources)
            try:
                with pytest.raises(
                    RecoveryRequired, match="console_snapshot_owner_changed"
                ):
                    project_dispatch_registry(controller)
                with pytest.raises(
                    RecoveryRequired, match="console_snapshot_owner_changed"
                ):
                    guard()
                with pytest.raises(
                    RecoveryRequired, match="console_snapshot_owner_changed"
                ):
                    confirm(snapshot)
            finally:
                backend.domain.sources = sources
        # The original frozen binding guard still revokes even a read->write change.
        registry.save_runtime_binding(
            replace(binding, metadata={**binding.metadata, "access": "rw"})
        )
        assert not guard()
        assert confirm(snapshot) == "cancel"
    finally:
        if pending is not None and not pending.done():
            for decision_id in tuple(backend.runtime._project_dispatch_decisions):
                backend.runtime.resolve_project_instruction_dispatch(
                    decision_id, "cancel"
                )
            await asyncio.shield(pending)
        if monitoring_slot is not None:
            sys.monitoring.set_local_events(monitoring_slot, write_code, 0)
            sys.monitoring.register_callback(
                monitoring_slot, sys.monitoring.events.PY_START, None
            )
            sys.monitoring.free_tool_id(monitoring_slot)
        if local_app_context is not None:
            await local_app_context.__aexit__(None, None, None)
        await backend.aclose()
