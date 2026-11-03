"""Private native Task8E controls; no test authority is installed in the product."""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

os.environ["TLDW_AGENTS_WORKFLOW_RUNS_ENABLED"] = "true"
os.environ["TLDW_AGENTS_AUTOWAKE_ENABLED"] = "false"
network_guard.install()
real_profile_guard.install()


async def exercise():
    import tldw_chatbook.Widgets.compact_model_bar  # noqa: F401
    from Tests.Agents.workflow_support import make_workflow_request, plan_data
    from Tests.Chat.test_automatic_provider_budget import resolution, response
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
    from tldw_chatbook import app as app_source
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowDecision, WorkflowPlan
    from tldw_chatbook.Agents.workflow_run_service import _workflow_root_identity
    from tldw_chatbook.Chat.console_chat_controller import (
        capture_run_admitted_workspace_roots,
    )
    from tldw_chatbook.Chat.console_goal_runs import goal_provider_ref
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.console_workflow_runs import _accept_workflow_phase
    from tldw_chatbook.Chat.console_workflow_wave import run_workflow_wave
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.change_review_consent import RootReadinessState

    creators = OriginalPrivateAppCreators(
        app_source, Path(os.environ["XDG_DATA_HOME"]).parent
    )
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    app._ui_ready = True  # Native headless runtime; no mounted presentation claim.
    runtime = app.console_runtime
    mode = sys.argv[1]
    entered, release = threading.Event(), threading.Event()
    calls, prepared = {}, []
    wave_task = None

    def provider(**kwargs):
        prompt = "\n".join(
            str(m.get("content", ""))
            for m in kwargs.get("messages", kwargs.get("messages_payload", ()))
        )
        task = next(
            name for name in ("left", "right") if f"Inspect {name}.txt" in prompt
        )
        calls[task] = calls.get(task, 0) + 1
        assert current_execution_owner() is not None
        if task == "left" and calls[task] == 1:
            entered.set()
            assert release.wait(20), "native control did not return"
        result = response()
        if calls[task] == 1:
            command = {
                "name": "fs_read",
                "arguments": {"path": f"{task}.txt", "root_alias": binding.binding_id},
            }
        elif calls[task] == 2:
            command = {
                "name": "fs_edit",
                "arguments": {
                    "path": f"{task}.txt",
                    "root_alias": binding.binding_id,
                    "old_string": f"original {task}",
                    "new_string": f"updated {task}",
                },
            }
        else:
            command = None
        result["choices"][0]["message"]["content"] = (
            "```tool_call\n" + json.dumps(command) + "\n```"
            if command
            else f"Native {task} result and required handoff."
        )
        return result

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    try:
        root = Path(os.environ["XDG_DATA_HOME"]).parent / "project"
        root.mkdir()
        for name in ("left", "right"):
            (root / f"{name}.txt").write_text(f"original {name}\n")
        registry = app.workspace_registry_service
        registry.create_workspace(workspace_id="pause-native", name="Pause native")
        registry.add_folder_binding("pause-native", root, allow_write=True)
        registry.set_change_review_enabled("pause-native", True)
        store = runtime.ensure_chat_store()
        session = store.create_session(
            workspace_id="pause-native",
            project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
        )
        conversation = store.persist_session_if_needed(session.id)
        controller = runtime.ensure_chat_controller(
            store=store, provider_gateway=gateway, agent_runtime_enabled=True
        )
        bridge = runtime.ensure_agent_bridge(
            store_factory=lambda: store,
            provider_gateway_factory=lambda: gateway,
            skills_service=app.skills_scope_service,
        )
        controller._agent_bridge = bridge
        runtime.ensure_hook_permissions()
        service = runtime.ensure_workflow_service()
        controller._fleet_wake.start_recovery()
        assert await controller._fleet_wake.wait_for_recovery()
        consent = app.change_review_consent_service
        consent.admit_turn(session.workspace_id)
        async with asyncio.timeout(10):
            while not any(
                item.state is RootReadinessState.READY
                for item in consent.status(session.workspace_id).roots
            ):
                await asyncio.sleep(0.01)
        binding = capture_run_admitted_workspace_roots(
            session=session, registry=registry
        )[0]
        selected = resolution(model="test-model")
        request = make_workflow_request(root).model_copy(
            update={
                "conversation_id": conversation,
                "targets": (goal_provider_ref(selected),),
                "binding": make_workflow_request(root).binding.model_copy(
                    update={
                        "workspace_id": session.workspace_id,
                        "binding_id": binding.binding_id,
                    }
                ),
            }
        )
        data = plan_data(root)
        data["request"] = request.model_dump(mode="json")
        data["root_identity"] = _workflow_root_identity(capture_directory_chain(root))
        data["tools"] = ["fs_read", "fs_edit"]
        original = data["phases"][0]["tasks"][0]
        data["phases"][0]["tasks"] = [
            dict(
                original,
                id=name,
                instructions=f"Inspect {name}.txt and report the exact result.",
                target=request.targets[0].model_dump(mode="json"),
                footprint=[
                    {"path": f"{name}.txt", "access": "write", "subtree": False}
                ],
                inputs=[],
                outputs=[f"{name}.txt"],
                allowed_tools=["fs_read", "fs_edit"],
            )
            for name in ("left", "right")
        ]
        db = bridge.runs_db
        with owned_goal_storage(service):
            snapshot = db.workflow_runs.create(request, launch_id="pause-native")
            snapshot = db.workflow_runs.save_plan(
                snapshot.id,
                WorkflowPlan.model_validate_json(json.dumps(data)),
                expected_control_revision=snapshot.control_revision,
            )
            attempt = db.automatic_work.prepare_workflow_phase(
                snapshot.id,
                revision_id=1,
                phase_id="phase-1",
                owner_id=controller._fleet_wake.runtime_owner_id,
                session_id=session.id,
                command_id="initial-native-approval",
            )
            phase = _accept_workflow_phase(
                controller, attempt.id, session_id=session.id
            )
        assert phase is not None
        item = await bridge._prepare_workflow_reply(
            phase=phase,
            task_id="left",
            binding=binding,
            resolution=selected,
            should_cancel=lambda: False,
        )
        prepared.append(item)

        def grant():
            provider = item.kwargs["local_provider"]
            for name in ("fs_read", "fs_edit"):
                tool = provider.hub_tool_for(name)
                app.unified_mcp_service.set_tool_state(
                    tool.server_key, name, "allow", tool=tool
                )

        await item._read(grant)
        if mode in {"prebind-pause", "before-commit"}:
            from tldw_chatbook.Agents.agent_service import AgentService
            from tldw_chatbook.Agents.workflow_task_scope import (
                WorkflowTaskAuthorization,
            )

            code = (
                WorkflowTaskAuthorization.bind_native.__code__
                if mode == "prebind-pause"
                else AgentService._run_owned.__code__
            )

            def hold_native_boundary(frame, event, _argument):
                if event != "call" or frame.f_code is not code:
                    return
                authorization = (
                    frame.f_locals.get("self")
                    if mode == "prebind-pause"
                    else frame.f_locals.get("workflow_task_authorization")
                )
                if authorization is item.authorization:
                    entered.set()
                    assert release.wait(20), "native control did not release boundary"

            threading.setprofile(hold_native_boundary)
        wave_task = asyncio.create_task(run_workflow_wave(phase, (item,)))
        async with asyncio.timeout(15):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        with owned_goal_storage(service), db.connection() as conn:
            assert conn.execute(
                "SELECT state FROM automatic_work_reservations WHERE id=?",
                (item.authorization.claim.reservation_id,),
            ).fetchone()[0] == ("reserved" if mode == "before-commit" else "committed")
        print("E_NATIVE_BOUNDARY_HELD", mode, flush=True)
        if mode in {"prebind-pause", "before-commit"}:
            assert not item.authorization._binding_state and not calls
        decision = WorkflowDecision(
            id="pause-original",
            workflow_id=snapshot.id,
            plan_revision=1,
            control_revision=snapshot.control_revision,
            phase_id="phase-1",
            action="stop" if mode == "stop" else "pause",
            reason="user_control",
        )
        control = getattr(phase, "_control", None)
        assert callable(control), (
            "missing original native Pause/Stop control producer after actual committed child"
        )
        with owned_goal_storage(service):
            print("E_NATIVE_CONTROL_ENTER", flush=True)
            if mode == "passive-pause":
                with db.automatic_work.transaction() as conn:
                    db.workflow_runs._save_decision(conn, decision, status="paused")
                controlled = service.get(snapshot.id)
            else:
                controlled = control(decision)
            print("E_NATIVE_CONTROL_RETURN", flush=True)
        assert controlled.status == ("stopped" if mode == "stop" else "paused")
        if mode == "bound-pause":
            try:
                await service.call(
                    service.settle_execution_phase, phase, outcome="cancelled"
                )
            except ValueError as error:
                assert str(error) == "workflow_original_source_pending"
            else:
                raise AssertionError("held original wave was treated as retired")
        if mode == "live-limit":
            os.environ["TLDW_AGENTS_MAX_WORKFLOW_OUTPUT_TOKENS"] = "0"
        release.set()
        async with asyncio.timeout(20):
            await wave_task
        assert not runtime._workflow_waves and not runtime._workflow_preparations
        with owned_goal_storage(service):
            completed = service.get(snapshot.id)
        if mode in {"stop", "before-commit", "passive-pause", "live-limit"}:
            assert calls.get("left", 0) == (0 if mode == "before-commit" else 1)
            assert (root / "left.txt").read_text() == "original left\n"
        else:
            assert calls["left"] == 3, calls
            assert (root / "left.txt").read_text() == "updated left\n"
            assert completed.results[0].status == "completed"
        print("E_NATIVE_ORIGINAL_CONTINUATION_AND_REQUIRED_WAVE_RETURN", flush=True)
        if mode == "live-limit":
            os.environ.pop("TLDW_AGENTS_MAX_WORKFLOW_OUTPUT_TOKENS")
        if mode in {"remaining", "remaining-cancel"}:
            settle = getattr(service, "settle_execution_phase", None)
            assert callable(settle), "missing exact original drained partial settlement"
            partial = await service.call(settle, phase, outcome="cancelled")
            assert partial.status == "awaiting_resume"
            assert (
                len(partial.results) == 1
                and partial.results[0].attempt_id == item.authorization.claim.id
            )
            with owned_goal_storage(service), db.connection() as conn:
                assert [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT task_id,ordinal,status,retirement FROM workflow_task_attempts WHERE workflow_id=? ORDER BY task_id",
                        (snapshot.id,),
                    )
                ] == [
                    ("left", 1, "completed", "drained"),
                    ("right", 1, "closed", "drained"),
                ]
            reviewed = await service.call(
                service.resume,
                snapshot.id,
                expected_control_revision=partial.control_revision,
                command_id="resume-inert",
            )
            assert reviewed.status == "awaiting_approval"
            decision = await service.call(
                service.prepare_remaining_decision, snapshot.id, phase_id="phase-1"
            )
            assert decision.result_refs == (item.authorization.claim.id,)
            from tldw_chatbook.Chat.console_workflow_runs import (
                _accept_remaining_workflow_phase,
            )

            if mode == "remaining-cancel":

                def cancel_host(frame, event, _arg):
                    if (
                        event == "call"
                        and frame.f_code.co_qualname
                        == "_accept_remaining_workflow_phase.<locals>.handoff"
                    ):
                        assert frame.f_locals["cell"].phase is not None
                        frame.f_locals["issuing"].cancel()

                sys.setprofile(cancel_host)
                try:
                    await asyncio.create_task(
                        _accept_remaining_workflow_phase(
                            controller,
                            decision,
                            command_id="cancelled-remaining",
                            session_id=session.id,
                            binding=binding,
                        )
                    )
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError(
                        "host cancellation delivered execution custody"
                    )
                finally:
                    sys.setprofile(None)
                assert (
                    not controller._preparation_reads and not runtime._preparation_reads
                )
                assert runtime.execution_capacity.snapshot().executions == ()
                with owned_goal_storage(service):
                    cancelled = service.get(snapshot.id)
                assert cancelled.status == "awaiting_resume"
                assert cancelled.results == partial.results
                assert cancelled.accounting.used["generation"] == 2
                assert cancelled.accounting.used["child_launch"] == 1
                assert calls == {"left": 3}
                print("E_NATIVE_ACCEPTED_CELL_CANCELLED_AFTER_HOST_HANDOFF", flush=True)
                return
            next_phase = await _accept_remaining_workflow_phase(
                controller,
                decision,
                command_id="remaining-original",
                session_id=session.id,
                binding=binding,
            )
            assert next_phase is not None and next_phase is not phase
            for changed in (
                decision.model_copy(update={"id": "different-decision"}),
                decision.model_copy(update={"result_refs": ()}),
            ):
                try:
                    await _accept_remaining_workflow_phase(
                        controller,
                        changed,
                        command_id="remaining-original",
                        session_id=session.id,
                        binding=binding,
                    )
                except ValueError as error:
                    assert str(error) == "workflow_command_conflict"
                else:
                    raise AssertionError(
                        "same command accepted a different full decision"
                    )
            assert (
                await _accept_remaining_workflow_phase(
                    controller,
                    decision,
                    command_id="remaining-original",
                    session_id=session.id,
                    binding=binding,
                )
                is None
            )
            next_item = await bridge._prepare_workflow_reply(
                phase=next_phase,
                task_id="right",
                binding=binding,
                resolution=selected,
                should_cancel=lambda: False,
            )
            prepared.append(next_item)
            await run_workflow_wave(next_phase, (next_item,))
            await service.call(
                service.settle_execution_phase, next_phase, outcome="completed"
            )
            quality = await service.call(
                service.prepare_quality_review, snapshot.id, phase_id="phase-1"
            )
            assert quality.evidence.ready and set(quality.result_refs) == {
                item.authorization.claim.id,
                next_item.authorization.claim.id,
            }
            with owned_goal_storage(service):
                final = service.get(snapshot.id)
            assert final.accounting.used["generation"] == 2
            assert final.accounting.used["child_launch"] == 2
            assert calls == {"left": 3, "right": 3}
            with owned_goal_storage(service), db.connection() as conn:
                assert [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT task_id,ordinal,status FROM workflow_task_attempts WHERE workflow_id=? ORDER BY task_id,ordinal",
                        (snapshot.id,),
                    )
                ] == [
                    ("left", 1, "completed"),
                    ("right", 1, "closed"),
                    ("right", 2, "completed"),
                ]
            print(
                "E_NATIVE_RETAINED_SIBLING_EXACT_COMPLEMENT_ACCOUNTING_QUALITY",
                flush=True,
            )
    finally:
        release.set()
        threading.setprofile(None)
        if wave_task is not None:
            await asyncio.gather(wave_task, return_exceptions=True)
        for item in prepared:
            await item.close()
        await runtime.dispose()
        assert runtime.execution_capacity.snapshot().executions == ()
        creators.retire()
        print("retired and reopened", flush=True)


with user_fixture_default_owner():
    asyncio.run(exercise())
