"""Private-process native wave controls; never loaded by the product."""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import pytest

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

os.environ["TLDW_AGENTS_WORKFLOW_RUNS_ENABLED"] = "true"
os.environ["TLDW_AGENTS_AUTOWAKE_ENABLED"] = "false"
os.environ["TLDW_AGENTS_MAX_LIVE_SUBAGENTS"] = "2"
if sys.argv[1] in {
    "manual-gap",
    "earlier-manual",
    "manual-before-child",
    "ready-queue",
    "window-refused",
}:
    os.environ["OPENAI_API_KEY"] = "private-fixture-key"
network_guard.install()
real_profile_guard.install()


async def exercise():
    import tldw_chatbook.Widgets.compact_model_bar  # noqa: F401 - native constructor dependency
    from Tests.Agents.workflow_support import make_workflow_request, plan_data
    from Tests.Chat.test_automatic_provider_budget import resolution, response
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
    from tldw_chatbook import app as app_source
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan
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
    # This headless runtime fixture has no Textual mount. Only its UI-ready
    # predicate is supplied; the original recovery audit still runs in full.
    app._ui_ready = True
    if sys.argv[1] in {
        "manual-gap",
        "earlier-manual",
        "manual-before-child",
        "ready-queue",
        "window-refused",
    }:
        # Keep Textual's original cross-thread callback/context implementation.
        # Headless presentation uses this real owning loop, without mounting UI.
        app._loop = asyncio.get_running_loop()
    runtime = app.console_runtime
    entered = {name: threading.Event() for name in ("left", "right")}
    release = threading.Event()
    owners = {}
    calls = {name: 0 for name in entered}
    prepared = []
    cancelled = threading.Event()
    cancel_after_baseline = sys.argv[1] == "cancel-after-baseline"
    shutdown_held = sys.argv[1] == "shutdown-held"
    held_tool = sys.argv[1] == "held-tool-return"
    capture_off = sys.argv[1] == "capture-off"
    manual_gap = sys.argv[1] == "manual-gap"
    manual_enabled = sys.argv[1] in {
        "manual-gap",
        "earlier-manual",
        "manual-before-child",
        "ready-queue",
        "window-refused",
    }
    earlier_manual = sys.argv[1] in {"earlier-manual", "ready-queue"}
    ready_queue = sys.argv[1] == "ready-queue"
    manual_before_child = sys.argv[1] == "manual-before-child"
    manual_entered = threading.Event()
    manual_release = threading.Event()
    launch_entered = threading.Event()
    launch_release = threading.Event()
    manual_calls = []
    publication_failure = sys.argv[1] in {
        "publisher-first-fails",
        "publisher-both-fail",
    }
    both_publishers_fail = sys.argv[1] == "publisher-both-fail"
    native_create_failure = sys.argv[1] == "native-create-fails"
    held_publisher = sys.argv[1] == "publisher-return"
    publisher_entered = threading.Event()
    publisher_release = threading.Event()
    tools_returning = {name: threading.Event() for name in entered}
    tool_release = threading.Event()

    def provider(**kwargs):
        messages = kwargs.get("messages", kwargs.get("messages_payload", ()))
        prompt = "\n".join(str(row.get("content", "")) for row in messages)
        if "Manual gap edit" in prompt:
            manual_calls.append(current_execution_owner())
            result = response()
            if earlier_manual or sys.argv[1] == "window-refused":
                manual_entered.set()
                assert manual_release.wait(20)
                return result
            if len(manual_calls) == 1:
                call = {"name": "find_tools", "arguments": {"query": "fs_edit"}}
            elif len(manual_calls) == 2:
                tool_id = prepared[0].kwargs["local_provider"]._tool_id("fs_edit")
                assert tool_id in prompt
                call = {"name": "load_tools", "arguments": {"ids": [tool_id]}}
            elif len(manual_calls) == 3:
                call = {
                    "name": "fs_edit",
                    "arguments": {
                        "path": "right.txt",
                        "root_alias": binding.binding_id,
                        "old_string": "updated right",
                        "new_string": "manual right",
                    },
                }
            else:
                call = None
            if call is not None:
                result["choices"][0]["message"]["content"] = (
                    "```tool_call\n" + json.dumps(call) + "\n```"
                )
            return result
        task = next(name for name in entered if f"Inspect {name}.txt" in prompt)
        calls[task] += 1
        owners[task] = current_execution_owner()
        if calls[task] == 1:
            entered[task].set()
            assert release.wait(15), "other native task did not enter"
        result = response()
        if held_tool and calls[task] > 1:
            result["choices"][0]["message"]["content"] = (
                "Native read timed out; stopped without further changes."
            )
        elif calls[task] <= 2:
            arguments = {
                "path": "right.txt" if task == "followup" else f"{task}.txt",
                "root_alias": binding.binding_id,
            }
            if calls[task] == 2:
                arguments.update(
                    old_string="manual right"
                    if task == "followup"
                    else f"original {task}",
                    new_string="after gap right"
                    if task == "followup"
                    else f"updated {task}",
                )
            name = "fs_read" if calls[task] == 1 else "fs_edit"
            result["choices"][0]["message"]["content"] = (
                "```tool_call\n"
                + json.dumps({"name": name, "arguments": arguments})
                + "\n```"
            )
        else:
            result["choices"][0]["message"]["content"] = (
                f"Updated {task}.txt; native task handoff."
            )
        return result

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    try:
        root = Path(os.environ["XDG_DATA_HOME"]).parent / "project"
        root.mkdir()
        if manual_before_child:
            (root / "nested").mkdir()
            alias = root.parent / "alias"
            alias.symlink_to(root, target_is_directory=True)
            unrelated = root.parent / "unrelated"
            unrelated.mkdir()
        for name in entered:
            (root / f"{name}.txt").write_text(f"original {name}\n")
        registry = app.workspace_registry_service
        registry.create_workspace(workspace_id="wave-native", name="Wave native")
        registry.add_folder_binding("wave-native", root, allow_write=True)
        registry.set_change_review_enabled("wave-native", not capture_off)
        store = runtime.ensure_chat_store()
        session = store.create_session(
            workspace_id="wave-native",
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
        selected = resolution(model="test-model")
        service = runtime.ensure_workflow_service()
        controller._fleet_wake.start_recovery()
        assert await controller._fleet_wake.wait_for_recovery()
        consent = app.change_review_consent_service
        if not capture_off:
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
        if manual_enabled:
            from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection

            controller._confirm_project_instruction_dispatch = lambda _notice: "proceed"
            controller.update_provider_selection(
                ConsoleProviderSelection(
                    provider="openai", explicit_model="test-model", streaming=False
                )
            )
            store.set_session_project_instruction_state(
                session.id,
                ProjectInstructionControlState(
                    project_instructions_enabled=True,
                    working_folder_binding_id=binding.binding_id,
                    working_folder_locator_fingerprint=binding.locator_fingerprint,
                ),
            )
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
        original_task = data["phases"][0]["tasks"][0]
        data["phases"][0]["tasks"] = [
            dict(
                original_task,
                id=name,
                instructions=f"Inspect {name}.txt and report the exact result.",
                target=request.targets[0].model_dump(mode="json"),
                footprint=[
                    {"path": f"{name}.txt", "access": "write", "subtree": False}
                ],
                inputs=[f"{name}.txt"],
                outputs=[f"{name}.txt"],
                allowed_tools=["fs_read", "fs_edit"],
            )
            for name in entered
        ]
        if manual_gap:
            data["phases"][0]["tasks"].append(
                dict(
                    data["phases"][0]["tasks"][1],
                    id="followup",
                    instructions="Inspect followup.txt using right.txt and report.",
                    dependencies=["left"],
                )
            )
        db = bridge.runs_db
        with owned_goal_storage(service):
            workflow = db.workflow_runs.create(request, launch_id="native-wave")
            workflow = db.workflow_runs.save_plan(
                workflow.id,
                WorkflowPlan.model_validate_json(json.dumps(data)),
                expected_control_revision=workflow.control_revision,
            )
            attempt = db.automatic_work.prepare_workflow_phase(
                workflow.id,
                revision_id=1,
                phase_id="phase-1",
                owner_id=controller._fleet_wake.runtime_owner_id,
                session_id=session.id,
                command_id="native-wave-approval",
            )
            phase = _accept_workflow_phase(
                controller, attempt.id, session_id=session.id
            )
        assert phase is not None
        assert capture_off or bridge._change_finalization_coordinator is not None
        print("WAVE_NATIVE_ACCEPTED", workflow.id, attempt.id, flush=True)
        for task_id in entered:
            from tldw_chatbook.Agents.agent_models import RunBudget

            prepared.append(
                await bridge._prepare_workflow_reply(
                    phase=phase,
                    task_id=task_id,
                    binding=binding,
                    resolution=selected,
                    should_cancel=cancelled.is_set,
                    **(
                        {
                            "confirm_project_instruction_dispatch": controller._confirm_project_instruction_dispatch
                        }
                        if manual_enabled
                        else {}
                    ),
                    **(
                        {"run_budget": RunBudget(max_tool_call_seconds=0.05)}
                        if held_tool
                        else {}
                    ),
                )
            )

        def grant_original_tools():
            provider = prepared[0].kwargs["local_provider"]
            for name in ("fs_read", "fs_edit"):
                tool = provider.hub_tool_for(name)
                app.unified_mcp_service.set_tool_state(
                    tool.server_key, name, "allow", tool=tool
                )

        await prepared[0]._read(grant_original_tools)
        if publication_failure:
            with owned_goal_storage(service), db.transaction() as conn:
                conn.execute(
                    "CREATE TRIGGER wave_test_publication_failure BEFORE INSERT ON change_snapshots "
                    + (
                        ""
                        if both_publishers_fail
                        else "WHEN COALESCE(NEW.tracking_error,'')='' "
                    )
                    + "BEGIN SELECT RAISE(ABORT, 'native publication storage failure'); END"
                )
        if native_create_failure:
            with owned_goal_storage(service), db.transaction() as conn:
                conn.execute(
                    "CREATE TRIGGER wave_test_run_failure BEFORE INSERT ON agent_runs BEGIN SELECT RAISE(ABORT, 'native creation storage failure'); END"
                )
        from tldw_chatbook.Chat.console_workflow_wave import run_workflow_wave

        if sys.argv[1] == "window-owned-close-unknown":
            import sqlite3

            from tldw_chatbook.DB import private_sqlite
            from tldw_chatbook.DB.store_kernel import SQLiteStoreKernel
            from tldw_chatbook.DB.workflow_runs import _WindowUnpublished

            with owned_goal_storage(service), db.transaction() as conn:
                conn.execute(
                    "CREATE TRIGGER wave_test_window_refusal BEFORE INSERT ON workflow_change_windows "
                    "BEGIN SELECT RAISE(ABORT, 'native refused window insert'); END"
                )
            original_open = db.workflow_runs.open_change_window.__func__.__code__
            original_close = SQLiteStoreKernel.close.__code__
            workspace_db = registry.db
            unpublished = []
            fault = []
            fault_attempts = []
            injected_error = OSError("native owned workspace close refused")

            def fail_original_owned_close(frame, event, argument):
                if event == "return" and frame.f_code is original_open:
                    if type(argument) is _WindowUnpublished:
                        unpublished.append(argument)
                    return
                if (
                    not unpublished
                    or event != "c_call"
                    or getattr(argument, "__name__", None) != "close"
                ):
                    return
                receiver = getattr(argument, "__self__", None)
                current = frame
                while current is not None:
                    if (
                        current.f_code is original_close
                        and current.f_locals.get("self") is workspace_db
                    ):
                        if receiver is not current.f_locals.get("conn"):
                            return
                        lease = private_sqlite._ordinary_connections.get(receiver)
                        participant = getattr(lease, "resource_participant", None)
                        if fault and receiver is not fault[0][0]:
                            return
                        if not fault:
                            fault.append(
                                (
                                    receiver,
                                    vars(workspace_db._thread_local),
                                    lease,
                                    participant,
                                )
                            )
                        fault_attempts.append(threading.get_ident())
                        raise injected_error
                    current = current.f_back

            def rearm_original_close_observer(frame, event, argument):
                # CPython disables a profile callback after it raises. Observe
                # the original catch passively so the same fault also reaches
                # the outer owner's genuine retry, rather than healing it.
                if (
                    frame.f_code is original_close
                    and event == "exception"
                    and argument[1] is injected_error
                ):
                    sys.setprofile(fail_original_owned_close)
                return rearm_original_close_observer

            threading.settrace_all_threads(rearm_original_close_observer)
            threading.setprofile_all_threads(fail_original_owned_close)
            outcome = None
            try:
                await run_workflow_wave(phase, tuple(prepared))
            except BaseException as error:  # noqa: BLE001 - isolated fault receipt records the actual native outcome
                outcome = (type(error).__name__, str(error))
            finally:
                threading.setprofile_all_threads(None)
                threading.settrace_all_threads(None)
            facts = {
                "fault_count": len(fault),
                "fault_attempts": len(fault_attempts),
                "unpublished_count": len(unpublished),
                "waves": len(runtime._workflow_waves),
                "fences": len(runtime.worktree_activity._waves),
                "sources": runtime._first_send_sources,
                "preparations": len(runtime._workflow_preparations),
                "calls": calls,
                "owners": len(owners),
                "claims": len(runtime.worktree_activity._claims),
                "outcome": outcome,
            }
            if fault:
                connection, local_values, lease, participant = fault[0]
                try:
                    sqlite3.Connection.in_transaction.__get__(connection)
                    live = True
                except sqlite3.ProgrammingError:
                    live = False
                facts.update(
                    original_workspace_receiver=True,
                    live_handle=live,
                    cached_original=local_values.get("conn") is connection,
                    retained_lease=lease is not None
                    and private_sqlite._ordinary_connections.get(connection) is lease,
                    retained_participant=participant is not None
                    and connection in participant.connections,
                )
            if runtime._workflow_waves:
                try:
                    await prepared[0].close(_unpublished=unpublished[0])
                except PermissionError as error:
                    facts["repeat_close_refused"] = (
                        str(error) == "unpublished frame close cannot be replayed"
                    )
                except RuntimeError as error:
                    facts["repeat_close_refused"] = (
                        str(error) == "original workflow reader return unproven"
                    )
                else:
                    facts["repeat_close_refused"] = False
            with owned_goal_storage(service), db.connection() as conn:
                facts["windows"] = conn.execute(
                    "SELECT COUNT(*) FROM workflow_change_windows"
                ).fetchone()[0]
                facts["native_runs"] = len(db.list_runs(conversation))
            print("WAVE_OWNED_CLOSE_FACTS " + json.dumps(facts), flush=True)
            # An original failed native close intentionally retains its lease.
            # Isolated process exit is not runtime/creator or physical drain.
            os._exit(0 if len(fault) == len(unpublished) == 1 else 2)

        if sys.argv[1] in {"window-ack-unknown", "window-unwind-unknown"}:
            uncertain_ack = sys.argv[1] == "window-ack-unknown"
            original_open = db.workflow_runs.open_change_window.__func__.__code__
            original_transaction = (
                db.automatic_work.transaction.__func__.__wrapped__.__code__
            )
            observed = []
            if not uncertain_ack:
                with owned_goal_storage(service), db.transaction() as conn:
                    conn.execute(
                        "CREATE TRIGGER wave_test_window_refusal BEFORE INSERT ON workflow_change_windows "
                        "BEGIN SELECT RAISE(ABORT, 'native refused window insert'); END"
                    )

            def unknown_return(frame, event, argument):
                if (
                    uncertain_ack
                    and event == "return"
                    and frame.f_code is original_open
                    or not uncertain_ack
                    and event == "c_return"
                    and frame.f_code is original_transaction
                    and getattr(argument, "__name__", None) == "rollback"
                ):
                    observed.append(event)
                    raise RuntimeError("native window return unconfirmed")

            sys.setprofile(unknown_return)
            try:
                try:
                    await run_workflow_wave(phase, tuple(prepared))
                except RuntimeError as error:
                    assert str(error) == "native window return unconfirmed", error
                else:
                    raise AssertionError("unknown native window return was accepted")
            finally:
                sys.setprofile(None)
            assert observed == (["return"] if uncertain_ack else ["c_return"])
            assert not owners and not any(calls.values())
            with owned_goal_storage(service), db.connection() as conn:
                rows = conn.execute(
                    "SELECT state,baseline,ending FROM workflow_change_windows"
                ).fetchall()
                assert [tuple(row) for row in rows] == (
                    [("active", "pending", "pending")] if uncertain_ack else []
                )
            assert (
                len(runtime._workflow_waves)
                == len(runtime.worktree_activity._waves)
                == 1
            )
            assert runtime._first_send_sources == len(prepared)
            assert [
                state.phase
                for state in bridge._change_finalization_coordinator._states.values()
            ] == ["registering"]
            assert not runtime.worktree_activity._claims
            assert not runtime.execution_capacity.snapshot().executions
            print("WAVE_UNKNOWN_WINDOW_RETURN_RETAINED", sys.argv[1], flush=True)
            # The normal fixture disposer below is not proof of retained-custody drain.
            os._exit(0)

        if sys.argv[1] in {"alias-handler", "alias-handler-after-permission"}:
            from dataclasses import replace

            from tldw_chatbook.Agents import local_tool_dispatch

            provider = prepared[0].kwargs["local_provider"]
            aliases = provider._path_specs_by_alias[binding.binding_id]
            original = aliases["fs_read"]
            escaped = root / "outside-approved-footprint.txt"

            def changed_handler(args):
                print("WAVE_CHANGED_ALIAS_HANDLER_ENTERED", flush=True)
                escaped.write_text("undeclared effect")
                return original.handler(args)

            changed = replace(original, handler=changed_handler)
            after_permission = sys.argv[1] == "alias-handler-after-permission"
            permission_return = provider._verdict_for.__func__.__code__
            handler_dispatch = next(
                code
                for code in local_tool_dispatch._invoke_detailed.__code__.co_consts
                if getattr(code, "co_name", None) == "_invoke_allowed"
            )
            refused = []

            def change_at_original_permission_return(frame, event, result):
                if event != "return" or frame.f_locals.get("self") is not provider:
                    return
                if (
                    frame.f_code is permission_return
                    and frame.f_locals.get("name") == "fs_read"
                ):
                    aliases["fs_read"] = changed
                elif frame.f_code is handler_dispatch and aliases["fs_read"] is changed:
                    refused.append(result)
                    # Restore only after the real invocation has returned. This
                    # fixture mutation must not spoil the original frame's close.
                    aliases["fs_read"] = original

            if after_permission:
                threading.setprofile(change_at_original_permission_return)
            else:
                aliases["fs_read"] = changed
            release.set()
            try:
                try:
                    await run_workflow_wave(phase, tuple(prepared))
                except (RuntimeError, PermissionError):
                    pass
                assert not escaped.exists(), (
                    "changed receiving handler performed undeclared write"
                )
                if after_permission:
                    assert len(refused) == 1 and refused[0].dispatch_started is False
                    assert refused[0].result.ok is False
                    assert calls == {"left": 3, "right": 3}
            finally:
                threading.setprofile(None)
                # Fixture-only mutation is restored for the original frame's close.
                aliases["fs_read"] = original
            print("WAVE_CHANGED_ALIAS_HANDLER_REFUSED", flush=True)
            return

        if sys.argv[1] == "window-refused":
            import sqlite3

            with owned_goal_storage(service), db.transaction() as conn:
                conn.execute(
                    "CREATE TRIGGER wave_test_window_refusal BEFORE INSERT ON workflow_change_windows "
                    "BEGIN SELECT RAISE(ABORT, 'native refused window insert'); END"
                )
            try:
                await run_workflow_wave(phase, tuple(prepared))
            except sqlite3.IntegrityError as error:
                assert "native refused window insert" in str(error)
            else:
                raise AssertionError("refused window insert was accepted")
            assert not owners and not any(calls.values())
            with owned_goal_storage(service), db.transaction() as conn:
                assert not conn.execute(
                    "SELECT 1 FROM workflow_change_windows"
                ).fetchone()
                assert not db.list_runs(conversation)
                conn.execute("DROP TRIGGER wave_test_window_refusal")
            assert not runtime.worktree_activity._claims
            assert not runtime.execution_capacity.snapshot().executions
            assert not runtime._workflow_waves, "known rollback retained an inert wave"
            assert not runtime.worktree_activity._waves
            assert not bridge._change_finalization_coordinator._states
            assert all(item._closed and item._native_closed for item in prepared)
            assert (
                not runtime._workflow_preparations and not runtime._first_send_sources
            )
            manual_release.set()
            sent = await controller.submit_draft(
                "Manual gap edit", session_id=session.id
            )
            assert sent.accepted, sent
            async with asyncio.timeout(15):
                while (
                    session.id in controller._active_stream_tasks
                    or runtime.worktree_activity._claims
                ):
                    await asyncio.sleep(0.01)
            assert manual_calls and all(not owner.child for owner in manual_calls)
            print("WAVE_KNOWN_ROLLBACK_RETIRED_THEN_NATIVE_ORDINARY_RETURN", flush=True)
            return

        if earlier_manual:
            from tldw_chatbook.Agents.agent_service import _WorkflowChildWait

            manual_session = store.create_session(
                workspace_id=session.workspace_id,
                project_instruction_state=session.project_instruction_state,
            )
            store.persist_session_if_needed(manual_session.id)
            from tldw_chatbook import config
            from tldw_chatbook.Chat.console_turn_context import (
                ConsoleTurnCustodyRequest,
            )

            before_configuration = config.current_config_identity()
            store.set_session_draft(manual_session.id, "Manual gap edit")
            configuration = await controller.capture_turn_configuration_snapshot(
                manual_session.id
            )
            turn = runtime.accept_turn(
                ConsoleTurnCustodyRequest(
                    turn_id="wave-earlier-manual",
                    session_id=manual_session.id,
                    draft="Manual gap edit",
                    configuration=configuration,
                )
            )
            manual_task = runtime._turn_custody[turn].task
            queue = controller.prompt_queue_coordinator
            drain_entered, drain_release = asyncio.Event(), asyncio.Event()
            original_drain = queue._drain_waiting
            if ready_queue:

                async def hold_original_drain(*args, **kwargs):
                    drain_entered.set()
                    await drain_release.wait()
                    return await original_drain(*args, **kwargs)

                queue._drain_waiting = hold_original_drain
            async with asyncio.timeout(10):
                while not manual_entered.is_set():
                    if manual_task.done():
                        await manual_task
                    await asyncio.sleep(0.01)
            if ready_queue:
                snapshot = queue.registry.snapshot(manual_session.id)
                queued_configuration = (
                    await controller.capture_turn_configuration_snapshot(
                        manual_session.id
                    )
                )
                queued = queue.admit(
                    manual_session.id,
                    text="Manual gap edit queued",
                    expected_revision=snapshot.revision,
                    custody_request=ConsoleTurnCustodyRequest(
                        turn_id="wave-ready-queued",
                        session_id=manual_session.id,
                        draft="Manual gap edit queued",
                        configuration=queued_configuration,
                    ),
                )
                assert queued.applied
                manual_release.set()
                await asyncio.wait_for(drain_entered.wait(), 15)
                async with asyncio.timeout(10):
                    while runtime.worktree_activity._claims:
                        await asyncio.sleep(0.01)
                assert runtime.worktree_activity._ready_manual
            waiting = await run_workflow_wave(
                phase, tuple(prepared), _retain_waits=True
            )
            assert waiting.closed and waiting.reservation is None
            assert set(waiting.unlaunched) == set(prepared)
            assert all(
                isinstance(error, _WorkflowChildWait)
                for error in waiting.unlaunched.values()
            )
            assert not owners and not any(calls.values())
            with owned_goal_storage(service), db.connection() as conn:
                assert not conn.execute(
                    "SELECT 1 FROM workflow_change_windows"
                ).fetchone()
            manual_release.set()
            drain_release.set()
            assert (await manual_task).accepted
            if ready_queue:
                assert len(manual_calls) == 2
                assert not runtime.worktree_activity._ready_manual
                queue._drain_waiting = original_drain
                print(
                    "WAVE_ORIGINAL_READY_MANUAL_QUEUE_WINS_WITH_NO_PHYSICAL_CLAIM",
                    flush=True,
                )
            assert runtime._first_send_sources == len(prepared)
            assert (
                runtime._first_send_task is not None
                and not runtime._first_send_task.done()
            )
            assert not runtime._first_send_publishing
            assert config.current_config_identity() == before_configuration
            for item in prepared:
                await item.close()
            await asyncio.wait_for(asyncio.shield(runtime._first_send_task), 15)
            assert runtime._first_send_sources == 0
            assert config.current_config_identity() != before_configuration
            print(
                "WAVE_EARLIER_NATIVE_MANUAL_TREE_PREVENTS_ANY_B_ORIGINAL_CONFIG_WAITS_FOR_PREPARATION_RETURN",
                flush=True,
            )
            return

        if manual_before_child:
            from types import CodeType

            from tldw_chatbook.Chat.console_workflow_wave import _WorkflowWave

            launch_code = next(
                code
                for code in _WorkflowWave._launch.__code__.co_consts
                if isinstance(code, CodeType) and code.co_name == "launch"
            )

            def hold_original_launch_entry(frame, event, _arg):
                if event == "call" and frame.f_code is launch_code:
                    launch_entered.set()
                    assert launch_release.wait(40)

            from tldw_chatbook.Agents.execution_capacity import WorkOrigin

            finite_source = await prepared[0]._read(
                lambda: bridge._change_finalization_coordinator._native.capture(
                    bridge,
                    session_id=session.id,
                    conversation_id=conversation,
                    origin=WorkOrigin.MANUAL,
                    roots=(unrelated,),
                )
            )
            threading.setprofile_all_threads(hold_original_launch_entry)

        if held_publisher:
            from tldw_chatbook.DB.workflow_runs import WorkflowRunsStore

            publisher_code = WorkflowRunsStore.publish_change_window.__code__

            def hold_original_publisher_return(frame, event, _arg):
                if event == "return" and frame.f_code is publisher_code:
                    publisher_entered.set()
                    assert publisher_release.wait(20), (
                        "original publisher return not released"
                    )

            threading.setprofile(hold_original_publisher_return)

        if held_tool:
            from tldw_chatbook.Agents.execution_capacity import OwnedOperation

            def hold_original_tool_finish(frame, event, _arg):
                if (
                    event == "return"
                    and frame.f_code is OwnedOperation.finish.__code__
                    and threading.current_thread().name == "tool-fs_read"
                ):
                    operation = frame.f_locals["self"]
                    name = next(
                        name
                        for name, owner in owners.items()
                        if owner is operation.owner
                    )
                    tools_returning[name].set()
                    assert tool_release.wait(20), (
                        "original tool finish was not released"
                    )

            threading.setprofile(hold_original_tool_finish)

        if cancel_after_baseline:
            from tldw_chatbook.Workspaces.change_review_ownership import _SnapshotTicket

            def cancel_on_actual_baseline_return(frame, event, _arg):
                if (
                    event == "return"
                    and frame.f_code is _SnapshotTicket.returned.__code__
                ):
                    ticket = frame.f_locals.get("self")
                    completed = getattr(ticket, "_workflow_completion", None)
                    if (
                        completed
                        and completed[1] == "baseline"
                        and ticket.retired.is_set()
                    ):
                        cancelled.set()

            threading.setprofile_all_threads(cancel_on_actual_baseline_return)
            try:
                await run_workflow_wave(phase, tuple(prepared))
            except PermissionError as error:
                assert str(error) == "workflow native source changed", error
            else:
                raise AssertionError("cancelled original member launched")
            finally:
                threading.setprofile_all_threads(None)
            assert cancelled.is_set() and not owners and not any(calls.values()), (
                cancelled.is_set(),
                owners,
                calls,
            )
            with owned_goal_storage(service), db.connection() as conn:
                assert [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT state,baseline,ending,anchor_attempt_id FROM workflow_change_windows WHERE workflow_id=?",
                        (workflow.id,),
                    )
                ] == [("unused", "completed", "unused", None)]
                assert not db.list_runs(conversation)
            assert all(item._closed for item in prepared)
            assert not runtime._workflow_waves and not runtime._workflow_preparations
            print("WAVE_ACTUAL_B_RETURN_CANCELLED_NO_CHILD_UNUSED", flush=True)
            return

        wave_task = asyncio.create_task(run_workflow_wave(phase, tuple(prepared)))
        if manual_before_child:
            from tldw_chatbook.Chat.attachment_core import PendingAttachment
            from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin

            async with asyncio.timeout(15):
                while not launch_entered.is_set():
                    if wave_task.done():
                        await wave_task
                    await asyncio.sleep(0.01)
            assert not owners and not runtime.worktree_activity._claims
            from tldw_chatbook.Workspaces.worktree_activity import (
                WorktreeFootprint,
                WorktreeScope,
            )

            for path in (root, alias, root / "nested"):
                chain = capture_directory_chain(path)
                footprint = WorktreeFootprint(
                    reads=(WorktreeScope(chain.canonical_root, True, chain),)
                )
                with runtime.worktree_activity.prelaunch(footprint=footprint) as claim:
                    assert claim is None
            with runtime.worktree_activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                assert claim is None
            other_chain = capture_directory_chain(unrelated)
            with runtime.worktree_activity.prelaunch(
                footprint=WorktreeFootprint(
                    writes=(
                        WorktreeScope(other_chain.canonical_root, True, other_chain),
                    )
                )
            ) as claim:
                assert claim is not None
            coordinator = bridge._change_finalization_coordinator
            finite = coordinator.register((unrelated,), native_source=finite_source)
            assert finite is not None
            assert await asyncio.to_thread(coordinator.await_baseline, finite, 10)
            async with asyncio.timeout(10):
                while finite.baseline_completion.value == "pending":
                    await asyncio.sleep(0.01)
            assert finite.baseline_completion.value == "observed"
            assert coordinator.finalize(finite, run_id="", kind="finite_control")
            async with asyncio.timeout(10):
                while finite.completion.value == "pending":
                    await asyncio.sleep(0.01)
            assert finite.completion.value == "published"
            assert not runtime.worktree_activity._claims and not owners
            print(
                "WAVE_CROSS_ROOT_NATIVE_FINITE_B_E_RETURNS_WITH_FOREIGN_FENCE_OPEN",
                flush=True,
            )
            original_head = store._active_leaf_by_session.get(session.id)
            draft = "Manual gap edit"
            store.set_session_draft(session.id, draft)
            attachment = PendingAttachment(
                file_path="original.txt",
                display_name="original.txt",
                file_type="text",
                insert_mode="inline",
                data=b"original attachment",
                attachment_id="held-manual",
            )
            assert store.add_pending_attachment(session.id, attachment)
            refused = await controller.submit_draft(draft, session_id=session.id)
            assert not refused.accepted and not refused.should_clear_draft
            assert refused.origin is ConsoleSubmissionOrigin.MANUAL
            assert store.session_draft(session.id) == draft
            assert store._active_leaf_by_session.get(session.id) == original_head
            assert store.pending_attachments(session.id)[0].data == attachment.data
            assert not manual_calls and not owners
            launch_release.set()
            print(
                "WAVE_FENCE_BEFORE_FIRST_CHILD_REFUSES_NATIVE_MANUAL_WITH_ORIGINAL_INPUT",
                flush=True,
            )
        if native_create_failure:
            from tldw_chatbook.Agents.agent_service import _WorkflowChildLaunchFailed

            try:
                await asyncio.wait_for(asyncio.shield(wave_task), 10)
            except _WorkflowChildLaunchFailed:
                pass
            else:
                raise AssertionError("refused native create was accepted")
            assert not owners and not any(calls.values())
            assert all(item._closed and item._native_closed for item in prepared)
            assert not runtime.worktree_activity._claims and not runtime._workflow_waves
            with owned_goal_storage(service), db.transaction() as conn:
                assert [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT state,snapshot_count FROM workflow_change_windows WHERE workflow_id=?",
                        (workflow.id,),
                    )
                ] == [("unused", 0)]
                assert not db.list_runs(conversation)
                conn.execute("DROP TRIGGER wave_test_run_failure")
            print("WAVE_FAILED_NATIVE_CREATE_ORIGINAL_CLOSE_RETURN_UNUSED", flush=True)
            return
        try:
            async with asyncio.timeout(10):
                while not all(event.is_set() for event in entered.values()):
                    if wave_task.done():
                        await wave_task
                    await asyncio.sleep(0.01)
            assert len(set(owners.values())) == 2
            assert all(owner.child and owner.run_id for owner in owners.values())
            if shutdown_held:
                shutdown = asyncio.create_task(runtime.dispose())
                await asyncio.sleep(0.05)
                assert not shutdown.done()
                assert not runtime.execution_capacity.snapshot().closed
        finally:
            release.set()
        if held_tool:
            async with asyncio.timeout(10):
                while not all(event.is_set() for event in tools_returning.values()):
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.15)
            assert not wave_task.done()
            assert all(
                not item._native_closed and item._close_operation is not None
                for item in prepared
            )
            assert runtime._workflow_waves and runtime._workflow_preparations
            tool_release.set()
        if both_publishers_fail:
            try:
                await wave_task
            except PermissionError as error:
                assert str(error) == "workflow E required publication unresolved", error
            else:
                raise AssertionError("failed required E was accepted")
            assert len(runtime._workflow_waves) == 1
            retained = next(iter(runtime._workflow_waves))
            assert retained.ending_return.done() and not retained.closed
            assert not runtime.worktree_activity._claims
            assert runtime.worktree_activity._waves.get(retained) is retained.fence
            assert all(
                item._closed and item._source_scope is not None for item in prepared
            )
            assert not runtime.execution_capacity.snapshot().executions
            with owned_goal_storage(service), db.transaction() as conn:
                assert [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT state,baseline,ending FROM workflow_change_windows WHERE workflow_id=?",
                        (workflow.id,),
                    )
                ] == [("active", "completed", "pending")]
                assert not conn.execute(
                    "SELECT 1 FROM workflow_change_window_refs"
                ).fetchone()
                assert db.workflow_runs._active(conn, workflow.id)
                conn.execute("DROP TRIGGER wave_test_publication_failure")
            print(
                "WAVE_BOTH_E_WRITES_FAILED_PHYSICAL_RETURN_REQUIRED_CUSTODY_RETAINED",
                flush=True,
            )
            for item in prepared:
                with pytest.raises(RuntimeError, match="configuration return unproven"):
                    await item.close()
            os._exit(0)
        if held_publisher:
            async with asyncio.timeout(40):
                while not publisher_entered.is_set():
                    if wave_task.done():
                        await wave_task
                    await asyncio.sleep(0.01)
            assert not wave_task.done() and runtime._workflow_waves
            assert runtime.worktree_activity._claims
            assert all(
                item._closed and item._source_scope is not None for item in prepared
            )
            with owned_goal_storage(service), db.connection() as conn:
                original_checkpoints = [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT attempt_id,payload_json FROM workflow_checkpoints ORDER BY attempt_id"
                    )
                ]
                assert len(original_checkpoints) == 2
                assert all(
                    json.loads(row[1])["change_window_refs"] == []
                    for row in original_checkpoints
                )
                assert [
                    row[0]
                    for row in conn.execute(
                        "SELECT snapshot_count FROM workflow_change_windows WHERE workflow_id=?",
                        (workflow.id,),
                    )
                ] == [1]
            publisher_release.set()
        window = await wave_task
        if held_publisher:
            with owned_goal_storage(service), db.connection() as conn:
                assert [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT attempt_id,payload_json FROM workflow_checkpoints ORDER BY attempt_id"
                    )
                ] == original_checkpoints
            assert not runtime._workflow_waves and not runtime.worktree_activity._claims
            print(
                "WAVE_PUBLISHER_ACTUAL_RETURN_AFTER_IMMUTABLE_TASK_CHECKPOINTS",
                flush=True,
            )
        if held_tool:
            assert all(
                item._native_closed
                and all(not worker.is_alive() for worker in item._tool_workers)
                for item in prepared
            )
            assert all(
                (root / f"{name}.txt").read_text() == f"original {name}\n"
                for name in entered
            )
            with owned_goal_storage(service), db.connection() as conn:
                assert [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT state,snapshot_count FROM workflow_change_windows WHERE workflow_id=?",
                        (workflow.id,),
                    )
                ] == [("completed", 0)]
                assert not conn.execute(
                    "SELECT 1 FROM workflow_change_window_refs"
                ).fetchone()
            print("WAVE_ORIGINAL_TOOL_FINISH_RETURN_PRECEDES_FRAME_CLOSE", flush=True)
            return
        if shutdown_held:
            await shutdown
            assert calls == {"left": 1, "right": 1}
            with owned_goal_storage(service), db.connection() as conn:
                assert [
                    row[0]
                    for row in conn.execute(
                        "SELECT state FROM workflow_change_windows WHERE workflow_id=?",
                        (workflow.id,),
                    )
                ] == ["completed"]
                assert all(
                    row[0] == "failed"
                    for row in conn.execute(
                        "SELECT status FROM workflow_task_attempts WHERE workflow_id=?",
                        (workflow.id,),
                    )
                )
            print("WAVE_SHUTDOWN_RETAINS_ORIGINAL_MODEL_THROUGH_E", flush=True)
            return
        assert calls == {"left": 3, "right": 3}, calls
        for name in entered:
            assert (root / f"{name}.txt").read_text() == f"updated {name}\n"
        with owned_goal_storage(service), db.connection() as conn:
            windows = conn.execute(
                "SELECT id,state FROM workflow_change_windows WHERE workflow_id=?",
                (workflow.id,),
            ).fetchall()
            assert [tuple(row) for row in windows] == (
                []
                if capture_off
                else [
                    (
                        window.id,
                        "disclosed_failure" if publication_failure else "completed",
                    )
                ]
            )
            assert len(db.list_runs(conversation)) == 2
            if not capture_off:
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM workflow_change_window_refs WHERE window_id=?",
                        (window.id,),
                    ).fetchone()[0]
                    == 1
                )
            if publication_failure:
                assert conn.execute(
                    "SELECT tracking_error FROM change_snapshots"
                ).fetchone()[0]
                conn.execute("DROP TRIGGER wave_test_publication_failure")
                conn.commit()
        print(
            "WAVE_TWO_NATIVE_TASKS_ONE_WINDOW"
            if not capture_off
            else "WAVE_CAPTURE_OFF_ORIGINAL_SOURCE_RETIRED",
            window.id if window else None,
            flush=True,
        )
        if sys.argv[1] == "retention":
            with owned_goal_storage(service):
                assert db.delete_change_snapshots_older_than("9999") == 0
                current = db.workflow_runs.get(workflow.id)
                try:
                    db.workflow_runs.remove_payloads(
                        workflow.id, expected_control_revision=current.control_revision
                    )
                except ValueError as error:
                    assert str(error) == "workflow_has_active_evidence"
                else:
                    raise AssertionError("accepted phase evidence removed")
                assert db.automatic_work.settle_workflow_phase(
                    attempt.id, owner_id=phase.attempt.owner_id, drained=True
                )
                current = db.workflow_runs.set_status(
                    workflow.id,
                    status="stopped",
                    expected_control_revision=current.control_revision,
                )
                db.workflow_runs.remove_payloads(
                    workflow.id, expected_control_revision=current.control_revision
                )
                with db.connection() as conn:
                    assert [
                        tuple(row)
                        for row in conn.execute(
                            "SELECT state,roots_json,snapshot_count,reserved_bytes FROM workflow_change_windows WHERE workflow_id=?",
                            (workflow.id,),
                        )
                    ] == [("removed", "[]", 0, 0)]
                    assert not conn.execute(
                        "SELECT 1 FROM workflow_change_window_refs"
                    ).fetchone()
                assert db.delete_change_snapshots_older_than("9999") == 1
            print(
                "WAVE_REQUIRED_REFS_RETAINED_UNTIL_REAL_SETTLEMENT_REMOVAL", flush=True
            )
        if sys.argv[1] == "complete-archive":
            from Tests.Chat.workflow_wave_archive import capture_profile, durable_facts

            with owned_goal_storage(service), db.connection() as conn:
                facts = durable_facts(conn, workflow.id)
            (Path.home() / "wave.json").write_text(
                json.dumps({"workflow_id": workflow.id, "facts": facts})
            )
            await capture_profile(
                app, Path.home() / "retained-temporary.tldw-backup.zip"
            )
            print("WAVE_NATIVE_COMPLETE_PROFILE_ARCHIVED", flush=True)
        if sys.argv[1] == "archive-controls":
            import sqlite3
            from contextlib import closing

            from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
            from tldw_chatbook.Backup_Recovery.sqlite_validation import (
                validate_candidate,
            )

            candidate = root.parent / "native-wave.db"
            with (
                owned_goal_storage(service),
                db.connection() as conn,
                closing(sqlite3.connect(candidate)) as copy,
            ):
                conn.backup(copy)
            assert (
                validate_candidate(owner(), candidate, threading.Event(), migrate=False)
                == ()
            )
            damage_cases = (
                (
                    "DELETE FROM workflow_change_window_refs",
                    "invalid_workflow_window_refs",
                ),
                (
                    "UPDATE workflow_change_windows SET snapshot_count=0",
                    "invalid_workflow_window_refs",
                ),
                (
                    "UPDATE change_snapshots SET root='/foreign-root'",
                    "invalid_workflow_window_refs",
                ),
                (
                    'UPDATE workflow_change_windows SET roots_json=\'["duplicate","duplicate"]\'',
                    "invalid_workflow_window",
                ),
                (
                    "UPDATE workflow_change_windows SET state='active',ending='pending',snapshot_count=NULL",
                    "invalid_workflow_window_refs",
                ),
            )
            for index, (damage, expected) in enumerate(damage_cases):
                damaged = candidate.with_name(f"damaged-{index}.db")
                with (
                    closing(sqlite3.connect(candidate)) as original,
                    closing(sqlite3.connect(damaged)) as copy,
                ):
                    original.backup(copy)
                    copy.execute(damage)
                    copy.commit()
                before = damaged.read_bytes()
                assert validate_candidate(
                    owner(), damaged, threading.Event(), migrate=False
                ) == (expected,)
                assert damaged.read_bytes() == before
            print("WAVE_NATIVE_ARCHIVE_REQUIRED_REF_ABSENCE_REJECTED", flush=True)
        if manual_gap:
            assert not runtime._workflow_waves and not runtime._workflow_preparations
            sent = await controller.submit_draft(
                "Manual gap edit", session_id=session.id
            )
            assert sent.accepted, sent
            async with asyncio.timeout(15):
                while (
                    session.id in controller._active_stream_tasks
                    or runtime.worktree_activity._claims
                ):
                    await asyncio.sleep(0.01)
            assert manual_calls and (root / "right.txt").read_text() == "manual right\n"
            assert runtime.run_hooks_engine is not None
            entered["followup"] = threading.Event()
            calls["followup"] = 0
            followup = await bridge._prepare_workflow_reply(
                phase=phase,
                task_id="followup",
                binding=binding,
                resolution=selected,
                confirm_project_instruction_dispatch=controller._confirm_project_instruction_dispatch,
            )
            prepared.append(followup)
            second = await run_workflow_wave(phase, (followup,))
            assert second.id != window.id and calls["followup"] == 3
            assert (root / "right.txt").read_text() == "after gap right\n"
            with owned_goal_storage(service), db.connection() as conn:
                snapshots = [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT w.ordinal,s.baseline_sha,s.end_sha FROM workflow_change_windows w JOIN workflow_change_window_refs r ON r.window_id=w.id JOIN change_snapshots s ON s.id=r.snapshot_id WHERE w.workflow_id=? ORDER BY w.ordinal",
                        (workflow.id,),
                    )
                ]
                assert len(snapshots) == 2 and snapshots[0][2] != snapshots[1][1]
            from tldw_chatbook import config
            from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
            from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService

            def read_actual_diff():
                repo = ShadowRepoService(
                    user_data_dir(config.load_settings()) / "change_review"
                ).repo_for_root(root)
                assert (
                    repo.file_bytes(snapshots[1][1], "right.txt") == b"manual right\n"
                )
                assert (
                    repo.file_bytes(snapshots[1][2], "right.txt")
                    == b"after gap right\n"
                )
                diff = repo.diff_text(snapshots[1][1], snapshots[1][2], "right.txt")
                assert "-manual right" in diff and "+after gap right" in diff
                assert "-updated right" not in diff

            await asyncio.to_thread(read_actual_diff)
            with owned_goal_storage(service), db.connection() as conn:
                assert [
                    row[0]
                    for row in conn.execute(
                        "SELECT state FROM workflow_change_windows WHERE workflow_id=?",
                        (workflow.id,),
                    )
                ] == ["completed", "completed"]
            print("WAVE_MANUAL_GAP_ORIGINAL_SEND_THEN_FRESH_B", flush=True)
    finally:
        release.set()
        manual_release.set()
        launch_release.set()
        tool_release.set()
        publisher_release.set()
        threading.setprofile(None)
        if manual_before_child:
            threading.setprofile_all_threads(None)
        for item in prepared:
            await item.close()
        await runtime.dispose()
        assert runtime.execution_capacity.snapshot().executions == ()
        creators.retire()
        print("retired and reopened", flush=True)


with user_fixture_default_owner():
    asyncio.run(exercise())
