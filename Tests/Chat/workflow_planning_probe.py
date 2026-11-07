"""Private App/Runtime/SQLite planning control, without test-issued custody."""

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
os.environ["OPENAI_API_KEY"] = "private-native-planning-test"
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
    from tldw_chatbook.Agents.workflow_run_service import _workflow_root_identity
    from tldw_chatbook.Agents.workflow_task_scope import current_task_authorization
    from tldw_chatbook.Chat.console_chat_controller import (
        capture_run_admitted_workspace_roots,
    )
    from tldw_chatbook.Chat.console_goal_runs import goal_provider_ref
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

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
    app._ui_ready = True
    runtime = app.console_runtime
    mode = sys.argv[1]
    calls = []
    received_system = []
    required_system = (
        "Use the selected Chat system context exactly: répondre en français."
    )
    repair_instruction = "Keep the completed work; revise the remaining tests to cover Unicode paths.\nPreserve the original objective."
    entered, release = threading.Event(), threading.Event()
    prepared = None
    retained_failure = False
    consent_calls = []
    forbidden_cleanup_calls = []
    compaction_calls = []
    disposal = None
    maintenance_entered, maintenance_release = threading.Event(), threading.Event()
    maintenance_return = {}
    if mode == "maintenance-tail":
        import inspect
        from concurrent.futures.thread import _WorkItem

        from tldw_chatbook.Backup_Recovery import (
            storage_admission as maintenance_storage,
        )
        from tldw_chatbook.Chat.console_trace_maintenance import LegacyTraceMaintenance
        from tldw_chatbook.DB.base_db import run_owned_db_call

        maintenance_code = LegacyTraceMaintenance.run_batch.__code__
        invoke_code = next(
            code
            for code in run_owned_db_call.__code__.co_consts
            if inspect.iscode(code) and code.co_name == "invoke"
        )

        def hold_original_maintenance(frame, event, argument):
            if (
                event != "return"
                or frame.f_code is not maintenance_code
                or maintenance_entered.is_set()
            ):
                return
            instance = frame.f_locals["self"]
            if instance.db is not app.chachanotes_db:
                return
            parent = frame.f_back
            assert (
                parent.f_code is invoke_code
                and parent.f_locals["database"] is instance.db
            )
            callback = parent.f_locals["operation"]
            assert (
                callback.__self__ is instance
                and callback.__func__ is LegacyTraceMaintenance.run_batch
            )
            while parent is not None and parent.f_code is not _WorkItem.run.__code__:
                parent = parent.f_back
            assert parent is not None
            item = parent.f_locals["self"]
            assert type(item) is _WorkItem and item.future.running()
            operation = maintenance_storage._operation_local.operation
            participant = instance.db._maintenance_participant
            task = runtime._legacy_trace_maintenance_task
            assert task is not None and task is maintenance_return["task"]
            with maintenance_storage._lock:
                assert operation in maintenance_storage._operations
                assert operation.participant is participant
                assert (
                    operation.thread is threading.current_thread()
                    and operation.task is None
                )
            maintenance_return.update(
                future=item.future, operation=operation, participant=participant
            )
            print(
                "F_FIX1_ORIGINAL_MAINTENANCE_HOLD",
                json.dumps(
                    {
                        "task": id(task),
                        "callback": id(callback.__func__),
                        "instance": id(instance),
                        "database": id(instance.db),
                        "participant": id(participant),
                        "operation": id(operation),
                        "future": id(item.future),
                        "worker": threading.current_thread().name,
                        "wrapper_code": id(invoke_code),
                        "created_before_planning": True,
                    }
                ),
                flush=True,
            )
            maintenance_entered.set()
            assert maintenance_release.wait(45), (
                "original maintenance return was not released"
            )

        threading.setprofile_all_threads(hold_original_maintenance)

    def provider(**kwargs):
        owner = current_execution_owner()
        assert owner is not None and owner.is_admitted and not owner.child
        assert current_task_authorization() is None
        with owned_goal_storage(service), bridge.runs_db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT state FROM automatic_wake_attempts WHERE id=?",
                    (attempt.id,),
                ).fetchone()[0]
                == "accepted"
            )
        calls.append(owner)
        if (
            mode
            in {
                "executor-replaced",
                "reader-replaced",
                "executor-hash-replaced",
                "executor-resolver-replaced",
            }
            and len(calls) == 1
        ):
            import inspect

            from tldw_chatbook.Tools import local_tool_impls

            def changed_receiver(*_args, **_kwargs):
                (root / "changed-receiver-wrote.txt").write_text(
                    "forbidden planning effect"
                )
                raise AssertionError("changed receiver invoked")

            if mode in {"executor-replaced", "executor-hash-replaced"}:
                _, native_provider = (
                    prepared.native.service.registry.resolve_owner_for_name("fs_list")
                )
                handler = native_provider._path_specs_by_alias[binding.alias][
                    "fs_list"
                ].handler
                executor = inspect.getclosurevars(handler).nonlocals[
                    "workspace_executor"
                ]
                if mode == "executor-hash-replaced":
                    type(executor).__hash__ = changed_receiver
                else:
                    executor.execute = changed_receiver
            elif mode == "executor-resolver-replaced":
                from tldw_chatbook.Tools import workspace_tool_executor

                original_resolver = workspace_tool_executor.resolve_workspace_path

                def changed_resolver(*_args, **_kwargs):
                    (root / "changed-receiver-wrote.txt").write_text(
                        "forbidden planning effect"
                    )
                    print("F_CHANGED_EXECUTOR_RESOLVER_ENTERED", flush=True)
                    return original_resolver(*_args, **_kwargs)

                workspace_tool_executor.resolve_workspace_path = changed_resolver
            else:
                local_tool_impls._read_bounded_bytes = changed_receiver
        if mode == "system-context":
            received_system.append(
                required_system in json.dumps(kwargs, ensure_ascii=False)
            )
        if mode == "direct-guard":
            registry = prepared.native.service.registry
            denied = registry.invoke_by_name(
                "fs_write", {"path": "forbidden.txt", "content": "must not write"}
            )
            assert denied.error
            assert not (root / "forbidden.txt").exists()
        if mode == "instructions":
            assert "ORIGINAL_PROJECT_GUIDANCE" in json.dumps(kwargs)
            assert len(consent_calls) == 1
        if mode in {"repair", "repair-instruction"} and len(calls) == 2:
            seed = json.loads(prepared.planning_seed)[0]["content"]
            assert '"previous_plans": [{' in seed
            assert '"unrun_tasks": ["task-1"]' in seed
            if mode == "repair-instruction":
                assert json.dumps(repair_instruction, ensure_ascii=False) in seed
                assert request.objective in seed
                assert any(
                    row.get("role") == "user" and row.get("content") == seed
                    for row in kwargs["messages_payload"]
                ), "exact repair input was not delivered to the native provider"
                print("F_REPAIR_EXACT_BODY_PROVIDER_RECEIVED", flush=True)
        result = response()
        result["choices"][0]["message"]["content"] = json.dumps(data)
        if mode == "wrapper-changed":
            wrapper.write_text("#!/bin/sh\nexit 1\n")
        if mode == "invalid-json":
            result["choices"][0]["message"]["content"] = (
                "```json\n" + json.dumps(data) + "\n```"
            )
        if (
            mode
            in {
                "tools",
                "executor-replaced",
                "reader-replaced",
                "executor-hash-replaced",
                "executor-resolver-replaced",
            }
            and len(calls) == 1
        ):
            result["choices"][0]["message"] = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "native-list",
                        "type": "function",
                        "function": {
                            "name": "fs_list",
                            "arguments": json.dumps({"path": "."}),
                        },
                    },
                    {
                        "id": "native-read",
                        "type": "function",
                        "function": {
                            "name": "fs_read",
                            "arguments": json.dumps({"path": "source.py"}),
                        },
                    },
                    {
                        "id": "native-calc",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"expression": "2 * 3"}),
                        },
                    },
                ],
            }
        if mode == "tools" and len(calls) == 2:
            assert "original source" in json.dumps(kwargs), (
                "native bounded fs_read result missing"
            )
        if mode == "response-close-failure":

            class Response:
                def __iter__(self):
                    yield result

                def close(self):
                    raise RuntimeError("original provider response close failed")

            return Response()
        return result

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    try:
        root = Path(os.environ["XDG_DATA_HOME"]).parent / "project"
        root.mkdir()
        (root / "source.py").write_text("original source\n")
        registry = app.workspace_registry_service
        registry.create_workspace(
            workspace_id="planning-native", name="Planning native"
        )
        registry.add_folder_binding("planning-native", root, allow_write=True)
        store = runtime.ensure_chat_store()
        if mode == "maintenance-tail":
            maintenance_return["task"] = runtime._legacy_trace_maintenance_task
            assert maintenance_return["task"] is not None and prepared is None
        session = store.create_session(
            workspace_id="planning-native",
            settings=ConsoleSessionSettings(
                provider="openai",
                model="test-model",
                source="user",
                system_prompt=(
                    required_system
                    if mode == "system-context"
                    else "REQUIRED_SYSTEM_CONTEXT_abcdef0123456789\n" * 40000
                    if mode == "system-oversize"
                    else None
                ),
            ),
            project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
        )
        conversation = (
            store.persist_session_if_needed(session.id)
            if mode != "new-unsaved"
            else None
        )
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
        binding = capture_run_admitted_workspace_roots(
            session=session, registry=registry
        )[0]
        if mode == "instructions":
            (root / "AGENTS.md").write_text(
                "ORIGINAL_PROJECT_GUIDANCE: keep the selected objective.\n"
            )
            store.set_session_project_instruction_state(
                session.id,
                ProjectInstructionControlState(
                    project_instructions_enabled=True,
                    working_folder_binding_id=binding.binding_id,
                    working_folder_locator_fingerprint=binding.locator_fingerprint,
                ),
            )

            def consent(notice):
                assert not runtime.worktree_activity._claims
                assert current_task_authorization() is None
                consent_calls.append(notice)
                return "proceed"

            controller._confirm_project_instruction_dispatch = consent
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
        data["phases"][0]["tasks"][0]["target"] = request.targets[0].model_dump(
            mode="json"
        )
        factory_calls = []
        if mode == "factory":
            from tldw_chatbook.Chat.console_workflow_preparation import (
                current_planning_preparation,
            )
            from tldw_chatbook.Tools.watchlists_tool_service import (
                WatchlistsToolService,
            )

            constructor_code = WatchlistsToolService.__init__.__code__

            def observe_factory(frame, event, _argument):
                if (
                    event == "call"
                    and frame.f_code is constructor_code
                    and current_planning_preparation() is not None
                ):
                    factory_calls.append(True)

            sys.setprofile(observe_factory)
            threading.setprofile(observe_factory)
        from dataclasses import replace

        from tldw_chatbook.Chat.console_workflow_preparation import (
            prepare_workflow_planning,
        )

        if mode in {"wrapper", "wrapper-changed", "wrapper-unavailable"}:
            import hashlib

            wrapper = root / "verify.sh"
            wrapper.write_text("#!/bin/sh\nexit 0\n")
            wrapper.chmod(0o700)
            if mode != "wrapper-unavailable":
                app.app_config.setdefault("console", {})["raw_cli_permitted"] = True
                assert app.raw_cli_runtime.arm().armed
                assert runtime.raw_cli_source.raw is app.raw_cli_runtime
            data["tools"].append("shell_exec")
            task = data["phases"][0]["tasks"][0]
            task["allowed_tools"].append("shell_exec")
            task["execution_mode"] = "exclusive"
            task["footprint"].append(
                {"path": "verify.sh", "access": "read", "subtree": False}
            )
            task["expected_checks"] = ["native-check"]
            task["verifiers"] = [
                {
                    "id": "native-check",
                    "executor_tool_id": "shell_exec",
                    "verifier_path": str(wrapper),
                    "verifier_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                    "input_paths": ["source.py"],
                }
            ]
        if mode == "tools":
            app.unified_mcp_service.set_global_default("allow")
        if mode == "cleanup-source":

            async def unknown_cleanup():
                forbidden_cleanup_calls.append(True)

            gateway.aclose_current_loop = unknown_cleanup
        if mode == "direct-guard":
            controller._micro_compaction_cadence = 1
            compaction_code = (
                controller._maybe_schedule_micro_compaction.__func__.__code__
            )

            def observe_compaction(frame, event, argument):
                if event == "call" and frame.f_code is compaction_code:
                    compaction_calls.append(True)

            sys.setprofile(observe_compaction)
        if mode in {"worker-tail", "shutdown-worker-tail", "worker-failure"}:
            import inspect

            from tldw_chatbook.Agents.activation import worker_guard

            decorator = next(
                code
                for code in worker_guard.__code__.co_consts
                if inspect.iscode(code) and code.co_name == "decorate"
            )
            worker_code = next(
                code
                for code in decorator.co_consts
                if inspect.iscode(code) and code.co_name == "call"
            )

            def hold_original_worker_return(frame, event, argument):
                if (
                    event == "return"
                    and frame.f_code is worker_code
                    and frame.f_locals.get("service") is service.db.automatic_work
                    and not entered.is_set()
                ):
                    entered.set()
                    if mode == "worker-failure":
                        raise RuntimeError("original provider worker return failed")
                    assert release.wait(15), (
                        "original provider worker return was not released"
                    )

            threading.setprofile_all_threads(hold_original_worker_return)
        if mode in {"cancel-cleanup", "maintenance-tail"}:
            from tldw_chatbook.Chat.console_agent_bridge import _ModelCallLifeline

            cleanup_code = _ModelCallLifeline._cleanup.__code__

            def hold_original_cleanup(frame, event, argument):
                if mode == "maintenance-tail":
                    hold_original_maintenance(frame, event, argument)
                if (
                    event == "call"
                    and frame.f_code is cleanup_code
                    and not entered.is_set()
                ):
                    entered.set()
                    assert release.wait(15), "native cleanup was not released"

            if mode == "maintenance-tail":
                threading.setprofile_all_threads(hold_original_cleanup)
            else:
                threading.setprofile(hold_original_cleanup)
        prepare_return = {}
        if mode == "prepare-return-error":
            import inspect
            from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger

            prepare_code = inspect.unwrap(AutomaticWorkLedger.prepare_workflow_planning).__code__

            def interrupt_original_prepare_return(frame, event, result):
                if event == "return" and frame.f_code is prepare_code and result is not None:
                    caller = frame.f_back
                    while caller is not None and caller.f_code.co_name != "commit_request":
                        caller = caller.f_back
                    assert caller is not None
                    prepare_return.update(prepared=caller.f_locals["prepared"], attempt=result)
                    raise RuntimeError("original prepare return interrupted")

            threading.setprofile_all_threads(interrupt_original_prepare_return)
        try:
            prepared = await prepare_workflow_planning(
                controller,
                session_id=session.id,
                binding=binding,
                objective=request.objective,
                policy=replace(request.policy, model_calls=0)
                if mode == "budget-zero"
                else request.policy,
                command_id="initial-native",
            )
        except Exception as error:
            from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

            if mode == "prepare-return-error":
                threading.setprofile_all_threads(None)
                original = prepare_return["prepared"]
                attempt = prepare_return["attempt"]
                assert original._planning_prepare_entered and original._planning_attempt is None
                assert not calls
                assert controller._fleet_wake._automatic_primary_claims[session.id] is original._planning_token
                with owned_goal_storage(service):
                    actual = service.db.automatic_work.read_workflow_attempt(attempt.id, owner_id=attempt.owner_id)
                assert actual.id == attempt.id and actual.state == "prepared"
                retained_failure = True
                print("F_ORIGINAL_PREPARE_RETURN_INTERRUPTED_CUSTODY_RETAINED", flush=True)
                return
            if mode == "system-oversize":
                assert isinstance(
                    error, PermissionError
                ) and "required task seed does not fit" in str(error)
                assert not calls
                assert (
                    controller._fleet_wake._automatic_primary_claims.get(session.id)
                    is None
                )
                return
            if mode == "cleanup-source":
                assert isinstance(
                    error, PermissionError
                ) and "client cleanup changed" in str(error)
                assert not calls and not forbidden_cleanup_calls
                assert (
                    controller._fleet_wake._automatic_primary_claims.get(session.id)
                    is None
                )
                return
            if (
                mode != "budget-zero"
                or not isinstance(error, AutomaticWorkRefused)
                or error.reason != "model_call_budget"
            ):
                raise
            assert not calls
            assert (
                controller._fleet_wake._automatic_primary_claims.get(session.id) is None
            )
            with owned_goal_storage(service), service.db.connection() as conn:
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM automatic_wake_attempts WHERE state='accepted'"
                    ).fetchone()[0]
                    == 0
                )
            return
        attempt = prepared.phase.attempt
        request = prepared.planning_request
        data["request"] = request.model_dump(mode="json")
        data["phases"][0]["tasks"][0]["target"] = request.targets[0].model_dump(
            mode="json"
        )
        assert request.conversation_id == session.persisted_conversation_id
        assert request.conversation_id != session.id
        if mode == "late-source-return":
            source_code = prepared._planning_return_source.__func__.__code__

            def fail_original_source_return(frame, event, argument):
                if (
                    event == "call"
                    and frame.f_code is source_code
                    and frame.f_locals.get("self") is prepared
                    and prepared._planning_terminal_returned
                    and threading.current_thread() is not threading.main_thread()
                    and not entered.is_set()
                ):
                    entered.set()
                    assert release.wait(15)
                    raise RuntimeError("late original tracked source return failed")

            threading.setprofile_all_threads(fail_original_source_return)
            driver = asyncio.create_task(prepared.run_planning())
            assert await asyncio.to_thread(entered.wait, 15)
            driver.cancel()
            release.set()
            try:
                await driver
            except (asyncio.CancelledError, RuntimeError):
                threading.setprofile_all_threads(None)
                assert prepared.primary_owner.is_admitted, (
                    "cancelled caller hid original producer failure and retired root"
                )
                assert not prepared._planning_settled
                assert (
                    controller._fleet_wake._automatic_primary_claims.get(session.id)
                    is prepared._planning_token
                )
                assert prepared.primary_claim.id in runtime.worktree_activity._claims
                with owned_goal_storage(service):
                    saved = service.get(attempt.workflow_id)
                assert saved.plans and saved.accounting.used["generation"] == 1
                retained_failure = True
                raise RuntimeError("F_REQUIRED_CUSTODY_RETAINED_" + mode) from None
            raise AssertionError("late tracked failure returned native completion")
        if mode == "unused":
            await prepared.close()
            assert not calls and prepared._closed
            assert (
                controller._fleet_wake._automatic_primary_claims.get(session.id) is None
            )
            return
        if mode in {
            "required-write",
            "required-return",
            "worker-failure",
            "response-close-failure",
        }:
            if mode == "required-write":
                with owned_goal_storage(service), service.db.connection() as conn:
                    conn.execute(
                        "CREATE TRIGGER planning_required_write_failure BEFORE INSERT ON workflow_plan_revisions BEGIN SELECT RAISE(ABORT, 'required plan write failed'); END"
                    )
            elif mode == "required-return":
                terminal_code = type(
                    service.db.automatic_work
                ).settle_workflow_phase.__code__

                def fail_required_return(frame, event, argument):
                    if (
                        event == "return"
                        and frame.f_code is terminal_code
                        and frame.f_locals.get("attempt_id") == attempt.id
                        and argument is True
                    ):
                        raise RuntimeError("required terminal return unknown")

                threading.setprofile_all_threads(fail_required_return)
            try:
                await prepared.run_planning()
            except Exception:  # noqa: BLE001 - retain original uncertainty or inspect exact native failure.
                threading.setprofile_all_threads(None)
                assert prepared.primary_owner.is_admitted
                if mode == "worker-failure":
                    assert entered.is_set() and prepared._planning_provider_failure
                assert not prepared.primary_owner._root_finished
                assert not prepared._planning_settled
                assert not prepared._planning_terminal_returned
                assert (
                    controller._fleet_wake._automatic_primary_claims.get(session.id)
                    is prepared._planning_token
                )
                assert prepared.primary_claim.id in runtime.worktree_activity._claims
                with owned_goal_storage(service):
                    saved = service.get(attempt.workflow_id)
                assert bool(saved.plans) == (mode == "required-return")
                assert saved.accounting.used["generation"] == 1
                retained_failure = True
                raise RuntimeError("F_REQUIRED_CUSTODY_RETAINED_" + mode) from None
            raise AssertionError("required failure unexpectedly returned completion")
        if mode in {"worker-tail", "shutdown-worker-tail", "prepare-return-error"}:
            driver = asyncio.create_task(prepared.run_planning())
            assert await asyncio.to_thread(entered.wait, 15)
            if mode == "shutdown-worker-tail":
                disposal = asyncio.create_task(runtime.dispose())
                try:
                    await asyncio.wait_for(asyncio.shield(disposal), timeout=0.1)
                except TimeoutError:
                    pass
                else:
                    raise AssertionError(
                        "runtime shutdown returned before provider worker"
                    )
            try:
                await asyncio.wait_for(asyncio.shield(driver), timeout=5)
            except TimeoutError:
                assert prepared.primary_owner.is_admitted
                assert not prepared._planning_terminal_returned
            else:
                raise AssertionError(
                    "planning retired before original provider worker returned"
                )
            finally:
                release.set()
            if mode == "shutdown-worker-tail":
                from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

                original_owner, original_claim = prepared.primary_owner, prepared.primary_claim
                try:
                    await driver
                except AutomaticWorkRefused as error:
                    assert error.reason == "workflow_not_settled"
                else:
                    raise AssertionError("Stop cut invented a complete usage receipt")
                assert prepared._planning_physical_returned and prepared._native_closed
                assert prepared.primary_owner is original_owner and original_owner.is_admitted
                assert prepared.primary_claim is original_claim
                assert original_claim.id in runtime.worktree_activity._claims
                assert not prepared._planning_terminal_returned
                assert prepared._source_scope is not None and prepared._planning_token is not None
                with owned_goal_storage(service), service.db.connection() as conn:
                    saved = service.get(attempt.workflow_id)
                    assert saved.accounting.uncertain and not saved.plans
                    assert saved.accounting.used["generation"] == 1
                    assert conn.execute("SELECT state FROM automatic_work_reservations WHERE chain_id=? AND kind='tokens'", (attempt.chain_id,)).fetchone()[0] == "uncertain"
                retained_failure = True
                print("F_STOP_CUT_ORIGINAL_WORKER_RETURNED_USAGE_UNKNOWN_CUSTODY_RETAINED", flush=True)
                raise RuntimeError("F_REQUIRED_CUSTODY_RETAINED_" + mode)
            if not driver.done():
                await driver
            result = driver.result()
        elif mode in {"cancel-cleanup", "maintenance-tail"}:
            if mode == "maintenance-tail":
                assert await asyncio.to_thread(maintenance_entered.wait, 15)
            driver = asyncio.create_task(prepared.run_planning())
            assert await asyncio.to_thread(entered.wait, 15)
            assert prepared.primary_owner.is_admitted
            driver.cancel()
            release.set()
            try:
                await driver
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("native cancellation was not delivered")
            with owned_goal_storage(service):
                result = service.get(attempt.workflow_id)
        else:
            result = await prepared.run_planning()
        if mode in {"repair", "repair-instruction"}:
            first = result
            prepared = await prepare_workflow_planning(
                controller,
                session_id=session.id,
                binding=binding,
                workflow_id=first.id,
                revision_id=len(first.plans),
                command_id="repair-native",
                repair_instruction=repair_instruction,
            )
            attempt = prepared.phase.attempt
            result = await prepared.run_planning()
            assert len(result.plans) == 2
            if mode == "repair-instruction":
                from tldw_chatbook.Agents.workflow_models import WorkflowPhaseAttempt

                repeated = await prepare_workflow_planning(
                    controller,
                    session_id=session.id,
                    binding=binding,
                    command_id="repair-native",
                    workflow_id=attempt.workflow_id,
                    revision_id=1,
                    repair_instruction=repair_instruction,
                )
                assert (
                    type(repeated) is WorkflowPhaseAttempt and repeated.id == attempt.id
                )
                assert repeated.state == "completed"
                try:
                    await prepare_workflow_planning(
                        controller,
                        session_id=session.id,
                        binding=binding,
                        command_id="repair-native",
                        workflow_id=attempt.workflow_id,
                        revision_id=1,
                        repair_instruction=repair_instruction + " ",
                    )
                except ValueError as exc:
                    assert "workflow_command_conflict" in str(exc)
                else:
                    raise AssertionError("changed repair command was adopted")
                assert len(calls) == 2
        if mode == "system-context":
            assert received_system == [True], (
                "selected native system context was omitted"
            )
        if mode == "system-oversize":
            assert not calls, "oversized mandatory system context reached MODEL"
        if mode in {
            "executor-replaced",
            "reader-replaced",
            "executor-hash-replaced",
            "executor-resolver-replaced",
        }:
            assert len(calls) == 1
            assert not (root / "changed-receiver-wrote.txt").exists(), (
                "changed receiver performed an effect"
            )
            with owned_goal_storage(service):
                receipt = service.db.automatic_work.read_workflow_attempt(
                    attempt.id, owner_id=attempt.owner_id
                )
            assert receipt.state == "completed" and receipt.outcome == "failed"
            assert (
                prepared._planning_physical_returned
                and prepared._planning_terminal_returned
            )
            assert prepared.primary_owner._root_finished
            assert prepared.primary_claim.id not in runtime.worktree_activity._claims
            print("F_CHANGED_RECEIVER_REFUSED_WITH_REQUIRED_RETURN", flush=True)
        assert not compaction_calls, "planning scheduled optional compaction"
        assert not factory_calls, "planning invoked an optional tool-service factory"
        assert len(calls) == (
            0
            if mode == "budget-zero"
            else 2
            if mode in {"tools", "repair", "repair-instruction"}
            else 1
        )
        if mode in {
            "invalid-json",
            "budget-zero",
            "cancel-cleanup",
            "maintenance-tail",
            "shutdown-worker-tail",
            "wrapper-changed",
            "wrapper-unavailable",
            "executor-replaced",
            "reader-replaced",
            "executor-hash-replaced",
            "executor-resolver-replaced",
        }:
            assert not result.plans, "invalid or cancelled planning published a plan"
        else:
            assert result.plans, "native verifier plan was refused"
            assert result.plans[-1].request == request
        assert result.accounting.used["generation"] == (
            2 if mode in {"repair", "repair-instruction"} else 1
        )
        assert result.accounting.used["child_launch"] == 0
        assert prepared._planning_settled
        assert controller._fleet_wake._automatic_primary_claims.get(session.id) is None
        assert (root / "source.py").read_text() == "original source\n"
        print("F_NATIVE_PRIMARY_REQUIRED_PLAN_RETURN", flush=True)
    finally:
        release.set()
        if mode in {"worker-tail", "shutdown-worker-tail"}:
            threading.setprofile_all_threads(None)
        sys.setprofile(None)
        threading.setprofile(None)
        if retained_failure:
            # This isolated negative deliberately retains the actual root/claim.
            # Process exit is not evidence of retirement and no cleanup is forged.
            raise RuntimeError("F_REQUIRED_CUSTODY_RETAINED_" + mode)
        if prepared is not None:
            await prepared.close()
        if mode == "maintenance-tail":
            assert maintenance_entered.is_set()
            original = maintenance_return["operation"]
            disposal = asyncio.create_task(runtime.dispose())
            try:
                try:
                    await asyncio.wait_for(asyncio.shield(disposal), timeout=0.2)
                except TimeoutError:
                    pass
                else:
                    with maintenance_storage._lock:
                        assert original in maintenance_storage._operations
                    print(
                        "F_FIX1_RUNTIME_RETURNED_WITH_ORIGINAL_MAINTENANCE", flush=True
                    )
                    creators.retire()  # Original unchanged assertion at the actual early-return boundary.
                    raise AssertionError(
                        "runtime retired before original maintenance operation returned"
                    )
            finally:
                maintenance_release.set()
                maintenance_return["future"].result(timeout=15)
                await disposal
                with maintenance_storage._lock:
                    assert original not in maintenance_storage._operations
                threading.setprofile_all_threads(None)
                capacity = runtime._execution_capacity
                assert capacity is None or capacity.snapshot().executions == ()
                creators.retire()
                print("F_FIX1_ORIGINAL_MAINTENANCE_PHYSICAL_RETURN", flush=True)
                print("retired and reopened", flush=True)
        else:
            if disposal is None:
                await runtime.dispose()
            else:
                await disposal
            capacity = runtime._execution_capacity
            assert capacity is None or capacity.snapshot().executions == ()
            creators.retire()
            print("retired and reopened", flush=True)


with user_fixture_default_owner():
    asyncio.run(exercise())
