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
    entered, release = threading.Event(), threading.Event()
    prepared = None
    retained_failure = False
    consent_calls = []
    forbidden_cleanup_calls = []
    compaction_calls = []
    disposal = None

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
        if mode == "repair" and len(calls) == 2:
            seed = json.loads(prepared.planning_seed)[0]["content"]
            assert '"previous_plans": [{' in seed
            assert '"unrun_tasks": ["task-1"]' in seed
        result = response()
        result["choices"][0]["message"]["content"] = json.dumps(data)
        if mode == "wrapper-changed":
            wrapper.write_text("#!/bin/sh\nexit 1\n")
        if mode == "invalid-json":
            result["choices"][0]["message"]["content"] = (
                "```json\n" + json.dumps(data) + "\n```"
            )
        if mode == "tools" and len(calls) == 1:
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
        session = store.create_session(
            workspace_id="planning-native",
            settings=ConsoleSessionSettings(
                provider="openai", model="test-model", source="user"
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
        if mode == "cancel-cleanup":
            from tldw_chatbook.Chat.console_agent_bridge import _ModelCallLifeline

            cleanup_code = _ModelCallLifeline._cleanup.__code__

            def hold_original_cleanup(frame, event, argument):
                if (
                    event == "call"
                    and frame.f_code is cleanup_code
                    and not entered.is_set()
                ):
                    entered.set()
                    assert release.wait(15), "native cleanup was not released"

            threading.setprofile(hold_original_cleanup)
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
        if mode in {"worker-tail", "shutdown-worker-tail"}:
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
                if not driver.done():
                    await driver
            result = driver.result()
        elif mode == "cancel-cleanup":
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
        if mode == "repair":
            first = result
            prepared = await prepare_workflow_planning(
                controller,
                session_id=session.id,
                binding=binding,
                workflow_id=first.id,
                revision_id=len(first.plans),
                command_id="repair-native",
            )
            attempt = prepared.phase.attempt
            result = await prepared.run_planning()
            assert len(result.plans) == 2
        assert not compaction_calls, "planning scheduled optional compaction"
        assert not factory_calls, "planning invoked an optional tool-service factory"
        assert len(calls) == (
            0 if mode == "budget-zero" else 2 if mode in {"tools", "repair"} else 1
        )
        if mode in {
            "invalid-json",
            "budget-zero",
            "cancel-cleanup",
            "shutdown-worker-tail",
            "wrapper-changed",
            "wrapper-unavailable",
        }:
            assert not result.plans, "invalid or cancelled planning published a plan"
        else:
            assert result.plans, "native verifier plan was refused"
            assert result.plans[-1].request == request
        assert result.accounting.used["generation"] == (2 if mode == "repair" else 1)
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
