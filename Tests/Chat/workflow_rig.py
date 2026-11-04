"""Native workflow integration owners and independently addressed provider replies."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


class WorkflowProvider:
    """Address by exact task instruction; concurrent tasks never share a cursor."""

    def __init__(self, *, hold=()):
        self.plan = None
        self.calls = []
        self.hold = frozenset(hold)
        self.entered = {}
        self.released = threading.Event()
        self._lock = threading.Lock()
        self.entry_times = {}

    def __call__(self, **kwargs):
        messages = kwargs.get("messages", kwargs.get("messages_payload", ()))
        task = next(
            (
                task
                for phase in self.plan["phases"]
                for task in phase["tasks"]
                if any(row.get("content") == task["instructions"] for row in messages)
            ),
            None,
        )
        key = task["id"] if task else "planning"
        with self._lock:
            self.entry_times[key] = time.monotonic()
            print("G_PROVIDER_ENTRY", key, self.entry_times[key], flush=True)
            self.calls.append((key, kwargs))
            self.entered.setdefault(key, threading.Event()).set()
        if key in self.hold:
            assert self.released.wait(10), f"held {key} provider was not released"
        from Tests.Chat.test_automatic_provider_budget import response

        result = response()
        result["choices"][0]["message"]["content"] = (
            json.dumps(self.plan) if task is None else f"Completed {key}"
        )
        return result


@dataclass
class WorkflowRig:
    workflow: Any
    service: Any
    coordinator: Any
    controller: Any
    store: Any
    gateway: Any
    runtime: Any
    provider: Any
    creators: Any
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    on_status_timeout: Any = None

    def observe(self, snapshot):
        print("G_RUNTIME_SNAPSHOT", snapshot.status, time.monotonic(), flush=True)
        if snapshot.id == getattr(self.workflow, "id", None):
            self.changed.set()

    async def wait_for_status(self, status, *, timeout=5):
        async def wait():
            while True:
                self.changed.clear()
                try:
                    snapshot = await self.coordinator.get(self.workflow.id)
                except ValueError as error:
                    if (
                        type(error) is not ValueError
                        or str(error) != "workflow_snapshot_superseded"
                    ):
                        raise
                    continue
                if snapshot.status == status:
                    return snapshot
                await self.changed.wait()

        print("G_STATUS_WAIT_START", status, time.monotonic(), flush=True)
        try:
            return await asyncio.wait_for(wait(), timeout)
        except TimeoutError:
            print("G_STATUS_WAIT_TIMEOUT", status, time.monotonic(), flush=True)
            run = self.coordinator._runs.get(self.workflow.id)
            if self.on_status_timeout is not None:
                self.on_status_timeout(run)
            if (
                status == "awaiting_result_review"
                and run is not None
                and run.driver is not None
            ):
                await asyncio.wait_for(asyncio.shield(run.driver), 10)
                print(
                    "G_AFTER_ORIGINAL_STATUS_TIMEOUT",
                    (await self.coordinator.get(self.workflow.id)).status,
                    time.monotonic(),
                    flush=True,
                )
            raise

    def provider_release(self):
        self.provider.released.set()

    async def wait_for_provider_entry(self, key="planning"):
        print("G_PROVIDER_WAIT_START", key, time.monotonic(), flush=True)

        async def wait():
            while not self.provider.entered.get(key, threading.Event()).is_set():
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait(), 10)
        finally:
            print(
                "G_PROVIDER_WAIT_END",
                key,
                time.monotonic(),
                self.provider.entry_times,
                flush=True,
            )

    async def close(self):
        self.provider_release()
        await self.runtime.dispose()
        assert self.runtime.execution_capacity.snapshot().executions == ()
        await self.gateway.aclose()
        self.creators.retire()
        print("retired and reopened", flush=True)


async def build_workflow_rig(
    tmp_path,
    monkeypatch,
    provider,
    *,
    target_models=None,
    session_base_url=None,
    change_review=False,
    public_coordinator=True,
):
    """Construct the real private App and keep its original creator retirement."""
    import tldw_chatbook.Widgets.compact_model_bar  # noqa: F401
    from Tests.Agents.workflow_support import make_workflow_request, plan_data
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
    from tldw_chatbook import app as app_source
    from tldw_chatbook.Agents.workflow_run_service import _workflow_root_identity
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

    creators = OriginalPrivateAppCreators(app_source, tmp_path)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    app._ui_ready = True
    runtime = app.console_runtime
    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    try:
        registry = app.workspace_registry_service
        registry.create_workspace(workspace_id="w", name="Workflow")
        root = tmp_path / "project"
        root.mkdir()
        registry.add_folder_binding("w", root, allow_write=True)
        if change_review:
            registry.set_change_review_enabled("w", True)
        store = runtime.ensure_chat_store()
        session = store.create_session(
            session_id="workflow-test-chat",
            workspace_id="w",
            settings=ConsoleSessionSettings(
                provider="openai",
                model="test-model",
                base_url=session_base_url,
                source="user",
            ),
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
        if change_review:
            from tldw_chatbook.Workspaces.change_review_consent import (
                RootReadinessState,
            )

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
        selected = await gateway.resolve_for_send(
            controller._provider_selection_for_session(session.id)
        )
        assert selected.ready
        targets = (goal_provider_ref(selected),)
        if target_models is not None:
            from dataclasses import replace

            selection = controller._provider_selection_for_session(session.id)
            actual = [
                await gateway.resolve_for_send(
                    replace(
                        selection,
                        explicit_model=model,
                        configured_model=None,
                        base_url=selection.base_url if model == "test-model" else None,
                    )
                )
                for model in target_models
            ]
            assert all(item.ready for item in actual)
            targets = tuple(goal_provider_ref(item) for item in actual)
        template = make_workflow_request(root)
        request = template.model_copy(
            update={
                "conversation_id": conversation,
                "targets": targets,
                "binding": template.binding.model_copy(
                    update={"binding_id": binding.binding_id}
                ),
            }
        )
        provider.plan = plan_data(root)
        provider.plan["request"] = request.model_dump(mode="json")
        provider.plan["root_identity"] = _workflow_root_identity(
            capture_directory_chain(root)
        )
        provider.plan["phases"][0]["tasks"][0].update(
            target=request.targets[0].model_dump(mode="json"),
            instructions="Execute task-1",
            allowed_tools=[],
            footprint=[],
            outputs=[],
        )
        coordinator = (
            runtime.ensure_workflow_coordinator() if public_coordinator else None
        )
        rig = WorkflowRig(
            SimpleNamespace(request=request, launch_id="workflow-start"),
            service,
            coordinator,
            controller,
            store,
            gateway,
            runtime,
            provider,
            creators,
        )
        if coordinator is not None:
            coordinator.subscribe(rig.observe)
        return rig
    except BaseException:
        provider.released.set()
        await runtime.dispose()
        await gateway.aclose()
        creators.retire()
        raise
