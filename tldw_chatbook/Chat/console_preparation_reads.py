"""Console source preparation and compatibility names for finite reads."""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager, asynccontextmanager
from typing import TYPE_CHECKING

from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import (
    ConsolePreparationRead as ConsolePreparationRead,
    _HookSourceBinding as _HookSourceBinding,
    _READS_LOCK as _READS_LOCK,
    _current_task as _current_task,
    _observe_locked as _observe_locked,
    _retire as _retire,
    _source_binding as _source_binding,
    await_finite_read as await_finite_read,
    drain_preparation_reads as drain_preparation_reads,
    observe_preparation_reads as observe_preparation_reads,
    preparation_read_for as preparation_read_for,
    preparation_reads_for as preparation_reads_for,
    preparation_source_for as preparation_source_for,
    preparation_tasks_for as preparation_tasks_for,
    run_preparation_read as run_preparation_read,
)

if TYPE_CHECKING:
    from tldw_chatbook.Agents.native_source_storage import (
        NativeAgentSources as _AgentSources,
    )


def current_workflow_preparation(controller=None):
    from tldw_chatbook.Agents.run_context import current_workflow_preparation as current

    return current(controller)


def current_planning_preparation(controller=None):
    from tldw_chatbook.Agents.run_context import current_planning_preparation as current

    return current(controller)


def issuing_agent_sources(controller: object) -> AbstractContextManager[None]:
    from tldw_chatbook.Agents.run_context import issuing_agent_sources as issue

    return issue(controller)


def current_agent_sources(controller: object | None = None) -> _AgentSources | None:
    from tldw_chatbook.Agents.run_context import current_agent_sources as current

    return current(controller)


async def prepare_agent_sources(
    controller: object, session: object, configuration: object
) -> None:
    """Select the original source tuple before the first SessionStart effect."""
    from .console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Utils.platform_files import supports_native_process_groups

    from tldw_chatbook.Agents.run_context import current_agent_source_issuance

    holder = current_agent_source_issuance()
    runtime = getattr(controller, "_hooks_v2_runtime", None)
    bridge = controller._agent_bridge
    if (
        runtime is None
        or type(bridge) is not ConsoleAgentBridge
        or not supports_native_process_groups()
        or not controller._agent_runtime_enabled
        or session.assistant_kind == "character"
    ):
        return
    if holder is None or holder[:2] != [controller, asyncio.current_task()]:
        raise RuntimeError("original agent source issuer unavailable")
    if holder[3] is not None:
        raise RuntimeError("agent storage already selected")
    from tldw_chatbook.Agents.native_source_storage import (
        NativeAgentSources as _AgentSources,
    )

    source = _AgentSources(controller, runtime, bridge, session, configuration)
    from .console_received_turn import received_turn_claim_for

    source.received_claim = received_turn_claim_for(controller.store, session.id)
    await run_preparation_read(
        source.prepare,
        creator=controller,
        session_id=session.id,
        reads=controller._preparation_reads,
        observers=(runtime._preparation_reads,),
        require_current=source.check_memory,
    )
    source.check_memory()
    holder[3] = source
    holder[2].enter_context(source.activate())


class NativePluginHookPreparation:
    """Sticky original Runtime attachment; construction only retains metadata."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.app = runtime._app
        self.capacity = runtime._execution_capacity
        self.activity = runtime._worktree_activity

    def pin_resources(self):
        if self.capacity is None:
            self.capacity = self.runtime._execution_capacity
        if self.activity is None:
            self.activity = self.runtime._worktree_activity

    @asynccontextmanager
    async def capture(self, source, service, configuration, review):
        from tldw_chatbook.Agents.hooks_v2.ownership import join_hook_worker

        from .console_plugin_hook_preparation import _Preparation

        request = _Preparation(self, source, service, configuration, review)
        read = ConsolePreparationRead(
            self.runtime,
            request.invoke,
            source,
            asyncio.current_task(),
            source.session_id,
            asyncio.get_running_loop().create_future(),
        )
        with _READS_LOCK:
            _observe_locked(read, self.runtime._preparation_reads)
            observer = getattr(source.controller, "_preparation_reads", None)
            if observer is not None:
                _observe_locked(read, observer)
        try:
            # Admission and original release callbacks stay off the host loop.
            await join_hook_worker(
                asyncio.create_task(asyncio.to_thread(request.admit)), request.cancel
            )
            read._producer = asyncio.create_task(request.invoke())
            result = await join_hook_worker(read._producer, request.cancel)
            request.check(service)
            yield result, request
            request.check(service)
        finally:
            try:
                await join_hook_worker(asyncio.create_task(asyncio.to_thread(request.finish)))
            finally:
                _retire(read)
