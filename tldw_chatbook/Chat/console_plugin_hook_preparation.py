"""Finite native custody for the original pre-engine plugin hook admission."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

from tldw_chatbook.Agents.activation import _guard, execution
from tldw_chatbook.Agents.execution_capacity import (
    CapacityRefused,
    WorkOrigin,
    current_execution_owner,
)
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

from .console_hook_preparation import hook_preparation_source_current
from .console_preparation_reads import (
    _READS_LOCK,
    ConsolePreparationRead,
    _observe_locked,
    _retire,
)
from .console_turn_context import _freeze

_submission = ContextVar("native_plugin_hook_submission", default=None)


@contextmanager
def submission_origin(controller, session, configuration, origin):
    """Bind only the original validated submission task, never event labels."""
    token = _submission.set(
        (asyncio.current_task(), controller, session, configuration, origin)
    )
    try:
        yield
    finally:
        _submission.reset(token)


async def _joined(task, cancel=None):
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancelled = error
            if cancel is not None:
                cancel()
        except BaseException:  # noqa: BLE001 - consume the original task below.
            break
    try:
        result = task.result()
    except BaseException:
        if cancelled is not None:
            raise cancelled from None
        raise
    if cancelled is not None:
        raise cancelled
    return result


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
            await _joined(
                asyncio.create_task(asyncio.to_thread(request.admit)), request.cancel
            )
            read._producer = asyncio.create_task(request.invoke())
            result = await _joined(read._producer, request.cancel)
            request.check(service)
            yield result, request
            request.check(service)
        finally:
            try:
                await _joined(asyncio.create_task(asyncio.to_thread(request.finish)))
            finally:
                _retire(read)


class _Preparation:
    # This request has no new durable payload. Plugin callbacks retain their
    # existing required writes and infrastructure ownership.
    required_writes = ()

    def __init__(self, issuer, source, service, configuration, review):
        self.issuer, self.source, self.service = issuer, source, service
        self.configuration, self.maximum, self.review = (
            configuration,
            configuration.skill_context_maximum,
            review,
        )
        self.skills = source.controller._skills_service
        self.local = self.skills.local_service
        self.service_state = tuple(
            getattr(service, name, None)
            for name in (
                "_thread",
                "_loop",
                "_ready",
                "_coordinator",
                "_admission",
                "fences",
                "profile_root",
                "workspace_lookup",
            )
        )
        self.callback = service.hook_configuration
        self.coordinator = getattr(service, "_coordinator", None)
        self.infrastructure = tuple(
            getattr(self.coordinator, name, None)
            for name in ("owner", "registry", "authority")
        )
        self.maximum_pin = _freeze(self.maximum)
        self.snapshot_store = getattr(service, "_snapshots", None)
        self.snapshots = tuple(
            (
                row.get("plugin_ceiling"),
                row.get("plugin_component_id"),
                row.get("plugin_workspace_id"),
                self.snapshot_store.get(row.get("plugin_ceiling")),
            )
            for row in self.maximum.get("available_skills", ())
            if row.get("plugin_owned") and self.snapshot_store is not None
        )
        self.sources = _guard.captured_sources(None)
        self.claim = self.owner = self.operation = None
        self.cancelled = False
        bound = _submission.get()
        parent = current_execution_owner()
        self.origin = WorkOrigin.AUTOMATIC
        from .console_chat_models import ConsoleSubmissionOrigin

        if bound is not None and (
            bound[0] is asyncio.current_task()
            and bound[1] is source.controller
            and bound[2] is source.session
            and bound[3] is configuration
            and type(bound[4]) is ConsoleSubmissionOrigin
        ):
            self.origin = (
                WorkOrigin.AUTOMATIC if bound[4].automatic else WorkOrigin.MANUAL
            )
        elif (
            parent is not None
            and parent.capacity is issuer.capacity
            and parent.is_admitted
        ):
            self.origin = parent.origin

    def check_memory(self, service):
        """No external reads or user callbacks under the plugin publication lock."""
        issuer, runtime, source = self.issuer, self.issuer.runtime, self.source
        for _key, component, workspace, snapshot in self.snapshots:
            if snapshot is None or not service.published_current(snapshot):
                raise PermissionError("Plugin hook ceiling changed.")
            service.fences.check(snapshot.installation_id, workspace)
            if component not in snapshot.selection:
                raise PermissionError("Plugin hook component unavailable.")
        if (
            self.cancelled
            or service is not self.service
            or runtime._native_plugin_hook_preparation is not issuer
            or runtime._app is not issuer.app
            or not hook_preparation_source_current(runtime, source)
            or source.controller._skills_service is not self.skills
            or self.skills.local_service is not self.local
            or vars(self.local).get(
                "_plugin_service", vars(self.local).get("plugin_service")
            )
            is not service
            or service.hook_configuration != self.callback
            or any(
                getattr(self.coordinator, name, None) is not original
                for name, original in zip(
                    ("owner", "registry", "authority"), self.infrastructure
                )
            )
            or self.configuration.skill_context_maximum is not self.maximum
            or _freeze(self.maximum) != self.maximum_pin
            or getattr(service, "_snapshots", None) is not self.snapshot_store
            or any(
                self.snapshot_store.get(key) is not snapshot
                for key, _component, _workspace, snapshot in self.snapshots
            )
            or issuer.capacity is not None
            and runtime._execution_capacity is not issuer.capacity
            or issuer.activity is not None
            and runtime._worktree_activity is not issuer.activity
            or tuple(
                getattr(service, name, None)
                for name in (
                    "_thread",
                    "_loop",
                    "_ready",
                    "_coordinator",
                    "_admission",
                    "fences",
                    "profile_root",
                    "workspace_lookup",
                )
            )
            != self.service_state
        ):
            raise RuntimeError("Original plugin hook preparation changed.")
        current = source.permissions._published
        if (
            source.permissions._closed.is_set()
            or current is None
            or current.config.config_path != self.review.config.config_path
            or current.config.section_stamp != self.review.config.section_stamp
            or current.store_path != self.review.store_path
            or (str(current.store_path), str(current.config.config_path))
            in source.permissions._refresh_pending
        ):
            raise PermissionError("Plugin hook configuration changed.")

    def cancel(self):
        self.cancelled = True
        if self.operation is not None:
            self.operation.mark_stopping()

    async def join(self, future):
        """Retain the original service Future; cancellation cannot cancel its job."""
        return await _joined(asyncio.wrap_future(future), self.cancel)

    def check(self, service):
        self.check_memory(service)
        if _guard.captured_sources(None) != self.sources:
            raise RuntimeError("Plugin hook activation source changed.")

    def admit(self):
        self.check(self.service)
        issuer = self.issuer
        with execution(sources=self.sources):
            if issuer.capacity is None:
                issuer.capacity = issuer.runtime.execution_capacity
            if issuer.activity is None:
                issuer.activity = issuer.runtime.worktree_activity
            self.check(self.service)
            with issuer.activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                if claim is None:
                    raise CapacityRefused("hook_preparation_activity_busy")
                session = self.source.session
                owner = issuer.capacity.begin_execution(
                    origin=self.origin,
                    conversation_id=session.persisted_conversation_id or session.id,
                )
                try:
                    operation = owner.reserve_tool()
                    try:
                        self.check(self.service)
                        claim.bind_owner(owner)
                    except BaseException:
                        operation.finish()
                        raise
                except BaseException:
                    owner.finish_root()
                    raise
                self.claim, self.owner, self.operation = claim, owner, operation

    async def invoke(self):
        from tldw_chatbook.Plugins.service import _hook_preparation

        token = _hook_preparation.set(self)
        try:
            with execution(sources=self.sources), self.owner.activate():
                self.check(self.service)
                runtime = self.issuer.runtime
                return await self.callback(
                    self.maximum,
                    scratch_manager=runtime._scratch_spaces,
                    scratch_snapshot=runtime._scratch_spaces.snapshot(self.source.session_id),
                )
        finally:
            _hook_preparation.reset(token)

    def finish(self):
        if self.operation is not None:
            self.operation.finish()
            self.claim.settle(durable=True)
            self.owner.finish_root()
