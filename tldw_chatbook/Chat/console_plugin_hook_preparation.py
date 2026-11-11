"""Finite native custody for the original pre-engine plugin hook admission."""

from __future__ import annotations

import asyncio

from tldw_chatbook.app_services import read_mcp_service
from tldw_chatbook.Agents.activation import captured_sources, execution
from tldw_chatbook.Agents.execution_capacity import (
    CapacityRefused,
    WorkOrigin,
    current_execution_owner,
)
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

from .console_preparation_reads import NativePluginHookPreparation as NativePluginHookPreparation
from .console_turn_context import _freeze

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
        self.sources = captured_sources(None)
        self.claim = self.owner = self.operation = None
        from tldw_chatbook.Agents.run_hooks import native_hook_submission as _submission
        from .console_ordinary_tree import current_ordinary_tree

        self.ordinary_tree = current_ordinary_tree(source.controller)
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
        from .console_hook_preparation import hook_preparation_source_current

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
                    ("owner", "registry", "authority"),
                    self.infrastructure,
                    strict=True,
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
        from tldw_chatbook.Agents.hooks_v2.ownership import join_hook_worker

        return await join_hook_worker(asyncio.wrap_future(future), self.cancel)

    def check(self, service):
        self.check_memory(service)
        if captured_sources(None) != self.sources:
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
            if self.ordinary_tree is not None:
                self.owner = self.ordinary_tree.borrow(issuer.runtime)
                self.operation = self.owner.reserve_tool()
                return
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

    def require_original_mcp(self):
        from .console_configuration_preparation import (
            original_mcp_capture_matches,
            pending_mcp_witness,
        )

        self.check(self.service)
        controller, session = self.source.controller, self.source.session
        configuration = self.configuration
        witness = pending_mcp_witness(configuration)
        if (
            type(witness) is not tuple
            or witness[-4] is not configuration
            or not original_mcp_capture_matches(
                controller, session, configuration, pending=True
            )
        ):
            raise PermissionError("Original compaction MCP capture changed.")

    async def resolve_compaction_mcp(self):
        """Capture the original MCP ceiling under this hook's live reservation."""
        from dataclasses import replace
        from functools import partial
        from tldw_chatbook.MCP.console_snapshot import (
            _CapturedSources,
            _capture_console_definition_selection,
            standard_console_sources,
        )
        from .console_configuration_capture import (
            CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
            _capture_mcp_definition_selection,
        )
        from .console_configuration_preparation import (
            consume_native_mcp_permission_publication,
        )
        from .console_preparation_reads import run_preparation_read

        configuration = self.configuration
        controller, session = self.source.controller, self.source.session
        mcp = read_mcp_service(controller.app)
        permission = getattr(mcp, "_permission_store", None)
        def reservation_current():
            self.check(self.service)
            if (
                not self.owner.is_admitted
                or self.operation is None
                or read_mcp_service(controller.app) is not mcp
            ):
                raise PermissionError("Original compaction MCP reservation changed.")

        def current():
            reservation_current()
            self.require_original_mcp()
            if getattr(mcp, "_permission_store", None) is not permission:
                raise PermissionError("Original compaction MCP permission source changed.")

        with execution(sources=self.sources), self.owner.activate():
            current()

            async def run_native(callback, *, check=current):
                return await run_preparation_read(
                    callback,
                    creator=controller,
                    session_id=session.id,
                    reads=controller._preparation_reads,
                    observers=(self.issuer.runtime._preparation_reads,),
                    require_current=check,
                )

            if permission is None:
                if not standard_console_sources(mcp):
                    raise PermissionError("Original compaction MCP permission source unavailable.")
                publisher = _CapturedSources(mcp)

                def publication_current():
                    reservation_current()
                    publisher.require_current()

                permission = await run_native(publisher.permission_owner, check=publication_current)
                publication = publisher.native_permission_publication
                if publication is None or publication[2] is not permission:
                    raise PermissionError("Original compaction MCP publication unavailable.")
                self.permission = permission
                self.permission_path = publication[4]
                self.native_permission_publication = publication
                consume_native_mcp_permission_publication(self)
                current()

            policy = {
                "profile_id": configuration.tool_policy_profile_id,
                "persona_policy_rules": configuration.persona_policy_rules,
                "_admitted": True,
            }
            if standard_console_sources(mcp):
                selected = await _capture_console_definition_selection(
                    mcp, CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
                    _run_native=run_native, **policy,
                )
            else:
                selected = await run_native(partial(
                    _capture_mcp_definition_selection, controller.app, **policy,
                ))
            current()
        return replace(
            configuration,
            mcp_tool_maximum=frozenset(selected.definitions),
            mcp_definition_maximum=selected.definitions,
            mcp_definition_sources=selected.sources,
            mcp_definition_complete=selected.complete,
            mcp_capture_pending=False,
        )

    def finish(self):
        if self.operation is not None:
            self.operation.finish()
            if self.claim is not None:
                self.claim.settle(durable=True)
                self.owner.finish_root()
