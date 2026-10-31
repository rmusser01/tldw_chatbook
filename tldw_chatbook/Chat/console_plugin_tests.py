"""Reviewed explicit tests using the existing Console admission and runtime owners."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True, eq=False)
class PluginTestReview:
    token: str
    session_id: str
    workspace_id: str | None
    installation_id: str
    component_id: str
    kind: str
    revision: str
    mappings: str
    configuration: object
    event_name: str | None
    data_json: str
    expires_at: float


class ConsolePluginTests:
    """Review/task retention only; execution belongs to the existing Console owners."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.reviews = {}
        self.tasks = {}
        self._owners = {}

    def _session(self, session_id):
        self.runtime._raise_if_disposed_or_session_fenced(session_id)
        controller = self.runtime._chat_controller
        if controller is None:
            raise ValueError("existing_console_session_required")
        session = next(
            (row for row in controller.store.sessions() if row.id == session_id), None
        )
        if session is None:
            raise ValueError("existing_console_session_required")
        return controller, session

    async def prepare(
        self,
        session_id,
        installation_id,
        component_id,
        *,
        kind,
        event_name=None,
        data=None,
    ):
        controller, session = self._session(session_id)
        service = controller._skills_service.local_service.plugin_service
        owners = (controller, session, controller.store, controller.app, service)
        details = await service.management_details(
            installation_id, workspace_id=session.workspace_id
        )
        component = details["inspection"].inventory.get(component_id)
        if component is None or component.kind != {
            "hook": "hook",
            "connection": "mcp",
        }.get(kind):
            raise ValueError("test_component_kind_mismatch")
        configuration = await controller.capture_turn_configuration_snapshot(session_id)
        current_controller, current_session = self._session(session_id)
        current = (
            current_controller,
            current_session,
            current_controller.store,
            current_controller.app,
            current_controller._skills_service.local_service.plugin_service,
        )
        if any(actual is not expected for actual, expected in zip(current, owners)):
            raise PermissionError("plugin_test_session_changed")
        if session.workspace_id != details["workspace_id"]:
            raise PermissionError("plugin_test_workspace_changed")
        entries = configuration.skill_context_maximum.get("available_skills", ())
        if not any(
            row.get("plugin_owned")
            and row.get("plugin_installation_id") == installation_id
            and row.get("plugin_component_id") == component_id
            for row in entries
        ):
            raise PermissionError("plugin_test_not_admitted")
        if kind == "hook":
            definition = json.loads(component.definition_json)
            declared = definition.get("event")
            event_name = event_name or declared
            if event_name != declared:
                raise ValueError("test_event_not_declared")
            from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

            lifecycle_event(
                event_name,
                session_id,
                workspace_id=session.workspace_id,
                data=data or {},
            )
        encoded = json.dumps(data or {}, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode()) > 65536:
            raise ValueError("test_data_too_large")
        now = time.monotonic()
        self.reviews = {
            key: value
            for key, value in self.reviews.items()
            if value.expires_at >= now or key in self.tasks
        }
        if len(self.reviews) >= 1000:
            raise RuntimeError("plugin_test_review_limit")
        review = PluginTestReview(
            uuid4().hex,
            session_id,
            session.workspace_id,
            installation_id,
            component_id,
            kind,
            details["inspection"].effective_digest,
            json.dumps(details["mappings"], sort_keys=True),
            configuration,
            event_name,
            encoded,
            now + 900,
        )
        self._owners = {
            key: value for key, value in self._owners.items() if key in self.reviews
        }
        self.reviews[review.token] = review
        self._owners[review.token] = owners
        return review

    def _current(self, review):
        controller, session = self._session(review.session_id)
        owners = self._owners[review.token]
        service = controller._skills_service.local_service.plugin_service
        current = (controller, session, controller.store, controller.app, service)
        if any(actual is not expected for actual, expected in zip(current, owners)):
            raise PermissionError("plugin_test_session_changed")
        if session.workspace_id != review.workspace_id:
            raise PermissionError("plugin_test_workspace_changed")
        return controller, session, service

    def _issued(self, review):
        if (
            type(review) is not PluginTestReview
            or self.reviews.get(review.token) is not review
        ):
            raise ValueError("plugin_test_review_unavailable")

    def discard(self, review):
        self._issued(review)
        if review.token in self.tasks:
            raise RuntimeError("plugin_test_already_started")
        self.reviews.pop(review.token)
        self._owners.pop(review.token)

    async def execute(self, review):
        self._issued(review)
        task = self.tasks.get(review.token)
        if task is None:
            if review.expires_at < time.monotonic():
                raise ValueError("plugin_test_review_expired")
            task = self.tasks[review.token] = asyncio.create_task(
                self._execute(review), name="plugin-test"
            )
        return await asyncio.shield(task)

    async def _execute(self, review):
        controller, _session, service = self._current(review)
        details = await service.management_details(
            review.installation_id, workspace_id=review.workspace_id
        )
        if (
            details["inspection"].effective_digest != review.revision
            or json.dumps(details["mappings"], sort_keys=True) != review.mappings
        ):
            raise PermissionError("plugin_test_definition_changed")
        fresh = await controller.capture_turn_configuration_snapshot(review.session_id)
        self._current(review)
        for key in (
            "workspace_roots",
            "project_authority",
            "tool_policy_profile_id",
            "persona_policy_rules",
            "mcp_tool_maximum",
            "mcp_definition_maximum",
        ):
            if getattr(fresh, key) != getattr(review.configuration, key):
                raise PermissionError("plugin_test_authority_changed")
        maximum = review.configuration.skill_context_maximum
        try:
            async with controller.plugin_test_admission(
                review.configuration
            ) as lifecycle:
                self._current(review)
                if review.kind == "hook":
                    if lifecycle is None:
                        raise PermissionError("plugin_test_hook_unavailable")
                    engine = lifecycle.engine
                    from tldw_chatbook.Plugins.hooks import hook_id

                    native = getattr(engine, "native_plugins", None)
                    matches = (
                        [
                            (identity, snapshot)
                            for identity, (snapshot, key) in native.owners.items()
                            if snapshot.installation_id == review.installation_id
                            and key == review.component_id
                        ]
                        if native is not None
                        else []
                    )
                    if len(matches) != 1:
                        raise PermissionError("plugin_test_hook_not_admitted")
                    handler_id, snapshot = matches[0]
                    if handler_id != hook_id(snapshot, review.component_id):
                        raise PermissionError("plugin_test_hook_not_admitted")
                    # Host identity remains real; origin explicitly labels synthetic input.
                    event = lifecycle.event(
                        review.event_name, data=json.loads(review.data_json)
                    ).model_copy(update={"origin": "host_plugin_test"})
                    execution = engine.begin_event(event)
                    try:
                        return await engine.fire_handler_async(
                            execution, handler_id, event
                        )
                    finally:
                        execution.close()
                from tldw_chatbook.Agents.persona_policy import (
                    parse_persona_policy_from_rules,
                )

                policy = parse_persona_policy_from_rules(
                    review.configuration.persona_policy_rules
                )
                from tldw_chatbook.Plugins.mcp_provider import (
                    MCPProviderGroup,
                    PluginMCPProvider,
                )

                providers = await controller._compose_mcp_provider(
                    review.session_id,
                    publish_counts=False,
                    plugin_maximum=maximum,
                    _retain_empty_plugin_providers=True,
                    profile_id_provider=lambda: review.configuration.tool_policy_profile_id,
                    persona_policy_provider=lambda: policy,
                    maximum_tool_ids=review.configuration.mcp_tool_maximum,
                    maximum_definition_hashes=review.configuration.mcp_definition_maximum,
                )
                self._current(review)
                if not isinstance(providers, MCPProviderGroup):
                    raise PermissionError("plugin_test_connection_not_admitted")
                provider = next(
                    (
                        row
                        for row in providers.providers
                        if isinstance(row, PluginMCPProvider)
                        and row.snapshot.installation_id == review.installation_id
                        and review.component_id in row.snapshot.selection
                    ),
                    None,
                )
                if provider is None:
                    raise PermissionError("plugin_test_connection_not_admitted")
                ownership = provider.ownership
                owner_id = ownership._owner_id(provider.snapshot)
                before = {
                    key
                    for key, row in ownership.connections.items()
                    if owner_id in row.owners
                }
                try:
                    discovery = await provider.connect(review.component_id)
                finally:
                    acquired = tuple(
                        key
                        for key, row in ownership.connections.items()
                        if owner_id in row.owners and key not in before
                    )
                    for key in acquired:
                        await ownership.detach(key, owner_id)
                return {
                    "discovery": discovery,
                    "cleanup_pending": tuple(
                        key
                        for key in acquired
                        if key in ownership.connections
                        and not ownership.connections[key].leases_settled
                    ),
                }
        finally:
            pending_id = maximum.get("plugin_run_id")
            if pending_id:
                await service.retire_pending(pending_id)


@contextlib.asynccontextmanager
async def plugin_test_admission(self, configuration):
    """Reserve a real idle Console session without accepting a conversation turn."""
    from .console_automatic_submission import _begin_hook_submission
    from .console_chat_models import ConsoleRunState, ConsoleRunStatus
    from .console_received_turn import received_turn_claim_for

    session_id = configuration.session_id
    if not any(row.id == session_id for row in self.store.sessions()):
        raise ValueError("existing_console_session_required")
    if (
        received_turn_claim_for(self.store, session_id) is not None
        or self.store.received_turn_for_session(session_id) is not None
    ):
        raise RuntimeError("console_received_turn_pending")
    busy = self._live_busy_session_ids()
    if session_id in busy or len(busy) >= self.max_parallel_runs:
        raise RuntimeError("console_session_busy")
    runtime = getattr(self, "_hooks_v2_runtime", None)
    if runtime is None:
        raise RuntimeError("console_runtime_unavailable")
    prior_engine = runtime.get_hooks_v2(session_id)
    _begin_hook_submission(self, session_id)
    task = asyncio.current_task()
    try:
        lifecycle = await runtime.prepare_hooks_v2(
            session_id, configuration=configuration, test_only=True
        )
        yield lifecycle
    finally:
        reservation = self._hooks_v2_submissions.pop(task, None)
        if reservation is not None:
            lifecycle, scope, _ = reservation
            if lifecycle is not None and scope is not None:
                lifecycle.close_scope(scope)
                if lifecycle.turn_scope is scope:
                    lifecycle.turn_scope = None
        try:
            created = getattr(runtime, "_plugin_test_created_engines", {}).pop(
                task, None
            )
            if (
                prior_engine is None
                and created is not None
                and runtime.get_hooks_v2(session_id) is created
            ):
                await runtime.close_hooks_v2(session_id)
                if runtime.get_hooks_v2(session_id) is None:
                    runtime._hooks_v2_lifecycles.pop(session_id, None)
                    runtime._hooks_v2_configured.pop(session_id, None)
        finally:
            if self.run_state_for(session_id).status is ConsoleRunStatus.VALIDATING:
                self._set_run_state(
                    ConsoleRunState(ConsoleRunStatus.STOPPED, "Plugin test ended."),
                    session_id=session_id,
                )
