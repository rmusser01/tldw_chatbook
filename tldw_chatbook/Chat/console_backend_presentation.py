"""Concrete backend endpoint; parent views and their authority stay remote."""

from __future__ import annotations

from inspect import getattr_static
from types import FunctionType

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

from .console_presentation_port import ConsoleParentAssociation, ConsolePresentationPort


class ConsoleBackendPresentation:
    """Bind original domain producers to one actual contained service owner."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self._runtime = None
        self._controller = None
        self._controller_app = None
        self._app = None
        self._view = None
        self._visit = 0
        self._attachment_generation = None
        self._hook_sources = ()
        self.association = ConsoleParentAssociation(object(), 0)
        self._display_full_entries = {}
        self._display_context_demands = {}
        self._display_character_demands = {}
        self.parent_attachment = None
        self._view_checkpoints = {}
        from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle

        self.view_lifecycle = ConsoleBackendViewLifecycle(self)
        self._projection_views = {}
        from .console_backend_readiness import ConsoleBackendReadiness

        self.readiness = ConsoleBackendReadiness(backend)
        from .console_backend_tool_counts import ConsoleBackendToolCounts

        self.tool_counts = ConsoleBackendToolCounts(self)

    def admit_view_checkpoint(self, message):
        """Register one closed parent lifecycle request before its task can await."""
        self.require_association(self.association)
        if message.operation != "ReadProjection" or "view_checkpoint" not in message.payload:
            return None
        value = message.payload["view_checkpoint"]
        checkpoint = self.view_lifecycle.admit(
            value["attachment_id"], value["visit"], value["action"],
        )
        self._view_checkpoints[message.identity.request_id] = (message.identity, checkpoint)
        return checkpoint

    def release_view_checkpoint(self, identity, checkpoint) -> None:
        entry = self._view_checkpoints.get(identity.request_id)
        if entry is not None and entry[0] is identity and entry[1] is checkpoint:
            del self._view_checkpoints[identity.request_id]

    async def prepare_view_checkpoint(self, identity) -> None:
        """Join only the exact request's original lifecycle producer."""
        self.require_association(self.association)
        entry = self._view_checkpoints.get(identity.request_id)
        if entry is None:
            return
        # Local command identities are detached from the wire identity but all
        # fixed channel/source/session/request fields must still match.
        if any(getattr(entry[0], field) != getattr(identity, field) for field in (
            "profile_epoch", "service_epoch", "sequence", "request_id",
            "source_revision", "session_id", "incarnation_id",
        )):
            raise RuntimeError("Console view checkpoint identity changed")
        try:
            await self.view_lifecycle.prepare(entry[1])
            self.require_association(self.association)
        finally:
            self.release_view_checkpoint(entry[0], entry[1])

    async def drain_view_lifecycle(self) -> None:
        """Join retained original mount seeds before the backend can close."""
        from types import MethodType
        from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle, _METHODS
        from .console_domain_services import _definition_current

        owner = self.view_lifecycle
        method = owner.drain
        record = dict(_METHODS)["drain"]
        if (type(owner) is not ConsoleBackendViewLifecycle
            or owner.presentation is not self
            or type(method) is not MethodType or method.__self__ is not owner
            or method.__func__ is not record[0] or not _definition_current(record)):
            raise RuntimeError("Console lifecycle drain source changed")
        await method()

    async def prepare_readiness(self, identity) -> None:
        """Join the fixed readiness read while its original command owns intake."""
        self.require_association(self.association)
        await self.readiness.warm(identity)

    def capture_projection_view(self, controls, identity, projection, *, started_at: float) -> None:
        """Retain inert row facts at the original synchronous projection boundary."""
        from .console_control_dispatch import (
            ConsoleLocalControls,
            require_control_source,
        )
        from .console_execution_projection_values import encode_session_projection
        from .console_execution_projection_result import encode_pressed_inputs
        from .console_execution_readiness_values import encode_settings_readiness
        from .console_execution_context_summary_values import encode_context_summary
        from .console_execution_agent_rail_values import encode_agent_rail_projection
        from .console_execution_projection_values import _settings
        import time

        self.require_association(self.association)
        source = require_control_source(self._runtime)
        if (
            type(controls) is not ConsoleLocalControls
            or controls is not self.backend.controls
            or source.identity is not identity
            or identity.source_revision != controls.source_revision
            or projection.source_revision != identity.source_revision
            or projection.session_id != identity.session_id
            or projection.incarnation_id != identity.incarnation_id
            or identity.request_id in self._projection_views
        ):
            raise RuntimeError("Backend projection capture changed")
        store = controls._store
        sessions = store.sessions()
        effective, readiness, provider, model, provider_chip = self.readiness.current_value(identity)
        composition = (
            self._capture_composition(store._sessions[store.active_session_id])
            if store.active_session_id is not None else None
        )
        from .console_execution_workspace_values import (
            ConsoleWorkspaceSessionFacts, ConsoleWorkspaceSurface, encode_workspace_surface,
        )
        from .console_chat_store import ConsoleChatSession
        from .console_domain_services import _definition_current
        from types import MethodType

        workspace_facts = []
        for row in sessions:
            original = store._sessions[row.id]
            character_reader = original.local_character_id
            if (
                type(original) is not ConsoleChatSession
                or type(character_reader) is not MethodType
                or character_reader.__self__ is not original
                or character_reader.__func__ is not _CHARACTER_READER[0]
                or not _definition_current(_CHARACTER_READER)
            ):
                raise RuntimeError("Console workspace session reader changed")
            launch = original.agent_handoff_launch
            workspace_facts.append(ConsoleWorkspaceSessionFacts(
                session_id=row.id, updated_at=str(original.updated_at or ""),
                character_id=character_reader(), character_name=original.character_name,
                handoff_label=launch.label if launch is not None else None,
                queued_count=self._controller.activity_for(row.id).queued_count,
                run_marker=self._controller.run_marker_for(row.id),
            ))
        captured_at = time.monotonic()
        body = {
            "sessions": [encode_session_projection(session) for session in sessions],
            "active_session_id": store.active_session_id,
            "fingerprints": [],
            "pressed_inputs": encode_pressed_inputs(
                store.session_input_snapshot(identity.session_id),
                store._sessions[identity.session_id],
            ),
            "readiness": {
                "effective_settings": _settings(effective, False),
                "state": encode_settings_readiness(readiness),
                "context_summary": encode_context_summary(
                    self.readiness.context_summary.estimate,
                    self.readiness.context_summary.summary,
                    compaction_pause_expected=self.readiness.context_summary.compaction_pause_expected,
                    compaction_forecast_unavailable=self.readiness.context_summary.compaction_forecast_unavailable,
                ),
                "agent_rail": encode_agent_rail_projection(
                    self.readiness.agent_projection(store._sessions[identity.session_id])
                ),
                "prompt_history": self.readiness.prompt_history_payload,
                "provider_label": provider,
                "control_provider_label": provider_chip,
                "model": model,
                "remaining_seconds": max(0.0, 1.0 - (captured_at - self.readiness.at)),
                "projection_elapsed_seconds": captured_at - started_at,
            },
            "composition": composition,
            "workspace": encode_workspace_surface(ConsoleWorkspaceSurface(
                active_workspace_id=store.workspace_context.active_workspace_id,
                controller_present=True, sessions=tuple(workspace_facts),
            )),
            "session_surface": {
                "streaming_session_id": self._controller.streaming_session_id(),
                "run_markers": [
                    {"session_id": item.session_id, "marker": item.run_marker.value}
                    for item in workspace_facts
                ],
                "queue_counts": [
                    {"session_id": item.session_id, "count": item.queued_count}
                    for item in workspace_facts
                ],
            },
        }
        for session in sessions:
            count, length = store.session_fingerprint(session.id)
            body["fingerprints"].append(
                {
                    "session_id": session.id,
                    "count": count,
                    "last_content_length": length,
                }
            )
        self._projection_views[identity.request_id] = (identity, projection, body)

    def _capture_composition(self, session) -> dict:
        """Read the original resident controller facts without exporting an owner."""
        from types import MethodType
        from .console_chat_controller import ConsoleChatController
        from .console_chat_start import ConsoleChatStartCoordinator
        from .console_agent_bridge import ConsoleAgentBridge
        from .console_execution_composition_values import encode_session_composition
        from .console_domain_services import _definition_current

        self.require_association(self.association)
        self.backend._require_owner_loop()
        controller = self._controller
        if (
            type(controller) is not ConsoleChatController
            or controller is not self.backend.controller
            or controller.store is not self.backend.store
            or controller.store._sessions.get(session.id) is not session
            or controller.store.active_session_id != session.id
            or type(controller._chat_start) is not ConsoleChatStartCoordinator
            or controller._chat_start._controller is not controller
            or type(controller._agent_bridge) is not ConsoleAgentBridge
            or controller._agent_bridge is not self.backend.runtime.agent_bridge
        ):
            raise RuntimeError("Console composition owner changed")
        for receiver, names in (
            (controller, _COMPOSITION_CONTROLLER_METHODS),
            (controller._chat_start, _COMPOSITION_START_METHODS),
            (self._runtime, _COMPOSITION_RUNTIME_METHODS),
        ):
            for name, record in names:
                bound = getattr(receiver, name, None)
                if (
                    type(bound) is not MethodType or bound.__self__ is not receiver
                    or bound.__func__ is not record[0]
                    or not _definition_current(record)
                ):
                    raise RuntimeError("Console composition reader changed")
        for owner, name, descriptor, record in _COMPOSITION_PROPERTIES:
            if getattr_static(owner, name, None) is not descriptor or not _definition_current(record):
                raise RuntimeError("Console composition property changed")
        for owner, name, record in _COMPOSITION_COPY:
            if getattr_static(owner, name, None) is not record[0] or not _definition_current(record):
                raise RuntimeError("Console composition copy changed")
        from .console_backend_tool_counts import _RESOLVER, backend_tool_counts

        if backend_tool_counts is not _RESOLVER[0] or not _definition_current(_RESOLVER):
            raise RuntimeError("Backend tool-count resolver changed")
        counts_owner = backend_tool_counts(controller)
        if counts_owner is not self.tool_counts:
            raise RuntimeError("Backend tool-count owner changed")
        counts = counts_owner.snapshot(controller, session.id)
        launch, staged_revision, sent_notice = self._runtime.snapshot_console_staged_evidence(session.id)
        pending = controller.has_pending_approval_round(session.id)
        return encode_session_composition(
            session,
            pending_approval_count=controller.pending_round_count(session.id),
            pending_interrupt_copy=(
                _COMPOSITION_COPY[0][2][0](controller, session.id) if pending else ""
            ),
            blocked_turn_reason=_COMPOSITION_COPY[1][2][0](controller),
            is_stop_allowed=controller.is_stop_allowed,
            chat_start_prepared=controller._chat_start.is_prepared(session.id),
            change_tracking_enabled=controller._agent_bridge.change_tracking_enabled,
            tools_published=counts.published,
            builtin_tool_count=counts.builtin_count,
            mcp_tool_count=counts.mcp_count,
            mcp_not_connected_count=counts.not_connected_count,
            selected_tool_count=counts.selected_count,
            staged_evidence_present=launch is not None,
            staged_evidence_revision=staged_revision,
            evidence_sent_notice=sent_notice,
        )

    def projection_view(self, identity, projection) -> dict:
        """Return only this completed command's already-captured row values."""
        self.require_association(self.association)
        captured = self._projection_views.get(identity.request_id)
        if (
            captured is None
            or captured[0] is not identity
            or captured[1] is not projection
        ):
            raise RuntimeError("Backend projection result association changed")
        return captured[2]

    def bind_runtime(self, runtime) -> None:
        if self._runtime is not None or runtime._app is not None:
            raise RecoveryRequired("console_snapshot_owner_changed")
        self._runtime = runtime

    def definitions_current(self) -> bool:
        from .console_domain_services import _definition_current

        return type(self) is ConsoleBackendPresentation and all(
            getattr_static(self, name, None) is record[0]
            and _definition_current(record)
            for name, record in _DEFINITIONS
        )

    def require_association(self, association) -> None:
        from .console_execution_backend import ConsoleExecutionBackend

        backend = self.backend
        if (
            not self.definitions_current()
            or type(backend) is not ConsoleExecutionBackend
            or backend.presentation is not self
            or association is not self.association
            or self._runtime is None
            or backend.runtime is not self._runtime
            or self._runtime._app is not None
            or self._runtime.domain_services is not backend.domain
            or backend.domain.presentation is not self
            or self._runtime._disposed
            or self._runtime.view is not None
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")
        backend.require_sources()

    def _require_display_owner(self, owner):
        from .console_domain_display import ConsoleDomainDisplayOwner
        from .console_display_phases import ConsoleDisplayPhaseRefused

        self.require_association(self.association)
        if (
            type(owner) is not ConsoleDomainDisplayOwner
            or owner.domain is not self.backend.domain
            or owner.runtime is not self._runtime
            or owner.sources is not self.backend.domain.sources
            or owner.store is not self._runtime._chat_store
            or owner.controller is not self._runtime._chat_controller
        ):
            raise ConsoleDisplayPhaseRefused("Backend display association changed")
        return None

    def publish_commit_plan(self, controls, pending) -> None:
        """Emit only the original registered pre-ACK plan's inert projection."""
        import asyncio
        from .console_control_dispatch import (
            ConsoleLocalControls,
            _PendingPlan,
            require_control_source,
        )
        from .console_execution_worker import _BackendService

        self.require_association(self.association)
        if (
            type(controls) is not ConsoleLocalControls
            or controls is not self.backend.controls
        ):
            raise RuntimeError("Backend commit controls changed")
        source = require_control_source(self._runtime)
        service = self.backend._worker_service
        if (
            type(pending) is not _PendingPlan
            or controls._plans.get(
                (pending.projection.turn_id, pending.projection.ordinal)
            )
            is not pending
            or pending.task is not asyncio.current_task()
            or pending.state != "waiting"
            or pending.acknowledged.done()
            or pending.source_revision != controls.source_revision
            or source.identity.request_id != pending.projection.received_request_id
            or type(service) is not _BackendService
            or service.backend is not self.backend
        ):
            raise RuntimeError("Backend commit plan association changed")
        service.publish_commit_plan(source.identity, pending)

    # These exact original bodies only inspect a qualified viewless owner and
    # the real local FULL/checkpoint. They do not discover an App or issue I/O.
    capture_display = ConsolePresentationPort.capture_display
    display_run_active = ConsolePresentationPort.display_run_active
    display_character_key = ConsolePresentationPort.display_character_key
    require_display_checkpoint = ConsolePresentationPort.require_display_checkpoint
    require_display_publication = ConsolePresentationPort.require_display_publication
    publish_display_character = ConsolePresentationPort.publish_display_character
    display_appearance_current = ConsolePresentationPort.display_appearance_current
    publish_display_appearance = ConsolePresentationPort.publish_display_appearance
    clear_view_hooks = ConsolePresentationPort.clear_view_hooks
    _decision_setter = ConsolePresentationPort._decision_setter
    decision_available = ConsolePresentationPort.decision_available
    pending_router_available = ConsolePresentationPort.pending_router_available

    def bind_view_hooks(self) -> None:
        self.require_association(self.association)
        if self.parent_attachment is not None:
            raise RuntimeError("Backend parent attachment requires its fixed hooks")
        self.clear_view_hooks()

    def wire_wake(self, wake, projection) -> None:
        wake.wire(app=None, startup_ready=self.backend.is_ready)
        wake._recovery_projection = projection

    def register_fleet_attention(self, bridge, receipt_service) -> None:
        from .console_fleet_attention import register_fleet_attention

        register_fleet_attention(bridge, None, receipt_service=receipt_service)

    def prepare_default_convergence(self, session_id, checked_config):
        self.require_association(self.association)
        if self.parent_attachment is not None:
            raise RuntimeError("Backend defaults require the exact parent checkpoint")
        return None  # Original viewless branch; never an attached-view witness.

    def archive_send_inflight(self, record):
        raise RuntimeError("Backend archive admission requires its parent checkpoint")

    def notice(self, message: str, *, severity: str = "warning") -> bool:
        self.require_association(self.association)
        self.backend.retain_notice(message, severity=severity)
        return False  # Receipt/adoption by the real parent is a separate fact.

    def present_pending_projection(self, projection) -> bool:
        self.require_association(self.association)
        return False

    def present_decision(self, kind, payload) -> None:
        self.require_association(self.association)
        if self.parent_attachment is not None:
            raise RuntimeError("Backend decision requires its parent presentation")

    def clear_decision(self, kind) -> None:
        self.present_decision(kind, None)

    def project_pending(self) -> None:
        self.require_association(self.association)

    def announce_hidden(self, message: str) -> bool:
        return self.notice(message)

    def project_attention(self, host, *, ring: bool) -> None:
        self.require_association(self.association)


from .console_domain_services import _definition

_DEFINITIONS = tuple(
    (name, _definition(value))
    for name, value in vars(ConsoleBackendPresentation).items()
    if type(value) is FunctionType
)

# Fixed presentation readers are pinned separately from the endpoint methods.
from . import console_chat_models as _composition_models
from .console_chat_controller import ConsoleChatController as _CompositionController
from .console_chat_start import ConsoleChatStartCoordinator as _CompositionStart
from .console_agent_bridge import ConsoleAgentBridge as _CompositionBridge
from tldw_chatbook.UI.Console_Modules import provider_continuation_recovery as _composition_recovery

_COMPOSITION_CONTROLLER_METHODS = tuple(
    (name, _definition(getattr(_CompositionController, name)))
    for name in (
        "has_pending_approval_round", "pending_round_count", "pending_round_kinds",
        "trace_call_recovery_preparation", "context_compaction_hold", "preparation_outcome",
        "activity_for", "run_marker_for", "streaming_session_id",
    )
)
_COMPOSITION_START_METHODS = (("is_prepared", _definition(_CompositionStart.is_prepared)),)
_COMPOSITION_PROPERTIES = tuple(
    (owner, name, descriptor, _definition(descriptor.fget))
    for owner, name in (
        (_CompositionController, "is_stop_allowed"),
        (_CompositionBridge, "change_tracking_enabled"),
    )
    for descriptor in (getattr_static(owner, name),)
)
_COMPOSITION_COPY = tuple(
    (owner, name, _definition(getattr(owner, name)))
    for owner, name in (
        (_composition_models, "console_pending_round_copy_for"),
        (_composition_recovery, "blocked_turn_reason"),
    )
)

from .console_runtime import ConsoleRuntime as _CompositionRuntime
from .console_chat_store import ConsoleChatSession as _CompositionSession

_CHARACTER_READER = _definition(_CompositionSession.local_character_id)

_COMPOSITION_RUNTIME_METHODS = tuple(
    (name, _definition(getattr(_CompositionRuntime, name)))
    for name in ("snapshot_console_staged_evidence", "_staged_evidence_state")
)
