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
            or self.parent_attachment is not None
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
