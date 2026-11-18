"""Actual parent-modal visibility for the original runtime's project decisions."""

from __future__ import annotations

import time
from inspect import getattr_static

from . import console_runtime as runtime_source
from .console_chat_controller import ConsoleChatController
from .console_domain_services import _definition, _definition_current
from .console_execution_project_values import ConsoleProjectDecisionHead
from .console_interrupt_rounds import InterruptRoundHost


class ConsoleBackendProjectDecisions:
    """Keep only the currently shown original decision and its actual visit."""

    def __init__(self, presentation) -> None:
        self.presentation = presentation
        self._shown = None
        self._hidden_checkpoint = None

    def _require(self):
        presentation = self.presentation
        presentation.require_association(presentation.association)
        presentation.backend._require_owner_loop()
        runtime = presentation.backend.runtime
        if (
            type(runtime) is not runtime_source.ConsoleRuntime
            or presentation.project_decisions is not self
            or any(
                getattr_static(runtime, name) is not record[0]
                or not _definition_current(record)
                for name, record in _RUNTIME_METHODS
            )
            or any(
                getattr_static(self, name) is not record[0]
                or not _definition_current(record)
                for name, record in _METHODS
            )
            or any(
                getattr_static(runtime._chat_controller, name) is not record[0]
                or not _definition_current(record)
                for name, record in _CONTROLLER_METHODS
            )
        ):
            raise RuntimeError("Console project decision owner changed")
        return runtime

    def _head_locked(self, runtime, session_id):
        return min(
            (
                decision for decision in (
                    *runtime._project_binding_decisions.values(),
                    *runtime._project_dispatch_decisions.values(),
                )
                if decision.session_id == session_id and not decision.completed.is_set()
            ),
            key=lambda decision: decision.order,
            default=None,
        )

    def pause(self) -> None:
        """Pause the exact prior notice using the original runtime timer helper."""
        runtime = self._require()
        with runtime._project_decision_lock:
            if self._shown is not None:
                runtime._pause_project_dispatch_locked(self._shown[2])
                self._shown = None
            self._hidden_checkpoint = None

    def _hide_console_view(self, attachment, head) -> None:
        """Revoke the prepared Console token for this original modal owner."""
        from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle, _METHODS

        runtime = self._require()
        lifecycle = self.presentation.view_lifecycle
        record = dict(_METHODS)["admit"]
        if (
            type(lifecycle) is not ConsoleBackendViewLifecycle
            or lifecycle.presentation is not self.presentation
            or getattr_static(lifecycle, "admit", None) is not record[0]
            or not _definition_current(record)
            or self.presentation.parent_attachment != attachment
        ):
            raise RuntimeError("Console project visibility owner changed")
        retained = self._hidden_checkpoint
        if (
            retained is not None and retained[:2] == attachment
            and retained[2] is head and retained[3] is lifecycle._current
        ):
            self.presentation.require_view_checkpoint(retained[3])
        else:
            self.presentation.require_view_checkpoint(lifecycle._current)
            checkpoint = record[0](lifecycle, *attachment, "hidden")
            self._hidden_checkpoint = (*attachment, head, checkpoint)
        runtime._chat_controller.on_console_view_visibility_changed(False)
        runtime._chat_controller.set_answerable_decision(head.session_id, None)

    def require_modal_display_checkpoint(self, identity, attachment_id, visit, *, _captured=None):
        """Qualify only resident decision reads for the actual shown modal."""
        runtime = self._require()
        attachment = (attachment_id, visit)
        with runtime._project_decision_lock:
            session = runtime._chat_store._sessions.get(identity.session_id)
            retained = self._hidden_checkpoint
            if (
                session is None or session.incarnation_id != identity.incarnation_id
                or runtime._chat_store.active_session_id != session.id
                or identity.source_revision != self.presentation.backend.controls.source_revision
                or not self.presentation.backend.controls._intake_open
                or retained is None or retained[:2] != attachment
                or retained[2].session_id != identity.session_id
                or (_captured is None and self._shown != (*attachment, retained[2]))
                or (_captured is not None and _captured is not retained[3])
                or (self._shown is not None and self._shown != (*attachment, retained[2]))
                or self.presentation.parent_attachment != attachment
                or retained[3].action != "hidden"
                or runtime._chat_controller._interrupt_host.view_visible is not False
            ):
                raise RuntimeError("Console project modal display scope changed")
            self.presentation.require_view_checkpoint(retained[3])
            return retained[3]

    def capture(self, identity) -> ConsoleProjectDecisionHead | None:
        """Capture the same active-session ordered head as the original view."""
        runtime = self._require()
        store = runtime._chat_store
        session = store._sessions.get(identity.session_id)
        if (
            session is None or session.incarnation_id != identity.incarnation_id
            or store.active_session_id != identity.session_id
        ):
            raise RuntimeError("Console project decision session changed")
        with runtime._project_decision_lock:
            head = self._head_locked(runtime, identity.session_id)
            if self._shown is not None and (
                self._shown[2] is not head
                or self._shown[:2] != self.presentation.parent_attachment
            ):
                runtime._pause_project_dispatch_locked(self._shown[2])
                self._shown = None
            if head is None:
                return None
            if type(head) is runtime_source._ProjectDispatchDecision:
                return ConsoleProjectDecisionHead(
                    session.id, session.incarnation_id, head.decision_id, head.order,
                    "dispatch", head.notice, (),
                )
            if type(head) is runtime_source._ProjectBindingDecision:
                return ConsoleProjectDecisionHead(
                    session.id, session.incarnation_id, head.decision_id, head.order,
                    "binding", None, head.choices,
                )
            raise TypeError("Original Console project decision is required")

    def read_head(self, identity, attachment_id, visit):
        """Read resident decision owners without entering unrelated FULL work."""
        from .console_execution_decision_head import ConsoleDecisionHead

        runtime = self._require()
        presentation = self.presentation
        backend = presentation.backend
        sources = backend.domain.sources
        backend.require_sources()
        backend.domain.require_current(sources)
        lifecycle = presentation.view_lifecycle
        checkpoint = lifecycle.capture_display_checkpoint()
        modal_checkpoint = None
        if checkpoint is None:
            modal_checkpoint = self.require_modal_display_checkpoint(identity, attachment_id, visit)
        host = runtime._chat_controller._interrupt_host
        if (
            type(attachment_id) is not str or not attachment_id
            or type(visit) is not int or visit < 1
            or presentation.parent_attachment != (attachment_id, visit)
            or not backend.controls._intake_open
            or identity.source_revision != backend.controls.source_revision
            or (modal_checkpoint is None and not lifecycle.display_checkpoint_current(checkpoint))
            or type(host) is not InterruptRoundHost
            or any(
                getattr_static(host, name) is not record[0]
                or not _definition_current(record)
                for name, record in _HEAD_METHODS
            )
        ):
            raise RuntimeError("Console decision read scope changed")
        # Each existing owner keeps its original lock and FIFO. There is no new
        # merged queue or timer; capture also pauses an obsolete project clock.
        project = self.capture(identity)
        decision = runtime._chat_controller.pending_decision_projection(identity.session_id)
        backend.require_sources()
        backend.domain.require_current(sources)
        self._require()
        if modal_checkpoint is not None and self.require_modal_display_checkpoint(
            identity, attachment_id, visit, _captured=modal_checkpoint,
        ) is not modal_checkpoint:
            raise RuntimeError("Console modal decision checkpoint changed")
        if (
            presentation.parent_attachment != (attachment_id, visit)
            or (modal_checkpoint is None and not lifecycle.display_checkpoint_current(checkpoint))
            or runtime._chat_controller._interrupt_host is not host
            or any(getattr_static(host, name) is not record[0]
                   or not _definition_current(record) for name, record in _HEAD_METHODS)
            or identity.source_revision != backend.controls.source_revision
            or runtime._chat_store.active_session_id != identity.session_id
            or runtime._chat_store._sessions[identity.session_id].incarnation_id != identity.incarnation_id
        ):
            raise RuntimeError("Console decision read changed during capture")
        return ConsoleDecisionHead(
            identity.source_revision, identity.session_id, identity.incarnation_id,
            attachment_id, visit, decision, project,
        )

    def present(self, identity, attachment_id, visit, decision_id, action) -> bool:
        """Apply one actual modal mount/hidden checkpoint; never resolve a choice."""
        runtime = self._require()
        attachment = (attachment_id, visit)
        if (
            type(attachment_id) is not str or not attachment_id
            or type(visit) is not int or visit < 1
            or type(decision_id) is not str or not decision_id
        ):
            raise ValueError("Invalid Console project presentation identity")
        if action == "hidden":
            with runtime._project_decision_lock:
                head = self._head_locked(runtime, identity.session_id)
                if (
                    self.presentation.parent_attachment == attachment
                    and head is not None and head.decision_id == decision_id
                ):
                    self._hide_console_view(attachment, head)
                if self._shown is not None and (
                    self._shown[:2] == attachment
                    and self._shown[2].decision_id == decision_id
                    and self._shown[2].session_id == identity.session_id
                ):
                    runtime._pause_project_dispatch_locked(self._shown[2])
                    self._shown = None
                    return True
            return False
        if action != "mounted":
            raise ValueError("Invalid Console project presentation action")
        if (
            not self.presentation.backend.controls._intake_open
            or identity.source_revision != self.presentation.backend.controls.source_revision
        ):
            return False
        if self.presentation.parent_attachment != attachment:
            return False
        checkpoint = self.presentation.view_lifecycle._current
        if checkpoint is None:
            return False
        self.presentation.require_view_checkpoint(checkpoint)
        current = self.capture(identity)
        if current is None or current.decision_id != decision_id:
            return False
        with runtime._project_decision_lock:
            head = self._head_locked(runtime, identity.session_id)
            if head is None or head.decision_id != decision_id:
                return False
            if checkpoint.action == "hidden" and (
                self._hidden_checkpoint is None
                or self._hidden_checkpoint[:2] != attachment
                or self._hidden_checkpoint[2] is not head
                or self._hidden_checkpoint[3] is not checkpoint
            ):
                return False
            if self._shown == (*attachment, head):
                return True
            if self._shown is not None:
                runtime._pause_project_dispatch_locked(self._shown[2])
            self._hide_console_view(attachment, head)
            self._shown = (*attachment, head)
            if type(head) is runtime_source._ProjectDispatchDecision:
                head.answerable_since = time.monotonic()
                head.ever_answerable = True
            return True


_RUNTIME_METHODS = tuple(
    (name, _definition(getattr(runtime_source.ConsoleRuntime, name)))
    for name in (
        "_pause_project_dispatch_locked", "_project_decision_for_id_locked",
        "resolve_project_instruction_dispatch", "resolve_project_instruction_binding",
    )
)
_CONTROLLER_METHODS = tuple(
    (name, _definition(getattr(ConsoleChatController, name)))
    for name in ("on_console_view_visibility_changed", "set_answerable_decision", "pending_decision_projection")
)
_HEAD_METHODS = tuple(
    (name, _definition(getattr(InterruptRoundHost, name)))
    for name in ("pending_decision_projection", "_pending_round_states_snapshot", "_pending_decision_payloads_locked")
)
_METHODS = tuple(
    (name, _definition(getattr(ConsoleBackendProjectDecisions, name)))
    for name in (
        "_require", "_head_locked", "pause", "capture", "read_head", "present",
        "_hide_console_view", "require_modal_display_checkpoint",
    )
)
