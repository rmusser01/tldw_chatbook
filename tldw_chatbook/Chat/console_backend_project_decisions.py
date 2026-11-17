"""Actual parent-modal visibility for the original runtime's project decisions."""

from __future__ import annotations

import time
from inspect import getattr_static

from . import console_runtime as runtime_source
from .console_chat_controller import ConsoleChatController
from .console_domain_services import _definition, _definition_current
from .console_execution_project_values import ConsoleProjectDecisionHead


class ConsoleBackendProjectDecisions:
    """Keep only the currently shown original decision and its actual visit."""

    def __init__(self, presentation) -> None:
        self.presentation = presentation
        self._shown = None

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
                getattr_static(type(self), name) is not record[0]
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
                    runtime._chat_controller.on_console_view_visibility_changed(False)
                    runtime._chat_controller.set_answerable_decision(identity.session_id, None)
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
        if checkpoint is None or checkpoint.action == "hidden":
            return False
        current = self.capture(identity)
        if current is None or current.decision_id != decision_id:
            return False
        with runtime._project_decision_lock:
            head = self._head_locked(runtime, identity.session_id)
            if head is None or head.decision_id != decision_id:
                return False
            if self._shown == (*attachment, head):
                return True
            if self._shown is not None:
                runtime._pause_project_dispatch_locked(self._shown[2])
            runtime._chat_controller.on_console_view_visibility_changed(False)
            runtime._chat_controller.set_answerable_decision(identity.session_id, None)
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
    for name in ("on_console_view_visibility_changed", "set_answerable_decision")
)
_METHODS = tuple(
    (name, _definition(getattr(ConsoleBackendProjectDecisions, name)))
    for name in ("_require", "_head_locked", "pause", "capture", "present")
)
