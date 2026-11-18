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
        from .console_backend_project_decisions import ConsoleBackendProjectDecisions

        self.project_decisions = ConsoleBackendProjectDecisions(self)
        self._projection_views = {}
        self._marker_projection = None
        self._marker_session = None
        from .console_backend_readiness import ConsoleBackendReadiness

        self.readiness = ConsoleBackendReadiness(backend)
        from .console_backend_tool_counts import ConsoleBackendToolCounts

        self.tool_counts = ConsoleBackendToolCounts(self)
        from .console_backend_inspector_scopes import ConsoleBackendInspectorScopes

        self.inspector_scopes = ConsoleBackendInspectorScopes(backend)

    def admit_view_checkpoint(self, message):
        """Register one closed parent lifecycle request before its task can await."""
        self.require_association(self.association)
        if message.operation != "ReadProjection":
            return None
        value = message.payload.get("view_checkpoint")
        if value is None:
            from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle, _METHODS
            from .console_domain_services import _definition_current

            # Capture before exposing the actual control Task. A modal can
            # revoke visibility before that already-received Task first runs.
            lifecycle = self.view_lifecycle
            record = dict(_METHODS)["capture_display_checkpoint"]
            if (
                type(lifecycle) is not ConsoleBackendViewLifecycle
                or lifecycle.presentation is not self
                or getattr_static(lifecycle, "capture_display_checkpoint", None) is not record[0]
                or not _definition_current(record)
            ):
                raise RuntimeError("Console display checkpoint capture changed")
            checkpoint = record[0](lifecycle)
            if checkpoint is None:
                current = self.view_lifecycle._current
                if current is not None and current.action == "hidden":
                    raise RuntimeError("Console hidden view cannot start a display read")
                return None
        else:
            checkpoint = self.view_lifecycle.admit(
                value["attachment_id"], value["visit"], value["action"],
            )
            if value["action"] in {"mount", "hidden"}:
                self.project_decisions.pause()
        self._view_checkpoints[message.identity.request_id] = (
            message.identity, checkpoint, value is not None,
        )
        return checkpoint

    def release_view_checkpoint(self, identity, checkpoint) -> None:
        entry = self._view_checkpoints.get(identity.request_id)
        if entry is not None and entry[0] is identity and entry[1] is checkpoint:
            del self._view_checkpoints[identity.request_id]

    def require_view_checkpoint(self, checkpoint) -> None:
        """Validate this exact original token without renewing visibility."""
        from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle, _METHODS
        from .console_domain_services import _definition_current

        self.require_association(self.association)
        lifecycle = self.view_lifecycle
        record = dict(_METHODS)["_require_checkpoint"]
        if (
            type(lifecycle) is not ConsoleBackendViewLifecycle
            or lifecycle.presentation is not self
            or getattr_static(lifecycle, "_require_checkpoint", None) is not record[0]
            or not _definition_current(record)
        ):
            raise RuntimeError("Console view checkpoint validator changed")
        record[0](lifecycle, checkpoint)

    async def prepare_view_checkpoint(self, identity):
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
            if entry[2]:
                await self.view_lifecycle.prepare(entry[1])
            else:
                self.require_view_checkpoint(entry[1])
            self.require_association(self.association)
            return entry[1]
        finally:
            self.release_view_checkpoint(entry[0], entry[1])

    async def drain_view_lifecycle(self) -> None:
        """Join retained original mount seeds before the backend can close."""
        from types import MethodType
        from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle, _METHODS
        from .console_domain_services import _definition_current

        if self.project_decisions._shown is not None:
            self.project_decisions.pause()
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
        from .console_backend_inspector_scopes import ConsoleBackendInspectorScopes, _SOURCES
        from .console_domain_services import _definition_current

        self.require_association(self.association)
        if type(self.inspector_scopes) is not ConsoleBackendInspectorScopes or any(
            getattr_static(self.inspector_scopes, name, None) is not descriptor
            or not _definition_current(record)
            for owner, name, descriptor, record in _SOURCES
            if owner is ConsoleBackendInspectorScopes
        ):
            raise RuntimeError("Console Inspector scope producer changed")
        await self.inspector_scopes.warm(identity)
        await self.readiness.warm_workspace_counts(identity)
        await self.readiness.warm(identity)

    async def prepare_run_history(self, identity):
        """Read original run metadata only for this explicit projection request."""
        from .console_execution_protocol import encode_message
        from .console_domain_services import _definition_current

        self.require_association(self.association)
        message = self.backend.adapter._receipts[identity.request_id][0]
        if message.operation != "ReadProjection":
            raise RuntimeError("Console history request owner changed")
        request = encode_message(message)["payload"].get("run_history")
        if request is None:
            return None
        if (
            _history_module.ConsoleBackendRunHistoryRead is not _HistoryRead
            or any(
                getattr(_HistoryRead, name) is not record[0]
                or not _definition_current(record)
                for name, record in _HISTORY_METHODS
            )
        ):
            raise RuntimeError("Console original history producer changed")
        return await _HistoryRead(self, identity, request).run()

    def _require_marker_projection(self, identity, projection) -> None:
        from types import MethodType
        from .console_domain_services import _definition_current

        self.require_association(self.association)
        self.backend._require_owner_loop()
        session = self._controller.store._sessions.get(identity.session_id)
        if (
            session is not self._marker_session
            or session is None
            or session.incarnation_id != identity.incarnation_id
            or projection is not self._marker_projection
            or type(projection) is not _MarkerProjection
            or _marker_module.ConsoleChangeReviewProjection is not _MarkerProjection
            or _MarkerProjection._same_selection is not _MARKER_SAME[0]
            or not _definition_current(_MARKER_SAME)
            or any(getattr(module, name) is not record[0] or not _definition_current(record)
                   for module, name, record in _MARKER_HELPERS)
        ):
            raise RuntimeError("Console marker projection owner changed")
        for name, record in _MARKER_METHODS:
            method = getattr(projection, name, None)
            if (
                type(method) is not MethodType or method.__self__ is not projection
                or method.__func__ is not record[0]
                or not _definition_current(record)
            ):
                raise RuntimeError("Console marker projection definition changed")

    async def prepare_transcript(self, identity):
        """Prepare original Change Review markers before the message snapshot."""
        self.require_association(self.association)
        session = self._controller.store._sessions.get(identity.session_id)
        if session is None or session.incarnation_id != identity.incarnation_id:
            raise RuntimeError("Console marker session changed")
        if self._marker_session is not session:
            from .console_domain_services import _definition_current

            if (_marker_module.ConsoleChangeReviewProjection is not _MarkerProjection
                or any(getattr(_MarkerProjection, name) is not record[0]
                       or not _definition_current(record)
                       for name, record in _MARKER_METHODS)):
                raise RuntimeError("Console marker constructor source changed")
            self._marker_projection = _MarkerProjection(
                runtime_accessor=lambda: self._runtime,
                conversation_id_accessor=lambda: session.persisted_conversation_id,
            )
            self._marker_session = session
        projection = self._marker_projection
        self._require_marker_projection(identity, projection)
        if not await projection.prepare():
            raise RuntimeError("Console marker preparation became obsolete")
        self._require_marker_projection(identity, projection)
        selected = projection._selection()
        # The stock worker owns saved marker reads. Never fall through to the
        # original synchronous/custom reader on this process presentation path.
        if selected[1] is not None and selected[4] is not None and not projection._prepared_stock:
            raise RuntimeError("Console marker worker source is unavailable")
        return projection, selected

    def project_transcript(self, identity, prepared, messages):
        projection, selected = prepared
        self._require_marker_projection(identity, projection)
        if not projection._same_selection(selected, projection._selection()):
            raise RuntimeError("Console marker publication changed")
        rows = tuple(projection.project(list(messages)))
        if not projection._same_selection(selected, projection._selection()):
            raise RuntimeError("Console marker publication changed")
        return rows, selected[5]

    def register_marker_read(self, source, read):
        """Enroll the actual original marker worker in this command's effect."""
        import asyncio
        from types import MethodType
        from .console_change_review_read import _MarkerRead
        from .console_domain_services import _definition_current
        from .console_execution_retirement import _ReadResources

        self._require_marker_projection(source.identity, read.creator)
        controls = self.backend.controls
        callback = read.callback
        if (
            source.controls is not controls
            or read.task is not asyncio.current_task()
            or controls._settings_command_tasks.get(read.task) is not source
            or read not in read.creator._preparation_reads
            or type(callback) is not MethodType
            or type(callback.__self__) is not _MarkerRead
            or callback.__func__ is not dict(_MARKER_READ_METHODS)["read"][0]
            or callback.__self__.runtime is not self._runtime
            or callback.__self__.conversation_id != self._marker_session.persisted_conversation_id
        ):
            raise RuntimeError("Console original marker handoff changed")
        for name, record in _MARKER_READ_METHODS:
            method = getattr(callback.__self__, name, None)
            if (type(method) is not MethodType or method.__self__ is not callback.__self__
                or method.__func__ is not record[0] or not _definition_current(record)):
                raise RuntimeError("Console original marker reader changed")
        observation = _ReadResources(source, read)
        observation.effect = controls._begin_effect("transcript_marker_read", observation, source)
        return observation

    def capture_projection_view(self, controls, identity, projection) -> None:
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
            ConsoleWorkspaceCount, ConsoleWorkspaceReceipt,
            ConsoleWorkspaceSessionFacts, ConsoleWorkspaceSurface, encode_workspace_surface,
        )
        from .console_chat_store import ConsoleChatSession
        from .console_domain_services import _definition_current
        from types import MethodType

        bridge = self._runtime._agent_bridge
        if (getattr_static(type(self._runtime), "activity_receipts") is not _WORKSPACE_RECEIPTS_PROPERTY
            or not _definition_current(_WORKSPACE_RECEIPTS_GETTER)):
            raise RuntimeError("Console workspace receipts getter changed")
        receipts = self._runtime.activity_receipts
        receipt_reader = _WORKSPACE_RECEIPTS
        delegate = None
        if type(receipts) is _LazyConsoleActivityReceiptService:
            receipt_reader = _LAZY_WORKSPACE_RECEIPTS
            delegate = receipts._delegate
        elif receipts is not None and type(receipts) is not ConsoleActivityReceiptService:
            raise RuntimeError("Console workspace receipts owner changed")
        if delegate is not None and type(delegate) is not ConsoleActivityReceiptService:
            raise RuntimeError("Console workspace receipts delegate changed")
        for receiver, name, record in (
            (bridge, "progress_counts", _WORKSPACE_PROGRESS),
            (receipts, "unseen_snapshot", receipt_reader),
            (delegate, "unseen_snapshot", _WORKSPACE_RECEIPTS),
        ):
            if receiver is None:
                continue
            method = getattr(receiver, name, None)
            if (type(method) is not MethodType or method.__self__ is not receiver
                or method.__func__ is not record[0] or not _definition_current(record)):
                raise RuntimeError("Console workspace display source changed")
        progress_counts = bridge.progress_counts() if bridge is not None else {}
        unseen_receipts = receipts.unseen_snapshot() if receipts is not None else ()
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
                attention=tuple(self._controller.conversation_attention_for(row.id)),
            ))
        from .console_execution_project_values import encode_project_head

        project_head = encode_project_head(self.project_decisions.capture(identity))
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
                    cost=self.readiness.context_summary.cost,
                    cost_cache_state=self.readiness.context_summary.cost_cache_state,
                ),
                "agent_rail": encode_agent_rail_projection(
                    self.readiness.agent_projection(store._sessions[identity.session_id])
                ),
                "prompt_history": self.readiness.prompt_history_payload,
                "send_selection": self.readiness.send_selection,
                "project_head": project_head,
                "provider_label": provider,
                "control_provider_label": provider_chip,
                "model": model,
                "remaining_seconds": max(0.0, 1.0 - (captured_at - self.readiness.at)),
                "captured_at_monotonic": captured_at,
            },
            "composition": composition,
            "workspace": encode_workspace_surface(ConsoleWorkspaceSurface(
                active_workspace_id=store.workspace_context.active_workspace_id,
                controller_present=True, sessions=tuple(workspace_facts),
                progress_counts=tuple(ConsoleWorkspaceCount(key, value)
                                      for key, value in progress_counts.items()),
                subagent_counts=tuple(ConsoleWorkspaceCount(key, value)
                                      for key, value in self.readiness.workspace_counts.items()),
                subagent_count_ids=self.readiness.workspace_count_ids,
                unseen_receipts=tuple(ConsoleWorkspaceReceipt(
                    item.conversation_id, item.session_id, item.status,
                ) for item in unseen_receipts),
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
        from .console_chat_models import FEEDBACK_ACTIVE_RUN_STATUSES
        from .console_execution_composition_values import encode_session_composition
        from .console_control_dispatch import require_control_source
        from .console_domain_services import _definition_current
        from .console_prompt_queue import ConsolePromptQueueRegistry, PromptQueueMode, PromptQueuePauseReason
        from .console_prompt_queue_coordinator import ConsolePromptQueueCoordinator
        from .console_fleet_wake import ConsoleFleetWakeCoordinator
        from .console_execution_queue_values import ConsoleQueueDisplay

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
            or type(controller.prompt_queue_registry) is not ConsolePromptQueueRegistry
            or type(controller.prompt_queue_coordinator) is not ConsolePromptQueueCoordinator
            or controller.prompt_queue_coordinator.registry is not controller.prompt_queue_registry
            or type(controller._fleet_wake) is not ConsoleFleetWakeCoordinator
            or controller._fleet_wake._controller is not controller
            or (controller._hooks_v2_runtime is not None and controller._hooks_v2_runtime is not self._runtime)
        ):
            raise RuntimeError("Console composition owner changed")
        for receiver, names in (
            (controller, _COMPOSITION_CONTROLLER_METHODS),
            (controller.store, _COMPOSITION_STORE_METHODS),
            (controller._chat_start, _COMPOSITION_START_METHODS),
            (self._runtime, _COMPOSITION_RUNTIME_METHODS),
            (controller.prompt_queue_registry, _QUEUE_REGISTRY_METHODS),
            (controller.prompt_queue_coordinator, _QUEUE_COORDINATOR_METHODS),
            (controller._fleet_wake, _QUEUE_WAKE_METHODS),
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
        activity = controller.activity_for(session.id)
        blocked = _COMPOSITION_COPY[1][2][0](controller)
        if (
            _recovery_values.capture_recovery_display is not _RECOVERY_CAPTURE[0]
            or not _definition_current(_RECOVERY_CAPTURE)
        ):
            raise RuntimeError("Console recovery capture changed")
        recovery = _RECOVERY_CAPTURE[0](controller, session)
        rules = controller.response_rules
        rule_state = None
        if rules is not None:
            read_state = getattr(rules, "state", None)
            if (
                type(rules) is not _ResponseRuleRuntime
                or rules is not self._runtime._response_rules
                or type(read_state) is not MethodType
                or read_state.__self__ is not rules
                or read_state.__func__ is not _RULE_STATE[0]
                or not _definition_current(_RULE_STATE)
            ):
                raise RuntimeError("Console response-rule display owner changed")
            rule_state = read_state(session.id)
        for module, name, record in _RUN_DISPLAY_HELPERS:
            if getattr(module, name) is not record[0] or not _definition_current(record):
                raise RuntimeError("Console run display source changed")
        run_state = controller.run_state_for(session.id)
        for name, record in _QUEUE_HELPERS:
            if getattr(_queue_display, name) is not record[0] or not _definition_current(record):
                raise RuntimeError("Console queue display source changed")
        queue_snapshot = controller.prompt_queue_registry.snapshot(session.id)
        recoveries = self._runtime.recoveries_for_session(session.id)
        failed_turn = (
            _queue_display.read_queue_recovery_turn(controller.store, session.id, action="retry-failed")
            if queue_snapshot.mode is PromptQueueMode.PAUSED
            and queue_snapshot.pause_reason is PromptQueuePauseReason.FAILED else None
        )
        queue = ConsoleQueueDisplay(
            queue_snapshot, activity,
            controller.prompt_queue_coordinator.dispatch_recovery_blocks_queue(session.id),
            controller.store.dispatch_recovery_blocks_submission(session.id),
            recoveries[0].turn_id if recoveries else None,
            recoveries[0].reason if recoveries else "",
            failed_turn.preview if failed_turn is not None else None,
            bool(controller._hooks_v2_runtime is not None and self._runtime.has_received_intents(
                session.id, unpromoted_only=True,
            )),
            controller._chat_start.is_prepared(session.id),
            controller._chat_start.is_accepted(session.id),
            run_state.is_stop_allowed,
            session.id in controller._fleet_wake.delivering_session_ids(),
        )
        feedback = _run_display.console_send_activity_copy(run_state)
        if run_state.status.value == "streaming":
            bridge_read = controller._agent_bridge.live_snapshot
            if (type(bridge_read) is not MethodType
                or bridge_read.__self__ is not controller._agent_bridge
                or bridge_read.__func__ is not _RUN_SNAPSHOT[0]
                or not _definition_current(_RUN_SNAPSHOT)):
                raise RuntimeError("Console run snapshot source changed")
            feedback = _run_display.console_send_activity_copy(
                run_state,
                snapshot=bridge_read(str(session.persisted_conversation_id or "")),
                reply_started=controller.store.has_live_reply_output(session.id),
            )
        control = session.project_instruction_state
        project_status = _composition_display.build_console_project_instruction_state(
            control,
            binding_label=control.working_folder_binding_id or "",
            activation_events=controller.project_instruction_activation_events(session.id),
        )
        effective = self.readiness.value[0]
        vision = self.readiness.vision_capability
        return encode_session_composition(
            session,
            pending_approval_count=controller.pending_round_count(session.id),
            pending_interrupt_copy=(
                _COMPOSITION_COPY[0][2][0](controller, session.id) if pending else ""
            ),
            response_rule_phase=rule_state.phase if rule_state is not None else None,
            response_rule_copy=_rule_display.rule_status(rule_state) if rule_state is not None else "",
            run_feedback_copy=feedback,
            blocked_turn_reason=blocked,
            is_stop_allowed=controller.is_stop_allowed,
            acknowledgement_idle=not (
                activity.accepted_live_turn or activity.occupies_slot or activity.queued_count
                or controller.run_state_for(session.id).is_stop_allowed or blocked
            ),
            draft_spend_run_active=(
                run_state.status in FEEDBACK_ACTIVE_RUN_STATUSES
                or self._runtime.has_custodied_turns(session.id)
            ),
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
            project_status=project_status,
            recovery=recovery,
            queue=queue,
            vision_capable=(vision[2] if vision is not None
                and vision[:2] == (effective.provider, effective.model) else None),
            inspector_scope=self.inspector_scopes.current_value(
                require_control_source(self._runtime).identity
            ),
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
        self._require_association_graph(association)
        self.backend.require_sources()

    def _require_association_graph(self, association) -> None:
        """Keep the original structural checks during an issued source Apply."""
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

    def capture_display_reuse_checkpoint(self, owner):
        """Qualify completed remote visibility, never an attached Screen."""
        from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle, _METHODS
        from .console_domain_services import _definition_current

        self._require_display_owner(owner)
        lifecycle = self.view_lifecycle
        record = dict(_METHODS)["capture_display_checkpoint"]
        if (type(lifecycle) is not ConsoleBackendViewLifecycle
            or lifecycle.presentation is not self
            or getattr_static(lifecycle, "capture_display_checkpoint", None) is not record[0]
            or not _definition_current(record)):
            raise RuntimeError("Backend display checkpoint capture changed")
        return record[0](lifecycle)

    def display_reuse_checkpoint_current(self, owner, checkpoint) -> bool:
        """Validate the exact completed lifecycle token alongside FULL guards."""
        from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle, _METHODS
        from .console_domain_services import _definition_current

        self._require_display_owner(owner)
        lifecycle = self.view_lifecycle
        record = dict(_METHODS)["display_checkpoint_current"]
        if (type(lifecycle) is not ConsoleBackendViewLifecycle
            or lifecycle.presentation is not self
            or getattr_static(lifecycle, "display_checkpoint_current", None) is not record[0]
            or not _definition_current(record)):
            raise RuntimeError("Backend display checkpoint validation changed")
        return record[0](lifecycle, checkpoint)

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

    def project_received_input(self, record, *, draft_committed: bool) -> None:
        """Publish the original input CAS checkpoint for its exact received turn."""
        import asyncio
        from .console_control_dispatch import require_control_source
        from .console_execution_worker import _BackendService

        self.require_association(self.association)
        self.backend._require_owner_loop()
        source = require_control_source(self._runtime)
        controls = self.backend.controls
        entry = controls._turns.get(record.turn_id)
        service = self.backend._worker_service
        if (
            type(draft_committed) is not bool
            or entry is None or entry[0] is not source.identity
            or entry[1] is not record or entry[2] is not asyncio.current_task()
            or self._runtime._turn_custody.get(record.turn_id) is not record
            or record.session_id != source.identity.session_id
            or type(service) is not _BackendService
            or service.backend is not self.backend
        ):
            raise RuntimeError("Backend received input checkpoint changed")
        retained = self.backend.adapter._receipts.get(source.identity.request_id)
        if (
            retained is None or retained[0].operation != "ReceiveTurn"
            or retained[0].payload["turn_id"] != record.turn_id
            or retained[1].work is not entry[2]
        ):
            raise RuntimeError("Backend received input receipt changed")
        service.emit(retained[0].identity, "input_projected", "ReceiveTurn", {
            "turn_id": record.turn_id, "draft_committed": draft_committed,
        })

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
        # The original host retains the payload. ReadProjection publishes its
        # exact mixed head; only the real card's checkpoint claims answerability.
        self.notify_decision_change()

    def clear_decision(self, kind) -> None:
        self.present_decision(kind, None)

    def project_pending(self) -> None:
        self.require_association(self.association)
        self.notify_decision_change()

    def notify_decision_change(self) -> None:
        """Wake the remote view without publishing decision values or authority."""
        from .console_execution_worker import _BackendService

        self.require_association(self.association)
        service = self.backend._worker_service
        if service is None:
            return  # Original in-process graph has no remote observer.
        if type(service) is not _BackendService or service.backend is not self.backend:
            raise RuntimeError("Backend decision hint channel changed")
        service.queue_decision_changed(self)

    def announce_hidden(self, message: str) -> bool:
        return self.notice(message)

    def project_attention(self, host, *, ring: bool) -> None:
        self.require_association(self.association)
        self.notify_decision_change()


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
from . import console_display_state as _composition_display

_COMPOSITION_CONTROLLER_METHODS = tuple(
    (name, _definition(getattr(_CompositionController, name)))
    for name in (
        "has_pending_approval_round", "pending_round_count", "pending_round_kinds",
        "trace_call_recovery_preparation", "context_compaction_hold", "preparation_outcome",
        "activity_for", "run_marker_for", "streaming_session_id", "run_state_for",
        "conversation_attention_for",
        "provider_continuation_recovery_message", "provider_continuation_replay_available",
        "provider_continuation_owner_is_live", "trace_call_provider_contact",
        "project_instruction_activation_events",
    )
)
_COMPOSITION_START_METHODS = tuple(
    (name, _definition(getattr(_CompositionStart, name)))
    for name in ("is_prepared", "is_accepted")
)
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
        (_composition_display, "build_console_project_instruction_state"),
    )
)

from .console_runtime import ConsoleRuntime as _CompositionRuntime
from .console_chat_store import ConsoleChatSession as _CompositionSession

_CHARACTER_READER = _definition(_CompositionSession.local_character_id)

from tldw_chatbook.UI.Console_Modules import transcript as _marker_module

_MarkerProjection = _marker_module.ConsoleChangeReviewProjection
_MARKER_METHODS = tuple(
    (name, _definition(value))
    for name, value in vars(_MarkerProjection).items()
    if type(value) is FunctionType and name != "_same_selection"
)
from .console_change_review_read import _MarkerRead

_MARKER_READ_METHODS = tuple(
    (name, _definition(getattr(_MarkerRead, name)))
    for name in ("read", "run", "require_current")
)
_MARKER_SAME = _definition(_MarkerProjection._same_selection)
from . import console_change_review_read as _marker_read_module
from . import console_agent_bridge as _marker_bridge_module

_MARKER_HELPERS = tuple(
    (module, name, _definition(getattr(module, name)))
    for module, name in (
        (_marker_read_module, "capture_marker_read"),
        (_marker_bridge_module, "inject_resume_agent_markers"),
    )
)

_COMPOSITION_RUNTIME_METHODS = tuple(
    (name, _definition(getattr(_CompositionRuntime, name)))
    for name in (
        "snapshot_console_staged_evidence", "_staged_evidence_state", "has_custodied_turns",
        "recoveries_for_session", "has_received_intents",
    )
)

from .console_activity_receipts import ConsoleActivityReceiptService
from .console_runtime import _LazyConsoleActivityReceiptService

_WORKSPACE_PROGRESS = _definition(_CompositionBridge.progress_counts)
_WORKSPACE_RECEIPTS = _definition(ConsoleActivityReceiptService.unseen_snapshot)
_LAZY_WORKSPACE_RECEIPTS = _definition(_LazyConsoleActivityReceiptService.unseen_snapshot)
_WORKSPACE_RECEIPTS_PROPERTY = getattr_static(_CompositionRuntime, "activity_receipts")
_WORKSPACE_RECEIPTS_GETTER = _definition(_WORKSPACE_RECEIPTS_PROPERTY.fget)

from . import console_backend_run_history as _history_module

_HistoryRead = _history_module.ConsoleBackendRunHistoryRead
_HISTORY_METHODS = _history_module._METHODS

from . import console_execution_recovery_values as _recovery_values
from .console_chat_store import ConsoleChatStore as _CompositionStore

_RECOVERY_CAPTURE = _definition(_recovery_values.capture_recovery_display)
_COMPOSITION_STORE_METHODS = tuple(
    (name, _definition(getattr(_CompositionStore, name)))
    for name in ("dispatch_recovery_for_presentation", "dispatch_recovery_for_session",
                 "provider_continuation_recovery_message", "has_live_reply_output",
                 "dispatch_recovery_blocks_submission", "iter_messages_newest_first")
)

from .console_prompt_queue import ConsolePromptQueueRegistry as _QueueRegistry
from .console_prompt_queue_coordinator import ConsolePromptQueueCoordinator as _QueueCoordinator
from .console_fleet_wake import ConsoleFleetWakeCoordinator as _QueueWake
from ..UI.Console_Modules import prompt_queue as _queue_display

_QUEUE_REGISTRY_METHODS = (("snapshot", _definition(_QueueRegistry.snapshot)),)
_QUEUE_COORDINATOR_METHODS = (("dispatch_recovery_blocks_queue", _definition(_QueueCoordinator.dispatch_recovery_blocks_queue)),)
_QUEUE_WAKE_METHODS = (("delivering_session_ids", _definition(_QueueWake.delivering_session_ids)),)
_QUEUE_HELPERS = tuple(
    (name, _definition(getattr(_queue_display, name)))
    for name in ("read_queue_recovery_turn", "derive_prompt_queue_presentation", "with_prompt_queue_start_state")
)

from .response_rules.runtime import ResponseRuleRuntime as _ResponseRuleRuntime
from ..UI.Console_Modules import response_rules as _rule_display
from ..UI.Console_Modules import agent as _run_display

_RULE_STATE = _definition(_ResponseRuleRuntime.state)
_RUN_SNAPSHOT = _definition(_CompositionBridge.live_snapshot)
_RUN_DISPLAY_HELPERS = tuple(
    (module, name, _definition(getattr(module, name)))
    for module, name in (
        (_rule_display, "rule_status"), (_rule_display, "reason_copy"),
        (_run_display, "console_send_activity_copy"),
    )
)
