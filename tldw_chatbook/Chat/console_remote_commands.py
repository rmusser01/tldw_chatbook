"""Command-owned input and explicit targets over the native Console owners."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING

from .console_chat_models import ConsoleWorkspaceContext
from .console_remote_models import ConsoleHostIdentity, ConsoleTarget

if TYPE_CHECKING:
    from .console_runtime import ConsoleRuntime


REVISION_FIELDS = (
    "settings_revision",
    "generation_settings_revision",
    "context_policy_revision",
    "identity_revision",
    "conversation_binding_revision",
)


class ConsoleRemoteCommands:
    """Route commands without borrowing a view, composer or execution authority."""

    def __init__(
        self, runtime: ConsoleRuntime, host_identity: ConsoleHostIdentity
    ) -> None:
        self.runtime = runtime
        self.host_identity = host_identity

    def _resolve(self, target: ConsoleTarget):
        if not isinstance(target, ConsoleTarget) or any(
            getattr(target, key) != getattr(self.host_identity, key)
            for key in ("server_epoch", "console_id")
        ):
            raise ValueError("Console target owner changed.")
        runtime = self.runtime
        runtime._raise_if_disposed_or_session_fenced(target.session_id)
        owner = runtime._execution_owner
        if owner is not None:
            owner.require_current(owner._path)
        store, controller = runtime._chat_store, runtime._chat_controller
        if store is None or controller is None or controller.store is not store:
            raise RuntimeError("Console owner is unavailable.")
        session = next(
            (row for row in store.sessions() if row.id == target.session_id), None
        )
        if session is None or session.incarnation_id != target.incarnation_id:
            raise ValueError("Console target incarnation changed.")
        if controller._disposed or controller._shutdown_requested.is_set():
            raise RuntimeError("Console owner is shutting down.")
        return store, controller, session

    @staticmethod
    def _revisions(session, expected: dict[str, int]) -> None:
        if (
            type(expected) is not dict
            or set(expected) != set(REVISION_FIELDS)
            or any(type(value) is not int or value < 0 for value in expected.values())
        ):
            raise ValueError("Expected all five non-negative native revisions.")
        if any(getattr(session, key) != expected[key] for key in REVISION_FIELDS):
            raise ValueError(
                "Console target configuration changed; refresh before Send."
            )

    async def send(
        self,
        target: ConsoleTarget,
        text: str,
        command_id: str,
        *,
        expected_revisions: dict[str, int],
    ) -> str:
        """Receive immutable text; the returned native turn is still Preparing."""
        from .console_configuration_preparation import (
            ConsoleTurnCaptureSelection,
            capture_console_received_configuration_preparation,
        )
        from .console_received_intent import ConsoleReceivedTurnIntent

        store, controller, session = self._resolve(target)
        if target.handle_id is not None or target.round_id is not None:
            raise ValueError("Send requires a chat target.")
        self._revisions(session, expected_revisions)
        inputs = store.command_input_snapshot(
            session.id, text=text, command_id=command_id
        )
        activity = controller.activity_for(session.id)
        queue = controller.prompt_queue_registry.snapshot(session.id)
        if activity.preparing_before_acceptance and not queue.total_count:
            raise RuntimeError("Preparing the current turn; Send is not available yet.")
        selection = ConsoleTurnCaptureSelection(
            provider_selection=replace(
                controller._provider_selection_for_session(session.id),
                workspace_context=ConsoleWorkspaceContext(
                    active_workspace_id=store.session_workspace_id(session.id)
                ),
            ),
            presentation_context=controller._presentation_context_for(session.id),
            rag_defaults=None,
            tool_configuration=None,
            skill_workspace_id=store.session_workspace_id(session.id),
            project_bindings_eligible=controller._agent_dispatch_is_eligible(
                session, prefill=None
            ),
            agent_runtime_enabled=controller._agent_runtime_enabled,
        )
        preparation = capture_console_received_configuration_preparation(
            controller.app, store, controller, session_id=session.id
        )
        if preparation is None:
            raise RuntimeError("Console configuration source is unavailable.")
        self._resolve(target)
        self._revisions(session, expected_revisions)
        intent = ConsoleReceivedTurnIntent(
            turn_id=command_id,
            session_id=session.id,
            inputs=inputs,
            selection=selection,
            staged_evidence_launch=None,
            staged_evidence_revision=0,
            view_attachment_generation=0,
            queue_revision=(
                queue.revision
                if activity.accepted_live_turn or queue.total_count
                else None
            ),
        )
        return self.runtime.accept_received_intent(
            intent, _configuration_preparation=preparation
        )

    async def execute(
        self,
        action: str,
        target: ConsoleHostIdentity | ConsoleTarget,
        payload: dict,
        command_id: str,
    ) -> dict:
        """Return body-free native submission facts; queued never means consumed."""
        from .console_turn_preparation import _validate_identifier

        try:
            _validate_identifier(command_id, "command ID")
            if type(payload) is not dict:
                raise ValueError("Command payload must be an object.")
            _store, controller, session = self._resolve(target)
            if action == "decision":
                if set(payload) != {"kind", "answer"} or not isinstance(
                    payload["kind"], str
                ):
                    raise ValueError("Decision requires kind and answer only.")
                loop = asyncio.get_running_loop()

                def on_outcome(outcome):
                    state = (
                        "applied"
                        if outcome == "decided"
                        else "outcome_unknown"
                        if outcome == "unavailable"
                        else "refused"
                    )
                    loop.call_soon_threadsafe(
                        lambda: self.runtime._observe_remote_custody(
                            command_id,
                            state=state,
                            native_id=target.round_id,
                            reason="decision_" + outcome,
                        )
                    )

                return controller._interrupt_host.resolve_remote_round(
                    target,
                    kind=payload["kind"],
                    answer=payload["answer"],
                    _on_outcome=on_outcome,
                )
            if action == "send":
                if set(payload) != {"text", "expected_revisions"}:
                    raise ValueError("Send requires text and expected_revisions only.")
                turn_id = await self.send(
                    target,
                    payload["text"],
                    command_id,
                    expected_revisions=payload["expected_revisions"],
                )
                return {"state": "pending", "native_id": turn_id, "reason": "Preparing"}
            if action not in {
                "steer",
                "redirect",
                "stop",
                "subagent_guidance",
                "subagent_stop",
            }:
                raise ValueError("Unsupported Console command.")
            text_action = action in {"steer", "redirect", "subagent_guidance"}
            if set(payload) != ({"text"} if text_action else set()):
                raise ValueError("Unexpected Console command fields.")
            if text_action and not isinstance(payload["text"], str):
                raise ValueError("Command text must be a string.")
            if target.round_id is not None:
                raise ValueError("Agent controls cannot target a decision round.")
            if action.startswith("subagent_"):
                bridge = controller._agent_bridge
                if not target.handle_id or not target.run_id or bridge is None:
                    raise ValueError(
                        "Subagent command requires an exact handle and run."
                    )
                conversation_id = session.persisted_conversation_id
                if not conversation_id:
                    raise ValueError("Subagent conversation is unavailable.")
                if action == "subagent_guidance":
                    applied = bridge.steer_subagent(
                        conversation_id,
                        target.handle_id,
                        payload["text"],
                        expected_run_id=target.run_id,
                    )
                    state, reason = (
                        "queued",
                        "Guidance queued; consumption is not confirmed.",
                    )
                else:
                    applied = bridge.cancel_subagent(
                        conversation_id, target.handle_id, expected_run_id=target.run_id
                    )
                    state, reason = (
                        "accepted",
                        "Stop requested; physical exit is not confirmed.",
                    )
                if not applied:
                    raise ValueError(
                        "Subagent target is no longer available or refused the command."
                    )
                return {"state": state, "native_id": target.run_id, "reason": reason}
            if target.handle_id is not None:
                raise ValueError("Main controls require a primary target.")
            if (
                target.turn_id is not None
                and controller._current_control_turn_id(session.id) != target.turn_id
            ):
                raise ValueError("Console target turn changed.")
            if action == "stop":
                applied = controller.stop_run(
                    session.id,
                    expected_run_id=target.run_id,
                    expected_turn_id=target.turn_id,
                )
                if not applied:
                    raise ValueError("Exact turn or run is no longer stoppable.")
                return {
                    "state": "accepted",
                    "native_id": target.run_id or target.turn_id,
                    "reason": "Stop requested; physical exit is not confirmed.",
                }
            if not target.run_id:
                raise ValueError("Main guidance requires an exact run.")
            method = (
                controller.steer_run if action == "steer" else controller.redirect_run
            )
            refusal = method(session.id, payload["text"], expected_run_id=target.run_id)
            if refusal is not None:
                return {"state": "refused", "reason": refusal}
            return {
                "state": "queued",
                "native_id": target.run_id,
                "reason": "Correction queued; consumption is not confirmed.",
            }
        except (ValueError, TypeError, RuntimeError, PermissionError) as error:
            return {"state": "refused", "reason": str(error)}
