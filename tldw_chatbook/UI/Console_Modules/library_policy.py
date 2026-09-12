"""Controller for per-conversation Console Library policy UI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tldw_chatbook.Chat.console_display_state import (
    ConsoleLibraryPolicyDisplayState,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicySnapshot,
    ConsoleLibraryPolicyWriteStatus,
)
from tldw_chatbook.Widgets.Console.console_library_access_modal import (
    ConsoleLibraryAccessModal,
    ConsoleLibraryPolicySaveOutcome,
)


def _destination_copy(session: Any) -> str:
    destination = session.library_destination_runtime.resolved_destination
    if destination is None:
        return "Resolved destination: not used yet"
    egress = str(destination.egress_class.value).replace("_", " ")
    model = f" / {destination.model}" if destination.model else ""
    return (
        f"Resolved destination: {destination.provider}{model} · {egress} · "
        f"{destination.endpoint_identity}"
    )


class ConsoleLibraryPolicyController:
    """Bind the access modal to the active session's holder/coordinator."""

    def __init__(
        self,
        *,
        app_instance: Any,
        active_session: Callable[[], Any | None],
        ensure_store: Callable[[], Any],
        direct_library_tools: Callable[[], bool],
        push_screen: Callable[[Any], Any],
        request_control_bar_sync: Callable[[], None],
    ) -> None:
        self.app_instance = app_instance
        self._active_session = active_session
        self._ensure_store = ensure_store
        self._direct_library_tools = direct_library_tools
        self._push_screen = push_screen
        self._request_control_bar_sync = request_control_bar_sync

    def _display_state(
        self,
        session: Any,
    ) -> ConsoleLibraryPolicyDisplayState:
        mode = "Direct" if self._direct_library_tools() else "RAG"
        return ConsoleLibraryPolicyDisplayState.from_snapshot(
            session.library_policy_holder.snapshot,
            provider_intent_label=f"Library tool mode: {mode}",
            resolved_destination_label=_destination_copy(session),
        )

    def open_access(self) -> None:
        """Open the policy-only modal for the currently active session."""
        session = self._active_session()
        if session is None:
            self.app_instance.notify(
                "Open a Console conversation before editing Library access.",
                severity="warning",
            )
            return
        snapshot = session.library_policy_holder.snapshot
        self._push_screen(
            ConsoleLibraryAccessModal(
                snapshot=snapshot,
                state=self._display_state(session),
                save_policy=self._save,
                reload_policy=self._reload,
            )
        )

    async def _save(
        self,
        candidate: ConsoleLibraryPolicyCandidate,
    ) -> ConsoleLibraryPolicySaveOutcome:
        session = self._active_session()
        if session is None:
            return ConsoleLibraryPolicySaveOutcome(
                status="unavailable",
                snapshot=_unavailable_snapshot(),
                copy="The active conversation is no longer available.",
            )
        store = self._ensure_store()
        session_id = session.id
        prior_snapshot = session.library_policy_holder.snapshot
        prior_staged = session.library_policy_holder.explicitly_staged
        store.stage_session_library_policy(session_id, candidate)

        if session.ephemeral or session.persisted_conversation_id is None:
            self._request_control_bar_sync()
            return ConsoleLibraryPolicySaveOutcome(
                status="saved",
                snapshot=session.library_policy_holder.snapshot,
                copy=(
                    "Applied to this temporary chat. It remains local and will "
                    "be saved only if the chat is saved."
                    if session.ephemeral
                    else "Applied to this new chat. It will be saved with the conversation."
                ),
            )

        try:
            result = await store.save_session_library_policy(session_id)
        except Exception:
            session.library_policy_holder.snapshot = prior_snapshot
            session.library_policy_holder.explicitly_staged = prior_staged
            self._request_control_bar_sync()
            return ConsoleLibraryPolicySaveOutcome(
                status="error",
                snapshot=prior_snapshot,
                copy="Save failed. The previously saved policy is still active.",
            )

        if result.status is ConsoleLibraryPolicyWriteStatus.COMMITTED:
            self._request_control_bar_sync()
            return ConsoleLibraryPolicySaveOutcome(
                status="saved",
                snapshot=result.snapshot,
                copy="Saved on this device. This policy is not synced.",
            )

        session.library_policy_holder.snapshot = prior_snapshot
        session.library_policy_holder.explicitly_staged = prior_staged
        self._request_control_bar_sync()
        if result.status is ConsoleLibraryPolicyWriteStatus.CONFLICT:
            return ConsoleLibraryPolicySaveOutcome(
                status="conflict",
                snapshot=prior_snapshot,
                copy="This policy changed elsewhere. Reload or compare and retry.",
            )
        if result.status is ConsoleLibraryPolicyWriteStatus.MISSING_CONVERSATION:
            return ConsoleLibraryPolicySaveOutcome(
                status="unavailable",
                snapshot=prior_snapshot,
                copy="This conversation no longer exists. Your choices were not saved.",
            )
        return ConsoleLibraryPolicySaveOutcome(
            status="unavailable",
            snapshot=prior_snapshot,
            copy="Policy storage is unavailable. The previously saved policy is active.",
        )

    async def _reload(self) -> ConsoleLibraryPolicySnapshot:
        session = self._active_session()
        if session is None:
            raise RuntimeError("The active conversation is no longer available.")
        if session.persisted_conversation_id is None:
            return session.library_policy_holder.snapshot
        snapshot = await self._ensure_store().hydrate_session_library_policy(
            session.id
        )
        self._request_control_bar_sync()
        return snapshot


def _unavailable_snapshot() -> ConsoleLibraryPolicySnapshot:
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
    )

    return ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=None,
        source="unavailable",
        error_code="missing_session",
    )


__all__ = ["ConsoleLibraryPolicyController"]
