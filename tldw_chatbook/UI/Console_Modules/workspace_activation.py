"""Existing owned operations, loaded only on their actual execution path."""

from __future__ import annotations

import tldw_chatbook.UI.Console_Modules.workspace as host


async def _open_character_conversation_activation(
    self,
    request: host.LocalCharacterConversationTarget
    | host.CharacterConversationActivationRequest,
) -> host.ConsoleActivationCommit:
    """Open a target and capture only a runtime session created by this call."""
    from ...Chat.console_conversation_activation import (
        CharacterConversationActivationRequest,
        ConsoleActivationCommit,
    )

    target = (
        request.target
        if isinstance(request, CharacterConversationActivationRequest)
        else request
    )
    store = self._ensure_console_chat_store()
    from ...Chat.console_chat_store import ConsoleSavedSessionCleanup

    cleanup_owner = object()
    opened = bool(
        await self.open_console_workspace_conversation(
            target.conversation_id, restore_cleanup_owner=cleanup_owner
        )
    )
    session = next(
        (
            row
            for row in store.sessions()
            if row.persisted_conversation_id == target.conversation_id
        ),
        None,
    )
    owned = (
        ConsoleSavedSessionCleanup(session, cleanup_owner)
        if session is not None
        else None
    )
    if opened:
        # Widget.focus() is queued by Textual. This committed Character
        # boundary must establish focus before the immediate exact-visible
        # proof, including when activation began in a Context row.
        try:
            # A whole-Console sync may coalesce into an already-running
            # pass. Publish this target before the exact-ready proof.
            await self._screen._sync_native_console_transcript()
            self._screen.set_focus(self._screen.query_one("#console-native-composer"))
        except host.asyncio.CancelledError:
            # The caller has not received ownership yet. Settle only this
            # exact cold runtime synchronously, then let it restore prior UI.
            if owned is not None and owned.matches(
                store._sessions.get(owned.session.id)
            ):
                store.rollback_restored_session(
                    owned.session.id,
                    expected_session=owned.session,
                    prior_active_session_id=None,
                    cleanup_owner=owned.owner,
                )
            raise
        except Exception:  # noqa: BLE001 - preserve the owned token for rollback
            host.logger.opt(exception=True).warning(
                "Could not focus the committed character conversation"
            )
            opened = False
    return ConsoleActivationCommit(opened=opened, owned_runtime_token=owned)


async def _rollback_character_conversation_activation(
    self, owned_runtime: object
) -> bool:
    """Remove only the exact hydrated runtime owned by this activation."""

    from ...Chat.console_chat_store import ConsoleSavedSessionCleanup

    if not isinstance(owned_runtime, ConsoleSavedSessionCleanup):
        return False
    store = self._ensure_console_chat_store()
    if not owned_runtime.matches(store._sessions.get(owned_runtime.session.id)):
        return False
    return store.rollback_restored_session(
        owned_runtime.session.id,
        expected_session=owned_runtime.session,
        prior_active_session_id=None,
        cleanup_owner=owned_runtime.owner,
    )


def _commit_character_conversation_activation(self, owned_runtime: object) -> None:
    """Publish only after the outer exact visibility/focus attempt commits."""
    from ...Chat.console_chat_store import ConsoleSavedSessionCleanup

    if isinstance(owned_runtime, ConsoleSavedSessionCleanup):
        store = self._ensure_console_chat_store()
        if not owned_runtime.matches(store._sessions.get(owned_runtime.session.id)):
            raise RuntimeError(
                "Character conversation owner changed before publication."
            )
        store.publish_restored_session(
            owned_runtime.session, cleanup_owner=owned_runtime.owner
        )
