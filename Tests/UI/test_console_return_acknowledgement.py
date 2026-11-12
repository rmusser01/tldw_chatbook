"""Only a captured unread reminder needs an acknowledgement-only FULL pass."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["no_session", "unsaved", "read", "refused", "unread"]
)
async def test_explicit_return_syncs_only_for_a_captured_unread_token(state):
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

    effects = []
    token = object()
    session = SimpleNamespace(
        id="chat", persisted_conversation_id=None if state == "unsaved" else "saved"
    )
    store = SimpleNamespace(
        active_session_id=None if state == "no_session" else session.id,
        sessions=lambda: [session],
    )

    def unread_token(conversation_id):
        effects.append(("read", conversation_id))
        if state == "refused":
            raise RuntimeError("original read refused")
        return token if state == "unread" else None

    async def sync():
        effects.append("sync")

    owner = ConsoleSessionController.__new__(ConsoleSessionController)
    owner.app_instance = SimpleNamespace(
        conversation_local_marks_service=SimpleNamespace(unread_token=unread_token)
    )
    owner._chat_store_accessor = lambda: store
    owner._sync_native_console_chat_ui_fn = sync
    owner._manual_navigation_generation = 0
    owner._pending_manual_read_visit = None
    owner._manual_read_scheduled = False
    owner._screen = SimpleNamespace(
        call_after_refresh=lambda callback: effects.append("after_paint")
    )

    await owner.acknowledge_explicit_console_return()

    if state == "unread":
        assert effects == [("read", "saved"), "sync", "after_paint"]
        assert owner._pending_manual_read_visit.token is token
        assert owner._pending_manual_read_visit.session_id == session.id
    else:
        assert effects == (
            [] if state in {"no_session", "unsaved"} else [("read", "saved")]
        )
        assert owner._pending_manual_read_visit is None
