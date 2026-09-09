"""A hidden run keeps its Canvas owner without gaining browser authority."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Canvas.models import CanvasScope
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


@pytest.mark.parametrize(
    "invalid_owner", (None, "session", "conversation", "branch", "closed")
)
def test_hidden_run_scope_preserves_exact_owner_and_browser_guard(invalid_owner):
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=None))
    store = runtime.ensure_chat_store()
    session = store.create_session(ephemeral=True)
    store.append_message(session.id, role="user", content="Make a Canvas")
    scope = CanvasScope(
        session_id=session.id,
        conversation_id=session.id,
        active_message_ids=store.canvas_active_path_message_ids(session.id),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="owned-run",
    )

    def visible_scope(session_id):
        if session_id != store.active_session_id:
            raise RuntimeError("Canvas session is no longer active")
        return scope

    authority = runtime.ensure_canvas_native_authority(scope_resolver=visible_scope)
    try:
        created = authority.import_html(session_id=session.id, source="<p>owned</p>")
        authority.gateway_scope(
            session_id=session.id,
            browser_session_id="owned-browser",
            canvas_id=created.canvas_id,
            revision_id=created.revision_id,
            follow_latest=False,
        )
        other = store.create_session(ephemeral=True)
        if invalid_owner == "session":
            scope = replace(scope, session_id=other.id)
        elif invalid_owner == "conversation":
            scope = replace(scope, conversation_id=other.id)
        elif invalid_owner == "branch":
            scope = replace(scope, active_message_ids=("not-this-branch",))
        elif invalid_owner == "closed":
            store.close_session(session.id)

        if invalid_owner is None:
            captured = runtime.canvas_controller.capture_selected_scope(scope)
            # Switching session invalidates the old browser pin. Run capture
            # preserves the exact branch and never substitutes the new view.
            assert captured == scope
            assert store.active_session_id == other.id
        else:
            with pytest.raises((RuntimeError, ValueError, KeyError)):
                runtime.canvas_controller.capture_selected_scope(scope)

        # A queued turn's scope must never relax browser interaction authority.
        with pytest.raises(RuntimeError, match="no longer active"):
            authority.import_html(session_id=session.id, source="<p>stale browser</p>")
    finally:
        asyncio.run(runtime.dispose())
