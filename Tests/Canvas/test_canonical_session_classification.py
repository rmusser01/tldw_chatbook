"""Native canonical saved IDs do not imply temporary Canvas storage."""

from dataclasses import replace

import pytest

from tldw_chatbook.Canvas.models import CanvasScope
from tldw_chatbook.Canvas.native_authority import NativeConsoleCanvasAuthority
from tldw_chatbook.Canvas.service import CanvasService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore, ConsoleMessageRole
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _scope(store, session):
    return CanvasScope(
        session.id,
        session.persisted_conversation_id or session.id,
        store.canvas_active_path_message_ids(session.id),
        None,
        None,
        "canonical-control",
        ephemeral=session.ephemeral,
    )


def test_real_temporary_promotion_and_canonical_saved_scope(tmp_path):
    db = CharactersRAGDB(tmp_path / "canonical.sqlite", "canonical-canvas")
    controller = ConsoleCanvasController(durable_service=CanvasService(db))
    store = ConsoleChatStore(
        persistence=ChatPersistenceService(db),
        canvas_promotion_participant=controller,
        canvas_turn_controller=controller,
    )
    try:
        session = store.create_session(ephemeral=True)
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="Canvas"
        )
        current = _scope(store, session)
        authority = NativeConsoleCanvasAuthority(
            scope_resolver=lambda _: current, canvas_controller=controller
        )
        created = authority.import_html(
            session_id=session.id, source="<!doctype html><p>native source</p>"
        )
        selected = authority.gateway_scope(
            session_id=session.id,
            browser_session_id="before-promotion",
            canvas_id=created.canvas_id,
            revision_id=created.revision_id,
            follow_latest=False,
        )
        assert authority.describe_selection(selected).temporary is True
        conversation_id = store.promote_ephemeral_session(session.id)
        assert conversation_id is not None
        current = _scope(store, session)
        assert current.ephemeral is False
        authority.sync_live_context(session.id)
        pinned = controller.capture_selected_scope(current)
        assert pinned.selected_revision_id == created.revision_id
        assert authority.describe_selection(selected).temporary is False

        # The restored native identity is deliberately the canonical durable ID.
        from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
        from tldw_chatbook.Chat.console_conversation_hydration import (
            console_messages_from_conversation_tree,
        )

        tree = ChatConversationService(db).get_conversation_tree(
            conversation_id, depth_cap=10_000, root_limit=10_000
        )
        restored_store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        restored = restored_store.restore_persisted_session(
            title="Canonical",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=console_messages_from_conversation_tree(tree, db=db),
            active_leaf_persisted_id=store.get_message(message.id).persisted_message_id,
        )
        scope = _scope(restored_store, restored)
        assert scope.session_id == scope.conversation_id == conversation_id
        assert scope.ephemeral is False
        reopened = NativeConsoleCanvasAuthority(
            scope_resolver=lambda _: scope,
            canvas_controller=ConsoleCanvasController(
                durable_service=CanvasService(db)
            ),
        )
        exact = reopened.gateway_scope(
            session_id=restored.id,
            browser_session_id="canonical-reopen",
            canvas_id=created.canvas_id,
            revision_id=created.revision_id,
            follow_latest=False,
        )
        assert reopened.describe_selection(exact).temporary is False
        assert reopened.read_source(exact).source.endswith("<p>native source</p>")
        renamed = reopened.navigate(exact, action="rename", title="Durable renamed")
        assert renamed.projection.title == "Durable renamed"
        assert (
            CanvasService(db)
            .read_canvas(
                replace(
                    scope,
                    selected_canvas_id=created.canvas_id,
                    selected_revision_id=renamed.scope.revision_id,
                ),
                created.canvas_id,
            )
            .revision.title
            == "Durable renamed"
        )
    finally:
        db.close_connection()


def test_native_authority_refuses_unknown_state_and_same_ids_changed_state():
    controller = ConsoleCanvasController()
    controller.activate_session("same")
    current = CanvasScope(
        "same", "same", ("message",), None, None, "run", ephemeral=True
    )
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _: current, canvas_controller=controller
    )
    captured = current
    current = replace(current, ephemeral=False)
    with pytest.raises(RuntimeError, match="^canvas_scope_unavailable$"):
        controller.capture_selected_scope(captured)
    current = replace(current, ephemeral=None)
    with pytest.raises(RuntimeError, match="^canvas_scope_unavailable$"):
        authority.import_html(session_id="same", source="<p>must refuse</p>")
    with pytest.raises(TypeError, match="Canvas ephemeral state"):
        replace(captured, ephemeral=1)


def test_registered_run_rejects_classification_substitution():
    controller = ConsoleCanvasController()
    controller.activate_session("same")
    scope = CanvasScope("same", "same", ("message",), None, None, "run", ephemeral=True)
    run = controller.register_run(scope, assistant_message_id="message", temporary=True)
    assert run.is_scope_current(scope)
    assert not run.is_scope_current(replace(scope, ephemeral=False))
    controller.activate_session("same")
    assert not run.is_scope_current(scope)
