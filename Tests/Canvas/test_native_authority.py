"""Production native Canvas authority and temporary-promotion coverage."""

from dataclasses import replace

from tldw_chatbook.Canvas.gateway import (
    BridgeConfirmationRequest,
    CanvasGatewayScope,
)
from tldw_chatbook.Canvas.models import CanvasBridgeRequest, CanvasScope
from tldw_chatbook.Canvas.native_authority import NativeConsoleCanvasAuthority
from tldw_chatbook.Canvas.service import CanvasService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleMessageRole,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _scope(session_id: str) -> CanvasScope:
    return CanvasScope(
        session_id=session_id,
        conversation_id=session_id,
        active_message_ids=("user-1", "assistant-1"),
        selected_canvas_id=None,
        selected_revision_id=None,
        run_id="interaction-1",
    )


def test_temporary_import_rename_previous_and_promotion_share_one_history():
    session_id = "temporary-session"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    scopes = {session_id: _scope(session_id)}
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda requested: scopes[requested],
        canvas_controller=controller,
    )

    root = authority.import_html(
        session_id=session_id,
        source="<!doctype html><title>First</title><p>private</p>",
        create_new=True,
    )
    scopes[session_id] = replace(scopes[session_id], run_id="interaction-2")
    gateway_scope = authority.gateway_scope(
        session_id=session_id,
        browser_session_id="browser-temporary",
        canvas_id=root.canvas_id,
        revision_id=root.revision_id,
    )
    renamed = authority.navigate(gateway_scope, action="rename", title="Renamed")

    assert renamed.projection.title == "Renamed"
    assert renamed.projection.parent_revision_id == root.revision_id
    assert authority.read_source(renamed.scope).source.endswith("<p>private</p>")
    previous = authority.navigate(renamed.scope, action="previous")
    assert previous.scope.revision_id == root.revision_id
    assert previous.projection.following is False

    contribution = controller.promotion_contribution(session_id)
    assert contribution is not None
    assert contribution.revision_count == 2
    rows = contribution.turn.revisions
    assert [row.actor_kind for row in rows] == ["user_import", "user_rename"]
    assert rows[0].source == rows[1].source
    assert rows[1].info.parent_revision_id == rows[0].info.revision_id


def test_temporary_canvas_history_is_destroyed_with_session():
    session_id = "temporary-discard"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    scope = _scope(session_id)
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: scope,
        canvas_controller=controller,
    )
    created = authority.import_html(
        session_id=session_id,
        source="<!doctype html><title>Temporary</title>",
        create_new=True,
    )

    controller.discard_session(session_id)

    assert controller.promotion_contribution(session_id) is None
    try:
        authority.gateway_scope(
            session_id=session_id,
            browser_session_id="browser-discarded",
            canvas_id=created.canvas_id,
        )
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("discarded temporary Canvas remained reachable")


def test_temporary_rename_and_selection_survive_atomic_promotion_and_restart(
    tmp_path,
):
    db = CharactersRAGDB(tmp_path / "canvas-native-promotion.sqlite", "canvas-native")
    controller = ConsoleCanvasController()
    store = ConsoleChatStore(
        persistence=ChatPersistenceService(db),
        canvas_promotion_participant=controller,
        canvas_turn_controller=controller,
    )
    try:
        session = store.create_session(ephemeral=True)
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Canvas is ready.",
        )
        scopes = {
            session.id: CanvasScope(
                session_id=session.id,
                conversation_id=session.id,
                active_message_ids=(assistant.id,),
                selected_canvas_id=None,
                selected_revision_id=None,
                run_id="user-import",
            )
        }
        authority = NativeConsoleCanvasAuthority(
            scope_resolver=lambda requested: scopes[requested],
            canvas_controller=controller,
        )
        created = authority.import_html(
            session_id=session.id,
            source="<!doctype html><title>Imported</title><main>private</main>",
            create_new=True,
        )
        scopes[session.id] = replace(scopes[session.id], run_id="user-rename")
        selected = authority.gateway_scope(
            session_id=session.id,
            browser_session_id="browser-before-promotion",
            canvas_id=created.canvas_id,
            revision_id=created.revision_id,
        )
        renamed = authority.navigate(selected, action="rename", title="Final title")

        conversation_id = store.promote_ephemeral_session(session.id)
        persisted_message_id = store._message_or_raise(
            assistant.id
        ).persisted_message_id
        assert conversation_id is not None
        assert persisted_message_id is not None

        rows = (
            db.get_connection()
            .execute(
                """
            SELECT id, parent_revision_id, actor_kind, origin_message_id, title
            FROM canvas_revisions
            WHERE canvas_id = ?
            ORDER BY sequence
            """,
                (created.canvas_id,),
            )
            .fetchall()
        )
        assert [row[2] for row in rows] == ["user_import", "user_rename"]
        assert rows[1][1] == rows[0][0] == created.revision_id
        assert [row[3] for row in rows] == [persisted_message_id] * 2
        assert rows[1][4] == "Final title"

        restarted_scope = CanvasScope(
            session_id="restarted-session",
            conversation_id=conversation_id,
            active_message_ids=(persisted_message_id,),
            selected_canvas_id=created.canvas_id,
            selected_revision_id=renamed.scope.revision_id,
            run_id="after-restart",
        )
        restarted = CanvasService(db).read_canvas(restarted_scope, created.canvas_id)
        assert restarted.revision.revision_id == renamed.scope.revision_id
        assert restarted.revision.title == "Final title"
        assert restarted.source.endswith("<main>private</main>")
    finally:
        db.close_connection()


def test_completed_tool_mutation_publishes_and_requests_first_auto_open():
    session_id = "temporary-tool"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    scope = _scope(session_id)
    opened = []
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: scope,
        canvas_controller=controller,
        auto_open=lambda requested, info: opened.append((requested, info.revision_id)),
    )
    controller.add_mutation_listener(authority.on_tool_mutation)
    run = controller.register_run(
        scope, assistant_message_id="assistant-1", temporary=True
    )

    created = run.create_canvas(
        scope,
        tool_call_id="canvas-create-call",
        title="Tool result",
        html="<!doctype html><h1>Tool result</h1>",
    )

    assert opened == [(session_id, created.revision.revision_id)]
    browser_scope = authority.gateway_scope(
        session_id=session_id,
        browser_session_id="browser-tool",
        canvas_id=created.revision.canvas_id,
        revision_id=created.revision.revision_id,
    )
    events = authority.read_events(browser_scope, after_event_id=None)
    assert events[-1].kind == "selection_changed"
    assert events[-1].revision_id == created.revision.revision_id


def test_confirmed_json_submit_reaches_composer_as_valid_json_text():
    class Settlement:
        def try_settle(self, callback):
            callback()
            return True

    drafts = []
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: _scope("temporary-bridge"),
        canvas_controller=ConsoleCanvasController(),
        bridge_sink=drafts.append,
    )
    request = BridgeConfirmationRequest(
        approved=True,
        request=CanvasBridgeRequest(
            version="canvas-v1",
            request_id="bridge-json",
            kind="submit",
            value={"answer": 42, "ok": True},
        ),
    )

    response = authority.confirm_bridge(
        CanvasGatewayScope(
            browser_session_id="browser-bridge",
            conversation_session_id="temporary-bridge",
            canvas_id="canvas-bridge",
            revision_id="revision-bridge",
        ),
        request,
        settlement=Settlement(),
    )

    assert response.status == "confirmed"
    assert drafts == ['{"answer":42,"ok":true}']
