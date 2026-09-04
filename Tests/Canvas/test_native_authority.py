"""Production native Canvas authority and temporary-promotion coverage."""

from dataclasses import replace

import pytest

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


def _branch_scope(
    session_id: str,
    conversation_id: str,
    active_message_ids: tuple[str, ...],
    run_id: str,
    *,
    canvas_id: str | None = None,
    revision_id: str | None = None,
) -> CanvasScope:
    return CanvasScope(
        session_id=session_id,
        conversation_id=conversation_id,
        active_message_ids=active_message_ids,
        selected_canvas_id=canvas_id,
        selected_revision_id=revision_id,
        run_id=run_id,
    )


def test_temporary_exact_left_revision_is_unreachable_from_right_sibling_branch():
    session_id = "temporary-siblings"
    conversation_id = "temporary-siblings"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    root_scope = _branch_scope(
        session_id, conversation_id, ("root-message",), "root-import"
    )
    root = controller.interactive_create_canvas(
        root_scope,
        origin_message_id="root-message",
        title="Sibling test",
        html="<!doctype html><p>root</p>",
        temporary=True,
    )
    left_scope = _branch_scope(
        session_id,
        conversation_id,
        ("root-message", "left-message"),
        "left-update",
    )
    left = controller.interactive_update_canvas(
        left_scope,
        origin_message_id="left-message",
        canvas_id=root.revision.canvas_id,
        expected_parent_revision_id=root.revision.revision_id,
        html="<!doctype html><p>left branch private source</p>",
        temporary=True,
    )
    assert hasattr(left, "revision")
    right_scope = _branch_scope(
        session_id,
        conversation_id,
        ("root-message", "right-message"),
        "right-update",
    )
    right = controller.interactive_update_canvas(
        right_scope,
        origin_message_id="right-message",
        canvas_id=root.revision.canvas_id,
        expected_parent_revision_id=root.revision.revision_id,
        html="<!doctype html><p>right branch</p>",
        temporary=True,
    )
    assert hasattr(right, "revision")

    scopes = {session_id: left_scope}
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda requested: scopes[requested],
        canvas_controller=controller,
    )
    authority.gateway_scope(
        session_id=session_id,
        browser_session_id="browser-siblings",
        canvas_id=root.revision.canvas_id,
        revision_id=left.revision.revision_id,
        follow_latest=False,
    )
    scopes[session_id] = right_scope
    selected_left = replace(
        right_scope,
        selected_canvas_id=root.revision.canvas_id,
        selected_revision_id=left.revision.revision_id,
    )
    gateway_left = CanvasGatewayScope(
        browser_session_id="browser-siblings",
        conversation_session_id=session_id,
        canvas_id=root.revision.canvas_id,
        revision_id=left.revision.revision_id,
    )

    rejected: list[str] = []
    for operation in (
        lambda: controller.read_session_canvas(
            selected_left, root.revision.canvas_id, temporary=True
        ),
        lambda: authority.gateway_scope(
            session_id=session_id,
            browser_session_id="browser-siblings-new",
            canvas_id=root.revision.canvas_id,
            revision_id=left.revision.revision_id,
            follow_latest=False,
        ),
        lambda: authority.read_source(gateway_left),
        lambda: authority.navigate(gateway_left, action="rename", title="Stolen"),
        lambda: authority.import_html(
            session_id=session_id,
            source="<!doctype html><p>replacement</p>",
            create_new=False,
        ),
    ):
        with pytest.raises((RuntimeError, ValueError)) as error:
            operation()
        rejected.append(str(error.value))

    assert all("left branch private source" not in error for error in rejected)


def test_temporary_exact_read_rejects_wrong_conversation_uncommitted_discarded_and_reused_owner():
    session_id = "temporary-exact-lifecycle"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    committed_scope = _branch_scope(
        session_id, session_id, ("committed-message",), "committed-import"
    )
    committed = controller.interactive_create_canvas(
        committed_scope,
        origin_message_id="committed-message",
        title="Committed",
        html="<!doctype html><p>committed private source</p>",
        temporary=True,
    )
    exact = replace(
        committed_scope,
        selected_canvas_id=committed.revision.canvas_id,
        selected_revision_id=committed.revision.revision_id,
    )
    wrong_conversation = replace(exact, conversation_id="other-conversation")
    with pytest.raises(RuntimeError, match="canvas_base_unavailable") as error:
        controller.read_session_canvas(
            wrong_conversation, committed.revision.canvas_id, temporary=True
        )
    assert "private source" not in str(error.value)

    open_scope = _branch_scope(
        session_id, session_id, ("open-message",), "open-tool-run"
    )
    open_run = controller.register_run(
        open_scope, assistant_message_id="open-message", temporary=True
    )
    uncommitted = open_run.create_canvas(
        open_scope,
        tool_call_id="open-create",
        title="Open",
        html="<!doctype html><p>uncommitted private source</p>",
    )
    uncommitted_exact = replace(
        open_scope,
        selected_canvas_id=uncommitted.revision.canvas_id,
        selected_revision_id=uncommitted.revision.revision_id,
    )
    assert controller.list_session_canvases(open_scope, temporary=True) == ()
    with pytest.raises(RuntimeError, match="canvas_base_unavailable"):
        controller.read_session_canvas(
            uncommitted_exact, uncommitted.revision.canvas_id, temporary=True
        )
    open_run.finish_assistant_run(
        "open-message", actual_run_id="open-tool-run", terminal_status="failed"
    )
    with pytest.raises(RuntimeError, match="canvas_base_unavailable"):
        controller.read_session_canvas(
            uncommitted_exact, uncommitted.revision.canvas_id, temporary=True
        )

    controller.activate_session(session_id)
    with pytest.raises(RuntimeError, match="canvas_base_unavailable"):
        controller.read_session_canvas(
            exact, committed.revision.canvas_id, temporary=True
        )


def test_live_branch_sync_moves_following_selection_but_keeps_pinned_selection():
    session_id = "temporary-live-sync"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    root_scope = _branch_scope(session_id, session_id, ("root",), "root")
    root = controller.interactive_create_canvas(
        root_scope,
        origin_message_id="root",
        title="Live sync",
        html="<!doctype html><p>root</p>",
        temporary=True,
    )
    left_scope = _branch_scope(session_id, session_id, ("root", "left"), "left")
    left = controller.interactive_update_canvas(
        left_scope,
        origin_message_id="left",
        canvas_id=root.revision.canvas_id,
        expected_parent_revision_id=root.revision.revision_id,
        html="<!doctype html><p>left</p>",
        temporary=True,
    )
    right_scope = _branch_scope(session_id, session_id, ("root", "right"), "right")
    right = controller.interactive_update_canvas(
        right_scope,
        origin_message_id="right",
        canvas_id=root.revision.canvas_id,
        expected_parent_revision_id=root.revision.revision_id,
        html="<!doctype html><p>right</p>",
        temporary=True,
    )
    assert hasattr(left, "revision") and hasattr(right, "revision")
    scopes = {session_id: left_scope}
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda requested: scopes[requested],
        canvas_controller=controller,
    )
    left_gateway = authority.gateway_scope(
        session_id=session_id,
        browser_session_id="browser-live-sync",
        canvas_id=root.revision.canvas_id,
        revision_id=left.revision.revision_id,
    )

    scopes[session_id] = right_scope
    authority.sync_live_context(session_id)
    following = authority.navigate(left_gateway, action="follow")
    assert following.scope.revision_id == right.revision.revision_id

    scopes[session_id] = left_scope
    pinned = authority.navigate(
        replace(left_gateway, revision_id=left.revision.revision_id), action="pin"
    )
    scopes[session_id] = right_scope
    authority.sync_live_context(session_id)
    assert authority._selection[session_id].revision_id == pinned.scope.revision_id
    events = authority.read_events(pinned.scope, after_event_id=None)
    assert events[-1].revision_id == right.revision.revision_id


def test_live_context_sync_is_a_noop_before_any_canvas_selection():
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: (_ for _ in ()).throw(
            RuntimeError("empty transcript has no Canvas scope")
        ),
        canvas_controller=ConsoleCanvasController(),
    )

    authority.sync_live_context("empty-session")


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


def test_parsed_block_identity_is_idempotent_branch_bound_and_preserves_origin():
    session_id = "temporary-parsed-block"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    scopes = {
        session_id: _branch_scope(
            session_id,
            session_id,
            ("user-root", "assistant-origin"),
            "canvas-open-interaction",
        )
    }
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda requested: scopes[requested],
        canvas_controller=controller,
    )
    arguments = {
        "session_id": session_id,
        "source": "<!doctype html><title>Parsed</title><p>one</p>",
        "source_message_id": "assistant-origin",
        "origin_message_id": "assistant-origin",
        "source_turn_id": "assistant-turn",
        "block_index": 0,
        "block_identity": "assistant-origin:canvas-html:0",
    }

    first = authority.import_html(**arguments)
    replay = authority.import_html(**arguments)
    scopes[session_id] = replace(
        scopes[session_id], run_id="canvas-open-as-new-interaction"
    )
    explicit_new = authority.import_html(**arguments, create_new=True)

    assert replay == first
    assert explicit_new.canvas_id != first.canvas_id
    assert first.origin.message_id == "assistant-origin"
    assert first.origin.run_id == "assistant-turn"
    assert controller.promotion_contribution(session_id).revision_count == 2

    scopes[session_id] = _branch_scope(
        session_id,
        session_id,
        ("user-root", "right-assistant"),
        "right-branch-interaction",
    )
    with pytest.raises(RuntimeError, match="source message") as error:
        authority.import_html(**arguments)
    assert "<p>one</p>" not in str(error.value)


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

        restarted_authority = NativeConsoleCanvasAuthority(
            scope_resolver=lambda _requested: replace(
                restarted_scope,
                selected_canvas_id=None,
                selected_revision_id=None,
            ),
            canvas_controller=ConsoleCanvasController(
                durable_service=CanvasService(db)
            ),
        )
        reopened = restarted_authority.import_html(
            session_id="restarted-session",
            source="<!doctype html><title>Imported</title><main>private</main>",
            source_message_id=persisted_message_id,
            origin_message_id=persisted_message_id,
            source_turn_id="user-import",
            block_index=0,
            block_identity=f"{persisted_message_id}:canvas-html:0",
        )

        assert reopened.revision_id == created.revision_id
        assert reopened.canvas_id == created.canvas_id
        assert len(CanvasService(db).list_canvases(restarted_scope)) == 1
    finally:
        db.close_connection()


def test_completed_tool_mutation_publishes_and_requests_first_auto_open_after_commit():
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
    controller.add_settlement_listener(authority.on_settlement_publication)
    run = controller.register_run(
        scope, assistant_message_id="assistant-1", temporary=True
    )

    created = run.create_canvas(
        scope,
        tool_call_id="canvas-create-call",
        title="Tool result",
        html="<!doctype html><h1>Tool result</h1>",
    )

    assert opened == []
    assert authority._events == {}
    settlement = run.finish_assistant_run(
        "assistant-1", actual_run_id=scope.run_id, terminal_status="done"
    )
    assert settlement is not None
    assert opened == []
    assert controller.confirm_exact_settlement(settlement) is True
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
