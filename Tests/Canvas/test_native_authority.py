"""Production native Canvas authority and temporary-promotion coverage."""

import threading
from dataclasses import replace

import pytest

from tldw_chatbook.Canvas.gateway import (
    BridgeConfirmationRequest,
    CanvasGatewayScope,
)
from tldw_chatbook.Canvas.limits import CanvasRepositoryLimits
from tldw_chatbook.Canvas.models import CanvasBridgeRequest, CanvasScope
from tldw_chatbook.Canvas.native_authority import NativeConsoleCanvasAuthority
from tldw_chatbook.Canvas.service import CanvasService
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_canvas_controller import (
    CanvasSettlementPublication,
    ConsoleCanvasController,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
)
from tldw_chatbook.Chat.console_message_actions import (
    assistant_canvas_html_blocks,
    canvas_block_origin_turn_id,
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
        replace(left_gateway, revision_id=root.revision.revision_id), action="pin"
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


def test_store_branch_transition_invalidates_unreachable_canvas_until_reopened():
    controller = ConsoleCanvasController()
    authority_holder = {}
    store = ConsoleChatStore(
        canvas_promotion_participant=controller,
        canvas_turn_controller=controller,
        on_canvas_context_changed=lambda session_id: authority_holder.get(
            "authority"
        )
        and authority_holder["authority"].sync_live_context(session_id),
    )
    session = store.create_session(ephemeral=True)
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="root"
    )
    left = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="left"
    )

    def resolve(requested: str) -> CanvasScope:
        if store.active_session_id != requested:
            raise RuntimeError("Canvas session is no longer active")
        return CanvasScope(
            session_id=requested,
            conversation_id=requested,
            active_message_ids=tuple(store.active_path_message_ids(requested)),
            selected_canvas_id=None,
            selected_revision_id=None,
            run_id="branch-transition",
        )

    invalidated = []
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=resolve,
        canvas_controller=controller,
    )
    authority.bind_gateway_invalidator(invalidated.append)
    authority_holder["authority"] = authority
    created = authority.import_html(
        session_id=session.id,
        source="<!doctype html><h1>left only</h1>",
        source_message_id=left.id,
        origin_message_id=left.id,
        source_turn_id="left-import",
        block_index=0,
        block_identity=f"{left.id}:canvas-html:0",
    )
    browser_scope = authority.gateway_scope(
        session_id=session.id,
        browser_session_id="browser-unreachable",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
    )
    authority.navigate(browser_scope, action="pin")

    right = store.create_sibling(
        left.id, role=ConsoleMessageRole.ASSISTANT, content="right"
    )

    assert invalidated == ["browser-unreachable"]
    events = authority.read_events(browser_scope, after_event_id=None)
    assert events[-1].kind == "disconnected"
    assert events[-1].metadata == {"notice": "unavailable_on_branch"}
    assert session.id not in authority._selection

    store.set_active_leaf(session.id, left.id)
    assert invalidated == ["browser-unreachable"]
    reopened = authority.gateway_scope(
        session_id=session.id,
        browser_session_id="browser-reopened",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
    )
    assert reopened.revision_id == created.revision_id
    assert store.get_message(right.id).content == "right"


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


def test_durable_parsed_block_identity_survives_real_store_hydration(tmp_path):
    db = CharactersRAGDB(tmp_path / "canvas-native-hydration.sqlite", "canvas-native")
    try:
        conversations = ChatConversationService(db)
        conversation_id = conversations.create_conversation(
            id="canvas-hydrated-conversation",
            title="Hydrated Canvas",
            scope_type="global",
            state="in-progress",
        )
        user_id = db.add_message(
            {
                "id": "canvas-hydrated-user",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "Show two examples.",
            }
        )
        assistant_id = db.add_message(
            {
                "id": "canvas-hydrated-assistant",
                "conversation_id": conversation_id,
                "parent_message_id": user_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "```html\n<!doctype html><p>same</p>\n```\n"
                "```html\n<!doctype html><p>same</p>\n```",
            }
        )
        db.set_conversation_active_cursor(
            conversation_id,
            active_leaf_message_id=assistant_id,
            before_message_id=None,
        )

        def hydrate():
            tree = conversations.get_conversation_tree(
                conversation_id, depth_cap=10_000, root_limit=10_000
            )
            nodes = console_messages_from_conversation_tree(tree, db=db)
            store = ConsoleChatStore(persistence=ChatPersistenceService(db))
            session = store.restore_persisted_session(
                title="Hydrated Canvas",
                workspace_id=None,
                persisted_conversation_id=conversation_id,
                all_nodes=nodes,
                active_leaf_persisted_id=assistant_id,
            )
            assistant = next(
                item
                for item in nodes
                if item.persisted_message_id == assistant_id
            )
            return store, session, assistant

        def scope_for(store, session):
            return CanvasScope(
                session_id=session.id,
                conversation_id=conversation_id,
                active_message_ids=tuple(
                    store.get_message(message_id).persisted_message_id
                    or message_id
                    for message_id in store.active_path_message_ids(session.id)
                ),
                selected_canvas_id=None,
                selected_revision_id=None,
                run_id="hydrated-import",
            )

        first_store, first_session, first_assistant = hydrate()
        first_scope = scope_for(first_store, first_session)
        first_authority = NativeConsoleCanvasAuthority(
            scope_resolver=lambda _requested: first_scope,
            canvas_controller=ConsoleCanvasController(
                durable_service=CanvasService(db)
            ),
        )
        first_blocks = assistant_canvas_html_blocks(first_assistant)
        imported = []
        for block in first_blocks:
            imported.append(
                first_authority.import_html(
                    session_id=first_session.id,
                    source=block.html,
                    source_message_id=first_assistant.id,
                    origin_message_id=assistant_id,
                    source_turn_id=canvas_block_origin_turn_id(
                        first_assistant, block.index
                    ),
                    block_index=block.index,
                    block_identity=block.identity,
                )
            )
        assert imported[0].revision_id != imported[1].revision_id

        restarted_store, restarted_session, restarted_assistant = hydrate()
        assert restarted_assistant.id != first_assistant.id
        restarted_scope = scope_for(restarted_store, restarted_session)
        restarted_authority = NativeConsoleCanvasAuthority(
            scope_resolver=lambda _requested: restarted_scope,
            canvas_controller=ConsoleCanvasController(
                durable_service=CanvasService(db)
            ),
        )
        restarted_blocks = assistant_canvas_html_blocks(restarted_assistant)
        reopened = []
        for block in restarted_blocks:
            reopened.append(
                restarted_authority.import_html(
                    session_id=restarted_session.id,
                    source=block.html,
                    source_message_id=restarted_assistant.id,
                    origin_message_id=assistant_id,
                    source_turn_id=canvas_block_origin_turn_id(
                        restarted_assistant, block.index
                    ),
                    block_index=block.index,
                    block_identity=block.identity,
                )
            )

        assert [item.revision_id for item in reopened] == [
            item.revision_id for item in imported
        ]
        assert len(CanvasService(db).list_canvases(restarted_scope)) == 1
    finally:
        db.close_connection()


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


def test_completed_tool_update_hot_reloads_without_requesting_auto_open():
    session_id = "temporary-tool-update"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    created_scope = _branch_scope(
        session_id, session_id, ("assistant-root",), "interactive-root"
    )
    created = controller.interactive_create_canvas(
        created_scope,
        origin_message_id="assistant-root",
        title="Existing",
        html="<!doctype html><p>root</p>",
        temporary=True,
    )
    update_scope = _branch_scope(
        session_id,
        session_id,
        ("assistant-root", "assistant-update"),
        "tool-update",
    )
    opened: list[str] = []
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: update_scope,
        canvas_controller=controller,
        auto_open=lambda _requested, info: opened.append(info.revision_id),
    )
    controller.add_settlement_listener(authority.on_settlement_publication)
    run = controller.register_run(
        update_scope, assistant_message_id="assistant-update", temporary=True
    )
    updated = run.update_canvas(
        update_scope,
        tool_call_id="canvas-update-call",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<!doctype html><p>updated</p>",
    )
    settlement = run.finish_assistant_run(
        "assistant-update", actual_run_id=update_scope.run_id, terminal_status="done"
    )

    assert settlement is not None
    assert controller.confirm_exact_settlement(settlement) is True
    assert opened == []
    assert authority._events[(session_id, created.revision.canvas_id)][-1].revision_id == (
        updated.revision.revision_id
    )


def test_completed_tool_update_preserves_same_revision_historical_pin():
    session_id = "temporary-tool-update-pinned"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    created_scope = _branch_scope(
        session_id, session_id, ("assistant-root",), "interactive-root"
    )
    created = controller.interactive_create_canvas(
        created_scope,
        origin_message_id="assistant-root",
        title="Existing",
        html="<!doctype html><p>root</p>",
        temporary=True,
    )
    update_scope = _branch_scope(
        session_id,
        session_id,
        ("assistant-root", "assistant-update"),
        "tool-update",
    )
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: update_scope,
        canvas_controller=controller,
    )
    controller.add_settlement_listener(authority.on_settlement_publication)
    browser_scope = authority.gateway_scope(
        session_id=session_id,
        browser_session_id="browser-tool-pinned",
        canvas_id=created.revision.canvas_id,
        revision_id=created.revision.revision_id,
    )
    authority.navigate(browser_scope, action="pin")

    run = controller.register_run(
        update_scope, assistant_message_id="assistant-update", temporary=True
    )
    updated = run.update_canvas(
        update_scope,
        tool_call_id="canvas-update-call",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        html="<!doctype html><p>updated</p>",
    )
    settlement = run.finish_assistant_run(
        "assistant-update", actual_run_id=update_scope.run_id, terminal_status="done"
    )

    assert settlement is not None
    assert controller.confirm_exact_settlement(settlement) is True
    selection = authority._selection[session_id]
    assert selection.revision_id == created.revision.revision_id
    assert selection.following is False
    assert authority._events[(session_id, created.revision.canvas_id)][
        -1
    ].revision_id == (updated.revision.revision_id)

    following = authority.navigate(browser_scope, action="follow")
    assert following.scope.revision_id == updated.revision.revision_id
    assert following.projection.following is True


def test_disable_before_settlement_confirmation_suppresses_browser_effects():
    session_id = "temporary-disabled-publication"
    enabled = [True]
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    scope = _scope(session_id)
    opened: list[str] = []
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: scope,
        canvas_controller=controller,
        auto_open=lambda _requested, info: opened.append(info.revision_id),
        enabled_reader=lambda: enabled[0],
    )
    controller.add_settlement_listener(authority.on_settlement_publication)
    run = controller.register_run(
        scope, assistant_message_id="assistant-1", temporary=True
    )
    run.create_canvas(
        scope,
        tool_call_id="disabled-create",
        title="Preserved",
        html="<!doctype html><p>preserved</p>",
    )
    settlement = run.finish_assistant_run(
        "assistant-1", actual_run_id=scope.run_id, terminal_status="done"
    )

    enabled[0] = False
    assert settlement is not None
    assert controller.confirm_exact_settlement(settlement) is True
    assert opened == []
    assert authority._events == {}
    assert controller.promotion_contribution(session_id).revision_count == 1


def test_settlement_listener_retry_only_retries_incomplete_auto_open():
    session_id = "temporary-partial-publication"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    scope = _scope(session_id)
    attempts = 0
    opened = []

    def flaky_open(requested, info):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected open failure")
        opened.append((requested, info.revision_id))

    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: scope,
        canvas_controller=controller,
        auto_open=flaky_open,
    )
    controller.add_settlement_listener(authority.on_settlement_publication)
    run = controller.register_run(
        scope, assistant_message_id="assistant-1", temporary=True
    )
    created = run.create_canvas(
        scope,
        tool_call_id="partial-create",
        title="Partial listener",
        html="<!doctype html><h1>partial</h1>",
    )
    settlement = run.finish_assistant_run(
        "assistant-1", actual_run_id=scope.run_id, terminal_status="done"
    )
    assert settlement is not None

    assert controller.confirm_exact_settlement(settlement) is True
    first_events = authority._events[(session_id, created.revision.canvas_id)]
    assert len(first_events) == 1
    assert attempts == 1
    assert opened == []

    assert controller.confirm_exact_settlement(settlement) is True
    final_events = authority._events[(session_id, created.revision.canvas_id)]
    assert len(final_events) == 1
    assert attempts == 2
    assert opened == [(session_id, created.revision.revision_id)]
    assert controller.promotion_contribution(session_id).revision_count == 1


def test_publication_capacity_never_evicts_incomplete_receipt_or_replays_event():
    session_id = "temporary-publication-capacity"
    controller = ConsoleCanvasController(
        repository_limits=CanvasRepositoryLimits(max_canvases_per_conversation=300)
    )
    controller.activate_session(session_id)
    opener_recovered = False
    attempts: list[str] = []
    opened: list[str] = []

    def systemic_open(_requested, info):
        attempts.append(info.revision_id)
        if not opener_recovered:
            raise RuntimeError("injected systemic opener failure")
        opened.append(info.revision_id)

    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: _scope(session_id),
        canvas_controller=controller,
        auto_open=systemic_open,
        publication_guard=lambda _publication: True,
    )
    controller.add_settlement_listener(authority.on_settlement_publication)
    settlements = []
    created = []
    for index in range(257):
        assistant_id = f"assistant-{index}"
        scope = _branch_scope(
            session_id,
            session_id,
            (assistant_id,),
            f"interaction-{index}",
        )
        run = controller.register_run(
            scope, assistant_message_id=assistant_id, temporary=True
        )
        mutation = run.create_canvas(
            scope,
            tool_call_id=f"create-{index}",
            title=f"Capacity {index}",
            html=f"<!doctype html><h1>Capacity {index}</h1>",
        )
        settlement = run.finish_assistant_run(
            assistant_id,
            actual_run_id=scope.run_id,
            terminal_status="done",
        )
        assert settlement is not None
        assert controller.confirm_exact_settlement(settlement) is True
        settlements.append(settlement)
        created.append(mutation.revision)

    oldest = created[0]
    newest = created[-1]
    assert len(authority._publication_receipts) == 256
    assert len(authority._events[(session_id, oldest.canvas_id)]) == 1
    assert (session_id, newest.canvas_id) not in authority._events
    assert controller.run_revision_count("interaction-0") == 1
    assert len(attempts) == 256

    newest_publication = controller._runs["interaction-256"].publication
    assert newest_publication is not None
    with pytest.raises(
        RuntimeError, match="canvas_publication_capacity_exhausted"
    ) as capacity_error:
        authority.on_settlement_publication(newest_publication)
    assert newest.revision_id not in str(capacity_error.value)
    assert "Capacity 256" not in str(capacity_error.value)

    opener_recovered = True
    assert controller.confirm_exact_settlement(settlements[0]) is True
    assert controller.confirm_exact_settlement(settlements[0]) is True
    assert len(authority._events[(session_id, oldest.canvas_id)]) == 1
    assert opened == [oldest.revision_id]
    assert len(authority._publication_receipts) == 256

    assert controller.confirm_exact_settlement(settlements[-1]) is True
    assert len(authority._events[(session_id, newest.canvas_id)]) == 1
    assert opened == [oldest.revision_id, newest.revision_id]
    assert len(authority._publication_receipts) == 256


def test_publication_capacity_churn_evicts_only_controller_delivered_receipts():
    session_id = "temporary-publication-success-churn"
    controller = ConsoleCanvasController(
        repository_limits=CanvasRepositoryLimits(max_canvases_per_conversation=300)
    )
    controller.activate_session(session_id)
    opened: list[str] = []
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: _scope(session_id),
        canvas_controller=controller,
        auto_open=lambda _requested, info: opened.append(info.revision_id),
        publication_guard=lambda _publication: True,
    )
    controller.add_settlement_listener(authority.on_settlement_publication)
    first_settlement = None
    first_revision_id = ""
    for index in range(257):
        assistant_id = f"assistant-success-{index}"
        scope = _branch_scope(
            session_id,
            session_id,
            (assistant_id,),
            f"interaction-success-{index}",
        )
        run = controller.register_run(
            scope, assistant_message_id=assistant_id, temporary=True
        )
        mutation = run.create_canvas(
            scope,
            tool_call_id=f"create-success-{index}",
            title=f"Successful capacity {index}",
            html=f"<!doctype html><h1>Successful capacity {index}</h1>",
        )
        settlement = run.finish_assistant_run(
            assistant_id,
            actual_run_id=scope.run_id,
            terminal_status="done",
        )
        assert settlement is not None
        assert controller.confirm_exact_settlement(settlement) is True
        if index == 0:
            first_settlement = settlement
            first_revision_id = mutation.revision.revision_id

    assert first_settlement is not None
    assert len(opened) == 257
    assert len(authority._publication_receipts) == 256

    assert controller.confirm_exact_settlement(first_settlement) is True
    assert opened.count(first_revision_id) == 1


def test_publication_receipt_concurrent_retry_is_source_free_and_exactly_once():
    session_id = "temporary-publication-concurrency"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    scope = _branch_scope(
        session_id,
        session_id,
        ("assistant-concurrent",),
        "interaction-concurrent",
    )
    created = controller.interactive_create_canvas(
        scope,
        origin_message_id="assistant-concurrent",
        title="Concurrent publication",
        html="<!doctype html><h1>Concurrent publication</h1>",
        temporary=True,
    )
    publication = CanvasSettlementPublication(
        publication_id="publication-" + "a" * 64,
        scope=scope,
        assistant_message_id="assistant-concurrent",
        revisions=(created.revision,),
    )
    entered = threading.Event()
    release = threading.Event()
    opened: list[str] = []

    def blocking_open(_requested, info):
        entered.set()
        assert release.wait(timeout=5)
        opened.append(info.revision_id)

    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: scope,
        canvas_controller=controller,
        auto_open=blocking_open,
        publication_guard=lambda _publication: True,
    )
    worker_errors: list[Exception] = []

    def publish() -> None:
        try:
            authority.on_settlement_publication(publication)
        except Exception as exc:  # pragma: no cover - asserted below
            worker_errors.append(exc)

    worker = threading.Thread(target=publish)
    worker.start()
    assert entered.wait(timeout=5)
    with pytest.raises(
        RuntimeError, match="canvas_publication_already_opening"
    ) as retry_error:
        authority.on_settlement_publication(publication)
    assert publication.publication_id not in str(retry_error.value)
    assert created.revision.revision_id not in str(retry_error.value)
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert worker_errors == []

    authority.on_settlement_publication(publication)
    assert len(authority._events[(session_id, created.revision.canvas_id)]) == 1
    assert opened == [created.revision.revision_id]


def test_publication_receipts_are_cleared_and_fenced_on_authority_dispose():
    session_id = "temporary-publication-dispose"
    controller = ConsoleCanvasController()
    controller.activate_session(session_id)
    scope = _branch_scope(
        session_id,
        session_id,
        ("assistant-dispose",),
        "interaction-dispose",
    )
    created = controller.interactive_create_canvas(
        scope,
        origin_message_id="assistant-dispose",
        title="Dispose publication",
        html="<!doctype html><h1>Dispose publication</h1>",
        temporary=True,
    )
    publication = CanvasSettlementPublication(
        publication_id="publication-" + "b" * 64,
        scope=scope,
        assistant_message_id="assistant-dispose",
        revisions=(created.revision,),
    )
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: scope,
        canvas_controller=controller,
        auto_open=lambda _requested, _info: (_ for _ in ()).throw(
            RuntimeError("injected opener failure")
        ),
        publication_guard=lambda _publication: True,
    )

    with pytest.raises(RuntimeError, match="injected opener failure"):
        authority.on_settlement_publication(publication)
    assert len(authority._publication_receipts) == 1

    authority.dispose()

    assert authority._publication_receipts == {}
    with pytest.raises(
        RuntimeError, match="canvas_publication_authority_disposed"
    ) as disposed_error:
        authority.on_settlement_publication(publication)
    assert publication.publication_id not in str(disposed_error.value)


def test_confirmed_json_submit_reaches_composer_as_valid_json_text():
    class Settlement:
        def try_settle(self, callback):
            callback()
            return True

    drafts = []
    controller = ConsoleCanvasController()
    controller.activate_session("temporary-bridge")
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: _scope("temporary-bridge"),
        canvas_controller=controller,
        bridge_sink=lambda target, text: drafts.append((target, text)),
    )
    created = authority.import_html(
        session_id="temporary-bridge",
        source="<!doctype html><p>bridge</p>",
        create_new=True,
    )
    browser_scope = authority.gateway_scope(
        session_id="temporary-bridge",
        browser_session_id="browser-bridge",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
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

    presentation, prepared = authority.prepare_bridge(browser_scope, request.request)
    response = authority.confirm_bridge(
        browser_scope,
        request,
        settlement=Settlement(),
        preparation=prepared,
    )

    assert presentation.complete_text == '{"answer":42,"ok":true}'
    assert response.status == "confirmed"
    assert drafts[0][1] == '{"answer":42,"ok":true}'


def test_submit_confirmation_refuses_when_captured_composer_changed() -> None:
    class Settlement:
        def try_settle(self, callback):
            callback()
            return True

    composer = {"revision": 1}
    drafts: list[str] = []

    def capture(_target):
        captured = composer["revision"]

        def apply(text: str) -> None:
            if composer["revision"] != captured:
                raise RuntimeError("Canvas composer changed")
            drafts.append(text)

        return apply

    controller = ConsoleCanvasController()
    controller.activate_session("temporary-changed-composer")
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: _scope("temporary-changed-composer"),
        canvas_controller=controller,
        bridge_prepare=capture,
    )
    created = authority.import_html(
        session_id="temporary-changed-composer",
        source="<!doctype html><p>bridge</p>",
        create_new=True,
    )
    browser_scope = authority.gateway_scope(
        session_id="temporary-changed-composer",
        browser_session_id="browser-changed-composer",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
    )
    bridge_request = CanvasBridgeRequest(
        version="canvas-v1",
        request_id="bridge-changed-composer",
        kind="submit",
        value="replacement",
    )

    _presentation, prepared = authority.prepare_bridge(browser_scope, bridge_request)
    composer["revision"] += 1
    response = authority.confirm_bridge(
        browser_scope,
        BridgeConfirmationRequest(approved=True, request=bridge_request),
        settlement=Settlement(),
        preparation=prepared,
    )

    assert response.status == "refused"
    assert drafts == []


def test_download_preparation_exposes_safe_metadata_without_retaining_bytes() -> None:
    controller = ConsoleCanvasController()
    controller.activate_session("temporary-download")
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: _scope("temporary-download"),
        canvas_controller=controller,
    )
    created = authority.import_html(
        session_id="temporary-download",
        source="<!doctype html><p>download</p>",
        create_new=True,
    )
    browser_scope = authority.gateway_scope(
        session_id="temporary-download",
        browser_session_id="browser-download",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
    )
    request = CanvasBridgeRequest(
        version="canvas-v1",
        request_id="bridge-download",
        kind="download",
        value={
            "filename": " result.json ",
            "mime_type": "application/json",
            "data": '{"answer":42}',
        },
    )

    presentation, prepared = authority.prepare_bridge(browser_scope, request)

    assert presentation.filename == "result.json"
    assert presentation.mime_type == "application/json"
    assert presentation.byte_size == 13
    assert presentation.complete_text == '{"answer":42}'
    assert presentation.canvas_title == "Canvas"
    assert presentation.revision_number == 1
    assert '{"answer":42}' not in repr(prepared)


def test_bridge_submit_is_fenced_to_captured_session_branch_and_pinned_view():
    active = {"session_id": "bridge-session"}
    scopes = {
        "bridge-session": _branch_scope(
            "bridge-session",
            "bridge-session",
            ("user-root", "assistant-left"),
            "bridge-run",
        ),
        "other-session": _branch_scope(
            "other-session",
            "other-session",
            ("other-user", "other-assistant"),
            "other-run",
        ),
    }

    def resolve(requested: str) -> CanvasScope:
        if requested != active["session_id"]:
            raise RuntimeError("Canvas session is no longer active")
        return scopes[requested]

    controller = ConsoleCanvasController()
    controller.activate_session("bridge-session")
    drafts = []
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=resolve,
        canvas_controller=controller,
        bridge_sink=lambda target, text: drafts.append((target, text)),
    )
    created = authority.import_html(
        session_id="bridge-session",
        source="<!doctype html><p>bridge</p>",
        create_new=True,
    )
    browser = authority.gateway_scope(
        session_id="bridge-session",
        browser_session_id="bridge-browser",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
        follow_latest=False,
    )
    request = BridgeConfirmationRequest(
        approved=True,
        request=CanvasBridgeRequest(
            version="canvas-v1",
            request_id="bridge-session-fence",
            kind="submit",
            value="confirmed draft",
        ),
    )

    _presentation, prepared = authority.prepare_bridge(browser, request.request)
    exact = authority.confirm_bridge(
        browser, request, settlement=_Settlement(), preparation=prepared
    )
    assert exact.status == "confirmed"
    assert drafts[0][0].session_id == "bridge-session"
    assert drafts[0][0].active_message_ids == ("user-root", "assistant-left")
    assert drafts[0][1] == "confirmed draft"

    drafts.clear()
    scopes["bridge-session"] = replace(
        scopes["bridge-session"],
        active_message_ids=("user-root", "assistant-right"),
    )
    sibling = authority.confirm_bridge(
        browser, request, settlement=_Settlement(), preparation=prepared
    )
    assert sibling.status == "refused"
    assert drafts == []

    scopes["bridge-session"] = replace(
        scopes["bridge-session"],
        active_message_ids=("user-root", "assistant-left"),
    )
    active["session_id"] = "other-session"
    switched = authority.confirm_bridge(
        browser, request, settlement=_Settlement(), preparation=prepared
    )
    assert switched.status == "refused"
    assert drafts == []


def test_bridge_effect_revalidates_after_switch_during_settlement():
    scope = _branch_scope(
        "bridge-race",
        "bridge-race",
        ("root", "left"),
        "bridge-race-run",
    )
    current = {"scope": scope}
    controller = ConsoleCanvasController()
    controller.activate_session(scope.session_id)
    drafts = []
    authority = NativeConsoleCanvasAuthority(
        scope_resolver=lambda _requested: current["scope"],
        canvas_controller=controller,
        bridge_sink=lambda target, text: drafts.append((target, text)),
    )
    created = authority.import_html(
        session_id=scope.session_id,
        source="<!doctype html><p>race</p>",
        create_new=True,
    )
    browser = authority.gateway_scope(
        session_id=scope.session_id,
        browser_session_id="bridge-race-browser",
        canvas_id=created.canvas_id,
        revision_id=created.revision_id,
    )
    request = BridgeConfirmationRequest(
        approved=True,
        request=CanvasBridgeRequest(
            version="canvas-v1",
            request_id="bridge-race-request",
            kind="submit",
            value={"safe": True},
        ),
    )

    class SwitchDuringSettlement:
        def try_settle(self, callback):
            current["scope"] = replace(
                scope, active_message_ids=("root", "right")
            )
            callback()
            return True

    _presentation, prepared = authority.prepare_bridge(browser, request.request)
    response = authority.confirm_bridge(
        browser,
        request,
        settlement=SwitchDuringSettlement(),
        preparation=prepared,
    )

    assert response.status == "refused"
    assert drafts == []


class _Settlement:
    def try_settle(self, callback):
        callback()
        return True
