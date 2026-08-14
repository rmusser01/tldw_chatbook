"""PROBE P1 (task-15860 Task 0) -- Console history continuity across a
nav-away, by EXECUTION.

NOT a regression test. This file exists to answer one question the
headless-wake plan's recommended design (A: app-owned Console runtime)
rests on, and which was READ rather than RUN:

    Are DB rows appended to a Console conversation while no Console screen
    is mounted visible at the next Console mount, in the transcript AND in
    the next send's provider payload?

The claim under probe: `ChatScreen.save_state` ->
`_serialize_native_console_state` snapshots `sessions`/`messages_by_session`
into `ScreenStateStore`, and `_restore_native_console_state` rebuilds the
store from THAT payload and never re-reads ChaChaNotes -- so a headless
(DB-only) writer is invisible and divergent at remount.

Everything here runs through the REAL navigation API
(`app.handle_screen_navigation(NavigateToScreen(...))`), the real
`ChatScreen`, a real on-disk ChaChaNotes DB, and the production
`ChatPersistenceService` for the out-of-band appends.
"""
from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.Chat.test_console_fleet_wake import _RecordingWakeGateway

HEADLESS_SYSTEM = "HEADLESS-SYSTEM-ROW-P1"
HEADLESS_ASSISTANT = "HEADLESS-ASSISTANT-ROW-P1"


def _report(title: str, lines: list[str]) -> None:
    print(f"\n===== {title} =====")
    for line in lines:
        print(f"  {line}")


@pytest.mark.asyncio
async def test_probe_p1_db_rows_written_while_console_is_unmounted(tmp_path):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _RecordingWakeGateway(reply="assistant one")
    app.console_provider_gateway_factory = lambda: gateway

    findings: list[str] = []

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller = chat._ensure_console_chat_controller()
        store = chat._console_chat_store
        session = store.sessions()[0]
        session_id = session.id

        outcome = await controller.submit_draft(
            "first user message", session_id=session_id
        )
        findings.append(f"pre-nav submit accepted={outcome.accepted}")
        assert outcome.accepted, "the seeding send must be accepted"

        conversation_id = store.sessions()[0].persisted_conversation_id
        findings.append(f"persisted conversation_id={conversation_id}")
        assert conversation_id, "the probe needs a PERSISTED conversation"

        db = app.chachanotes_db
        persistence = store.persistence
        rows_before = db.get_messages_for_conversation(conversation_id, limit=100)
        leaf_before = db.get_conversation_active_leaf(conversation_id)
        findings.append(
            "pre-nav DB rows: "
            + ", ".join(f"{r['sender']}:{r['content'][:24]!r}" for r in rows_before)
        )
        findings.append(f"pre-nav active_leaf={leaf_before}")
        in_memory_before = [
            (m.role.value, m.content[:24])
            for m in store.messages_for_session(session_id)
        ]
        findings.append(f"pre-nav in-memory: {in_memory_before}")

        # ---- navigate away through the REAL navigation API -------------
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        findings.append(f"after nav-away screen={type(app.screen).__name__}")
        assert chat not in app.screen_stack, "Console must actually unmount"
        assert controller._shutdown_requested.is_set(), (
            "leaving Console must shut the controller down"
        )

        # ---- append rows to the conversation with NO Console mounted ----
        last_id = rows_before[-1]["id"] if rows_before else None
        system_id = persistence.create_message(
            conversation_id=conversation_id,
            sender="system",
            content=HEADLESS_SYSTEM,
            parent_message_id=last_id,
        )
        assistant_id = persistence.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            content=HEADLESS_ASSISTANT,
            parent_message_id=system_id,
        )
        findings.append(
            f"headless appends: system={system_id} assistant={assistant_id}"
        )
        leaf_after_append = db.get_conversation_active_leaf(conversation_id)
        findings.append(
            "active_leaf after DB-only appends="
            f"{leaf_after_append} (unchanged={leaf_after_append == leaf_before})"
        )

        # ---- navigate BACK to Console -----------------------------------
        gateway.payloads.clear()
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        chat2 = app.screen
        assert isinstance(chat2, ChatScreen), type(chat2).__name__
        assert chat2 is not chat, "screens are never cached"
        await _wait_for_selector(chat2, pilot, "#console-native-composer")

        store2 = chat2._console_chat_store
        restored = store2.messages_for_session(session_id)
        rendered = [(m.role.value, m.content[:40]) for m in restored]
        findings.append(f"post-return in-memory transcript: {rendered}")

        # (a) do the rows render?
        transcript_text = "\n".join(m.content for m in restored)
        system_visible = HEADLESS_SYSTEM in transcript_text
        assistant_visible = HEADLESS_ASSISTANT in transcript_text
        findings.append(
            f"(a) headless rows in restored transcript: "
            f"system={system_visible} assistant={assistant_visible}"
        )

        # ...and in the actual rendered DOM?
        dom_text = " ".join(
            str(getattr(node, "renderable", ""))
            for node in chat2.query("*")
        )
        findings.append(
            "(a-dom) headless rows in rendered widgets: "
            f"system={HEADLESS_SYSTEM in dom_text} "
            f"assistant={HEADLESS_ASSISTANT in dom_text}"
        )

        # (b) are they in the next send's provider payload?
        controller2 = chat2._ensure_console_chat_controller()
        session2 = next(
            s for s in store2.sessions() if s.id == session_id
        )
        findings.append(
            "post-return session persisted_conversation_id="
            f"{session2.persisted_conversation_id}"
        )
        outcome2 = await controller2.submit_draft(
            "second user message", session_id=session_id
        )
        findings.append(f"post-return submit accepted={outcome2.accepted}")
        assert gateway.payloads, "the second send never reached the provider"
        payload = gateway.payloads[-1]
        payload_shape = [(m["role"], str(m["content"])[:40]) for m in payload]
        findings.append(f"(b) provider payload: {payload_shape}")
        payload_text = "\n".join(str(m["content"]) for m in payload)
        findings.append(
            "(b) headless rows in provider payload: "
            f"system={HEADLESS_SYSTEM in payload_text} "
            f"assistant={HEADLESS_ASSISTANT in payload_text}"
        )

        # (c) active leaf + does the next persisted append fork the tree?
        leaf_final = db.get_conversation_active_leaf(conversation_id)
        rows_after = db.get_messages_for_conversation(conversation_id, limit=100)
        by_id = {r["id"]: r for r in rows_after}
        findings.append("(c) final DB rows (id | sender | parent | content):")
        for r in rows_after:
            findings.append(
                f"      {r['id'][:8]} | {r['sender']:9} | "
                f"{str(r.get('parent_message_id'))[:8]:8} | {r['content'][:34]!r}"
            )
        findings.append(f"(c) final active_leaf={leaf_final}")
        leaf_row = by_id.get(leaf_final) if leaf_final else None
        findings.append(
            "(c) active_leaf points at: "
            + (
                f"{leaf_row['sender']} {leaf_row['content'][:34]!r}"
                if leaf_row
                else repr(leaf_final)
            )
        )
        new_user_rows = [
            r
            for r in rows_after
            if r["sender"] == "user" and "second user message" in r["content"]
        ]
        if new_user_rows:
            parent = new_user_rows[0].get("parent_message_id")
            parent_row = by_id.get(parent)
            forked = parent != assistant_id
            findings.append(
                "(c) next persisted append parent="
                + (
                    f"{parent_row['sender']} {parent_row['content'][:34]!r}"
                    if parent_row
                    else repr(parent)
                )
                + f" -> FORKS away from the headless rows: {forked}"
            )
        else:
            findings.append("(c) the second send persisted NO user row")

    _report("P1 -- DB rows written while Console is unmounted", findings)


@pytest.mark.asyncio
async def test_probe_p1_variant_headless_writer_also_moves_the_active_leaf(tmp_path):
    """The obvious objection to the main probe, closed.

    A real design-B/C headless writer would not stop at raw rows: it would
    also write through the local-only active-leaf pointer the way
    `ConsoleChatStore._persist_active_leaf` does. Does maintaining the
    pointer make the headless rows survive the remount?
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _RecordingWakeGateway(reply="assistant one")
    app.console_provider_gateway_factory = lambda: gateway

    findings: list[str] = []

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")
        controller = chat._ensure_console_chat_controller()
        store = chat._console_chat_store
        session_id = store.sessions()[0].id
        await controller.submit_draft("first user message", session_id=session_id)
        conversation_id = store.sessions()[0].persisted_conversation_id
        db = app.chachanotes_db

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()

        rows = db.get_messages_for_conversation(conversation_id, limit=100)
        system_id = store.persistence.create_message(
            conversation_id=conversation_id,
            sender="system",
            content=HEADLESS_SYSTEM,
            parent_message_id=rows[-1]["id"],
        )
        assistant_id = store.persistence.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            content=HEADLESS_ASSISTANT,
            parent_message_id=system_id,
        )
        # ...and write through the active-leaf pointer, as the store would.
        db.set_conversation_active_leaf(conversation_id, assistant_id)
        findings.append(
            f"active_leaf written through to the headless assistant: "
            f"{db.get_conversation_active_leaf(conversation_id) == assistant_id}"
        )

        gateway.payloads.clear()
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        chat2 = app.screen
        await _wait_for_selector(chat2, pilot, "#console-native-composer")
        store2 = chat2._console_chat_store
        restored = "\n".join(
            m.content for m in store2.messages_for_session(session_id)
        )
        findings.append(
            "headless rows in the restored transcript (leaf maintained): "
            f"system={HEADLESS_SYSTEM in restored} "
            f"assistant={HEADLESS_ASSISTANT in restored}"
        )
        controller2 = chat2._ensure_console_chat_controller()
        await controller2.submit_draft("second user message", session_id=session_id)
        payload_text = "\n".join(str(m["content"]) for m in gateway.payloads[-1])
        findings.append(
            "headless rows in the next provider payload (leaf maintained): "
            f"system={HEADLESS_SYSTEM in payload_text} "
            f"assistant={HEADLESS_ASSISTANT in payload_text}"
        )
        rows_after = db.get_messages_for_conversation(conversation_id, limit=100)
        by_id = {r["id"]: r for r in rows_after}
        new_user = [
            r
            for r in rows_after
            if r["sender"] == "user" and "second user message" in r["content"]
        ]
        if new_user:
            parent = new_user[0].get("parent_message_id")
            findings.append(
                "next persisted append parent="
                + repr(by_id.get(parent, {}).get("content", parent))
                + f" -> FORKS away from the headless rows: {parent != assistant_id}"
            )
        findings.append(
            "final active_leaf still the headless assistant: "
            f"{db.get_conversation_active_leaf(conversation_id) == assistant_id}"
        )

    _report("P1 variant -- headless writer maintains the active leaf", findings)


@pytest.mark.asyncio
async def test_probe_p1_control_rows_written_while_console_is_mounted(tmp_path):
    """Control: the SAME out-of-band DB append with Console still mounted.

    Separates "the snapshot hides it" from "nothing ever reads DB appends".
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _RecordingWakeGateway(reply="assistant one")
    app.console_provider_gateway_factory = lambda: gateway

    findings: list[str] = []

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller = chat._ensure_console_chat_controller()
        store = chat._console_chat_store
        session_id = store.sessions()[0].id
        await controller.submit_draft("first user message", session_id=session_id)
        conversation_id = store.sessions()[0].persisted_conversation_id
        db = app.chachanotes_db
        rows = db.get_messages_for_conversation(conversation_id, limit=100)
        store.persistence.create_message(
            conversation_id=conversation_id,
            sender="assistant",
            content=HEADLESS_ASSISTANT,
            parent_message_id=rows[-1]["id"] if rows else None,
        )
        await pilot.pause()
        for _ in range(10):
            await pilot.pause()
        live = "\n".join(m.content for m in store.messages_for_session(session_id))
        findings.append(
            "mounted-Console DB append visible in the live store: "
            f"{HEADLESS_ASSISTANT in live}"
        )
        gateway.payloads.clear()
        await controller.submit_draft("second user message", session_id=session_id)
        payload_text = "\n".join(
            str(m["content"]) for m in gateway.payloads[-1]
        )
        findings.append(
            "mounted-Console DB append visible in the next payload: "
            f"{HEADLESS_ASSISTANT in payload_text}"
        )

    _report("P1 control -- DB append with Console MOUNTED", findings)
