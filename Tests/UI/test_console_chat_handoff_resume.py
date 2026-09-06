"""Real cached-Console navigation for the CHAT handoff channel."""

import asyncio

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_live_work_handoffs import _wait_for_production_chat_screen
from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
    close_owned_console_test_apps as close_owned_console_test_apps,
)
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel


async def _wait_until(pilot, condition):
    deadline = asyncio.get_running_loop().time() + 6.0
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await pilot.pause(0.02)
    assert condition(), "Console navigation or handoff did not settle"


async def _navigate(app, pilot, destination, screen_name):
    app.post_message(NavigateToScreen(destination))
    await _wait_until(
        pilot,
        lambda: (
            type(app.screen).__name__ == screen_name
            and not app._screen_navigation_in_progress()
        ),
    )


def _character_app(tmp_path):
    app = _build_test_app()
    db = CharactersRAGDB(tmp_path / "resume-character.sqlite", "resume-test")
    app.chachanotes_db = db
    app.local_character_persona_service.db = db
    app.local_chat_dictionary_service.db = db
    character_id = db.add_character_card(
        {"name": "Resume Ada", "first_message": "Hello from Resume Ada."}
    )
    payload = ChatHandoffPayload(
        source="personas",
        item_type="character-card",
        title="Resume Ada",
        body="",
        runtime_backend="local",
        source_owner="local",
        source_selector_state="local",
        metadata={
            "intent": "start_chat",
            "selected_kind": "character",
            "backend": "local",
            "selected_record_id": str(character_id),
            "selected_target_id": f"local:character:{character_id}",
            "selected_name": "Resume Ada",
        },
    )
    return app, character_id, payload


@pytest.mark.parametrize("new_handoff", [False, True], ids=["no-new", "new"])
@pytest.mark.asyncio
async def test_warm_chat_handoff_creates_one_character_session(tmp_path, new_handoff):
    app, character_id, payload = _character_app(tmp_path)
    async with app.run_test(size=(180, 40)) as pilot:
        console = await _wait_for_production_chat_screen(app, pilot)
        store = console._ensure_console_chat_store()
        # Preserve an existing conversation; a pristine initial session may
        # legitimately be repurposed by the Start Chat contract (covered in UAT).
        original_session = store.active_session_id
        store.append_message(
            original_session, role=ConsoleMessageRole.USER, content="Keep this chat."
        )
        before = {session.id for session in store.sessions()}
        await _navigate(app, pilot, "home", "HomeScreen")
        if new_handoff:
            revision = app.pending_handoffs.stage(HandoffChannel.CHAT, payload)
        await pilot.pause(0.2)
        assert {session.id for session in store.sessions()} == before
        assert app.pending_handoffs.has_pending(HandoffChannel.CHAT) is new_handoff

        await _navigate(app, pilot, "chat", "ChatScreen")
        assert app.screen is console
        await _wait_until(
            pilot,
            lambda: (
                not app.pending_handoffs.has_pending(HandoffChannel.CHAT)
                and not console._handoff_consumption_in_progress
            ),
        )
        created = [session for session in store.sessions() if session.id not in before]
        assert len(created) == int(new_handoff)
        assert (
            store.messages_for_session(original_session)[0].content == "Keep this chat."
        )
        if new_handoff:
            assert (
                app.pending_handoffs.exact_revision_status(
                    HandoffChannel.CHAT, revision
                )
                == "settled"
            )
            session = created[0]
            assert str(session.character_id) == str(character_id)
            assert session.title == "Chat with Resume Ada"
            assert any(
                message.content == "Hello from Resume Ada."
                for message in store.messages_for_session(session.id)
            )
        # Another real leave/return must not replay an acknowledged handoff.
        after = {session.id for session in store.sessions()}
        await _navigate(app, pilot, "home", "HomeScreen")
        await _navigate(app, pilot, "chat", "ChatScreen")
        await pilot.pause(0.2)
        assert app.screen is console
        assert {session.id for session in store.sessions()} == after
        assert app.pending_handoffs.claim(HandoffChannel.CHAT) is None


@pytest.mark.asyncio
async def test_suspending_again_stops_the_pending_chat_resume_timer(
    tmp_path, monkeypatch
):
    app, character_id, payload = _character_app(tmp_path)
    async with app.run_test(size=(180, 40)) as pilot:
        console = await _wait_for_production_chat_screen(app, pilot)
        store = console._ensure_console_chat_store()
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.USER,
            content="Keep this chat.",
        )
        before = {session.id for session in store.sessions()}
        await _navigate(app, pilot, "home", "HomeScreen")
        revision = app.pending_handoffs.stage(HandoffChannel.CHAT, payload)
        timers = []
        set_timer = console.set_timer

        def hold_chat_timer(delay, callback, **kwargs):
            timer = set_timer(delay, callback, **kwargs)
            if callback == console._consume_pending_chat_handoff:
                # Hold only delivery, not the real Timer's suspend/stop lifecycle.
                timer.pause()
                timers.append(timer)
            return timer

        monkeypatch.setattr(console, "set_timer", hold_chat_timer)
        await _navigate(app, pilot, "chat", "ChatScreen")
        assert app.screen is console
        assert len(timers) == 1
        assert timers[0] in console._console_resume_handoff_timers
        await _navigate(app, pilot, "home", "HomeScreen")
        assert console._console_resume_handoff_timers == []
        timers[0].resume()
        await pilot.pause(0.2)
        assert app.pending_handoffs.has_pending(HandoffChannel.CHAT)
        assert {session.id for session in store.sessions()} == before

        monkeypatch.setattr(console, "set_timer", set_timer)
        await _navigate(app, pilot, "chat", "ChatScreen")
        await _wait_until(
            pilot,
            lambda: (
                app.pending_handoffs.exact_revision_status(
                    HandoffChannel.CHAT, revision
                )
                == "settled"
            ),
        )
        created = [session for session in store.sessions() if session.id not in before]
        assert len(created) == 1
        assert str(created[0].character_id) == str(character_id)
