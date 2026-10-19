"""A Send pressed while a conversation resume is activating is refused visibly.

Console UAT 2026-10-17 finding 1 (round 11, ``qa/console-uat-2026-10-17/``):
pressing Send shortly after clicking a workspace-rail conversation row was
**silently dropped** -- no provider call, no persisted turn, no notice. The
resume's slow path (``open_console_workspace_conversation`` ->
``_resume_console_workspace_conversation``) occupies the screen pump across
multiple awaits while the active session AND the composer binding still name
the OLD chat, so ``refuse_send_from_unbound_composer`` sees bound == active ==
OLD and cannot fire; the decoupled send task then dies pinned to the OLD chat
in branches that return without any notice.

These tests hold the resume mid-flight (the ``Gate`` pattern of
``test_console_tab_switch_binds_composer.py``) and demand the same visible
refusal the tab-switch race got in TASK-33622.7: nothing dispatches during
the window, the user is told, and a fresh Enter after the resume settles
sends to the newly activated chat.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import (
    _configure_native_ready_console,
    _select_llamacpp_console,
)
from Tests.UI.test_console_send_acknowledgement import press, until
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules import workspace as workspace_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile


@dataclass
class Gate:
    """One held await: ``entered`` when reached, waits for ``release``."""

    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


_GATES: list[Gate] = []


@contextlib.asynccontextmanager
async def _running(host):
    """Run the Console; release every held await before the app shuts down."""
    try:
        async with host.run_test(size=(160, 48)) as pilot:
            try:
                yield pilot
            finally:
                for gate in _GATES:
                    gate.release.set()
                _GATES.clear()
    finally:
        # ConsoleHarness stops the view, while its borrowed app owns storage.
        app = host.app_instance
        await app._shutdown_app_owned_lifecycles()
        await app.evaluation_orchestrator.aclose()
        app.local_workspace_db.close()
        app.local_library_collections_db.close()
        app.subscriptions_db.close()


def _record_dispatch(console) -> list[tuple[str | None, str]]:
    """Record the chat each send is dispatched to, instead of sending it."""
    dispatched: list[tuple[str | None, str]] = []

    async def dispatch(draft, stash=None, *, session_id=None):
        dispatched.append((session_id, draft))
        return True

    console._dispatch_console_draft_send = dispatch
    return dispatched


def _record_notices(console) -> list[str]:
    notices: list[str] = []
    real = console.app_instance.notify

    def notify(message, *args, **kwargs):
        notices.append(str(message))
        return real(message, *args, **kwargs)

    console.app_instance.notify = notify
    return notices


def _refused(notices: list[str]) -> bool:
    """Whether the switch guard refused a send (TASK-33622.7 copy)."""
    return any("still switching chats" in notice.lower() for notice in notices)


def _hold_resume_hydration() -> Gate:
    """Hold the next conversation resume at its conversation-tree DB read."""
    gate = Gate()
    _GATES.append(gate)
    real = workspace_module.load_console_conversation_tree

    async def held(*args, **kwargs):
        if not gate.entered.is_set():
            gate.entered.set()
            await gate.release.wait()
        return await real(*args, **kwargs)

    workspace_module.load_console_conversation_tree = held
    return gate




async def until_result(factory, *, timeout: float = 15.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = factory()
        if value is not None:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {factory}")

def _resumed_session(store, conversation_id: str):
    """The store's active session once it owns the resumed conversation."""
    for item in store.sessions():
        if (
            item.id == store.active_session_id
            and item.persisted_conversation_id == conversation_id
        ):
            return item
    return None

def _host() -> ConsoleHarness:
    app = _build_test_app()
    # The durable persistence adapter a persisted conversation needs. The DB
    # must be FILE-backed at the sandbox's resolved path: the resume path's
    # conversation-tree loader re-opens the database by path, and a fresh
    # connection to ":memory:" would see an empty schema.
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.config import get_chachanotes_db_path

    app.chachanotes_db = CharactersRAGDB(
        str(get_chachanotes_db_path()), "resume-guard-test"
    )
    # The conversation scope service wires from app.chachanotes_db during
    # __init__ (app.py), when the factory app still has None; the resume
    # path's tree loader reads through it, so rewire after the attach.
    app._wire_chat_conversation_services()
    # chat_defaults on disk decide the resumed session's base settings
    # (hydration starts from chat_defaults, not the seeding session), so a
    # send-ready llama.cpp default keeps the resumed chat ready too.
    _configure_native_ready_console(app, model="test-model")
    return ConsoleHarness(app)


async def _seed_saved_conversation(console, store):
    """Persist one saved conversation, closed, and return (session, conv id).

    The conversation must be durable but NOT open: ``open_console_workspace_
    conversation`` only walks the slow (multi-await) resume path for a
    conversation no live native session holds.
    """
    active = next(
        item for item in store.sessions() if item.id == store.active_session_id
    )
    saved = store.create_session(
        title="Saved chat", settings=active.settings, activate=False
    )
    conversation_id = store.persist_session_if_needed(saved.id)
    assert conversation_id, "seed session must persist a durable conversation"
    # Persist the settings snapshot the way production does on a settings
    # change: hydration restores the conversation's metadata settings, so
    # without this the resumed session would boot on chat_defaults.
    assert store.persistence.update_conversation_console_session_settings(
        conversation_id=conversation_id, settings=saved.settings
    )
    store.close_session(saved.id)
    assert all(item.id != saved.id for item in store.sessions())
    return saved, conversation_id


@pytest.mark.asyncio
async def test_enter_during_a_conversation_resume_is_refused_visibly():
    """AC#1/#2: nothing dispatches mid-resume and the user is told why."""
    host = _host()
    async with _running(host) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        saved, conversation_id = await _seed_saved_conversation(console, store)

        dispatched = _record_dispatch(console)
        notices = _record_notices(console)
        composer.load_draft("typed during the resume")
        gate = _hold_resume_hydration()

        resume = asyncio.create_task(
            console._workspace.open_console_workspace_conversation(conversation_id)
        )
        await until(gate.entered.is_set, timeout=10)
        press(host, "enter")
        await pilot.pause(0.2)

        assert _refused(notices), (
            "Enter during a resume activation must refuse visibly, got notices: "
            f"{notices}"
        )
        assert dispatched == [], (
            "nothing may dispatch while the resume activation is in flight; "
            f"dispatched={dispatched!r}"
        )

        gate.release.set()
        await until(lambda: resume.done(), timeout=15)
        resumed = await until_result(
            lambda: _resumed_session(store, conversation_id), timeout=15
        )

        # After the resume settles, a fresh Enter goes to the resumed chat.
        composer.load_draft("after the resume")
        press(host, "enter")
        await until(lambda: bool(dispatched), timeout=15)
        assert dispatched[0][0] == resumed.id
        assert dispatched[0][1] == "after the resume"


@pytest.mark.asyncio
async def test_enter_right_after_a_resume_settles_sends_to_the_resumed_chat():
    """AC#3: once the resume has settled, Enter sends into the resumed chat."""
    host = _host()
    async with _running(host) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        saved, conversation_id = await _seed_saved_conversation(console, store)

        dispatched = _record_dispatch(console)
        opened = await console._workspace.open_console_workspace_conversation(
            conversation_id
        )
        assert opened is True
        resumed = await until_result(
            lambda: _resumed_session(store, conversation_id), timeout=15
        )

        from textual.widgets import Button as _Button

        composer.load_draft("sent right after the resume")
        await until(
            lambda: not console.query_one("#console-send-message", _Button).disabled,
            timeout=15,
        )
        press(host, "enter")
        await until(lambda: bool(dispatched), timeout=15)
        assert dispatched[0][0] == resumed.id
        assert dispatched[0][1] == "sent right after the resume"
