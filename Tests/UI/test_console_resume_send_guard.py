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

    async def dispatch(draft, stash=None, *, session_id=None, _captured_inputs=None):
        assert _captured_inputs is not None
        assert _captured_inputs.session_id == session_id
        assert _captured_inputs.draft == draft
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


def _hold_resume_hydration(monkeypatch) -> Gate:
    """Hold the next conversation resume at its conversation-tree DB read."""
    gate = Gate()
    _GATES.append(gate)
    real = workspace_module.load_console_conversation_tree

    async def held(*args, **kwargs):
        if not gate.entered.is_set():
            gate.entered.set()
            await gate.release.wait()
        return await real(*args, **kwargs)

    monkeypatch.setattr(workspace_module, "load_console_conversation_tree", held)
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
@pytest.mark.parametrize("entrypoint", ("rail", "library", "switcher"))
async def test_enter_during_a_conversation_resume_is_refused_visibly(
    monkeypatch, entrypoint
):
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
        composer.focus()
        await until(lambda: host.focused is composer)
        gate = _hold_resume_hydration(monkeypatch)

        if entrypoint == "library":
            from tldw_chatbook.UI.Console_Modules.archive import (
                consume_conversation_resume,
            )
            from tldw_chatbook.UI.Navigation.pending_handoff_store import (
                ConsoleConversationResumeIntent,
                HandoffChannel,
            )

            host.app_instance.pending_handoffs.stage(
                HandoffChannel.CONSOLE_CONVERSATION_RESUME,
                ConsoleConversationResumeIntent(conversation_id),
            )
            operation = consume_conversation_resume(console)
        elif entrypoint == "switcher":
            from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
                ConsoleSwitcherChoice,
            )

            page = await console._workspace.load_console_session_switcher_history(
                query="", offset=0, limit=50
            )
            entry = next(
                item for item in page.entries if item.conversation_id == conversation_id
            )
            operation = console._session._apply_console_switcher_choice(
                ConsoleSwitcherChoice("activate", entry)
            )
        else:
            operation = console._workspace.open_console_workspace_conversation(
                conversation_id
            )
        resume = asyncio.create_task(operation)
        try:
            await until(gate.entered.is_set, timeout=10)
            press(host, "enter")
            await until(lambda: _refused(notices), timeout=10)
            assert composer.draft_text() == "typed during the resume"
            assert dispatched == [], (
                "nothing may dispatch while the resume activation is in flight; "
                f"dispatched={dispatched!r}"
            )
        finally:
            gate.release.set()
            # The test owns this detached resume, including its native reads.
            # Retire it before the harness can close the executor/databases.
            await resume
        assert console._session._conversation_activation_depth == 0
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


def test_resume_refusal_records_only_code_owned_send_diagnostics(caplog):
    import logging
    from types import SimpleNamespace
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor

    notices = []
    monitor = UIResponsivenessMonitor(enabled=False)
    session = SimpleNamespace(
        _conversation_activation_depth=1,
        app_instance=SimpleNamespace(
            ui_responsiveness_monitor=monitor,
            notify=lambda message, **kwargs: notices.append(message),
        ),
    )
    with caplog.at_level(logging.INFO, logger="tldw_chatbook.diagnostics.console"):
        try:
            assert ConsoleSessionController.refuse_send_from_unbound_composer(session)
        finally:
            monitor.close()
    assert _refused(notices)
    assert not monitor._diagnostic_sink_failed
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "tldw_chatbook.diagnostics.console"
    ]
    assert messages == [
        "event=console_send_stage component=console "
        "operation=conversation_resume_activation phase=ui_submit status=not_dispatched"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ("pending_send_token_mismatch", "pending_send_slot_busy")
)
async def test_pending_send_refusals_reach_the_metadata_sink(reason, caplog):
    import logging
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_send_diagnostics import send_diagnostic_scope
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor

    notices = []
    monitor = UIResponsivenessMonitor(enabled=False)
    screen = SimpleNamespace(
        _console_pending_send=(
            None
            if reason == "pending_send_token_mismatch"
            else SimpleNamespace(session_id="test-session")
        ),
        _console_visible_send_session_id=lambda: "test-session",
        app_instance=SimpleNamespace(
            notify=lambda text, **kwargs: notices.append(text)
        ),
    )
    with caplog.at_level(logging.INFO, logger="tldw_chatbook.diagnostics.console"):
        try:
            async with send_diagnostic_scope("ui_action", monitor):
                sent = (
                    await ChatScreen._send_console_message_from_visible_action_observed(
                        screen,
                        pending_send_token=(
                            object()
                            if reason == "pending_send_token_mismatch"
                            else None
                        ),
                    )
                )
            assert not sent
        finally:
            await asyncio.to_thread(monitor.close)
    assert notices
    assert not monitor._diagnostic_sink_failed
    assert any(
        f"operation={reason}" in record.getMessage()
        and "status=not_dispatched" in record.getMessage()
        for record in caplog.records
    )
