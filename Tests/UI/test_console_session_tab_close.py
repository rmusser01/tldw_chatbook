"""Closing a Console session tab from the real tab strip (TASK-33621.15).

On dev @ 64579cce2c no close attempt ever closed a tab: the close worker
called ``self._console_runtime()`` on ``ConsoleSessionController``, which
only has the injected ``_console_runtime_accessor``, and the worker's
``exit_on_error=False`` swallowed the ``AttributeError`` with no toast and
no log line. The routing tests in ``test_console_button_routing.py`` pinned
the right outcome but call ``Button.press()`` directly, so every case here
goes through a real Pilot mouse click on a mounted ``ChatScreen`` and lets
the real close worker run against the real chat store and ``ConsoleRuntime``.

Each case runs under ``@private_profile_test``: ``_build_test_app`` reloads
config, which the per-test config sandbox refuses locally with
``RecoveryRequired: raw_source_selection_changed`` (see the Tests/UI
``RecoveryRequired`` lesson in ``backlog/docs/lessons-testing-evidence.md``).
"""

from __future__ import annotations

import time

import pytest
from loguru import logger
from textual.css.query import NoMatches
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_button_routing import (
    _mounted_console,
    _wait_for_confirmation,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

#: Wide enough that every tab and its ✕ is mounted and hit-testable.
_SIZE = (160, 44)
_TAB_PREFIX = "console-session-tab-"
_SAVED_TITLE = "Saved notes"


def _ready_app():
    """A test app whose user has already sent a first message.

    That is the persisted flag the app sets after a real first send; without
    it the first-run setup card covers the workbench and a real click on the
    tab strip lands on its backdrop.
    """

    app = _build_test_app()
    app.app_config.setdefault("console", {})["onboarding"] = {
        "first_send_completed": True
    }
    return app


def _open_tab_ids(console) -> set[str]:
    """Session ids of the tabs the strip currently renders."""

    return {
        str(button.id).removeprefix(_TAB_PREFIX)
        for button in console.query(Button)
        if str(button.id or "").startswith(_TAB_PREFIX)
    }


async def _await_tabs(console, pilot, expected: set[str]) -> None:
    """Wait until the strip renders exactly ``expected`` tabs, laid out.

    The strip mounts its buttons asynchronously after a sync, so a click
    straight after one can land before the ✕ has a region.
    """

    def laid_out() -> bool:
        if _open_tab_ids(console) != expected:
            return False
        try:
            return all(
                console.query_one(
                    f"#console-close-session-tab-{session_id}"
                ).region.area
                for session_id in expected
            )
        except NoMatches:
            return False

    assert await _settle(pilot, laid_out), (
        f"tab strip never settled: {_open_tab_ids(console)} != {expected}"
    )
    await pilot.pause()


async def _show_tabs(console, pilot, expected: set[str]) -> None:
    """Re-render the strip from the store and wait for ``expected`` tabs."""

    await console._sync_native_console_chat_ui()
    await _await_tabs(console, pilot, expected)


async def _click(pilot, selector: str, **kwargs) -> None:
    """Really click ``selector`` and fail loudly if something else was hit.

    Waits out the button's own ``-active`` press effect first: Textual's
    ``Button._on_click`` ignores a click while it is set (about 0.2 s),
    which a person never re-clicks inside but a test easily does.
    """

    target = pilot.app.screen.query_one(selector)
    assert await _settle(pilot, lambda: not target.has_class("-active"))
    region = target.region
    under, _ = pilot.app.get_widget_at(*region.offset)
    hit = await pilot.click(selector, **kwargs)
    assert hit, f"{selector} not hit-testable: region={region} under={under!r}"


async def _settle(pilot, predicate, *, timeout: float = 10.0) -> bool:
    """Poll ``predicate`` while the close worker runs; no fixed sleeps.

    Deadline-based rather than a fixed number of polls: the close does real
    DB work, and on a loaded machine a 3 s budget was measured too short.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.01)
    return bool(predicate())


def _session_ids(store) -> list[str]:
    """Open sessions in tab order, from the store (the strip can be mid-render)."""

    return [session.id for session in store.sessions()]


def _saved_conversation(app, tmp_path):
    """Back one saved conversation with a real ChaChaNotes DB file."""

    db = CharactersRAGDB(tmp_path / "tab-close.db", client_id="tab-close")
    app.chachanotes_db = db
    app.local_chat_conversation_service = ChatConversationService(db)
    conversation_id = db.add_conversation({"title": _SAVED_TITLE})
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "saved"}
    )
    return db, conversation_id, message_id


def _restore_saved_tab(store, conversation_id, message_id):
    """Open the saved conversation as an idle Console tab."""

    return store.restore_persisted_session(
        title=_SAVED_TITLE,
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=[
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER,
                content="saved",
                persisted_message_id=message_id,
            )
        ],
    )


def _record_notifications(app) -> list[tuple[str, str]]:
    """Record what the Console tells the user (the toast sink, not a seam)."""

    notes: list[tuple[str, str]] = []

    def notify(message, *, severity="information", **_kwargs):
        notes.append((str(message), str(severity)))

    app.notify = notify
    return notes


@pytest.mark.asyncio
@private_profile_test
async def test_clicking_x_closes_an_idle_saved_tab_and_a_blank_tab(request, tmp_path):
    """AC #1 / #5: the real ✕ click runs the close worker and the tab goes."""

    app = _ready_app()
    db, conversation_id, message_id = _saved_conversation(app, tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper = store.active_session_id
        saved = _restore_saved_tab(store, conversation_id, message_id)
        blank = store.create_session()
        store.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, saved.id, blank.id})

        await _click(pilot, f"#console-close-session-tab-{saved.id}")
        closed = await _settle(pilot, lambda: saved.id not in _session_ids(store))
        assert closed, "clicking ✕ on an idle saved tab left it open"
        assert _session_ids(store) == [keeper, blank.id]
        await _await_tabs(console, pilot, {keeper, blank.id})
        assert not isinstance(host.screen_stack[-1], ConfirmationDialog)
        assert store.active_session_id == keeper
        # Closing the tab never touches the saved history in Library.
        retained = db.get_conversation_by_id(conversation_id)
        assert retained is not None and not retained["deleted"]

        await _click(pilot, f"#console-close-session-tab-{blank.id}")
        closed = await _settle(pilot, lambda: blank.id not in _session_ids(store))
        assert closed, "clicking ✕ on a blank never-sent tab left it open"
        await _await_tabs(console, pilot, {keeper})
        assert not isinstance(host.screen_stack[-1], ConfirmationDialog)
    db.close_connection()


@pytest.mark.asyncio
@private_profile_test
async def test_at_risk_tab_dialog_stay_keeps_it_and_close_closes_it(request):
    """AC #2: Stay keeps the tab and its draft; Close really closes it."""

    app = _ready_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper = store.active_session_id
        drafted = store.create_session()
        store.set_session_draft(drafted.id, "unsent draft")
        store.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, drafted.id})

        await _click(pilot, f"#console-close-session-tab-{drafted.id}")
        first = await _wait_for_confirmation(host)
        assert "Unsent draft: yes" in first.message
        await _click(pilot, "#cancel-button")
        assert await _settle(
            pilot, lambda: not isinstance(host.screen_stack[-1], ConfirmationDialog)
        )
        await _await_tabs(console, pilot, {keeper, drafted.id})
        assert store.session_draft(drafted.id) == "unsent draft"

        await _click(pilot, f"#console-close-session-tab-{drafted.id}")
        second = await _wait_for_confirmation(host, previous=first)
        await _click(pilot, "#confirm-button")
        closed = await _settle(pilot, lambda: drafted.id not in _session_ids(store))
        assert closed, "confirming Close left the at-risk tab open"
        await _await_tabs(console, pilot, {keeper})
        assert store.active_session_id == keeper
        assert second is not host.screen_stack[-1]


@pytest.mark.asyncio
@private_profile_test
async def test_middle_click_closes_a_tab_without_switching_to_it(request):
    """AC #3: a middle-click closes the tab; it never activates it first.

    Closing the ACTIVE tab activates its right-hand neighbour, so a
    middle-click that switched to ``doomed`` before closing it would leave
    ``other`` active instead of ``keeper``.
    """

    app = _ready_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper = store.active_session_id
        doomed = store.create_session()
        other = store.create_session()
        store.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, doomed.id, other.id})
        assert _session_ids(store) == [keeper, doomed.id, other.id]

        await _click(pilot, f"#{_TAB_PREFIX}{doomed.id}", button=2)
        closed = await _settle(pilot, lambda: doomed.id not in _session_ids(store))
        assert closed, "middle-clicking a tab left it open"
        assert store.active_session_id == keeper
        await _await_tabs(console, pilot, {keeper, other.id})


@pytest.mark.asyncio
@private_profile_test
async def test_refused_close_names_the_tab_and_reason_and_logs_the_error_type(
    request, tmp_path
):
    """AC #4: a close the runtime refuses is shown and logged, never silent.

    The runtime refuses to close a session it has already fenced
    (``Console session is closed.``). After the refusal the tab stays, the
    user is told which tab and why, the log carries the exception type, and
    the next ✕ press is not swallowed by the in-flight guard.
    """

    app = _ready_app()
    db, conversation_id, message_id = _saved_conversation(app, tmp_path)
    notes = _record_notifications(app)
    records: list[str] = []
    sink = logger.add(records.append, level="WARNING", format="{message}")
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=_SIZE) as pilot:
            console = await _mounted_console(host, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            keeper = store.active_session_id
            saved = _restore_saved_tab(store, conversation_id, message_id)
            store.switch_session(keeper)
            await _show_tabs(console, pilot, {keeper, saved.id})
            runtime = console._console_runtime()
            runtime._admission_fenced_sessions.add(saved.id)

            await _click(pilot, f"#console-close-session-tab-{saved.id}")
            assert await _settle(pilot, lambda: bool(notes)), "refused close was silent"
            message, severity = notes[-1]
            assert f'"{_SAVED_TITLE}"' in message
            assert "Console session is closed" in message
            assert severity == "error"
            assert any(
                "Console session close failed" in record
                and "error_type=RuntimeError" in record
                for record in records
            ), records
            assert saved.id in _session_ids(store)
            await _await_tabs(console, pilot, {keeper, saved.id})

            runtime._admission_fenced_sessions.discard(saved.id)
            await _click(pilot, f"#console-close-session-tab-{saved.id}")
            closed = await _settle(pilot, lambda: saved.id not in _session_ids(store))
            assert closed, "the retry after a refused close was dropped"
            await _await_tabs(console, pilot, {keeper})
    finally:
        logger.remove(sink)
        db.close_connection()


class _UndrainedVoiceOwner:
    """A voice publication that does not drain within the close grace period.

    Stands in for ``VoicePromotionOwner`` only on the drain wait, the one
    real path where ``ConsoleRuntime.close_session`` returns without
    closing anything.
    """

    def __init__(self) -> None:
        self.aborted = 0

    def begin_session_close(self, session_id: str) -> object:
        return object()

    async def wait_for_session(self, session_id: str, timeout: float) -> bool:
        return False

    def complete_session_close(self, token: object) -> None:
        raise AssertionError("an undrained close must not complete")

    def abort_session_close(self, token: object) -> None:
        self.aborted += 1


@pytest.mark.asyncio
@private_profile_test
async def test_close_that_does_not_finish_is_reported_and_keeps_tab_state(
    request, tmp_path
):
    """AC #4: a runtime close that returns without closing is not success.

    The tab stays open, so its per-tab state (the composer undo history)
    must stay too, and the user is told the close did not finish.
    """

    app = _ready_app()
    db, conversation_id, message_id = _saved_conversation(app, tmp_path)
    notes = _record_notifications(app)
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=_SIZE) as pilot:
            console = await _mounted_console(host, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            keeper = store.active_session_id
            saved = _restore_saved_tab(store, conversation_id, message_id)
            store.switch_session(keeper)
            console._console_undo_histories[saved.id] = ([], [])
            await _show_tabs(console, pilot, {keeper, saved.id})
            runtime = console._console_runtime()
            previous_owner = runtime._voice_promotion_owner
            owner = _UndrainedVoiceOwner()
            runtime._voice_promotion_owner = owner
            try:
                await _click(pilot, f"#console-close-session-tab-{saved.id}")
                assert await _settle(pilot, lambda: bool(notes)), (
                    "unfinished close was silent"
                )
            finally:
                runtime._voice_promotion_owner = previous_owner
            message, severity = notes[-1]
            assert f'"{_SAVED_TITLE}"' in message
            assert "did not finish closing" in message
            assert severity == "error"
            assert owner.aborted == 1
            assert saved.id in _session_ids(store)
            await _await_tabs(console, pilot, {keeper, saved.id})
            assert saved.id in console._console_undo_histories
    finally:
        db.close_connection()


@pytest.mark.asyncio
@private_profile_test
async def test_close_flow_that_cannot_start_tells_the_user(request, monkeypatch):
    """AC #4: even a close worker that cannot be scheduled is not silent."""

    app = _ready_app()
    notes = _record_notifications(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper = store.active_session_id
        blank = store.create_session()
        store.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, blank.id})
        blank_title = next(s.title for s in store.sessions() if s.id == blank.id)

        def refuse_worker(*_args, **_kwargs):
            raise RuntimeError("worker manager is closed")

        with monkeypatch.context() as patch:
            patch.setattr(console, "run_worker", refuse_worker)
            await _click(pilot, f"#console-close-session-tab-{blank.id}")
            assert await _settle(pilot, lambda: bool(notes)), (
                "unstartable close was silent"
            )
        message, severity = notes[-1]
        assert f'"{blank_title}"' in message
        assert severity == "error"
        assert blank.id in _session_ids(store)
        await _await_tabs(console, pilot, {keeper, blank.id})

        await _click(pilot, f"#console-close-session-tab-{blank.id}")
        closed = await _settle(pilot, lambda: blank.id not in _session_ids(store))
        assert closed, "the retry after an unstartable close was dropped"
        await _await_tabs(console, pilot, {keeper})
