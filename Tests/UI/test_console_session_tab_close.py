"""Closing a Console session tab from the real tab strip (TASK-33621.15).

On dev @ 64579cce2c no close attempt ever closed a tab: the close worker
called ``self._console_runtime()`` on ``ConsoleSessionController``, which
only has the injected ``_console_runtime_accessor``, and the worker's
``exit_on_error=False`` swallowed the ``AttributeError`` with no toast and
no log line. The routing tests in ``test_console_button_routing.py`` pinned
the right outcome but call ``Button.press()`` directly, so every case here
goes through a real Pilot mouse click on a mounted ``ChatScreen`` and lets
the real close worker run against the real chat store and ``ConsoleRuntime``.

Each public group runs under ``@private_profile_test``; navigation and
recovery scenarios retain fresh apps and cleanup within their group.
``_build_test_app`` reloads config, which the per-test config sandbox refuses locally with
``RecoveryRequired: raw_source_selection_changed`` (see the Tests/UI
``RecoveryRequired`` lesson in ``backlog/docs/lessons-testing-evidence.md``).
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
import warnings
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path

import pytest
from loguru import logger
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI._child_creation_original_admission import (
    original_child_creation_admission,  # noqa: F401 - pytest fixture registration.
)
from Tests.UI._original_prepared_fleet_db_lifetime import (
    original_prepared_fleet_db_lifetime,  # noqa: F401 - pytest fixture registration.
)
from Tests.UI.app_factory import (
    _build_test_app,
    drain_active_service_patches,
    drain_created_dirs,
)
from Tests.UI.console_fixture_ownership import owned_console_apps  # noqa: F401
from Tests.UI.test_console_button_routing import (
    _mounted_console,
    _wait_for_confirmation,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleLifecycleImpact,
    ConsoleMessageRole,
    ConsoleRunMarker,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
    ConsoleDispatchReconstructability,
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionCloseImpact
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

#: Wide enough that every tab and its ✕ is mounted and hit-testable.
_SIZE = (160, 44)
_TAB_PREFIX = "console-session-tab-"
_SAVED_TITLE = "Saved notes"


class ProductionConsoleHarness(ConsoleHarness):
    """Use shipping Console startup sheets and inherited widget/modal defaults."""

    CSS_PATH = TldwCli.CSS_PATH

    async def _shutdown(self) -> None:
        from tldw_chatbook.UI.Console_Modules.view_workers import (
            capture_console_view_workers,
            drain_console_view_workers,
        )

        self._exit = True
        drain_error = None
        try:
            # The actual host owns current and detached nodes in these groups.
            captured = capture_console_view_workers(self)
            view = self.screen
            view._console_chat_tearing_down = True
            await drain_console_view_workers(captured)
        except BaseException as error:
            drain_error = error
        try:
            await super()._shutdown()
        except BaseException as error:
            if drain_error is not None:
                error.add_note(
                    "Captured Console view drain also failed before host shutdown"
                )
            raise
        if drain_error is not None:
            raise drain_error


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

    assert await _settle(
        pilot, laid_out
    ), f"tab strip never settled: {_open_tab_ids(console)} != {expected}"
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


def _saved_conversation(app, tmp_path, request):
    """Back one saved conversation with a real ChaChaNotes DB file."""

    db = CharactersRAGDB(tmp_path / "tab-close.db", client_id="tab-close")
    request.getfixturevalue("owned_console_apps")(app.console_runtime, db)
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


def _failure_toasts(notes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Error and warning toasts: a close that worked, or a Stay, shows none."""

    return [note for note in notes if note[1] in {"error", "warning"}]


async def _verify_clicking_x_closes_an_idle_saved_tab_and_a_blank_tab(
    request, tmp_path
):
    """AC #1 / #5: the real ✕ click runs the close worker and the tab goes."""

    app = _ready_app()
    db, conversation_id, message_id = _saved_conversation(app, tmp_path, request)
    notes = _record_notifications(app)
    host = ProductionConsoleHarness(app)
    try:
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
            assert _failure_toasts(notes) == []
    finally:
        db.close_connection()


async def _verify_at_risk_tab_dialog_stay_keeps_it_and_close_closes_it(request):
    """AC #2: Stay keeps the tab and its draft; Close really closes it."""

    app = _ready_app()
    notes = _record_notifications(app)
    host = ProductionConsoleHarness(app)
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
        assert _failure_toasts(notes) == [], "Stay is not a failed close"

        await _click(pilot, f"#console-close-session-tab-{drafted.id}")
        second = await _wait_for_confirmation(host, previous=first)
        await _click(pilot, "#confirm-button")
        closed = await _settle(pilot, lambda: drafted.id not in _session_ids(store))
        assert closed, "confirming Close left the at-risk tab open"
        await _await_tabs(console, pilot, {keeper})
        assert store.active_session_id == keeper
        assert second is not host.screen_stack[-1]
        assert _failure_toasts(notes) == []


async def _verify_middle_click_closes_a_tab_without_switching_to_it(request):
    """AC #3: a middle-click closes the tab; it never activates it first.

    Closing the ACTIVE tab activates its right-hand neighbour, so a
    middle-click that switched to ``doomed`` before closing it would leave
    ``other`` active instead of ``keeper``.
    """

    app = _ready_app()
    notes = _record_notifications(app)
    host = ProductionConsoleHarness(app)
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
        assert _failure_toasts(notes) == []

        # A plain click still activates: the tab prevents Textual's default
        # only after its own press has run.
        await _click(pilot, f"#{_TAB_PREFIX}{other.id}")
        assert await _settle(pilot, lambda: store.active_session_id == other.id)
        assert set(_session_ids(store)) == {keeper, other.id}


async def _verify_internal_close_error_names_the_tab_but_never_the_error_text(
    request, tmp_path, monkeypatch
):
    """AC #4: a close that fails inside the runtime is shown and logged.

    An unrelated internal failure may contain private bytes. The toast
    names only the error type; the log carries type and origin without the
    exception text (TASK-15103). The next ✕ press remains available.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Temporary directory for the private database fixture.
        monkeypatch: Fixture injecting an unrelated runtime Close failure.
    """

    app = _ready_app()
    db, conversation_id, message_id = _saved_conversation(app, tmp_path, request)
    notes = _record_notifications(app)
    records: list[str] = []
    sink = logger.add(records.append, level="WARNING", format="{message}")
    host = ProductionConsoleHarness(app)
    try:
        async with host.run_test(size=_SIZE) as pilot:
            console = await _mounted_console(host, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            keeper = store.active_session_id
            saved = _restore_saved_tab(store, conversation_id, message_id)
            store.switch_session(keeper)
            await _show_tabs(console, pilot, {keeper, saved.id})
            runtime = console._console_runtime()
            close_session = runtime.close_session
            private_error_text = "private-runtime-close-error-bytes"

            async def fail_close(_session_id, **_kwargs):
                raise RuntimeError(private_error_text)

            monkeypatch.setattr(runtime, "close_session", fail_close)
            await _click(pilot, f"#console-close-session-tab-{saved.id}")
            assert await _settle(pilot, lambda: bool(notes)), "refused close was silent"
            assert notes[-1] == (
                f'Couldn\'t close tab "{_SAVED_TITLE}": '
                "An unexpected error occurred (RuntimeError).",
                "error",
            )
            assert any(
                "Console session close failed (stage=close," in record
                and "error_type=RuntimeError" in record
                for record in records
            ), records
            assert not any(private_error_text in record for record in records)
            assert not any(private_error_text in text for text, _severity in notes)
            assert saved.id in _session_ids(store)
            await _await_tabs(console, pilot, {keeper, saved.id})

            monkeypatch.setattr(runtime, "close_session", close_session)
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


async def _verify_close_that_does_not_finish_is_reported_and_keeps_tab_state(
    request, tmp_path
):
    """AC #4: a runtime close that returns without closing is not success.

    The tab stays open, so its per-tab state (the composer undo history)
    must stay too, and the user is told the close did not finish.
    """

    app = _ready_app()
    db, conversation_id, message_id = _saved_conversation(app, tmp_path, request)
    notes = _record_notifications(app)
    host = ProductionConsoleHarness(app)
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
                assert await _settle(
                    pilot, lambda: bool(notes)
                ), "unfinished close was silent"
            finally:
                runtime._voice_promotion_owner = previous_owner
            message, severity = notes[-1]
            assert f'"{_SAVED_TITLE}"' in message
            assert "The close did not finish. Try again in a moment." in message
            assert severity == "error"
            assert owner.aborted == 1
            assert saved.id in _session_ids(store)
            await _await_tabs(console, pilot, {keeper, saved.id})
            assert saved.id in console._console_undo_histories
    finally:
        db.close_connection()


async def _verify_close_flow_that_cannot_start_tells_the_user(request, monkeypatch):
    """AC #4: even a close worker that cannot be scheduled is not silent."""

    app = _ready_app()
    notes = _record_notifications(app)
    host = ProductionConsoleHarness(app)
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
            assert await _settle(
                pilot, lambda: bool(notes)
            ), "unstartable close was silent"
        assert notes[-1] == (
            f'Couldn\'t close tab "{blank_title}": '
            "The close could not start. Try again in a moment.",
            "error",
        )
        assert blank.id in _session_ids(store)
        await _await_tabs(console, pilot, {keeper, blank.id})

        await _click(pilot, f"#console-close-session-tab-{blank.id}")
        closed = await _settle(pilot, lambda: blank.id not in _session_ids(store))
        assert closed, "the retry after an unstartable close was dropped"
        await _await_tabs(console, pilot, {keeper})


def _pending_temporary_turn(store) -> str:
    """A Temporary chat whose accepted turn has not finished (no SQL rows).

    The controller refuses to close it with copy the user can act on:
    "Finish or discard the pending turn before closing this chat."
    """

    session = store.create_session(title="Temporary", ephemeral=True)
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hello", persist=False
    )
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=False
    )
    store.register_ephemeral_dispatch_recovery(
        session.id,
        user_message_id=user.id,
        assistant_message_id=assistant.id,
        preparation_id="pending-preparation",
        attempt_id="attempt-1",
        checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
        origin="manual",
        queue_entry_id=None,
        frozen_authority=ConsoleTurnLibraryAuthority(
            policy=ConsoleLibraryPolicySnapshot(
                auto_retrieve=ConsoleAutoRetrieve.NEVER,
                assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                policy_revision=None,
                source="temporary",
            ),
            direct_library_tools=False,
            source_types=("notes", "media", "conversations"),
            scope_snapshot=ConsoleLibraryItemScopeSnapshot(
                note_ids=(), media_ids=(), conversations_allowed=True
            ),
            provider_intent=ConsoleProviderIntent(
                provider="llama_cpp",
                model="test-model",
                endpoint="http://127.0.0.1:9099",
            ),
            attempt_id="attempt-1",
        ),
        resolved_destination=ConsoleResolvedDestination(
            provider="llama_cpp",
            model="test-model",
            endpoint_identity="http://127.0.0.1:9099",
            egress_class=ConsoleEgressClass.ON_DEVICE,
        ),
        reconstructability=ConsoleDispatchReconstructability(
            attachments_reconstructable=True,
            evidence_reconstructable=True,
            prefill_reconstructable=True,
            opaque_reference="opaque:pending-turn",
        ),
    )
    return session.id


async def _verify_a_refusal_the_user_can_act_on_shows_its_own_reason(request):
    """An actionable pending-turn refusal leaves the tab accessible.

    Args:
        request: Pytest request selecting the isolated private-profile child.
    """

    app = _ready_app()
    notes = _record_notifications(app)
    host = ProductionConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper = store.active_session_id
        pending = _pending_temporary_turn(store)
        store.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, pending})

        await _click(pilot, f"#console-close-session-tab-{pending}")
        await _wait_for_confirmation(host)
        await _click(pilot, "#confirm-button")
        assert await _settle(pilot, lambda: bool(notes)), "refused close was silent"
        assert notes[-1] == (
            'Couldn\'t close tab "Temporary": '
            "Finish or discard the pending turn before closing this chat.",
            "error",
        )
        assert pending in _session_ids(store)
        assert await _settle(
            pilot, lambda: pending not in console._session._closing_session_requests
        ), "an actionable pending-turn refusal kept Close covering the tab"
        assert not isinstance(host.screen, ConfirmationDialog)
        assert store.dispatch_recovery_for_session(pending).recovery_needed
        assert store.active_session_id == keeper
        assert len(notes) == 1
        await _await_tabs(console, pilot, {keeper, pending})


async def _verify_failure_after_the_close_landed_says_so_and_leaves_no_dead_tab(
    request,
):
    """AC #4: a failure after the store closed the session is not a failed close.

    The close landed, so "Couldn't close" would be false. The failure here is
    the strip refresh itself, the teardown step a closed tab depends on to
    disappear: once, the report's own re-render heals the strip; twice, the
    closed tab's next ✕ re-renders it away instead of doing nothing.
    """

    app = _ready_app()
    notes = _record_notifications(app)
    host = ProductionConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper = store.active_session_id
        healed = store.create_session()
        stale = store.create_session()
        store.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, healed.id, stale.id})
        titles = {session.id: session.title for session in store.sessions()}

        controller = console._session
        real_sync = controller._sync_native_console_chat_ui_fn
        failures_left = [0]

        def failing_sync():
            if failures_left[0]:
                failures_left[0] -= 1
                raise ValueError("tab strip refresh failed")
            return real_sync()

        controller._sync_native_console_chat_ui_fn = failing_sync
        try:
            failures_left[0] = 1
            await _click(pilot, f"#console-close-session-tab-{healed.id}")
            assert await _settle(
                pilot, lambda: bool(notes)
            ), "teardown failure was silent"
            assert notes == [
                (
                    f'Closed tab "{titles[healed.id]}", but the Console did not '
                    "finish updating (ValueError).",
                    "warning",
                )
            ]
            assert healed.id not in _session_ids(store)
            await _await_tabs(console, pilot, {keeper, stale.id})

            failures_left[0] = 2
            await _click(pilot, f"#console-close-session-tab-{stale.id}")
            assert await _settle(pilot, lambda: len(notes) == 2)
            assert notes[-1][0].startswith(f'Closed tab "{titles[stale.id]}", but')
            assert stale.id not in _session_ids(store)
            await pilot.pause()
            assert stale.id in _open_tab_ids(console), "strip healed unexpectedly"

            await _click(pilot, f"#console-close-session-tab-{stale.id}")
            await _await_tabs(console, pilot, {keeper})
            assert len(notes) == 2, notes
            assert failures_left[0] == 0
        finally:
            controller._sync_native_console_chat_ui_fn = real_sync


@asynccontextmanager
async def _pending_close_app(request, kind, *, surviving_child=False):
    """Give prepared new-chat cases an exact runtime and database lifetime."""
    import shutil
    import sys
    from tempfile import mkdtemp
    from Tests.UI._prepared_close_owned_resources import PreparedCloseOwnedResources

    app = _ready_app()
    if kind != "chat_create":
        from Tests.UI._prepared_close_runtime_owner import PreparedCloseRuntimeOwner

        runtime_owner = PreparedCloseRuntimeOwner(app)
        try:
            yield app
        finally:
            primary = sys.exception()
            try:
                await runtime_owner.dispose_runtime()
            except BaseException as cleanup:
                if primary is None:
                    raise
                reasons = {
                    "prepared_close_runtime_wrong_thread",
                    "prepared_close_runtime_wrong_loop",
                    "prepared_close_runtime_owner_changed",
                    "prepared_close_runtime_not_retired",
                }
                reason = (
                    cleanup.args[0]
                    if type(cleanup) is RuntimeError
                    and len(cleanup.args) == 1
                    and type(cleanup.args[0]) is str  # noqa: E721 - reject custom static error argument types
                    and cleanup.args[0] in reasons
                    else None
                )
                primary.add_note(
                    "prepared_close_cleanup_error:"
                    + type(cleanup).__name__
                    + (":" + reason if reason is not None else "")
                )
        return
    temporary_root = request.getfixturevalue("tmp_path").resolve()
    directory = Path(mkdtemp(prefix="prepared-close-", dir=temporary_root)).absolute()
    assert directory.resolve().parent == temporary_root
    db = CharactersRAGDB(directory / "chats.sqlite", client_id="prepared-close")
    app.chachanotes_db = db
    app.local_chat_conversation_service = ChatConversationService(db)
    runs = None
    if surviving_child:
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        runs = AgentRunsDB(directory / "runs.sqlite")
        app._pending_close_runs = runs
    owner = PreparedCloseOwnedResources(app, directory, db, runs)
    app._pending_close_owned_resources = owner
    try:
        yield app
    finally:
        primary = sys.exception()
        try:
            try:
                # The host is gone, but the original owner loop is still alive.
                await owner.dispose_runtime()
            finally:
                if owner.runtime_terminal:
                    owner.close_creators()
                    assert directory.resolve().parent == temporary_root
                    shutil.rmtree(directory)
        except BaseException as cleanup:
            if primary is None:
                raise
            # Only fixed helper refusal codes may accompany the primary failure.
            reasons = {
                "prepared_close_disposal_wrong_thread",
                "prepared_close_runtime_runs_not_retained",
                "prepared_close_finalization_wrong_thread",
                "prepared_close_runtime_not_disposed",
                "prepared_close_database_owner_changed",
                "prepared_close_database_not_retired",
                "prepared_close_directory_has_live_storage",
            }
            reason = (
                cleanup.args[0]
                if type(cleanup) is RuntimeError
                and len(cleanup.args) == 1
                and type(cleanup.args[0]) is str  # noqa: E721 - reject custom static error argument types
                and cleanup.args[0] in reasons
                else None
            )
            primary.add_note(
                "prepared_close_cleanup_error:"
                + type(cleanup).__name__
                + (":" + reason if reason is not None else "")
            )


def _prepare_surviving_child(controller, session_id):
    """Keep existing child authority live after the primary turn has ended."""
    from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    source = controller.store._sessions[session_id]
    source.persisted_conversation_id = controller.store.persistence.create_conversation(
        conversation_title=source.title
    )
    runs = controller.app._pending_close_runs
    parent = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="primary",
        assistant_message_id="finished-parent-message",
    )
    child = runs.create_run(
        conversation_id=source.persisted_conversation_id,
        agent_kind="subagent",
        parent_run_id=parent,
        run_id=f"close-chat_create-{session_id}",
    )
    controller._agent_bridge = ConsoleAgentBridge(
        agent_runs_db=runs,
        store=controller.store,
        provider_gateway=controller.provider_gateway,
    )
    controller._hooks_v2_runtime.set_agent_bridge(controller._agent_bridge)
    actor = CurrentRunActor("subagent", child, parent)
    with use_run_actor(actor):
        prepared = controller.prepare_agent_chat_create(
            {
                "tool": "new_chat",
                "session_id": session_id,
                "source_run_id": child,
                "source_message_id": "finished-parent-message",
                "title": "private close chat",
                "destination": "same_workspace",
                "mode": "draft",
            }
        )
    return prepared, actor


def _assert_surviving_creation_is_live(controller, pending):
    """Require prepared child authority to survive a normal Console refresh."""
    from tldw_chatbook.Agents.run_context import use_run_actor

    prepared = pending._prepared_creation_payload
    with use_run_actor(pending._prepared_creation_actor):
        controller._hooks_v2_runtime.view._sync_console_chat_core_state()
        assert controller._chat_creation_source_live(prepared)
        record = controller._chat_creation_record(prepared)
        assert record is controller._chat_creation_records[prepared["_creation_token"]]
        assert record["payload"] == prepared
        assert not record["approved"]


async def _arm_pending_round(
    controller, kind: str, session_id: str, *, surviving_child=False
):
    """Arm a real blocking round without executing the proposed tool."""

    actor = None
    if kind == "chat_create":
        if surviving_child:
            prepared, actor = _prepare_surviving_child(controller, session_id)
            creation_run = actor.run_id
        else:
            from Tests.Chat.test_console_chat_create_integration import (
                _prepare_close_new_chat,
            )

            prepared, creation_run = _prepare_close_new_chat(
                controller, session_id, "private close chat"
            )
        owner = getattr(controller.app, "_pending_close_owned_resources", None)
        if owner is not None:
            assert owner.app is controller.app
            owner.adopt_runtime_runs()
        prepared_runs = getattr(controller._agent_bridge, "runs_db", None)

    def request():
        if kind == "approval":
            return controller.request_mcp_approvals(
                [
                    MCPPendingCall(
                        llm_name="close-call",
                        server_key="local:fixture",
                        tool_name="search",
                        server_label="Close fixture",
                        arguments={"query": "private close query"},
                        reason="ask",
                    )
                ],
                session_id=session_id,
            )
        if kind == "question":
            return controller.request_user_questions(
                [
                    {
                        "header": "Choice",
                        "question": "Private choice?",
                        "options": [
                            {"label": "One", "description": "First choice"},
                            {"label": "Two", "description": "Second choice"},
                        ],
                    }
                ],
                session_id=session_id,
            )
        if kind == "chat_create":
            scope = (
                owner.request_connection_scope(prepared_runs)
                if owner is not None
                else nullcontext()
            )
            with scope:
                return controller.request_chat_create_confirm(
                    prepared,
                    session_id=session_id,
                )
        if kind == "worktree_merge":
            return controller.request_worktree_merge_confirm(
                {"run_id": "private-child", "action": "merge"}, session_id=session_id
            )
        if kind == "skill_install":
            return controller.request_skill_install_confirm(
                "https://example.com/private-skill", session_id=session_id
            )
        return controller.request_skill_script_confirm(
            {
                "skill_name": "Close fixture",
                "script_path": "scripts/example.py",
                "mechanism": "python",
                "args": [],
            },
            session_id=session_id,
        )

    from tldw_chatbook.Agents.run_context import use_run_actor

    context = (
        use_run_actor(actor)
        if actor is not None
        else use_run_id(
            creation_run if kind == "chat_create" else f"close-{kind}-{session_id}"
        )
    )
    with context:
        pending = asyncio.create_task(asyncio.to_thread(request))
    if kind == "chat_create":
        pending._prepared_creation_payload = prepared
        pending._prepared_creation_actor = actor
    return pending


async def _verify_background_pending_close_names_consequences_and_cancels_only_its_owner(
    request,
):
    """TASK-33621.16: real background rounds, run cancellation and physical Close.

    Args:
        request: Pytest request selecting the isolated private-profile child.
    """
    for kind, consequence, result in [
        ("approval", "Tool approvals: denied; runs cancelled.", {"close-call": "deny"}),
        (
            "chat_create",
            "Chat creation: declined; no chat created.",
            {"allow": False, "remember": False},
        ),
        (
            "worktree_merge",
            "Worktree decisions: cancelled; no merge or discard.",
            {"allow": False},
        ),
        (
            "question",
            "Questions: cancelled without an answer.",
            {"answered": False, "reason": "cancelled"},
        ),
        (
            "skill_install",
            "Skill installs: declined; runs cancelled.",
            False,
        ),
        (
            "skill_script",
            "Skill scripts: declined; runs cancelled.",
            {"allow": False, "remember": False},
        ),
    ]:
        async with _pending_close_app(request, kind) as app:
            host = ProductionConsoleHarness(app)
            async with host.run_test(size=_SIZE) as pilot:
                console = await _mounted_console(
                    host, pilot, "#console-native-composer"
                )
                controller = console._ensure_console_chat_controller()
                app.call_from_thread = host.call_from_thread
                store = controller.store
                keeper = store.active_session_id
                assert _session_ids(store) == [keeper]
                assert not controller.pending_round_kinds(keeper)
                assert not controller._active_stream_tasks
                doomed = controller.new_session(title="Pending [notes]")
                controller.switch_session(keeper)
                assistant = store.append_message(
                    doomed.id, role=ConsoleMessageRole.ASSISTANT, content=""
                )
                cancelled = asyncio.Event()
                controller._active_cancel_events[doomed.id] = threading.Event()
                if kind == "chat_create":
                    controller._active_assistant_message_ids[doomed.id] = assistant.id
                round_task = await _arm_pending_round(controller, kind, doomed.id)

                async def waiting_run(
                    controller=controller,
                    doomed=doomed,
                    assistant=assistant,
                    round_task=round_task,
                    cancelled=cancelled,
                ):
                    task = asyncio.current_task()
                    controller._active_stream_tasks[doomed.id] = task
                    controller._active_assistant_message_ids[doomed.id] = assistant.id
                    controller._set_run_state(
                        ConsoleRunState(ConsoleRunStatus.STREAMING, "Waiting"),
                        session_id=doomed.id,
                    )
                    try:
                        await asyncio.shield(round_task)
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                    finally:
                        controller._active_stream_tasks.pop(doomed.id, None)
                        controller._active_assistant_message_ids.pop(doomed.id, None)
                        controller._active_cancel_events.pop(doomed.id, None)

                run_task = asyncio.create_task(waiting_run())
                try:
                    armed = await _settle(
                        pilot,
                        lambda kind=kind, controller=controller, doomed=doomed: (
                            kind in controller.pending_round_kinds(doomed.id)
                            and doomed.id in controller._active_stream_tasks
                        ),
                    )
                    if not armed:
                        import faulthandler

                        print(
                            "pending round timeout:",
                            kind,
                            "request done:",
                            round_task.done(),
                            "request cancelled:",
                            round_task.cancelled(),
                            "run done:",
                            run_task.done(),
                            flush=True,
                        )
                        if round_task.done() and not round_task.cancelled():
                            # Surface a real request exception before cleanup
                            # can replace it with a secondary ownership error.
                            print(
                                "pending round result:", round_task.result(), flush=True
                            )
                        else:
                            faulthandler.dump_traceback(file=2)
                    assert (
                        armed
                    ), "the actual pending round and its owning run did not arm"
                    await _show_tabs(console, pilot, {keeper, doomed.id})
                    assert (
                        controller.run_marker_for(doomed.id)
                        is ConsoleRunMarker.NEEDS_APPROVAL
                    )
                    await _click(pilot, f"#console-close-session-tab-{doomed.id}")
                    dialog = await _wait_for_confirmation(host)
                    assert "Pending [notes]" in dialog.title
                    assert consequence in dialog.message
                    assert "private close" not in dialog.message
                    for zero_row in (
                        "Unsent draft:",
                        "Pending attachments:",
                        "Delegated agents:",
                        "Unsent queued prompts:",
                    ):
                        assert zero_row not in dialog.message
                    assert await _settle(
                        pilot,
                        lambda dialog=dialog: (
                            dialog.query_one("#cancel-button").has_focus
                        ),
                    )
                    await _click(pilot, "#confirm-button")
                    assert await _settle(
                        pilot,
                        lambda doomed=doomed, store=store: (
                            doomed.id not in _session_ids(store)
                        ),
                    )
                    await _await_tabs(console, pilot, {keeper})
                    assert await _settle(
                        pilot,
                        lambda cancelled=cancelled, round_task=round_task: (
                            cancelled.is_set() and round_task.done()
                        ),
                    ), {
                        "pending_kind": kind,
                        "run_cancel_seen": cancelled.is_set(),
                        "round_done": round_task.done(),
                        "run_done": run_task.done(),
                        "run_cancelled": run_task.cancelled(),
                        "run_cancel_requests": run_task.cancelling(),
                        "registered_run_is_exact_owner": (
                            controller._active_stream_tasks.get(doomed.id) is run_task
                        ),
                    }
                    assert await round_task == result
                    assert run_task.cancelled()
                    assert not controller.has_pending_approval_round(doomed.id)
                    if kind == "chat_create":
                        assert not controller.pending_chat_create_ids()
                        assert not controller._parked_chat_create_payloads
                    else:
                        assert (
                            controller._interrupt_host.session_round_payloads(
                                kind, doomed.id
                            )
                            == []
                        )
                        assert not controller._interrupt_host.registries[kind]
                    assert store.active_session_id == keeper
                    assert not console._console_runtime().console_needs_attention
                finally:
                    # A copy assertion can fail before Close; release the real merge
                    # worker's cancellation signal before dropping its owning task.
                    cancel_event = controller._active_cancel_events.get(doomed.id)
                    if cancel_event is not None:
                        cancel_event.set()
                    controller.revoke_approval_rounds_for_run(
                        f"close-{kind}-{doomed.id}"
                    )
                    controller._cancel_pending_decisions_for_session(doomed.id)
                    run_task.cancel()
                    await asyncio.gather(run_task, return_exceptions=True)
                    await asyncio.wait_for(asyncio.shield(round_task), 5)


async def _verify_background_pending_close_releases_round_without_an_active_turn(
    request: pytest.FixtureRequest,
) -> None:
    """Release both real standalone decisions without an owning cancel signal.

    Args:
        request: Pytest request selecting the isolated private-profile child.
    """
    for kind, expected in (
        ("question", {"answered": False, "reason": "cancelled"}),
        ("chat_create", {"allow": False, "remember": False}),
    ):
        async with _pending_close_app(
            request, kind, surviving_child=kind == "chat_create"
        ) as app:
            host = ProductionConsoleHarness(app)
            async with host.run_test(size=_SIZE) as pilot:
                console = await _mounted_console(
                    host, pilot, "#console-native-composer"
                )
                controller = console._ensure_console_chat_controller()
                app.call_from_thread = host.call_from_thread
                store = controller.store
                keeper = store.active_session_id
                doomed = controller.new_session(title="Pending decision")
                controller.switch_session(keeper)
                sibling = await _arm_pending_round(controller, "question", keeper)
                pending = await _arm_pending_round(
                    controller, kind, doomed.id, surviving_child=kind == "chat_create"
                )
                try:
                    read_ids = (
                        controller.pending_question_ids
                        if kind == "question"
                        else controller.pending_chat_create_ids
                    )
                    assert await _settle(
                        pilot,
                        lambda read_ids=read_ids, kind=kind: (
                            len(read_ids()) == (2 if kind == "question" else 1)
                        ),
                    ), "both real decision rounds must be armed before closing"
                    if kind == "chat_create":
                        assert await _settle(
                            pilot, lambda: bool(controller._parked_chat_create_payloads)
                        )
                        request_id = controller.pending_chat_create_ids()[0]
                        card = controller._parked_chat_create_payloads[request_id]
                        assert card["session_id"] == doomed.id
                        assert card["request_id"] == request_id
                        assert (
                            card["_creation_token"]
                            is pending._prepared_creation_payload["_creation_token"]
                        )
                        assert not controller._pending_chat_create_rounds[request_id][
                            "event"
                        ].is_set()
                        assert controller.pending_round_kinds(doomed.id) == {
                            "chat_create"
                        }
                        _assert_surviving_creation_is_live(controller, pending)
                    assert doomed.id not in controller._active_cancel_events
                    assert doomed.id not in controller._active_assistant_message_ids
                    await _show_tabs(console, pilot, {keeper, doomed.id})
                    await _click(pilot, f"#console-close-session-tab-{doomed.id}")
                    await _wait_for_confirmation(host)
                    await _click(pilot, "#confirm-button")
                    assert await _settle(
                        pilot,
                        lambda doomed=doomed, store=store: (
                            doomed.id not in _session_ids(store)
                        ),
                    )
                    await _await_tabs(console, pilot, {keeper})
                    assert await _settle(
                        pilot, pending.done, timeout=2
                    ), "closed session left its decision armed without an owning turn"
                    assert await pending == expected
                    assert (
                        not sibling.done()
                    ), "closing the background tab answered the viewed tab"
                    assert len(controller.pending_question_ids()) == 1
                    assert controller.pending_round_kinds(keeper) == {"question"}
                    assert not controller.has_pending_approval_round(doomed.id)
                finally:
                    controller.revoke_approval_rounds_for_run(
                        f"close-question-{keeper}"
                    )
                    controller.revoke_approval_rounds_for_run(
                        f"close-{kind}-{doomed.id}"
                    )
                    await asyncio.wait_for(asyncio.gather(sibling, pending), 5)


async def _verify_chat_create_enrichment_cannot_arm_after_its_session_closes(
    request, monkeypatch
):
    """A worker returning from fork enrichment must observe the committed Close.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        monkeypatch: Pause the real bridge at its payload-enrichment boundary.
    """
    async with _pending_close_app(request, "chat_create", surviving_child=True) as app:
        host = ProductionConsoleHarness(app)
        async with host.run_test(size=_SIZE) as pilot:
            console = await _mounted_console(host, pilot, "#console-native-composer")
            controller = console._ensure_console_chat_controller()
            app.call_from_thread = host.call_from_thread
            keeper = controller.store.active_session_id
            doomed = controller.new_session(title="Closing before confirmation")
            controller.switch_session(keeper)
            entered = threading.Event()
            release = threading.Event()
            enrich = controller._enrich_chat_create_confirm_payload

            def paused_enrichment(payload):
                entered.set()
                assert release.wait(10), "Close never released the enrichment boundary"
                return enrich(payload)

            monkeypatch.setattr(
                controller, "_enrich_chat_create_confirm_payload", paused_enrichment
            )
            pending = await _arm_pending_round(
                controller, "chat_create", doomed.id, surviving_child=True
            )
            try:
                assert await _settle(pilot, entered.is_set)
                _assert_surviving_creation_is_live(controller, pending)
                assert not controller.pending_round_kinds(doomed.id)
                assert doomed.id not in controller._active_assistant_message_ids
                assert doomed.id not in controller._active_cancel_events
                await _show_tabs(console, pilot, {keeper, doomed.id})
                await _click(pilot, f"#console-close-session-tab-{doomed.id}")
                assert await _settle(
                    pilot, lambda: doomed.id not in _session_ids(controller.store)
                )
                assert doomed.id in controller._session_close_generations
                release.set()
                assert await _settle(
                    pilot, pending.done, timeout=2
                ), "chat-create confirmation armed after the committed Close sweep"
                assert await pending == {"allow": False, "remember": False}
                assert not controller.pending_chat_create_ids()
                assert not controller._parked_chat_create_payloads
                assert not controller.pending_round_kinds(doomed.id)
                assert doomed.id not in controller._chat_create_session_grants
                assert controller.store.active_session_id == keeper
            finally:
                release.set()
                controller.revoke_approval_rounds_for_run(
                    f"close-chat_create-{doomed.id}"
                )
                await asyncio.wait_for(pending, 5)


async def _verify_failed_confirmed_close_reoffers_confirmation_without_retrying(
    request,
):
    """A refused at-risk close keeps work and offers a fresh, explicit retry.

    Args:
        request: Pytest request selecting the isolated private-profile child.
    """
    app = _ready_app()
    notes = _record_notifications(app)
    host = ProductionConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper = store.active_session_id
        doomed = store.create_session(title="Retry notes")
        store.set_session_draft(doomed.id, "private draft")
        store.switch_session(keeper)
        await _show_tabs(console, pilot, {keeper, doomed.id})
        runtime = console._console_runtime()
        previous_owner = runtime._voice_promotion_owner
        owner = _UndrainedVoiceOwner()
        runtime._voice_promotion_owner = owner
        try:
            await _click(pilot, f"#console-close-session-tab-{doomed.id}")
            first = await _wait_for_confirmation(host)
            await _click(pilot, "#confirm-button")
            second = await _wait_for_confirmation(host, previous=first)
            assert notes[-1] == (
                'Couldn\'t close tab "Retry notes": The close did not finish. Try again in a moment.',
                "error",
            )
            assert owner.aborted == 1, "failure must not retry the close automatically"
            assert store.session_draft(doomed.id) == "private draft"
            assert "Retry notes" in second.title
            assert await _settle(
                pilot, lambda: second.query_one("#cancel-button").has_focus
            )
            runtime._voice_promotion_owner = previous_owner
            await _click(pilot, "#confirm-button")
            assert await _settle(pilot, lambda: doomed.id not in _session_ids(store))
            await _await_tabs(console, pilot, {keeper})
        finally:
            runtime._voice_promotion_owner = previous_owner


async def _verify_all_close_consequences_keep_named_title_and_actions_painted_at_80x24(
    request,
):
    """Keep named Close controls reachable while long consequences scroll.

    Args:
        request: Pytest request selecting the isolated private-profile child.
    """
    for title in ("Pending [notes]", "A" * 60):
        app = _ready_app()
        host = ProductionConsoleHarness(app)
        async with host.run_test(size=(80, 24)) as pilot:
            console = await _mounted_console(host, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            assert _session_ids(store) == [store.active_session_id]
            session = store.create_session(title=title)
            impact = ConsoleSessionCloseImpact(
                session_id=session.id,
                transcript_message_count=1,
                lifecycle=ConsoleLifecycleImpact(
                    revision=1,
                    live_run_count=1,
                    queued_session_count=1,
                    unsent_prompt_count=1,
                    delegated_child_count=1,
                ),
                has_draft=True,
                pending_attachment_count=1,
                pending_round_kinds=frozenset(
                    {
                        "approval",
                        "question",
                        "skill_install",
                        "skill_script",
                        "worktree_merge",
                        "chat_create",
                    }
                ),
            )
            worker = console.run_worker(
                console._session._confirm_session_close(impact), exit_on_error=False
            )
            dialog = await _wait_for_confirmation(host)
            assert await _settle(
                pilot,
                lambda dialog=dialog: dialog.query_one("#cancel-button").has_focus,
            )
            try:
                container = dialog.query_one("#confirmation-dialog", VerticalScroll)
                assert container.region.width == 60
                controls = {
                    selector: dialog.query_one(selector)
                    for selector in ("#cancel-button", "#confirm-button")
                }
                viewport = dialog.region
                assert viewport.intersection(container.region) == container.region, (
                    container.region,
                    viewport,
                )
                for selector, text in (
                    (".dialog-title", "Close tab"),
                    ("#cancel-button", "Stay"),
                    ("#confirm-button", "Close"),
                ):
                    control = dialog.query_one(selector)
                    region, clip = dialog._compositor.visible_widgets[control]
                    assert region.area and region.intersection(clip) == region
                    assert region.intersection(viewport) == region
                    painted = "\n".join(
                        strip.crop(region.x, region.right).text
                        for strip in dialog._compositor.render_strips()[
                            region.y : region.bottom
                        ]
                    )
                    assert text in painted, (selector, painted)
                assert title in dialog.title
                assert (
                    "Worktree decisions: cancelled; no merge or discard."
                    in dialog.message
                )
                assert "Chat creation: declined; no chat created." in dialog.message
                await pilot.resize_terminal(80, 18)
                await pilot.pause()
                assert dialog.region.intersection(container.region) == container.region
                for selector in ("#cancel-button", "#confirm-button"):
                    control = dialog.query_one(selector)
                    region, clip = dialog._compositor.visible_widgets[control]
                    assert region.area and region.intersection(clip) == region
                    assert region.intersection(dialog.region) == region
                    under, _ = host.get_widget_at(*region.center)
                    assert under is control
                assert controls["#cancel-button"].has_focus
                await pilot.press("shift+tab")
                assert container.has_focus
                await pilot.press("end")
                assert await _settle(
                    pilot, lambda container=container: container.scroll_y > 0
                )
                await pilot.wait_for_scheduled_animations()
                painted = "\n".join(
                    strip.text for strip in dialog._compositor.render_strips()
                )
                assert "no merge or discard." in painted
                assert "Stay" in painted and "Close" in painted
                await pilot.press("home", "tab")
                assert controls["#cancel-button"].has_focus
                await pilot.resize_terminal(160, 44)
                await pilot.pause()
                assert container.region.width == 60
                for selector, control in controls.items():
                    assert dialog.query_one(selector) is control
                    region, clip = dialog._compositor.visible_widgets[control]
                    assert region.area and region.intersection(clip) == region
                    assert region.intersection(dialog.region) == region
            finally:
                dialog.dismiss(False)
                assert await worker.wait() is False


async def _verify_progress_close_failure_reconciles_fleet_before_confirmed_retry(
    request,
    tmp_path,
    monkeypatch,
):
    """A failed provisional callback must not strand or replace an exact fence.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Temporary directory for the private database fixture.
        monkeypatch: Pytest patch fixture injecting the provisional cleanup failure.
    """
    from Tests.Chat.test_fleet_usage_reattach import _resolution, _turn_signals
    from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
    from tldw_chatbook.Chat.console_agent_bridge import FleetDrained, SettledChild

    for rollback_refused in (False, True):
        case_path = tmp_path / str(rollback_refused)
        case_path.mkdir()
        with monkeypatch.context() as patch:
            app = _ready_app()
            _attach_real_dbs(app, case_path)
            request.getfixturevalue("owned_console_apps")(
                app.console_runtime, app.chachanotes_db
            )
            notes = _record_notifications(app)
            host = ProductionConsoleHarness(app)
            async with host.run_test(size=_SIZE) as pilot:
                console = await _mounted_console(
                    host, pilot, "#console-native-composer"
                )
                controller = console._ensure_console_chat_controller()
                store = controller.store
                keeper = store.active_session_id
                assert _session_ids(store) == [keeper]
                assert not controller._session_close_states
                assert not controller._session_close_generations
                assert not controller._failed_session_close_generations
                doomed = controller.new_session(title="Progress retry")
                assistant = store.append_message(
                    doomed.id,
                    role=ConsoleMessageRole.ASSISTANT,
                    content="Completed parent",
                )
                signals = _turn_signals(prompt=2, completion=1)
                resolution = _resolution()
                controller._attach_stream_usage(
                    assistant.id, signals, resolution, partial=False
                )
                controller._fleet_usage_reattach_sources[assistant.id] = (
                    signals,
                    resolution,
                    False,
                )
                signals.record_usage_payload(
                    {"prompt_tokens": 2, "completion_tokens": 1}
                )
                signals.close_usage_call()
                doomed.persisted_conversation_id = "saved-progress-retry"
                conversation_id = controller.conversation_id_for_session(doomed.id)
                store.set_session_draft(doomed.id, "private draft")
                controller.switch_session(keeper)
                await _show_tabs(console, pilot, {keeper, doomed.id})
                bridge = controller._agent_bridge
                assert bridge is not None
                close_progress = bridge.begin_close_progress
                abort_fence = bridge.abort_fleet_fence
                calls = []

                def fail_once(
                    session_id,
                    *,
                    conversation_id,
                    calls=calls,
                    close_progress=close_progress,
                ):
                    calls.append(session_id)
                    if len(calls) == 1:
                        raise RuntimeError("progress cleanup unavailable")
                    return close_progress(session_id, conversation_id=conversation_id)

                patch.setattr(bridge, "begin_close_progress", fail_once)
                if rollback_refused:
                    patch.setattr(
                        bridge, "abort_fleet_fence", lambda *_args, **_kwargs: False
                    )
                try:
                    await _click(pilot, f"#console-close-session-tab-{doomed.id}")
                    first = await _wait_for_confirmation(host)
                    await _click(pilot, "#confirm-button")
                    assert await _settle(pilot, lambda notes=notes: bool(notes))
                    generation = controller._session_close_generation
                    assert calls == [doomed.id], "failure must not retry automatically"
                    assert notes[-1][1] == "error"
                    assert doomed.id in _session_ids(store)
                    assert store.session_draft(doomed.id) == "private draft"
                    assert not controller._session_close_states
                    assert doomed.id not in controller._session_close_generations
                    assert not controller._fleet_wake._conversation_fences
                    if rollback_refused:
                        # A provisional failure never cancelled the fleet. Its later
                        # deterministic drain must still fold usage into the open tab.
                        bridge._fleet_drain_fanout.fire(
                            FleetDrained(
                                conversation_id=conversation_id,
                                children=(
                                    SettledChild(
                                        run_id="surviving-child",
                                        status="done",
                                        session_id=doomed.id,
                                        assistant_message_id=assistant.id,
                                    ),
                                ),
                            )
                        )
                        assert await _settle(
                            pilot,
                            lambda store=store, assistant=assistant: (
                                store.get_message(assistant.id).usage.total_tokens == 6
                            ),
                        ), "provisional close failure dropped surviving-child usage"
                        assert bridge._fleet_fence_generations == {
                            doomed.id: generation,
                            conversation_id: generation,
                        }
                        assert controller._failed_session_close_generations == {
                            doomed.id: generation
                        }
                        assert await _settle(
                            pilot,
                            lambda session=console._session, doomed=doomed: (
                                doomed.id not in session._closing_session_requests
                            ),
                        ), "unrecoverable close kept a replacement confirmation open"
                        assert not isinstance(host.screen, ConfirmationDialog)
                        assert "Progress retry" in notes[-1][0]
                        assert "restart" in notes[-1][0].casefold()
                        note_count = len(notes)
                        await _click(pilot, f"#console-close-session-tab-{doomed.id}")
                        retry = await _wait_for_confirmation(host, previous=first)
                        assert await _settle(
                            pilot,
                            lambda retry=retry: (
                                retry.query_one("#cancel-button").has_focus
                            ),
                        )
                        await _click(pilot, "#confirm-button")
                        assert await _settle(
                            pilot,
                            lambda notes=notes, note_count=note_count: (
                                len(notes) == note_count + 1
                            ),
                        )
                        assert await _settle(
                            pilot,
                            lambda session=console._session, doomed=doomed: (
                                doomed.id not in session._closing_session_requests
                            ),
                        )
                        assert not isinstance(host.screen, ConfirmationDialog)
                        assert "Progress retry" in notes[-1][0]
                        assert "restart" in notes[-1][0].casefold()
                        assert controller._session_close_generation == generation
                        assert calls == [doomed.id]
                        assert bridge._fleet_fence_generations == {
                            doomed.id: generation,
                            conversation_id: generation,
                        }
                        assert controller._failed_session_close_generations == {
                            doomed.id: generation
                        }
                        assert doomed.id in _session_ids(store)
                        assert store.session_draft(doomed.id) == "private draft"
                        assert store.get_message(assistant.id).usage.total_tokens == 6
                    else:
                        second = await _wait_for_confirmation(host, previous=first)
                        assert await _settle(
                            pilot,
                            lambda second=second: (
                                second.query_one("#cancel-button").has_focus
                            ),
                        )
                        assert not bridge._fleet_fence_generations
                        assert not controller._failed_session_close_generations
                        assert doomed.id not in controller._session_close_generations
                        await _click(pilot, "#confirm-button")
                        assert await _settle(
                            pilot,
                            lambda doomed=doomed, store=store: (
                                doomed.id not in _session_ids(store)
                            ),
                        )
                        await _await_tabs(console, pilot, {keeper})
                        assert calls == [doomed.id, doomed.id]
                        generation = controller._session_close_generations[doomed.id]
                        assert bridge._fleet_fence_generations == {
                            doomed.id: generation
                        }
                        assert controller._fleet_wake._conversation_fences == {
                            doomed.id: generation
                        }
                        assert not controller._session_close_states

                        # Low-level recreation can reuse the native ID, but it
                        # cannot retire the old close/late-usage authority.
                        generation = controller._session_close_generations[doomed.id]
                        reopened = store.create_session(
                            session_id=doomed.id,
                            title="Reopened Progress retry",
                            activate=False,
                        )
                        reopened.persisted_conversation_id = conversation_id
                        restored = store.append_message(
                            reopened.id,
                            role=ConsoleMessageRole.ASSISTANT,
                            content="Restored parent",
                            message_id=assistant.id,
                        )
                        controller._attach_stream_usage(
                            restored.id,
                            _turn_signals(prompt=2, completion=1),
                            resolution,
                            partial=False,
                        )
                        stale_drain = FleetDrained(
                            conversation_id=conversation_id,
                            children=(
                                SettledChild(
                                    run_id="old-incarnation-child",
                                    status="done",
                                    session_id=doomed.id,
                                    assistant_message_id=assistant.id,
                                ),
                            ),
                        )
                        assert controller._fleet_event_is_stale(stale_drain)
                        bridge._fleet_drain_fanout.fire(stale_drain)
                        # Pin the second guard too: a drain may have already
                        # been queued on the app loop before close committed.
                        controller._reattach_fleet_usage_guarded(stale_drain)
                        await pilot.pause()
                        assert store.get_message(restored.id).usage.total_tokens == 3
                        assert (
                            controller._fleet_usage_reattach_sources[assistant.id][0]
                            is signals
                        )
                        store.set_session_draft(reopened.id, "restored private draft")
                        await _show_tabs(console, pilot, {keeper, reopened.id})
                        note_count = len(notes)
                        await _click(pilot, f"#console-close-session-tab-{reopened.id}")
                        refusal_dialog = await _wait_for_confirmation(
                            host, previous=second
                        )
                        assert await _settle(
                            pilot,
                            lambda refusal_dialog=refusal_dialog: (
                                refusal_dialog.query_one("#cancel-button").has_focus
                            ),
                        )
                        await _click(pilot, "#confirm-button")
                        assert await _settle(
                            pilot,
                            lambda notes=notes, note_count=note_count: (
                                len(notes) == note_count + 1
                            ),
                        )
                        assert await _settle(
                            pilot,
                            lambda session=console._session, reopened=reopened: (
                                reopened.id not in session._closing_session_requests
                            ),
                        ), "reused ID close kept a replacement confirmation open"
                        assert not isinstance(host.screen, ConfirmationDialog)
                        assert "Reopened Progress retry" in notes[-1][0]
                        assert "restart" in notes[-1][0].casefold()
                        assert notes[-1][1] == "error"
                        assert controller._session_close_generations == {
                            doomed.id: generation
                        }
                        assert controller._session_close_generation == generation
                        assert not controller._session_close_states
                        assert reopened.id in (
                            console._console_runtime()._admission_fenced_sessions
                        )
                        assert bridge._fleet_fence_generations == {
                            doomed.id: generation
                        }
                        assert controller._fleet_wake._conversation_fences == {
                            doomed.id: generation
                        }
                        assert calls == [doomed.id, doomed.id]
                        assert reopened.id in _session_ids(store)
                        assert (
                            store.session_draft(reopened.id) == "restored private draft"
                        )
                        assert store.get_message(restored.id).usage.total_tokens == 3
                finally:
                    if isinstance(host.screen, ConfirmationDialog):
                        host.screen.dismiss(False)
                    patch.setattr(bridge, "abort_fleet_fence", abort_fence)
                    for fenced_id in (doomed.id, conversation_id):
                        generation = bridge._fleet_fence_generations.get(fenced_id)
                        if generation is not None:
                            abort_fence(fenced_id, generation=generation)


@pytest.mark.asyncio
@private_profile_test
async def test_session_close_navigation_journeys(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Run three real tab-strip journeys with independent app lifetimes.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Parent directory for the saved-history scenario's database.
    """
    saved_dir = tmp_path / "saved-history"
    saved_dir.mkdir()
    for verify, kwargs in (
        (
            _verify_clicking_x_closes_an_idle_saved_tab_and_a_blank_tab,
            {"tmp_path": saved_dir},
        ),
        (_verify_at_risk_tab_dialog_stay_keeps_it_and_close_closes_it, {}),
        (_verify_middle_click_closes_a_tab_without_switching_to_it, {}),
    ):
        try:
            await verify(request, **kwargs)
        finally:
            drain_active_service_patches()
            drain_created_dirs()
            gc.unfreeze()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                gc.collect()


@pytest.mark.asyncio
@private_profile_test
async def test_session_close_failure_and_retry_journeys(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run six Close recovery journeys with fresh apps and scoped patches.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Parent directory for distinct saved-history databases.
        monkeypatch: Fixture providing a separate patch context per journey.
    """
    error_dir = tmp_path / "internal-error"
    error_dir.mkdir()
    unfinished_dir = tmp_path / "unfinished-close"
    unfinished_dir.mkdir()
    for verify, kwargs in (
        (
            _verify_internal_close_error_names_the_tab_but_never_the_error_text,
            {"tmp_path": error_dir, "monkeypatch": monkeypatch},
        ),
        (
            _verify_close_that_does_not_finish_is_reported_and_keeps_tab_state,
            {"tmp_path": unfinished_dir},
        ),
        (
            _verify_close_flow_that_cannot_start_tells_the_user,
            {"monkeypatch": monkeypatch},
        ),
        (_verify_a_refusal_the_user_can_act_on_shows_its_own_reason, {}),
        (_verify_failure_after_the_close_landed_says_so_and_leaves_no_dead_tab, {}),
        (_verify_failed_confirmed_close_reoffers_confirmation_without_retrying, {}),
    ):
        try:
            with monkeypatch.context() as patch:
                if "monkeypatch" in kwargs:
                    kwargs["monkeypatch"] = patch
                await verify(request, **kwargs)
        finally:
            drain_active_service_patches()
            drain_created_dirs()
            gc.unfreeze()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                gc.collect()


@pytest.mark.asyncio
@private_profile_test
async def test_session_close_pending_race_and_fleet_journeys(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run pending, race, geometry and fleet scenarios with fresh owners.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Parent directory for the fleet scenarios' private databases.
        monkeypatch: Fixture providing a separate patch context per scenario.
    """
    fleet_dir = tmp_path / "fleet-close"
    fleet_dir.mkdir()
    for verify, kwargs in (
        (
            _verify_background_pending_close_names_consequences_and_cancels_only_its_owner,
            {},
        ),
        (_verify_background_pending_close_releases_round_without_an_active_turn, {}),
        (
            _verify_chat_create_enrichment_cannot_arm_after_its_session_closes,
            {"monkeypatch": monkeypatch},
        ),
        (
            _verify_all_close_consequences_keep_named_title_and_actions_painted_at_80x24,
            {},
        ),
        (
            _verify_progress_close_failure_reconciles_fleet_before_confirmed_retry,
            {"tmp_path": fleet_dir, "monkeypatch": monkeypatch},
        ),
    ):
        try:
            with monkeypatch.context() as patch:
                if "monkeypatch" in kwargs:
                    kwargs["monkeypatch"] = patch
                await verify(request, **kwargs)
        finally:
            drain_active_service_patches()
            drain_created_dirs()
            gc.unfreeze()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                gc.collect()


@pytest.fixture(autouse=True)
def _observe_original_tab_sync_lifetime(request):
    """Opt-in passive child-only observation; original journeys stay intact."""
    import hashlib
    import json
    import os
    import sys

    from Tests.private_profile import is_private_profile_child

    if (
        os.environ.get("TLDW_TEST_TAB_SYNC_LIFETIME") != "1"
        or request.node.name
        not in {
            "test_session_close_navigation_journeys",
            "test_session_close_failure_and_retry_journeys",
            "test_session_close_pending_race_and_fleet_journeys",
        }
        or not is_private_profile_child(request)
    ):
        yield
        return

    from Tests.UI._tab_sync_original_lifetime import OriginalTabSyncLifetime

    # Keep the diagnostic beside this exact original child's XML unless the
    # root supplies an explicit metadata-only Evidence receipt destination.
    output = Path(
        os.environ.get("TLDW_TEST_TAB_SYNC_RECEIPT")
        or str(
            Path(os.environ["TLDW_TEST_CONFIG_ROOT"]).parent
            / (request.node.name + ".tab-sync-lifetime.json")
        )
    )
    observer = OriginalTabSyncLifetime()
    source_paths = {name: Path(module.__file__) for name, module in observer.modules}
    helper_module = sys.modules[OriginalTabSyncLifetime.__module__]
    source_paths[helper_module.__name__] = Path(helper_module.__file__)
    before = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    stop_error = None
    try:
        observer.start()
        yield
    finally:
        try:
            if observer.active:
                observer.stop()
        except BaseException as error:
            stop_error = type(error).__name__
            raise
        finally:
            facts = observer.receipt()
            after = {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in source_paths.items()
            }
            facts.update(
                original_selected_node=request.node.nodeid,
                source_hashes_before=before,
                source_hashes_after=after,
                all_source_hashes_stable=(before == after),
                observer_stop_error_type=stop_error,
                original_test_bodies_assertions_and_deadlines_unchanged=True,
            )
            output.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
            print(
                "tab-sync-lifetime receipt="
                + str(output)
                + " events="
                + str(len(facts["events"]))
                + " overflow="
                + str(facts["overflow"])
                + " unmatched="
                + str(facts["live_original_frames"])
                + " restored="
                + str(facts["restoration"])
            )
