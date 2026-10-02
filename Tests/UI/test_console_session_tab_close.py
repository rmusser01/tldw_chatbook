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

import asyncio
import threading
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
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Agents.run_context import use_run_id
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


def _failure_toasts(notes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Error and warning toasts: a close that worked, or a Stay, shows none."""

    return [note for note in notes if note[1] in {"error", "warning"}]


@pytest.mark.asyncio
@private_profile_test
async def test_clicking_x_closes_an_idle_saved_tab_and_a_blank_tab(request, tmp_path):
    """AC #1 / #5: the real ✕ click runs the close worker and the tab goes."""

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


@pytest.mark.asyncio
@private_profile_test
async def test_at_risk_tab_dialog_stay_keeps_it_and_close_closes_it(request):
    """AC #2: Stay keeps the tab and its draft; Close really closes it."""

    app = _ready_app()
    notes = _record_notifications(app)
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


@pytest.mark.asyncio
@private_profile_test
async def test_middle_click_closes_a_tab_without_switching_to_it(request):
    """AC #3: a middle-click closes the tab; it never activates it first.

    Closing the ACTIVE tab activates its right-hand neighbour, so a
    middle-click that switched to ``doomed`` before closing it would leave
    ``other`` active instead of ``keeper``.
    """

    app = _ready_app()
    notes = _record_notifications(app)
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
        assert _failure_toasts(notes) == []

        # A plain click still activates: the tab prevents Textual's default
        # only after its own press has run.
        await _click(pilot, f"#{_TAB_PREFIX}{other.id}")
        assert await _settle(pilot, lambda: store.active_session_id == other.id)
        assert set(_session_ids(store)) == {keeper, other.id}


@pytest.mark.asyncio
@private_profile_test
async def test_internal_close_error_names_the_tab_but_never_the_error_text(
    request, tmp_path
):
    """AC #4: a close that fails inside the runtime is shown and logged.

    The runtime refuses to close a session it has already fenced with an
    internal ``RuntimeError("Console session is closed.")`` -- text that
    would contradict the still-open tab, so the toast names only the error
    type, and the log carries the type and origin but never the text
    (TASK-15103). The next ✕ press is not swallowed by the in-flight guard.
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
            assert not any("Console session is closed" in r for r in records)
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
            assert "The close did not finish. Try again in a moment." in message
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


@pytest.mark.asyncio
@private_profile_test
async def test_a_refusal_the_user_can_act_on_shows_its_own_reason(request):
    """AC #4: a refusal meant for the user is shown in its own words."""

    app = _ready_app()
    notes = _record_notifications(app)
    host = ConsoleHarness(app)
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
        await _await_tabs(console, pilot, {keeper, pending})


@pytest.mark.asyncio
@private_profile_test
async def test_failure_after_the_close_landed_says_so_and_leaves_no_dead_tab(
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
    host = ConsoleHarness(app)
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
            assert await _settle(pilot, lambda: bool(notes)), (
                "teardown failure was silent"
            )
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


async def _arm_pending_round(controller, kind: str, session_id: str):
    """Arm a real blocking round without executing the proposed tool."""

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

    with use_run_id(f"close-{kind}-{session_id}"):
        return asyncio.create_task(asyncio.to_thread(request))


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("kind", "consequence", "result"),
    [
        ("approval", "Tool approvals: denied; runs cancelled.", {"close-call": "deny"}),
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
    ],
)
async def test_background_pending_close_names_consequences_and_cancels_only_its_owner(
    request,
    kind,
    consequence,
    result,
):
    """TASK-33621.16: real background rounds, run cancellation and physical Close."""
    app = _ready_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        app.call_from_thread = host.call_from_thread
        store = controller.store
        keeper = store.active_session_id
        doomed = controller.new_session(title="Pending [notes]")
        controller.switch_session(keeper)
        assistant = store.append_message(
            doomed.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        cancelled = asyncio.Event()
        controller._active_cancel_events[doomed.id] = threading.Event()
        round_task = await _arm_pending_round(controller, kind, doomed.id)

        async def waiting_run():
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
            assert await _settle(
                pilot,
                lambda: (
                    kind in controller.pending_round_kinds(doomed.id)
                    and doomed.id in controller._active_stream_tasks
                ),
            ), "the actual pending round and its owning run did not arm"
            await _show_tabs(console, pilot, {keeper, doomed.id})
            assert (
                controller.run_marker_for(doomed.id) is ConsoleRunMarker.NEEDS_APPROVAL
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
                pilot, lambda: dialog.query_one("#cancel-button").has_focus
            )
            await _click(pilot, "#confirm-button")
            assert await _settle(pilot, lambda: doomed.id not in _session_ids(store))
            await _await_tabs(console, pilot, {keeper})
            assert await _settle(
                pilot, lambda: cancelled.is_set() and round_task.done()
            )
            assert await round_task == result
            assert run_task.cancelled()
            assert not controller.has_pending_approval_round(doomed.id)
            assert (
                controller._interrupt_host.session_round_payloads(kind, doomed.id) == []
            )
            assert not controller._interrupt_host.registries[kind]
            assert store.active_session_id == keeper
            assert not console._console_runtime().console_needs_attention
        finally:
            controller.revoke_approval_rounds_for_run(f"close-{kind}-{doomed.id}")
            controller._cancel_pending_decisions_for_session(doomed.id)
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
            await asyncio.wait_for(asyncio.shield(round_task), 5)


@pytest.mark.asyncio
@private_profile_test
async def test_background_question_close_releases_round_without_an_active_turn(request):
    """A closed session's question must not survive when no turn cancel signal exists."""
    app = _ready_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        app.call_from_thread = host.call_from_thread
        store = controller.store
        keeper = store.active_session_id
        doomed = controller.new_session(title="Question")
        controller.switch_session(keeper)
        sibling = await _arm_pending_round(controller, "question", keeper)
        pending = await _arm_pending_round(controller, "question", doomed.id)
        try:
            assert await _settle(
                pilot, lambda: len(controller.pending_question_ids()) == 2
            ), "both real question rounds must be armed before closing"
            await _show_tabs(console, pilot, {keeper, doomed.id})
            await _click(pilot, f"#console-close-session-tab-{doomed.id}")
            await _wait_for_confirmation(host)
            await _click(pilot, "#confirm-button")
            assert await _settle(pilot, lambda: doomed.id not in _session_ids(store))
            await _await_tabs(console, pilot, {keeper})
            assert await _settle(pilot, pending.done, timeout=2), (
                "closed session left its question armed without an owning turn"
            )
            assert await pending == {"answered": False, "reason": "cancelled"}
            assert not sibling.done(), (
                "closing the background tab answered the viewed tab"
            )
            assert len(controller.pending_question_ids()) == 1
            assert controller.pending_round_kinds(keeper) == {"question"}
            assert not controller.has_pending_approval_round(doomed.id)
        finally:
            for session_id in (keeper, doomed.id):
                controller.revoke_approval_rounds_for_run(
                    f"close-question-{session_id}"
                )
            await asyncio.wait_for(asyncio.gather(sibling, pending), 5)


@pytest.mark.asyncio
@private_profile_test
async def test_failed_confirmed_close_reoffers_confirmation_without_retrying(request):
    """A refused at-risk close keeps work and offers a fresh, explicit retry."""
    app = _ready_app()
    notes = _record_notifications(app)
    host = ConsoleHarness(app)
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


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("title", ["Pending [notes]", "A" * 60])
async def test_all_close_consequences_keep_named_title_and_actions_painted_at_80x24(
    request, title
):
    """All loss categories must fit the real dialog at the minimum terminal size."""
    app = _ready_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
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
                {"approval", "question", "skill_install", "skill_script"}
            ),
        )
        worker = console.run_worker(
            console._session._confirm_session_close(impact), exit_on_error=False
        )
        dialog = await _wait_for_confirmation(host)
        assert await _settle(
            pilot, lambda: dialog.query_one("#cancel-button").has_focus
        )
        try:
            container = dialog.query_one("#confirmation-dialog")
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
        finally:
            dialog.dismiss(False)
            assert await worker.wait() is False
