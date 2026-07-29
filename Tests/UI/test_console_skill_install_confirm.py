from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace

import pytest

from Tests.UI.test_console_native_chat_flow import (
    RestoredConsoleHarness,
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    _visible_text,
)

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState


class _FakeApp:
    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _controller():
    store = ConsoleChatStore()
    return ConsoleChatController(store=store, provider_gateway=object()), store


@pytest.mark.asyncio
async def test_confirm_round_trip_allow():
    controller, _ = _controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_skill_install = received.append

    async def resolve_soon():
        await asyncio.sleep(0.05)
        assert received and received[0] is not None
        assert received[0]["url"] == "https://github.com/o/r"
        assert received[0]["request_id"]
        controller.resolve_pending_skill_install(
            True, request_id=received[0]["request_id"]
        )

    task = asyncio.create_task(
        asyncio.to_thread(controller.request_skill_install_confirm, "https://github.com/o/r")
    )
    await resolve_soon()
    allowed = await task
    assert allowed is True
    assert received[-1] is None  # card cleared afterwards


@pytest.mark.asyncio
async def test_confirm_round_trip_deny():
    controller, _ = _controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_skill_install = received.append

    async def resolve_soon():
        await asyncio.sleep(0.05)
        assert received and received[0] is not None
        controller.resolve_pending_skill_install(
            False, request_id=received[0]["request_id"]
        )

    task = asyncio.create_task(
        asyncio.to_thread(controller.request_skill_install_confirm, "https://x/y")
    )
    await resolve_soon()
    assert (await task) is False


def test_confirm_timeout_denies():
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    controller.skill_install_confirm_timeout_seconds = lambda: 0.05
    started = time.monotonic()
    allowed = controller.request_skill_install_confirm("https://x/y")
    assert allowed is False
    assert time.monotonic() - started < 2.5


def test_confirm_no_app_denies_immediately():
    """No UI bridge wired (``controller.app`` is None) -> fail closed with no stall.

    Without an app, the marshal in ``_marshal_pending_skill_install`` is a
    no-op and nothing could ever set the Event, so the 120s poll loop would
    never resolve. The early guard must return False well before the
    timeout, without any resolver thread ever being set up.
    """
    controller, _ = _controller()
    assert controller.app is None  # no UI bridge wired at all
    started = time.monotonic()
    allowed = controller.request_skill_install_confirm("https://x/y")
    elapsed = time.monotonic() - started
    assert allowed is False
    assert elapsed < 0.5


def test_own_session_cancel_event_denies_the_round():
    """TASK-910: session B's OWN cancel event (as `stop_active_run`/
    `close_session` would set via `_signal_stop` if B were the viewed/
    closing session) still correctly denies B's round when
    `session_id=B` is threaded through -- mirrors `request_mcp_approvals`'
    identical test."""
    controller, store = _controller()
    session_b = store.ensure_session(title="B").id
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    controller.skill_install_confirm_timeout_seconds = lambda: 30.0
    cancel_event = threading.Event()
    controller._active_cancel_events[session_b] = cancel_event

    def _cancel_soon() -> None:
        time.sleep(0.05)
        cancel_event.set()

    threading.Thread(target=_cancel_soon).start()
    allowed = controller.request_skill_install_confirm(
        "https://x/y", session_id=session_b
    )

    assert allowed is False


def test_unrelated_session_stop_does_not_deny():
    """TASK-910: stopping a DIFFERENT session must not deny THIS session's
    in-flight install confirm -- mirrors `request_mcp_approvals`'
    `test_request_mcp_approvals_unrelated_session_stop_does_not_cross_
    cancel`."""
    controller, store = _controller()
    owning = store.ensure_session(title="Owning").id
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    controller.skill_install_confirm_timeout_seconds = lambda: 30.0

    def _stop_unrelated_soon() -> None:
        time.sleep(0.05)
        controller._signal_stop(session_id="unrelated-session-id")

    def _resolve_soon() -> None:
        time.sleep(0.2)
        request_id = controller.pending_skill_install_ids()[0]
        controller.resolve_pending_skill_install(True, request_id=request_id)

    threading.Thread(target=_stop_unrelated_soon).start()
    threading.Thread(target=_resolve_soon).start()
    allowed = controller.request_skill_install_confirm(
        "https://x/y", session_id=owning
    )

    assert allowed is True


def test_shutdown_denies_pending_confirm():
    """TASK-910: real process teardown (`_shutdown_requested`) still denies
    -- switching context alone no longer does (AC#2), but shutdown must be
    unchanged (AC#4)."""
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    controller.skill_install_confirm_timeout_seconds = lambda: 30.0

    def _shutdown_soon() -> None:
        time.sleep(0.05)
        controller._shutdown_requested.set()

    threading.Thread(target=_shutdown_soon).start()
    allowed = controller.request_skill_install_confirm("https://x/y")

    assert allowed is False


def test_request_skill_install_confirm_parks_for_a_non_active_session():
    """TASK-910: a round whose `session_id` differs from the store's ACTIVE
    session parks -- no card mount (`set_pending_skill_install` never
    called with a real payload), the run-marker pending flag flips, and
    `park_pending_approval` fires exactly once. Visiting (switching to) the
    owning session later mounts the SAME retained payload and lets it
    resolve normally."""
    controller, store = _controller()
    viewed = store.create_session(title="Viewed").id
    background = store.create_session(title="Background").id
    store.switch_session(viewed)  # keep viewing the first session
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_skill_install = mounted.append
    parked: list[str] = []
    controller.park_pending_approval = parked.append
    controller.skill_install_confirm_timeout_seconds = lambda: 30.0

    result_holder: dict[str, bool] = {}

    def _run_round() -> None:
        result_holder["allowed"] = controller.request_skill_install_confirm(
            "https://x/y", session_id=background
        )

    worker = threading.Thread(target=_run_round)
    worker.start()
    time.sleep(0.1)

    assert parked == [background]
    assert mounted == []  # never mounted -- the active session's card is untouched
    assert background in controller._pending_approvals
    assert controller.run_marker_for(background) is ConsoleRunMarker.NEEDS_APPROVAL

    # Visiting + deciding resolves it.
    controller.switch_session(background)
    assert mounted and mounted[-1] is not None
    request_id = mounted[-1]["request_id"]
    controller.resolve_pending_skill_install(True, request_id=request_id)
    worker.join(timeout=2.0)

    assert result_holder["allowed"] is True
    assert background not in controller._pending_approvals
    assert mounted[-1] is None


def test_switch_session_no_longer_denies_a_pending_skill_install_confirm():
    """AC#2/#1: switching away from the OWNING session must park, not
    deny, and switching back re-mounts the SAME round (mirrors
    `test_switch_session_parks_rather_than_denies_a_pending_approval_
    round`)."""
    controller, store = _controller()
    owning_session = store.create_session(title="Owning").id
    other_session = store.create_session(title="Other").id
    store.switch_session(owning_session)
    controller.app = _FakeApp()
    mounted: list[dict | None] = []
    controller.set_pending_skill_install = mounted.append
    controller.skill_install_confirm_timeout_seconds = lambda: 30.0

    result_holder: dict[str, bool] = {}

    def _run_round() -> None:
        result_holder["allowed"] = controller.request_skill_install_confirm(
            "https://x/y", session_id=owning_session
        )

    worker = threading.Thread(target=_run_round)
    worker.start()
    time.sleep(0.1)
    # `owning_session` WAS active at round-start, so it mounted immediately.
    assert mounted and mounted[-1] is not None

    controller.switch_session(other_session)
    time.sleep(0.05)
    assert "allowed" not in result_holder  # not denied by the switch
    assert mounted[-1] is None  # the departing session's card is cleared

    controller.switch_session(owning_session)
    assert mounted[-1] is not None  # re-mounted, same round
    request_id = mounted[-1]["request_id"]
    controller.resolve_pending_skill_install(True, request_id=request_id)
    worker.join(timeout=2.0)

    assert result_holder["allowed"] is True


def test_resolve_pending_skill_install_ignores_a_stale_or_unknown_request_id():
    """AC#3: a mismatched/unknown request_id is a safe no-op leaving the
    round pending -- mirrors `resolve_pending_skill_script`'s identical
    contract."""
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    controller.skill_install_confirm_timeout_seconds = lambda: 30.0
    result: dict[str, bool] = {}

    def worker() -> None:
        result["allowed"] = controller.request_skill_install_confirm("https://x/y")

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.1)
    real_id = controller.pending_skill_install_ids()[0]

    controller.resolve_pending_skill_install(True, request_id="not-a-real-id")
    time.sleep(0.1)
    assert t.is_alive(), "a stale/unknown request_id must not resolve the round"

    controller.resolve_pending_skill_install(True, request_id=real_id)
    t.join(timeout=3.0)
    assert result["allowed"] is True


def test_resolve_pending_skill_install_with_no_request_id_is_dropped():
    """AC#3/#4: a resolve carrying no id at all must never auto-approve."""
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    controller.skill_install_confirm_timeout_seconds = lambda: 30.0
    result: dict[str, bool] = {}

    def worker() -> None:
        result["allowed"] = controller.request_skill_install_confirm("https://x/y")

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.1)

    controller.resolve_pending_skill_install(True)  # request_id omitted
    time.sleep(0.1)
    assert t.is_alive(), "an id-less resolve must not resolve the armed round"

    controller.resolve_pending_skill_install(
        True, request_id=controller.pending_skill_install_ids()[0]
    )
    t.join(timeout=3.0)
    assert result["allowed"] is True


def test_task_resume_state_pending_skill_install_serializes_while_live():
    """``to_dict`` stays a faithful snapshot of a live in-session install card.

    ``pending_skill_install`` is a normal dataclass field while a real
    ``ConsoleChatController`` round is armed in THIS screen instance -- only
    the ``from_dict`` direction of the round-trip drops it (see below).
    """
    s = TaskResumeState()
    assert s.has_pending_skill_install() is False
    s2 = replace(
        s,
        pending_skill_install={
            "url": "https://x/y",
            "timeout_seconds": 120.0,
            "request_id": "r1",
        },
    )
    assert s2.has_pending_skill_install() is True
    assert s2.to_dict()["pending_skill_install"] == {
        "url": "https://x/y",
        "timeout_seconds": 120.0,
        "request_id": "r1",
    }


def test_restored_state_drops_the_pending_install_so_no_dead_card_appears():
    """A restored pending install must never come back as an actionable card.

    TASK-1130: mirrors ``test_skill_script_confirm_card.py::
    test_restored_state_drops_the_pending_script_so_no_dead_card_appears``.
    The confirm it belongs to is a live, blocked worker round keyed by
    ``request_id``, and ``ConsoleChatController.resolve_pending_skill_install``
    strict-matches that id against the currently-armed round. A round that
    survived a save/restore cannot still be armed (a fresh
    ``ConsoleChatController`` is built on every navigation, and TASK-1143's
    navigation guard denies any busy round on teardown besides), so a
    restored card's buttons would all be silently dropped -- a dead card the
    user could click forever while it misrepresents an abandoned request as
    awaiting them. This was TASK-910's round-trip fidelity contract before
    TASK-1130 flipped it to pin the drop instead, mirroring TASK-1051's
    identical script-side decision.
    """
    state = TaskResumeState(
        summary="Keep me",
        pending_skill_install={
            "url": "https://x/y",
            "timeout_seconds": 120.0,
            "request_id": "r1",
        },
    )
    restored = TaskResumeState.from_dict(state.to_dict())
    assert restored.pending_skill_install is None
    assert restored.has_pending_skill_install() is False
    # ...and only the skill-confirm fields are affected.
    assert restored.summary == "Keep me"


@pytest.mark.asyncio
async def test_skill_install_card_allow_and_deny():
    from textual import on
    from textual.app import App, ComposeResult
    from textual.widgets import Button
    from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
        SkillInstallConfirmCard,
    )

    class _Host(App[None]):
        def __init__(self):
            super().__init__()
            self.decided = []

        def compose(self) -> ComposeResult:
            yield SkillInstallConfirmCard()

        @on(SkillInstallConfirmCard.InstallDecided)
        def _cap(self, event: SkillInstallConfirmCard.InstallDecided) -> None:
            self.decided.append((event.allow, event.request_id))

    app = _Host()
    async with app.run_test() as pilot:
        card = app.query_one(SkillInstallConfirmCard)
        # A URL containing Rich-markup-like text must render literally.
        card.set_install(
            {
                "url": "https://github.com/o/[bold]r[/]",
                "timeout_seconds": 120.0,
                "request_id": "round-1",
            }
        )
        await pilot.pause()
        assert card.display is True
        from textual.widgets import Static
        url_text = str(app.query_one("#skill-install-url", Static).render())
        assert "[bold]" in url_text  # not interpreted as markup
        app.query_one("#skill-install-allow", Button).press()
        await pilot.pause()
        assert app.decided == [(True, "round-1")]
        card.set_install(
            {"url": "https://github.com/o/r", "timeout_seconds": 120.0, "request_id": "round-2"}
        )
        await pilot.pause()
        app.query_one("#skill-install-deny", Button).press()
        await pilot.pause()
        assert app.decided == [(True, "round-1"), (False, "round-2")]


def test_chat_screen_forwards_install_decided_to_controller_with_request_id(
    monkeypatch,
):
    """The one contract this bridge exists to protect: a decision must
    carry the pending round's exact request_id through to
    resolve_pending_skill_install, or the resolve is silently dropped."""
    from unittest.mock import Mock

    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
        SkillInstallConfirmCard,
    )

    mock_chat_host = Mock()
    mock_chat_host.chat_right_sidebar_collapsed = False
    mock_chat_host.notify = Mock()
    mock_chat_host.run_worker = Mock()
    mock_chat_host.bell = Mock()

    screen = ChatScreen(mock_chat_host)
    controller = Mock()
    screen._console_chat_controller = controller

    event = SkillInstallConfirmCard.InstallDecided(True, request_id="round-9")
    screen.handle_console_skill_install_decided(event)

    controller.resolve_pending_skill_install.assert_called_once_with(
        True, request_id="round-9"
    )


@pytest.mark.asyncio
async def test_restored_pending_install_never_mounts_an_actionable_card_through_the_real_screen():
    """Drive the drop through the REAL screen boundary, not just `from_dict` in isolation.

    `test_restored_state_drops_the_pending_install_so_no_dead_card_appears`
    (above) pins the contract at the `TaskResumeState.from_dict` unit level.
    This test proves the same thing through the seam a real tab switch
    actually uses -- `ChatScreen.restore_state`, which `app.py`'s
    `handle_screen_navigation` calls on every navigation -- so a regression
    that plumbed `pending_skill_install` back through `from_dict` would show
    up here as an actually-mounted, actually-visible `SkillInstallConfirmCard`,
    not merely a field on a dataclass nobody rendered.

    The snapshot shape mirrors what `ChatScreen.save_state` /
    `_serialize_native_console_state` actually produce (one session is
    required -- `_restore_native_console_state` returns before touching
    `task_resume_state` at all when `sessions` is empty, which would make
    this test pass for the wrong reason).
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
        SkillInstallConfirmCard,
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    session_id = "restored-session-1"
    saved_state = {
        "interface_type": "native_console",
        "native_console_state": {
            "version": 1,
            "active_session_id": session_id,
            "sessions": [
                {
                    "id": session_id,
                    "title": "Restored session",
                    "workspace_id": "default",
                    "persisted_conversation_id": None,
                    "draft": "",
                    "settings": None,
                    "updated_at": None,
                    "character_id": None,
                    "character_name": None,
                }
            ],
            "messages_by_session": {session_id: []},
            "task_resume_state": {
                "summary": "Restored summary text should survive",
                "last_step": "",
                "pending_approval": None,
                "pending_skill_install": {
                    "url": "https://example.com/skill",
                    "timeout_seconds": 120.0,
                    "request_id": "restored-round-1",
                },
                "pending_skill_script": None,
                "diff_summary": "",
                "next_action": "",
            },
            "image_view_modes": {},
        },
    }

    host = RestoredConsoleHarness(app, saved_state)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause()

        # The restore-side drop itself: a snapshot's pending_skill_install
        # must never repopulate the live screen's task-resume state.
        assert console._task_resume_state.pending_skill_install is None
        assert console._task_resume_state.has_pending_skill_install() is False

        # ...and therefore no actionable card is mounted/visible.
        install_card = console.query_one(
            "#chat-skill-install-card", SkillInstallConfirmCard
        )
        assert install_card.display is False

        # An unrelated resume field from the SAME snapshot must still survive
        # the restore -- the drop is scoped to the two skill-confirm fields.
        assert (
            console._task_resume_state.summary
            == "Restored summary text should survive"
        )
        assert "Restored summary text should survive" in _visible_text(console)
