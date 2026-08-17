"""task-17500: a headless approval card must PAINT answerable on first open.

The close-out live pass (2026-08-17, real tty) opened Console on a round
armed headless and saw the card VISIBLE AND EMPTY -- the "Approval
required" title, no tool row, no arguments, no decision controls -- on
both headless paths (nav-away and launch). A session-tab click repaired
it; a round armed with Console mounted rendered fully.

The mechanism, proven by state reachability before any code was changed:
the only writer sequence that can produce the observed state
(`ChatApprovalCard.display=True` + `#approval-batch-body.display=False`
with the surrounding `ChatTaskCards` shown) is a COMPLETED
`set_batch(calls)` followed by `_hide_batch_body` -- the card's
mount-deferred initial hide (`on_mount` -> `call_after_refresh`) landing
AFTER the screen's one-shot 0.05s mount sync (`ChatScreen.on_mount` ->
`set_timer(0.05, sync_task_resume_state)`). On a real terminal the fresh
Console screen's first paint takes longer than 50ms, so the deferred hide
runs LAST and unrenders the batch it was never meant to touch. Textual
gives no ordering guarantee between a screen timer and a widget's
after-refresh callback; `Tests/UI/test_approval_row_information_budget.py`
had already documented the same race as TEST flakiness (TASK-1900's
`_show_batch` workaround: "land `set_batch` first and the hide runs
afterwards") without anyone recognising it as a production defect.

The `run_test` harness resolves the race FAVOURABLY -- measured by
`test_probe_17500_first_open_ordering.py`: at first sample after the real
navigation the card is already fully painted -- which is exactly why the
merged e2e (`test_console_headless_approval.py`) never saw the bug: its
`.approval-row` query and `_rendered` walk also find rows inside a
`display: none` container, the "data arrived" trap. So the tests here do
two things differently:

1. they assert on the PAINTED FRAME (`export_screenshot` `<text>` nodes,
   the compositor-honest idiom) and on the display chain, never on
   queries that see through `display: none`;
2. they make the live ordering deterministic by capturing the card's
   `call_after_refresh` callbacks and delivering them only after the
   mount sync has rendered the card -- the same callbacks, the same
   production scheduling seam, delivered at the time a slow first paint
   delivers them. After the fix the card defers no mount work at all, so
   there is nothing to capture and the delivery step is inert.
"""

from __future__ import annotations

import functools
import re
import time
from html import unescape

import pytest
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button

from Tests.Chat.test_console_fleet_wake import _drain, _settle, _survivor
from Tests.UI.test_console_headless_approval import (
    _arm,
    _build_console_app,
    _risk_row,
    _round_is_claimable,
)
from Tests.UI.test_console_launch_wake import (
    _assert_console_never_mounted,
    _fixture_tree,
    _launch_app,
    _seed_a_finished_background_job,
)
from Tests.UI.test_console_store_continuity import (
    _drain_from_child_thread,
    _navigate,
    _seed_console,
    _terminal_survivor_run,
)

from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards


# ---------------------------------------------------------------------------
# rendered-frame helpers
# ---------------------------------------------------------------------------


def _compositor_text(svg: str) -> str:
    """Rejoin an exported-screenshot SVG's `<text>` nodes into plain text.

    The established compositor-honest idiom (see
    `test_library_media_trash._compositor_text`): content hidden by
    `display: none` never becomes a `<text>` node at all, unlike a
    widget's `.renderable`/`query` results, which exist regardless of
    paint.
    """
    joined = "".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    return unescape(joined).replace("\xa0", " ")


def _painted(app) -> str:
    return _compositor_text(app.export_screenshot(simplify=True))


def _card_paint_state(app) -> str:
    """One-line summary quoted by every failure message."""
    painted = _painted(app)
    return (
        f"title_painted={'Approval required' in painted} "
        f"tool_painted={'write_file (high risk)' in painted} "
        f"controls_painted={'Approve all' in painted}"
    )


class _DeferredCardWork:
    """Capture `ChatApprovalCard.call_after_refresh` work; deliver it late.

    This is the deterministic stand-in for a slow first paint: Textual
    parks after-refresh callbacks on `Screen._callbacks` until the screen
    has actually repainted (`Screen._on_idle` early-returns while dirty),
    so on a real terminal a fresh Console screen delivers the card's
    mount-deferred work AFTER the 0.05s mount sync. `run_test` paints in
    microseconds and always delivers it BEFORE. Capturing the callback at
    the production scheduling seam and invoking it after the sync
    reproduces the terminal's ordering exactly -- same callable, same
    seam, later delivery, which is a delivery time Textual explicitly
    permits.
    """

    def __init__(self) -> None:
        self.work: list[functools.partial] = []
        self._real = ChatApprovalCard.call_after_refresh

    def install(self) -> None:
        capture = self

        def deferred(self, callback, *args, **kwargs):  # noqa: ANN001
            capture.work.append(functools.partial(callback, *args, **kwargs))
            return True

        ChatApprovalCard.call_after_refresh = deferred

    def restore(self) -> None:
        ChatApprovalCard.call_after_refresh = self._real

    def deliver(self) -> int:
        delivered = len(self.work)
        for callback in self.work:
            callback()
        self.work.clear()
        return delivered


async def _assert_card_paints_answerable(app, pilot, chat_screen, box) -> None:
    """The shared verdict: rendered on the frame, then answerable by press."""
    deadline_msg = _card_paint_state(app)
    painted = _painted(app)
    assert "Approval required" in painted, (
        f"the approval card is not painted at all ({deadline_msg}); the round "
        "is invisible and unanswerable"
    )
    assert "write_file (high risk)" in painted, (
        "task-17500's live pane: the card painted TITLE-ONLY -- no tool row, "
        f"no arguments, no controls ({deadline_msg})"
    )
    assert "Approve all" in painted, (
        f"the decision controls are not painted ({deadline_msg}); the user "
        "was told to come and answer and cannot"
    )
    body = chat_screen.query_one("#approval-batch-body")
    assert body.display is True, (
        "the batch body is display:none -- the mount-deferred initial hide "
        "unrendered the batch the mount sync had just set"
    )
    # Answerable, through the rendered control.
    chat_screen.query_one(".approval-row-fast-approve", Button).press()
    await pilot.pause()
    assert await _settle(lambda: "decisions" in box, seconds=10.0), (
        "pressing the painted card's Approve never resolved the round"
    )
    assert box["decisions"] == {"builtin__write_file": "approve_once"}, box[
        "decisions"
    ]


# ---------------------------------------------------------------------------
# THE RED -- both headless paths, through the real navigation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_open_paints_an_answerable_card_nav_away_path(tmp_path):
    """AC#1: wake armed with Console unmounted (left via real navigation);
    the FIRST open must paint the full card -- no session switch.

    RED before the fix: after the deferred mount work lands (as it does
    last on a real terminal), the frame shows 'Approval required' and
    nothing else, exactly the close-out pane.
    """
    app, gateway = _build_console_app(tmp_path)
    deferred = _DeferredCardWork()

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_id = _terminal_survivor_run(runs_db, conversation_id)

        gateway.stall = True
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
        )
        assert await _settle(lambda: gateway.entered_stall.is_set(), seconds=10.0), (
            "harness precondition: the wake turn must be in flight"
        )

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack, "Console must actually unmount"

        thread, box = _arm(controller, session_id, call=_risk_row())
        assert await _settle(
            lambda: _round_is_claimable(controller, session_id), seconds=5.0
        ), "harness precondition: the round must be registered AND payload-retained"

        deferred.install()
        try:
            chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
            assert chat2 is not chat, "screens are never cached"
            assert await _settle(
                lambda: bool(list(chat2.query(".approval-row"))), seconds=5.0
            ), "harness precondition: the mount sync must have built the rows"
            assert await _settle(
                lambda: "write_file (high risk)" in _painted(app), seconds=5.0
            ), (
                "harness precondition: the card must be PAINTED before the "
                "deferred work is delivered, or the delivery proves nothing: "
                f"{_card_paint_state(app)}"
            )
            # The slow first paint completes: the card's mount-deferred work
            # (none, after the fix) lands after the sync.
            deferred.deliver()
        finally:
            deferred.restore()
        await pilot.pause()
        await pilot.pause(0.1)

        await _assert_card_paints_answerable(app, pilot, chat2, box)
        thread.join(timeout=5)
        gateway.release.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_first_ever_open_paints_an_answerable_card_launch_path(tmp_path):
    """AC#2: the round armed before ANY ChatScreen ever existed in the
    process (launch wake, Console never opened); the first-ever open must
    paint the full card.
    """
    conversation_id, run_id, rows = await _seed_a_finished_background_job(tmp_path)
    app, marks, gateway = _launch_app(
        tmp_path, tree=_fixture_tree(conversation_id, rows)
    )
    deferred = _DeferredCardWork()

    async with app.run_test(size=(160, 48)) as pilot:
        gateway.stall = True
        # The launch wake fires on its own; wait for the launch-built
        # controller and its hydrated session, with Console never mounted.
        assert await _settle(
            lambda: getattr(app.console_runtime, "chat_controller", None) is not None,
            seconds=10.0,
        ), "harness precondition: the launch never built a controller"
        controller = app.console_runtime.chat_controller
        store = app.console_runtime.chat_store

        def _hydrated_session_id():
            for session in store.sessions():
                if session.persisted_conversation_id == conversation_id:
                    return session.id
            return None

        assert await _settle(
            lambda: _hydrated_session_id() is not None, seconds=10.0
        ), "harness precondition: the launch never hydrated the owed session"
        session_id = _hydrated_session_id()
        assert await _settle(lambda: gateway.entered_stall.is_set(), seconds=10.0), (
            "harness precondition: the launch wake turn must be in flight"
        )
        _assert_console_never_mounted(app)

        thread, box = _arm(controller, session_id, call=_risk_row())
        assert await _settle(
            lambda: _round_is_claimable(controller, session_id), seconds=5.0
        ), "harness precondition: the round must be registered AND payload-retained"

        deferred.install()
        try:
            chat = await _navigate(app, pilot, "chat", expect="ChatScreen")
            assert store.active_session_id == session_id, (
                "harness precondition: the woken session must be the active "
                "one at first open -- a session SWITCH is the repair path "
                "this test must not take"
            )
            assert await _settle(
                lambda: bool(list(chat.query(".approval-row"))), seconds=5.0
            ), "harness precondition: the mount sync must have built the rows"
            assert await _settle(
                lambda: "write_file (high risk)" in _painted(app), seconds=5.0
            ), (
                "harness precondition: the card must be PAINTED before the "
                f"deferred work is delivered: {_card_paint_state(app)}"
            )
            deferred.deliver()
        finally:
            deferred.restore()
        await pilot.pause()
        await pilot.pause(0.1)

        await _assert_card_paints_answerable(app, pilot, chat, box)
        thread.join(timeout=5)
        gateway.release.set()
        await pilot.pause()


# ---------------------------------------------------------------------------
# the mechanism, pinned at the widget layer
# ---------------------------------------------------------------------------


class _SurfaceHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield ChatTaskCards(id="console-task-surface")


def _approval_payload() -> dict:
    """The parked-payload shape `remount_pending_approval_for_active_session`
    and `switch_session` push into the view seam."""
    return {
        "calls": [
            {
                "llm_name": "builtin__write_file",
                "server_key": "agent:builtin",
                "tool_name": "write_file",
                "reason": "risk_floored",
                "arguments": {"path": "essay.txt", "content": "x"},
            }
        ],
        "timeout_seconds": 0.0,
        "round_id": "round-17500",
    }


@pytest.mark.asyncio
async def test_deferred_mount_work_cannot_unrender_a_live_batch():
    """The mechanism in one widget: whatever work the card defers at mount
    must not unrender a batch that `sync_state` has ALREADY built.

    RED before the fix: the deferred `_hide_batch_body` hides
    `#approval-batch-body` after `set_batch` showed it -- title-only.
    """
    deferred = _DeferredCardWork()
    deferred.install()
    app = _SurfaceHarness()
    try:
        async with app.run_test(size=(120, 36)) as pilot:
            surface = app.query_one("#console-task-surface", ChatTaskCards)
            assert await _settle(
                lambda: bool(app.query("#approval-batch-body")), seconds=5.0
            ), "harness precondition: the card's children must be attached"
            await pilot.pause()
            state = TaskResumeState(pending_approval=_approval_payload())
            surface.sync_state(state)
            await pilot.pause()
            assert "write_file (high risk)" in _painted(app), (
                f"harness precondition: sync must paint first: {_painted(app)!r}"
            )
            deferred.deliver()
            await pilot.pause()
            painted = _painted(app)
            assert "write_file (high risk)" in painted and "Approve all" in painted, (
                "the card's mount-deferred work unrendered a live batch: "
                f"painted={painted!r}"
            )
            assert app.query_one("#approval-batch-body").display is True
    finally:
        deferred.restore()


def test_a_constructed_card_shows_nothing_until_a_batch_is_set():
    """The initial hide must be CONSTRUCTION state, not deferred mount work.

    A hide applied at mount (or after a refresh) is a write racing every
    other writer of the same slot; a hide applied at construction cannot
    race anything. RED before the fix: both widgets are built visible and
    only become hidden when their mount handlers eventually run.
    """
    assert ChatApprovalCard().display is False, (
        "a freshly constructed approval card is visible before its mount "
        "handler runs -- the initial hide is deferred, so it can land after "
        "a sync and unrender it"
    )
    assert ChatTaskCards().display is False, (
        "a freshly constructed task surface is visible before its mount "
        "handler runs"
    )


@pytest.mark.asyncio
async def test_a_mounted_but_never_synced_surface_paints_nothing():
    """The initial-hide contract itself (what the deferred hide existed
    for): with no batch ever set, nothing of the card reaches the frame.
    Kills the mutation that drops the construction-time hides outright.
    """
    app = _SurfaceHarness()
    async with app.run_test(size=(120, 36)) as pilot:
        assert await _settle(
            lambda: bool(app.query("#approval-batch-body")), seconds=5.0
        ), "harness precondition: the card's children must be attached"
        await pilot.pause()
        painted = _painted(app)
        assert "Approval required" not in painted and "Approve all" not in painted, (
            f"an empty task surface painted card chrome: {painted!r}"
        )
        assert app.query_one("#approval-batch-body").display is False


def test_set_batch_does_not_half_apply_when_the_body_is_missing():
    """All-or-nothing: a `set_batch` that cannot reach its containers must
    not leave the card title-only.

    `sync_task_resume_state` swallows `QueryError`, so a `set_batch` that
    raises AFTER flipping `self.display = True` strands a visible, empty
    card with no retry -- the same user-visible state as the deferred-hide
    clobber, through a different writer. RED before the fix: the display
    flip happened before the body query.
    """
    card = ChatApprovalCard()
    with pytest.raises(NoMatches):
        card.set_batch(
            _approval_payload()["calls"], timeout_seconds=0.0, round_id="r"
        )
    assert card.display is False, (
        "a set_batch that could not reach its containers left the card "
        "visible and empty -- title-only, unanswerable"
    )
    assert card._batch_round_id is None, (
        "a failed set_batch stashed the new round id without rendering it"
    )
