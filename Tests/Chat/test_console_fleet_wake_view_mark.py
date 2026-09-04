"""task-15971: a wake that completes off-view must leave the ◈ mark set.

The coordinator's design ruling (wake-integrity arc, ledger progress.md):
OFF-SCREEN DELIVERY IS THE INTENDED BEHAVIOR when the Chat screen is
mounted-but-hidden -- the supervisor acts immediately; staging exists only
for a genuinely-unmounted Console (restart / first boot). The requirement
that replaces staging for the hidden case: the user must still LEARN of
the delivery. Live (residue arc, run 3c0cdfaf): the wake delivered while
Library was displayed and the FLEET_UNSEEN mark did not survive -- the
delivery commit cleared it (and the hidden screen's sync tick, pinned
UI-side in ``test_console_fleet_wake_hidden_screen.py``, consumed what
was left). The user was told nothing beyond a transient toast.

Pinned here, at the delivery commit (``_deliver``'s acceptance branch):

- completing while the conversation is NOT in view (screen-wired
  ``wake_conversation_in_view`` probe returns False) SETS the mark
  through the named seam, so the ◈ badge points at the delivered result
  until the user views it (view-clear then applies normally -- a
  delivered wake has nothing pending);
- completing IN view keeps the historical clear;
- a raising probe keeps the mark (fail toward the badge: a stale badge
  self-heals on the next viewed sync tick; a silently-cleared one is the
  live bug);
- an unwired probe (controller doubles, the pre-screen rig) keeps the
  historical clear-on-delivery.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _settle,
    _survivor,
    _terminal_subagent_run,
)
from tldw_chatbook.Chat.console_fleet_attention import (
    FLEET_UNSEEN_REVISION_ATTR,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)


def _marked(app, conversation_id) -> bool:
    return app.conversation_local_marks_service.has_mark(
        conversation_id, ConversationLocalMarksService.FLEET_UNSEEN
    )


async def _deliver_one(runs_db, app, session, gateway, controller):
    """Drive one survivor settle through to an accepted, stamped delivery."""
    _parent, run_id = _terminal_subagent_run(runs_db, session.id)
    app.conversation_local_marks_service.set_mark(
        session.id, ConversationLocalMarksService.FLEET_UNSEEN
    )
    wake = controller.fleet_wake
    wake.on_fleet_drained(_drain(session.id, _survivor(run_id, session_id=session.id)))
    assert await _settle(lambda: gateway.payloads), "the wake never delivered"
    assert await _settle(lambda: not wake.has_pending(session.id))
    assert runs_db.get_run(run_id).get("wake_delivered_at"), (
        "harness precondition: the delivery must commit (stamped ledger)"
    )


@pytest.mark.asyncio
async def test_a_wake_completing_off_view_leaves_the_mark_set(tmp_path):
    """The ruling's core requirement, RED against unmodified production:
    the delivery commit cleared the mark regardless of visibility, so an
    off-screen delivery left the user nothing."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = _controller_rig(
        tmp_path
    )
    try:
        controller.wake_conversation_in_view = lambda conversation_id, session_id: False
        revision_before = int(getattr(app, FLEET_UNSEEN_REVISION_ATTR, 0))
        await _deliver_one(runs_db, app, session, gateway, controller)
        assert _marked(app, session.id), (
            "a wake completing OFF-VIEW must leave the FLEET_UNSEEN mark "
            "set -- the ◈ badge is how the user learns a supervisor turn "
            "ran and delivered while they were elsewhere"
        )
        assert int(getattr(app, FLEET_UNSEEN_REVISION_ATTR, 0)) > revision_before, (
            "the off-view mark must bump the badge revision so screen caches repaint"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_wake_completing_in_view_still_clears_the_mark(tmp_path):
    """The preserved side: the user watched the wake land; nothing is
    unseen; the historical clear stands."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = _controller_rig(
        tmp_path
    )
    try:
        controller.wake_conversation_in_view = lambda conversation_id, session_id: True
        await _deliver_one(runs_db, app, session, gateway, controller)
        assert not _marked(app, session.id), (
            "a wake the user watched land is not unseen -- the delivery "
            "commit must clear the mark as before"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_raising_view_probe_keeps_the_mark(tmp_path):
    """Uncertainty resolves toward the badge: a kept mark on a viewed
    conversation self-heals on the next displayed sync tick; a cleared
    mark on an unviewed delivery is the live silent-delivery bug."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = _controller_rig(
        tmp_path
    )
    try:

        def _broken_probe(conversation_id, session_id):
            raise RuntimeError("view probe broke")

        controller.wake_conversation_in_view = _broken_probe
        await _deliver_one(runs_db, app, session, gateway, controller)
        assert _marked(app, session.id), (
            "a raising view probe must keep the mark (fail toward the "
            "badge, never toward silence)"
        )
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_an_unwired_view_probe_keeps_the_historical_clear(tmp_path):
    """Controller doubles and the pre-screen rig have no view probe; they
    keep the pre-15971 clear-on-delivery (the screen always wires the
    probe in production)."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = _controller_rig(
        tmp_path
    )
    try:
        assert getattr(controller, "wake_conversation_in_view", None) is None
        await _deliver_one(runs_db, app, session, gateway, controller)
        assert not _marked(app, session.id)
    finally:
        chacha.close()
