"""PROBE (task-17500): measure the harness's natural first-open ordering.

The live pass observed a headless approval card mount VISIBLE AND EMPTY
("Approval required" and nothing else) on first Console open. By state
reachability, the only writer sequence that produces (card.display=True,
#approval-batch-body.display=False) is `set_batch(calls)` followed by
`_hide_batch_body` -- i.e. the card's mount-deferred initial hide landing
AFTER the screen's 0.05s mount sync. This probe measures which order the
run_test harness produces naturally, sampling the display chain over time
and capturing the painted frame at the end. It asserts nothing beyond
harness preconditions; its output is the finding.
"""

from __future__ import annotations

import re
import time
from html import unescape

import pytest

from Tests.Chat.test_console_fleet_wake import _drain, _quiet, _settle, _survivor
from Tests.UI.test_console_headless_approval import (
    _arm,
    _armed_round_ids,
    _build_console_app,
    _risk_row,
    _round_is_claimable,
)
from Tests.UI.test_console_store_continuity import (
    _drain_from_child_thread,
    _navigate,
    _seed_console,
    _terminal_survivor_run,
)

from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard


def _compositor_text(svg: str) -> str:
    joined = "".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    return unescape(joined).replace("\xa0", " ")


def _snap(screen):
    try:
        card = screen.query_one(ChatApprovalCard)
    except Exception:  # noqa: BLE001
        return ("no-card",)
    try:
        body_disp = card.query_one("#approval-batch-body").display
    except Exception:  # noqa: BLE001
        body_disp = "body-missing"
    try:
        tc_disp = screen.query_one("#console-task-surface").display
    except Exception:  # noqa: BLE001
        tc_disp = "surface-missing"
    rows = len(list(card.query(".approval-row")))
    return (tc_disp, card.display, body_disp, rows)


@pytest.mark.asyncio
async def test_probe_first_open_natural_ordering(tmp_path):
    app, gateway = _build_console_app(tmp_path)
    samples: list[tuple[float, tuple]] = []
    painted = ""

    async with app.run_test(size=(160, 48), notifications=True) as pilot:
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
        assert await _settle(lambda: gateway.entered_stall.is_set(), seconds=10.0)

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack

        thread, box = _arm(controller, session_id, call=_risk_row())
        assert await _settle(
            lambda: _round_is_claimable(controller, session_id), seconds=5.0
        )

        chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
        t0 = time.monotonic()
        last = None
        while time.monotonic() - t0 < 3.0:
            snap = _snap(chat2)
            if snap != last:
                samples.append((round(time.monotonic() - t0, 4), snap))
                last = snap
            await pilot.pause(0.01)
        painted = _compositor_text(app.export_screenshot(simplify=True))

        # Resolve the round so the worker thread does not outlive the app.
        from textual.widgets import Button

        buttons = list(chat2.query(".approval-row-fast-approve"))
        if buttons:
            buttons[0].press()
            await pilot.pause()
        else:
            controller.resolve_pending_approval({"builtin__write_file": "deny"})
        await _settle(lambda: "decisions" in box, seconds=10.0)
        thread.join(timeout=5)
        gateway.release.set()
        await pilot.pause()

    print("PROBE samples (t, (surface.display, card.display, body.display, rows)):")
    for entry in samples:
        print("  ", entry)
    print("PROBE painted-frame markers:")
    print("   'Approval required' painted:", "Approval required" in painted)
    print("   'write_file' painted:", "write_file" in painted)
    print("   'Approve' painted:", "Approve" in painted)
