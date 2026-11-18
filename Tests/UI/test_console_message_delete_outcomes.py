"""A Console Delete or Undo reports what was saved, not what the refresh did.

TASK-33628.5 review follow-ups. The receipt opens at confirm and waits for
the save; its outcome is the save's outcome:

* A delete that committed must offer Undo even when the Console refresh
  after it fails. Before, any exception after the save -- the transcript
  sync, say -- dismissed the receipt as "failed" ("nothing was deleted"):
  no Undo, no word to the user, and held recovered-media references waited
  for the next start-up.
* An Undo that committed must close as restored. Before, a refresh failure
  after it mapped to "retry" and offered Undo again for rows already back.
* Undo is refused, and stays on offer, while a dispatch is pending on the
  conversation. Nothing pinned that refusal on the off-loop path: dropping
  its branch transaction still passed every delete suite.
* Escape while the receipt switches to "Restoring N messages..." (Undo just
  pressed, the progress panel not mounted yet) says the save can't be
  cancelled instead of raising ``NoMatches`` out of a key action.
* The receipt counts what the delete saved.

Every case drives the real ``ChatScreen`` with the real Console store over
an in-memory ChaChaNotes database (``:memory:`` runs the durable half inline;
the off-loop hand-off itself is pinned by
``test_console_message_delete_off_loop.py``), and reads what was painted.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import attach_chachanotes_db
from Tests.UI.test_console_message_delete_undo import (
    _confirm_delete,
    _deleted_flags,
    _open_rows,
    _painted,
    _wait_until,
)
from Tests.UI.test_console_native_chat_flow import _wait_for_selector
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_message_delete_receipt import (
    ConsoleMessageDeleteReceiptModal,
)

# The real ChatScreen/store goes through config-participant admission, which
# the per-test sandbox refuses (RecoveryRequired); keep the collection-time
# profile.
pytestmark = pytest.mark.bootstrap_profile

_ROWS = [
    ("u1", "user", None),
    ("a1", "assistant", "u1"),
    ("u2", "user", "a1"),
    ("a2", "assistant", "u2"),
]
_PENDING_DISPATCH = "Resolve pending dispatch before changing conversation branches."


def _fail_one_refresh_once(controller: Any, saved: Any) -> list[str]:
    """Make the delete flow's next Console refresh raise once ``saved()``.

    Args:
        controller: The Console message controller the delete flow runs on.
        saved: Returns True once the write under test has committed.

    Returns:
        A list that gains one entry when the refresh raised.
    """
    real = controller._sync_native_console_chat_ui_fn
    raised: list[str] = []

    async def refresh() -> None:
        if saved() and not raised:
            raised.append("refresh")
            raise RuntimeError("transcript refresh failed")
        await real()

    controller._sync_native_console_chat_ui_fn = refresh
    return raised


async def _open(host: Any, pilot: Any, db: Any) -> tuple[Any, dict[str, Any]]:
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-transcript")
    return console, await _open_rows(console, db, _ROWS, "a2")


@pytest.mark.asyncio
async def test_a_refresh_failure_after_the_delete_saved_still_offers_undo():
    app = _build_test_app()
    db = attach_chachanotes_db(app)
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console, seeded = await _open(host, pilot, db)
        conversation_id = seeded["conversation_id"]
        raised = _fail_one_refresh_once(
            console._message, lambda: _deleted_flags(db, conversation_id)["u2"] == 1
        )

        await _confirm_delete(console, pilot, host, seeded["native"]["u2"])
        assert raised == ["refresh"]
        await _wait_until(pilot, lambda: "Deleted 2 messages" in _painted(host))
        assert _deleted_flags(db, conversation_id) == {
            "u1": 0,
            "a1": 0,
            "u2": 1,
            "a2": 1,
        }

        # The delete was saved, so Undo is on offer and puts it back.
        host.screen.query_one("#console-delete-receipt-undo", Button).press()
        await _wait_until(
            pilot, lambda: set(_deleted_flags(db, conversation_id).values()) == {0}
        )
        await _wait_until(
            pilot, lambda: not host.screen.query("#console-delete-receipt-box")
        )
        assert any("Restored 2 messages" in notice for notice in notices), notices


@pytest.mark.asyncio
async def test_a_refresh_failure_after_undo_restored_closes_as_restored():
    app = _build_test_app()
    db = attach_chachanotes_db(app)
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console, seeded = await _open(host, pilot, db)
        conversation_id = seeded["conversation_id"]
        await _confirm_delete(console, pilot, host, seeded["native"]["u2"])
        receipt = host.screen
        assert isinstance(receipt, ConsoleMessageDeleteReceiptModal)
        raised = _fail_one_refresh_once(
            console._message,
            lambda: set(_deleted_flags(db, conversation_id).values()) == {0},
        )

        receipt.query_one("#console-delete-receipt-undo", Button).press()
        await _wait_until(pilot, lambda: raised == ["refresh"])
        # The rows are back, so the receipt closes; it must not offer Undo
        # again for messages that are no longer deleted.
        await _wait_until(
            pilot, lambda: not host.screen.query("#console-delete-receipt-box")
        )
        assert set(_deleted_flags(db, conversation_id).values()) == {0}
        assert any("Restored 2 messages" in notice for notice in notices), notices
        assert not any("already back" in notice for notice in notices), notices


@pytest.mark.asyncio
async def test_undo_is_refused_and_stays_on_offer_while_a_dispatch_is_pending():
    app = _build_test_app()
    db = attach_chachanotes_db(app)
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console, seeded = await _open(host, pilot, db)
        conversation_id = seeded["conversation_id"]
        await _confirm_delete(console, pilot, host, seeded["native"]["u2"])

        # A Console send was accepted on this conversation and has not
        # settled: no branch change may land under it, Undo included.
        with db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO console_dispatch_checkpoints (assistant_message_id, "
                "user_message_id, conversation_id, preparation_id, attempt_id, "
                "state, user_message_version, assistant_message_version, origin, "
                "frozen_authority_json, resolved_destination_json, "
                "reconstructability_json) VALUES "
                "(?, ?, ?, ?, ?, 'accepted', 1, 1, 'manual', '{}', '{}', '{}')",
                ("a1", "u1", conversation_id, "prep-pending", "attempt-pending"),
            )
        host.screen.query_one("#console-delete-receipt-undo", Button).press()
        await _wait_until(pilot, lambda: _PENDING_DISPATCH in notices)
        assert _deleted_flags(db, conversation_id)["u2"] == 1
        # Nothing changed, so Undo is offered again.
        await _wait_until(
            pilot, lambda: bool(host.screen.query("#console-delete-receipt-undo"))
        )
        assert "Deleted 2 messages" in _painted(host)

        with db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM console_dispatch_checkpoints WHERE conversation_id = ?",
                (conversation_id,),
            )
        host.screen.query_one("#console-delete-receipt-undo", Button).press()
        await _wait_until(
            pilot, lambda: set(_deleted_flags(db, conversation_id).values()) == {0}
        )


@pytest.mark.asyncio
async def test_escape_while_the_undo_progress_mounts_is_refused_not_raised(
    monkeypatch,
):
    app = _build_test_app()
    db = attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console, seeded = await _open(host, pilot, db)
        conversation_id = seeded["conversation_id"]
        await _confirm_delete(console, pilot, host, seeded["native"]["u2"])
        receipt = host.screen
        assert isinstance(receipt, ConsoleMessageDeleteReceiptModal)

        # Hold the switch to the "Restoring..." panel: Undo has been pressed
        # (the receipt is working) but its status line is not mounted yet.
        real_show = receipt._show
        switching, release = asyncio.Event(), asyncio.Event()

        async def held_show(panel: Any) -> bool:
            if panel.id == "console-delete-receipt-progress":
                switching.set()
                await release.wait()
            return await real_show(panel)

        monkeypatch.setattr(receipt, "_show", held_show)
        receipt.query_one("#console-delete-receipt-undo", Button).press()
        try:
            # Plain asyncio waits: Pilot.pause() waits for every message pump
            # to go idle, and the receipt's stays busy until the release.
            await asyncio.wait_for(switching.wait(), timeout=5.0)
            assert receipt._working == "undo"
            assert not receipt.query("#console-delete-receipt-status")

            # An Escape dispatched in that window: refused, never raised.
            await receipt.request_safe_cancel(source="escape")
            assert host.screen is receipt
            assert _deleted_flags(db, conversation_id)["u2"] == 1
        finally:
            release.set()
        await _wait_until(
            pilot, lambda: set(_deleted_flags(db, conversation_id).values()) == {0}
        )
        await _wait_until(
            pilot, lambda: not host.screen.query("#console-delete-receipt-box")
        )


@pytest.mark.asyncio
async def test_the_receipt_counts_what_the_delete_saved():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async def saved_five() -> int:
        return 5

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        # Confirmed for three; the save removed five (the subtree grew
        # between the confirmation and the write).
        receipt = ConsoleMessageDeleteReceiptModal(count=3, delete=saved_five)
        await host.push_screen(receipt)
        await _wait_until(pilot, lambda: bool(receipt.query("#console-delete-receipt")))
        painted = _painted(host)
        assert "Deleted 5 messages" in painted
        assert "and 4 later messages" in painted
        assert "Deleted 3 messages" not in painted
