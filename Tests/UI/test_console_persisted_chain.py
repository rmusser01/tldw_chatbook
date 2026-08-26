"""Every accepted Console turn must EXTEND the persisted chain, never fork it.

Regression cover for the durable dispatch checkpoint added by `a26cdafd8`
("resume Library-gated sends"). That path writes its USER and assistant rows
with raw SQL (`console_dispatch_repository.insert_with_messages`) and took the
parent from `echoed_user.parent_message_id` -- but that field holds the
*persisted* parent id, which `ConsoleChatStore._persist_new_message` is the
only writer of. The optimistic echo is appended with `persist=False`, so the
field was ALWAYS None and every checkpointed turn landed as a fresh DB root.

Measured on dev `8ef5bf12e` before the fix: an ordinary SECOND send in a
single Console visit -- no navigation, no wake, no Library gate -- persisted
its user row with `parent_message_id=None` while the conversation's real leaf
sat one row above it. The transcript still looked right, because the fork is
only visible in the durable tree that reload, rewind, branching and trajectory
export all walk.

Both shapes are covered: a second send inside one visit, and a send after the
user has navigated away and back (which is how the fork was first seen).
"""

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_console_store_continuity import (
    _StallingWakeGateway,
    _configure_native_ready_console,
    _db_chain,
    _navigate,
    _seed_console,
)


def _build_console_app():
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    return app, gateway


def _assert_extends(new_rows, previous_rows) -> None:
    assert new_rows, "the send persisted nothing"
    assert new_rows[0][3] == previous_rows[-1][0], (
        "the accepted turn FORKED the persisted conversation: its user row "
        f"parents to {new_rows[0][3]!r}, not to the chain's leaf "
        f"{previous_rows[-1][0]!r}"
    )


@pytest.mark.asyncio
async def test_a_second_send_in_one_visit_extends_the_persisted_chain(tmp_path):
    """The everyday case: two turns in a row, no navigation involved."""
    app, gateway = _build_console_app()
    _attach_real_dbs(app, tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        _, controller, _, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        db = app.chachanotes_db
        before = _db_chain(db, conversation_id)
        outcome = await controller.submit_draft("second turn", session_id=session_id)
        assert outcome.accepted, outcome.visible_copy
        after = _db_chain(db, conversation_id)
        _assert_extends(after[len(before) :], before)


@pytest.mark.asyncio
async def test_a_send_after_navigating_away_extends_the_persisted_chain(tmp_path):
    """The same invariant across the app-owned runtime's unmount/remount."""
    app, gateway = _build_console_app()
    _attach_real_dbs(app, tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        _, _, _, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        db = app.chachanotes_db
        before = _db_chain(db, conversation_id)
        await _navigate(app, pilot, "library", expect="LibraryScreen")
        chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
        await _wait_for_selector(chat2, pilot, "#console-native-composer")
        await pilot.pause()
        controller2 = chat2._ensure_console_chat_controller()
        outcome = await controller2.submit_draft("after nav", session_id=session_id)
        assert outcome.accepted, outcome.visible_copy
        after = _db_chain(db, conversation_id)
        _assert_extends(after[len(before) :], before)
