"""PROBE (task-15860 Task 6, throwaway): what a launch actually runs.

Three questions, each answered by execution rather than by reading:

P1. Does a `_build_test_app()` under `app.run_test()` actually reach
    `_post_mount_setup` -> `_schedule_deferred_startup_work`? That is the
    only place a launch-time hook could live.
P2. With Console never opened, what does the app hold: is there a
    controller, a bridge, a wake coordinator?
P3. Is the ephemeral/stale mark leak real -- does a completion in an
    UNSAVED (ephemeral) session write a durable FLEET_UNSEEN mark whose
    conversation id names nothing after a restart?
"""

from __future__ import annotations

import time

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)


async def _settle(pilot, predicate, seconds: float = 8.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return bool(predicate())


@pytest.mark.asyncio
async def test_probe_p1_deferred_startup_runs_and_console_is_absent(tmp_path):
    app = _build_test_app("library")
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    seen: list[str] = []
    real = type(app)._schedule_deferred_startup_work

    def recording(self):
        seen.append("deferred")
        return real(self)

    type(app)._schedule_deferred_startup_work = recording
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            reached = await _settle(pilot, lambda: bool(seen), seconds=15.0)
            print(f"\nPROBE P1 deferred-startup reached: {reached}")
            print(f"PROBE P1 screen stack: {[type(s).__name__ for s in app.screen_stack]}")
            runtime = app.console_runtime
            print(f"PROBE P2 runtime: {runtime!r}")
            print(f"PROBE P2 chat_controller: {runtime.chat_controller!r}")
            print(f"PROBE P2 chat_store: {runtime.chat_store!r}")
            print(f"PROBE P2 agent_bridge: {runtime.agent_bridge!r}")
            print(f"PROBE P2 ui_ready: {getattr(app, '_ui_ready', None)}")
    finally:
        type(app)._schedule_deferred_startup_work = real
    assert reached, "the app never reached _schedule_deferred_startup_work"


@pytest.mark.asyncio
async def test_probe_p4_the_real_conversation_service_can_be_wired_after_the_db(
    tmp_path,
):
    """Can a launch test use the PRODUCTION tree loader instead of a double?

    `_build_test_app` patches `get_chachanotes_db_lazy` to None, so
    `_wire_chat_conversation_services` runs at `__init__` with no DB and
    leaves `local_chat_conversation_service` None -- which is why every
    resume test in the suite injects a `StaticConversationTreeService`.
    Re-running the wiring after `_attach_real_dbs` should give the real one.
    """
    from tldw_chatbook.Chat.console_conversation_hydration import (
        load_console_conversation_tree,
    )

    app = _build_test_app("library")
    _attach_real_dbs(app, tmp_path)
    print(f"\nPROBE P4 before rewire: {app.local_chat_conversation_service!r}")
    app._wire_chat_conversation_services()
    print(f"PROBE P4 after rewire: {app.local_chat_conversation_service!r}")
    app.chachanotes_db.add_conversation({"id": "conv-real", "title": "Real"})
    app.chachanotes_db.add_message(
        {
            "id": "msg-real",
            "conversation_id": "conv-real",
            "sender": "user",
            "content": "a real persisted message",
        }
    )
    tree = await load_console_conversation_tree(app, "conv-real")
    print(f"PROBE P4 tree keys: {sorted(tree) if tree else tree!r}")
    if tree:
        print(f"PROBE P4 root_threads: {tree.get('root_threads')!r}")
    assert tree is not None, "the real service could not load a real conversation"


def test_probe_p3a_an_unsaved_session_is_keyed_by_its_own_session_id(tmp_path):
    """The PRODUCTION source of an unresolvable mark, executed.

    `ConsoleChatController._agent_conversation_id` is what the agent bridge
    (and therefore the fleet drain, and therefore the mark writer) keys a
    conversation by. For an UNSAVED session it returns the ephemeral
    `session.id`, which names no ChaChaNotes conversation.
    """
    from Tests.Chat.test_console_fleet_wake import _controller_rig

    chacha, _app, runs_db, _store, session, _gw, _bridge, controller = _controller_rig(
        tmp_path
    )
    try:
        keyed = controller._agent_conversation_id(session.id)
        print(f"\nPROBE P3a session.persisted_conversation_id: "
              f"{session.persisted_conversation_id!r}")
        print(f"PROBE P3a keyed conversation id: {keyed!r}")
        print(
            "PROBE P3a ChaChaNotes row for that id: "
            f"{chacha.get_conversation_by_id(keyed)!r}"
        )
        assert session.persisted_conversation_id is None
        assert keyed == session.id
        assert chacha.get_conversation_by_id(keyed) is None
    finally:
        runs_db.close()
        chacha.close_connection()


@pytest.mark.asyncio
async def test_probe_p3_ephemeral_conversation_id_leaks_a_permanent_mark(tmp_path):
    """The stale-mark shape: mark an id that names no ChaChaNotes row (an
    ephemeral session id is exactly that), then 'restart' over the same DB
    and see whether anything can resolve it."""
    app = _build_test_app("library")
    marks = _attach_real_dbs(app, tmp_path)
    ephemeral_id = "console-session-abc123"
    marks.set_mark(ephemeral_id, ConversationLocalMarksService.FLEET_UNSEEN)

    # "restart": a second app over the SAME on-disk DB.
    app2 = _build_test_app("library")
    marks2 = _attach_real_dbs(app2, tmp_path)
    listed = marks2.list_marked_conversation_ids(
        ConversationLocalMarksService.FLEET_UNSEEN
    )
    row = app2.chachanotes_db.get_conversation_by_id(ephemeral_id)
    print(f"\nPROBE P3 marks after restart: {listed}")
    print(f"PROBE P3 ChaChaNotes row for that id: {row!r}")
    assert ephemeral_id in listed
