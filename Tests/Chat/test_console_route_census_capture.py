"""TASK-42552: the route census's capture column matches what a send does.

``CONSOLE_REQUEST_ROUTE_CENSUS`` says, per logical route, whether its calls are
traced. This drives each user action through the real controller, store,
gateway preparation and trace runtime of a Capture-On chat (only the provider
adapter is a recorder, as in ``test_console_trace_rewritten_history.py``) and
checks that the action reserves a trace call exactly when the census says its
route is traced.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_send_diagnostics import sinks  # noqa: F401
from Tests.Chat.test_console_trace_current_turn_transforms import (  # noqa: F401
    make_database,
    make_gateway,
    trace_harness,
)
from Tests.Chat.test_console_trace_rewritten_history import (  # noqa: F401
    SESSION,
    _assert_captured,
    _send,
    harness,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    CONSOLE_REQUEST_ROUTE_CENSUS,
    ConsoleRequestRoute,
    ConsoleRouteCaptureDisposition,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def census_harness(harness):  # noqa: F811
    """The shared harness; on the agent path, close the runs database it opens."""
    yield harness
    bridge = getattr(harness.controller, "_agent_bridge", None)
    if harness.path == "agent" and bridge is not None:
        bridge._db.close()


async def _act(chat, action: str, first) -> None:
    controller = chat.controller
    if action == "fresh":
        await _send(chat, "q1")
    elif action == "edit":
        await controller.edit_and_resend_message(first.user_message_id, "edited")
    elif action == "regenerate":
        await controller.regenerate_message(first.assistant_message_id)
    elif action == "continue":
        await controller.continue_from_message(first.assistant_message_id)
    elif action == "impersonate":
        await controller.impersonate_user_reply(SESSION)
    else:
        raise AssertionError(action)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "route"),
    [
        ("fresh", ConsoleRequestRoute.FRESH),
        ("edit", ConsoleRequestRoute.EDIT),
        ("regenerate", ConsoleRequestRoute.REGENERATE),
        ("continue", ConsoleRequestRoute.CONTINUE),
        ("retry", ConsoleRequestRoute.RETRY),
        ("impersonate", ConsoleRequestRoute.IMPERSONATE),
    ],
)
async def test_a_captured_chats_action_is_traced_as_the_route_census_says(
    census_harness, action, route
):
    chat = census_harness
    first, sent, reserved = await _send(chat, "q0")
    _assert_captured(chat, first, sent, reserved)
    if action == "retry":
        chat.adapter_error = True
        failed, _sent, _reserved = await _send(chat, "q1")
        chat.adapter_error = False
        message = chat.store.get_message(failed.assistant_message_id)
        assert "failed" in (message.status, message.assistant_generation_state)
    calls, reservations = len(chat.entries), len(chat.boundaries)

    if action == "retry":
        await chat.controller.retry_message(failed.assistant_message_id)
    else:
        await _act(chat, action, first)

    record = next(item for item in CONSOLE_REQUEST_ROUTE_CENSUS if item.route is route)
    traced = record.capture is ConsoleRouteCaptureDisposition.CONVERSATION_TRACE
    assert len(chat.entries) - calls == 1, "the action must reach the provider once"
    assert chat.failures == []
    assert len(chat.boundaries) - reservations == int(traced), (
        f"{route.value}: census says traced={traced}"
    )
