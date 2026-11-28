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
from tldw_chatbook.Chat.citation_repair import CitationRepairContract
from tldw_chatbook.Chat.citation_trace_models import MarkerNamespace
from tldw_chatbook.Chat.console_trace_provenance import (
    CONSOLE_REQUEST_ROUTE_CENSUS,
    ConsoleRequestRoute,
    ConsoleRouteCaptureDisposition,
)

pytestmark = pytest.mark.bootstrap_profile


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
    elif action == "citation_repair":
        # The send carries a citation contract and the reply cites nothing,
        # so the controller makes one repair call after the fresh one.
        await _send(chat, "q1")
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
        ("citation_repair", ConsoleRequestRoute.CITATION_REPAIR),
    ],
)
async def test_a_captured_chats_action_is_traced_as_the_route_census_says(
    harness, action, route, monkeypatch  # noqa: F811 - imported pytest fixture
):
    chat = harness
    first, sent, reserved = await _send(chat, "q0")
    _assert_captured(chat, first, sent, reserved)
    if action == "retry":
        chat.adapter_error = True
        failed, _sent, _reserved = await _send(chat, "q1")
        chat.adapter_error = False
        message = chat.store.get_message(failed.assistant_message_id)
        assert "failed" in (message.status, message.assistant_generation_state)
    if action == "citation_repair":
        contract = CitationRepairContract(
            schema_version=1,
            marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
            allowed_ordinals=(1,),
            evidence_context="[S1] MEDIA - Repair source\nExact repair evidence.",
        )

        async def staged_evidence(_draft, turn_context=None, origin=None):
            return contract.evidence_context, None, None, contract

        original_capture = chat.controller._capture_rag_context
        monkeypatch.setattr(chat.controller, "_capture_rag_context", staged_evidence)
    calls, reservations = len(chat.entries), len(chat.boundaries)

    if action == "retry":
        await chat.controller.retry_message(failed.assistant_message_id)
    else:
        await _act(chat, action, first)

    record = next(item for item in CONSOLE_REQUEST_ROUTE_CENSUS if item.route is route)
    traced = record.capture is ConsoleRouteCaptureDisposition.CONVERSATION_TRACE
    # A citation repair follows its own traced fresh call.
    before = int(action == "citation_repair")
    assert len(chat.entries) - calls == 1 + before, "one provider call per route"
    if before:
        assert chat.entries[-1][-1]["content"].startswith("UNTRUSTED EVIDENCE BEGIN")
    assert chat.failures == []
    assert len(chat.boundaries) - reservations == before + int(traced), (
        f"{route.value}: census says traced={traced}"
    )

    # The User Guide: after each of these, the next send is captured as usual.
    if action == "citation_repair":
        monkeypatch.setattr(chat.controller, "_capture_rag_context", original_capture)
    after, sent, reserved = await _send(chat, "after")
    _assert_captured(chat, after, sent, reserved)
