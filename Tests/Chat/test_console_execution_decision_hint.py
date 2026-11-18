"""Inert decision wakes never replace the original projection or answerability."""

import asyncio
import io
from dataclasses import replace

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_execution_protocol import (
    ConsoleProtocolError,
    ConsoleWireMessage,
    WireIdentity,
    decode_message,
    encode_message,
)


def test_decision_hint_is_closed_scoped_and_distinct_from_command_receipts():
    identity = WireIdentity(
        "profile", "service", 1, "decision-change:2", "session", "incarnation", 3
    )
    payload = {"attachment_id": "view", "visit": 1, "decision_revision": 2}
    value = ConsoleWireMessage(identity, "decision_changed", "ReadProjection", payload)
    assert decode_message(encode_message(value)) == value
    for invalid in (
        dict(payload, decision_revision=0),
        dict(payload, decision_revision=True),
        dict(payload, visit=0),
        dict(payload, attachment_id=""),
        dict(payload, decision={}),
        dict(payload, retired=True),
    ):
        with pytest.raises(ConsoleProtocolError):
            ConsoleWireMessage(identity, "decision_changed", "ReadProjection", invalid)
    for invalid_identity in (
        replace(identity, request_id="ordinary-request"),
        replace(identity, session_id="", incarnation_id=""),
    ):
        with pytest.raises(ConsoleProtocolError):
            ConsoleWireMessage(
                invalid_identity, "decision_changed", "ReadProjection", payload
            )
    for kind, operation in (
        ("command", "ReadProjection"),
        ("received", "ReadProjection"),
        ("decision_changed", "ReceiveTurn"),
    ):
        with pytest.raises(ConsoleProtocolError):
            ConsoleWireMessage(identity, kind, operation, payload)


@private_profile_test
@pytest.mark.asyncio
async def test_actual_project_decision_wakes_current_view_without_answerability(
    request,
):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_chat_controller import (
        ProjectInstructionDispatchNotice,
    )
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_execution_worker import (
        _BackendChannel,
        _BackendService,
    )

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="decision-hint"
    )
    pending = None
    try:
        backend.start()
        session = backend.store.create_session(title="Decision wake")
        lifecycle = backend.presentation.view_lifecycle
        await lifecycle.prepare(lifecycle.admit("actual-client", 1, "mount"))
        # Retain the actual channel and encoder; no pipe threads are required
        # to inspect the inert output it queues from this real backend owner.
        channel = _BackendChannel(io.BytesIO(), io.BytesIO())
        service = _BackendService(backend, channel)
        identity = ControlIdentity(
            "profile",
            "decision-hint",
            1,
            "capture",
            session.id,
            session.incarnation_id,
            0,
        )
        notice = ProjectInstructionDispatchNotice(
            session.id, "Local endpoint", "AGENTS.md", ".", 42, ("active",), ()
        )
        pending = asyncio.create_task(
            asyncio.to_thread(
                backend.runtime._confirm_project_instruction_dispatch, notice
            )
        )

        async def next_hint():
            while channel.outbox.empty():
                if channel.failure.done():
                    raise channel.failure.result()
                await asyncio.sleep(0.001)
            return decode_message(channel.outbox.get_nowait())

        hint = await asyncio.wait_for(next_hint(), 10)
        assert (hint.kind, hint.operation) == ("decision_changed", "ReadProjection")
        assert hint.identity.session_id == session.id
        assert hint.identity.incarnation_id == session.incarnation_id
        assert hint.identity.source_revision == backend.controls.source_revision
        assert dict(hint.payload) == {
            "attachment_id": "actual-client",
            "visit": 1,
            "decision_revision": 1,
        }
        head = backend.presentation.project_decisions.capture(identity)
        original = backend.runtime._project_dispatch_decisions[head.decision_id]
        assert original.answerable_since is None and not original.ever_answerable
        assert not pending.done()
        assert service._received_requests == set()
        assert backend.adapter._receipts == {}

        # The original mixed-head marshal can issue several change callbacks;
        # one loop turn produces one bounded wake, never a projection body.
        host = backend.controller._interrupt_host
        host._marshal_pending_decision_projection()
        host._marshal_pending_decision_projection()
        second = await asyncio.wait_for(next_hint(), 10)
        assert second.payload["decision_revision"] == 2
        assert second.identity.sequence == hint.identity.sequence + 1
        await asyncio.sleep(0)
        assert channel.outbox.empty()

        # Capture the old visit, then hide before either queued callback runs.
        host._marshal_pending_decision_projection()
        await lifecycle.prepare(lifecycle.admit("actual-client", 1, "hidden"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert channel.outbox.empty()
        assert service._decision_revision == 2
        assert original.answerable_since is None
        assert not channel.failure.done()
        with pytest.raises(RuntimeError):
            service.queue_decision_changed(object())
    finally:
        if pending is not None and not pending.done():
            for decision_id in tuple(backend.runtime._project_dispatch_decisions):
                backend.runtime.resolve_project_instruction_dispatch(
                    decision_id, "cancel"
                )
            await asyncio.shield(pending)
        await backend.aclose()
