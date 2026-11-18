"""Original modal revocation retires old displays without hidden native work."""

import asyncio
import io
import sys

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before_task", "queued", "full_entry", "negative_retirement", "unknown_retirement"])
async def test_original_modal_revokes_captured_display_before_native_entry(request, boundary):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_backend_view_lifecycle import ConsoleBackendViewSuperseded
    from tldw_chatbook.Chat.console_chat_controller import ProjectInstructionDispatchNotice
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity, ConsoleLocalControls
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayVisibilityChanged
    from tldw_chatbook.Chat.console_domain_display import ConsoleDisplayFull
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile
    from tldw_chatbook.Chat.console_execution_protocol import ConsoleWireMessage, WireIdentity
    from tldw_chatbook.Chat.console_execution_worker import _BackendChannel, _BackendService

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(expected_config_path=path, expected_data_directory=directory)
    backend = ConsoleExecutionBackend(profile, profile_epoch="profile", service_epoch="modal-display")
    pending = None
    receipts, fulls = [], []
    lock_owned = False
    negative = boundary in {"negative_retirement", "unknown_retirement"}
    try:
        backend.start()
        channel = _BackendChannel(io.BytesIO(), io.BytesIO())
        service = _BackendService(backend, channel)
        session = backend.store.create_session(title="Actual modal display")
        owner = backend.runtime._execution_service.prepare_display_owner()
        project = backend.presentation.project_decisions
        lifecycle = backend.presentation.view_lifecycle

        def command(operation, payload):
            sequence = backend._sequence + 1
            message = ConsoleWireMessage(
                WireIdentity("profile", "modal-display", sequence, f"command-{sequence}",
                             session.id, session.incarnation_id, backend.controls.source_revision),
                "command", operation, payload,
            )
            receipt = backend.dispatch(message)
            receipts.append(receipt)
            return message, receipt

        def read(action=None):
            payload = {"known_display_revision": None}
            if action is not None:
                payload["view_checkpoint"] = {"attachment_id": "actual-client", "visit": 1, "action": action}
            return command("ReadProjection", payload)

        _, mount = read("mount")
        await asyncio.shield(mount.applied)
        assert await mount.retired is True
        visible = lifecycle.capture_display_checkpoint()
        canceled_message, canceled = read()
        assert canceled_message.identity.request_id in backend.presentation._view_checkpoints
        canceled.work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await canceled.applied
        assert await canceled.retired is True
        await asyncio.sleep(0)
        assert canceled_message.identity.request_id not in backend.presentation._view_checkpoints
        identity = ControlIdentity("profile", "modal-display", 1, "capture", session.id, session.incarnation_id, 0)
        notice = ProjectInstructionDispatchNotice(session.id, "Local endpoint", "AGENTS.md", ".", 42, ("active",), ())
        pending = asyncio.create_task(asyncio.to_thread(backend.runtime._confirm_project_instruction_dispatch, notice))
        async with asyncio.timeout(10):
            while (head := project.capture(identity)) is None:
                await asyncio.sleep(0.001)
        modal = {"attachment_id": "actual-client", "visit": 1, "decision_id": head.decision_id}

        def hide():
            _, hidden = command("ProjectDecisionPresentation", {**modal, "action": "hidden"})
            # Original False means no shown notice was paused; it does not
            # undo the original Console visibility revocation that just ran.
            assert hidden.applied.result() is False
            assert not backend.controller._interrupt_host.view_visible
            assert lifecycle._current is not visible
            assert lifecycle.capture_display_checkpoint() is None

        def starting(frame):
            full = frame.f_locals["self"]
            if full.owner is owner:
                fulls.append(full)
                if boundary == "full_entry" or negative:
                    hide()

        def finished(frame):
            full = frame.f_locals["full"]
            if negative and full in fulls:
                # Contradict the positive resource receipt to exercise refusal;
                # this is a proof-corruption control, not native-close evidence.
                full.effect.physically_retired = False if boundary == "negative_retirement" else None

        if boundary == "full_entry" or negative:
            with (
                _observe(ConsoleDisplayFull.run.__code__, sys.monitoring.events.PY_START, starting),
                _observe(ConsoleLocalControls._finish_display_effect.__code__, sys.monitoring.events.PY_RETURN, finished),
            ):
                message, prior = read()
                expected = ConsoleDisplayVisibilityChanged if negative else ConsoleBackendViewSuperseded
                with pytest.raises(expected):
                    await asyncio.shield(prior.applied)
            assert len(fulls) == 1
            full = fulls[0]
            assert full.logical is None and not full.tasks and not full.retirements
            assert not full.sections and not full.phase_pending and full.phase_issuance_closed
            assert full.effect.completed.result() is True
        else:
            with _observe(ConsoleDisplayFull.run.__code__, sys.monitoring.events.PY_START, starting):
                if boundary == "queued":
                    await backend.controls._projection_read_lock.acquire()
                    lock_owned = True
                message, prior = read()
                if boundary == "queued":
                    async with asyncio.timeout(10):
                        while message.identity.request_id in backend.presentation._view_checkpoints:
                            await asyncio.sleep(0)
                    assert not prior.applied.done()
                hide()
                if lock_owned:
                    backend.controls._projection_read_lock.release()
                    lock_owned = False
                with pytest.raises(ConsoleBackendViewSuperseded):
                    await asyncio.shield(prior.applied)
            assert not fulls and owner._full is None
        positive = not negative
        assert await prior.retired is positive
        assert not backend.controls.admitted_effects
        await service.finish(message, backend.adapter._commands[message.identity.request_id], prior)
        frames = []
        while not channel.outbox.empty():
            frame = channel.outbox.get_nowait()
            if frame["identity"]["request_id"] == message.identity.request_id:
                frames.append(frame)
        assert [frame["kind"] for frame in frames] == ["failed", "retired"]
        assert frames[0]["payload"] == {
            "code": "view_checkpoint_superseded" if positive else "command_failed", "stage": "applied",
        }
        assert frames[1]["payload"]["retired"] is positive
        # A new request under hidden visibility has no displaced visible token
        # and must refuse, rather than start a cold FULL or claim supersession.
        with pytest.raises(RuntimeError, match="hidden view"):
            read()
        assert owner._full is None
        _, mounted = command("ProjectDecisionPresentation", {**modal, "action": "mounted"})
        assert mounted.applied.result() is True
        hidden_token = project.require_modal_display_checkpoint(identity, "actual-client", 1)
        assert hidden_token.action == "hidden"
        _, resolved = command("ResolveDecision", {"decision": {
            "kind": "ProjectDispatchDecision", "decision_id": head.decision_id, "action": "proceed",
        }})
        assert resolved.applied.result() is True
        assert await asyncio.wait_for(pending, 10) == "proceed"
        # The resident read can discover the resolved head without re-enabling
        # Console visibility; only the subsequent original visible checkpoint can.
        assert project.require_modal_display_checkpoint(identity, "actual-client", 1) is hidden_token
        assert project.capture(identity) is None
        assert project.require_modal_display_checkpoint(identity, "actual-client", 1, _captured=hidden_token) is hidden_token
        assert not backend.controller._interrupt_host.view_visible
        _, resumed = read("visible")
        await asyncio.shield(resumed.applied)
        assert await resumed.retired is True
        assert backend.controller._interrupt_host.view_visible
        assert lifecycle.capture_display_checkpoint() is not visible
        with pytest.raises(RuntimeError):
            project.require_modal_display_checkpoint(identity, "actual-client", 1, _captured=hidden_token)
    finally:
        if lock_owned:
            backend.controls._projection_read_lock.release()
        if pending is not None and not pending.done():
            for decision_id in tuple(backend.runtime._project_dispatch_decisions):
                backend.runtime.resolve_project_instruction_dispatch(decision_id, "cancel")
            await asyncio.shield(pending)
        await asyncio.gather(*(receipt.work for receipt in receipts if receipt.work is not None), return_exceptions=True)
        await backend.aclose()
