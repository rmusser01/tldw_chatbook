"""A real hidden checkpoint cannot overtake an owned display's native tail."""

import asyncio
import io
import sys
import threading

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiting", [False, True])
async def test_hidden_checkpoint_revokes_before_waiting_for_actual_display_retirement(request, cancel_waiting):
    from Tests.Chat.test_console_control_settings import _entered
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat import console_native_commit
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_control_dispatch import ConsoleLocalControls
    from tldw_chatbook.Chat.console_backend_view_lifecycle import ConsoleBackendViewSuperseded
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayVisibilityChanged
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile
    from tldw_chatbook.Chat.console_execution_protocol import ConsoleWireMessage, WireIdentity
    from tldw_chatbook.Chat.console_execution_worker import _BackendChannel, _BackendService

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(expected_config_path=path, expected_data_directory=directory)
    backend = ConsoleExecutionBackend(profile, profile_epoch="profile", service_epoch="projection-overlap")
    release, entered, expired = threading.Event(), threading.Event(), threading.Event()
    hidden_prepared = asyncio.Event()
    receipts, messages, fulls, held = [], [], [], []
    try:
        backend.start()
        channel = _BackendChannel(io.BytesIO(), io.BytesIO())
        service = _BackendService(backend, channel)
        session = backend.store.create_session(title="Actual overlapping display")
        owner = backend.runtime._execution_service.prepare_display_owner()

        def read(action=None):
            sequence = len(receipts) + 1
            payload = {"known_display_revision": None}
            if action is not None:
                payload["view_checkpoint"] = {"attachment_id": "actual-client", "visit": 1, "action": action}
            message = ConsoleWireMessage(
                WireIdentity("profile", "projection-overlap", sequence, f"projection-{sequence}",
                             session.id, session.incarnation_id, backend.controls.source_revision),
                "command", "ReadProjection", payload,
            )
            receipt = backend.dispatch(message)
            messages.append(message)
            receipts.append(receipt)
            return receipt

        def admitted(frame):
            if frame.f_locals["self"] is backend.controls:
                fulls.append(frame.f_locals["full"])

        def native_returned(frame):
            full = owner._full
            if (entered.is_set() or full is None or len(fulls) != 2
                or not full.phase_pending):
                return
            assert threading.current_thread() is not threading.main_thread()
            held.append(full)
            entered.set()
            if not release.wait(10):
                expired.set()

        def visibility_returned(frame):
            if frame.f_locals["self"] is backend.controller and not frame.f_locals["visible"]:
                hidden_prepared.set()

        with (
            _observe(ConsoleLocalControls._begin_display_effect.__code__, sys.monitoring.events.PY_RETURN, admitted),
            _observe(console_native_commit._invoke_plan_native.__code__, sys.monitoring.events.PY_RETURN, native_returned),
            _observe(ConsoleChatController.on_console_view_visibility_changed.__code__, sys.monitoring.events.PY_RETURN, visibility_returned),
        ):
            mount = read("mount")
            await asyncio.shield(mount.applied)
            assert await mount.retired is True
            assert backend.controller._interrupt_host.view_visible
            # A real settings mutation makes the next display collect fresh
            # inputs while retaining the same authenticated visible checkpoint.
            backend.store.set_session_system_prompt(session.id, "Changed selected context")
            prior = read()
            await _entered(entered)
            assert len(held) == 1 and held[0] is owner._full
            full = held[0]
            assert full.reuse_backend_character
            assert not full.effect.completed.done()
            hidden = read("hidden")
            await asyncio.wait_for(hidden_prepared.wait(), 10)
            assert not backend.controller._interrupt_host.view_visible
            assert owner._full is full and len(fulls) == 2
            assert not hidden.applied.done() and not hidden.retired.done()
            if cancel_waiting:
                hidden.work.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await hidden.applied
                assert await hidden.retired is True  # Actual no-entry command.
                assert owner._full is full and not full.effect.completed.done()
            release.set()
            with pytest.raises(ConsoleBackendViewSuperseded) as superseded:
                await asyncio.shield(prior.applied)
            assert type(superseded.value.__context__) is ConsoleDisplayVisibilityChanged
            assert superseded.value.__context__.full is full
            assert await prior.retired is True
            assert full.phase_issuance_closed and full.physically_retired
            assert full.effect.completed.result() is True
            assert full.logical.retired.done()
            assert all(task.done() for task in full.tasks.values())
            assert not expired.is_set()
            message = messages[1]
            command = backend.adapter._receipts[message.identity.request_id][2]
            await service.finish(message, command, prior)
            frames = []
            while not channel.outbox.empty():
                frame = channel.outbox.get_nowait()
                if frame["identity"]["request_id"] == message.identity.request_id:
                    frames.append(frame)
            assert [frame["kind"] for frame in frames] == ["failed", "retired"]
            assert frames[0]["payload"] == {"code": "view_checkpoint_superseded", "stage": "applied"}
            assert frames[1]["payload"]["retired"] is True
            if not cancel_waiting:
                projection = await asyncio.shield(hidden.applied)
                assert projection.session_id == session.id
                assert await hidden.retired is True
                assert len(fulls) == 3 and fulls[-1].effect.completed.result() is True
            assert owner._full is None and not backend.controls.admitted_effects
            assert not backend.controller._interrupt_host.view_visible
    finally:
        release.set()
        await asyncio.gather(*(receipt.work for receipt in receipts), return_exceptions=True)
        await backend.aclose()
