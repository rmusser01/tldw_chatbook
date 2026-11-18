"""Real Send configuration capture without unrelated display admission."""

import asyncio
import io
import sys
import threading
from dataclasses import replace
from types import CodeType

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["none", "inputs", "source", "config", "hidden", "uncertain"]
)
async def test_owned_send_selection_joins_config_before_claim_without_full(
    request, change
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_backend_send_capture as capture_source
    from tldw_chatbook.Chat.console_domain_display import ConsoleDomainDisplayOwner
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )
    from tldw_chatbook.Chat.console_execution_projection_result import (
        encode_pressed_inputs,
    )
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage,
        WireIdentity,
        decode_message,
    )
    from tldw_chatbook.Chat.console_execution_retirement import _ReadResources
    from tldw_chatbook.Chat.console_execution_worker import (
        _BackendChannel,
        _BackendService,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="send-capture"
    )
    loop = asyncio.get_running_loop()
    original_sources = None
    generation = config._CONFIG_GENERATION
    captures, fulls, read_threads = [], [], []
    handler = None
    receipt = None
    lock_held = False
    read_code = next(
        code
        for code in capture_source.capture_backend_send_selection.__code__.co_consts
        if type(code) is CodeType and code.co_name == "read"
    )

    def loaded(frame):
        if frame.f_back.f_code is not read_code:
            return
        capture = frame.f_back.f_locals["capture"]
        captures.append(capture)
        read_threads.append(threading.current_thread())
        if change not in {"inputs", "source", "config", "hidden"}:
            return
        done, failures = threading.Event(), []

        def mutate():
            try:
                if change == "inputs":
                    backend.store.set_session_draft(
                        session.id, "newer draft", authored_token=(2, 1)
                    )
                elif change == "source":
                    backend.domain.sources = replace(original_sources)
                elif change == "config":
                    config._CONFIG_GENERATION += 1
                else:
                    backend.presentation.view_lifecycle.admit("send-visit", 1, "hidden")
            except BaseException as error:
                failures.append(error)
            finally:
                done.set()

        loop.call_soon_threadsafe(mutate)
        assert done.wait(10)
        if failures:
            raise failures[0]

    def negative(frame):
        observation = frame.f_locals["self"]
        if (
            change == "uncertain"
            and type(observation.read.creator) is capture_source._CapturedSendSelection
        ):
            observation.invalid = True

    try:
        backend.start()
        original_sources = backend.domain.sources
        session = backend.store.create_session(
            title="Original finite Send",
            assistant_kind="generic",
            settings=ConsoleSessionSettings(
                provider="openai", model=None, source="user"
            ),
        )
        backend.store.set_session_draft(
            session.id, "captured draft", authored_token=(1, 1)
        )
        inputs = backend.store.session_input_snapshot(session.id)
        lifecycle = backend.presentation.view_lifecycle
        await lifecycle.prepare(lifecycle.admit("send-visit", 1, "mount"))
        channel = _BackendChannel(io.BytesIO(), io.BytesIO())
        service = _BackendService(backend, channel)
        message = ConsoleWireMessage(
            WireIdentity(
                "profile",
                "send-capture",
                1,
                "finite-selection",
                session.id,
                session.incarnation_id,
                backend.controls.source_revision,
            ),
            "command",
            "ReceiveTurn",
            {
                "turn_id": "finite-selection-turn",
                "draft_revision": inputs.draft_revision,
                "selection": {
                    "source": "backend",
                    "attachment_id": "send-visit",
                    "visit": 1,
                    "rag_source_types": ["notes"],
                },
                "expected_settings_revision": inputs.settings_revision,
                "expected_attachment_revision": inputs.attachment_revision,
                "expected_prefill_revision": inputs.prefill_revision,
                "pressed_inputs": encode_pressed_inputs(inputs, session),
            },
        )
        await backend.controls._projection_read_lock.acquire()
        lock_held = True
        with (
            _observe(
                config.load_settings.__code__, sys.monitoring.events.PY_START, loaded
            ),
            _observe(
                _ReadResources.confirmed.__code__,
                sys.monitoring.events.PY_START,
                negative,
            ),
            _observe(
                ConsoleDomainDisplayOwner.read.__code__,
                sys.monitoring.events.PY_START,
                lambda _frame: fulls.append(True),
            ),
        ):
            handler = service.accept(message)
            assert handler is not None
            receipt = backend.adapter._receipts[message.identity.request_id][1]
            assert message.identity.request_id not in service._received_requests
            if change == "none":
                await asyncio.shield(receipt.applied)
                assert message.identity.request_id in service._received_requests
                record = backend.controls._turns["finite-selection-turn"][1]
                assert record.received_intent.selection is captures[0].selection
                assert tuple(
                    record.received_intent.selection.rag_defaults["source_types"]
                ) == ("notes",)
                cancelled = backend.dispatch(
                    ConsoleWireMessage(
                        WireIdentity(
                            "profile",
                            "send-capture",
                            2,
                            "cancel-finite",
                            session.id,
                            session.incarnation_id,
                            backend.controls.source_revision,
                        ),
                        "command",
                        "CancelTurn",
                        {"turn_id": "finite-selection-turn"},
                    )
                )
                await asyncio.shield(cancelled.applied)
            else:
                with pytest.raises(RuntimeError):
                    await asyncio.shield(receipt.applied)
                assert "finite-selection-turn" not in backend.runtime._turn_custody
                assert message.identity.request_id not in service._received_requests
            await asyncio.shield(handler)
        assert len(captures) == 1 and fulls == []
        capture = captures[0]
        assert read_threads[0] is not threading.current_thread()
        assert capture.effect.completed.result() is (change != "uncertain")
        assert not capture.reads
        frames = []
        while not channel.outbox.empty():
            frame = decode_message(channel.outbox.get_nowait())
            if frame.identity.request_id == message.identity.request_id:
                frames.append(frame)
        assert frames[-1].kind == "retired"
        if change == "none":
            assert frames[0].kind == "received" and any(
                frame.kind == "applied" for frame in frames
            )
        else:
            assert [frame.kind for frame in frames] == ["failed", "retired"]
            assert frames[0].payload["code"] == (
                "send_selection_unsettled"
                if change == "uncertain"
                else "send_selection_refused"
            )
            assert frames[-1].payload["retired"] is (change != "uncertain")
            assert session.draft == (
                "newer draft" if change == "inputs" else "captured draft"
            )
        request.node.user_properties.append(("send_selection_full_reads", len(fulls)))
    finally:
        if lock_held:
            backend.controls._projection_read_lock.release()
        if original_sources is not None:
            backend.domain.sources = original_sources
        config._CONFIG_GENERATION = generation
        if receipt is not None and receipt.turn_retirement is not None:
            receipt.work.cancel()
        if handler is not None:
            await asyncio.gather(handler, return_exceptions=True)
        if change == "uncertain" and captures:
            with pytest.raises(RuntimeError):
                await backend.aclose()
        else:
            await backend.aclose()


def test_backend_send_selection_variant_is_closed_and_explicit_selection_unchanged():
    from Tests.Chat.test_console_execution_protocol import receive_payload
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleProtocolError,
        _detach,
        _send_selection,
    )

    explicit = receive_payload()["selection"]
    _send_selection(_detach(explicit, "selection", 0, set()), "selection")
    backend = {
        "source": "backend",
        "attachment_id": "visit",
        "visit": 1,
        "rag_source_types": [],
    }
    _send_selection(_detach(backend, "selection", 0, set()), "selection")
    for changed in (
        {**backend, "visit": 0},
        {**backend, "provider": "injected"},
        {**backend, "rag_source_types": [False]},
    ):
        with pytest.raises(ConsoleProtocolError):
            _send_selection(_detach(changed, "selection", 0, set()), "selection")
