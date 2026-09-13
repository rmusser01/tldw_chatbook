"""Reversible shared STT and Console dictation maintenance behavior."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sys, threading, time
from types import SimpleNamespace
from Tests.STT.test_dispatch_coordinator import FakeExecutor, _begin, _library_kwargs
from tldw_chatbook.STT import dispatch_coordinator as module
from tldw_chatbook.STT.executor import ExecutorBusyError
from tldw_chatbook.STT.contracts import TranscriptionFailureCode

case = sys.argv[1]
async def main():
    executor = FakeExecutor()
    owner = module.LocalSTTDispatchCoordinator(executor)
    if case == "idle_callback":
        entered, release = threading.Event(), threading.Event()
        def idle():
            entered.set()
            assert release.wait(3)
        owner = module.LocalSTTDispatchCoordinator(executor, on_dictation_idle=idle)
        capture = _begin(module, owner)
        owner.maintenance_close_admission()
        finishing = asyncio.create_task(asyncio.to_thread(capture.finish))
        assert await asyncio.to_thread(entered.wait,2)
        assert not await owner.maintenance_drain(time.monotonic())
        release.set()
        await finishing
        assert await owner.maintenance_drain(time.monotonic()+1)
    elif case in ("callback", "callback_error", "event"):
        entered, release = threading.Event(), threading.Event()
        def callback(result):
            entered.set()
            assert release.wait(3)
            if case == "callback_error": raise ValueError("callback failed")
        owner.submit_library(**_library_kwargs(**({"on_event":callback} if case == "event" else {"on_result":callback})))
        owner.maintenance_close_admission()
        event_thread = None
        if case == "event":
            event_thread = threading.Thread(target=lambda:executor.emit_event(1))
            event_thread.start()
            assert await asyncio.to_thread(entered.wait, 2)
        executor.succeed()
        assert await asyncio.to_thread(entered.wait, 2)
        assert not await owner.maintenance_drain(time.monotonic())
        try: owner.submit_library(**_library_kwargs(attempt_id="refused"))
        except ExecutorBusyError: pass
        else: raise AssertionError("accepted during maintenance")
        waiter = asyncio.create_task(owner.maintenance_drain(time.monotonic()+2))
        await asyncio.sleep(0)
        waiter.cancel()
        try: await waiter
        except asyncio.CancelledError: pass
        assert not owner.maintenance_ready
        release.set()
        if event_thread is not None: await asyncio.to_thread(event_thread.join, 2)
        assert await owner.maintenance_drain(time.monotonic()+2)
        owner.maintenance_resume()
        owner.submit_library(**_library_kwargs(attempt_id="resumed"))
        executor.succeed()
    elif case in ("capture", "retry"):
        texts = []
        capture = _begin(module, owner, callback=lambda sequence,text:texts.append(text))
        owner.maintenance_close_admission()
        assert not await owner.maintenance_drain(time.monotonic())
        try: _begin(module, owner, capture_generation=8)
        except ExecutorBusyError: pass
        else: raise AssertionError("opened another capture")
        capture.append_segment(b"\x01\x00")
        capture.finish()
        if case == "capture": executor.succeed()
        else: executor.fail(TranscriptionFailureCode.INFERENCE_FAILED, recovery_actions=("retry_faster_whisper",))
        assert await asyncio.to_thread(capture._capture.done.wait, 2)
        if case == "retry":
            assert not await owner.maintenance_drain(time.monotonic())
            retry = capture.take_retry_buffer()
            assert retry.source.audio == b"\x01\x00"
        else: assert texts == ["ok"]
        assert await owner.maintenance_drain(time.monotonic()+2)
        assert not executor.cancelled_attempts and not executor.force_stopped_attempts
    else:
        from tldw_chatbook.UI.Console_Modules.dictation import ConsoleDictationController, ConsoleDictationEvent
        published = []
        tail_entered, tail_release = asyncio.Event(), asyncio.Event()
        async def tail(origin):
            tail_entered.set()
            await tail_release.wait()
            published.append("tail")
        noop = lambda *a,**k:None
        screen = SimpleNamespace(is_mounted=True, post_message=lambda m:published.append(m) or True)
        owner = ConsoleDictationController(screen, app_instance=SimpleNamespace(), composer_accessor=noop,
            chat_store_accessor=noop, speak_status=noop, hands_free_session_accessor=noop,
            set_hands_free_vad_degraded=noop, enter_hands_free_loop=noop, hands_free_force_immediate_send=noop,
            deliver_hands_free_capture_ended=noop, realtime_session_accessor=noop,
            realtime_adopt_transcript=lambda t:False, run_pending_voice_action=tail,
            undo_histories_accessor=lambda:{}, visible_draft_session_id_accessor=noop)
        owner.maintenance_close_admission()
        owner._request_console_dictation_start()
        await owner._start_console_dictation()
        assert owner._console_dictation_state == "idle" and owner._console_dictation_session is None
        if case == "ui_orphan":
            from pytest import MonkeyPatch
            from Tests.Audio.test_dictation_lazy_transcription import _build_service, _FakeTranscriptionService
            from tldw_chatbook.UI.Console_Modules.dictation import ConsoleStreamingDictationSession
            patch = MonkeyPatch()
            service = _build_service(patch, _FakeTranscriptionService())
            service.stop_join_timeout_seconds = 0.01
            entered, release = threading.Event(), threading.Event()
            def inference():
                entered.set()
                assert release.wait(4)
                owner._emit_console_dictation_event(None, object())
            service._processing_loop = inference
            session = ConsoleStreamingDictationSession(on_event=owner._emit_console_dictation_event, service_factory=lambda **kw:service)
            def start(**kwargs):
                assert session._build_service().start_dictation()
            session.start = start
            owner._create_console_dictation_session=lambda:session
            owner._set_console_dictation_state=lambda state:setattr(owner,"_console_dictation_state",state)
            screen.set_timer = screen.set_interval = lambda *a,**kw:None
            owner.maintenance_resume()
            await owner._start_console_dictation()
            assert await asyncio.to_thread(entered.wait,2)
            owner.maintenance_close_admission()
            outcome = await asyncio.to_thread(service.stop_dictation)
            assert not outcome.transcription_complete
            await owner.teardown()
            assert owner._console_dictation_session is None and owner._console_dictation_state == "idle"
            assert not await owner.maintenance_drain(time.monotonic())
            release.set()
            await asyncio.to_thread(service.processing_thread.join,2)
            assert not owner.maintenance_ready
            owner._handle_console_dictation_event(published.pop())
            assert await owner.maintenance_drain(time.monotonic()+1)
            patch.undo()
        elif case == "ui_event":
            event = object()
            owner._emit_console_dictation_event(None, event)
            assert not await owner.maintenance_drain(time.monotonic())
            owner._handle_console_dictation_event(published.pop())
            assert await owner.maintenance_drain(time.monotonic()+1)
        else:
            entered, release = threading.Event(), threading.Event()
            def transcribe():
                entered.set()
                assert release.wait(3)
                return "words"
            session = SimpleNamespace(stop_and_transcribe=transcribe, retry_available=False, clear_retry=noop, discard=noop)
            owner._console_dictation_session=session
            owner._console_dictation_state="recording"
            assert not await owner.maintenance_drain(time.monotonic())
            owner._set_console_dictation_state=lambda state:setattr(owner,"_console_dictation_state",state)
            owner._insert_console_dictation=lambda **kwargs:published.append(kwargs["transcript"])
            running = asyncio.create_task(owner._stop_console_dictation(session))
            assert await asyncio.to_thread(entered.wait,2)
            assert not await owner.maintenance_drain(time.monotonic())
            if case == "ui_cancel":
                running.cancel()
                try: await running
                except asyncio.CancelledError: pass
                await owner.teardown()
                assert owner._console_dictation_state == "idle"
                assert not await owner.maintenance_drain(time.monotonic())
                release.set()
                assert await owner.maintenance_drain(time.monotonic()+2)
                assert published == []
                print("retired and reopened")
                return
            release.set()
            await asyncio.wait_for(tail_entered.wait(),2)
            assert owner._console_dictation_state == "idle"
            assert not await owner.maintenance_drain(time.monotonic())
            tail_release.set()
            await running
            assert await owner.maintenance_drain(time.monotonic()+1)
            assert published == ["words", "tail"]
            session.retry_available = True
            assert not await owner.maintenance_drain(time.monotonic())
            assert session.retry_available
            session.retry_available = False
        owner.maintenance_resume()
    print("retired and reopened")
asyncio.run(main())
"""


@pytest.mark.parametrize(
    "route",
    [
        "idle_callback",
        "callback",
        "callback_error",
        "event",
        "capture",
        "retry",
        "ui_event",
        "ui_stop",
        "ui_cancel",
        "ui_orphan",
    ],
)
def test_dictation_maintenance(tmp_path, route):
    _run(tmp_path, route, "maintenance", script=_SCRIPT)
