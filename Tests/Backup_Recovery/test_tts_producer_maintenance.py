"""Reversible handler and shared TTS resource settlement."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sys, time, threading
from pathlib import Path
from types import SimpleNamespace
from Tests.TTS.adapter_fakes import FakeAdapterFactory, provider_spec
from Tests.TTS.test_tts_registry_service import tts_request
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.TTS_Generation import TTSService
from tldw_chatbook.TTS.adapter_types import TTSRegistryClosedError
case = sys.argv[1]
async def main():
    if case in ("registry_ticket", "registry_transition"):
        factory = FakeAdapterFactory("openai")
        registry = TTSAdapterRegistry(specs=(provider_spec("openai",factory),), aliases={})
        lease = await registry.acquire("openai")
        adapter = lease.adapter
        await lease.release()
        if case == "registry_ticket":
            ticket = await registry.begin_reconfigure_provider("openai", {"version": 2})
            registry.maintenance_close_admission()
            # No yield between accepted ticket creation and the readiness probe.
            assert not ticket.completion.done()
            assert not await registry.maintenance_drain(time.monotonic())
            await ticket.completion
            assert adapter.close_calls == 1
            assert registry._slots["openai"].config == {"version": 2}
        else:
            lock = registry._slots["openai"].transition_lock
            await lock.acquire()
            lock_held_by_test = True
            entered, release = asyncio.Event(), asyncio.Event()
            async def draining(): entered.set(); await release.wait()
            async def action(): pass
            caller = asyncio.create_task(registry.run_exclusive_provider_transition(
                "openai", on_draining=draining, action=action, apply_staged=False))
            try:
                await asyncio.sleep(0)
                caller.cancel()
                await asyncio.gather(caller, return_exceptions=True)
                registry.maintenance_close_admission()
                assert registry._transition_tasks and not entered.is_set()
                assert not await registry.maintenance_drain(time.monotonic())
                lock.release()
                lock_held_by_test = False
                await asyncio.wait_for(entered.wait(), 1)
                assert not await registry.maintenance_drain(time.monotonic())
            finally:
                if lock_held_by_test: lock.release()
                release.set()
                results = await asyncio.gather(*registry._transition_tasks, return_exceptions=True)
                assert not any(isinstance(result, BaseException) for result in results)
            assert adapter.close_calls == 0
        assert await registry.maintenance_drain(time.monotonic()+1)
        registry.maintenance_resume()
        lease = await registry.acquire("openai")
        await lease.release()
    elif case in ("response", "preflight", "registry", "native_publication"):
        factory = FakeAdapterFactory("openai")
        registry = TTSAdapterRegistry(specs=(provider_spec("openai",factory),), aliases={})
        service = TTSService(registry)
        if case == "native_publication":
            from tldw_chatbook.TTS.TTS_Generation import TTSSettingsPersistenceOutcome
            from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
            entered, release=threading.Event(),threading.Event()
            path=Path.home()/"publication.txt"
            def persist():
                entered.set(); assert release.wait(4); path.write_text("committed")
                return TTSSettingsPersistenceOutcome(file_replaced=True,caches_reloaded=True,failure_phase=None)
            ticket=service.begin_preferences_publication(TTSPreferencesSnapshot.from_settings({}),{},persist)
            assert await asyncio.to_thread(entered.wait,2)
            service.maintenance_close_admission()
            ticket.completion.cancel()
            await asyncio.sleep(0.02)
            assert not await service.maintenance_drain(time.monotonic())
            release.set()
            await asyncio.gather(ticket.completion,return_exceptions=True)
            assert await service.maintenance_drain(time.monotonic()+2)
            assert path.read_text()=="committed"
            print("retired and reopened")
            return
        if case == "registry":
            owner = registry
            lease = await registry.acquire("openai")
            owner.maintenance_close_admission()
            assert not await owner.maintenance_drain(time.monotonic())
            try: await registry.acquire("openai")
            except TTSRegistryClosedError: pass
            else: raise AssertionError("registry admitted new lease")
            await lease.release()
        else:
            owner = service
            entered, release = asyncio.Event(), asyncio.Event()
            if case == "preflight":
                lease = await registry.acquire("openai")
                adapter = lease.adapter
                await lease.release()
                original = adapter.ensure_ready
                async def ready():
                    entered.set()
                    await release.wait()
                    await original()
                adapter.ensure_ready = ready
                task=asyncio.create_task(service.synthesize(tts_request()))
                await entered.wait()
                owner.maintenance_close_admission()
                assert not await owner.maintenance_drain(time.monotonic())
                release.set()
                response = await task
            else:
                response=await service.synthesize(tts_request())
                owner.maintenance_close_admission()
            try: await service.synthesize(tts_request())
            except TTSRegistryClosedError: pass
            else: raise AssertionError("service admitted new request")
            assert not await owner.maintenance_drain(time.monotonic())
            await response.aclose()
        assert await owner.maintenance_drain(time.monotonic()+2)
        owner.maintenance_resume()
        lease=await registry.acquire("openai")
        await lease.release()
    elif case == "legacy_playback":
        import tldw_chatbook.Event_Handlers.TTS_Events.tts_events as module
        import tldw_chatbook.TTS.audio_player as audio
        owner=module.TTSEventHandler()
        path=Path.home()/"clip.wav";path.write_bytes(b"audio")
        player=SimpleNamespace(state=audio.PlaybackState.PLAYING)
        player.get_current_file=lambda:path
        player.get_state=lambda:player.state
        audio.get_audio_player=lambda:player
        module.play_audio_file=lambda p:None
        owner._audio_files["m"]=path
        cleanup_entered,cleanup_release=asyncio.Event(),asyncio.Event()
        async def cleanup(*a,**kw):cleanup_entered.set();await cleanup_release.wait()
        owner._cleanup_audio_file=cleanup
        await owner.handle_tts_playback(module.TTSPlaybackEvent("play","m"))
        owner.maintenance_close_admission()
        await cleanup_entered.wait()
        assert not await owner.maintenance_drain(time.monotonic())
        player.state=audio.PlaybackState.FINISHED
        await asyncio.sleep(.08)
        assert not await owner.maintenance_drain(time.monotonic())
        cleanup_release.set()
        assert await owner.maintenance_drain(time.monotonic()+1)
        assert path.exists()
    elif case == "same_tick":
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler,TTSRequestEvent
        owner=TTSEventHandler()
        entered,release=asyncio.Event(),asyncio.Event()
        async def request(event): entered.set();await release.wait()
        owner.handle_tts_request=request
        owner.on_tts_request_event(TTSRequestEvent("words","m"))
        owner.maintenance_close_admission()
        assert not owner.maintenance_ready
        await entered.wait();release.set()
        assert await owner.maintenance_drain(time.monotonic()+1)
    elif case == "native":
        import tldw_chatbook.Event_Handlers.TTS_Events.tts_events as module
        owner=module.TTSEventHandler()
        entered, release=threading.Event(),threading.Event()
        def play(*a,**kw):
            entered.set(); assert release.wait(4); return True
        module._play_legacy_clip_and_await_completion=play
        module._TTS_IO_CANCELLATION_JOIN_TIMEOUT_SECONDS=0.01
        lifecycle=module.TTSPlaybackLifecycle(message_id="m",request_id=1,validator=lambda:True,callback=lambda state:None)
        task=asyncio.create_task(owner._run_owned_file_playback("m",Path.home()/"a.wav",lifecycle,threading.Event(),threading.Event()))
        owner._retain_active_task(task)
        assert await asyncio.to_thread(entered.wait,2)
        owner.maintenance_close_admission()
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        assert not await owner.maintenance_drain(time.monotonic())
        release.set()
        assert await owner.maintenance_drain(time.monotonic()+2)
    elif case == "settings":
        from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler
        owner=STTSEventHandler()
        entered, release, observed, final=asyncio.Event(),asyncio.Event(),asyncio.Event(),asyncio.Event()
        results=[]
        async def persist(event):
            entered.set(); await release.wait()
            async def observer():
                observed.set(); await final.wait(); results.append("published")
            owner._start_event_task(observer())
        owner._persist_settings=persist
        event=SimpleNamespace(_abandon_publication_lease=lambda:results.append("lease released"))
        task=asyncio.create_task(owner.handle_settings_save(event))
        await entered.wait();owner.maintenance_close_admission();release.set();await task
        await asyncio.wait_for(observed.wait(),1)
        assert not await owner.maintenance_drain(time.monotonic())
        final.set();assert await owner.maintenance_drain(time.monotonic()+1)
        assert results==["lease released","published"]
    elif case.startswith("tts"):
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler, TTSRequestEvent, TTSGlobalOverrideDecisionEvent
        owner=TTSEventHandler()
        outcomes=[]
        entered, release=asyncio.Event(),asyncio.Event()
        async def prepare(*a,**kw):
            entered.set(); await release.wait(); return "words"
        async def generate(*a,**kw): outcomes.append("published")
        owner._prepare_tts_text=prepare
        owner._generate_tts=generate
        if case=="tts_refusal":
            owner.maintenance_close_admission()
            owner._pending_global_overrides["a"*32]=object()
            await owner.speak_utterance("words",on_finished=outcomes.append)
            await owner.handle_tts_global_override_decision(TTSGlobalOverrideDecisionEvent("a"*32,True))
            assert outcomes==[False] and not entered.is_set()
            assert "a"*32 in owner._pending_global_overrides
        else:
            task=asyncio.create_task(owner.speak_utterance("words",on_finished=outcomes.append))
            await entered.wait()
            owner.maintenance_close_admission()
            assert not await owner.maintenance_drain(time.monotonic())
            release.set(); await task
            assert "published" in outcomes
        assert await owner.maintenance_drain(time.monotonic()+2)
        owner.maintenance_resume()
    else:
        from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler
        owner=STTSEventHandler()
        path=Path.home()/"held.wav";path.write_bytes(b"audio")
        owner._playground_audio_files.add(path)
        owner._playground_operation_files["operation"]={path}
        owner._current_playground_artifact=SimpleNamespace(operation_id="operation",path=path)
        assert owner.lease_playground_result("operation",path)
        owner.maintenance_close_admission()
        assert not owner.lease_playground_result("operation",path)
        assert not await owner.maintenance_drain(time.monotonic())
        owner._current_audio_file=path
        owner.release_playground_result("operation",path)
        assert await owner.maintenance_drain(time.monotonic()+1)
        assert path.exists()
        owner.maintenance_resume()
    print("retired and reopened")
asyncio.run(main())
"""


@pytest.mark.parametrize(
    "case",
    [
        "response",
        "preflight",
        "registry",
        "tts_refusal",
        "tts_admitted",
        "stts_lease",
        "native",
        "settings",
        "same_tick",
        "native_publication",
        "legacy_playback",
        "registry_ticket",
        "registry_transition",
    ],
)
def test_tts_producer_maintenance(tmp_path, case):
    _run(tmp_path, case, "success", script=_SCRIPT)
