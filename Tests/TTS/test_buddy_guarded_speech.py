"""Buddy speech reuses TTS identity, destination admission and exact playback owner."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSEventHandler,
    TTSPlaybackEvent,
    TTSPlaybackLifecycle,
)
from tldw_chatbook.Persona_Buddy.speech import BuddySpeechItem, BuddySpeechQueue

DESTINATION = "sha256:" + "a" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_owner", [False, True])
@pytest.mark.parametrize("cancel_replacement", [False, True])
async def test_superseded_generation_settles_buddy_queue_without_stopping_replacement(
    monkeypatch,
    stale_owner,
    cancel_replacement,
):
    handler = TTSEventHandler()
    synthesis_entered = asyncio.Queue()
    next_selected, release_next = asyncio.Event(), asyncio.Event()
    utterance_results, replacement_states, replacement_outcomes = [], [], []
    old_is_current = [True]
    cleanup_entered, cleanup_cancelled, release_cleanup, cleanup_finished = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    replacement_admission = None

    async def resolve(**_kwargs):
        return SimpleNamespace(source="global")

    async def prepare(text, *_args, **_kwargs):
        return text

    async def destination(_resolution):
        return SimpleNamespace(fingerprint=DESTINATION)

    async def post(_event):
        return True

    async def synthesize(*, text, **_kwargs):
        # Hold the provider before its first response. Admission, task ownership,
        # cancellation/join, lifecycle transitions and exact Stop stay real.
        await synthesis_entered.put(text)
        try:
            await asyncio.Event().wait()
        finally:
            if cancel_replacement and text == "Research. Old reply":
                cleanup_entered.set()
                while not release_cleanup.is_set():
                    try:
                        await release_cleanup.wait()
                    except asyncio.CancelledError:
                        # Owned provider cleanup must finish even when its
                        # cancellation waiter is cancelled in turn.
                        cleanup_cancelled.set()
                cleanup_finished.set()

    monkeypatch.setattr(handler, "_resolve_speech_request_identity", resolve)
    monkeypatch.setattr(handler, "_prepare_tts_text", prepare)
    monkeypatch.setattr(handler, "_destination_for_resolution", destination)
    monkeypatch.setattr(handler, "_post_tts_message", post)
    handler._tts_service = SimpleNamespace(
        synthesize_default=synthesize,
        preferences_snapshot=lambda: SimpleNamespace(provider_id="kokoro", speed=1),
    )

    async def play(item, current):
        if item.key == "next":
            # Observe queue continuation before its next admission legitimately
            # replaces the manual request whose ownership we are checking.
            next_selected.set()
            await release_next.wait()
        result = await handler.speak_guarded_utterance(
            item.named_text,
            assistant_kind="generic",
            character_ref=None,
            expected_destination_fingerprint=DESTINATION,
            validator=current,
        )
        utterance_results.append((item.key, result))
        return result

    queue = BuddySpeechQueue(play)
    replacement = TTSPlaybackLifecycle(
        message_id=f"manual-replacement-{stale_owner}-{cancel_replacement}",
        request_id=1,
        validator=lambda: True,
        callback=replacement_states.append,
    )
    try:
        assert queue.enqueue(
            BuddySpeechItem(
                "old",
                "Research",
                "Old reply",
                is_current=lambda: old_is_current[0],
            )
        )
        assert queue.enqueue(BuddySpeechItem("next", "Research", "Next reply"))
        assert (
            await asyncio.wait_for(synthesis_entered.get(), 2) == "Research. Old reply"
        )
        old = handler._console_generation_owner
        assert old is not None and old.lifecycle.state == "generating"
        old_is_current[0] = not stale_owner
        replacement_admission = asyncio.create_task(
            handler._admit_tts_generation(
                text="Manual speech",
                message_id=replacement.message_id,
                voice=None,
                resolution=SimpleNamespace(source="global"),
                playback_lifecycle=replacement,
                outcome_callback=replacement_outcomes.append,
            )
        )
        if cancel_replacement:
            await asyncio.wait_for(cleanup_entered.wait(), 2)
            replacement_admission.cancel()
            await asyncio.wait_for(cleanup_cancelled.wait(), 2)
            assert not old.task.done() and not replacement_admission.done()
            assert old.lifecycle.state == "generating"
            assert utterance_results == [] and not next_selected.is_set()
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(replacement_admission, 2)
            assert cleanup_finished.is_set()
            assert synthesis_entered.empty()
        else:
            await asyncio.wait_for(replacement_admission, 2)
            assert await asyncio.wait_for(synthesis_entered.get(), 2) == "Manual speech"
        assert old.task.cancelled()
        await asyncio.wait_for(next_selected.wait(), 2)
        assert utterance_results == [("old", False)]
        assert old.lifecycle.state == "stopped"
        manual = handler._console_generation_owner
        if cancel_replacement:
            assert manual is None
        else:
            assert manual is not None and manual.lifecycle is replacement
            assert not manual.task.done()
        assert replacement.state == "generating"
        assert replacement_outcomes == []

        # A late old-owner Stop cannot cancel the manual request. Repeated exact
        # Stops settle only once, retaining successful supersession bookkeeping.
        await handler.handle_tts_playback(
            TTSPlaybackEvent(
                "stop",
                old.lifecycle.message_id,
                playback_lifecycle=old.lifecycle,
            )
        )
        assert handler._console_generation_owner is manual
        if cancel_replacement:
            assert replacement_states == [] and replacement_outcomes == []
        else:
            assert not manual.task.done()
            stop = TTSPlaybackEvent(
                "stop",
                replacement.message_id,
                playback_lifecycle=replacement,
            )
            await handler.handle_tts_playback(stop)
            await handler.handle_tts_playback(stop)
            assert manual.task.cancelled()
            assert replacement_states == ["stopped"]
            assert replacement_outcomes == [True]

        release_next.set()
        assert (
            await asyncio.wait_for(synthesis_entered.get(), 2) == "Research. Next reply"
        )
        next_owner = handler._console_generation_owner
        assert next_owner is not None and next_owner is not old
        await handler.handle_tts_playback(
            TTSPlaybackEvent(
                "stop",
                next_owner.lifecycle.message_id,
                playback_lifecycle=next_owner.lifecycle,
            )
        )
        await asyncio.wait_for(queue.wait_idle(), 2)
        assert utterance_results == [("old", False), ("next", False)]
        assert handler._console_generation_owner is None
        assert queue.state.queued == 0 and queue.state.current_title == ""
        assert not any(not task.done() for task in handler._active_tasks)
    finally:
        release_cleanup.set()
        release_next.set()
        if replacement_admission is not None:
            replacement_admission.cancel()
            await asyncio.gather(replacement_admission, return_exceptions=True)
        await asyncio.wait_for(queue.aclose(), 2)
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate", [None, "resolution", "preparation"])
async def test_guarded_utterance_revalidates_before_admission_and_waits_playback(
    invalidate,
):
    handler = TTSEventHandler()
    valid, admitted, stopped = [True], [], []

    async def resolve(**kwargs):
        if invalidate == "resolution":
            valid[0] = False
        return SimpleNamespace(source="global")

    async def prepare(text, *_args, **_kwargs):
        if invalidate == "preparation":
            valid[0] = False
        return text

    async def admit(**kwargs):
        admitted.append(kwargs)
        kwargs["outcome_callback"](
            True
        )  # Generation success is not playback completion.

    async def stop(event):
        stopped.append(event)

    handler._resolve_speech_request_identity = resolve
    handler._prepare_tts_text = prepare
    handler._admit_tts_generation = admit
    handler.handle_tts_playback = stop
    task = asyncio.create_task(
        handler.speak_guarded_utterance(
            "Research. Result",
            assistant_kind="generic",
            character_ref=None,
            expected_destination_fingerprint=DESTINATION,
            validator=lambda: valid[0],
        )
    )
    await asyncio.sleep(0)
    if invalidate:
        assert await task is False
        assert admitted == []
    else:
        assert not task.done()
        assert admitted[0]["expected_destination_fingerprint"] == DESTINATION
        owner = admitted[0]["playback_lifecycle"]
        owner.report("playing")
        owner.report_terminal("stopped")
        assert await task is True
        assert stopped[-1].playback_lifecycle is owner


@pytest.mark.asyncio
async def test_guarded_utterance_cancel_stops_only_its_exact_owner():
    handler = TTSEventHandler()
    admitted, stops = [], []

    async def resolve(**kwargs):
        return SimpleNamespace(source="global")

    async def prepare(text, *_args, **_kwargs):
        return text

    async def admit(**kwargs):
        admitted.append(kwargs)

    async def stop(event):
        stops.append(event)

    handler._resolve_speech_request_identity = resolve
    handler._prepare_tts_text = prepare
    handler._admit_tts_generation = admit
    handler.handle_tts_playback = stop
    task = asyncio.create_task(
        handler.speak_guarded_utterance(
            "Named question",
            assistant_kind="generic",
            character_ref=None,
            expected_destination_fingerprint=DESTINATION,
            validator=lambda: True,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(stops) == 1 and stops[0].message_id.startswith("buddy-")
    assert stops[0].playback_lifecycle is admitted[0]["playback_lifecycle"]


@pytest.mark.asyncio
async def test_provider_admission_rejects_lifecycle_revoked_after_progress_await():
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
        TTSPlaybackLifecycle,
        TTSProgressEvent,
    )

    handler = TTSEventHandler()
    valid, denied = [True], []

    async def post(event):
        if isinstance(event, TTSProgressEvent):
            valid[0] = False

    async def synthesize(**kwargs):
        denied.append(
            not kwargs["admission_authorizer"]("openai", "https://api.openai.com/v1")
        )
        raise ValueError("Test ends at provider admission")

    handler._post_tts_message = post
    handler._tts_service = SimpleNamespace(
        synthesize_default=synthesize,
        preferences_snapshot=lambda: SimpleNamespace(provider_id="openai"),
    )
    owner = TTSPlaybackLifecycle(
        message_id="buddy-test",
        request_id=1,
        validator=lambda: valid[0],
        callback=lambda _state: None,
    )
    await handler._generate_tts(
        "Named response", "buddy-test", None, playback_lifecycle=owner
    )
    assert denied == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_closing", [False, True])
async def test_app_shutdown_joins_buddy_before_shared_tts_owners(cancel_closing):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Persona_Buddy.speech import BuddySpeechItem, BuddySpeechQueue

    entered, cleanup_entered, release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    events = []

    async def play(_item, _current):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await release.wait()
            events.append("buddy-stopped")

    async def profiles():
        events.append("profiles")

    async def service():
        events.append("service")

    queue = BuddySpeechQueue(play)
    queue.enqueue(BuddySpeechItem("reply", "Name", "Response"))
    await entered.wait()
    owner = SimpleNamespace(
        buddy_speech_coordinator=queue,
        _close_tts_profile_repository=profiles,
        _close_tts_service=service,
    )
    closing = asyncio.create_task(TldwCli._close_owned_tts_resources(owner))
    try:
        await cleanup_entered.wait()
        if cancel_closing:
            closing.cancel("shutdown cancelled")
            await asyncio.sleep(0)
        assert not closing.done() and events == []
        release.set()
        if cancel_closing:
            with pytest.raises(asyncio.CancelledError, match="shutdown cancelled"):
                await closing
        else:
            await closing
    finally:
        release.set()
        await asyncio.gather(closing, return_exceptions=True)
        await queue.wait_idle()
    assert events == ["buddy-stopped", "profiles", "service"]
