"""Buddy speech reuses TTS identity, destination admission and exact playback owner."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

DESTINATION = "sha256:" + "a" * 64


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
