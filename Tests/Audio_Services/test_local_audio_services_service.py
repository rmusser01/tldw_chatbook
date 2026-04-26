from __future__ import annotations

import pytest

from tldw_chatbook.Audio_Services_Interop.local_audio_services_service import LocalAudioServicesService


@pytest.mark.asyncio
async def test_local_audio_service_generates_speech_and_persists_history(tmp_path):
    calls = []

    async def fake_generator(**kwargs):
        calls.append(kwargs)
        return b"local-audio"

    service = LocalAudioServicesService(
        tts_audio_generator=fake_generator,
        history_store_path=tmp_path / "audio-history.json",
    )

    speech = await service.create_audio_speech(
        {
            "model": "kokoro",
            "input": "Hello from local Chatbook.",
            "voice": "af_heart",
            "response_format": "wav",
            "speed": 1.25,
            "stream": False,
        }
    )

    assert speech["content"] == b"local-audio"
    assert speech["content_type"] == "audio/wav"
    assert speech["filename"] == "local_speech_1.wav"
    assert speech["history_id"] == 1
    assert calls == [
        {
            "text": "Hello from local Chatbook.",
            "model": "kokoro",
            "voice": "af_heart",
            "response_format": "wav",
            "stream": False,
            "speed": 1.25,
        }
    ]

    history = await service.list_tts_history(limit=10, offset=0)
    assert history["total"] == 1
    assert history["items"] == [
        {
            "id": 1,
            "created_at": history["items"][0]["created_at"],
            "provider": "kokoro",
            "model": "kokoro",
            "voice": "af_heart",
            "response_format": "wav",
            "filename": "local_speech_1.wav",
            "content_type": "audio/wav",
            "size_bytes": len(b"local-audio"),
            "input_characters": len("Hello from local Chatbook."),
            "text_preview": "Hello from local Chatbook.",
            "favorite": False,
            "has_text": True,
        }
    ]

    detail = await service.get_tts_history_entry(1)
    assert detail["text"] == "Hello from local Chatbook."
    assert detail["content"] == b"local-audio"

    reloaded = LocalAudioServicesService(history_store_path=tmp_path / "audio-history.json")
    reloaded_detail = await reloaded.get_tts_history_entry(1)
    assert reloaded_detail["content"] == b"local-audio"
    assert reloaded_detail["text"] == "Hello from local Chatbook."


@pytest.mark.asyncio
async def test_local_audio_service_updates_and_deletes_history_entries(tmp_path):
    async def fake_generator(**kwargs):
        return b"audio"

    service = LocalAudioServicesService(
        tts_audio_generator=fake_generator,
        history_store_path=tmp_path / "audio-history.json",
    )
    await service.create_audio_speech({"model": "kokoro", "input": "One", "response_format": "mp3"})

    updated = await service.update_tts_history_favorite(1, {"favorite": True})
    assert updated["favorite"] is True

    deleted = await service.delete_tts_history_entry(1)
    assert deleted == {"id": 1, "deleted": True}
    assert await service.list_tts_history() == {"items": [], "limit": 50, "offset": 0, "total": 0}


@pytest.mark.asyncio
async def test_local_audio_service_rejects_empty_text_before_generation():
    async def fake_generator(**kwargs):
        raise AssertionError("generator should not be called")

    service = LocalAudioServicesService(tts_audio_generator=fake_generator)

    with pytest.raises(ValueError, match="local_tts_input_required"):
        await service.create_audio_speech({"model": "kokoro", "input": "   "})
