from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    AudioJobResponse,
    AudioSpeechJobArtifactsResponse,
    AudioSpeechJobCreateResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioTranslationRequest,
    OpenAISpeechRequest,
    ReadingExportResponse,
    SubmitAudioJobRequest,
    TLDWAPIClient,
    TTSHealthResponse,
    TTSHistoryDetailResponse,
    TTSHistoryFavoriteUpdate,
    TTSHistoryListResponse,
    TTSProvidersResponse,
    TTSVoicesResponse,
)


def _job() -> dict:
    return {
        "id": 9,
        "uuid": "job-uuid",
        "job_type": "audio_download",
        "status": "queued",
        "priority": 5,
        "retry_count": 0,
        "max_retries": 3,
        "owner_user_id": "user-1",
        "available_at": None,
        "started_at": None,
        "leased_until": None,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
        "completed_at": None,
    }


def _history_item() -> dict:
    return {
        "id": 3,
        "created_at": "2026-04-25T12:00:00Z",
        "has_text": True,
        "text_preview": "Hello",
        "provider": "kokoro",
        "model": "kokoro",
        "voice_id": "af_heart",
        "voice_name": "Heart",
        "voice_info": {"lang": "en"},
        "duration_ms": 1200,
        "format": "mp3",
        "status": "completed",
        "favorite": False,
        "job_id": 9,
        "output_id": 44,
        "artifact_deleted_at": None,
    }


@pytest.mark.asyncio
async def test_audio_routes_wire_speech_jobs_history_and_transcription(monkeypatch, tmp_path):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "status": "healthy",
                "providers": {"total": 1, "available": 1, "details": {"kokoro": {"status": "enabled"}}},
                "capabilities": {"kokoro": {"formats": ["mp3"]}},
                "timestamp": "2026-04-25T12:00:00Z",
            },
            {
                "status": "healthy",
                "model": "whisper-1",
                "provider": "whisper",
                "warm": False,
                "timestamp": "2026-04-25T12:00:00Z",
            },
            {"providers": {"kokoro": {"available": True}}, "voices": {"kokoro": [{"id": "af_heart"}]}, "timestamp": "2026-04-25T12:00:00Z"},
            {"kokoro": [{"id": "af_heart", "name": "Heart"}]},
            {"job_id": 12, "status": "queued"},
            {
                "job_id": 12,
                "artifacts": [
                    {
                        "output_id": 44,
                        "format": "mp3",
                        "type": "tts_audio",
                        "title": "Speech",
                        "download_url": "/api/v1/outputs/44/download",
                        "metadata": {"voice": "af_heart"},
                    }
                ],
            },
            {"id": 9, "uuid": "job-uuid", "domain": "audio", "queue": "default", "job_type": "audio_download", "status": "queued"},
            _job(),
            {"items": [_history_item()], "total": 1, "limit": 10, "offset": 0, "next_cursor": None},
            {**_history_item(), "text": "Hello world", "text_length": 11, "generation_time_ms": 300, "params_json": {}, "segments_json": None, "artifact_ids": [44], "error_message": None},
            {"id": 3, "favorite": True},
            {},
            {"text": "Transcript", "language": "en", "duration": 1.5, "words": None, "segments": []},
            {"text": "Translated", "language": "en", "duration": 1.5, "words": None, "segments": []},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)
    audio_payload = ReadingExportResponse(
        content=b"mp3-bytes",
        content_type="audio/mpeg",
        content_disposition='attachment; filename="speech.mp3"',
        filename="speech.mp3",
    )
    binary = AsyncMock(return_value=audio_payload)
    monkeypatch.setattr(client, "_binary_request", binary)

    async def fake_sse(method, endpoint, params=None, headers=None):
        yield {"type": "snapshot", "status": "queued"}
        yield {"type": "complete", "status": "completed"}

    monkeypatch.setattr(client, "_sse_request", fake_sse)

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF")

    health = await client.get_tts_health()
    stt_health = await client.get_stt_health(model="whisper-1", warm=False)
    providers = await client.list_tts_providers()
    voices = await client.list_tts_voices(provider="kokoro")
    speech = await client.create_audio_speech(
        OpenAISpeechRequest(model="kokoro", input="Hello", voice="af_heart", stream=False)
    )
    speech_job = await client.create_audio_speech_job(
        OpenAISpeechRequest(model="kokoro", input="Long text", voice="af_heart", stream=False)
    )
    speech_artifacts = await client.list_audio_speech_job_artifacts(12)
    submitted = await client.submit_audio_job(
        SubmitAudioJobRequest(url="https://example.com/audio.mp3", perform_analysis=True)
    )
    job = await client.get_audio_job(9)
    events = [event async for event in client.stream_audio_job_progress(9, after_id=7)]
    history = await client.list_tts_history(favorite=True, provider="kokoro", include_total=True, limit=10)
    detail = await client.get_tts_history_entry(3)
    favorite = await client.update_tts_history_favorite(3, TTSHistoryFavoriteUpdate(favorite=True))
    delete_response = await client.delete_tts_history_entry(3)
    transcription = await client.create_audio_transcription(
        str(audio_file),
        AudioTranscriptionRequest(model="whisper-1", language="en", response_format="json"),
    )
    translation = await client.create_audio_translation(
        str(audio_file),
        AudioTranslationRequest(model="whisper-1", response_format="json"),
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/audio/health")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/audio/transcriptions/health")
    assert mocked.await_args_list[1].kwargs["params"] == {"model": "whisper-1", "warm": "false"}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/audio/providers")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/audio/voices/catalog")
    assert mocked.await_args_list[3].kwargs["params"] == {"provider": "kokoro"}
    assert binary.await_args.args[:2] == ("POST", "/api/v1/audio/speech")
    assert binary.await_args.kwargs["json_data"]["model"] == "kokoro"
    assert binary.await_args.kwargs["json_data"]["stream"] is False
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/audio/speech/jobs")
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/audio/speech/jobs/12/artifacts")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/audio/jobs/submit")
    assert mocked.await_args_list[6].kwargs["json_data"]["perform_analysis"] is True
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/audio/jobs/9")
    assert mocked.await_args_list[8].args[:2] == ("GET", "/api/v1/audio/history")
    assert mocked.await_args_list[8].kwargs["params"] == {
        "favorite": "true",
        "provider": "kokoro",
        "limit": 10,
        "offset": 0,
        "include_total": "true",
    }
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/audio/history/3")
    assert mocked.await_args_list[10].args[:2] == ("PATCH", "/api/v1/audio/history/3")
    assert mocked.await_args_list[10].kwargs["json_data"] == {"favorite": True}
    assert mocked.await_args_list[11].args[:2] == ("DELETE", "/api/v1/audio/history/3")
    assert mocked.await_args_list[12].args[:2] == ("POST", "/api/v1/audio/transcriptions")
    assert mocked.await_args_list[12].kwargs["data"] == {
        "model": "whisper-1",
        "language": "en",
        "response_format": "json",
        "temperature": "0.0",
        "timestamp_granularities": ["segment"],
    }
    assert mocked.await_args_list[12].kwargs["files"][0][0] == "file"
    assert mocked.await_args_list[13].args[:2] == ("POST", "/api/v1/audio/translations")
    assert mocked.await_args_list[13].kwargs["data"] == {
        "model": "whisper-1",
        "response_format": "json",
        "temperature": "0.0",
    }
    assert isinstance(health, TTSHealthResponse)
    assert stt_health.status == "healthy"
    assert isinstance(providers, TTSProvidersResponse)
    assert isinstance(voices, TTSVoicesResponse)
    assert speech.content == b"mp3-bytes"
    assert isinstance(speech_job, AudioSpeechJobCreateResponse)
    assert isinstance(speech_artifacts, AudioSpeechJobArtifactsResponse)
    assert submitted.id == 9
    assert isinstance(job, AudioJobResponse)
    assert events[-1]["type"] == "complete"
    assert isinstance(history, TTSHistoryListResponse)
    assert isinstance(detail, TTSHistoryDetailResponse)
    assert favorite["favorite"] is True
    assert delete_response == {}
    assert isinstance(transcription, AudioTranscriptionResponse)
    assert isinstance(translation, AudioTranscriptionResponse)
    assert translation.text == "Translated"
