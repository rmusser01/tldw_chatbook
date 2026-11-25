"""Verify native speech owner receipts and strict route selection."""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_server_audio_retries_a_known_job_and_reads_one_owned_native_artifact():
    from Tests.Research_Workspace.test_server_studio import SERVER, server_request
    from dataclasses import replace
    from tldw_chatbook.Research_Workspace.server_studio_audio import ServerStudioAudio
    from tldw_chatbook.Research_Workspace.studio_generation import StudioGeneratedOutput
    from tldw_chatbook.Research_Workspace.studio_models import (
        ResearchGenerationConfig,
        StudioSpeechChoice,
    )

    selected = StudioSpeechChoice("gateway:speaker", "reader", "voice-a", "mp3", True)

    class Speech:
        def __init__(self):
            self.submissions = []
            self.polls = 0

        async def get_tts_health(self, **kwargs):
            return {"status": "healthy"}

        async def list_tts_providers(self, **kwargs):
            return {
                "supports_explicit_backend": True,
                "providers": {
                    "gateway:speaker": {
                        "discovery": {"stale": False, "status": "fresh"},
                        "default_model": "reader",
                        "model_capabilities": {
                            "reader": {
                                "formats": ["wav", "mp3"],
                                "voices": ["voice-a"],
                                "supports_speed": True,
                            }
                        },
                    }
                },
            }

        async def list_audio_speech_job_artifacts(self, *, job_id, **kwargs):
            return {
                "job_id": job_id,
                "artifacts": []
                if job_id == 0
                else [
                    {
                        "type": "tts_audio",
                        "output_id": 51,
                        "format": "mp3",
                        "download_url": "/api/v1/outputs/51/download",
                        "metadata": {
                            "provider": "gateway:speaker",
                            "model": "reader",
                            "voice": "voice-a",
                        },
                    }
                ],
            }

        async def create_audio_speech_job(self, *, request_data, **kwargs):
            self.submissions.append(request_data)
            return {"job_id": 17, "status": "queued"}

        async def get_audio_job(self, *, job_id, **kwargs):
            self.polls += 1
            if self.polls == 1:
                raise ConnectionError("temporary failed read")
            return {
                "id": job_id,
                "status": "completed",
                "job_type": "tts_longform",
                "uuid": "uuid",
            }

    speech = Speech()
    owner = ServerStudioAudio(speech)
    assert selected in await owner.choices()
    request = replace(
        server_request(),
        output_kind="audio_summary",
        generation=ResearchGenerationConfig(speech=selected, speech_speed=1.1),
    )
    script = StudioGeneratedOutput(
        "complete", "The source describes a native plant. [S1]"
    )
    with pytest.raises(ConnectionError):
        await owner.prepare(request, script, asyncio.Event())
    result = await owner.prepare(request, script, asyncio.Event())
    assert result.native_owner == ("17:51", 1)
    [submission] = speech.submissions
    assert (submission.backend, submission.model, submission.voice) == (
        "gateway:speaker",
        "reader",
        "voice-a",
    )
    assert submission.allow_fallback is False
    assert "[S1]" not in submission.input
    assert (await owner.detail(SERVER, "17:51"))["output_id"] == 51
