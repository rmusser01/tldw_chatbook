from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    AlignmentPayload,
    AlignmentWord,
    AudiobookArtifactsResponse,
    AudiobookChapterListResponse,
    AudiobookJobCreateResponse,
    AudiobookJobRequest,
    AudiobookJobStatusResponse,
    AudiobookParseRequest,
    AudiobookParseResponse,
    AudiobookProjectListResponse,
    AudiobookProjectResponse,
    ChapterSelection,
    OutputOptions,
    QueueOptions,
    ReadingExportResponse,
    SourceRef,
    SubtitleExportRequest,
    SubtitleOptions,
    TLDWAPIClient,
    VoiceProfileCreateRequest,
    VoiceProfileDeleteResponse,
    VoiceProfileListResponse,
    VoiceProfileResponse,
)


@pytest.mark.asyncio
async def test_audiobook_routes_wire_jobs_projects_profiles_and_subtitles(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "project_id": "abk_1",
                "normalized_text": "Chapter 1\nHello",
                "chapters": [
                    {
                        "chapter_id": "ch_001",
                        "title": "Chapter 1",
                        "start_offset": 0,
                        "end_offset": 15,
                        "word_count": 2,
                    }
                ],
                "metadata": {"source_type": "txt"},
            },
            {"job_id": 77, "project_id": "abk_1", "status": "queued"},
            {
                "job_id": 77,
                "project_id": "abk_1",
                "status": "processing",
                "progress": {"stage": "audiobook_tts", "percent": 50},
                "errors": [],
            },
            {
                "project_id": "abk_1",
                "artifacts": [
                    {
                        "artifact_type": "audio",
                        "format": "mp3",
                        "scope": "chapter",
                        "chapter_id": "ch_001",
                        "output_id": 501,
                        "download_url": "/api/v1/outputs/501/download",
                    }
                ],
            },
            {
                "projects": [
                    {
                        "project_db_id": 12,
                        "project_id": "abk_1",
                        "title": "Book",
                        "status": "completed",
                        "source_ref": {"input_type": "txt"},
                        "settings": {},
                        "created_at": "2026-04-25T12:00:00Z",
                        "updated_at": "2026-04-25T12:01:00Z",
                    }
                ]
            },
            {
                "project": {
                    "project_db_id": 12,
                    "project_id": "abk_1",
                    "title": "Book",
                    "status": "completed",
                    "source_ref": {"input_type": "txt"},
                    "settings": {},
                    "created_at": "2026-04-25T12:00:00Z",
                    "updated_at": "2026-04-25T12:01:00Z",
                }
            },
            {
                "project_id": "abk_1",
                "chapters": [
                    {
                        "id": 1,
                        "chapter_index": 0,
                        "title": "Chapter 1",
                        "start_offset": 0,
                        "end_offset": 15,
                        "voice_profile_id": None,
                        "speed": 1.0,
                        "metadata": {"chapter_id": "ch_001"},
                    }
                ],
            },
            {
                "project_id": "abk_1",
                "artifacts": [
                    {
                        "artifact_type": "subtitle",
                        "format": "srt",
                        "scope": "chapter",
                        "chapter_id": "ch_001",
                        "output_id": 502,
                        "download_url": "/api/v1/outputs/502/download",
                    }
                ],
            },
            {
                "profile_id": "vp_1",
                "name": "Narrator",
                "default_voice": "af_heart",
                "default_speed": 1.0,
                "chapter_overrides": [],
            },
            {
                "profiles": [
                    {
                        "profile_id": "vp_1",
                        "name": "Narrator",
                        "default_voice": "af_heart",
                        "default_speed": 1.0,
                        "chapter_overrides": [],
                    }
                ]
            },
            {"profile_id": "vp_1", "deleted": True},
        ]
    )
    binary = AsyncMock(
        return_value=ReadingExportResponse(
            content=b"1\n00:00:00,000 --> 00:00:00,500\nHello\n",
            content_type="text/plain; charset=utf-8",
            content_disposition=None,
            filename=None,
        )
    )
    monkeypatch.setattr(client, "_request", mocked)
    monkeypatch.setattr(client, "_binary_request", binary)

    source = SourceRef(input_type="txt", raw_text="Chapter 1\nHello")
    parsed = await client.parse_audiobook_source(AudiobookParseRequest(source=source, detect_chapters=True))
    job = await client.create_audiobook_job(
        AudiobookJobRequest(
            project_title="Book",
            source=source,
            chapters=[ChapterSelection(chapter_id="ch_001", include=True)],
            output=OutputOptions(merge=True, per_chapter=True, formats=["mp3"]),
            subtitles=SubtitleOptions(formats=["srt"], mode="sentence", variant="wide"),
            queue=QueueOptions(priority=6, batch_group="batch-1"),
        )
    )
    status = await client.get_audiobook_job_status(77)
    job_artifacts = await client.list_audiobook_job_artifacts(77)
    projects = await client.list_audiobook_projects(limit=10, offset=5)
    project = await client.get_audiobook_project("abk_1")
    chapters = await client.list_audiobook_project_chapters("abk_1", limit=25, offset=0)
    project_artifacts = await client.list_audiobook_project_artifacts("abk_1", limit=25, offset=0)
    profile = await client.create_audiobook_voice_profile(
        VoiceProfileCreateRequest(name="Narrator", default_voice="af_heart", default_speed=1.0)
    )
    profiles = await client.list_audiobook_voice_profiles()
    deleted = await client.delete_audiobook_voice_profile("vp_1")
    subtitles = await client.export_audiobook_subtitles(
        SubtitleExportRequest(
            format="srt",
            mode="sentence",
            variant="wide",
            alignment=AlignmentPayload(
                engine="kokoro",
                sample_rate=24000,
                words=[AlignmentWord(word="Hello", start_ms=0, end_ms=500)],
            ),
        )
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/audiobooks/parse")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/audiobooks/jobs")
    assert mocked.await_args_list[1].kwargs["json_data"]["queue"] == {"priority": 6, "batch_group": "batch-1"}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/audiobooks/jobs/77")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/audiobooks/jobs/77/artifacts")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/audiobooks/projects")
    assert mocked.await_args_list[4].kwargs["params"] == {"limit": 10, "offset": 5}
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/audiobooks/projects/abk_1")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/audiobooks/projects/abk_1/chapters")
    assert mocked.await_args_list[6].kwargs["params"] == {"limit": 25, "offset": 0}
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/audiobooks/projects/abk_1/artifacts")
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/audiobooks/voices/profiles")
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/audiobooks/voices/profiles")
    assert mocked.await_args_list[10].args[:2] == ("DELETE", "/api/v1/audiobooks/voices/profiles/vp_1")
    assert binary.await_args_list[0].args[:2] == ("POST", "/api/v1/audiobooks/subtitles")
    assert isinstance(parsed, AudiobookParseResponse)
    assert parsed.chapters[0].chapter_id == "ch_001"
    assert isinstance(job, AudiobookJobCreateResponse)
    assert isinstance(status, AudiobookJobStatusResponse)
    assert isinstance(job_artifacts, AudiobookArtifactsResponse)
    assert isinstance(projects, AudiobookProjectListResponse)
    assert isinstance(project, AudiobookProjectResponse)
    assert isinstance(chapters, AudiobookChapterListResponse)
    assert isinstance(project_artifacts, AudiobookArtifactsResponse)
    assert isinstance(profile, VoiceProfileResponse)
    assert isinstance(profiles, VoiceProfileListResponse)
    assert isinstance(deleted, VoiceProfileDeleteResponse)
    assert subtitles.content.startswith(b"1\n")
