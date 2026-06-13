from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    MediaIngestSubmitRequest,
    ReprocessMediaRequest,
    TLDWAPIClient,
)


def _job_status(job_id: int = 11) -> dict:
    return {
        "id": job_id,
        "uuid": "job-uuid",
        "status": "queued",
        "job_type": "media_ingest",
        "owner_user_id": "user-1",
        "created_at": "2026-04-22T12:00:00Z",
        "started_at": None,
        "completed_at": None,
        "cancelled_at": None,
        "cancellation_reason": None,
        "progress_percent": 0.0,
        "progress_message": "Queued",
        "result": None,
        "error_message": None,
        "media_type": "pdf",
        "source": "https://example.com/a.pdf",
        "source_kind": "url",
        "batch_id": "batch-1",
    }


@pytest.mark.asyncio
async def test_media_ingest_jobs_client_routes_submit_status_list_cancel_and_reprocess(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "batch_id": "batch-1",
                "jobs": [{"id": 11, "uuid": "job-uuid", "source": "https://example.com/a.pdf", "source_kind": "url", "status": "queued"}],
                "errors": [],
            },
            _job_status(),
            {"batch_id": "batch-1", "jobs": [_job_status()]},
            {"success": True, "job_id": 11, "status": "cancelled", "message": "Job cancellation requested"},
            {
                "success": True,
                "batch_id": "batch-1",
                "requested": 1,
                "cancelled": 1,
                "already_terminal": 0,
                "failed": 0,
                "message": "Cancellation requested for 1 job(s)",
            },
            {
                "media_id": 7,
                "status": "completed",
                "message": "Reprocess completed.",
                "chunks_created": 4,
                "embeddings_started": False,
                "job_id": None,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    submitted = await client.submit_media_ingest_jobs(
        MediaIngestSubmitRequest(
            media_type="pdf",
            urls=["https://example.com/a.pdf"],
            keywords=["paper"],
            chunk_size=600,
        )
    )
    status = await client.get_media_ingest_job(11)
    listed = await client.list_media_ingest_jobs("batch-1", limit=50)
    cancelled = await client.cancel_media_ingest_job(11, reason="user requested")
    batch_cancelled = await client.cancel_media_ingest_batch(batch_id="batch-1", reason="user requested")
    reprocessed = await client.reprocess_media(
        7,
        ReprocessMediaRequest(
            perform_chunking=True,
            generate_embeddings=False,
            chunk_size=600,
            chunk_overlap=100,
        ),
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/media/ingest/jobs")
    assert mocked.await_args_list[0].kwargs["data"] == {
        "media_type": "pdf",
        "urls": ["https://example.com/a.pdf"],
        "keywords": ["paper"],
        "chunk_size": 600,
        "chunk_overlap": 200,
        "perform_chunking": True,
        "generate_embeddings": False,
        "force_regenerate_embeddings": False,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/media/ingest/jobs/11")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/media/ingest/jobs")
    assert mocked.await_args_list[2].kwargs["params"] == {"batch_id": "batch-1", "limit": 50}
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/media/ingest/jobs/11")
    assert mocked.await_args_list[3].kwargs["params"] == {"reason": "user requested"}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/media/ingest/jobs/cancel")
    assert mocked.await_args_list[4].kwargs["params"] == {"batch_id": "batch-1", "reason": "user requested"}
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/media/7/reprocess")
    assert mocked.await_args_list[5].kwargs["json_data"] == {
        "perform_chunking": True,
        "chunk_size": 600,
        "chunk_overlap": 100,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "auto_apply_template": False,
        "enable_contextual_chunking": False,
        "hierarchical_chunking": False,
        "generate_embeddings": False,
        "force_regenerate_embeddings": False,
    }

    assert submitted.batch_id == "batch-1"
    assert status.id == 11
    assert listed.jobs[0].id == 11
    assert cancelled.success is True
    assert batch_cancelled.cancelled == 1
    assert reprocessed.chunks_created == 4


@pytest.mark.asyncio
async def test_media_ingest_jobs_client_streams_sse_events(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    streamed = AsyncMock()
    monkeypatch.setattr(
        client,
        "_sse_request",
        lambda *args, **kwargs: _fake_sse([{"event": "status", "data": {"id": 11, "status": "completed"}}], streamed, args, kwargs),
    )

    events = [event async for event in client.stream_media_ingest_job_events(batch_id="batch-1", after_id=4)]

    assert streamed.await_args.args[0] == ("GET", "/api/v1/media/ingest/jobs/events/stream")
    assert streamed.await_args.args[1]["params"] == {"batch_id": "batch-1", "after_id": 4}
    assert events == [{"event": "status", "data": {"id": 11, "status": "completed"}}]


async def _fake_sse(events, recorder, args, kwargs):
    await recorder(args, kwargs)
    for event in events:
        yield event
