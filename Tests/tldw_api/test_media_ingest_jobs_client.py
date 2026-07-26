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


#: A real list response for a *completed* job, captured verbatim from a live
#: server (batch 32b5bce2, 2026-07-26). Kept literal because the bug it pins was
#: invisible to hand-written fixtures: every fake used a ``result`` shaped like
#: the model we had, so the model never met the shape the server sends.
_LIVE_COMPLETED_LIST_RESPONSE = {
    "batch_id": "32b5bce2-8818-493f-ab57-bd379396c402",
    "jobs": [
        {
            "id": 281,
            "uuid": "aacbb0d9-111e-43d1-9a2a-3d57fcf97596",
            "status": "completed",
            "job_type": "media_ingest_item",
            "owner_user_id": "1",
            "created_at": "2026-07-26 13:52:27",
            "started_at": "2026-07-26 13:52:28",
            "completed_at": "2026-07-26 13:52:28",
            "cancelled_at": None,
            "cancellation_reason": None,
            "progress_percent": 100.0,
            "progress_message": "completed",
            "result": {
                "status": "Success",
                "media_id": 1125,
                "media_uuid": "bee6d9a0-931f-4c11-a87a-07ddebc66443",
                "error": None,
                "warnings": None,
                "db_message": "Media 'gettysburg' already exists. Overwrite not enabled.",
            },
            "error_message": None,
            "media_type": "document",
            "source": "gettysburg.txt",
            "source_kind": "file",
            "batch_id": "32b5bce2-8818-493f-ab57-bd379396c402",
            "collection_id": None,
            "planned_item_id": None,
            "idempotency_key": None,
        }
    ],
    "limit": 100,
    "offset": 0,
    "has_more": False,
    "next_offset": None,
    "pagination": {
        "mode": "offset",
        "limit": 100,
        "offset": 0,
        "total": None,
        "has_more": False,
        "next_offset": None,
    },
}


def test_a_completed_job_from_a_live_server_validates():
    """A finished ingest job's own status response must parse.

    ``MediaIngestJobStatus.result`` was typed ``ReadingImportResponse`` -- the
    *reading-list* import result (``source``/``imported``/``updated``/
    ``skipped``), a different domain reused by mistake. A media ingest job
    reports ``{"status": ..., "media_id": ..., ...}`` instead, so validation
    failed with four missing fields on every completed job.

    That made the failure invisible until a job actually finished: the Library's
    remote poller treats a raised list call as transient, logs at debug and
    retries, so jobs would have sat "queued" in the UI forever while the poller
    churned. The endpoint documents ``result`` as a free-form object.
    """
    from tldw_chatbook.tldw_api.media_reading_schemas import (
        MediaIngestJobListResponse,
    )

    listed = MediaIngestJobListResponse.model_validate(
        _LIVE_COMPLETED_LIST_RESPONSE
    )

    job = listed.jobs[0]
    assert job.status == "completed"
    assert job.result["media_id"] == 1125


def test_the_completed_spelling_reaches_a_terminal_local_state():
    """``completed`` is the server's real success spelling -- observed live.

    The endpoint types ``status`` as a bare string with no enum, so only a live
    run can confirm the vocabulary. An unmapped spelling would be ignored by the
    reconciler (unknown deliberately moves nothing), leaving a finished job
    displayed as unfinished.
    """
    from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
    from tldw_chatbook.Library.server_ingest_status import (
        is_terminal_server_status,
        local_state_for_server_status,
    )

    assert local_state_for_server_status("completed") is IngestJobState.DONE
    assert is_terminal_server_status("completed") is True


@pytest.mark.asyncio
async def test_listing_jobs_sends_offset_so_later_pages_are_reachable(monkeypatch):
    """``offset`` must reach the wire, or only the first page is ever readable.

    The endpoint documents ``limit`` (default 100, max 500) *and* ``offset``
    (default 0, max 10000), and its response carries ``has_more``/
    ``next_offset`` to drive paging. This client only ever sent ``limit``, so
    the Library's remote-ingest poller -- which asks for successive offsets --
    raised ``TypeError`` on every call and silently fell back to a single page
    (task-684.2). Verified against the live server's OpenAPI document.
    """
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "batch_id": "batch-1",
            "jobs": [_job_status()],
            "limit": 100,
            "offset": 100,
            "has_more": False,
            "next_offset": None,
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_media_ingest_jobs("batch-1", limit=100, offset=100)

    assert mocked.await_args.kwargs["params"] == {
        "batch_id": "batch-1",
        "limit": 100,
        "offset": 100,
    }
    # The pagination fields must survive validation; the response model dropped
    # them once already, which made the poller read a field that could not exist.
    assert listed.offset == 100
    assert listed.has_more is False


@pytest.mark.asyncio
async def test_listing_jobs_defaults_to_the_first_page(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"batch_id": "batch-1", "jobs": []})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_media_ingest_jobs("batch-1")

    assert mocked.await_args.kwargs["params"]["offset"] == 0


@pytest.mark.asyncio
async def test_media_ingest_jobs_client_routes_submit_status_list_cancel_and_reprocess(
    monkeypatch,
):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "batch_id": "batch-1",
                "jobs": [
                    {
                        "id": 11,
                        "uuid": "job-uuid",
                        "source": "https://example.com/a.pdf",
                        "source_kind": "url",
                        "status": "queued",
                    }
                ],
                "errors": [],
            },
            _job_status(),
            {"batch_id": "batch-1", "jobs": [_job_status()]},
            {
                "success": True,
                "job_id": 11,
                "status": "cancelled",
                "message": "Job cancellation requested",
            },
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
    batch_cancelled = await client.cancel_media_ingest_batch(
        batch_id="batch-1", reason="user requested"
    )
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
    assert mocked.await_args_list[2].kwargs["params"] == {
        "batch_id": "batch-1",
        "limit": 50,
        "offset": 0,
    }
    assert mocked.await_args_list[3].args[:2] == (
        "DELETE",
        "/api/v1/media/ingest/jobs/11",
    )
    assert mocked.await_args_list[3].kwargs["params"] == {"reason": "user requested"}
    assert mocked.await_args_list[4].args[:2] == (
        "POST",
        "/api/v1/media/ingest/jobs/cancel",
    )
    assert mocked.await_args_list[4].kwargs["params"] == {
        "batch_id": "batch-1",
        "reason": "user requested",
    }
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
        lambda *args, **kwargs: _fake_sse(
            [{"event": "status", "data": {"id": 11, "status": "completed"}}],
            streamed,
            args,
            kwargs,
        ),
    )

    events = [
        event
        async for event in client.stream_media_ingest_job_events(
            batch_id="batch-1", after_id=4
        )
    ]

    assert streamed.await_args.args[0] == (
        "GET",
        "/api/v1/media/ingest/jobs/events/stream",
    )
    assert streamed.await_args.args[1]["params"] == {
        "batch_id": "batch-1",
        "after_id": 4,
    }
    assert events == [{"event": "status", "data": {"id": 11, "status": "completed"}}]


async def _fake_sse(events, recorder, args, kwargs):
    await recorder(args, kwargs)
    for event in events:
        yield event
