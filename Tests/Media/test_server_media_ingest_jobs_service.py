import pytest

from tldw_chatbook.Media import ServerMediaReadingService


class FakeIngestJobsClient:
    def __init__(self):
        self.calls = []

    async def submit_media_ingest_jobs(self, request_data, file_paths=None):
        self.calls.append(("submit_media_ingest_jobs", request_data.model_dump(exclude_none=True, mode="json"), file_paths))
        return {"batch_id": "batch-1", "jobs": [], "errors": []}

    async def get_media_ingest_job(self, job_id):
        self.calls.append(("get_media_ingest_job", job_id))
        return {"id": job_id, "status": "queued"}

    async def list_media_ingest_jobs(self, batch_id, limit=100):
        self.calls.append(("list_media_ingest_jobs", batch_id, limit))
        return {"batch_id": batch_id, "jobs": []}

    async def cancel_media_ingest_job(self, job_id, reason=None):
        self.calls.append(("cancel_media_ingest_job", job_id, reason))
        return {"success": True, "job_id": job_id, "status": "cancelled"}

    async def cancel_media_ingest_batch(self, *, batch_id=None, session_id=None, reason=None):
        self.calls.append(("cancel_media_ingest_batch", batch_id, session_id, reason))
        return {"success": True, "batch_id": batch_id or session_id, "requested": 1, "cancelled": 1, "already_terminal": 0}

    async def reprocess_media(self, media_id, request_data):
        self.calls.append(("reprocess_media", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "status": "completed", "message": "ok"}


@pytest.mark.asyncio
async def test_server_media_service_routes_ingest_jobs_and_reprocess_operations():
    client = FakeIngestJobsClient()
    service = ServerMediaReadingService(client=client)

    submitted = await service.submit_ingest_jobs(media_type="pdf", urls=["https://example.com/a.pdf"], chunk_size=600)
    status = await service.get_ingest_job(11)
    listed = await service.list_ingest_jobs("batch-1", limit=50)
    cancelled = await service.cancel_ingest_job(11, reason="user requested")
    batch_cancelled = await service.cancel_ingest_batch(batch_id="batch-1", reason="user requested")
    reprocessed = await service.reprocess_media(7, perform_chunking=True, generate_embeddings=False)

    assert submitted["batch_id"] == "batch-1"
    assert status["id"] == 11
    assert listed["batch_id"] == "batch-1"
    assert cancelled["success"] is True
    assert batch_cancelled["cancelled"] == 1
    assert reprocessed["status"] == "completed"
    assert client.calls == [
        (
            "submit_media_ingest_jobs",
            {
                "media_type": "pdf",
                "urls": ["https://example.com/a.pdf"],
                "chunk_size": 600,
                "chunk_overlap": 200,
                "perform_chunking": True,
                "generate_embeddings": False,
                "force_regenerate_embeddings": False,
            },
            None,
        ),
        ("get_media_ingest_job", 11),
        ("list_media_ingest_jobs", "batch-1", 50),
        ("cancel_media_ingest_job", 11, "user requested"),
        ("cancel_media_ingest_batch", "batch-1", None, "user requested"),
        (
            "reprocess_media",
            7,
            {
                "perform_chunking": True,
                "chunk_size": 500,
                "chunk_overlap": 200,
                "use_adaptive_chunking": False,
                "use_multi_level_chunking": False,
                "auto_apply_template": False,
                "enable_contextual_chunking": False,
                "hierarchical_chunking": False,
                "generate_embeddings": False,
                "force_regenerate_embeddings": False,
            },
        ),
    ]
