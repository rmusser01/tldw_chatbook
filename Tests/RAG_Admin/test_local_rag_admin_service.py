from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService


class FakeMediaService:
    def __init__(self):
        self.calls = []

    def reprocess_media(self, media_id, **options):
        self.calls.append(("reprocess_media", media_id, options))
        return {"status": "queued", "media_id": media_id, "job_id": "local-job-1", "options": options}


def test_local_rag_admin_service_delegates_reprocess_to_local_media_service():
    media_service = FakeMediaService()
    service = LocalRAGAdminService(None, media_service=media_service)

    result = service.reprocess_media(
        9,
        perform_chunking=True,
        generate_embeddings=True,
    )

    assert result["status"] == "queued"
    assert result["job_id"] == "local-job-1"
    assert media_service.calls == [
        ("reprocess_media", 9, {"perform_chunking": True, "generate_embeddings": True})
    ]
