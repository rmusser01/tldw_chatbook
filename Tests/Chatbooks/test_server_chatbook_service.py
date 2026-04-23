import pytest

from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ContentItem,
    ContentType,
)
from tldw_chatbook.Chatbooks.server_chatbook_service import (
    ServerChatbookService,
    build_server_import_selections_from_manifest,
    build_server_job_record,
    get_server_import_blockers_from_manifest,
    get_server_job_records,
    record_server_job,
)
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import (
    ChatbookExportJobListResponse,
    ChatbookExportJobResponse,
    ChatbookImportJobListResponse,
    ChatbookImportJobResponse,
    ChatbookJobMutationResponse,
)


def test_service_rejects_server_unsupported_import_content_types():
    service = ServerChatbookService(client=None)

    unsupported = service.validate_server_import_selection(
        {"prompt": ["1"], "note": ["2"], "evaluation": ["3"]}
    )

    assert unsupported == ["evaluation", "prompt"]


def test_service_normalizes_export_selection_keys():
    service = ServerChatbookService(client=None)

    payload = service.build_export_request_payload(
        name="Pack",
        description="Desc",
        selections={
            ContentType.CONVERSATION: [1],
            "prompt": ["2"],
        },
    )

    assert payload.content_selections["conversation"] == ["1"]
    assert payload.content_selections["prompt"] == ["2"]


def test_service_rejects_import_request_with_unsupported_content_types():
    service = ServerChatbookService(client=None)

    with pytest.raises(ValueError, match="Unsupported server import content types"):
        service.build_import_request_payload(
            {
                "conversation": ["1"],
                ContentType.MEDIA: ["2"],
            }
        )


@pytest.mark.asyncio
async def test_service_delegates_chatbook_api_calls():
    class FakeClient:
        def __init__(self):
            self.preview_path = None
            self.export_request = None
            self.import_path = None
            self.import_request = None

        async def preview_chatbook(self, chatbook_file_path: str):
            self.preview_path = chatbook_file_path
            return {"success": True, "manifest": {"name": "Preview"}}

        async def export_chatbook(self, request_data):
            self.export_request = request_data
            return {"job_id": "export-job-1", "status": "queued"}

        async def import_chatbook(self, chatbook_file_path: str, request_data):
            self.import_path = chatbook_file_path
            self.import_request = request_data
            return {"job_id": "import-job-1", "status": "queued"}

        async def get_chatbook_export_job(self, job_id: str):
            return {"job_id": job_id, "status": "completed", "progress_percentage": 100}

        async def get_chatbook_import_job(self, job_id: str):
            return {"job_id": job_id, "status": "completed", "progress_percentage": 100}

    client = FakeClient()
    service = ServerChatbookService(client=client)

    preview = await service.preview_chatbook("/tmp/demo.chatbook.zip")
    export_request = service.build_export_request_payload(
        name="Pack",
        description="Desc",
        selections={"conversation": ["1"]},
        async_mode=True,
    )
    export_result = await service.export_chatbook(export_request)

    import_request = service.build_import_request_payload(
        {"conversation": ["1"]},
        conflict_resolution="rename",
        prefix_imported=True,
        async_mode=True,
    )
    import_result = await service.import_chatbook("/tmp/demo.chatbook.zip", import_request)
    export_job = await service.get_export_job("export-job-1")
    import_job = await service.get_import_job("import-job-1")

    assert preview["success"] is True
    assert client.preview_path == "/tmp/demo.chatbook.zip"
    assert client.export_request.content_selections == {"conversation": ["1"]}
    assert export_result["job_id"] == "export-job-1"
    assert client.import_path == "/tmp/demo.chatbook.zip"
    assert client.import_request.conflict_resolution == "rename"
    assert client.import_request.prefix_imported is True
    assert import_result["job_id"] == "import-job-1"
    assert export_job["status"] == "completed"
    assert import_job["status"] == "completed"


@pytest.mark.asyncio
async def test_service_exposes_remote_job_admin_as_plain_dicts():
    class FakeClient:
        async def list_chatbook_export_jobs(self, limit: int = 100, offset: int = 0):
            return ChatbookExportJobListResponse(
                jobs=[
                    ChatbookExportJobResponse(
                        job_id="export-job-1",
                        status="completed",
                        chatbook_name="Pack",
                        progress_percentage=100,
                    )
                ],
                total=1,
            )

        async def list_chatbook_import_jobs(self, limit: int = 100, offset: int = 0):
            return ChatbookImportJobListResponse(
                jobs=[
                    ChatbookImportJobResponse(
                        job_id="import-job-1",
                        status="completed",
                        progress_percentage=100,
                    )
                ],
                total=1,
            )

        async def get_chatbook_export_job(self, job_id: str):
            return ChatbookExportJobResponse(
                job_id=job_id,
                status="completed",
                chatbook_name="Pack",
                progress_percentage=100,
            )

        async def get_chatbook_import_job(self, job_id: str):
            return ChatbookImportJobResponse(
                job_id=job_id,
                status="completed",
                progress_percentage=100,
            )

        async def cancel_chatbook_export_job(self, job_id: str):
            return ChatbookJobMutationResponse(success=True, message="cancelled", job_id=job_id)

        async def cancel_chatbook_import_job(self, job_id: str):
            return ChatbookJobMutationResponse(success=True, message="cancelled", job_id=job_id)

        async def remove_chatbook_export_job(self, job_id: str):
            return ChatbookJobMutationResponse(success=True, message="removed", job_id=job_id)

        async def remove_chatbook_import_job(self, job_id: str):
            return ChatbookJobMutationResponse(success=True, message="removed", job_id=job_id)

    service = ServerChatbookService(client=FakeClient())

    export_jobs = await service.list_export_jobs(limit=25, offset=5)
    import_jobs = await service.list_import_jobs(limit=25, offset=5)
    export_job = await service.get_export_job("export-job-1")
    import_job = await service.get_import_job("import-job-1")
    cancel_export = await service.cancel_export_job("export-job-1")
    cancel_import = await service.cancel_import_job("import-job-1")
    remove_export = await service.remove_export_job("export-job-1")
    remove_import = await service.remove_import_job("import-job-1")

    assert export_jobs["jobs"][0]["job_id"] == "export-job-1"
    assert import_jobs["jobs"][0]["job_id"] == "import-job-1"
    assert export_job.get("status") == "completed"
    assert import_job.get("status") == "completed"
    assert cancel_export == {"success": True, "message": "cancelled", "job_id": "export-job-1"}
    assert cancel_import == {"success": True, "message": "cancelled", "job_id": "import-job-1"}
    assert remove_export == {"success": True, "message": "removed", "job_id": "export-job-1"}
    assert remove_import == {"success": True, "message": "removed", "job_id": "import-job-1"}


@pytest.mark.asyncio
async def test_service_downloads_export_job_to_destination(tmp_path):
    class FakeClient:
        async def download_chatbook_export(self, job_id: str):
            assert job_id == "export-job-1"
            return b"chatbook-zip"

    service = ServerChatbookService(client=FakeClient())
    destination = tmp_path / "Pack.chatbook.zip"

    result = await service.download_export_job("export-job-1", destination)

    assert result == destination
    assert destination.read_bytes() == b"chatbook-zip"


def test_service_builds_server_import_selections_from_manifest():
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Pack",
        description="Desc",
        content_items=[
            ContentItem(id="conv-1", type=ContentType.CONVERSATION, title="Conversation"),
            ContentItem(id="note-1", type=ContentType.NOTE, title="Note"),
            ContentItem(id="media-1", type=ContentType.MEDIA, title="Media"),
            ContentItem(id="embed-1", type=ContentType.EMBEDDING, title="Embedding"),
        ],
    )

    selections = build_server_import_selections_from_manifest(
        manifest,
        import_media=False,
        import_embeddings=False,
    )

    assert selections == {
        "conversation": ["conv-1"],
        "note": ["note-1"],
    }


def test_service_reports_server_import_blockers_from_manifest():
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Pack",
        description="Desc",
        content_items=[
            ContentItem(id="prompt-1", type=ContentType.PROMPT, title="Prompt"),
            ContentItem(id="eval-1", type=ContentType.EVALUATION, title="Eval"),
            ContentItem(id="media-1", type=ContentType.MEDIA, title="Media"),
        ],
    )

    blockers = get_server_import_blockers_from_manifest(
        manifest,
        import_media=True,
        import_embeddings=False,
    )

    assert blockers == ["evaluation", "media", "prompt"]


def test_service_records_server_jobs_on_app_state():
    class FakeApp:
        pass

    app = FakeApp()

    record = build_server_job_record(
        "export",
        {
            "job_id": "job-123",
            "status": "completed",
            "progress_percentage": 100,
            "chatbook_name": "Pack",
            "download_url": "https://example.com/download.zip",
        },
    )
    record_server_job(app, record)

    jobs = get_server_job_records(app)

    assert len(jobs) == 1
    assert jobs[0]["job_type"] == "export"
    assert jobs[0]["job_id"] == "job-123"
    assert jobs[0]["chatbook_name"] == "Pack"
