import pytest

from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.Chatbooks.server_chatbook_service import ServerChatbookService


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
