from unittest.mock import Mock

import pytest

from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ContentItem,
    ContentType,
)
from tldw_chatbook.Chatbooks.server_chatbook_service import (
    ServerChatbookService,
    build_server_chatbook_service_from_config,
    build_server_import_selections_from_manifest,
    build_server_job_record,
    get_server_import_blockers_from_manifest,
    get_server_job_records,
    record_server_job,
)
from tldw_chatbook.runtime_policy import PolicyDecision, PolicyDeniedError
from tldw_chatbook.tldw_api import (
    ChatbookCleanupResponse,
    ChatbookContinueExportRequest,
    ChatbookExportJobListResponse,
    ChatbookExportJobResponse,
    ChatbookImportJobListResponse,
    ChatbookImportJobResponse,
    ChatbookJobMutationResponse,
    ReadingExportResponse,
)


class FakePolicyEnforcer:
    def __init__(self):
        self.calls = []

    def require_allowed(self, *, action_id: str):
        self.calls.append(action_id)


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


def test_build_server_chatbook_service_from_config_threads_policy_enforcer():
    policy = FakePolicyEnforcer()

    service, client = build_server_chatbook_service_from_config(
        {
            "tldw_api": {
                "base_url": "https://example.com/api/",
                "api_key": "secret-key",
            }
        },
        policy_enforcer=policy,
    )

    assert service.client is client
    assert service.policy_enforcer is policy


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

        async def continue_chatbook_export(self, request_data):
            self.continue_request = request_data
            return {"job_id": "continue-job-1", "status": "queued"}

        async def import_chatbook(self, chatbook_file_path: str, request_data):
            self.import_path = chatbook_file_path
            self.import_request = request_data
            return {"job_id": "import-job-1", "status": "queued"}

        async def get_chatbook_export_job(self, job_id: str):
            return {"job_id": job_id, "status": "completed", "progress_percentage": 100}

        async def download_chatbook_export(self, job_id: str, *, token=None, exp=None):
            self.download_args = (job_id, token, exp)
            return ReadingExportResponse(
                content=b"chatbook-bytes",
                content_type="application/zip",
                content_disposition="attachment; filename=pack.chatbook.zip",
                filename="pack.chatbook.zip",
            )

        async def get_chatbook_import_job(self, job_id: str):
            return {"job_id": job_id, "status": "completed", "progress_percentage": 100}

        async def list_chatbook_export_jobs(self, *, limit: int = 100, offset: int = 0):
            return {"items": [{"job_id": "export-job-1"}], "limit": limit, "offset": offset}

        async def list_chatbook_import_jobs(self, *, limit: int = 100, offset: int = 0):
            return {"items": [{"job_id": "import-job-1"}], "limit": limit, "offset": offset}

        async def cancel_chatbook_export_job(self, job_id: str):
            return {"job_id": job_id, "cancelled": True}

        async def cancel_chatbook_import_job(self, job_id: str):
            return {"job_id": job_id, "cancelled": True}

        async def remove_chatbook_export_job(self, job_id: str):
            return {"job_id": job_id, "removed": True}

        async def remove_chatbook_import_job(self, job_id: str):
            return {"job_id": job_id, "removed": True}

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
    continue_result = await service.continue_chatbook_export(
        {
            "export_id": "exp-1",
            "continuations": [{"type": "evaluation", "cursor": "next"}],
            "name": "Continuation",
        }
    )

    import_request = service.build_import_request_payload(
        {"conversation": ["1"]},
        conflict_resolution="rename",
        prefix_imported=True,
        async_mode=True,
    )
    import_result = await service.import_chatbook("/tmp/demo.chatbook.zip", import_request)
    export_job = await service.get_export_job("export-job-1")
    downloaded = await service.download_export("export-job-1", token="signed", exp=12345)
    import_job = await service.get_import_job("import-job-1")
    export_jobs = await service.list_export_jobs(limit=25, offset=5)
    import_jobs = await service.list_import_jobs(limit=10, offset=2)
    cancelled_export = await service.cancel_export_job("export-job-1")
    cancelled_import = await service.cancel_import_job("import-job-1")
    removed_export = await service.remove_export_job("export-job-1")
    removed_import = await service.remove_import_job("import-job-1")

    assert preview["success"] is True
    assert client.preview_path == "/tmp/demo.chatbook.zip"
    assert client.export_request.content_selections == {"conversation": ["1"]}
    assert export_result["job_id"] == "export-job-1"
    assert client.continue_request.export_id == "exp-1"
    assert continue_result["job_id"] == "continue-job-1"
    assert client.import_path == "/tmp/demo.chatbook.zip"
    assert client.import_request.conflict_resolution == "rename"
    assert client.import_request.prefix_imported is True
    assert import_result["job_id"] == "import-job-1"
    assert export_job["status"] == "completed"
    assert client.download_args == ("export-job-1", "signed", 12345)
    assert downloaded["job_id"] == "export-job-1"
    assert downloaded["content"] == b"chatbook-bytes"
    assert downloaded["filename"] == "pack.chatbook.zip"
    assert import_job["status"] == "completed"
    assert export_jobs["items"][0]["job_id"] == "export-job-1"
    assert import_jobs["items"][0]["job_id"] == "import-job-1"
    assert cancelled_export["cancelled"] is True
    assert cancelled_import["cancelled"] is True
    assert removed_export["removed"] is True
    assert removed_import["removed"] is True


@pytest.mark.asyncio
async def test_service_accepts_dict_payloads_from_scope_adapter():
    class FakeClient:
        def __init__(self):
            self.export_request = None
            self.import_request = None

        async def export_chatbook(self, request_data):
            self.export_request = request_data
            return {"job_id": "export-job-1", "status": "queued"}

        async def import_chatbook(self, chatbook_file_path: str, request_data):
            self.import_request = request_data
            return {"job_id": "import-job-1", "status": "queued"}

    client = FakeClient()
    service = ServerChatbookService(client=client)

    await service.export_chatbook(
        {
            "name": "Pack",
            "description": "Desc",
            "content_selections": {"conversation": ["1"]},
        }
    )
    await service.import_chatbook(
        "/tmp/demo.chatbook.zip",
        {
            "content_selections": {"conversation": ["1"]},
            "conflict_resolution": "rename",
        },
    )

    assert client.export_request.content_selections == {"conversation": ["1"]}
    assert client.import_request.conflict_resolution == "rename"


@pytest.mark.asyncio
async def test_server_chatbook_service_enforces_policy_actions():
    class FakeClient:
        def __init__(self):
            self.calls = []

        async def preview_chatbook(self, chatbook_file_path: str):
            self.calls.append(("preview_chatbook", chatbook_file_path))
            return {"success": True, "manifest": {"name": "Preview"}}

        async def export_chatbook(self, request_data):
            self.calls.append(("export_chatbook", request_data))
            return {"job_id": "export-job-1", "status": "queued"}

        async def continue_chatbook_export(self, request_data):
            self.calls.append(("continue_chatbook_export", request_data))
            return {"job_id": "continue-job-1", "status": "queued"}

        async def import_chatbook(self, chatbook_file_path: str, request_data):
            self.calls.append(("import_chatbook", chatbook_file_path, request_data))
            return {"job_id": "import-job-1", "status": "queued"}

        async def get_chatbook_export_job(self, job_id: str):
            self.calls.append(("get_chatbook_export_job", job_id))
            return {"job_id": job_id, "status": "completed"}

        async def download_chatbook_export(self, job_id: str, *, token=None, exp=None):
            self.calls.append(("download_chatbook_export", job_id, token, exp))
            return ReadingExportResponse(content=b"chatbook-bytes")

        async def get_chatbook_import_job(self, job_id: str):
            self.calls.append(("get_chatbook_import_job", job_id))
            return {"job_id": job_id, "status": "completed"}

        async def list_chatbook_export_jobs(self, *, limit: int = 100, offset: int = 0):
            self.calls.append(("list_chatbook_export_jobs", limit, offset))
            return {"items": []}

        async def list_chatbook_import_jobs(self, *, limit: int = 100, offset: int = 0):
            self.calls.append(("list_chatbook_import_jobs", limit, offset))
            return {"items": []}

        async def cancel_chatbook_export_job(self, job_id: str):
            self.calls.append(("cancel_chatbook_export_job", job_id))
            return {"job_id": job_id, "cancelled": True}

        async def cancel_chatbook_import_job(self, job_id: str):
            self.calls.append(("cancel_chatbook_import_job", job_id))
            return {"job_id": job_id, "cancelled": True}

        async def remove_chatbook_export_job(self, job_id: str):
            self.calls.append(("remove_chatbook_export_job", job_id))
            return {"job_id": job_id, "removed": True}

        async def remove_chatbook_import_job(self, job_id: str):
            self.calls.append(("remove_chatbook_import_job", job_id))
            return {"job_id": job_id, "removed": True}

    policy = Mock()
    service = ServerChatbookService(client=FakeClient(), policy_enforcer=policy)

    await service.preview_chatbook("/tmp/demo.chatbook.zip")
    await service.export_chatbook({"name": "Pack", "description": "Desc", "content_selections": {"conversation": ["1"]}})
    await service.continue_chatbook_export({"export_id": "exp-1", "continuations": [{"type": "evaluation"}]})
    await service.import_chatbook("/tmp/demo.chatbook.zip", {"content_selections": {"conversation": ["1"]}})
    await service.get_export_job("export-job-1")
    await service.download_export("export-job-1")
    await service.get_import_job("import-job-1")
    await service.list_export_jobs()
    await service.list_import_jobs()
    await service.cancel_export_job("export-job-1")
    await service.cancel_import_job("import-job-1")
    await service.remove_export_job("export-job-1")
    await service.remove_import_job("import-job-1")

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "chatbooks.detail.server",
        "chatbooks.export.server",
        "chatbooks.export.server",
        "chatbooks.import.server",
        "chatbooks.export_jobs.detail.server",
        "chatbooks.export_jobs.export.server",
        "chatbooks.import_jobs.detail.server",
        "chatbooks.export_jobs.list.server",
        "chatbooks.import_jobs.list.server",
        "chatbooks.export_jobs.update.server",
        "chatbooks.import_jobs.update.server",
        "chatbooks.export_jobs.delete.server",
        "chatbooks.import_jobs.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_chatbook_service_hard_stops_denied_ui_policy_decision():
    class FakeClient:
        def __init__(self):
            self.calls = []

        async def preview_chatbook(self, chatbook_file_path: str):
            self.calls.append(("preview_chatbook", chatbook_file_path))
            return {"success": True}

    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeClient()
    service = ServerChatbookService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.preview_chatbook("/tmp/demo.chatbook.zip")

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []


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
async def test_service_exposes_chatbook_continue_and_cleanup_as_plain_dicts():
    class FakeClient:
        def __init__(self):
            self.continue_request = None

        async def continue_chatbook_export(self, request_data):
            self.continue_request = request_data
            return {"success": True, "job_id": "continued-job-1"}

        async def cleanup_chatbook_exports(self):
            return ChatbookCleanupResponse(deleted_count=2, message="Cleaned expired exports")

    client = FakeClient()
    service = ServerChatbookService(client=client)
    request = ChatbookContinueExportRequest(
        export_id="export-job-1",
        continuations=[{"content_type": "notes", "cursor": "cursor-2"}],
    )

    continued = await service.continue_chatbook_export(request)
    cleanup = await service.cleanup_expired_exports()

    assert client.continue_request == request
    assert continued == {"success": True, "job_id": "continued-job-1"}
    assert cleanup == {"deleted_count": 2, "message": "Cleaned expired exports"}


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


@pytest.mark.asyncio
async def test_service_enforces_action_policy_for_server_chatbook_operations(tmp_path):
    class FakeClient:
        async def preview_chatbook(self, chatbook_file_path: str):
            return {"success": True}

        async def export_chatbook(self, request_data):
            return {"job_id": "export-job-1"}

        async def continue_chatbook_export(self, request_data):
            return {"job_id": "continued-job-1"}

        async def import_chatbook(self, chatbook_file_path: str, request_data):
            return {"job_id": "import-job-1"}

        async def list_chatbook_export_jobs(self, limit: int = 100, offset: int = 0):
            return {"jobs": [], "total": 0}

        async def get_chatbook_export_job(self, job_id: str):
            return {"job_id": job_id}

        async def cancel_chatbook_export_job(self, job_id: str):
            return {"success": True, "job_id": job_id}

        async def remove_chatbook_export_job(self, job_id: str):
            return {"success": True, "job_id": job_id}

        async def download_chatbook_export(self, job_id: str):
            return b"chatbook-zip"

        async def cleanup_chatbook_exports(self):
            return {"deleted_count": 0}

        async def list_chatbook_import_jobs(self, limit: int = 100, offset: int = 0):
            return {"jobs": [], "total": 0}

        async def get_chatbook_import_job(self, job_id: str):
            return {"job_id": job_id}

        async def cancel_chatbook_import_job(self, job_id: str):
            return {"success": True, "job_id": job_id}

        async def remove_chatbook_import_job(self, job_id: str):
            return {"success": True, "job_id": job_id}

    policy = FakePolicyEnforcer()
    service = ServerChatbookService(client=FakeClient(), policy_enforcer=policy)
    export_request = service.build_export_request_payload(
        name="Pack",
        description="Desc",
        selections={"conversation": ["1"]},
    )
    import_request = service.build_import_request_payload({"conversation": ["1"]})
    continue_request = ChatbookContinueExportRequest(
        export_id="export-job-1",
        continuations=[{"content_type": "notes", "cursor": "cursor-2"}],
    )

    await service.preview_chatbook("/tmp/demo.chatbook.zip")
    await service.export_chatbook(export_request)
    await service.continue_chatbook_export(continue_request)
    await service.import_chatbook("/tmp/demo.chatbook.zip", import_request)
    await service.list_export_jobs()
    await service.get_export_job("export-job-1")
    await service.cancel_export_job("export-job-1")
    await service.remove_export_job("export-job-1")
    await service.download_export_job("export-job-1", tmp_path / "Pack.chatbook.zip")
    await service.cleanup_expired_exports()
    await service.list_import_jobs()
    await service.get_import_job("import-job-1")
    await service.cancel_import_job("import-job-1")
    await service.remove_import_job("import-job-1")

    assert policy.calls == [
        "chatbooks.detail.server",
        "chatbooks.export.server",
        "chatbooks.export.server",
        "chatbooks.import.server",
        "chatbooks.export_jobs.list.server",
        "chatbooks.export_jobs.detail.server",
        "chatbooks.export_jobs.update.server",
        "chatbooks.export_jobs.delete.server",
        "chatbooks.export_jobs.export.server",
        "chatbooks.export_jobs.delete.server",
        "chatbooks.import_jobs.list.server",
        "chatbooks.import_jobs.detail.server",
        "chatbooks.import_jobs.update.server",
        "chatbooks.import_jobs.delete.server",
    ]


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
