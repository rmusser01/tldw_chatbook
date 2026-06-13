import pytest

from tldw_chatbook.Media import ServerMediaReadingService


class FakeMediaVersionsClient:
    def __init__(self):
        self.calls = []

    async def list_media_versions(self, media_id, **params):
        self.calls.append(("list_media_versions", media_id, params))
        return [{"media_id": media_id, "version_number": 1}]

    async def get_media_version(self, media_id, version_number, **params):
        self.calls.append(("get_media_version", media_id, version_number, params))
        return {"media_id": media_id, "version_number": version_number}

    async def create_media_version(self, media_id, request_data):
        self.calls.append(("create_media_version", media_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"media_id": media_id, "versions": []}

    async def delete_media_version(self, media_id, version_number):
        self.calls.append(("delete_media_version", media_id, version_number))
        return {}


@pytest.mark.asyncio
async def test_server_media_service_routes_document_version_operations():
    client = FakeMediaVersionsClient()
    service = ServerMediaReadingService(client=client)

    versions = await service.list_document_versions(7, include_content=True, limit=25, page=2)
    version = await service.get_document_version(7, 2, include_content=False)
    saved = await service.save_analysis_version(
        7,
        content="Body",
        prompt="Summarize",
        analysis_content="Analysis",
        safe_metadata={"reviewed": True},
    )
    overwritten = await service.overwrite_analysis_version(
        7,
        content="Body v2",
        prompt="Refresh",
        analysis_content="Analysis v2",
    )
    deleted = await service.delete_document_version(7, 2)

    assert versions == [{"media_id": 7, "version_number": 1}]
    assert version == {"media_id": 7, "version_number": 2}
    assert saved == {"media_id": 7, "versions": []}
    assert overwritten == {"media_id": 7, "versions": []}
    assert deleted == {}
    assert client.calls == [
        ("list_media_versions", 7, {"include_content": True, "limit": 25, "page": 2}),
        ("get_media_version", 7, 2, {"include_content": False}),
        (
            "create_media_version",
            7,
            {
                "content": "Body",
                "prompt": "Summarize",
                "analysis_content": "Analysis",
                "safe_metadata": {"reviewed": True},
            },
        ),
        (
            "create_media_version",
            7,
            {
                "content": "Body v2",
                "prompt": "Refresh",
                "analysis_content": "Analysis v2",
            },
        ),
        ("delete_media_version", 7, 2),
    ]


@pytest.mark.asyncio
async def test_server_media_service_rejects_deleted_version_listing_and_uuid_delete_shape():
    service = ServerMediaReadingService(client=FakeMediaVersionsClient())

    with pytest.raises(ValueError, match="deleted document versions"):
        await service.list_document_versions(7, include_deleted=True)

    with pytest.raises(ValueError, match="requires media_id and version_number"):
        await service.delete_analysis_version("11111111-1111-1111-1111-111111111111")
