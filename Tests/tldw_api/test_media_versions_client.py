from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    MediaAdvancedVersionUpsertRequest,
    MediaMetadataPatchRequest,
    MediaVersionCreateRequest,
    MediaVersionRollbackRequest,
    TLDWAPIClient,
)


def _version_response() -> dict:
    return {
        "uuid": "11111111-1111-1111-1111-111111111111",
        "media_id": 7,
        "version_number": 2,
        "created_at": "2026-04-22T12:00:00Z",
        "prompt": "Summarize",
        "analysis_content": "Analysis",
        "safe_metadata": {"reviewed": True},
        "content": "Body",
    }


@pytest.mark.asyncio
async def test_media_versions_client_routes_core_version_operations(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    media_detail = {"media_id": 7, "versions": []}
    mocked = AsyncMock(
        side_effect=[
            [_version_response()],
            _version_response(),
            media_detail,
            {},
            media_detail,
            media_detail,
            media_detail,
            media_detail,
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    versions = await client.list_media_versions(7, include_content=True, limit=25, page=2)
    version = await client.get_media_version(7, 2, include_content=False)
    created = await client.create_media_version(
        7,
        MediaVersionCreateRequest(
            content="Body",
            prompt="Summarize",
            analysis_content="Analysis",
            safe_metadata={"reviewed": True},
        ),
    )
    deleted = await client.delete_media_version(7, 2)
    rolled_back = await client.rollback_media_version(7, MediaVersionRollbackRequest(version_number=1))
    patched = await client.patch_media_metadata(
        7,
        MediaMetadataPatchRequest(safe_metadata={"topic": "paper"}, merge=False, new_version=True),
    )
    version_patched = await client.put_media_version_metadata(
        7,
        2,
        MediaMetadataPatchRequest(safe_metadata={"topic": "paper"}, merge=True),
    )
    advanced = await client.upsert_media_version_advanced(
        7,
        MediaAdvancedVersionUpsertRequest(
            content="Body v3",
            prompt="Refresh",
            analysis_content="Analysis v3",
            safe_metadata={"topic": "paper"},
            new_version=True,
        ),
    )

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/media/7/versions")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "include_content": "true",
        "limit": 25,
        "page": 2,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/media/7/versions/2")
    assert mocked.await_args_list[1].kwargs["params"] == {"include_content": "false"}
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/media/7/versions")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "content": "Body",
        "prompt": "Summarize",
        "analysis_content": "Analysis",
        "safe_metadata": {"reviewed": True},
    }
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/media/7/versions/2")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/media/7/versions/rollback")
    assert mocked.await_args_list[4].kwargs["json_data"] == {"version_number": 1}
    assert mocked.await_args_list[5].args[:2] == ("PATCH", "/api/v1/media/7/metadata")
    assert mocked.await_args_list[6].args[:2] == ("PUT", "/api/v1/media/7/versions/2/metadata")
    assert mocked.await_args_list[7].args[:2] == ("POST", "/api/v1/media/7/versions/advanced")
    assert mocked.await_args_list[7].kwargs["json_data"] == {
        "content": "Body v3",
        "prompt": "Refresh",
        "analysis_content": "Analysis v3",
        "safe_metadata": {"topic": "paper"},
        "merge": True,
        "new_version": True,
    }

    assert versions[0].version_number == 2
    assert version.safe_metadata == {"reviewed": True}
    assert created == media_detail
    assert deleted == {}
    assert rolled_back == media_detail
    assert patched == media_detail
    assert version_patched == media_detail
    assert advanced == media_detail
