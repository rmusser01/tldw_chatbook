"""Tests for writing manuscript project endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ManuscriptProjectCreateRequest,
    ManuscriptProjectListResponse,
    ManuscriptProjectResponse,
    ManuscriptProjectUpdateRequest,
    TLDWAPIClient,
)


def _project_payload(*, project_id: str = "proj-1", version: int = 1) -> dict:
    return {
        "id": project_id,
        "title": "Novel Draft",
        "subtitle": "A Beginning",
        "author": "Jane Doe",
        "genre": "Fantasy",
        "status": "draft",
        "synopsis": "An epic journey",
        "target_word_count": 90000,
        "settings": {"tone": "dark"},
        "word_count": 0,
        "created_at": "2026-04-22T00:00:00Z",
        "last_modified": "2026-04-22T00:00:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": version,
    }


@pytest.mark.asyncio
async def test_list_manuscript_projects_uses_server_prefix():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(return_value={"projects": [], "total": 0})

    response = await client.list_manuscript_projects(status="draft", limit=25, offset=5)

    assert isinstance(response, ManuscriptProjectListResponse)
    assert response.total == 0
    client._request.assert_awaited_once_with(
        "GET",
        "/api/v1/writing/manuscripts/projects",
        params={"status": "draft", "limit": 25, "offset": 5},
    )


@pytest.mark.asyncio
async def test_create_manuscript_project_wires_payload_and_returns_typed_response():
    client = TLDWAPIClient("http://example.test", "token")
    payload = _project_payload()
    client._request = AsyncMock(return_value=payload)

    response = await client.create_manuscript_project(
        ManuscriptProjectCreateRequest(
            title="Novel Draft",
            subtitle="A Beginning",
            author="Jane Doe",
            genre="Fantasy",
            status="draft",
            synopsis="An epic journey",
            target_word_count=90000,
            settings={"tone": "dark"},
        )
    )

    assert isinstance(response, ManuscriptProjectResponse)
    client._request.assert_awaited_once_with(
        "POST",
        "/api/v1/writing/manuscripts/projects",
        json_data={
            "title": "Novel Draft",
            "subtitle": "A Beginning",
            "author": "Jane Doe",
            "genre": "Fantasy",
            "status": "draft",
            "synopsis": "An epic journey",
            "target_word_count": 90000,
            "settings": {"tone": "dark"},
        },
    )


@pytest.mark.asyncio
async def test_get_manuscript_project_returns_typed_response():
    client = TLDWAPIClient("http://example.test", "token")
    payload = _project_payload(project_id="proj-99", version=2)
    client._request = AsyncMock(return_value=payload)

    response = await client.get_manuscript_project("proj-99")

    assert isinstance(response, ManuscriptProjectResponse)
    assert response.id == "proj-99"
    client._request.assert_awaited_once_with(
        "GET",
        "/api/v1/writing/manuscripts/projects/proj-99",
    )


@pytest.mark.asyncio
async def test_update_manuscript_project_passes_expected_version_header():
    client = TLDWAPIClient("http://example.test", "token")
    payload = _project_payload(project_id="proj-2", version=4)
    client._request = AsyncMock(return_value=payload)

    response = await client.update_manuscript_project(
        "proj-2",
        ManuscriptProjectUpdateRequest(status="writing", synopsis="Updated synopsis"),
        expected_version=3,
    )

    assert isinstance(response, ManuscriptProjectResponse)
    assert response.version == 4
    client._request.assert_awaited_once_with(
        "PATCH",
        "/api/v1/writing/manuscripts/projects/proj-2",
        json_data={"status": "writing", "synopsis": "Updated synopsis"},
        headers={"expected-version": "3"},
    )


@pytest.mark.asyncio
async def test_delete_manuscript_project_passes_expected_version_header():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(return_value={})

    response = await client.delete_manuscript_project("proj-2", expected_version=3)

    assert response == {}
    client._request.assert_awaited_once_with(
        "DELETE",
        "/api/v1/writing/manuscripts/projects/proj-2",
        headers={"expected-version": "3"},
    )
