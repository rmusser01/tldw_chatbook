from unittest.mock import AsyncMock

import httpx
import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.exceptions import APIResponseError, AuthenticationError


@pytest.mark.asyncio
async def test_request_401_preserves_structured_response_data_for_auth_classification(
    monkeypatch,
):
    client = TLDWAPIClient("http://localhost:8000")
    request = httpx.Request("GET", "http://localhost:8000/api/v1/evals")
    response = httpx.Response(
        401,
        request=request,
        json={"code": "session_invalid", "detail": "Session expired"},
    )
    http_error = httpx.HTTPStatusError(
        "401 Unauthorized", request=request, response=response
    )

    mocked_http_client = AsyncMock()
    mocked_http_client.request = AsyncMock(side_effect=http_error)
    monkeypatch.setattr(
        client, "_get_client", AsyncMock(return_value=mocked_http_client)
    )

    with pytest.raises(AuthenticationError) as exc:
        await client._request("GET", "/api/v1/evals")

    assert exc.value.response_data == {
        "code": "session_invalid",
        "detail": "Session expired",
    }
    assert "Session expired" in str(exc.value)


@pytest.mark.asyncio
async def test_request_409_dict_detail_surfaces_the_server_message(monkeypatch):
    """tldw_server returns its deterministic 4xx refusals as a STRUCTURED
    detail object, not a string (schedules task 6 round 2, D9):

        {"detail": {"code": "scheduled_task_definition_archived",
                    "message": "Scheduled task definition is archived.",
                    "details": {...}, "retryable": false}}

    The extraction handled `detail` as a list (pydantic) or a str, but a
    dict fell through to the raw httpx text ("Client error '409
    Conflict' for url ... For more information check: https://..."), so
    the server's own explanation never reached the caller and the UI
    could only report a generic failure.
    """
    client = TLDWAPIClient("http://localhost:8000")
    url = "http://localhost:8000/api/v1/scheduled-tasks/definitions/abc/mark-solved"
    request = httpx.Request("POST", url)
    body = {
        "detail": {
            "code": "scheduled_task_definition_archived",
            "message": "Scheduled task definition is archived.",
            "details": {"reason": "definition_archived"},
            "retryable": False,
        }
    }
    response = httpx.Response(409, request=request, json=body)
    http_error = httpx.HTTPStatusError(
        "409 Conflict", request=request, response=response
    )

    mocked_http_client = AsyncMock()
    mocked_http_client.request = AsyncMock(side_effect=http_error)
    monkeypatch.setattr(
        client, "_get_client", AsyncMock(return_value=mocked_http_client)
    )

    with pytest.raises(APIResponseError) as exc:
        await client._request("POST", "/api/v1/scheduled-tasks/definitions/abc/mark-solved")

    assert exc.value.status_code == 409
    assert "Scheduled task definition is archived." in str(exc.value)
    # Never the raw httpx text, which is what a user would otherwise see.
    assert "developer.mozilla.org" not in str(exc.value)
    # The whole structured body still reaches callers that want the code.
    assert exc.value.response_data == body


@pytest.mark.asyncio
async def test_request_dict_detail_without_message_falls_back_to_code(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    request = httpx.Request("POST", "http://localhost:8000/api/v1/x")
    response = httpx.Response(
        409, request=request, json={"detail": {"code": "some_conflict"}}
    )
    http_error = httpx.HTTPStatusError(
        "409 Conflict", request=request, response=response
    )
    mocked_http_client = AsyncMock()
    mocked_http_client.request = AsyncMock(side_effect=http_error)
    monkeypatch.setattr(
        client, "_get_client", AsyncMock(return_value=mocked_http_client)
    )

    with pytest.raises(APIResponseError) as exc:
        await client._request("POST", "/api/v1/x")

    assert "some_conflict" in str(exc.value)
