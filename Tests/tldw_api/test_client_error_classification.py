from unittest.mock import AsyncMock

import httpx
import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.exceptions import AuthenticationError


@pytest.mark.asyncio
async def test_request_401_preserves_structured_response_data_for_auth_classification(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    request = httpx.Request("GET", "http://localhost:8000/api/v1/evals")
    response = httpx.Response(
        401,
        request=request,
        json={"code": "session_invalid", "detail": "Session expired"},
    )
    http_error = httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)

    mocked_http_client = AsyncMock()
    mocked_http_client.request = AsyncMock(side_effect=http_error)
    monkeypatch.setattr(client, "_get_client", AsyncMock(return_value=mocked_http_client))

    with pytest.raises(AuthenticationError) as exc:
        await client._request("GET", "/api/v1/evals")

    assert exc.value.response_data == {
        "code": "session_invalid",
        "detail": "Session expired",
    }
    assert "Session expired" in str(exc.value)
