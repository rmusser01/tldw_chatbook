from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import Text2SQLRequest, Text2SQLResponse, TLDWAPIClient


@pytest.mark.asyncio
async def test_text2sql_client_routes_query(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "sql": "SELECT title FROM media LIMIT 2",
            "columns": ["title"],
            "rows": [{"title": "A"}, {"title": "B"}],
            "row_count": 2,
            "duration_ms": 15,
            "target_id": "media_db",
            "guardrail": {"limit_injected": True, "limit_clamped": False},
            "truncated": False,
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.query_text2sql(
        Text2SQLRequest(
            query="SELECT title FROM media",
            target_id="media_db",
            max_rows=2,
            timeout_ms=1500,
            include_sql=True,
        )
    )

    assert mocked.await_args.args[:2] == ("POST", "/api/v1/text2sql/query")
    assert mocked.await_args.kwargs["json_data"] == {
        "query": "SELECT title FROM media",
        "target_id": "media_db",
        "max_rows": 2,
        "timeout_ms": 1500,
        "include_sql": True,
    }
    assert isinstance(result, Text2SQLResponse)
    assert result.guardrail.limit_injected is True
    assert result.rows == [{"title": "A"}, {"title": "B"}]
