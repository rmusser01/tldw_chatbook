from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    TLDWAPIClient,
    TranslateRequest,
    TranslateResponse,
)


@pytest.mark.asyncio
async def test_translate_text_posts_to_translate_endpoint(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "translated_text": "Bonjour",
            "detected_source_language": "English",
            "target_language": "French",
            "model_used": "openai_default",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.translate_text(
        TranslateRequest(
            text="Hello",
            target_language="French",
            source_language="English",
        )
    )

    mocked.assert_awaited_once()
    assert mocked.await_args.args[:2] == ("POST", "/api/v1/translate")
    assert mocked.await_args.kwargs["json_data"] == {
        "text": "Hello",
        "target_language": "French",
        "source_language": "English",
    }
    assert isinstance(result, TranslateResponse)
    assert result.translated_text == "Bonjour"
    assert result.target_language == "French"
