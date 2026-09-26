"""The prompt-collection raw-dict fallback covers a SCHEMA mismatch only.

Tier-2 review S06, P2 [D1]: the four `*_prompt_collection` methods wrap
`Model.model_validate(response)` in a bare `except Exception: return
response`. The union return type `PromptCollectionResponse | Dict[str,
Any]` is a designed contract and its callers in
`Prompt_Management/prompt_scope_service.py` genuinely handle both shapes
-- but `except Exception` also swallows any bug of *ours* raised during
validation, degrading it into a raw dict that then fails far from the
cause with no trace of the original error.

Narrowed to `ValidationError`: the server-schema-drift case keeps its
documented fallback; anything else propagates.
"""

from __future__ import annotations

import httpx
import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient

_MISMATCHED_BODY = {"unexpected": "shape", "collection_id": 7}


def _client() -> TLDWAPIClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_MISMATCHED_BODY)

    client = TLDWAPIClient("http://api.test", "secret")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


@pytest.mark.asyncio
async def test_a_schema_mismatch_still_falls_back_to_the_raw_dict():
    """The documented union return, unchanged."""
    result = await _client().get_prompt_collection(7)

    assert result == _MISMATCHED_BODY


@pytest.mark.asyncio
async def test_a_bug_during_validation_is_not_swallowed_into_a_raw_dict(
    monkeypatch,
):
    from tldw_chatbook.tldw_api import prompt_chatbook_schemas

    def boom(_payload):
        raise RuntimeError("model_validate is broken")

    monkeypatch.setattr(
        prompt_chatbook_schemas.PromptCollectionResponse, "model_validate", boom
    )

    with pytest.raises(RuntimeError, match="model_validate is broken"):
        await _client().get_prompt_collection(7)
