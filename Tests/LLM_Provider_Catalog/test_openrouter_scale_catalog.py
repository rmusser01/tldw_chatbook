"""TASK-32925: large catalogs (OpenRouter) must list fully and survive restarts.

Live measurement 2026-09-23: openrouter.ai/api/v1/models returned 456 models in
748,851 bytes (~1,756 bytes per record). The old bounds sat just above that:
discovery failed closed past 512 models or 1 MiB (listing NOTHING), and the
disk cache refused any snapshot over 100 models, so OpenRouter's list was
never persisted. These tests use an OpenRouter-shaped catalog three times
today's size so normal catalog growth trips a test, not users.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tldw_chatbook.Chat import local_server_discovery
from tldw_chatbook.LLM_Provider_Catalog import (
    model_discovery_disk_cache,
    openai_compatible_model_discovery,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_cache import (
    ModelDiscoveryCache,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
    ModelCatalogDiskStore,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    discover_openai_compatible_models,
)

OBSERVED_OPENROUTER_MODELS = 456
SCALE_MODELS = OBSERVED_OPENROUTER_MODELS * 3


def _openrouter_record(index: int) -> dict:
    """One record with the live API's field set and ~1.7 KB serialized size."""
    model_id = f"vendor-{index % 60}/model-{index}-instruct"
    return {
        "id": model_id,
        "canonical_slug": model_id,
        "hugging_face_id": None,
        "name": f"Vendor {index % 60}: Model {index} Instruct",
        "created": 1790000000 + index,
        "description": "A capable general model for chat, tools and code. " * 18,
        "context_length": 131072,
        "architecture": {
            "modality": "text+image->text",
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "tokenizer": "Other",
            "instruct_type": None,
        },
        "pricing": {"prompt": "0.0000003", "completion": "0.0000012"},
        "top_provider": {
            "context_length": 131072,
            "max_completion_tokens": 16384,
            "is_moderated": False,
        },
        "per_request_limits": None,
        "supported_parameters": [
            "max_tokens",
            "response_format",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ],
        "default_parameters": {},
        "supported_voices": None,
        "knowledge_cutoff": None,
        "expiration_date": None,
        "links": {"details": f"/api/v1/models/{model_id}/endpoints"},
    }


def _catalog_body(count: int) -> bytes:
    return json.dumps({"data": [_openrouter_record(i) for i in range(count)]}).encode()


@pytest.mark.asyncio
async def test_openrouter_scale_catalog_discovers_every_model():
    body = _catalog_body(SCALE_MODELS)
    assert len(body) > 1024 * 1024, "fixture must exceed the old 1 MiB bound"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        result = await discover_openai_compatible_models(
            provider="openrouter",
            provider_list_key="openrouter",
            endpoint="https://openrouter.ai/api/v1",
            api_key=None,
            client=client,
        )

    assert result.status == "success", result.error
    assert len(result.models) == SCALE_MODELS


def test_bounds_keep_headroom_over_the_live_catalog():
    discovery = openai_compatible_model_discovery
    per_model_bytes = 1756  # live average, 2026-09-23
    assert discovery.DISCOVERED_MODEL_MAX_COUNT >= 8 * OBSERVED_OPENROUTER_MODELS
    assert (
        discovery.MODEL_DISCOVERY_RESPONSE_MAX_BYTES
        >= discovery.DISCOVERED_MODEL_MAX_COUNT * per_model_bytes
    ), "a full-count catalog of live-sized records must fit the byte bound"
    assert (
        local_server_discovery.MODEL_PROBE_RESPONSE_MAX_BYTES
        >= discovery.MODEL_DISCOVERY_RESPONSE_MAX_BYTES
    ), "the Settings endpoint probe reads the same catalog body"
    # Every stage after discovery must hold what discovery accepts, or the
    # list is silently lost there instead (the old 100-model disk bound).
    assert (
        model_discovery_disk_cache.MODEL_CATALOG_DISK_MAX_MODELS_PER_ENTRY
        >= discovery.DISCOVERED_MODEL_MAX_COUNT
    )
    assert ModelDiscoveryCache()._max_models >= 2 * discovery.DISCOVERED_MODEL_MAX_COUNT


def test_openrouter_scale_list_survives_a_restart(tmp_path):
    ids = [_openrouter_record(i)["id"] for i in range(SCALE_MODELS)]
    endpoint = "https://openrouter.ai/api/v1"
    store = ModelCatalogDiskStore(tmp_path / "catalog.json")
    store.record("openrouter", endpoint, ids)
    store.save()

    cache = ModelDiscoveryCache()
    ModelCatalogDiskStore(tmp_path / "catalog.json").load_into(cache)

    assert [m.model_id for m in cache.list("openrouter", endpoint)] == ids


def test_a_full_list_does_not_evict_other_providers():
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        DiscoveredModel,
    )

    def models(provider: str, count: int):
        return [
            DiscoveredModel(
                provider=provider,
                provider_list_key=provider,
                model_id=f"m-{i}",
                display_name=f"m-{i}",
                source="runtime_discovered",
                endpoint_fingerprint="https://x.test/v1",
                discovered_at="2026-09-23T00:00:00Z",
            )
            for i in range(count)
        ]

    cache = ModelDiscoveryCache()
    full = openai_compatible_model_discovery.DISCOVERED_MODEL_MAX_COUNT
    cache.replace("openai", "https://x.test/v1", models("openai", 128))
    cache.replace("openrouter", "https://x.test/v1", models("openrouter", full))
    assert len(cache.list("openai", "https://x.test/v1")) == 128
    assert len(cache.list("openrouter", "https://x.test/v1")) == full


async def _anthropic_discovery(handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await discover_openai_compatible_models(
            provider="anthropic",
            provider_list_key="Anthropic",
            endpoint="https://api.anthropic.com/v1",
            api_key="sk-ant-test",
            client=client,
        )


def _anthropic_pages(total: int, page_size: int, *, never_ends: bool = False):
    """Handler serving `total` models in `page_size` pages via after_id."""

    def handler(request: httpx.Request) -> httpx.Response:
        after = request.url.params.get("after_id")
        start = 0 if after is None else int(after.rsplit("-", 1)[1]) + 1
        ids = [f"claude-{i}" for i in range(start, min(start + page_size, total))]
        more = never_ends or start + page_size < total
        return httpx.Response(
            200,
            json={
                "data": [{"id": i} for i in ids],
                "has_more": more,
                "last_id": ids[-1],
            },
        )

    return handler


@pytest.mark.asyncio
async def test_paginated_catalog_over_one_thousand_models_lists_every_model():
    """Qodo review on #2821: pagination stopped at ten 100-model pages, so a
    1,001-4,096 model catalog silently kept only its first 1,000."""
    result = await _anthropic_discovery(_anthropic_pages(1500, 100))
    assert result.status == "success", result.error
    assert len(result.models) == 1500


@pytest.mark.asyncio
async def test_pagination_that_never_finishes_fails_closed():
    """Short pages that keep saying has_more must not cache a partial list."""
    result = await _anthropic_discovery(_anthropic_pages(10**6, 10, never_ends=True))
    assert result.status == "error"
    assert result.models == ()


@pytest.mark.asyncio
async def test_paged_responses_share_one_byte_budget():
    """Qodo review on #2821: the byte bound applied per page, so many pages
    could hold several times MODEL_DISCOVERY_RESPONSE_MAX_BYTES at once."""
    budget = openai_compatible_model_discovery.MODEL_DISCOVERY_RESPONSE_MAX_BYTES
    pad = b" " * (budget // 3)  # JSON whitespace: each page fits, three do not

    def handler(request: httpx.Request) -> httpx.Response:
        after = request.url.params.get("after_id")
        page = 0 if after is None else int(after.rsplit("-", 1)[1]) + 1
        payload = {
            "data": [{"id": f"claude-{page}"}],
            "has_more": True,
            "last_id": f"claude-{page}",
        }
        return httpx.Response(200, content=json.dumps(payload).encode() + pad)

    result = await _anthropic_discovery(handler)
    assert result.status == "error"
    assert "large" in result.error.message.casefold()
