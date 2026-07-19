"""Selector merge-cap tests for resolve_provider_model_options (ADR-019).

Discovered entries merge into dropdown options only when the provider's total
discovered catalog is at or below SELECTOR_MERGE_CAP; oversized catalogs stay
saved-list-only (they remain reachable via the search picker). The cap never
gates the transient current-model option.
"""

import pytest

from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import SELECTOR_MERGE_CAP
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import MergedModelEntry
from tldw_chatbook.UI.Screens.provider_model_resolution import (
    resolve_provider_model_options,
)


class _FakeScope:
    def __init__(self, entries):
        self._entries = entries

    async def merge_saved_and_discovered_models(self, *, mode, provider):
        return self._entries


class _FakeApp:
    def __init__(self, providers_models, entries):
        self.providers_models = providers_models
        self.llm_provider_catalog_scope_service = _FakeScope(entries)


def _entries(provider, ids, source="runtime_discovered"):
    return tuple(
        MergedModelEntry(
            provider=provider, provider_list_key=provider, model_id=m,
            display_name=m, source=source, capability_status="unknown", persisted=False,
        )
        for m in ids
    )


@pytest.mark.asyncio
async def test_small_discovered_catalog_merges_with_label():
    app = _FakeApp({"OpenAI": ["saved-1"]}, _entries("OpenAI", ["new-1"]))
    options = await resolve_provider_model_options(app, provider="OpenAI")
    assert [o.model_id for o in options] == ["saved-1", "new-1"]
    assert "runtime discovered" in options[1].label


@pytest.mark.asyncio
async def test_oversized_catalog_stays_saved_only():
    app = _FakeApp({"OpenRouter": ["saved-1"]},
                   _entries("OpenRouter", [f"v/m{i}" for i in range(60)]))
    options = await resolve_provider_model_options(app, provider="OpenRouter")
    assert [o.model_id for o in options] == ["saved-1"]


@pytest.mark.asyncio
async def test_catalog_at_cap_boundary_merges_in():
    app = _FakeApp({"OpenRouter": ["saved-1"]},
                   _entries("OpenRouter", [f"v/m{i}" for i in range(SELECTOR_MERGE_CAP)]))
    options = await resolve_provider_model_options(app, provider="OpenRouter")
    assert [o.model_id for o in options] == ["saved-1"] + [f"v/m{i}" for i in range(SELECTOR_MERGE_CAP)]


@pytest.mark.asyncio
async def test_catalog_one_over_cap_stays_saved_only():
    app = _FakeApp({"OpenRouter": ["saved-1"]},
                   _entries("OpenRouter", [f"v/m{i}" for i in range(SELECTOR_MERGE_CAP + 1)]))
    options = await resolve_provider_model_options(app, provider="OpenRouter")
    assert [o.model_id for o in options] == ["saved-1"]


@pytest.mark.asyncio
async def test_uncapped_returns_full_catalog_for_picker():
    app = _FakeApp({"OpenRouter": ["saved-1"]},
                   _entries("OpenRouter", [f"v/m{i}" for i in range(60)]))
    options = await resolve_provider_model_options(app, provider="OpenRouter", merge_cap=None)
    assert len(options) == 61


@pytest.mark.asyncio
async def test_current_model_inserted_as_transient_when_missing():
    app = _FakeApp({"OpenAI": ["saved-1"]}, ())
    options = await resolve_provider_model_options(
        app, provider="OpenAI", current_model="picked-elsewhere")
    assert options[0].model_id == "picked-elsewhere"


@pytest.mark.asyncio
async def test_oversized_catalog_still_includes_current_model_transient():
    app = _FakeApp({"OpenRouter": ["saved-1"]},
                   _entries("OpenRouter", [f"v/m{i}" for i in range(60)]))
    options = await resolve_provider_model_options(
        app, provider="OpenRouter", current_model="picked-elsewhere")
    assert [o.model_id for o in options] == ["picked-elsewhere", "saved-1"]
