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


def _entries(provider, ids, source="runtime_discovered"):
    return tuple(
        MergedModelEntry(
            provider=provider,
            provider_list_key=provider,
            model_id=m,
            display_name=m,
            source=source,
            capability_status="unknown",
            persisted=False,
        )
        for m in ids
    )


@pytest.mark.asyncio
async def test_small_discovered_catalog_merges_with_label():
    options = await resolve_provider_model_options(
        {"OpenAI": ["saved-1"]},
        _FakeScope(_entries("OpenAI", ["new-1"])),
        provider="OpenAI",
    )
    assert [o.model_id for o in options] == ["saved-1", "new-1"]
    assert "runtime discovered" in options[1].label


@pytest.mark.asyncio
async def test_oversized_catalog_stays_saved_only():
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved-1"]},
        _FakeScope(_entries("OpenRouter", [f"v/m{i}" for i in range(60)])),
        provider="OpenRouter",
    )
    assert [o.model_id for o in options] == ["saved-1"]


@pytest.mark.asyncio
async def test_catalog_at_cap_boundary_merges_in():
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved-1"]},
        _FakeScope(
            _entries("OpenRouter", [f"v/m{i}" for i in range(SELECTOR_MERGE_CAP)])
        ),
        provider="OpenRouter",
    )
    assert [o.model_id for o in options] == ["saved-1"] + [
        f"v/m{i}" for i in range(SELECTOR_MERGE_CAP)
    ]


@pytest.mark.asyncio
async def test_catalog_one_over_cap_stays_saved_only():
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved-1"]},
        _FakeScope(
            _entries(
                "OpenRouter",
                [f"v/m{i}" for i in range(SELECTOR_MERGE_CAP + 1)],
            )
        ),
        provider="OpenRouter",
    )
    assert [o.model_id for o in options] == ["saved-1"]


@pytest.mark.asyncio
async def test_uncapped_returns_full_catalog_for_picker():
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved-1"]},
        _FakeScope(_entries("OpenRouter", [f"v/m{i}" for i in range(60)])),
        provider="OpenRouter",
        merge_cap=None,
    )
    assert len(options) == 61


@pytest.mark.asyncio
async def test_current_model_inserted_as_transient_when_missing():
    options = await resolve_provider_model_options(
        {"OpenAI": ["saved-1"]},
        _FakeScope(()),
        provider="OpenAI",
        current_model="picked-elsewhere",
    )
    assert options[0].model_id == "picked-elsewhere"


@pytest.mark.asyncio
async def test_oversized_catalog_still_includes_current_model_transient():
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved-1"]},
        _FakeScope(_entries("OpenRouter", [f"v/m{i}" for i in range(60)])),
        provider="OpenRouter",
        current_model="picked-elsewhere",
    )
    assert [o.model_id for o in options] == ["picked-elsewhere", "saved-1"]


class _RaisingScope:
    """Local catalog that does not cover the provider (FB-09)."""

    async def merge_saved_and_discovered_models(self, *, mode, provider):
        raise ValueError(f"Unknown or ambiguous local LLM provider: {provider}")


@pytest.mark.asyncio
async def test_empty_local_catalog_degrades_to_saved_only_without_raising():
    """FB-09 (TASK-2154.18): a provider the local catalog does not cover
    (a cloud-only provider, or an empty local catalog) makes the local
    service raise ValueError("Unknown or ambiguous local LLM provider").
    The merge must degrade to saved-only quietly instead of tracebacking
    through the Alt+M popover path (logged via logger.exception there) or
    the model search picker (which has no exception handler at all)."""
    options = await resolve_provider_model_options(
        {"OpenAI": ["saved-1"]},
        _RaisingScope(),
        provider="OpenAI",
    )
    assert [o.model_id for o in options] == ["saved-1"]


class _BuggyScope:
    async def merge_saved_and_discovered_models(self, *, mode, provider):
        raise RuntimeError("catalog exploded")


@pytest.mark.asyncio
async def test_unexpected_merge_errors_still_propagate():
    """Only the known 'provider absent from the local catalog' ValueError
    degrades quietly; genuine catalog bugs must keep propagating to the
    caller's traceback logging (FB-09 review guard)."""
    with pytest.raises(RuntimeError, match="catalog exploded"):
        await resolve_provider_model_options(
            {"OpenAI": ["saved-1"]},
            _BuggyScope(),
            provider="OpenAI",
        )
