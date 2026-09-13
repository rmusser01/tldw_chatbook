"""Selector authority and merge-cap tests (ADR-020).

Discovered entries merge into dropdown options only when the provider's total
discovered catalog is at or below SELECTOR_MERGE_CAP; oversized catalogs stay
saved-list-only (they remain reachable via the search picker). The cap never
gates the transient current-model option.
"""

import pytest

from tldw_chatbook.UI.Screens import provider_model_resolution
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
async def test_cloud_catalog_excludes_saved_only_models_and_keeps_live_label():
    options = await resolve_provider_model_options(
        {"OpenAI": ["saved-1"]},
        _FakeScope(_entries("OpenAI", ["new-1"])),
        provider="OpenAI",
    )
    assert [o.model_id for o in options] == ["new-1"]
    assert "runtime discovered" in options[0].label


@pytest.mark.asyncio
async def test_cloud_catalog_provenance_never_claims_the_exact_endpoint() -> None:
    """A provider-only cache cannot prove an unsaved draft endpoint served a model."""
    options = await resolve_provider_model_options(
        {"OpenAI": ["saved-old"]},
        _FakeScope(_entries("OpenAI", ["catalog-current"])),
        provider="OpenAI",
    )

    assert hasattr(provider_model_resolution, "ConsoleModelProvenance")
    provenance = provider_model_resolution.ConsoleModelProvenance
    assert [(item.model_id, item.provenance) for item in options] == [
        ("catalog-current", provenance.CURRENT_CATALOG),
    ]
    assert options[0].verified_for_connection is False
    assert options[0].provenance is not provenance.SERVED_NOW


class _EmptySnapshotScope:
    async def merge_saved_and_discovered_models(self, **_kwargs):
        return ()

    async def has_discovered_model_snapshot(self, **_kwargs):
        return True


@pytest.mark.asyncio
async def test_empty_cloud_snapshot_is_authoritative_over_saved_history():
    options = await resolve_provider_model_options(
        {"Anthropic": ["retired-model"]},
        _EmptySnapshotScope(),
        provider="Anthropic",
    )

    assert options == []


@pytest.mark.asyncio
async def test_empty_cloud_snapshot_preserves_current_model_as_unlisted():
    options = await resolve_provider_model_options(
        {"Anthropic": ["retired-model"]},
        _EmptySnapshotScope(),
        provider="Anthropic",
        current_model="retired-model",
    )

    assert [option.model_id for option in options] == ["retired-model"]
    assert options[0].source == "current_unlisted"


@pytest.mark.asyncio
async def test_current_unlisted_model_has_custom_unverified_provenance() -> None:
    options = await resolve_provider_model_options(
        {"Anthropic": ["retired-model"]},
        _EmptySnapshotScope(),
        provider="Anthropic",
        current_model="session-only",
    )

    assert hasattr(provider_model_resolution, "ConsoleModelProvenance")
    provenance = provider_model_resolution.ConsoleModelProvenance
    assert options[0].model_id == "session-only"
    assert options[0].provenance is provenance.CUSTOM_UNVERIFIED
    assert options[0].verified_for_connection is False


@pytest.mark.asyncio
async def test_oversized_cloud_catalog_does_not_restore_saved_only_models():
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved-1"]},
        _FakeScope(_entries("OpenRouter", [f"v/m{i}" for i in range(60)])),
        provider="OpenRouter",
    )
    assert options == []


@pytest.mark.asyncio
async def test_catalog_at_cap_boundary_merges_in():
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved-1"]},
        _FakeScope(
            _entries("OpenRouter", [f"v/m{i}" for i in range(SELECTOR_MERGE_CAP)])
        ),
        provider="OpenRouter",
    )
    assert [o.model_id for o in options] == [
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
    assert options == []


@pytest.mark.asyncio
async def test_uncapped_returns_full_catalog_for_picker():
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved-1"]},
        _FakeScope(_entries("OpenRouter", [f"v/m{i}" for i in range(60)])),
        provider="OpenRouter",
        merge_cap=None,
    )
    assert len(options) == 60


@pytest.mark.asyncio
async def test_qwencloud_full_catalog_remains_searchable_in_model_popover():
    discovered = _entries(
        "QwenCloud",
        [f"qwen-model-{index}" for index in range(SELECTOR_MERGE_CAP + 10)],
    )

    dropdown = await resolve_provider_model_options(
        {"QwenCloud": ["saved-model"]},
        _FakeScope(discovered),
        provider="QwenCloud",
    )
    searchable = await resolve_provider_model_options(
        {"QwenCloud": ["saved-model"]},
        _FakeScope(discovered),
        provider="QwenCloud",
        merge_cap=None,
    )

    assert dropdown == []
    assert [option.model_id for option in searchable] == [
        f"qwen-model-{index}" for index in range(SELECTOR_MERGE_CAP + 10)
    ]


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
    assert [o.model_id for o in options] == ["picked-elsewhere"]
    assert options[0].source == "current_unlisted"
    assert "not in latest catalog" in options[0].label


@pytest.mark.asyncio
async def test_endpoint_confirmed_saved_model_survives_cloud_pruning():
    entries = (
        MergedModelEntry(
            provider="Anthropic",
            provider_list_key="Anthropic",
            model_id="claude-current",
            display_name="claude-current",
            source="persisted_discovered",
            capability_status="known",
            persisted=True,
        ),
        MergedModelEntry(
            provider="Anthropic",
            provider_list_key="Anthropic",
            model_id="claude-retired",
            display_name="claude-retired",
            source="saved",
            capability_status="known",
            persisted=True,
        ),
    )

    options = await resolve_provider_model_options(
        {"Anthropic": ["claude-current", "claude-retired"]},
        _FakeScope(entries),
        provider="Anthropic",
    )

    assert [option.model_id for option in options] == ["claude-current"]
    assert options[0].source == "persisted_discovered"


@pytest.mark.asyncio
async def test_manual_local_discovery_remains_additive():
    entries = (
        MergedModelEntry(
            provider="llama_cpp",
            provider_list_key="llama_cpp",
            model_id="saved-alias",
            display_name="saved-alias",
            source="saved",
            capability_status="known",
            persisted=True,
        ),
        *_entries("llama_cpp", ["server-model"]),
    )

    options = await resolve_provider_model_options(
        {"llama_cpp": ["saved-alias"]},
        _FakeScope(entries),
        provider="llama_cpp",
    )

    assert [option.model_id for option in options] == [
        "saved-alias",
        "server-model",
    ]


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
