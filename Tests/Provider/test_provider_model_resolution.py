from __future__ import annotations

import pytest

from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    SELECTOR_MERGE_CAP,
)
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    MergedModelEntry,
)
from tldw_chatbook.UI.Screens.provider_model_resolution import (
    MODEL_CAPABILITY_UNKNOWN_WARNING,
    resolve_effective_provider_model,
    resolve_provider_model_options,
)


class CatalogScopeFixture:
    """Narrow catalog collaborator used by the explicit resolver boundary."""

    def __init__(self, entries: tuple[MergedModelEntry, ...]) -> None:
        self.entries = entries
        self.calls: list[tuple[str, str]] = []

    async def merge_saved_and_discovered_models(
        self,
        *,
        mode: str,
        provider: str,
    ) -> tuple[MergedModelEntry, ...]:
        self.calls.append((mode, provider))
        return self.entries


def _entry(
    model_id: str,
    *,
    source: str,
    capability_status: str = "unknown",
    persisted: bool = False,
) -> MergedModelEntry:
    return MergedModelEntry(
        provider="OpenAI",
        provider_list_key="OpenAI",
        model_id=model_id,
        display_name=model_id,
        source=source,
        capability_status=capability_status,
        persisted=persisted,
    )


def _runtime_entries(count: int) -> tuple[MergedModelEntry, ...]:
    return tuple(
        _entry(f"runtime-{index}", source="runtime_discovered")
        for index in range(count)
    )


def test_effective_values_use_settings_then_console_then_persisted_defaults() -> None:
    persisted_defaults = {"provider": "OpenAI", "model": "gpt-4o"}

    resolved = resolve_effective_provider_model(
        persisted_defaults,
        console_provider="Anthropic",
        console_model="claude-3-5-sonnet",
        settings_provider="Groq",
    )

    assert resolved.provider == "Groq"
    assert resolved.provider_source == "settings_draft"
    assert resolved.model == "claude-3-5-sonnet"
    assert resolved.model_source == "console_session"


def test_effective_values_fall_back_to_persisted_defaults_without_app_state() -> None:
    persisted_defaults = {"provider": "Ollama", "model": "qwen2.5"}

    resolved = resolve_effective_provider_model(persisted_defaults)

    assert resolved.provider == "Ollama"
    assert resolved.provider_source == "chat_defaults"
    assert resolved.model == "qwen2.5"
    assert resolved.model_source == "chat_defaults"


def test_effective_resolver_rejects_non_mapping_defaults() -> None:
    with pytest.raises(TypeError, match="mapping"):
        resolve_effective_provider_model(object())


@pytest.mark.asyncio
async def test_model_options_merge_explicit_saved_models_and_catalog_scope() -> None:
    scope = CatalogScopeFixture(
        (
            _entry(
                "gpt-4o",
                source="saved",
                capability_status="known",
                persisted=True,
            ),
            _entry("runtime-private", source="runtime_discovered"),
        )
    )

    options = await resolve_provider_model_options(
        {"OpenAI": ["gpt-4o"]},
        scope,
        provider=" openAI ",
    )

    assert scope.calls == [("local", "openai")]
    assert [option.model_id for option in options] == ["gpt-4o", "runtime-private"]
    runtime = options[1]
    assert runtime.source == "runtime_discovered"
    assert runtime.warning == MODEL_CAPABILITY_UNKNOWN_WARNING


@pytest.mark.asyncio
async def test_model_options_keep_discovery_out_of_capped_dropdowns() -> None:
    scope = CatalogScopeFixture(
        (
            _entry("runtime-one", source="runtime_discovered"),
            _entry("runtime-two", source="runtime_discovered"),
        )
    )

    dropdown = await resolve_provider_model_options(
        {"OpenAI": ["saved"]},
        scope,
        provider="OpenAI",
        merge_cap=1,
    )
    search = await resolve_provider_model_options(
        {"OpenAI": ["saved"]},
        scope,
        provider="OpenAI",
        merge_cap=None,
    )

    assert [option.model_id for option in dropdown] == ["saved"]
    assert [option.model_id for option in search] == [
        "saved",
        "runtime-one",
        "runtime-two",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("discovered_count", "expected_count"),
    [
        (SELECTOR_MERGE_CAP, SELECTOR_MERGE_CAP + 1),
        (SELECTOR_MERGE_CAP + 1, 1),
    ],
)
async def test_model_options_enforce_the_configured_merge_cap_boundary(
    discovered_count: int,
    expected_count: int,
) -> None:
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved"]},
        CatalogScopeFixture(_runtime_entries(discovered_count)),
        provider="OpenRouter",
    )

    assert len(options) == expected_count
    assert options[0].model_id == "saved"


@pytest.mark.asyncio
async def test_oversized_catalog_still_preserves_the_current_model() -> None:
    options = await resolve_provider_model_options(
        {"OpenRouter": ["saved"]},
        CatalogScopeFixture(_runtime_entries(SELECTOR_MERGE_CAP + 1)),
        provider="OpenRouter",
        current_model="session-only",
    )

    assert [option.model_id for option in options] == ["session-only", "saved"]


@pytest.mark.asyncio
async def test_model_options_preserve_current_model_without_catalog_service() -> None:
    options = await resolve_provider_model_options(
        {"OpenAI": ["saved"]},
        None,
        provider="OpenAI",
        current_model="session-only",
    )

    assert [option.model_id for option in options] == ["session-only", "saved"]


@pytest.mark.asyncio
async def test_model_options_reject_non_mapping_saved_catalog() -> None:
    with pytest.raises(TypeError, match="mapping"):
        await resolve_provider_model_options(
            object(),
            None,
            provider="OpenAI",
        )


def test_explicit_api_has_no_application_parameter() -> None:
    annotations = resolve_effective_provider_model.__annotations__

    assert "app_instance" not in annotations
    assert annotations["persisted_defaults"] == "Mapping[str, Any]"
