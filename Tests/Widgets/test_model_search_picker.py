"""Tests for ModelSearchPicker (ADR-019 uncapped catalog type-to-filter search).

The picker searches the full (uncapped) provider catalog via
``resolve_provider_model_options(..., merge_cap=None)`` so models hidden by the
dropdown's SELECTOR_MERGE_CAP stay reachable. Results are mapped by
``option_index`` into the widget's ``_matches`` list; model IDs (which contain
``/`` and ``:``) must never be used as Option ids.
"""

import pytest

from textual import on
from textual.app import App
from textual.widgets import Input, OptionList, Select

from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import SELECTOR_MERGE_CAP
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import MergedModelEntry
from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker


class _FakeScope:
    """Minimal llm_provider_catalog_scope_service stand-in."""

    def __init__(self, entries):
        self._entries = entries

    async def merge_saved_and_discovered_models(self, *, mode, provider):
        return self._entries


def _entries(provider, ids):
    return tuple(
        MergedModelEntry(
            provider=provider,
            provider_list_key=provider,
            model_id=m,
            display_name=m,
            source="runtime_discovered",
            capability_status="unknown",
            persisted=False,
        )
        for m in ids
    )


class PickerTestApp(App[None]):
    """Minimal host app exposing #chat-api-provider and the catalog scope."""

    def __init__(self, providers_models, entries, provider="OpenRouter"):
        super().__init__()
        self.providers_models = providers_models
        self.llm_provider_catalog_scope_service = _FakeScope(entries)
        self._provider = provider
        self.selected_models: list[str] = []

    def compose(self):
        yield Select(
            [("OpenRouter", "OpenRouter"), ("OpenAI", "OpenAI")],
            id="chat-api-provider",
            value=self._provider,
            allow_blank=False,
        )
        yield ModelSearchPicker(id="model-search-picker")

    @on(ModelSearchPicker.ModelSelected)
    def _record_selected(self, event: ModelSearchPicker.ModelSelected) -> None:
        self.selected_models.append(event.model_id)


async def _set_query(pilot, query: str) -> None:
    search_input = pilot.app.query_one("#model-search-picker-input", Input)
    search_input.value = query
    await pilot.pause()


def _results(app) -> OptionList:
    return app.query_one("#model-search-picker-results", OptionList)


def _result_prompts(results: OptionList) -> list[str]:
    return [str(option.prompt) for option in results.options]


async def _select_option(pilot, index: int) -> None:
    results = _results(pilot.app)
    option = results.get_option_at_index(index)
    results.post_message(OptionList.OptionSelected(results, option, index))
    await pilot.pause()


@pytest.mark.asyncio
async def test_substring_filter_matches_provider_prefix():
    """Query 'anthropic' in an OpenRouter catalog shows only anthropic/ IDs."""
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"]),
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "anthropic")
        results = _results(app)
        assert results.display
        assert _result_prompts(results) == ["anthropic/claude-x"]
        assert app.query_one(ModelSearchPicker)._matches == ["anthropic/claude-x"]


@pytest.mark.asyncio
async def test_empty_query_hides_results():
    """Clearing the query hides the results list and clears options."""
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x"]),
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "claude")
        assert _results(app).display
        await _set_query(pilot, "")
        results = _results(app)
        assert not results.display
        assert results.option_count == 0
        assert app.query_one(ModelSearchPicker)._matches == []


@pytest.mark.asyncio
async def test_results_hidden_on_mount():
    """Results list starts hidden before any query."""
    app = PickerTestApp({"OpenRouter": []}, ())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not _results(app).display


@pytest.mark.asyncio
async def test_selection_posts_model_selected_with_model_id():
    """Picking a result posts ModelSelected with the model ID from _matches."""
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x", "anthropic/claude-y"]),
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "anthropic")
        await _select_option(pilot, 1)
        assert app.selected_models == ["anthropic/claude-y"]


@pytest.mark.asyncio
async def test_model_ids_never_used_as_option_ids():
    """Model IDs contain '/' and ':' (invalid DOM ids) — Option ids stay None."""
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-3.7:beta", "anthropic/claude-x"]),
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "anthropic")
        results = _results(app)
        assert results.option_count == 2
        for option in results.options:
            assert option.id is None


@pytest.mark.asyncio
async def test_over_cap_catalog_fully_searchable():
    """Catalogs over SELECTOR_MERGE_CAP are fully searchable (merge_cap=None)."""
    deep_ids = [f"vendor/m{i:02d}" for i in range(SELECTOR_MERGE_CAP + 10)]
    target = deep_ids[-1]
    app = PickerTestApp({"OpenRouter": []}, _entries("OpenRouter", deep_ids))
    async with app.run_test() as pilot:
        await _set_query(pilot, target.lower())
        results = _results(app)
        assert results.display
        assert _result_prompts(results) == [target]
        await _select_option(pilot, 0)
        assert app.selected_models == [target]


class PickerCustomSelectApp(App[None]):
    """Host app exposing a non-default provider select id (popover-style)."""

    def __init__(self, providers_models, entries):
        super().__init__()
        self.providers_models = providers_models
        self.llm_provider_catalog_scope_service = _FakeScope(entries)
        self.selected_models: list[str] = []

    def compose(self):
        yield Select(
            [("OpenRouter", "OpenRouter")],
            id="console-popover-provider",
            value="OpenRouter",
            allow_blank=False,
        )
        yield ModelSearchPicker(
            id="model-search-picker",
            provider_select_id="#console-popover-provider",
        )

    @on(ModelSearchPicker.ModelSelected)
    def _record_selected(self, event: ModelSearchPicker.ModelSelected) -> None:
        self.selected_models.append(event.model_id)


@pytest.mark.asyncio
async def test_custom_provider_select_id():
    """A custom provider_select_id points the picker at a different select."""
    app = PickerCustomSelectApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"]),
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "openai")
        results = _results(app)
        assert results.display
        assert _result_prompts(results) == ["openai/gpt-y"]
        await _select_option(pilot, 0)
        assert app.selected_models == ["openai/gpt-y"]


@pytest.mark.asyncio
async def test_selection_clears_search_input():
    """Picking a result resets the search input to empty."""
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x"]),
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "claude")
        await _select_option(pilot, 0)
        search_input = app.query_one("#model-search-picker-input", Input)
        assert search_input.value == ""
        assert app.selected_models == ["anthropic/claude-x"]
