"""Tests for the shared ADR-020 searchable model picker.

The picker searches the full (uncapped) provider catalog via
``resolve_provider_model_options(..., merge_cap=None)`` so models hidden by the
dropdown's SELECTOR_MERGE_CAP stay reachable. Results are mapped by
``option_index`` into the widget's ``_matches`` list; model IDs (which contain
``/`` and ``:``) must never be used as Option ids.
"""

import asyncio

import pytest

from textual import on
from textual.app import App
from textual.widgets import Button, Input, OptionList, Select

from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import SELECTOR_MERGE_CAP
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import MergedModelEntry
from tldw_chatbook.UI.Screens.provider_model_resolution import (
    ConsoleModelProvenance,
    ResolvedProviderModelOption,
)
from tldw_chatbook.Widgets.model_search_picker import (
    MODEL_ID_MAX_LENGTH,
    ModelSearchPicker,
)


class _FakeScope:
    """Minimal llm_provider_catalog_scope_service stand-in."""

    def __init__(self, entries):
        self._entries = entries
        self.calls = []

    async def merge_saved_and_discovered_models(self, *, mode, provider):
        self.calls.append({"mode": mode, "provider": provider})
        if isinstance(self._entries, BaseException):
            raise self._entries
        if isinstance(self._entries, dict):
            return self._entries.get(provider, ())
        return self._entries


class _BlockingScope(_FakeScope):
    """Catalog scope whose response can be released after the user types."""

    def __init__(self, entries):
        super().__init__(entries)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def merge_saved_and_discovered_models(self, *, mode, provider):
        self.calls.append({"mode": mode, "provider": provider})
        self.started.set()
        await self.release.wait()
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


def _provenance_option(
    model_id: str,
    provenance: ConsoleModelProvenance,
    *,
    verified: bool = False,
) -> ResolvedProviderModelOption:
    return ResolvedProviderModelOption(
        label=model_id,
        model_id=model_id,
        source="test",
        capability_status="known",
        persisted=False,
        provenance=provenance,
        verified_for_connection=verified,
    )


class PickerTestApp(App[None]):
    """Minimal host app exposing #chat-api-provider and the catalog scope."""

    def __init__(
        self,
        providers_models,
        entries,
        provider="OpenRouter",
        current_model=None,
    ):
        super().__init__()
        self.providers_models = providers_models
        self.llm_provider_catalog_scope_service = _FakeScope(entries)
        self._provider = provider
        self._current_model = current_model
        self.selected_models: list[str] = []

    def compose(self):
        yield Select(
            [("OpenRouter", "OpenRouter"), ("OpenAI", "OpenAI")],
            id="chat-api-provider",
            value=self._provider,
            allow_blank=False,
        )
        yield ModelSearchPicker(
            id="model-search-picker",
            current_model=self._current_model,
        )
        yield Button("Apply", id="apply")

    @on(ModelSearchPicker.ModelSelected)
    def _record_selected(self, event: ModelSearchPicker.ModelSelected) -> None:
        self.selected_models.append(event.model_id)


async def _set_query(pilot, query: str) -> None:
    search_input = pilot.app.query_one("#model-search-picker-input", Input)
    search_input.value = query
    await pilot.pause()


async def _wait_for_catalog(pilot) -> None:
    for _ in range(40):
        status = pilot.app.query_one("#model-search-picker-status")
        if str(status.renderable) != "Loading models...":
            return
        await pilot.pause(0.01)
    raise AssertionError("model picker catalog did not finish loading")


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
async def test_enter_commits_single_keyboard_filtered_result():
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"]),
    )
    async with app.run_test() as pilot:
        search_input = app.query_one("#model-search-picker-input", Input)
        app.set_focus(search_input)
        await pilot.pause()
        for character in "claude":
            await pilot.press(character)
        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one(ModelSearchPicker).value == "anthropic/claude-x"
        assert search_input.value == "anthropic/claude-x"
        assert app.selected_models == ["anthropic/claude-x"]


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
async def test_selection_commits_model_into_the_shared_input():
    """Picking a result leaves the committed model visible in the one control."""
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x"]),
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "claude")
        await _select_option(pilot, 0)
        search_input = app.query_one("#model-search-picker-input", Input)
        assert search_input.value == "anthropic/claude-x"
        assert app.selected_models == ["anthropic/claude-x"]


@pytest.mark.asyncio
async def test_typing_filters_cached_catalog_without_reloading_provider():
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"]),
    )
    async with app.run_test() as pilot:
        picker = app.query_one(ModelSearchPicker)
        await _set_query(pilot, "c")
        await _set_query(pilot, "cl")
        await _set_query(pilot, "claude")

        assert app.llm_provider_catalog_scope_service.calls == [
            {"mode": "local", "provider": "openrouter"}
        ]
        assert picker._load_counts == {"openrouter": 1}
        assert _result_prompts(_results(app)) == ["anthropic/claude-x"]


@pytest.mark.asyncio
async def test_query_typed_during_catalog_reload_populates_when_load_finishes():
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["initial/model"]),
    )
    async with app.run_test() as pilot:
        picker = app.query_one(ModelSearchPicker)
        blocking_scope = _BlockingScope(
            _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"])
        )
        app.llm_provider_catalog_scope_service = blocking_scope
        picker.refresh_provider("OpenRouter", force=True)
        await blocking_scope.started.wait()

        search_input = app.query_one("#model-search-picker-input", Input)
        app.set_focus(search_input)
        await pilot.pause()
        search_input.value = "claude"
        await pilot.pause()
        assert not _results(app).display

        blocking_scope.release.set()
        await _wait_for_catalog(pilot)
        await pilot.pause()

        assert _result_prompts(_results(app)) == ["anthropic/claude-x"]


@pytest.mark.asyncio
async def test_no_matches_status_names_recovery_actions():
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x"]),
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "does-not-exist")

        assert not _results(app).display
        status = app.query_one("#model-search-picker-status")
        assert str(status.renderable) == (
            "No matching models. Clear the filter or use Custom ID."
        )


@pytest.mark.asyncio
async def test_manual_discovery_overlay_is_searchable_without_catalog_reload():
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["anthropic/claude-x"]),
    )
    async with app.run_test() as pilot:
        picker = app.query_one(ModelSearchPicker)
        picker.set_discovered_models(
            "OpenRouter", ["local/probed-model", "local/probed-model"]
        )
        await _set_query(pilot, "probed")

        assert _result_prompts(_results(app)) == ["local/probed-model"]
        assert len(app.llm_provider_catalog_scope_service.calls) == 1


@pytest.mark.asyncio
async def test_escape_clears_filter_without_losing_committed_model():
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["anthropic/claude-x"]),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "claude")
        app.set_focus(app.query_one("#model-search-picker-input", Input))
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        picker = app.query_one(ModelSearchPicker)
        search_input = app.query_one("#model-search-picker-input", Input)
        assert picker.value == "saved-model"
        assert search_input.value == "saved-model"
        assert not _results(app).display


@pytest.mark.asyncio
async def test_blur_restores_committed_model_after_uncommitted_filter():
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["anthropic/claude-x"]),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        search_input = app.query_one("#model-search-picker-input", Input)
        app.set_focus(search_input)
        await pilot.pause()
        assert search_input.value == ""

        search_input.value = "claude"
        await pilot.pause()
        assert _results(app).display

        await pilot.click("#apply")
        for _ in range(20):
            await pilot.pause(0.01)
            if search_input.value == "saved-model":
                break

        picker = app.query_one(ModelSearchPicker)
        assert picker.value == "saved-model"
        assert search_input.value == "saved-model"
        assert not _results(app).display


@pytest.mark.asyncio
async def test_empty_catalog_names_custom_id_recovery():
    app = PickerTestApp({"OpenRouter": []}, ())
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        status = app.query_one("#model-search-picker-status")
        assert "No models reported" in str(status.renderable)
        assert "Custom ID" in str(status.renderable)


@pytest.mark.asyncio
async def test_unavailable_catalog_names_configured_and_custom_recovery():
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        RuntimeError("catalog offline"),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        status = app.query_one("#model-search-picker-status")
        assert "Catalog unavailable" in str(status.renderable)
        assert "configured model or Custom ID" in str(status.renderable)


@pytest.mark.asyncio
async def test_current_model_not_in_latest_catalog_is_explicit():
    app = PickerTestApp(
        {"OpenRouter": ["retired-model"]},
        _entries("OpenRouter", ["openai/current-model"]),
        current_model="retired-model",
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        status = app.query_one("#model-search-picker-status")
        assert "Current model is not in the latest catalog" in str(status.renderable)
        assert app.query_one(ModelSearchPicker).value == "retired-model"


@pytest.mark.asyncio
async def test_custom_id_escape_hatch_commits_typed_value():
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["openai/current-model"]),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        await pilot.click("#model-search-picker-custom")
        search_input = app.query_one("#model-search-picker-input", Input)
        search_input.value = "vendor/private-model"
        await pilot.pause()

        picker = app.query_one(ModelSearchPicker)
        assert picker.custom_mode is True
        assert picker.value == "vendor/private-model"
        assert "Custom model ID" in str(
            app.query_one("#model-search-picker-status").renderable
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_model_id",
    [
        "vendor/model\nsecond-line",
        "vendor/model\n",
        "x" * (MODEL_ID_MAX_LENGTH + 1),
        "vendor/<script-model",
    ],
)
async def test_custom_id_rejects_invalid_text(invalid_model_id):
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["openai/current-model"]),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        await pilot.click("#model-search-picker-custom")
        search_input = app.query_one("#model-search-picker-input", Input)
        search_input.value = invalid_model_id
        await pilot.pause()

        picker = app.query_one(ModelSearchPicker)
        assert picker.custom_mode is True
        assert picker.value is None
        assert "Invalid model ID" in str(
            app.query_one("#model-search-picker-status").renderable
        )


@pytest.mark.asyncio
async def test_provider_switch_uses_target_catalog_and_drops_previous_model():
    entries = {
        "openrouter": _entries("OpenRouter", ["anthropic/claude-x"]),
        "openai": _entries("OpenAI", ["gpt-5"]),
    }
    app = PickerTestApp(
        {"OpenRouter": [], "OpenAI": []},
        entries,
        current_model="anthropic/claude-x",
    )
    async with app.run_test() as pilot:
        picker = app.query_one(ModelSearchPicker)
        await picker.load_provider("OpenAI", current_model=None)
        await _set_query(pilot, "gpt")

        assert picker.value is None
        assert _result_prompts(_results(app)) == ["gpt-5"]
        assert "anthropic/claude-x" not in picker._catalog_model_ids()


@pytest.mark.asyncio
async def test_provenance_groups_are_disabled_and_select_by_option_identity() -> None:
    """Interleaved headings must never shift a model selection to another row."""
    app = PickerTestApp({"OpenRouter": []}, ())
    async with app.run_test() as pilot:
        picker = app.query_one(ModelSearchPicker)
        picker.set_provenance_options(
            "OpenRouter",
            (
                _provenance_option(
                    "saved/model", ConsoleModelProvenance.SAVED_FALLBACK
                ),
                _provenance_option(
                    "served/model",
                    ConsoleModelProvenance.SERVED_NOW,
                    verified=True,
                ),
                _provenance_option(
                    "catalog/model", ConsoleModelProvenance.CURRENT_CATALOG
                ),
                _provenance_option(
                    "custom/model", ConsoleModelProvenance.CUSTOM_UNVERIFIED
                ),
            ),
        )
        picker.focus_input()
        await pilot.pause()

        results = _results(app)
        assert _result_prompts(results) == [
            "Served now",
            "served/model",
            "Current catalog",
            "catalog/model",
            "Saved fallback",
            "saved/model",
            "Custom / unverified",
            "custom/model",
        ]
        assert [option.disabled for option in results.options] == [
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        await pilot.press("down")
        await pilot.pause()
        assert results.highlighted == 1

        await _select_option(pilot, 5)
        assert app.selected_models == ["saved/model"]


@pytest.mark.asyncio
async def test_unverified_served_now_option_is_grouped_as_unverified() -> None:
    """A provenance label alone must not assert endpoint-specific evidence."""
    app = PickerTestApp({"OpenRouter": []}, ())
    async with app.run_test() as pilot:
        picker = app.query_one(ModelSearchPicker)
        picker.set_provenance_options(
            "OpenRouter",
            (
                _provenance_option(
                    "stale-probe/model",
                    ConsoleModelProvenance.SERVED_NOW,
                    verified=False,
                ),
            ),
        )
        picker.focus_input()
        await pilot.pause()

        assert _result_prompts(_results(app)) == [
            "Custom / unverified",
            "stale-probe/model",
        ]


@pytest.mark.asyncio
async def test_provenance_filter_only_renders_non_empty_groups() -> None:
    """Filtering must not leave orphan headings for groups with no matches."""
    app = PickerTestApp({"OpenRouter": []}, ())
    async with app.run_test() as pilot:
        picker = app.query_one(ModelSearchPicker)
        picker.set_provenance_options(
            "OpenRouter",
            (
                _provenance_option(
                    "served/alpha", ConsoleModelProvenance.SERVED_NOW, verified=True
                ),
                _provenance_option(
                    "catalog/beta", ConsoleModelProvenance.CURRENT_CATALOG
                ),
            ),
        )
        await _set_query(pilot, "beta")

        assert _result_prompts(_results(app)) == [
            "Current catalog",
            "catalog/beta",
        ]


@pytest.mark.asyncio
async def test_provenance_model_ids_render_as_literal_text() -> None:
    """Provider model IDs must not be interpreted as Rich markup."""
    app = PickerTestApp({"OpenRouter": []}, ())
    async with app.run_test() as pilot:
        picker = app.query_one(ModelSearchPicker)
        picker.set_provenance_options(
            "OpenRouter",
            (
                _provenance_option(
                    "vendor/[bold]literal[/bold]",
                    ConsoleModelProvenance.CURRENT_CATALOG,
                ),
            ),
        )
        picker.focus_input()
        await pilot.pause()

        results = _results(app)
        assert _result_prompts(results) == [
            "Current catalog",
            "vendor/[bold]literal[/bold]",
        ]
        assert results.get_option_at_index(1).id is not None
