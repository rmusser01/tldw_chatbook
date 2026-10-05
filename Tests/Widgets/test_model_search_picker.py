"""Tests for the shared ADR-020 searchable model picker.

The picker searches the full (uncapped) provider catalog via
``resolve_provider_model_options(..., merge_cap=None)`` so models hidden by the
dropdown's SELECTOR_MERGE_CAP stay reachable. Results are mapped by
``option_index`` into the widget's ``_matches`` list; model IDs (which contain
``/`` and ``:``) must never be used as Option ids.
"""

import asyncio

import pytest

from textual import events, on
from textual.app import App
from textual.widgets import Button, Input, OptionList, Select
from textual.widgets._input import Selection

from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import SELECTOR_MERGE_CAP
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    MergedModelEntry,
)
from tldw_chatbook.UI.Screens.provider_model_resolution import (
    ConsoleModelProvenance,
    ResolvedProviderModelOption,
)
from tldw_chatbook.Widgets.model_search_picker import (
    CURRENT_MARK,
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

        # TASK-33007.9: the choice is selected like the Provider's, so the
        # next key filters afresh instead of landing at the filter's index.
        await pilot.press("o")
        assert search_input.value == "o"


@pytest.mark.asyncio
async def test_enter_on_an_exactly_typed_id_selects_it_so_the_next_key_searches_afresh():
    """TASK-33007.9 (review round 3): Enter on text that already equals the
    chosen id leaves the field's text unchanged, and that choice is selected
    too, so the next key replaces it."""
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"]),
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        search_input = app.query_one("#model-search-picker-input", Input)
        app.set_focus(search_input)
        await pilot.pause()
        await pilot.press(*"openai/gpt-y", "enter")
        await pilot.pause()

        assert app.selected_models == ["openai/gpt-y"]
        assert search_input.value == "openai/gpt-y"
        await pilot.press("o")
        assert search_input.value == "o"


@pytest.mark.asyncio
async def test_keyboard_result_commit_restores_visible_input_focus():
    """Enter from a focused flat result cannot strand focus on the hidden list."""
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"]),
    )
    async with app.run_test() as pilot:
        search = app.query_one("#model-search-picker-input", Input)
        search.focus()
        await pilot.pause()
        search.value = "claude"
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()

        assert app.query_one(ModelSearchPicker).value == "anthropic/claude-x"
        assert app.focused is search
        assert search.display and not search.disabled
        assert not _results(app).display


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
    """Host app exposing a non-default provider select id.

    The Alt+M popover used to be this host; Switch model (TASK-33004.4) no
    longer embeds the picker, so the id is a neutral one.
    """

    def __init__(self, providers_models, entries):
        super().__init__()
        self.providers_models = providers_models
        self.llm_provider_catalog_scope_service = _FakeScope(entries)
        self.selected_models: list[str] = []

    def compose(self):
        yield Select(
            [("OpenRouter", "OpenRouter")],
            id="custom-provider-select",
            value="OpenRouter",
            allow_blank=False,
        )
        yield ModelSearchPicker(
            id="model-search-picker",
            provider_select_id="#custom-provider-select",
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

        # TASK-33007.9: the restored model is selected, not left under the
        # filter's selection, so the next key replaces it whole.
        await pilot.press("o")
        assert search_input.value == "o"


@pytest.mark.asyncio
async def test_keyboard_escape_from_results_restores_model_and_input_focus():
    """Closing keyboard results must not strand focus on the hidden list."""
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["anthropic/claude-x"]),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        await _set_query(pilot, "claude")
        app.query_one("#model-search-picker-input", Input).focus()
        await pilot.pause()
        app.query_one("#model-search-picker-input", Input).value = "claude"
        await pilot.pause()
        await pilot.press("down")
        assert app.focused is _results(app)

        await pilot.press("escape")
        await pilot.pause()

        search = app.query_one("#model-search-picker-input", Input)
        assert app.focused is search
        assert search.value == "saved-model"
        assert not _results(app).display


@pytest.mark.asyncio
async def test_model_picker_accessible_metadata_names_search_and_actions():
    """Search, options, and the custom escape hatch expose bounded descriptions."""
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["anthropic/claude-x"]),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        search = app.query_one("#model-search-picker-input", Input)
        results = _results(app)
        custom = app.query_one("#model-search-picker-custom", Button)

        assert search.name == "model-search"
        assert search.tooltip == "Choose or search the model for this provider."
        assert results.name == "model-options"
        assert results.tooltip == "Matching models; use arrow keys and Enter to select."
        assert custom.name == "custom-model-id"
        assert custom.tooltip == "Enter an exact model ID that is not in the list."


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
        # TASK-33001.7 AC#5: focus no longer blanks the committed model.
        assert search_input.value == "saved-model"

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

        results.focus()
        await pilot.pause()
        await _select_option(pilot, 5)
        assert app.selected_models == ["saved/model"]
        assert picker.value == "saved/model"
        assert app.focused is app.query_one("#model-search-picker-input", Input)
        assert not results.display


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
@pytest.mark.parametrize(
    ("served_now", "group"),
    [(False, "Custom / unverified"), (True, "Served now")],
)
async def test_discovered_overlay_groups_as_served_now_only_when_its_host_says_so(
    served_now, group
) -> None:
    """TASK-33007.3: Settings drops its listing whenever the endpoint changes,
    so its overlay's new ids are served now; the default stays unverified."""
    app = PickerTestApp({"OpenRouter": []}, ())
    async with app.run_test() as pilot:
        picker = app.query_one(ModelSearchPicker)
        picker.set_provenance_options(
            "OpenRouter",
            (
                _provenance_option(
                    "catalog/model", ConsoleModelProvenance.CURRENT_CATALOG
                ),
            ),
        )
        picker.set_discovered_models(
            "OpenRouter", ["catalog/model", "listed/model"], served_now=served_now
        )
        picker.focus_input()
        await pilot.pause()

        prompts = _result_prompts(_results(app))
        assert prompts[prompts.index(group) + 1] == "listed/model", prompts
        assert prompts[prompts.index("Current catalog") + 1] == "catalog/model"
        assert picker.provenance_for_model("listed/model") == (
            ConsoleModelProvenance.SERVED_NOW
            if served_now
            else ConsoleModelProvenance.CUSTOM_UNVERIFIED
        )


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


@pytest.mark.asyncio
async def test_superseded_catalog_refresh_does_not_leak_unawaited_coroutine():
    import gc
    import warnings

    app = PickerTestApp(
        {"OpenRouter": ["old-model"], "OpenAI": ["current-model"]},
        (),
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        async with app.run_test() as pilot:
            await pilot.pause()
            picker = app.query_one(ModelSearchPicker)
            picker.refresh_provider("OpenRouter", current_model="old-model", force=True)
            picker.refresh_provider("OpenAI", current_model="current-model", force=True)
            await pilot.pause()
            assert picker._provider == "OpenAI"
            assert picker.value == "current-model"
        gc.collect()
    assert not [
        warning for warning in recorded if "was never awaited" in str(warning.message)
    ]


def _status_text(app) -> str:
    return str(app.query_one("#model-search-picker-status").renderable)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_ids", "expected"),
    [
        (["only/model"], "1 model available. Type to filter."),
        (["a/one", "b/two"], "2 models available. Type to filter."),
    ],
)
async def test_catalog_status_counts_models_in_singular_and_plural(model_ids, expected):
    """TASK-33001.7 AC#1: one model is "1 model", never "1 models"."""
    app = PickerTestApp({"OpenRouter": []}, _entries("OpenRouter", model_ids))
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        assert _status_text(app) == expected


@pytest.mark.asyncio
async def test_truncated_results_say_how_many_matched_and_that_typing_narrows():
    """TASK-33001.7 AC#2: the 20-row cap is never silent."""
    cap = ModelSearchPicker.MAX_RESULTS
    model_ids = [f"vendor/m{index:02d}" for index in range(cap + 5)]
    app = PickerTestApp({"OpenRouter": []}, _entries("OpenRouter", model_ids))
    async with app.run_test() as pilot:
        await _set_query(pilot, "vendor")
        assert len(_result_prompts(_results(app))) == cap
        assert _status_text(app) == (
            f"Showing {cap} of {cap + 5} matching models. Type to narrow the list."
        )

        # Narrowing under the cap drops the note: the list is complete again.
        await _set_query(pilot, "vendor/m2")
        assert _result_prompts(_results(app)) == [
            f"vendor/m{index}" for index in range(20, cap + 5)
        ]
        assert _status_text(app) == f"{cap + 5} models available. Type to filter."


_OVER_CAP_IDS = [
    f"vendor/m{index:02d}" for index in range(ModelSearchPicker.MAX_RESULTS + 5)
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "entries", "current", "discovered", "expected"),
    [
        (
            ["retired-model"],
            _entries("OpenRouter", _OVER_CAP_IDS),
            "retired-model",
            (),
            "Current model is not in the latest catalog. Choose another or keep it.",
        ),
        (
            # The catalog service failed; an endpoint probe's overlay still
            # fills the list past the cap.
            [],
            RuntimeError("catalog offline"),
            None,
            _OVER_CAP_IDS,
            "Catalog unavailable. Use a configured model or Custom ID.",
        ),
        (
            _OVER_CAP_IDS,
            (),
            _OVER_CAP_IDS[0],
            (),
            f"Live catalog unavailable. Showing {len(_OVER_CAP_IDS)} configured models.",
        ),
    ],
    ids=["current-unlisted", "load-error", "saved-only"],
)
async def test_catalog_health_warning_outranks_the_result_cap_note_on_focus(
    configured, entries, current, discovered, expected
):
    """Final-review I2: focus opens the capped full list, but on a catalog
    larger than the cap the health warning must still show; the cap note
    only replaces the plain "N models available" line."""
    app = PickerTestApp({"OpenRouter": configured}, entries, current_model=current)
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        picker = app.query_one(ModelSearchPicker)
        if discovered:
            picker.set_discovered_models("OpenRouter", discovered)
        assert _status_text(app) == expected
        picker.focus_input()
        await pilot.pause()

        assert len(_result_prompts(_results(app))) == ModelSearchPicker.MAX_RESULTS
        assert _status_text(app) == expected


@pytest.mark.asyncio
async def test_truncated_provenance_results_say_how_many_matched():
    """The grouped (provenance) list obeys the same cap and says so."""
    cap = ModelSearchPicker.MAX_RESULTS
    app = PickerTestApp({"OpenRouter": []}, ())
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        picker = app.query_one(ModelSearchPicker)
        picker.set_provenance_options(
            "OpenRouter",
            tuple(
                _provenance_option(
                    f"vendor/m{index:02d}", ConsoleModelProvenance.CURRENT_CATALOG
                )
                for index in range(cap + 3)
            ),
        )
        picker.focus_input()
        await pilot.pause()

        assert _status_text(app) == (
            f"Showing {cap} of {cap + 3} matching models. Type to narrow the list."
        )


@pytest.mark.asyncio
async def test_focus_keeps_committed_model_selected_so_typing_replaces_it():
    """TASK-33001.7 AC#5: focus opens the full list without blanking the value."""
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"]),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        search_input = app.query_one("#model-search-picker-input", Input)
        app.set_focus(search_input)
        await pilot.pause()

        assert search_input.value == "saved-model"
        # TASK-33007.9: the whole id is selected with the caret at its head.
        assert search_input.selection == Selection(len("saved-model"), 0)
        assert "openai/gpt-y" in _result_prompts(_results(app))

        await pilot.press("c", "l")
        assert search_input.value == "cl"
        assert _result_prompts(_results(app)) == ["anthropic/claude-x"]


@pytest.mark.asyncio
async def test_focusing_click_selects_committed_model_so_typing_replaces_it():
    """TASK-33001.7: a click that focuses the field behaves like Tab.

    Input.select_on_focus alone loses to Input._on_mouse_down, which moves
    the caret to the click point, so click-then-type edited the committed
    model instead of searching.
    """
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"]),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        search_input = app.query_one("#model-search-picker-input", Input)
        await pilot.click("#model-search-picker-input", offset=(4, 1))
        await pilot.pause()
        assert search_input.value == "saved-model"
        assert search_input.selection == Selection(len("saved-model"), 0)

        await pilot.press("c", "l")
        assert search_input.value == "cl"

        # A second click on the focused field places the caret AT the click
        # point (Input._on_mouse_down), not merely somewhere empty: typing
        # left the caret at the end, so the click must land inside the text.
        clicked_index = 1
        assert 0 < clicked_index < len(search_input.value)
        click_x = search_input.gutter.left + clicked_index
        await pilot.click("#model-search-picker-input", offset=(click_x, 1))
        await pilot.pause()
        assert search_input.selection == Selection.cursor(clicked_index)


def _painted_text(app) -> str:
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


#: Wider than the field at the default 80-column test size.
_WIDE_HEAD = "head-of-the-id/"
_WIDE_MODEL = _WIDE_HEAD + "x" * 90 + "/its-tail"


def _painted_field(app, field) -> str:
    """The cells the compositor paints for ``field``'s text area."""
    area = field.content_region
    return _painted_text(app).splitlines()[area.y][area.x : area.right]


async def _until(pilot, condition) -> None:
    """Pause until ``condition()`` holds (the picker's blur timer is 50 ms)."""
    for _ in range(100):
        if condition():
            return
        await pilot.pause(0.02)


async def _leave_the_picker(pilot, field) -> None:
    pilot.app.query_one("#apply", Button).focus()
    await _until(pilot, lambda: not field.has_focus)
    await pilot.pause(0.2)  # the picker's blur timer


@pytest.mark.asyncio
async def test_a_model_id_wider_than_the_field_reads_from_its_head():
    """TASK-33007.9 (review round 2): Input keeps a cell for the caret after
    the last character and scrolls to it, so a select-all that ended there
    pushed the head of a wide model id out of view. Focus now selects with
    the caret at the head, and a field without focus comes to rest on the
    head: after the caret was left at the end, after a value is set while it
    rests, and for a Custom ID."""
    shorter = _WIDE_MODEL[:-20]
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", [_WIDE_MODEL, shorter]),
        current_model=_WIDE_MODEL,
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        picker = app.query_one(ModelSearchPicker)
        field = app.query_one("#model-search-picker-input", Input)
        assert field.content_region.width < len(shorter)
        assert _painted_field(app, field).startswith(_WIDE_HEAD)  # as mounted

        picker.focus_input()
        await pilot.pause()
        assert field.selection == Selection(len(_WIDE_MODEL), 0)
        assert _painted_field(app, field).startswith(_WIDE_HEAD)

        # The caret is the user's to move while the field is theirs.
        await pilot.press("end")
        await pilot.pause()
        assert not _painted_field(app, field).startswith(_WIDE_HEAD)
        await _leave_the_picker(pilot, field)
        assert _painted_field(app, field).startswith(_WIDE_HEAD)

        # A shorter id set at rest moves the caret, which Input scrolls to.
        picker.set_model_value(shorter)
        await pilot.pause()
        assert field.value == shorter
        assert _painted_field(app, field).startswith(_WIDE_HEAD)

        picker.toggle_custom_mode()
        await pilot.pause()
        await pilot.press("end", "z")
        await pilot.pause()
        assert field.value == shorter + "z"
        assert not _painted_field(app, field).startswith(_WIDE_HEAD)
        await _leave_the_picker(pilot, field)
        assert field.value == shorter + "z"
        assert _painted_field(app, field).startswith(_WIDE_HEAD)


@pytest.mark.asyncio
async def test_a_returning_window_focus_selects_the_model_unless_text_is_being_typed():
    """TASK-33007.9 (review round 2): Input keeps the caret when the window
    regains focus. Here the blur has dropped the filter and put the committed
    model back by then, so that focus selects it like any other and the next
    key replaces it. Text still being typed keeps its caret: a filter when the
    window was away for less than the blur timer, with the list it filtered,
    and a Custom ID always, scrolled back into view after the blur showed the
    head of an id wider than the field."""
    app = PickerTestApp(
        {"OpenRouter": ["saved-model"]},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y"]),
        current_model="saved-model",
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        picker = app.query_one(ModelSearchPicker)
        field = app.query_one("#model-search-picker-input", Input)
        picker.focus_input()
        await pilot.pause()
        await pilot.press("c", "l")
        assert field.value == "cl" and _results(app).display

        # Away for less than the blur timer: both events are queued at once.
        app.post_message(events.AppBlur())
        app.post_message(events.AppFocus())
        await pilot.pause(0.3)
        assert field.has_focus and _results(app).display
        assert field.value == "cl"
        assert field.selection == Selection.cursor(2)
        assert _result_prompts(_results(app)) == ["anthropic/claude-x"]

        # Round 3: a filter that matches nothing hides the list, and is still
        # being typed: it keeps its caret, the hidden list and its status, on
        # a short window blur and on Tab out to Custom ID and Shift+Tab back.
        no_match = "No matching models. Clear the filter or use Custom ID."
        await pilot.press("backspace", "backspace", "z", "z", "z")
        assert field.value == "zzz" and not _results(app).display
        app.post_message(events.AppBlur())
        app.post_message(events.AppFocus())
        await pilot.pause(0.3)
        assert field.has_focus and field.value == "zzz"
        assert field.selection == Selection.cursor(3)
        assert not _results(app).display and _status_text(app) == no_match
        await pilot.press("q")
        assert field.value == "zzzq"
        await pilot.press("tab")
        await _until(pilot, lambda: not field.has_focus)
        await pilot.press("shift+tab")
        await _until(pilot, lambda: field.has_focus)
        await pilot.pause()
        assert field.value == "zzzq" and not _results(app).display
        assert _status_text(app) == no_match

        app.post_message(events.AppBlur())
        await _until(
            pilot, lambda: field.value == "saved-model" and not _results(app).display
        )
        assert field.value == "saved-model" and not field.has_focus
        app.post_message(events.AppFocus())
        await _until(pilot, lambda: field.has_focus)
        await pilot.pause()
        assert field.selection == Selection(len("saved-model"), 0)
        await pilot.press("o")
        assert field.value == "o"

        await pilot.click("#model-search-picker-custom")
        await _until(pilot, lambda: field.has_focus)
        await pilot.press(*_WIDE_MODEL)
        assert picker.custom_mode and field.value == _WIDE_MODEL
        assert field.content_region.width < len(_WIDE_MODEL)
        assert _painted_field(app, field).rstrip().endswith("/its-tail")
        app.post_message(events.AppBlur())
        await _until(pilot, lambda: not field.has_focus)
        await pilot.pause(0.2)
        assert _painted_field(app, field).startswith(_WIDE_HEAD)  # at rest
        app.post_message(events.AppFocus())
        await _until(pilot, lambda: field.has_focus)
        await pilot.pause()
        assert field.value == _WIDE_MODEL
        assert field.selection == Selection.cursor(len(_WIDE_MODEL))
        # The view is back on the caret, and so is the terminal cursor: an end
        # caret sits in the cell just past the last painted character.
        assert _painted_field(app, field).rstrip().endswith("/its-tail")
        assert app.cursor_position == field.cursor_screen_offset
        area = field.content_region
        assert area.x <= app.cursor_position.x <= area.right


@pytest.mark.asyncio
async def test_the_committed_model_is_marked_current_in_words():
    """TASK-33004.6 AC#3: the committed model's result says ● CURRENT in its
    text, so the mark reads without colour; no other result carries it, and
    the mark moves with the next commit."""
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["anthropic/claude-x", "openai/gpt-y", "vendor/z"]),
        current_model="openai/gpt-y",
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        picker = app.query_one(ModelSearchPicker)
        picker.focus_input()
        await pilot.pause()

        marked = f"openai/gpt-y  {CURRENT_MARK}"
        prompts = _result_prompts(_results(app))
        assert [prompt for prompt in prompts if CURRENT_MARK in prompt] == [marked]
        assert "anthropic/claude-x" in prompts and "vendor/z" in prompts
        assert marked in _painted_text(app)

        await _select_option(pilot, prompts.index("vendor/z"))
        app.query_one("#apply", Button).focus()
        await pilot.pause()
        picker.focus_input()
        await pilot.pause()
        assert [
            prompt
            for prompt in _result_prompts(_results(app))
            if CURRENT_MARK in prompt
        ] == [f"vendor/z  {CURRENT_MARK}"]


@pytest.mark.asyncio
async def test_down_highlights_the_committed_model_first():
    """TASK-33004.6 AC#3: Down lands on the committed model, not the first row;
    when a filter hides it, Down falls back to the first result."""
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", ["a/one", "b/two", "c/three"]),
        current_model="c/three",
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        picker = app.query_one(ModelSearchPicker)
        picker.focus_input()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()

        results = _results(app)
        assert app.focused is results
        prompts = _result_prompts(results)
        assert prompts[results.highlighted] == f"c/three  {CURRENT_MARK}"
        assert results.highlighted != 0

        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("t", "w", "o", "down")
        await pilot.pause()
        assert _result_prompts(results) == ["b/two"]
        assert results.highlighted == 0


@pytest.mark.asyncio
async def test_grouped_results_mark_and_highlight_the_committed_model():
    """TASK-33004.6 AC#3 for the provenance-grouped list Chat settings shows:
    the committed model is marked in its group and Down skips headings to it."""
    app = PickerTestApp({"OpenRouter": []}, (), current_model="saved/model")
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        picker = app.query_one(ModelSearchPicker)
        picker.set_provenance_options(
            "OpenRouter",
            (
                _provenance_option(
                    "served/model", ConsoleModelProvenance.SERVED_NOW, verified=True
                ),
                _provenance_option(
                    "catalog/model", ConsoleModelProvenance.CURRENT_CATALOG
                ),
                _provenance_option(
                    "saved/model", ConsoleModelProvenance.SAVED_FALLBACK
                ),
            ),
        )
        picker.focus_input()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()

        results = _results(app)
        assert _result_prompts(results) == [
            "Served now",
            "served/model",
            "Current catalog",
            "catalog/model",
            "Saved fallback",
            f"saved/model  {CURRENT_MARK}",
        ]
        assert results.highlighted == 5
        await pilot.press("enter")
        await pilot.pause()
        assert app.selected_models == ["saved/model"]


@pytest.mark.asyncio
async def test_a_committed_model_past_the_result_cap_stays_marked_and_first():
    """TASK-33004.6 AC#3 on a large catalog (review fix round 1): a committed
    model past MAX_RESULTS takes the last visible slot, in the flat and the
    grouped list, so it keeps its mark and Down still lands on it."""
    cap = ModelSearchPicker.MAX_RESULTS
    model_ids = _OVER_CAP_IDS
    committed = model_ids[-1]
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", model_ids),
        current_model=committed,
    )
    async with app.run_test() as pilot:
        await _wait_for_catalog(pilot)
        picker = app.query_one(ModelSearchPicker)
        picker.focus_input()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()

        results = _results(app)
        prompts = _result_prompts(results)
        assert len(prompts) == cap
        assert prompts[: cap - 1] == model_ids[: cap - 1]
        assert prompts[-1] == f"{committed}  {CURRENT_MARK}"
        assert results.highlighted == cap - 1
        await pilot.press("enter")
        await pilot.pause()
        assert app.selected_models == [committed]

        # Grouped (Chat settings): the unlisted current model is appended last.
        app.query_one("#apply", Button).focus()
        await pilot.pause()
        picker.set_provenance_options(
            "OpenRouter",
            tuple(
                _provenance_option(model_id, ConsoleModelProvenance.CURRENT_CATALOG)
                for model_id in model_ids[:-1]
            )
            + (_provenance_option(committed, ConsoleModelProvenance.SAVED_FALLBACK),),
        )
        picker.focus_input()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        prompts = _result_prompts(results)
        assert prompts[-2:] == ["Saved fallback", f"{committed}  {CURRENT_MARK}"]
        assert len(picker._matches) == cap
        assert results.highlighted == len(prompts) - 1
