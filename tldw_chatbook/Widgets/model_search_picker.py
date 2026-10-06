"""Controlled searchable model picker over the full provider catalog."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.geometry import Region
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.input import Selection
from textual.widgets.option_list import Option

from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
)
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input
from tldw_chatbook.UI.Screens.provider_model_resolution import (
    ConsoleModelProvenance,
    ResolvedProviderModelOption,
)
from tldw_chatbook.Widgets.status_line import set_status_line


_CLOUD_CATALOG_PROVIDER_KEYS = {
    provider_config_key(provider) for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS
}
MODEL_ID_MAX_LENGTH = 256
#: The committed model's row says so in words, not colour alone (shared with
#: Switch model).
CURRENT_MARK = "● CURRENT"
_BLUR_RESTORE_DELAY_SECONDS = 0.05
_PROVENANCE_GROUP_LABELS = {
    ConsoleModelProvenance.SERVED_NOW: "Served now",
    ConsoleModelProvenance.CURRENT_CATALOG: "Current catalog",
    ConsoleModelProvenance.SAVED_FALLBACK: "Saved fallback",
    ConsoleModelProvenance.CUSTOM_UNVERIFIED: "Custom / unverified",
}


def normalize_model_id(value: object | None) -> str | None:
    """Return a model id as bounded single-line text, or None (TASK-14812 AC#7).

    Shared by this picker and Switch model's typed-id row.
    """
    raw_text = str(value or "")
    text = raw_text.strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    if (
        sanitize_string(raw_text, max_length=MODEL_ID_MAX_LENGTH) != raw_text
        or any(character in raw_text for character in "\r\n\t")
        or not validate_text_input(
            raw_text,
            max_length=MODEL_ID_MAX_LENGTH,
            allow_html=False,
        )
    ):
        return None
    return text


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


class PickerSearchInput(Input):
    """Combobox input whose focus selects its value from the head.

    Tab, the click that focuses it and the window regaining focus all select
    the committed value, so the next key replaces it. Compose it with
    ``select_on_focus=False``: Input's own select-on-focus leaves the caret
    after the value, and this field selects with the caret at the head
    (``select_all``).

    ``Input._on_mouse_down`` moves the caret to the click point, so
    click-then-type edited the value instead of replacing it (TASK-33001.7).
    Same mechanism as the Library rail's ``SelectAllOnFocusingClickInput``,
    which is off the UI-ready path.
    """

    _select_on_focusing_click = False

    def select_all(self) -> None:
        """Select the whole value with the caret at its head.

        Input keeps one cell for the caret after the last character and
        scrolls to it, so a caret at the end pushes the head of a value as
        wide as the field out of view (TASK-33007.9). Typing still replaces
        the selection.
        """
        self.selection = Selection(len(self.value), 0)

    def _typing(self) -> bool:
        """Whether the field holds text its user is still typing.

        Returns:
            False here. A subclass says True while its owner has not put the
            committed value back, e.g. while it holds a typed filter.
        """
        return False

    def _on_focus(self, event: events.Focus) -> None:
        # Screen focuses a widget before forwarding the click that focused it,
        # so that MouseDown is the next message; the refresh disarms a Tab focus.
        self._select_on_focusing_click = True
        self.call_after_refresh(self._disarm_focusing_click)
        # Input keeps the caret when the window regains focus. Here the owner
        # has put the committed value back by then, so that focus selects too,
        # unless the text is still the user's to finish (TASK-33007.9).
        if not (event.from_app_focus and self._typing()):
            self.select_all()
        else:  # back to the caret the blur's rest at the head scrolled away
            self.scroll_to_region(
                Region(self._cursor_offset, 0, width=1, height=1),
                force=True,
                animate=False,
            )

    def _disarm_focusing_click(self) -> None:
        self._select_on_focusing_click = False

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        # No super() call on purpose: Textual's dispatch walks the MRO and runs
        # Input._on_mouse_down itself, so a click in an already-focused field
        # still places the caret; super() would run it a second time.
        if self._select_on_focusing_click:
            self._select_on_focusing_click = False
            self.select_all()
            event.prevent_default()  # skip Input._on_mouse_down's caret move


class ModelPickerInput(PickerSearchInput):
    """Input that lets the compound picker own Escape semantics."""

    class EscapePressed(Message):
        """Posted before Input consumes Escape as an edit rollback."""

    def _typing(self) -> bool:
        # A Custom ID stays as typed; a filter (even one that matches nothing
        # and so hides the list) is held until the picker's blur timer puts
        # the committed model back.
        picker = self.query_ancestor(ModelSearchPicker)
        return picker.custom_mode or self.value != (picker.value or "")

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.post_message(self.EscapePressed())
            return
        await super()._on_key(event)


class ModelSearchPicker(Widget):
    """One keyboard-first model control shared by Console settings surfaces.

    The uncapped provider catalog is loaded once per provider and retained for
    the lifetime of the widget. Input changes only filter that in-memory list;
    they never trigger discovery or catalog resolution.
    """

    MAX_RESULTS = 20

    DEFAULT_CSS = """
    ModelSearchPicker {
        height: auto;
        width: 1fr;
    }

    ModelSearchPicker .model-search-picker-control {
        height: 3;
        width: 100%;
    }

    ModelSearchPicker #model-search-picker-input {
        width: 1fr;
        min-width: 0;
    }

    ModelSearchPicker #model-search-picker-custom {
        width: 12;
        min-width: 12;
    }

    ModelSearchPicker #model-search-picker-status {
        height: auto;
        color: $text-muted;
    }

    ModelSearchPicker #model-search-picker-results {
        max-height: 10;
    }
    """

    class ModelSelected(Message):
        """Posted when the user commits a catalog model."""

        def __init__(self, model_id: str) -> None:
            super().__init__()
            self.model_id = model_id

    class ModelValueChanged(Message):
        """Posted while a custom model ID is edited."""

        def __init__(self, model_id: str | None, *, custom: bool) -> None:
            super().__init__()
            self.model_id = model_id
            self.custom = custom

    class ProvenanceOptionsChanged(Message):
        """Posted when source-aware options become available for a provider."""

        def __init__(self, provider: str) -> None:
            super().__init__()
            self.provider = provider

    def __init__(
        self,
        *,
        id: str | None = None,
        provider_select_id: str = "#chat-api-provider",
        current_model: str | None = None,
        providers_models: Mapping[str, object] | None = None,
        show_custom_button: bool = True,
        show_provenance: bool = False,
        catalog_scope_service: object | None = None,
    ) -> None:
        """Initialize the controlled picker.

        Args:
            id: Optional Textual widget ID.
            provider_select_id: Provider Select whose value scopes the catalog.
            current_model: Model committed when the picker opens.
            providers_models: Optional catalog mapping supplied by the surface.
                The app catalog remains the fallback and the discovery scope is
                still used to obtain the full uncapped endpoint snapshot.
            show_custom_button: Whether this widget renders its own custom-ID
                action. Full settings reuses its existing adjacent action.
            show_provenance: Whether to group resolved options by their typed
                model provenance. Existing callers retain flat results.
            catalog_scope_service: The catalog service a surface owns; the
                app's ``llm_provider_catalog_scope_service`` is the fallback.
        """
        super().__init__(id=id)
        self._provider_select_id = provider_select_id
        self._initial_providers_models = providers_models
        self._show_custom_button = show_custom_button
        self._show_provenance = show_provenance
        self._catalog_scope_service = catalog_scope_service
        self._provider = ""
        self._selected_model = self._normalize_model(current_model)
        self._model_before_custom = self._selected_model
        self._custom_mode = False
        self._suppress_input_events = False
        self._matches: list[str] | list[ResolvedProviderModelOption] = []
        self._options_by_provider: dict[str, tuple[object, ...]] = {}
        self._provenance_provider_keys: set[str] = set()
        self._result_model_ids_by_option_id: dict[str, str] = {}
        self._committed_index: int | None = None
        self._discovered_model_ids: dict[str, tuple[str, ...]] = {}
        self._served_now_provider_keys: set[str] = set()
        self._load_errors: dict[str, bool] = {}
        self._load_counts: dict[str, int] = {}
        self._preserve_committed_on_next_input_focus = False

    @property
    def value(self) -> str | None:
        """Return the committed catalog model or current custom model ID."""
        if self._custom_mode and self.is_mounted:
            return self._normalize_model(
                self.query_one("#model-search-picker-input", Input).value
            )
        return self._selected_model

    @property
    def custom_mode(self) -> bool:
        """Return whether the custom-ID escape hatch is active."""
        return self._custom_mode

    def compose(self) -> ComposeResult:
        """Compose the editable combobox, custom action, status, and results."""
        with Horizontal(classes="model-search-picker-control"):
            yield ModelPickerInput(
                value=self._selected_model or "",
                placeholder="Choose or search models",
                id="model-search-picker-input",
                name="model-search",
                tooltip="Choose or search the model for this provider.",
                select_on_focus=False,
            )
            custom_button = Button(
                "Custom ID",
                id="model-search-picker-custom",
                name="custom-model-id",
                compact=True,
                tooltip="Enter an exact model ID that is not in the list.",
            )
            if not self._show_custom_button:
                # Only the hidden state is inline, so a host's stylesheet can
                # still decide when a shown button is displayed (Settings).
                custom_button.display = False
            yield custom_button
        yield Static("Loading models...", id="model-search-picker-status", markup=False)
        results = OptionList(
            id="model-search-picker-results",
            name="model-options",
        )
        results.tooltip = "Matching models; use arrow keys and Enter to select."
        yield results

    def on_mount(self) -> None:
        """Start the initial catalog load without blocking Textual's message pump."""
        self.query_one("#model-search-picker-results", OptionList).display = False
        provider = self._current_provider()
        if provider:
            self.refresh_provider(provider, current_model=self._selected_model)
        else:
            self._set_status("Choose a provider first.")

    _normalize_model = staticmethod(normalize_model_id)

    def _current_provider(self) -> str | None:
        try:
            provider_select = self.screen.query_one(self._provider_select_id, Select)
        except Exception:
            return None
        # TASK-32533: a blank provider select holds `Select.NULL`, a truthy
        # sentinel whose str() is "Select.NULL" -- not a provider key.
        if provider_select.value is Select.NULL:
            return None
        value = str(provider_select.value or "").strip()
        return value or None

    def _providers_models(self) -> Mapping[str, object]:
        if isinstance(self._initial_providers_models, Mapping):
            return self._initial_providers_models
        app_models = getattr(self.app, "providers_models", {})
        return app_models if isinstance(app_models, Mapping) else {}

    async def load_provider(
        self,
        provider: str,
        *,
        current_model: str | None = None,
        force: bool = False,
    ) -> None:
        """Load one provider once, or switch immediately to its cached catalog."""
        normalized_provider = str(provider or "").strip()
        if not normalized_provider:
            self._provider = ""
            self._selected_model = None
            self._set_input_value("")
            self._hide_results()
            self._set_status("Choose a provider first.")
            return

        self._provider = normalized_provider
        self._custom_mode = False
        self._selected_model = self._normalize_model(current_model)
        self._model_before_custom = self._selected_model
        self._set_input_value(self._selected_model or "")
        self._sync_custom_button()
        self._hide_results()

        cache_key = provider_config_key(normalized_provider)
        if not force and cache_key in self._options_by_provider:
            self._render_catalog_status()
            if self._show_provenance:
                self.post_message(self.ProvenanceOptionsChanged(normalized_provider))
            return

        self._set_status("Loading models...")
        self._load_counts[cache_key] = self._load_counts.get(cache_key, 0) + 1
        try:
            from tldw_chatbook.UI.Screens.provider_model_resolution import (
                resolve_provider_model_options,
            )

            options = await resolve_provider_model_options(
                self._providers_models(),
                self._catalog_scope_service
                or getattr(self.app, "llm_provider_catalog_scope_service", None),
                provider=normalized_provider,
                current_model=self._selected_model,
                merge_cap=None,
            )
        except Exception:
            options = []
            self._load_errors[cache_key] = True
        else:
            self._load_errors[cache_key] = False
        if provider_config_key(self._provider) != cache_key:
            return
        if self._show_provenance:
            self.set_provenance_options(normalized_provider, options)
        else:
            self._options_by_provider[cache_key] = tuple(options)
        self._render_catalog_status()
        input_widget = self.query_one("#model-search-picker-input", Input)
        input_shows_committed_model = bool(self._selected_model) and (
            input_widget.value == self._selected_model
        )
        if (
            input_widget.has_focus
            and not self._custom_mode
            and not input_shows_committed_model
        ):
            self._render_matches(input_widget.value, show_empty_query=True)

    def refresh_provider(
        self,
        provider: str,
        *,
        current_model: str | None = None,
        force: bool = False,
    ) -> None:
        """Schedule a provider switch without blocking the parent event handler."""
        self.run_worker(
            partial(
                self.load_provider,
                provider,
                current_model=current_model,
                force=force,
            ),
            exclusive=True,
            group=f"model-picker-load-{self.id or 'default'}",
        )

    def focus_input(self) -> None:
        """Focus the shared searchable input."""
        self.query_one("#model-search-picker-input", Input).focus()

    def set_model_value(self, model_id: str | None) -> None:
        """Synchronize a committed model from a compatibility adapter."""
        self._custom_mode = False
        self._selected_model = self._normalize_model(model_id)
        self._model_before_custom = self._selected_model
        self._set_input_value(self._selected_model or "")
        self._sync_custom_button()
        self._hide_results()
        self._render_catalog_status()

    def set_custom_value(self, model_id: str | None) -> None:
        """Synchronize a custom model draft from a compatibility adapter."""
        self._custom_mode = True
        self._selected_model = self._normalize_model(model_id)
        self._set_input_value(self._selected_model or "")
        self._sync_custom_button()
        self._hide_results()
        self._render_catalog_status()

    def set_discovered_models(
        self,
        provider: str,
        model_ids: tuple[str, ...] | list[str],
        *,
        notify: bool = True,
        served_now: bool = False,
    ) -> None:
        """Merge models returned by an explicit endpoint probe into the picker.

        Manual discovery in Console settings probes the user's unsaved base URL,
        so those results are not yet present in the application catalog service.
        Keep them as a provider-scoped overlay without reloading the endpoint.

        Args:
            provider: Provider whose overlay is changing.
            model_ids: Discovered model identifiers.
            notify: Post a parent-facing provenance refresh message. The modal
                disables this only for its own derived overlay updates.
            served_now: The listing is the current endpoint's own, dropped by
                its host whenever that endpoint changes (Settings), so its
                new ids group as "Served now" instead of custom.
        """
        cache_key = provider_config_key(provider)
        if served_now:
            self._served_now_provider_keys.add(cache_key)
        else:
            self._served_now_provider_keys.discard(cache_key)
        normalized_ids: list[str] = []
        for model_id in model_ids:
            normalized = self._normalize_model(model_id)
            if normalized and normalized not in normalized_ids:
                normalized_ids.append(normalized)
        self._discovered_model_ids[cache_key] = tuple(normalized_ids)
        if provider_config_key(self._provider) != cache_key:
            return
        self._render_catalog_status()
        input_widget = self.query_one("#model-search-picker-input", Input)
        if input_widget.has_focus and not self._custom_mode:
            self._render_matches(input_widget.value, show_empty_query=True)
        if notify and self._uses_provenance_options():
            self.post_message(self.ProvenanceOptionsChanged(provider))

    def provenance_for_model(
        self,
        model_id: str | None,
        *,
        provider: str | None = None,
    ) -> ConsoleModelProvenance | None:
        """Return the source category for a model in the active provider."""
        normalized = self._normalize_model(model_id)
        if normalized is None:
            return None
        if provider is not None and provider_config_key(
            provider
        ) != provider_config_key(self._provider):
            return None
        if self._custom_mode:
            return ConsoleModelProvenance.CUSTOM_UNVERIFIED
        if not self._uses_provenance_options():
            return None
        for option in self._typed_provider_options():
            if option.model_id == normalized:
                return self._display_provenance(option)
        return ConsoleModelProvenance.CUSTOM_UNVERIFIED

    @staticmethod
    def _display_provenance(
        option: ResolvedProviderModelOption,
    ) -> ConsoleModelProvenance:
        """Fail closed if a served-now option lacks connection verification."""
        if (
            option.provenance == ConsoleModelProvenance.SERVED_NOW
            and not option.verified_for_connection
        ):
            return ConsoleModelProvenance.CUSTOM_UNVERIFIED
        return option.provenance

    def set_provenance_options(
        self,
        provider: str,
        options: Sequence[ResolvedProviderModelOption],
        *,
        notify: bool = True,
    ) -> None:
        """Replace one provider's results with source-aware model choices.

        This opt-in path leaves existing catalog-loading callers on their
        original flat result list. Model IDs are normalized and deduplicated,
        while the complete typed option remains available for grouping and
        selection provenance.

        Args:
            provider: Provider whose source-aware options are changing.
            options: Complete typed model projection.
            notify: Post a parent-facing refresh message. The modal disables
                this only while applying its own derived provenance overlay.
        """
        cache_key = provider_config_key(provider)
        normalized_options: list[ResolvedProviderModelOption] = []
        seen_model_ids: set[str] = set()
        for option in options:
            if not isinstance(option, ResolvedProviderModelOption):
                raise TypeError(
                    "provenance options must be ResolvedProviderModelOption values"
                )
            model_id = self._normalize_model(option.model_id)
            if not model_id or model_id in seen_model_ids:
                continue
            normalized_options.append(
                replace(
                    option,
                    model_id=model_id,
                )
            )
            seen_model_ids.add(model_id)
        self._options_by_provider[cache_key] = tuple(normalized_options)
        self._provenance_provider_keys.add(cache_key)
        if provider_config_key(self._provider) != cache_key:
            return
        self._render_catalog_status()
        input_widget = self.query_one("#model-search-picker-input", Input)
        if input_widget.has_focus and not self._custom_mode:
            self._render_matches(input_widget.value, show_empty_query=True)
        if notify:
            self.post_message(self.ProvenanceOptionsChanged(provider))

    def toggle_custom_mode(self) -> None:
        """Toggle the explicit custom-ID escape hatch."""
        if self._custom_mode:
            self._custom_mode = False
            self._selected_model = self._normalize_model(
                self.query_one("#model-search-picker-input", Input).value
            )
            self._model_before_custom = self._selected_model
        else:
            self._custom_mode = True
            self._model_before_custom = self._selected_model
        self._set_input_value(self._selected_model or "")
        self._sync_custom_button()
        self._hide_results()
        self._render_catalog_status()
        self.focus_input()

    def _provider_options(self) -> tuple[object, ...]:
        return self._options_by_provider.get(provider_config_key(self._provider), ())

    def _uses_provenance_options(self) -> bool:
        return provider_config_key(self._provider) in self._provenance_provider_keys

    def _typed_provider_options(self) -> list[ResolvedProviderModelOption]:
        options = [
            option
            for option in self._provider_options()
            if isinstance(option, ResolvedProviderModelOption)
        ]
        seen_model_ids = {option.model_id for option in options}
        cache_key = provider_config_key(self._provider)
        served_now = cache_key in self._served_now_provider_keys
        for model_id in self._discovered_model_ids.get(cache_key, ()):
            if model_id in seen_model_ids:
                continue
            options.append(
                ResolvedProviderModelOption(
                    label=model_id,
                    model_id=model_id,
                    source=(
                        "manual_discovery_exact"
                        if served_now
                        else "manual_discovery_unfenced"
                    ),
                    capability_status="unknown",
                    persisted=False,
                    provenance=(
                        ConsoleModelProvenance.SERVED_NOW
                        if served_now
                        else ConsoleModelProvenance.CUSTOM_UNVERIFIED
                    ),
                    verified_for_connection=served_now,
                )
            )
            seen_model_ids.add(model_id)
        return options

    def _catalog_model_ids(self) -> list[str]:
        model_ids: list[str] = []
        for option in self._provider_options():
            model_id = self._normalize_model(getattr(option, "model_id", None))
            if model_id and model_id not in model_ids:
                model_ids.append(model_id)
        for model_id in self._discovered_model_ids.get(
            provider_config_key(self._provider), ()
        ):
            if model_id not in model_ids:
                model_ids.append(model_id)
        return model_ids

    def _render_catalog_status(self, matched: int | None = None) -> None:
        """Set the status line; catalog-health warnings outrank the cap note.

        Args:
            matched: Rows that matched the filter when the MAX_RESULTS cap cut
                the list short; replaces only the plain "N models" line.
        """
        if self._custom_mode:
            if self.is_mounted:
                custom_value = self.query_one("#model-search-picker-input", Input).value
                if custom_value and self._normalize_model(custom_value) is None:
                    self._set_status(
                        "Invalid model ID. Use a single-line value of at most "
                        f"{MODEL_ID_MAX_LENGTH} characters."
                    )
                    return
            self._set_status(
                "Custom model ID. Enter the exact ID expected by this provider."
            )
            return
        cache_key = provider_config_key(self._provider)
        if self._load_errors.get(cache_key, False):
            self._set_status(
                "Catalog unavailable. Use a configured model or Custom ID."
            )
            return
        options = self._provider_options()
        model_ids = self._catalog_model_ids()
        if not model_ids:
            self._set_status(
                "No models reported for this provider. Use Custom ID if needed."
            )
            return
        current_unlisted = any(
            str(getattr(option, "source", "")) == "current_unlisted"
            and self._normalize_model(getattr(option, "model_id", None))
            == self._selected_model
            for option in options
        )
        if current_unlisted:
            self._set_status(
                "Current model is not in the latest catalog. Choose another or keep it."
            )
            return
        sources = {str(getattr(option, "source", "")) for option in options}
        if (
            cache_key in _CLOUD_CATALOG_PROVIDER_KEYS
            and sources
            and sources <= {"saved"}
        ):
            self._set_status(
                f"Live catalog unavailable. Showing {_count(len(model_ids), 'configured model')}."
            )
            return
        if matched is not None:
            self._set_status(
                f"Showing {len(self._matches)} of {matched} matching models. "
                "Type to narrow the list."
            )
            return
        self._set_status(
            f"{_count(len(model_ids), 'model')} available. Type to filter."
        )

    def _set_status(self, copy: str) -> None:
        set_status_line(self, "#model-search-picker-status", copy)

    def _set_input_value(self, value: str) -> None:
        if not self.is_mounted:
            return
        input_widget = self.query_one("#model-search-picker-input", Input)
        self._suppress_input_events = True
        try:
            with input_widget.prevent(Input.Changed):
                input_widget.value = value
        finally:
            self._suppress_input_events = False
        if not input_widget.has_focus:
            self._rest_at_head(input_widget)
        else:
            # Every focused caller puts a committed value back (a choice, Esc,
            # a mode switch, a provider re-scope or a host sync), so the next
            # key replaces it, even when Enter chose exactly the text typed,
            # not lands at the filter's caret (TASK-33007.9). A new caller
            # that sets an uncommitted value under focus is selected too.
            input_widget.select_all()

    @staticmethod
    def _rest_at_head(input_widget: Input) -> None:
        # At rest the id reads from its head, wherever the caret was left
        # (force: Textual does not scroll a disabled widget without it).
        input_widget.scroll_to(x=0, animate=False, force=True)

    def _hide_results(self) -> None:
        if not self.is_mounted:
            return
        results = self.query_one("#model-search-picker-results", OptionList)
        self._matches = []
        self._result_model_ids_by_option_id = {}
        self._committed_index = None
        results.clear_options()
        results.display = False

    def _render_matches(self, query: str, *, show_empty_query: bool = False) -> None:
        results = self.query_one("#model-search-picker-results", OptionList)
        normalized_query = query.strip().lower()
        if self._uses_provenance_options():
            self._render_provenance_matches(
                results,
                normalized_query,
                show_empty_query=show_empty_query,
            )
            return
        catalog_model_ids = self._catalog_model_ids()
        model_ids = catalog_model_ids
        if normalized_query:
            model_ids = [
                model_id
                for model_id in model_ids
                if normalized_query in model_id.lower()
            ]
        elif not show_empty_query:
            self._hide_results()
            return
        self._matches = self._capped(model_ids, str)
        self._result_model_ids_by_option_id = {}
        self._committed_index = None
        results.clear_options()
        for model_id in self._matches:
            self._add_result(results, model_id)
        results.display = bool(self._matches)
        self._render_match_status(normalized_query, len(model_ids))

    def _render_match_status(self, normalized_query: str, matched: int) -> None:
        """Name an empty filter, or a list the MAX_RESULTS cap cut short."""
        if (
            normalized_query
            and not matched
            and self._catalog_model_ids()
            and not self._load_errors.get(provider_config_key(self._provider), False)
        ):
            self._set_status("No matching models. Clear the filter or use Custom ID.")
        else:
            self._render_catalog_status(
                matched if matched > len(self._matches) else None
            )

    def _render_provenance_matches(
        self,
        results: OptionList,
        normalized_query: str,
        *,
        show_empty_query: bool,
    ) -> None:
        """Render only populated provenance groups with stable option IDs."""
        if not normalized_query and not show_empty_query:
            self._hide_results()
            return
        options = self._typed_provider_options()
        if normalized_query:
            options = [
                option
                for option in options
                if normalized_query in option.model_id.lower()
                or normalized_query in option.label.lower()
            ]
        ordered_options = [
            option
            for provenance in _PROVENANCE_GROUP_LABELS
            for option in options
            if self._display_provenance(option) == provenance
        ]
        self._matches = self._capped(ordered_options, lambda option: option.model_id)
        self._result_model_ids_by_option_id = {}
        self._committed_index = None
        results.clear_options()
        for provenance, group_label in _PROVENANCE_GROUP_LABELS.items():
            group = [
                option
                for option in self._matches
                if self._display_provenance(option) == provenance
            ]
            if not group:
                continue
            results.add_option(
                Option(
                    group_label,
                    id=f"model-provenance-heading-{provenance.value}",
                    disabled=True,
                )
            )
            for option in group:
                option_id = f"model-provenance-option-{len(self._result_model_ids_by_option_id)}"
                self._result_model_ids_by_option_id[option_id] = option.model_id
                self._add_result(results, option.model_id, option_id)
        results.display = bool(self._matches)
        self._render_match_status(normalized_query, len(ordered_options))

    def _capped(self, entries: list, model_id_of) -> list:
        """The first MAX_RESULTS entries; a committed model past the cap takes
        the last slot, so large catalogs keep its mark and Down (C7(b))."""
        shown = entries[: self.MAX_RESULTS]
        committed = next(
            (
                entry
                for entry in entries[self.MAX_RESULTS :]
                if model_id_of(entry) == self._selected_model
            ),
            None,
        )
        if committed is not None and self._selected_model not in map(
            model_id_of, shown
        ):
            shown[-1] = committed
        return shown

    def _add_result(
        self, results: OptionList, model_id: str, option_id: str | None = None
    ) -> None:
        """Add one literal model row; the committed model says so in words."""
        prompt = Text(model_id)
        if model_id == self._selected_model:
            self._committed_index = results.option_count
            prompt.append(f"  {CURRENT_MARK}")
        results.add_option(Option(prompt, id=option_id))

    def _commit_catalog_model(self, model_id: str) -> None:
        normalized = self._normalize_model(model_id)
        if not normalized:
            return
        results = self.query_one("#model-search-picker-results", OptionList)
        results_had_focus = results.has_focus
        self._custom_mode = False
        self._selected_model = normalized
        self._model_before_custom = normalized
        self._set_input_value(normalized)
        self._sync_custom_button()
        self._hide_results()
        self._render_catalog_status()
        if results_had_focus:
            self._preserve_committed_on_next_input_focus = True
            self.focus_input()
        self.post_message(self.ModelSelected(normalized))

    def _sync_custom_button(self) -> None:
        if self.is_mounted:
            self.query_one("#model-search-picker-custom", Button).label = (
                "Model list" if self._custom_mode else "Custom ID"
            )

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if getattr(event.control, "id", None) != "model-search-picker-input":
            return
        if self._preserve_committed_on_next_input_focus:
            self._preserve_committed_on_next_input_focus = False
            return
        if self._custom_mode:
            return
        # TASK-33001.7: keep the committed model painted. The input selects
        # it on focus, so the first keystroke replaces it. A typed filter the
        # blur timer has not dropped yet (a short window blur, Tab to Custom
        # ID and back) keeps its own list, empty or not (TASK-33007.9).
        value = event.control.value
        self._render_matches(
            "" if value == (self._selected_model or "") else value,
            show_empty_query=True,
        )

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        """Restore committed copy after focus leaves the compound picker."""
        self.set_timer(
            _BLUR_RESTORE_DELAY_SECONDS,
            self._restore_committed_display_after_blur,
        )

    def _restore_committed_display_after_blur(self) -> None:
        """Keep visible and committed catalog values aligned after an edit."""
        if not self.is_mounted:
            return
        focused = self.app.focused
        if focused is not None and self in focused.ancestors_with_self:
            return
        input_widget = self.query_one("#model-search-picker-input", Input)
        self._rest_at_head(input_widget)  # a Custom ID too
        if self._custom_mode:
            return
        if input_widget.value != (self._selected_model or "") or self._matches:
            self._set_input_value(self._selected_model or "")
            self._hide_results()
            self._render_catalog_status()

    @on(Input.Changed, "#model-search-picker-input")
    def _handle_query(self, event: Input.Changed) -> None:
        if self._suppress_input_events:
            return
        if self._custom_mode:
            self._selected_model = self._normalize_model(event.value)
            self._hide_results()
            self._render_catalog_status()
            self.post_message(self.ModelValueChanged(self._selected_model, custom=True))
            return
        self._render_matches(event.value)

    @on(Input.Submitted, "#model-search-picker-input")
    def _input_submitted(self, event: Input.Submitted) -> None:
        if self._custom_mode:
            self._selected_model = self._normalize_model(event.value)
            self.post_message(self.ModelValueChanged(self._selected_model, custom=True))
            return
        query = event.value.strip().lower()
        match_model_ids = [
            option.model_id
            if isinstance(option, ResolvedProviderModelOption)
            else option
            for option in self._matches
        ]
        exact = next(
            (model_id for model_id in match_model_ids if model_id.lower() == query),
            None,
        )
        if exact is not None:
            self._commit_catalog_model(exact)
        elif len(match_model_ids) == 1:
            self._commit_catalog_model(match_model_ids[0])

    @on(OptionList.OptionSelected, "#model-search-picker-results")
    def _handle_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            model_id = self._result_model_ids_by_option_id.get(event.option.id)
            if model_id is not None:
                self._commit_catalog_model(model_id)
            return
        index = event.option_index
        if index is None or not (0 <= index < len(self._matches)):
            return
        model = self._matches[index]
        if isinstance(model, ResolvedProviderModelOption):
            self._commit_catalog_model(model.model_id)
        else:
            self._commit_catalog_model(model)

    @on(Button.Pressed, "#model-search-picker-custom")
    def _toggle_custom(self, event: Button.Pressed) -> None:
        event.stop()
        self.toggle_custom_mode()

    def _cancel_edit(self, event: Message) -> None:
        input_widget = self.query_one("#model-search-picker-input", Input)
        results = self.query_one("#model-search-picker-results", OptionList)
        results_had_focus = results.has_focus
        if self._custom_mode:
            self._custom_mode = False
            self._selected_model = self._model_before_custom
            self._set_input_value(self._selected_model or "")
            self._sync_custom_button()
            self._hide_results()
            self._render_catalog_status()
            self.post_message(
                self.ModelValueChanged(self._selected_model, custom=False)
            )
            if results_had_focus:
                self._preserve_committed_on_next_input_focus = True
                self.focus_input()
            event.stop()
            return
        if input_widget.value != (self._selected_model or "") or self._matches:
            self._set_input_value(self._selected_model or "")
            self._hide_results()
            self._render_catalog_status()
            if results_had_focus:
                self._preserve_committed_on_next_input_focus = True
                self.focus_input()
            event.stop()

    @on(ModelPickerInput.EscapePressed)
    def _input_escape_pressed(self, event: ModelPickerInput.EscapePressed) -> None:
        self._cancel_edit(event)

    @on(events.Key)
    def _handle_key(self, event: events.Key) -> None:
        if event.key == "down" and self._matches:
            results = self.query_one("#model-search-picker-results", OptionList)
            results.focus()
            # The committed model first (C7(b)), else the first enabled row.
            results.highlighted = next(
                (
                    index
                    for index, option in enumerate(results.options)
                    if index == self._committed_index
                    or (self._committed_index is None and not option.disabled)
                ),
                None,
            )
            event.stop()
            return
        if event.key != "escape":
            return
        self._cancel_edit(event)
