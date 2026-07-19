"""Type-to-filter model picker over the full (uncapped) provider catalog."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, OptionList, Select
from textual.widgets._option_list import Option


class ModelSearchPicker(Widget):
    """Substring search across saved + discovered models for the active provider.

    OpenRouter model IDs embed the upstream provider prefix
    (``anthropic/claude-3.7-sonnet``), so provider search works naturally (ADR-019).
    """

    MAX_RESULTS = 20

    DEFAULT_CSS = """
    ModelSearchPicker {
        height: auto;
    }

    ModelSearchPicker #model-search-picker-results {
        max-height: 10;
    }
    """

    class ModelSelected(Message):
        """Posted when the user picks a model from the search results."""

        def __init__(self, model_id: str) -> None:
            super().__init__()
            self.model_id = model_id

    def __init__(
        self,
        *,
        id: str | None = None,
        provider_select_id: str = "#chat-api-provider",
    ) -> None:
        super().__init__(id=id)
        self._provider_select_id = provider_select_id
        self._matches: list[str] = []

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search all models…", id="model-search-picker-input")
        yield OptionList(id="model-search-picker-results")

    def on_mount(self) -> None:
        self.query_one("#model-search-picker-results", OptionList).display = False

    def _current_provider(self) -> str | None:
        # Query via self.screen (not self.app): App queries resolve against the
        # default screen, which misses selects inside a modal like the popover.
        try:
            provider_select = self.screen.query_one(self._provider_select_id, Select)
        except Exception:
            return None
        value = str(provider_select.value or "").strip()
        return value or None

    @on(Input.Changed, "#model-search-picker-input")
    async def _handle_query(self, event: Input.Changed) -> None:
        results = self.query_one("#model-search-picker-results", OptionList)
        query = event.value.strip().lower()
        self._matches = []
        results.clear_options()
        if not query:
            results.display = False
            return
        provider = self._current_provider()
        if not provider:
            results.display = False
            return
        from tldw_chatbook.UI.Screens.provider_model_resolution import (
            resolve_provider_model_options,
        )
        options = await resolve_provider_model_options(
            self.app, provider=provider, merge_cap=None,
        )
        self._matches = [
            option.model_id for option in options if query in option.model_id.lower()
        ][: self.MAX_RESULTS]
        for model_id in self._matches:
            results.add_option(Option(model_id))
        results.display = bool(self._matches)

    @on(OptionList.OptionSelected, "#model-search-picker-results")
    def _handle_selected(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        if index is None or not (0 <= index < len(self._matches)):
            return
        model_id = self._matches[index]
        self.query_one("#model-search-picker-input", Input).value = ""
        self.post_message(self.ModelSelected(model_id))
