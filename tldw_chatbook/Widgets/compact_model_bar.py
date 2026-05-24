# compact_model_bar.py
# Description: Compact inline model selector bar shown above the chat log.
# Provides quick access to Provider, Model, Temperature without opening the sidebar.
#
# Imports
from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Select, Input, Static, Label
from textual.reactive import reactive
from textual import on
from textual.css.query import NoMatches

from ..config import get_cli_providers_and_models, get_cli_setting, resolve_provider_name

if TYPE_CHECKING:
    from ..app import TldwCli

logger = logger.bind(module="CompactModelBar")

#######################################################################################################################


class CompactModelBar(Horizontal):
    """Compact inline bar showing Provider, Model, Temperature and a sidebar toggle.

    Uses unique IDs (compact-api-provider, compact-api-model) to avoid collision
    with sidebar widgets (chat-api-provider, chat-api-model).
    """

    def __init__(
        self,
        app_instance: 'TldwCli',
        on_sidebar_toggle_requested: Callable[[], Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.on_sidebar_toggle_requested = on_sidebar_toggle_requested

    def compose(self) -> ComposeResult:
        """Compose the compact model bar."""
        config = self.app_instance.app_config
        defaults = config.get("chat_defaults", {})
        providers_models = get_cli_providers_and_models()
        available_providers = list(providers_models.keys())
        default_provider = resolve_provider_name(
            defaults.get("provider", available_providers[0] if available_providers else ""),
            providers_models,
        )

        # Provider select
        provider_options = [(p, p) for p in available_providers]
        yield Select(
            options=provider_options,
            prompt="Provider",
            allow_blank=False,
            id="compact-api-provider",
        )

        # Model select
        initial_models = providers_models.get(default_provider, [])
        model_options = [(m, m) for m in initial_models]
        yield Select(
            options=model_options,
            prompt="Model",
            allow_blank=True,
            id="compact-api-model",
        )

        # Temperature input
        yield Label("Temp:", classes="compact-bar-label")
        yield Input(
            placeholder="0.7",
            id="compact-temperature",
            value=str(defaults.get("temperature", 0.7)),
            classes="compact-bar-temp",
        )

        # Sidebar toggle button
        yield Button(
            "⚙",
            id="compact-sidebar-toggle",
            classes="compact-bar-toggle",
            tooltip="Toggle settings sidebar (Ctrl+[)",
        )

    def on_mount(self) -> None:
        """Set default values after widgets are mounted."""
        config = self.app_instance.app_config
        defaults = config.get("chat_defaults", {})
        providers_models = get_cli_providers_and_models()
        available_providers = list(providers_models.keys())
        default_provider = resolve_provider_name(
            defaults.get("provider", available_providers[0] if available_providers else ""),
            providers_models,
        )
        # Set provider
        try:
            provider_select = self.query_one("#compact-api-provider", Select)
            if default_provider in available_providers:
                provider_select.value = default_provider
        except NoMatches:
            pass
        # Set model
        initial_models = providers_models.get(default_provider, [])
        default_model = defaults.get("model", "")
        try:
            model_select = self.query_one("#compact-api-model", Select)
            if default_model in initial_models:
                model_select.value = default_model
            elif initial_models:
                model_select.value = initial_models[0]
        except NoMatches:
            pass

    @on(Select.Changed, "#compact-api-provider")
    async def handle_compact_provider_change(self, event: Select.Changed) -> None:
        """Handle provider change in compact bar and sync to sidebar."""
        new_provider = str(event.value)
        logger.info(f"Compact bar: provider changed to {new_provider}")

        providers_models = get_cli_providers_and_models()
        available_models = providers_models.get(new_provider, [])

        # Update compact model select
        try:
            compact_model = self.query_one("#compact-api-model", Select)
            new_options = [(m, m) for m in available_models]
            compact_model.set_options(new_options)
            if available_models:
                compact_model.value = available_models[0]
            else:
                compact_model.value = Select.BLANK
        except NoMatches:
            pass

        # Sync to sidebar provider select
        try:
            sidebar_provider = self.app.query_one("#chat-api-provider", Select)
            sidebar_provider.value = event.value
        except NoMatches:
            logger.debug("Sidebar provider select not found for sync")

    @on(Select.Changed, "#compact-api-model")
    async def handle_compact_model_change(self, event: Select.Changed) -> None:
        """Sync model change to sidebar."""
        try:
            sidebar_model = self.app.query_one("#chat-api-model", Select)
            sidebar_model.value = event.value
        except NoMatches:
            logger.debug("Sidebar model select not found for sync")
        except Exception as e:
            logger.debug(f"Sidebar model select could not accept compact model value {event.value!r}: {e}")

    @on(Input.Changed, "#compact-temperature")
    async def handle_compact_temp_change(self, event: Input.Changed) -> None:
        """Sync temperature change to sidebar."""
        try:
            sidebar_temp = self.app.query_one("#chat-temperature", Input)
            sidebar_temp.value = event.value
        except NoMatches:
            logger.debug("Sidebar temperature input not found for sync")

    @on(Button.Pressed, "#compact-sidebar-toggle")
    async def handle_sidebar_toggle(self, event: Button.Pressed) -> None:
        """Toggle the settings sidebar."""
        event.stop()
        if self.on_sidebar_toggle_requested:
            result = self.on_sidebar_toggle_requested()
            if hasattr(result, "__await__"):
                await result
            return

        # Find the ChatWindowEnhanced parent and toggle via its handler
        try:
            from ..UI.Chat_Window_Enhanced import ChatWindowEnhanced
            chat_window = self.ancestors_with_self
            for ancestor in chat_window:
                if isinstance(ancestor, ChatWindowEnhanced):
                    ancestor._sidebar_collapsed = not ancestor._sidebar_collapsed
                    ancestor.app_instance.chat_sidebar_collapsed = ancestor._sidebar_collapsed
                    try:
                        sidebar = ancestor.query_one("#chat-left-sidebar")
                        sidebar.display = not ancestor._sidebar_collapsed
                    except NoMatches:
                        pass
                    break
        except Exception as e:
            logger.error(f"Error toggling sidebar from compact bar: {e}")

    def sync_from_sidebar(self, provider: str = None, model: str = None, temperature: str = None) -> None:
        """Sync values from sidebar to compact bar (called when sidebar values change)."""
        try:
            compact_model = None
            providers_models = get_cli_providers_and_models()
            available_models: list[str] | None = None
            if provider is not None:
                compact_provider = self.query_one("#compact-api-provider", Select)
                if compact_provider.value != provider:
                    compact_provider.value = provider
                available_models = providers_models.get(provider, [])
                compact_model = self.query_one("#compact-api-model", Select)
                compact_model.set_options([(m, m) for m in available_models])
            if model is not None:
                if compact_model is None:
                    compact_model = self.query_one("#compact-api-model", Select)
                if available_models is None:
                    try:
                        compact_provider = self.query_one("#compact-api-provider", Select)
                        current_provider = None if compact_provider.value == Select.BLANK else str(compact_provider.value)
                    except NoMatches:
                        current_provider = None
                    available_models = providers_models.get(current_provider, []) if current_provider else []
                if model not in available_models:
                    available_models = [*available_models, model]
                    compact_model.set_options([(m, m) for m in available_models])
                if compact_model.value != model:
                    compact_model.value = model
            if temperature is not None:
                compact_temp = self.query_one("#compact-temperature", Input)
                if compact_temp.value != temperature:
                    compact_temp.value = temperature
        except NoMatches:
            pass


#
# End of compact_model_bar.py
#######################################################################################################################
