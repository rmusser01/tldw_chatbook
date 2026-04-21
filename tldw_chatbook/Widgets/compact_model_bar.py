# compact_model_bar.py
# Description: Compact inline model selector bar shown above the chat log.
# Provides quick access to Provider, Model, Temperature without opening the sidebar.
#
# Imports
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Label, Select

from ..config import get_cli_providers_and_models

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
        app_instance: "TldwCli | None" = None,
        on_sidebar_toggle_requested: Optional[Callable[[], Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.on_sidebar_toggle_requested = on_sidebar_toggle_requested

    def _resolve_app(self):
        if self.app_instance is not None:
            return self.app_instance
        try:
            return self.app
        except Exception:
            return None

    def _sync_compact_provider_model(
        self,
        provider: str,
        selected_model: Optional[str] = None,
    ) -> None:
        providers_models = get_cli_providers_and_models()
        available_models = providers_models.get(provider, [])

        compact_model = self.query_one("#compact-api-model", Select)
        current_model = None if compact_model.value == Select.BLANK else str(compact_model.value)

        new_options = [(m, m) for m in available_models]
        compact_model.set_options(new_options)

        desired_model = selected_model if selected_model in available_models else None
        if desired_model is None and current_model in available_models:
            desired_model = current_model
        if desired_model is None and available_models:
            desired_model = available_models[0]

        compact_model.value = desired_model if desired_model is not None else Select.BLANK

    def _set_provider_value(self, provider: str) -> None:
        compact_provider = self.query_one("#compact-api-provider", Select)
        if compact_provider.value != provider:
            compact_provider.value = provider

    def compose(self) -> ComposeResult:
        """Compose the compact model bar."""
        app = self._resolve_app()
        config = getattr(app, "app_config", {}) if app is not None else {}
        defaults = config.get("chat_defaults", {})
        providers_models = get_cli_providers_and_models()
        available_providers = list(providers_models.keys())
        default_provider = defaults.get("provider", available_providers[0] if available_providers else "")

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
        app = self._resolve_app()
        config = getattr(app, "app_config", {}) if app is not None else {}
        defaults = config.get("chat_defaults", {})
        providers_models = get_cli_providers_and_models()
        available_providers = list(providers_models.keys())
        default_provider = defaults.get("provider", available_providers[0] if available_providers else "")

        try:
            if default_provider in available_providers:
                self._sync_compact_provider_model(default_provider, defaults.get("model", ""))
                self._set_provider_value(default_provider)
        except NoMatches:
            pass

        try:
            temperature_input = self.query_one("#compact-temperature", Input)
            temperature_input.value = str(defaults.get("temperature", 0.7))
        except NoMatches:
            pass

    @on(Select.Changed, "#compact-api-provider")
    async def handle_compact_provider_change(self, event: Select.Changed) -> None:
        """Handle provider change in compact bar and sync to sidebar."""
        new_provider = str(event.value)
        logger.info(f"Compact bar: provider changed to {new_provider}")

        try:
            self._sync_compact_provider_model(new_provider)
        except NoMatches:
            pass

        # Sync to sidebar provider select
        app = self._resolve_app()
        if app is not None:
            try:
                sidebar_provider = app.query_one("#chat-api-provider", Select)
                sidebar_provider.value = event.value
            except NoMatches:
                logger.debug("Sidebar provider select not found for sync")

    @on(Select.Changed, "#compact-api-model")
    async def handle_compact_model_change(self, event: Select.Changed) -> None:
        """Sync model change to sidebar."""
        app = self._resolve_app()
        if app is not None:
            try:
                sidebar_model = app.query_one("#chat-api-model", Select)
                sidebar_model.value = event.value
            except NoMatches:
                logger.debug("Sidebar model select not found for sync")

    @on(Input.Changed, "#compact-temperature")
    async def handle_compact_temp_change(self, event: Input.Changed) -> None:
        """Sync temperature change to sidebar."""
        app = self._resolve_app()
        if app is not None:
            try:
                sidebar_temp = app.query_one("#chat-temperature", Input)
                sidebar_temp.value = event.value
            except NoMatches:
                logger.debug("Sidebar temperature input not found for sync")

    @on(Button.Pressed, "#compact-sidebar-toggle")
    async def handle_sidebar_toggle(self, event: Button.Pressed) -> None:
        """Toggle the settings sidebar."""
        event.stop()
        callback = self.on_sidebar_toggle_requested
        if callback is None:
            logger.debug("Sidebar toggle requested but no host callback is configured")
            return
        try:
            result = callback()
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.error(f"Error handling sidebar toggle request from compact bar: {e}")

    def sync_from_sidebar(
        self,
        provider: str = None,
        model: str = None,
        temperature: str = None,
    ) -> None:
        """Sync values from sidebar to compact bar (called when sidebar values change)."""
        try:
            if provider is not None:
                self._sync_compact_provider_model(provider, model)
                self._set_provider_value(provider)
            elif model is not None:
                compact_model = self.query_one("#compact-api-model", Select)
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
