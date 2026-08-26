# compact_model_bar.py
# Description: Compact inline model selector bar shown above the chat log.
# Provides quick access to Provider, Model, Temperature without opening the sidebar.
#
# Imports
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Select, Input, Label
from textual import on
from textual.css.query import NoMatches

from ..config import get_cli_providers_and_models, resolve_provider_name

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
        app_instance: "TldwCli",
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
        # task-16474: no arbitrary first-provider fallback. When
        # chat_defaults.provider is missing or unresolvable the select stays
        # on its prompt until the user chooses -- the first [providers] key
        # in file order is not a selection anyone made.
        default_provider = resolve_provider_name(
            defaults.get("provider", ""),
            providers_models,
        )

        # Provider select. allow_blank=True (task-16474): Textual's Select
        # force-picks options[0] at mount and on set_options when blank is
        # disallowed, which both fabricates a selection nobody made and fires
        # a Changed event the provider mirror would treat as user intent.
        # Blank now means "nothing chosen" and the screen handler already
        # ignores empty values.
        provider_options = [(p, p) for p in available_providers]
        yield Select(
            options=provider_options,
            prompt="Provider",
            allow_blank=True,
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
        """Set default values after widgets are mounted.

        task-16474: population is programmatic, so every value set here is
        wrapped in ``prevent(Select.Changed)`` -- the screen's provider/model
        mirrors must only track genuine user selections, never the mount
        burst (which used to write them on every recompose and silently
        revert values the user had applied).
        """
        config = self.app_instance.app_config
        defaults = config.get("chat_defaults", {})
        providers_models = get_cli_providers_and_models()
        available_providers = list(providers_models.keys())
        default_provider = resolve_provider_name(
            defaults.get("provider", ""),
            providers_models,
        )
        # Set provider
        try:
            provider_select = self.query_one("#compact-api-provider", Select)
            if default_provider in available_providers:
                with provider_select.prevent(Select.Changed):
                    provider_select.value = default_provider
        except NoMatches:
            pass
        # Set model
        initial_models = providers_models.get(default_provider, [])
        default_model = defaults.get("model", "")
        try:
            model_select = self.query_one("#compact-api-model", Select)
            if default_model in initial_models:
                with model_select.prevent(Select.Changed):
                    model_select.value = default_model
            elif initial_models:
                with model_select.prevent(Select.Changed):
                    model_select.value = initial_models[0]
        except NoMatches:
            pass
        # The suppressed population events used to reach the screen's
        # Select.Changed handler, whose coalesced control-bar sync kept the
        # rail/inspector fresh after the bar (re)mounted. Population no
        # longer carries user intent, so request that same sync explicitly
        # instead of resurrecting the ambient events (task-16474).
        screen = self.screen
        request_sync = getattr(screen, "_request_console_control_bar_sync", None)
        if callable(request_sync):
            request_sync()

    @on(Select.Changed, "#compact-api-provider")
    async def handle_compact_provider_change(self, event: Select.Changed) -> None:
        """Handle provider change in compact bar and sync to sidebar."""
        # task-16474: the provider select allows blank ("nothing chosen");
        # a blank change carries no provider to sync. Covers Textual's
        # BLANK and NULL sentinels the same way the screen's handler does.
        if (
            event.value is None
            or event.value == Select.BLANK
            or str(event.value).startswith("Select.")
        ):
            return
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
            logger.debug(
                f"Sidebar model select could not accept compact model value {event.value!r}: {e}"
            )

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
        """Toggle the settings sidebar.

        ``ChatWindowEnhanced`` is retired, so the only live host wiring this
        widget always passes ``on_sidebar_toggle_requested`` (the Console
        control bar routes it to ``ChatScreen._toggle_console_chat_sidebar``);
        the callback is the sole toggle path now.

        Args:
            event: The compact-bar sidebar-toggle button press.
        """
        event.stop()
        if self.on_sidebar_toggle_requested:
            result = self.on_sidebar_toggle_requested()
            if hasattr(result, "__await__"):
                await result

    def sync_from_sidebar(
        self, provider: str = None, model: str = None, temperature: str = None
    ) -> None:
        """Sync values from sidebar to compact bar (called when sidebar values change)."""
        try:
            compact_model = None
            providers_models = get_cli_providers_and_models()
            available_models: list[str] | None = None
            if provider is not None:
                compact_provider = self.query_one("#compact-api-provider", Select)
                if compact_provider.value != provider:
                    with compact_provider.prevent(Select.Changed):
                        compact_provider.value = provider
                available_models = providers_models.get(provider, [])
                compact_model = self.query_one("#compact-api-model", Select)
                with compact_model.prevent(Select.Changed):
                    compact_model.set_options([(m, m) for m in available_models])
            if model is not None:
                if compact_model is None:
                    compact_model = self.query_one("#compact-api-model", Select)
                if available_models is None:
                    try:
                        compact_provider = self.query_one(
                            "#compact-api-provider", Select
                        )
                        current_provider = (
                            None
                            if compact_provider.value == Select.BLANK
                            else str(compact_provider.value)
                        )
                    except NoMatches:
                        current_provider = None
                    available_models = (
                        providers_models.get(current_provider, [])
                        if current_provider
                        else []
                    )
                if model not in available_models:
                    available_models = [*available_models, model]
                    with compact_model.prevent(Select.Changed):
                        compact_model.set_options([(m, m) for m in available_models])
                if compact_model.value != model:
                    with compact_model.prevent(Select.Changed):
                        compact_model.value = model
            if temperature is not None:
                compact_temp = self.query_one("#compact-temperature", Input)
                if compact_temp.value != temperature:
                    with compact_temp.prevent(Input.Changed):
                        compact_temp.value = temperature
        except NoMatches:
            pass


#
# End of compact_model_bar.py
#######################################################################################################################
