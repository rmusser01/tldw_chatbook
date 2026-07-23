"""Settings-native splash screen browser and preview widget."""

from __future__ import annotations

from typing import Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input, OptionList, Select, Static, Switch
from textual.widgets.option_list import Option

from ..config import get_cli_setting, save_setting_to_cli_config
from ..Utils.Splash_Screens.card_definitions import get_all_card_definitions
from ..Widgets.splash_screen import SplashScreen


DEFAULT_SPLASH_CONFIG: dict[str, Any] = {
    "enabled": True,
    "duration": 2.5,
    "skip_on_keypress": True,
    "card_selection": "random",
    "show_progress": True,
    "fade_in_duration": 0.3,
    "fade_out_duration": 0.2,
    "animation_speed": 1.0,
}


class SettingsSplashScreenViewer(Vertical):
    """Splash screen gallery and defaults editor styled for Settings."""

    class SplashConfigChanged(Message):
        """Message sent when a splash config value is changed and saved."""

        def __init__(self, section: str, key: str, value: Any) -> None:
            self.section = section
            self.key = key
            self.value = value
            super().__init__()

    selected_card: reactive[str] = reactive("default")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cards: dict[str, dict[str, Any]] = {}
        self._config: dict[str, Any] = {}

    def _load_config(self) -> dict[str, Any]:
        try:
            config = get_cli_setting("splash_screen", DEFAULT_SPLASH_CONFIG)
        except Exception as exc:
            logger.warning("Failed to load splash_screen config: {}. Using defaults.", exc)
            config = dict(DEFAULT_SPLASH_CONFIG)
        if not isinstance(config, dict):
            config = dict(DEFAULT_SPLASH_CONFIG)
        for key, value in DEFAULT_SPLASH_CONFIG.items():
            config.setdefault(key, value)
        return config

    def _card_options(self) -> list[Option]:
        options: list[Option] = []
        for card_name, card_data in self._cards.items():
            card_type = card_data.get("type", "static")
            title = card_data.get("title", card_name)
            label = f"{title} ({card_name}) [{card_type}]"
            options.append(Option(label, id=card_name))
        if not options:
            options.append(Option("No splash screens found", id="__none__"))
        return options

    def _default_select_options(self) -> list[tuple[str, str]]:
        options = [("Random", "random")]
        for card_name, card_data in self._cards.items():
            title = card_data.get("title", card_name)
            options.append((f"{title} ({card_name})", card_name))
        return options

    def compose(self) -> ComposeResult:
        """Compose the splash screen settings widget.

        Yields:
            ComposeResult: The splash screen settings UI sections.
        """
        self._config = self._load_config()
        try:
            self._cards = get_all_card_definitions()
        except Exception as exc:
            logger.error("Failed to load splash card definitions: {}", exc)
            self._cards = {}

        with Vertical(id="settings-splash-card", classes="settings-focus-card"):
            yield Static("Startup defaults", classes="destination-section")
            with Horizontal(classes="settings-input-row settings-select-row"):
                yield Static("Default card", classes="settings-input-label")
                yield Select(
                    self._default_select_options(),
                    value=str(self._config.get("card_selection", "random")),
                    id="settings-splash-default-select",
                    classes="settings-compact-select",
                    allow_blank=False,
                    compact=True,
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Enabled", classes="settings-input-label")
                yield Switch(
                    value=bool(self._config.get("enabled", True)),
                    id="settings-splash-enabled",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Show progress", classes="settings-input-label")
                yield Switch(
                    value=bool(self._config.get("show_progress", True)),
                    id="settings-splash-show-progress",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Skip on keypress", classes="settings-input-label")
                yield Switch(
                    value=bool(self._config.get("skip_on_keypress", True)),
                    id="settings-splash-skip-on-keypress",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Duration", classes="settings-input-label")
                yield Input(
                    value=str(self._config.get("duration", 2.5)),
                    id="settings-splash-duration",
                    classes="settings-compact-input",
                    placeholder="seconds",
                    restrict=r"^[0-9]*\.?[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Animation speed", classes="settings-input-label")
                yield Input(
                    value=str(self._config.get("animation_speed", 1.0)),
                    id="settings-splash-animation-speed",
                    classes="settings-compact-input",
                    placeholder="multiplier",
                    restrict=r"^[0-9]*\.?[0-9]*$",
                )

            yield Static("Gallery", classes="destination-section")
            with Horizontal(id="settings-splash-gallery", classes="settings-splash-gallery"):
                yield OptionList(*self._card_options(), id="settings-splash-card-list")
                with VerticalScroll(id="settings-splash-preview-scroll"):
                    yield Static(
                        "Select a card to preview",
                        id="settings-splash-preview-placeholder",
                    )

            with Horizontal(classes="settings-action-row"):
                yield Button("Play selected", id="settings-splash-play", variant="primary")
                yield Button("Set as default", id="settings-splash-set-default")

            yield Static(
                "",
                id="settings-splash-status",
                classes="settings-status-row",
            )

    def on_mount(self) -> None:
        card_list = self.query_one("#settings-splash-card-list", OptionList)
        if self._cards:
            card_list.highlighted = 0

    def _update_status(self, message: str) -> None:
        status = self.query_one("#settings-splash-status", Static)
        status.update(message)

    def _save_config_value(self, key: str, value: Any) -> bool:
        try:
            save_setting_to_cli_config("splash_screen", key, value)
            self._config[key] = value
            self.post_message(self.SplashConfigChanged("splash_screen", key, value))
            return True
        except Exception as exc:
            logger.error("Failed to save splash_screen.{}: {}", key, exc)
            self._update_status(f"Error saving {key}: {exc}")
            return False

    def _float_or_default(self, raw: str, default: float) -> float:
        raw = raw.strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @on(Switch.Changed, "#settings-splash-enabled")
    def handle_enabled_changed(self, event: Switch.Changed) -> None:
        if self._save_config_value("enabled", event.value):
            self._update_status("Splash screen enabled setting saved.")

    @on(Switch.Changed, "#settings-splash-show-progress")
    def handle_show_progress_changed(self, event: Switch.Changed) -> None:
        if self._save_config_value("show_progress", event.value):
            self._update_status("Show progress setting saved.")

    @on(Switch.Changed, "#settings-splash-skip-on-keypress")
    def handle_skip_on_keypress_changed(self, event: Switch.Changed) -> None:
        if self._save_config_value("skip_on_keypress", event.value):
            self._update_status("Skip on keypress setting saved.")

    @on(Select.Changed, "#settings-splash-default-select")
    def handle_default_changed(self, event: Select.Changed) -> None:
        value = str(event.value) if event.value is not None else "random"
        if self._save_config_value("card_selection", value):
            self._update_status(f"Default splash card set to {value}.")

    @on(Input.Submitted, "#settings-splash-duration")
    def handle_duration_submitted(self, event: Input.Submitted) -> None:
        value = self._float_or_default(event.value, DEFAULT_SPLASH_CONFIG["duration"])
        if value < 0:
            value = 0
        if self._save_config_value("duration", value):
            self._update_status(f"Splash duration set to {value}s.")

    @on(Input.Submitted, "#settings-splash-animation-speed")
    def handle_animation_speed_submitted(self, event: Input.Submitted) -> None:
        value = self._float_or_default(
            event.value, DEFAULT_SPLASH_CONFIG["animation_speed"]
        )
        if value <= 0:
            value = DEFAULT_SPLASH_CONFIG["animation_speed"]
        if self._save_config_value("animation_speed", value):
            self._update_status(f"Animation speed set to {value}x.")

    @on(OptionList.OptionHighlighted, "#settings-splash-card-list")
    def handle_card_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        option_id = event.option_id
        if not option_id or option_id == "__none__":
            return
        self.selected_card = option_id
        self._mount_preview(option_id)

    @on(Button.Pressed, "#settings-splash-play")
    def handle_play_pressed(self) -> None:
        self._mount_preview(self.selected_card)
        self._update_status(f"Playing preview of {self.selected_card}.")

    @on(Button.Pressed, "#settings-splash-set-default")
    def handle_set_default_pressed(self) -> None:
        if self.selected_card and self.selected_card != "__none__":
            select = self.query_one("#settings-splash-default-select", Select)
            select.value = self.selected_card
            if self._save_config_value("card_selection", self.selected_card):
                self._update_status(f"{self.selected_card} set as default splash card.")

    def _mount_preview(self, card_name: str) -> None:
        container = self.query_one("#settings-splash-preview-scroll", VerticalScroll)
        for child in list(container.children):
            if isinstance(child, SplashScreen):
                child.close()
        container.remove_children()
        if card_name not in self._cards:
            container.mount(
                Static(
                    "Select a card to preview",
                    id="settings-splash-preview-placeholder",
                )
            )
            return

        card_data = self._cards[card_name]
        try:
            preview = SplashScreen(
                card_name=card_name,
                duration=0,
                show_progress=False,
                skip_on_keypress=False,
                classes="settings-splash-preview",
            )
            container.mount(preview)
        except Exception as exc:
            logger.error("Failed to mount splash preview for {}: {}", card_name, exc)
            container.mount(
                Static(
                    f"Preview unavailable for {card_name}: {exc}",
                    id="settings-splash-preview-placeholder",
                )
            )

    def on_unmount(self) -> None:
        try:
            container = self.query_one("#settings-splash-preview-scroll", VerticalScroll)
        except Exception:
            return
        for child in list(container.children):
            if isinstance(child, SplashScreen):
                child.close()
