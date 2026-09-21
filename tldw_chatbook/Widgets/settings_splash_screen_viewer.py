"""Settings-native splash screen browser and preview widget."""

from __future__ import annotations

from typing import Any

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import QueryError
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from ..Constants import DEFAULT_SPLASH_DURATION_SECONDS
from ..config import (
    ConfigMutationResult,
    apply_settings_mutation_to_cli_config,
    get_cli_setting,
)
from ..Utils.Splash_Screens.card_definitions import get_all_card_definitions
from ..Widgets.splash_screen import SplashScreen


DEFAULT_SPLASH_CONFIG: dict[str, Any] = {
    "enabled": True,
    "duration": DEFAULT_SPLASH_DURATION_SECONDS,
    "skip_on_keypress": True,
    "card_selection": "random",
    "show_progress": True,
    "fade_in_duration": 0.3,
    "fade_out_duration": 0.2,
    "animation_speed": 1.0,
}


_EFFECTS_KEYS = {"fade_in_duration", "fade_out_duration", "animation_speed"}


def _config_section(key: str) -> str:
    return "splash_screen.effects" if key in _EFFECTS_KEYS else "splash_screen"


def switch_state_label(value: bool) -> str:
    """On/Off word beside a toggle: the widget alone carries state by
    position/color only, which is unreadable in reduced-color terminals and
    violates the text-labeled-states rule (task-1561)."""
    return "On" if value else "Off"


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
        self._pending_values: dict[str, Any] = {}
        self._status_revision = 0
        self._closing = False

    def _load_config(self) -> dict[str, Any]:
        try:
            # TASK-32804.8: one section read, then index it (same shape as
            # splash_screen.py's loader) instead of a call per key.
            section = get_cli_setting("splash_screen", default={})
            section = section if isinstance(section, dict) else {}
            effects = section.get("effects", {})
            effects = effects if isinstance(effects, dict) else {}
            config = {
                key: (effects if key in _EFFECTS_KEYS else section).get(key, value)
                for key, value in DEFAULT_SPLASH_CONFIG.items()
            }
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
            # task-1341: splash defaults persist on change; label the
            # instant-apply commit model inline (staged is the default).
            # Mirrors INSTANT_APPLY_BEHAVIOR_COPY in
            # UI/Screens/settings_screen.py (the widget must not import the
            # screen); the Enter clause covers the duration/animation-speed
            # Inputs, which persist on Input.Submitted, not per keystroke.
            yield Static(
                "applies immediately - no Save needed; text fields apply on Enter",
                id="settings-splash-instant-hint",
                classes="settings-instant-apply-hint",
            )
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
                label_static = Static("Enabled", classes="settings-input-label")
                # task-1561: the shared label column truncates longer
                # labels ("Skip on keypress" showed as "Skip on").
                label_static.add_class("w-20")
                yield label_static
                yield Checkbox(
                    value=bool(self._config.get("enabled", True)),
                    id="settings-splash-enabled",
                )
                yield Static(
                    switch_state_label(bool(self._config.get("enabled", True))),
                    id="settings-splash-enabled-state",
                    classes="settings-toggle-state",
                )
            with Horizontal(classes="settings-input-row"):
                label_static = Static("Show progress", classes="settings-input-label")
                # task-1561: the shared label column truncates longer
                # labels ("Skip on keypress" showed as "Skip on").
                label_static.add_class("w-20")
                yield label_static
                yield Checkbox(
                    value=bool(self._config.get("show_progress", True)),
                    id="settings-splash-show-progress",
                )
                yield Static(
                    switch_state_label(bool(self._config.get("show_progress", True))),
                    id="settings-splash-show-progress-state",
                    classes="settings-toggle-state",
                )
            with Horizontal(classes="settings-input-row"):
                label_static = Static("Skip on keypress", classes="settings-input-label")
                # task-1561: the shared label column truncates longer
                # labels ("Skip on keypress" showed as "Skip on").
                label_static.add_class("w-20")
                yield label_static
                yield Checkbox(
                    value=bool(self._config.get("skip_on_keypress", True)),
                    id="settings-splash-skip-on-keypress",
                )
                yield Static(
                    switch_state_label(bool(self._config.get("skip_on_keypress", True))),
                    id="settings-splash-skip-on-keypress-state",
                    classes="settings-toggle-state",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Duration (s)", classes="settings-input-label")
                yield Input(
                    value=str(self._config.get("duration", 2.5)),
                    id="settings-splash-duration",
                    classes="settings-compact-input",
                    placeholder="seconds",
                    restrict=r"^[0-9]*\.?[0-9]*$",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Animation speed (x)", classes="settings-input-label")
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

            yield Static(
                "",
                id="settings-splash-status",
                classes="settings-status-row",
            )

    def on_mount(self) -> None:
        """Initialize after composed descendants are mounted."""
        self.call_after_refresh(self._initialize_card_list)

    def _initialize_card_list(self) -> None:
        """Select the first available splash card."""
        try:
            card_list = self.query_one("#settings-splash-card-list", OptionList)
        except QueryError:
            # Settings can recompose while this callback is queued. A stale,
            # detached viewer must not fail the replacement screen.
            return
        if self._cards:
            card_list.highlighted = 0

    def _update_status(self, message: str) -> None:
        if self._closing or not self.is_attached:
            return
        try:
            self.query_one("#settings-splash-status", Static).update(message)
        except QueryError:
            # Navigation can detach the category before a write finishes.
            return

    def _control_for_key(self, key: str):
        suffix = "default-select" if key == "card_selection" else key.replace("_", "-")
        return self.query_one(f"#settings-splash-{suffix}")

    def _sync_control_value(self, key: str, value: Any) -> None:
        control = self._control_for_key(key)
        with self.prevent(Checkbox.Changed, Select.Changed, Input.Changed):
            control.value = str(value) if isinstance(control, Input) else value
        if isinstance(control, Checkbox):
            self.query_one(f"#{control.id}-state", Static).update(
                switch_state_label(bool(value))
            )

    def _save_config_value(self, key: str, value: Any) -> bool:
        """Keep the confirmed value while one write per control is pending."""
        if key in self._pending_values:
            if not isinstance(self._control_for_key(key), Input):
                self._sync_control_value(key, self._pending_values[key])
            return False
        self._sync_control_value(key, value)
        if self._config.get(key) == value:
            return False
        self._pending_values[key] = value
        self._status_revision += 1
        self._update_status(f"Saving {key.replace('_', ' ')}…")
        self._persist_splash_config_value(key, value, self._status_revision, self.app)
        return True

    @work(thread=True)
    def _persist_splash_config_value(
        self, key: str, value: Any, revision: int, app
    ) -> None:
        """Write off the event loop and preserve the owner's two-phase outcome."""
        error = None
        try:
            result = apply_settings_mutation_to_cli_config(
                {_config_section(key): {key: value}}
            )
        except Exception as exc:
            logger.error("Failed to save splash setting {}: {}", key, exc)
            result = ConfigMutationResult(False, False, "before_replace")
            error = str(exc)
        try:
            app.call_from_thread(
                self._finish_persist, key, value, revision, result, error
            )
        except RuntimeError:
            # The file outcome remains authoritative after app shutdown.
            logger.debug("Splash write finished after the app stopped")

    def _finish_persist(
        self,
        key: str,
        value: Any,
        revision: int,
        result: ConfigMutationResult,
        error: str | None,
    ) -> None:
        self._pending_values.pop(key, None)
        if result.file_replaced:
            self._config[key] = value
            self.post_message(
                self.SplashConfigChanged(_config_section(key), key, value)
            )
        if self._closing or not self.is_attached:
            return
        try:
            control = self._control_for_key(key)
            newer_input = isinstance(control, Input) and control.value != str(value)
            if not newer_input:
                self._sync_control_value(key, self._config[key])
        except QueryError:
            return
        if revision != self._status_revision:
            return
        label = key.replace("_", " ")
        if not result.file_replaced:
            message = (
                f"Error saving {key}: {error or 'configuration write was not accepted'}"
            )
        elif not result.caches_reloaded:
            message = f"Saved {label}, but configuration refresh failed. Reopen Settings to refresh."
        else:
            message = f"Saved {label}."
        if newer_input:
            message += " Press Enter to save your newer value."
        self._update_status(message)

    def _float_or_default(self, raw: str, default: float) -> float:
        raw = raw.strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @on(Checkbox.Changed, "#settings-splash-enabled")
    def handle_enabled_changed(self, event: Checkbox.Changed) -> None:
        self.query_one("#settings-splash-enabled-state", Static).update(
            switch_state_label(bool(event.value))
        )
        self._save_config_value("enabled", event.value)

    @on(Checkbox.Changed, "#settings-splash-show-progress")
    def handle_show_progress_changed(self, event: Checkbox.Changed) -> None:
        self.query_one("#settings-splash-show-progress-state", Static).update(
            switch_state_label(bool(event.value))
        )
        self._save_config_value("show_progress", event.value)

    @on(Checkbox.Changed, "#settings-splash-skip-on-keypress")
    def handle_skip_on_keypress_changed(self, event: Checkbox.Changed) -> None:
        self.query_one("#settings-splash-skip-on-keypress-state", Static).update(
            switch_state_label(bool(event.value))
        )
        self._save_config_value("skip_on_keypress", event.value)

    @on(Select.Changed, "#settings-splash-default-select")
    def handle_default_changed(self, event: Select.Changed) -> None:
        value = str(event.value) if event.value is not None else "random"
        self._save_config_value("card_selection", value)

    @on(Input.Submitted, "#settings-splash-duration")
    def handle_duration_submitted(self, event: Input.Submitted) -> None:
        value = self._float_or_default(event.value, DEFAULT_SPLASH_CONFIG["duration"])
        if value < 0:
            value = 0
        self._save_config_value("duration", value)

    @on(Input.Submitted, "#settings-splash-animation-speed")
    def handle_animation_speed_submitted(self, event: Input.Submitted) -> None:
        value = self._float_or_default(
            event.value, DEFAULT_SPLASH_CONFIG["animation_speed"]
        )
        if value <= 0:
            value = DEFAULT_SPLASH_CONFIG["animation_speed"]
        self._save_config_value("animation_speed", value)

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
        self._closing = True
        try:
            container = self.query_one("#settings-splash-preview-scroll", VerticalScroll)
        except Exception:
            return
        for child in list(container.children):
            if isinstance(child, SplashScreen):
                child.close()
