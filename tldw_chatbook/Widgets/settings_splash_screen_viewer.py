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

    def _load_config(self) -> dict[str, Any]:
        try:
            _EFFECTS_KEYS = {"fade_in_duration", "fade_out_duration", "animation_speed"}
            config = {
                key: get_cli_setting(
                    "splash_screen.effects" if key in _EFFECTS_KEYS else "splash_screen",
                    key,
                    value,
                )
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
                label_static.styles.width = 20
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
                label_static.styles.width = 20
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
                label_static.styles.width = 20
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
        status = self.query_one("#settings-splash-status", Static)
        status.update(message)

    def _save_config_value(self, key: str, value: Any) -> bool:
        """Apply one splash_screen setting and persist it off the event loop.

        task-15470: six Checkbox/Select/Input handlers below all funnel
        through this one method, and it used to call
        ``save_setting_to_cli_config`` -- a full config.toml read+atomic-
        rewrite+cache-reload -- synchronously on the event loop, once per
        click/submit, returning True/False for real, synchronously-known
        success/failure. The in-memory ``_config`` update and
        ``SplashConfigChanged`` message (both pure, no I/O) still happen
        immediately here, optimistically; only the disk write is deferred
        to a worker, so this can no longer report a confirmed outcome --
        the return value now means "dispatched", not "saved". If the
        write actually fails, `_persist_splash_config_value` reverts
        `_config[key]` back to `previous` (so memory never diverges from
        what is actually on disk) and overwrites the status line with the
        error, so an optimistic "... saved." message a caller shows right
        after this call is always eventually corrected rather than
        silently wrong. Callers no longer gate their "saved" message on
        this return value (review round: it was always True, making six
        `if` guards dead weight) -- they call it unconditionally now.
        """
        previous = self._config.get(key)
        self._config[key] = value
        self.post_message(self.SplashConfigChanged("splash_screen", key, value))
        self._persist_splash_config_value(key, value, previous)
        return True

    @work(thread=True)
    def _persist_splash_config_value(
        self, key: str, value: Any, previous: Any
    ) -> None:
        """Write one ``[splash_screen]`` config value on a worker thread.

        On failure, hands the correction back to the main thread via
        ``self.app.call_from_thread`` -- ``call_from_thread`` is an ``App``
        method, not available on a ``Widget``; calling it as
        ``self.call_from_thread`` crashed the whole app on any write
        failure here, since the resulting ``AttributeError`` was raised
        from inside this ``except`` block, uncaught, inside a
        ``@work(thread=True)`` worker, where an uncaught exception is
        fatal by default (``exit_on_error=True``).
        """
        try:
            save_setting_to_cli_config("splash_screen", key, value)
        except Exception as exc:
            logger.error("Failed to save splash_screen.{}: {}", key, exc)
            self.app.call_from_thread(
                self._handle_persist_failure, key, previous, exc
            )

    def _handle_persist_failure(self, key: str, previous: Any, exc: Exception) -> None:
        """Revert the optimistic in-memory value and surface the error.

        Runs on the main thread (via ``call_from_thread``). Reverting
        `_config[key]` closes the memory/disk divergence the optimistic
        update in `_save_config_value` would otherwise leave behind on a
        failed write; the corrective `SplashConfigChanged` message mirrors
        the optimistic one posted there, for the same reason.
        """
        self._config[key] = previous
        self.post_message(self.SplashConfigChanged("splash_screen", key, previous))
        self._update_status(f"Error saving {key}: {exc}")

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
        self._update_status("Splash screen enabled setting saved.")

    @on(Checkbox.Changed, "#settings-splash-show-progress")
    def handle_show_progress_changed(self, event: Checkbox.Changed) -> None:
        self.query_one("#settings-splash-show-progress-state", Static).update(
            switch_state_label(bool(event.value))
        )
        self._save_config_value("show_progress", event.value)
        self._update_status("Show progress setting saved.")

    @on(Checkbox.Changed, "#settings-splash-skip-on-keypress")
    def handle_skip_on_keypress_changed(self, event: Checkbox.Changed) -> None:
        self._save_config_value("skip_on_keypress", event.value)
        self._update_status("Skip on keypress setting saved.")

    @on(Select.Changed, "#settings-splash-default-select")
    def handle_default_changed(self, event: Select.Changed) -> None:
        value = str(event.value) if event.value is not None else "random"
        self._save_config_value("card_selection", value)
        self._update_status(f"Default splash card set to {value}.")

    @on(Input.Submitted, "#settings-splash-duration")
    def handle_duration_submitted(self, event: Input.Submitted) -> None:
        value = self._float_or_default(event.value, DEFAULT_SPLASH_CONFIG["duration"])
        if value < 0:
            value = 0
        self._save_config_value("duration", value)
        self._update_status(f"Splash duration set to {value}s.")

    @on(Input.Submitted, "#settings-splash-animation-speed")
    def handle_animation_speed_submitted(self, event: Input.Submitted) -> None:
        value = self._float_or_default(
            event.value, DEFAULT_SPLASH_CONFIG["animation_speed"]
        )
        if value <= 0:
            value = DEFAULT_SPLASH_CONFIG["animation_speed"]
        self._save_config_value("animation_speed", value)
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
        try:
            container = self.query_one("#settings-splash-preview-scroll", VerticalScroll)
        except Exception:
            return
        for child in list(container.children):
            if isinstance(child, SplashScreen):
                child.close()
