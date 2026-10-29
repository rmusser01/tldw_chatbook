"""The first-run wizard's Appearance step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.

TASK-34100.13 (coverage-15, fulltrack-09, fulltrack-13): the splash-card
gallery is gone. The step offers the theme (display names with a tone tag),
"Startup animation: Off / Short / Full", "Reduce motion" and "Plain ASCII
status marks", and names the current splash card read-only. Facts and
commits live in ``first_run_appearance_state``.
"""

from __future__ import annotations

import os
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
)

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import (
    Button,
    Checkbox,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

from tldw_chatbook.UI.Wizards import first_run_appearance_state as look
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupCheckbox,
    SetupRadioButton,
    SetupRadioSet,
    SetupStep,
)


class AppearanceStep(SetupStep):
    """Theme, startup animation and motion comfort. Theme applies live on commit."""

    selected_theme: str = ""

    def __init__(
        self,
        wizard=None,
        config=None,
        *,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._environ = os.environ if environ is None else environ
        self._comfort_touched = False
        self._read_current()

    def _app_config(self) -> Mapping[str, Any]:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        return app_config if isinstance(app_config, Mapping) else {}

    def _read_current(self) -> None:
        """What is saved now, from the full TOML (load_settings has no splash table)."""
        sections = wizard_state.wizard_config_sections(
            self.wizard, "splash_screen", "appearance"
        )
        self._current_startup = look.startup_choice(sections["splash_screen"])
        self._current_reduce, self._current_ascii = look.read_comfort(
            sections["appearance"]
        )
        self._ascii_start = look.ascii_checkbox_value(
            sections["appearance"],
            self._environ,
            rerun=bool(getattr(self.wizard, "rerun", False)),
        )
        self._card_selection = str(
            sections["splash_screen"].get("card_selection") or ""
        )

    def compose_step(self) -> ComposeResult:
        # Re-run prefill: pre-select the persisted theme (Bug-2a: a re-run
        # that never touches the theme radio must not clobber it on commit).
        prefill = wizard_state.read_wizard_prefill(self._app_config())
        self.selected_theme = prefill.default_theme
        with Vertical(classes="setup-appearance"):
            yield Static("Appearance", classes="setup-title")
            yield Label("Theme", classes="setup-field-label")
            with SetupRadioSet(
                id="setup-theme-choice", classes="setup-choice-list setup-choice-fit"
            ):
                yield from self._theme_buttons(self._theme_shortlist())
            yield Button(
                "Show all themes…",
                id="setup-theme-show-all",
                classes="setup-tertiary-button",
            )
            yield Label("Startup animation", classes="setup-field-label")
            with SetupRadioSet(
                id="setup-startup-choice", classes="setup-choice-list setup-choice-fit"
            ):
                for choice in look.STARTUP_CHOICES:
                    yield SetupRadioButton(
                        look.startup_label(choice),
                        value=choice == self._current_startup,
                        id=f"setup-startup-{choice}",
                    )
            yield SetupCheckbox(
                "Reduce motion — still frames instead of animation",
                self._current_reduce,
                id="setup-appearance-reduce-motion",
            )
            yield SetupCheckbox(
                "Plain ASCII status marks — for fonts without symbols (+ - ! x)",
                self._ascii_start,
                id="setup-appearance-ascii",
            )
            yield Static(
                look.splash_card_line(self._card_selection),
                id="setup-appearance-splash-card",
                classes="setup-step-note",
                markup=False,
            )

    def on_show(self) -> None:
        super().on_show()
        # Welcome's "Reduce motion" applies live to app_config; mirror it
        # here until the user touches this step's own checkbox.
        if self._comfort_touched:
            return
        appearance = self._app_config().get("appearance")
        if not isinstance(appearance, Mapping):
            return
        live, _ = look.read_comfort(appearance)
        try:
            box = self.query_one("#setup-appearance-reduce-motion", Checkbox)
        except Exception:
            return
        with box.prevent(Checkbox.Changed):
            box.value = live
        # Welcome saved it on its Next: the shown value is the saved one.
        self._current_reduce = live

    @on(Checkbox.Changed, "#setup-appearance-reduce-motion")
    def _on_reduce_motion(self, event: Checkbox.Changed) -> None:
        event.stop()
        self._comfort_touched = True

    # -- themes ------------------------------------------------------------

    def _theme_label(self, theme_name: str) -> str:
        """fulltrack-13: a display name with a tone tag, never the raw id."""
        try:
            themes = self.app.available_themes
        except Exception:
            themes = None
        return look.theme_display_label(
            theme_name, dark=look.theme_tone(theme_name, themes)
        )

    def _theme_buttons(self, names: list[str]):
        """Radio rows for theme names, marking the persisted one "(current)".

        TASK-1500: the clean theme name rides on the button as
        ``_theme_name`` so previews and commits never see display text.
        """
        for theme_name in names:
            label = self._theme_label(theme_name)
            if theme_name == self.selected_theme and theme_name:
                label = f"{label}   (current)"
            button = SetupRadioButton(label, value=(theme_name == self.selected_theme))
            button._theme_name = theme_name
            yield button

    # TASK-1500: flagship candidates for the shortlist, in preference order.
    # Filtered against what this Textual build actually registers; the two
    # stock themes are always present.
    _FLAGSHIP_THEMES = ("nord", "gruvbox", "tokyo-night", "catppuccin-mocha")

    def _theme_names(self) -> list[str]:
        try:
            return sorted(self.app.available_themes)
        except Exception:
            return ["textual-dark", "textual-light"]

    def _theme_shortlist(self) -> list[str]:
        """Curated first screen: current + stock defaults + a few flagships."""
        available = self._theme_names()
        shortlist: list[str] = []
        for name in (
            self.selected_theme,
            "textual-dark",
            "textual-light",
            *self._FLAGSHIP_THEMES,
        ):
            if name and name in available and name not in shortlist:
                shortlist.append(name)
        return shortlist or available[:6]

    @on(Button.Pressed, "#setup-theme-show-all")
    async def _on_show_all_themes(self, event: Button.Pressed) -> None:
        event.stop()
        radio_set = self.query_one("#setup-theme-choice", RadioSet)
        # The full list is a catalog again: back to the capped, scrolling box.
        radio_set.remove_class("setup-choice-fit")
        await radio_set.remove_children()
        await radio_set.mount_all(self._theme_buttons(self._theme_names()))
        self.query_one("#setup-theme-show-all", Button).display = False

    #: Theme active before the first preview; None = nothing to revert.
    _preview_original: Optional[str] = None

    @on(RadioSet.Changed, "#setup-theme-choice")
    def _on_theme(self, event: RadioSet.Changed) -> None:
        if event.pressed is None:
            return
        # Clean value, never the "(current)"-decorated label.
        self.selected_theme = str(
            getattr(event.pressed, "_theme_name", event.pressed.label)
        )
        self._preview_theme(self.selected_theme)

    def _preview_theme(self, theme_name: str) -> None:
        """TASK-1500: selecting a theme applies it immediately as a preview."""
        if not theme_name:
            return
        try:
            if self._preview_original is None:
                self._preview_original = str(self.app.theme)
            self.app.theme = theme_name
        except Exception:
            logger.debug("Theme preview failed for %s", theme_name, exc_info=True)

    def revert_preview(self) -> None:
        """Restore the pre-preview theme (no-op when nothing was previewed)."""
        if self._preview_original is not None:
            try:
                self.app.theme = self._preview_original
            except Exception:
                logger.debug("Theme preview revert failed", exc_info=True)
            self._preview_original = None

    # -- state -------------------------------------------------------------

    def _pressed_startup(self) -> str:
        try:
            choice = self.query_one("#setup-startup-choice", RadioSet)
        except Exception:
            return self._current_startup
        pressed = choice.pressed_button
        if pressed is None or pressed not in choice.query(RadioButton):
            return self._current_startup
        return (pressed.id or "").removeprefix("setup-startup-")

    def _checked(self, selector: str, default: bool) -> bool:
        try:
            return bool(self.query_one(selector, Checkbox).value)
        except Exception:
            return default

    def restore_draft(
        self,
        values: Mapping[str, Any],
        restore_radio: Callable[[RadioSet, Callable[[RadioButton], bool]], None],
    ) -> None:
        """Put a resumed draft's Appearance choices back on the controls."""
        if "theme" in values:
            theme = str(values["theme"])
            self.selected_theme = theme
            restore_radio(
                self.query_one("#setup-theme-choice", RadioSet),
                lambda button: getattr(button, "_theme_name", "") == theme,
            )
        if values.get("startup") in look.STARTUP_CHOICES:
            startup = str(values["startup"])
            restore_radio(
                self.query_one("#setup-startup-choice", RadioSet),
                lambda button: button.id == f"setup-startup-{startup}",
            )
        for key, selector in (
            ("reduce_motion", "#setup-appearance-reduce-motion"),
            ("ascii_glyphs", "#setup-appearance-ascii"),
        ):
            if isinstance(values.get(key), bool):
                box = self.query_one(selector, Checkbox)
                with box.prevent(Checkbox.Changed):
                    box.value = values[key]
                self._comfort_touched = True

    async def commit(self) -> tuple[bool, str]:
        # Review V2: the delta is against what the controls were built from
        # (``_current_*``), never a fresh read -- a config that moved since
        # then would make an untouched step write its shown values back.
        prefill = wizard_state.read_wizard_prefill(self._app_config())
        commit: dict[str, dict[str, Any]] = {}
        # Bug-2b: write the theme only when it differs from what is saved.
        if self.selected_theme and self.selected_theme != prefill.default_theme:
            commit["general"] = {"default_theme": self.selected_theme}
        startup = self._pressed_startup()
        commit.update(
            look.build_startup_commit(choice=startup, current=self._current_startup)
        )
        reduce_motion = self._checked(
            "#setup-appearance-reduce-motion", self._current_reduce
        )
        ascii_glyphs = self._checked("#setup-appearance-ascii", self._current_ascii)
        commit.update(
            look.build_comfort_commit(
                reduce_motion=reduce_motion,
                ascii_glyphs=ascii_glyphs,
                current_reduce_motion=self._current_reduce,
                current_ascii_glyphs=self._current_ascii,
            )
        )
        if not commit:
            return True, ""
        ok = await self.wizard.commit_config(commit)
        if not ok:
            return False, "Saving appearance settings failed."
        if "general" in commit:
            try:
                self.app.theme = self.selected_theme
            except Exception:
                logger.debug("Live theme apply failed; persisted value still wins")
            # TASK-1500: the commit made the previewed theme real.
            self._preview_original = None
        if "ascii_glyphs" in commit.get("appearance", {}):
            from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode

            set_ascii_glyph_mode(ascii_glyphs)
        # What is saved now is what the controls show.
        self._current_startup = startup
        self._current_reduce, self._current_ascii = reduce_motion, ascii_glyphs
        return True, ""

    def get_step_data(self) -> Dict[str, Any]:
        return {
            "theme": self.selected_theme,
            "startup": self._pressed_startup(),
            "reduce_motion": self._checked(
                "#setup-appearance-reduce-motion", self._current_reduce
            ),
            "ascii_glyphs": self._checked(
                "#setup-appearance-ascii", self._current_ascii
            ),
        }
