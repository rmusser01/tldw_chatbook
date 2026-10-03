"""The first-run wizard's Appearance step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
)

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import (
    Button,
    Label,
    RadioSet,
    Static,
)

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupRadioButton,
    SetupRadioSet,
    SetupStep,
)


class AppearanceStep(SetupStep):
    """Theme and splash card. Applies the theme live on commit (best effort)."""

    selected_theme: str = ""
    selected_splash_card: str = ""
    # Bug-2 fix: True only when the user EXPLICITLY re-picked "Surprise me"
    # this run (see _on_card) -- distinct from selected_splash_card=="",
    # which is ALSO true on a fresh mount where nothing was ever chosen
    # (RadioSet does not fire Changed for its own initial pre-selection).
    _picked_surprise_me: bool = False

    def compose_step(self) -> ComposeResult:
        # Re-run prefill: pre-select the theme RadioButton matching the
        # persisted default_theme, when it's in the rendered list. First-run
        # has no general.default_theme, so prefill.default_theme is "" and
        # nothing matches -- identical to the old always-unselected render.
        prefill = wizard_state.read_wizard_prefill(
            getattr(self.wizard.app_instance, "app_config", {}) or {}
        )
        # Bug-2a fix: initialize selected_theme from the persisted value.
        # RadioSet does not emit Changed for its own initial pre-selection
        # (only _on_theme below updates selected_theme), so without this a
        # rerun that never touches the theme radio left selected_theme=="",
        # and commit()'s old "fall back to textual-dark" default would
        # clobber the persisted theme just because some OTHER field (e.g.
        # only the splash card) changed on this step.
        self.selected_theme = prefill.default_theme
        with Vertical(classes="setup-appearance"):
            yield Static("Appearance", classes="setup-title")
            yield Label("Theme", classes="setup-field-label")
            with SetupRadioSet(id="setup-theme-choice", classes="setup-choice-list"):
                yield from self._theme_buttons(self._theme_shortlist())
            yield Button(
                "Show all themes…",
                id="setup-theme-show-all",
                classes="setup-tertiary-button",
            )
            yield Label("Splash screen card", classes="setup-field-label")
            with SetupRadioSet(id="setup-splash-choice", classes="setup-choice-list"):
                yield SetupRadioButton("Surprise me (random)", value=True)
                # TASK-21149 (UAT G-5): human names in the list; the raw
                # card id rides on the button (same pattern as _theme_name)
                # so commits never see display text.
                yield from self._card_buttons(self._card_names()[:10])
            yield Button(
                "Show all cards…",
                id="setup-splash-show-all",
                classes="setup-tertiary-button",
            )

    def _theme_buttons(self, names: list[str]):
        """Radio rows for theme names, marking the persisted one "(current)".

        TASK-1500: like the model rows, the label may carry decoration; the
        clean theme name rides on the button as ``_theme_name`` so previews
        and commits never see display text.
        """
        for theme_name in names:
            label = (
                f"{theme_name}   (current)"
                if theme_name == self.selected_theme and theme_name
                else theme_name
            )
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
        """Curated first screen: current + stock defaults + a few flagships.

        The full alphabetical wall (novelty themes first) buried the sane
        choices; "Show all themes…" swaps in the complete list on demand.
        """
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
        await radio_set.remove_children()
        await radio_set.mount_all(self._theme_buttons(self._theme_names()))
        self.query_one("#setup-theme-show-all", Button).display = False

    @staticmethod
    def _card_names() -> list[str]:
        try:
            from tldw_chatbook.Utils.Splash_Screens.card_definitions import (
                get_all_card_definitions,
            )

            return sorted(get_all_card_definitions())
        except Exception:
            return []

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
        """TASK-1500: selecting a theme applies it immediately as a preview.

        The pre-preview theme is remembered once so `revert_preview` can
        restore it if the user backs out (finish-later) without committing.
        A successful commit clears the revert obligation — the new theme is
        then the persisted one.
        """
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

    @staticmethod
    def _card_display_name(card_name: str) -> str:
        """Human name for a snake_case splash card id (UAT G-5)."""
        return card_name.replace("_", " ").strip().title()

    def _card_buttons(self, names: list[str]):
        """Radio rows for splash cards, pressing the retained selection.

        Args:
            names: Raw snake_case card ids in display order.

        Yields:
            SetupRadioButton rows with the human name as label and the raw
            id riding as ``_card_name`` (Qodo review: without value= here,
            show-all rebuilds and draft restoration rendered every card
            unpressed even when one was selected).
        """
        for card_name in names:
            button = SetupRadioButton(
                self._card_display_name(card_name),
                value=bool(card_name)
                and card_name == self.selected_splash_card,
            )
            button._card_name = card_name
            yield button

    @on(Button.Pressed, "#setup-splash-show-all")
    async def _on_show_all_cards(self, event: Button.Pressed) -> None:
        """UAT G-5: parity with themes — the first ten cards are a teaser."""
        event.stop()
        radio_set = self.query_one("#setup-splash-choice", RadioSet)
        keep_surprise = SetupRadioButton(
            "Surprise me (random)", value=not self.selected_splash_card
        )
        await radio_set.remove_children()
        await radio_set.mount(keep_surprise)
        await radio_set.mount_all(self._card_buttons(self._card_names()))
        self.query_one("#setup-splash-show-all", Button).display = False

    @on(RadioSet.Changed, "#setup-splash-choice")
    def _on_card(self, event: RadioSet.Changed) -> None:
        card_name = getattr(event.pressed, "_card_name", "")
        if not card_name:
            self.selected_splash_card = ""
            self._picked_surprise_me = True
        else:
            self.selected_splash_card = card_name
            self._picked_surprise_me = False

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            build_appearance_commit,
            read_wizard_prefill,
        )

        prefill = read_wizard_prefill(
            getattr(self.wizard.app_instance, "app_config", {}) or {}
        )
        # Bug-2c fix: only reset to "random" when the user EXPLICITLY
        # re-picked Surprise-me this run over a config that currently names
        # a specific card -- a fresh/no-op run (nothing pressed, or already
        # "random") must not write anything.
        reset_to_random = (
            self._picked_surprise_me
            and bool(prefill.card_selection)
            and prefill.card_selection != "random"
        )
        if (
            not self.selected_theme
            and not self.selected_splash_card
            and not reset_to_random
        ):
            return True, ""
        # Bug-2b fix: delta-aware theme write -- only persist default_theme
        # when the chosen theme actually differs from what's already on
        # disk, so a rerun that only changes the splash card (theme radio
        # left at its prefilled, already-persisted position) leaves the
        # persisted theme untouched instead of rewriting it (or a stale
        # "textual-dark" fallback) back over itself.
        chosen_theme = self.selected_theme or "textual-dark"
        theme_to_persist = (
            chosen_theme if chosen_theme != prefill.default_theme else None
        )
        ok = await self.wizard.commit_config(
            build_appearance_commit(
                default_theme=theme_to_persist,
                splash_card=self.selected_splash_card or None,
                reset_splash_to_random=reset_to_random,
            )
        )
        if ok and self.selected_theme:
            try:
                self.app.theme = self.selected_theme
            except Exception:
                logger.debug("Live theme apply failed; persisted value still wins")
            # TASK-1500: the commit made the previewed theme real — nothing
            # to revert on cancel any more.
            self._preview_original = None
        return (True, "") if ok else (False, "Saving appearance settings failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"theme": self.selected_theme, "splash_card": self.selected_splash_card}
