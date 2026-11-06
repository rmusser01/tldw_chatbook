"""The first-run wizard's Welcome step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.

TASK-34100.13 (entry-exit-handoff-18, E7): the copy no longer oversells
(the approved setup-shape spec's D1 time line, a Full label that says what
Full adds, privacy scoped to local models), the empty error band is gone,
and Welcome's two quiet rows -- "Reduce motion" and "Moving from another
computer? [Restore a backup]" -- sit below the nav bar, so Tab reaches Next
first (Textual's focus order is visual order).
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)

from textual import on
from textual.app import ComposeResult
from textual.containers import (
    Horizontal,
    Vertical,
)
from textual.widgets import (
    Button,
    Checkbox,
    RadioButton,
    Static,
)

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupCheckbox,
    SetupRadioButton,
    SetupRadioSet,
    SetupStep,
)

#: Approved setup-shape spec, D1 (amended 2026-10-03): the measured claim
#: with its condition, and the true "change it later" route.
WELCOME_TIME_LINE = (
    "Quick takes about 2 minutes if you have an API key or a local AI server "
    "running. You can change any of this later by running setup again."
)
#: D1's pitch: privacy is scoped to where the model runs, never "nothing
#: leaves your machine" while cloud voices and model-list refresh exist.
WELCOME_PITCH = (
    "Chat with cloud or local AI models, keep notes, and work with your own "
    "documents — all in your terminal. With a model running on this "
    "computer, your messages are answered on it."
)
#: What the Full track adds to today's Quick track (Voice and key protection
#: are on both), instead of "configure everything".
FULL_TRACK_LABEL = "Full setup — adds search, dictation, tools, appearance"


def _live_app_config(app_instance: Any) -> Optional[dict]:
    """The app's in-memory config when it can be changed live, else None."""
    app_config = getattr(app_instance, "app_config", None)
    return app_config if isinstance(app_config, dict) else None


class WelcomeFooter(Vertical):
    """Welcome's quiet rows, placed below the nav bar.

    E7: "Reduce motion" applies at once, to the in-memory settings the live
    animated surfaces read (Buddy, character expressions); WelcomeStep writes
    it on Next, as its own change, and the splash honours the saved value from
    the next launch. It is offered only where it can apply live.
    """

    DEFAULT_CLASSES = "setup-welcome-footer"

    def __init__(self, app_instance: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._app_instance = app_instance

    def compose(self) -> ComposeResult:
        # One row, so the step body keeps its height at 80x24 (live: two rows
        # pushed the track choice below the fold there).
        with Horizontal(classes="setup-welcome-restore-row"):
            app_config = _live_app_config(self._app_instance)
            if app_config is not None:
                appearance = app_config.get("appearance")
                current = bool(
                    isinstance(appearance, Mapping) and appearance.get("reduce_motion")
                )
                box = SetupCheckbox(
                    "Reduce motion",
                    current,
                    id="setup-welcome-reduce-motion",
                    compact=True,
                )
                box.tooltip = "Fewer animations, starting now."
                yield box
            yield Static(
                "Moving from another computer?",
                classes="setup-welcome-restore-label",
            )
            yield Button(
                "Restore a backup",
                id="setup-backup-restore",
                classes="setup-tertiary-button",
            )

    @on(Checkbox.Changed, "#setup-welcome-reduce-motion")
    def _on_reduce_motion(self, event: Checkbox.Changed) -> None:
        event.stop()
        app_config = _live_app_config(self._app_instance)
        if app_config is None:
            return
        appearance = app_config.get("appearance")
        if not isinstance(appearance, dict):
            appearance = {}
            app_config["appearance"] = appearance
        appearance["reduce_motion"] = bool(event.value)

    @on(Button.Pressed, "#setup-backup-restore")
    def open_backup_restore(self, event: Button.Pressed) -> None:
        """Keep setup choices intact while the app owns the recovery view."""
        event.stop()
        self.app.action_backup_restore("inspect")  # TASK-34100.16: open on Inspect


class WelcomeStep(SetupStep):
    """Track choice: Quick / Full / Skip."""

    _footer: Optional[WelcomeFooter] = None
    _reduce_motion_at_mount: bool = False

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-welcome"):
            yield Static("Welcome to tldw chatbook", classes="setup-title")
            yield Static(WELCOME_PITCH, classes="setup-subtitle")
            with SetupRadioSet(id="setup-track-choice", classes="setup-choice-list"):
                # TASK-2154.9 (FR-02): name the steps the tracker will show
                # (Welcome is this one; Provider, Model and Summary follow)
                # so the "Step 1 of 4" count is not a surprise after picking
                # what read as a two-item "provider & model" track.
                yield SetupRadioButton(
                    "Quick setup — provider, model, voice, protection (recommended)",
                    value=True,
                    id="setup-track-quick",
                )
                yield SetupRadioButton(FULL_TRACK_LABEL, id="setup-track-full")
            yield Static(WELCOME_TIME_LINE, classes="setup-subtitle")

    def _app_instance(self) -> Any:
        wizard = self.wizard
        if wizard is not None:
            return getattr(wizard, "app_instance", None)
        return self.app

    def on_mount(self) -> None:
        app_config = _live_app_config(self._app_instance()) or {}
        appearance = app_config.get("appearance")
        self._reduce_motion_at_mount = bool(
            isinstance(appearance, Mapping) and appearance.get("reduce_motion")
        )
        self._footer = WelcomeFooter(self._app_instance(), id="setup-welcome-footer")
        # The wizard's nav bar mounts in the same batch as this step; place
        # the rows once it is in the DOM.
        self.call_after_refresh(self._place_footer)

    def _place_footer(self) -> None:
        """Below the wizard's nav bar, so Tab reaches Next before these rows.

        A Welcome mounted on its own (no wizard container) keeps them in its
        own body.
        """
        footer = self._footer
        if footer is None or footer.parent is not None or not self.is_attached:
            return
        footer.display = not self.has_class("hidden")
        container = self.wizard if hasattr(self.wizard, "mount") else None
        nav = None
        if container is not None:
            nav = next(iter(container.query(".wizard-navigation")), None)
        if container is not None and nav is not None:
            container.mount(footer, after=nav)
        else:
            self.mount(footer)

    def on_unmount(self) -> None:
        """The rows live in the wizard container, so they leave with the step.

        A failed Welcome replaced by ``_retry_failed_step`` must not orphan
        them, or the new step's rows collide with the old id (review F11).
        """
        footer, self._footer = self._footer, None
        if footer is not None and footer.parent is not None and footer.parent is not self:
            footer.remove()

    def on_show(self) -> None:
        super().on_show()
        if self._footer is not None:
            self._footer.display = True
            self._show_saved_reduce_motion()

    def _show_saved_reduce_motion(self) -> None:
        """Review G13-R1-03: show Reduce motion as it is saved now.

        Appearance edits the same setting, and the wizard mirrors what it
        saves into ``app_config``; Back to Welcome showed the value from
        mount against a stale baseline, so Next there wrote nothing.
        """
        app_config = _live_app_config(self._app_instance())
        if app_config is None or self._footer is None:
            return
        try:
            box = self._footer.query_one("#setup-welcome-reduce-motion", Checkbox)
        except Exception:
            return  # not mounted yet: compose read the same value
        appearance = app_config.get("appearance")
        saved = bool(isinstance(appearance, Mapping) and appearance.get("reduce_motion"))
        with box.prevent(Checkbox.Changed):
            box.value = saved
        self._reduce_motion_at_mount = saved

    def on_hide(self) -> None:
        super().on_hide()
        if self._footer is not None:
            self._footer.display = False

    def _reduce_motion_choice(self) -> Optional[bool]:
        if self._footer is None:
            return None
        try:
            box = self._footer.query_one("#setup-welcome-reduce-motion", Checkbox)
        except Exception:
            return None
        return bool(box.value)

    async def commit(self) -> tuple[bool, str]:
        """Write "Reduce motion" only when it changed here (spec §4.3)."""
        chosen = self._reduce_motion_choice()
        if chosen is None or chosen == self._reduce_motion_at_mount:
            return True, ""
        ok = await self.wizard.commit_config({"appearance": {"reduce_motion": chosen}})
        if not ok:
            return False, "Saving the reduce-motion setting failed."
        self._reduce_motion_at_mount = chosen
        return True, ""

    def get_step_data(self) -> Dict[str, Any]:
        return {"track": self.chosen_track()}

    def busy_label(self) -> str:
        """What a slow Next from Welcome is doing (TASK-34100.1)."""
        full = self.chosen_track() == wizard_state.TRACK_FULL
        return f"Preparing the {'Full' if full else 'Quick'} setup…"

    def chosen_track(self) -> str:
        try:
            full = self.query_one("#setup-track-full", RadioButton).value
        except Exception:
            full = False
        return wizard_state.TRACK_FULL if full else wizard_state.TRACK_QUICK
