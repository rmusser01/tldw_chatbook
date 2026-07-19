"""Blocking first-run setup modal for the native Console workbench.

While provider setup is incomplete (``ConsoleSetupCardState.mode == "card"``),
this widget overlays the Console workbench (rail + transcript + composer) with a
dim backdrop and a centered "Get started" card. The card shows the three live
setup steps and a single primary action that routes to provider recovery. It is
Console-scoped: it lives inside the ChatScreen (never an app-level modal), so the
top navigation tab bar stays reachable. The modal dismisses automatically the
moment readiness + model are satisfied (the guidance sync drives it).

The backdrop itself renders a drifting snow effect (``ConsoleSetupBackdrop``,
styled after the classic ZSNES emulator background) behind the card. Textual's
alpha-background compositing only blends a widget's background with its
*ancestor* style chain (see ``DOMNode.background_colors`` /
``Widget.opacity``) -- it does not re-composite the actual rendered pixels of
sibling widgets sitting on a lower layer. That was verified directly against
this Textual build: a same-color alpha fill over an identically-colored
ancestor is a no-op (which is why the previous ``$ds-surface-panel 80%`` fill
read as fully opaque), while a distinctly different token (``$background``,
darker than the Console shell's ``$ds-surface-panel``) blended at the same
layer produces a real, measurably darker fill. The Console workbench text
itself cannot "show through" the overlay under this widget architecture; the
snow backdrop is the closest achievable dim + motion flourish given that
constraint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_CARD_TITLE,
    ConsoleDetectedServerAction,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested


CONSOLE_SETUP_MODAL_STEP_COUNT = 3
CONSOLE_SETUP_MODAL_ACTION_ID = "console-setup-modal-action"
CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID = "console-setup-modal-detected-action"
CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION = "use-detected-local-server"
CONSOLE_SETUP_MODAL_BACKDROP_ID = "console-setup-modal-snow"
_DEFAULT_ACTION_LABEL = "Choose model"
_DEFAULT_ACTION_TOOLTIP = "Choose the provider and model for this Console session."

# Snow tuning: modest density (~1 flake per 30-50 cells), gentle tick cadence
# (0.15-0.25s), varied fall speed + a little horizontal wobble so the field
# doesn't look mechanical. Mirrors the composer cursor-blink timer discipline:
# created paused on mount, resumed only while blocking.
_SNOW_TICK_INTERVAL = 0.2
_SNOW_FLAKE_GLYPHS = ("·", "•", "*")  # ·, •, *
_SNOW_DENSITY_CELLS = 40
_SNOW_MIN_SPEED = 0.4
_SNOW_MAX_SPEED = 1.4
_SNOW_MAX_WOBBLE = 0.4


@dataclass
class _SnowFlake:
    """Mutable position/velocity state for a single falling glyph."""

    x: float
    y: float
    speed: float
    wobble: float
    glyph: str


class ConsoleSetupBackdrop(Static):
    """Dimmed backdrop behind the setup card with a drifting snow effect.

    Renders a grid of falling glyphs (mixed ``·`` / ``•`` / ``*``) that drift
    downward with a slight horizontal wobble, wrapping back to the top once
    past the bottom edge. Flake state is seeded from an injectable
    ``random.Random`` so tests can assert deterministic frames; production
    code leaves ``rng`` unset (default-seeded, non-deterministic) since the
    effect is purely decorative.

    The tick timer is created paused on mount and only resumed while the
    owning modal is actually blocking -- no background churn while the
    Console is idle or the modal is hidden, matching the composer cursor
    blink timer's discipline (``set_interval(..., pause=True)``, resumed /
    paused alongside visibility).
    """

    # Fallback sizing so the widget still fills its host when mounted in a
    # bare test harness (no app-level stylesheet loaded); the real Console
    # stylesheet's ``.console-setup-modal-backdrop-snow`` rule (width/height
    # 100%) takes precedence wherever it is loaded.
    DEFAULT_CSS = """
    ConsoleSetupBackdrop {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, *, rng: random.Random | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("id", CONSOLE_SETUP_MODAL_BACKDROP_ID)
        classes = kwargs.pop("classes", "")
        kwargs["classes"] = f"console-setup-modal-backdrop-snow {classes}".strip()
        super().__init__(**kwargs)
        self._rng = rng if rng is not None else random.Random()
        self._flakes: list[_SnowFlake] = []
        self._field_width = 0
        self._field_height = 0
        self._snow_timer: Timer | None = None
        # Intent flag: tracks whether the timer *should* be running, even
        # before the timer object exists (on_mount() runs after __init__, so
        # resume_snow()/pause_snow() can be called first -- see on_mount()).
        self._snow_should_run = False

    @property
    def flake_count(self) -> int:
        """Number of flakes currently tracked in the field."""
        return len(self._flakes)

    @property
    def timer_paused(self) -> bool:
        """Whether the snow-tick timer is currently paused."""
        return not self._snow_should_run

    def on_mount(self) -> None:
        self._snow_timer = self.set_interval(
            _SNOW_TICK_INTERVAL,
            self._tick,
            pause=True,
        )
        # Apply any resume intent recorded before the timer existed -- a
        # resume_snow() call that raced ahead of on_mount() must not be lost.
        if self._snow_should_run:
            self._snow_timer.resume()
        self._resize_flake_field()

    def on_resize(self, event: object) -> None:
        self._resize_flake_field()

    def resume_snow(self) -> None:
        """Resume the tick timer -- called while the modal is blocking."""
        self._snow_should_run = True
        if self._snow_timer is not None:
            self._snow_timer.resume()

    def pause_snow(self) -> None:
        """Pause the tick timer -- called while the modal is hidden."""
        self._snow_should_run = False
        if self._snow_timer is not None:
            self._snow_timer.pause()

    def _resize_flake_field(self) -> None:
        """Adapt the flake field to the widget's current size.

        Clamps existing flakes into the new bounds and tops up / trims the
        field to the target density. Degrades to an empty, crash-free field
        at zero/negative sizes (e.g. transient layout passes).
        """
        width, height = int(self.size.width), int(self.size.height)
        if width <= 0 or height <= 0:
            self._field_width = 0
            self._field_height = 0
            self._flakes = []
            self._render_flakes()
            return
        self._field_width = width
        self._field_height = height
        target_count = max(1, (width * height) // _SNOW_DENSITY_CELLS)
        for flake in self._flakes:
            flake.x = min(flake.x, float(width - 1))
            flake.y = min(flake.y, float(height - 1))
        if len(self._flakes) > target_count:
            self._flakes = self._flakes[:target_count]
        else:
            while len(self._flakes) < target_count:
                self._flakes.append(self._new_flake(seed_y=True))
        self._render_flakes()

    def _new_flake(self, *, seed_y: bool) -> _SnowFlake:
        width = max(self._field_width, 1)
        height = max(self._field_height, 1)
        return _SnowFlake(
            x=self._rng.uniform(0, max(width - 1, 0)),
            y=self._rng.uniform(0, max(height - 1, 0)) if seed_y else 0.0,
            speed=self._rng.uniform(_SNOW_MIN_SPEED, _SNOW_MAX_SPEED),
            wobble=self._rng.uniform(-_SNOW_MAX_WOBBLE, _SNOW_MAX_WOBBLE),
            glyph=self._rng.choice(_SNOW_FLAKE_GLYPHS),
        )

    def _tick(self) -> None:
        """Advance every flake one step and repaint the field."""
        width, height = self._field_width, self._field_height
        if width <= 0 or height <= 0:
            return
        for flake in self._flakes:
            flake.y += flake.speed
            flake.x += flake.wobble
            if flake.x < 0:
                flake.x = 0.0
                flake.wobble = abs(flake.wobble) or _SNOW_MAX_WOBBLE
            elif flake.x > width - 1:
                flake.x = float(width - 1)
                flake.wobble = -(abs(flake.wobble) or _SNOW_MAX_WOBBLE)
            if flake.y >= height:
                # Past the bottom: wrap to the top with a fresh x so the
                # field doesn't look like it's raining in vertical lines.
                flake.y = 0.0
                flake.x = self._rng.uniform(0, max(width - 1, 0))
        self._render_flakes()

    def _render_flakes(self) -> None:
        width, height = self._field_width, self._field_height
        if width <= 0 or height <= 0:
            self.update("")
            return
        rows = [[" "] * width for _ in range(height)]
        for flake in self._flakes:
            fx, fy = int(flake.x), int(flake.y)
            if 0 <= fx < width and 0 <= fy < height:
                rows[fy][fx] = flake.glyph
        self.update("\n".join("".join(row) for row in rows))


def _coerce_card_state(value: object) -> ConsoleSetupCardState:
    """Guard against a transiently non-``ConsoleSetupCardState`` value."""
    if isinstance(value, ConsoleSetupCardState):
        return value
    return ConsoleSetupCardState(mode="quiet")


class ConsoleSetupModal(Vertical):
    """Console-scoped blocking overlay carrying the first-run setup card."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "console-setup-modal")
        classes = kwargs.pop("classes", "")
        kwargs["classes"] = f"console-setup-modal-backdrop {classes}".strip()
        super().__init__(**kwargs)
        self._card_state = ConsoleSetupCardState(mode="quiet")
        self._action_label = _DEFAULT_ACTION_LABEL
        self._action_tooltip = _DEFAULT_ACTION_TOOLTIP
        self._detected_action: ConsoleDetectedServerAction | None = None
        # Hidden until a card-mode state is synced in.
        self.display = False

    @property
    def detected_server_action(self) -> ConsoleDetectedServerAction | None:
        """Return the currently offered detected-local-server action."""
        return self._detected_action

    @property
    def is_blocking(self) -> bool:
        """Return whether the modal is currently overlaying the workbench."""
        return self._card_state.mode == "card"

    def compose(self) -> ComposeResult:
        # Children mirror the container's blocking state so hidden-modal copy
        # never leaks into visible-text scrapes before the first guidance sync.
        blocking = self.is_blocking
        yield ConsoleSetupBackdrop(id=CONSOLE_SETUP_MODAL_BACKDROP_ID)
        card = Vertical(
            id="console-setup-modal-card",
            classes="console-setup-modal-card",
        )
        card.display = blocking
        with card:
            title = Static(
                CONSOLE_SETUP_CARD_TITLE,
                id="console-setup-modal-title",
                classes="console-setup-modal-title",
            )
            title.display = blocking
            yield title
            for index in range(1, CONSOLE_SETUP_MODAL_STEP_COUNT + 1):
                step = self._step_at(index)
                step_row = Static(
                    self._step_text(index, step),
                    id=f"console-setup-step-{index}",
                    classes=self._step_classes(step),
                    markup=False,
                )
                step_row.display = blocking
                yield step_row
            action = Button(
                self._action_label,
                id=CONSOLE_SETUP_MODAL_ACTION_ID,
                classes="console-setup-modal-action",
                compact=True,
            )
            action.tooltip = self._action_tooltip
            action.display = blocking
            yield action
            detected = Button(
                self._detected_action_label(),
                id=CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID,
                classes="console-setup-modal-action console-setup-modal-detected-action",
                compact=True,
            )
            detected.tooltip = self._detected_action_tooltip()
            detected.display = blocking and self._detected_action is not None
            yield detected

    def on_mount(self) -> None:
        self._sync_snow_timer()

    def _sync_snow_timer(self) -> None:
        """Resume the backdrop's snow tick only while actually blocking."""
        try:
            backdrop = self.query_one(
                f"#{CONSOLE_SETUP_MODAL_BACKDROP_ID}", ConsoleSetupBackdrop
            )
        except Exception:
            return
        if self.is_blocking:
            backdrop.resume_snow()
        else:
            backdrop.pause_snow()

    def sync_card_state(
        self,
        card_state: ConsoleSetupCardState,
        *,
        action_label: str = "",
        action_tooltip: str = "",
    ) -> None:
        """Refresh steps + primary action and toggle overlay visibility in place."""
        self._card_state = _coerce_card_state(card_state)
        self._action_label = action_label.strip() or _DEFAULT_ACTION_LABEL
        self._action_tooltip = action_tooltip.strip() or _DEFAULT_ACTION_TOOLTIP
        blocking = self.is_blocking
        self.display = blocking
        if not self.is_mounted:
            return
        self._sync_snow_timer()
        for index in range(1, CONSOLE_SETUP_MODAL_STEP_COUNT + 1):
            step = self._step_at(index)
            try:
                widget = self.query_one(f"#console-setup-step-{index}", Static)
            except Exception:
                continue
            widget.update(self._step_text(index, step))
            widget.set_classes(self._step_classes(step))
            # Own display must track blocking so hidden-modal content does not
            # leak into visible-text scrapes while the overlay is dismissed.
            widget.display = blocking
        for selector in ("#console-setup-modal-title", "#console-setup-modal-card"):
            try:
                self.query_one(selector).display = blocking
            except Exception:
                continue
        self._sync_detected_action_button()
        try:
            action = self.query_one(f"#{CONSOLE_SETUP_MODAL_ACTION_ID}", Button)
        except Exception:
            return
        action.label = self._action_label
        action.tooltip = self._action_tooltip
        action.display = blocking

    def sync_detected_server_action(
        self,
        action: ConsoleDetectedServerAction | None,
    ) -> None:
        """Offer (or withdraw) the detected-local-server secondary action.

        Args:
            action: Affordance built by ``build_console_detected_server_action``
                or ``None`` when no detected server should be offered.
        """
        self._detected_action = action if isinstance(action, ConsoleDetectedServerAction) else None
        if self.is_mounted:
            self._sync_detected_action_button()

    def _sync_detected_action_button(self) -> None:
        """Refresh the secondary detected-server button in place."""
        try:
            detected = self.query_one(
                f"#{CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID}", Button
            )
        except Exception:
            return
        detected.label = self._detected_action_label()
        detected.tooltip = self._detected_action_tooltip()
        detected.display = self.is_blocking and self._detected_action is not None

    def _detected_action_label(self) -> str:
        """Return the escaped label for the detected-server button."""
        if self._detected_action is None:
            return ""
        # Server-derived text (provider display + endpoint) must never be
        # interpreted as console markup inside a Button label.
        return escape_markup(self._detected_action.label)

    def _detected_action_tooltip(self) -> str:
        """Return the escaped tooltip for the detected-server button."""
        if self._detected_action is None:
            return ""
        # Tooltips render markup too; model ids/urls must stay literal.
        return escape_markup(self._detected_action.tooltip)

    def focus_primary_action(self) -> None:
        """Move focus to the modal's primary action button while blocking."""
        if not (self.is_mounted and self.is_blocking):
            return
        try:
            self.query_one(f"#{CONSOLE_SETUP_MODAL_ACTION_ID}", Button).focus()
        except Exception:
            return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route card actions through the owning Workbench screen."""
        if event.button.id == CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID:
            event.stop()
            self.post_message(
                WorkbenchActionRequested(CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION)
            )
            return
        if event.button.id != CONSOLE_SETUP_MODAL_ACTION_ID:
            return
        event.stop()
        self.post_message(WorkbenchActionRequested("provider-recovery"))

    def _step_at(self, index: int) -> ConsoleSetupStep:
        steps = self._card_state.steps
        if 1 <= index <= len(steps):
            return steps[index - 1]
        return ConsoleSetupStep(state="pending", label="")

    @staticmethod
    def _step_text(index: int, step: ConsoleSetupStep) -> str:
        if not step.label:
            return ""
        text = f"{index}. {step.glyph} {step.label}"
        if step.detail:
            text = f"{text}  {step.detail}"
        return text

    @staticmethod
    def _step_classes(step: ConsoleSetupStep) -> str:
        return f"console-setup-step console-setup-step-{step.state}"
