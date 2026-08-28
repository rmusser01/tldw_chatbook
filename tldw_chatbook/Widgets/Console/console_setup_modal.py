"""Blocking first-run setup modal for the native Console workbench.

While provider setup is incomplete (``ConsoleSetupCardState.mode == "card"``),
this widget overlays the Console workbench (rail + transcript + composer) with a
dim backdrop and a centered "Get started" card. The card shows the three live
setup steps and a single primary action that routes to provider recovery. It is
Console-scoped: it lives inside the ChatScreen (never an app-level modal), so the
top navigation tab bar stays reachable. The modal dismisses automatically the
moment readiness + model are satisfied (the guidance sync drives it).

The backdrop itself renders a still snow field (``ConsoleSetupBackdrop``,
styled after the classic ZSNES emulator background) behind the card -- drawn
once per (re)size, never on a clock; see the comment above
``_SNOW_FLAKE_GLYPHS`` for the measurements that retired the animation
(TASK-23021). Textual's
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
snow backdrop is the closest achievable dim + decoration given that
constraint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Key
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_CARD_SUBTITLE,
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

#: TASK-2154.8 (FR-09): one informational toast per blocking episode when the
#: user types while the composer is locked. Not per keystroke -- a toast on
#: every letter would be spam; one per episode makes the lock perceivable.
_TYPING_LOCKED_HINT = (
    "Typing is locked until setup finishes — press Enter to continue setup."
)

# Snow tuning: modest density (~1 flake per 40 cells). The field is a STILL
# frame -- drawn when the widget (re)sizes, never on a clock.
#
# TASK-21134 halved the old tick's rate (5 Hz -> 2.5 Hz) and dropped its
# layout pass; the tick itself got cheap (30 ms per 15 s) and the burn
# survived one layer down. TASK-23021 measured where it lived: this backdrop
# spans the whole Console shell, so each tick's `Static.update` dirtied a
# full-viewport region and Textual's compositor re-rendered every widget
# overlapping the dirty crop -- 124 widget renders x 44 rows, 13-16 ms per
# repaint inside `Screen._on_timer_update`, 3.6-4.3% of a core at idle on the
# first screen every new user sees, against a 0.04% floor with the tick
# neutralised. Shrinking the dirty region does not help: Textual's
# `Compositor.render_partial_update` crops to the *bounding box* of the dirty
# cells, and flakes span the field, so a per-cell-dirty variant (the
# render_line shape) measured 2.7-3.6% -- statistically the same burn. Even a
# single 3x3-cell repaint at the same 2.5 Hz measured ~0.55%, an order of
# magnitude above the floor, because ~30 widgets stack under any cell of this
# overlay. On this screen ANY repeating repaint is too expensive for a
# decoration, so the animation is retired: the flakes hold still, the field
# re-scatters only on resize, and idle cost is zero. `appearance.
# reduce_motion` keeps its meaning -- the field it used to freeze is now
# frozen for everyone.
_SNOW_FLAKE_GLYPHS = ("·", "•", "*")  # ·, •, *
_SNOW_DENSITY_CELLS = 40


@dataclass
class _SnowFlake:
    """Position state for a single glyph in the still flake field.

    Kept as a mutable dataclass (not a bare tuple) because a resize clamps
    existing flakes into the new bounds in place, so the field stays visually
    stable across resizes instead of re-scattering wholesale.
    """

    x: float
    y: float
    glyph: str


class ConsoleSetupBackdrop(Static):
    """Dimmed backdrop behind the setup card with a still snow field.

    Renders a static scatter of glyphs (mixed ``·`` / ``•`` / ``*``) over the
    dim fill -- the ZSNES-style flourish, minus the motion (see the module
    comment above ``_SNOW_FLAKE_GLYPHS`` for the measured reason,
    TASK-23021). The field is (re)drawn only when the widget's size changes;
    between resizes the widget arms no timers, performs no repaints, and
    dirties nothing. Flake placement is seeded from an injectable
    ``random.Random`` so tests can assert deterministic frames; production
    code leaves ``rng`` unset (default-seeded, non-deterministic) since the
    effect is purely decorative.
    """

    # Fallback sizing so the widget still fills its host when mounted in a
    # bare test harness (no app-level stylesheet loaded); the real Console
    # stylesheet's ``.console-setup-modal-backdrop-snow`` rule (width/height
    # 100%) takes precedence wherever it is loaded.
    BUNDLED_CSS = """
    ConsoleSetupBackdrop {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("id", CONSOLE_SETUP_MODAL_BACKDROP_ID)
        classes = kwargs.pop("classes", "")
        kwargs["classes"] = f"console-setup-modal-backdrop-snow {classes}".strip()
        # TASK-21134: the field is only spaces and the three flake glyphs, none
        # of which is markup. Parsing it as console markup on every repaint is
        # pure waste, and a glyph set that grew a "[" would otherwise silently
        # become a markup tag.
        kwargs.setdefault("markup", False)
        super().__init__(**kwargs)
        self._rng = rng if rng is not None else random.Random()
        self._flakes: list[_SnowFlake] = []
        self._field_width = 0
        self._field_height = 0

    @property
    def flake_count(self) -> int:
        """Number of flakes currently tracked in the field."""
        return len(self._flakes)

    def on_mount(self) -> None:
        self._resize_flake_field()

    def on_resize(self, event: object) -> None:
        self._resize_flake_field()

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
                self._flakes.append(self._new_flake())
        self._render_flakes()

    def _new_flake(self) -> _SnowFlake:
        width = max(self._field_width, 1)
        height = max(self._field_height, 1)
        return _SnowFlake(
            x=self._rng.uniform(0, max(width - 1, 0)),
            y=self._rng.uniform(0, max(height - 1, 0)),
            glyph=self._rng.choice(_SNOW_FLAKE_GLYPHS),
        )

    def _render_flakes(self) -> None:
        """Draw the still flake field.

        Reached only from the mount/resize path, where the field's dimensions
        really did just change -- so the default ``layout=True`` of
        ``Static.update`` is correct here, and there is no repeating caller
        left to need the TASK-21134 ``layout=False`` opt-out.
        """
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
        # Task-2852: a receipt line for a pending "Use in Console" handoff
        # (e.g. Library Search/RAG evidence) staged while setup is still
        # incomplete -- see `sync_card_state`'s `staged_evidence_notice`.
        self._staged_evidence_notice = ""
        #: FR-09 (TASK-2154.8): whether the typing-locked toast already fired
        #: for the current blocking episode; re-arms when the block lifts.
        self._typing_hint_shown = False
        #: TASK-2154.10 (AC-04): the app's `appearance.reduce_motion` setting,
        #: written by the ChatScreen on every guidance sync. Since TASK-23021
        #: the snow backdrop renders a still frame for everyone, so the flag
        #: no longer changes what this modal paints -- it is kept as the
        #: recorded preference (and the conduit, should animation ever
        #: return) rather than silently dropped from the screen's sync path.
        self._reduced_motion = False
        # Hidden until a card-mode state is synced in.
        self.display = False

    @property
    def detected_server_action(self) -> ConsoleDetectedServerAction | None:
        """Return the currently offered detected-local-server action."""
        return self._detected_action

    @property
    def reduced_motion(self) -> bool:
        """The recorded `appearance.reduce_motion` preference.

        The snow backdrop is a still frame for everyone since TASK-23021, so
        this no longer selects between an animated and a static presentation
        -- see the attribute comment in ``__init__``.
        """
        return self._reduced_motion

    @reduced_motion.setter
    def reduced_motion(self, value: bool) -> None:
        """Record the app's reduced-motion preference."""
        self._reduced_motion = bool(value)

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
            # FR-10 (TASK-2154.8): plain-language explainer so the steps below
            # ("Connect a provider…") land for a first-time user.
            subtitle = Static(
                CONSOLE_SETUP_CARD_SUBTITLE,
                id="console-setup-modal-subtitle",
                classes="console-setup-modal-subtitle",
                markup=False,
            )
            subtitle.display = blocking
            yield subtitle
            staged_notice = Static(
                self._staged_evidence_notice,
                id="console-setup-modal-staged-notice",
                classes="console-setup-modal-staged-notice",
                markup=False,
            )
            staged_notice.display = blocking and bool(self._staged_evidence_notice)
            yield staged_notice
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

    def sync_card_state(
        self,
        card_state: ConsoleSetupCardState,
        *,
        action_label: str = "",
        action_tooltip: str = "",
        staged_evidence_notice: str = "",
    ) -> None:
        """Refresh steps + primary action and toggle overlay visibility in place.

        Args:
            card_state: The current first-run setup card state.
            action_label: Primary action button label.
            action_tooltip: Primary action button tooltip.
            staged_evidence_notice: Task-2852 receipt line for a "Use in
                Console" handoff staged while setup is incomplete (e.g.
                ``"Library Search/RAG evidence staged — finish provider
                setup to use it."``), or ``""`` when nothing is staged.
                Only ever shown while the card is actually blocking.
        """
        self._card_state = _coerce_card_state(card_state)
        self._action_label = action_label.strip() or _DEFAULT_ACTION_LABEL
        self._action_tooltip = action_tooltip.strip() or _DEFAULT_ACTION_TOOLTIP
        self._staged_evidence_notice = staged_evidence_notice.strip()
        blocking = self.is_blocking
        if not blocking:
            # FR-09 (TASK-2154.8): block lifted -- re-arm the typing toast for
            # the next blocking episode.
            self._typing_hint_shown = False
        self.display = blocking
        if not self.is_mounted:
            return
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
        for selector in (
            "#console-setup-modal-title",
            "#console-setup-modal-subtitle",
            "#console-setup-modal-card",
        ):
            try:
                self.query_one(selector).display = blocking
            except Exception:
                continue
        try:
            staged_widget = self.query_one(
                "#console-setup-modal-staged-notice", Static
            )
        except Exception:
            pass
        else:
            staged_widget.update(self._staged_evidence_notice)
            staged_widget.display = blocking and bool(self._staged_evidence_notice)
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
        self._detected_action = (
            action if isinstance(action, ConsoleDetectedServerAction) else None
        )
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

    def on_key(self, event: Key) -> None:
        """Surface visible feedback when the user types while setup locks the composer.

        FR-09 (TASK-2154.8): with the workbench covered, printable keystrokes
        used to vanish silently (focus sits on the card's action button, which
        ignores them). While blocking, consume printable character keys -- this
        also keeps the screen's transcript j/k/c/e/r bindings inert under the
        overlay -- and raise one informational toast per blocking episode.
        Enter/Tab/Escape and other non-printables pass through untouched.
        """
        if not self.is_blocking:
            return
        character = event.character
        if not character or not character.isprintable():
            return
        event.stop()
        event.prevent_default()
        if self._typing_hint_shown:
            return
        self._typing_hint_shown = True
        try:
            self.app.notify(_TYPING_LOCKED_HINT, severity="information")
        except Exception:  # pragma: no cover - notify must never break key handling
            pass

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
