"""Blocking first-run setup modal for the native Console workbench.

While provider setup is incomplete (``ConsoleSetupCardState.mode == "card"``),
this widget overlays the Console workbench (rail + transcript + composer) with a
dim backdrop and a centered "Get started" card. The card shows the three live
setup steps and up to three actions: provider recovery, a note in Library, and
the detected local server. It is Console-scoped: it lives inside the ChatScreen
(never an app-level modal), so the top navigation tab bar stays reachable. The
modal dismisses automatically the moment readiness + model are satisfied (the
guidance sync drives it).

The dim is the modal's own flat fill (``$background 60%``) and nothing is
drawn on it but the card (TASK-33008.4 retired the snow field that used to
sit there). Textual's alpha-background compositing only blends a widget's
background with its *ancestor* style chain (see ``DOMNode.background_colors``
/ ``Widget.opacity``); it does not re-composite the rendered pixels of
sibling widgets on a lower layer. So the fill hides the workbench beneath
rather than tinting it, and it needs a token distinctly darker than the
Console shell's ``$ds-surface-panel``: a same-color alpha fill over an
identically-colored ancestor is a no-op.

Tab and Shift+Tab cycle the actions the card shows, and wrap (TASK-32572).
"""

from __future__ import annotations

from typing import Any, ClassVar

from tldw_chatbook.Utils.input_validation import escape_markup
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Key
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_CARD_SUBTITLE,
    CONSOLE_SETUP_CARD_TITLE,
    CONSOLE_SETUP_NOTES_ACTION_LABEL,
    CONSOLE_SETUP_NOTES_ACTION_TOOLTIP,
    CONSOLE_SETUP_STEP_DETAIL_INDENT,
    ConsoleDetectedServerAction,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested


CONSOLE_SETUP_MODAL_STEP_COUNT = 3
CONSOLE_SETUP_MODAL_ACTION_ID = "console-setup-modal-action"
CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID = "console-setup-modal-detected-action"
CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION = "use-detected-local-server"
# TASK-34100.6 (review entry-exit-handoff-21): look for a local server again
# on request -- a quiet card never polls.
CONSOLE_SETUP_MODAL_LOOK_AGAIN_ID = "console-setup-modal-look-again"
CONSOLE_SETUP_MODAL_LOOK_AGAIN_WORKBENCH_ACTION = "look-again-local-server"
# task-32140: a needs-no-provider secondary action alongside the detected-
# server one -- same "extra affordance while blocking" shape.
CONSOLE_SETUP_MODAL_NOTES_ACTION_ID = "console-setup-modal-notes-action"
CONSOLE_SETUP_MODAL_NOTES_WORKBENCH_ACTION = "write-note-library"
_DEFAULT_ACTION_LABEL = "Choose model"
_DEFAULT_ACTION_TOOLTIP = "Choose the provider and model for this Console session."

#: TASK-2154.8 (FR-09): one informational toast per blocking episode when the
#: user types while the composer is locked. Not per keystroke -- a toast on
#: every letter would be spam; one per episode makes the lock perceivable.
_TYPING_LOCKED_HINT = (
    "Typing is locked until setup finishes — press Enter to use the highlighted action."
)


def _coerce_card_state(value: object) -> ConsoleSetupCardState:
    """Guard against a transiently non-``ConsoleSetupCardState`` value."""
    if isinstance(value, ConsoleSetupCardState):
        return value
    return ConsoleSetupCardState(mode="quiet")


class ConsoleSetupModal(Vertical):
    """Console-scoped blocking overlay carrying the first-run setup card."""

    # TASK-32572: the modal is the focused action's ancestor, so these run
    # before ChatScreen's own Tab, which traps focus on the card but cannot
    # move it between the card's actions.
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("tab", "cycle_action(1)", show=False),
        Binding("shift+tab", "cycle_action(-1)", show=False),
    ]

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "console-setup-modal")
        classes = kwargs.pop("classes", "")
        kwargs["classes"] = f"console-setup-modal-backdrop {classes}".strip()
        super().__init__(**kwargs)
        self._card_state = ConsoleSetupCardState(mode="quiet")
        self._action_label = _DEFAULT_ACTION_LABEL
        self._action_tooltip = _DEFAULT_ACTION_TOOLTIP
        self._detected_action: ConsoleDetectedServerAction | None = None
        #: TASK-33008.4: a newly offered detected action takes focus once.
        self._detected_focus_pending = False
        # Task-2852: a receipt line for a pending "Use in Console" handoff
        # (e.g. Library Search/RAG evidence) staged while setup is still
        # incomplete -- see `sync_card_state`'s `staged_evidence_notice`.
        self._staged_evidence_notice = ""
        #: FR-09 (TASK-2154.8): whether the typing-locked toast already fired
        #: for the current blocking episode; re-arms when the block lifts.
        self._typing_hint_shown = False
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
        """Build the setup card, including its action buttons.

        Returns:
            The setup card with its title, steps, and action buttons
            (provider recovery, notes, and any detected-server action).
        """
        # Children mirror the container's blocking state so hidden-modal copy
        # never leaks into visible-text scrapes before the first guidance sync.
        blocking = self.is_blocking
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
            # task-32555 AC#1: real button chrome. ``compact=True`` used to
            # render these as bare text lines (Textual's compact class forces
            # ``border: none !important``, so no sheet could give them an
            # edge); the border and the focus outline live in the Console
            # sheet's ``.console-setup-modal-action`` rules.
            action = Button(
                self._action_label,
                id=CONSOLE_SETUP_MODAL_ACTION_ID,
                classes="console-setup-modal-action",
            )
            action.tooltip = self._action_tooltip
            action.display = blocking
            yield action
            # task-32140: needs no provider -- shown whenever the card is
            # blocking, unlike the detected-server action which depends on
            # discovery finding something.
            notes_action = Button(
                CONSOLE_SETUP_NOTES_ACTION_LABEL,
                id=CONSOLE_SETUP_MODAL_NOTES_ACTION_ID,
                classes="console-setup-modal-action",
            )
            notes_action.tooltip = CONSOLE_SETUP_NOTES_ACTION_TOOLTIP
            notes_action.display = blocking
            yield notes_action
            detected = Button(
                self._detected_action_label(),
                id=CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID,
                classes="console-setup-modal-action console-setup-modal-detected-action",
            )
            detected.tooltip = self._detected_action_tooltip()
            detected.display = blocking and self._detected_action is not None
            yield detected
            look_again = Button(
                "Look again for a local server",
                id=CONSOLE_SETUP_MODAL_LOOK_AGAIN_ID,
                classes="console-setup-modal-action",
            )
            look_again.tooltip = "Checks this computer for a running model server."
            look_again.display = blocking and self._detected_action is None
            yield look_again

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
            # TASK-33620.15.7: this sync runs on every 0.2 s run tick; an
            # unchanged step (the hidden card, most of all) must not request
            # a whole-screen reflow.
            text = self._step_text(index, step)
            if widget.content != text:
                widget.update(text)
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
            staged_widget = self.query_one("#console-setup-modal-staged-notice", Static)
        except Exception:
            pass
        else:
            if staged_widget.content != self._staged_evidence_notice:
                staged_widget.update(self._staged_evidence_notice)
            staged_widget.display = blocking and bool(self._staged_evidence_notice)
        self._sync_detected_action_button()
        try:
            notes_action = self.query_one(
                f"#{CONSOLE_SETUP_MODAL_NOTES_ACTION_ID}", Button
            )
        except Exception:
            pass
        else:
            notes_action.display = blocking
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
        action = action if isinstance(action, ConsoleDetectedServerAction) else None
        if action != self._detected_action:
            self._detected_focus_pending = action is not None
        self._detected_action = action
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
        try:
            look_again = self.query_one(f"#{CONSOLE_SETUP_MODAL_LOOK_AGAIN_ID}", Button)
        except Exception:
            return
        look_again.display = self.is_blocking and self._detected_action is None

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
        """Focus the card's first action while blocking.

        That is "Use detected <server>" while one is offered (TASK-33008.3:
        Enter, Enter is the local first run), and the primary action
        otherwise. The button order on the card does not change.

        Every guidance sync calls this, so focus already on a shown action
        stays where Tab put it (TASK-33008.4). The one exception: a newly
        offered detected action takes focus once. The card's own screen is
        read, not the app's: under a dialog ``app.focused`` is the dialog's,
        and the sync moved the Console's remembered Tab stop.
        """
        actions = self._displayed_actions()
        if not actions:
            return
        if self.screen.focused in actions and not self._detected_focus_pending:
            return
        self._detected_focus_pending = False
        action_id = (
            CONSOLE_SETUP_MODAL_ACTION_ID
            if self._detected_action is None
            else CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID
        )
        try:
            self.query_one(f"#{action_id}", Button).focus()
        except Exception:
            return

    def _displayed_actions(self) -> list[Button]:
        """The card's shown action buttons, in card order; none unless blocking."""
        if not (self.is_mounted and self.is_blocking):
            return []
        return [button for button in self.query(Button) if button.display]

    def action_cycle_action(self, step: int) -> None:
        """Move focus ``step`` actions along the card's shown actions, wrapping.

        Args:
            step: 1 for Tab, -1 for Shift+Tab.

        Raises:
            SkipAction: The card shows no action, so the screen's Tab runs.
        """
        actions = self._displayed_actions()
        if not actions:
            raise SkipAction()
        focused = self.app.focused
        index = (
            actions.index(focused) if focused in actions else (-1 if step > 0 else 0)
        )
        actions[(index + step) % len(actions)].focus()

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
        """Route card actions through the owning Workbench screen.

        Args:
            event: The button-press event; ``event.button.id`` selects
                which ``WorkbenchActionRequested`` action id gets posted
                (detected-server, notes, or provider recovery).
        """
        if event.button.id == CONSOLE_SETUP_MODAL_DETECTED_ACTION_ID:
            event.stop()
            self.post_message(
                WorkbenchActionRequested(CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION)
            )
            return
        if event.button.id == CONSOLE_SETUP_MODAL_LOOK_AGAIN_ID:
            event.stop()
            self.post_message(
                WorkbenchActionRequested(CONSOLE_SETUP_MODAL_LOOK_AGAIN_WORKBENCH_ACTION)
            )
            return
        if event.button.id == CONSOLE_SETUP_MODAL_NOTES_ACTION_ID:
            event.stop()
            self.post_message(
                WorkbenchActionRequested(CONSOLE_SETUP_MODAL_NOTES_WORKBENCH_ACTION)
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
            # TASK-33005.3: the active step's detail is the readiness word; it
            # gets its own line under the label so a wrap never splits it.
            joiner = (
                f"\n{CONSOLE_SETUP_STEP_DETAIL_INDENT}"
                if step.state == "active"
                else "  "
            )
            text = f"{text}{joiner}{step.detail}"
        return text

    @staticmethod
    def _step_classes(step: ConsoleSetupStep) -> str:
        return f"console-setup-step console-setup-step-{step.state}"
