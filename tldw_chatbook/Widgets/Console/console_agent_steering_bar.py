"""The Agent rail drill-in's steering input + honest queued-count line.

Supervisor fleet PR 3b, Task 3 (spec `Docs/superpowers/specs/2026-08-08-
supervisor-agent-fleet-design.md` §7: "phase 3 adds the steering input +
mailbox 'queued' state"). A compact ``Input`` over two one-line Statics:

* the **queued line** — the mailbox's honest latency state ("steering
  queued (N)", spec §6): a posted entry is *queued*, delivered only before
  the child's next model turn, and this line says so until the child's
  drain consumes it;
* the **note line** — this producer's own refusal copy (today: oversize),
  because the plan places text validation at each producer's boundary with
  its own user-facing copy, which the bridge's silent bool cannot carry.

Visibility is NOT this widget's decision: `ConsoleAgentController.
_console_agent_steering_state` derives it (drilled into a LIVE child only
— spec §1's owner pin: the panel watches/steers, never launches), and the
widget just applies whatever ``ConsoleAgentSteeringState`` it is handed —
at construction (so a rail recompose mid-drill-in paints correctly without
waiting for the next sync, the same reason `ConsoleLeftRail` takes
``agent_drilldown_active`` at construction) and via ``sync_state`` on the
screen's equality-guarded Agent-section sync.

Submit discipline (all three guards BEFORE the message posts, so the
screen handler and controller never see junk):

* empty-after-strip → inert (no message, no refusal copy — an empty send
  is a non-action, not an error);
* empty ``target_id`` → inert (Task 2's report bound this task: an empty
  id must never draw an unknown-id refusal naming ``''``);
* over ``MAX_STEERING_CHARS`` → refused here with this bar's own copy,
  draft kept in the input for shortening.

Sizing is explicit throughout (`width: 100%`, `height: auto`/3): a bare
``Static``/``Input`` defaulting to ``1fr`` in a laid-out container has
pushed siblings off-screen in this rail before while display-only tests
stayed green — every child pins both dimensions.
"""

from __future__ import annotations

import dataclasses

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, Static

from tldw_chatbook.Agents.agent_models import MAX_STEERING_CHARS

#: DOM ids -- module constants so the screen sync, the rail compose and the
#: tests all speak one vocabulary.
STEERING_BAR_ID = "console-agent-steering-bar"
STEERING_INPUT_ID = "console-agent-steering-input"
STEERING_QUEUED_ID = "console-agent-steering-queued"
STEERING_NOTE_ID = "console-agent-steering-note"

__all__ = [
    "ConsoleAgentSteeringBar",
    "ConsoleAgentSteeringState",
    "STEERING_BAR_ID",
    "STEERING_INPUT_ID",
    "STEERING_NOTE_ID",
    "STEERING_QUEUED_ID",
]


@dataclasses.dataclass(frozen=True)
class ConsoleAgentSteeringState:
    """Atomic snapshot of everything the steering bar renders from.

    One value object, not independent kwargs — the same anti-drift
    discipline ``ConsoleInspectorSectionState`` records (its task-3 review
    round 1/3 findings): all three fields are REQUIRED, so "I forgot a
    dimension" is a ``TypeError``, never silent data loss. A frozen
    dataclass compares by value, which is what lets it ride the Agent
    section payload's existing ``==`` equality guard unchanged.

    Attributes:
        visible: Whether the bar is on the live surface at all — True only
            while drilled into a LIVE child (never finished/historical,
            never the overview).
        target_id: The steering target the submit message will carry —
            the drilled-in child's identity in whichever vocabulary the
            controller holds (``ConsoleAgentBridge.steer_subagent``
            resolves both). ``""`` disarms submit entirely.
        queued: The child's honest mailbox depth (``FleetHandle.
            queued_steering``, computed onto coordinator copies) — 0 hides
            the queued line.
    """

    visible: bool
    target_id: str
    queued: int


#: The hidden default: what a bar shows when nothing is drilled into.
STEERING_STATE_HIDDEN = ConsoleAgentSteeringState(
    visible=False, target_id="", queued=0
)


def _queued_text(queued: int) -> str:
    """The queued line's exact grammar; ``""`` when nothing waits."""
    return f"steering queued ({queued})" if queued > 0 else ""


class ConsoleAgentSteeringBar(Vertical):
    """Compact steering input + queued-count line for one LIVE child."""

    class SteeringSubmitted(Message):
        """A validated steering submit: non-empty text, non-empty target,
        within the cap — the three guards ran before this posted.

        Attributes:
            target_id: The drilled-in child's id (handle or run
                vocabulary — the bridge resolves both).
            text: The stripped steering text.
        """

        def __init__(self, target_id: str, text: str) -> None:
            super().__init__()
            self.target_id = target_id
            self.text = text

    def __init__(
        self,
        state: ConsoleAgentSteeringState | None = None,
        **kwargs,
    ) -> None:
        """Create the bar from an already-derived steering state.

        Args:
            state: The initial ``ConsoleAgentSteeringState``; ``None``
                (bare test constructions) means hidden. The production
                construction site (``ConsoleLeftRail.compose``) always
                passes the controller's current derivation, so a rail
                recompose while drilled into a live child paints the bar
                correctly without waiting for the next sync tick.
            kwargs: Forwarded to ``Vertical`` (id, classes, ...).
        """
        super().__init__(**kwargs)
        self._state = state if state is not None else STEERING_STATE_HIDDEN
        # Explicit sizing (module docstring): never inherit a 1fr default.
        self.styles.width = "100%"
        self.styles.height = "auto"
        self.styles.display = "block" if self._state.visible else "none"

    def compose(self) -> ComposeResult:
        steer_input = Input(
            placeholder="Steer this sub-agent…",
            id=STEERING_INPUT_ID,
        )
        steer_input.styles.width = "100%"
        steer_input.styles.height = 3
        yield steer_input
        queued = Static(
            _queued_text(self._state.queued),
            id=STEERING_QUEUED_ID,
            markup=False,
        )
        queued.styles.width = "100%"
        queued.styles.height = "auto"
        queued.styles.display = "block" if self._state.queued > 0 else "none"
        yield queued
        note = Static("", id=STEERING_NOTE_ID, markup=False)
        note.styles.width = "100%"
        note.styles.height = "auto"
        note.styles.display = "none"
        yield note

    def sync_state(self, state: ConsoleAgentSteeringState) -> None:
        """Apply a freshly-derived state to the mounted widgets.

        Called by the screen's equality-guarded Agent-section sync — by
        the time this runs the payload HAS changed, so the writes below
        are always warranted. A target change also clears any standing
        refusal note (it referred to a draft aimed at the previous
        child) AND the input's own draft (Qodo audit S3, PR 1793):
        visibility toggles via ``styles.display``, so without the clear a
        draft typed for child A survived drill-out/drill-in and submit
        paired it with the LATEST ``target_id`` — steering child B with
        A's text. A sync for the SAME target (queued-count change,
        elapsed repaint) keeps the draft: a routine tick must never eat
        text the user is mid-typing.
        """
        previous = self._state
        self._state = state
        self.styles.display = "block" if state.visible else "none"
        try:
            queued = self.query_one(f"#{STEERING_QUEUED_ID}", Static)
        except Exception:
            # Not composed yet -- compose() renders from self._state.
            return
        queued.update(_queued_text(state.queued))
        queued.styles.display = "block" if state.queued > 0 else "none"
        if state.target_id != previous.target_id:
            self._set_note("")
            try:
                steer_input = self.query_one(f"#{STEERING_INPUT_ID}", Input)
            except Exception:
                pass  # not composed: compose() starts the input empty
            else:
                steer_input.value = ""

    def _set_note(self, text: str) -> None:
        try:
            note = self.query_one(f"#{STEERING_NOTE_ID}", Static)
        except Exception:
            return
        note.update(text)
        note.styles.display = "block" if text else "none"

    def clear_draft(self) -> None:
        """Clear the input after a submit the bridge actually QUEUED.

        Qodo audit minor batch: the input used to clear at post time, so
        a submit the bridge then refused (unknown/terminal target, dead
        coordinator) destroyed the user's text with nothing delivered and
        nothing shown. The screen's ``SteeringSubmitted`` handler calls
        this only on a ``True`` return from the steering route; a refusal
        keeps the draft in place for retry.
        """
        try:
            steer_input = self.query_one(f"#{STEERING_INPUT_ID}", Input)
        except Exception:
            return
        steer_input.value = ""

    @on(Input.Submitted, f"#{STEERING_INPUT_ID}")
    def _on_steering_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        text = (event.value or "").strip()
        if not text:
            # Inert, deliberately: an empty send is a non-action. No
            # refusal copy -- and never one naming ''.
            return
        if not self._state.target_id:
            # The empty-target guard (Task 2's report, concern (d)): a
            # submit with no target must never reach the bridge and draw
            # an unknown-id refusal naming ''. Pinned at THIS layer by
            # `test_submit_with_an_empty_target_never_posts_a_message_at_
            # all`'s post_message spy -- the controller and bridge also
            # refuse an empty target, so an outcome-only assertion cannot
            # see this guard (first-round mutation finding).
            return
        if len(text) > MAX_STEERING_CHARS:
            self._set_note(
                f"Steering is too long ({len(text)} chars; the cap is "
                f"{MAX_STEERING_CHARS}). Shorten it and press Enter again."
            )
            return
        self._set_note("")
        # The draft is NOT cleared here: the screen handler clears it via
        # `clear_draft()` only once the bridge reports the entry queued
        # (see that method's docstring for the refusal case).
        self.post_message(self.SteeringSubmitted(self._state.target_id, text))
