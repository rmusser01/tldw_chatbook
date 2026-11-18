"""Personas notice for a stuck persona Actor Pack import (TASK-41114).

Owner decision 2026-11-16: "Clear the interrupted import (Recommended)". The
notice's one action deletes only the quarantined intent rows; persona records
stay exactly as they are, and new persona Actor Pack imports work again.
"""

from __future__ import annotations

import asyncio

from textual import on
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, Static

NOTICE_ID = "personas-actor-pack-recovery-notice"
CLEAR_BUTTON_ID = "personas-actor-pack-recovery-clear"
NOTICE_TEXT = (
    "An earlier persona Actor Pack import was interrupted and is still "
    "recorded, so persona Actor Packs can't be imported or created. Your "
    "personas are not affected."
)
CLEAR_LABEL = "Keep my current personas and clear the interrupted import"
_CLEARED = (
    "The interrupted import was cleared. You can import persona Actor Packs again."
)
_CLEAR_ERRORS = {
    "actor_pack_storage_unavailable": (
        "Backup or restore is using this profile. Clear the interrupted import "
        "when it finishes."
    ),
}
_CLEAR_FAILED = "Couldn't clear the interrupted import. Restart the app and try again."


class ActorPackRecoveryNotice(Horizontal):
    """One keyboard-reachable notice and its clear action."""

    BUNDLED_CSS = """
    ActorPackRecoveryNotice {
        height: auto;
        padding: 0 1;
        border: round $warning;
    }
    ActorPackRecoveryNotice > Static {
        width: 1fr;
    }
    ActorPackRecoveryNotice > Button {
        width: auto;
    }
    """

    def __init__(self, coordinator: object, **kwargs) -> None:
        super().__init__(**kwargs)
        self.coordinator = coordinator

    def compose(self):
        yield Static(NOTICE_TEXT, markup=False)
        yield Button(CLEAR_LABEL, id=CLEAR_BUTTON_ID)

    @on(Button.Pressed, f"#{CLEAR_BUTTON_ID}")
    def _clear_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        event.button.disabled = True
        self.run_worker(self._clear(event.button), exclusive=True, group=NOTICE_ID)

    async def _clear(self, button: Button) -> None:
        from tldw_chatbook.Actor_Packs.persona_coordinator import (
            PersonaActorPackCoordinatorError,
        )
        from tldw_chatbook.Backup_Recovery.participants import (
            run_finite_local_worker,
        )

        try:
            await asyncio.to_thread(
                run_finite_local_worker, self.coordinator.clear_blocked_intents
            )
        except PersonaActorPackCoordinatorError as error:
            button.disabled = False
            self.app.notify(
                _CLEAR_ERRORS.get(error.category, _CLEAR_FAILED),
                severity="error",
                markup=False,
            )
            return
        try:
            recovery_error = self.app.actor_pack_recovery_error
        except AttributeError:
            recovery_error = None
        if recovery_error == "actor_pack_recovery_blocked":
            self.app.actor_pack_recovery_error = None
        self.app.notify(_CLEARED, markup=False)
        self.remove()  # Not awaited: this worker belongs to the notice.


async def show_actor_pack_recovery_notice(screen: Widget) -> None:
    """Mount the notice under the Personas purpose line, once, while the
    app's persona Actor Pack coordinator has a blocked intent."""

    app = getattr(screen, "app_instance", None) or screen.app
    try:
        coordinator = app.persona_actor_pack_coordinator
    except AttributeError:
        return
    if (
        coordinator is None
        or not getattr(coordinator, "blocked_intent_ids", ())
        or screen.query(f"#{NOTICE_ID}")
    ):
        return
    await screen.query_one("#personas-shell").mount(
        ActorPackRecoveryNotice(coordinator, id=NOTICE_ID), after="#personas-purpose"
    )
