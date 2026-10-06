"""Exit setup when its progress cannot be saved (TASK-34100.4, protect-summary-04).

"Exit setup" saves a resume checkpoint, then closes the wizard. When that save
kept failing -- the 2026-10-02 review hit it after a second password stranded
the keys, so every config write refused -- the wizard only toasted "Setup
progress could not be saved. Retry Exit setup." and stayed open. Next, Voice
and Skip setup failed the same way, so the only way out was Ctrl+Q.

Now a failed save asks once more: "Exit without saving progress" closes setup
without the checkpoint (steps already saved stay saved; the next launch simply
has no resume point), "Keep going" returns to the wizard.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

EXIT_UNSAVED_TITLE = "Setup progress wasn't saved"
EXIT_UNSAVED_MESSAGE = (
    "chatbook couldn't save where you are in setup. You can exit anyway: "
    "settings you already saved stay saved, but setup won't resume from "
    "this step. Rerun it any time from Settings ▸ Diagnostics."
)
EXIT_UNSAVED_LABEL = "Exit without saving progress"


async def finish_later(container: Any) -> None:
    """Save the resume checkpoint and close setup, or offer to exit anyway.

    Args:
        container: The wizard's ``SetupWizardContainer``.
    """
    try:
        if container.hold_provider_save_settlement():
            return
        saved = await container.persist_current_checkpoint()
    except Exception:
        logger.warning("Setup finish-later checkpoint failed (category=runtime)")
        saved = False
    if saved:
        container._dismiss_screen(None)
        return

    def _exit_anyway(confirmed: bool | None) -> None:
        if confirmed:
            container._dismiss_screen(None)

    container.app.push_screen(
        ConfirmationDialog(
            title=EXIT_UNSAVED_TITLE,
            message=EXIT_UNSAVED_MESSAGE,
            confirm_label=EXIT_UNSAVED_LABEL,
            cancel_label="Keep going",
        ),
        _exit_anyway,
    )
