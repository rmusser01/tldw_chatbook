"""Exit setup when its progress cannot be saved (TASK-34100.4, protect-summary-04).

"Exit setup" saves a resume checkpoint, then closes the wizard. When that save
kept failing -- the 2026-10-02 review hit it after a second password stranded
the keys, so every config write refused -- the wizard only toasted "Setup
progress could not be saved. Retry Exit setup." and stayed open. Next, Voice
and Skip setup failed the same way, so the only way out was Ctrl+Q.

Now a failed save asks once more: "Exit without saving progress" closes setup
without the checkpoint (steps already saved stay saved; the next launch simply
has no resume point), "Keep going" returns to the wizard.

Review round 1 (R1-F6): "Skip setup" (Esc or the cancel button on Welcome)
saves completion instead of a checkpoint, and when that save failed it only
said "Retry before closing", so a re-run on an unwritable config could still
trap the user on Welcome. It now gets the same offer; exiting there leaves
setup unfinished, so it opens again at the next launch.
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
SKIP_UNSAVED_MESSAGE = (
    "chatbook couldn't save that setup is finished, so it will open again the "
    "next time chatbook starts. You can exit for now anyway: settings you "
    "already saved stay saved."
)


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
    offer_exit_without_saving(container, EXIT_UNSAVED_MESSAGE)


async def skip_setup(container: Any) -> None:
    """Mark setup finished and close it, or offer to exit without saving.

    Args:
        container: The wizard's ``SetupWizardContainer``.
    """
    async with container._draft_mutation_lock:
        saved = await container._complete_setup_locked()
    if saved:
        container._dismiss_screen({"completed": True, "exit_route": None})
        return
    container._show_completion_save_error()
    if container.is_attached:
        offer_exit_without_saving(container, SKIP_UNSAVED_MESSAGE)


def offer_exit_without_saving(container: Any, message: str) -> None:
    """Ask whether to close setup although its save failed.

    "Exit without saving progress" closes the wizard with no result (the
    finish-later outcome); "Keep going" leaves it open on the same step.

    Args:
        container: The wizard's ``SetupWizardContainer``.
        message: Why nothing was saved and what exiting means.
    """

    def _exit_anyway(confirmed: bool | None) -> None:
        if confirmed:
            container._dismiss_screen(None)

    container.app.push_screen(
        ConfirmationDialog(
            title=EXIT_UNSAVED_TITLE,
            message=message,
            confirm_label=EXIT_UNSAVED_LABEL,
            cancel_label="Keep going",
        ),
        _exit_anyway,
    )
