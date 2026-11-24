"""Replay the Library's selected report while its row and profile remain current."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...Library.library_artifacts_state import ArtifactKey
from ...Subscriptions.briefing_audio import briefing_audio_dir
from ...Utils.path_validation import validate_path

if TYPE_CHECKING:
    from ..file_output_attempt import FileOutputAttempt
    from .library_artifacts_controller import LibraryArtifactsController


async def play_report_audio(
    controller: LibraryArtifactsController,
    key: ArtifactKey,
    revision: str,
    profile: tuple[int, ...],
    presentation: int,
    attempt: FileOutputAttempt,
) -> None:
    """Resolve the selected report's file and submit its original play intent."""
    screen = controller.screen
    db = controller.app_instance.subscriptions_db

    def current() -> bool:
        return (
            controller.active()
            and not controller.suspended
            and controller.presentation == presentation
            and controller.profile() == profile
            and controller.selected == key
            and controller.detail is not None
            and controller.detail.revision == revision
            and screen.is_mounted
            and screen.is_attached
            and (screen.app.screen is screen or controller._file_output.confirming())
        )

    raw = await screen._run_library_service_call(db.get_artifact_audio_path, key)
    if not current() or not attempt.ticket.is_current():
        return
    if raw:
        path = validate_path(raw, briefing_audio_dir(), redact_paths=True)
        if path.is_file():
            await controller._file_output.play(attempt, path, "Library report", current)
            return
    controller.notify("This report's audio is no longer available.", "warning")
