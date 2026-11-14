"""The generated video's Save-to-disk screens, guarded against Ctrl+Q.

TASK-33622.15. A generated video that missed managed storage waits in
``ConsoleVideoCapacityModal``, whose ``confirm_quit`` asks "Discard generated
video and quit?". Choosing **Save to disk** closes that modal and opens a file
picker, then -- for an existing or vanished destination -- a Replace or
Destination-changed confirmation. The video is still only in memory while
those are open, but the quit walk found no hook on them, so Ctrl+Q discarded
it without a word.

These subclasses change nothing but that: each answers ``confirm_quit`` with
the capacity modal's own prompt. ``UI/Console_Modules/video.py`` pushes them
in place of the plain picker and dialog, and imports this module only when
the user chooses Save to disk, so it stays off the boot path.
"""

from __future__ import annotations

from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console.console_video_capacity_modal import (
    confirm_quit_discarding_generated_video,
)
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave


class GeneratedVideoFileSave(EnhancedFileSave):
    """The Save-to-disk picker for a generated video awaiting a choice."""

    async def confirm_quit(self) -> bool:
        """Ask before Ctrl+Q discards the video this picker would save.

        Returns:
            True to let the quit proceed; False to stay in the picker.
        """
        return await confirm_quit_discarding_generated_video(self)


class GeneratedVideoConfirmation(ConfirmationDialog):
    """A Replace / Destination-changed question about a generated video's save."""

    async def confirm_quit(self) -> bool:
        """Ask before Ctrl+Q discards the video this question is about.

        Returns:
            True to let the quit proceed; False to stay on the question.
        """
        return await confirm_quit_discarding_generated_video(self)
