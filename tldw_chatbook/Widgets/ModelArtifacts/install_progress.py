"""Persistent progress display and worker-to-UI message boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import ProgressBar, Static

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef


class InstallProgressed(Message):
    """Carry one acquisition progress event onto the Textual event loop."""

    def __init__(self, progress: AcquisitionProgress) -> None:
        """Create a progress message.

        Args:
            progress: Immutable event emitted by the acquisition worker.
        """
        super().__init__()
        self.progress = progress


class InstallStatusChanged(Message):
    """Report the start or completion of one managed-model install."""

    def __init__(
        self,
        reference: ArtifactRef,
        *,
        active: bool,
        succeeded: bool | None = None,
    ) -> None:
        """Create a cross-view install-state message.

        Args:
            reference: Root model being installed.
            active: Whether provisioning is still running.
            succeeded: Completion result, or ``None`` while active.
        """
        super().__init__()
        self.reference = reference
        self.active = active
        self.succeeded = succeeded


def make_progress_callback(
    post_message: Callable[[Message], object],
) -> Callable[[AcquisitionProgress], None]:
    """Build a worker-safe callback that posts immutable progress messages.

    Args:
        post_message: Host screen's message-posting method.

    Returns:
        A callback suitable for ``ArtifactAcquisitionService.provision``.
    """

    def callback(progress: AcquisitionProgress) -> None:
        post_message(InstallProgressed(progress))

    return callback


def _bytes(size: int) -> str:
    """Format a nonnegative byte count compactly."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


class ModelInstallProgress(Widget):
    """Show the current phase and determinate transfer/hash progress."""

    DEFAULT_CSS = """
    ModelInstallProgress {
        height: auto;
    }

    ModelInstallProgress ProgressBar {
        margin-top: 1;
    }
    """

    _PHASE_LABELS = {
        "fetch": "Downloading model",
        "pre-verify": "Checking download",
        "verify-install": "Verifying and installing model",
        "activate": "Activating model",
    }

    def __init__(
        self,
        initial: AcquisitionProgress | None = None,
        *,
        id: str | None = None,
    ) -> None:
        """Create a progress display, optionally restoring its latest event.

        Args:
            initial: Latest event retained by the host across recomposition.
            id: Optional Textual widget id.
        """
        self._initial = initial
        super().__init__(id=id)

    def compose(self) -> ComposeResult:
        """Compose the stable progress display."""
        yield Static("Waiting to install", id="model-install-progress-phase")
        yield Static("", id="model-install-progress-detail", markup=False)
        yield ProgressBar(
            total=None,
            show_eta=False,
            id="model-install-progress-bar",
        )

    def on_mount(self) -> None:
        """Restore retained progress or hide the idle determinate bar."""
        if self._initial is not None:
            self.update_progress(self._initial)
            return
        self.query_one("#model-install-progress-bar", ProgressBar).display = False

    def update_progress(self, event: AcquisitionProgress) -> None:
        """Render one acquisition progress event on the event loop.

        Args:
            event: Event delivered through ``InstallProgressed``.
        """
        self.query_one("#model-install-progress-phase", Static).update(
            self._PHASE_LABELS[event.phase]
        )
        detail = self.query_one("#model-install-progress-detail", Static)
        bar = self.query_one("#model-install-progress-bar", ProgressBar)
        byte_phase = event.phase in {"fetch", "pre-verify"}
        if byte_phase:
            filename = event.file or "Model files"
            detail.update(
                f"{filename} — {_bytes(event.bytes_done)} / {_bytes(event.bytes_total)}"
            )
            bar.display = True
            bar.update(total=max(event.bytes_total, 1), progress=event.bytes_done)
            return

        detail.update("")
        bar.display = False
