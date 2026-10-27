"""Persistent progress display and worker-to-UI message boundary."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, ProgressBar, Static

from tldw_chatbook.UI.Screens.model_browser_state import (
    TransferRateEstimator,
    transfer_summary,
    waiting_for_copy,
)
from tldw_chatbook.Widgets.pausable_progress import PausableProgressBar

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactRef,
        LocalGGUFImportProgress,
    )


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


class ModelInstallProgress(Widget):
    """Show the current phase and determinate transfer/hash progress.

    TASK-34100.14: a transfer also shows its speed and time left, estimated
    from successive progress events, so every host (Lab ▸ Models, Library,
    the first-run wizard) gets them. A host that can really stop the
    transfer (it passed a ``DownloadControl`` to ``provision()``) opts into
    a "Cancel download" button with ``cancellable=True`` and handles
    :class:`ModelInstallProgress.CancelRequested`; it is shown only while
    the download is waiting, transferring or being checked.
    """

    class CancelRequested(Message):
        """The user asked to cancel the download this widget shows."""

        def __init__(self, progress: ModelInstallProgress) -> None:
            """Create the request.

            Args:
                progress: The widget whose Cancel was pressed.
            """
            super().__init__()
            self.progress = progress

        @property
        def control(self) -> ModelInstallProgress:
            """The widget that posted the request (for ``@on`` selectors)."""
            return self.progress

    BUNDLED_CSS = """
    ModelInstallProgress {
        height: auto;
    }

    ModelInstallProgress ProgressBar {
        margin-top: 1;
    }

    ModelInstallProgress .model-install-cancel {
        margin-top: 1;
    }
    """

    _CANCELLABLE_PHASES = frozenset({"queued", "fetch", "pre-verify"})
    _RATE_PHASES = frozenset({"fetch", "copy"})

    _PHASE_LABELS = {
        "fetch": "Downloading model",
        "pre-verify": "Checking download",
        "verify-install": "Verifying and installing model",
        "activate": "Activating model",
        "copy": "Copying model into Chatbook",
        "inspect": "Checking GGUF structure",
        "verify": "Verifying managed copy",
        "finalize": "Finalizing managed model",
    }

    def __init__(
        self,
        initial: AcquisitionProgress | LocalGGUFImportProgress | None = None,
        *,
        id: str | None = None,
        cancellable: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a progress display, optionally restoring its latest event.

        Args:
            initial: Latest event retained by the host across recomposition.
            id: Optional Textual widget id.
            cancellable: Offer "Cancel download" (the host must handle
                ``CancelRequested`` by cancelling its ``DownloadControl``).
            clock: Monotonic seconds source for the speed estimate (tests).
        """
        self._initial = initial
        self._cancellable = cancellable
        self._cancel_pending = False
        self._estimator = TransferRateEstimator(clock)
        self._transfer_line = ""
        super().__init__(id=id)

    @property
    def transfer_line(self) -> str:
        """The latest ``"X of Y MiB · speed · time left"`` clause, or ``""``."""
        return self._transfer_line

    @property
    def cancel_pending(self) -> bool:
        """Whether the user has asked to cancel (the request was posted)."""
        return self._cancel_pending

    def compose(self) -> ComposeResult:
        """Compose the stable progress display."""
        yield Static("Waiting to install", id="model-install-progress-phase")
        yield Static("", id="model-install-progress-detail", markup=False)
        # TASK-23022: this bar is mounted hidden on every screen that embeds
        # ModelInstallProgress. A stock indeterminate ProgressBar arms a 15 Hz
        # refresh plus a 1 Hz ETA sampler that `display = False` does not
        # stop; PausableProgressBar keeps both clocks silent until shown.
        bar = PausableProgressBar(
            total=None,
            show_eta=False,
            id="model-install-progress-bar",
        )
        bar.display = False
        yield bar
        if self._cancellable:
            cancel = Button(
                "Cancel download",
                classes="model-install-cancel",
                variant="default",
            )
            cancel.display = False
            yield cancel

    def on_mount(self) -> None:
        """Restore retained progress or hide the idle determinate bar."""
        if self._initial is not None:
            self.update_progress(self._initial)

    def reset(self) -> None:
        """Return to the idle display before a host reuses this widget.

        A host that keeps one widget across downloads (the first-run Voice
        step composes its OmniVoice progress once) calls this when a new
        download starts: a cancel requested during the previous one would
        otherwise keep Cancel disabled and every phase reading "Cancelling
        download…", so the next download could not be stopped
        (TASK-34100.14 review F1).
        """
        self._cancel_pending = False
        self._estimator.reset()
        self._transfer_line = ""
        self.query_one("#model-install-progress-phase", Static).update(
            "Waiting to install"
        )
        self.query_one("#model-install-progress-detail", Static).update("")
        self.query_one("#model-install-progress-bar", ProgressBar).display = False
        if self._cancellable:
            cancel = self.query_one(".model-install-cancel", Button)
            self._hand_off_focus(cancel)
            cancel.display = False
            cancel.disabled = False

    def update_progress(
        self,
        event: AcquisitionProgress | LocalGGUFImportProgress,
    ) -> None:
        """Render one acquisition progress event on the event loop.

        Args:
            event: Event delivered through ``InstallProgressed``.
        """
        phase = self.query_one("#model-install-progress-phase", Static)
        if self._cancel_pending:
            phase.update("Cancelling download…")
        elif event.phase == "queued":
            phase.update(waiting_for_copy(getattr(event, "waiting_for", None)))
        else:
            phase.update(self._PHASE_LABELS[event.phase])
        self._sync_cancel(event.phase)
        detail = self.query_one("#model-install-progress-detail", Static)
        bar = self.query_one("#model-install-progress-bar", ProgressBar)
        byte_phase = event.phase in {"fetch", "pre-verify", "copy"}
        if byte_phase:
            filename = event.file or "Model files"
            estimator = None
            if event.phase in self._RATE_PHASES:
                self._estimator.observe(event.phase, event.bytes_done)
                estimator = self._estimator
            self._transfer_line = transfer_summary(
                event.bytes_done, event.bytes_total, estimator
            )
            detail.update(f"{filename} — {self._transfer_line}")
            bar.display = True
            bar.update(total=max(event.bytes_total, 1), progress=event.bytes_done)
            return

        self._estimator.reset()
        self._transfer_line = ""
        detail.update("")
        bar.display = False

    def _sync_cancel(self, phase: str) -> None:
        """Show Cancel only while there is still a transfer to stop."""
        if not self._cancellable:
            return
        cancel = self.query_one(".model-install-cancel", Button)
        show = phase in self._CANCELLABLE_PHASES
        if not show or self._cancel_pending:
            self._hand_off_focus(cancel)
        cancel.display = show
        cancel.disabled = self._cancel_pending

    def _hand_off_focus(self, cancel: Button) -> None:
        """Move focus off Cancel before it is disabled or hidden.

        Textual blurs a focused widget that becomes disabled or hidden, and
        nothing else in this widget can take focus, so it fell to None: the
        first-run wizard's Ctrl+N/Ctrl+B, bound on its container, then did
        nothing until Tab (TASK-34100.14 live recheck). The next control in
        the screen's focus chain takes it instead.
        """
        if cancel.has_focus:
            self.screen.focus_next()

    @on(Button.Pressed, ".model-install-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        """Ask the host to cancel, once, and say so until the host reacts."""
        event.stop()
        if self._cancel_pending:
            return
        self._cancel_pending = True
        self._hand_off_focus(event.button)
        event.button.disabled = True
        self.query_one("#model-install-progress-phase", Static).update(
            "Cancelling download…"
        )
        self.post_message(self.CancelRequested(self))
