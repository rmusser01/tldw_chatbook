"""One managed download's user-facing identity and cancel switch (TASK-34100.14).

Every managed download in the app runs as ``asyncio.run(provision(...))``
inside a Textual thread worker (the first-run wizard's Speech and Voice
steps, Lab ▸ Models, Library). ``Worker.cancel()`` cannot reach that inner
event loop, so a host that wants a real "Cancel download" hands
``provision()`` one of these instead: the UI thread calls :meth:`cancel`,
and the acquisition layer polls :attr:`cancel_requested` from the worker
thread's own loop and stops the transfer cooperatively (see
``ArtifactAcquisitionService.provision``).

The same object names the download for OTHER installs: while it holds the
process-wide acquisition session, a second install that has to wait (or
fails fast as busy) is told which model it is waiting for, by ``label``.

This module deliberately imports nothing network-capable, so UI modules can
build a control without loading ``acquisition``/``httpx`` at import time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(eq=False)
class DownloadControl:
    """A thread-safe handle a host keeps for one managed download.

    Attributes:
        label: User-visible model name, e.g. ``"Parakeet v2 (English,
            INT8)"``. Other installs that have to wait for this one name it.
        wait_if_busy: When another install already holds the acquisition
            session, wait for it (reporting a ``queued`` progress event)
            instead of failing fast with ``AcquisitionBusyError``.
    """

    label: str
    wait_if_busy: bool = False
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> None:
        """Ask the running (or waiting) download to stop; safe from any thread."""
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        """Whether :meth:`cancel` has been called."""
        return self._cancel.is_set()
