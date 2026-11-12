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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _started: bool = field(default=False, repr=False)

    def cancel(self) -> None:
        """Ask the running (or waiting) download to stop; safe from any thread."""
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        """Whether :meth:`cancel` has been called."""
        return self._cancel.is_set()

    def mark_started(self) -> None:
        """Record that this download now holds the acquisition session.

        Called by the acquisition layer once its turn has come (the session
        lease is taken), before anything is staged or fetched.
        """
        with self._lock:
            self._started = True

    def cancel_if_waiting(self) -> bool:
        """Cancel only a download that has not started yet; safe from any thread.

        A host that goes away (the first-run wizard closing) calls this so a
        download still waiting in line -- or not yet picked up by its worker
        -- never starts later with no UI left to show or stop it. A download
        already transferring is left alone (TASK-34100.14 review F4).

        Returns:
            Whether this call cancelled the download.
        """
        with self._lock:
            if self._started:
                return False
            self._cancel.set()
            return True
