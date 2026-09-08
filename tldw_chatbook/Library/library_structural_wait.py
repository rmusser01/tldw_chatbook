"""One structural wait: its label, its age, and its way out (task-32055).

A "structural wait" is a Library operation the user cannot work around while
it runs -- changing the File Notes folder, importing a skill, writing an
export bundle. Critique #8 found one of them (a folder change) sitting on
``Changing folder…`` with no progress, no deadline and no exit. This helper
holds the two things every such wait owes the user: an honest status line
that starts admitting it is slow after a short patience window, and the
callable that abandons it.

It is deliberately pure -- no widgets, no timers, no clock of its own -- so
every surface renders it into whatever status slot it already has, and the
copy cannot drift between them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

#: How long a wait may stay quiet before it must admit it is still working.
STRUCTURAL_WAIT_PATIENCE_SECONDS = 3.0

#: Surfaces that own a structural wait. The screen keeps one registry slot,
#: so each surface renders only the wait it started.
WAIT_OWNER_FILE_NOTES = "file-notes"
WAIT_OWNER_SKILL_IMPORT = "skill-import"
WAIT_OWNER_EXPORT = "export"


@dataclass
class StructuralWait:
    """One in-flight structural operation and its optional escape hatch.

    Attributes:
        label: What is happening, with no trailing ellipsis (the status
            line adds it) -- e.g. ``"Changing folder"``.
        started_at: ``time.monotonic()`` reading taken when the wait began.
        cancel: Callable that abandons the operation, or ``None`` when the
            operation genuinely cannot be abandoned. A ``None`` cancel is
            never advertised as one.
        owner: Which Library surface started it. The screen keeps one
            registry slot for every surface, so each renders the wait only
            when it is its own -- a running export must never show up as a
            folder change's status line.
    """

    label: str
    started_at: float
    cancel: Callable[[], None] | None = None
    owner: str = ""
    _cancelled: bool = field(default=False, init=False, repr=False)

    def status_line(
        self,
        now: float,
        patience_seconds: float = STRUCTURAL_WAIT_PATIENCE_SECONDS,
    ) -> str:
        """Return the status copy for this wait at ``now``.

        Args:
            now: Current ``time.monotonic()`` reading.
            patience_seconds: How long the wait stays quiet before it
                reports that it is still working.

        Returns:
            ``"<label>…"`` inside the patience window; after it,
            ``"<label>… · still working"`` plus ``" · Cancel"`` when this
            wait can actually be cancelled.
        """
        line = f"{self.label}…"
        if now - self.started_at < patience_seconds:
            return line
        line = f"{line} · still working"
        return f"{line} · Cancel" if self.cancel is not None else line

    def request_cancel(self) -> bool:
        """Run the cancel callable at most once.

        Returns:
            True when this call actually cancelled the wait; False when
            the wait has no cancel or was already cancelled.
        """
        if self._cancelled or self.cancel is None:
            return False
        self._cancelled = True
        self.cancel()
        return True

    @property
    def cancelled(self) -> bool:
        """Whether this wait has already been cancelled."""
        return self._cancelled
