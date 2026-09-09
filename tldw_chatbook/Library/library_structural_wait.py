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

    def is_slow(
        self,
        now: float,
        patience_seconds: float = STRUCTURAL_WAIT_PATIENCE_SECONDS,
    ) -> bool:
        """Whether this wait has outlived its patience window at ``now``.

        Args:
            now: Current ``time.monotonic()`` reading, on the same clock as
                ``started_at``.
            patience_seconds: How long the wait stays quiet before it
                reports that it is still working.

        Returns:
            True once ``now`` is at least ``patience_seconds`` past
            ``started_at``.
        """
        return now - self.started_at >= patience_seconds

    def with_patience_suffix(
        self,
        base: str,
        now: float,
        patience_seconds: float = STRUCTURAL_WAIT_PATIENCE_SECONDS,
    ) -> str:
        """Append the still-working suffix to a caller-owned status line.

        A surface with something better to say than the label -- an export's
        per-phase progress, a refused second import -- keeps its own line and
        the wait only adds to it once it stops being quiet.

        Args:
            base: The surface's own line, already complete.
            now: Current ``time.monotonic()`` reading.
            patience_seconds: How long the wait stays quiet before it
                reports that it is still working.

        Returns:
            ``base`` inside the patience window; after it, ``base`` plus
            ``" · still working"`` and ``" · Cancel"`` when this wait can
            actually be cancelled.
        """
        if not self.is_slow(now, patience_seconds):
            return base
        line = f"{base} · still working"
        return f"{line} · Cancel" if self.cancel is not None else line

    def status_line(
        self,
        now: float,
        patience_seconds: float = STRUCTURAL_WAIT_PATIENCE_SECONDS,
    ) -> str:
        """Return the status copy for a surface with no line of its own.

        Args:
            now: Current ``time.monotonic()`` reading.
            patience_seconds: How long the wait stays quiet before it
                reports that it is still working.

        Returns:
            ``"<label>…"`` inside the patience window; after it,
            ``"<label>… · still working"`` plus ``" · Cancel"`` when this
            wait can actually be cancelled.
        """
        return self.with_patience_suffix(f"{self.label}…", now, patience_seconds)

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
