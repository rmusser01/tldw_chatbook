"""Generated takes, newest first, so options can be compared.

A pane showing only the latest result asks the user to remember what the
previous one sounded like. The screen exists to identify which option works
best, which means hearing two and choosing -- so every take keeps its own
row and its own Play/Export rather than being replaced by the next one.

Session-scoped: takes are not persisted across restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static


@dataclass(frozen=True)
class SpeechTake:
    """One generated audio take.

    Attributes:
        take_id: Stable id, used to build the per-row control ids.
        voice: Voice used, one of the variables being compared.
        fmt: Audio format, e.g. ``"mp3"``.
        duration_s: Length in seconds.
        created_label: Short display time, e.g. ``"14:02"``.
    """

    take_id: str
    voice: str
    fmt: str
    duration_s: float
    created_label: str

    @property
    def summary(self) -> str:
        """Return the one-line description shown in the history row.

        States the variables the user is choosing between -- voice and
        format -- because a row that omits them cannot be compared against
        the row above it.

        Returns:
            e.g. ``"14:02  Nova · wav · 1:31"``.
        """
        minutes, seconds = divmod(int(self.duration_s), 60)
        return (
            f"{self.created_label}  {self.voice} · {self.fmt} · "
            f"{minutes}:{seconds:02d}"
        )


class SpeechResultHistory(Vertical):
    """The takes generated this session, newest first."""

    def __init__(self, *, takes: Iterable[SpeechTake] = (), **kwargs: Any) -> None:
        """Create the history.

        Args:
            takes: Existing takes, oldest first; rendered newest first.
            kwargs: Forwarded to ``Vertical``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-history {classes}".strip(), **kwargs)
        self.takes: list[SpeechTake] = list(takes)

    def compose(self) -> ComposeResult:
        """Yield the section head, then one row per take, newest first."""
        yield Static("Result", classes="speech-section-head")

        if not self.takes:
            yield Static(
                "No takes yet. Generate to synthesize the text.",
                id="speech-history-empty",
                classes="speech-result-state",
                markup=False,
            )
            return

        for take in reversed(self.takes):
            with Horizontal(classes="speech-take-row"):
                yield Static(take.summary, classes="speech-take-summary", markup=False)
                yield Button(
                    "Play",
                    id=f"speech-take-play-{take.take_id}",
                    classes="workbench-action speech-take-action",
                    compact=True,
                )
                yield Button(
                    "Export",
                    id=f"speech-take-export-{take.take_id}",
                    classes="workbench-action speech-take-action",
                    compact=True,
                )

    def add_take(self, take: SpeechTake) -> None:
        """Append a take and rebuild the list.

        Recomposes rather than mounting a row directly: the empty state has
        to be removed on the first take, and ordering is newest-first, so a
        single append would put the row in the wrong place and leave the
        placeholder behind.

        Args:
            take: The newly generated take.
        """
        self.takes.append(take)
        self.refresh(recompose=True)
