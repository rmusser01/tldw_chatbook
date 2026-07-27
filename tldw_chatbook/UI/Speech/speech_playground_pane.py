"""TTS Playground in the Console grammar.

Replaces the legacy playground's stacked form. That form measured **93 rows
in a 34-row viewport** -- every label/input pair cost 4 rows and two
explanatory notices cost 5 each -- which put `Generate Speech` at y=60,
21 rows below the fold on arrival.

The Console idiom this follows, read off the running Console screen rather
than invented here:

- one row per thing, never a box per control
- visible commands as a text action strip (``CommandStrip``), not chunky
  buttons stacked down the page
- settings as ``Label: value`` chips on one line, the way Console states
  provider/model/character/RAG above its composer
- a single bordered input, like Console's composer
- recovery copy as one line, not a five-row block

Nothing here wraps a legacy STTS widget: the point is the grammar, and the
old widgets carry the old rhythm with them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, TextArea

from ..Workbench.workbench_state import WorkbenchAction
from ..Workbench.workbench_widgets import CommandStrip

if TYPE_CHECKING:
    pass

#: Below this pane width the split stacks. Two panes need roughly 30 cells
#: each to stay readable; the Lab rail and inspector already take their
#: share, so at 80 columns the body is ~41 and side-by-side would give each
#: about 20. Mirrors `PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH`'s approach --
#: a measured threshold, toggled from `on_resize`, because Textual has no
#: media queries.
SPEECH_SPLIT_MIN_WIDTH = 64

#: The playground's visible commands, in the order Console orders its own:
#: the thing you came to do first, then what you do to the result.
PLAYGROUND_ACTIONS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(
        id="speech-generate",
        label="Generate",
        tooltip="Synthesize the text above",
        primary=True,
    ),
    WorkbenchAction(id="speech-play", label="Play", tooltip="Play the last result"),
    WorkbenchAction(id="speech-pause", label="Pause", tooltip="Pause playback"),
    WorkbenchAction(id="speech-stop", label="Stop", tooltip="Stop playback"),
    WorkbenchAction(id="speech-export", label="Export", tooltip="Save the audio file"),
    WorkbenchAction(id="speech-clear", label="Clear", tooltip="Clear the text"),
)

#: Actions that operate on the synthesized result, kept beside it rather
#: than mixed into the strip that acts on the text.
RESULT_ACTIONS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(
        id="speech-export-result", label="Export", tooltip="Save the audio file"
    ),
    WorkbenchAction(
        id="speech-copy-path", label="Copy path", tooltip="Copy the file path"
    ),
)


class SpeechChip(Static):
    """One `Label: value` cell in the settings line.

    A chip, not a form field. The legacy screen spent four rows on each of
    these -- a label row, an input row, and two rows of padding -- for
    values that are one short token each.
    """

    def __init__(self, label: str, value: str, **kwargs: Any) -> None:
        """Create a settings chip.

        Args:
            label: The setting's name, e.g. ``"Voice"``.
            value: Its current value, rendered plainly beside the label.
            kwargs: Forwarded to ``Static``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(
            f"{label}: {value}",
            classes=f"speech-chip {classes}".strip(),
            markup=False,
            **kwargs,
        )


class SpeechPlaygroundPane(Vertical):
    """The TTS Playground body: title, actions, input, settings, status."""

    def __init__(self, **kwargs: Any) -> None:
        """Create the pane."""
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-pane {classes}".strip(), **kwargs)
        #: None until first measured, so the first sync always applies.
        self._stacked: bool | None = None

    def on_resize(self) -> None:
        """Stack the split when the pane is too narrow to hold two columns."""
        self._sync_split_layout()

    def on_mount(self) -> None:
        """Apply the split layout once the pane has a real width."""
        self._sync_split_layout()

    def _sync_split_layout(self) -> None:
        """Toggle the stacked class from the pane's measured width."""
        stacked = self.size.width < SPEECH_SPLIT_MIN_WIDTH
        if stacked == self._stacked:
            return
        self._stacked = stacked
        self.set_class(stacked, "speech-split-stacked")

    def compose(self) -> ComposeResult:
        """Compose the pane.

        Returns:
            A ``ComposeResult`` yielding, in order: the view title, the
            command strip, the text input, the settings chip line, and a
            one-line capability status.
        """
        yield Static("🎤 TTS Playground", classes="speech-pane-title")

        yield CommandStrip(PLAYGROUND_ACTIONS, id="speech-playground-actions")

        # Settings sit ABOVE the input, where Console puts its status chips
        # relative to the composer: what you are about to do with, then the
        # thing you type into.
        with Horizontal(id="speech-settings-line", classes="speech-chip-row"):
            yield SpeechChip("Provider", "audio.cpp", id="speech-chip-provider")
            yield SpeechChip("Voice", "Server default", id="speech-chip-voice")
            yield SpeechChip("Format", "mp3", id="speech-chip-format")
            yield SpeechChip("Speed", "1.0", id="speech-chip-speed")

        # Text and Result side by side: the pane has width to spend and does
        # not have 34 rows of content to spend vertically. Stacks below
        # SPEECH_SPLIT_MIN_WIDTH -- see `on_resize`.
        with Horizontal(id="speech-split", classes="speech-split"):
            with Vertical(id="speech-text-pane", classes="speech-split-pane"):
                yield Static("Text", classes="speech-section-head")
                editor = TextArea(
                    "",
                    id="speech-input",
                    classes="speech-input",
                    placeholder="Type or paste the text to synthesize...",
                )
                editor.show_line_numbers = False
                yield editor

            with Vertical(id="speech-result-pane", classes="speech-split-pane"):
                yield Static("Result", classes="speech-section-head")
                # The old screen's player row and audio container both sat
                # below the fold. Without somewhere for the result to land,
                # Generate has no visible consequence.
                yield Static(
                    "No audio yet.",
                    id="speech-result-state",
                    classes="speech-result-state",
                    markup=False,
                )
                yield SpeechChip("Voice", "Server default", id="speech-res-voice")
                yield SpeechChip("Format", "mp3", id="speech-res-format")
                yield SpeechChip("Size", "--", id="speech-res-size")
                yield CommandStrip(RESULT_ACTIONS, id="speech-result-actions")

        yield Static(
            "audio.cpp is unavailable — open TTS Settings to configure it.",
            id="speech-capability-line",
            classes="speech-status-line",
            markup=False,
        )
