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
from .speech_axis_row import SpeechAxisRow
from .speech_param_group import SpeechParamGroup
from .speech_result_history import SpeechResultHistory, SpeechTake

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
#: The text actions. Ids are the legacy control ids on purpose: the rebuild
#: re-sites controls, it does not rename them, so wiring is an id lookup
#: rather than a translation table.
PLAYGROUND_ACTIONS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(
        id="tts-generate-btn",
        label="Generate",
        tooltip="Synthesize the text",
        primary=True,
    ),
    WorkbenchAction(
        id="tts-random-text-btn", label="Random text", tooltip="Insert sample text"
    ),
    WorkbenchAction(id="tts-clear-text-btn", label="Clear", tooltip="Clear the text"),
    WorkbenchAction(
        id="tts-refresh-catalog-btn",
        label="Refresh catalog",
        tooltip="Re-read available models and voices",
    ),
)

#: Playback actions, kept beside the result rather than mixed into the strip
#: that acts on the text.
PLAYER_ACTIONS: tuple[WorkbenchAction, ...] = (
    WorkbenchAction(id="audio-play-btn", label="Play", tooltip="Play"),
    WorkbenchAction(id="pause-audio-btn", label="Pause", tooltip="Pause"),
    WorkbenchAction(id="stop-audio-btn", label="Stop", tooltip="Stop"),
    WorkbenchAction(id="audio-export-btn", label="Export", tooltip="Save the audio"),
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

    def __init__(
        self,
        *,
        provider: str = "audio_cpp",
        axis_values: dict[str, str] | None = None,
        axis_defaults: dict[str, str] | None = None,
        takes: Any = None,
        capability_line: str = "",
        **kwargs: Any,
    ) -> None:
        """Create the pane.

        Args:
            provider: Selected provider; decides the parameter group.
            axis_values: Effective value per comparison axis.
            axis_defaults: Persisted default per axis, for override marking.
            takes: Takes generated this session.
            capability_line: One-line local-speech status.
            kwargs: Forwarded to ``Vertical``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-pane {classes}".strip(), **kwargs)
        #: None until first measured, so the first sync always applies.
        self._stacked: bool | None = None
        #: Effective axis values for this session, and the persisted
        #: defaults they are compared against. The pane never writes the
        #: defaults -- overrides are session-scoped by design.
        self.axis_values: dict[str, str] = dict(axis_values or {})
        self.axis_defaults: dict[str, str] = dict(axis_defaults or {})
        #: Selected provider, which decides the parameter group's contents.
        self.provider = provider
        #: Takes generated this session.
        self.takes: list[SpeechTake] = list(takes or ())
        #: One-line capability status, sourced from lab_speech_status by the
        #: screen rather than re-derived here.
        self.capability_line = capability_line

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
        """Compose the Playground.

        Order is the comparison loop: what you are about to do, the
        variables you are comparing, the text, then the results and the
        knobs you rarely touch.

        Returns:
            A ``ComposeResult`` yielding the title, action strip, axis
            chips, the Text/Result split, the collapsed provider parameters
            and the capability line.
        """
        yield Static("🎤 TTS Playground", classes="speech-pane-title")

        yield CommandStrip(PLAYGROUND_ACTIONS, id="speech-playground-actions")

        yield SpeechAxisRow(
            values=self.axis_values,
            defaults=self.axis_defaults,
            id="speech-axis-row",
        )

        with Horizontal(id="speech-split", classes="speech-split"):
            with Vertical(id="speech-text-pane", classes="speech-split-pane"):
                yield Static("Text", classes="speech-section-head")
                editor = TextArea(
                    "",
                    id="tts-text-input",
                    classes="speech-input",
                    placeholder="Type or paste the text to synthesize...",
                )
                editor.show_line_numbers = False
                yield editor

            with Vertical(id="speech-result-pane", classes="speech-split-pane"):
                yield SpeechResultHistory(
                    takes=self.takes, id="speech-result-history"
                )
                yield CommandStrip(PLAYER_ACTIONS, id="speech-result-actions")

        yield SpeechParamGroup(
            provider=self.provider, id="speech-param-group"
        )

        yield Static(
            self.capability_line,
            id="speech-capability-line",
            classes="speech-status-line",
            markup=False,
        )
