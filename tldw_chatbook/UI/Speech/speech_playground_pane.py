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

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Collapsible,
    ProgressBar,
    RichLog,
    Select,
    Static,
    TextArea,
)

from ..Workbench.workbench_state import WorkbenchAction
from .speech_action_strip import SpeechActionStrip
from .speech_axis_row import SpeechAxisRow
from .speech_catalog_mixin import SpeechCatalogMixin
from .speech_synthesis_mixin import SpeechSynthesisMixin
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

#: Providers that synthesize from a reference clip, so the clip picker is
#: only mounted for them -- legacy mounted it always and hid it, which left
#: it focusable while inert.
REFERENCE_AUDIO_PROVIDERS = ("chatterbox",)

#: Providers with their own voice-sample upload flow.
VOICE_UPLOAD_PROVIDERS = ("higgs",)

#: The playground's visible commands, in the order Console orders its own:
#: the thing you came to do first, then what you do to the result.
#: The text actions. Ids are the legacy control ids, and `SpeechActionStrip`
#: mounts them verbatim: the handler matches `event.button.id` against these
#: exact strings, so a strip that renames its buttons never fires.
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


class SpeechPlaygroundPane(SpeechSynthesisMixin, SpeechCatalogMixin, Vertical):
    """The TTS Playground body: title, actions, input, settings, status.

    Synthesis comes from `SpeechSynthesisMixin`, shared with the legacy
    widget rather than reimplemented: the generate path is 322 lines of
    provider resolution and request building, and a second copy would drift
    from the first. The mixin queries its controls by id, which is why this
    rebuild kept the legacy ids.
    """

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
        self.init_synthesis_state()
        self.init_catalog_state()

    @on(Button.Pressed, "#tts-generate-btn")
    def _on_generate_pressed(self, event: Button.Pressed) -> None:
        """Run the shared synthesis path.

        Args:
            event: The press, stopped here so it does not also reach a
                host screen's own button handling.
        """
        event.stop()
        self._generate_tts()

    @on(Select.Changed)
    def on_tts_provider_select_changed(self, event: Select.Changed) -> None:
        """Delegate to the shared catalog mixin.

        The decorator has to live on this class: Textual collects `@on`
        handlers per-class in its metaclass, so one declared in the mixin is
        never registered.

        Args:
            event: Any Select change in this pane.
        """
        self.handle_provider_select_changed(event)

    def _show_provider_specific_controls(self, provider: str) -> None:
        """Re-scope the parameter group and clip picker to `provider`.

        This is the one method in the catalog closure that was bound to the
        legacy layout, where it toggled a `hidden` class on five per-provider
        container boxes. Those containers are gone -- `SpeechParamGroup`
        mounts only the selected provider's knobs -- so the override does the
        equivalent thing for this layout.

        Deliberately NOT a `Select.Changed` handler of its own. The mixin's
        `on_tts_provider_select_changed` is decorated `@on(Select.Changed)`
        with no selector, so a second handler here would also fire, and a
        recompose would destroy the widgets the mixin is midway through
        populating.

        Args:
            provider: The newly selected provider id.
        """
        if provider == self.provider:
            return
        self.provider = provider
        self.refresh(recompose=True)

    def _refresh_provider_ids(self) -> None:
        """Record the provider values on offer.

        `_generation_readiness_error` distinguishes a selection that has gone
        stale from one that was never valid, and needs the current option
        set to do it.
        """
        try:
            select = self.query_one("#tts-provider-select", Select)
        except Exception:  # noqa: BLE001 - not yet mounted
            return
        options = getattr(select, "_options", ()) or ()
        self._provider_ids = frozenset(
            value for _label, value in options if isinstance(value, str)
        )

    def on_resize(self) -> None:
        """Stack the split when the pane is too narrow to hold two columns."""
        self._sync_split_layout()

    def on_mount(self) -> None:
        """Apply the split layout, then load the catalog.

        The catalog is what fills the axes. Without this call the pane
        renders correct, empty selects -- controls with nothing to choose --
        which looks like a finished screen and cannot synthesize anything.
        """
        self._sync_split_layout()
        self._refresh_provider_ids()
        self._load_provider_catalog(initialize=True)

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

        yield SpeechActionStrip(PLAYGROUND_ACTIONS, id="speech-playground-actions")

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
                yield from self._compose_voice_source()

            with Vertical(id="speech-result-pane", classes="speech-split-pane"):
                yield SpeechResultHistory(
                    takes=self.takes, id="speech-result-history"
                )
                yield from self._compose_player()

        yield from self._compose_generation_status()

        yield SpeechParamGroup(
            provider=self.provider, id="speech-param-group"
        )

        yield Static(
            self.capability_line,
            id="speech-capability-line",
            classes="speech-status-line",
            markup=False,
        )
        yield Static(
            "",
            id="tts-provider-status",
            classes="speech-status-line",
            markup=False,
        )
        yield Static(
            "",
            id="tts-audio-cpp-restrictions",
            classes="speech-status-line",
            markup=False,
        )

    def _compose_voice_source(self) -> ComposeResult:
        """Yield the clip picker for providers that synthesize from one.

        Mounted only for the providers that use it. Legacy mounted both
        rows for every provider and toggled a `hidden` class, which leaves
        a focusable control that does nothing -- the same reason the
        parameter group scopes knobs by provider.

        Returns:
            A ``ComposeResult`` yielding a reference-audio row, a voice
            upload row, or nothing.
        """
        if self.provider in REFERENCE_AUDIO_PROVIDERS:
            with Horizontal(id="reference-audio-row", classes="speech-source-row"):
                yield Static(
                    "No reference clip",
                    id="reference-audio-status",
                    classes="speech-source-status",
                    markup=False,
                )
                yield SpeechActionStrip(
                    (
                        WorkbenchAction(
                            id="reference-audio-btn",
                            label="Choose clip",
                            tooltip="Pick a reference audio file",
                        ),
                        WorkbenchAction(
                            id="clear-reference-audio-btn",
                            label="Clear",
                            tooltip="Forget the reference clip",
                        ),
                    ),
                    id="speech-reference-actions",
                )

        if self.provider in VOICE_UPLOAD_PROVIDERS:
            with Horizontal(id="higgs-voice-upload-row", classes="speech-source-row"):
                yield Static(
                    "No voice sample",
                    id="higgs-voice-status",
                    classes="speech-source-status",
                    markup=False,
                )
                yield SpeechActionStrip(
                    (
                        WorkbenchAction(
                            id="higgs-voice-upload-btn",
                            label="Upload voice",
                            tooltip="Pick a voice sample to clone",
                        ),
                        WorkbenchAction(
                            id="higgs-clear-voice-btn",
                            label="Clear",
                            tooltip="Forget the voice sample",
                        ),
                    ),
                    id="speech-voice-upload-actions",
                )

    def _compose_player(self) -> ComposeResult:
        """Yield the playback region for the take being auditioned.

        Returns:
            A ``ComposeResult`` yielding the player container and its
            transport actions.
        """
        with Vertical(id="audio-player-container", classes="speech-player"):
            yield Static(
                "Nothing loaded",
                id="audio-player-status",
                classes="speech-player-status",
                markup=False,
            )
            with Horizontal(classes="speech-player-transport"):
                yield ProgressBar(
                    id="audio-progress-bar",
                    show_eta=False,
                    show_percentage=False,
                )
                yield Static(
                    "0:00 / 0:00",
                    id="audio-time-display",
                    classes="speech-player-time",
                    markup=False,
                )
            yield SpeechActionStrip(PLAYER_ACTIONS, id="speech-result-actions")

    def _compose_generation_status(self) -> ComposeResult:
        """Yield the generation status line, progress, and the log.

        The log is collapsed: it is diagnostic, and the defect this rebuild
        exists to fix was rows spent on things nobody reads by default.

        Returns:
            A ``ComposeResult`` yielding the status container and the log.
        """
        with Horizontal(
            id="generation-status-container", classes="speech-generation-status"
        ):
            yield Static(
                "Ready to generate",
                id="generation-status-text",
                classes="speech-generation-text",
                markup=False,
            )
            yield ProgressBar(
                id="generation-progress", show_eta=True, show_percentage=True
            )

        yield Collapsible(
            RichLog(id="tts-generation-log", classes="speech-log", markup=False),
            title="Generation log",
            id="speech-log-group",
            collapsed=True,
        )
