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
from textual.binding import Binding
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS import (
    TTSPlaygroundSelectionPreset,
    TTSPreferencesSnapshot,
)
from tldw_chatbook.UI.stts_playground_catalog import (
    SERVER_DEFAULT_VOICE_ID,
)

from textual.widgets import (
    Button,
    Collapsible,
    ProgressBar,
    RichLog,
    Input,
    Select,
    Static,
    Switch,
    TextArea,
)

from tldw_chatbook.UI.stts_playground_catalog import UNAVAILABLE_SELECT_VALUE

from ..Workbench.workbench_state import WorkbenchAction
from .speech_action_strip import SpeechActionStrip
from .speech_axis_row import AXIS_EMPTY_PROMPTS, DEFAULT_SPEED, SpeechAxisRow
from .speech_catalog_mixin import SpeechCatalogMixin
from .speech_playback_mixin import EXAMPLE_TEXTS, SpeechPlaybackMixin
from .speech_playground_model import AXIS_CONTROLS
from .speech_profile_mixin import SpeechProfileMixin
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
        id="tts-random-text-btn", label="Random", tooltip="Insert sample text"
    ),
    WorkbenchAction(id="tts-clear-text-btn", label="Clear", tooltip="Clear the text"),
    WorkbenchAction(
        id="tts-refresh-catalog-btn",
        label="Refresh",
        tooltip="Re-read available models and voices",
    ),
    WorkbenchAction(
        id="tts-save-default-btn",
        label="Save default",
        tooltip="Keep these axes as the app-wide defaults",
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


class SpeechPlaygroundPane(
    SpeechSynthesisMixin,
    SpeechCatalogMixin,
    SpeechPlaybackMixin,
    SpeechProfileMixin,
    Vertical,
):
    """The TTS Playground body: title, actions, input, settings, status.

    Synthesis comes from `SpeechSynthesisMixin`, shared with the legacy
    widget rather than reimplemented: the generate path is 322 lines of
    provider resolution and request building, and a second copy would drift
    from the first. The mixin queries its controls by id, which is why this
    rebuild kept the legacy ids.
    """

    #: The legacy widget's shortcuts, carried over verbatim. Deleting it
    #: took its BINDINGS with it while the `action_*` methods survived in
    #: `SpeechPlaybackMixin` -- five shortcuts silently stopped working, and
    #: the screen still advertised them. Nothing in the per-view tests could
    #: have noticed: the methods were all still there and still callable.
    BINDINGS = [
        Binding("ctrl+g", "generate_tts", "Generate Speech"),
        Binding("ctrl+r", "random_text", "Random Text"),
        Binding("ctrl+l", "clear_text", "Clear Text"),
        Binding("ctrl+p", "play_audio", "Play Audio"),
        Binding("ctrl+s", "stop_audio", "Stop Audio"),
    ]

    def __init__(
        self,
        *,
        provider: str = "audio_cpp",
        profile_preset: Any = None,
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
        self.init_playback_state()
        self.init_profile_state(profile_preset)

    @on(Button.Pressed, "#tts-save-default-btn")
    def _on_save_default_pressed(self, event: Button.Pressed) -> None:
        """Commit the current axes as defaults.

        Declared here rather than added to the shared `on_button_pressed`:
        that dispatcher is the legacy playground's, and this action is new
        to the rebuild.

        Args:
            event: The press.
        """
        event.stop()
        self._save_axes_as_default()

    @on(Switch.Changed)
    def on_tts_option_switch_changed(self, event: Switch.Changed) -> None:
        """Delegate to the shared catalog mixin.

        Args:
            event: Any Switch change in this pane.
        """
        self.handle_option_switch_changed(event)

    @on(Input.Changed)
    def on_tts_speed_changed(self, event: Input.Changed) -> None:
        """Mirror an axis edit, then delegate to the shared catalog mixin.

        dev wired the delegation up while the rebuild was in flight -- it
        had been an undispatched name, which is why an earlier version of
        this branch deleted it as dead. It is live now, so the pane carries
        it. The mirroring is this ruling's addition: the speed axis is the
        only Input among the axes, so this is also the pane's one
        Input.Changed chokepoint.

        Args:
            event: Any Input change in this pane.
        """
        self._mirror_axis_edit(event.input.id, event.value)
        self.handle_speed_changed(event)

    @on(TextArea.Changed)
    def on_tts_text_changed(self, event: TextArea.Changed) -> None:
        """Delegate to the shared catalog mixin.

        Args:
            event: Any TextArea change in this pane.
        """
        self.handle_text_changed(event)

    @on(Select.Changed)
    def on_tts_provider_select_changed(self, event: Select.Changed) -> None:
        """Mirror an axis edit, then delegate to the shared catalog mixin.

        The decorator has to live on this class: Textual collects `@on`
        handlers per-class in its metaclass, so one declared in the mixin is
        never registered. Despite the name, this sees every Select in the
        pane, not only the provider one -- which is exactly why it is also
        the pane's one Select.Changed chokepoint for mirroring axis edits.

        Args:
            event: Any Select change in this pane.
        """
        self._mirror_axis_edit(event.select.id, event.value)
        self.handle_provider_select_changed(event)

    def _mirror_axis_edit(self, control_id: str | None, value: object) -> None:
        """Mirror a user's direct edit of an axis control into the model.

        No `_applying_catalog_controls` guard: an earlier version of this
        method skipped mirroring while that flag was True, on the theory
        that it would otherwise double-mirror a catalog-driven write. It
        cannot, and never did -- `_apply_controls` and
        `_prime_profile_preset_controls` never hold the flag True across an
        `await`, so it is always back to False by the time Textual delivers
        the `Select.Changed`/`Input.Changed` that write produced (a
        five-scenario instrumented check found zero guard-True entries).
        Reusing that dead branch would have masked the real reason mirroring
        one of those deferred, catalog-driven messages is harmless: a
        widget's own `Changed` queue is FIFO, so this always sees the LAST
        value written to that control, and `_apply_controls`/
        `_prime_profile_preset_controls` already wrote that identical value
        into `axis_values` synchronously, before the message was even
        queued -- mirroring it again is a redundant no-op, not a
        correctness risk.

        Args:
            control_id: The changed widget's id, or ``None``.
            value: The widget's new value. Ignored unless it is a plain
                ``str`` -- the sentinel values `Select` uses for "loading"
                and "unavailable" are not session edits.
        """
        if control_id not in AXIS_CONTROLS or not isinstance(value, str):
            return
        self.axis_values[control_id] = value
        self._refresh_axis_markers()

    def _refresh_axis_markers(self) -> None:
        """Repaint the axis row's override markers from the current model.

        Tolerates the row not being mounted yet (`NoMatches`): this is
        called from `_apply_controls` and `_prime_profile_preset_controls`,
        which can run before `compose()` has yielded `#speech-axis-row`, or
        in a unit-test context that never mounts it.
        """
        try:
            row = self.query_one("#speech-axis-row", SpeechAxisRow)
        except NoMatches:
            return
        row.update_values(self.axis_values)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Re-evaluate whether Generate is available.

        Args:
            event: The text change.
        """
        if event.text_area.id == "tts-text-input":
            self._sync_generate_enabled()

    def _save_axes_as_default(self) -> None:
        """Persist the current axes as the app-wide defaults.

        The one path by which the Playground writes a persisted value, and
        it exists because the screen's purpose is to identify what works
        best: a comparison you cannot keep is half a tool. Everything else
        here stays session-scoped.

        Reuses the snapshot and event Settings posts rather than writing
        config directly, so both views commit defaults exactly one way.
        """
        provider = self._get_select_key(
            self.query_one("#tts-provider-select", Select)
        )
        model = self._get_select_key(self.query_one("#tts-model-select", Select))
        voice = self._get_select_key(self.query_one("#tts-voice-select", Select))
        fmt = self._get_select_key(self.query_one("#tts-format-select", Select))

        if not isinstance(provider, str) or not isinstance(fmt, str):
            # Refuse rather than persist a sentinel as though it were a
            # choice. Before a catalog loads there is nothing to commit.
            self.app.notify(
                "Choose a provider and format before saving them as default",
                severity="warning",
            )
            return

        try:
            speed = float(
                self.query_one("#tts-speed-input", Input).value or DEFAULT_SPEED
            )
        except ValueError:
            speed = float(DEFAULT_SPEED)

        preferences = TTSPreferencesSnapshot(
            provider_id=provider,
            model_mode="exact" if isinstance(model, str) else "first_available",
            model_id=model if isinstance(model, str) else None,
            voice_mode=(
                "exact"
                if isinstance(voice, str) and voice is not SERVER_DEFAULT_VOICE_ID
                else "server_default"
            ),
            voice_id=(
                voice
                if isinstance(voice, str) and voice is not SERVER_DEFAULT_VOICE_ID
                else None
            ),
            response_format=fmt,
            speed=speed,
        )
        self.app.post_message(
            STTSSettingsSaveEvent({}, preferences=preferences)
        )
        self.app.notify("Saved as default", severity="information")

    def _show_provider_specific_controls(self, provider: str) -> None:
        """Re-scope the parameter group and clip picker to `provider`.

        This is the one method in the catalog closure that was bound to the
        legacy layout, where it toggled a `hidden` class on five per-provider
        container boxes. Those containers are gone -- `SpeechParamGroup`
        mounts only the selected provider's knobs -- so the override does the
        equivalent thing for this layout.

        Swaps the two provider-scoped regions rather than recomposing the
        pane. `refresh(recompose=True)` destroys and rebuilds every child,
        including the axis selects the catalog loader is in the middle of
        populating -- the exact hazard `SpeechCatalogMixin` documents, which
        this method previously walked into. It cost five provider-switch
        tests, each timing out waiting for state written to a widget that no
        longer existed.

        Args:
            provider: The newly selected provider id.
        """
        # Before the early return: this runs on catalog application as well
        # as on user change, and the language axis needs settling either way.
        self._settle_language_axis(provider)

        if provider == self.provider:
            return
        self.provider = provider

        try:
            group = self.query_one("#speech-param-group")
            text_pane = self.query_one("#speech-text-pane")
        except NoMatches:
            return

        # Textual's `remove()` is DEFERRED: the widget is still mounted
        # when this returns, so mounting the replacement immediately raises
        # DuplicateIds on `speech-param-group`. The shared parameters
        # (`AUDIO_PARAMS`, `REQUEST_PARAMS`) are in every provider's group,
        # so their ids collide too.
        #
        # The callback RECONCILES to whatever provider is current rather than
        # mounting the one captured here. Two provider changes in quick
        # succession -- which is normal, since loading a catalog can trigger
        # another -- otherwise queue two mounts and the second duplicates the
        # first.
        group.remove()
        for row in self.query(".speech-source-row"):
            row.remove()
        self.call_after_refresh(self._reconcile_provider_regions)

    def _reconcile_provider_regions(self) -> None:
        """Mount the provider-scoped regions if they are not already there.

        Idempotent by design: it is scheduled once per provider change but
        several may be pending, and only the current provider should end up
        on screen.
        """
        if not self.is_mounted:
            return
        try:
            anchor = self.query_one("#speech-log-group")
            text_pane = self.query_one("#speech-text-pane")
        except NoMatches:
            return

        if not self.query("#speech-param-group"):
            self.mount(
                SpeechParamGroup(provider=self.provider, id="speech-param-group"),
                after=anchor,
            )
        if not self.query(".speech-source-row"):
            for widget in self._compose_voice_source():
                text_pane.mount(widget)

    def _settle_language_axis(self, provider: str) -> None:
        """Say something terminal on the language axis once a catalog is in.

        Language is Kokoro-only; the legacy screen hid the row entirely for
        every other provider. As a permanent axis it cannot be hidden, so it
        has to say what is true instead -- and "Waiting for provider…" stops
        being true the moment that provider's catalog has arrived without
        languages. Left alone it waits forever.

        Args:
            provider: The provider whose catalog is being applied.
        """
        if provider == "kokoro":
            return
        try:
            select = self.query_one("#tts-language-select", Select)
        except NoMatches:
            return
        label = AXIS_EMPTY_PROMPTS["tts-language-select"]
        select.set_options([(label, UNAVAILABLE_SELECT_VALUE)])
        select.value = UNAVAILABLE_SELECT_VALUE
        select.disabled = True

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
        self._rehydrate_handler_state()
        # dev's mount sequence for the profile preset, kept in order: a pane
        # opened on an exact profile shows that profile immediately, before
        # discovery runs, rather than showing a catalog the user did not ask
        # for and then replacing it.
        if self._profile_preset is not None:
            self.call_after_refresh(self._finish_profile_preset_mount)
        else:
            self._sync_profile_preview_status()
            self._load_provider_catalog(initialize=True)

    def _finish_profile_preset_mount(self) -> None:
        """Prime an exact preset after nested Select children are mounted."""

        if not self.is_mounted:
            return
        if self._profile_preset is not None:
            self._prime_profile_preset_controls()
            self.query_one("#tts-text-input", TextArea).focus()
        else:
            self._sync_profile_preview_status()
        self._load_provider_catalog(initialize=True)

    def apply_profile_preset(
        self,
        preset: TTSPlaygroundSelectionPreset,
    ) -> None:
        """Apply one exact process-local preset to this mounted Playground."""

        if type(preset) is not TTSPlaygroundSelectionPreset:
            raise TypeError("preset must be TTSPlaygroundSelectionPreset")
        self.init_profile_state(preset)
        if self.is_mounted:
            self.call_after_refresh(self._finish_profile_preset_mount)

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
        # dev's profile-preview status line and save action. Both start
        # hidden and are revealed by `SpeechProfileMixin`, which queries
        # them by these exact ids -- so the rebuild mounts them unchanged.
        yield Static(
            "",
            id="tts-profile-preview-status",
            classes="speech-status-line hidden",
            markup=False,
        )

        yield SpeechActionStrip(PLAYGROUND_ACTIONS, id="speech-playground-actions")

        yield SpeechAxisRow(
            values=self.axis_values,
            defaults=self.axis_defaults,
            id="speech-axis-row",
        )

        with Horizontal(id="speech-split", classes="speech-split"):
            with Vertical(id="speech-text-pane", classes="speech-split-pane"):
                yield Static("Text", classes="speech-section-head")
                # Seeded with the first example, as legacy was. The screen
                # exists to let someone hear the options and pick one, so it
                # should be one keypress from audio -- not a blank box that
                # asks you to think of something to say first.
                editor = TextArea(
                    EXAMPLE_TEXTS[0],
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
        # Both carry their legacy copy. The shared catalog code only toggles
        # these lines' visibility -- the text itself lived in the legacy
        # compose(), so mounting them empty left a blank status line and a
        # restriction notice that never said anything.
        yield Static(
            "Loading TTS providers…",
            id="tts-provider-status",
            classes="speech-status-line",
            markup=False,
        )
        yield Static(
            "audio.cpp returns one complete WAV and currently uses speed 1.0.",
            id="tts-audio-cpp-restrictions",
            classes="speech-status-line hidden",
            markup=False,
        )

    def _compose_voice_source(self) -> ComposeResult:
        """Yield the clip picker for providers that synthesize from one.

        Mounted only for the providers that use it. Legacy mounted both rows
        for every provider and toggled a `hidden` class, which leaves a
        focusable control that does nothing -- the same reason the parameter
        group scopes knobs by provider.

        Builds each row by CONSTRUCTION rather than with the `with
        Horizontal(...)` compose idiom. That idiom only works inside
        `compose()`, and this is also called from
        `_reconcile_provider_regions` when the provider changes -- where it
        raised IndexError off Textual's empty compose stack.

        Returns:
            A ``ComposeResult`` yielding a reference-audio row, a voice
            upload row, or nothing.
        """
        if self.provider in REFERENCE_AUDIO_PROVIDERS:
            yield Horizontal(
                Static(
                    "No reference clip",
                    id="reference-audio-status",
                    classes="speech-source-status",
                    markup=False,
                ),
                SpeechActionStrip(
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
                ),
                id="reference-audio-row",
                classes="speech-source-row",
            )

        if self.provider in VOICE_UPLOAD_PROVIDERS:
            yield Horizontal(
                Static(
                    "No voice sample",
                    id="higgs-voice-status",
                    classes="speech-source-status",
                    markup=False,
                ),
                SpeechActionStrip(
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
                ),
                id="higgs-voice-upload-row",
                classes="speech-source-row",
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
                # `total` and `hidden` both matter. Without `total` a
                # ProgressBar renders its indeterminate pulse, so an idle
                # screen animates two bars forever; without `hidden` it is
                # on screen with nothing to report. Legacy carried both and
                # the rebuild dropped them -- visible only on a live run.
                yield ProgressBar(
                    id="audio-progress-bar",
                    total=100,
                    show_eta=False,
                    show_percentage=False,
                    classes="audio-progress hidden",
                )
                yield Static(
                    "0:00 / 0:00",
                    id="audio-time-display",
                    classes="speech-player-time",
                    markup=False,
                )
            yield SpeechActionStrip(PLAYER_ACTIONS, id="speech-result-actions")
            yield Button(
                "Save result as profile",
                id="audio-save-profile-btn",
                classes="workbench-action hidden",
                disabled=True,
                compact=True,
            )

    def _compose_generation_status(self) -> ComposeResult:
        """Yield the generation status line, progress, and the log.

        The log is collapsed: it is diagnostic, and the defect this rebuild
        exists to fix was rows spent on things nobody reads by default.

        Returns:
            A ``ComposeResult`` yielding the status container and the log.
        """
        # Hidden until a generation starts, as legacy was: otherwise the
        # placeholder ETA (`--% --:--:--`) sits on an idle screen.
        with Horizontal(
            id="generation-status-container",
            classes="speech-generation-status hidden",
        ):
            yield Static(
                "Ready to generate",
                id="generation-status-text",
                classes="speech-generation-text",
                markup=False,
            )
            yield ProgressBar(
                id="generation-progress",
                total=100,
                show_eta=True,
                show_percentage=True,
            )

        yield Collapsible(
            RichLog(id="tts-generation-log", classes="speech-log", markup=False),
            title="Generation log",
            id="speech-log-group",
            collapsed=True,
        )
