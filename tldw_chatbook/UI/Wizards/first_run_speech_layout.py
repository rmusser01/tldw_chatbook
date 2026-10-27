"""The Speech step's widget tree (TASK-34100.13, voice-speech-12/15).

Moved out of ``first_run_speech_step.py`` (pinned by the module-size ratchet)
so the step could gain its engine choice without growing. Pure: it builds
from the step's in-memory state only; ``SpeechSetupStep.compose_step`` yields
from ``compose_speech_step``.

Layout, top to bottom: title; the engine radio and its detail line; then,
only under "On this computer -- Parakeet", the choice-then-action block --
a "Recommended:" card, the saved-state line, a one-row Language select, the
status line with "Review and install…" directly below it, and an Advanced
collapsible holding Precision (plain labels), "Use model from disk…" and the
transcribe.cpp GGUF option. With the language now one row, the primary
action stays visible at 120x40 (the old "status first" order is retired).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Label, Select, Static

from tldw_chatbook.UI.Wizards import first_run_speech_engine as engines
from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupRadioButton,
    SetupRadioSet,
)
from tldw_chatbook.Utils.input_validation import escape_markup
from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

if TYPE_CHECKING:
    from tldw_chatbook.UI.Wizards.first_run_speech_step import SpeechSetupStep

#: voice-speech-12: plain words first, the technical name in brackets.
PRECISION_LABELS = {
    "int8": "Smaller — recommended (INT8)",
    "f32": "Full precision, larger (F32)",
}


class EngineRadioSet(SetupRadioSet):
    """The engine radio, its highlight kept on the chosen engine across a rebuild.

    Review F7 (TASK-34100.13): RadioSet seats its highlight on the first row
    when it mounts, so after any rebuild of the step the first Down press
    re-selected the engine already chosen.
    """

    def on_mount(self) -> None:
        # Runs before RadioSet's own mount handler, which seats row 0.
        self.call_after_refresh(self._highlight_choice)

    def _highlight_choice(self) -> None:
        # The step may have been rebuilt again before this ran.
        nodes = list(self.children)
        if self.is_attached and self._pressed_button in nodes:
            self._selected = nodes.index(self._pressed_button)


def focus_inside(step: "SpeechSetupStep") -> str | None:
    """The id to give focus back to after a rebuild, or None to leave it.

    Returns:
        The focused widget's id (``""`` when it has none) while the focus is
        in the step or the step is still arriving (its first-show reload
        runs before the wizard's own focus call lands); None otherwise, so a
        rebuild never takes the focus from somewhere the user put it.
    """
    focused = step.app.focused if step.is_attached else None
    if focused is not None and step in focused.ancestors_with_self:
        step._arriving = False  # the keyboard reached the step: arrival is over
        return focused.id or ""
    return "" if getattr(step, "_arriving", False) else None


def restore_focus(step: "SpeechSetupStep", focused_id: str | None) -> None:
    """Give the keyboard back after the step rebuilt its own widgets.

    Live (TASK-34100.13 resume): the wizard focused the engine radio on
    arrival, then the first-show reload rebuilt the step and the focus fell
    to Next, so Down and Space did nothing on the step's one choice. The
    rebuilt widget with the same id gets it back, else the engine radio.
    """
    if focused_id is None or not step.display or step.has_class("hidden"):
        return
    focused = step.app.focused
    if focused is not None and step in focused.ancestors_with_self:
        return
    for selector in (f"#{focused_id}" if focused_id else "", "#setup-speech-engine"):
        target = next(iter(step.query(selector)), None) if selector else None
        if target is not None and target.focusable and target.display:
            target.focus()
            return


def engine_locked(step: "SpeechSetupStep") -> bool:
    """Whether the engine may not change now: an install, an activation or a
    from-disk check is under way (its Cancel lives under Parakeet)."""
    return bool(
        step._operation is not None
        or step._external_busy
        or step._external_commit_pending
    )


def show_engine_choice(step: "SpeechSetupStep") -> None:
    """Show the chosen engine's detail line and (under Parakeet) its block.

    Updates in place: rebuilding the step for an engine change cost keyboard
    users a dead Down press after every change (review F7).
    """
    choices = engines.engine_choices(step._engine_facts)
    detail = next((c.detail for c in choices if c.key == step._engine), "")
    for line in step.query("#setup-speech-engine-detail").results(Static):
        line.update(detail)
    for block in step.query("#setup-speech-parakeet"):
        block.display = step._engine == engines.ENGINE_PARAKEET


def _engine_block(step: "SpeechSetupStep") -> Iterator[Widget]:
    choices = engines.engine_choices(step._engine_facts)
    yield Label("Dictation engine", classes="setup-field-label")
    with EngineRadioSet(
        id="setup-speech-engine",
        classes="setup-choice-list setup-choice-fit",
        disabled=engine_locked(step),
    ):
        for choice in choices:
            yield SetupRadioButton(
                choice.label,
                value=choice.key == step._engine,
                id=f"setup-speech-engine-{choice.key}",
            )
    detail = next((c.detail for c in choices if c.key == step._engine), "")
    yield Static(
        detail, id="setup-speech-engine-detail", classes="setup-step-note", markup=False
    )


def _language_select(step: "SpeechSetupStep") -> Select:
    options = [
        (
            option.display_name + (" (recommended)" if option.code == "en" else ""),
            option.code,
        )
        for option in speech_state.speech_language_options(
            curated_model_ids=step._curated_model_ids()
        )
        if option.selectable
    ]
    return Select(
        options,
        value=step._selected_language,
        allow_blank=False,
        id="setup-speech-language",
        disabled=step._lifecycle_pending,
    )


def _advanced_block(step: "SpeechSetupStep", action_disabled: bool) -> Iterator[Widget]:
    yield Label("Precision", classes="setup-field-label")
    with SetupRadioSet(
        id="setup-speech-precision-choice",
        classes="setup-choice-list setup-choice-fit",
    ):
        for option in speech_state.speech_precision_options(
            model_id=step._selection().model_id,
            curated_selections=step._curated_selections(),
        ):
            label = PRECISION_LABELS.get(option.value, option.display_name)
            if not option.selectable:
                label += " — not yet available for managed install"
            # Minor 8: pre-press only the selected option.
            yield SetupRadioButton(
                label,
                id=f"setup-speech-precision-{option.value}",
                value=option.selectable and option.value == step._selected_precision,
                disabled=not option.selectable or step._lifecycle_pending,
            )
    yield Button(
        "Use model from disk…",
        id="setup-speech-use-from-disk",
        variant="primary" if action_disabled else "default",
        disabled=step._external_busy or step._lifecycle_pending,
    )
    yield Static(
        "Existing local transcribe.cpp GGUF configured."
        if step._transcribe_cpp_configured
        else "No existing local transcribe.cpp GGUF configured.",
        id="setup-speech-transcribe-cpp-status",
        classes="setup-subtitle",
        markup=False,
    )
    yield Button(
        "Choose another GGUF…"
        if step._transcribe_cpp_configured
        else "Use an existing transcribe.cpp GGUF…",
        id="setup-speech-choose-transcribe-cpp-gguf",
        disabled=step._external_commit_pending or step._lifecycle_pending,
    )


def compose_speech_step(step: "SpeechSetupStep") -> Iterator[Widget]:
    """Yield the Speech step's widgets for its current state."""
    with Vertical(classes="setup-speech"):
        yield Static("Speech transcription (optional)", classes="setup-title")
        yield from _engine_block(step)
        parakeet = Vertical(id="setup-speech-parakeet")
        parakeet.display = step._engine == engines.ENGINE_PARAKEET
        with parakeet:
            yield Static(
                engines.recommendation_line(
                    step._model_label(),
                    step._parakeet_mib(),
                    chosen=step._acted_this_run,
                ),
                id="setup-speech-recommendation",
                classes="setup-subtitle",
                markup=False,
            )
            # A from-disk check in progress reports (and cancels) up here, by
            # the card, not inside Advanced below the fold.
            external_status = Static(
                step._external_status,
                id="setup-speech-external-status",
                classes="setup-subtitle",
                markup=False,
            )
            external_status.display = bool(step._external_status)
            yield external_status
            if step._external_busy:
                yield Button(
                    "Cancel external setup",
                    id="setup-speech-cancel-external",
                    variant="default",
                )
            prefill_text = step._prefill_status_text()
            if prefill_text:
                # NEW-1: persisted values may contain literal "[...]".
                yield Static(
                    prefill_text,
                    id="setup-speech-prefill",
                    classes="setup-subtitle",
                    markup=False,
                )
            yield Label("Language", classes="setup-field-label")
            yield _language_select(step)
            status_text, action_widget = step._status_and_action()
            yield Static(
                status_text, id="setup-speech-status", classes="setup-subtitle", markup=False
            )
            if not step._runtime_installed():
                with Horizontal(classes="setup-rag-extras-row"):
                    yield Static(engines.PARAKEET_EXTRAS_PACKAGE, markup=False)
                    yield Button(
                        "Copy package name",
                        id="setup-speech-copy-extras",
                        classes="setup-tertiary-button",
                    )
            progress = ModelInstallProgress(
                step._progress, id="setup-speech-install-progress", cancellable=True
            )
            progress.display = step._operation == "install" and step._progress is not None
            yield progress
            if action_widget is not None:
                yield action_widget
            if step._use_as_default_offer():
                # Review NEW-2: installed + active + configured elsewhere.
                yield Button(
                    f"Use {escape_markup(step._model_label())} as my default",
                    id="setup-speech-use-as-default",
                    variant="primary",
                )
            action_disabled = action_widget is None or bool(
                getattr(action_widget, "disabled", False)
            )
            with Collapsible(
                title="Advanced",
                collapsed=not step._advanced_open(),
                id="setup-speech-advanced",
            ):
                yield from _advanced_block(step, action_disabled)
