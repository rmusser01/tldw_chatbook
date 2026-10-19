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


def _engine_block(step: "SpeechSetupStep") -> Iterator[Widget]:
    choices = engines.engine_choices(step._engine_facts)
    yield Label("Dictation engine", classes="setup-field-label")
    with SetupRadioSet(
        id="setup-speech-engine", classes="setup-choice-list setup-choice-fit"
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
