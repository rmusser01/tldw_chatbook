"""Provider tuning knobs, collapsed and scoped to the selected provider.

Only the selected provider's parameters are *mounted*, not merely hidden:
a hidden control is still in the DOM and still in the focus chain, so a
keyboard user tabbing through a Chatterbox session would land on ElevenLabs'
stability field.

Collapsed by default because these are set once per provider. Expanded they
would push the text input and the primary action back down the page, which
is the defect the redesign exists to fix -- the legacy form was 93 rows with
`Generate` at y=60.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Collapsible, Input, Static, Switch

from .speech_playground_model import params_for_provider

#: Human label per parameter id. Every id `params_for_provider` can return
#: must appear here -- a control rendered with a bare id is not a control a
#: user can reason about, and a test asserts the coverage.
PARAM_LABELS: dict[str, str] = {
    # ElevenLabs
    "tts-stability-input": "Stability",
    "tts-similarity-input": "Similarity",
    "tts-style-input": "Style",
    "tts-speaker-boost-switch": "Speaker boost",
    # Chatterbox
    "tts-exaggeration-input": "Exaggeration",
    "tts-cfg-weight-input": "CFG weight",
    "tts-temperature-input": "Temperature",
    "tts-num-candidates-input": "Candidates",
    "tts-validate-whisper-switch": "Validate with Whisper",
    "tts-random-seed-input": "Seed",
    # Higgs
    "tts-higgs-temperature-input": "Temperature",
    "tts-higgs-top-p-input": "Top-p",
    "tts-higgs-repetition-penalty-input": "Repetition penalty",
    "tts-higgs-voice-cloning-switch": "Voice cloning",
    "tts-higgs-multi-speaker-switch": "Multi-speaker",
    "tts-higgs-delimiter-input": "Delimiter",
    # Kokoro
    "tts-kokoro-use-onnx": "Use ONNX",
    # Shared audio post-processing
    "tts-preprocess-text-switch": "Preprocess text",
    "tts-normalize-audio-switch": "Normalize audio",
    "tts-target-db-input": "Target dB",
    # Shared per-request options
    "tts-download-format-select": "Download format",
    "tts-norm-unit-switch": "Expand units",
    "tts-norm-url-switch": "Expand URLs",
    "tts-norm-email-switch": "Expand emails",
    "tts-norm-plural-switch": "Expand plurals",
    "tts-norm-phone-switch": "Expand phone numbers",
}

#: Suffixes that mean the control is a boolean rather than a text field.
_SWITCH_SUFFIXES = ("-switch", "-onnx")


def _is_switch(param: str) -> bool:
    """Report whether a parameter renders as a Switch rather than an Input.

    Args:
        param: The parameter control id.

    Returns:
        True for boolean parameters.
    """
    return param.endswith(_SWITCH_SUFFIXES)


def _param_rows(provider: str) -> list[Horizontal]:
    """Build one labelled row per parameter this provider has.

    Args:
        provider: The selected provider key.

    Returns:
        The rows, ready to pass to ``Collapsible`` as children.
    """
    rows: list[Horizontal] = []
    for param in params_for_provider(provider):
        control = (
            Switch(id=param, classes="speech-param-control")
            if _is_switch(param)
            else Input(id=param, classes="speech-param-control")
        )
        rows.append(
            Horizontal(
                Static(
                    PARAM_LABELS.get(param, param),
                    classes="speech-param-label",
                    markup=False,
                ),
                control,
                classes="speech-param-row",
            )
        )
    return rows


class SpeechParamGroup(Collapsible):
    """The selected provider's tuning knobs, collapsed by default.

    Children are passed to ``Collapsible`` rather than yielded from an
    overridden ``compose()``. Overriding it replaces Collapsible's own
    composition -- the title row and the contents container it toggles -- so
    the group rendered fully expanded with no title while still reporting
    `collapsed is True`. The attribute was right and the screen was wrong,
    which is why the test now asserts what renders.
    """

    def __init__(self, *, provider: str, **kwargs: Any) -> None:
        """Create the group.

        Args:
            provider: The selected provider key, e.g. ``"chatterbox"``.
            kwargs: Forwarded to ``Collapsible``.
        """
        self.provider = provider
        kwargs.setdefault("title", f"{provider} parameters")
        kwargs.setdefault("collapsed", True)
        classes = kwargs.pop("classes", "")
        super().__init__(
            *_param_rows(provider),
            classes=f"speech-param-group {classes}".strip(),
            **kwargs,
        )
