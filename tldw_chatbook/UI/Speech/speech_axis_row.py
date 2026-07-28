"""The comparison axes, as one always-visible chip row.

These are the six variables a user changes to compare options, so they never
collapse and never move: provider, model, voice, language, format, speed.
Everything else about a provider is a tuning knob and lives in the collapsed
group beside them.

The Playground owns session-scoped values that never write back to persisted
defaults, so an override has to be *visible*. A control that looks identical
whether or not it differs from the saved default makes deliberate variation
indistinguishable from configuration -- which is the one thing this screen
exists to let a user see.

These are **controls, not chips**. An earlier pass rendered them as `Static`
chips borrowed from Console's status bar, which is a genuine read-only
status strip -- but the axes are the variables a user changes to compare, so
read-only text made the screen unusable for its own purpose. They carry the
legacy control ids (`tts-provider-select` and friends), which is also what
lets the existing synthesis handler read them unchanged.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal
from textual.widgets import Input, Select, Static

from .speech_playground_model import AXIS_CONTROLS

#: Chip label per axis. Keys are exactly :data:`AXIS_CONTROLS`.
#: What an axis says when it has options but none is chosen yet.
AXIS_PROMPTS: dict[str, str] = {
    "tts-provider-select": "Choose a provider",
    "tts-model-select": "Choose a model",
    "tts-voice-select": "Choose a voice",
    "tts-language-select": "Choose a language",
    "tts-format-select": "Choose a format",
    "tts-speed-input": "",
}

#: What it says when the selected provider offers nothing for that axis.
#: Textual's default prompt is the bare word "Select", which on a provider
#: with no languages reads as an instruction the user cannot follow.
AXIS_EMPTY_PROMPTS: dict[str, str] = {
    "tts-provider-select": "No providers available",
    "tts-model-select": "No models for this provider",
    "tts-voice-select": "No voices for this model",
    "tts-language-select": "Not used by this provider",
    "tts-format-select": "No formats available",
    "tts-speed-input": "",
}

AXIS_LABELS: dict[str, str] = {
    "tts-provider-select": "Provider",
    "tts-model-select": "Model",
    "tts-voice-select": "Voice",
    "tts-language-select": "Language",
    "tts-format-select": "Format",
    "tts-speed-input": "Speed",
}

#: Shown when an axis has no value yet. Not blank: a chip that renders as
#: `"Voice: "` reads as broken rather than unset.
UNSET_VALUE = "unset"

#: Appended to an overridden chip. Carried in the text, not only the colour,
#: because colour is not available to every reader.
OVERRIDE_MARKER = " *"


def axis_chip_id(axis: str) -> str:
    """Return the stable chip id for one axis.

    Args:
        axis: The axis control id, e.g. ``"tts-voice-select"``.

    Returns:
        ``"speech-axis-<axis>"``.
    """
    return f"speech-axis-{axis}"


class SpeechAxisRow(Grid):
    """The comparison axes as chips, reflowed rather than truncated.

    A single Horizontal could not hold six chips: at 120 columns the Lab
    body is ~81 cells and the chips need more, so "Format" truncated and
    "Speed" fell off the right edge entirely. A Grid wraps them onto a
    second line instead -- the axes are the one thing on this screen that
    must never be cut off, since they are what the user is comparing.
    """

    def __init__(
        self,
        *,
        values: dict[str, str],
        defaults: dict[str, str],
        options: dict[str, tuple[tuple[str, str], ...]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the row.

        Args:
            values: Effective value per axis for this session.
            defaults: Persisted default per axis, used only to detect
                overrides. This widget never writes to it.
            options: Selectable ``(label, value)`` pairs per axis. Empty for
                an axis whose catalog has not loaded yet, which renders as a
                blank Select rather than omitting the control.
            kwargs: Forwarded to ``Horizontal``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-axis-grid {classes}".strip(), **kwargs)
        self.values = dict(values)
        self.defaults = dict(defaults)
        self.options = dict(options or {})

    def is_override(self, axis: str) -> bool:
        """Report whether this axis differs from its persisted default.

        An axis with no saved default is **not** an override. Marking it
        would flag every axis on a first run, when nothing has been
        configured and nothing has been deliberately changed.

        Args:
            axis: The axis control id.

        Returns:
            True only when a default exists and the effective value differs.
        """
        if axis not in self.defaults:
            return False
        return self.values.get(axis) != self.defaults[axis]

    def _control_for(self, axis: str):
        """Build the editable control for one axis.

        Args:
            axis: The axis control id.

        Returns:
            An ``Input`` for speed (a free number) or a ``Select`` for the
            rest. The id is the legacy control id, unchanged, so the
            existing synthesis handler reads it without translation.
        """
        if axis == "tts-speed-input":
            return Input(
                value=self.values.get(axis, ""),
                id=axis,
                classes="speech-axis-control",
            )
        options = self.options.get(axis, ())
        current = self.values.get(axis)
        select: Select[str] = Select(
            [(label, value) for label, value in options],
            id=axis,
            classes="speech-axis-control",
            allow_blank=True,
            prompt=AXIS_PROMPTS[axis] if options else AXIS_EMPTY_PROMPTS[axis],
        )
        # Only set a value we can honour. `Select.BLANK` is itself falsy and
        # passing it explicitly is rejected as an illegal value; leaving the
        # default (`Select.NULL`) is how "nothing selected yet" is expressed,
        # which is the normal state before a provider catalog has loaded.
        if current is not None and current in {value for _, value in options}:
            select.value = current
        return select

    def compose(self) -> ComposeResult:
        """Yield a labelled, editable control per axis, marking overrides."""
        for axis in AXIS_CONTROLS:
            override = self.is_override(axis)
            with Horizontal(classes="speech-axis-cell"):
                label = Static(
                    AXIS_LABELS[axis] + (OVERRIDE_MARKER if override else ""),
                    id=axis_chip_id(axis),
                    classes="speech-chip"
                    + (" speech-chip-override" if override else ""),
                    markup=False,
                )
                label.tooltip = (
                    "Session override — saved default is "
                    f"{self.defaults.get(axis, UNSET_VALUE)}"
                    if override
                    else "Matches the saved default "
                    f"({self.defaults.get(axis, UNSET_VALUE)})"
                )
                yield label
                yield self._control_for(axis)
