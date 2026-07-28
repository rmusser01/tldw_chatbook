"""The comparison axes, as one always-visible chip row.

These are the six variables a user changes to compare options, so they never
collapse and never move: provider, model, voice, language, format, speed.
Everything else about a provider is a tuning knob and lives in the collapsed
group beside them.

The Playground owns session-scoped values that never write back to persisted
defaults, so an override has to be *visible*. A chip that looks identical
whether or not it differs from the saved default makes deliberate variation
indistinguishable from configuration -- which is the one thing this screen
exists to let a user see.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from .speech_playground_model import AXIS_CONTROLS

#: Chip label per axis. Keys are exactly :data:`AXIS_CONTROLS`.
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


class SpeechAxisRow(Horizontal):
    """One row of ``Label: value`` chips for the comparison axes."""

    def __init__(
        self,
        *,
        values: dict[str, str],
        defaults: dict[str, str],
        **kwargs: Any,
    ) -> None:
        """Create the row.

        Args:
            values: Effective value per axis for this session.
            defaults: Persisted default per axis, used only to detect
                overrides. This widget never writes to it.
            kwargs: Forwarded to ``Horizontal``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-chip-row {classes}".strip(), **kwargs)
        self.values = dict(values)
        self.defaults = dict(defaults)

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

    def compose(self) -> ComposeResult:
        """Yield one chip per axis, in comparison order, marking overrides."""
        for axis in AXIS_CONTROLS:
            label = AXIS_LABELS[axis]
            value = self.values.get(axis, UNSET_VALUE)
            override = self.is_override(axis)

            chip = Static(
                f"{label}: {value}" + (OVERRIDE_MARKER if override else ""),
                id=axis_chip_id(axis),
                classes="speech-chip" + (" speech-chip-override" if override else ""),
                markup=False,
            )
            chip.tooltip = (
                f"Session override — saved default is "
                f"{self.defaults.get(axis, UNSET_VALUE)}"
                if override
                else f"Matches the saved default ({self.defaults.get(axis, UNSET_VALUE)})"
            )
            yield chip
