"""The first-run Voice step's Service radio (TASK-34100.8).

Five segments: "No voice for now" leads, then the four services. They share
one row in proportion to each label's width (``_wizards.tcss``), so every
label fits wherever the row has room for all five. Review round 2 (R2-F3 /
G8-R2-F4): at 80x24, a supported size, one row offers about 64 cells to
labels that need 71, and every label clipped ("No voice for …",
"PocketTT…"). There the radio wraps to two rows instead (the ``-stacked``
class), keeping each label whole.
"""

from __future__ import annotations

from textual import events
from textual.widgets import RadioButton

from tldw_chatbook.UI.Wizards import first_run_voice_step_state as voice_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import SetupRadioSet

#: Service radio: (button id, label, preset). "No voice for now" leads.
SERVICES = (
    ("setup-voice-preset-none", "No voice for now", voice_state.VOICE_PRESET_NONE),
    ("setup-voice-preset-pocket", "PocketTTS", voice_state.VOICE_PRESET_POCKET_TTS),
    ("setup-voice-preset-official", "OpenAI", voice_state.VOICE_PRESET_OFFICIAL_OPENAI),
    ("setup-voice-preset-custom", "Custom", voice_state.VOICE_PRESET_CUSTOM),
    ("setup-voice-preset-omnivoice", "OmniVoice", voice_state.VOICE_PRESET_OMNIVOICE),
)
PRESET_BY_BUTTON = {button_id: preset for button_id, _label, preset in SERVICES}
BUTTON_BY_PRESET = {preset: button_id for button_id, _label, preset in SERVICES}
#: Set while one row would clip a label; the stylesheet wraps the radio.
STACKED_CLASS = "-stacked"


class VoiceServiceRadioSet(SetupRadioSet):
    """The Service radio: one row when every label fits, two when not."""

    def on_mount(self) -> None:
        # A step mounted already visible lays this out without ever sending
        # it a Resize (measured), so measure once the first layout is done.
        self.call_after_refresh(self.fit_labels)

    def on_show(self) -> None:
        self.call_after_refresh(self.fit_labels)

    def on_resize(self, _event: events.Resize) -> None:
        self.fit_labels()

    def fit_labels(self) -> None:
        """Wrap to two rows exactly when one row is narrower than the labels.

        It cannot flip back and forth: wrapping changes the radio's height,
        never its width, and a taller step can only gain a scrollbar, which
        narrows the row further, so a wrapped row never measures wide enough
        to unwrap at the same terminal size.
        """
        needed = sum(
            button.get_content_width(button.size, self.app.size)
            for button in self.query(RadioButton)
        )
        self.set_class(0 < self.content_size.width < needed, STACKED_CLASS)


__all__ = [
    "BUTTON_BY_PRESET",
    "PRESET_BY_BUTTON",
    "SERVICES",
    "STACKED_CLASS",
    "VoiceServiceRadioSet",
]
