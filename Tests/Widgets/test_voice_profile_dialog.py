"""TASK-16841: `VoiceProfileDialog`'s `#language-select` is backwards.

Found by the repo-wide AST sweep for TASK-16841 (the backwards-Select bug
class fixed piecemeal in TASK-15772/TASK-15991/TASK-16841's own
`#auth-type-select`). `Widgets/voice_profile_dialog.py::compose` built
`#language-select` with ``(value, label)``-ordered option tuples --
``("en", "English"), ("es", "Spanish"), ...`` -- backwards against Textual's
``(label, value)`` contract, AND passed ``value=self.profile_data.get(
"language") or "en"`` as the Select's initial value. Since the Select's
*real* legal values were the display labels ("English", "Spanish", ...),
mounting the dialog with the default "en" (i.e. every "New Voice Profile"
open, not just edits of a saved profile) raised
``InvalidSelectValueError`` from `Select._on_mount` -> `_init_selected_
option` -> the reactive `value` setter.

This dialog is reachable: Lab > Speech > Voice Cloning ("view-voice-cloning-
btn" in `UI/STTS_Window.py`) > "New Profile" -> `Voice_Cloning_Window.
_create_new_profile` picks a reference audio file, then pushes
`VoiceProfileDialog(str(path), on_submit=handle_profile_data)`
(`UI/Voice_Cloning_Window.py:514-515`) -- so opening this dialog at all
would have crashed.

The consumer confirms the intended direction: `on_button_pressed` reads
``language = self.query_one("#language-select", Select).value`` and passes
it straight through as ``profile_data["language"]`` to
`VoiceBackendManager.create_profile(..., language=...)`
(`Voice_Cloning_Window._create_new_profile`'s `handle_profile_data`) -- a
machine-readable language code, not the English display name.

Born red at HEAD: pushing the dialog with default `profile_data` (the
"New Voice Profile" path) raised `InvalidSelectValueError`. Green once the
tuples are swapped to `(label, value)`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Select
from textual.widgets._select import InvalidSelectValueError

from tldw_chatbook.Widgets.voice_profile_dialog import VoiceProfileDialog

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = str(BUNDLE)


@pytest.mark.asyncio
async def test_new_profile_dialog_mounts_without_raising() -> None:
    """AC born-red: the default "New Voice Profile" open must not crash.

    `profile_data` is None/{} here -- the exact path `_create_new_profile`
    takes -- so the Select's initial value falls back to `"en"`.
    """
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(VoiceProfileDialog("reference.wav"))
        await pilot.pause()

        dialog = app.screen
        assert isinstance(dialog, VoiceProfileDialog)
        language_select = dialog.query_one("#language-select", Select)
        assert language_select.value == "en"


@pytest.mark.asyncio
async def test_editing_a_profile_restores_its_saved_language() -> None:
    """A saved profile's language code round-trips through the Select."""
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(
            VoiceProfileDialog(
                "reference.wav",
                profile_data={
                    "name": "narrator_1",
                    "display_name": "Narrator One",
                    "language": "fr",
                },
            )
        )
        await pilot.pause()

        dialog = app.screen
        assert isinstance(dialog, VoiceProfileDialog)
        assert dialog.query_one("#language-select", Select).value == "fr"


@pytest.mark.asyncio
async def test_display_labels_are_not_valid_select_values() -> None:
    """Pins the fix's direction: labels must not be legal `.value`s."""
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(VoiceProfileDialog("reference.wav"))
        await pilot.pause()

        dialog = app.screen
        assert isinstance(dialog, VoiceProfileDialog)
        language_select = dialog.query_one("#language-select", Select)
        with pytest.raises(InvalidSelectValueError):
            language_select.value = "English"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
