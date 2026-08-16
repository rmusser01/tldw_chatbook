"""TASK-16841: `VoiceCloningWindow`'s `#test-profile-select` is backwards.

Found by the repo-wide AST sweep for TASK-16841. `_update_profile_display`
(`UI/Voice_Cloning_Window.py::_update_profile_display`) built the Voice
Testing tab's profile selector with:

    test_options.append((profile["name"], profile["display_name"]))

-- `(machine id, human label)` order, backwards against Textual's
`(label, value)` contract. `profile["name"]` is the profile's machine
identifier (used elsewhere in this same method as
`self.selected_profile = profile["name"]`, and by the profile-management
table's first column); `profile["display_name"]` is the human-facing label
shown in the "Display Name" field of `VoiceProfileDialog`.

The consumer, `_test_generate_voice`, reads
``test_profile = self.query_one("#test-profile-select", Select).value``
and sends ``voice = f"profile:{test_profile}"`` straight to the TTS
backend -- so with the tuples reversed, selecting a profile by its name in
the dropdown (which, backwards, DISPLAYED the internal name, not the
display name) would send `profile:{display_name}` instead of
`profile:{name}`, silently generating audio with the wrong voice (or none,
if no profile happens to share its name and display_name).

No `InvalidSelectValueError` here (the Select starts with `options=[]` and
no explicit initial `value=`), so this bug produced wrong BEHAVIOR, not a
crash -- confirmed directly by asserting what the dropdown shows and what
`.value` resolves to, and by driving the exact `_test_generate_voice` call
site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = str(BUNDLE)

    def compose(self) -> ComposeResult:
        yield VoiceCloningWindow()


_PROFILES: list[dict[str, Any]] = [
    {
        "name": "narrator_1",
        "display_name": "Narrator One",
        "language": "en",
        "created_at": "2026-01-01T00:00:00",
        "tags": [],
    },
    {
        "name": "villain_2",
        "display_name": "Villain Two",
        "language": "en",
        "created_at": "2026-01-01T00:00:00",
        "tags": [],
    },
]


@pytest.mark.asyncio
async def test_profile_select_shows_display_name_and_values_the_machine_id() -> None:
    """AC born-red: the dropdown must render display_name and value name."""
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        widget = app.query_one(VoiceCloningWindow)
        # `on_mount` schedules a one-shot `self.set_timer(0.1, self._load_profiles)`
        # against the (empty, real) backend directories in the isolated test
        # home. Let that fire and finish FIRST so this test's direct call
        # below is the last write and its result is what sticks -- otherwise
        # the timer's empty-profile update races this call and can clobber it.
        await pilot.pause(0.2)
        await app.workers.wait_for_complete()

        widget._update_profile_display(_PROFILES)
        await pilot.pause()

        test_select = widget.query_one("#test-profile-select", Select)
        # Textual's SelectType is opaque about ordering at the API surface,
        # so read the same private option list `_test_generate_voice`'s
        # `.value` reads are validated against. Index 0 is the blank
        # placeholder Select always injects when `allow_blank` (the
        # default here).
        rendered_labels = [str(prompt) for prompt, _value in test_select._options[1:]]
        option_values = [value for _prompt, value in test_select._options[1:]]

        assert rendered_labels == ["Narrator One", "Villain Two"]
        assert option_values == ["narrator_1", "villain_2"]

        test_select.value = "narrator_1"
        assert test_select.value == "narrator_1"


@pytest.mark.asyncio
async def test_test_generate_voice_sends_the_profile_name_not_the_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drives the real `_test_generate_voice` call site end-to-end."""
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        widget = app.query_one(VoiceCloningWindow)
        await pilot.pause(0.2)
        await app.workers.wait_for_complete()

        widget._update_profile_display(_PROFILES)
        await pilot.pause()

        test_select = widget.query_one("#test-profile-select", Select)
        test_select.value = "villain_2"

        sent_events: list[Any] = []
        monkeypatch.setattr(app, "post_message", lambda event: sent_events.append(event))

        await widget._test_generate_voice()

        assert len(sent_events) == 1
        request = sent_events[0].request
        assert request.voice_id == "profile:villain_2"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
