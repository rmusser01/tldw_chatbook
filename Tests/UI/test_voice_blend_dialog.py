"""Direct function tests for voice-blend dialog behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.Widgets.voice_blend_dialog import VoiceBlendDialog
from tldw_chatbook.UI.Speech import speech_settings_mixin
from tldw_chatbook.UI.Speech import speech_settings_pane


def _voice_entry(index: int, voice: str, weight: str):
    fields = {
        f"#voice-select-{index}": SimpleNamespace(value=voice),
        f"#weight-input-{index}": SimpleNamespace(value=weight),
    }
    return SimpleNamespace(
        index=index,
        query_one=Mock(side_effect=lambda selector, *_args: fields[selector]),
    )


def _dialog_like(*, name="", description="", entries=()):
    fields = {
        "#blend-name-input": SimpleNamespace(value=name),
        "#blend-description-input": SimpleNamespace(value=description),
    }
    return SimpleNamespace(
        query_one=Mock(side_effect=lambda selector, *_args: fields[selector]),
        voice_entries=list(entries),
        app=SimpleNamespace(notify=Mock()),
        dismiss=Mock(),
    )


def test_voice_destinations_and_blend_actions_have_unambiguous_labels() -> None:
    destination_actions = getattr(
        speech_settings_pane,
        "VOICE_DESTINATION_ACTIONS",
        (),
    )
    blend_actions = getattr(speech_settings_pane, "VOICE_BLEND_ACTIONS", ())

    assert [(action.id, action.label) for action in destination_actions] == [
        ("voice-profiles", "Voice Profiles"),
        ("voice-blends", "Voice Blends"),
    ]
    assert [(action.id, action.label) for action in blend_actions] == [
        ("add-voice-blend-btn", "Add Voice Blend"),
        ("import-blends-btn", "Import Voice Blends"),
        ("export-blends-btn", "Export Voice Blends"),
    ]


def test_non_kokoro_provider_invalidates_a_blend_selection() -> None:
    normalize = getattr(
        speech_settings_mixin,
        "normalize_provider_voice_selection",
        None,
    )

    assert callable(normalize), "Voice selection needs provider-scoped validation"
    assert normalize("openai", "blend:duet", ("alloy", "nova")) == "alloy"
    assert normalize("kokoro", "blend:duet", ("af_bella", "blend:duet")) == (
        "blend:duet"
    )


def test_voice_blend_dialog_create_normalizes_result():
    dialog = _dialog_like(
        name="My Test Blend",
        description="A test voice blend",
        entries=(_voice_entry(0, "af_bella", "1.0"),),
    )

    VoiceBlendDialog.save_blend(dialog)

    dialog.dismiss.assert_called_once_with(
        {
            "name": "My Test Blend",
            "description": "A test voice blend",
            "voices": [("af_bella", 1.0)],
            "metadata": {"created_by": "TUI", "normalized": True},
        }
    )


def test_voice_blend_dialog_normalizes_multiple_voice_weights():
    dialog = _dialog_like(
        name="Pair",
        entries=(
            _voice_entry(0, "af_bella", "1.5"),
            _voice_entry(1, "am_michael", "0.5"),
        ),
    )

    VoiceBlendDialog.save_blend(dialog)

    result = dialog.dismiss.call_args.args[0]
    assert result["voices"] == [("af_bella", 0.75), ("am_michael", 0.25)]


@pytest.mark.parametrize(
    ("name", "weight", "message"),
    [
        ("", "1.0", "Blend name is required"),
        ("Blend", "invalid", "Invalid weight value"),
        ("Blend", "0", "All weights must be positive"),
    ],
)
def test_voice_blend_dialog_validation(name, weight, message):
    dialog = _dialog_like(
        name=name,
        entries=(_voice_entry(0, "af_bella", weight),),
    )

    VoiceBlendDialog.save_blend(dialog)

    dialog.dismiss.assert_not_called()
    dialog.app.notify.assert_called_once_with(message, severity="error")


@pytest.mark.asyncio
async def test_add_voice_entry_tracks_and_mounts_real_entry():
    voice_list = SimpleNamespace(mount=AsyncMock())
    dialog = SimpleNamespace(
        voice_entries=[],
        next_index=0,
        query_one=Mock(return_value=voice_list),
    )

    await VoiceBlendDialog.add_voice_entry(dialog, "am_michael", 0.5)

    assert len(dialog.voice_entries) == 1
    assert dialog.voice_entries[0].initial_voice == "am_michael"
    assert dialog.voice_entries[0].initial_weight == 0.5
    assert dialog.next_index == 1
    voice_list.mount.assert_awaited_once_with(dialog.voice_entries[0])


@pytest.mark.asyncio
async def test_remove_voice_entry_keeps_at_least_one():
    first = SimpleNamespace(remove=AsyncMock())
    second = SimpleNamespace(remove=AsyncMock())
    dialog = SimpleNamespace(voice_entries=[first, second])

    await VoiceBlendDialog.on_voice_blend_entry_removed(
        dialog,
        SimpleNamespace(entry=first),
    )
    await VoiceBlendDialog.on_voice_blend_entry_removed(
        dialog,
        SimpleNamespace(entry=second),
    )

    assert dialog.voice_entries == [second]
    first.remove.assert_awaited_once_with()
    second.remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_button_dismisses_without_result():
    dialog = SimpleNamespace(
        add_voice_entry=AsyncMock(),
        save_blend=Mock(),
        dismiss=Mock(),
    )

    await VoiceBlendDialog.on_button_pressed(
        dialog,
        SimpleNamespace(button=SimpleNamespace(id="cancel-btn")),
    )

    dialog.dismiss.assert_called_once_with(None)
