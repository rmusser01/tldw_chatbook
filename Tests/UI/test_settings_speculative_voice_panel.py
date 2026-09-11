"""Canonical speculative-pipeline settings ownership and UI contracts."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static, Switch

from tldw_chatbook.UI.Screens.settings_speech_tts import (
    GlobalSpeechTTSValidationError,
    load_global_speech_tts_state,
)
from tldw_chatbook.Widgets.Settings_Widgets import (
    speech_tts_settings_panel as panel_module,
)
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSSettingsPanel,
)


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield SpeechTTSSettingsPanel(
            state=load_global_speech_tts_state({}),
            id="panel",
        )


@pytest.fixture
def pipeline_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    values = {
        ("dictation", "response_eagerness_ms"): 700,
        ("dictation", "pipeline_aec_enabled"): True,
    }

    def get_setting(section: str, key: str, default: object) -> object:
        return values.get((section, key), default)

    monkeypatch.setattr(panel_module, "get_cli_setting", get_setting)
    yield


def _panel_copy(panel: SpeechTTSSettingsPanel) -> str:
    return "\n".join(str(widget.render()) for widget in panel.query(Static))


@pytest.mark.asyncio
async def test_pipeline_block_is_canonical_plain_language_ui(
    pipeline_config: None,
) -> None:
    async with _Harness().run_test(size=(190, 80)) as pilot:
        await pilot.pause()
        panel = pilot.app.query_one("#panel", SpeechTTSSettingsPanel)
        copy = _panel_copy(panel)

        assert "Pipeline conversation" in copy
        assert "Native live" in copy
        assert "rolling-window fallback" in copy
        assert "overlapping audio" in copy
        assert "discarded" in copy
        assert "Fast: 700 ms" in copy
        assert "Balanced: 1200 ms" in copy
        assert "Deliberate: 2000 ms" in copy
        assert "500-3000 ms" in copy
        assert "troubleshooting only" in copy.lower()
        assert "half duplex" in copy.lower()
        assert "qualification gate" not in copy.lower()
        assert (
            panel.query_one(
                "#settings-speech-pipeline-response-eagerness-ms", Input
            ).value
            == "700"
        )
        assert (
            panel.query_one("#settings-speech-pipeline-aec-enabled", Switch).value
            is True
        )


@pytest.mark.asyncio
async def test_opening_pipeline_settings_does_not_write_config(
    pipeline_config: None,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[dictation]\nresponse_eagerness_ms = 700\npipeline_aec_enabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    before = config_path.read_bytes()
    writes: list[object] = []
    monkeypatch.setattr(
        panel_module,
        "save_settings_to_cli_config",
        lambda *args, **kwargs: writes.append((args, kwargs)) or True,
    )

    async with _Harness().run_test(size=(190, 80)) as pilot:
        await pilot.pause()

    assert writes == []
    assert config_path.read_bytes() == before


@pytest.mark.asyncio
async def test_named_response_presets_update_the_numeric_draft(
    pipeline_config: None,
) -> None:
    async with _Harness().run_test(size=(190, 80)) as pilot:
        field = pilot.app.query_one(
            "#settings-speech-pipeline-response-eagerness-ms", Input
        )
        for preset, expected in (
            ("fast", "700"),
            ("balanced", "1200"),
            ("deliberate", "2000"),
        ):
            pilot.app.query_one(f"#settings-speech-pipeline-preset-{preset}").press()
            await pilot.pause()
            assert field.value == expected


@pytest.mark.asyncio
async def test_pipeline_only_change_writes_only_canonical_pipeline_keys(
    pipeline_config: None,
) -> None:
    async with _Harness().run_test(size=(190, 80)) as pilot:
        await pilot.pause()
        panel = pilot.app.query_one("#panel", SpeechTTSSettingsPanel)
        panel.query_one(
            "#settings-speech-pipeline-response-eagerness-ms", Input
        ).value = "1200"
        panel.query_one("#settings-speech-pipeline-aec-enabled", Switch).value = False
        panel._collect_visible_state()

        payload = panel._validated_realtime_payload()

        assert payload is not None
        assert payload.section_values == {
            "dictation": {
                "response_eagerness_ms": 1200,
                "pipeline_aec_enabled": False,
            }
        }
        assert payload.delete_keys == {}
        assert payload.persisted_pipeline_draft.snapshot() == ("1200", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["499", "3001", "700.5", "words"])
async def test_pipeline_eagerness_validation_is_bounded(
    pipeline_config: None,
    value: str,
) -> None:
    async with _Harness().run_test(size=(190, 80)) as pilot:
        await pilot.pause()
        panel = pilot.app.query_one("#panel", SpeechTTSSettingsPanel)
        panel.query_one(
            "#settings-speech-pipeline-response-eagerness-ms", Input
        ).value = value
        panel._collect_visible_state()

        with pytest.raises(GlobalSpeechTTSValidationError) as caught:
            panel._validated_realtime_payload()

        assert caught.value.provider_id == "pipeline"
        assert caught.value.field_id == "response_eagerness_ms"


@pytest.mark.asyncio
async def test_pipeline_draft_round_trips_through_panel_snapshot(
    pipeline_config: None,
) -> None:
    async with _Harness().run_test(size=(190, 80)) as pilot:
        await pilot.pause()
        panel = pilot.app.query_one("#panel", SpeechTTSSettingsPanel)
        panel.query_one(
            "#settings-speech-pipeline-response-eagerness-ms", Input
        ).value = "2000"
        panel.query_one("#settings-speech-pipeline-aec-enabled", Switch).value = False

        snapshot = panel.draft_snapshot()

        assert snapshot.pipeline_voice_draft.snapshot() == ("2000", False)
        assert panel.has_unsaved_changes() is True
