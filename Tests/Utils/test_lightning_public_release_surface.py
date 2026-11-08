"""Public metadata and error projections retain the categorical deferral."""

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_release_extras_omit_lightning_without_removing_legacy_parakeet_bundle():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert "transcription_lightning_whisper" not in extras
    assert not any(
        "lightning-whisper-mlx" in item for items in extras.values() for item in items
    )
    assert "parakeet-mlx; sys_platform == 'darwin'" in extras["mlx_whisper"]
    assert "tiktoken==0.14.0" in project["dependencies"]


def test_public_feature_metadata_has_no_deferred_install_recipe():
    from tldw_chatbook.Utils.optional_deps import OPTIONAL_FEATURES

    assert "transcription_lightning_whisper" not in OPTIONAL_FEATURES
    assert OPTIONAL_FEATURES["mlx_whisper"].package_dependencies == ("parakeet-mlx",)


def test_audio_inventory_refuses_stale_lightning_without_probing_it(monkeypatch):
    from tldw_chatbook.Utils import optional_deps as deps

    monkeypatch.setattr(deps, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(
        deps, "DEPENDENCIES_AVAILABLE", dict(deps.DEPENDENCIES_AVAILABLE)
    )
    monkeypatch.setitem(deps.DEPENDENCIES_AVAILABLE, "lightning_whisper_mlx", True)
    monkeypatch.setattr(deps, "check_dependency", lambda *args: True)

    def installed(name):
        assert name != "lightning_whisper_mlx", "deferred provider must not be probed"
        return name == "parakeet_mlx"

    monkeypatch.setattr(deps, "_check_dependency_installed", installed)
    assert deps.check_audio_processing_deps() is True
    assert deps.DEPENDENCIES_AVAILABLE["lightning_whisper_mlx"] is False
    assert deps.DEPENDENCIES_AVAILABLE["parakeet_mlx"] is True


def test_refreshed_speech_status_is_not_ready_with_only_deferred_lightning(monkeypatch):
    from tldw_chatbook.UI.Lab_Modules import lab_speech_status as status

    monkeypatch.setattr(
        status,
        "_speech_dependency_installed",
        lambda name: name == "lightning_whisper_mlx",
    )
    assert status.speech_local_dependency_availability(refresh=True).stt is False


@pytest.mark.parametrize("provider", ["lightning-whisper-mlx", "lightning-whisper"])
def test_library_field_preserves_complete_deferred_selection_copy(provider):
    from tldw_chatbook.Library.ingest_capabilities import OptionField
    from tldw_chatbook.Library.library_ingest_state import validate_ingest_option_value

    field = OptionField("transcription_provider", "Provider", "select")
    message = validate_ingest_option_value(field, provider)
    assert "Lightning Whisper MLX is deferred for Chatbook 0.2.4." in message
    assert "Your saved settings have not been changed." in message


@pytest.mark.parametrize("provider", ["lightning-whisper-mlx", "lightning-whisper"])
def test_stale_lightning_install_warning_is_suppressed_without_erasing_other_evidence(
    provider,
):
    from tldw_chatbook.Library.ingest_capabilities import selected_stt_warnings

    warnings = [
        {"feature": "lightning_whisper_mlx", "command": "obsolete install hint"},
        {"feature": "soundfile", "command": "retained prerequisite"},
    ]
    assert selected_stt_warnings(warnings, provider) == [warnings[1]]
    assert len(warnings) == 2
