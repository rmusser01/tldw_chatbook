"""Release availability never grants effects from a retained local identity."""

import sys
from types import ModuleType

import pytest

from tldw_chatbook.Utils import local_stt_providers as providers


@pytest.mark.parametrize("provider", ["lightning-whisper-mlx", "lightning-whisper"])
@pytest.mark.parametrize("preloaded", [False, True])
def test_deferred_lightning_is_unavailable_before_any_package_probe(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    preloaded: bool,
) -> None:
    monkeypatch.setattr(providers.sys, "platform", "darwin")
    if preloaded:
        monkeypatch.setitem(
            sys.modules, "lightning_whisper_mlx", ModuleType("lightning_whisper_mlx")
        )
    monkeypatch.setattr(
        providers,
        "module_installed",
        lambda name: pytest.fail(f"deferred provider probed package {name}"),
    )

    assert providers.provider_installed(provider) is False
    canonical = providers.normalize_provider_id(provider)
    assert canonical in providers.LOCAL_STT_PROVIDERS
    assert providers.provider_is_local(canonical)


@pytest.mark.parametrize("provider", ["lightning-whisper-mlx", "lightning-whisper"])
def test_deferred_selection_explains_unchanged_saved_settings(provider: str) -> None:
    assert providers.provider_release_deferral(provider) == (
        "Lightning Whisper MLX is deferred for Chatbook 0.2.4. Select another "
        "supported transcription provider, such as faster-whisper. Your saved "
        "settings have not been changed."
    )


@pytest.mark.parametrize(
    "provider", [None, "faster-whisper", "parakeet-mlx", "remote-whisper"]
)
def test_release_deferral_does_not_reclassify_other_providers(
    provider: str | None,
) -> None:
    assert providers.provider_release_deferral(provider) is None


def test_supported_provider_keeps_its_existing_package_availability(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        providers, "module_installed", lambda name: name == "faster_whisper"
    )
    assert providers.provider_installed("faster-whisper") is True


def test_generic_module_probe_still_reports_physical_presence(monkeypatch) -> None:
    monkeypatch.setattr(providers.importlib.util, "find_spec", lambda name: object())
    assert providers.module_installed("lightning_whisper_mlx") is True
