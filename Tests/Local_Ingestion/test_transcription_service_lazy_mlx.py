from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_MODULE = "tldw_chatbook.Local_Ingestion.transcription_service"
LOADER_CASES = (
    pytest.param(
        "_ensure_lightning_whisper_mlx_import",
        "LIGHTNING_WHISPER_AVAILABLE",
        "LightningWhisperMLX",
        "lightning_whisper_mlx",
        "LightningWhisperMLX",
        id="lightning-whisper",
    ),
    pytest.param(
        "_ensure_parakeet_mlx_import",
        "PARAKEET_MLX_AVAILABLE",
        "parakeet_from_pretrained",
        "parakeet_mlx",
        "from_pretrained",
        id="parakeet",
    ),
)


@pytest.fixture(scope="module")
def service_module():
    """Import safely while preserving the pre-fix RED test path."""
    if SERVICE_MODULE in sys.modules:
        return sys.modules[SERVICE_MODULE]
    lightning_module = ModuleType("lightning_whisper_mlx")
    lightning_module.LightningWhisperMLX = object()
    parakeet_module = ModuleType("parakeet_mlx")
    parakeet_module.from_pretrained = object()
    with patch.dict(
        sys.modules,
        {
            "lightning_whisper_mlx": lightning_module,
            "parakeet_mlx": parakeet_module,
        },
    ):
        return importlib.import_module(SERVICE_MODULE)


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    private_home = tmp_path / "home"
    private_data = tmp_path / "data"
    private_config = tmp_path / "config"
    private_temp = tmp_path / "tmp"
    for directory in (private_home, private_data, private_config, private_temp):
        directory.mkdir(parents=True, mode=0o700)
    env.update(
        {
            "HOME": str(private_home),
            "USERPROFILE": str(private_home),
            "XDG_DATA_HOME": str(private_data),
            "XDG_CONFIG_HOME": str(private_config),
            "TLDW_CONFIG_PATH": str(private_config / "config.toml"),
            "TMPDIR": str(private_temp),
            "PYTHONPATH": str(PROJECT_ROOT),
        }
    )
    return env


def _service(service_module, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        service_module, "get_cli_setting", lambda _key, default=None: default
    )
    return service_module._LegacyTranscriptionBackend()


def _install_fake_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    mlx_module = ModuleType("mlx")
    core_module = ModuleType("mlx.core")
    core_module.float32 = object()
    core_module.float16 = object()
    core_module.bfloat16 = object()
    mlx_module.core = core_module
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", core_module)


def test_service_import_discovers_mlx_without_importing_it(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import builtins
        import importlib.machinery
        import importlib.util
        import json
        import sys

        guarded = {"parakeet_mlx", "lightning_whisper_mlx"}
        original_find_spec = importlib.util.find_spec
        original_import = builtins.__import__

        def find_spec(name, package=None):
            if name in guarded:
                return importlib.machinery.ModuleSpec(name, loader=None)
            return original_find_spec(name, package)

        def reject_runtime_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in guarded:
                raise AssertionError(f"optional runtime imported: {name}")
            return original_import(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        builtins.__import__ = reject_runtime_import
        sys.platform = "darwin"

        from tldw_chatbook.Local_Ingestion import transcription_service

        print(
            json.dumps(
                {
                    "parakeet_available":
                        transcription_service.PARAKEET_MLX_AVAILABLE,
                    "lightning_available":
                        transcription_service.LIGHTNING_WHISPER_AVAILABLE,
                    "parakeet_loaded":
                        transcription_service.parakeet_from_pretrained is not None,
                    "lightning_loaded":
                        transcription_service.LightningWhisperMLX is not None,
                }
            )
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=_isolated_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "parakeet_available": True,
        "lightning_available": True,
        "parakeet_loaded": False,
        "lightning_loaded": False,
    }


@pytest.mark.parametrize(
    (
        "loader_name",
        "available_name",
        "symbol_name",
        "module_name",
        "export_name",
    ),
    LOADER_CASES,
)
def test_mlx_loader_imports_once_and_caches_symbol(
    service_module,
    monkeypatch: pytest.MonkeyPatch,
    loader_name: str,
    available_name: str,
    symbol_name: str,
    module_name: str,
    export_name: str,
) -> None:
    expected_symbol = object()
    required_modules: list[str] = []
    optional_deps = importlib.import_module("tldw_chatbook.Utils.optional_deps")

    def require_dependency(name: str):
        required_modules.append(name)
        return SimpleNamespace(**{export_name: expected_symbol})

    monkeypatch.setattr(service_module, available_name, True)
    monkeypatch.setattr(service_module, symbol_name, None)
    monkeypatch.setattr(optional_deps, "require_dependency", require_dependency)
    monkeypatch.setattr(
        service_module.importlib,
        "import_module",
        Mock(side_effect=AssertionError("optional_deps helper bypassed")),
    )
    ensure_import = getattr(service_module, loader_name)

    assert ensure_import() is expected_symbol
    assert ensure_import() is expected_symbol
    assert required_modules == [module_name]


@pytest.mark.parametrize(
    (
        "loader_name",
        "available_name",
        "symbol_name",
        "module_name",
        "_export_name",
    ),
    LOADER_CASES,
)
def test_mlx_loader_failure_is_cached(
    service_module,
    monkeypatch: pytest.MonkeyPatch,
    loader_name: str,
    available_name: str,
    symbol_name: str,
    module_name: str,
    _export_name: str,
) -> None:
    required_modules: list[str] = []
    unsafe_runtime = RuntimeError("unsafe mlx")
    optional_deps = importlib.import_module("tldw_chatbook.Utils.optional_deps")

    def require_dependency(name: str):
        required_modules.append(name)
        raise unsafe_runtime

    monkeypatch.setattr(service_module, available_name, True)
    monkeypatch.setattr(service_module, symbol_name, None)
    monkeypatch.setattr(optional_deps, "require_dependency", require_dependency)
    monkeypatch.setattr(
        service_module.importlib,
        "import_module",
        Mock(side_effect=AssertionError("optional_deps helper bypassed")),
    )
    ensure_import = getattr(service_module, loader_name)

    with pytest.raises(service_module.TranscriptionError) as first_error:
        ensure_import()
    with pytest.raises(service_module.TranscriptionError) as second_error:
        ensure_import()

    assert first_error.value.__cause__ is unsafe_runtime
    assert second_error.value.__cause__ is None
    assert getattr(service_module, available_name) is False
    assert getattr(service_module, symbol_name) is None
    assert required_modules == [module_name]


def test_lightning_file_model_construction_uses_loader(
    service_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(service_module, monkeypatch)
    sentinel = RuntimeError("lightning loader reached")
    ensure_import = Mock(side_effect=sentinel)
    monkeypatch.setattr(service_module, "LIGHTNING_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(
        service_module, "_ensure_lightning_whisper_mlx_import", ensure_import
    )

    with pytest.raises(service_module.TranscriptionError) as error:
        service._transcribe_with_lightning_whisper_mlx("audio.wav")

    assert error.value.__cause__ is sentinel
    ensure_import.assert_called_once_with()


def test_parakeet_file_model_construction_uses_loader(
    service_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    debug_messages: list[str] = []
    monkeypatch.setattr(
        service_module.logger,
        "debug",
        lambda message, *args, **kwargs: debug_messages.append(str(message)),
    )
    service = _service(service_module, monkeypatch)
    sentinel = RuntimeError("parakeet loader reached")
    ensure_import = Mock(side_effect=sentinel)
    soundfile = Mock()
    soundfile.read.return_value = (service_module.np.zeros(1), 16000)
    soundfile.info.return_value = SimpleNamespace(duration=0.0)
    _install_fake_mlx(monkeypatch)
    monkeypatch.setattr(service_module, "PARAKEET_MLX_AVAILABLE", True)
    monkeypatch.setattr(service_module, "sf", soundfile)
    monkeypatch.setattr(service_module, "_ensure_parakeet_mlx_import", ensure_import)

    with pytest.raises(service_module.TranscriptionError) as error:
        service._transcribe_with_parakeet_mlx("audio.wav")

    assert error.value.__cause__ is sentinel
    ensure_import.assert_called_once_with()
    assert not any(
        "parakeet_from_pretrained function:" in message for message in debug_messages
    )


def test_parakeet_buffer_model_construction_uses_loader(
    service_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(service_module, monkeypatch)
    sentinel = RuntimeError("parakeet loader reached")
    ensure_import = Mock(side_effect=sentinel)
    _install_fake_mlx(monkeypatch)
    monkeypatch.setattr(service_module, "PARAKEET_MLX_AVAILABLE", True)
    monkeypatch.setattr(service_module, "_ensure_parakeet_mlx_import", ensure_import)

    with pytest.raises(service_module.TranscriptionError) as error:
        service._transcribe_buffer_with_parakeet_mlx(
            b"\x00\x00",
            sample_rate=16000,
            channels=1,
            sample_width=2,
            model=None,
            language=None,
        )

    assert error.value.__cause__ is sentinel
    ensure_import.assert_called_once_with()


def test_parakeet_streaming_model_construction_uses_loader(
    service_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(service_module, monkeypatch)
    ensure_import = Mock(side_effect=RuntimeError("parakeet loader reached"))
    _install_fake_mlx(monkeypatch)
    monkeypatch.setattr(service_module, "PARAKEET_MLX_AVAILABLE", True)
    monkeypatch.setattr(service_module, "_ensure_parakeet_mlx_import", ensure_import)

    assert service.create_streaming_transcriber(provider="parakeet-mlx") is None
    ensure_import.assert_called_once_with()
