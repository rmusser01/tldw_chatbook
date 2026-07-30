from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        (("parakeet_mlx", "lightning_whisper_mlx"), "parakeet-mlx"),
        (("lightning_whisper_mlx",), "lightning-whisper-mlx"),
        ((), "faster-whisper"),
    ],
)
def test_macos_stt_default_probes_packages_without_importing_them(
    tmp_path: Path,
    installed: tuple[str, ...],
    expected: str,
) -> None:
    """Choosing an STT default must not initialize optional native runtimes."""

    script = textwrap.dedent(
        f"""
        import builtins
        import importlib.machinery
        import importlib.util
        import json
        import sys

        installed = {installed!r}
        guarded = {{"parakeet_mlx", "lightning_whisper_mlx"}}
        original_find_spec = importlib.util.find_spec
        original_import = builtins.__import__

        def find_spec(name, package=None):
            if name in guarded:
                if name in installed:
                    return importlib.machinery.ModuleSpec(name, loader=None)
                return None
            return original_find_spec(name, package)

        def reject_optional_runtime_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in guarded:
                raise AssertionError(f"optional runtime imported: {{name}}")
            return original_import(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        builtins.__import__ = reject_optional_runtime_import
        sys.platform = "darwin"

        from tldw_chatbook.config import settings

        print(json.dumps(settings["STT_settings"]["default_stt_provider"]))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=_hermetic_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == f'"{expected}"'


def _hermetic_env(tmp_path: Path) -> dict:
    """Env vars that keep a `config.py` import from touching the real home dir."""
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


def test_non_macos_stt_default_fallback_is_hyphenated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every provider id actually dispatched on is hyphenated --
    `console_voice_input.py`'s `LOCAL_PROVIDER_MODULES` (`"faster-whisper"`)
    and `transcription_service.py`'s provider-branch matching. On a non-macOS
    platform neither `if sys.platform == "darwin"` branch below the initial
    assignment in `config.py` ever runs, so that initial value is exactly
    what `STT_settings.default_stt_provider` resolves to.

    This machine is darwin, so `sys.platform` is patched to prove the
    fallback path (an unpatched test would prove nothing). It is patched
    *in-process*, around a `load_settings(force_reload=True)` call on the
    already-imported `config` module, not before a fresh interpreter imports
    it: spawning a subprocess with `sys.platform` forced to a fake value
    *before* `import config` (and everything it transitively imports --
    `loguru`, `psutil`, ...) reliably crashes the interpreter itself, because
    several of those packages pick a platform-specific C extension or probe
    `sysconfig` at import time using the real build's platform tag. Patching
    only around the call keeps every other import on the real platform and
    exercises just the one `sys.platform` read this fix touches.

    Args:
        tmp_path: pytest's per-test temporary directory; used as the isolated
            `TLDW_CONFIG_PATH` so this test never touches a real config file.
        monkeypatch: pytest's monkeypatch fixture; sets `TLDW_CONFIG_PATH`
            and forces `config_module.sys.platform` to `"linux"`.
    """
    from tldw_chatbook import config as config_module

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    config_module._CONFIG_CACHE = None
    config_module._CONFIG_CACHE_SOURCE = None
    config_module._SETTINGS_CACHE = None
    config_module._SETTINGS_CACHE_SOURCE = None
    monkeypatch.setattr(config_module.sys, "platform", "linux")

    settings = config_module.load_settings(force_reload=True)

    assert settings["STT_settings"]["default_stt_provider"] == "faster-whisper"
