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
        ((), "faster_whisper"),
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

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == f'"{expected}"'
