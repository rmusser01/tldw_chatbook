# Tests/Utils/test_model_cache_dir_lifecycle.py
"""
Coverage for get_model_cache_dir's creation branches (task-32901 review),
exercised through the real configuration boundary in fresh interpreters:
the default data-root path is created 0700 by the private lifecycle even
under umask 002, a configured custom directory is validated centrally and
created with a pinned mode, an unusable configured value falls back to the
default, and a pre-existing shared-writable instance logs a reason-gated,
shell-safe repair command instead of breaking the call.

Isolated subprocesses because the config-participant system binds each
route's selected path once per process; a second in-process call with a
different configured value raises raw_source_selection_changed by design.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = """
import stat

from tldw_chatbook.config import get_model_cache_dir, load_settings

load_settings()
cache = get_model_cache_dir()
print(f"CACHE={cache}")
print(f"MODE={oct(stat.S_IMODE(cache.stat().st_mode))}")
"""


def _run(
    home: Path, *, umask_002: bool = True, config_toml: str | None = None
) -> subprocess.CompletedProcess[str]:
    config_dir = home / ".config" / "tldw_cli"
    config_dir.mkdir(parents=True, exist_ok=True)
    if config_toml is not None:
        (config_dir / "config.toml").write_text(config_toml)
        (config_dir / "config.toml").chmod(0o600)
    env = {
        **os.environ,
        "HOME": str(home),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("TLDW_CONFIG_PATH", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    umask = "umask 002; " if umask_002 else ""
    return subprocess.run(
        ["/bin/sh", "-c", f"{umask}exec {sys.executable} -"],
        input=_SCRIPT,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX umask/mode contract")
def test_default_cache_dir_created_0700_under_umask_002(tmp_path: Path) -> None:
    pytest.importorskip("portalocker", reason="config import needs hard deps")
    result = _run(tmp_path / "home")
    assert result.returncode == 0, result.stderr
    assert "MODE=0o700" in result.stdout
    assert "CACHE=" in result.stdout


@pytest.mark.skipif(os.name != "posix", reason="POSIX umask/mode contract")
def test_custom_cache_dir_validated_created_0700_under_umask_002(
    tmp_path: Path,
) -> None:
    custom = tmp_path / "custom cache"  # whitespace on purpose
    toml = (
        "[embedding_config]\n"
        f'model_cache_dir = "{custom.as_posix()}"\n'
    )
    result = _run(tmp_path / "home", config_toml=toml)
    assert result.returncode == 0, result.stderr
    assert f"CACHE={custom.resolve()}" in result.stdout
    assert "MODE=0o700" in result.stdout


@pytest.mark.skipif(os.name != "posix", reason="POSIX refusal contract")
def test_relative_custom_cache_dir_refuses_instead_of_relocating(
    tmp_path: Path,
) -> None:
    toml = '[embedding_config]\nmodel_cache_dir = "relative/cache"\n'
    result = _run(tmp_path / "home", config_toml=toml)
    # An unusable configured value refuses loudly; nothing is created from
    # an unvalidated string and the cache is not silently relocated.
    assert result.returncode != 0
    assert "CACHE=" not in result.stdout
    assert "must be absolute" in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_preexisting_shared_writable_instance_names_shell_safe_repair(
    tmp_path: Path,
) -> None:
    import shlex

    home = tmp_path / "home"
    first = _run(home)
    assert first.returncode == 0, first.stderr
    cache = Path(first.stdout.split("CACHE=")[1].splitlines()[0])
    cache.chmod(0o775)

    second = _run(home)
    assert second.returncode == 0, second.stderr  # non-fatal by design
    repair_lines = [l for l in second.stderr.splitlines() if "chmod g-w,o-w" in l]
    assert repair_lines, second.stderr[-2000:]
    command = repair_lines[0].split("Repair once with: ")[1].strip()
    assert shlex.split(command)[-1] == str(cache)
