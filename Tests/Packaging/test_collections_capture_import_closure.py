"""Import-closure guards for deferred Collections capture composition."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run one import probe against an isolated profile."""
    home = tmp_path / "home"
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    for directory in (home, data_home, config_home):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "PYTHONPATH": str(REPO_ROOT),
        "TLDW_TEST_MODE": "1",
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


_CAPTURE_IMPLEMENTATIONS = (
    "tldw_chatbook.Library.collections_capture_repository",
    "tldw_chatbook.Library.collections_capture_service",
    "tldw_chatbook.Library.collections_legacy_recovery",
    "tldw_chatbook.Library.collections_offline_store",
    "tldw_chatbook.Library.server_collections_capture_service",
)


def test_library_package_preserves_capture_exports_without_eager_imports(
    tmp_path: Path,
) -> None:
    """The package facade must defer capture implementations until access."""
    code = f"""
import sys
import tldw_chatbook.Library as library

forbidden = {_CAPTURE_IMPLEMENTATIONS!r}
resident = [name for name in forbidden if name in sys.modules]
assert not resident, resident

exported = library.CollectionsCaptureRepository
from tldw_chatbook.Library.collections_capture_repository import (
    CollectionsCaptureRepository,
)
assert exported is CollectionsCaptureRepository
assert library.CollectionsCaptureRepository is CollectionsCaptureRepository
print("LIBRARY_CAPTURE_FACADE_OK")
"""
    result = _run_isolated_python(tmp_path, code)
    assert result.returncode == 0, (
        f"Library capture facade probe failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-3000:]}"
    )
    assert "LIBRARY_CAPTURE_FACADE_OK" in result.stdout


def test_app_import_defers_capture_implementation_graph(tmp_path: Path) -> None:
    """Importing the app must not compose capture persistence or adapters."""
    code = f"""
import sys
import tldw_chatbook.app  # noqa: F401

forbidden = {_CAPTURE_IMPLEMENTATIONS!r}
resident = [name for name in forbidden if name in sys.modules]
assert not resident, resident
print("APP_CAPTURE_IMPORT_DIET_OK")
"""
    result = _run_isolated_python(tmp_path, code)
    assert result.returncode == 0, (
        f"app capture import-diet probe failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "APP_CAPTURE_IMPORT_DIET_OK" in result.stdout


def test_headless_capture_controller_keeps_service_implementation_lazy(
    tmp_path: Path,
) -> None:
    """The controller may use service protocols without importing backends."""
    code = f"""
import sys
import tldw_chatbook.UI.Library_Modules.library_collections_capture_controller

forbidden = {_CAPTURE_IMPLEMENTATIONS!r}
resident = [name for name in forbidden if name in sys.modules]
assert not resident, resident
assert "tldw_chatbook.Library.collections_capture_models" in sys.modules
print("CAPTURE_CONTROLLER_IMPORT_DIET_OK")
"""
    result = _run_isolated_python(tmp_path, code)
    assert result.returncode == 0, (
        f"capture controller import-diet probe failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-3000:]}"
    )
    assert "CAPTURE_CONTROLLER_IMPORT_DIET_OK" in result.stdout
