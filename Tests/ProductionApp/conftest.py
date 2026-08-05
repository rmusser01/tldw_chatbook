"""Collection-time privacy boundary for production-application tests."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile


_PRIVATE_TEST_ROOT = Path(tempfile.mkdtemp(prefix="tldw_production_app_")).resolve()
_PRIVATE_HOME = _PRIVATE_TEST_ROOT / "home"
_PRIVATE_DATA = _PRIVATE_TEST_ROOT / "data"
_PRIVATE_CONFIG = _PRIVATE_TEST_ROOT / "config"
_PRIVATE_TEMP = _PRIVATE_TEST_ROOT / "tmp"

for _directory in (_PRIVATE_HOME, _PRIVATE_DATA, _PRIVATE_CONFIG, _PRIVATE_TEMP):
    _directory.mkdir(parents=True, mode=0o700)

os.environ.update(
    {
        "HOME": str(_PRIVATE_HOME),
        "USERPROFILE": str(_PRIVATE_HOME),
        "XDG_DATA_HOME": str(_PRIVATE_DATA),
        "XDG_CONFIG_HOME": str(_PRIVATE_CONFIG),
        "TLDW_CONFIG_PATH": str(_PRIVATE_CONFIG / "config.toml"),
        "TMPDIR": str(_PRIVATE_TEMP),
    }
)


def pytest_sessionfinish() -> None:
    """Remove the private filesystem after the production-app test session."""

    shutil.rmtree(_PRIVATE_TEST_ROOT, ignore_errors=True)
