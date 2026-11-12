"""Collection-time privacy boundary for production-application tests.

HOME, the XDG roots and TLDW_CONFIG_PATH are Tests/conftest.py's: it points
them at its sandbox before anything imports the app. This file used to point
them somewhere else again at import (TASK-33001.11). An app import after that
raised RecoveryRequired('raw_source_selection_changed'), so a run that named
a file here collected nothing, and a private-profile child could not collect
its own test.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

import pytest


_PRIVATE_TEST_ROOT = Path(tempfile.mkdtemp(prefix="tldw_production_app_")).resolve()
_PRIVATE_TEMP = _PRIVATE_TEST_ROOT / "tmp"
_PRIVATE_TEMP.mkdir(mode=0o700)
os.environ["TMPDIR"] = str(_PRIVATE_TEMP)


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh(isolate_test_environment, monkeypatch) -> None:
    """Keep unrelated production-app tests off the catalog network seam."""

    async def _offline_refresh(_app) -> None:
        return None

    monkeypatch.setattr(
        "tldw_chatbook.app.TldwCli._refresh_model_catalogs",
        _offline_refresh,
    )


def pytest_sessionfinish() -> None:
    """Remove the private filesystem after the production-app test session."""

    shutil.rmtree(_PRIVATE_TEST_ROOT, ignore_errors=True)
