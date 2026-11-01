"""Private recovery fixtures and package-owned age worker resources."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def private_recovery_temp(tmp_path, monkeypatch):
    """Give recovery workers private temporary and native app-data roots."""
    from tldw_chatbook.Backup_Recovery.native_files import create_private_directory

    temporary = tmp_path / "private-temp"
    create_private_directory(temporary)
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    for name, leaf in (
        ("APPDATA", "fixture-roaming"),
        ("LOCALAPPDATA", "fixture-local"),
    ):
        appdata = tmp_path / leaf
        create_private_directory(appdata)
        monkeypatch.setenv(name, str(appdata))


@pytest.fixture(scope="session")
def helper_resource_root(tmp_path_factory):
    package_root = tmp_path_factory.mktemp("age-package")
    package_root.chmod(0o700)
    worker = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "Backup_Recovery"
        / "age_worker.py"
    )
    shutil.copy2(worker, package_root / "age_worker.py")
    return package_root
