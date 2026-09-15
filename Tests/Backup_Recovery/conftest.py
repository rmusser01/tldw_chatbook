"""Copy the package-owned Python age worker for transport tests."""

import shutil
from pathlib import Path

import pytest


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
