"""Build real qualification binaries; never skip a missing required tool."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope="session")
def helper_resource_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("age-resource")
    root.chmod(0o700)
    helper_source = Path(__file__).resolve().parents[2] / "Packaging" / "backup_age"
    binary = root / ("backup-age.exe" if os.name == "nt" else "backup-age")
    subprocess.run(
        ["go", "build", "-o", str(binary), "."], cwd=helper_source, check=True
    )
    info = json.loads(subprocess.check_output([str(binary), "info"]))
    info["sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(info))
    return root


@pytest.fixture(scope="session")
def official_age(tmp_path_factory):
    root = tmp_path_factory.mktemp("official-age")
    binary = root / "age"
    helper_source = Path(__file__).resolve().parents[2] / "Packaging" / "backup_age"
    subprocess.run(
        ["go", "build", "-o", str(binary), "filippo.io/age/cmd/age"],
        cwd=helper_source,
        check=True,
    )
    return binary
