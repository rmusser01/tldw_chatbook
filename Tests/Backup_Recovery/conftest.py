"""Build real qualification binaries; never skip a missing required tool."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope="session")
def helper_resource_root(tmp_path_factory):
    package_root = tmp_path_factory.mktemp("age-package")
    package_root.chmod(0o700)
    root = package_root / "_age"
    root.mkdir(mode=0o700)
    helper_source = Path(__file__).resolve().parents[2] / "Packaging" / "backup_age"
    binary = root / ("backup-age.exe" if os.name == "nt" else "backup-age")
    subprocess.run(
        ["go", "build", "-o", str(binary), "."], cwd=helper_source, check=True
    )
    info = json.loads(subprocess.check_output([str(binary), "info"]))
    info["sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    source_manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "tldw_chatbook/Backup_Recovery/helper_manifest.json"
        ).read_text(encoding="utf-8")
    )
    qualified = {
        "status": "qualified",
        **info,
        "resource": f"_age/{binary.name}",
    }
    source_manifest["helpers"] = [
        qualified
        if (entry["os"], entry["arch"]) == (info["os"], info["arch"])
        else entry
        for entry in source_manifest["helpers"]
    ]
    (package_root / "helper_manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
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
