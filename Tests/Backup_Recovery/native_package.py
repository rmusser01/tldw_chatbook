"""Build and preserve an installed package containing the Python worker."""

import hashlib
import json
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def native_package(tmp_path_factory):
    """Install the application wheel and preserve every installed file."""
    from Tests.Packaging.test_backup_helper_distribution import (
        _build_wheel,
        _copy_build_source,
    )

    root = tmp_path_factory.mktemp("f9-native-package")
    source = root / "source"
    source.mkdir()
    _copy_build_source(source)
    wheel = _build_wheel(source, root / "wheels")
    installed = root / "installed"
    # The current interpreter installs the locally built wheel with fixed arguments.
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--disable-pip-version-check",
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    worker = installed / "tldw_chatbook/Backup_Recovery/age_worker.py"
    assert worker.is_file() and not worker.is_symlink()
    files = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in installed.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    (root / "native-package.json").write_text(
        json.dumps(
            {
                "wheel": str(wheel),
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "installed": str(installed),
                "installed_files": len(files),
            },
            indent=2,
        )
    )
    yield installed
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in files.items()
    )
