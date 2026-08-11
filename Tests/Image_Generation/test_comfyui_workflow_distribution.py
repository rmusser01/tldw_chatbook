"""Distribution tests for the packaged H3 image-edit workflow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from Tests.Image_Generation.test_comfyui_workflow_assets import (
    EXPECTED_NODE_CLASSES,
    WORKFLOW_FILENAME,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_MEMBER = (
    "tldw_chatbook/Image_Generation/workflows/" + WORKFLOW_FILENAME
)
SDIST_INCLUDE = "recursive-include tldw_chatbook/Image_Generation/workflows *.json"


def _copy_build_inputs(destination: Path) -> None:
    ignored = shutil.ignore_patterns(
        ".git",
        ".venv",
        "__pycache__",
        "*.pyc",
        "build",
        "dist",
        "*.egg-info",
    )
    shutil.copytree(REPO_ROOT / "tldw_chatbook", destination / "tldw_chatbook", ignore=ignored)
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "CLAUDE.md",
        "CHANGELOG.md",
        "requirements.txt",
    ):
        source = REPO_ROOT / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _build_distributions(source_root: Path) -> tuple[Path, Path]:
    dist_dir = source_root / "dist"
    command = [
        sys.executable,
        "-m",
        "build",
        "--sdist",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(sdists) == 1
    assert len(wheels) == 1
    return sdists[0], wheels[0]


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        files = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in files}
    assert len(roots) == 1
    return {name.split("/", 1)[1] for name in files if "/" in name}


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


def _image_workflow_members(members: set[str]) -> set[str]:
    prefix = "tldw_chatbook/Image_Generation/workflows/"
    return {
        name
        for name in members
        if name.startswith(prefix) and name.endswith(".json")
    }


def _install_wheel(wheel: Path, target: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(target),
        str(wheel),
    ]
    completed = subprocess.run(
        command,
        cwd=target.parent,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _load_installed_workflow(target: Path) -> dict[str, object]:
    script = r"""
import json
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
checkout = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(target))
for entry in sys.path:
    if not entry:
        continue
    resolved_entry = Path(entry).resolve()
    assert resolved_entry != checkout
    assert not resolved_entry.is_relative_to(checkout)

from tldw_chatbook.Image_Generation.adapters import comfyui_image_adapter as adapter

assert Path(adapter.__file__).resolve().is_relative_to(target)
first = adapter._load_packaged_workflow()
second = adapter._load_packaged_workflow()
assert first is not second
first["114"]["inputs"]["image"] = "mutation"
assert second["114"]["inputs"]["image"] == "h3_edit_input.png"
for invalid_key in ("other", "../minimax_h3_image_edit", "nested/key", "nested\\key"):
    try:
        adapter._load_packaged_workflow(invalid_key)
    except ValueError:
        pass
    else:
        raise AssertionError(f"unconfined workflow key accepted: {invalid_key!r}")
print(json.dumps({"classes": {key: node["class_type"] for key, node in second.items()}}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(target),
            str(REPO_ROOT),
        ],
        cwd=target.parent,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def test_workflow_ships_in_wheel_and_sdist_and_loads_from_wheel(
    tmp_path: Path,
) -> None:
    manifest_lines = {
        line.strip()
        for line in (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    }
    assert SDIST_INCLUDE in manifest_lines

    source_root = tmp_path / "source"
    source_root.mkdir()
    _copy_build_inputs(source_root)
    sdist, wheel = _build_distributions(source_root)

    assert _image_workflow_members(_sdist_members(sdist)) == {WORKFLOW_MEMBER}
    assert _image_workflow_members(_wheel_members(wheel)) == {WORKFLOW_MEMBER}

    installed = tmp_path / "installed"
    installed.mkdir()
    _install_wheel(wheel, installed)
    loaded = _load_installed_workflow(installed)

    assert loaded == {"classes": EXPECTED_NODE_CLASSES}
