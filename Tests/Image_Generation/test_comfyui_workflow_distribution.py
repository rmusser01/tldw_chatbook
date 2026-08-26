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
    EXPECTED_DIRECT_LINKS,
    EXPECTED_NEUTRAL_LITERALS,
    EXPECTED_NODE_CLASSES,
    WORKFLOW_FILENAME,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_MEMBER = (
    "tldw_chatbook/Image_Generation/workflows/" + WORKFLOW_FILENAME
)
SDIST_INCLUDE = "recursive-include tldw_chatbook/Image_Generation/workflows *.json"
BUILD_ERROR = "H3 workflow distributions could not be built"
SDIST_ERROR = "H3 workflow sdist inventory does not match the approved contract"
WHEEL_ERROR = "H3 workflow wheel inventory does not match the approved contract"
INSTALL_ERROR = "H3 workflow wheel could not be installed"
INSTALLED_PROBE_ERROR = "Installed H3 workflow probe failed"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
    _require(completed.returncode == 0, BUILD_ERROR)
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    _require(len(sdists) == 1 and len(wheels) == 1, BUILD_ERROR)
    return sdists[0], wheels[0]


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        files = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in files}
    _require(len(roots) == 1, SDIST_ERROR)
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
    _require(completed.returncode == 0, INSTALL_ERROR)


def _probe_installed_workflow(target: Path) -> None:
    script = r"""
import json
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
checkout = Path(sys.argv[2]).resolve()
expected = json.loads(sys.argv[3])

def require(condition, message):
    if not condition:
        raise AssertionError(message)

sys.path.insert(0, str(target))
for entry in sys.path:
    if not entry:
        continue
    resolved_entry = Path(entry).resolve()
    require(resolved_entry != checkout, "Installed probe import confinement failed")
    require(
        not resolved_entry.is_relative_to(checkout),
        "Installed probe import confinement failed",
    )

from tldw_chatbook.Image_Generation.adapters import comfyui_image_adapter as adapter

require(
    Path(adapter.__file__).resolve().is_relative_to(target),
    "Installed adapter did not load from the wheel target",
)
first = adapter._load_packaged_workflow()
second = adapter._load_packaged_workflow()
require(first is not second, "Installed loader did not return a fresh workflow")
first["114"]["inputs"]["image"] = "mutation"
require(
    second["114"]["inputs"]["image"] == expected["neutral"]["114.image"],
    "Installed loader returned shared workflow state",
)
for invalid_key in (
    "",
    None,
    "other",
    "minimax_h3_image_edit.json",
    "../minimax_h3_image_edit",
    "nested/key",
    "nested\\key",
):
    try:
        adapter._load_packaged_workflow(invalid_key)
    except ValueError:
        pass
    else:
        raise AssertionError("Installed loader accepted a forbidden workflow key")

classes = {key: node.get("class_type") for key, node in second.items()}
require(classes == expected["classes"], "Installed workflow node classes mismatch")

def walk(value, parts):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, [*parts, str(key)])
    else:
        yield ".".join(parts), value

def direct_link(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )

leaves = {}
for node_id, node in second.items():
    leaves.update(walk(node.get("inputs", {}), [node_id]))
links = {path: value for path, value in leaves.items() if direct_link(value)}
require(links == expected["links"], "Installed workflow direct links mismatch")
for path, approved in expected["neutral"].items():
    require(leaves.get(path) == approved, "Installed workflow neutral controls mismatch")

restored_output = (
    second["165"]["inputs"]["images"] == ["149", 0]
    and second["149"]["inputs"]["input"] == ["144", 0]
    and second["149"]["inputs"]["resize_type.width"] == ["150", 0]
    and second["149"]["inputs"]["resize_type.height"] == ["150", 1]
    and second["150"]["inputs"]["image"] == ["114", 0]
)
require(restored_output, "Installed workflow restored output path mismatch")
print("PASS")
"""
    expected = json.dumps(
        {
            "classes": EXPECTED_NODE_CLASSES,
            "links": {
                path: list(source) for path, source in EXPECTED_DIRECT_LINKS.items()
            },
            "neutral": EXPECTED_NEUTRAL_LITERALS,
        },
        sort_keys=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(target),
            str(REPO_ROOT),
            expected,
        ],
        cwd=target.parent,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    _require(completed.returncode == 0, INSTALLED_PROBE_ERROR)
    _require(completed.stdout.strip() == "PASS", INSTALLED_PROBE_ERROR)


def test_workflow_ships_in_wheel_and_sdist_and_loads_from_wheel(
    tmp_path: Path,
) -> None:
    manifest_lines = {
        line.strip()
        for line in (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    }
    _require(SDIST_INCLUDE in manifest_lines, SDIST_ERROR)

    source_root = tmp_path / "source"
    source_root.mkdir()
    _copy_build_inputs(source_root)
    sdist, wheel = _build_distributions(source_root)

    _require(
        _image_workflow_members(_sdist_members(sdist)) == {WORKFLOW_MEMBER},
        SDIST_ERROR,
    )
    _require(
        _image_workflow_members(_wheel_members(wheel)) == {WORKFLOW_MEMBER},
        WHEEL_ERROR,
    )

    installed = tmp_path / "installed"
    installed.mkdir()
    _install_wheel(wheel, installed)
    _probe_installed_workflow(installed)
