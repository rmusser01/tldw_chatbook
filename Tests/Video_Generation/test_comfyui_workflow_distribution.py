"""Built-distribution coverage for the shipped ComfyUI H3 workflows."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


REPO_ROOT = Path(__file__).parents[2]
WORKFLOW_PATHS = {
    "tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v.json",
    "tldw_chatbook/Video_Generation/workflows/minimax_h3_t2v_spectrum.json",
}
OBSOLETE_WORKFLOW_PATHS = {
    "tldw_chatbook/Video_Generation/workflows/wan22_t2v.json",
    "tldw_chatbook/Video_Generation/workflows/svd_xt_i2v.json",
}


def _validated_directory(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    assert directory.is_dir()
    assert not directory.is_symlink()
    assert directory.resolve().is_relative_to(root.resolve())
    return directory


def _copy_build_inputs(destination: Path) -> None:
    ignored = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".DS_Store",
        "build",
        "dist",
        "*.egg-info",
    )
    shutil.copytree(
        REPO_ROOT / "tldw_chatbook",
        destination / "tldw_chatbook",
        ignore=ignored,
    )
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "CLAUDE.md",
        "CHANGELOG.md",
        "requirements.txt",
    ):
        shutil.copy2(REPO_ROOT / name, destination / name)


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        files = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in files}
    assert len(roots) == 1
    return {name.split("/", 1)[1] for name in files if "/" in name}


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


def _workflow_members(members: set[str]) -> set[str]:
    prefix = "tldw_chatbook/Video_Generation/workflows/"
    return {
        name
        for name in members
        if name.startswith(prefix) and name.endswith(".json")
    }


def _run(command: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def test_h3_workflows_ship_in_sdist_and_installed_wheel(tmp_path: Path) -> None:
    source_root = _validated_directory(tmp_path, "source")
    dist_dir = _validated_directory(tmp_path, "dist")
    _copy_build_inputs(source_root)

    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ],
        cwd=source_root,
        timeout=300,
    )
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(sdists) == 1
    assert len(wheels) == 1

    sdist_workflows = _workflow_members(_sdist_members(sdists[0]))
    wheel_workflows = _workflow_members(_wheel_members(wheels[0]))
    assert sdist_workflows == WORKFLOW_PATHS
    assert wheel_workflows == WORKFLOW_PATHS
    assert OBSOLETE_WORKFLOW_PATHS.isdisjoint(sdist_workflows)
    assert OBSOLETE_WORKFLOW_PATHS.isdisjoint(wheel_workflows)

    install_root = _validated_directory(tmp_path, "installed")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(install_root),
            str(wheels[0]),
        ],
        cwd=tmp_path,
        timeout=180,
    )
    probe_root = _validated_directory(tmp_path, "probe")
    probe = """
import os
from pathlib import Path

from tldw_chatbook.Video_Generation.adapters import comfyui_video_adapter as cva

target = Path(os.environ["H3_WHEEL_TARGET"]).resolve()
assert Path(cva.__file__).resolve().is_relative_to(target)
data_root = Path.cwd() / "data"
data_root.mkdir()
cva.get_user_data_dir = lambda: data_root
adapter = cva.ComfyUIVideoAdapter.__new__(cva.ComfyUIVideoAdapter)
for workflow_name in ("minimax_h3_t2v.json", "minimax_h3_t2v_spectrum.json"):
    graph = adapter._load_workflow(workflow_name)
    assert cva.ComfyUIVideoAdapter._is_h3_workflow(graph)
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=probe_root,
        env={
            **os.environ,
            "H3_WHEEL_TARGET": str(install_root),
            "PYTHONPATH": str(install_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
