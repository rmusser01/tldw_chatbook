from __future__ import annotations

import configparser
from email.parser import Parser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import NamedTuple
import zipfile

import pytest


pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_NAMES = {
    "academic_paper",
    "code_documentation",
    "conversation",
    "ebook_chapters",
    "json",
    "legal_document",
    "paragraphs",
    "rolling_summarize",
    "semantic",
    "sentences",
    "tokens",
    "words",
    "xml",
}


class BuiltDistributions(NamedTuple):
    source_root: Path
    dist_dir: Path
    sdist: Path
    wheel: Path


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
    for name in ("tldw_chatbook", "Packaging"):
        shutil.copytree(REPO_ROOT / name, destination / name, ignore=ignored)

    seen_test_trees: set[tuple[int, int]] = set()
    for name in ("Tests", "tests", "STests"):
        source = REPO_ROOT / name
        if not source.is_dir():
            continue
        stat = source.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen_test_trees:
            continue
        seen_test_trees.add(identity)
        shutil.copytree(source, destination / name, ignore=ignored)

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


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> BuiltDistributions:
    source_root = tmp_path_factory.mktemp("distribution-source")
    _copy_build_inputs(source_root)
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
        f"command: {command}\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "`project.license` as a TOML table is deprecated" not in (
        completed.stdout + completed.stderr
    )
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(sdists) == 1
    assert len(wheels) == 1
    return BuiltDistributions(source_root, dist_dir, sdists[0], wheels[0])


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        files = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in files}
    assert len(roots) == 1
    return {name.split("/", 1)[1] for name in files if "/" in name}


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


def test_built_artifacts_match_distribution_contract(
    built_distributions: BuiltDistributions,
) -> None:
    sdist_members = _sdist_members(built_distributions.sdist)
    wheel_members = _wheel_members(built_distributions.wheel)

    required_sdist = {
        "LICENSE",
        "README.md",
        "CLAUDE.md",
        "CHANGELOG.md",
        "MANIFEST.in",
        "pyproject.toml",
        "requirements.txt",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/components/stats_screen.css",
        "tldw_chatbook/Config_Files/rag_pipelines.toml",
        "tldw_chatbook/Evals/config/eval_config.yaml",
        "tldw_chatbook/Third_Party/aider/LICENSE.txt",
        "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
    }
    required_wheel = {
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/Config_Files/rag_pipelines.toml",
        "tldw_chatbook/Evals/config/eval_config.yaml",
        "tldw_chatbook/Third_Party/aider/LICENSE.txt",
        "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
    }
    assert not required_sdist - sdist_members
    assert not required_wheel - wheel_members

    wheel_templates = {
        Path(name).stem
        for name in wheel_members
        if name.startswith("tldw_chatbook/Chunking/templates/")
        and name.endswith(".json")
    }
    assert wheel_templates == TEMPLATE_NAMES

    forbidden_wheel = {
        "tldw_chatbook/css/components/stats_screen.css",
        "tldw_chatbook/Config_Files/embedding_configs_examples.toml",
        "tldw_chatbook/Config_Files/pipeline_configs/custom_pipelines_example.toml",
        "tldw_chatbook/Chunking/templates/README.md",
        "tldw_chatbook/Chunking/templates/example_usage.py",
        "tldw_chatbook/Evals/DEVELOPER_GUIDE.md",
    }
    assert forbidden_wheel.isdisjoint(wheel_members)
    for members in (sdist_members, wheel_members):
        assert not any(
            name.startswith(("Tests/", "tests/", "STests/"))
            or "/__pycache__/" in name
            or name.endswith((".pyc", ".pyo", ".DS_Store"))
            for name in members
        )

    with zipfile.ZipFile(built_distributions.wheel) as archive:
        metadata_name = next(
            name for name in wheel_members if name.endswith(".dist-info/METADATA")
        )
        entry_points_name = next(
            name
            for name in wheel_members
            if name.endswith(".dist-info/entry_points.txt")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(entry_points_name).decode("utf-8"))

    with tarfile.open(built_distributions.sdist, "r:gz") as archive:
        pkg_info = next(
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/PKG-INFO")
        )
        pkg_info_stream = archive.extractfile(pkg_info)
        assert pkg_info_stream is not None
        sdist_metadata = Parser().parsestr(
            pkg_info_stream.read().decode("utf-8")
        )

    assert metadata["Metadata-Version"] == "2.4"
    assert metadata["License-Expression"] == "AGPL-3.0-or-later"
    assert "LICENSE" in (metadata.get_all("License-File") or [])
    assert sdist_metadata["Metadata-Version"] == "2.4"
    assert sdist_metadata["License-Expression"] == "AGPL-3.0-or-later"
    assert "LICENSE" in (sdist_metadata.get_all("License-File") or [])
    assert any(
        name.endswith(".dist-info/licenses/LICENSE") for name in wheel_members
    )
    assert dict(entry_points["console_scripts"]) == {
        "tldw-cli": "tldw_chatbook.cli:main_cli_runner",
        "tldw-serve": "tldw_chatbook.Web_Server.serve:main",
    }
