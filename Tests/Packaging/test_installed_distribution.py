from __future__ import annotations

import configparser
from email.parser import Parser
import hashlib
import json
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

INSTALLED_PROBE = r"""
from pathlib import Path
import json
import os
import tomllib

import tldw_chatbook
from tldw_chatbook.Chunking.chunking_templates import ChunkingTemplateManager
from tldw_chatbook.Evals.config_loader import EvalConfigLoader
from tldw_chatbook.RAG_Search.pipeline_loader import PipelineLoader
from tldw_chatbook.app import TldwCli, get_app

package_root = Path(tldw_chatbook.__file__).resolve().parent
expected_target = Path(os.environ["EXPECTED_TARGET"]).resolve()
expected_templates = set(json.loads(os.environ["EXPECTED_TEMPLATES"]))
assert package_root.is_relative_to(expected_target)
assert (package_root / "css" / "tldw_cli_modular.tcss").is_file()

with (package_root / "Config_Files" / "rag_pipelines.toml").open("rb") as stream:
    assert "plain" in tomllib.load(stream)["pipelines"]

loader = PipelineLoader(config_dir=package_root / "Config_Files")
loader.load_pipeline_config()
assert "plain" in loader.pipelines
assert set(ChunkingTemplateManager().get_available_templates()) == expected_templates
assert "code_execution" in EvalConfigLoader().get_task_types()
assert (package_root / "Third_Party" / "aider" / "LICENSE.txt").is_file()
assert (
    package_root / "Third_Party" / "textual_fspicker" / "LICENSE"
).is_file()
assert isinstance(get_app(), TldwCli)
print(package_root)
"""


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


def _run_manifest_checker(
    built: BuiltDistributions,
    dist_dir: Path,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(built.source_root / "Packaging" / "check_manifest.py"),
            str(dist_dir),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _install_wheel(
    built: BuiltDistributions,
    target: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(target),
        str(built.wheel),
    ]
    completed = subprocess.run(
        command,
        cwd=target.parent,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _target_hashes(target: Path) -> dict[str, str]:
    return {
        path.relative_to(target).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }


def _private_child_env(state_root: Path, target: Path) -> dict[str, str]:
    state_root = state_root.resolve(strict=True)
    config_root = state_root / "config"
    data_root = state_root / "data"
    temp_root = state_root / "tmp"
    for path in (config_root, data_root, temp_root):
        path.mkdir(parents=True, mode=0o700, exist_ok=True)

    env = os.environ.copy()
    for name in ("TLDW_TEST_CONFIG_ROOT", "TLDW_TEST_CONFIG_ROOT_OWNER"):
        env.pop(name, None)
    env.update(
        {
            "HOME": str(state_root),
            "USERPROFILE": str(state_root),
            "APPDATA": str(data_root),
            "LOCALAPPDATA": str(data_root),
            "XDG_CONFIG_HOME": str(config_root),
            "XDG_DATA_HOME": str(data_root),
            "TLDW_CONFIG_PATH": str(config_root / "config.toml"),
            "TMPDIR": str(temp_root),
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(target.resolve(strict=True)),
            "EXPECTED_TARGET": str(target.resolve(strict=True)),
            "EXPECTED_TEMPLATES": json.dumps(sorted(TEMPLATE_NAMES)),
        }
    )
    return env


def _run_child(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return completed


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


def test_release_checker_accepts_fresh_artifacts(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    result = _run_manifest_checker(
        built_distributions,
        built_distributions.dist_dir,
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_checker_rejects_multiple_wheels(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    shutil.copy2(
        built_distributions.wheel,
        dist_dir / f"duplicate-{built_distributions.wheel.name}",
    )

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert "exactly one wheel" in (result.stdout + result.stderr).lower()


def test_release_checker_rejects_sdist_only_css_in_wheel(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    wheel = next(dist_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(
            "tldw_chatbook/css/components/stats_screen.css",
            "forbidden",
        )

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert "stats_screen.css" in result.stdout + result.stderr


def test_release_checker_rejects_missing_runtime_data(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "dist"
    shutil.copytree(built_distributions.dist_dir, dist_dir)
    wheel = next(dist_dir.glob("*.whl"))
    rewritten = wheel.with_suffix(".rewritten")
    missing = "tldw_chatbook/Evals/config/eval_config.yaml"
    with (
        zipfile.ZipFile(wheel) as source,
        zipfile.ZipFile(rewritten, "w") as destination,
    ):
        for member in source.infolist():
            if member.filename != missing:
                destination.writestr(member, source.read(member.filename))
    rewritten.replace(wheel)

    result = _run_manifest_checker(built_distributions, dist_dir, tmp_path)

    assert result.returncode == 1
    assert missing in result.stdout + result.stderr


def test_installed_wheel_loaders_entry_points_and_assets_are_immutable(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    state_root = tmp_path / "state"
    run_root = tmp_path / "run"
    state_root.mkdir(mode=0o700)
    run_root.mkdir()
    _install_wheel(built_distributions, target)
    env = _private_child_env(state_root, target)
    before = _target_hashes(target)
    results = [
        _run_child([sys.executable, "-c", INSTALLED_PROBE], run_root, env)
    ]

    script_path = os.pathsep.join(
        str(path) for path in (target / "bin", target / "Scripts")
    )
    for name in ("tldw-cli", "tldw-serve"):
        script = shutil.which(name, path=script_path)
        assert script is not None, (
            f"missing installed script {name!r}; "
            f"target files: {sorted(_target_hashes(target))}"
        )
        results.append(_run_child([script, "--help"], run_root, env))

    after = _target_hashes(target)
    process_text = "\n".join(
        result.stdout + "\n" + result.stderr for result in results
    )
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in state_root.rglob("*.log*")
        if path.is_file()
    )
    observed_text = process_text + "\n" + log_text
    for forbidden in (
        "Building modular CSS",
        "Failed to build modular CSS",
        "Error handling CSS file",
    ):
        assert forbidden not in observed_text
    assert after == before
