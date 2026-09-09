"""Active package and qualification surfaces share the Python 3.12 floor."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _job_block(workflow: str, job_name: str) -> str:
    start = workflow.index(f"  {job_name}:")
    following = re.search(r"^  [a-z0-9-]+:\s*$", workflow[start + 1 :], re.MULTILINE)
    if following is None:
        return workflow[start:]
    return workflow[start : start + 1 + following.start()]


def test_project_metadata_declares_python_312_floor() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    project = metadata["project"]
    assert project["requires-python"] == ">=3.12"
    assert "Programming Language :: Python :: 3.11" not in project["classifiers"]
    assert "Programming Language :: Python :: 3.12" in project["classifiers"]
    assert metadata["tool"]["mypy"]["python_version"] == "3.12"


def test_active_install_and_packaging_guidance_requires_python_312() -> None:
    readme = _text("README.md")
    assert "Python 3.12+" in readme
    assert "Python `>=3.12`" in readme
    assert "Python 3.11+" not in readme
    assert "Python `>=3.11`" not in readme

    for contributor_guide in ("AGENTS.md", "CLAUDE.md"):
        guide = _text(contributor_guide)
        assert "Python ≥3.12" in guide
        assert "Python ≥3.11" not in guide

    packaging = _text("Packaging/README.md")
    assert "Python 3.12 or later" in packaging
    assert packaging.count("python-version: '3.12'") == 2
    assert "Python version (3.12+ required)" in packaging
    assert "Python 3.11 or later" not in packaging


def test_active_runtime_checks_require_python_312() -> None:
    windows_builder = _text("Packaging/windows/build_windows.py")
    assert "sys.version_info < (3, 12)" in windows_builder
    assert "ERROR: Python 3.12+ is required" in windows_builder

    test_runner = _text("run_all_tests_with_report.py")
    assert "sys.version_info < (3, 12)" in test_runner
    assert "Error: Python 3.12+ required" in test_runner

    preflight = _text("scripts/preflight.sh")
    assert "PYTHON_FLOOR_MAJOR=3" in preflight
    assert "PYTHON_FLOOR_MINOR=12" in preflight
    assert "python3.14 python3.13 python3.12 python" in preflight
    assert "python3.11" not in preflight


def test_active_terminal_qualification_example_uses_python_312() -> None:
    qualification = _text("scripts/terminal_qualification/README.md")

    assert "tldw-task-22512-macos-arm64-py312" in qualification
    assert (
        "python3.12 scripts/terminal_qualification/common.py prepare-row"
        in qualification
    )
    assert "--row-id macos-arm64-py312" in qualification
    assert "native Windows x64 CPython 3.11 qualification row" in qualification


def test_ci_jobs_that_install_or_parse_chatbook_use_python_312() -> None:
    derived = _text(".github/workflows/derived-artifacts.yml")
    for job_name in ("pr-fast-lane", "derived-artifacts"):
        assert "python-version: '3.12'" in _job_block(derived, job_name)

    comprehensive = _text(".github/workflows/test.yml")
    assert 'python-version: ["3.12"]' in _job_block(
        comprehensive, "artifact-lease-spike"
    )
    assert 'python-version: "3.12"' in _job_block(comprehensive, "textual-minimum")
    assert 'python-version: "3.11"' in _job_block(comprehensive, "artifact-lease-shape")

    css_guard = _text(".github/workflows/css-bundle-guard.yml")
    assert "python-version: '3.12'" in _job_block(css_guard, "css-bundle-reproducible")


def test_nightly_matrix_starts_at_python_312_without_duplicate_floor_row() -> None:
    nightly = _text(".github/workflows/nightly-deep.yml")
    nightly_job = _job_block(nightly, "nightly-deep")
    matrix = nightly_job[
        nightly_job.index("        include:") : nightly_job.index("    steps:")
    ]

    assert 'python-version: "3.11"' not in matrix
    assert matrix.count('python-version: "3.12"') == 3
    assert matrix.count('python-version: "3.13"') == 1
    assert matrix.count("os: ubuntu-latest") == 2
    assert matrix.count("os: macos-latest") == 1
    assert matrix.count("os: windows-latest") == 1
    assert "io-encoding: cp1252" in matrix
