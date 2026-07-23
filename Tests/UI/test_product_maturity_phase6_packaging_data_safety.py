"""Product maturity Phase 6.6 packaging/configuration/data-safety validation."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = Path("pyproject.toml")
README = Path("README.md")
RECOVERY_DOC = Path("Docs/Development/release-recovery-setup.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_6_README = Path("Docs/superpowers/qa/product-maturity/phase-6/README.md")
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-6/2026-05-16-phase-6-6-packaging-config-data-safety.md"
)
CONFIG = Path("tldw_chatbook/config.py")
CHACHANOTES_DB = Path("tldw_chatbook/DB/ChaChaNotes_DB.py")
MEDIA_DB = Path("tldw_chatbook/DB/Client_Media_DB_v2.py")
TASK_13 = Path(
    "backlog/tasks/task-13 - Product-Maturity-Phase-6-Release-Hardening-And-Documentation.md"
)
TASK_13_6 = Path(
    "backlog/tasks/task-13.6 - Phase-6.6-Packaging-configuration-and-data-safety-validation.md"
)

REQUIRED_VALIDATION_AREAS = {
    "packaging",
    "configuration",
    "migration",
    "data-safety",
}
LOCAL_PATH_PREFIXES = (
    "/Users/",
    "/home/",
    "/var/home/",
    "/private/var/folders/",
    "C:\\Users\\",
    "C:/Users/",
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _metadata(text: str) -> dict:
    match = re.search(
        r"<!-- PHASE_6_6_PACKAGING_DATA_SAFETY_METADATA:BEGIN -->\s*```json\s*(.*?)\s*```\s*"
        r"<!-- PHASE_6_6_PACKAGING_DATA_SAFETY_METADATA:END -->",
        text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _markdown_table_row(markdown: str, first_cell_text: str) -> list[str]:
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or first_cell_text not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] == first_cell_text:
            return cells
    raise AssertionError(f"Missing markdown table row for {first_cell_text!r}")


def _validation_matrix_rows(evidence: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for raw_line in evidence.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] in REQUIRED_VALIDATION_AREAS:
            rows[cells[0]] = cells
    return rows


def _assert_no_local_path_prefixes(text: str) -> None:
    leaked_prefixes = [prefix for prefix in LOCAL_PATH_PREFIXES if prefix in text]
    assert not leaked_prefixes, (
        f"evidence contains local filesystem prefix(es): {leaked_prefixes}"
    )


@pytest.mark.parametrize("inherited_root", [False, True])
def test_ui_collect_only_does_not_write_to_caller_home(
    tmp_path,
    inherited_root,
) -> None:
    caller_home = tmp_path / "caller-home"
    caller_home.mkdir(mode=0o700)
    env = os.environ.copy()
    env["HOME"] = str(caller_home)
    env.pop("TMPDIR", None)
    for name in (
        "TLDW_CONFIG_PATH",
        "TLDW_TEST_CONFIG_ROOT",
        "TLDW_TEST_CONFIG_ROOT_OWNER",
    ):
        env.pop(name, None)
    if inherited_root:
        trusted_root = tmp_path / "trusted-root"
        trusted_root.mkdir(mode=0o700)
        root_alias = tmp_path / "trusted-root-alias"
        root_alias.symlink_to(trusted_root, target_is_directory=True)
        env["TLDW_TEST_CONFIG_ROOT"] = str(root_alias)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "Tests/UI/test_tools_settings_window.py",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(caller_home.iterdir()) == []


def test_mixed_root_and_ui_nodes_share_config_fixture_setup_safely(
    tmp_path,
) -> None:
    caller_home = tmp_path / "caller-home"
    caller_home.mkdir(mode=0o700)
    env = os.environ.copy()
    env["HOME"] = str(caller_home)
    env.pop("TMPDIR", None)
    for name in (
        "TLDW_CONFIG_PATH",
        "TLDW_TEST_CONFIG_ROOT",
        "TLDW_TEST_CONFIG_ROOT_OWNER",
    ):
        env.pop(name, None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            (
                "Tests/test_config_private_bootstrap.py"
                "::test_first_config_creation_is_private"
            ),
            (
                "Tests/UI/test_product_maturity_phase6_packaging_data_safety.py"
                "::test_phase6_packaging_config_and_data_safety_source_seams_are_present"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(caller_home.iterdir()) == []


@pytest.mark.parametrize("inherited_root", [False, True])
def test_mixed_conftests_restore_caller_environment_and_cleanup_owned_root(
    tmp_path,
    inherited_root,
) -> None:
    plugin_dir = tmp_path / "probe-plugin"
    plugin_dir.mkdir(mode=0o700)
    report_path = tmp_path / "fixture-report.json"
    plugin_path = plugin_dir / "mixed_fixture_probe.py"
    plugin_path.write_text(
        """
import json
import os
from pathlib import Path

_ENV_NAMES = (
    "HOME",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "TLDW_CONFIG_PATH",
    "TLDW_TEST_CONFIG_ROOT",
    "TLDW_TEST_CONFIG_ROOT_OWNER",
)
_bootstrap_root = None
_hook_order = {}


def pytest_sessionstart(session):
    global _bootstrap_root, _hook_order
    _bootstrap_root = os.environ.get("TLDW_TEST_CONFIG_ROOT")
    for implementation in (
        session.config.pluginmanager.hook.pytest_sessionfinish.get_hookimpls()
    ):
        plugin_file = getattr(implementation.plugin, "__file__", None)
        plugin_path = Path(plugin_file).as_posix() if plugin_file else ""
        if plugin_path.endswith("/Tests/conftest.py"):
            _hook_order["root"] = {
                "tryfirst": implementation.tryfirst,
                "trylast": implementation.trylast,
            }
        elif plugin_path.endswith("/Tests/UI/conftest.py"):
            _hook_order["ui"] = {
                "tryfirst": implementation.tryfirst,
                "trylast": implementation.trylast,
            }


def pytest_unconfigure(config):
    sentinel = os.environ.get("TLDW_FIXTURE_SENTINEL")
    payload = {
        "env": {name: os.environ.get(name) for name in _ENV_NAMES},
        "bootstrap_root": _bootstrap_root,
        "bootstrap_exists": (
            Path(_bootstrap_root).exists() if _bootstrap_root else None
        ),
        "sentinel_exists": Path(sentinel).exists() if sentinel else None,
        "hook_order": _hook_order,
    }
    Path(os.environ["TLDW_FIXTURE_REPORT"]).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
""".lstrip(),
        encoding="utf-8",
    )

    caller_home = tmp_path / "caller-home"
    caller_data = tmp_path / "caller-data"
    caller_config_home = tmp_path / "caller-config-home"
    for directory in (caller_home, caller_data, caller_config_home):
        directory.mkdir(mode=0o700)
    previous_env = {
        "HOME": str(caller_home),
        "XDG_DATA_HOME": str(caller_data),
        "XDG_CONFIG_HOME": str(caller_config_home),
        "TLDW_CONFIG_PATH": str(caller_config_home / "caller.toml"),
        "TLDW_TEST_CONFIG_ROOT": None,
        "TLDW_TEST_CONFIG_ROOT_OWNER": None,
    }

    env = os.environ.copy()
    env.pop("TMPDIR", None)
    env.update(
        {name: value for name, value in previous_env.items() if value is not None}
    )
    env.pop("TLDW_TEST_CONFIG_ROOT", None)
    env.pop("TLDW_TEST_CONFIG_ROOT_OWNER", None)
    env["TLDW_FIXTURE_REPORT"] = str(report_path)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(plugin_dir), env.get("PYTHONPATH")) if value
    )

    sentinel = None
    if inherited_root:
        external_root = tmp_path / "external-root"
        external_root.mkdir(mode=0o700)
        sentinel = external_root / "sentinel.txt"
        sentinel.write_text("preserve me", encoding="utf-8")
        root_alias = tmp_path / "external-root-alias"
        root_alias.symlink_to(external_root, target_is_directory=True)
        previous_env["TLDW_TEST_CONFIG_ROOT"] = str(root_alias)
        previous_env["TLDW_TEST_CONFIG_ROOT_OWNER"] = "caller-owned"
        env["TLDW_TEST_CONFIG_ROOT"] = str(root_alias)
        env["TLDW_TEST_CONFIG_ROOT_OWNER"] = "caller-owned"
        env["TLDW_FIXTURE_SENTINEL"] = str(sentinel)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "mixed_fixture_probe",
            (
                "Tests/test_config_private_bootstrap.py"
                "::test_first_config_creation_is_private"
            ),
            (
                "Tests/UI/test_product_maturity_phase6_packaging_data_safety.py"
                "::test_phase6_packaging_config_and_data_safety_source_seams_are_present"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["env"] == previous_env
    assert report["bootstrap_exists"] is inherited_root
    assert report["sentinel_exists"] is (True if inherited_root else None)
    assert report["hook_order"] == {
        "root": {"tryfirst": False, "trylast": True},
        "ui": {"tryfirst": True, "trylast": False},
    }
    if sentinel is not None:
        assert sentinel.read_text(encoding="utf-8") == "preserve me"


def test_root_conftest_restores_inherited_bootstrap_environment(
    monkeypatch,
    tmp_path,
) -> None:
    trusted_root = tmp_path / "trusted-root"
    trusted_root.mkdir(mode=0o700)
    root_alias = tmp_path / "trusted-root-alias"
    root_alias.symlink_to(trusted_root, target_is_directory=True)
    previous_env = {
        "TLDW_CONFIG_PATH": "caller-config.toml",
        "TLDW_TEST_CONFIG_ROOT": str(root_alias),
        "TLDW_TEST_CONFIG_ROOT_OWNER": "caller-owned",
    }
    for name, value in previous_env.items():
        monkeypatch.setenv(name, value)

    spec = importlib.util.spec_from_file_location(
        "_task488_root_conftest_probe",
        REPO_ROOT / "Tests" / "conftest.py",
    )
    assert spec is not None
    assert spec.loader is not None
    root_conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_conftest)

    assert os.environ["TLDW_TEST_CONFIG_ROOT"] == str(trusted_root.resolve())
    assert os.environ["TLDW_CONFIG_PATH"] == str(
        trusted_root.resolve() / "config" / "config.toml"
    )

    root_conftest.pytest_sessionfinish(None, 0)

    assert {name: os.environ.get(name) for name in previous_env} == previous_env


def test_phase6_packaging_config_and_data_safety_source_seams_are_present() -> None:
    pyproject = tomllib.loads(_text(PYPROJECT))
    readme = _text(README)
    recovery_doc = _text(RECOVERY_DOC)
    config = _text(CONFIG)
    chachanotes_db = _text(CHACHANOTES_DB)
    media_db = _text(MEDIA_DB)

    project = pyproject["project"]
    assert project["name"] == "tldw_chatbook"
    assert project["requires-python"] == ">=3.11"
    assert "textual>=3.3.0" in project["dependencies"]
    assert "tldw-cli" in project["scripts"]
    # The supported launcher is the lightweight shim in cli.py (it defers the
    # heavy app import until invocation); app:main_cli_runner remains the
    # underlying runner the shim delegates to.
    assert project["scripts"]["tldw-cli"] == "tldw_chatbook.cli:main_cli_runner"
    assert "tldw-serve" in project["scripts"]
    assert project["scripts"]["tldw-serve"] == "tldw_chatbook.Web_Server.serve:main"
    for extra in ("dev", "embeddings_rag", "mcp", "web"):
        assert extra in project["optional-dependencies"]

    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "tldw_chatbook.css" in package_data
    assert "tldw_chatbook.Config_Files" in package_data

    for required_copy in (
        "Local-first baseline",
        "Advanced optional capability groups",
        "python3 -m venv .venv",
        "pip install -e .",
        'pip install -e ".[dev]"',
        'pip install "tldw_chatbook[embeddings_rag]"',
        "tldw-cli",
        "tldw-serve",
        "Configuration File",
        "Environment Variables",
    ):
        assert required_copy in readme

    for optional_area in (
        "RAG and retrieval",
        "Media ingestion and transcription",
        "MCP integration",
        "Local inference",
        "Web access",
    ):
        assert optional_area in readme

    assert "TLDW_CONFIG_PATH" in config
    assert "_get_effective_config_path" in config
    assert "_CONFIG_CACHE_SOURCE == config_path" in config
    assert "atomic_write_text(DEFAULT_CONFIG_PATH" in config
    assert "Do not use machine-specific absolute paths" in recovery_doc

    for required_migration_signal in (
        "db_schema_version",
        "_initialize_schema",
        "migration_steps",
        "_migrate_from_v15_to_v16",
        "SchemaError",
        "backup_database",
        "check_integrity",
        "transaction",
        "rollback",
    ):
        assert required_migration_signal in chachanotes_db
    assert "PRAGMA foreign_keys = ON" in chachanotes_db
    assert "PRAGMA journal_mode=WAL" in chachanotes_db

    for required_media_signal in (
        "schema_version",
        "_initialize_schema",
        "backup_database",
        "check_integrity",
        "transaction",
    ):
        assert required_media_signal in media_db
