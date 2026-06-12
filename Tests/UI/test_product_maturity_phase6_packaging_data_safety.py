"""Product maturity Phase 6.6 packaging/configuration/data-safety validation."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path


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
    assert not leaked_prefixes, f"evidence contains local filesystem prefix(es): {leaked_prefixes}"


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
    assert project["scripts"]["tldw-cli"] == "tldw_chatbook.app:main_cli_runner"
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
        "pip install -e \".[dev]\"",
        "pip install \"tldw_chatbook[embeddings_rag]\"",
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


