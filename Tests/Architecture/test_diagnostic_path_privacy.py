"""Born-red contracts for the TASK-19864 diagnostic path scanner."""

from __future__ import annotations

import json
import re
from textwrap import dedent

import pytest

from scripts.check_persistent_diagnostic_inventory import (
    render_diff,
    scan_path_diagnostic_candidates,
)


def _has_path_expression_label(labels: list[str], expected: str) -> bool:
    if expected != "row.get(root)":
        return expected in labels
    return any(re.fullmatch(r"row\.get\((['\"])root\1\)", label) for label in labels)


@pytest.mark.parametrize(
    ("source", "filename", "method", "scope", "path_expression"),
    [
        pytest.param(
            'logger.info(f"Opened {file_path}")',
            "f_string.py",
            "info",
            "<module>",
            "file_path",
            id="f-string",
        ),
        pytest.param(
            'logger.info("Opened {}", file_path)',
            "loguru_positional.py",
            "info",
            "<module>",
            "file_path",
            id="loguru-positional",
        ),
        pytest.param(
            'logger.warning("Workspace {root}", root=workspace_root)',
            "loguru_keyword.py",
            "warning",
            "<module>",
            "workspace_root",
            id="loguru-keyword",
        ),
        pytest.param(
            'logger.debug("Opened %s" % file_path)',
            "percent_format.py",
            "debug",
            "<module>",
            "file_path",
            id="percent-format",
        ),
        pytest.param(
            'logger.error("Opened {}".format(file_path))',
            "dot_format.py",
            "error",
            "<module>",
            "file_path",
            id="dot-format",
        ),
        pytest.param(
            dedent(
                """
                def emit(output_path):
                    logger.error(
                        "Output path: {}",
                        output_path,
                    )
                """
            ),
            "multiline.py",
            "error",
            "emit",
            "output_path",
            id="multiline-call",
        ),
        pytest.param(
            'logger.info("Workspace {}", row.get("root"))',
            "mapping_root.py",
            "info",
            "<module>",
            "row.get(root)",
            id="mapping-root-key",
        ),
        pytest.param(
            dedent(
                """
                class Store:
                    def emit(self):
                        logger.error("Database {}", self.db_path_str)
                """
            ),
            "database_path.py",
            "error",
            "Store.emit",
            "self.db_path_str",
            id="database-path-attribute",
        ),
    ],
)
def test_path_shaped_diagnostic_inputs_are_candidates(
    source: str,
    filename: str,
    method: str,
    scope: str,
    path_expression: str,
) -> None:
    candidates = scan_path_diagnostic_candidates(source, filename=filename)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["method"] == method
    assert candidate["scope"] == scope
    assert candidate["status"] == "legacy_unreviewed"
    assert candidate["call_digest"]
    assert _has_path_expression_label(candidate["path_expressions"], path_expression)


@pytest.mark.parametrize(
    ("source", "path_expression"),
    [
        pytest.param(
            dedent(
                """
                def emit(configured):
                    raw = configured or os.getcwd()
                    logger.info("Workspace {}", raw)
                """
            ),
            "raw",
            id="configured-or-cwd",
        ),
        pytest.param(
            dedent(
                """
                def emit(raw):
                    target = validate_path_simple(raw)
                    logger.info("Target {}", target)
                """
            ),
            "target",
            id="validated-path-alias",
        ),
        pytest.param(
            dedent(
                """
                def emit(raw):
                    target = Path(raw)
                    logger.info("Target {}", target)
                """
            ),
            "target",
            id="path-constructor-alias",
        ),
        pytest.param(
            dedent(
                """
                def emit():
                    target = Path.home()
                    logger.info("Target {}", target)
                """
            ),
            "target",
            id="path-home-alias",
        ),
        pytest.param(
            dedent(
                """
                def emit(raw):
                    target = raw.resolve()
                    logger.info("Target {}", target)
                """
            ),
            "target",
            id="resolved-path-alias",
        ),
    ],
)
def test_simple_assignment_taint_reaches_diagnostic_arguments(
    source: str, path_expression: str
) -> None:
    candidates = scan_path_diagnostic_candidates(source, filename="assignment.py")

    assert len(candidates) == 1
    assert path_expression in candidates[0]["path_expressions"]


def test_assignment_taint_reaches_a_scope_local_fixed_point() -> None:
    source = dedent(
        """
        def emit(raw):
            final = intermediate
            intermediate: str = Path(raw)
            logger.info("Target {}", final)
        """
    )

    candidates = scan_path_diagnostic_candidates(source, filename="fixed_point.py")

    assert len(candidates) == 1
    assert candidates[0]["path_expressions"] == ["final"]


def test_assignment_taint_does_not_bleed_between_lexical_scopes() -> None:
    source = dedent(
        """
        def discover():
            value = os.getcwd()

        def emit(value):
            logger.info("Value {}", value)
        """
    )

    assert scan_path_diagnostic_candidates(source, filename="scopes.py") == []


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'logger.info("Path ref {}", content_fingerprint(path))',
            id="content-fingerprint",
        ),
        pytest.param(
            'logger.info(f"Path {redact_user_paths(path)}")',
            id="path-redaction",
        ),
        pytest.param(
            'logger.info("File kind {}", path.suffix)',
            id="suffix",
        ),
        pytest.param(
            'logger.info("Path count {}", len(paths))',
            id="cardinality",
        ),
        pytest.param(
            'logger.error("Failure type {}", type(exc).__name__)',
            id="exception-type",
        ),
    ],
)
def test_safe_path_transforms_are_not_candidates(source: str) -> None:
    assert scan_path_diagnostic_candidates(source, filename="safe.py") == []


def test_path_like_substrings_without_bounded_path_identifiers_are_ignored() -> None:
    source = dedent(
        """
        logger.info("Root cause {}", root_cause)
        logger.info("Directory count {}", directory_count)
        """
    )

    assert scan_path_diagnostic_candidates(source, filename="negative_names.py") == []


def _inventory_with_path_candidates(
    rows: list[dict[str, object]], *, candidate_count: int
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "scope": "tldw_chatbook/**/*.py",
        "classification_rules": {},
        "path_privacy_rules": {},
        "reviewed_exclusions": [],
        "summary": {"path_privacy_candidate_calls": candidate_count},
        "owners": [],
        "persistent_sink_topology": [],
        "path_privacy_candidates": rows,
    }


def test_path_candidate_report_preserves_all_files_and_duplicate_findings() -> None:
    duplicate_source = dedent(
        """
        logger.warning("Workspace root {}", workspace_root)
        logger.warning("Workspace root {}", workspace_root)
        """
    )
    other_source = 'logger.error("Database path {}", database_path)'
    duplicate_candidates = scan_path_diagnostic_candidates(
        duplicate_source, filename="alpha.py"
    )
    other_candidates = scan_path_diagnostic_candidates(other_source, filename="beta.py")

    assert len(duplicate_candidates) == 2
    assert len(other_candidates) == 1
    duplicate_digest = duplicate_candidates[0]["call_digest"]
    assert duplicate_candidates[1]["call_digest"] == duplicate_digest
    other_digest = other_candidates[0]["call_digest"]

    committed = _inventory_with_path_candidates([], candidate_count=0)
    rebuilt = _inventory_with_path_candidates(
        [
            {"path": "alpha.py", "candidates": duplicate_candidates},
            {"path": "beta.py", "candidates": other_candidates},
        ],
        candidate_count=3,
    )
    report = render_diff(json.dumps(committed), rebuilt)

    assert "alpha.py" in report
    assert "beta.py" in report
    assert duplicate_digest in report
    assert other_digest in report
    assert "x2" in report


def test_path_candidate_report_counts_additions_removals_and_changes() -> None:
    old = scan_path_diagnostic_candidates(
        'logger.warning("Old root {}", workspace_root)', filename="old.py"
    )[0]
    new = scan_path_diagnostic_candidates(
        'logger.error("New path {}", output_path)', filename="new.py"
    )[0]
    committed = _inventory_with_path_candidates(
        [
            {"path": "removed.py", "candidates": [old, old]},
            {"path": "changed.py", "candidates": [old]},
            {"path": "multiplicity.py", "candidates": [old, old]},
        ],
        candidate_count=5,
    )
    rebuilt = _inventory_with_path_candidates(
        [
            {"path": "added.py", "candidates": [new, new]},
            {"path": "changed.py", "candidates": [new]},
            {"path": "multiplicity.py", "candidates": [old]},
        ],
        candidate_count=4,
    )

    report = render_diff(json.dumps(committed), rebuilt)

    for path in ("removed.py", "added.py", "changed.py", "multiplicity.py"):
        assert path in report
    assert old["call_digest"] in report
    assert new["call_digest"] in report
    assert "x2" in report
    assert "x2 -> x1" in report
    assert "unresolved" in report
    assert "not approved" in report


def test_path_privacy_rules_are_inventory_metadata() -> None:
    committed = _inventory_with_path_candidates([], candidate_count=0)
    rebuilt = _inventory_with_path_candidates([], candidate_count=0)
    committed["path_privacy_rules"] = {"candidate_status": "old"}
    rebuilt["path_privacy_rules"] = {"candidate_status": "legacy_unreviewed"}

    report = render_diff(json.dumps(committed), rebuilt)

    assert "path_privacy_rules" in report
    assert "legacy_unreviewed" in report
