#!/usr/bin/env python3
"""Normalize and aggregate bounded TASK-601 native process-tree evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
EVIDENCE_LABEL = "task601_native_process_tree"
AGGREGATE_LABEL = "task601_native_process_tree_matrix"
EXPECTED_REPOSITORY = "rmusser01/tldw_chatbook"
REQUIRED_NODES = (
    "Tests/STT/test_executor_process_tree.py::test_native_force_stop_removes_worker_and_descendant_before_scratch_cleanup",
    "Tests/STT/test_executor_process_tree.py::test_native_crashed_leader_reaps_descendant_before_scratch_cleanup",
    "Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch",
)
EXPECTED_PLATFORMS = {
    "linux-x86_64": ("Linux", "x86_64"),
    "windows-x86_64": ("Windows", "x86_64"),
    "macos-x86_64": ("Darwin", "x86_64"),
}

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_DIGITS = re.compile(r"[0-9]{1,20}\Z")
_PYTHON_VERSION = re.compile(r"3\.12\.[0-9]+\Z")
_PYTEST_OUTCOMES = frozenset({"success", "failure", "cancelled", "skipped"})
_NODE_OUTCOMES = frozenset({"passed", "failed", "skipped"})
_FAILURE_CODES = frozenset({"not_run", "dependency_install", "test_execution"})
_FAILURE_STAGES = frozenset({"initialize", "dependency_install", "test_execution"})
_MAX_DURATION_SECONDS = 1_200.0
_MAX_STRING_LENGTH = 256
_PRIVATE_KEYS = frozenset(
    {
        "cmd",
        "command",
        "exception",
        "file",
        "handle",
        "path",
        "pid",
        "process_id",
        "process_group_id",
        "traceback",
        "user",
        "user_name",
        "username",
    }
)
_NODE_BY_TESTCASE = {
    (
        path.removesuffix(".py").replace("/", "."),
        name,
    ): node
    for node in REQUIRED_NODES
    for path, name in (node.split("::", 1),)
}
_REQUIRED_TEST_NAMES = frozenset(name for _, name in _NODE_BY_TESTCASE)


def _run_url(run_id: str) -> str:
    return f"https://github.com/{EXPECTED_REPOSITORY}/actions/runs/{run_id}"


def current_run_identity() -> dict[str, str]:
    """Return the checked-out commit and bounded GitHub workflow identity.

    Returns:
        The tested commit and canonical workflow-run fields.

    Raises:
        ValueError: If Git or the workflow environment lacks a valid identity.
    """

    try:
        tested_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("git could not identify the checked-out commit") from error
    if not _SHA.fullmatch(tested_commit):
        raise ValueError("git returned an invalid checked-out commit")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not _DIGITS.fullmatch(run_id) or not _DIGITS.fullmatch(attempt):
        raise ValueError("workflow run ID and attempt must be numeric")
    return {
        "tested_commit": tested_commit,
        "workflow_run_id": run_id,
        "workflow_run_attempt": attempt,
        "workflow_run_url": _run_url(run_id),
    }


def _host_result(evidence_name: str) -> dict[str, str]:
    expected = EXPECTED_PLATFORMS.get(evidence_name)
    if expected is None:
        raise ValueError("unknown evidence_name")
    system = platform.system()
    machine = platform.machine()
    architecture = "x86_64" if system == "Windows" and machine == "AMD64" else machine
    if (system, architecture) != expected:
        raise ValueError("evidence_name does not match the executing host")
    return {
        "system": system,
        "architecture": architecture,
        "python": platform.python_version(),
    }


def failure_result(
    run_identity: Mapping[str, str],
    *,
    evidence_name: str,
    failure_code: str,
    failure_stage: str,
    pytest_outcome: str = "skipped",
    duration_seconds: float = 0.0,
    required_nodes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return a bounded failed result without exception text or local paths.

    Args:
        run_identity: Canonical commit and workflow-run fields.
        evidence_name: Allowlisted platform evidence name.
        failure_code: Stable failure category.
        failure_stage: Stable workflow stage.
        pytest_outcome: Bounded pytest process outcome.
        duration_seconds: Bounded pytest duration.
        required_nodes: Outcomes already known for required nodes.

    Returns:
        A validated failed platform-evidence document.

    Raises:
        ValueError: If any supplied evidence field is invalid.
    """

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": EVIDENCE_LABEL,
        "evidence_name": evidence_name,
        "status": "failed",
        "failure_code": failure_code,
        "failure_stage": failure_stage,
        "run": dict(run_identity),
        "host": _host_result(evidence_name),
        "pytest": {
            "outcome": pytest_outcome,
            "duration_seconds": duration_seconds,
            "required_nodes": dict(required_nodes or {}),
        },
    }
    validate_result(result, require_pass=False)
    return result


def _duration(root: ET.Element) -> float:
    raw = root.get("time")
    if raw is None:
        value = sum(float(case.get("time", "0")) for case in root.iter("testcase"))
    else:
        value = float(raw)
    if not 0 <= value <= _MAX_DURATION_SECONDS:
        raise ValueError("JUnit duration is outside the evidence bound")
    return value


def result_from_junit(
    path: Path,
    *,
    pytest_outcome: str,
    evidence_name: str,
) -> dict[str, object]:
    """Normalize selected JUnit cases without copying file or failure details.

    Args:
        path: JUnit XML file produced by the bounded pytest run.
        pytest_outcome: Captured pytest process outcome.
        evidence_name: Allowlisted platform evidence name.

    Returns:
        A validated bounded platform-evidence document.

    Raises:
        ValueError: If the run identity or platform identity is invalid.
    """

    run_identity = current_run_identity()
    if pytest_outcome not in _PYTEST_OUTCOMES:
        raise ValueError("pytest outcome is not allowlisted")
    required: dict[str, str] = {}
    duration_seconds = 0.0
    selected_failure = False
    duplicate = False
    unexpected_required_identity = False
    try:
        root = ET.parse(path).getroot()
        duration_seconds = _duration(root)
        selected_failure = any(item.tag in {"failure", "error"} for item in root.iter())
        counts = dict.fromkeys(REQUIRED_NODES, 0)
        for case in root.iter("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            node = _NODE_BY_TESTCASE.get((classname, name))
            if node is None:
                if name in _REQUIRED_TEST_NAMES or any(
                    name.startswith(f"{required_name}[")
                    for required_name in _REQUIRED_TEST_NAMES
                ):
                    unexpected_required_identity = True
                continue
            counts[node] += 1
            if case.find("failure") is not None or case.find("error") is not None:
                required[node] = "failed"
            elif case.find("skipped") is not None:
                required[node] = "skipped"
            else:
                required[node] = "passed"
        duplicate = any(count != 1 for count in counts.values())
    except (OSError, UnicodeError, ET.ParseError, TypeError, ValueError):
        return failure_result(
            run_identity,
            evidence_name=evidence_name,
            failure_code="test_execution",
            failure_stage="test_execution",
            pytest_outcome=pytest_outcome,
        )

    passed = (
        pytest_outcome == "success"
        and not selected_failure
        and not duplicate
        and not unexpected_required_identity
        and required == dict.fromkeys(REQUIRED_NODES, "passed")
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": EVIDENCE_LABEL,
        "evidence_name": evidence_name,
        "status": "passed" if passed else "failed",
        "failure_code": None if passed else "test_execution",
        "failure_stage": None if passed else "test_execution",
        "run": run_identity,
        "host": _host_result(evidence_name),
        "pytest": {
            "outcome": pytest_outcome,
            "duration_seconds": duration_seconds,
            "required_nodes": required,
        },
    }
    validate_result(result, require_pass=False)
    return result


def _require_fields(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    missing = expected - set(value)
    unexpected = set(value) - expected
    if missing:
        raise ValueError(f"{context} is missing fields")
    if unexpected:
        raise ValueError(f"{context} contains an unknown field")


def _require_object(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _reject_private_strings(value: object) -> None:
    if isinstance(value, str):
        if len(value) > _MAX_STRING_LENGTH:
            raise ValueError("evidence contains an unbounded string")
        if value in REQUIRED_NODES or re.fullmatch(
            rf"https://github\.com/{re.escape(EXPECTED_REPOSITORY)}/actions/runs/[0-9]{{1,20}}",
            value,
        ):
            return
        if "/" in value or "\\" in value:
            raise ValueError(
                "evidence contains path-like or noncanonical slash content"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("evidence object keys must be strings")
            if key.lower() in _PRIVATE_KEYS:
                raise ValueError("evidence contains a private process or user field")
            _reject_private_strings(key)
            _reject_private_strings(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_private_strings(item)


def _validate_run(run: Mapping[str, object], *, include_attempt: bool) -> None:
    expected = {"tested_commit", "workflow_run_id", "workflow_run_url"}
    if include_attempt:
        expected.add("workflow_run_attempt")
    _require_fields(run, expected, "run")
    commit = run.get("tested_commit")
    run_id = run.get("workflow_run_id")
    if not isinstance(commit, str) or not _SHA.fullmatch(commit):
        raise ValueError(
            "run.tested_commit must be 40 lowercase hexadecimal characters"
        )
    if not isinstance(run_id, str) or not _DIGITS.fullmatch(run_id):
        raise ValueError("run.workflow_run_id must be a digits string")
    if include_attempt:
        attempt = run.get("workflow_run_attempt")
        if not isinstance(attempt, str) or not _DIGITS.fullmatch(attempt):
            raise ValueError("run.workflow_run_attempt must be a digits string")
    if run.get("workflow_run_url") != _run_url(run_id):
        raise ValueError("run.workflow_run_url must be the canonical repository URL")


def _validate_host(host: Mapping[str, object], evidence_name: object) -> None:
    _require_fields(host, {"system", "architecture", "python"}, "host")
    if not isinstance(evidence_name, str) or evidence_name not in EXPECTED_PLATFORMS:
        raise ValueError("evidence_name is not allowlisted")
    expected_system, expected_architecture = EXPECTED_PLATFORMS[evidence_name]
    if (
        host.get("system") != expected_system
        or host.get("architecture") != expected_architecture
    ):
        raise ValueError("host does not match evidence_name")
    python_version = host.get("python")
    if not isinstance(python_version, str) or not _PYTHON_VERSION.fullmatch(
        python_version
    ):
        raise ValueError("host.python must identify Python 3.12")


def validate_result(result: Mapping[str, object], *, require_pass: bool = True) -> None:
    """Validate one exact platform result and optionally require release success.

    Args:
        result: Candidate platform-evidence document.
        require_pass: Whether failed-but-well-formed evidence must be rejected.

    Raises:
        ValueError: If the document violates the schema or required outcome.
    """

    if not isinstance(result, Mapping):
        raise ValueError("evidence must be an object")
    _reject_private_strings(result)
    _require_fields(
        result,
        {
            "schema_version",
            "evidence_label",
            "evidence_name",
            "status",
            "failure_code",
            "failure_stage",
            "run",
            "host",
            "pytest",
        },
        "result",
    )
    if (
        type(result.get("schema_version")) is not int
        or result.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported schema_version")
    if result.get("evidence_label") != EVIDENCE_LABEL:
        raise ValueError("unexpected evidence_label")
    status = result.get("status")
    if status not in {"passed", "failed"}:
        raise ValueError("status must be passed or failed")
    evidence_name = result.get("evidence_name")
    run = _require_object(result, "run")
    _validate_run(run, include_attempt=True)
    _validate_host(_require_object(result, "host"), evidence_name)

    pytest_result = _require_object(result, "pytest")
    _require_fields(
        pytest_result,
        {"outcome", "duration_seconds", "required_nodes"},
        "pytest",
    )
    outcome = pytest_result.get("outcome")
    if outcome not in _PYTEST_OUTCOMES:
        raise ValueError("pytest.outcome is not allowlisted")
    duration = pytest_result.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not 0 <= duration <= _MAX_DURATION_SECONDS
    ):
        raise ValueError("pytest.duration_seconds is outside the evidence bound")
    nodes = _require_object(pytest_result, "required_nodes")
    if not set(nodes).issubset(REQUIRED_NODES):
        raise ValueError("pytest.required_nodes contains an unknown node")
    if any(node_outcome not in _NODE_OUTCOMES for node_outcome in nodes.values()):
        raise ValueError("pytest.required_nodes contains an invalid outcome")

    if status == "passed":
        if (
            result.get("failure_code") is not None
            or result.get("failure_stage") is not None
        ):
            raise ValueError("passed evidence cannot contain failure fields")
        if outcome != "success":
            raise ValueError("passed evidence requires successful pytest outcome")
        if dict(nodes) != dict.fromkeys(REQUIRED_NODES, "passed"):
            raise ValueError("every required node must be passed")
    else:
        if result.get("failure_code") not in _FAILURE_CODES:
            raise ValueError("failure_code must be a stable allowlisted code")
        if result.get("failure_stage") not in _FAILURE_STAGES:
            raise ValueError("failure_stage must be a stable allowlisted stage")
        if require_pass:
            raise ValueError("TASK-601 evidence did not pass")


def aggregate_results(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate exactly one passing document for each required native platform.

    Args:
        results: Passing platform documents from one workflow run and commit.

    Returns:
        The validated three-platform aggregate.

    Raises:
        ValueError: If platform coverage or run identity is inconsistent.
    """

    if len(results) != len(EXPECTED_PLATFORMS):
        raise ValueError("aggregate requires exactly three platform results")
    by_name: dict[str, Mapping[str, object]] = {}
    for result in results:
        validate_result(result)
        name = result.get("evidence_name")
        if not isinstance(name, str) or name in by_name:
            raise ValueError("aggregate contains duplicate platform evidence")
        by_name[name] = result
    if set(by_name) != set(EXPECTED_PLATFORMS):
        raise ValueError("aggregate platforms must be exact")

    first_run = _require_object(next(iter(by_name.values())), "run")
    commit = first_run["tested_commit"]
    run_id = first_run["workflow_run_id"]
    run_url = first_run["workflow_run_url"]

    aggregate: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": AGGREGATE_LABEL,
        "status": "passed",
        "run": {
            "tested_commit": commit,
            "workflow_run_id": run_id,
            "workflow_run_url": run_url,
        },
        "platforms": {name: by_name[name] for name in EXPECTED_PLATFORMS},
    }
    validate_aggregate(aggregate)
    return aggregate


def validate_aggregate(aggregate: Mapping[str, object]) -> None:
    """Validate the strict same-commit, same-run three-platform aggregate.

    Args:
        aggregate: Candidate aggregate evidence document.

    Raises:
        ValueError: If schema, platform coverage, or run identity is invalid.
    """

    if not isinstance(aggregate, Mapping):
        raise ValueError("aggregate evidence must be an object")
    _reject_private_strings(aggregate)
    _require_fields(
        aggregate,
        {"schema_version", "evidence_label", "status", "run", "platforms"},
        "aggregate",
    )
    if (
        type(aggregate.get("schema_version")) is not int
        or aggregate.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported aggregate schema_version")
    if aggregate.get("evidence_label") != AGGREGATE_LABEL:
        raise ValueError("unexpected aggregate evidence_label")
    if aggregate.get("status") != "passed":
        raise ValueError("aggregate status must be passed")
    run = _require_object(aggregate, "run")
    _validate_run(run, include_attempt=False)
    platforms = _require_object(aggregate, "platforms")
    if set(platforms) != set(EXPECTED_PLATFORMS):
        raise ValueError("aggregate platforms must be exact")
    for name in EXPECTED_PLATFORMS:
        result = platforms.get(name)
        if not isinstance(result, Mapping):
            raise ValueError("aggregate platform must be an object")
        validate_result(result)
        if result.get("evidence_name") != name:
            raise ValueError("aggregate platform key does not match evidence_name")
        platform_run = _require_object(result, "run")
        if platform_run.get("tested_commit") != run.get("tested_commit"):
            raise ValueError("aggregate tested commit does not match platform evidence")
        if platform_run.get("workflow_run_id") != run.get(
            "workflow_run_id"
        ) or platform_run.get("workflow_run_url") != run.get("workflow_run_url"):
            raise ValueError("aggregate workflow run does not match platform evidence")


def _write_result(output: Path, result: Mapping[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)


def _load_result(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("evidence must be an object")
    return loaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--initialize", action="store_true")
    operation.add_argument("--record-failure")
    operation.add_argument("--from-junit", type=Path)
    operation.add_argument("--validate", type=Path)
    operation.add_argument("--aggregate", nargs=3, type=Path)
    operation.add_argument("--validate-aggregate", type=Path)
    parser.add_argument("--failure-stage")
    parser.add_argument("--evidence-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pytest-outcome")
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.initialize:
        required = (args.evidence_name, args.output)
        forbidden = (args.failure_stage, args.pytest_outcome)
    elif args.record_failure is not None:
        required = (args.failure_stage, args.evidence_name, args.output)
        forbidden = (args.pytest_outcome,)
    elif args.from_junit is not None:
        required = (args.pytest_outcome, args.evidence_name, args.output)
        forbidden = (args.failure_stage,)
    elif args.aggregate is not None:
        required = (args.output,)
        forbidden = (args.failure_stage, args.evidence_name, args.pytest_outcome)
    else:
        required = ()
        forbidden = (
            args.failure_stage,
            args.evidence_name,
            args.output,
            args.pytest_outcome,
        )
    if any(value is None for value in required) or any(
        value is not None for value in forbidden
    ):
        raise ValueError("arguments do not match the selected evidence operation")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evidence normalization, validation, or aggregation operation.

    Args:
        argv: Optional argument vector; defaults to process arguments.

    Returns:
        Zero for a valid completed operation, otherwise one.
    """

    args = _parser().parse_args(argv)
    try:
        _validate_cli_args(args)
        if args.validate is not None:
            validate_result(_load_result(args.validate))
            return 0
        if args.validate_aggregate is not None:
            validate_aggregate(_load_result(args.validate_aggregate))
            return 0
        if args.aggregate is not None:
            aggregate = aggregate_results(
                [_load_result(path) for path in args.aggregate]
            )
            _write_result(args.output, aggregate)
            return 0
        if args.initialize:
            result = failure_result(
                current_run_identity(),
                evidence_name=args.evidence_name,
                failure_code="not_run",
                failure_stage="initialize",
            )
        elif args.record_failure is not None:
            result = failure_result(
                current_run_identity(),
                evidence_name=args.evidence_name,
                failure_code=args.record_failure,
                failure_stage=args.failure_stage,
            )
        elif args.from_junit is not None:
            result = result_from_junit(
                args.from_junit,
                pytest_outcome=args.pytest_outcome,
                evidence_name=args.evidence_name,
            )
        else:
            raise ValueError("one evidence operation is required")
        _write_result(args.output, result)
        return 0
    except (
        OSError,
        UnicodeError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        ValueError,
    ):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
