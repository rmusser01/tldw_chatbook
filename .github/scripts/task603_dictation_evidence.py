#!/usr/bin/env python3
"""Normalize and aggregate bounded TASK-603 dictation evidence."""

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
EVIDENCE_LABEL = "task603_bounded_dictation"
AGGREGATE_LABEL = "task603_bounded_dictation_matrix"
EXPECTED_REPOSITORY = "rmusser01/tldw_chatbook"
REQUIRED_NODES = (
    "Tests/STT/test_dispatch_coordinator.py::test_pcm_byte_limit_is_derived_from_the_single_sixty_second_ceiling",
    "Tests/STT/test_dispatch_coordinator.py::test_waiting_segments_coalesce_into_one_source_with_ordered_boundaries",
    "Tests/STT/test_dispatch_coordinator.py::test_pending_cancel_clears_gate_once_without_preempting_batch",
    "Tests/STT/test_dispatch_coordinator.py::test_processing_thread_exits_within_join_bound_while_dictation_waits",
    "Tests/Library/test_library_ingest_runner.py::test_dictation_reservation_gates_only_heavy_library_work",
    "Tests/Library/test_library_ingest_runner.py::test_library_terminal_hands_executor_to_pending_dictation_before_top_up",
    "Tests/Library/test_library_ingest_runner.py::test_shutdown_cooperatively_cancels_active_dictation_before_executor_close",
    "Tests/STT/test_transcription_service_facade.py::test_parakeet_streaming_reports_unsupported_without_consulting_the_bridge",
    "Tests/UI/test_console_dictation.py::test_console_mic_has_strict_wall_timer_and_visible_limit_transition",
    "Tests/UI/test_console_hands_free_wiring.py::test_hands_free_limit_exits_without_reopen_until_a_physical_mic_press",
)
EXPECTED_PLATFORMS = {
    "linux-x86_64": ("Linux", "x86_64"),
    "linux-aarch64": ("Linux", "aarch64"),
    "windows-x86_64": ("Windows", "x86_64"),
    "macos-arm64": ("Darwin", "arm64"),
    "macos-x86_64": ("Darwin", "x86_64"),
}

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_DIGITS = re.compile(r"[0-9]{1,20}\Z")
_PYTHON_VERSION = re.compile(r"3\.12\.[0-9]+\Z")
_RUN_URL = re.compile(
    rf"https://github\.com/{EXPECTED_REPOSITORY}/actions/runs/[0-9]{{1,20}}\Z"
)
_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/]")
_POSIX_PRIVATE_PATH = re.compile(
    r"/(?:Users|home|private|tmp|var|opt|mnt|workspace|github)(?:/|\Z)"
)
_PYTEST_OUTCOMES = frozenset({"success", "failure", "cancelled", "skipped"})
_NODE_OUTCOMES = frozenset({"passed", "failed", "skipped"})
_FAILURE_CODES = frozenset({"not_run", "dependency_install", "test_execution"})
_FAILURE_STAGES = frozenset({"initialize", "dependency_install", "test_execution"})
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
        "traceback",
        "user",
        "username",
    }
)
_MAX_DURATION_SECONDS = 1_800
_MAX_STRING_LENGTH = 512
_NODE_BY_TESTCASE = {
    (path.removesuffix(".py").replace("/", "."), name): node
    for node in REQUIRED_NODES
    for path, name in (node.split("::", 1),)
}
_REQUIRED_NAMES = frozenset(name for _, name in _NODE_BY_TESTCASE)


def _run_url(run_id: str) -> str:
    return f"https://github.com/{EXPECTED_REPOSITORY}/actions/runs/{run_id}"


def current_run_identity() -> dict[str, str]:
    """Return the exact checked-out commit and GitHub workflow identity."""

    try:
        tested_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("git could not identify the tested commit") from error
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not _SHA.fullmatch(tested_commit):
        raise ValueError("tested commit is invalid")
    if not _DIGITS.fullmatch(run_id) or not _DIGITS.fullmatch(attempt):
        raise ValueError("workflow identity is invalid")
    return {
        "tested_commit": tested_commit,
        "workflow_run_id": run_id,
        "workflow_run_attempt": attempt,
        "workflow_run_url": _run_url(run_id),
    }


def _normalized_architecture(system: str, machine: str) -> str:
    value = machine.lower()
    if value in {"amd64", "x86_64"}:
        return "x86_64"
    if value in {"arm64", "aarch64"}:
        return "arm64" if system == "Darwin" else "aarch64"
    return value


def _host_result(evidence_name: str) -> dict[str, str]:
    expected = EXPECTED_PLATFORMS.get(evidence_name)
    if expected is None:
        raise ValueError("unknown evidence name")
    system = platform.system()
    architecture = _normalized_architecture(system, platform.machine())
    if (system, architecture) != expected:
        raise ValueError("evidence name does not match the executing host")
    return {
        "system": system,
        "architecture": architecture,
        "python": platform.python_version(),
    }


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _duration(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("duration is not numeric")
    if not 0 <= value <= _MAX_DURATION_SECONDS:
        raise ValueError("duration is outside the evidence bound")
    return value


def _validate_run(run: object) -> dict[str, str]:
    if not isinstance(run, dict):
        raise ValueError("run must be an object")
    _exact_keys(
        run,
        {
            "tested_commit",
            "workflow_run_id",
            "workflow_run_attempt",
            "workflow_run_url",
        },
        "run",
    )
    if not all(isinstance(value, str) for value in run.values()):
        raise ValueError("run values must be strings")
    if not _SHA.fullmatch(run["tested_commit"]):
        raise ValueError("tested commit is invalid")
    if not _DIGITS.fullmatch(run["workflow_run_id"]):
        raise ValueError("workflow run ID is invalid")
    if not _DIGITS.fullmatch(run["workflow_run_attempt"]):
        raise ValueError("workflow run attempt is invalid")
    if run["workflow_run_url"] != _run_url(run["workflow_run_id"]):
        raise ValueError("workflow run URL is not canonical")
    return run


def _validate_host(host: object, evidence_name: str) -> None:
    if not isinstance(host, dict):
        raise ValueError("host must be an object")
    _exact_keys(host, {"system", "architecture", "python"}, "host")
    system, architecture = EXPECTED_PLATFORMS[evidence_name]
    if host["system"] != system or host["architecture"] != architecture:
        raise ValueError("host does not match the evidence lane")
    if not isinstance(host["python"], str) or not _PYTHON_VERSION.fullmatch(
        host["python"]
    ):
        raise ValueError("Python version is invalid")


def _validate_privacy(value: object, *, key: str | None = None) -> None:
    if key is not None and key.lower() in _PRIVATE_KEYS:
        raise ValueError("private evidence key is forbidden")
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise ValueError("evidence keys must be strings")
            _validate_privacy(nested_value, key=nested_key)
        return
    if isinstance(value, list):
        for item in value:
            _validate_privacy(item)
        return
    if isinstance(value, str):
        if len(value) > _MAX_STRING_LENGTH:
            raise ValueError("evidence string is too long")
        if value in REQUIRED_NODES or _RUN_URL.fullmatch(value):
            return
        if _WINDOWS_PATH.search(value) or _POSIX_PRIVATE_PATH.search(value):
            raise ValueError("private path is forbidden")


def validate_result(result: object, *, require_pass: bool) -> None:
    """Validate one path-private platform result."""

    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    _exact_keys(
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
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema version is invalid")
    if result["evidence_label"] != EVIDENCE_LABEL:
        raise ValueError("evidence label is invalid")
    evidence_name = result["evidence_name"]
    if not isinstance(evidence_name, str) or evidence_name not in EXPECTED_PLATFORMS:
        raise ValueError("evidence name is invalid")
    status = result["status"]
    if not isinstance(status, str) or status not in {"passed", "failed"}:
        raise ValueError("status is invalid")
    if require_pass and status != "passed":
        raise ValueError("passing evidence is required")
    _validate_run(result["run"])
    _validate_host(result["host"], evidence_name)

    pytest_result = result["pytest"]
    if not isinstance(pytest_result, dict):
        raise ValueError("pytest must be an object")
    _exact_keys(
        pytest_result,
        {"outcome", "duration_seconds", "required_nodes"},
        "pytest",
    )
    outcome = pytest_result["outcome"]
    if not isinstance(outcome, str) or outcome not in _PYTEST_OUTCOMES:
        raise ValueError("pytest outcome is invalid")
    _duration(pytest_result["duration_seconds"])
    required_nodes = pytest_result["required_nodes"]
    if not isinstance(required_nodes, dict):
        raise ValueError("required nodes must be an object")
    if not set(required_nodes).issubset(REQUIRED_NODES):
        raise ValueError("required node identity is invalid")
    if any(
        not isinstance(value, str) or value not in _NODE_OUTCOMES
        for value in required_nodes.values()
    ):
        raise ValueError("required node outcome is invalid")

    if status == "passed":
        if result["failure_code"] is not None or result["failure_stage"] is not None:
            raise ValueError("passed evidence cannot have failure fields")
        if outcome != "success":
            raise ValueError("passed evidence requires successful pytest")
        if required_nodes != dict.fromkeys(REQUIRED_NODES, "passed"):
            raise ValueError("all exact required nodes must pass once")
    else:
        failure_code = result["failure_code"]
        if not isinstance(failure_code, str) or failure_code not in _FAILURE_CODES:
            raise ValueError("failure code is invalid")
        failure_stage = result["failure_stage"]
        if not isinstance(failure_stage, str) or failure_stage not in _FAILURE_STAGES:
            raise ValueError("failure stage is invalid")
    _validate_privacy(result)


def failure_result(
    run_identity: Mapping[str, str],
    *,
    evidence_name: str,
    failure_code: str,
    failure_stage: str,
    pytest_outcome: str = "skipped",
    duration_seconds: int | float = 0,
    required_nodes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a validated failure without exception or filesystem details."""

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


def _junit_duration(root: ET.Element) -> int | float:
    raw = root.get("time")
    value = (
        float(raw)
        if raw is not None
        else sum(float(case.get("time", "0")) for case in root.iter("testcase"))
    )
    return _duration(value)


def result_from_junit(
    path: Path,
    *,
    pytest_outcome: str,
    evidence_name: str,
) -> dict[str, object]:
    """Normalize exact JUnit cases without copying native failure details."""

    run_identity = current_run_identity()
    if pytest_outcome not in _PYTEST_OUTCOMES:
        raise ValueError("pytest outcome is invalid")
    required: dict[str, str] = {}
    duration_seconds: int | float = 0
    malformed = False
    try:
        root = ET.parse(path).getroot()
        duration_seconds = _junit_duration(root)
        counts = dict.fromkeys(REQUIRED_NODES, 0)
        unexpected_required_identity = False
        selected_failure = any(item.tag in {"failure", "error"} for item in root.iter())
        for case in root.iter("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            node = _NODE_BY_TESTCASE.get((classname, name))
            if node is None:
                if name in _REQUIRED_NAMES:
                    unexpected_required_identity = True
                continue
            counts[node] += 1
            if case.find("failure") is not None or case.find("error") is not None:
                required[node] = "failed"
            elif case.find("skipped") is not None:
                required[node] = "skipped"
            else:
                required[node] = "passed"
        malformed = (
            selected_failure
            or unexpected_required_identity
            or any(count != 1 for count in counts.values())
            or required != dict.fromkeys(REQUIRED_NODES, "passed")
        )
    except (OSError, UnicodeError, ET.ParseError, TypeError, ValueError, OverflowError):
        malformed = True
        required = {}
        duration_seconds = 0

    if pytest_outcome != "success" or malformed:
        return failure_result(
            run_identity,
            evidence_name=evidence_name,
            failure_code="test_execution",
            failure_stage="test_execution",
            pytest_outcome=pytest_outcome,
            duration_seconds=duration_seconds,
            required_nodes=required,
        )
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": EVIDENCE_LABEL,
        "evidence_name": evidence_name,
        "status": "passed",
        "failure_code": None,
        "failure_stage": None,
        "run": run_identity,
        "host": _host_result(evidence_name),
        "pytest": {
            "outcome": pytest_outcome,
            "duration_seconds": duration_seconds,
            "required_nodes": required,
        },
    }
    validate_result(result, require_pass=True)
    return result


def validate_aggregate(aggregate: object) -> None:
    """Validate a passing five-platform TASK-603 matrix."""

    if not isinstance(aggregate, dict):
        raise ValueError("aggregate must be an object")
    _exact_keys(
        aggregate,
        {"schema_version", "evidence_label", "status", "run", "platforms"},
        "aggregate",
    )
    if aggregate["schema_version"] != SCHEMA_VERSION:
        raise ValueError("aggregate schema version is invalid")
    if aggregate["evidence_label"] != AGGREGATE_LABEL:
        raise ValueError("aggregate label is invalid")
    if aggregate["status"] != "passed":
        raise ValueError("aggregate must pass")
    run = _validate_run(aggregate["run"])
    platforms = aggregate["platforms"]
    if not isinstance(platforms, dict) or set(platforms) != set(EXPECTED_PLATFORMS):
        raise ValueError("aggregate platform matrix is invalid")
    for name, result in platforms.items():
        validate_result(result, require_pass=True)
        if result["evidence_name"] != name or result["run"] != run:
            raise ValueError("aggregate result identity is inconsistent")
    _validate_privacy(aggregate)


def aggregate_results(paths: Sequence[Path]) -> dict[str, object]:
    """Load and aggregate exactly one passing document for each platform."""

    documents: dict[str, dict[str, object]] = {}
    common_run: dict[str, str] | None = None
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_result(document, require_pass=True)
        name = document["evidence_name"]
        if name in documents:
            raise ValueError("duplicate platform evidence")
        run = document["run"]
        if common_run is None:
            common_run = dict(run)
        elif run != common_run:
            raise ValueError("platform evidence run identity differs")
        documents[name] = document
    if set(documents) != set(EXPECTED_PLATFORMS):
        raise ValueError("platform evidence matrix is incomplete")
    ordered = {name: documents[name] for name in EXPECTED_PLATFORMS}
    aggregate: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": AGGREGATE_LABEL,
        "status": "passed",
        "run": common_run,
        "platforms": ordered,
    }
    validate_aggregate(aggregate)
    return aggregate


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--initialize", action="store_true")
    modes.add_argument("--record-failure", choices=sorted(_FAILURE_CODES - {"not_run"}))
    modes.add_argument("--from-junit", type=Path)
    modes.add_argument("--validate", type=Path)
    modes.add_argument("--aggregate", type=Path, nargs="+")
    parser.add_argument("--failure-stage", choices=sorted(_FAILURE_STAGES))
    parser.add_argument("--pytest-outcome", choices=sorted(_PYTEST_OUTCOMES))
    parser.add_argument("--evidence-name", choices=EXPECTED_PLATFORMS)
    parser.add_argument("--output", type=Path)
    return parser


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} is required")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one strict evidence operation, returning nonzero without traceback."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.initialize:
            _require(args.evidence_name, "evidence name")
            _require(args.output, "output")
            if args.failure_stage is not None or args.pytest_outcome is not None:
                raise ValueError("initialize received unrelated arguments")
            value = failure_result(
                current_run_identity(),
                evidence_name=args.evidence_name,
                failure_code="not_run",
                failure_stage="initialize",
            )
            _write_json(args.output, value)
            return 0
        if args.record_failure is not None:
            _require(args.evidence_name, "evidence name")
            _require(args.failure_stage, "failure stage")
            _require(args.output, "output")
            if args.pytest_outcome is not None:
                raise ValueError("record failure received unrelated arguments")
            value = failure_result(
                current_run_identity(),
                evidence_name=args.evidence_name,
                failure_code=args.record_failure,
                failure_stage=args.failure_stage,
            )
            _write_json(args.output, value)
            return 0
        if args.from_junit is not None:
            _require(args.evidence_name, "evidence name")
            _require(args.pytest_outcome, "pytest outcome")
            _require(args.output, "output")
            if args.failure_stage is not None:
                raise ValueError("JUnit normalization received unrelated arguments")
            value = result_from_junit(
                args.from_junit,
                pytest_outcome=args.pytest_outcome,
                evidence_name=args.evidence_name,
            )
            _write_json(args.output, value)
            return 0
        if args.validate is not None:
            if any(
                value is not None
                for value in (
                    args.evidence_name,
                    args.failure_stage,
                    args.pytest_outcome,
                    args.output,
                )
            ):
                raise ValueError("validation received unrelated arguments")
            value = _read_json(args.validate)
            if (
                isinstance(value, dict)
                and value.get("evidence_label") == AGGREGATE_LABEL
            ):
                validate_aggregate(value)
            else:
                validate_result(value, require_pass=True)
            return 0
        _require(args.output, "output")
        if any(
            value is not None
            for value in (args.evidence_name, args.failure_stage, args.pytest_outcome)
        ):
            raise ValueError("aggregation received unrelated arguments")
        value = aggregate_results(args.aggregate)
        _write_json(args.output, value)
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
