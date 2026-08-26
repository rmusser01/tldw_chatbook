#!/usr/bin/env python3
"""Validate and aggregate bounded TASK-602 native Parakeet evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
EVIDENCE_LABEL = "task602_native_parakeet"
AGGREGATE_LABEL = "task602_native_parakeet_matrix"
EXPECTED_REPOSITORY = "rmusser01/tldw_chatbook"
EXPECTED_PLATFORMS = {
    "linux-x86_64": ("Linux", "x86_64"),
    "linux-aarch64": ("Linux", "aarch64"),
    "windows-x86_64": ("Windows", "x86_64"),
    "macos-arm64": ("Darwin", "arm64"),
    "macos-x86_64": ("Darwin", "x86_64"),
}
REQUIRED_CHECKS = (
    "package_resolution",
    "runtime_probe",
    "v2_int8_cpu",
    "v3_int8_cpu",
    "long_form_vad",
    "cancellation",
    "batch_reuse",
    "retry_wiring",
)
EXPECTED_PACKAGES = (
    "onnx-asr",
    "onnxruntime",
    "faster-whisper",
    "ctranslate2",
)
EXPECTED_ARTIFACTS = {
    "v2_int8": {
        "artifact_id": "parakeet-v2",
        "revision": "0bbb45a3365852604aef28b538a8f066f4ccaa85-vad-b3e3ee3cce4c",
        "variant": "int8",
    },
    "v3_int8": {
        "artifact_id": "parakeet-v3",
        "revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce-vad-b3e3ee3cce4c",
        "variant": "int8",
    },
    "vad": {
        "artifact_id": "silero-vad",
        "revision": "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
        "variant": "f32",
    },
}
DURATION_FIELDS = (
    "acquisition",
    "v2_int8_cpu",
    "v3_int8_cpu",
    "long_form_vad",
    "total",
)

_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DIGITS = re.compile(r"[0-9]{1,20}\Z")
_PYTHON = re.compile(r"3\.12\.[0-9]+\Z")
_VERSION = re.compile(r"[0-9][0-9A-Za-z.+_-]{0,63}\Z")
_SMOKE_OUTCOMES = frozenset({"success", "failure", "cancelled", "skipped"})
_CHECK_OUTCOMES = frozenset({"passed", "failed"})
_FAILURE_CODES = frozenset(
    {
        "not_run",
        "dependency_install",
        "fixture_download",
        "artifact_acquisition",
        "smoke_execution",
        "cleanup",
    }
)
_FAILURE_STAGES = frozenset(
    {
        "initialize",
        "dependency_install",
        "fixture_download",
        "artifact_acquisition",
        "runtime_smoke",
        "cleanup",
    }
)
_SMOKE_FAILURES = {
    "fixture_download": "fixture_download",
    "artifact_acquisition": "artifact_acquisition",
    "smoke_execution": "runtime_smoke",
    "cleanup": "cleanup",
}
_PRIVATE_KEY_PARTS = (
    "command",
    "credential",
    "exception",
    "handle",
    "password",
    "path",
    "pid",
    "secret",
    "token",
    "traceback",
    "username",
)
_MAX_DURATION = 2_700.0
_MAX_STRING = 256


def _run_url(run_id: str) -> str:
    return f"https://github.com/{EXPECTED_REPOSITORY}/actions/runs/{run_id}"


def current_run_identity() -> dict[str, str]:
    """Return the checked-out commit and canonical GitHub run identity."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("git could not identify the checked-out commit") from error
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not _SHA40.fullmatch(commit):
        raise ValueError("git returned an invalid checked-out commit")
    if not _DIGITS.fullmatch(run_id) or not _DIGITS.fullmatch(attempt):
        raise ValueError("workflow run identity must be numeric")
    return {
        "tested_commit": commit,
        "workflow_run_id": run_id,
        "workflow_run_attempt": attempt,
        "workflow_run_url": _run_url(run_id),
    }


def _host_result(evidence_name: str) -> dict[str, str]:
    expected = EXPECTED_PLATFORMS.get(evidence_name)
    if expected is None:
        raise ValueError("unknown evidence name")
    system = platform.system()
    machine = platform.machine()
    aliases = {"AMD64": "x86_64", "arm64": "arm64", "aarch64": "aarch64"}
    architecture = aliases.get(machine, machine)
    if (system, architecture) != expected:
        raise ValueError("evidence name does not match the executing host")
    return {
        "system": system,
        "architecture": architecture,
        "python": platform.python_version(),
    }


def _exact_mapping(value: object, keys: set[str], name: str) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} schema is not exact")
    return value


def _bounded_duration(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not 0 <= value <= _MAX_DURATION:
        raise ValueError(f"{name} is outside the duration bound")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} is outside the duration bound")
    return float(value)


def _validate_run(value: object) -> Mapping[str, object]:
    run = _exact_mapping(
        value,
        {
            "tested_commit",
            "workflow_run_id",
            "workflow_run_attempt",
            "workflow_run_url",
        },
        "run",
    )
    commit = run["tested_commit"]
    run_id = run["workflow_run_id"]
    attempt = run["workflow_run_attempt"]
    if type(commit) is not str or not _SHA40.fullmatch(commit):
        raise ValueError("tested commit is invalid")
    if type(run_id) is not str or not _DIGITS.fullmatch(run_id):
        raise ValueError("workflow run ID is invalid")
    if type(attempt) is not str or not _DIGITS.fullmatch(attempt):
        raise ValueError("workflow run attempt is invalid")
    if run["workflow_run_url"] != _run_url(run_id):
        raise ValueError("workflow run URL does not match the run ID")
    return run


def _validate_host(value: object, evidence_name: str) -> None:
    host = _exact_mapping(value, {"system", "architecture", "python"}, "host")
    expected = EXPECTED_PLATFORMS.get(evidence_name)
    if expected is None or (host["system"], host["architecture"]) != expected:
        raise ValueError("platform host does not match evidence name")
    python_version = host["python"]
    if type(python_version) is not str or not _PYTHON.fullmatch(python_version):
        raise ValueError("platform Python version is invalid")


def _validate_packages(value: object) -> None:
    packages = _exact_mapping(value, set(EXPECTED_PACKAGES), "packages")
    if packages["onnx-asr"] != "0.12.0":
        raise ValueError("onnx-asr must be the pinned version")
    for name, version in packages.items():
        if type(version) is not str or not _VERSION.fullmatch(version):
            raise ValueError(f"package version is invalid: {name}")


def _validate_reference(value: object, expected: Mapping[str, str]) -> None:
    reference = _exact_mapping(
        value, {"artifact_id", "revision", "variant"}, "artifact reference"
    )
    if reference != expected:
        raise ValueError("artifact reference is not the exact pinned identity")


def _validate_artifacts(value: object) -> None:
    artifacts = _exact_mapping(value, {"v2_int8", "v3_int8", "vad"}, "artifacts")
    for key in ("v2_int8", "v3_int8"):
        root = _exact_mapping(artifacts[key], {"reference", "closure_fingerprint"}, key)
        _validate_reference(root["reference"], EXPECTED_ARTIFACTS[key])
        fingerprint = root["closure_fingerprint"]
        if type(fingerprint) is not str or not _SHA256.fullmatch(fingerprint):
            raise ValueError("closure fingerprint is invalid")
    _validate_reference(artifacts["vad"], EXPECTED_ARTIFACTS["vad"])


def _validate_checks(value: object, *, require_pass: bool) -> None:
    checks = _exact_mapping(value, set(REQUIRED_CHECKS), "checks")
    if any(
        type(outcome) is not str or outcome not in _CHECK_OUTCOMES
        for outcome in checks.values()
    ):
        raise ValueError("check outcome is invalid")
    if require_pass and checks != dict.fromkeys(REQUIRED_CHECKS, "passed"):
        raise ValueError("every required check must pass")


def _validate_durations(value: object) -> None:
    durations = _exact_mapping(value, set(DURATION_FIELDS), "durations")
    for name, duration in durations.items():
        _bounded_duration(duration, str(name))


def _validate_privacy(value: object, *, key: str | None = None) -> None:
    if key is not None and any(part in key.casefold() for part in _PRIVATE_KEY_PARTS):
        raise ValueError("private evidence key is forbidden")
    if isinstance(value, dict):
        for child_key, child in value.items():
            if type(child_key) is not str or len(child_key) > _MAX_STRING:
                raise ValueError("evidence key is invalid")
            _validate_privacy(child, key=child_key)
    elif isinstance(value, list):
        for child in value:
            _validate_privacy(child)
    elif isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise ValueError("evidence string is too long")
        folded = value.casefold()
        if (
            value.startswith(("/", "\\"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
            or "/home/" in folded
            or "/users/" in folded
            or "/private/" in folded
            or "/tmp/" in folded
            or "hf_secret" in folded
        ):
            raise ValueError("private evidence value is forbidden")


def failure_result(
    run_identity: Mapping[str, str],
    *,
    evidence_name: str,
    failure_code: str,
    failure_stage: str,
) -> dict[str, object]:
    """Create a bounded path-private failure document."""

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": EVIDENCE_LABEL,
        "evidence_name": evidence_name,
        "status": "failed",
        "failure_code": failure_code,
        "failure_stage": failure_stage,
        "run": dict(run_identity),
        "host": _host_result(evidence_name),
    }
    validate_result(result, require_pass=False)
    return result


def _validate_failure(result: Mapping[str, object]) -> None:
    _exact_mapping(
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
        },
        "failure evidence",
    )
    if (
        type(result["failure_code"]) is not str
        or result["failure_code"] not in _FAILURE_CODES
    ):
        raise ValueError("failure code is invalid")
    if (
        type(result["failure_stage"]) is not str
        or result["failure_stage"] not in _FAILURE_STAGES
    ):
        raise ValueError("failure stage is invalid")


def validate_result(result: object, *, require_pass: bool = True) -> None:
    """Validate one platform document and optionally require success."""

    if type(result) is not dict:
        raise ValueError("evidence must be an object")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema version is invalid")
    if result.get("evidence_label") != EVIDENCE_LABEL:
        raise ValueError("evidence label is invalid")
    evidence_name = result.get("evidence_name")
    if type(evidence_name) is not str or evidence_name not in EXPECTED_PLATFORMS:
        raise ValueError("evidence name is invalid")
    status = result.get("status")
    if type(status) is not str or status not in {"passed", "failed"}:
        raise ValueError("evidence status is invalid")
    _validate_run(result.get("run"))
    _validate_host(result.get("host"), evidence_name)
    if status == "failed" and "packages" not in result:
        _validate_failure(result)
    else:
        _exact_mapping(
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
                "packages",
                "execution_provider",
                "artifacts",
                "checks",
                "durations_seconds",
                "cleanup",
            },
            "platform evidence",
        )
        if status == "passed":
            if (
                result["failure_code"] is not None
                or result["failure_stage"] is not None
            ):
                raise ValueError("passing evidence cannot contain a failure")
        elif (
            result["failure_code"] != "smoke_execution"
            or result["failure_stage"] != "runtime_smoke"
        ):
            raise ValueError("full failure evidence has an invalid failure identity")
        _validate_packages(result["packages"])
        if result["execution_provider"] != "CPUExecutionProvider":
            raise ValueError("CPUExecutionProvider is required")
        _validate_artifacts(result["artifacts"])
        _validate_checks(result["checks"], require_pass=status == "passed")
        _validate_durations(result["durations_seconds"])
        if result["cleanup"] != "passed":
            raise ValueError("cleanup did not pass")
    _validate_privacy(result)
    if require_pass and status != "passed":
        raise ValueError("platform evidence did not pass")


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("evidence JSON could not be read") from error


def result_from_smoke(
    path: Path,
    *,
    smoke_outcome: str,
    evidence_name: str,
) -> dict[str, object]:
    """Bind a bounded smoke observation to the checked-out workflow identity."""

    if smoke_outcome not in _SMOKE_OUTCOMES:
        raise ValueError("smoke outcome is invalid")
    run = current_run_identity()
    if smoke_outcome != "success":
        try:
            failure = _exact_mapping(
                _load_json(path),
                {"schema_version", "status", "failure_code", "failure_stage"},
                "smoke failure",
            )
            code = failure["failure_code"]
            stage = failure["failure_stage"]
            if (
                failure["schema_version"] != SCHEMA_VERSION
                or failure["status"] != "failed"
                or type(code) is not str
                or type(stage) is not str
                or _SMOKE_FAILURES.get(code) != stage
            ):
                raise ValueError("smoke failure identity is invalid")
            return failure_result(
                run,
                evidence_name=evidence_name,
                failure_code=code,
                failure_stage=stage,
            )
        except ValueError:
            pass
        return failure_result(
            run,
            evidence_name=evidence_name,
            failure_code="smoke_execution",
            failure_stage="runtime_smoke",
        )
    try:
        payload = _load_json(path)
        smoke = _exact_mapping(
            payload,
            {
                "schema_version",
                "status",
                "failure_code",
                "failure_stage",
                "packages",
                "execution_provider",
                "artifacts",
                "checks",
                "durations_seconds",
                "cleanup",
            },
            "smoke payload",
        )
        if smoke["schema_version"] != SCHEMA_VERSION:
            raise ValueError("smoke schema version is invalid")
        if type(smoke["status"]) is not str or smoke["status"] not in {
            "passed",
            "failed",
        }:
            raise ValueError("smoke status is invalid")
        result = {
            "schema_version": SCHEMA_VERSION,
            "evidence_label": EVIDENCE_LABEL,
            "evidence_name": evidence_name,
            "status": "passed",
            "failure_code": None,
            "failure_stage": None,
            "run": run,
            "host": _host_result(evidence_name),
            "packages": smoke["packages"],
            "execution_provider": smoke["execution_provider"],
            "artifacts": smoke["artifacts"],
            "checks": smoke["checks"],
            "durations_seconds": smoke["durations_seconds"],
            "cleanup": smoke["cleanup"],
        }
        checks = result["checks"]
        if smoke["status"] != "passed" or checks != dict.fromkeys(
            REQUIRED_CHECKS, "passed"
        ):
            result["status"] = "failed"
            result["failure_code"] = "smoke_execution"
            result["failure_stage"] = "runtime_smoke"
        validate_result(result, require_pass=False)
        return result
    except ValueError:
        return failure_result(
            run,
            evidence_name=evidence_name,
            failure_code="smoke_execution",
            failure_stage="runtime_smoke",
        )


def aggregate_results(paths: Sequence[Path]) -> dict[str, object]:
    """Validate and aggregate exactly five same-run passed platform results."""

    if len(paths) != len(EXPECTED_PLATFORMS):
        raise ValueError("aggregate requires exactly five platform results")
    platforms: dict[str, object] = {}
    commits: set[str] = set()
    run_ids: set[str] = set()
    attempts: set[str] = set()
    urls: set[str] = set()
    for path in paths:
        result = _load_json(path)
        validate_result(result)
        assert isinstance(result, dict)
        name = result["evidence_name"]
        if name in platforms:
            raise ValueError("aggregate platform is duplicated")
        platforms[name] = result
        run = result["run"]
        assert isinstance(run, dict)
        commits.add(run["tested_commit"])
        run_ids.add(run["workflow_run_id"])
        attempts.add(run["workflow_run_attempt"])
        urls.add(run["workflow_run_url"])
    if set(platforms) != set(EXPECTED_PLATFORMS):
        raise ValueError("aggregate platform set is not exact")
    if len(commits) != 1:
        raise ValueError("aggregate commits do not match")
    if len(run_ids) != 1 or len(attempts) != 1 or len(urls) != 1:
        raise ValueError("aggregate workflow run identity does not match")
    result = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": AGGREGATE_LABEL,
        "status": "passed",
        "tested_commit": next(iter(commits)),
        "workflow_run_id": next(iter(run_ids)),
        "workflow_run_attempt": next(iter(attempts)),
        "workflow_run_url": next(iter(urls)),
        "platforms": {name: platforms[name] for name in sorted(platforms)},
    }
    validate_aggregate(result)
    return result


def validate_aggregate(result: object) -> None:
    """Validate an exact same-run five-platform aggregate."""

    aggregate = _exact_mapping(
        result,
        {
            "schema_version",
            "evidence_label",
            "status",
            "tested_commit",
            "workflow_run_id",
            "workflow_run_attempt",
            "workflow_run_url",
            "platforms",
        },
        "aggregate",
    )
    if aggregate["schema_version"] != SCHEMA_VERSION:
        raise ValueError("aggregate schema version is invalid")
    if aggregate["evidence_label"] != AGGREGATE_LABEL:
        raise ValueError("aggregate label is invalid")
    if aggregate["status"] != "passed":
        raise ValueError("aggregate did not pass")
    platforms = _exact_mapping(
        aggregate["platforms"], set(EXPECTED_PLATFORMS), "aggregate platforms"
    )
    for name, platform_result in platforms.items():
        validate_result(platform_result)
        if platform_result["evidence_name"] != name:
            raise ValueError("aggregate platform key does not match result")
        run = platform_result["run"]
        if (
            run["tested_commit"] != aggregate["tested_commit"]
            or run["workflow_run_id"] != aggregate["workflow_run_id"]
            or run["workflow_run_attempt"] != aggregate["workflow_run_attempt"]
            or run["workflow_run_url"] != aggregate["workflow_run_url"]
        ):
            raise ValueError("aggregate run identity does not match platform")
    _validate_run(
        {
            "tested_commit": aggregate["tested_commit"],
            "workflow_run_id": aggregate["workflow_run_id"],
            "workflow_run_attempt": aggregate["workflow_run_attempt"],
            "workflow_run_url": aggregate["workflow_run_url"],
        }
    )
    _validate_privacy(aggregate)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--initialize", action="store_true")
    modes.add_argument("--record-failure", choices=sorted(_FAILURE_CODES - {"not_run"}))
    modes.add_argument("--from-smoke", type=Path)
    modes.add_argument("--validate", type=Path)
    modes.add_argument("--aggregate", nargs=5, type=Path)
    modes.add_argument("--validate-aggregate", type=Path)
    parser.add_argument("--evidence-name", choices=sorted(EXPECTED_PLATFORMS))
    parser.add_argument("--failure-stage", choices=sorted(_FAILURE_STAGES))
    parser.add_argument("--smoke-outcome", choices=sorted(_SMOKE_OUTCOMES))
    parser.add_argument("--output", type=Path)
    return parser


def _reject_companions(args: argparse.Namespace, names: Sequence[str]) -> None:
    if any(getattr(args, name) is not None for name in names):
        raise ValueError("CLI mode received an irrelevant companion argument")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one strict evidence CLI mode without emitting private tracebacks."""

    try:
        args = _parser().parse_args(argv)
        if args.initialize:
            _reject_companions(
                args,
                ("failure_stage", "smoke_outcome"),
            )
            if args.evidence_name is None or args.output is None:
                raise ValueError("initialize requires evidence name and output")
            value = failure_result(
                current_run_identity(),
                evidence_name=args.evidence_name,
                failure_code="not_run",
                failure_stage="initialize",
            )
            _write_json(args.output, value)
        elif args.record_failure is not None:
            _reject_companions(args, ("smoke_outcome",))
            if (
                args.evidence_name is None
                or args.failure_stage is None
                or args.output is None
            ):
                raise ValueError("record failure requires identity, stage, and output")
            value = failure_result(
                current_run_identity(),
                evidence_name=args.evidence_name,
                failure_code=args.record_failure,
                failure_stage=args.failure_stage,
            )
            _write_json(args.output, value)
        elif args.from_smoke is not None:
            _reject_companions(args, ("failure_stage",))
            if (
                args.evidence_name is None
                or args.smoke_outcome is None
                or args.output is None
            ):
                raise ValueError(
                    "smoke normalization requires identity, outcome, and output"
                )
            value = result_from_smoke(
                args.from_smoke,
                smoke_outcome=args.smoke_outcome,
                evidence_name=args.evidence_name,
            )
            _write_json(args.output, value)
        elif args.validate is not None:
            _reject_companions(
                args,
                ("evidence_name", "failure_stage", "smoke_outcome", "output"),
            )
            validate_result(_load_json(args.validate))
        elif args.aggregate is not None:
            _reject_companions(
                args,
                ("evidence_name", "failure_stage", "smoke_outcome"),
            )
            if args.output is None:
                raise ValueError("aggregation requires output")
            _write_json(args.output, aggregate_results(args.aggregate))
        else:
            _reject_companions(
                args,
                ("evidence_name", "failure_stage", "smoke_outcome", "output"),
            )
            validate_aggregate(_load_json(args.validate_aggregate))
        return 0
    except (OSError, TypeError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
