"""Generate source-bound speculative-voice rollout authority."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from email.parser import Parser
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from zipfile import BadZipFile, ZipFile

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from Packaging.compute_voice_source_digest import compute_voice_source_digest
from Packaging.speculative_voice_history_gate import (
    _locator_distribution,
    completed_pair_locator_inventory,
)
from Packaging.voice_physical_reports import (
    PhysicalVoiceReportError,
    read_physical_report,
)


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATH_LIST = _ROOT / "Packaging/speculative_voice_source_paths.txt"
_SCHEMA_PATH = Path(__file__).with_name("voice_qualification_manifest.schema.json")
_CORPUS_MANIFEST = _ROOT / "Tests/Audio/fixtures/voice_aec/manifest.json"
_OWNER_INVENTORY = (
    _ROOT / "Docs/Development/TTS/speculative-voice-durable-owner-inventory.md"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)?")
_PYTHON_VERSION = re.compile(r"3\.(11|12|13)(?:\.\d+)?")
_WHEEL_PYTHON_TAG = re.compile(r"-(cp3(?:11|12|13))-\1-")
_OWNER_ROW = re.compile(r"^\| `([^`]+)` \|")
PLATFORM_KEYS = (
    "macos-arm64",
    "macos-x86_64",
    "windows-x86_64",
    "linux-x86_64",
    "linux-aarch64",
)
_DEVICE_CLASSES = ("builtin", "usb", "bluetooth")
_FORBIDDEN_KEYS = {
    "audio",
    "capture_body",
    "credential",
    "device_id",
    "device_identifier",
    "device_name",
    "pcm",
    "request_body",
    "response",
    "response_text",
    "transcript",
}
_AUTOMATED_KEYS = {
    "schema_version",
    "scenario",
    "source_tree_digest",
    "interpreter",
    "companion",
    "corpus",
    "latency_distributions",
    "completed_pair_history_gate",
    "lifecycle",
    "durable_owners",
    "passed",
}
_CORPUS_THRESHOLDS = {
    "median_erle_db_min": 20.0,
    "p10_erle_db_min": 10.0,
    "false_barge_events_max": 1,
    "false_barge_render_minutes_min": 30.0,
    "double_talk_recall_min": 0.95,
}
_LATENCY_THRESHOLDS = {
    "eos_to_dispatch_p95_ms_max": 850.0,
    "barge_to_audible_stop_p95_ms_max": 150.0,
    "added_eos_to_replacement_dispatch_p95_ms_max": 850.0,
    "eos_to_first_assistant_audio_median_ms_max": 1_500.0,
    "eos_to_first_assistant_audio_p95_ms_max": 2_500.0,
}
_CLOCK_BOUNDARIES = {
    "eos": "post_aec_admitted_speech_end",
    "dispatch": "attempt_dispatch_handoff",
    "audible_stop": "transport_abort_fence",
    "first_assistant_audio": "transport_playback_started",
}
_HISTORY_THRESHOLDS = {
    "large_p95_ms": 100.0,
    "median_paired_delta_ms": 2.0,
    "p95_paired_delta_ms": 10.0,
}
_LIFECYCLE_TESTS = (
    "Tests/Chat/test_console_voice_attempts.py",
    "Tests/Chat/test_console_voice_effect_barrier.py",
    "Tests/Chat/test_console_voice_capture.py",
    "Tests/integration/test_speculative_voice_pipeline.py",
)
_DURABLE_OWNER_TEST = "Tests/Chat/test_console_voice_ephemerality.py"


class VoiceManifestError(RuntimeError):
    """Raised when rollout evidence or generated authority is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise VoiceManifestError(f"{label} is missing or unsafe")
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VoiceManifestError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise VoiceManifestError(f"{label} must be an object")
    _reject_forbidden_fields(value)
    return value, raw


def _reject_forbidden_fields(value: object, *, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise VoiceManifestError("qualification evidence keys must be strings")
            if key.casefold() in _FORBIDDEN_KEYS:
                raise VoiceManifestError(
                    f"privacy-forbidden qualification field: {path}.{key}"
                )
            _reject_forbidden_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, path=f"{path}[{index}]")


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VoiceManifestError("qualification metric is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise VoiceManifestError("qualification metric is not finite")
    return number


def _percentile(samples: Sequence[float], percentile: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def _validate_sample_summary(value: object, *, count: int) -> list[float]:
    if not isinstance(value, Mapping) or set(value) != {
        "samples_ms",
        "median_ms",
        "p95_ms",
    }:
        raise VoiceManifestError("history gate sample summary is invalid")
    raw_samples = value["samples_ms"]
    if not isinstance(raw_samples, list) or len(raw_samples) != count:
        raise VoiceManifestError("history gate sample count is invalid")
    samples = [_number(sample) for sample in raw_samples]
    expected_median = round(statistics.median(samples), 6)
    expected_p95 = round(_percentile(samples, 0.95), 6)
    if not math.isclose(
        _number(value["median_ms"]), expected_median, abs_tol=1e-6
    ) or not math.isclose(_number(value["p95_ms"]), expected_p95, abs_tol=1e-6):
        raise VoiceManifestError("history gate sample derivation is invalid")
    return samples


def _validate_history_operation(value: object, *, trial_count: int) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "baseline",
        "large",
        "paired_delta",
        "thresholds",
        "passed",
    }:
        raise VoiceManifestError("history gate operation is invalid")
    if value["thresholds"] != _HISTORY_THRESHOLDS:
        raise VoiceManifestError("history gate thresholds changed")
    baseline = _validate_sample_summary(value["baseline"], count=trial_count)
    large = _validate_sample_summary(value["large"], count=trial_count)
    deltas = _validate_sample_summary(value["paired_delta"], count=trial_count)
    if any(
        not math.isclose(delta, large_ms - baseline_ms, abs_tol=2e-6)
        for baseline_ms, large_ms, delta in zip(
            baseline,
            large,
            deltas,
            strict=True,
        )
    ):
        raise VoiceManifestError("history gate paired deltas are invalid")
    derived = bool(
        _percentile(large, 0.95) <= _HISTORY_THRESHOLDS["large_p95_ms"]
        and statistics.median(deltas) <= _HISTORY_THRESHOLDS["median_paired_delta_ms"]
        and _percentile(deltas, 0.95) <= _HISTORY_THRESHOLDS["p95_paired_delta_ms"]
    )
    if value["passed"] is not derived or not derived:
        raise VoiceManifestError("history gate operation did not pass")


def validate_completed_pair_history_gate(report: object) -> None:
    """Independently validate every fixed large-history predicate."""

    try:
        if not isinstance(report, Mapping) or set(report) != {
            "inventory",
            "query_plans",
            "scan_count",
            "row_counts",
            "physical_forbidden_rows",
            "distribution",
            "distribution_sha256",
            "warmup_count",
            "trial_count",
            "measurement_order",
            "sqlite",
            "fresh_commit",
            "uncertain_retry",
            "passed",
        }:
            raise VoiceManifestError("history gate shape is invalid")
        expected_inventory = completed_pair_locator_inventory()
        if report["inventory"] != expected_inventory:
            raise VoiceManifestError("history gate inventory is invalid")
        expected_distribution = _locator_distribution()
        if report["distribution"] != expected_distribution or report[
            "distribution_sha256"
        ] != _canonical_sha256(expected_distribution):
            raise VoiceManifestError("history gate distribution is invalid")
        if report["scan_count"] != 0:
            raise VoiceManifestError("history gate query plan scanned")
        if report["row_counts"] != {"baseline": 1_000, "large": 100_000}:
            raise VoiceManifestError("history gate row counts are invalid")
        physical_rows = report["physical_forbidden_rows"]
        if (
            not isinstance(physical_rows, Mapping)
            or set(physical_rows) != {"baseline", "large"}
            or any(type(item) is not int or item < 1 for item in physical_rows.values())
        ):
            raise VoiceManifestError("history gate physical rows are invalid")
        if (
            report["warmup_count"] != 10
            or report["trial_count"] != 40
            or report["measurement_order"] != "ABBA"
        ):
            raise VoiceManifestError("history gate trial contract is invalid")
        query_plans = report["query_plans"]
        if not isinstance(query_plans, Mapping) or set(query_plans) != {
            "baseline",
            "large",
        }:
            raise VoiceManifestError("history gate query plans are invalid")
        expected_locators = {
            (item["table"], item["column"]) for item in expected_inventory["locators"]
        }
        for size in ("baseline", "large"):
            plans = query_plans[size]
            if not isinstance(plans, list) or len(plans) != len(expected_locators):
                raise VoiceManifestError("history gate query plans are incomplete")
            observed: set[tuple[object, object]] = set()
            for plan in plans:
                if not isinstance(plan, Mapping) or set(plan) != {
                    "table",
                    "column",
                    "details",
                    "uses_search",
                    "uses_scan",
                }:
                    raise VoiceManifestError("history gate query plan is invalid")
                details = plan["details"]
                if (
                    not isinstance(details, list)
                    or not details
                    or any(not isinstance(item, str) for item in details)
                    or plan["uses_search"] is not True
                    or plan["uses_scan"] is not False
                ):
                    raise VoiceManifestError("history gate query plan is unsafe")
                observed.add((plan["table"], plan["column"]))
            if observed != expected_locators:
                raise VoiceManifestError("history gate query plan coverage is invalid")
        sqlite = report["sqlite"]
        if (
            not isinstance(sqlite, Mapping)
            or set(sqlite)
            != {
                "version",
                "pragmas_match",
                "baseline_pragmas",
                "large_pragmas",
            }
            or not isinstance(sqlite["version"], str)
            or sqlite["pragmas_match"] is not True
            or sqlite["baseline_pragmas"] != sqlite["large_pragmas"]
        ):
            raise VoiceManifestError("history gate SQLite identity is invalid")
        _validate_history_operation(report["fresh_commit"], trial_count=40)
        _validate_history_operation(report["uncertain_retry"], trial_count=40)
        if report["passed"] is not True:
            raise VoiceManifestError("history gate did not pass")
    except (KeyError, TypeError, ValueError) as exc:
        raise VoiceManifestError("history gate is malformed") from exc


def _validate_corpus(corpus: object) -> None:
    if not isinstance(corpus, Mapping) or set(corpus) != {
        "schema_version",
        "thresholds",
        "median_erle_db",
        "p10_erle_db",
        "false_barge_events",
        "false_barge_render_minutes",
        "double_talk_recall",
        "case_results",
        "unqualified_case_ids",
        "effective_mode",
        "passed",
    }:
        raise VoiceManifestError("automated corpus shape is invalid")
    _validate_corpus_cases(corpus["case_results"])
    passed = bool(
        corpus["schema_version"] == 1
        and corpus["thresholds"] == _CORPUS_THRESHOLDS
        and _number(corpus["median_erle_db"]) >= 20.0
        and _number(corpus["p10_erle_db"]) >= 10.0
        and type(corpus["false_barge_events"]) is int
        and 0 <= corpus["false_barge_events"] <= 1
        and _number(corpus["false_barge_render_minutes"]) >= 30.0
        and _number(corpus["double_talk_recall"]) >= 0.95
        and isinstance(corpus["unqualified_case_ids"], list)
        and corpus["unqualified_case_ids"] == []
        and corpus["effective_mode"] == "full-duplex"
    )
    if corpus["passed"] is not passed or not passed:
        raise VoiceManifestError("automated corpus did not pass")


def _validate_corpus_cases(value: object) -> None:
    manifest, _raw = _read_json(_CORPUS_MANIFEST, label="AEC corpus manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise VoiceManifestError("AEC corpus manifest case inventory is invalid")
    expected: set[tuple[str, str]] = set()
    for case in cases:
        if (
            not isinstance(case, Mapping)
            or not isinstance(case.get("id"), str)
            or not case["id"]
            or not isinstance(case.get("kind"), str)
            or not case["kind"]
        ):
            raise VoiceManifestError("AEC corpus manifest case inventory is invalid")
        expected.add((case["id"], case["kind"]))
    if (
        len(expected) != len(cases)
        or not isinstance(value, list)
        or len(value) != len(expected)
    ):
        raise VoiceManifestError("automated corpus case inventory is invalid")
    observed: set[tuple[str, str]] = set()
    for result in value:
        if not isinstance(result, Mapping) or set(result) != {
            "id",
            "kind",
            "passed",
            "error_class",
            "frames",
            "median_erle_db",
            "false_barge_events",
            "double_talk_recall",
        }:
            raise VoiceManifestError("automated corpus case result is invalid")
        if (
            not isinstance(result["id"], str)
            or not result["id"]
            or not isinstance(result["kind"], str)
            or not result["kind"]
        ):
            raise VoiceManifestError("automated corpus case identity is invalid")
        identity = (result["id"], result["kind"])
        if (
            identity in observed
            or result["passed"] is not True
            or result["error_class"] is not None
            or type(result["frames"]) is not int
            or result["frames"] < 1
            or type(result["false_barge_events"]) is not int
            or result["false_barge_events"] < 0
        ):
            raise VoiceManifestError("automated corpus case did not pass")
        _number(result["median_erle_db"])
        _number(result["double_talk_recall"])
        observed.add(identity)
    if observed != expected:
        raise VoiceManifestError("automated corpus case inventory is invalid")


def _validate_pytest_gate(value: object, *, tests: Sequence[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "suite_sha256",
        "test_path_count",
        "exit_code",
        "passed",
    }:
        raise VoiceManifestError("automated pytest gate shape is invalid")
    if value != {
        "suite_sha256": _canonical_sha256({"tests": list(tests)}),
        "test_path_count": len(tests),
        "exit_code": 0,
        "passed": True,
    }:
        raise VoiceManifestError("automated pytest gate did not pass")


def _validate_durable_owners(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "inventory_sha256",
        "owner_ids_sha256",
        "owner_count",
        "probe",
        "report_sha256",
        "passed",
    }:
        raise VoiceManifestError("automated durable-owner gate shape is invalid")
    try:
        inventory_bytes = _OWNER_INVENTORY.read_bytes()
        owner_ids = sorted(
            match.group(1)
            for line in inventory_bytes.decode("utf-8").splitlines()
            if (match := _OWNER_ROW.match(line)) is not None
        )
    except (OSError, UnicodeError) as exc:
        raise VoiceManifestError("durable-owner inventory is unavailable") from exc
    _validate_pytest_gate(value["probe"], tests=(_DURABLE_OWNER_TEST,))
    core = {
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "owner_ids_sha256": _canonical_sha256(owner_ids),
        "owner_count": len(owner_ids),
        "probe": value["probe"],
    }
    if (
        value["inventory_sha256"] != core["inventory_sha256"]
        or value["owner_ids_sha256"] != core["owner_ids_sha256"]
        or value["owner_count"] != 14
        or len(owner_ids) != 14
        or value["report_sha256"] != _canonical_sha256(core)
        or value["passed"] is not True
    ):
        raise VoiceManifestError("automated durable-owner gate did not pass")


def _validate_latency(latency: object) -> None:
    metric_names = (
        "eos_to_dispatch",
        "barge_to_audible_stop",
        "added_eos_to_replacement_dispatch",
        "eos_to_first_assistant_audio",
    )
    if not isinstance(latency, Mapping) or set(latency) != {
        "trial_count",
        "clock_boundaries",
        "thresholds",
        *metric_names,
        "passed",
    }:
        raise VoiceManifestError("automated latency shape is invalid")
    if (
        latency["trial_count"] != 40
        or latency["clock_boundaries"] != _CLOCK_BOUNDARIES
        or latency["thresholds"] != _LATENCY_THRESHOLDS
    ):
        raise VoiceManifestError("automated latency contract is invalid")
    summaries = {
        name: _validate_sample_summary(latency[name], count=40) for name in metric_names
    }
    passed = bool(
        _percentile(summaries["eos_to_dispatch"], 0.95) <= 850.0
        and _percentile(summaries["barge_to_audible_stop"], 0.95) <= 150.0
        and _percentile(summaries["added_eos_to_replacement_dispatch"], 0.95) <= 850.0
        and statistics.median(summaries["eos_to_first_assistant_audio"]) <= 1_500.0
        and _percentile(summaries["eos_to_first_assistant_audio"], 0.95) <= 2_500.0
    )
    if latency["passed"] is not passed or not passed:
        raise VoiceManifestError("automated latency did not pass")


def _validate_automated_report(
    report: Mapping[str, object],
    *,
    source_tree_digest: str,
    app_version: str,
    upstream_commit: str,
    wheel_sha256: str,
    extension_sha256: str,
) -> Mapping[str, object]:
    allowed = _AUTOMATED_KEYS | {"git_revision"}
    if set(report) - allowed or not _AUTOMATED_KEYS <= set(report):
        raise VoiceManifestError("automated report has unknown or missing fields")
    if (
        report["schema_version"] != 1
        or report["scenario"] != "automated"
        or report["source_tree_digest"] != source_tree_digest
    ):
        raise VoiceManifestError("automated report source-tree digest is invalid")
    _interpreter_identity(report["interpreter"])
    companion = report["companion"]
    if not isinstance(companion, Mapping) or set(companion) != {
        "version",
        "upstream_commit",
        "wheel_sha256",
        "extension_sha256",
        "passed",
    }:
        raise VoiceManifestError("automated companion identity is invalid")
    if companion != {
        "version": app_version,
        "upstream_commit": upstream_commit,
        "wheel_sha256": wheel_sha256,
        "extension_sha256": extension_sha256,
        "passed": True,
    }:
        raise VoiceManifestError("automated companion identity does not match wheel")
    _validate_corpus(report["corpus"])
    _validate_latency(report["latency_distributions"])
    history = report["completed_pair_history_gate"]
    validate_completed_pair_history_gate(history)
    _validate_pytest_gate(report["lifecycle"], tests=_LIFECYCLE_TESTS)
    _validate_durable_owners(report["durable_owners"])
    if report["passed"] is not True:
        raise VoiceManifestError("automated report did not pass")
    assert isinstance(history, Mapping)
    return history


def _validate_soak_report(
    report: Mapping[str, object],
    *,
    scenario: str,
    source_tree_digest: str,
    platform_key: str,
    python_tag: str,
) -> None:
    required = {
        "schema_version",
        "scenario",
        "source_tree_digest",
        "interpreter",
        "soak",
        "passed",
    }
    if set(report) - (required | {"git_revision"}) or not required <= set(report):
        raise VoiceManifestError("soak report has unknown or missing fields")
    soak = report["soak"]
    if (
        report["schema_version"] != 1
        or report["scenario"] != scenario
        or report["source_tree_digest"] != source_tree_digest
        or report["passed"] is not True
        or not isinstance(soak, Mapping)
    ):
        raise VoiceManifestError("soak report did not pass the 30-minute gate")
    report_python_tag, report_platform_key = _interpreter_identity(
        report["interpreter"]
    )
    if (report_python_tag, report_platform_key) != (python_tag, platform_key):
        raise VoiceManifestError("soak report interpreter does not match qualification")
    common_keys = {
        "requested_duration_seconds",
        "elapsed_seconds",
        "iterations",
        "native_process",
        "tasks",
        "post_fence_callbacks_accepted",
        "passed",
    }
    scenario_keys = (
        {"audio_buffers", "device_handles"}
        if scenario == "duplex-soak"
        else {"cancellations", "orphans", "obsolete_cleanups_final"}
    )
    if set(soak) != common_keys | scenario_keys:
        raise VoiceManifestError("soak report has unknown or missing metrics")
    native = soak["native_process"]
    tasks = soak["tasks"]
    requested_duration = _number(soak["requested_duration_seconds"])
    elapsed = _number(soak["elapsed_seconds"])
    if (
        requested_duration < 1_800.0
        or elapsed < requested_duration
        or type(soak["iterations"]) is not int
        or soak["iterations"] < 1
        or not isinstance(native, Mapping)
        or native
        != {
            "started": True,
            "exit_code": 0,
            "forced_termination": False,
            "reaped": True,
            "passed": True,
        }
        or not isinstance(tasks, Mapping)
        or set(tasks) != {"baseline", "maximum", "final", "maximum_delta_limit"}
        or any(type(value) is not int or value < 0 for value in tasks.values())
        or tasks["maximum"] < tasks["baseline"]
        or tasks["maximum"] < tasks["final"]
        or tasks["final"] > tasks["baseline"]
        or tasks["maximum"] - tasks["baseline"] > tasks["maximum_delta_limit"]
        or soak["post_fence_callbacks_accepted"] != 0
        or soak["passed"] is not True
    ):
        raise VoiceManifestError("soak report did not pass the lifecycle gate")
    if scenario == "duplex-soak":
        _validate_duplex_soak(soak)
    else:
        _validate_cancellation_soak(soak)


def _validate_duplex_soak(soak: Mapping[str, object]) -> None:
    buffers = soak["audio_buffers"]
    handles = soak["device_handles"]
    occupancy_keys = {
        "capture_frames",
        "render_frames",
        "render_reference_frames",
        "control_events",
    }
    if (
        not isinstance(buffers, Mapping)
        or set(buffers) != {"capacities", "maximum", "final", "overflow_count"}
        or not isinstance(handles, Mapping)
        or set(handles) != {"opened", "stopped", "closed", "leaked"}
    ):
        raise VoiceManifestError("duplex soak metric shape is invalid")
    sections: dict[str, Mapping[str, object]] = {}
    for key in ("capacities", "maximum", "final"):
        value = buffers[key]
        if (
            not isinstance(value, Mapping)
            or set(value) != occupancy_keys
            or any(type(item) is not int or item < 0 for item in value.values())
        ):
            raise VoiceManifestError("duplex soak buffer metrics are invalid")
        sections[key] = value
    if (
        any(value < 1 for value in sections["capacities"].values())
        or any(
            sections["maximum"][key] > sections["capacities"][key]
            for key in occupancy_keys
        )
        or any(sections["final"].values())
        or buffers["overflow_count"] != 0
        or any(type(value) is not int or value < 0 for value in handles.values())
        or handles["opened"] < 1
        or handles["opened"] != handles["stopped"]
        or handles["opened"] != handles["closed"]
        or handles["leaked"] != 0
    ):
        raise VoiceManifestError("duplex soak safety gate did not pass")


def _validate_cancellation_soak(soak: Mapping[str, object]) -> None:
    orphans = soak["orphans"]
    if (
        type(soak["cancellations"]) is not int
        or soak["cancellations"] < 2
        or soak["cancellations"] != soak["iterations"] * 2
        or not isinstance(orphans, Mapping)
        or set(orphans) != {"maximum", "final", "limit"}
        or any(type(value) is not int or value < 0 for value in orphans.values())
        or orphans["limit"] != 2
        or orphans["maximum"] > orphans["limit"]
        or orphans["final"] != 0
        or soak["obsolete_cleanups_final"] != 0
    ):
        raise VoiceManifestError("cancellation soak safety gate did not pass")


def _platform_from_wheel_name(name: str) -> str | None:
    lowered = name.casefold()
    if "macosx" in lowered:
        if "arm64" in lowered:
            return "macos-arm64"
        if "x86_64" in lowered:
            return "macos-x86_64"
    if "win_amd64" in lowered:
        return "windows-x86_64"
    if "linux" in lowered:
        if "aarch64" in lowered:
            return "linux-aarch64"
        if "x86_64" in lowered:
            return "linux-x86_64"
    return None


def _interpreter_identity(value: object) -> tuple[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"implementation", "version", "system", "machine"}
        or str(value["implementation"]).casefold() != "cpython"
        or not isinstance(value["version"], str)
        or not isinstance(value["system"], str)
        or not isinstance(value["machine"], str)
    ):
        raise VoiceManifestError("qualification interpreter identity is invalid")
    match = _PYTHON_VERSION.fullmatch(value["version"])
    if match is None:
        raise VoiceManifestError("qualification Python ABI is unsupported")
    system = value["system"].casefold()
    machine = value["machine"].casefold()
    architecture = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "arm64" if system == "darwin" else "aarch64",
    }.get(machine, machine)
    prefix = {"darwin": "macos", "windows": "windows", "linux": "linux"}.get(system)
    platform_key = f"{prefix}-{architecture}" if prefix is not None else "unknown"
    if platform_key not in PLATFORM_KEYS:
        raise VoiceManifestError("qualification platform identity is unsupported")
    return f"cp3{match.group(1)}", platform_key


def _python_tag_from_wheel_name(name: str) -> str | None:
    match = _WHEEL_PYTHON_TAG.search(name.casefold())
    return match.group(1) if match is not None else None


def _wheel_identity(
    wheel_path: Path,
    *,
    app_version: str,
    upstream_commit: str,
) -> tuple[str, str]:
    wheel = Path(wheel_path)
    if not wheel.is_file() or wheel.is_symlink() or wheel.suffix != ".whl":
        raise VoiceManifestError("qualified wheel is missing or unsafe")
    try:
        with ZipFile(wheel) as archive:
            names = archive.namelist()
            extensions = [
                name
                for name in names
                if name.startswith("tldw_voice_aec/_native")
                and Path(name).suffix.casefold() in {".so", ".pyd", ".dylib"}
            ]
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(extensions) != 1 or len(metadata_names) != 1:
                raise VoiceManifestError("qualified wheel contents are invalid")
            metadata = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
            provenance = json.loads(
                archive.read("tldw_voice_aec/provenance/UPSTREAM.json")
            )
            extension_bytes = archive.read(extensions[0])
    except (BadZipFile, KeyError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VoiceManifestError("qualified wheel contents are invalid") from exc
    if (
        metadata.get("Name", "").casefold() != "tldw-voice-aec"
        or metadata.get("Version") != app_version
        or not isinstance(provenance, Mapping)
        or provenance.get("commit") != upstream_commit
    ):
        raise VoiceManifestError("qualified wheel identity is invalid")
    return (
        hashlib.sha256(wheel.read_bytes()).hexdigest(),
        hashlib.sha256(extension_bytes).hexdigest(),
    )


def _qualified_wheels(wheelhouse: Path) -> dict[tuple[str, str], Path]:
    directory = Path(wheelhouse)
    if not directory.is_dir() or directory.is_symlink():
        raise VoiceManifestError("qualified wheelhouse is missing or unsafe")
    wheels: dict[tuple[str, str], Path] = {}
    for candidate in sorted(directory.glob("*.whl")):
        platform_key = _platform_from_wheel_name(candidate.name)
        python_tag = _python_tag_from_wheel_name(candidate.name)
        if platform_key is None or python_tag is None:
            continue
        key = (platform_key, python_tag)
        if key in wheels:
            raise VoiceManifestError(
                "multiple qualified wheels target one platform ABI"
            )
        wheels[key] = candidate
    if {platform_key for platform_key, _tag in wheels} != set(PLATFORM_KEYS):
        raise VoiceManifestError("qualified wheel platform matrix is incomplete")
    return wheels


def _validate_physical_matrix(
    physical_dir: Path,
    *,
    platform_key: str,
    source_tree_digest: str,
    app_version: str,
    upstream_commit: str,
    companion_version: str,
    expected_prerequisites: Mapping[str, str],
) -> str:
    hashes: dict[str, str] = {}
    for device_class in _DEVICE_CLASSES:
        path = Path(physical_dir) / f"{platform_key}-{device_class}.json"
        try:
            report, raw = read_physical_report(path)
        except PhysicalVoiceReportError as exc:
            raise VoiceManifestError("physical report is invalid") from exc
        companion = report.get("companion")
        if (
            report.get("evidence_kind") != "physical"
            or report.get("passed") is not True
            or report.get("source_tree_digest") != source_tree_digest
            or report.get("platform") != platform_key
            or report.get("device_class") != device_class
            or report.get("app_version") != app_version
            or not isinstance(companion, Mapping)
            or companion.get("version") != companion_version
            or companion.get("upstream_commit") != upstream_commit
            or report.get("automated_prerequisites") != expected_prerequisites
        ):
            raise VoiceManifestError("physical report identity or safety is invalid")
        hashes[device_class] = hashlib.sha256(raw).hexdigest()
    return _canonical_sha256(hashes)


def validate_voice_qualification_manifest(
    manifest: Mapping[str, object],
    *,
    schema_path: Path = _SCHEMA_PATH,
) -> None:
    """Validate strict manifest shape and derived rollout predicates."""

    _reject_forbidden_fields(manifest)
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        error = next(iter(Draft202012Validator(schema).iter_errors(manifest)), None)
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaError) as exc:
        raise VoiceManifestError(
            "voice qualification manifest schema is invalid"
        ) from exc
    if error is not None:
        raise VoiceManifestError("voice qualification manifest validation failed")
    platforms = manifest["platforms"]
    assert isinstance(platforms, Mapping)
    if set(platforms) != set(PLATFORM_KEYS):
        raise VoiceManifestError("voice qualification platform matrix is invalid")
    for value in platforms.values():
        assert isinstance(value, Mapping)
        if value["qualified"] is True and value["history_gate_passed"] is not True:
            raise VoiceManifestError(
                "qualified platform lacks passing history evidence"
            )


def _write_canonical(path: Path, value: Mapping[str, object]) -> bytes:
    output = Path(path)
    if output.exists() and output.is_symlink():
        raise VoiceManifestError("rollout authority output cannot be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(value) + b"\n"
    output.write_bytes(encoded)
    return encoded


def generate_voice_qualification_manifest(
    *,
    source_tree_digest: str,
    app_version: str,
    aec_upstream: str,
    wheelhouse: Path,
    automated_dir: Path,
    physical_dir: Path,
    output: Path,
    build_identity_output: Path,
    root: Path = _ROOT,
    path_list: Path = _SOURCE_PATH_LIST,
    _source_digest_reader: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Validate complete evidence and write deterministic packaged authority."""

    if not _SHA256.fullmatch(source_tree_digest):
        raise VoiceManifestError("source-tree digest must be lowercase SHA-256")
    if not _VERSION.fullmatch(app_version) or not _COMMIT.fullmatch(aec_upstream):
        raise VoiceManifestError("app or companion identity is invalid")
    digest_reader = _source_digest_reader or (
        lambda: compute_voice_source_digest(root=Path(root), path_list=Path(path_list))
    )
    try:
        computed_digest = digest_reader()
    except Exception as exc:
        raise VoiceManifestError("source-tree digest could not be proven") from exc
    if computed_digest != source_tree_digest:
        raise VoiceManifestError("source-tree digest does not match included source")
    wheels = _qualified_wheels(wheelhouse)
    platforms: dict[str, object] = {}
    for platform_key in PLATFORM_KEYS:
        automated_path = Path(automated_dir) / f"{platform_key}-automated.json"
        automated, automated_raw = _read_json(
            automated_path,
            label="automated report",
        )
        python_tag, report_platform_key = _interpreter_identity(
            automated.get("interpreter")
        )
        if report_platform_key != platform_key:
            raise VoiceManifestError("automated report platform identity is invalid")
        wheel_path = wheels.get((platform_key, python_tag))
        if wheel_path is None:
            raise VoiceManifestError(
                "qualified wheel for report interpreter is missing"
            )
        wheel_sha256, extension_sha256 = _wheel_identity(
            wheel_path,
            app_version=app_version,
            upstream_commit=aec_upstream,
        )
        history = _validate_automated_report(
            automated,
            source_tree_digest=source_tree_digest,
            app_version=app_version,
            upstream_commit=aec_upstream,
            wheel_sha256=wheel_sha256,
            extension_sha256=extension_sha256,
        )
        soak_hashes: dict[str, str] = {}
        for scenario in ("duplex-soak", "cancellation-soak"):
            soak, soak_raw = _read_json(
                Path(automated_dir) / f"{platform_key}-{scenario}.json",
                label="soak report",
            )
            _validate_soak_report(
                soak,
                scenario=scenario,
                source_tree_digest=source_tree_digest,
                platform_key=platform_key,
                python_tag=python_tag,
            )
            soak_hashes[f"{scenario.replace('-', '_')}_report_sha256"] = hashlib.sha256(
                soak_raw
            ).hexdigest()
        prerequisites = {
            "source_tree_digest": source_tree_digest,
            "automated_report_sha256": hashlib.sha256(automated_raw).hexdigest(),
            "corpus_section_sha256": _canonical_sha256(automated["corpus"]),
            "latency_section_sha256": _canonical_sha256(
                automated["latency_distributions"]
            ),
            **soak_hashes,
        }
        physical_hash = _validate_physical_matrix(
            physical_dir,
            platform_key=platform_key,
            source_tree_digest=source_tree_digest,
            app_version=app_version,
            upstream_commit=aec_upstream,
            companion_version=app_version,
            expected_prerequisites=prerequisites,
        )
        platforms[platform_key] = {
            "qualified": True,
            "python_tag": python_tag,
            "wheel_sha256": wheel_sha256,
            "extension_sha256": extension_sha256,
            "automated_report_sha256": hashlib.sha256(automated_raw).hexdigest(),
            "physical_report_sha256": physical_hash,
            "history_gate_sha256": _canonical_sha256(history),
            "history_gate_passed": True,
        }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "pipeline_version": 1,
        "app_version": app_version,
        "source_tree_digest": source_tree_digest,
        "aec_package": {
            "name": "tldw-voice-aec",
            "version": app_version,
            "upstream_commit": aec_upstream,
        },
        "platforms": platforms,
    }
    validate_voice_qualification_manifest(manifest)
    manifest_bytes = _write_canonical(output, manifest)
    build_identity = {
        "schema_version": 1,
        "pipeline_version": 1,
        "app_version": app_version,
        "source_tree_digest": source_tree_digest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    _write_canonical(build_identity_output, build_identity)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Generate one manifest from exact source-bound evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tree-digest", required=True)
    parser.add_argument("--app-version", required=True)
    parser.add_argument("--aec-upstream", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--automated-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-identity-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        generate_voice_qualification_manifest(
            source_tree_digest=args.source_tree_digest,
            app_version=args.app_version,
            aec_upstream=args.aec_upstream,
            wheelhouse=args.wheelhouse,
            automated_dir=args.automated_dir,
            physical_dir=args.physical_dir,
            output=args.output,
            build_identity_output=args.build_identity_output,
        )
    except VoiceManifestError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover - release entry point
    raise SystemExit(main())


__all__ = [
    "PLATFORM_KEYS",
    "VoiceManifestError",
    "generate_voice_qualification_manifest",
    "validate_completed_pair_history_gate",
    "validate_voice_qualification_manifest",
]
