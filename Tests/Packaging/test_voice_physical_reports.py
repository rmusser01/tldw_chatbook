"""Schema, safety, and privacy tests for physical voice evidence."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import statistics
import stat
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from Packaging import voice_physical_reports
from Packaging.physical_voice_runner import (
    LiveTrialConfig,
    PhysicalVoiceRunnerError,
    PhysicalVoiceTrialRunner,
    RouteIdentity,
)
from Packaging.voice_physical_reports import (
    PhysicalVoiceReportError,
    _canonical_bytes,
    build_physical_report,
    load_automated_prerequisites,
    load_fixture_observations,
    main,
    validate_physical_report,
    write_physical_report,
)
from Tests.Audio.fakes.fake_duplex_backend import FakeDuplexBackend
from tldw_chatbook.Audio.duplex_transport import DuplexAudioTransport


_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).with_name("fixtures") / "speculative_voice_physical"
_DIGEST = "d" * 64


def test_public_exports_include_physical_report_reader() -> None:
    assert "read_physical_report" in voice_physical_reports.__all__
    assert callable(voice_physical_reports.read_physical_report)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _prerequisites(
    tmp_path: Path,
    *,
    digest: str = _DIGEST,
    platform_key: str | None = None,
):
    prerequisite_platform = (
        platform_key or voice_physical_reports._runtime_platform_key()
    )
    automated = {
        "schema_version": 1,
        "scenario": "automated",
        "source_tree_digest": digest,
        "interpreter": {"implementation": "cpython"},
        "companion": {
            "version": "0.1.8.0",
            "upstream_commit": "1" * 40,
            "wheel_sha256": "2" * 64,
            "extension_sha256": "3" * 64,
            "passed": True,
        },
        "corpus": {"thresholds": {}, "passed": True},
        "latency_distributions": {"trial_count": 40, "passed": True},
        "completed_pair_history_gate": {"passed": True},
        "lifecycle": {"passed": True},
        "durable_owners": {"passed": True},
        "passed": True,
    }
    duplex = {
        "schema_version": 1,
        "scenario": "duplex-soak",
        "source_tree_digest": digest,
        "soak": {"passed": True},
        "passed": True,
    }
    cancellation = {
        "schema_version": 1,
        "scenario": "cancellation-soak",
        "source_tree_digest": digest,
        "soak": {"passed": True},
        "passed": True,
    }
    automated_path = tmp_path / "automated.json"
    _write_json(automated_path, automated)
    _write_json(tmp_path / f"{prerequisite_platform}-duplex-soak.json", duplex)
    _write_json(
        tmp_path / f"{prerequisite_platform}-cancellation-soak.json",
        cancellation,
    )
    return load_automated_prerequisites(
        automated_path,
        expected_source_tree_digest=digest,
        platform_key=prerequisite_platform,
    )


def _cli_args(
    tmp_path: Path,
    *,
    output: Path | None = None,
    device_class: str = "usb",
    platform_key: str | None = None,
) -> list[str]:
    return [
        "--source-tree-digest",
        _DIGEST,
        "--platform",
        platform_key or voice_physical_reports._runtime_platform_key(),
        "--device-class",
        device_class,
        "--automated-report",
        str(tmp_path / "automated.json"),
        "--output",
        str(output or tmp_path / "physical.json"),
    ]


def _report(
    tmp_path: Path,
    fixture: str,
    *,
    salt: bytes = b"0123456789abcdef",
) -> dict[str, object]:
    observations = load_fixture_observations(_FIXTURES / f"{fixture}.json")
    return build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class=(
            "bluetooth"
            if "half" in fixture or "unsafe" in fixture
            else "usb"
            if "isolated" in fixture
            else "builtin"
        ),
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=salt,
    )


@pytest.mark.parametrize(
    ("fixture", "expected_passed"),
    [
        ("safe_full_duplex", True),
        ("safe_isolated_full_duplex", True),
        ("safe_half_duplex", True),
        ("unsafe_unsuppressed", False),
    ],
)
def test_synthetic_safety_cases_validate_with_derived_outcome(
    tmp_path: Path,
    fixture: str,
    expected_passed: bool,
) -> None:
    report = _report(tmp_path, fixture)

    validate_physical_report(report)

    assert report["passed"] is expected_passed


def test_version_two_report_rejects_version_one(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    assert report["schema_version"] == 2
    report["schema_version"] = 1

    with pytest.raises(PhysicalVoiceReportError, match="schema"):
        validate_physical_report(report)


@pytest.mark.parametrize(
    "safety_path",
    ["warming", "full-duplex", "isolated", "AEC", ""],
)
def test_safety_path_is_a_closed_three_value_contract(
    tmp_path: Path,
    safety_path: str,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    report["safety_path"] = safety_path

    with pytest.raises(PhysicalVoiceReportError, match="schema"):
        validate_physical_report(report)


def test_isolation_accepts_null_erle_only_with_operational_processor(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path, "safe_isolated_full_duplex")
    aec = report["aec"]
    assert isinstance(aec, dict)
    assert aec == {
        "delay_estimate_available": False,
        "delay_estimate_refined": False,
        "health_path": "warming",
        "processor_operational": True,
        "erle_samples_db": [],
        "median_erle_db": None,
        "p10_erle_db": None,
    }

    validate_physical_report(report)

    aec["processor_operational"] = False
    with pytest.raises(PhysicalVoiceReportError, match="schema"):
        validate_physical_report(report)


@pytest.mark.parametrize(
    "path",
    [
        ("aec", "delay_estimate_available"),
        ("aec", "delay_estimate_refined"),
        ("isolation", "warmup_duration_ms"),
        ("isolation", "demotion_reasons"),
        ("trials", "device_switch", "route_generation_trials"),
        ("thresholds", "isolation_warmup_duration_ms_min"),
        ("thresholds", "route_round_trips_required"),
    ],
)
def test_version_two_requires_every_new_safety_evidence_field(
    tmp_path: Path,
    path: tuple[str, ...],
) -> None:
    report = _report(tmp_path, "safe_isolated_full_duplex")
    target = report
    for key in path[:-1]:
        next_target = target[key]
        assert isinstance(next_target, dict)
        target = next_target
    target.pop(path[-1])

    with pytest.raises(PhysicalVoiceReportError, match="schema"):
        validate_physical_report(report)


def test_isolation_window_count_is_derived_from_paired_samples(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_isolated_full_duplex")
    isolation = report["isolation"]
    assert isinstance(isolation, dict)
    assert isolation["eligible_windows"] == 5
    assert isolation["eligible_windows"] == len(isolation["correlation_samples"])
    assert isolation["eligible_windows"] == len(isolation["leakage_db_samples"])

    isolation["eligible_windows"] = 4
    report["passed"] = False
    with pytest.raises(PhysicalVoiceReportError, match="derivation"):
        validate_physical_report(report)


def test_builder_rejects_unpaired_isolation_samples(tmp_path: Path) -> None:
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    observations["leakage_db_samples"] = [-48.0] * 4

    with pytest.raises(PhysicalVoiceReportError, match="paired"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="usb",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


def test_strict_validator_rejects_unpaired_isolation_samples(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_isolated_full_duplex")
    isolation = report["isolation"]
    assert isinstance(isolation, dict)
    isolation["leakage_db_samples"].pop()
    isolation["p95_leakage_db"] = -45.0
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="paired"):
        validate_physical_report(report)


def test_delay_refinement_requires_an_available_estimate(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_isolated_full_duplex")
    aec = report["aec"]
    assert isinstance(aec, dict)
    aec["delay_estimate_refined"] = True

    with pytest.raises(PhysicalVoiceReportError, match="schema|delay"):
        validate_physical_report(report)


def test_aec_path_requires_available_refined_delay_evidence(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    aec = report["aec"]
    assert isinstance(aec, dict)
    assert aec["delay_estimate_available"] is True
    assert aec["delay_estimate_refined"] is True
    aec["delay_estimate_refined"] = False
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="schema|delay"):
        validate_physical_report(report)


def test_isolation_requires_five_seconds_of_warmup_to_pass(tmp_path: Path) -> None:
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    observations["isolation_warmup_duration_ms"] = 4_999.999999

    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="usb",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )

    assert report["passed"] is False


def test_demotion_count_and_closed_reason_records_are_derived(tmp_path: Path) -> None:
    report = _report(tmp_path, "unsafe_unsuppressed")
    isolation = report["isolation"]
    assert isinstance(isolation, dict)
    assert isolation["demotions"] == 1
    assert isolation["demotion_reasons"] == [
        {"count": 1, "reason": "correlated-render"}
    ]

    isolation["demotions"] = 2
    with pytest.raises(PhysicalVoiceReportError, match="demotion.*derivation"):
        validate_physical_report(report)


@pytest.mark.parametrize(
    "reason_records",
    [
        [{"count": 1, "reason": "invented"}],
        [
            {"count": 1, "reason": "saturation"},
            {"count": 2, "reason": "saturation"},
        ],
        [
            {"count": 1, "reason": "saturation"},
            {"count": 1, "reason": "correlated-render"},
        ],
    ],
)
def test_demotion_reason_records_are_closed_unique_and_canonical(
    tmp_path: Path,
    reason_records: list[dict[str, object]],
) -> None:
    report = _report(tmp_path, "safe_isolated_full_duplex")
    isolation = report["isolation"]
    assert isinstance(isolation, dict)
    isolation["demotion_reasons"] = reason_records
    isolation["demotions"] = sum(int(record["count"]) for record in reason_records)
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="schema|demotion"):
        validate_physical_report(report)


def test_consistent_demotion_record_is_valid_failed_evidence(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_isolated_full_duplex")
    isolation = report["isolation"]
    assert isinstance(isolation, dict)
    isolation["demotion_reasons"] = [{"count": 1, "reason": "saturation"}]
    isolation["demotions"] = 1
    report["passed"] = False

    validate_physical_report(report)


def test_route_trial_aggregates_are_derived_from_bounded_records(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    device_switch = report["trials"]["device_switch"]
    assert isinstance(device_switch, dict)
    records = device_switch["route_generation_trials"]
    assert isinstance(records, list)
    assert device_switch["trials"] == len(records) == 3
    assert device_switch["safe_fallbacks"] == 3

    device_switch["safe_fallbacks"] = 2
    report["passed"] = False
    with pytest.raises(PhysicalVoiceReportError, match="route.*derivation"):
        validate_physical_report(report)


def test_bad_route_generation_record_is_valid_failed_evidence(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    device_switch = report["trials"]["device_switch"]
    assert isinstance(device_switch, dict)
    records = device_switch["route_generation_trials"]
    assert isinstance(records, list)
    first = records[0]
    assert isinstance(first, dict)
    first["alternate_generation"] = first["start_generation"]
    device_switch["safe_fallbacks"] = 2
    report["passed"] = False

    validate_physical_report(report)


@pytest.mark.parametrize(
    ("device_class", "transport"),
    [("builtin", "usb"), ("usb", "bluetooth"), ("bluetooth", "wired")],
)
def test_strict_validator_rejects_impossible_device_transport_pairs(
    tmp_path: Path,
    device_class: str,
    transport: str,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    report["device_class"] = device_class
    device = report["device"]
    assert isinstance(device, dict)
    device["transport"] = transport
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="schema|transport"):
        validate_physical_report(report)


@pytest.mark.parametrize(("trials", "detected"), [(0, 1), (2, 3)])
def test_strict_validator_rejects_impossible_double_talk_counts(
    tmp_path: Path,
    trials: int,
    detected: int,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    double_talk = report["trials"]["double_talk"]
    assert isinstance(double_talk, dict)
    double_talk.update({"trials": trials, "detected": detected, "recall": 0.0})
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="double-talk"):
        validate_physical_report(report)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_strict_validator_rejects_non_finite_numbers_recursively(
    tmp_path: Path,
    non_finite: float,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    report["trials"]["soak"]["minutes"] = non_finite
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="finite"):
        validate_physical_report(report)


def test_report_writer_never_serializes_non_standard_nan(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    report["trials"]["soak"]["minutes"] = float("nan")
    output = tmp_path / "physical.json"

    with pytest.raises(PhysicalVoiceReportError, match="finite"):
        write_physical_report(output, report)
    assert not output.exists()


@pytest.mark.parametrize(
    ("fixture", "path", "integer_valued_float"),
    [
        ("safe_full_duplex", ("schema_version",), 2.0),
        ("safe_full_duplex", ("device", "sample_rate_hz"), 48_000.0),
        ("safe_full_duplex", ("device", "channels"), 1.0),
        ("safe_full_duplex", ("device", "frame_duration_ms"), 10.0),
        ("safe_full_duplex", ("isolation", "eligible_windows"), 0.0),
        ("safe_full_duplex", ("isolation", "required_windows"), 5.0),
        ("safe_full_duplex", ("isolation", "render_only_vad_events"), 0.0),
        ("safe_full_duplex", ("isolation", "demotions"), 0.0),
        ("safe_full_duplex", ("transport_health", "capture_overflows"), 0.0),
        ("safe_full_duplex", ("transport_health", "render_overflows"), 0.0),
        ("safe_full_duplex", ("transport_health", "reference_overflows"), 0.0),
        ("safe_full_duplex", ("transport_health", "control_overflows"), 0.0),
        ("safe_full_duplex", ("transport_health", "saturation_events"), 0.0),
        (
            "safe_full_duplex",
            ("trials", "rendered_speech", "false_barge_events"),
            0.0,
        ),
        ("safe_full_duplex", ("trials", "double_talk", "trials"), 100.0),
        ("safe_full_duplex", ("trials", "double_talk", "detected"), 98.0),
        ("safe_full_duplex", ("trials", "interruption", "trials"), 4.0),
        ("safe_full_duplex", ("trials", "silence", "false_barge_events"), 0.0),
        ("safe_full_duplex", ("trials", "device_switch", "trials"), 3.0),
        (
            "safe_full_duplex",
            ("trials", "device_switch", "safe_fallbacks"),
            3.0,
        ),
        ("safe_full_duplex", ("trials", "soak", "process_leaks"), 0.0),
        ("safe_full_duplex", ("trials", "soak", "device_handle_leaks"), 0.0),
        ("safe_full_duplex", ("trials", "soak", "post_fence_callbacks"), 0.0),
        (
            "safe_full_duplex",
            (
                "trials",
                "device_switch",
                "route_generation_trials",
                0,
                "start_generation",
            ),
            0.0,
        ),
        (
            "safe_full_duplex",
            (
                "trials",
                "device_switch",
                "route_generation_trials",
                0,
                "alternate_generation",
            ),
            1.0,
        ),
        (
            "safe_full_duplex",
            (
                "trials",
                "device_switch",
                "route_generation_trials",
                0,
                "return_generation",
            ),
            2.0,
        ),
        ("unsafe_unsuppressed", ("isolation", "demotion_reasons", 0, "count"), 1.0),
        (
            "safe_full_duplex",
            ("thresholds", "isolation_eligible_windows_min"),
            5.0,
        ),
        (
            "safe_full_duplex",
            ("thresholds", "isolation_render_only_vad_events_max"),
            0.0,
        ),
        (
            "safe_full_duplex",
            ("thresholds", "route_round_trips_required"),
            3.0,
        ),
    ],
)
def test_strict_validator_rejects_integer_valued_floats(
    tmp_path: Path,
    fixture: str,
    path: tuple[str | int, ...],
    integer_valued_float: float,
) -> None:
    report = _report(tmp_path, fixture)
    target: object = report
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = integer_valued_float
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="integer"):
        validate_physical_report(report)


@pytest.mark.parametrize(
    "path",
    [
        ("isolation", "warmup_duration_ms"),
        ("aec", "erle_samples_db", 0),
        ("trials", "interruption", "stop_latency_samples_ms", 0),
    ],
)
def test_strict_validator_and_writer_reject_negative_zero(
    tmp_path: Path,
    path: tuple[str | int, ...],
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    target: object = report
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = -0.0
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="negative zero"):
        validate_physical_report(report)
    output = tmp_path / "negative-zero.json"
    with pytest.raises(PhysicalVoiceReportError, match="negative zero"):
        write_physical_report(output, report)
    assert not output.exists()


def test_builder_normalizes_every_allowed_negative_zero(tmp_path: Path) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_half_duplex.json")
    observations.update(
        {
            "erle_samples_db": [-0.0],
            "correlation_samples": [-0.0],
            "leakage_db_samples": [-0.0],
            "isolation_warmup_duration_ms": -0.0,
            "stop_latency_samples_ms": [-0.0],
        }
    )

    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="bluetooth",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )

    numeric_zeroes = [
        report["aec"]["erle_samples_db"][0],
        report["aec"]["median_erle_db"],
        report["aec"]["p10_erle_db"],
        report["isolation"]["warmup_duration_ms"],
        report["isolation"]["correlation_samples"][0],
        report["isolation"]["p95_correlation"],
        report["isolation"]["leakage_db_samples"][0],
        report["isolation"]["p95_leakage_db"],
        report["trials"]["interruption"]["stop_latency_samples_ms"][0],
        report["trials"]["interruption"]["p95_stop_latency_ms"],
    ]
    assert all(value == 0.0 for value in numeric_zeroes)
    assert all(math.copysign(1.0, value) == 1.0 for value in numeric_zeroes)
    assert "-0.0" not in _canonical_bytes(report).decode("ascii")


def test_builder_normalizes_derived_median_negative_zero(tmp_path: Path) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_half_duplex.json")
    observations["erle_samples_db"] = [-0.000002, 0.000001]

    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="bluetooth",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )

    median = report["aec"]["median_erle_db"]
    assert median == 0.0
    assert math.copysign(1.0, median) == 1.0


@pytest.mark.parametrize(
    "observation_key",
    [
        "render_only_vad_events",
        "capture_overflows",
        "render_overflows",
        "reference_overflows",
        "control_overflows",
        "saturation_events",
        "rendered_false_barge_events",
        "double_talk_trials",
        "double_talk_detected",
        "silence_false_barge_events",
        "process_leaks",
        "device_handle_leaks",
        "post_fence_callbacks",
    ],
)
def test_builder_bounds_every_event_count(
    tmp_path: Path,
    observation_key: str,
) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    observations[observation_key] = 1_000_001
    if observation_key == "double_talk_detected":
        observations["double_talk_trials"] = 1_000_000

    with pytest.raises(PhysicalVoiceReportError, match="bounds|integer"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="builtin",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


def test_event_and_demotion_counts_accept_limit_but_reject_limit_plus_one(
    tmp_path: Path,
) -> None:
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    observations["rendered_false_barge_events"] = 1_000_000
    observations["demotion_reason_counts"] = {"saturation": 1_000_000}
    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="usb",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )
    assert report["trials"]["rendered_speech"]["false_barge_events"] == 1_000_000
    assert report["isolation"]["demotions"] == 1_000_000

    observations["demotion_reason_counts"] = {"saturation": 1_000_001}
    with pytest.raises(PhysicalVoiceReportError, match="demotion"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="usb",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )

    observations["demotion_reason_counts"] = {
        "saturation": 600_000,
        "route-mismatch": 600_000,
    }
    with pytest.raises(PhysicalVoiceReportError, match="demotion"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="usb",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


@pytest.mark.parametrize(
    ("observation_key", "maximum"),
    [
        ("sample_rate_hz", 192_000),
        ("channels", 2),
        ("frame_duration_ms", 100),
    ],
)
def test_device_integer_bounds_accept_limit_only(
    tmp_path: Path,
    observation_key: str,
    maximum: int,
) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    observations[observation_key] = maximum
    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="builtin",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )
    assert report["device"][observation_key] == maximum

    observations[observation_key] = maximum + 1
    with pytest.raises(PhysicalVoiceReportError, match="bounds|integer"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="builtin",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


def test_route_generation_accepts_signed_64_bit_limit_only(tmp_path: Path) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    route_trials = observations["route_generation_trials"]
    route_trials[-1]["alternate_generation"] = (1 << 63) - 2
    route_trials[-1]["return_generation"] = (1 << 63) - 1
    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="builtin",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )
    assert (
        report["trials"]["device_switch"]["route_generation_trials"][-1][
            "return_generation"
        ]
        == (1 << 63) - 1
    )

    observations["route_generation_trials"][-1]["return_generation"] = 1 << 63
    with pytest.raises(PhysicalVoiceReportError, match="generation"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="builtin",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


@pytest.mark.parametrize(
    ("fixture", "path", "over_limit"),
    [
        (
            "safe_full_duplex",
            ("transport_health", "capture_overflows"),
            1_000_001,
        ),
        (
            "safe_full_duplex",
            (
                "trials",
                "device_switch",
                "route_generation_trials",
                0,
                "start_generation",
            ),
            1 << 63,
        ),
        (
            "unsafe_unsuppressed",
            ("isolation", "demotion_reasons", 0, "count"),
            1_000_001,
        ),
    ],
)
def test_strict_validator_rejects_every_integer_domain_over_limit(
    tmp_path: Path,
    fixture: str,
    path: tuple[str | int, ...],
    over_limit: int,
) -> None:
    report = _report(tmp_path, fixture)
    target: object = report
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = over_limit
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="schema|integer"):
        validate_physical_report(report)


def test_schema_bounds_every_declared_integer_field() -> None:
    schema = json.loads(
        (_ROOT / "Packaging/voice_physical_report.schema.json").read_text(
            encoding="utf-8"
        )
    )
    stack = [schema]
    integer_schemas: list[dict[str, object]] = []
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if value.get("type") == "integer":
                integer_schemas.append(value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)

    assert integer_schemas
    assert all("minimum" in value and "maximum" in value for value in integer_schemas)


@pytest.mark.parametrize(
    "observation_key",
    [
        "rendered_speech_minutes",
        "erle_samples_db",
        "rendered_false_barge_events",
    ],
)
def test_malformed_huge_observations_never_leak_raw_numeric_errors(
    tmp_path: Path,
    observation_key: str,
) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    huge_integer = 10**10_000
    observations[observation_key] = (
        [huge_integer] if observation_key == "erle_samples_db" else huge_integer
    )

    with pytest.raises(PhysicalVoiceReportError):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="builtin",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


@pytest.mark.parametrize(
    "demotion_reason_counts",
    [{1: 1}, {"saturation": 1, 2: 1}],
)
def test_mixed_or_non_string_demotion_keys_are_domain_errors(
    tmp_path: Path,
    demotion_reason_counts: dict[object, int],
) -> None:
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    observations["demotion_reason_counts"] = demotion_reason_counts

    with pytest.raises(PhysicalVoiceReportError, match="demotion"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="usb",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


def test_surrogate_device_identifier_is_a_domain_error(tmp_path: Path) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    observations["device_identifier"] = "invalid-\ud800-device"

    with pytest.raises(PhysicalVoiceReportError, match="device identifier"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="builtin",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


def test_canonical_serialization_translates_hostile_value_errors() -> None:
    circular: list[object] = []
    circular.append(circular)
    too_deep: object = 0
    for _ in range(sys.getrecursionlimit() + 10):
        too_deep = [too_deep]

    for hostile in ({"value": object()}, circular, too_deep):
        with pytest.raises(PhysicalVoiceReportError, match="canonical"):
            _canonical_bytes(hostile)


def test_non_finite_automated_prerequisite_is_rejected_before_hashing(
    tmp_path: Path,
) -> None:
    _prerequisites(tmp_path)
    automated_path = tmp_path / "automated.json"
    automated = json.loads(automated_path.read_text(encoding="utf-8"))
    automated["corpus"]["non_finite"] = float("nan")
    _write_json(automated_path, automated)

    with pytest.raises(PhysicalVoiceReportError, match="finite"):
        load_automated_prerequisites(
            automated_path,
            expected_source_tree_digest=_DIGEST,
        )


@pytest.mark.parametrize("kind", ["depth", "nodes"])
def test_hostile_report_tree_limits_are_domain_errors(
    tmp_path: Path,
    kind: str,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    if kind == "depth":
        nested: dict[str, object] = {}
        report["unknown"] = nested
        for _ in range(33):
            child: dict[str, object] = {}
            nested["child"] = child
            nested = child
    else:
        report["unknown"] = [0] * 50_001

    with pytest.raises(PhysicalVoiceReportError, match="complex|depth|nodes"):
        validate_physical_report(report)


def test_oversized_json_report_is_rejected_before_parse(tmp_path: Path) -> None:
    fixture = tmp_path / "oversized.json"
    _write_json(fixture, {"padding": "x" * (2 * 1024 * 1024)})

    with pytest.raises(PhysicalVoiceReportError, match="large|size"):
        load_fixture_observations(fixture)


def test_huge_json_integer_parse_failure_is_a_domain_error(tmp_path: Path) -> None:
    fixture = tmp_path / "huge-integer.json"
    fixture.write_text('{"value":' + "9" * 10_000 + "}", encoding="utf-8")

    with pytest.raises(PhysicalVoiceReportError, match="JSON"):
        load_fixture_observations(fixture)


def test_validator_writer_and_reader_share_the_same_canonical_byte_limit(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    byte_limit = 2 * 1024 * 1024
    report["app_version"] = "1.1.1"
    extra_bytes = byte_limit - (len(_canonical_bytes(report)) + 1)
    report["app_version"] = "1." + "1" * (extra_bytes + 1) + ".1"
    assert len(_canonical_bytes(report)) + 1 == byte_limit

    validate_physical_report(report)
    exact_output = tmp_path / "exact-limit-output.json"
    write_physical_report(exact_output, report)
    assert exact_output.stat().st_size == byte_limit
    validate_physical_report(load_fixture_observations(exact_output))

    report["app_version"] += "1"
    output = tmp_path / "oversized-output.json"

    with pytest.raises(PhysicalVoiceReportError, match="large|size"):
        validate_physical_report(report)
    with pytest.raises(PhysicalVoiceReportError, match="large|size"):
        write_physical_report(output, report)
    assert not output.exists()


def test_maximum_sized_sample_report_remains_valid(tmp_path: Path) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    observations.update(
        {
            "erle_samples_db": [20.0] * 3_600,
            "correlation_samples": [0.0] * 3_600,
            "leakage_db_samples": [-30.0] * 3_600,
            "stop_latency_samples_ms": [150.0] * 20,
        }
    )

    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="builtin",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )

    assert len(report["aec"]["erle_samples_db"]) == 3_600
    assert len(report["isolation"]["correlation_samples"]) == 3_600
    assert len(report["isolation"]["leakage_db_samples"]) == 3_600
    assert len(report["trials"]["interruption"]["stop_latency_samples_ms"]) == 20
    validate_physical_report(report)
    output = tmp_path / "maximum-report.json"
    write_physical_report(output, report)
    assert output.stat().st_size <= 2 * 1024 * 1024
    serialized = load_fixture_observations(output)
    validate_physical_report(serialized)


@pytest.mark.parametrize(
    "fixture",
    ["safe_full_duplex", "safe_isolated_full_duplex"],
)
def test_safe_bluetooth_full_duplex_can_qualify(
    tmp_path: Path,
    fixture: str,
) -> None:
    observations = load_fixture_observations(_FIXTURES / f"{fixture}.json")
    observations["transport"] = "bluetooth"

    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="bluetooth",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )

    assert report["passed"] is True
    validate_physical_report(report)


@pytest.mark.parametrize(
    ("fixture", "sample_key", "unsafe_samples"),
    [
        ("safe_full_duplex", "erle_samples_db", [19.0] * 5),
        ("safe_isolated_full_duplex", "correlation_samples", [0.13] * 5),
    ],
)
def test_bluetooth_full_duplex_still_requires_path_thresholds(
    tmp_path: Path,
    fixture: str,
    sample_key: str,
    unsafe_samples: list[float],
) -> None:
    observations = load_fixture_observations(_FIXTURES / f"{fixture}.json")
    observations["transport"] = "bluetooth"
    observations[sample_key] = unsafe_samples

    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="bluetooth",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )

    assert report["passed"] is False
    validate_physical_report(report)


def test_healthy_aec_cannot_be_forged_as_acoustic_isolation(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_isolated_full_duplex")
    aec = report["aec"]
    assert isinstance(aec, dict)
    aec["health_path"] = "healthy"

    with pytest.raises(PhysicalVoiceReportError, match="schema|health"):
        validate_physical_report(report)


def test_bluetooth_transport_must_still_match_device_class(tmp_path: Path) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    observations["transport"] = "bluetooth"

    with pytest.raises(PhysicalVoiceReportError, match="transport"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="usb",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


def test_aec_path_requires_serialized_erle_and_thresholds(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    aec = report["aec"]
    assert isinstance(aec, dict)
    assert aec["median_erle_db"] == 25.0
    assert aec["p10_erle_db"] == 15.0

    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    observations["erle_samples_db"] = [9.9, 19.9, 19.9]
    failed = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="builtin",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )
    assert failed["passed"] is False


def test_safe_half_duplex_requires_manual_stop_not_acoustic_stop_latency(
    tmp_path: Path,
) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_half_duplex.json")
    observations["stop_latency_samples_ms"] = [400.0]

    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="bluetooth",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )

    assert report["passed"] is True


@pytest.mark.parametrize(
    ("section", "aggregate", "value"),
    [
        ("aec", "median_erle_db", 99.0),
        ("aec", "p10_erle_db", 99.0),
        ("isolation", "p95_correlation", 0.0),
        ("isolation", "p95_leakage_db", -99.0),
        ("interruption", "p95_stop_latency_ms", 0.0),
    ],
)
def test_aggregate_forgery_is_rejected_against_serialized_samples(
    tmp_path: Path,
    section: str,
    aggregate: str,
    value: float,
) -> None:
    fixture = "safe_full_duplex" if section == "aec" else "safe_isolated_full_duplex"
    report = _report(tmp_path, fixture)
    target = report[section] if section != "interruption" else report["trials"][section]
    assert isinstance(target, dict)
    target[aggregate] = value

    with pytest.raises(PhysicalVoiceReportError, match="derivation"):
        validate_physical_report(report)


def test_builder_rounds_samples_once_before_deriving_summaries(
    tmp_path: Path,
) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    observations["erle_samples_db"] = [0.04858251, 0.06012551]

    report = build_physical_report(
        evidence_kind="synthetic-fixture",
        source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
        device_class="builtin",
        app_version="0.1.8.0",
        prerequisites=_prerequisites(tmp_path),
        observations=observations,
        salt=b"0123456789abcdef",
    )
    aec = report["aec"]
    assert isinstance(aec, dict)
    assert aec["erle_samples_db"] == [0.048583, 0.060126]
    assert aec["median_erle_db"] == round(
        statistics.median(aec["erle_samples_db"]),
        6,
    )
    assert aec["median_erle_db"] == 0.054355


@pytest.mark.parametrize(
    ("fixture", "section", "key", "unsafe_value"),
    [
        ("safe_full_duplex", "transport_health", "capture_overflows", 1),
        ("safe_full_duplex", "transport_health", "render_overflows", 1),
        ("safe_full_duplex", "transport_health", "reference_overflows", 1),
        ("safe_full_duplex", "transport_health", "control_overflows", 1),
        ("safe_full_duplex", "transport_health", "saturation_events", 1),
        ("safe_full_duplex", "soak", "post_fence_callbacks", 1),
        ("safe_full_duplex", "device_switch", "safe_fallbacks", 2),
        ("safe_isolated_full_duplex", "isolation", "demotions", 1),
    ],
)
def test_any_transport_or_route_safety_fault_forces_failure(
    tmp_path: Path,
    fixture: str,
    section: str,
    key: str,
    unsafe_value: int,
) -> None:
    report = _report(tmp_path, fixture)
    if section == "transport_health":
        target = report[section]
    elif section == "isolation":
        target = report[section]
        target["demotion_reasons"] = [{"count": unsafe_value, "reason": "saturation"}]
    elif section == "device_switch":
        target = report["trials"][section]
        records = target["route_generation_trials"]
        assert isinstance(records, list)
        first = records[0]
        assert isinstance(first, dict)
        first["alternate_admission_closed"] = False
    else:
        target = report["trials"][section]
    assert isinstance(target, dict)
    target[key] = unsafe_value
    report["passed"] = False

    validate_physical_report(report)
    assert report["passed"] is False


@pytest.mark.parametrize(
    ("fixture", "mutate"),
    [
        (
            "safe_isolated_full_duplex",
            lambda report: report["isolation"].__setitem__("eligible_windows", 4),
        ),
        (
            "safe_isolated_full_duplex",
            lambda report: report["isolation"].__setitem__("p95_correlation", 0.13),
        ),
        (
            "safe_isolated_full_duplex",
            lambda report: report["isolation"].__setitem__("p95_leakage_db", -29.9),
        ),
        (
            "safe_isolated_full_duplex",
            lambda report: report["isolation"].__setitem__("render_only_vad_events", 1),
        ),
        (
            "safe_isolated_full_duplex",
            lambda report: report["trials"]["double_talk"].__setitem__("detected", 18),
        ),
        (
            "safe_isolated_full_duplex",
            lambda report: report["trials"]["interruption"].__setitem__(
                "stop_latency_samples_ms", [151.0]
            ),
        ),
        (
            "safe_half_duplex",
            lambda report: report["degradation"].__setitem__(
                "playback_speech_admission_closed", False
            ),
        ),
        (
            "safe_half_duplex",
            lambda report: report["degradation"].__setitem__(
                "manual_interruption_available", False
            ),
        ),
    ],
)
def test_path_specific_safety_condition_cannot_be_forged_passing(
    tmp_path: Path,
    fixture: str,
    mutate,
) -> None:
    report = _report(tmp_path, fixture)
    mutate(report)
    report["passed"] = True

    with pytest.raises(PhysicalVoiceReportError, match="schema|derivation|safety"):
        validate_physical_report(report)


def test_strict_validator_rejects_split_manual_interruption_evidence(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path, "safe_half_duplex")
    interruption = report["trials"]["interruption"]
    assert isinstance(interruption, dict)
    interruption["manual_available"] = False
    report["passed"] = False

    with pytest.raises(PhysicalVoiceReportError, match="manual.*derivation"):
        validate_physical_report(report)


@pytest.mark.parametrize(
    ("field", "count"),
    [
        ("erle_samples_db", 3_601),
        ("correlation_samples", 3_601),
        ("leakage_db_samples", 3_601),
        ("stop_latency_samples_ms", 21),
    ],
)
def test_observation_sample_arrays_are_bounded(
    tmp_path: Path,
    field: str,
    count: int,
) -> None:
    observations = load_fixture_observations(_FIXTURES / "safe_full_duplex.json")
    observations[field] = [1.0] * count

    with pytest.raises(PhysicalVoiceReportError, match="samples|bounds"):
        build_physical_report(
            evidence_kind="synthetic-fixture",
            source_tree_digest=_DIGEST,
            platform_key="macos-arm64",
            device_class="builtin",
            app_version="0.1.8.0",
            prerequisites=_prerequisites(tmp_path),
            observations=observations,
            salt=b"0123456789abcdef",
        )


def test_forged_passing_unsafe_report_is_rejected(tmp_path: Path) -> None:
    report = _report(tmp_path, "unsafe_unsuppressed")
    report["passed"] = True

    with pytest.raises(PhysicalVoiceReportError, match="schema|safety"):
        validate_physical_report(report)


def test_missing_prerequisite_hash_is_rejected_even_when_passed(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    prerequisites = report["automated_prerequisites"]
    assert isinstance(prerequisites, dict)
    prerequisites.pop("duplex_soak_report_sha256")

    with pytest.raises(PhysicalVoiceReportError, match="schema"):
        validate_physical_report(report)


def test_mixed_source_digest_or_failed_automated_report_is_rejected(
    tmp_path: Path,
) -> None:
    _prerequisites(tmp_path)
    automated_path = tmp_path / "automated.json"
    automated = json.loads(automated_path.read_text(encoding="utf-8"))
    automated["source_tree_digest"] = "e" * 64
    _write_json(automated_path, automated)

    with pytest.raises(PhysicalVoiceReportError, match="source-tree digest"):
        load_automated_prerequisites(
            automated_path,
            expected_source_tree_digest=_DIGEST,
        )

    automated["source_tree_digest"] = _DIGEST
    automated["passed"] = False
    _write_json(automated_path, automated)
    with pytest.raises(PhysicalVoiceReportError, match="did not pass"):
        load_automated_prerequisites(
            automated_path,
            expected_source_tree_digest=_DIGEST,
        )


def test_platform_named_soaks_win_over_other_platform_and_fallback_reports(
    tmp_path: Path,
) -> None:
    _prerequisites(tmp_path, platform_key="macos-arm64")
    for scenario in ("duplex-soak", "cancellation-soak"):
        _write_json(
            tmp_path / f"macos-arm64-{scenario}.json",
            {
                "schema_version": 1,
                "scenario": scenario,
                "source_tree_digest": _DIGEST,
                "soak": {"platform_marker": "macos-arm64", "passed": True},
                "passed": True,
            },
        )
        _write_json(
            tmp_path / f"windows-x86_64-{scenario}.json",
            {
                "schema_version": 1,
                "scenario": scenario,
                "source_tree_digest": _DIGEST,
                "soak": {"platform_marker": "windows-x86_64", "passed": True},
                "passed": True,
            },
        )

    evidence = load_automated_prerequisites(
        tmp_path / "automated.json",
        expected_source_tree_digest=_DIGEST,
        platform_key="macos-arm64",
    )

    assert (
        evidence.report_hashes["duplex_soak_report_sha256"]
        == hashlib.sha256(
            (tmp_path / "macos-arm64-duplex-soak.json").read_bytes()
        ).hexdigest()
    )
    assert (
        evidence.report_hashes["cancellation_soak_report_sha256"]
        == hashlib.sha256(
            (tmp_path / "macos-arm64-cancellation-soak.json").read_bytes()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "requested_platform,other_platform",
    [
        ("windows-x86_64", "macos-arm64"),
        ("linux-x86_64", "windows-x86_64"),
    ],
)
def test_platform_prerequisites_require_exact_named_soak_reports(
    tmp_path: Path,
    requested_platform: str,
    other_platform: str,
) -> None:
    _prerequisites(tmp_path, platform_key=other_platform)

    with pytest.raises(PhysicalVoiceReportError, match="exact.*prerequisite"):
        load_automated_prerequisites(
            tmp_path / "automated.json",
            expected_source_tree_digest=_DIGEST,
            platform_key=requested_platform,
        )


def test_invalid_json_schema_is_rejected_before_report_validation(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    schema_path = tmp_path / "invalid-schema.json"
    _write_json(schema_path, {"type": "not-a-json-schema-type"})

    with pytest.raises(PhysicalVoiceReportError, match="schema is invalid"):
        validate_physical_report(report, schema_path=schema_path)


@pytest.mark.parametrize(
    "forbidden",
    [
        "device_name",
        "transcript",
        "response_text",
        "pcm",
        "credential",
        "request_body",
    ],
)
def test_privacy_forbidden_fields_are_rejected(
    tmp_path: Path,
    forbidden: str,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    report[forbidden] = "VOICE-POISON"

    with pytest.raises(PhysicalVoiceReportError, match="forbidden|schema"):
        validate_physical_report(report)


def test_raw_device_identity_is_salted_and_never_serialized(tmp_path: Path) -> None:
    first = _report(tmp_path, "safe_full_duplex", salt=b"a" * 16)
    second = _report(tmp_path, "safe_full_duplex", salt=b"b" * 16)
    first_device = first["device"]
    second_device = second["device"]
    assert isinstance(first_device, dict) and isinstance(second_device, dict)

    assert first_device["identity_sha256"] != second_device["identity_sha256"]
    assert (
        first_device["identity_sha256"]
        == hashlib.sha256(b"a" * 16 + b"synthetic-safe-full-duplex").hexdigest()
    )
    assert "synthetic-safe-full-duplex" not in json.dumps(first, sort_keys=True)


def test_unknown_fields_and_threshold_changes_are_rejected(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    report["unknown"] = True
    with pytest.raises(PhysicalVoiceReportError, match="schema"):
        validate_physical_report(report)

    report = _report(tmp_path, "safe_full_duplex")
    thresholds = report["thresholds"]
    assert isinstance(thresholds, dict)
    thresholds["median_erle_db_min"] = 19.0
    with pytest.raises(PhysicalVoiceReportError, match="schema"):
        validate_physical_report(report)


def test_full_duplex_threshold_failure_cannot_be_forged_passing(tmp_path: Path) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    broken = copy.deepcopy(report)
    aec = broken["aec"]
    assert isinstance(aec, dict)
    aec["p10_erle_db"] = 9.9
    broken["passed"] = True

    with pytest.raises(PhysicalVoiceReportError, match="schema|derivation|safety"):
        validate_physical_report(broken)


@pytest.mark.parametrize(
    ("fixture", "device_class", "expected_exit", "expected_passed"),
    [
        ("safe-full-duplex", "builtin", 0, True),
        ("safe-isolated-full-duplex", "usb", 0, True),
        ("safe-half-duplex", "bluetooth", 0, True),
        ("unsafe-unsuppressed", "bluetooth", 1, False),
    ],
)
def test_fixture_cli_writes_valid_json_and_content_free_summary(
    tmp_path: Path,
    fixture: str,
    device_class: str,
    expected_exit: int,
    expected_passed: bool,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / f"{fixture}-physical.json"

    result = main(
        [
            "--source-tree-digest",
            _DIGEST,
            "--platform",
            "macos-arm64",
            "--device-class",
            device_class,
            "--automated-report",
            str(tmp_path / "automated.json"),
            "--output",
            str(output),
            "--fixture",
            fixture,
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    validate_physical_report(report)
    summary = output.with_suffix(".summary.txt").read_text(encoding="utf-8")
    assert result == expected_exit
    assert report["passed"] is expected_passed
    assert report["evidence_kind"] == "synthetic-fixture"
    assert "synthetic-safe" not in summary


def test_normal_cli_validates_prerequisites_before_one_live_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    events: list[str] = []
    run_count = 0
    real_load = voice_physical_reports.load_automated_prerequisites

    def compute_digest(**_kwargs: object) -> str:
        events.append("source-digest")
        return _DIGEST

    def load_prerequisites(*args: object, **kwargs: object):
        events.append("automated-prerequisites")
        return real_load(*args, **kwargs)

    async def collect_live(**kwargs: object) -> dict[str, object]:
        events.append("live-runner")
        assert kwargs == {"device_class": "usb"}
        return observations

    def forbid_manual_collection(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("normal mode must not collect operator-entered metrics")

    def forbid_cli_input(_prompt: str = "") -> str:
        raise AssertionError("the CLI must not prompt for measured values")

    real_asyncio_run = asyncio.run

    def run_once(coroutine: object) -> object:
        nonlocal run_count
        run_count += 1
        return real_asyncio_run(coroutine)  # type: ignore[arg-type]

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        compute_digest,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "load_automated_prerequisites",
        load_prerequisites,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        collect_live,
        raising=False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_guided_observations",
        forbid_manual_collection,
        raising=False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "asyncio",
        SimpleNamespace(run=run_once),
        raising=False,
    )
    monkeypatch.setattr("builtins.input", forbid_cli_input)

    result = main(_cli_args(tmp_path))

    report = load_fixture_observations(tmp_path / "physical.json")
    validate_physical_report(report)
    assert result == 0
    assert run_count == 1
    assert events == ["source-digest", "automated-prerequisites", "live-runner"]
    assert report["evidence_kind"] == "physical"
    assert report["safety_path"] == "acoustic-isolation"
    assert report["passed"] is True


@pytest.mark.parametrize(
    ("runtime", "machine", "expected"),
    [
        ("darwin", "arm64", "macos-arm64"),
        ("darwin", "x86_64", "macos-x86_64"),
        ("win32", "AMD64", "windows-x86_64"),
        ("win32", "x86_64", "windows-x86_64"),
        ("linux", "x86_64", "linux-x86_64"),
        ("linux", "aarch64", "linux-aarch64"),
        ("linux", "arm64", "linux-aarch64"),
    ],
)
def test_runtime_platform_key_canonicalizes_supported_architectures(
    runtime: str,
    machine: str,
    expected: str,
) -> None:
    assert voice_physical_reports._runtime_platform_key(runtime, machine) == expected


@pytest.mark.parametrize(
    ("runtime", "machine"),
    [("freebsd13", "x86_64"), ("darwin", "ppc64"), ("win32", "ARM64")],
)
def test_runtime_platform_key_rejects_unsupported_runtime_or_architecture(
    runtime: str,
    machine: str,
) -> None:
    with pytest.raises(PhysicalVoiceReportError, match="unsupported"):
        voice_physical_reports._runtime_platform_key(runtime, machine)


def test_normal_cli_rejects_wrong_platform_before_prerequisites_or_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    args = _cli_args(tmp_path, output=output)
    runtime_key = args[args.index("--platform") + 1]
    requested_key = "linux-x86_64" if runtime_key != "linux-x86_64" else "macos-arm64"
    args[args.index("--platform") + 1] = requested_key

    monkeypatch.setattr(
        voice_physical_reports,
        "_runtime_platform_key",
        lambda *_args: runtime_key,
        raising=False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "load_automated_prerequisites",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prerequisites must not load")
        ),
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("hardware must not open")
        ),
    )

    with pytest.raises(SystemExit) as raised:
        main(args)

    assert raised.value.code == 2
    assert not output.exists()
    assert not output.with_suffix(".summary.txt").exists()


def test_fixture_cli_may_simulate_non_runtime_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_key = voice_physical_reports._runtime_platform_key()
    simulated_key = (
        "windows-x86_64" if runtime_key != "windows-x86_64" else "macos-arm64"
    )
    _prerequisites(tmp_path, platform_key=simulated_key)
    monkeypatch.setattr(
        voice_physical_reports,
        "_runtime_platform_key",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("fixture mode must not inspect runtime platform")
        ),
        raising=False,
    )

    assert (
        main(
            [
                *_cli_args(
                    tmp_path,
                    device_class="builtin",
                    platform_key=simulated_key,
                ),
                "--fixture",
                "safe-full-duplex",
            ]
        )
        == 0
    )


def test_completed_unsafe_live_trial_writes_valid_failed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    observations = load_fixture_observations(_FIXTURES / "unsafe_unsuppressed.json")
    output = tmp_path / "unsafe-physical.json"
    live_calls = 0

    async def collect_live(**kwargs: object) -> dict[str, object]:
        nonlocal live_calls
        live_calls += 1
        assert kwargs == {"device_class": "bluetooth"}
        return observations

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        collect_live,
        raising=False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_guided_observations",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual observations are forbidden")
        ),
        raising=False,
    )

    result = main(_cli_args(tmp_path, output=output, device_class="bluetooth"))

    report = load_fixture_observations(output)
    validate_physical_report(report)
    assert result == 1
    assert live_calls == 1
    assert report["evidence_kind"] == "physical"
    assert report["passed"] is False
    assert "Result: FAIL" in output.with_suffix(".summary.txt").read_text(
        encoding="utf-8"
    )


def test_live_infrastructure_abort_writes_no_report_or_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"

    async def fail_live(**_kwargs: object) -> dict[str, object]:
        raise PhysicalVoiceRunnerError("microphone permission denied")

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        fail_live,
        raising=False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_guided_observations",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual observations are forbidden")
        ),
        raising=False,
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert not output.exists()
    assert not output.with_suffix(".summary.txt").exists()


@pytest.mark.parametrize("target_kind", ["symlink", "broken-symlink", "directory"])
def test_cli_rejects_unsafe_summary_target_before_writing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    _prerequisites(tmp_path)
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    output = tmp_path / "physical.json"
    summary_path = output.with_suffix(".summary.txt")
    summary_target = tmp_path / "existing-summary.txt"
    if target_kind == "symlink":
        summary_target.write_text("unchanged", encoding="utf-8")
        summary_path.symlink_to(summary_target)
    elif target_kind == "broken-symlink":
        summary_path.symlink_to(summary_target)
    else:
        summary_path.mkdir()

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: asyncio.sleep(0, result=observations),
        raising=False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_guided_observations",
        lambda **_kwargs: observations,
        raising=False,
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert not output.exists()
    if target_kind == "symlink":
        assert summary_target.read_text(encoding="utf-8") == "unchanged"
    elif target_kind == "broken-symlink":
        assert not summary_target.exists()
    else:
        assert summary_path.is_dir()


def test_fixture_cli_is_synchronous_and_does_not_touch_live_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixture mode must not invoke the live runner")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "asyncio",
        SimpleNamespace(
            run=lambda _coroutine: (_ for _ in ()).throw(
                AssertionError("fixture mode must remain synchronous")
            )
        ),
        raising=False,
    )

    result = main(
        [
            *_cli_args(tmp_path),
            "--fixture",
            "safe-isolated-full-duplex",
        ]
    )

    assert result == 0
    assert "sounddevice" not in sys.modules


@pytest.mark.parametrize("failure_point", ["open", "start"])
def test_raw_transport_startup_failure_is_a_controlled_cli_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"

    class FailingOpenBackend(FakeDuplexBackend):
        def open_stream(self, **kwargs: object):
            self.open_calls.append(dict(kwargs))
            raise OSError("raw stream open failure")

    backend = (
        FailingOpenBackend()
        if failure_point == "open"
        else FakeDuplexBackend(start_failures=1)
    )
    route = RouteIdentity(1, 2, "input", "output", 1, 1, 48_000)
    runner = PhysicalVoiceTrialRunner(
        device_class="usb",
        config=LiveTrialConfig(
            soak_seconds=0.01,
            route_round_trips=1,
            interruption_trials=1,
            double_talk_trials=1,
        ),
        transport_factory=lambda: DuplexAudioTransport(backend=backend),
        route_reader=lambda: route,
    )

    async def collect_live(**_kwargs: object) -> dict[str, object]:
        return await runner.run()

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports, "collect_live_observations", collect_live
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert backend.open_count == 1
    assert not output.exists()
    assert not output.with_suffix(".summary.txt").exists()


def test_programming_assertion_from_live_runner_is_not_hidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)

    async def collect_live(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("programming defect")

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports, "collect_live_observations", collect_live
    )

    with pytest.raises(AssertionError, match="programming defect"):
        main(_cli_args(tmp_path))


def test_raw_live_value_error_is_a_controlled_cli_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)

    async def collect_live(**_kwargs: object) -> dict[str, object]:
        raise ValueError("invalid native stream latency")

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports, "collect_live_observations", collect_live
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path))

    assert raised.value.code == 2
    assert not (tmp_path / "physical.json").exists()


def test_cli_rejects_preexisting_report_pair_before_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    summary = output.with_suffix(".summary.txt")
    existing_report = _report(tmp_path, "safe_full_duplex")
    write_physical_report(output, existing_report)
    summary.write_text(
        voice_physical_reports.physical_report_summary(existing_report),
        encoding="utf-8",
    )
    existing_report_bytes = output.read_bytes()
    existing_summary_bytes = summary.read_bytes()
    live_calls = 0

    async def collect_live(**_kwargs: object) -> dict[str, object]:
        nonlocal live_calls
        live_calls += 1
        raise AssertionError("hardware must not open")

    monkeypatch.setattr(
        voice_physical_reports, "collect_live_observations", collect_live
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert live_calls == 0
    assert output.read_bytes() == existing_report_bytes
    assert summary.read_bytes() == existing_summary_bytes


@pytest.mark.parametrize("target_name", ["report", "summary"])
@pytest.mark.parametrize("link_kind", ["live", "broken"])
def test_cli_rejects_every_symlink_target_before_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    link_kind: str,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    summary = output.with_suffix(".summary.txt")
    target = output if target_name == "report" else summary
    victim = tmp_path / f"{target_name}-victim"
    if link_kind == "live":
        victim.write_bytes(b"unchanged")
    target.symlink_to(victim)
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("hardware opened")),
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert target.is_symlink()
    if link_kind == "live":
        assert victim.read_bytes() == b"unchanged"
    else:
        assert not victim.exists()


def _assert_no_publication_staging(tmp_path: Path) -> None:
    assert not [path for path in tmp_path.iterdir() if ".voice-stage-" in path.name]


def test_summary_symlink_swap_during_collection_rolls_back_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    summary = output.with_suffix(".summary.txt")
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"unchanged")
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )

    async def collect_live(**_kwargs: object) -> dict[str, object]:
        summary.symlink_to(victim)
        return observations

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports, "collect_live_observations", collect_live
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert not output.exists()
    assert summary.is_symlink()
    assert victim.read_bytes() == b"unchanged"
    _assert_no_publication_staging(tmp_path)


@pytest.mark.parametrize(
    "failure_point",
    ["report-stage", "summary-stage", "first-publish", "second-publish"],
)
def test_publication_failure_leaves_neither_artifact_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    real_open = os.open
    real_link = os.link
    stage_calls = 0
    publish_calls = 0

    def failing_open(path: object, flags: int, mode: int = 0o777, **kwargs: object):
        nonlocal stage_calls
        if ".voice-stage-" in os.fspath(path):
            stage_calls += 1
            if failure_point == (
                "report-stage" if stage_calls == 1 else "summary-stage"
            ):
                raise PermissionError("injected staging failure")
        return real_open(path, flags, mode, **kwargs)

    def failing_link(src: object, dst: object, **kwargs: object):
        nonlocal publish_calls
        publish_calls += 1
        if failure_point == (
            "first-publish" if publish_calls == 1 else "second-publish"
        ):
            raise PermissionError("injected publication failure")
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os, "open", failing_open)
    monkeypatch.setattr(os, "link", failing_link)
    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: asyncio.sleep(0, result=observations),
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert not os.path.lexists(output)
    assert not os.path.lexists(output.with_suffix(".summary.txt"))
    _assert_no_publication_staging(tmp_path)


def test_unsafe_trial_second_publication_failure_rolls_back_failed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "unsafe.json"
    observations = load_fixture_observations(_FIXTURES / "unsafe_unsuppressed.json")
    real_link = os.link
    calls = 0

    def fail_second_link(src: object, dst: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("injected summary publication failure")
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os, "link", fail_second_link)
    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: asyncio.sleep(0, result=observations),
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output, device_class="bluetooth"))

    assert raised.value.code == 2
    assert not os.path.lexists(output)
    assert not os.path.lexists(output.with_suffix(".summary.txt"))
    _assert_no_publication_staging(tmp_path)


def test_public_writer_rejects_broken_symlink_and_replaces_regular_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    output = tmp_path / "report.json"
    victim = tmp_path / "missing-victim.json"
    output.symlink_to(victim)
    with pytest.raises(PhysicalVoiceReportError, match="symlink"):
        write_physical_report(output, report)
    assert not victim.exists()

    output.unlink()
    output.write_bytes(b"old")
    real_replace = os.replace
    monkeypatch.setattr(
        os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("injected replace failure")
        ),
    )
    with pytest.raises(PhysicalVoiceReportError):
        write_physical_report(output, report)
    assert output.read_bytes() == b"old"
    _assert_no_publication_staging(tmp_path)
    monkeypatch.setattr(os, "replace", real_replace)
    write_physical_report(output, report)
    validate_physical_report(json.loads(output.read_bytes()))


def test_public_writer_restores_existing_report_when_durability_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_report = _report(tmp_path, "safe_full_duplex")
    new_report = _report(
        tmp_path,
        "safe_full_duplex",
        salt=b"fedcba9876543210",
    )
    output = tmp_path / "report.json"
    write_physical_report(output, old_report)
    old_bytes = output.read_bytes()
    real_replace = os.replace
    real_fsync = os.fsync
    replacement_started = False
    injected = False

    def record_replace(src: object, dst: object, **kwargs: object):
        nonlocal replacement_started
        replacement_started = True
        return real_replace(src, dst, **kwargs)

    def fail_replacement_sync(descriptor: int) -> None:
        nonlocal injected
        if (
            replacement_started
            and not injected
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
        ):
            injected = True
            raise PermissionError("replacement durability denied")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "replace", record_replace)
    monkeypatch.setattr(os, "fsync", fail_replacement_sync)

    with pytest.raises(PhysicalVoiceReportError):
        write_physical_report(output, new_report)

    assert output.read_bytes() == old_bytes
    _assert_no_publication_staging(tmp_path)
    assert not [path for path in tmp_path.iterdir() if ".voice-backup-" in path.name]


def test_public_writer_restore_failure_is_reported_truthfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_report = _report(tmp_path, "safe_full_duplex")
    new_report = _report(
        tmp_path,
        "safe_full_duplex",
        salt=b"fedcba9876543210",
    )
    output = tmp_path / "report.json"
    write_physical_report(output, old_report)
    old_bytes = output.read_bytes()
    real_replace = os.replace
    real_fsync = os.fsync
    replacement_started = False
    injected = False

    def fail_backup_restore(src: object, dst: object, **kwargs: object):
        nonlocal replacement_started
        replacement_started = True
        if ".voice-backup-" in os.fspath(src):
            raise PermissionError("backup restore denied")
        return real_replace(src, dst, **kwargs)

    def fail_replacement_sync(descriptor: int) -> None:
        nonlocal injected
        if (
            replacement_started
            and not injected
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
        ):
            injected = True
            raise PermissionError("replacement durability denied")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "replace", fail_backup_restore)
    monkeypatch.setattr(os, "fsync", fail_replacement_sync)

    with pytest.raises(PhysicalVoiceReportError, match="restore|rollback"):
        write_physical_report(output, new_report)

    backups = [path for path in tmp_path.iterdir() if ".voice-backup-" in path.name]
    assert output.read_bytes() != old_bytes
    assert len(backups) == 1
    assert backups[0].read_bytes() == old_bytes


def test_public_writer_rejects_post_replace_swap_without_touching_attacker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_report = _report(tmp_path, "safe_full_duplex")
    new_report = _report(
        tmp_path,
        "safe_full_duplex",
        salt=b"fedcba9876543210",
    )
    output = tmp_path / "report.json"
    write_physical_report(output, old_report)
    old_bytes = output.read_bytes()
    attacker_bytes = b"attacker-owned\n"
    real_replace = os.replace
    swapped = False

    def replace_then_swap(src: object, dst: object, **kwargs: object):
        nonlocal swapped
        result = real_replace(src, dst, **kwargs)
        if not swapped and ".voice-stage-" in os.fspath(src):
            swapped = True
            output.unlink()
            output.write_bytes(attacker_bytes)
        return result

    monkeypatch.setattr(os, "replace", replace_then_swap)

    with pytest.raises(PhysicalVoiceReportError, match="publication|identity"):
        write_physical_report(output, new_report)

    backups = [path for path in tmp_path.iterdir() if ".voice-backup-" in path.name]
    assert swapped is True
    assert output.read_bytes() == attacker_bytes
    assert len(backups) == 1
    assert backups[0].read_bytes() == old_bytes
    _assert_no_publication_staging(tmp_path)


def test_public_writer_removes_fresh_report_when_durability_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    output = tmp_path / "report.json"
    real_fsync = os.fsync

    def fail_directory_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PermissionError("replacement durability denied")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_sync)

    with pytest.raises(PhysicalVoiceReportError):
        write_physical_report(output, report)

    assert not output.exists()
    _assert_no_publication_staging(tmp_path)


def test_summary_is_durably_published_before_authoritative_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not voice_physical_reports._NATIVE_DIRECTORY_FD_OPERATIONS:
        pytest.skip("directory fsync ordering requires descriptor-relative support")
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    events: list[tuple[str, str]] = []
    real_link = os.link
    real_fsync = os.fsync

    def record_link(src: object, dst: object, **kwargs: object):
        events.append(("link", Path(os.fspath(dst)).name))
        return real_link(src, dst, **kwargs)

    def record_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "link", record_link)
    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: asyncio.sleep(0, result=observations),
    )

    assert main(_cli_args(tmp_path, output=output)) == 0

    summary_link = events.index(("link", "physical.summary.txt"))
    report_link = events.index(("link", "physical.json"))
    assert summary_link < report_link
    assert ("fsync", "directory") in events[summary_link + 1 : report_link]


def test_summary_publication_failure_never_attempts_authoritative_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    destinations: list[str] = []

    def fail_first_link(_src: object, dst: object, **_kwargs: object):
        destinations.append(Path(os.fspath(dst)).name)
        raise PermissionError("summary publication denied")

    monkeypatch.setattr(os, "link", fail_first_link)
    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: asyncio.sleep(0, result=observations),
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert destinations == ["physical.summary.txt"]
    assert not output.exists()


def test_summary_finalization_failure_never_attempts_authoritative_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not voice_physical_reports._NATIVE_DIRECTORY_FD_OPERATIONS:
        pytest.skip("directory fsync failure requires descriptor-relative support")
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    real_fsync = os.fsync
    real_link = os.link
    destinations: list[str] = []
    directory_syncs = 0

    def fail_first_directory_sync(descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
            if directory_syncs == 1:
                raise PermissionError("summary finalization denied")
        real_fsync(descriptor)

    def record_link(src: object, dst: object, **kwargs: object):
        destinations.append(Path(os.fspath(dst)).name)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os, "fsync", fail_first_directory_sync)
    monkeypatch.setattr(os, "link", record_link)
    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: asyncio.sleep(0, result=observations),
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert destinations == ["physical.summary.txt"]
    assert not output.exists()
    assert not output.with_suffix(".summary.txt").exists()
    _assert_no_publication_staging(tmp_path)


def test_report_publish_and_summary_rollback_failures_leave_no_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    summary = output.with_suffix(".summary.txt")
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    real_link = os.link
    real_unlink = os.unlink

    def fail_report_link(src: object, dst: object, **kwargs: object):
        if Path(os.fspath(dst)).name == output.name:
            raise PermissionError("report publication denied")
        return real_link(src, dst, **kwargs)

    def fail_summary_cleanup(path: object, **kwargs: object):
        if Path(os.fspath(path)).name == summary.name:
            raise PermissionError("summary rollback denied")
        return real_unlink(path, **kwargs)

    monkeypatch.setattr(os, "link", fail_report_link)
    monkeypatch.setattr(os, "unlink", fail_summary_cleanup)
    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: asyncio.sleep(0, result=observations),
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert not output.exists()
    assert summary.exists()
    assert "rollback" in capsys.readouterr().err


def test_report_finalization_and_report_rollback_failures_retain_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not voice_physical_reports._NATIVE_DIRECTORY_FD_OPERATIONS:
        pytest.skip("directory fsync failure requires descriptor-relative support")
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    summary = output.with_suffix(".summary.txt")
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    real_fsync = os.fsync
    real_unlink = os.unlink
    directory_syncs = 0

    def fail_second_directory_sync(descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
            if directory_syncs == 2:
                raise PermissionError("report finalization denied")
        real_fsync(descriptor)

    def fail_report_rollback(path: object, **kwargs: object):
        if Path(os.fspath(path)).name == output.name:
            raise PermissionError("report rollback denied")
        return real_unlink(path, **kwargs)

    monkeypatch.setattr(os, "fsync", fail_second_directory_sync)
    monkeypatch.setattr(os, "unlink", fail_report_rollback)
    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: asyncio.sleep(0, result=observations),
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert output.is_file()
    assert summary.is_file()
    assert "rollback" in capsys.readouterr().err


@pytest.mark.parametrize("replacement_kind", ["directory", "symlink"])
def test_parent_replacement_during_collection_cannot_receive_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    _prerequisites(tmp_path)
    parent = tmp_path / "qualification"
    output = parent / "physical.json"
    moved_parent = tmp_path / "qualification-moved"
    victim = tmp_path / "victim"
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )

    async def replace_parent(**_kwargs: object) -> dict[str, object]:
        parent.rename(moved_parent)
        if replacement_kind == "directory":
            parent.mkdir()
        else:
            victim.mkdir()
            parent.symlink_to(victim, target_is_directory=True)
        return observations

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports, "collect_live_observations", replace_parent
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert not (moved_parent / output.name).exists()
    assert not (moved_parent / "physical.summary.txt").exists()
    replacement = parent if replacement_kind == "directory" else victim
    assert not (replacement / output.name).exists()
    assert not (replacement / "physical.summary.txt").exists()
    _assert_no_publication_staging(moved_parent)


def test_parent_swap_between_staging_and_publish_uses_pinned_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not voice_physical_reports._NATIVE_DIRECTORY_FD_OPERATIONS:
        pytest.skip("pinned cleanup requires descriptor-relative support")
    _prerequisites(tmp_path)
    parent = tmp_path / "qualification"
    output = parent / "physical.json"
    moved_parent = tmp_path / "qualification-moved"
    victim = tmp_path / "victim"
    victim.mkdir()
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )
    real_link = os.link
    swapped = False

    def swap_then_link(src: object, dst: object, **kwargs: object):
        nonlocal swapped
        if not swapped:
            swapped = True
            parent.rename(moved_parent)
            parent.symlink_to(victim, target_is_directory=True)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os, "link", swap_then_link)
    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        lambda **_kwargs: asyncio.sleep(0, result=observations),
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert not (moved_parent / output.name).exists()
    assert not (moved_parent / "physical.summary.txt").exists()
    assert not (victim / output.name).exists()
    assert not (victim / "physical.summary.txt").exists()
    _assert_no_publication_staging(moved_parent)


class _FakeWindowsDirectoryPin:
    def __init__(self, *, fail_move: int | None = None) -> None:
        self.closed = False
        self.fail_move = fail_move
        self.moves: list[tuple[str, str, bool]] = []

    def close(self) -> None:
        self.closed = True

    def revalidate(self) -> None:
        if self.closed:
            raise PhysicalVoiceReportError("directory pin is closed")

    def deny_parent_swap(self) -> None:
        if not self.closed:
            raise PermissionError("directory handle denies delete sharing")

    def move_write_through(
        self,
        source: Path,
        destination: Path,
        *,
        replace: bool,
    ) -> None:
        self.moves.append((source.name, destination.name, replace))
        if self.fail_move == len(self.moves):
            raise PermissionError("injected write-through move failure")
        if replace:
            os.replace(source, destination)
            return
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(source, destination)


def test_windows_pin_uses_write_through_moves_and_closes_handle() -> None:
    moves: list[tuple[str, str, int]] = []
    closed: list[int] = []
    identity = voice_physical_reports._WindowsFileIdentity(
        7,
        b"\x00" * 15 + b"\x80",
    )
    pin = voice_physical_reports._WindowsDirectoryPin(
        path=Path("parent"),
        identity=identity,
        handle=41,
        _open_and_identify=lambda _path: (42, identity),
        _close_handle=lambda handle: closed.append(handle) or 1,
        _move_file_ex=lambda source, destination, flags: (
            moves.append((source, destination, flags)) or 1
        ),
        _get_last_error=lambda: 0,
    )

    pin.revalidate()
    pin.move_write_through(Path("stage"), Path("summary"), replace=False)
    pin.move_write_through(Path("backup"), Path("report"), replace=True)
    pin.close()

    assert moves == [
        ("stage", "summary", 0x00000008),
        ("backup", "report", 0x00000009),
    ]
    assert closed == [42, 41]


@pytest.mark.parametrize(
    "verifier_identity",
    [
        pytest.param(
            (8, b"\x00" * 15 + b"\x80"),
            id="volume-mismatch",
        ),
        pytest.param(
            (7, b"\x00" * 14 + b"\x01\x80"),
            id="128-bit-file-id-mismatch",
        ),
    ],
)
def test_windows_pin_revalidation_uses_exact_win32_identity_and_closes_verifier(
    verifier_identity: tuple[int, bytes],
) -> None:
    closed: list[int] = []
    identity = voice_physical_reports._WindowsFileIdentity(
        7,
        b"\x00" * 15 + b"\x80",
    )
    pin = voice_physical_reports._WindowsDirectoryPin(
        path=Path("parent"),
        identity=identity,
        handle=41,
        _open_and_identify=lambda _path: (
            42,
            voice_physical_reports._WindowsFileIdentity(*verifier_identity),
        ),
        _close_handle=lambda handle: closed.append(handle) or 1,
        _move_file_ex=lambda *_args: 1,
        _get_last_error=lambda: 0,
    )

    with pytest.raises(PhysicalVoiceReportError, match="identity changed"):
        pin.revalidate()

    assert closed == [42]
    pin.close()
    assert closed == [42, 41]


def _install_fake_windows_directory_pin(
    monkeypatch: pytest.MonkeyPatch,
    pin: _FakeWindowsDirectoryPin,
) -> None:
    monkeypatch.setattr(
        voice_physical_reports,
        "_supports_pinned_directory_operations",
        lambda: False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "_directory_backend",
        lambda: "windows",
        raising=False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "_open_windows_directory",
        lambda _path: pin,
        raising=False,
    )


@pytest.mark.skipif(os.name != "nt", reason="requires the real Windows API")
def test_real_windows_directory_pin_reports_identity_and_denies_rename(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "pinned"
    parent.mkdir()
    moved = tmp_path / "moved"
    pin = voice_physical_reports._open_windows_directory(parent)
    try:
        assert pin.identity.volume_serial_number >= 0
        assert len(pin.identity.file_id) == 16
        pin.revalidate()
        with pytest.raises(OSError):
            parent.rename(moved)
        with pytest.raises(OSError):
            parent.rmdir()
    finally:
        pin.close()

    parent.rename(moved)
    moved.rename(parent)


def test_unsupported_directory_backend_fails_before_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    live_calls = 0

    async def collect_live(**_kwargs: object) -> dict[str, object]:
        nonlocal live_calls
        live_calls += 1
        raise AssertionError("hardware must not open")

    monkeypatch.setattr(
        voice_physical_reports,
        "_supports_pinned_directory_operations",
        lambda: False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "_directory_backend",
        lambda: "unsupported",
        raising=False,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        collect_live,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=output))

    assert raised.value.code == 2
    assert live_calls == 0
    assert not output.exists()


def test_windows_directory_pin_is_held_across_live_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    pin = _FakeWindowsDirectoryPin()
    _install_fake_windows_directory_pin(monkeypatch, pin)
    observations = load_fixture_observations(
        _FIXTURES / "safe_isolated_full_duplex.json"
    )

    async def collect_live(**_kwargs: object) -> dict[str, object]:
        with pytest.raises(PermissionError, match="delete sharing"):
            pin.deny_parent_swap()
        return observations

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports, "collect_live_observations", collect_live
    )

    assert main(_cli_args(tmp_path, output=output)) == 0

    assert pin.closed is True
    assert [move[1:] for move in pin.moves] == [
        ("physical.summary.txt", False),
        ("physical.json", False),
    ]
    assert pin.moves[0][0].startswith(".physical.summary.txt.voice-stage-")
    assert pin.moves[1][0].startswith(".physical.json.voice-stage-")
    assert output.is_file()
    assert output.with_suffix(".summary.txt").is_file()


@pytest.mark.parametrize("failed_move", [1, 2])
def test_windows_write_through_publication_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_move: int,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    pin = _FakeWindowsDirectoryPin(fail_move=failed_move)
    _install_fake_windows_directory_pin(monkeypatch, pin)

    with pytest.raises(SystemExit) as raised:
        main(
            [
                *_cli_args(tmp_path, output=output),
                "--fixture",
                "safe-isolated-full-duplex",
            ]
        )

    assert raised.value.code == 2
    assert pin.closed is True
    assert [move[1:] for move in pin.moves] == (
        [("physical.summary.txt", False)]
        if failed_move == 1
        else [
            ("physical.summary.txt", False),
            ("physical.json", False),
        ]
    )
    assert not output.exists()
    assert not output.with_suffix(".summary.txt").exists()
    _assert_no_publication_staging(tmp_path)


def test_windows_stage_failure_cleans_stage_and_closes_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    pin = _FakeWindowsDirectoryPin()
    _install_fake_windows_directory_pin(monkeypatch, pin)
    real_open = os.open
    real_fsync = os.fsync
    denied_swap = False

    def race_stage_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        **kwargs: object,
    ) -> int:
        nonlocal denied_swap
        if ".voice-stage-" in os.fspath(path):
            with pytest.raises(PermissionError, match="delete sharing"):
                pin.deny_parent_swap()
            denied_swap = True
        return real_open(path, flags, mode, **kwargs)

    def fail_first_stage_sync(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PermissionError("injected stage sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "open", race_stage_open)
    monkeypatch.setattr(os, "fsync", fail_first_stage_sync)

    with pytest.raises(SystemExit) as raised:
        main(
            [
                *_cli_args(tmp_path, output=output),
                "--fixture",
                "safe-isolated-full-duplex",
            ]
        )

    assert raised.value.code == 2
    assert denied_swap is True
    assert pin.closed is True
    assert not output.exists()
    _assert_no_publication_staging(tmp_path)


@pytest.mark.parametrize(
    "primary",
    [
        KeyboardInterrupt("operator interrupt"),
        asyncio.CancelledError("collection cancelled"),
        AssertionError("programming failure"),
    ],
)
def test_main_close_failure_does_not_replace_primary_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary: BaseException,
) -> None:
    _prerequisites(tmp_path)
    real_close = voice_physical_reports._OutputDirectory.close

    async def fail_collection(**_kwargs: object) -> dict[str, object]:
        raise primary

    def close_then_fail(directory: object) -> None:
        real_close(directory)
        raise PhysicalVoiceReportError("directory close denied")

    monkeypatch.setattr(
        voice_physical_reports,
        "compute_voice_source_digest",
        lambda **_kwargs: _DIGEST,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "collect_live_observations",
        fail_collection,
    )
    monkeypatch.setattr(
        voice_physical_reports._OutputDirectory,
        "close",
        close_then_fail,
    )

    with pytest.raises(type(primary)) as raised:
        main(_cli_args(tmp_path))

    assert raised.value is primary
    assert any("close" in note for note in getattr(primary, "__notes__", ()))


def test_public_writer_close_failure_does_not_replace_primary_assertion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    primary = AssertionError("programming failure")
    real_close = voice_physical_reports._OutputDirectory.close

    def fail_stage(*_args: object, **_kwargs: object):
        raise primary

    def close_then_fail(directory: object) -> None:
        real_close(directory)
        raise PhysicalVoiceReportError("directory close denied")

    monkeypatch.setattr(voice_physical_reports, "_stage_bytes", fail_stage)
    monkeypatch.setattr(
        voice_physical_reports._OutputDirectory,
        "close",
        close_then_fail,
    )

    with pytest.raises(AssertionError) as raised:
        write_physical_report(tmp_path / "report.json", report)

    assert raised.value is primary
    assert any("close" in note for note in getattr(primary, "__notes__", ()))


def test_public_writer_annotates_primary_domain_error_with_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path, "safe_full_duplex")
    primary = PhysicalVoiceReportError("staging denied")
    real_close = voice_physical_reports._OutputDirectory.close

    def fail_stage(*_args: object, **_kwargs: object):
        raise primary

    def close_then_fail(directory: object) -> None:
        real_close(directory)
        raise PhysicalVoiceReportError("directory close denied")

    monkeypatch.setattr(voice_physical_reports, "_stage_bytes", fail_stage)
    monkeypatch.setattr(
        voice_physical_reports._OutputDirectory,
        "close",
        close_then_fail,
    )

    with pytest.raises(PhysicalVoiceReportError) as raised:
        write_physical_report(tmp_path / "report.json", report)

    assert raised.value is primary
    assert any("close" in note for note in getattr(primary, "__notes__", ()))


def test_pair_cleanup_and_close_failures_do_not_replace_primary_assertion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    output = tmp_path / "physical.json"
    primary = AssertionError("programming failure")
    real_close = voice_physical_reports._OutputDirectory.close

    def fail_publication(*_args: object, **_kwargs: object):
        raise primary

    def close_then_fail(directory: object) -> None:
        real_close(directory)
        raise PhysicalVoiceReportError("directory close denied")

    monkeypatch.setattr(
        voice_physical_reports,
        "_publish_exclusive",
        fail_publication,
    )
    monkeypatch.setattr(
        voice_physical_reports._OutputDirectory,
        "close",
        close_then_fail,
    )

    with pytest.raises(AssertionError) as raised:
        main(
            [
                *_cli_args(tmp_path, output=output),
                "--fixture",
                "safe-isolated-full-duplex",
            ]
        )

    assert raised.value is primary
    assert any("close" in note for note in getattr(primary, "__notes__", ()))
    assert not output.exists()
    assert not output.with_suffix(".summary.txt").exists()
    _assert_no_publication_staging(tmp_path)


def test_pair_rollback_assertion_does_not_replace_primary_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prerequisites(tmp_path)
    primary = asyncio.CancelledError("publication cancelled")

    def fail_publication(*_args: object, **_kwargs: object):
        raise primary

    def fail_cleanup(*_args: object, **_kwargs: object):
        raise AssertionError("rollback programming failure")

    monkeypatch.setattr(
        voice_physical_reports,
        "_publish_exclusive",
        fail_publication,
    )
    monkeypatch.setattr(
        voice_physical_reports,
        "_unlink_if_owned",
        fail_cleanup,
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        main(
            [
                *_cli_args(tmp_path),
                "--fixture",
                "safe-isolated-full-duplex",
            ]
        )

    assert raised.value is primary
    assert any("rollback" in note for note in getattr(primary, "__notes__", ()))


def test_windows_path_cleanup_failure_does_not_replace_primary_assertion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = _FakeWindowsDirectoryPin()
    primary = AssertionError("link programming failure")
    revalidations = 0
    directory = voice_physical_reports._OutputDirectory(
        tmp_path,
        voice_physical_reports._FileIdentity(
            tmp_path.stat().st_dev,
            tmp_path.stat().st_ino,
        ),
        None,
        pin,
    )
    real_revalidate = voice_physical_reports._OutputDirectory.revalidate

    def fail_second_revalidation(target: object) -> None:
        nonlocal revalidations
        revalidations += 1
        if revalidations == 2:
            raise PhysicalVoiceReportError("parent identity cleanup failed")
        real_revalidate(target)

    monkeypatch.setattr(
        voice_physical_reports._OutputDirectory,
        "revalidate",
        fail_second_revalidation,
    )
    monkeypatch.setattr(
        os, "link", lambda *_args, **_kwargs: (_ for _ in ()).throw(primary)
    )

    with pytest.raises(AssertionError) as raised:
        directory.link("source", "destination")

    assert raised.value is primary
    assert any("identity" in note for note in getattr(primary, "__notes__", ()))
    directory.close()


def test_stage_cleanup_failure_does_not_replace_primary_assertion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = AssertionError("programming failure")
    real_close = os.close
    stage_descriptor: int | None = None

    def fail_write(descriptor: int, _data: object) -> int:
        nonlocal stage_descriptor
        stage_descriptor = descriptor
        raise primary

    def fail_stage_close(descriptor: int) -> None:
        if descriptor == stage_descriptor:
            raise OSError("stage descriptor close denied")
        real_close(descriptor)

    monkeypatch.setattr(os, "write", fail_write)
    monkeypatch.setattr(os, "close", fail_stage_close)
    try:
        with voice_physical_reports._OutputDirectory.acquire(tmp_path) as directory:
            with pytest.raises(AssertionError) as raised:
                voice_physical_reports._stage_bytes(
                    directory,
                    "report.json",
                    b"{}\n",
                )

        assert raised.value is primary
        assert any("close" in note for note in getattr(primary, "__notes__", ()))
        _assert_no_publication_staging(tmp_path)
    finally:
        if stage_descriptor is not None:
            real_close(stage_descriptor)


def test_stage_close_failure_never_retries_a_reused_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_close = os.close
    real_open = os.open
    replacement_descriptor: int | None = None
    injected = False

    def close_release_reuse_then_fail(descriptor: int) -> None:
        nonlocal injected, replacement_descriptor
        if not injected and stat.S_ISREG(os.fstat(descriptor).st_mode):
            injected = True
            real_close(descriptor)
            replacement_descriptor = real_open(os.devnull, os.O_RDONLY)
            assert replacement_descriptor == descriptor
            raise OSError("close status is ambiguous after descriptor release")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close_release_reuse_then_fail)
    try:
        with voice_physical_reports._OutputDirectory.acquire(tmp_path) as directory:
            with pytest.raises(PhysicalVoiceReportError, match="staging"):
                voice_physical_reports._stage_bytes(
                    directory,
                    "report.json",
                    b"{}\n",
                )

        assert injected is True
        assert replacement_descriptor is not None
        os.fstat(replacement_descriptor)
        _assert_no_publication_staging(tmp_path)
    finally:
        if replacement_descriptor is not None:
            real_close(replacement_descriptor)


@pytest.mark.parametrize("unsafe_output", ["/", ".", "bad\0name.json"])
def test_invalid_output_paths_are_controlled_cli_errors(
    tmp_path: Path,
    unsafe_output: str,
) -> None:
    _prerequisites(tmp_path)

    with pytest.raises(SystemExit) as raised:
        main(_cli_args(tmp_path, output=Path(unsafe_output)))

    assert raised.value.code == 2
