"""Source-bound manifest generation and rollout-authority tests."""

from __future__ import annotations

import copy
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
import statistics
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from Packaging.generate_voice_qualification_manifest import (
    PLATFORM_KEYS,
    VoiceManifestError,
    generate_voice_qualification_manifest,
    validate_completed_pair_history_gate,
    validate_voice_qualification_manifest,
)
from Packaging.qualify_speculative_voice import (
    assemble_automated_report,
    assemble_soak_report,
    write_qualification_report,
)
from Packaging.speculative_voice_history_gate import (
    _locator_distribution,
    _sample_summary as history_sample_summary,
    completed_pair_locator_inventory,
)
from Packaging.speculative_voice_latency_gate import _summary as latency_sample_summary
from Packaging.voice_physical_reports import (
    build_physical_report,
    load_automated_prerequisites,
    load_fixture_observations,
    write_physical_report,
)


_ROOT = Path(__file__).resolve().parents[2]
_PHYSICAL_FIXTURES = Path(__file__).with_name("fixtures") / "speculative_voice_physical"
_DIGEST = "d" * 64
_VERSION = "0.1.8.0"
_UPSTREAM = "109e23c9cec3a44e67c08774874a409741b1e58a"
_WHEEL_NAMES = {
    "macos-arm64": "tldw_voice_aec-0.1.8.0-cp312-cp312-macosx_15_0_arm64.whl",
    "macos-x86_64": "tldw_voice_aec-0.1.8.0-cp312-cp312-macosx_13_0_x86_64.whl",
    "windows-x86_64": "tldw_voice_aec-0.1.8.0-cp312-cp312-win_amd64.whl",
    "linux-x86_64": "tldw_voice_aec-0.1.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    "linux-aarch64": "tldw_voice_aec-0.1.8.0-cp312-cp312-manylinux_2_28_aarch64.whl",
}


def _interpreter(platform_key: str) -> dict[str, object]:
    system_name, architecture = platform_key.split("-", 1)
    return {
        "implementation": "cpython",
        "version": "3.12.11",
        "system": {
            "macos": "Darwin",
            "windows": "Windows",
            "linux": "Linux",
        }[system_name],
        "machine": {
            "arm64": "arm64",
            "aarch64": "aarch64",
            "x86_64": "x86_64",
        }[architecture],
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_raw_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def _summary(values: list[float]) -> dict[str, object]:
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "p95_ms": _p95(values),
    }


@pytest.mark.parametrize(
    "summary",
    [history_sample_summary, latency_sample_summary],
)
def test_serialized_sample_summaries_derive_from_serialized_samples(
    summary: Callable[[list[float]], dict[str, object]],
) -> None:
    report = summary([0.04858251, 0.06012551])
    samples = report["samples_ms"]
    assert isinstance(samples, list)
    assert report["median_ms"] == round(statistics.median(samples), 6)
    assert report["p95_ms"] == round(_p95(samples), 6)


def _passing_history_gate() -> dict[str, object]:
    inventory = completed_pair_locator_inventory()
    distribution = _locator_distribution()
    plans = [
        {
            "table": locator["table"],
            "column": locator["column"],
            "details": ["SEARCH fixture USING INDEX"],
            "uses_search": True,
            "uses_scan": False,
        }
        for locator in inventory["locators"]
    ]
    baseline = [1.0] * 40
    large = [2.0] * 40
    delta = [1.0] * 40
    operation = {
        "baseline": _summary(baseline),
        "large": _summary(large),
        "paired_delta": _summary(delta),
        "thresholds": {
            "large_p95_ms": 100.0,
            "median_paired_delta_ms": 2.0,
            "p95_paired_delta_ms": 10.0,
        },
        "passed": True,
    }
    pragmas = {"journal_mode": "wal", "synchronous": 2}
    return {
        "inventory": inventory,
        "query_plans": {"baseline": plans, "large": copy.deepcopy(plans)},
        "scan_count": 0,
        "row_counts": {"baseline": 1_000, "large": 100_000},
        "physical_forbidden_rows": {"baseline": 1_000, "large": 100_000},
        "distribution": distribution,
        "distribution_sha256": _canonical_sha256(distribution),
        "warmup_count": 10,
        "trial_count": 40,
        "measurement_order": "ABBA",
        "sqlite": {
            "version": "3.49.1",
            "pragmas_match": True,
            "baseline_pragmas": pragmas,
            "large_pragmas": copy.deepcopy(pragmas),
        },
        "fresh_commit": operation,
        "uncertain_retry": copy.deepcopy(operation),
        "passed": True,
    }


def _passing_latency() -> dict[str, object]:
    return {
        "trial_count": 40,
        "clock_boundaries": {
            "eos": "post_aec_admitted_speech_end",
            "dispatch": "attempt_dispatch_handoff",
            "audible_stop": "transport_abort_fence",
            "first_assistant_audio": "transport_playback_started",
        },
        "thresholds": {
            "eos_to_dispatch_p95_ms_max": 850.0,
            "barge_to_audible_stop_p95_ms_max": 150.0,
            "added_eos_to_replacement_dispatch_p95_ms_max": 850.0,
            "eos_to_first_assistant_audio_median_ms_max": 1_500.0,
            "eos_to_first_assistant_audio_p95_ms_max": 2_500.0,
        },
        "eos_to_dispatch": _summary([700.0] * 40),
        "barge_to_audible_stop": _summary([100.0] * 40),
        "added_eos_to_replacement_dispatch": _summary([700.0] * 40),
        "eos_to_first_assistant_audio": _summary([1_200.0] * 40),
        "passed": True,
    }


def _passing_corpus_case_results() -> list[dict[str, object]]:
    manifest = json.loads(
        (_ROOT / "Tests/Audio/fixtures/voice_aec/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return [
        {
            "id": case["id"],
            "kind": case["kind"],
            "passed": True,
            "error_class": None,
            "frames": 1,
            "median_erle_db": 29.0,
            "false_barge_events": 0,
            "double_talk_recall": 1.0,
        }
        for case in manifest["cases"]
    ]


def _passing_pytest_gate(tests: list[str]) -> dict[str, object]:
    return {
        "suite_sha256": _canonical_sha256({"tests": tests}),
        "test_path_count": len(tests),
        "exit_code": 0,
        "passed": True,
    }


def _passing_durable_owners() -> dict[str, object]:
    inventory_bytes = (
        _ROOT / "Docs/Development/TTS/speculative-voice-durable-owner-inventory.md"
    ).read_bytes()
    owner_ids = sorted(
        line.split("`", 2)[1]
        for line in inventory_bytes.decode("utf-8").splitlines()
        if line.startswith("| `")
    )
    probe = _passing_pytest_gate(["Tests/Chat/test_console_voice_ephemerality.py"])
    core = {
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "owner_ids_sha256": _canonical_sha256(owner_ids),
        "owner_count": len(owner_ids),
        "probe": probe,
    }
    return {**core, "report_sha256": _canonical_sha256(core), "passed": True}


def _passing_soak(scenario: str) -> dict[str, object]:
    common = {
        "requested_duration_seconds": 1_800.0,
        "elapsed_seconds": 1_800.0,
        "iterations": 100,
        "native_process": {
            "started": True,
            "exit_code": 0,
            "forced_termination": False,
            "reaped": True,
            "passed": True,
        },
        "tasks": {
            "baseline": 10,
            "maximum": 12,
            "final": 10,
            "maximum_delta_limit": 16,
        },
        "post_fence_callbacks_accepted": 0,
        "passed": True,
    }
    if scenario == "duplex-soak":
        occupancy = {
            "capture_frames": 0,
            "render_frames": 0,
            "render_reference_frames": 0,
            "control_events": 0,
        }
        common.update(
            {
                "audio_buffers": {
                    "capacities": {
                        "capture_frames": 8,
                        "render_frames": 8,
                        "render_reference_frames": 8,
                        "control_events": 2,
                    },
                    "maximum": {
                        "capture_frames": 1,
                        "render_frames": 1,
                        "render_reference_frames": 1,
                        "control_events": 1,
                    },
                    "final": occupancy,
                    "overflow_count": 0,
                },
                "device_handles": {
                    "opened": 2,
                    "stopped": 2,
                    "closed": 2,
                    "leaked": 0,
                },
            }
        )
    else:
        common.update(
            {
                "cancellations": 200,
                "orphans": {"maximum": 2, "final": 0, "limit": 2},
                "obsolete_cleanups_final": 0,
            }
        )
    return common


def _write_wheel(path: Path, *, extension_bytes: bytes) -> None:
    extension_suffix = ".pyd" if "win_amd64" in path.name else ".so"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            f"tldw_voice_aec/_native.fixture{extension_suffix}", extension_bytes
        )
        archive.writestr(
            "tldw_voice_aec/provenance/UPSTREAM.json",
            json.dumps({"commit": _UPSTREAM}),
        )
        archive.writestr(
            "tldw_voice_aec-0.1.8.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: tldw-voice-aec\nVersion: 0.1.8.0\n",
        )


def _write_platform_evidence(
    *,
    root: Path,
    platform_key: str,
) -> None:
    automated_dir = root / "automated"
    physical_dir = root / "physical"
    wheelhouse = root / "wheels"
    automated_dir.mkdir(exist_ok=True)
    physical_dir.mkdir(exist_ok=True)
    wheelhouse.mkdir(exist_ok=True)
    extension_bytes = f"native-extension:{platform_key}".encode()
    wheel_path = wheelhouse / _WHEEL_NAMES[platform_key]
    _write_wheel(wheel_path, extension_bytes=extension_bytes)
    wheel_hash = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    extension_hash = hashlib.sha256(extension_bytes).hexdigest()
    corpus = {
        "schema_version": 1,
        "thresholds": {
            "median_erle_db_min": 20.0,
            "p10_erle_db_min": 10.0,
            "false_barge_events_max": 1,
            "false_barge_render_minutes_min": 30.0,
            "double_talk_recall_min": 0.95,
        },
        "median_erle_db": 29.0,
        "p10_erle_db": 20.0,
        "false_barge_events": 0,
        "false_barge_render_minutes": 30.0,
        "double_talk_recall": 0.98,
        "case_results": _passing_corpus_case_results(),
        "unqualified_case_ids": [],
        "effective_mode": "full-duplex",
        "passed": True,
    }
    automated = assemble_automated_report(
        source_tree_digest=_DIGEST,
        interpreter=_interpreter(platform_key),
        companion={
            "version": _VERSION,
            "upstream_commit": _UPSTREAM,
            "wheel_sha256": wheel_hash,
            "extension_sha256": extension_hash,
            "passed": True,
        },
        corpus=corpus,
        latency_distributions=_passing_latency(),
        completed_pair_history_gate=_passing_history_gate(),
        lifecycle=_passing_pytest_gate(
            [
                "Tests/Chat/test_console_voice_attempts.py",
                "Tests/Chat/test_console_voice_effect_barrier.py",
                "Tests/Chat/test_console_voice_capture.py",
                "Tests/integration/test_speculative_voice_pipeline.py",
            ]
        ),
        durable_owners=_passing_durable_owners(),
    )
    automated_path = automated_dir / f"{platform_key}-automated.json"
    write_qualification_report(automated_path, automated)
    for scenario in ("duplex-soak", "cancellation-soak"):
        soak = assemble_soak_report(
            scenario=scenario,
            source_tree_digest=_DIGEST,
            interpreter=_interpreter(platform_key),
            soak=_passing_soak(scenario),
        )
        write_qualification_report(
            automated_dir / f"{platform_key}-{scenario}.json", soak
        )
    prerequisites = load_automated_prerequisites(
        automated_path,
        expected_source_tree_digest=_DIGEST,
        platform_key=platform_key,
    )
    for device_class in ("builtin", "usb", "bluetooth"):
        fixture = (
            "safe_half_duplex" if device_class == "bluetooth" else "safe_full_duplex"
        )
        observations = load_fixture_observations(_PHYSICAL_FIXTURES / f"{fixture}.json")
        observations["device_identifier"] = f"fixture:{platform_key}:{device_class}"
        observations["transport"] = device_class
        report = build_physical_report(
            evidence_kind="physical",
            source_tree_digest=_DIGEST,
            platform_key=platform_key,
            device_class=device_class,
            app_version=_VERSION,
            prerequisites=prerequisites,
            observations=observations,
            salt=hashlib.sha256(f"{platform_key}:{device_class}".encode()).digest()[
                :16
            ],
        )
        write_physical_report(
            physical_dir / f"{platform_key}-{device_class}.json", report
        )


def _complete_evidence(root: Path) -> None:
    for platform_key in PLATFORM_KEYS:
        _write_platform_evidence(root=root, platform_key=platform_key)


def _manifest_generation_kwargs(root: Path) -> dict[str, object]:
    return {
        "source_tree_digest": _DIGEST,
        "app_version": _VERSION,
        "aec_upstream": _UPSTREAM,
        "wheelhouse": root / "wheels",
        "automated_dir": root / "automated",
        "physical_dir": root / "physical",
        "output": root / "manifest.json",
        "build_identity_output": root / "build-identity.json",
        "_source_digest_reader": lambda: _DIGEST,
    }


def test_history_gate_is_independently_recomputed() -> None:
    report = _passing_history_gate()
    validate_completed_pair_history_gate(report)

    forged = copy.deepcopy(report)
    forged["fresh_commit"]["large"]["samples_ms"] = [101.0] * 40
    forged["fresh_commit"]["passed"] = True
    forged["passed"] = True
    with pytest.raises(VoiceManifestError, match="history gate"):
        validate_completed_pair_history_gate(forged)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scan_count",), 1),
        (("row_counts", "large"), 99_999),
        (("warmup_count",), 9),
        (("trial_count",), 39),
        (("measurement_order",), "AB"),
        (("fresh_commit", "thresholds", "large_p95_ms"), 101.0),
        (("sqlite", "pragmas_match"), False),
    ],
)
def test_history_gate_rejects_changed_fixed_contract(
    path: tuple[str, ...],
    value: object,
) -> None:
    report = _passing_history_gate()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(VoiceManifestError, match="history gate"):
        validate_completed_pair_history_gate(report)


def test_generator_requires_complete_evidence_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    _complete_evidence(tmp_path)
    output = tmp_path / "manifest.json"
    identity = tmp_path / "build-identity.json"
    kwargs = {
        "source_tree_digest": _DIGEST,
        "app_version": _VERSION,
        "aec_upstream": _UPSTREAM,
        "wheelhouse": tmp_path / "wheels",
        "automated_dir": tmp_path / "automated",
        "physical_dir": tmp_path / "physical",
        "output": output,
        "build_identity_output": identity,
        "_source_digest_reader": lambda: _DIGEST,
    }

    manifest = generate_voice_qualification_manifest(**kwargs)
    first = (output.read_bytes(), identity.read_bytes())
    second_manifest = generate_voice_qualification_manifest(**kwargs)

    validate_voice_qualification_manifest(manifest)
    assert second_manifest == manifest
    assert (output.read_bytes(), identity.read_bytes()) == first
    assert set(manifest["platforms"]) == set(PLATFORM_KEYS)
    assert all(item["qualified"] is True for item in manifest["platforms"].values())
    assert all(
        item["history_gate_passed"] is True for item in manifest["platforms"].values()
    )
    assert {item["python_tag"] for item in manifest["platforms"].values()} == {"cp312"}


def test_generator_rejects_physical_report_padded_over_two_mib(
    tmp_path: Path,
) -> None:
    _complete_evidence(tmp_path)
    physical_path = tmp_path / "physical/macos-arm64-builtin.json"
    physical_path.write_bytes(physical_path.read_bytes() + b" " * (2 * 1024 * 1024))

    with pytest.raises(VoiceManifestError, match="physical"):
        generate_voice_qualification_manifest(**_manifest_generation_kwargs(tmp_path))


def test_generator_translates_deep_physical_json_to_domain_error(
    tmp_path: Path,
) -> None:
    _complete_evidence(tmp_path)
    physical_path = tmp_path / "physical/macos-arm64-builtin.json"
    raw = physical_path.read_bytes().rstrip()
    deep_value = b"[" * 1_100 + b"0" + b"]" * 1_100
    physical_path.write_bytes(raw[:-1] + b',"unknown":' + deep_value + b"}\n")

    with pytest.raises(VoiceManifestError, match="physical"):
        generate_voice_qualification_manifest(**_manifest_generation_kwargs(tmp_path))


def test_generator_hashes_original_valid_physical_report_bytes(
    tmp_path: Path,
) -> None:
    _complete_evidence(tmp_path)
    platform_key = "macos-arm64"
    raw_hashes = {
        device_class: hashlib.sha256(
            (tmp_path / f"physical/{platform_key}-{device_class}.json").read_bytes()
        ).hexdigest()
        for device_class in ("builtin", "usb", "bluetooth")
    }
    expected_matrix_hash = _canonical_sha256(raw_hashes)

    manifest = generate_voice_qualification_manifest(
        **_manifest_generation_kwargs(tmp_path)
    )

    assert (
        manifest["platforms"][platform_key]["physical_report_sha256"]
        == expected_matrix_hash
    )


def test_generator_selects_the_qualified_interpreter_abi_from_multi_abi_wheels(
    tmp_path: Path,
) -> None:
    _complete_evidence(tmp_path)
    for platform_key, cp312_name in _WHEEL_NAMES.items():
        for tag in ("cp311", "cp313"):
            _write_wheel(
                tmp_path / "wheels" / cp312_name.replace("cp312", tag),
                extension_bytes=f"native-extension:{platform_key}:{tag}".encode(),
            )

    manifest = generate_voice_qualification_manifest(
        source_tree_digest=_DIGEST,
        app_version=_VERSION,
        aec_upstream=_UPSTREAM,
        wheelhouse=tmp_path / "wheels",
        automated_dir=tmp_path / "automated",
        physical_dir=tmp_path / "physical",
        output=tmp_path / "manifest.json",
        build_identity_output=tmp_path / "identity.json",
        _source_digest_reader=lambda: _DIGEST,
    )

    assert {entry["python_tag"] for entry in manifest["platforms"].values()} == {
        "cp312"
    }


def test_generator_rejects_missing_device_class_and_synthetic_evidence(
    tmp_path: Path,
) -> None:
    _complete_evidence(tmp_path)
    missing = tmp_path / "physical" / "linux-aarch64-usb.json"
    missing.unlink()
    kwargs = {
        "source_tree_digest": _DIGEST,
        "app_version": _VERSION,
        "aec_upstream": _UPSTREAM,
        "wheelhouse": tmp_path / "wheels",
        "automated_dir": tmp_path / "automated",
        "physical_dir": tmp_path / "physical",
        "output": tmp_path / "manifest.json",
        "build_identity_output": tmp_path / "identity.json",
        "_source_digest_reader": lambda: _DIGEST,
    }
    with pytest.raises(VoiceManifestError, match="physical"):
        generate_voice_qualification_manifest(**kwargs)

    _write_platform_evidence(root=tmp_path, platform_key="linux-aarch64")
    physical_path = tmp_path / "physical" / "linux-aarch64-usb.json"
    report = json.loads(physical_path.read_text(encoding="utf-8"))
    report["evidence_kind"] = "synthetic-fixture"
    write_physical_report(physical_path, report)
    with pytest.raises(VoiceManifestError, match="physical"):
        generate_voice_qualification_manifest(**kwargs)


def test_generator_rejects_digest_version_history_and_privacy_forgery(
    tmp_path: Path,
) -> None:
    _complete_evidence(tmp_path)
    automated_path = tmp_path / "automated" / "macos-arm64-automated.json"
    original = json.loads(automated_path.read_text(encoding="utf-8"))
    kwargs = {
        "source_tree_digest": _DIGEST,
        "app_version": _VERSION,
        "aec_upstream": _UPSTREAM,
        "wheelhouse": tmp_path / "wheels",
        "automated_dir": tmp_path / "automated",
        "physical_dir": tmp_path / "physical",
        "output": tmp_path / "manifest.json",
        "build_identity_output": tmp_path / "identity.json",
        "_source_digest_reader": lambda: _DIGEST,
    }

    for mutate in (
        lambda report: report.__setitem__("source_tree_digest", "e" * 64),
        lambda report: report["companion"].__setitem__("version", "0.1.7.0"),
        lambda report: report["completed_pair_history_gate"].__setitem__(
            "scan_count", 1
        ),
        lambda report: report["interpreter"].__setitem__("system", "Linux"),
        lambda report: report["corpus"]["case_results"][0].__setitem__("passed", False),
        lambda report: report["corpus"]["case_results"][0].__setitem__("id", []),
        lambda report: report.__setitem__("transcript", "VOICE-POISON"),
        lambda report: report.__setitem__("unknown", True),
    ):
        forged = copy.deepcopy(original)
        mutate(forged)
        _write_raw_json(automated_path, forged)
        with pytest.raises(VoiceManifestError):
            generate_voice_qualification_manifest(**kwargs)
        write_qualification_report(automated_path, original)


def test_generator_recomputes_corpus_cases_instead_of_trusting_aggregate(
    tmp_path: Path,
) -> None:
    _complete_evidence(tmp_path)
    automated_path = tmp_path / "automated/macos-arm64-automated.json"
    report = json.loads(automated_path.read_text(encoding="utf-8"))
    report["corpus"]["case_results"][0]["passed"] = False
    assert report["corpus"]["passed"] is True
    _write_raw_json(automated_path, report)

    with pytest.raises(VoiceManifestError, match="corpus"):
        generate_voice_qualification_manifest(
            source_tree_digest=_DIGEST,
            app_version=_VERSION,
            aec_upstream=_UPSTREAM,
            wheelhouse=tmp_path / "wheels",
            automated_dir=tmp_path / "automated",
            physical_dir=tmp_path / "physical",
            output=tmp_path / "manifest.json",
            build_identity_output=tmp_path / "identity.json",
            _source_digest_reader=lambda: _DIGEST,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("lifecycle", {"exit_code": 1}, "pytest"),
        ("durable_owners", {"inventory_sha256": "0" * 64}, "durable-owner"),
    ],
)
def test_generator_recomputes_test_gates_instead_of_trusting_passed(
    tmp_path: Path,
    field: str,
    value: dict[str, object],
    match: str,
) -> None:
    _complete_evidence(tmp_path)
    automated_path = tmp_path / "automated/macos-arm64-automated.json"
    report = json.loads(automated_path.read_text(encoding="utf-8"))
    report[field].update(value)
    assert report[field]["passed"] is True
    _write_raw_json(automated_path, report)

    with pytest.raises(VoiceManifestError, match=match):
        generate_voice_qualification_manifest(
            source_tree_digest=_DIGEST,
            app_version=_VERSION,
            aec_upstream=_UPSTREAM,
            wheelhouse=tmp_path / "wheels",
            automated_dir=tmp_path / "automated",
            physical_dir=tmp_path / "physical",
            output=tmp_path / "manifest.json",
            build_identity_output=tmp_path / "identity.json",
            _source_digest_reader=lambda: _DIGEST,
        )


@pytest.mark.parametrize(
    ("scenario", "mutate"),
    [
        (
            "duplex-soak",
            lambda report: report["soak"]["audio_buffers"].__setitem__(
                "overflow_count", 1
            ),
        ),
        (
            "cancellation-soak",
            lambda report: report["soak"].__setitem__("elapsed_seconds", 1_799.0),
        ),
        (
            "duplex-soak",
            lambda report: report["soak"].update(
                {"requested_duration_seconds": 3_600.0, "elapsed_seconds": 1_800.0}
            ),
        ),
    ],
)
def test_generator_recomputes_soak_safety_instead_of_trusting_passed(
    tmp_path: Path,
    scenario: str,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    _complete_evidence(tmp_path)
    soak_path = tmp_path / "automated" / f"macos-arm64-{scenario}.json"
    report = json.loads(soak_path.read_text(encoding="utf-8"))
    mutate(report)
    assert report["passed"] is True
    assert report["soak"]["passed"] is True
    _write_raw_json(soak_path, report)

    with pytest.raises(VoiceManifestError, match="soak"):
        generate_voice_qualification_manifest(
            source_tree_digest=_DIGEST,
            app_version=_VERSION,
            aec_upstream=_UPSTREAM,
            wheelhouse=tmp_path / "wheels",
            automated_dir=tmp_path / "automated",
            physical_dir=tmp_path / "physical",
            output=tmp_path / "manifest.json",
            build_identity_output=tmp_path / "identity.json",
            _source_digest_reader=lambda: _DIGEST,
        )


def test_generator_rejects_claimed_digest_unequal_to_computed_source() -> None:
    with pytest.raises(VoiceManifestError, match="source-tree digest"):
        generate_voice_qualification_manifest(
            source_tree_digest=_DIGEST,
            app_version=_VERSION,
            aec_upstream=_UPSTREAM,
            wheelhouse=_ROOT,
            automated_dir=_ROOT,
            physical_dir=_ROOT,
            output=_ROOT / "never-written.json",
            build_identity_output=_ROOT / "never-written-identity.json",
            _source_digest_reader=lambda: "e" * 64,
        )


def test_generator_wraps_source_digest_failures() -> None:
    def fail_digest() -> str:
        raise RuntimeError("local path must not escape through CLI")

    with pytest.raises(VoiceManifestError, match="source-tree digest"):
        generate_voice_qualification_manifest(
            source_tree_digest=_DIGEST,
            app_version=_VERSION,
            aec_upstream=_UPSTREAM,
            wheelhouse=_ROOT,
            automated_dir=_ROOT,
            physical_dir=_ROOT,
            output=_ROOT / "never-written.json",
            build_identity_output=_ROOT / "never-written-identity.json",
            _source_digest_reader=fail_digest,
        )


def test_checked_in_authority_is_valid_hard_off_package_data() -> None:
    manifest_path = _ROOT / "tldw_chatbook/Audio/voice_qualification_manifest.json"
    identity_path = _ROOT / "tldw_chatbook/Audio/voice_build_identity.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    identity = json.loads(identity_path.read_bytes())

    validate_voice_qualification_manifest(manifest)
    assert all(
        entry["qualified"] is False and entry["history_gate_passed"] is False
        for entry in manifest["platforms"].values()
    )
    assert identity == {
        "schema_version": 1,
        "pipeline_version": 1,
        "app_version": manifest["app_version"],
        "source_tree_digest": manifest["source_tree_digest"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    configuration = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    assert configuration["tool"]["setuptools"]["package-data"][
        "tldw_chatbook.Audio"
    ] == [
        "voice_qualification_manifest.json",
        "voice_build_identity.json",
    ]


def test_release_workflow_regenerates_and_verifies_packaged_authority() -> None:
    workflow = (_ROOT / ".github/workflows/release-voice-aec.yml").read_text(
        encoding="utf-8"
    )
    verification = workflow.split("  verify-app-rollout-authority:", 1)[1].split(
        "  publish-exact-bytes:", 1
    )[0]

    for required in (
        "needs: verify-qualified-release",
        "actions/checkout@",
        "actions/download-artifact@",
        "Packaging/compute_voice_source_digest.py",
        "Packaging/generate_voice_qualification_manifest.py",
        "Artifacts/voice_qualification/automated",
        "Artifacts/voice_qualification/physical",
        "cmp --silent generated-authority/voice_qualification_manifest.json",
        "cmp --silent generated-authority/voice_build_identity.json",
        "python -m build --wheel --outdir app-dist",
        "Tests/Packaging/test_voice_qualification_manifest.py",
        "installed app wheel authority smoke",
    ):
        assert required in verification
    publish_header = workflow.split("  publish-exact-bytes:", 1)[1].split(
        "    steps:", 1
    )[0]
    assert "verify-app-rollout-authority" in publish_header
