"""Regression tests for the real-provider three-turn Console benchmark."""

from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest

from Tests.Performance import run_console_three_turn_profile as profile


def test_balanced_arm_order_rotates_complete_triples() -> None:
    balanced_arm_order = getattr(profile, "balanced_arm_order", None)

    assert callable(balanced_arm_order)
    assert balanced_arm_order(0) == ("control", "disabled", "enabled")
    assert balanced_arm_order(1) == ("disabled", "enabled", "control")
    assert balanced_arm_order(2) == ("enabled", "control", "disabled")
    assert balanced_arm_order(3) == balanced_arm_order(0)


def test_nearest_rank_percentile_uses_one_based_ceiling() -> None:
    nearest_rank_percentile = getattr(profile, "nearest_rank_percentile", None)

    assert callable(nearest_rank_percentile)
    assert nearest_rank_percentile(list(range(1, 31)), 0.95) == 29


def test_paired_p95_ratio_bounds_are_deterministic_and_resample_blocks() -> None:
    paired_p95_ratio_bounds = getattr(profile, "paired_p95_ratio_bounds", None)
    blocks = [
        {"control": 10.0 + index, "disabled": 12.0 + index, "enabled": 9.0 + index}
        for index in range(12)
    ]

    assert callable(paired_p95_ratio_bounds)
    first = paired_p95_ratio_bounds(blocks, "disabled", resamples=250, seed=17)
    second = paired_p95_ratio_bounds(blocks, "disabled", resamples=250, seed=17)

    assert first == second
    assert set(first) == {
        "two_sided_95",
        "one_sided_lower_95",
        "one_sided_upper_95",
    }
    assert first["two_sided_95"][0] <= first["two_sided_95"][1]
    assert first["one_sided_lower_95"] <= first["one_sided_upper_95"]


def test_paired_p95_ratio_bounds_preserve_constant_ratio() -> None:
    paired_p95_ratio_bounds = getattr(profile, "paired_p95_ratio_bounds", None)
    blocks = [
        {"control": float(index), "disabled": float(index * 2), "enabled": 1.0}
        for index in range(1, 11)
    ]

    assert callable(paired_p95_ratio_bounds)
    bounds = paired_p95_ratio_bounds(blocks, "disabled", resamples=100, seed=9)

    assert bounds == {
        "two_sided_95": (2.0, 2.0),
        "one_sided_lower_95": 2.0,
        "one_sided_upper_95": 2.0,
    }


@pytest.mark.parametrize(
    ("blocks", "candidate", "message"),
    [
        ([{"control": 1.0, "disabled": 1.0}], "disabled", "two blocks"),
        (
            [
                {"control": 1.0, "disabled": 1.0},
                {"control": 2.0, "disabled": 2.0},
            ],
            "enabled",
            "complete blocks",
        ),
        (
            [
                {"control": 0.0, "disabled": 1.0, "enabled": 1.0},
                {"control": 0.0, "disabled": 2.0, "enabled": 2.0},
            ],
            "disabled",
            "positive control",
        ),
    ],
)
def test_paired_p95_ratio_bounds_fail_closed(
    blocks: list[dict[str, float]], candidate: str, message: str
) -> None:
    paired_p95_ratio_bounds = getattr(profile, "paired_p95_ratio_bounds", None)

    assert callable(paired_p95_ratio_bounds)
    with pytest.raises(ValueError, match=message):
        paired_p95_ratio_bounds(blocks, candidate, resamples=10)


def test_sample_heartbeat_p95_reduces_each_sample_independently() -> None:
    sample_heartbeat_p95_ns = getattr(profile, "sample_heartbeat_p95_ns", None)

    assert callable(sample_heartbeat_p95_ns)
    assert sample_heartbeat_p95_ns([1, 2, 3, 100]) == 100.0
    with pytest.raises(ValueError, match="heartbeat"):
        sample_heartbeat_p95_ns([])


PAYLOAD_SHA = "a" * 64
PERMISSION_SHA = "b" * 64


def _valid_sample(arm: str, *, iteration: int = 0) -> dict[str, object]:
    review_events: dict[str, int]
    if arm == "control":
        review_events = {
            "baseline_started": 10,
            "baseline_ready": 20,
            "review_e_started": 110,
            "review_e_completed": 190,
        }
    elif arm == "enabled":
        review_events = {
            "baseline_started": 10,
            "baseline_ready": 20,
            "finalization_scheduled": 100,
            "review_e_started": 130,
            "review_e_completed": 210,
        }
    else:
        review_events = {}
    return {
        "sample_id": f"measured-{iteration}-{arm}",
        "phase": "measured",
        "iteration": iteration,
        "arm": arm,
        "status": "complete",
        "provider_round_counts": {"1": 1, "2": 3, "3": 1},
        "provider_usage": [
            {
                "prompt_tokens": 100 + index,
                "completion_tokens": 10,
                "total_tokens": 110 + index,
            }
            for index in range(5)
        ],
        "terminal_turn_2_provider_completed_ns": 100,
        "third_send_requested_ns": 115,
        "turn_2_release_ns": 120,
        "third_worker_started_ns": 125,
        "third_provider_started_ns": 130,
        "terminal_third_provider_completed_ns": 180,
        "terminal_third_assistant_ns": 200,
        "heartbeat_lateness_ns": [1, 2, 3],
        "prompt_loss_count": 0,
        "selected_binding_access": "rw",
        "expected_payload_sha256": PAYLOAD_SHA,
        "expected_permission_definition_hash": PERMISSION_SHA,
        "tool_calls": [
            {
                "name": "load_tools",
                "turn": 2,
                "provider_round": 1,
                "requested_tool_id": "local:fs_write",
                "permission": "allow",
                "definition_hash": PERMISSION_SHA,
            },
            {
                "name": "fs_write",
                "turn": 2,
                "provider_round": 2,
                "tool_id": "local:fs_write",
                "path": "measured/turn-two.txt",
                "payload_sha256": PAYLOAD_SHA,
                "permission": "allow",
                "definition_hash": PERMISSION_SHA,
            },
        ],
        "mutation": {
            "path": "measured/turn-two.txt",
            "payload_sha256": PAYLOAD_SHA,
            "success": True,
        },
        "review_events": review_events,
        "metrics": {
            "third_send_to_worker_ns": 10 + iteration,
            "event_loop_lag_p95_ns": 3 + iteration,
            "provider_total_ns": 1_000 + iteration,
            "conversation_wall_ns": 2_000 + iteration,
            "assistant_durable_to_release_ns": 5 + iteration,
            "terminal_to_third_provider_ns": 30 + iteration,
        },
    }


@pytest.mark.parametrize("arm", profile.ARMS)
def test_validate_sample_accepts_complete_arm_contracts(arm: str) -> None:
    validate_sample = getattr(profile, "validate_sample", None)

    assert callable(validate_sample)
    assert validate_sample(_valid_sample(arm)) == ()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (("terminal_third_assistant_ns", None), "terminal_third_assistant_missing"),
        (("terminal_third_provider_completed_ns", None), "terminal_third_provider_missing"),
        (("third_provider_started_ns", None), "third_provider_missing"),
        (("provider_round_counts", {"1": 1, "2": 2, "3": 1}), "provider_round_contract"),
        (("provider_usage", []), "provider_usage_contract"),
        (("third_send_requested_ns", 120), "third_send_not_queued"),
        (("heartbeat_lateness_ns", []), "heartbeat_missing"),
        (("heartbeat_lateness_ns", [1, "late"]), "heartbeat_contract"),
        (("selected_binding_access", "ro"), "workspace_binding_not_rw"),
        (("expected_payload_sha256", None), "expected_hash_contract"),
        (("expected_permission_definition_hash", None), "expected_hash_contract"),
        (("metrics", {}), "metrics_contract"),
    ],
)
def test_validate_sample_rejects_common_contract_mutants(
    mutation: tuple[str, object], expected_code: str
) -> None:
    validate_sample = getattr(profile, "validate_sample", None)
    row = _valid_sample("enabled")
    row[mutation[0]] = mutation[1]

    assert callable(validate_sample)
    assert expected_code in validate_sample(row)


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda row: row["tool_calls"].pop(), "tool_call_contract"),
        (
            lambda row: row["tool_calls"][0].update({"requested_tool_id": "local:fs_read"}),
            "load_tools_contract",
        ),
        (
            lambda row: row["tool_calls"][0].update({"provider_round": 2}),
            "load_tools_contract",
        ),
        (
            lambda row: row["tool_calls"][1].update({"path": "outside.txt"}),
            "fs_write_contract",
        ),
        (
            lambda row: row["tool_calls"][1].update({"payload_sha256": "c" * 64}),
            "fs_write_contract",
        ),
        (
            lambda row: row["tool_calls"][1].update({"definition_hash": "c" * 64}),
            "permission_contract",
        ),
        (
            lambda row: row["mutation"].update({"success": False}),
            "mutation_contract",
        ),
    ],
)
def test_validate_sample_rejects_tool_contract_mutants(mutate, expected_code: str) -> None:
    validate_sample = getattr(profile, "validate_sample", None)
    row = _valid_sample("enabled")
    mutate(row)

    assert callable(validate_sample)
    assert expected_code in validate_sample(row)


def test_validate_sample_treats_e_relationship_as_descriptive() -> None:
    validate_sample = getattr(profile, "validate_sample", None)

    assert callable(validate_sample)
    for started, completed in ((101, 110), (110, 125), (130, 210)):
        row = _valid_sample("enabled")
        row["review_events"].update(
            {"review_e_started": started, "review_e_completed": completed}
        )
        assert validate_sample(row) == ()


def test_validate_sample_enforces_arm_specific_review_events() -> None:
    validate_sample = getattr(profile, "validate_sample", None)
    disabled = _valid_sample("disabled")
    disabled["review_events"] = {"review_e_started": 130}
    control = _valid_sample("control")
    control["review_events"].pop("review_e_completed")
    enabled = _valid_sample("enabled")
    enabled["review_events"].pop("finalization_scheduled")

    assert callable(validate_sample)
    assert "review_event_prohibited" in validate_sample(disabled)
    assert "review_event_missing" in validate_sample(control)
    assert "review_event_missing" in validate_sample(enabled)


def test_validate_sample_rejects_incoherent_review_timing() -> None:
    validate_sample = getattr(profile, "validate_sample", None)
    row = _valid_sample("enabled")
    row["review_events"]["review_e_completed"] = 120

    assert callable(validate_sample)
    assert "review_event_timing" in validate_sample(row)


def _valid_run(iterations: int = 30) -> list[dict[str, object]]:
    warmups = []
    for arm in profile.ARMS:
        row = _valid_sample(arm, iteration=-1)
        row["sample_id"] = f"warmup-{arm}"
        row["phase"] = "warmup"
        warmups.append(row)
    measured = [
        _valid_sample(arm, iteration=iteration)
        for iteration in range(iterations)
        for arm in profile.ARMS
    ]
    return warmups + measured


def _valid_confirmation_rows() -> tuple[
    tuple[profile.SamplePlan, ...], list[dict[str, object]]
]:
    schedule = profile.sample_schedule(30, burn_in_blocks=5)
    rows = []
    for schedule_position, plan in enumerate(schedule):
        row = _valid_sample(plan.arm, iteration=plan.iteration)
        row.update(
            {
                "sample_id": f"{plan.phase}-{plan.iteration}-{plan.arm}",
                "phase": plan.phase,
                "schedule_position": schedule_position,
            }
        )
        rows.append(row)
    return schedule, rows


def test_validate_confirmation_rows_accepts_exact_sequence_before_filtering() -> None:
    schedule, rows = _valid_confirmation_rows()
    validate_confirmation_rows = getattr(profile, "validate_confirmation_rows", None)

    assert callable(validate_confirmation_rows)
    errors, filtered = validate_confirmation_rows(
        rows,
        schedule,
        validate_sample=profile.validate_sample,
    )

    assert len(rows) == 108
    assert errors == ()
    assert filtered == [row for row in rows if row["phase"] != "burn_in"]
    assert len(filtered) == 93


@pytest.mark.parametrize(
    "mutation",
    ("reordered", "missing", "extra", "missing_position", "wrong_position", "unknown_phase"),
)
def test_validate_confirmation_rows_rejects_sequence_mutants(mutation: str) -> None:
    schedule, rows = _valid_confirmation_rows()
    validate_confirmation_rows = getattr(profile, "validate_confirmation_rows", None)

    if mutation == "reordered":
        rows[10], rows[11] = rows[11], rows[10]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        extra = copy.deepcopy(rows[-1])
        extra.update(
            {
                "sample_id": "measured-30-control",
                "iteration": 30,
                "arm": "control",
                "schedule_position": len(rows),
            }
        )
        rows.append(extra)
    elif mutation == "missing_position":
        rows[10].pop("schedule_position")
    elif mutation == "wrong_position":
        rows[10]["schedule_position"] = 11
    else:
        rows[10]["phase"] = "unknown"

    assert callable(validate_confirmation_rows)
    errors, _filtered = validate_confirmation_rows(
        rows,
        schedule,
        validate_sample=profile.validate_sample,
    )

    assert errors == ("confirmation_schedule_contract",)


def test_validate_confirmation_rows_rejects_within_phase_duplicate_sample_ids() -> None:
    schedule, rows = _valid_confirmation_rows()
    rows[4]["sample_id"] = rows[3]["sample_id"]
    validate_confirmation_rows = getattr(profile, "validate_confirmation_rows", None)

    assert callable(validate_confirmation_rows)
    errors, _filtered = validate_confirmation_rows(
        rows,
        schedule,
        validate_sample=profile.validate_sample,
    )

    assert errors == (
        "confirmation_schedule_contract",
        "confirmation_sample_id_duplicate",
    )


def test_validate_confirmation_rows_rejects_cross_phase_duplicate_sample_ids() -> None:
    schedule, rows = _valid_confirmation_rows()
    rows[18]["sample_id"] = rows[3]["sample_id"]
    validate_confirmation_rows = getattr(profile, "validate_confirmation_rows", None)

    assert callable(validate_confirmation_rows)
    errors, _filtered = validate_confirmation_rows(
        rows,
        schedule,
        validate_sample=profile.validate_sample,
    )

    assert errors == (
        "confirmation_schedule_contract",
        "confirmation_sample_id_duplicate",
    )


def test_validate_confirmation_rows_handles_unhashable_duplicate_sample_ids() -> None:
    schedule, rows = _valid_confirmation_rows()
    rows[10]["sample_id"] = ["malformed"]
    rows[11]["sample_id"] = ["malformed"]
    seen_positions = []
    validate_confirmation_rows = getattr(profile, "validate_confirmation_rows", None)

    def injected_validate_sample(row: dict[str, object]) -> tuple[str, ...]:
        seen_positions.append(row["schedule_position"])
        return ()

    assert callable(validate_confirmation_rows)
    errors, _filtered = validate_confirmation_rows(
        rows,
        schedule,
        validate_sample=injected_validate_sample,
    )

    assert errors == (
        "confirmation_schedule_contract",
        "confirmation_sample_id_duplicate",
    )
    assert seen_positions == list(range(108))


def test_validate_confirmation_rows_validates_every_row_including_burn_in() -> None:
    schedule, rows = _valid_confirmation_rows()
    rows[3]["status"] = "failed"
    seen_positions = []
    validate_confirmation_rows = getattr(profile, "validate_confirmation_rows", None)

    def original_validate_sample(row: dict[str, object]) -> tuple[str, ...]:
        seen_positions.append(row["schedule_position"])
        return profile.validate_sample(row)

    assert callable(validate_confirmation_rows)
    errors, _filtered = validate_confirmation_rows(
        rows,
        schedule,
        validate_sample=original_validate_sample,
    )

    assert errors == ("confirmation_sample_contract",)
    assert seen_positions == list(range(108))


def _materialize_original_runner(candidate_root: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            "git",
            "show",
            f"{profile.ORIGINAL_HARNESS_SHA}:Tests/Performance/"
            "run_console_three_turn_profile.py",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    runner = candidate_root / "Tests/Performance/run_console_three_turn_profile.py"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(completed.stdout)
    return runner


def _copy_original_evidence(destination_root: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    relative = Path("Docs/superpowers/qa/console-three-turn-real-provider")
    destination = destination_root / relative
    destination.mkdir(parents=True)
    for name in profile.ORIGINAL_EVIDENCE_SHA256:
        (destination / name).write_bytes((repository_root / relative / name).read_bytes())
    return destination


def test_original_runner_and_evidence_pins_are_exact() -> None:
    assert profile.ORIGINAL_HARNESS_SHA == (
        "eb8225a32f88ea43c337aff99804d360384e7668"
    )
    assert profile.ORIGINAL_RUNNER_SHA256 == (
        "fbca69703b771f7b7b27fa78ef9bf095fb30712435743877e20fcb01bb6d06ae"
    )
    assert profile.CANDIDATE_SHA == profile.ORIGINAL_HARNESS_SHA
    assert profile.ORIGINAL_EVIDENCE_SHA256 == {
        "README.md": "724be0f80eff3c9a2eced35b86ae4ce2e6f9a7524d44016cd3f49b61752bd491",
        "real-provider-three-turn-summary.md": (
            "fdb4528bd82a33f244b4e6fbcfe3b739bd2374006cfea2df878f2e0d27a7d5c2"
        ),
        "real-provider-three-turn.manifest.json": (
            "f5dec9153845b585d32660ca87f8d4aef7ad31be4dc431bb52e64fdc29187bb6"
        ),
        "real-provider-three-turn.raw.jsonl": (
            "82150cd55ba701b5a2680f87fce43b15676004fc1609f477f458a7abb2078319"
        ),
        "real-provider-three-turn.summary.json": (
            "edec5d347427748e26c93d21da7ecf121cccedb41ea7d304fb6cdad684f3668a"
        ),
    }


def test_original_evidence_guard_rejects_altered_missing_and_extra_files(
    tmp_path: Path,
) -> None:
    verify_original_evidence = getattr(profile, "verify_original_evidence", None)
    evidence = _copy_original_evidence(tmp_path)

    assert callable(verify_original_evidence)
    assert verify_original_evidence(tmp_path) == profile.ORIGINAL_EVIDENCE_SHA256

    altered = evidence / "README.md"
    altered.write_bytes(altered.read_bytes() + b"altered")
    with pytest.raises(RuntimeError, match="original_evidence_hash_mismatch:README.md"):
        verify_original_evidence(tmp_path)
    altered.write_bytes(
        (Path(__file__).resolve().parents[2] / evidence.relative_to(tmp_path) / "README.md").read_bytes()
    )

    missing = evidence / "real-provider-three-turn.summary.json"
    missing.unlink()
    with pytest.raises(RuntimeError, match="original_evidence_missing"):
        verify_original_evidence(tmp_path)
    missing.write_bytes(
        (
            Path(__file__).resolve().parents[2]
            / evidence.relative_to(tmp_path)
            / missing.name
        ).read_bytes()
    )

    (evidence / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="original_evidence_set_mismatch"):
        verify_original_evidence(tmp_path)


def test_original_runner_loads_isolated_and_exposes_direct_statistics(
    tmp_path: Path,
) -> None:
    load_original_runner = getattr(profile, "load_original_runner", None)
    _materialize_original_runner(tmp_path)

    assert callable(load_original_runner)
    original = load_original_runner(tmp_path)

    assert original.__name__ == "task_20009_original_runner"
    assert original is not profile
    assert original.validate_sample(_valid_sample("control")) == ()
    rows = _valid_run()
    assert original.validate_run(rows) == ()
    assert original.build_summary(rows, bootstrap_resamples=10)["overall_verdict"]


def test_original_runner_rejects_digest_drift(tmp_path: Path) -> None:
    load_original_runner = getattr(profile, "load_original_runner", None)
    runner = _materialize_original_runner(tmp_path)
    runner.write_bytes(runner.read_bytes() + b"\n# altered\n")

    assert callable(load_original_runner)
    with pytest.raises(RuntimeError, match="original_runner_hash_mismatch"):
        load_original_runner(tmp_path)


def test_original_runner_contract_failure_removes_isolated_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _materialize_original_runner(tmp_path)
    invalid = SimpleNamespace()
    monkeypatch.setattr(profile.importlib.util, "module_from_spec", lambda _spec: invalid)
    monkeypatch.setattr(
        profile.importlib.util,
        "spec_from_file_location",
        lambda *_args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _module: None)),
    )

    with pytest.raises(RuntimeError, match="original_runner_contract_mismatch"):
        profile.load_original_runner(tmp_path)

    assert "task_20009_original_runner" not in sys.modules


def test_burn_in_summary_is_byte_equivalent_when_only_excluded_metrics_change(
    tmp_path: Path,
) -> None:
    load_original_runner = getattr(profile, "load_original_runner", None)
    _materialize_original_runner(tmp_path)
    original = load_original_runner(tmp_path)
    schedule, rows = _valid_confirmation_rows()

    errors, measured = profile.validate_confirmation_rows(
        rows,
        schedule,
        validate_sample=original.validate_sample,
    )
    assert errors == ()
    assert original.validate_run(measured) == ()
    before = json.dumps(
        original.build_summary(measured, bootstrap_resamples=20),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    for row in rows:
        if row["phase"] == "burn_in":
            row["metrics"] = {name: 10**18 for name in profile.REQUIRED_METRICS}
    errors, changed_measured = profile.validate_confirmation_rows(
        rows,
        schedule,
        validate_sample=original.validate_sample,
    )
    assert errors == ()
    after = json.dumps(
        original.build_summary(changed_measured, bootstrap_resamples=20),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert after == before


def _original_protocol() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (
            repository_root
            / "Docs/superpowers/qa/console-three-turn-real-provider/"
            "real-provider-three-turn.manifest.json"
        ).read_bytes()
    )
    return profile.confirmation_protocol(
        revisions=manifest["revisions"],
        provider_kind=manifest["provider"],
        provider_server=manifest["provider_server"],
        runtime=manifest["runtime"],
        model_alias=manifest["provider_server"]["model_alias"],
        workspace_content_tree_digest=manifest["fixture_hashes"][
            "workspace_content_tree_digest"
        ],
        tool_definition_sha256_by_arm=manifest["fixture_hashes"][
            "tool_definition_sha256_by_arm"
        ],
    )


def test_confirmation_protocol_pins_every_original_machine_contract() -> None:
    protocol = _original_protocol()

    assert protocol == {
        "revisions": {
            "control": profile.CONTROL_SHA,
            "candidate": profile.CANDIDATE_SHA,
        },
        "provider_kind": "llama_cpp",
        "provider_server": {
            "build_info": "b8795-c0de6eda7",
            "model_alias": "gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf",
            "total_slots": 1,
            "context_tokens": 64_000,
            "endpoints": {"metrics": False, "slots": True, "props": False},
            "is_sleeping": False,
            "modalities": {"vision": False, "audio": False},
        },
        "runtime": {
            "python": {"implementation": "cpython", "version": "3.12.11"},
            "sqlite": "3.49.1",
            "dependencies": {
                "httpx": "0.28.1",
                "pydantic": "2.12.5",
                "rich": "14.3.3",
                "textual": "8.2.8",
            },
        },
        "model_alias": "gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf",
        "request_settings": {
            "temperature": 0.0,
            "max_tokens": 512,
            "reasoning_effort": "none",
            "streaming": True,
            "include_usage": True,
        },
        "fixture_ids": {
            "turn_prompts": "task-19641-three-turn-prompts-v1",
            "tool_schema": "local:fs_write-target-definition-v1",
            "mutation": "task-19641-confined-fs-write-v1",
            "workspace_corpus": "task-19641-workspace-corpus-v1",
        },
        "fixture_hashes": {
            "turn_prompts_sha256": (
                "3f6b88ffa37b4f6b9673878288b93d81e965c50cba1fb2ce7bbb4dfadb5245ac"
            ),
            "mutation_sha256": (
                "04630906249e9f2fe123d406404dc8577d2b998484217f9c48d64f7368198ce6"
            ),
            "workspace_content_tree_digest": (
                "f7d3e52271c125417208dbad8c1f7a3aadc1e80a5c7f1856db787c9132873ea6"
            ),
            "tool_definition_sha256_by_arm": {
                arm: "be1dec3a1a1a7f31c8fd33956eab6ba5c6100c9b3649e55b928764911476d0bf"
                for arm in profile.ARMS
            },
        },
        "metric_names": sorted(profile.REQUIRED_METRICS),
        "primary_gate_names": sorted(profile.NON_REGRESSION_METRICS),
        "p95": {
            "method": "nearest_rank",
            "fraction": 0.95,
            "behavior_sha256": (
                "0d7a800a516a1394b7b86639678314b863603a22f3d139d57c2b41f0940fd742"
            ),
        },
        "measured_blocks": 30,
        "resampling": {
            "method": "paired_complete_blocks",
            "resamples": 10_000,
            "seed": 19_641,
            "behavior_sha256": (
                "79cd4bb711ed33737ebe3ed28f0369ae44b29b5383bb95849edabcbfe0fb53eb"
            ),
        },
        "confidence_bounds": [
            "two_sided_95",
            "one_sided_lower_95",
            "one_sided_upper_95",
        ],
        "non_regression_ceiling": 1.10,
        "improvement_ceiling": 1.00,
    }


def test_confirmation_protocol_rejects_wrong_current_nearest_rank_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profile,
        "nearest_rank_percentile",
        lambda values, _fraction: min(values),
    )

    with pytest.raises(RuntimeError, match="confirmation_protocol_statistics_invalid"):
        _original_protocol()


def test_confirmation_protocol_rejects_wrong_current_paired_block_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        profile,
        "paired_p95_ratio_bounds",
        lambda *_args, **_kwargs: {
            "two_sided_95": (1.0, 1.0),
            "one_sided_lower_95": 1.0,
            "one_sided_upper_95": 1.0,
        },
    )

    with pytest.raises(RuntimeError, match="confirmation_protocol_statistics_invalid"):
        _original_protocol()


def test_current_summary_behavior_drift_mismatches_pinned_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_original_evidence(tmp_path)
    _materialize_original_runner(tmp_path)
    expected = profile.load_original_protocol(tmp_path, tmp_path)
    real_build_summary = profile.build_summary

    @functools.wraps(real_build_summary)
    def drifted_summary(*args, **kwargs):
        result = real_build_summary(*args, **kwargs)
        result["overall_verdict"] = "drifted"
        return result

    monkeypatch.setattr(profile, "build_summary", drifted_summary)
    observed = _original_protocol()

    assert "protocol_resampling_mismatch" in profile.protocol_mismatches(
        expected, observed
    )


def test_current_resample_cap_mismatches_pinned_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_original_evidence(tmp_path)
    _materialize_original_runner(tmp_path)
    expected = profile.load_original_protocol(tmp_path, tmp_path)
    real_paired = profile.paired_p95_ratio_bounds

    @functools.wraps(real_paired)
    def capped_paired(blocks, candidate, *, resamples=10_000, seed=19_641):
        return real_paired(
            blocks,
            candidate,
            resamples=min(resamples, 32),
            seed=seed,
        )

    monkeypatch.setattr(profile, "paired_p95_ratio_bounds", capped_paired)
    observed = _original_protocol()

    assert "protocol_resampling_mismatch" in profile.protocol_mismatches(
        expected, observed
    )


def test_current_mean_summary_mismatches_pinned_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_original_evidence(tmp_path)
    _materialize_original_runner(tmp_path)
    expected = profile.load_original_protocol(tmp_path, tmp_path)
    real_build_summary = profile.build_summary

    @functools.wraps(real_build_summary)
    def mean_summary(rows, **kwargs):
        summary = real_build_summary(rows, **kwargs)
        measured = [row for row in rows if row.get("phase") == "measured"]
        for arm in profile.ARMS:
            arm_rows = [row for row in measured if row["arm"] == arm]
            for metric, distribution in summary["arms"][arm]["metrics"].items():
                values = [float(row["metrics"][metric]) for row in arm_rows]
                distribution["median"] = sum(values) / len(values)
        return summary

    monkeypatch.setattr(profile, "build_summary", mean_summary)
    observed = _original_protocol()

    assert "protocol_resampling_mismatch" in profile.protocol_mismatches(
        expected, observed
    )


@pytest.mark.parametrize(
    ("path", "replacement", "code"),
    [
        (("revisions", "candidate"), "0" * 40, "protocol_revisions_mismatch"),
        (("provider_kind",), "other", "protocol_provider_kind_mismatch"),
        *((
            ("provider_server", *path),
            replacement,
            "protocol_provider_server_mismatch",
        ) for path, replacement in [
            (("build_info",), "other"),
            (("model_alias",), "other.gguf"),
            (("total_slots",), 2),
            (("context_tokens",), 32_000),
            (("endpoints", "metrics"), True),
            (("endpoints", "slots"), False),
            (("endpoints", "props"), True),
            (("is_sleeping",), True),
            (("modalities", "vision"), True),
            (("modalities", "audio"), True),
        ]),
        *((
            ("runtime", *path),
            "changed",
            "protocol_runtime_mismatch",
        ) for path in [
            ("python", "implementation"),
            ("python", "version"),
            ("sqlite",),
            ("dependencies", "httpx"),
            ("dependencies", "pydantic"),
            ("dependencies", "rich"),
            ("dependencies", "textual"),
        ]),
        (("model_alias",), "other.gguf", "protocol_model_alias_mismatch"),
        *((
            ("request_settings", key),
            replacement,
            "protocol_request_settings_mismatch",
        ) for key, replacement in [
            ("temperature", 0.1),
            ("max_tokens", 511),
            ("reasoning_effort", "low"),
            ("streaming", False),
            ("include_usage", False),
        ]),
        *((
            ("fixture_ids", key),
            "changed",
            "protocol_fixture_ids_mismatch",
        ) for key in ("turn_prompts", "tool_schema", "mutation", "workspace_corpus")),
        *((
            ("fixture_hashes", *path),
            "0" * 64,
            "protocol_fixture_hashes_mismatch",
        ) for path in [
            ("turn_prompts_sha256",),
            ("mutation_sha256",),
            ("workspace_content_tree_digest",),
            ("tool_definition_sha256_by_arm", "control"),
            ("tool_definition_sha256_by_arm", "disabled"),
            ("tool_definition_sha256_by_arm", "enabled"),
        ]),
        (("metric_names",), [], "protocol_metric_names_mismatch"),
        (("primary_gate_names",), [], "protocol_primary_gate_names_mismatch"),
        (("p95", "method"), "linear", "protocol_p95_mismatch"),
        (("p95", "fraction"), 0.9, "protocol_p95_mismatch"),
        (("p95", "behavior_sha256"), "0" * 64, "protocol_p95_mismatch"),
        (("measured_blocks",), 29, "protocol_measured_blocks_mismatch"),
        (("resampling", "method"), "rows", "protocol_resampling_mismatch"),
        (("resampling", "resamples"), 9_999, "protocol_resampling_mismatch"),
        (("resampling", "seed"), 1, "protocol_resampling_mismatch"),
        (
            ("resampling", "behavior_sha256"),
            "0" * 64,
            "protocol_resampling_mismatch",
        ),
        (("confidence_bounds",), [], "protocol_confidence_bounds_mismatch"),
        (("non_regression_ceiling",), 1.11, "protocol_non_regression_ceiling_mismatch"),
        (("improvement_ceiling",), 1.01, "protocol_improvement_ceiling_mismatch"),
    ],
)
def test_protocol_comparison_fails_closed_with_stable_mismatch_codes(
    path: tuple[str, ...], replacement: object, code: str
) -> None:
    expected = _original_protocol()
    observed = copy.deepcopy(expected)
    target = observed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    assert profile.protocol_mismatches(expected, observed) == (code,)


def test_original_protocol_loader_never_parses_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_original_evidence(tmp_path)
    _materialize_original_runner(tmp_path)
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.suffix == ".md":
            pytest.fail("Markdown must not be parsed as protocol input")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    assert profile.load_original_protocol(tmp_path, tmp_path) == _original_protocol()


def test_original_protocol_uses_real_detached_runner_and_retained_evidence(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    git_repository = tmp_path / "repository"
    subprocess.run(
        ["git", "clone", "--no-checkout", str(repository_root), str(git_repository)],
        check=True,
        capture_output=True,
    )
    run_root = tmp_path / "run"
    candidate = profile.prepare_target_worktree(
        git_repository,
        run_root,
        name="candidate",
        revision=profile.ORIGINAL_HARNESS_SHA,
    )
    try:
        assert not (
            candidate
            / "Docs/superpowers/qa/console-three-turn-real-provider/"
            "real-provider-three-turn.manifest.json"
        ).exists()
        assert profile.load_original_protocol(candidate, repository_root) == _original_protocol()
        with pytest.raises(RuntimeError, match="original_evidence_missing"):
            profile.load_original_protocol(candidate, candidate)
    finally:
        profile._remove_target_worktree(
            git_repository, run_root, name="candidate"
        )


def test_original_protocol_is_independent_from_current_harness_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_original_evidence(tmp_path)
    _materialize_original_runner(tmp_path)
    monkeypatch.setattr(profile, "TURN_PROMPTS", ("drifted prompt",))
    monkeypatch.setattr(profile, "FIXED_MUTATION", b"drifted mutation")
    monkeypatch.setattr(
        profile,
        "REQUIRED_METRICS",
        ("drifted_metric", *profile.REQUIRED_METRICS),
    )
    monkeypatch.setattr(profile, "P95_FRACTION", 0.9)
    monkeypatch.setattr(profile, "MEASURED_BLOCKS", 29)
    monkeypatch.setattr(profile, "NON_REGRESSION_CEILING", 1.2)
    monkeypatch.setattr(profile, "IMPROVEMENT_CEILING", 0.9)
    current_build_summary = profile.build_summary

    def drifted_current_summary(
        rows, *, bootstrap_resamples=9_999, bootstrap_seed=7
    ):
        return current_build_summary(
            rows,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )

    monkeypatch.setattr(profile, "build_summary", drifted_current_summary)
    monkeypatch.setattr(
        profile,
        "REQUEST_SETTINGS",
        {
            "temperature": 0.1,
            "max_tokens": 511,
            "reasoning_effort": "low",
            "streaming": False,
            "include_usage": False,
        },
        raising=False,
    )
    expected = profile.load_original_protocol(tmp_path, tmp_path)
    observed = _original_protocol()

    assert expected["fixture_hashes"]["turn_prompts_sha256"] == (
        "3f6b88ffa37b4f6b9673878288b93d81e965c50cba1fb2ce7bbb4dfadb5245ac"
    )
    assert expected["metric_names"] == sorted([
        "assistant_durable_to_release_ns",
        "conversation_wall_ns",
        "event_loop_lag_p95_ns",
        "provider_total_ns",
        "terminal_to_third_provider_ns",
        "third_send_to_worker_ns",
    ])
    assert profile.protocol_mismatches(expected, observed) == (
        "protocol_request_settings_mismatch",
        "protocol_fixture_hashes_mismatch",
        "protocol_metric_names_mismatch",
        "protocol_p95_mismatch",
        "protocol_measured_blocks_mismatch",
        "protocol_resampling_mismatch",
        "protocol_non_regression_ceiling_mismatch",
        "protocol_improvement_ceiling_mismatch",
    )


def test_original_protocol_discriminates_through_pinned_statistics_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_original_evidence(tmp_path)
    _materialize_original_runner(tmp_path)
    original = profile.load_original_runner(tmp_path)
    calls: list[tuple[str, object]] = []

    for name in (
        "nearest_rank_percentile",
        "paired_p95_ratio_bounds",
        "build_summary",
    ):
        function = getattr(original, name)

        @functools.wraps(function)
        def wrapped(*args, __name=name, __function=function, **kwargs):
            calls.append((__name, kwargs.copy()))
            return __function(*args, **kwargs)

        monkeypatch.setattr(original, name, wrapped)
    monkeypatch.setattr(profile, "load_original_runner", lambda _root: original)

    protocol = profile.load_original_protocol(tmp_path, tmp_path)

    assert protocol["p95"]["method"] == "nearest_rank"
    assert protocol["p95"]["fraction"] == 0.95
    assert len(protocol["p95"]["behavior_sha256"]) == 64
    assert any(name == "nearest_rank_percentile" for name, _kwargs in calls)
    assert any(
        name == "paired_p95_ratio_bounds"
        and kwargs.get("resamples") == 1
        and kwargs.get("seed") == 0
        for name, kwargs in calls
    )
    assert any(
        name == "paired_p95_ratio_bounds"
        and kwargs.get("resamples") == 10_000
        and kwargs.get("seed") == 19_641
        for name, kwargs in calls
    )
    assert any(
        name == "build_summary"
        and kwargs.get("bootstrap_resamples") == 10_000
        and kwargs.get("bootstrap_seed") == 19_641
        for name, kwargs in calls
    )
    assert sum(name == "build_summary" for name, _kwargs in calls) > 2


def test_original_protocol_rejects_summary_percentile_behavior_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_original_evidence(tmp_path)
    _materialize_original_runner(tmp_path)
    original = profile.load_original_runner(tmp_path)
    real_build_summary = original.build_summary

    @functools.wraps(real_build_summary)
    def drifted_build_summary(rows, **kwargs):
        summary = real_build_summary(rows, **kwargs)
        measured = [row for row in rows if row.get("phase") == "measured"]
        for arm, arm_summary in summary["arms"].items():
            arm_rows = [row for row in measured if row["arm"] == arm]
            for metric, distribution in arm_summary["metrics"].items():
                distribution["p95"] = original.nearest_rank_percentile(
                    [row["metrics"][metric] for row in arm_rows], 0.9
                )
        return summary

    monkeypatch.setattr(original, "build_summary", drifted_build_summary)
    monkeypatch.setattr(profile, "load_original_runner", lambda _root: original)

    with pytest.raises(RuntimeError, match="original_protocol_invalid"):
        profile.load_original_protocol(tmp_path, tmp_path)


def test_original_threshold_discovery_preserves_exact_float_boundaries() -> None:
    non_regression = 1.104
    improvement = 0.997

    class Original:
        def paired_p95_ratio_bounds(self, *_args, **_kwargs):
            raise AssertionError("threshold helper must inject discriminating bounds")

        def build_summary(self, _rows, *, bootstrap_resamples):
            bounds = self.paired_p95_ratio_bounds()
            ratio = bounds["one_sided_upper_95"]
            claims = {"metric": {"disabled": {}}} if ratio < improvement else {}
            return {
                "arms": {
                    "disabled": {
                        "verdict": "pass" if ratio <= non_regression else "regression"
                    }
                },
                "critical_path_improvement_claims": claims,
            }

    original = Original()

    assert profile._original_thresholds(original, ()) == (
        non_regression,
        improvement,
    )
    assert math.nextafter(non_regression, -math.inf) < non_regression
    assert math.nextafter(improvement, math.inf) > improvement


@pytest.mark.parametrize(
    "overrides",
    [
        {"revisions": None},
        {"provider_kind": []},
        {"provider_server": []},
        {"runtime": "runtime"},
        {"model_alias": 1},
        {"workspace_content_tree_digest": None},
        {"tool_definition_sha256_by_arm": []},
        {"tool_definition_sha256_by_arm": {arm: None for arm in profile.ARMS}},
    ],
)
def test_confirmation_protocol_rejects_malformed_json_shapes_stably(
    overrides: dict[str, object],
) -> None:
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "Docs/superpowers/qa/console-three-turn-real-provider/"
            "real-provider-three-turn.manifest.json"
        ).read_bytes()
    )
    arguments = {
        "revisions": manifest["revisions"],
        "provider_kind": manifest["provider"],
        "provider_server": manifest["provider_server"],
        "runtime": manifest["runtime"],
        "model_alias": manifest["provider_server"]["model_alias"],
        "workspace_content_tree_digest": manifest["fixture_hashes"][
            "workspace_content_tree_digest"
        ],
        "tool_definition_sha256_by_arm": manifest["fixture_hashes"][
            "tool_definition_sha256_by_arm"
        ],
    }
    arguments.update(overrides)

    with pytest.raises(RuntimeError, match="^confirmation_protocol_invalid$"):
        profile.confirmation_protocol(**arguments)


@pytest.mark.parametrize("expected,observed", [(None, {}), ({}, []), ([], []), (1, "x")])
def test_protocol_mismatches_rejects_malformed_json_shapes_stably(
    expected: object, observed: object
) -> None:
    assert profile.protocol_mismatches(expected, observed) == (
        "protocol_schema_mismatch",
    )


def test_validate_run_requires_thirty_complete_unique_rotation_blocks() -> None:
    validate_run = getattr(profile, "validate_run", None)
    rows = _valid_run()

    assert callable(validate_run)
    assert validate_run(rows) == ()
    assert "sample_count" in validate_run(rows[:-1])
    duplicated = copy.deepcopy(rows)
    duplicated[-1]["sample_id"] = duplicated[0]["sample_id"]
    assert "sample_id_duplicate" in validate_run(duplicated)
    wrong_block = copy.deepcopy(rows)
    wrong_block[-1]["iteration"] = 0
    assert "rotation_block_contract" in validate_run(wrong_block)


def test_validate_run_requires_one_successful_warmup_per_arm() -> None:
    validate_run = getattr(profile, "validate_run", None)
    rows = _valid_run()

    assert callable(validate_run)
    assert "warmup_contract" in validate_run(rows[1:])
    rows[0]["status"] = "failed"
    assert "warmup_contract" in validate_run(rows)


def test_build_summary_requires_both_non_regression_gates() -> None:
    build_summary = getattr(profile, "build_summary", None)
    rows = _valid_run()
    for row in rows:
        if row["arm"] == "disabled":
            row["metrics"]["third_send_to_worker_ns"] *= 2
        if row["arm"] == "enabled":
            row["metrics"]["event_loop_lag_p95_ns"] *= 2

    assert callable(build_summary)
    summary = build_summary(rows, bootstrap_resamples=100, bootstrap_seed=7)

    assert summary["arms"]["disabled"]["verdict"] == "regression"
    assert summary["arms"]["enabled"]["verdict"] == "regression"
    assert summary["overall_verdict"] == "regression"
    assert "provider_total_ns" not in summary["critical_path_improvement_claims"]
    assert "conversation_wall_ns" not in summary["critical_path_improvement_claims"]


def test_build_summary_invalidates_before_statistics() -> None:
    build_summary = getattr(profile, "build_summary", None)
    rows = _valid_run()
    rows[0]["terminal_third_assistant_ns"] = None

    assert callable(build_summary)
    summary = build_summary(rows, bootstrap_resamples=10)
    assert summary["overall_verdict"] == "invalid"
    assert summary["validation_errors"]


@pytest.mark.parametrize(
    "payload",
    [
        {"api_key": "secret"},
        {"headers": {"Authorization": "Bearer secret"}},
        {"prompt": "synthetic but prohibited"},
        {"tool_result": "body"},
        {"error": "/Users/alice/work/app.py:12"},
        {"path": "/tmp/absolute.txt"},
        {"pid": 42},
        {"listener_pid": 42},
    ],
)
def test_privacy_violations_reject_sensitive_keys_and_absolute_paths(payload) -> None:
    privacy_violations = getattr(profile, "privacy_violations", None)

    assert callable(privacy_violations)
    assert privacy_violations(payload)


def test_normalize_text_replaces_known_roots_longest_first(tmp_path) -> None:
    normalize_text = getattr(profile, "normalize_text", None)
    candidate = tmp_path / "candidate"
    child = candidate / "sample"

    assert callable(normalize_text)
    normalized = normalize_text(
        f"failed at {child}/file.py under {candidate}",
        {"$CANDIDATE": candidate, "$RUN": child},
    )
    assert str(candidate) not in normalized
    assert normalized == "failed at $RUN/file.py under $CANDIDATE"


class _FlushCountingText(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def test_boundary_writer_flushes_but_heartbeat_buffer_never_writes() -> None:
    write_boundary_event = getattr(profile, "write_boundary_event", None)
    heartbeat_buffer_type = getattr(profile, "HeartbeatBuffer", None)
    write_terminal_sample = getattr(profile, "write_terminal_sample", None)
    destination = _FlushCountingText()

    assert callable(write_boundary_event)
    assert heartbeat_buffer_type is not None
    assert callable(write_terminal_sample)
    write_boundary_event(destination, {"event": "provider_started", "at_ns": 1})
    assert destination.flush_count == 1
    heartbeat = heartbeat_buffer_type(capacity=3)
    heartbeat.record(4)
    heartbeat.record(5)
    assert destination.flush_count == 1
    write_terminal_sample(destination, {"sample_id": "s"}, heartbeat)
    assert destination.flush_count == 2
    records = [json.loads(line) for line in destination.getvalue().splitlines()]
    assert records[-1]["heartbeat_lateness_ns"] == [4, 5]


def test_build_child_environment_is_strictly_allowlisted(tmp_path: Path) -> None:
    build_child_environment = getattr(profile, "build_child_environment", None)
    base = {
        "PATH": "/usr/bin",
        "LANG": "en_US.UTF-8",
        "TERM": "xterm-256color",
        "OPENAI_API_KEY": "secret",
        "HTTPS_PROXY": "http://proxy.invalid",
        "NO_PROXY": "localhost",
        "PYTHONPATH": "/candidate/that/must/not/leak",
        "HOME": "/Users/alice",
        "UNRELATED": "host-state",
    }

    assert callable(build_child_environment)
    environment = build_child_environment(base, tmp_path / "sample")

    assert environment["PATH"] == "/usr/bin"
    assert environment["LANG"] == "en_US.UTF-8"
    assert environment["TERM"] == "xterm-256color"
    assert environment["HOME"] == str(tmp_path / "sample" / "home")
    assert environment["XDG_CONFIG_HOME"] == str(tmp_path / "sample" / "config")
    assert environment["XDG_DATA_HOME"] == str(tmp_path / "sample" / "data")
    assert environment["XDG_CACHE_HOME"] == str(tmp_path / "sample" / "cache")
    assert environment["TMPDIR"] == str(tmp_path / "sample" / "tmp")
    assert environment["TLDW_TEST_MODE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert set(environment).isdisjoint(
        {"OPENAI_API_KEY", "HTTPS_PROXY", "NO_PROXY", "PYTHONPATH", "UNRELATED"}
    )


def test_sample_schedule_has_unmeasured_warmups_then_complete_rotations() -> None:
    sample_schedule = getattr(profile, "sample_schedule", None)

    assert callable(sample_schedule)
    schedule = sample_schedule(4)
    assert [(item.phase, item.arm) for item in schedule[:3]] == [
        ("warmup", "control"),
        ("warmup", "disabled"),
        ("warmup", "enabled"),
    ]
    assert [item.arm for item in schedule[3:6]] == list(profile.balanced_arm_order(0))
    assert [item.arm for item in schedule[6:9]] == list(profile.balanced_arm_order(1))
    assert sum(item.phase == "measured" for item in schedule) == 12


def test_confirmatory_schedule_continues_rotation_after_five_burn_in_blocks() -> None:
    schedule = profile.sample_schedule(30, burn_in_blocks=5)

    assert [(row.phase, row.arm, row.iteration) for row in schedule[:3]] == [
        ("warmup", "control", -1),
        ("warmup", "disabled", -1),
        ("warmup", "enabled", -1),
    ]
    burn_in = [row for row in schedule if row.phase == "burn_in"]
    assert len(burn_in) == 15
    for block in range(5):
        block_rows = burn_in[block * len(profile.ARMS) : (block + 1) * len(profile.ARMS)]
        assert [(row.arm, row.iteration) for row in block_rows] == [
            (arm, block) for arm in profile.balanced_arm_order(block)
        ]
    measured = [row for row in schedule if row.phase == "measured"]
    assert len(measured) == 90
    assert [row.arm for row in measured[:3]] == list(profile.balanced_arm_order(5))
    assert [row.iteration for row in measured[:3]] == [0, 0, 0]


def test_zero_burn_in_keeps_the_existing_schedule() -> None:
    assert profile.sample_schedule(4) == profile.sample_schedule(4, burn_in_blocks=0)


@pytest.mark.parametrize(
    ("iterations", "burn_in_blocks"),
    ((0, 0), (-1, 0), (1, -1)),
)
def test_sample_schedule_rejects_invalid_counts(
    iterations: int, burn_in_blocks: int
) -> None:
    with pytest.raises(
        ValueError,
        match="schedule counts must be nonnegative with measured iterations",
    ):
        profile.sample_schedule(iterations, burn_in_blocks=burn_in_blocks)


def _write_target_package(root: Path, marker: str) -> None:
    for relative in (
        "tldw_chatbook/__init__.py",
        "Tests/__init__.py",
        "Tests/UI/__init__.py",
        "Tests/UI/test_destination_shells.py",
        "Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"MARKER = {marker!r}\n", encoding="utf-8")


def test_target_imports_resolve_only_below_selected_root(tmp_path: Path) -> None:
    target_a = tmp_path / "target-a"
    target_b = tmp_path / "target-b"
    _write_target_package(target_a, "A")
    _write_target_package(target_b, "B")
    runner = Path(profile.__file__).resolve()
    program = textwrap.dedent(
        f"""
        import importlib.util, json, pathlib, sys
        sys.path.insert(0, {str(target_a)!r})
        import tldw_chatbook
        spec = importlib.util.spec_from_file_location("benchmark_runner", {str(runner)!r})
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        runner.install_target_root(pathlib.Path({str(target_b)!r}))
        resolved = runner.assert_target_modules(runner.TARGET_MODULES, pathlib.Path({str(target_b)!r}))
        print(json.dumps(resolved, sort_keys=True))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    resolved = json.loads(completed.stdout)
    assert all(Path(path).is_relative_to(target_b) for path in resolved.values())


def test_assert_target_modules_rejects_an_import_outside_target(tmp_path: Path) -> None:
    assert_target_modules = getattr(profile, "assert_target_modules", None)
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")

    assert callable(assert_target_modules)
    with pytest.raises(RuntimeError, match="target_import_mismatch"):
        assert_target_modules({"fake": outside}, tmp_path / "target")


def test_watchdog_preserves_last_event_on_nonzero_exit(tmp_path: Path) -> None:
    run_child_with_watchdog = getattr(profile, "run_child_with_watchdog", None)
    evidence = tmp_path / "events.jsonl"
    program = (
        "import json,pathlib,sys; "
        f"p=pathlib.Path({str(evidence)!r}); "
        "p.write_text(json.dumps({'event':'before_failure'})+'\\n'); sys.exit(7)"
    )

    assert callable(run_child_with_watchdog)
    result = run_child_with_watchdog(
        [sys.executable, "-c", program], evidence_path=evidence, timeout_seconds=2
    )
    assert result.status == "failed"
    assert result.returncode == 7
    assert result.last_event == {"event": "before_failure"}


def test_watchdog_reports_normal_exit(tmp_path: Path) -> None:
    run_child_with_watchdog = getattr(profile, "run_child_with_watchdog", None)
    evidence = tmp_path / "events.jsonl"

    assert callable(run_child_with_watchdog)
    result = run_child_with_watchdog(
        [sys.executable, "-c", "pass"],
        evidence_path=evidence,
        timeout_seconds=2,
    )
    assert result.status == "complete"
    assert result.returncode == 0
    assert result.last_event is None


def test_watchdog_terminates_then_kills_a_term_resistant_child(tmp_path: Path) -> None:
    run_child_with_watchdog = getattr(profile, "run_child_with_watchdog", None)
    evidence = tmp_path / "events.jsonl"
    program = textwrap.dedent(
        f"""
        import json, pathlib, signal, time
        signal.signal(signal.SIGTERM, lambda *_: None)
        path = pathlib.Path({str(evidence)!r})
        with path.open('w') as stream:
            stream.write(json.dumps({{'event': 'waiting'}}) + '\\n')
            stream.flush()
            while True:
                time.sleep(0.1)
        """
    )

    assert callable(run_child_with_watchdog)
    result = run_child_with_watchdog(
        [sys.executable, "-c", program],
        evidence_path=evidence,
        timeout_seconds=0.2,
        term_grace_seconds=0.1,
    )
    assert result.status == "timed_out_killed"
    assert result.last_event == {"event": "waiting"}


def test_write_child_config_pins_local_provider_and_disables_background_work(
    tmp_path: Path,
) -> None:
    write_child_config = getattr(profile, "write_child_config", None)

    assert callable(write_child_config)
    path = write_child_config(
        tmp_path / "sample",
        endpoint="http://127.0.0.1:9099",
        model="fixture-model.gguf",
    )
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    assert config["first_run"]["setup_completed"] is True
    assert config["splash_screen"]["enabled"] is False
    assert config["model_catalog"]["auto_refresh_enabled"] is False
    assert config["subscriptions"]["enable_background_checking"] is False
    assert config["console"] == {"local_tools_enabled": True, "workspace_root": ""}
    assert config["api_settings"]["llama_cpp"] == {
        "api_url": "http://127.0.0.1:9099",
        "model": "fixture-model.gguf",
        "temperature": 0.0,
        "max_tokens": 512,
        "reasoning_effort": "none",
        "timeout": 120,
        "retries": 0,
        "streaming": True,
    }


def test_resolve_revision_requires_exact_control_and_full_candidate_hash(
    tmp_path: Path,
) -> None:
    resolve_benchmark_revisions = getattr(profile, "resolve_benchmark_revisions", None)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        ref = command[-1]
        value = profile.CONTROL_SHA if ref.startswith(profile.CONTROL_SHA) else "c" * 40
        return subprocess.CompletedProcess(command, 0, stdout=value + "\n", stderr="")

    assert callable(resolve_benchmark_revisions)
    revisions = resolve_benchmark_revisions(
        tmp_path,
        control_ref=profile.CONTROL_SHA,
        candidate_ref="HEAD",
        run_command=fake_run,
    )
    assert revisions == {"control": profile.CONTROL_SHA, "candidate": "c" * 40}
    assert len(calls) == 2


def test_resolve_revision_rejects_a_different_control_hash(tmp_path: Path) -> None:
    resolve_benchmark_revisions = getattr(profile, "resolve_benchmark_revisions", None)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="d" * 40 + "\n", stderr="")

    assert callable(resolve_benchmark_revisions)
    with pytest.raises(RuntimeError, match="control_revision_mismatch"):
        resolve_benchmark_revisions(
            tmp_path,
            control_ref=profile.CONTROL_SHA,
            candidate_ref="HEAD",
            run_command=fake_run,
        )


def test_parse_arguments_pins_safe_benchmark_defaults(tmp_path: Path) -> None:
    parse_arguments = getattr(profile, "parse_arguments", None)

    assert callable(parse_arguments)
    arguments = parse_arguments(
        [
            "--endpoint",
            "http://127.0.0.1:9099",
            "--model",
            "fixture.gguf",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert arguments.iterations == 30
    assert arguments.control_sha == profile.CONTROL_SHA
    assert arguments.candidate_sha == "HEAD"
    assert arguments.sample_timeout == 900.0
    assert arguments.child_spec is None


def test_prepare_output_root_allows_only_retained_readme(tmp_path: Path) -> None:
    prepare_output_root = getattr(profile, "prepare_output_root", None)
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "README.md").write_text("instructions\n", encoding="utf-8")

    assert callable(prepare_output_root)
    prepare_output_root(output)
    assert (output / "README.md").is_file()

    (output / "stale.jsonl").write_text("stale\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output_root_not_empty"):
        prepare_output_root(output)


def test_remove_successful_sample_root_is_confined_to_run_samples(tmp_path: Path) -> None:
    remove_successful_sample_root = getattr(
        profile, "remove_successful_sample_root", None
    )
    run_root = tmp_path / "run"
    sample_root = run_root / "samples" / "000-measured-0-control"
    sample_root.mkdir(parents=True)
    (sample_root / "synthetic.bin").write_bytes(b"fixture")

    assert callable(remove_successful_sample_root)
    remove_successful_sample_root(run_root, sample_root)
    assert not sample_root.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(RuntimeError, match="sample_cleanup_refused"):
        remove_successful_sample_root(run_root, outside)
    assert outside.is_dir()


def test_preflight_provider_verifies_exact_model_without_credentials() -> None:
    preflight_provider = getattr(profile, "preflight_provider", None)
    requests: list[Request] = []

    class Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self.status = 200
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

    def fake_open(request: Request, timeout: float):
        requests.append(request)
        assert timeout > 0
        if request.full_url.endswith("/v1/models"):
            return Response({"data": [{"id": "fixture.gguf"}]})
        return Response(
            {"choices": [{"message": {"content": "synthetic-ok"}}]}
        )

    assert callable(preflight_provider)
    result = preflight_provider(
        "http://127.0.0.1:9099",
        "fixture.gguf",
        urlopen=fake_open,
    )

    assert result == {
        "status": "ready",
        "model": "fixture.gguf",
        "model_count": 1,
        "completion_chars": 12,
    }
    assert [request.full_url for request in requests] == [
        "http://127.0.0.1:9099/v1/models",
        "http://127.0.0.1:9099/v1/chat/completions",
    ]
    assert all("Authorization" not in request.headers for request in requests)
    payload = json.loads(requests[1].data)
    assert payload["model"] == "fixture.gguf"
    assert payload["temperature"] == 0.0
    assert payload["stream"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_extract_sse_usage_retains_only_exact_token_counts() -> None:
    extract_sse_usage = getattr(profile, "extract_sse_usage", None)

    assert callable(extract_sse_usage)
    assert extract_sse_usage(
        'data: {"usage":{"prompt_tokens":17,"completion_tokens":4,'
        '"total_tokens":21,"prompt_tokens_details":{"cached_tokens":2}},'
        '"choices":[],"secret":"drop-me"}'
    ) == {
        "prompt_tokens": 17,
        "completion_tokens": 4,
        "total_tokens": 21,
    }
    assert extract_sse_usage("data: [DONE]") is None
    assert extract_sse_usage(
        'data: {"usage":{"prompt_tokens":17,"completion_tokens":4,'
        '"total_tokens":20}}'
    ) is None


def test_runtime_and_host_metadata_are_content_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_metadata = getattr(profile, "runtime_metadata", None)
    host_load_snapshot = getattr(profile, "host_load_snapshot", None)

    assert callable(runtime_metadata)
    assert callable(host_load_snapshot)
    monkeypatch.setattr(profile.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(profile.os, "getloadavg", lambda: (1.25, 2.5, 3.75))

    runtime = runtime_metadata(version_lookup=lambda name: f"fixture-{name}")
    host = host_load_snapshot()

    assert runtime["dependencies"] == {
        name: f"fixture-{name}"
        for name in ("httpx", "pydantic", "rich", "textual")
    }
    assert isinstance(runtime["sqlite"], str)
    assert host == {"logical_cpu_count": 8, "load_average": [1.25, 2.5, 3.75]}
    assert not profile.privacy_violations({"runtime": runtime, "host": host})


def test_provider_server_metadata_is_sanitized_and_model_pinned() -> None:
    provider_server_metadata = getattr(profile, "provider_server_metadata", None)
    requests: list[Request] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "build_info": "b8795-c0de6eda7",
                    "model_alias": "fixture.gguf",
                    "model_path": "/private/secret/model.gguf",
                    "total_slots": 1,
                    "default_generation_settings": {"n_ctx": 64_000},
                    "endpoint_metrics": False,
                    "endpoint_slots": True,
                    "endpoint_props": False,
                    "is_sleeping": False,
                    "modalities": {"vision": False, "audio": False},
                    "chat_template": "private template body",
                }
            ).encode("utf-8")

    def fake_open(request: Request, timeout: float):
        requests.append(request)
        assert timeout > 0
        return Response()

    assert callable(provider_server_metadata)
    result = provider_server_metadata(
        "http://127.0.0.1:9099/v1",
        "fixture.gguf",
        urlopen=fake_open,
    )

    assert requests[0].full_url == "http://127.0.0.1:9099/props"
    assert result == {
        "build_info": "b8795-c0de6eda7",
        "model_alias": "fixture.gguf",
        "total_slots": 1,
        "context_tokens": 64_000,
        "endpoints": {"metrics": False, "slots": True, "props": False},
        "is_sleeping": False,
        "modalities": {"vision": False, "audio": False},
    }
    assert "model_path" not in json.dumps(result)


def test_listener_resource_snapshot_retains_counts_not_process_identity() -> None:
    listener_resource_snapshot = getattr(profile, "listener_resource_snapshot", None)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[0] == "lsof":
            return SimpleNamespace(returncode=0, stdout="42\n84\n", stderr="")
        values = {"42": " 1024  7.5\n", "84": " 2048  2.25\n"}
        return SimpleNamespace(returncode=0, stdout=values[command[-1]], stderr="")

    assert callable(listener_resource_snapshot)
    result = listener_resource_snapshot(
        "http://127.0.0.1:9099", run_command=fake_run
    )

    assert result == {
        "listener_count": 2,
        "processes": [
            {"rss_bytes": 1_048_576, "cpu_percent": 7.5},
            {"rss_bytes": 2_097_152, "cpu_percent": 2.25},
        ],
    }
    assert commands[0] == [
        "lsof",
        "-nP",
        "-t",
        "-iTCP:9099",
        "-sTCP:LISTEN",
    ]
    assert all("42" not in row and "84" not in row for row in result["processes"])


@pytest.mark.parametrize("status", (" M tracked.py\n", "?? untracked.txt\n"))
def test_current_harness_identity_refuses_any_dirty_file(
    tmp_path: Path, status: str
) -> None:
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=status, stderr="")

    runner = tmp_path / "runner.py"
    runner.write_text("fixture", encoding="utf-8")
    with pytest.raises(RuntimeError, match="harness_worktree_dirty"):
        profile.current_harness_identity(
            tmp_path,
            runner_path=runner,
            run_command=fake_run,
        )
    assert commands == [
        ["git", "status", "--porcelain", "--untracked-files=all"]
    ]


def test_current_harness_identity_retains_full_revision_and_runner_digest(
    tmp_path: Path,
) -> None:
    revision = "1" * 40
    runner = tmp_path / "runner.py"
    runner.write_bytes(b"runner fixture")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        stdout = "" if command[1] == "status" else f"{revision}\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    result = profile.current_harness_identity(
        tmp_path,
        runner_path=runner,
        run_command=fake_run,
    )

    assert result == {
        "revision": revision,
        "runner_sha256": hashlib.sha256(b"runner fixture").hexdigest(),
    }
    assert commands == [
        ["git", "status", "--porcelain", "--untracked-files=all"],
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
    ]


def test_parent_refuses_dirty_harness_before_mutating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        profile,
        "current_harness_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("harness_worktree_dirty")
        ),
    )
    monkeypatch.setattr(
        profile,
        "prepare_output_root",
        lambda *_args: pytest.fail("output mutation preceded harness guard"),
    )
    args = SimpleNamespace(output_root=output)

    with pytest.raises(RuntimeError, match="harness_worktree_dirty"):
        profile.run_parent_mode(args)

    assert not output.exists()


def test_parent_retains_harness_identity_and_splits_original_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = Path(profile.__file__).resolve().parents[2]
    output = tmp_path / "run"
    harness = {"revision": "1" * 40, "runner_sha256": "2" * 64}
    plans = tuple(
        profile.SamplePlan("measured", arm, 0) for arm in profile.ARMS
    )
    monkeypatch.setattr(profile, "current_harness_identity", lambda _root: harness)
    monkeypatch.setattr(
        profile,
        "resolve_benchmark_revisions",
        lambda *_args, **_kwargs: {
            "control": profile.CONTROL_SHA,
            "candidate": profile.CANDIDATE_SHA,
        },
    )
    for name in (
        "preflight_provider",
        "provider_server_metadata",
        "runtime_metadata",
        "host_load_snapshot",
        "listener_resource_snapshot",
    ):
        monkeypatch.setattr(profile, name, lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        profile,
        "listener_identity",
        lambda *_args, **_kwargs: {"fingerprint_sha256": "3" * 64},
    )
    monkeypatch.setattr(profile, "verify_listener_identity", lambda *_args: None)
    monkeypatch.setattr(profile, "sample_schedule", lambda _iterations: plans)
    monkeypatch.setattr(profile, "validate_sample", lambda _row: ())
    monkeypatch.setattr(profile, "validate_run", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(profile, "_remove_target_worktrees", lambda *_args, **_kwargs: None)

    def prepare(_repository, run_root, *, name, revision):
        target = run_root / name
        target.mkdir()
        return target

    monkeypatch.setattr(profile, "prepare_target_worktree", prepare)
    loaded: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        profile,
        "load_original_protocol",
        lambda runner_root, evidence_root: loaded.append((runner_root, evidence_root)),
    )
    monkeypatch.setattr(profile, "load_original_runner", lambda _root: profile)

    def child(command, **_kwargs):
        spec_path = Path(command[command.index("--child-spec") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        return profile.ChildResult(
            "complete",
            0,
            {
                "event": "sample",
                "sample_id": spec["sample_id"],
                "arm": spec["arm"],
                "workspace_content_tree_digest": "4" * 64,
                "expected_permission_definition_hash": "5" * 64,
            },
        )

    monkeypatch.setattr(profile, "run_child_with_watchdog", child)
    args = SimpleNamespace(
        output_root=output,
        control_sha=profile.CONTROL_SHA,
        candidate_sha=profile.CANDIDATE_SHA,
        endpoint="http://127.0.0.1:9099",
        model="fixture.gguf",
        iterations=1,
        sample_timeout=1.0,
    )

    assert profile.run_parent_mode(args) == 0
    manifest = json.loads(
        (output / "real-provider-three-turn.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["harness"] == harness
    assert loaded == [((output / "candidate").resolve(), repository_root)]


def test_parent_preserves_primary_failure_with_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        profile,
        "current_harness_identity",
        lambda _root: {"revision": "1" * 40, "runner_sha256": "2" * 64},
    )
    monkeypatch.setattr(
        profile,
        "resolve_benchmark_revisions",
        lambda *_args, **_kwargs: {
            "control": profile.CONTROL_SHA,
            "candidate": profile.CANDIDATE_SHA,
        },
    )
    for name in (
        "preflight_provider",
        "provider_server_metadata",
        "runtime_metadata",
        "host_load_snapshot",
        "listener_resource_snapshot",
    ):
        monkeypatch.setattr(profile, name, lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        profile,
        "listener_identity",
        lambda *_args, **_kwargs: {"fingerprint_sha256": "3" * 64},
    )
    calls = 0

    def prepare(_repository, run_root, *, name, revision):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("candidate-primary-failed")
        target = run_root / name
        target.mkdir()
        return target

    monkeypatch.setattr(profile, "prepare_target_worktree", prepare)
    cleaned: list[tuple[str, ...]] = []

    def cleanup(*_args, **kwargs):
        cleaned.append(tuple(kwargs["names"]))
        raise RuntimeError("cleanup-failed")

    monkeypatch.setattr(
        profile,
        "_remove_target_worktrees",
        cleanup,
    )
    args = SimpleNamespace(
        output_root=tmp_path / "run",
        control_sha=profile.CONTROL_SHA,
        candidate_sha=profile.CANDIDATE_SHA,
        endpoint="http://127.0.0.1:9099",
        model="fixture.gguf",
        iterations=1,
        sample_timeout=1.0,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        profile.run_parent_mode(args)

    assert [str(error) for error in caught.value.exceptions] == [
        "candidate-primary-failed",
        "cleanup-failed",
    ]
    assert cleaned == [("control",)]


def _listener_run(identity: str):
    def fake_run(command, **_kwargs):
        if command[0] == "lsof":
            return SimpleNamespace(returncode=0, stdout="42\n", stderr="")
        assert command == ["ps", "-o", "pid=,lstart=", "-p", "42"]
        return SimpleNamespace(
            returncode=0,
            stdout=f"42 {identity}\n",
            stderr="",
        )

    return fake_run


def test_listener_identity_hashes_pid_and_start_without_retaining_them() -> None:
    started = "Fri Aug 22 11:00:00 2026"
    result = profile.listener_identity(
        "http://127.0.0.1:9099",
        run_command=_listener_run(started),
    )

    assert result == {
        "listener_count": 1,
        "fingerprint_sha256": hashlib.sha256(
            f"42\0{started}".encode()
        ).hexdigest(),
    }
    retained = json.dumps(result, sort_keys=True)
    assert "pid" not in result
    assert started not in retained
    assert "command" not in retained
    assert "model" not in retained
    assert "/Users/" not in retained
    assert "secret" not in retained


def test_listener_identity_is_verified_at_repeated_boundaries() -> None:
    started = "Fri Aug 22 11:00:00 2026"
    first = profile.listener_identity(
        "http://127.0.0.1:9099",
        run_command=_listener_run(started),
    )

    for _boundary in range(6):
        assert profile.verify_listener_identity(
            "http://127.0.0.1:9099",
            first["fingerprint_sha256"],
            run_command=_listener_run(started),
        ) == first


def test_changed_listener_identity_invalidates_the_attempt() -> None:
    first = profile.listener_identity(
        "http://127.0.0.1:9099",
        run_command=_listener_run("Fri Aug 22 11:00:00 2026"),
    )

    with pytest.raises(RuntimeError, match="listener_identity_changed"):
        profile.verify_listener_identity(
            "http://127.0.0.1:9099",
            first["fingerprint_sha256"],
            run_command=_listener_run("Fri Aug 22 11:01:00 2026"),
        )


def test_listener_identity_rejects_malformed_inventory_with_stable_code() -> None:
    def fake_run(command, **_kwargs):
        assert command[0] == "lsof"
        return SimpleNamespace(returncode=0, stdout="not-a-pid\n", stderr="")

    with pytest.raises(RuntimeError, match="listener_identity_failed"):
        profile.listener_identity(
            "http://127.0.0.1:9099",
            run_command=fake_run,
        )


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://example.com",
        "http://user:secret@127.0.0.1:9099",
        "http://127.0.0.1:9099?token=secret",
    ),
)
def test_preflight_provider_rejects_nonlocal_or_credential_bearing_urls(
    endpoint: str,
) -> None:
    preflight_provider = getattr(profile, "preflight_provider", None)

    assert callable(preflight_provider)
    with pytest.raises(ValueError, match="preflight_endpoint_refused"):
        preflight_provider(endpoint, "fixture.gguf", urlopen=lambda *_a, **_k: None)


def test_main_preflight_only_emits_safe_json_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main = getattr(profile, "main", None)
    calls = []

    def fake_preflight(endpoint: str, model: str) -> dict[str, object]:
        calls.append((endpoint, model))
        return {
            "status": "ready",
            "model": model,
            "model_count": 1,
            "completion_chars": 12,
        }

    monkeypatch.setattr(profile, "preflight_provider", fake_preflight)
    assert callable(main)
    exit_code = main(
        [
            "--endpoint",
            "http://127.0.0.1:9099",
            "--model",
            "fixture.gguf",
            "--output-root",
            str(tmp_path),
            "--preflight-only",
        ]
    )

    assert exit_code == 0
    assert calls == [("http://127.0.0.1:9099", "fixture.gguf")]
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["event"] == "preflight"
    assert emitted["status"] == "ready"
    assert not profile.privacy_violations(emitted)


def test_assert_child_environment_requires_sample_owned_paths(tmp_path: Path) -> None:
    assert_child_environment = getattr(profile, "assert_child_environment", None)
    sample_root = tmp_path / "sample"
    environment = profile.build_child_environment({}, sample_root)

    assert callable(assert_child_environment)
    assert_child_environment(sample_root, environment)
    with_platform_key = dict(environment)
    with_platform_key["__CF_USER_TEXT_ENCODING"] = "0x1F5:0x0:0x0"
    assert_child_environment(sample_root, with_platform_key)

    escaped = dict(environment)
    escaped["XDG_DATA_HOME"] = str(tmp_path / "outside")
    with pytest.raises(RuntimeError, match="child_environment_mismatch"):
        assert_child_environment(sample_root, escaped)


def test_child_spec_round_trip_rejects_unknown_fields(tmp_path: Path) -> None:
    write_child_spec = getattr(profile, "write_child_spec", None)
    read_child_spec = getattr(profile, "read_child_spec", None)
    spec = {
        "sample_id": "measured-0-disabled",
        "phase": "measured",
        "iteration": 0,
        "arm": "disabled",
        "target_root": str((tmp_path / "target").resolve()),
        "sample_root": str((tmp_path / "sample").resolve()),
        "run_root": str(tmp_path.resolve()),
        "evidence_path": str((tmp_path / "raw.jsonl").resolve()),
    }

    assert callable(write_child_spec)
    assert callable(read_child_spec)
    path = tmp_path / "spec.json"
    write_child_spec(path, spec)
    assert read_child_spec(path) == spec

    path.write_text(json.dumps({**spec, "api_key": "forbidden"}))
    with pytest.raises(RuntimeError, match="child_spec_invalid"):
        read_child_spec(path)


@pytest.mark.parametrize("mode", ("sample", "protocol_preflight"))
def test_child_spec_accepts_only_exact_extended_modes(
    tmp_path: Path, mode: str
) -> None:
    spec = {
        "mode": mode,
        "sample_id": (
            "protocol_preflight-disabled"
            if mode == "protocol_preflight"
            else "burn_in-0-disabled"
        ),
        "phase": "protocol_preflight" if mode == "protocol_preflight" else "burn_in",
        "iteration": -1 if mode == "protocol_preflight" else 0,
        "arm": "disabled",
        "target_root": str((tmp_path / "target").resolve()),
        "sample_root": str((tmp_path / "sample").resolve()),
        "run_root": str(tmp_path.resolve()),
        "evidence_path": str((tmp_path / "raw.jsonl").resolve()),
    }
    path = tmp_path / f"{mode}.json"

    profile.write_child_spec(path, spec)
    assert profile.read_child_spec(path) == spec

    path.write_text(json.dumps({**spec, "mode": "unknown"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="child_spec_invalid"):
        profile.read_child_spec(path)


def test_protocol_preflight_derives_content_free_fingerprints_and_tears_down(
    tmp_path: Path,
) -> None:
    closed: list[str] = []
    runtime = SimpleNamespace(
        workspace_root=tmp_path / "workspace",
        permission_definition_hash="d" * 64,
        review_state="enabled",
        review_ready=True,
        close=lambda: closed.append("runtime"),
    )
    adapter = SimpleNamespace(
        revision_kind="candidate",
        close=lambda: closed.append("adapter"),
    )

    result = profile.protocol_preflight(
        tmp_path / "target",
        tmp_path / "sample",
        arm="enabled",
        adapter_factory=lambda *_args: adapter,
        runtime_factory=lambda *_args, **_kwargs: runtime,
        corpus_generator=lambda _root: {"content_tree_digest": "c" * 64},
    )

    assert result == {
        "event": "protocol_preflight",
        "arm": "enabled",
        "target_revision_kind": "candidate",
        "behavior_sha256": result["behavior_sha256"],
        "workspace_content_tree_digest": "c" * 64,
        "tool_definition_sha256": "d" * 64,
        "final_ownership": {"live_threads": 0},
    }
    assert len(result["behavior_sha256"]) == 64
    assert closed == ["runtime", "adapter"]
    assert not profile.privacy_violations(result)


def test_protocol_preflight_measures_ownership_only_after_cleanup(tmp_path: Path) -> None:
    closed: list[str] = []
    runtime = SimpleNamespace(
        workspace_root=tmp_path / "workspace",
        permission_definition_hash="d" * 64,
        review_state="disabled",
        review_ready=False,
        close=lambda: closed.append("runtime"),
    )
    adapter = SimpleNamespace(
        revision_kind="candidate",
        close=lambda: closed.append("adapter"),
    )

    def ownership_probe(_baseline):
        assert closed == ["runtime", "adapter"]
        return {"live_threads": 0}

    result = profile.protocol_preflight(
        tmp_path / "target",
        tmp_path / "sample",
        arm="disabled",
        adapter_factory=lambda *_args: adapter,
        runtime_factory=lambda *_args, **_kwargs: runtime,
        corpus_generator=lambda _root: {"content_tree_digest": "c" * 64},
        ownership_probe=ownership_probe,
    )

    assert result["final_ownership"] == ownership_probe(set())
    assert not {
        "provider_closed",
        "sqlite_closed",
        "shadow_operations_pending",
    } & result["final_ownership"].keys()


def test_protocol_preflight_restores_adapter_when_runtime_close_fails(
    tmp_path: Path,
) -> None:
    closed: list[str] = []

    def runtime_close() -> None:
        closed.append("runtime")
        raise RuntimeError("runtime-close-failed")

    runtime = SimpleNamespace(
        workspace_root=tmp_path / "workspace",
        permission_definition_hash="d" * 64,
        review_state="enabled",
        review_ready=True,
        close=runtime_close,
    )
    adapter = SimpleNamespace(
        revision_kind="candidate",
        close=lambda: closed.append("adapter"),
    )

    with pytest.raises(RuntimeError, match="runtime-close-failed"):
        profile.protocol_preflight(
            tmp_path / "target",
            tmp_path / "sample",
            arm="enabled",
            adapter_factory=lambda *_args: adapter,
            runtime_factory=lambda *_args, **_kwargs: runtime,
            corpus_generator=lambda _root: {"content_tree_digest": "c" * 64},
        )
    assert closed == ["runtime", "adapter"]


def test_protocol_preflight_preserves_primary_and_all_cleanup_failures(
    tmp_path: Path,
) -> None:
    closed: list[str] = []

    def fail(label: str):
        def action() -> None:
            closed.append(label)
            raise RuntimeError(f"{label}-failed")

        return action

    runtime = SimpleNamespace(
        workspace_root=tmp_path / "workspace",
        permission_definition_hash="invalid",
        review_state="enabled",
        review_ready=True,
        close=fail("runtime-close"),
    )
    adapter = SimpleNamespace(
        revision_kind="candidate",
        close=fail("adapter-close"),
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        profile.protocol_preflight(
            tmp_path / "target",
            tmp_path / "sample",
            arm="enabled",
            adapter_factory=lambda *_args: adapter,
            runtime_factory=lambda *_args, **_kwargs: runtime,
            corpus_generator=lambda _root: {"content_tree_digest": "c" * 64},
        )

    assert closed == ["runtime-close", "adapter-close"]
    assert [str(error) for error in caught.value.exceptions] == [
        "protocol_preflight_hash_invalid",
        "runtime-close-failed",
        "adapter-close-failed",
    ]


def test_protocol_preflight_child_never_mounts_a_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    sample_root = run_root / "preflight"
    target_root = tmp_path / "target"
    evidence = run_root / "raw.jsonl"
    sample_root.mkdir(parents=True)
    target_root.mkdir()
    config_path = sample_root / "config/tldw_cli/config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("", encoding="utf-8")
    spec_path = sample_root / "child-spec.json"
    profile.write_child_spec(
        spec_path,
        {
            "mode": "protocol_preflight",
            "sample_id": "protocol_preflight-enabled",
            "phase": "protocol_preflight",
            "iteration": -1,
            "arm": "enabled",
            "target_root": str(target_root),
            "sample_root": str(sample_root),
            "run_root": str(run_root),
            "evidence_path": str(evidence),
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(profile, "assert_child_environment", lambda *_args: None)
    monkeypatch.setattr(profile, "install_target_root", lambda *_args: None)
    monkeypatch.setattr(
        profile,
        "assert_target_modules",
        lambda *_args: {
            name: str(target_root / "fixture.py") for name in profile.TARGET_MODULES
        },
    )
    monkeypatch.setattr(
        profile,
        "protocol_preflight",
        lambda *_args, **_kwargs: {
            "event": "protocol_preflight",
            "arm": "enabled",
            "behavior_sha256": "b" * 64,
            "workspace_content_tree_digest": "c" * 64,
            "tool_definition_sha256": "d" * 64,
            "final_ownership": {
                "live_threads": 0,
                "provider_closed": True,
                "sqlite_closed": True,
                "shadow_operations_pending": 0,
            },
        },
        raising=False,
    )

    async def mounted(*_args, **_kwargs):
        pytest.fail("protocol preflight must not mount a conversation")

    monkeypatch.setattr(profile, "run_mounted_sample", mounted)
    args = SimpleNamespace(
        child_spec=spec_path,
        output_root=run_root,
        endpoint="http://127.0.0.1:9099",
        model="fixture.gguf",
    )

    assert profile.run_child_mode(args) == 0
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "child_start",
        "protocol_preflight",
    ]
    assert rows[-1]["sample_id"] == "protocol_preflight-enabled"


def test_child_mode_preserves_async_cancellation_as_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    sample_root = run_root / "sample"
    target_root = tmp_path / "target"
    evidence = run_root / "raw.jsonl"
    sample_root.mkdir(parents=True)
    target_root.mkdir()
    config_path = sample_root / "config" / "tldw_cli" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("", encoding="utf-8")
    spec_path = sample_root / "child-spec.json"
    profile.write_child_spec(
        spec_path,
        {
            "sample_id": "measured-0-enabled",
            "phase": "measured",
            "iteration": 0,
            "arm": "enabled",
            "target_root": str(target_root),
            "sample_root": str(sample_root),
            "run_root": str(run_root),
            "evidence_path": str(evidence),
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(profile, "assert_child_environment", lambda *_args: None)
    monkeypatch.setattr(profile, "install_target_root", lambda *_args: None)
    monkeypatch.setattr(
        profile,
        "assert_target_modules",
        lambda *_args: {name: str(target_root / "fixture.py") for name in profile.TARGET_MODULES},
    )
    monkeypatch.setattr(
        profile.TargetAdapter,
        "for_arm",
        classmethod(
            lambda _cls, _root, _arm: SimpleNamespace(revision_kind="candidate")
        ),
    )

    async def cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(profile, "run_mounted_sample", cancelled)
    args = SimpleNamespace(
        child_spec=spec_path,
        output_root=run_root,
        endpoint="http://127.0.0.1:9099",
        model="fixture.gguf",
    )

    assert profile.run_child_mode(args) == 1
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["child_start", "child_failure"]
    assert rows[-1]["error_type"] == "CancelledError"
    assert rows[-1]["error_code"] == "unclassified"
    assert rows[-1]["error_origin"] == "cancelled"


def test_owned_cleanup_suppresses_child_cancellation_but_not_task_cancellation() -> None:
    await_owned_cleanup = getattr(profile, "await_owned_cleanup", None)

    assert callable(await_owned_cleanup)

    async def child_cancelled() -> None:
        raise asyncio.CancelledError

    asyncio.run(await_owned_cleanup(child_cancelled()))

    async def externally_cancelled() -> None:
        never = asyncio.Event()
        task = asyncio.create_task(await_owned_cleanup(never.wait()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(externally_cancelled())


def test_owned_teardown_cancel_requires_completed_contract_and_live_caller() -> None:
    should_suppress = getattr(
        profile, "should_suppress_owned_teardown_cancel", None
    )

    assert callable(should_suppress)
    assert should_suppress(contract_complete=True, cancellation_count=0) is True
    assert should_suppress(contract_complete=False, cancellation_count=0) is False
    assert should_suppress(contract_complete=True, cancellation_count=1) is False


def test_main_dispatches_nonpreflight_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_spec = tmp_path / "child.json"
    base = [
        "--endpoint",
        "http://127.0.0.1:9099",
        "--model",
        "fixture.gguf",
        "--output-root",
        str(tmp_path),
    ]
    monkeypatch.setattr(profile, "run_child_mode", lambda _args: 7, raising=False)
    monkeypatch.setattr(profile, "run_parent_mode", lambda _args: 9, raising=False)

    assert profile.main([*base, "--child-spec", str(child_spec)]) == 7
    assert profile.main(base) == 9


def test_safe_error_code_retains_only_stable_body_free_tokens() -> None:
    safe_error_code = getattr(profile, "safe_error_code", None)

    assert callable(safe_error_code)
    assert safe_error_code(RuntimeError("mounted_sample_timeout")) == (
        "mounted_sample_timeout"
    )
    assert safe_error_code(
        RuntimeError("benchmark_owned_thread_survivor:change-review-baseline")
    ) == "benchmark_owned_thread_survivor:change-review-baseline"
    assert safe_error_code(RuntimeError("failed at /Users/person/secret")) == (
        "unclassified"
    )


@pytest.mark.parametrize(
    ("name", "revision"),
    (("control", profile.CONTROL_SHA), ("candidate", profile.CANDIDATE_SHA)),
)
def test_prepare_target_worktree_uses_fixed_name_and_detached_exact_hash(
    tmp_path: Path, name: str, revision: str
) -> None:
    prepare_target_worktree = getattr(profile, "prepare_target_worktree", None)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-2]).mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    assert callable(prepare_target_worktree)
    target = prepare_target_worktree(
        tmp_path / "repository",
        tmp_path / "run",
        name=name,
        revision=revision,
        run_command=fake_run,
    )
    assert target == (tmp_path / "run" / name).resolve()
    assert calls[0][0] == [
        "git",
        "worktree",
        "add",
        "--detach",
        str(target),
        revision,
    ]


@pytest.mark.parametrize(
    ("name", "revision"),
    (("../escape", profile.CONTROL_SHA), ("control", "HEAD")),
)
def test_prepare_target_worktree_rejects_unsafe_name_or_revision(
    tmp_path: Path, name: str, revision: str
) -> None:
    prepare_target_worktree = getattr(profile, "prepare_target_worktree", None)

    assert callable(prepare_target_worktree)
    with pytest.raises(RuntimeError, match="target_worktree_invalid"):
        prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name=name,
            revision=revision,
            run_command=lambda *_args, **_kwargs: pytest.fail("must not run git"),
        )


def test_prepare_target_worktree_preserves_command_failure(tmp_path: Path) -> None:
    prepare_target_worktree = getattr(profile, "prepare_target_worktree", None)

    def fake_run(command, **kwargs):
        if command == ["git", "worktree", "list", "--porcelain"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[2] == "remove":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="busy")

    assert callable(prepare_target_worktree)
    with pytest.raises(RuntimeError, match="target_worktree_failed:candidate"):
        prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )


def test_prepare_target_worktree_cleans_partial_add_failure(tmp_path: Path) -> None:
    target = (tmp_path / "run/candidate").resolve()
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[2] == "add":
            target.mkdir(parents=True)
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="busy")
        if command[2] == "remove":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert command == ["git", "worktree", "list", "--porcelain"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="target_worktree_failed:candidate"):
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert calls == [
        ["git", "worktree", "add", "--detach", str(target), profile.CANDIDATE_SHA],
        ["git", "worktree", "remove", "--force", str(target)],
        ["git", "worktree", "list", "--porcelain"],
    ]
    assert not target.exists()


def test_prepare_target_worktree_accepts_nonzero_remove_when_git_proves_absent(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "run/candidate").resolve()

    def fake_run(command, **_kwargs):
        if command[2] == "add":
            target.mkdir(parents=True)
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="add failed")
        if command[2] == "remove":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="already absent")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="^target_worktree_failed:candidate$"):
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert not target.exists()


def test_prepare_target_worktree_cleans_partial_directory_when_add_raises(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "run/candidate").resolve()

    def fake_run(command, **_kwargs):
        if command[2] == "add":
            target.mkdir(parents=True)
            raise RuntimeError("git-add-exploded")
        if command[2] == "remove":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="^git-add-exploded$"):
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert not target.exists()


def test_prepare_target_worktree_preserves_add_when_remove_and_list_raise(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "run/candidate").resolve()
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[2] == "add":
            target.mkdir(parents=True)
            raise RuntimeError("git-add-exploded")
        if command[2] == "remove":
            raise RuntimeError("git-remove-exploded")
        raise OSError("git-list-exploded")

    with pytest.raises(BaseExceptionGroup) as caught:
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert calls == [
        ["git", "worktree", "add", "--detach", str(target), profile.CANDIDATE_SHA],
        ["git", "worktree", "remove", "--force", str(target)],
        ["git", "worktree", "list", "--porcelain"],
    ]
    assert target.is_dir()
    assert [str(error) for error in caught.value.exceptions] == [
        "git-add-exploded",
        "git-remove-exploded",
        "git-list-exploded",
        "target_worktree_unregister_failed:candidate",
    ]


def test_prepare_target_worktree_keeps_registered_partial_add_directory(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "run/candidate").resolve()

    def fake_run(command, **_kwargs):
        if command[2] == "add":
            target.mkdir(parents=True)
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="add failed")
        if command[2] == "remove":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="busy")
        return subprocess.CompletedProcess(
            command, 0, stdout=f"worktree {target}\nHEAD {'1' * 40}\n\n", stderr=""
        )

    with pytest.raises(BaseExceptionGroup) as caught:
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert target.is_dir()
    assert [str(error) for error in caught.value.exceptions] == [
        "target_worktree_failed:candidate",
        "target_worktree_unregister_failed:candidate",
    ]


def test_prepare_target_worktree_keeps_partial_directory_when_registration_unknown(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "run/candidate").resolve()

    def fake_run(command, **_kwargs):
        if command[2] == "add":
            target.mkdir(parents=True)
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="add failed")
        if command[2] == "remove":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="list failed")

    with pytest.raises(BaseExceptionGroup) as caught:
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert target.is_dir()
    assert [str(error) for error in caught.value.exceptions] == [
        "target_worktree_failed:candidate",
        "target_worktree_registration_check_failed",
        "target_worktree_unregister_failed:candidate",
    ]


def test_remove_target_worktree_only_removes_owned_fixed_target(tmp_path: Path) -> None:
    remove_target_worktree = getattr(profile, "_remove_target_worktree", None)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    target.mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    assert callable(remove_target_worktree)
    remove_target_worktree(
        tmp_path / "repository",
        run_root,
        name="candidate",
        run_command=fake_run,
    )
    assert calls[0][0] == [
        "git",
        "worktree",
        "remove",
        "--force",
        str(target.resolve()),
    ]
    with pytest.raises(RuntimeError, match="target_worktree_invalid"):
        remove_target_worktree(
            tmp_path / "repository",
            run_root,
            name="outside",
            run_command=fake_run,
        )


def test_remove_target_worktree_preserves_remove_and_registration_failures(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    (run_root / "candidate").mkdir(parents=True)

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")

    with pytest.raises(BaseExceptionGroup) as caught:
        profile._remove_target_worktree(
            tmp_path / "repository",
            run_root,
            name="candidate",
            run_command=fake_run,
        )

    assert [str(error) for error in caught.value.exceptions] == [
        "target_worktree_remove_failed:candidate",
        "target_worktree_registration_check_failed",
    ]
    assert (run_root / "candidate").is_dir()


def test_remove_target_worktree_cleans_registered_missing_directory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    target = (run_root / "candidate").resolve()
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    profile._remove_target_worktree(
        tmp_path / "repository",
        run_root,
        name="candidate",
        run_command=fake_run,
    )

    assert calls == [["git", "worktree", "remove", "--force", str(target)]]


def test_remove_target_worktrees_attempts_both_when_first_cleanup_fails(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    for name in ("control", "candidate"):
        (run_root / name).mkdir(parents=True)
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command == ["git", "worktree", "list", "--porcelain"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"worktree {(run_root / 'candidate').resolve()}\n\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            1 if command[-1].endswith("candidate") else 0,
            stdout="",
            stderr="busy",
        )

    with pytest.raises(RuntimeError, match="target_worktree_unregister_failed:candidate"):
        profile._remove_target_worktrees(
            tmp_path / "repository",
            run_root,
            names=("candidate", "control"),
            run_command=fake_run,
        )

    removed_names = [Path(command[-1]).name for command in calls if command[2] == "remove"]
    assert removed_names == ["candidate", "control"]


def _write_fingerprint_tree(root: Path, *, candidate: bool) -> None:
    tracker = root / "tldw_chatbook/Workspaces/change_turn_tracker.py"
    tracker.parent.mkdir(parents=True, exist_ok=True)
    tracker.write_text(
        "class ChangeTurnTracker:\n"
        "    def begin_turn(self): pass\n"
        "    def populate_baseline(self): pass\n"
        "    def end_turn(self): pass\n"
        + ("    def finish_turn(self): pass\n" if candidate else "")
    )
    bridge = root / "tldw_chatbook/Chat/console_agent_bridge.py"
    bridge.parent.mkdir(parents=True, exist_ok=True)
    if candidate:
        bridge.write_text(
            "self._change_finalization_coordinator.register(roots)\n"
            "self._change_finalization_coordinator.finalize(reservation)\n"
        )
        (tracker.parent / "change_review_consent.py").write_text(
            "class ChangeReviewConsentService: pass\n"
        )
        (tracker.parent / "change_review_finalization.py").write_text(
            "class ChangeReviewFinalizationCoordinator:\n"
            "    def finalize(self): pass\n"
            "    def _worker_loop(self): pass\n"
        )
    else:
        bridge.write_text(
            "self._change_tracker.begin_turn(roots)\n"
            "self._change_tracker.end_turn(handle)\n"
        )


def test_target_adapter_accepts_only_the_expected_revision_shape(tmp_path: Path) -> None:
    target_adapter_type = getattr(profile, "TargetAdapter", None)
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    _write_fingerprint_tree(control, candidate=False)
    _write_fingerprint_tree(candidate, candidate=True)

    assert target_adapter_type is not None
    assert target_adapter_type.for_arm(control, "control").revision_kind == "legacy"
    assert target_adapter_type.for_arm(candidate, "disabled").revision_kind == "candidate"
    assert target_adapter_type.for_arm(candidate, "enabled").revision_kind == "candidate"
    with pytest.raises(RuntimeError, match="target_fingerprint_mismatch"):
        target_adapter_type.for_arm(candidate, "control")
    with pytest.raises(RuntimeError, match="target_fingerprint_mismatch"):
        target_adapter_type.for_arm(control, "enabled")

    candidate_finalization = (
        candidate
        / "tldw_chatbook/Workspaces/change_review_finalization.py"
    )
    candidate_finalization.write_text(
        "class ChangeReviewFinalizationCoordinator:\n"
        "    def finalize(self): pass\n"
    )
    with pytest.raises(RuntimeError, match="target_fingerprint_mismatch"):
        target_adapter_type.for_arm(candidate, "enabled")


def test_target_adapter_wraps_and_restores_legacy_review_boundaries(
    tmp_path: Path,
) -> None:
    class Handle:
        def await_baseline(self):
            return None

    class Tracker:
        def begin_turn(self):
            return Handle()

        def end_turn(self, _handle):
            return []

    originals = (Tracker.begin_turn, Tracker.end_turn)
    adapter = profile.TargetAdapter(tmp_path, "control", "legacy")

    adapter.install_timing_wrappers(tracker_type=Tracker)
    tracker = Tracker()
    handle = tracker.begin_turn()
    tracker.end_turn(handle)

    events = adapter.review_events()
    assert set(events) == {
        "baseline_started",
        "baseline_ready",
        "review_e_started",
        "review_e_completed",
    }
    assert events["baseline_started"] <= events["baseline_ready"]
    assert events["review_e_started"] <= events["review_e_completed"]

    adapter.close()
    assert (Tracker.begin_turn, Tracker.end_turn) == originals


def test_target_adapter_candidate_schedule_precedes_worker_review_e(
    tmp_path: Path,
) -> None:
    class Tracker:
        def finish_turn(self):
            return []

    class Coordinator:
        def __init__(self, tracker):
            self.tracker = tracker

        def register(self):
            return object()

        def await_baseline(self):
            return None

        def finalize(self):
            # Exercise the real race shape: scheduling may let the worker
            # enter E before finalize itself returns.
            self.tracker.finish_turn()
            return "scheduled"

    original_finalize = Coordinator.finalize
    adapter = profile.TargetAdapter(tmp_path, "enabled", "candidate")

    adapter.install_timing_wrappers(
        tracker_type=Tracker,
        coordinator_type=Coordinator,
    )
    coordinator = Coordinator(Tracker())
    coordinator.register()
    coordinator.await_baseline()
    coordinator.finalize()

    events = adapter.review_events()
    assert set(events) == set(profile.ARM_CONTRACTS["enabled"].required_review)
    assert events["finalization_scheduled"] <= events["review_e_started"]
    assert events["review_e_started"] <= events["review_e_completed"]

    adapter.close()
    assert Coordinator.finalize is original_finalize


@pytest.mark.parametrize(
    ("arm", "kind", "runtime", "expected_service"),
    (
        (
            "control",
            "legacy",
            SimpleNamespace(
                consent_service=None,
                review_state="legacy",
                review_ready=True,
            ),
            None,
        ),
        (
            "disabled",
            "candidate",
            SimpleNamespace(
                consent_service="disabled-service",
                review_state="disabled",
                review_ready=False,
            ),
            "disabled-service",
        ),
        (
            "enabled",
            "candidate",
            SimpleNamespace(
                consent_service="enabled-service",
                review_state="enabled",
                review_ready=True,
            ),
            "enabled-service",
        ),
    ),
)
def test_target_adapter_configures_only_its_matching_review_arm(
    tmp_path: Path,
    arm: str,
    kind: str,
    runtime: SimpleNamespace,
    expected_service: object,
) -> None:
    app = SimpleNamespace(change_review_consent_service="old")
    adapter = profile.TargetAdapter(tmp_path, arm, kind)

    adapter.configure_review(app, runtime)

    assert app.change_review_consent_service == expected_service


def test_generate_corpus_is_deterministic_and_uses_content_digest(tmp_path: Path) -> None:
    generate_corpus = getattr(profile, "generate_corpus", None)
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert callable(generate_corpus)
    first_result = generate_corpus(first, file_count=4, file_size=32, blob_size=128)
    second_result = generate_corpus(second, file_count=4, file_size=32, blob_size=128)
    assert first_result == second_result
    assert len(first_result["files"]) == 5
    assert (first / "measured").is_dir()
    assert first_result["content_tree_digest"] == profile.content_tree_digest(first)
    (second / "corpus/0001.bin").write_bytes(b"changed")
    assert first_result["content_tree_digest"] != profile.content_tree_digest(second)


def test_workspace_runtime_close_attempts_and_preserves_all_owned_cleanup() -> None:
    closed: list[str] = []

    def fail(label: str):
        def action(*_args, **_kwargs):
            closed.append(label)
            raise RuntimeError(f"{label}-failed")

        return action

    runtime = profile.WorkspaceRuntime(
        workspace_id="fixture",
        workspace_root=Path("workspace"),
        shadow_root=Path("shadow"),
        database=SimpleNamespace(close=fail("database-close")),
        registry=None,
        binding=None,
        consent_service=SimpleNamespace(shutdown=fail("consent-shutdown")),
        review_state="enabled",
        review_ready=True,
        control_plane=None,
        local_provider=None,
        hub=None,
        gate=None,
        permission_definition_hash="d" * 64,
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        runtime.close()

    assert closed == ["consent-shutdown", "database-close"]
    assert [str(error) for error in caught.value.exceptions] == [
        "consent-shutdown-failed",
        "database-close-failed",
    ]


@pytest.mark.parametrize("boundary", ("readiness", "permission_gate", "definition_hash"))
def test_prepare_workspace_runtime_cleans_resources_on_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.MCP import permission_store
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewConsentService,
        RootReadinessState,
    )

    databases: list[object] = []
    consents: list[object] = []
    closed: list[str] = []
    database_init = WorkspaceDB.__init__
    database_close = WorkspaceDB.close
    consent_init = ChangeReviewConsentService.__init__
    consent_shutdown = ChangeReviewConsentService.shutdown

    def track_database(self, *args, **kwargs):
        database_init(self, *args, **kwargs)
        databases.append(self)

    def close_database(self, *args, **kwargs):
        closed.append("database")
        return database_close(self, *args, **kwargs)

    def track_consent(self, *args, **kwargs):
        consent_init(self, *args, **kwargs)
        consents.append(self)

    def close_consent(self, *args, **kwargs):
        closed.append("consent")
        return consent_shutdown(self, *args, **kwargs)

    monkeypatch.setattr(WorkspaceDB, "__init__", track_database)
    monkeypatch.setattr(WorkspaceDB, "close", close_database)
    monkeypatch.setattr(ChangeReviewConsentService, "__init__", track_consent)
    monkeypatch.setattr(ChangeReviewConsentService, "shutdown", close_consent)
    arm = "enabled" if boundary == "readiness" else "disabled"
    expected = "change_review_initialization_failed"
    if boundary == "readiness":
        monkeypatch.setattr(
            ChangeReviewConsentService,
            "status",
            lambda *_args: SimpleNamespace(
                roots=[SimpleNamespace(state=RootReadinessState.FAILED)]
            ),
        )
    elif boundary == "permission_gate":
        expected = "permission-gate-failed"
        monkeypatch.setattr(
            UnifiedMCPControlPlaneService,
            "set_tool_state",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(expected)),
        )
    else:
        expected = "definition-hash-failed"
        monkeypatch.setattr(
            permission_store,
            "definition_hash",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(expected)),
        )

    try:
        with pytest.raises(RuntimeError, match=expected):
            profile.prepare_workspace_runtime(tmp_path / boundary, arm=arm)
        assert closed == ["consent", "database"]
    finally:
        if consents and "consent" not in closed:
            consent_shutdown(consents[0], timeout=2.0)
        if databases and "database" not in closed:
            database_close(databases[0])


def test_prepare_workspace_runtime_aggregates_primary_and_construction_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewConsentService,
    )

    databases: list[object] = []
    consents: list[object] = []
    database_init = WorkspaceDB.__init__
    database_close = WorkspaceDB.close
    consent_init = ChangeReviewConsentService.__init__
    consent_shutdown = ChangeReviewConsentService.shutdown

    def track_database(self, *args, **kwargs):
        database_init(self, *args, **kwargs)
        databases.append(self)

    def track_consent(self, *args, **kwargs):
        consent_init(self, *args, **kwargs)
        consents.append(self)

    def fail_database(self, *args, **kwargs):
        database_close(self, *args, **kwargs)
        raise RuntimeError("database-cleanup-failed")

    def fail_consent(self, *args, **kwargs):
        consent_shutdown(self, *args, **kwargs)
        raise RuntimeError("consent-cleanup-failed")

    monkeypatch.setattr(WorkspaceDB, "__init__", track_database)
    monkeypatch.setattr(WorkspaceDB, "close", fail_database)
    monkeypatch.setattr(ChangeReviewConsentService, "__init__", track_consent)
    monkeypatch.setattr(ChangeReviewConsentService, "shutdown", fail_consent)
    monkeypatch.setattr(
        UnifiedMCPControlPlaneService,
        "set_tool_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("permission-gate-failed")
        ),
    )

    try:
        with pytest.raises(BaseExceptionGroup) as caught:
            profile.prepare_workspace_runtime(tmp_path / "dual", arm="disabled")
        assert [str(error) for error in caught.value.exceptions] == [
            "permission-gate-failed",
            "consent-cleanup-failed",
            "database-cleanup-failed",
        ]
    finally:
        # The injected failures occur after the real close operations.
        if consents and any(
            worker.is_alive() for worker in consents[0]._workers
        ):
            consent_shutdown(consents[0], timeout=2.0)


def test_prepare_workspace_runtime_disabled_has_rw_allow_without_shadow(
    tmp_path: Path,
) -> None:
    prepare_workspace_runtime = getattr(profile, "prepare_workspace_runtime", None)

    assert callable(prepare_workspace_runtime)
    runtime = prepare_workspace_runtime(tmp_path / "disabled", arm="disabled")
    try:
        assert runtime.registry.get_active_workspace().workspace_id == runtime.workspace_id
        assert runtime.binding.metadata["access"] == "rw"
        assert runtime.review_state == "disabled"
        assert runtime.review_ready is False
        assert runtime.gate.state == "allow"
        assert runtime.gate.origin == "tool_override"
        assert runtime.hub.name == "fs_write"
        assert runtime.permission_definition_hash
        assert not runtime.shadow_root.exists()
    finally:
        runtime.close()


def test_prepare_workspace_runtime_enabled_waits_for_real_ready_snapshot(
    tmp_path: Path,
) -> None:
    prepare_workspace_runtime = getattr(profile, "prepare_workspace_runtime", None)

    assert callable(prepare_workspace_runtime)
    runtime = prepare_workspace_runtime(tmp_path / "enabled", arm="enabled")
    try:
        assert runtime.review_state == "enabled"
        assert runtime.review_ready is True
        assert runtime.consent_service.admit_turn(runtime.workspace_id).ready_roots == (
            str(runtime.workspace_root.resolve()),
        )
        assert any(runtime.shadow_root.rglob("HEAD"))
        assert runtime.gate.state == "allow"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write(
    tmp_path: Path,
) -> None:
    run_scripted_mounted_sample = getattr(profile, "run_scripted_mounted_sample", None)

    assert callable(run_scripted_mounted_sample)
    result = await run_scripted_mounted_sample(tmp_path / "mounted", arm="disabled")

    assert result["provider_round_counts"] == {"1": 1, "2": 3, "3": 1}
    assert result["tool_calls"] == ["load_tools", "fs_write"]
    assert result["third_send_requested_ns"] < result["turn_2_release_ns"]
    assert result["third_provider_started_ns"] is not None
    assert result["terminal_third_assistant"] == "turn-three-complete"
    assert (tmp_path / "mounted/workspace/measured/turn-two.txt").read_bytes() == (
        profile.FIXED_MUTATION
    )
