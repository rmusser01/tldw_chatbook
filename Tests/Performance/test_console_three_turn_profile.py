"""Regression tests for the real-provider three-turn Console benchmark."""

from __future__ import annotations

import asyncio
import copy
import io
import json
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


def test_prepare_control_worktree_uses_detached_exact_hash(tmp_path: Path) -> None:
    prepare_control_worktree = getattr(profile, "prepare_control_worktree", None)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-2]).mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    assert callable(prepare_control_worktree)
    target = prepare_control_worktree(
        tmp_path / "repository",
        tmp_path / "run",
        control_sha=profile.CONTROL_SHA,
        run_command=fake_run,
    )
    assert target == (tmp_path / "run" / "control-worktree").resolve()
    assert calls[0][0] == [
        "git",
        "worktree",
        "add",
        "--detach",
        str(target),
        profile.CONTROL_SHA,
    ]


def test_prepare_control_worktree_preserves_command_failure(tmp_path: Path) -> None:
    prepare_control_worktree = getattr(profile, "prepare_control_worktree", None)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="busy")

    assert callable(prepare_control_worktree)
    with pytest.raises(RuntimeError, match="control_worktree_failed"):
        prepare_control_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            control_sha=profile.CONTROL_SHA,
            run_command=fake_run,
        )


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
