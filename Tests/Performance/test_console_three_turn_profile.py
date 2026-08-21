"""Regression tests for the real-provider three-turn Console benchmark."""

from __future__ import annotations

import copy
import io
import json

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
    return [
        _valid_sample(arm, iteration=iteration)
        for iteration in range(iterations)
        for arm in profile.ARMS
    ]


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
