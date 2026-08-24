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
import shutil
import stat
import subprocess
import sys
import textwrap
import tomllib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest

from Tests.Performance import run_console_three_turn_profile as profile


REVIEWED_ARTIFACT_NAMES = {
    "README.md",
    "real-provider-three-turn.raw.jsonl",
    "real-provider-three-turn.manifest.json",
    "real-provider-three-turn.summary.json",
    "real-provider-three-turn-summary.md",
}
TASK5_PUBLISHED_NAMES = REVIEWED_ARTIFACT_NAMES | {
    "confirmatory-review-receipt.json"
}


def _write_immutable_foreign_publication(
    root: Path, *, root_mode: int = 0o711
) -> dict[str, tuple[bytes, int, int]]:
    root.mkdir()
    state: dict[str, tuple[bytes, int, int]] = {}
    for name in TASK5_PUBLISHED_NAMES:
        path = root / name
        path.write_bytes(f"foreign:{name}".encode())
        path.chmod(0o640)
        os.chflags(path, path.stat().st_flags | stat.UF_IMMUTABLE)
        metadata = path.stat()
        state[name] = (
            path.read_bytes(),
            stat.S_IMODE(metadata.st_mode),
            metadata.st_flags,
        )
    root.chmod(root_mode)
    return state


def _publication_state(root: Path) -> dict[str, tuple[bytes, int, int]]:
    return {
        name: (
            (root / name).read_bytes(),
            stat.S_IMODE((root / name).stat().st_mode),
            (root / name).stat().st_flags,
        )
        for name in TASK5_PUBLISHED_NAMES
    }


@pytest.fixture(autouse=True)
def _unseal_task5_test_directories(request: pytest.FixtureRequest):
    """Restore test-only temp permissions so pytest can remove sealed evidence."""
    yield
    tmp_path = request.node.funcargs.get("tmp_path")
    if not isinstance(tmp_path, Path) or not tmp_path.is_dir():
        return
    for directory, child_directories, files in os.walk(tmp_path):
        root = Path(directory)
        if not root.is_symlink():
            root.chmod(0o755)
        if sys.platform == "darwin":
            for name in files:
                path = root / name
                if path.is_file() and not path.is_symlink():
                    os.chflags(path, path.stat().st_flags & ~stat.UF_IMMUTABLE)
        child_directories[:] = [
            name for name in child_directories if not (root / name).is_symlink()
        ]


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


def _attempt_event(attempt_id: str, state: str, **extra: object) -> dict[str, object]:
    return {"attempt_id": attempt_id, "state": state, **extra}


def test_campaign_attempt_ids_are_sequential_and_staging_roots_are_fresh(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "attempts.jsonl"

    assert profile.next_attempt_id(()) == "attempt-0001"
    first_root = profile.create_attempt_root(tmp_path, "attempt-0001")
    assert first_root == tmp_path / "attempts" / "attempt-0001"
    assert first_root.is_dir()
    with pytest.raises(RuntimeError, match="campaign_attempt_root_exists"):
        profile.create_attempt_root(tmp_path, "attempt-0001")

    profile.append_attempt_state(
        ledger, _attempt_event("attempt-0001", "running")
    )
    profile.append_attempt_state(
        ledger,
        _attempt_event(
            "attempt-0001", "failed", reason_category="acquisition"
        ),
    )
    assert profile.next_attempt_id(profile.attempt_lineage(ledger)) == "attempt-0002"
    second_root = profile.create_attempt_root(tmp_path, "attempt-0002")
    assert second_root.is_dir()
    assert second_root != first_root


def test_campaign_ledger_appends_sorted_newline_json_and_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    modes: list[str] = []
    synced: list[int] = []
    real_open = Path.open

    def recording_open(path: Path, mode: str = "r", *args, **kwargs):
        if path == ledger:
            modes.append(mode)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    monkeypatch.setattr(profile.os, "fsync", synced.append)
    event = _attempt_event("attempt-0001", "running")

    profile.append_attempt_state(ledger, event)

    assert modes == ["a"]
    assert synced
    assert ledger.read_text(encoding="utf-8") == json.dumps(
        event, sort_keys=True
    ) + "\n"


def test_campaign_directory_fsync_helper_uses_directory_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        profile.os,
        "open",
        lambda path, flags: calls.append(("open", (path, flags))) or 91,
    )
    monkeypatch.setattr(
        profile.os, "fsync", lambda descriptor: calls.append(("fsync", descriptor))
    )
    monkeypatch.setattr(
        profile.os, "close", lambda descriptor: calls.append(("close", descriptor))
    )

    profile._fsync_directory(tmp_path)

    assert calls[0][0] == "open"
    assert calls[0][1][0] == tmp_path
    assert calls[1:] == [("fsync", 91), ("close", 91)]


def test_campaign_state_bearing_namespace_mutations_fsync_parents_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(
        profile, "_fsync_directory", lambda path: synced.append(Path(path)), raising=False
    )
    campaign = tmp_path / "campaign"

    attempt_root = profile.create_attempt_root(campaign, "attempt-0001")
    owner = _acquire_lock(campaign)
    profile.append_attempt_state(
        campaign / "attempts.jsonl",
        _attempt_event("attempt-0001", "running"),
    )
    profile.release_campaign_lock(campaign, owner)

    assert synced[:4] == [
        tmp_path,
        campaign,
        campaign / "attempts",
        campaign,
    ]
    owner_stage = synced[4]
    assert owner_stage.parent == campaign
    assert owner_stage.name.startswith(".campaign-owner-")
    assert synced[5:] == [
        campaign,
        campaign,
        campaign,
        campaign / ".campaign-release",
        campaign,
    ]
    assert attempt_root.is_dir()


def test_campaign_ledger_never_overwrites_or_truncates(tmp_path: Path) -> None:
    ledger = tmp_path / "attempts.jsonl"
    running = _attempt_event("attempt-0001", "running")
    failed = _attempt_event(
        "attempt-0001", "failed", reason_category="provider"
    )
    profile.append_attempt_state(ledger, running)
    original = ledger.read_bytes()

    profile.append_attempt_state(ledger, failed)

    assert ledger.read_bytes().startswith(original)
    assert ledger.read_text(encoding="utf-8").splitlines() == [
        json.dumps(running, sort_keys=True),
        json.dumps(failed, sort_keys=True),
    ]


def test_campaign_attempt_states_are_exact() -> None:
    assert profile.ATTEMPT_STATES == frozenset(
        {
            "running",
            "failed",
            "invalid",
            "complete_pending_review",
            "changes_required",
        }
    )
    assert profile.BLOCKING_ATTEMPT_STATES == frozenset(
        {"running", "complete_pending_review", "changes_required"}
    )


@pytest.mark.parametrize(
    "state",
    ("running", "complete_pending_review", "changes_required"),
)
def test_campaign_acquisition_is_blocked_by_current_attempt_state(
    tmp_path: Path, state: str
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    profile.append_attempt_state(
        ledger, _attempt_event("attempt-0001", "running")
    )
    if state == "complete_pending_review":
        profile.complete_attempt_measurement(
            ledger,
            "attempt-0001",
            verdict="pass",
            raw_sha256="1" * 64,
        )
    elif state == "changes_required":
        profile.complete_attempt_measurement(
            ledger,
            "attempt-0001",
            verdict="inconclusive",
            raw_sha256="1" * 64,
        )
        profile.append_attempt_state(
            ledger,
            _attempt_event(
                "attempt-0001",
                "changes_required",
                verdict="inconclusive",
                raw_sha256="1" * 64,
                reason_category="summary",
            ),
        )

    with pytest.raises(RuntimeError, match=f"campaign_acquisition_blocked:{state}"):
        profile.require_campaign_acquisition(ledger)


@pytest.mark.parametrize(
    ("terminal_state", "reason_category"),
    (
        ("failed", "provider"),
        ("failed", "acquisition"),
        ("failed", "interrupted"),
        ("invalid", "raw"),
        ("invalid", "product"),
        ("invalid", "completeness"),
        ("invalid", "isolation"),
        ("invalid", "privacy"),
        ("invalid", "ownership"),
    ),
)
def test_campaign_retry_is_allowed_only_after_uncorrectable_terminal_state(
    tmp_path: Path, terminal_state: str, reason_category: str
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    profile.append_attempt_state(
        ledger, _attempt_event("attempt-0001", "running")
    )
    profile.append_attempt_state(
        ledger,
        _attempt_event(
            "attempt-0001", terminal_state, reason_category=reason_category
        ),
    )

    assert profile.require_campaign_acquisition(ledger) == "attempt-0002"


@pytest.mark.parametrize("verdict", ("pass", "regression", "inconclusive"))
def test_campaign_measured_verdict_always_enters_pending_review(
    tmp_path: Path, verdict: str
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    profile.append_attempt_state(
        ledger, _attempt_event("attempt-0001", "running")
    )

    event = profile.complete_attempt_measurement(
        ledger,
        "attempt-0001",
        verdict=verdict,
        raw_sha256="a" * 64,
    )

    assert event == {
        "attempt_id": "attempt-0001",
        "state": "complete_pending_review",
        "verdict": verdict,
        "raw_sha256": "a" * 64,
    }
    with pytest.raises(RuntimeError, match="campaign_acquisition_blocked"):
        profile.require_campaign_acquisition(ledger)


def test_disposable_smoke_uses_explicit_nonstatistical_verdict(tmp_path: Path) -> None:
    ledger = tmp_path / "attempts.jsonl"
    profile.append_attempt_state(
        ledger, _attempt_event("attempt-0001", "running")
    )

    event = profile.complete_attempt_measurement(
        ledger,
        "attempt-0001",
        verdict="smoke",
        raw_sha256="a" * 64,
    )

    assert event["verdict"] == "smoke"
    with pytest.raises(RuntimeError, match="campaign_acquisition_blocked"):
        profile.require_campaign_acquisition(ledger)
    with pytest.raises(RuntimeError, match="^campaign_verdict_mismatch$"):
        profile.append_attempt_state(
            ledger,
            _attempt_event(
                "attempt-0001",
                "changes_required",
                verdict="pass",
                raw_sha256="a" * 64,
                reason_category="report",
            ),
        )


@pytest.mark.parametrize(
    ("iterations", "burn_in_blocks", "verdict"),
    ((1, 1, "pass"), (30, 5, "smoke")),
)
def test_acquisition_rejects_verdict_from_the_other_schedule_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    iterations: int,
    burn_in_blocks: int,
    verdict: str,
) -> None:
    campaign = tmp_path / "campaign"
    attempt = _acquire_attempt(campaign)
    monkeypatch.setattr(profile, "acquire_campaign_attempt", lambda _root: attempt)

    def parent(args) -> int:
        args.acquisition_pin_callback(verdict=verdict, raw_sha256="a" * 64)
        return 0

    monkeypatch.setattr(profile, "run_parent_mode", parent)

    with pytest.raises(RuntimeError, match="^campaign_schedule_verdict_mismatch$"):
        profile.run_acquisition_action(
            SimpleNamespace(
                campaign_root=campaign,
                iterations=iterations,
                burn_in_blocks=burn_in_blocks,
            )
        )

    assert profile.read_acquisition_pin(campaign, attempt.attempt_id) is None
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == {
        "attempt_id": attempt.attempt_id,
        "state": "invalid",
        "reason_category": "product",
    }
    assert not (campaign / ".campaign-lock").exists()


def test_campaign_correctable_derived_changes_preserve_raw_hash_and_lineage(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    raw_sha256 = "b" * 64
    profile.append_attempt_state(
        ledger, _attempt_event("attempt-0001", "running")
    )
    profile.complete_attempt_measurement(
        ledger,
        "attempt-0001",
        verdict="inconclusive",
        raw_sha256=raw_sha256,
    )
    profile.append_attempt_state(
        ledger,
        _attempt_event(
            "attempt-0001",
            "changes_required",
            verdict="inconclusive",
            raw_sha256=raw_sha256,
            reason_category="report",
        ),
    )

    with pytest.raises(RuntimeError, match="campaign_raw_hash_mismatch"):
        profile.complete_attempt_measurement(
            ledger,
            "attempt-0001",
            verdict="inconclusive",
            raw_sha256="c" * 64,
        )
    corrected = profile.complete_attempt_measurement(
        ledger,
        "attempt-0001",
        verdict="inconclusive",
        raw_sha256=raw_sha256,
    )

    assert corrected["state"] == "complete_pending_review"
    assert [event["state"] for event in profile.attempt_lineage(ledger)] == [
        "running",
        "complete_pending_review",
        "changes_required",
        "complete_pending_review",
    ]
    assert {
        event["raw_sha256"]
        for event in profile.attempt_lineage(ledger)
        if "raw_sha256" in event
    } == {raw_sha256}


def test_campaign_changes_required_preserves_first_measured_verdict(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    profile.append_attempt_state(
        ledger, _attempt_event("attempt-0001", "running")
    )
    profile.complete_attempt_measurement(
        ledger, "attempt-0001", verdict="pass", raw_sha256="a" * 64
    )

    with pytest.raises(RuntimeError, match="^campaign_verdict_mismatch$"):
        profile.append_attempt_state(
            ledger,
            _attempt_event(
                "attempt-0001",
                "changes_required",
                verdict="regression",
                raw_sha256="a" * 64,
                reason_category="report",
            ),
        )


def test_campaign_renewed_pending_review_preserves_first_measured_verdict(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    profile.append_attempt_state(
        ledger, _attempt_event("attempt-0001", "running")
    )
    profile.complete_attempt_measurement(
        ledger, "attempt-0001", verdict="pass", raw_sha256="a" * 64
    )
    profile.append_attempt_state(
        ledger,
        _attempt_event(
            "attempt-0001",
            "changes_required",
            verdict="pass",
            raw_sha256="a" * 64,
            reason_category="report",
        ),
    )

    with pytest.raises(RuntimeError, match="^campaign_verdict_mismatch$"):
        profile.complete_attempt_measurement(
            ledger,
            "attempt-0001",
            verdict="inconclusive",
            raw_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("event", "code"),
    [
        (
            _attempt_event("attempt-0001", "running", extra="field"),
            "campaign_event_fields_invalid",
        ),
        (
            _attempt_event("attempt-0001", "complete"),
            "campaign_attempt_state_invalid",
        ),
        (
            _attempt_event(
                "attempt-0001", "failed", reason_category="network_flake"
            ),
            "campaign_reason_category_invalid",
        ),
        (
            _attempt_event("attempt-1", "running"),
            "campaign_attempt_id_invalid",
        ),
        (
            _attempt_event(
                "attempt-0001",
                "complete_pending_review",
                verdict="pass",
                raw_sha256="not-a-hash",
            ),
            "campaign_raw_hash_invalid",
        ),
        (
            _attempt_event("attempt-0001", "running", prompt="secret"),
            "campaign_event_privacy_violation",
        ),
        (
            _attempt_event("attempt-0001", "running", path="/tmp/raw.jsonl"),
            "campaign_event_privacy_violation",
        ),
    ],
)
def test_campaign_attempt_event_schema_fails_closed(
    tmp_path: Path, event: dict[str, object], code: str
) -> None:
    with pytest.raises(RuntimeError, match=code):
        profile.append_attempt_state(tmp_path / "attempts.jsonl", event)


@pytest.mark.parametrize("value", (None, False, 7, [], {}))
@pytest.mark.parametrize(
    ("field", "event", "code"),
    (
        (
            "state",
            {"attempt_id": "attempt-0001", "state": "running"},
            "campaign_attempt_state_invalid",
        ),
        (
            "reason_category",
            {
                "attempt_id": "attempt-0001",
                "state": "failed",
                "reason_category": "provider",
            },
            "campaign_reason_category_invalid",
        ),
        (
            "verdict",
            {
                "attempt_id": "attempt-0001",
                "state": "complete_pending_review",
                "verdict": "pass",
                "raw_sha256": "a" * 64,
            },
            "campaign_verdict_invalid",
        ),
    ),
)
def test_campaign_enum_fields_reject_non_strings_stably(
    tmp_path: Path,
    value: object,
    field: str,
    event: dict[str, object],
    code: str,
) -> None:
    malformed = {**event, field: value}
    transition_ledger = tmp_path / "transition.jsonl"
    profile.append_attempt_state(
        transition_ledger, _attempt_event("attempt-0001", "running")
    )

    with pytest.raises(RuntimeError, match=f"^{code}$"):
        profile.append_attempt_state(transition_ledger, malformed)

    payload = json.dumps(malformed, sort_keys=True) + "\n"
    loaded_ledger = tmp_path / "loaded.jsonl"
    loaded_ledger.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match=f"^{code}$"):
        profile.attempt_lineage(loaded_ledger)

    admission_ledger = tmp_path / "admission.jsonl"
    admission_ledger.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match=f"^{code}$"):
        profile.require_campaign_acquisition(admission_ledger)


def test_campaign_attempt_lineage_rejects_malformed_and_out_of_order_records(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    ledger.write_text('{"state":"running"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="campaign_event_fields_invalid"):
        profile.attempt_lineage(ledger)

    ledger.write_text(
        json.dumps(_attempt_event("attempt-0002", "running"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="campaign_attempt_sequence_invalid"):
        profile.attempt_lineage(ledger)

    ledger.write_text(
        json.dumps(_attempt_event("attempt-0001", "running"), sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="campaign_ledger_malformed"):
        profile.attempt_lineage(ledger)

    ledger.write_bytes(b"\xff\n")
    with pytest.raises(RuntimeError, match="campaign_ledger_malformed"):
        profile.attempt_lineage(ledger)


_OWNER_START = "d" * 64
_OWNER_TOKEN = "e" * 64


def _acquire_lock(campaign_root: Path, *, pid: int = 123):
    return profile.acquire_campaign_lock(
        campaign_root,
        pid=pid,
        process_start_probe=lambda observed_pid: (
            _OWNER_START if observed_pid == pid else None
        ),
        owner_token_factory=lambda: _OWNER_TOKEN,
    )


def _acquire_attempt(campaign_root: Path, *, pid: int = 123):
    return profile.acquire_campaign_attempt(
        campaign_root,
        pid=pid,
        process_start_probe=lambda observed_pid: (
            _OWNER_START if observed_pid == pid else None
        ),
        owner_token_factory=lambda: _OWNER_TOKEN,
    )


def test_campaign_lock_uses_atomic_staged_directory_and_exact_private_owner_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_mkdir_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    published: list[tuple[str, profile.CampaignLockOwner]] = []
    real_mkdir = Path.mkdir
    real_rename = profile._rename_namespace

    def recording_mkdir(path: Path, *args, **kwargs):
        if path.name == ".campaign-lock":
            lock_mkdir_calls.append((args, kwargs))
        return real_mkdir(path, *args, **kwargs)

    def recording_rename(source: Path, target: Path) -> None:
        if target == tmp_path / ".campaign-lock":
            published.append((source.name, profile._read_lock_owner(source)))
        real_rename(source, target)

    monkeypatch.setattr(Path, "mkdir", recording_mkdir)
    monkeypatch.setattr(profile, "_rename_namespace", recording_rename)

    owner = _acquire_lock(tmp_path)

    assert lock_mkdir_calls == []
    assert len(published) == 1
    assert published[0][0].startswith(".campaign-owner-")
    assert published[0][1] == owner
    assert owner.pid == 123
    assert json.loads((tmp_path / ".campaign-lock" / "owner.json").read_bytes()) == {
        "owner_token": _OWNER_TOKEN,
        "pid": 123,
        "process_start_sha256": _OWNER_START,
    }
    owner_bytes = (tmp_path / ".campaign-lock" / "owner.json").read_text(
        encoding="utf-8"
    )
    assert all(
        forbidden not in owner_bytes
        for forbidden in ("api_key", "secret", "command", "environment", "environ")
    )


def test_campaign_owner_stage_rename_cannot_replace_complete_lock(
    tmp_path: Path,
) -> None:
    staged_owner = profile.CampaignLockOwner(111, _OWNER_START, "a" * 64)
    canonical_owner = profile.CampaignLockOwner(222, _OWNER_START, "b" * 64)
    stage = tmp_path / ".campaign-owner-proof"
    canonical = tmp_path / ".campaign-lock"
    _write_campaign_owner(stage, staged_owner)
    _write_campaign_owner(canonical, canonical_owner)

    with pytest.raises(OSError):
        profile._rename_namespace(stage, canonical)

    assert profile._read_lock_owner(stage) == staged_owner
    assert profile._read_lock_owner(canonical) == canonical_owner


def test_campaign_lock_owner_read_namespace_race_has_stable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _acquire_lock(tmp_path)
    canonical = tmp_path / ".campaign-lock"
    recovery = tmp_path / ".campaign-recovery"
    real_iterdir = Path.iterdir

    def move_before_iterdir(path: Path):
        if path == canonical:
            path.rename(recovery)
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", move_before_iterdir)

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_invalid$"):
        profile._read_lock_owner(canonical)


def test_campaign_owner_publish_rename_failure_never_exposes_canonical_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_rename = profile._rename_namespace

    def fail_publish(source: Path, target: Path) -> None:
        if target.name == ".campaign-lock":
            raise OSError("injected owner publication failure")
        real_rename(source, target)

    monkeypatch.setattr(profile, "_rename_namespace", fail_publish)

    with pytest.raises(RuntimeError, match="^campaign_lock_publish_failed$"):
        _acquire_lock(tmp_path)

    assert not (tmp_path / ".campaign-lock").exists()
    assert not tuple(tmp_path.glob(".campaign-owner-*"))


def test_campaign_owner_publish_preserves_unrelated_partial_temp(
    tmp_path: Path,
) -> None:
    unrelated = tmp_path / ".campaign-owner-unrelated"
    unrelated.mkdir()
    (unrelated / "owner.json").write_bytes(b'{"pid":')

    owner = _acquire_lock(tmp_path)

    assert profile._read_lock_owner(tmp_path / ".campaign-lock") == owner
    assert (unrelated / "owner.json").read_bytes() == b'{"pid":'


def test_campaign_competing_acquirer_cannot_lose_its_owner_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loser_ready = threading.Event()
    winner_done = threading.Event()
    real_rename = profile._rename_namespace
    results: dict[str, object] = {}

    def coordinated_rename(source: Path, target: Path) -> None:
        if (
            threading.current_thread().name == "campaign-owner-loser"
            and target.name == ".campaign-lock"
        ):
            loser_ready.set()
            assert winner_done.wait(5)
        real_rename(source, target)

    monkeypatch.setattr(profile, "_rename_namespace", coordinated_rename)

    def acquire(label: str, pid: int, token: str) -> None:
        try:
            results[label] = profile.acquire_campaign_lock(
                tmp_path,
                pid=pid,
                process_start_probe=lambda _pid: _OWNER_START,
                owner_token_factory=lambda: token,
            )
        except BaseException as exc:
            results[label] = exc
        finally:
            if label == "winner":
                winner_done.set()

    loser = threading.Thread(
        target=acquire,
        args=("loser", 111, "a" * 64),
        name="campaign-owner-loser",
    )
    winner = threading.Thread(
        target=acquire,
        args=("winner", 111, "a" * 64),
        name="campaign-owner-winner",
    )
    loser.start()
    assert loser_ready.wait(5)
    winner.start()
    winner.join(5)
    loser.join(5)

    assert not winner.is_alive()
    assert not loser.is_alive()
    assert isinstance(results["winner"], profile.CampaignLockOwner)
    assert isinstance(results["loser"], RuntimeError)
    assert str(results["loser"]) == "campaign_lock_held"
    assert not tuple(tmp_path.glob(".campaign-owner-*"))


def test_campaign_owner_publication_has_no_fixed_marker_aba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish_ready = {
        "campaign-owner-a": threading.Event(),
        "campaign-owner-b": threading.Event(),
    }
    allow_publish = {
        "campaign-owner-a": threading.Event(),
        "campaign-owner-b": threading.Event(),
    }
    real_rename = profile._rename_namespace
    results: dict[str, object] = {}

    def coordinated_rename(source: Path, target: Path) -> None:
        thread_name = threading.current_thread().name
        is_owner_publication = target.name == ".campaign-lock" or (
            target.name == "owner.json"
            and target.parent.name == ".campaign-lock"
        )
        if thread_name in publish_ready and is_owner_publication:
            publish_ready[thread_name].set()
            assert allow_publish[thread_name].wait(5)
        real_rename(source, target)

    monkeypatch.setattr(profile, "_rename_namespace", coordinated_rename)

    def acquire(label: str, pid: int, token: str) -> None:
        try:
            results[label] = profile.acquire_campaign_lock(
                tmp_path,
                pid=pid,
                process_start_probe=lambda _pid: _OWNER_START,
                owner_token_factory=lambda: token,
            )
        except BaseException as exc:
            results[label] = exc

    owner_a = threading.Thread(
        target=acquire,
        args=("a", 111, "a" * 64),
        name="campaign-owner-a",
    )
    owner_a.start()
    assert publish_ready["campaign-owner-a"].wait(5)

    try:
        results["recovery"] = profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: None
        )
    except BaseException as exc:
        results["recovery"] = exc

    owner_b = threading.Thread(
        target=acquire,
        args=("b", 222, "b" * 64),
        name="campaign-owner-b",
    )
    owner_b.start()
    assert publish_ready["campaign-owner-b"].wait(5)

    allow_publish["campaign-owner-a"].set()
    owner_a.join(5)
    allow_publish["campaign-owner-b"].set()
    owner_b.join(5)

    assert not owner_a.is_alive()
    assert not owner_b.is_alive()
    successes = [
        result
        for result in (results["a"], results["b"])
        if isinstance(result, profile.CampaignLockOwner)
    ]
    refusals = [
        result
        for result in (results["a"], results["b"])
        if isinstance(result, RuntimeError)
    ]
    assert len(successes) == 1
    assert [str(error) for error in refusals] == ["campaign_lock_held"]
    assert profile._read_lock_owner(tmp_path / ".campaign-lock") == successes[0]


def test_campaign_lock_refuses_second_owner_and_releases_only_exact_token(
    tmp_path: Path,
) -> None:
    owner = _acquire_lock(tmp_path)
    with pytest.raises(RuntimeError, match="campaign_lock_held"):
        _acquire_lock(tmp_path, pid=456)
    wrong = profile.CampaignLockOwner(
        pid=owner.pid,
        process_start_sha256=owner.process_start_sha256,
        owner_token="f" * 64,
    )

    with pytest.raises(RuntimeError, match="campaign_lock_owner_mismatch"):
        profile.release_campaign_lock(tmp_path, wrong)
    assert (tmp_path / ".campaign-lock").is_dir()

    profile.release_campaign_lock(tmp_path, owner)
    assert not (tmp_path / ".campaign-lock").exists()
    assert not (tmp_path / ".campaign-release").exists()


def _write_campaign_owner(lock_root: Path, owner) -> None:
    lock_root.mkdir()
    (lock_root / "owner.json").write_text(
        json.dumps(
            {
                "owner_token": owner.owner_token,
                "pid": owner.pid,
                "process_start_sha256": owner.process_start_sha256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_campaign_release_never_deletes_replacement_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _acquire_lock(tmp_path)
    replacement = profile.CampaignLockOwner(
        pid=456,
        process_start_sha256="f" * 64,
        owner_token="1" * 64,
    )
    real_read_owner = profile._read_lock_owner
    replaced = False

    def replace_after_validation(lock_root: Path):
        nonlocal replaced
        observed = real_read_owner(lock_root)
        if not replaced:
            replaced = True
            if lock_root.name == ".campaign-lock":
                (lock_root / "owner.json").unlink()
                lock_root.rmdir()
            _write_campaign_owner(tmp_path / ".campaign-lock", replacement)
        return observed

    monkeypatch.setattr(profile, "_read_lock_owner", replace_after_validation)

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_mismatch$"):
        profile.release_campaign_lock(tmp_path, original)

    assert real_read_owner(tmp_path / ".campaign-lock") == replacement
    assert not (tmp_path / ".campaign-release").exists()


def test_campaign_wrong_release_token_restores_original_canonical_lock(
    tmp_path: Path,
) -> None:
    owner = _acquire_lock(tmp_path)
    wrong = profile.CampaignLockOwner(
        pid=owner.pid,
        process_start_sha256=owner.process_start_sha256,
        owner_token="2" * 64,
    )

    with pytest.raises(RuntimeError, match="campaign_lock_owner_mismatch"):
        profile.release_campaign_lock(tmp_path, wrong)

    assert profile._read_lock_owner(tmp_path / ".campaign-lock") == owner
    assert not (tmp_path / ".campaign-release").exists()


def test_campaign_wrong_release_token_never_moves_canonical_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _acquire_lock(tmp_path)
    wrong = profile.CampaignLockOwner(
        pid=owner.pid,
        process_start_sha256=owner.process_start_sha256,
        owner_token="2" * 64,
    )
    real_rename = Path.rename
    renamed = False

    def recording_rename(path: Path, target: Path):
        nonlocal renamed
        if path.name == ".campaign-lock":
            renamed = True
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", recording_rename)

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_mismatch$"):
        profile.release_campaign_lock(tmp_path, wrong)

    assert not renamed
    assert profile._read_lock_owner(tmp_path / ".campaign-lock") == owner


def test_campaign_wrong_release_cannot_resurrect_after_exact_owner_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _acquire_lock(tmp_path)
    wrong = profile.CampaignLockOwner(
        pid=owner.pid,
        process_start_sha256=owner.process_start_sha256,
        owner_token="2" * 64,
    )
    real_read_owner = profile._read_lock_owner
    interleaved = False

    def finish_exact_release_after_claim(lock_root: Path):
        nonlocal interleaved
        observed = real_read_owner(lock_root)
        if lock_root.name in {".campaign-lock", ".campaign-release"} and not interleaved:
            interleaved = True
            profile.release_campaign_lock(tmp_path, owner)
        return observed

    monkeypatch.setattr(profile, "_read_lock_owner", finish_exact_release_after_claim)

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_mismatch$"):
        profile.release_campaign_lock(tmp_path, wrong)

    assert not (tmp_path / ".campaign-lock").exists()
    assert not (tmp_path / ".campaign-release").exists()


def test_campaign_release_conflict_remains_recoverable_by_exact_owner(
    tmp_path: Path,
) -> None:
    original = _acquire_lock(tmp_path)
    replacement = profile.CampaignLockOwner(
        pid=456,
        process_start_sha256="f" * 64,
        owner_token="4" * 64,
    )
    (tmp_path / ".campaign-lock").rename(tmp_path / ".campaign-release")
    _write_campaign_owner(tmp_path / ".campaign-lock", replacement)

    with pytest.raises(RuntimeError, match="campaign_release_in_progress"):
        profile.release_campaign_lock(tmp_path, original)
    profile._delete_exact_lock_root(tmp_path / ".campaign-lock", replacement)

    profile.release_campaign_lock(tmp_path, original)

    assert not (tmp_path / ".campaign-lock").exists()
    assert not (tmp_path / ".campaign-release").exists()


def test_campaign_release_fsync_failure_preserves_resumable_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _acquire_lock(tmp_path)
    fail = True
    real_fsync_directory = profile._fsync_directory

    def injected_fsync(path: Path) -> None:
        if fail and path == tmp_path:
            raise OSError("injected directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(profile, "_fsync_directory", injected_fsync)

    with pytest.raises(RuntimeError, match="^campaign_release_in_progress$"):
        profile.release_campaign_lock(tmp_path, owner)

    assert profile._read_lock_owner(tmp_path / ".campaign-release") == owner
    fail = False
    profile.release_campaign_lock(tmp_path, owner)
    assert not (tmp_path / ".campaign-release").exists()


def test_campaign_owner_unlink_fsync_failure_leaves_empty_resumable_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _acquire_lock(tmp_path)
    fail = True
    real_fsync_directory = profile._fsync_directory

    def injected_fsync(path: Path) -> None:
        if fail and path.name == ".campaign-release":
            raise OSError("injected directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(profile, "_fsync_directory", injected_fsync)

    with pytest.raises(OSError, match="injected directory fsync failure"):
        profile.release_campaign_lock(tmp_path, owner)

    assert (tmp_path / ".campaign-release").is_dir()
    assert not any((tmp_path / ".campaign-release").iterdir())
    fail = False
    profile.release_campaign_lock(tmp_path, owner)
    assert not (tmp_path / ".campaign-release").exists()


def test_campaign_release_marker_blocks_acquisition_and_recovery(
    tmp_path: Path,
) -> None:
    _acquire_attempt(tmp_path)
    (tmp_path / ".campaign-lock").rename(tmp_path / ".campaign-release")

    with pytest.raises(RuntimeError, match="campaign_release_in_progress"):
        _acquire_attempt(tmp_path, pid=456)
    with pytest.raises(RuntimeError, match="campaign_release_in_progress"):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: None
        )

    assert (tmp_path / ".campaign-release").is_dir()
    assert not (tmp_path / ".campaign-lock").exists()


def test_campaign_attempt_acquisition_leaves_running_lock_and_staging_evidence(
    tmp_path: Path,
) -> None:
    attempt = _acquire_attempt(tmp_path)

    assert attempt.attempt_id == "attempt-0001"
    assert attempt.root == tmp_path / "attempts" / "attempt-0001"
    assert attempt.root.is_dir()
    assert (tmp_path / ".campaign-lock").is_dir()
    assert profile.attempt_lineage(tmp_path / "attempts.jsonl") == (
        {"attempt_id": "attempt-0001", "state": "running"},
    )


def test_campaign_attempt_records_running_before_staging_root_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "attempts.jsonl"

    def fail_after_running(_campaign_root: Path, attempt_id: str) -> Path:
        assert attempt_id == "attempt-0001"
        assert profile.attempt_lineage(ledger) == (
            {"attempt_id": "attempt-0001", "state": "running"},
        )
        raise OSError("injected staging creation failure")

    monkeypatch.setattr(profile, "create_attempt_root", fail_after_running)

    with pytest.raises(OSError, match="injected staging creation failure"):
        _acquire_attempt(tmp_path)

    assert profile.attempt_lineage(ledger) == (
        {"attempt_id": "attempt-0001", "state": "running"},
        {
            "attempt_id": "attempt-0001",
            "state": "failed",
            "reason_category": "acquisition",
        },
    )
    assert not (tmp_path / ".campaign-lock").exists()


def test_campaign_recovery_records_legacy_single_orphan_attempt(
    tmp_path: Path,
) -> None:
    _acquire_lock(tmp_path)
    orphan = profile.create_attempt_root(tmp_path, "attempt-0001")

    event = profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    )

    assert event == _attempt_event(
        "attempt-0001", "failed", reason_category="interrupted"
    )
    assert profile.attempt_lineage(tmp_path / "attempts.jsonl") == (
        {"attempt_id": "attempt-0001", "state": "running"},
        event,
    )
    assert orphan.is_dir()
    assert _acquire_attempt(tmp_path).attempt_id == "attempt-0002"


def test_campaign_recovery_rejects_multiple_legacy_orphan_roots(
    tmp_path: Path,
) -> None:
    _acquire_lock(tmp_path)
    first = profile.create_attempt_root(tmp_path, "attempt-0001")
    second = profile.create_attempt_root(tmp_path, "attempt-0002")

    with pytest.raises(RuntimeError, match="^campaign_orphan_attempt_invalid$"):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: None
        )

    assert first.is_dir()
    assert second.is_dir()
    assert (tmp_path / ".campaign-lock").is_dir()
    assert not (tmp_path / "attempts.jsonl").exists()


def test_campaign_ledger_creation_fsync_failure_retains_running_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "attempts.jsonl"
    real_fsync_directory = profile._fsync_directory

    def injected_fsync(path: Path) -> None:
        if path == tmp_path and ledger.exists():
            raise OSError("injected ledger namespace fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(profile, "_fsync_directory", injected_fsync)

    with pytest.raises(OSError, match="injected ledger namespace fsync failure"):
        _acquire_attempt(tmp_path)

    assert profile.attempt_lineage(ledger) == (
        {"attempt_id": "attempt-0001", "state": "running"},
    )
    assert (tmp_path / ".campaign-lock").is_dir()


def test_campaign_acquisition_resumes_empty_canonical_lock(tmp_path: Path) -> None:
    (tmp_path / ".campaign-lock").mkdir()

    owner = _acquire_lock(tmp_path)

    assert profile._read_lock_owner(tmp_path / ".campaign-lock") == owner
    assert not (tmp_path / ".campaign-recovery").exists()


def test_campaign_recovery_releases_dead_lock_before_running_ledger(
    tmp_path: Path,
) -> None:
    _acquire_lock(tmp_path)

    event = profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    )

    assert event == {"state": "failed", "reason_category": "interrupted"}
    assert not (tmp_path / ".campaign-lock").exists()
    assert not (tmp_path / ".campaign-recovery").exists()
    assert not (tmp_path / "attempts.jsonl").exists()
    assert _acquire_attempt(tmp_path).attempt_id == "attempt-0001"


def test_campaign_recovery_marks_running_ledger_without_attempt_root_interrupted(
    tmp_path: Path,
) -> None:
    _acquire_lock(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    running = _attempt_event("attempt-0001", "running")
    profile.append_attempt_state(ledger, running)
    assert not (tmp_path / "attempts").exists()

    event = profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    )

    assert event == _attempt_event(
        "attempt-0001", "failed", reason_category="interrupted"
    )
    assert profile.attempt_lineage(ledger) == (running, event)
    assert not (tmp_path / ".campaign-lock").exists()
    assert not (tmp_path / ".campaign-recovery").exists()


@pytest.mark.parametrize(
    ("state", "reason_category"),
    (("failed", "provider"), ("invalid", "privacy")),
)
def test_campaign_recovery_releases_dead_owner_after_durable_terminal(
    tmp_path: Path, state: str, reason_category: str
) -> None:
    attempt = _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    terminal = _attempt_event(
        attempt.attempt_id, state, reason_category=reason_category
    )
    profile.append_attempt_state(ledger, terminal)
    before = ledger.read_bytes()

    assert profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    ) == terminal

    assert ledger.read_bytes() == before
    assert not (tmp_path / ".campaign-lock").exists()
    assert not (tmp_path / ".campaign-recovery").exists()


def test_campaign_recovery_refuses_live_owner_after_durable_terminal(
    tmp_path: Path,
) -> None:
    attempt = _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    terminal = _attempt_event(
        attempt.attempt_id, "failed", reason_category="provider"
    )
    profile.append_attempt_state(ledger, terminal)
    before = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_live$"):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: _OWNER_START
        )

    assert ledger.read_bytes() == before
    assert (tmp_path / ".campaign-lock").is_dir()
    assert not (tmp_path / ".campaign-recovery").exists()


@pytest.mark.parametrize("state", ("complete_pending_review", "changes_required"))
@pytest.mark.parametrize(
    "marker", (".campaign-lock", ".campaign-recovery", ".campaign-release")
)
def test_campaign_recovery_releases_dead_maintenance_owner_without_lineage_mutation(
    tmp_path: Path, state: str, marker: str
) -> None:
    attempt = _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    profile.complete_attempt_measurement(
        ledger,
        attempt.attempt_id,
        verdict="pass",
        raw_sha256="a" * 64,
    )
    profile.release_campaign_attempt(tmp_path, attempt)
    if state == "changes_required":
        profile.append_attempt_state(
            ledger,
            {
                "attempt_id": attempt.attempt_id,
                "state": "changes_required",
                "verdict": "pass",
                "raw_sha256": "a" * 64,
                "reason_category": "receipt",
            },
        )
    _acquire_lock(tmp_path)
    if marker != ".campaign-lock":
        (tmp_path / ".campaign-lock").rename(tmp_path / marker)
    before = ledger.read_bytes()

    recovered = profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    )

    assert recovered["state"] == state
    assert ledger.read_bytes() == before
    assert not any(
        (tmp_path / name).exists()
        for name in (".campaign-lock", ".campaign-recovery", ".campaign-release")
    )


@pytest.mark.parametrize("state", ("complete_pending_review", "changes_required"))
@pytest.mark.parametrize("marker", (".campaign-lock", ".campaign-recovery"))
def test_campaign_recovery_refuses_live_maintenance_owner(
    tmp_path: Path, state: str, marker: str
) -> None:
    attempt = _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    profile.complete_attempt_measurement(
        ledger,
        attempt.attempt_id,
        verdict="pass",
        raw_sha256="a" * 64,
    )
    profile.release_campaign_attempt(tmp_path, attempt)
    if state == "changes_required":
        profile.append_attempt_state(
            ledger,
            {
                "attempt_id": attempt.attempt_id,
                "state": "changes_required",
                "verdict": "pass",
                "raw_sha256": "a" * 64,
                "reason_category": "receipt",
            },
        )
    _acquire_lock(tmp_path)
    if marker == ".campaign-recovery":
        (tmp_path / ".campaign-lock").rename(tmp_path / marker)
    before = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_live$"):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: _OWNER_START
        )

    assert ledger.read_bytes() == before
    assert (tmp_path / marker).is_dir()


@pytest.mark.parametrize("state", ("complete_pending_review", "changes_required"))
@pytest.mark.parametrize("marker", (".campaign-recovery", ".campaign-release"))
def test_campaign_recovery_finishes_empty_maintenance_checkpoint_without_probe(
    tmp_path: Path, state: str, marker: str
) -> None:
    attempt = _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    profile.complete_attempt_measurement(
        ledger,
        attempt.attempt_id,
        verdict="pass",
        raw_sha256="a" * 64,
    )
    if state == "changes_required":
        profile.append_attempt_state(
            ledger,
            {
                "attempt_id": attempt.attempt_id,
                "state": "changes_required",
                "verdict": "pass",
                "raw_sha256": "a" * 64,
                "reason_category": "receipt",
            },
        )
    (tmp_path / ".campaign-lock").rename(tmp_path / marker)
    (tmp_path / marker / "owner.json").unlink()
    before = ledger.read_bytes()

    recovered = profile.recover_interrupted_attempt(
        tmp_path,
        process_start_probe=lambda _pid: pytest.fail("empty checkpoint was probed"),
    )

    assert recovered["state"] == state
    assert ledger.read_bytes() == before
    assert not (tmp_path / marker).exists()


def test_campaign_recovery_rejects_conflicting_maintenance_markers(
    tmp_path: Path,
) -> None:
    attempt = _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    profile.complete_attempt_measurement(
        ledger,
        attempt.attempt_id,
        verdict="pass",
        raw_sha256="a" * 64,
    )
    profile.release_campaign_attempt(tmp_path, attempt)
    _acquire_lock(tmp_path)
    _write_campaign_owner(
        tmp_path / ".campaign-recovery",
        profile.CampaignLockOwner(456, "c" * 64, "d" * 64),
    )
    before = ledger.read_bytes()

    with pytest.raises(RuntimeError, match="^campaign_recovery_owner_conflict$"):
        profile.recover_interrupted_attempt(
            tmp_path,
            process_start_probe=lambda _pid: pytest.fail("conflict was probed"),
        )

    assert ledger.read_bytes() == before
    assert (tmp_path / ".campaign-lock").is_dir()
    assert (tmp_path / ".campaign-recovery").is_dir()


@pytest.mark.parametrize("verdict", ("pass", "smoke"))
def test_campaign_recover_is_idempotent_after_pending_attempt_release(
    tmp_path: Path, verdict: str
) -> None:
    attempt = _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    pending = profile.write_acquisition_pin(
        tmp_path,
        attempt.attempt_id,
        verdict=verdict,
        raw_sha256="a" * 64,
    )
    profile.complete_attempt_measurement(
        ledger,
        attempt.attempt_id,
        verdict=verdict,
        raw_sha256="a" * 64,
    )
    profile.release_campaign_attempt(tmp_path, attempt)
    before = ledger.read_bytes()

    assert profile.recover_interrupted_attempt(
        tmp_path,
        process_start_probe=lambda _pid: pytest.fail("released owner was probed"),
    ) == pending

    assert ledger.read_bytes() == before
    assert not (tmp_path / ".campaign-lock").exists()
    assert not (tmp_path / ".campaign-recovery").exists()
    assert not (tmp_path / ".campaign-release").exists()


@pytest.mark.parametrize("verdict", ("pass", "smoke"))
@pytest.mark.parametrize("empty_release", (False, True))
def test_campaign_recover_finishes_pending_release_marker_without_probe(
    tmp_path: Path, verdict: str, empty_release: bool
) -> None:
    attempt = _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    pending = profile.write_acquisition_pin(
        tmp_path,
        attempt.attempt_id,
        verdict=verdict,
        raw_sha256="a" * 64,
    )
    profile.complete_attempt_measurement(
        ledger,
        attempt.attempt_id,
        verdict=verdict,
        raw_sha256="a" * 64,
    )
    release = tmp_path / ".campaign-release"
    (tmp_path / ".campaign-lock").rename(release)
    if empty_release:
        (release / "owner.json").unlink()
    before = ledger.read_bytes()

    assert profile.recover_interrupted_attempt(
        tmp_path,
        process_start_probe=lambda _pid: pytest.fail("terminal owner was probed"),
    ) == pending

    assert ledger.read_bytes() == before
    assert not release.exists()


@pytest.mark.parametrize(
    "conflict_marker",
    (".campaign-lock", ".campaign-recovery", ".campaign-rollback"),
)
def test_campaign_recover_rejects_conflicting_pending_release_markers(
    tmp_path: Path, conflict_marker: str
) -> None:
    attempt = _acquire_attempt(tmp_path)
    pending = profile.write_acquisition_pin(
        tmp_path,
        attempt.attempt_id,
        verdict="pass",
        raw_sha256="a" * 64,
    )
    profile.complete_attempt_measurement(
        tmp_path / "attempts.jsonl",
        attempt.attempt_id,
        verdict=str(pending["verdict"]),
        raw_sha256=str(pending["raw_sha256"]),
    )
    release = tmp_path / ".campaign-release"
    (tmp_path / ".campaign-lock").rename(release)
    _write_campaign_owner(
        tmp_path / conflict_marker,
        profile.CampaignLockOwner(456, "c" * 64, "d" * 64),
    )

    with pytest.raises(RuntimeError, match="^campaign_recovery_owner_conflict$"):
        profile.recover_interrupted_attempt(
            tmp_path,
            process_start_probe=lambda _pid: pytest.fail("conflict was probed"),
        )

    assert (tmp_path / conflict_marker).is_dir()
    assert release.is_dir()


@pytest.mark.parametrize("marker", (".campaign-recovery", ".campaign-release"))
@pytest.mark.parametrize("empty_marker", (False, True))
def test_campaign_recovery_finishes_terminal_lock_marker_without_appending(
    tmp_path: Path, marker: str, empty_marker: bool
) -> None:
    attempt = _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    terminal = _attempt_event(
        attempt.attempt_id, "invalid", reason_category="raw"
    )
    profile.append_attempt_state(ledger, terminal)
    before = ledger.read_bytes()
    (tmp_path / ".campaign-lock").rename(tmp_path / marker)
    if empty_marker:
        (tmp_path / marker / "owner.json").unlink()

    assert profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    ) == terminal

    assert ledger.read_bytes() == before
    assert not (tmp_path / marker).exists()


def test_campaign_recovery_finishes_owned_marker_after_interrupted_append(
    tmp_path: Path,
) -> None:
    _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    (tmp_path / ".campaign-lock").rename(tmp_path / ".campaign-recovery")
    failed = _attempt_event(
        "attempt-0001", "failed", reason_category="interrupted"
    )
    profile.append_attempt_state(ledger, failed)

    assert profile.recover_interrupted_attempt(
        tmp_path,
        process_start_probe=lambda _pid: pytest.fail("must not probe completed recovery"),
    ) == failed
    assert not (tmp_path / ".campaign-recovery").exists()
    assert profile.attempt_lineage(ledger) == (
        {"attempt_id": "attempt-0001", "state": "running"},
        failed,
    )
    assert _acquire_attempt(tmp_path).attempt_id == "attempt-0002"


def test_campaign_recovery_append_exception_preserves_resumable_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _acquire_attempt(tmp_path)
    real_append = profile.append_attempt_state

    def append_then_fail(ledger: Path, event: dict[str, object]) -> None:
        real_append(ledger, event)
        raise OSError("injected post-append failure")

    monkeypatch.setattr(profile, "append_attempt_state", append_then_fail)

    with pytest.raises(OSError, match="injected post-append failure"):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: None
        )

    assert (tmp_path / ".campaign-recovery").is_dir()
    assert not (tmp_path / ".campaign-lock").exists()
    monkeypatch.setattr(profile, "append_attempt_state", real_append)
    assert profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: pytest.fail("must not reprobe")
    ) == _attempt_event(
        "attempt-0001", "failed", reason_category="interrupted"
    )
    assert not (tmp_path / ".campaign-recovery").exists()


def test_campaign_recovery_resumes_owned_marker_before_interrupted_append(
    tmp_path: Path,
) -> None:
    _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    (tmp_path / ".campaign-lock").rename(tmp_path / ".campaign-recovery")

    event = profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    )

    assert event == _attempt_event(
        "attempt-0001", "failed", reason_category="interrupted"
    )
    assert [item["state"] for item in profile.attempt_lineage(ledger)] == [
        "running",
        "failed",
    ]
    assert not (tmp_path / ".campaign-recovery").exists()
    assert _acquire_attempt(tmp_path).attempt_id == "attempt-0002"


def test_campaign_release_and_recovery_resume_empty_markers(tmp_path: Path) -> None:
    release_campaign = tmp_path / "release"
    release_owner = _acquire_lock(release_campaign)
    (release_campaign / ".campaign-lock").rename(
        release_campaign / ".campaign-release"
    )
    (release_campaign / ".campaign-release" / "owner.json").unlink()

    profile.release_campaign_lock(release_campaign, release_owner)

    assert not (release_campaign / ".campaign-release").exists()
    assert _acquire_lock(release_campaign)

    recovery_campaign = tmp_path / "recovery"
    _acquire_attempt(recovery_campaign)
    recovery_ledger = recovery_campaign / "attempts.jsonl"
    (recovery_campaign / ".campaign-lock").rename(
        recovery_campaign / ".campaign-recovery"
    )
    failed = _attempt_event(
        "attempt-0001", "failed", reason_category="interrupted"
    )
    profile.append_attempt_state(recovery_ledger, failed)
    (recovery_campaign / ".campaign-recovery" / "owner.json").unlink()

    assert profile.recover_interrupted_attempt(recovery_campaign) == failed
    assert not (recovery_campaign / ".campaign-recovery").exists()
    assert _acquire_attempt(recovery_campaign).attempt_id == "attempt-0002"


def test_campaign_acquisition_finishes_empty_release_marker_without_owner_token(
    tmp_path: Path,
) -> None:
    attempt = _acquire_attempt(tmp_path)
    profile.append_attempt_state(
        tmp_path / "attempts.jsonl",
        _attempt_event(
            "attempt-0001", "failed", reason_category="acquisition"
        ),
    )
    (tmp_path / ".campaign-lock").rename(tmp_path / ".campaign-release")
    (tmp_path / ".campaign-release" / "owner.json").unlink()

    resumed = _acquire_attempt(tmp_path, pid=456)

    assert attempt.attempt_id == "attempt-0001"
    assert resumed.attempt_id == "attempt-0002"
    assert not (tmp_path / ".campaign-release").exists()


def test_campaign_acquisition_preserves_empty_recovery_for_running_attempt(
    tmp_path: Path,
) -> None:
    _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    before = ledger.read_bytes()
    (tmp_path / ".campaign-lock").rename(tmp_path / ".campaign-recovery")
    (tmp_path / ".campaign-recovery" / "owner.json").unlink()

    with pytest.raises(RuntimeError, match="^campaign_recovery_in_progress$"):
        _acquire_attempt(tmp_path, pid=456)

    assert (tmp_path / ".campaign-recovery").is_dir()
    assert not any((tmp_path / ".campaign-recovery").iterdir())
    assert not (tmp_path / ".campaign-lock").exists()
    assert ledger.read_bytes() == before


def test_campaign_live_exact_owner_recovery_refuses_without_mutation(
    tmp_path: Path,
) -> None:
    _acquire_attempt(tmp_path)
    before = (tmp_path / "attempts.jsonl").read_bytes()

    with pytest.raises(RuntimeError, match="campaign_lock_owner_live"):
        profile.recover_interrupted_attempt(
            tmp_path,
            process_start_probe=lambda pid: _OWNER_START if pid == 123 else None,
        )

    assert (tmp_path / ".campaign-lock").is_dir()
    assert (tmp_path / "attempts.jsonl").read_bytes() == before


def test_campaign_process_probe_failures_use_stable_codes_and_preserve_lock(
    tmp_path: Path,
) -> None:
    def failing_probe(_pid: int) -> str:
        raise RuntimeError("arbitrary-sensitive-probe-detail")

    with pytest.raises(RuntimeError, match="^campaign_process_identity_failed$"):
        profile.acquire_campaign_lock(
            tmp_path / "new",
            pid=123,
            process_start_probe=failing_probe,
            owner_token_factory=lambda: _OWNER_TOKEN,
        )
    assert not (tmp_path / "new" / ".campaign-lock").exists()

    _acquire_attempt(tmp_path / "existing")
    before = (tmp_path / "existing" / "attempts.jsonl").read_bytes()
    with pytest.raises(RuntimeError, match="^campaign_process_identity_failed$"):
        profile.recover_interrupted_attempt(
            tmp_path / "existing", process_start_probe=failing_probe
        )
    assert (tmp_path / "existing" / ".campaign-lock").is_dir()
    assert (tmp_path / "existing" / "attempts.jsonl").read_bytes() == before


def test_campaign_owner_token_generation_failure_uses_dedicated_code(
    tmp_path: Path,
) -> None:
    def failing_token() -> str:
        raise RuntimeError("sensitive token detail")

    with pytest.raises(RuntimeError, match="^campaign_owner_token_failed$"):
        profile.acquire_campaign_lock(
            tmp_path,
            pid=123,
            process_start_probe=lambda _pid: _OWNER_START,
            owner_token_factory=failing_token,
        )

    assert not (tmp_path / ".campaign-lock").exists()


def test_process_start_identity_returns_none_only_for_exact_missing_pid() -> None:
    def missing_pid(_command, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    assert profile.process_start_identity(123, run_command=missing_pid) is None


def test_process_start_identity_hashes_valid_ps_start_time() -> None:
    started = "Fri Aug 22 13:00:00 2026"

    assert profile.process_start_identity(
        123,
        run_command=lambda _command, **_kwargs: SimpleNamespace(
            returncode=0, stdout=started + "\n", stderr=""
        ),
    ) == hashlib.sha256(started.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "completed",
    (
        SimpleNamespace(returncode=2, stdout="", stderr="ps failed"),
        SimpleNamespace(returncode=1, stdout="", stderr="permission denied"),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="not-a-start-time\n", stderr=""),
    ),
)
def test_process_start_identity_rejects_operational_and_parse_failures(
    completed: SimpleNamespace,
) -> None:
    with pytest.raises(RuntimeError, match="^campaign_process_identity_failed$"):
        profile.process_start_identity(
            123, run_command=lambda _command, **_kwargs: completed
        )


def test_process_start_identity_normalizes_command_exception() -> None:
    def failing_command(_command, **_kwargs):
        raise OSError("sensitive command detail")

    with pytest.raises(RuntimeError, match="^campaign_process_identity_failed$"):
        profile.process_start_identity(123, run_command=failing_command)


@pytest.mark.parametrize(
    "second_outcome",
    (
        SimpleNamespace(returncode=2, stdout="", stderr="ps failed"),
        SimpleNamespace(returncode=1, stdout="", stderr="permission denied"),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        OSError("sensitive command detail"),
    ),
)
def test_campaign_recovery_ps_error_rolls_back_without_ledger_mutation(
    tmp_path: Path, second_outcome: object
) -> None:
    _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    before = ledger.read_bytes()
    calls = 0

    def run_command(_command, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if isinstance(second_outcome, BaseException):
            raise second_outcome
        return second_outcome

    with pytest.raises(RuntimeError, match="^campaign_process_identity_failed$"):
        profile.recover_interrupted_attempt(
            tmp_path,
            process_start_probe=lambda pid: profile.process_start_identity(
                pid, run_command=run_command
            ),
        )

    assert profile._read_lock_owner(tmp_path / ".campaign-lock").owner_token == (
        _OWNER_TOKEN
    )
    assert not (tmp_path / ".campaign-recovery").exists()
    assert ledger.read_bytes() == before


@pytest.mark.parametrize(
    "observed_identity",
    (None, 7, True, "", "not-a-hash", "g" * 64),
)
def test_campaign_acquisition_rejects_malformed_process_identity(
    tmp_path: Path, observed_identity: object
) -> None:
    with pytest.raises(RuntimeError, match="^campaign_process_identity_invalid$"):
        profile.acquire_campaign_lock(
            tmp_path,
            pid=123,
            process_start_probe=lambda _pid: observed_identity,
            owner_token_factory=lambda: _OWNER_TOKEN,
        )

    assert not (tmp_path / ".campaign-lock").exists()
    assert not (tmp_path / ".campaign-recovery").exists()
    assert not (tmp_path / ".campaign-release").exists()


@pytest.mark.parametrize("observed_identity", (7, True, "", "bad", "g" * 64))
def test_campaign_recovery_rejects_malformed_pre_takeover_identity(
    tmp_path: Path, observed_identity: object
) -> None:
    _acquire_attempt(tmp_path)
    before = (tmp_path / "attempts.jsonl").read_bytes()

    with pytest.raises(RuntimeError, match="^campaign_process_identity_invalid$"):
        profile.recover_interrupted_attempt(
            tmp_path,
            process_start_probe=lambda _pid: observed_identity,
        )

    assert profile._read_lock_owner(tmp_path / ".campaign-lock").owner_token == (
        _OWNER_TOKEN
    )
    assert not (tmp_path / ".campaign-recovery").exists()
    assert (tmp_path / "attempts.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    ("second_outcome", "code"),
    (
        (_OWNER_START, "campaign_lock_owner_live"),
        (7, "campaign_process_identity_invalid"),
        ("bad", "campaign_process_identity_invalid"),
        (RuntimeError("probe-failed"), "campaign_process_identity_failed"),
    ),
)
def test_campaign_recovery_rolls_back_after_second_probe_failure(
    tmp_path: Path, second_outcome: object, code: str
) -> None:
    _acquire_attempt(tmp_path)
    before = (tmp_path / "attempts.jsonl").read_bytes()
    calls = 0

    def probe(_pid: int):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        if isinstance(second_outcome, BaseException):
            raise second_outcome
        return second_outcome

    with pytest.raises(RuntimeError, match=f"^{code}$"):
        profile.recover_interrupted_attempt(tmp_path, process_start_probe=probe)

    assert profile._read_lock_owner(tmp_path / ".campaign-lock").owner_token == (
        _OWNER_TOKEN
    )
    assert not (tmp_path / ".campaign-recovery").exists()
    assert (tmp_path / "attempts.jsonl").read_bytes() == before


def test_campaign_recovery_rollback_owner_publication_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = _acquire_attempt(tmp_path)
    before = (tmp_path / "attempts.jsonl").read_bytes()
    calls = 0
    original_rename = profile._rename_namespace
    interrupted = False

    def probe(_pid: int) -> str | None:
        nonlocal calls
        calls += 1
        return None if calls == 1 else _OWNER_START

    def interrupted_owner_publish(source: Path, target: Path) -> None:
        nonlocal interrupted
        if (
            source.name.startswith(".campaign-owner-")
            and target.name == ".campaign-lock"
            and not interrupted
        ):
            interrupted = True
            raise OSError("simulated owner publication interruption")
        original_rename(source, target)

    monkeypatch.setattr(profile, "_rename_namespace", interrupted_owner_publish)

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_live$"):
        profile.recover_interrupted_attempt(tmp_path, process_start_probe=probe)

    canonical = tmp_path / ".campaign-lock"
    assert not canonical.exists()
    assert profile._read_lock_owner(tmp_path / ".campaign-recovery") == attempt.owner
    assert not tuple(tmp_path.glob(".campaign-owner-*"))
    assert (tmp_path / "attempts.jsonl").read_bytes() == before

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_live$"):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: _OWNER_START
        )

    assert profile._read_lock_owner(canonical) == attempt.owner
    assert not (tmp_path / ".campaign-recovery").exists()
    assert not (tmp_path / ".campaign-rollback").exists()


@pytest.mark.parametrize("duplicate_state", ("owned", "empty"))
def test_campaign_recovery_restarts_after_duplicate_rollback_checkpoint(
    tmp_path: Path, duplicate_state: str
) -> None:
    attempt = _acquire_attempt(tmp_path)
    recovery = tmp_path / ".campaign-recovery"
    if duplicate_state == "owned":
        _write_campaign_owner(recovery, attempt.owner)
    else:
        recovery.mkdir()

    event = profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    )

    assert event == _attempt_event(
        "attempt-0001", "failed", reason_category="interrupted"
    )
    assert profile.attempt_lineage(tmp_path / "attempts.jsonl") == (
        _attempt_event("attempt-0001", "running"),
        event,
    )
    assert not (tmp_path / ".campaign-lock").exists()
    assert not recovery.exists()


def test_campaign_recovery_refuses_different_canonical_and_recovery_owners(
    tmp_path: Path,
) -> None:
    _acquire_attempt(tmp_path)
    replacement = profile.CampaignLockOwner(
        pid=456,
        process_start_sha256="f" * 64,
        owner_token="3" * 64,
    )
    recovery = tmp_path / ".campaign-recovery"
    _write_campaign_owner(recovery, replacement)
    ledger = tmp_path / "attempts.jsonl"
    ledger_before = ledger.read_bytes()
    canonical_before = (tmp_path / ".campaign-lock" / "owner.json").read_bytes()
    recovery_before = (recovery / "owner.json").read_bytes()

    with pytest.raises(
        RuntimeError, match="^campaign_recovery_owner_conflict$"
    ):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: None
        )

    assert ledger.read_bytes() == ledger_before
    assert (tmp_path / ".campaign-lock" / "owner.json").read_bytes() == canonical_before
    assert (recovery / "owner.json").read_bytes() == recovery_before


@pytest.mark.parametrize("malformed_marker", ("canonical", "recovery"))
@pytest.mark.parametrize("payload", (b"not-json\n", b"{}\n"))
def test_campaign_recovery_preserves_malformed_dual_marker_owners(
    tmp_path: Path, malformed_marker: str, payload: bytes
) -> None:
    attempt = _acquire_attempt(tmp_path)
    canonical = tmp_path / ".campaign-lock"
    recovery = tmp_path / ".campaign-recovery"
    _write_campaign_owner(recovery, attempt.owner)
    malformed = canonical if malformed_marker == "canonical" else recovery
    (malformed / "owner.json").write_bytes(payload)
    ledger = tmp_path / "attempts.jsonl"
    ledger_before = ledger.read_bytes()
    canonical_before = (canonical / "owner.json").read_bytes()
    recovery_before = (recovery / "owner.json").read_bytes()
    probes = 0

    def probe(_pid: int) -> None:
        nonlocal probes
        probes += 1
        return None

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_invalid$"):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=probe
        )

    assert probes == 0
    assert ledger.read_bytes() == ledger_before
    assert (canonical / "owner.json").read_bytes() == canonical_before
    assert (recovery / "owner.json").read_bytes() == recovery_before


def test_campaign_recovery_rollback_conflict_preserves_both_locked_owners(
    tmp_path: Path,
) -> None:
    _acquire_attempt(tmp_path)
    before = (tmp_path / "attempts.jsonl").read_bytes()
    replacement = profile.CampaignLockOwner(
        pid=456,
        process_start_sha256="f" * 64,
        owner_token="3" * 64,
    )
    calls = 0

    def probe(_pid: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            _write_campaign_owner(tmp_path / ".campaign-lock", replacement)
            return _OWNER_START
        return None

    with pytest.raises(RuntimeError, match="^campaign_lock_owner_live$"):
        profile.recover_interrupted_attempt(tmp_path, process_start_probe=probe)

    assert profile._read_lock_owner(tmp_path / ".campaign-lock") == replacement
    assert profile._read_lock_owner(tmp_path / ".campaign-rollback").owner_token == (
        _OWNER_TOKEN
    )
    assert (tmp_path / "attempts.jsonl").read_bytes() == before
    with pytest.raises(RuntimeError, match="campaign_recovery_in_progress"):
        _acquire_attempt(tmp_path, pid=789)

    profile._delete_exact_lock_root(tmp_path / ".campaign-lock", replacement)
    with pytest.raises(RuntimeError, match="^campaign_recovery_rolled_back$"):
        _acquire_attempt(tmp_path, pid=789)

    assert profile._read_lock_owner(tmp_path / ".campaign-lock").owner_token == (
        _OWNER_TOKEN
    )
    assert not (tmp_path / ".campaign-recovery").exists()
    assert not (tmp_path / ".campaign-rollback").exists()
    assert (tmp_path / "attempts.jsonl").read_bytes() == before


def test_campaign_recovery_append_race_rolls_back_owned_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _acquire_attempt(tmp_path)
    ledger = tmp_path / "attempts.jsonl"
    real_append = profile.append_attempt_state

    def append_after_competing_completion(path: Path, event) -> None:
        if event["state"] == "failed":
            real_append(
                path,
                _attempt_event(
                    "attempt-0001",
                    "complete_pending_review",
                    verdict="pass",
                    raw_sha256="a" * 64,
                ),
            )
        real_append(path, event)

    monkeypatch.setattr(
        profile, "append_attempt_state", append_after_competing_completion
    )

    with pytest.raises(RuntimeError, match="^campaign_attempt_transition_invalid$"):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: None
        )

    assert profile._read_lock_owner(tmp_path / ".campaign-lock").owner_token == (
        _OWNER_TOKEN
    )
    assert not (tmp_path / ".campaign-recovery").exists()
    assert [event["state"] for event in profile.attempt_lineage(ledger)] == [
        "running",
        "complete_pending_review",
    ]


def test_campaign_dead_owner_recovery_appends_interrupted_and_preserves_raw(
    tmp_path: Path,
) -> None:
    attempt = _acquire_attempt(tmp_path)
    raw = attempt.root / "real-provider-three-turn.raw.jsonl"
    staging = attempt.root / "partial-staging.json"
    raw.write_bytes(b'{"event":"sample"}\n')
    staging.write_bytes(b"partial")
    raw_sha256 = hashlib.sha256(raw.read_bytes()).hexdigest()

    event = profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    )

    assert event == {
        "attempt_id": "attempt-0001",
        "state": "failed",
        "reason_category": "interrupted",
    }
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == raw_sha256
    assert staging.read_bytes() == b"partial"
    assert attempt.root.is_dir()
    assert not (tmp_path / ".campaign-lock").exists()
    assert not (tmp_path / ".campaign-recovery").exists()
    assert [row["state"] for row in profile.attempt_lineage(tmp_path / "attempts.jsonl")] == [
        "running",
        "failed",
    ]


def test_campaign_pid_reuse_is_stale_for_recorded_owner(tmp_path: Path) -> None:
    _acquire_attempt(tmp_path, pid=321)

    event = profile.recover_interrupted_attempt(
        tmp_path,
        process_start_probe=lambda pid: "f" * 64 if pid == 321 else None,
    )

    assert event["reason_category"] == "interrupted"


def test_campaign_concurrent_stale_recovery_has_one_winner(tmp_path: Path) -> None:
    _acquire_attempt(tmp_path)
    first_probe = threading.Barrier(2)
    probe_counts: dict[int, int] = {}
    probe_lock = threading.Lock()

    def dead_probe(_pid: int) -> None:
        thread_id = threading.get_ident()
        with probe_lock:
            count = probe_counts.get(thread_id, 0)
            probe_counts[thread_id] = count + 1
        if count == 0:
            first_probe.wait(timeout=5)
        return None

    def recover():
        return profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=dead_probe
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(recover) for _ in range(2)]
        outcomes: list[object] = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=5))
            except RuntimeError as exc:
                outcomes.append(exc)

    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, RuntimeError) for outcome in outcomes) == 1
    lineage = profile.attempt_lineage(tmp_path / "attempts.jsonl")
    assert [event["state"] for event in lineage] == ["running", "failed"]


def test_campaign_acquisition_cannot_slip_through_recovery_takeover(
    tmp_path: Path,
) -> None:
    _acquire_attempt(tmp_path)
    takeover_held = threading.Event()
    allow_recovery = threading.Event()
    call_count = 0

    def dead_probe(_pid: int) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            takeover_held.set()
            assert allow_recovery.wait(timeout=5)
        return None

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            profile.recover_interrupted_attempt,
            tmp_path,
            process_start_probe=dead_probe,
        )
        assert takeover_held.wait(timeout=5)
        assert (tmp_path / ".campaign-recovery").is_dir()
        with pytest.raises(RuntimeError, match="campaign_recovery_in_progress"):
            _acquire_attempt(tmp_path, pid=456)
        allow_recovery.set()
        assert future.result(timeout=5)["reason_category"] == "interrupted"


@pytest.mark.parametrize("state", ("complete_pending_review", "changes_required"))
def test_campaign_recovery_releases_dead_owner_after_review_terminal_state(
    tmp_path: Path, state: str
) -> None:
    _acquire_attempt(tmp_path)
    profile.complete_attempt_measurement(
        tmp_path / "attempts.jsonl",
        "attempt-0001",
        verdict="pass",
        raw_sha256="a" * 64,
    )
    if state == "changes_required":
        profile.append_attempt_state(
            tmp_path / "attempts.jsonl",
            _attempt_event(
                "attempt-0001",
                "changes_required",
                verdict="pass",
                raw_sha256="a" * 64,
                reason_category="digest",
            ),
        )

    before = (tmp_path / "attempts.jsonl").read_bytes()
    recovered = profile.recover_interrupted_attempt(
        tmp_path, process_start_probe=lambda _pid: None
    )

    assert recovered["state"] == state
    assert (tmp_path / "attempts.jsonl").read_bytes() == before
    assert not (tmp_path / ".campaign-lock").exists()


@pytest.mark.parametrize("owner_payload", (None, b"not-json\n", b"{}\n"))
def test_campaign_recovery_fails_closed_on_missing_or_malformed_owner(
    tmp_path: Path, owner_payload: bytes | None
) -> None:
    lock = tmp_path / ".campaign-lock"
    lock.mkdir()
    if owner_payload is not None:
        (lock / "owner.json").write_bytes(owner_payload)

    with pytest.raises(RuntimeError, match="campaign_lock_owner_invalid"):
        profile.recover_interrupted_attempt(
            tmp_path, process_start_probe=lambda _pid: None
        )

    assert lock.is_dir()


def test_campaign_attempt_cleanup_removes_only_owned_target_worktrees(
    tmp_path: Path,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    campaign = tmp_path / "campaign"
    attempt_root = profile.create_attempt_root(campaign, "attempt-0001")
    raw = attempt_root / "real-provider-three-turn.raw.jsonl"
    raw.write_bytes(b"retained\n")
    for name in ("control", "candidate"):
        _add_real_worktree(repository, attempt_root / name, branch=name)

    profile.cleanup_attempt_worktrees(
        repository,
        campaign,
        "attempt-0001",
    )

    assert profile._worktree_registrations(repository) == frozenset(
        {str(repository.resolve())}
    )
    assert raw.read_bytes() == b"retained\n"
    assert attempt_root.is_dir()


def test_campaign_attempt_cleanup_rejects_attempts_symlink_outside_campaign(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    outside = tmp_path / "outside"
    attempt = outside / "attempt-0001"
    for name in ("control", "candidate"):
        (attempt / name).mkdir(parents=True)
    (campaign / "attempts").symlink_to(outside, target_is_directory=True)
    calls: list[list[str]] = []

    def run_command(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    with pytest.raises(RuntimeError, match="^campaign_attempt_cleanup_refused$"):
        profile.cleanup_attempt_worktrees(
            tmp_path,
            campaign,
            "attempt-0001",
            run_command=run_command,
        )

    assert calls == []
    assert (attempt / "control").is_dir()
    assert (attempt / "candidate").is_dir()


def test_campaign_attempt_cleanup_rejects_attempt_root_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    attempt = profile.create_attempt_root(campaign, "attempt-0001")
    for name in ("control", "candidate"):
        (attempt / name).mkdir()
    outside = tmp_path / "outside" / "attempt-0001"
    for name in ("control", "candidate"):
        (outside / name).mkdir(parents=True)
    original_remove = profile._remove_target_worktree
    swapped = False
    calls: list[list[str]] = []

    def swap_then_remove(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            (campaign / "attempts").rename(campaign / "original-attempts")
            (campaign / "attempts").symlink_to(
                tmp_path / "outside", target_is_directory=True
            )
        return original_remove(*args, **kwargs)

    monkeypatch.setattr(profile, "_remove_target_worktree", swap_then_remove)

    def run_command(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    with pytest.raises(BaseExceptionGroup) as caught:
        profile.cleanup_attempt_worktrees(
            tmp_path,
            campaign,
            "attempt-0001",
            run_command=run_command,
        )

    assert {str(error) for error in caught.value.exceptions} == {
        "campaign_attempt_cleanup_refused"
    }
    assert calls == []
    assert (outside / "control").is_dir()
    assert (outside / "candidate").is_dir()


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
                "d3a1fc6fc6993704b14a3caae2ace886151229750956306ed12e42d3933a2e1c"
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


def test_current_enabled_bootstrap_routing_mismatches_pinned_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_original_evidence(tmp_path)
    _materialize_original_runner(tmp_path)
    expected = profile.load_original_protocol(tmp_path, tmp_path)
    real_paired = profile.paired_p95_ratio_bounds

    @functools.wraps(real_paired)
    def disabled_only(blocks, _candidate, **kwargs):
        return real_paired(blocks, "disabled", **kwargs)

    monkeypatch.setattr(profile, "paired_p95_ratio_bounds", disabled_only)
    observed = _original_protocol()

    assert "protocol_resampling_mismatch" in profile.protocol_mismatches(
        expected, observed
    )


def test_current_enabled_summary_routing_mismatches_pinned_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_original_evidence(tmp_path)
    _materialize_original_runner(tmp_path)
    expected = profile.load_original_protocol(tmp_path, tmp_path)
    real_build_summary = profile.build_summary

    @functools.wraps(real_build_summary)
    def disabled_only_summary(*args, **kwargs):
        summary = real_build_summary(*args, **kwargs)
        summary["arms"]["enabled"]["gates"] = copy.deepcopy(
            summary["arms"]["disabled"]["gates"]
        )
        summary["arms"]["enabled"]["verdict"] = summary["arms"]["disabled"][
            "verdict"
        ]
        return summary

    monkeypatch.setattr(profile, "build_summary", disabled_only_summary)
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


def test_parse_arguments_pins_confirmatory_campaign_defaults(tmp_path: Path) -> None:
    arguments = profile.parse_arguments(
        [
            "--endpoint",
            "http://127.0.0.1:9099",
            "--model",
            "fixture.gguf",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert arguments.campaign_action == "acquire"
    assert arguments.campaign_root is None
    assert arguments.burn_in_blocks == 0
    assert arguments.attempt_id is None
    assert arguments.review_receipt is None
    assert arguments.destination is None


def test_parse_arguments_accepts_confirmatory_acquisition_with_fixed_candidate(
    tmp_path: Path,
) -> None:
    arguments = profile.parse_arguments(
        [
            "--endpoint",
            "http://127.0.0.1:9099",
            "--model",
            "fixture.gguf",
            "--campaign-root",
            str(tmp_path),
            "--burn-in-blocks",
            "5",
            "--candidate-sha",
            profile.CANDIDATE_SHA,
        ]
    )

    assert arguments.output_root is None
    assert arguments.campaign_root == tmp_path
    assert arguments.burn_in_blocks == 5


@pytest.mark.parametrize(
    ("iterations", "burn_in_blocks", "kind", "conversation_count"),
    ((30, 5, "official", 108), (1, 1, "disposable_smoke", 9)),
)
def test_campaign_acquisition_accepts_only_registered_schedule_shapes(
    tmp_path: Path,
    iterations: int,
    burn_in_blocks: int,
    kind: str,
    conversation_count: int,
) -> None:
    arguments = profile.parse_arguments(
        [
            "--endpoint",
            "http://127.0.0.1:9099",
            "--model",
            "fixture.gguf",
            "--campaign-root",
            str(tmp_path),
            "--iterations",
            str(iterations),
            "--burn-in-blocks",
            str(burn_in_blocks),
            "--candidate-sha",
            profile.CANDIDATE_SHA,
        ]
    )

    assert profile.campaign_schedule_contract(
        arguments.iterations, arguments.burn_in_blocks
    ) == {
        "kind": kind,
        "iterations": iterations,
        "burn_in_blocks": burn_in_blocks,
        "conversation_count": conversation_count,
    }


@pytest.mark.parametrize(
    ("iterations", "burn_in_blocks"),
    ((30, 0), (1, 0), (30, 1), (2, 1), (29, 5), (31, 5), (1, 5)),
)
def test_campaign_acquisition_rejects_unregistered_schedule_shapes(
    tmp_path: Path, iterations: int, burn_in_blocks: int
) -> None:
    with pytest.raises(SystemExit):
        profile.parse_arguments(
            [
                "--endpoint",
                "http://127.0.0.1:9099",
                "--model",
                "fixture.gguf",
                "--campaign-root",
                str(tmp_path),
                "--iterations",
                str(iterations),
                "--burn-in-blocks",
                str(burn_in_blocks),
                "--candidate-sha",
                profile.CANDIDATE_SHA,
            ]
        )


def test_legacy_output_root_rejects_campaign_burn_in(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        profile.parse_arguments(
            [
                "--endpoint",
                "http://127.0.0.1:9099",
                "--model",
                "fixture.gguf",
                "--output-root",
                str(tmp_path),
                "--burn-in-blocks",
                "1",
                "--candidate-sha",
                profile.CANDIDATE_SHA,
            ]
        )


@pytest.mark.parametrize(
    ("iterations", "burn_in_blocks"), ((30, 5), (1, 1))
)
def test_confirmatory_child_cli_accepts_parent_burn_in(
    tmp_path: Path, iterations: int, burn_in_blocks: int
) -> None:
    child_spec = tmp_path / "child-spec.json"

    arguments = profile.parse_arguments(
        [
            "--endpoint",
            "http://127.0.0.1:9099",
            "--model",
            "fixture.gguf",
            "--output-root",
            str(tmp_path),
            "--iterations",
            str(iterations),
            "--burn-in-blocks",
            str(burn_in_blocks),
            "--child-spec",
            str(child_spec),
        ]
    )

    assert arguments.child_spec == child_spec
    assert arguments.burn_in_blocks == burn_in_blocks


@pytest.mark.parametrize(
    ("iterations", "burn_in_blocks"), ((30, 5), (1, 1))
)
def test_real_confirmatory_child_command_reaches_child_mode_after_parse(
    tmp_path: Path, iterations: int, burn_in_blocks: int
) -> None:
    run_root = tmp_path / "run"
    target_root = tmp_path / "empty-target"
    sample_root = run_root / "samples" / "parse-check"
    raw_path = run_root / "raw.jsonl"
    run_root.mkdir()
    target_root.mkdir()
    args = SimpleNamespace(
        endpoint="http://127.0.0.1:9099",
        model="fixture.gguf",
        iterations=iterations,
        burn_in_blocks=burn_in_blocks,
        sample_timeout=5.0,
    )
    spec = {
        "sample_id": "parse-check",
        "phase": "warmup",
        "iteration": -1,
        "arm": "control",
        "schedule_position": 0,
        "target_root": str(target_root),
        "sample_root": str(sample_root),
        "run_root": str(run_root),
        "evidence_path": str(raw_path),
    }

    result = profile._run_parent_child(
        args,
        runner=Path(profile.__file__).resolve(),
        run_root=run_root,
        raw_path=raw_path,
        revisions={
            "control": profile.CONTROL_SHA,
            "candidate": profile.CANDIDATE_SHA,
        },
        target_root=target_root,
        sample_root=sample_root,
        spec=spec,
    )

    assert result.status == "failed"
    assert result.returncode == 1
    assert result.last_event is not None
    assert result.last_event["event"] == "child_failure"


@pytest.mark.parametrize(
    ("action", "extra"),
    [
        ("recover", []),
        ("digest", ["--attempt-id", "attempt-0001"]),
        (
            "register-review",
            [
                "--attempt-id",
                "attempt-0001",
                "--review-receipt",
                "receipt.json",
            ],
        ),
        (
            "reopen-review",
            [
                "--attempt-id",
                "attempt-0001",
                "--review-receipt",
                "receipt.json",
                "--correction-id",
                "correction-001",
            ],
        ),
        (
            "promote",
            [
                "--attempt-id",
                "attempt-0001",
                "--review-receipt",
                "receipt.json",
                "--destination",
                "published",
            ],
        ),
    ],
)
def test_campaign_actions_parse_without_provider_contact_arguments(
    tmp_path: Path, action: str, extra: list[str]
) -> None:
    arguments = profile.parse_arguments(
        [
            "--campaign-action",
            action,
            "--campaign-root",
            str(tmp_path),
            *extra,
        ]
    )

    assert arguments.campaign_action == action
    assert arguments.endpoint is None
    assert arguments.model is None


def test_reopen_review_cli_dispatches_without_provider_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = tmp_path / "campaign/attempts/attempt-0001/reviews/review-002.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    observed: list[tuple[Path, str, Path, str]] = []
    monkeypatch.setattr(
        profile,
        "reopen_review_receipt",
        lambda campaign, attempt, review, *, correction_id: observed.append(
            (campaign, attempt, review, correction_id)
        )
        or {"artifact_set_sha256": "a" * 64, "decision": "changes_required"},
    )
    monkeypatch.setattr(
        profile,
        "preflight_provider",
        lambda *_args: pytest.fail("maintenance action contacted provider"),
    )

    assert profile.main(
        [
            "--campaign-action",
            "reopen-review",
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--attempt-id",
            "attempt-0001",
            "--review-receipt",
            str(receipt),
            "--correction-id",
            "correction-001",
        ]
    ) == 0
    assert observed == [
        (tmp_path / "campaign", "attempt-0001", receipt, "correction-001")
    ]
    assert json.loads(capsys.readouterr().out) == {
        "artifact_set_sha256": "a" * 64,
        "attempt_id": "attempt-0001",
        "correction_id": "correction-001",
        "decision": "changes_required",
        "event": "review_reopened",
    }


def _write_reviewed_artifacts(root: Path) -> dict[str, bytes]:
    root.mkdir(parents=True, exist_ok=True)
    values = {
        "README.md": b"# Confirmation\n",
        "real-provider-three-turn.raw.jsonl": b'{"event":"sample"}\n',
        "real-provider-three-turn.manifest.json": b"{}\n",
        "real-provider-three-turn.summary.json": b'{"overall_verdict":"pass"}\n',
        "real-provider-three-turn-summary.md": b"# Result\n",
    }
    for name, value in values.items():
        (root / name).write_bytes(value)
    return values


def test_canonical_artifact_digest_hashes_exact_sorted_five_file_map(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    values = _write_reviewed_artifacts(root)
    expected_map = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in sorted(values.items())
    }
    expected = hashlib.sha256(
        json.dumps(expected_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert profile.canonical_artifact_digest(root) == expected


@pytest.mark.parametrize("problem", ("missing", "extra", "symlink"))
def test_canonical_artifact_digest_rejects_noncanonical_set(
    tmp_path: Path, problem: str
) -> None:
    root = tmp_path / "attempt"
    _write_reviewed_artifacts(root)
    if problem == "missing":
        (root / "README.md").unlink()
    elif problem == "extra":
        (root / "notes.txt").write_text("extra", encoding="utf-8")
    else:
        (root / "README.md").unlink()
        (root / "README.md").symlink_to(root / "real-provider-three-turn-summary.md")

    with pytest.raises(RuntimeError, match="^review_artifact_set_invalid$"):
        profile.canonical_artifact_digest(root)


def test_canonical_artifact_digest_allows_receipts_outside_reviewed_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    _write_reviewed_artifacts(root)
    before = profile.canonical_artifact_digest(root)
    reviews = root / "reviews"
    reviews.mkdir()
    (reviews / "review-001.json").write_text("{}\n", encoding="utf-8")

    assert profile.canonical_artifact_digest(root) == before


def _write_empty_acquisition_namespaces(root: Path) -> None:
    (root / "control").mkdir()
    (root / "candidate").mkdir()
    samples = root / "samples"
    samples.mkdir()
    (samples / "000-warmup--1-control").mkdir()
    nested = samples / "001-measured-0-enabled"
    nested.mkdir()
    (nested / "retained-empty-tombstone").mkdir()


def test_canonical_artifact_digest_excludes_empty_acquisition_namespaces(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    values = _write_reviewed_artifacts(root)
    expected_hashes = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in sorted(values.items())
    }
    expected_digest = profile.canonical_artifact_digest(root)
    _write_empty_acquisition_namespaces(root)

    assert profile.canonical_artifact_hashes(root) == expected_hashes
    assert profile.canonical_artifact_digest(root) == expected_digest


@pytest.mark.parametrize(
    "problem",
    (
        "namespace_file",
        "namespace_symlink",
        "nested_file",
        "nested_symlink",
        "unknown_directory",
    ),
)
def test_canonical_artifact_digest_rejects_invalid_acquisition_namespaces(
    tmp_path: Path, problem: str
) -> None:
    root = tmp_path / "attempt"
    _write_reviewed_artifacts(root)
    if problem == "namespace_file":
        (root / "samples").write_text("not a directory", encoding="utf-8")
    elif problem == "namespace_symlink":
        (root / "samples").symlink_to(root / "reviews", target_is_directory=True)
    elif problem == "nested_file":
        (root / "samples").mkdir()
        (root / "samples" / "residue.txt").write_text("residue", encoding="utf-8")
    elif problem == "nested_symlink":
        (root / "samples").mkdir()
        (root / "samples" / "residue").symlink_to(root / "README.md")
    else:
        (root / "unexpected").mkdir()

    with pytest.raises(RuntimeError, match="^review_artifact_set_invalid$"):
        profile.canonical_artifact_digest(root)


def test_canonical_artifact_hashes_reject_absolute_artifact_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    _write_reviewed_artifacts(root)

    with pytest.raises(RuntimeError, match="^review_artifact_path_invalid$"):
        profile.canonical_artifact_hashes(
            root, [*(REVIEWED_ARTIFACT_NAMES - {"README.md"}), root / "README.md"]
        )


def _prepare_review_attempt(
    tmp_path: Path, *, verdict: str = "pass"
) -> tuple[Path, Path, str, str]:
    campaign = tmp_path / "campaign"
    attempt = profile.create_attempt_root(campaign, "attempt-0001")
    values = _write_reviewed_artifacts(attempt)
    raw_sha256 = hashlib.sha256(
        values["real-provider-three-turn.raw.jsonl"]
    ).hexdigest()
    profile.append_attempt_state(
        campaign / "attempts.jsonl",
        {"attempt_id": "attempt-0001", "state": "running"},
    )
    profile.complete_attempt_measurement(
        campaign / "attempts.jsonl",
        "attempt-0001",
        verdict=verdict,
        raw_sha256=raw_sha256,
    )
    return campaign, attempt, profile.canonical_artifact_digest(attempt), raw_sha256


def _write_review_receipt(
    artifact_root: Path,
    digest: str,
    *,
    decision: str = "approved",
    filename: str = "review-001.json",
    verdict: str = "pass",
    findings: list[dict[str, str]] | None = None,
) -> Path:
    if findings is None:
        findings = (
            []
            if decision == "approved"
            else [{"severity": "important", "code": "report_verdict_mismatch"}]
        )
    receipt = {
        "artifact_set_sha256": digest,
        "attempt_id": "attempt-0001",
        "decision": decision,
        "findings": findings,
        "privacy_confirmed": True,
        "reviewed_at": "2026-08-23T00:00:00Z",
        "reviewer": "independent-reviewer",
        "verdict": verdict,
    }
    reviews = artifact_root / "reviews"
    reviews.mkdir(exist_ok=True)
    path = reviews / filename
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _approve_review_attempt(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str, str]:
    campaign, attempt, digest, raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    profile.register_review_receipt(campaign, "attempt-0001", receipt)
    return campaign, attempt, receipt, digest, raw_sha256


def _reopen_approved_attempt(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str, str]:
    campaign, attempt, approved, digest, raw_sha256 = _approve_review_attempt(tmp_path)
    rejected = _write_review_receipt(
        attempt,
        digest,
        decision="changes_required",
        filename="review-002.json",
        findings=[
            {"severity": "important", "code": "implementation_base_revision_missing"}
        ],
    )
    profile.reopen_review_receipt(
        campaign,
        "attempt-0001",
        rejected,
        correction_id="correction-001",
    )
    return campaign, attempt, approved, digest, raw_sha256


def test_implementation_base_revision_resolves_only_the_fixed_ref(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="77c5e9f487af79391a479deb85e712163bfed909\n",
            stderr="",
        )

    assert profile.resolve_implementation_base_revision(
        tmp_path, run_command=run
    ) == "77c5e9f487af79391a479deb85e712163bfed909"
    assert calls == [
        [
            "git",
            "rev-parse",
            "--verify",
            "refs/benchmarks/task-20010-implementation-base^{commit}",
        ]
    ]


@pytest.mark.parametrize(
    ("returncode", "stdout", "code"),
    (
        (1, "", "implementation_base_revision_failed"),
        (0, "f" * 40, "implementation_base_revision_mismatch"),
    ),
)
def test_implementation_base_revision_fails_closed_on_missing_or_drifted_ref(
    tmp_path: Path, returncode: int, stdout: str, code: str
) -> None:
    def run(_command, **_kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError, match=f"^{code}$"):
        profile.resolve_implementation_base_revision(tmp_path, run_command=run)


def test_reopen_review_registers_later_rejection_and_correction_identity(
    tmp_path: Path,
) -> None:
    campaign, attempt, approved, digest, raw_sha256 = _approve_review_attempt(tmp_path)
    rejected = _write_review_receipt(
        attempt,
        digest,
        decision="changes_required",
        filename="review-002.json",
        findings=[
            {"severity": "important", "code": "implementation_base_revision_missing"}
        ],
    )

    receipt = profile.reopen_review_receipt(
        campaign,
        "attempt-0001",
        rejected,
        correction_id="correction-001",
    )

    assert receipt["decision"] == "changes_required"
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == {
        "attempt_id": "attempt-0001",
        "state": "changes_required",
        "verdict": "pass",
        "raw_sha256": raw_sha256,
        "reason_category": "manifest",
        "correction_id": "correction-001",
    }
    registry = attempt / "reviews" / ".registered"
    assert (
        registry
        / "reviews"
        / f"review-002.json--{hashlib.sha256(rejected.read_bytes()).hexdigest()}"
    ).is_file()
    assert approved.read_bytes() == (attempt / "reviews" / "review-001.json").read_bytes()


@pytest.mark.parametrize(
    ("problem", "code"),
    (
        ("no_approval", "review_reopen_prior_approval_required"),
        ("older_number", "review_reopen_receipt_order_invalid"),
        ("approved", "review_reopen_receipt_decision_invalid"),
        ("digest", "review_receipt_binding_mismatch"),
        ("raw", "review_receipt_binding_mismatch"),
        ("verdict", "review_receipt_binding_mismatch"),
        ("attempt", "review_receipt_binding_mismatch"),
        ("foreign", "review_receipt_location_invalid"),
        ("repeated", "review_attempt_already_reopened"),
    ),
)
def test_reopen_review_rejects_ambiguous_or_unbound_transition(
    tmp_path: Path, problem: str, code: str
) -> None:
    campaign, attempt, _approved, digest, _raw_sha256 = _approve_review_attempt(
        tmp_path
    )
    if problem == "no_approval":
        registry = attempt / "reviews" / ".registered"
        for marker in registry.iterdir():
            marker.unlink()
        registry.rmdir()
    filename = "review-000.json" if problem == "older_number" else "review-002.json"
    rejected = _write_review_receipt(
        attempt,
        "f" * 64 if problem == "digest" else digest,
        decision="approved" if problem == "approved" else "changes_required",
        filename=filename,
        verdict="regression" if problem == "verdict" else "pass",
    )
    if problem == "attempt":
        payload = json.loads(rejected.read_bytes())
        payload["attempt_id"] = "attempt-0002"
        rejected.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    if problem == "raw":
        lineage = (campaign / "attempts.jsonl").read_text(encoding="utf-8")
        (campaign / "attempts.jsonl").write_text(
            lineage.replace(
                hashlib.sha256(
                    (attempt / "real-provider-three-turn.raw.jsonl").read_bytes()
                ).hexdigest(),
                "e" * 64,
            ),
            encoding="utf-8",
        )
    if problem == "foreign":
        foreign = tmp_path / "review-002.json"
        foreign.write_bytes(rejected.read_bytes())
        rejected = foreign
    if problem == "repeated":
        profile.reopen_review_receipt(
            campaign,
            "attempt-0001",
            rejected,
            correction_id="correction-001",
        )

    with pytest.raises(RuntimeError, match=f"^{code}$"):
        profile.reopen_review_receipt(
            campaign,
            "attempt-0001",
            rejected,
            correction_id="correction-001",
        )


@pytest.mark.parametrize("name", sorted(REVIEWED_ARTIFACT_NAMES))
def test_manifest_correction_rejects_any_change_to_original_approved_files(
    tmp_path: Path, name: str
) -> None:
    campaign, attempt, _approved, _digest, _raw_sha256 = _reopen_approved_attempt(
        tmp_path
    )
    path = attempt / name
    path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match="^review_correction_original_changed$"):
        profile.prepare_manifest_correction(
            campaign,
            "attempt-0001",
            correction_id="correction-001",
            implementation_base_revision=profile.IMPLEMENTATION_BASE_SHA,
        )

    assert not (attempt / "corrections" / "correction-001").exists()
    assert not (attempt / "corrections" / ".correction-001-stage").exists()


def test_review_registry_rejects_duplicate_numeric_identity_across_roots(
    tmp_path: Path,
) -> None:
    _campaign, attempt, _approved, _digest, _raw_sha256 = _approve_review_attempt(
        tmp_path
    )
    correction = attempt / "corrections" / "correction-001"
    _write_reviewed_artifacts(correction)
    receipt = _write_review_receipt(
        correction,
        profile.canonical_artifact_digest(correction),
        filename="review-001.json",
    )
    marker = (
        attempt
        / "reviews"
        / ".registered"
        / "corrections"
        / "correction-001"
        / "reviews"
        / f"review-001.json--{hashlib.sha256(receipt.read_bytes()).hexdigest()}"
    )
    marker.parent.mkdir(parents=True)
    marker.touch()

    with pytest.raises(RuntimeError, match="^review_receipt_registry_invalid$"):
        profile._registered_review_receipts(attempt)


def test_reopen_review_rejects_lexical_path_traversal(
    tmp_path: Path,
) -> None:
    campaign, attempt, _approved, digest, _raw_sha256 = _approve_review_attempt(
        tmp_path
    )
    _write_review_receipt(
        attempt,
        digest,
        decision="changes_required",
        filename="review-002.json",
    )
    traversed = (
        attempt
        / "corrections"
        / "correction-001"
        / ".."
        / ".."
        / "reviews"
        / "review-002.json"
    )

    with pytest.raises(RuntimeError, match="^review_receipt_location_invalid$"):
        profile.reopen_review_receipt(
            campaign,
            "attempt-0001",
            traversed,
            correction_id="correction-001",
        )


def test_receipt_location_rejects_symlinked_correction_root(tmp_path: Path) -> None:
    campaign, attempt, _approved, _digest, _raw_sha256 = _approve_review_attempt(
        tmp_path
    )
    foreign = tmp_path / "foreign-correction"
    _write_reviewed_artifacts(foreign)
    receipt = _write_review_receipt(
        foreign,
        profile.canonical_artifact_digest(foreign),
        filename="review-003.json",
    )
    corrections = attempt / "corrections"
    corrections.mkdir()
    (corrections / "correction-001").symlink_to(foreign, target_is_directory=True)
    aliased = corrections / "correction-001" / "reviews" / receipt.name

    with pytest.raises(RuntimeError, match="^review_receipt_location_invalid$"):
        profile._validate_receipt_location(attempt, aliased)


def test_manifest_correction_rename_failure_leaves_no_partial_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, _approved, _digest, _raw_sha256 = _reopen_approved_attempt(
        tmp_path
    )
    original = {
        name: (attempt / name).read_bytes() for name in REVIEWED_ARTIFACT_NAMES
    }
    monkeypatch.setattr(
        profile,
        "_atomic_rename_directory_noreplace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected_correction_rename_failure")
        ),
    )

    with pytest.raises(RuntimeError, match="^injected_correction_rename_failure$"):
        profile.prepare_manifest_correction(
            campaign,
            "attempt-0001",
            correction_id="correction-001",
            implementation_base_revision=profile.IMPLEMENTATION_BASE_SHA,
        )

    assert not (attempt / "corrections" / "correction-001").exists()
    assert not (attempt / "corrections" / ".correction-001-stage").exists()
    assert all((attempt / name).read_bytes() == data for name, data in original.items())


@pytest.mark.parametrize(
    ("name", "code"),
    tuple(
        (
            name,
            "review_correction_manifest_invalid"
            if name == "real-provider-three-turn.manifest.json"
            else "review_correction_acquisition_mismatch",
        )
        for name in sorted(REVIEWED_ARTIFACT_NAMES)
    ),
)
def test_registered_correction_rejects_later_artifact_changes(
    tmp_path: Path, name: str, code: str
) -> None:
    campaign, _attempt, _approved, _digest, _raw_sha256 = _reopen_approved_attempt(
        tmp_path
    )
    correction = profile.prepare_manifest_correction(
        campaign,
        "attempt-0001",
        correction_id="correction-001",
        implementation_base_revision=profile.IMPLEMENTATION_BASE_SHA,
    )
    receipt = _write_review_receipt(
        correction,
        profile.canonical_artifact_digest(correction),
        filename="review-003.json",
    )
    profile.register_review_receipt(campaign, "attempt-0001", receipt)
    path = correction / name
    path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match=f"^{code}$"):
        profile.register_review_receipt(campaign, "attempt-0001", receipt)


def test_correction_approval_requires_a_still_later_review_number(
    tmp_path: Path,
) -> None:
    campaign, _attempt, _approved, _digest, _raw_sha256 = _reopen_approved_attempt(
        tmp_path
    )
    correction = profile.prepare_manifest_correction(
        campaign,
        "attempt-0001",
        correction_id="correction-001",
        implementation_base_revision=profile.IMPLEMENTATION_BASE_SHA,
    )
    stale = _write_review_receipt(
        correction,
        profile.canonical_artifact_digest(correction),
        filename="review-002.json",
    )

    with pytest.raises(RuntimeError, match="^review_receipt_order_invalid$"):
        profile.register_review_receipt(campaign, "attempt-0001", stale)


def test_manifest_correction_post_rename_failure_retains_only_complete_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, _approved, _digest, _raw_sha256 = _reopen_approved_attempt(
        tmp_path
    )
    real_fsync_directory = profile._fsync_directory
    corrections = attempt / "corrections"

    def fail_after_rename(path: Path) -> None:
        if path == corrections and (corrections / "correction-001").is_dir():
            raise OSError("injected_post_rename_fsync_failure")
        real_fsync_directory(path)

    monkeypatch.setattr(profile, "_fsync_directory", fail_after_rename)

    with pytest.raises(OSError, match="injected_post_rename_fsync_failure"):
        profile.prepare_manifest_correction(
            campaign,
            "attempt-0001",
            correction_id="correction-001",
            implementation_base_revision=profile.IMPLEMENTATION_BASE_SHA,
        )

    correction = corrections / "correction-001"
    assert correction.is_dir()
    assert not (corrections / ".correction-001-stage").exists()
    profile._validate_correction_contents(attempt, correction)


def test_versioned_manifest_correction_preserves_original_and_promotes_by_path(
    tmp_path: Path,
) -> None:
    campaign, attempt, approved, digest, raw_sha256 = _approve_review_attempt(tmp_path)
    original = {
        name: (attempt / name).read_bytes() for name in REVIEWED_ARTIFACT_NAMES
    }
    approved_bytes = approved.read_bytes()
    rejected = _write_review_receipt(
        attempt,
        digest,
        decision="changes_required",
        filename="review-002.json",
        findings=[
            {"severity": "important", "code": "implementation_base_revision_missing"}
        ],
    )
    profile.reopen_review_receipt(
        campaign,
        "attempt-0001",
        rejected,
        correction_id="correction-001",
    )

    correction = profile.prepare_manifest_correction(
        campaign,
        "attempt-0001",
        correction_id="correction-001",
        implementation_base_revision="77c5e9f487af79391a479deb85e712163bfed909",
    )

    assert correction == attempt / "corrections" / "correction-001"
    assert all(
        (attempt / name).read_bytes() == value for name, value in original.items()
    )
    assert approved.read_bytes() == approved_bytes
    for name in REVIEWED_ARTIFACT_NAMES - {"real-provider-three-turn.manifest.json"}:
        assert (correction / name).read_bytes() == original[name]
    corrected_manifest = json.loads(
        (correction / "real-provider-three-turn.manifest.json").read_bytes()
    )
    assert corrected_manifest["implementation_base_revision"] == (
        "77c5e9f487af79391a479deb85e712163bfed909"
    )
    corrected_digest = profile.canonical_artifact_digest(correction)
    assert corrected_digest != digest
    approved_correction = _write_review_receipt(
        correction,
        corrected_digest,
        filename="review-003.json",
    )

    profile.register_review_receipt(
        campaign, "attempt-0001", approved_correction
    )

    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == {
        "attempt_id": "attempt-0001",
        "state": "complete_pending_review",
        "verdict": "pass",
        "raw_sha256": raw_sha256,
    }
    marker = (
        attempt
        / "reviews"
        / ".registered"
        / "corrections"
        / "correction-001"
        / "reviews"
        / f"review-003.json--{hashlib.sha256(approved_correction.read_bytes()).hexdigest()}"
    )
    assert marker.is_file()
    destination = tmp_path / "published-correction"
    profile.promote_reviewed_artifacts(
        campaign,
        "attempt-0001",
        approved_correction,
        destination,
    )
    assert profile.canonical_artifact_digest(destination) == corrected_digest
    assert (destination / "real-provider-three-turn.raw.jsonl").read_bytes() == (
        original["real-provider-three-turn.raw.jsonl"]
    )


def test_correction_approval_rejects_wrong_root_or_unchanged_digest(
    tmp_path: Path,
) -> None:
    campaign, attempt, _approved, digest, _raw_sha256 = _approve_review_attempt(
        tmp_path
    )
    rejected = _write_review_receipt(
        attempt,
        digest,
        decision="changes_required",
        filename="review-002.json",
    )
    profile.reopen_review_receipt(
        campaign,
        "attempt-0001",
        rejected,
        correction_id="correction-001",
    )
    wrong = attempt / "corrections" / "correction-002"
    _write_reviewed_artifacts(wrong)
    receipt = _write_review_receipt(wrong, profile.canonical_artifact_digest(wrong), filename="review-003.json")

    with pytest.raises(RuntimeError, match="^review_correction_identity_mismatch$"):
        profile.register_review_receipt(campaign, "attempt-0001", receipt)


def test_review_receipt_has_exact_privacy_safe_schema(tmp_path: Path) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt_path = _write_review_receipt(attempt, digest)

    receipt = profile.register_review_receipt(campaign, "attempt-0001", receipt_path)

    assert set(receipt) == {
        "artifact_set_sha256",
        "attempt_id",
        "decision",
        "findings",
        "privacy_confirmed",
        "reviewed_at",
        "reviewer",
        "verdict",
    }
    assert receipt["decision"] == "approved"
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1]["state"] == (
        "complete_pending_review"
    )


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        ({"unknown": True}, "review_receipt_fields_invalid"),
        ({"api_key": "secret"}, "review_receipt_privacy_violation"),
        ({"privacy_confirmed": False}, "review_receipt_privacy_invalid"),
        ({"reviewer": "/Users/person"}, "review_receipt_privacy_violation"),
        (
            {"findings": [{"severity": "important", "code": "raw secret value"}]},
            "review_receipt_findings_invalid",
        ),
    ),
)
def test_review_receipt_rejects_unknown_or_sensitive_fields(
    tmp_path: Path, mutation: dict[str, object], code: str
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt_path = _write_review_receipt(attempt, digest)
    receipt = json.loads(receipt_path.read_bytes())
    receipt.update(mutation)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=f"^{code}$"):
        profile.register_review_receipt(campaign, "attempt-0001", receipt_path)


@pytest.mark.parametrize("field", ("attempt_id", "artifact_set_sha256", "verdict"))
def test_review_registration_rejects_attempt_digest_or_verdict_mismatch(
    tmp_path: Path, field: str
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt_path = _write_review_receipt(attempt, digest)
    receipt = json.loads(receipt_path.read_bytes())
    receipt[field] = {
        "attempt_id": "attempt-0002",
        "artifact_set_sha256": "f" * 64,
        "verdict": "regression",
    }[field]
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="^review_receipt_binding_mismatch$"):
        profile.register_review_receipt(campaign, "attempt-0001", receipt_path)


def test_changes_required_preserves_raw_and_blocks_reacquisition(
    tmp_path: Path,
) -> None:
    campaign, attempt, digest, raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt_path = _write_review_receipt(attempt, digest, decision="changes_required")

    profile.register_review_receipt(campaign, "attempt-0001", receipt_path)

    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == {
        "attempt_id": "attempt-0001",
        "state": "changes_required",
        "verdict": "pass",
        "raw_sha256": raw_sha256,
        "reason_category": "receipt",
    }
    with pytest.raises(
        RuntimeError, match="campaign_acquisition_blocked:changes_required"
    ):
        profile.require_campaign_acquisition(campaign / "attempts.jsonl")


@pytest.mark.parametrize("append_committed", (False, True))
def test_registered_changes_receipt_reconciles_append_failure_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    append_committed: bool,
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest, decision="changes_required")
    real_append = profile.append_attempt_state

    def fail_changes_append(ledger: Path, event: dict[str, object]) -> None:
        if event["state"] == "changes_required":
            if append_committed:
                real_append(ledger, event)
            raise OSError("injected changes append failure")
        real_append(ledger, event)

    monkeypatch.setattr(profile, "append_attempt_state", fail_changes_append)
    with pytest.raises(OSError, match="injected changes append failure"):
        profile.register_review_receipt(campaign, "attempt-0001", receipt)

    assert (attempt / "reviews" / ".registered").is_dir()
    monkeypatch.setattr(profile, "append_attempt_state", real_append)
    assert profile.register_review_receipt(
        campaign, "attempt-0001", receipt
    )["decision"] == "changes_required"
    assert [
        event["state"] for event in profile.attempt_lineage(campaign / "attempts.jsonl")
    ] == ["running", "complete_pending_review", "changes_required"]


def test_registered_rejection_blocks_unchanged_approval_and_promotion_after_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    rejected = _write_review_receipt(
        attempt, digest, decision="changes_required", filename="review-001.json"
    )
    real_append = profile.append_attempt_state

    def fail_changes_append(ledger: Path, event: dict[str, object]) -> None:
        if event["state"] == "changes_required":
            raise OSError("injected changes append failure")
        real_append(ledger, event)

    monkeypatch.setattr(profile, "append_attempt_state", fail_changes_append)
    with pytest.raises(OSError, match="injected changes append failure"):
        profile.register_review_receipt(campaign, "attempt-0001", rejected)
    monkeypatch.setattr(profile, "append_attempt_state", real_append)

    unchanged = _write_review_receipt(attempt, digest, filename="review-002.json")
    with pytest.raises(RuntimeError, match="^review_artifact_digest_not_changed$"):
        profile.register_review_receipt(campaign, "attempt-0001", unchanged)
    with pytest.raises(RuntimeError, match="^review_artifact_digest_not_changed$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", unchanged, tmp_path / "published"
        )

    assert not (tmp_path / "published").exists()
    assert [
        event["state"] for event in profile.attempt_lineage(campaign / "attempts.jsonl")
    ] == ["running", "complete_pending_review", "changes_required"]

    (attempt / "README.md").write_text("# Corrected after crash\n", encoding="utf-8")
    corrected_digest = profile.canonical_artifact_digest(attempt)
    corrected = _write_review_receipt(
        attempt, corrected_digest, filename="review-003.json"
    )
    corrected_destination = tmp_path / "published-corrected"
    profile.promote_reviewed_artifacts(
        campaign, "attempt-0001", corrected, corrected_destination
    )

    assert corrected_destination.is_dir()
    assert [
        event["state"] for event in profile.attempt_lineage(campaign / "attempts.jsonl")
    ] == [
        "running",
        "complete_pending_review",
        "changes_required",
        "complete_pending_review",
    ]


def test_dead_maintenance_lock_after_changes_append_recovers_without_ledger_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest, decision="changes_required")
    real_release = profile.release_campaign_lock

    monkeypatch.setattr(
        profile,
        "release_campaign_lock",
        lambda _root, _owner: (_ for _ in ()).throw(
            OSError("injected maintenance release crash")
        ),
    )
    with pytest.raises(OSError, match="injected maintenance release crash"):
        profile.register_review_receipt(campaign, "attempt-0001", receipt)
    before = (campaign / "attempts.jsonl").read_bytes()
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1]["state"] == (
        "changes_required"
    )
    assert (campaign / ".campaign-lock").is_dir()

    monkeypatch.setattr(profile, "release_campaign_lock", real_release)
    recovered = profile.recover_interrupted_attempt(
        campaign, process_start_probe=lambda _pid: None
    )

    assert recovered["state"] == "changes_required"
    assert (campaign / "attempts.jsonl").read_bytes() == before
    assert not (campaign / ".campaign-lock").exists()


def test_corrected_derived_artifacts_require_new_digest_and_receipt(
    tmp_path: Path,
) -> None:
    campaign, attempt, digest, raw_sha256 = _prepare_review_attempt(tmp_path)
    rejected = _write_review_receipt(attempt, digest, decision="changes_required")
    profile.register_review_receipt(campaign, "attempt-0001", rejected)

    unchanged = _write_review_receipt(attempt, digest, filename="review-002.json")
    with pytest.raises(RuntimeError, match="^review_artifact_digest_not_changed$"):
        profile.register_review_receipt(campaign, "attempt-0001", unchanged)

    (attempt / "README.md").write_text("# Corrected confirmation\n", encoding="utf-8")
    corrected_digest = profile.canonical_artifact_digest(attempt)
    approved = _write_review_receipt(
        attempt, corrected_digest, filename="review-003.json"
    )
    profile.register_review_receipt(campaign, "attempt-0001", approved)

    assert (
        hashlib.sha256(
            (attempt / "real-provider-three-turn.raw.jsonl").read_bytes()
        ).hexdigest()
        == raw_sha256
    )
    lineage = profile.attempt_lineage(campaign / "attempts.jsonl")
    assert [event["state"] for event in lineage] == [
        "running",
        "complete_pending_review",
        "changes_required",
        "complete_pending_review",
    ]
    assert lineage[-1]["raw_sha256"] == raw_sha256


def test_registered_receipt_is_immutable(tmp_path: Path) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    profile.register_review_receipt(campaign, "attempt-0001", receipt)
    payload = json.loads(receipt.read_bytes())
    payload["reviewed_at"] = "2026-08-23T00:00:01Z"
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="^review_receipt_changed$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, tmp_path / "published"
        )


def test_smoke_receipt_cannot_be_registered_or_promoted(tmp_path: Path) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(
        tmp_path, verdict="smoke"
    )
    receipt = _write_review_receipt(attempt, digest, verdict="smoke")
    artifact_bytes = {
        name: (attempt / name).read_bytes() for name in REVIEWED_ARTIFACT_NAMES
    }

    with pytest.raises(RuntimeError, match="^review_attempt_verdict_invalid$"):
        profile.register_review_receipt(campaign, "attempt-0001", receipt)
    with pytest.raises(RuntimeError, match="^review_attempt_verdict_invalid$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, tmp_path / "published"
        )

    assert not (tmp_path / "published").exists()
    assert {
        name: (attempt / name).read_bytes() for name in REVIEWED_ARTIFACT_NAMES
    } == artifact_bytes
    assert not (attempt / "reviews" / ".registered").exists()


@pytest.mark.parametrize("severity", ("critical", "important"))
def test_approved_receipt_rejects_blocking_findings_at_every_entrypoint(
    tmp_path: Path, severity: str
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(
        attempt,
        digest,
        findings=[{"severity": severity, "code": "evidence_mismatch"}],
    )

    with pytest.raises(
        RuntimeError, match="^review_receipt_approval_findings_invalid$"
    ):
        profile._read_review_receipt(receipt)
    with pytest.raises(
        RuntimeError, match="^review_receipt_approval_findings_invalid$"
    ):
        profile.register_review_receipt(campaign, "attempt-0001", receipt)
    with pytest.raises(
        RuntimeError, match="^review_receipt_approval_findings_invalid$"
    ):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, tmp_path / "published"
        )


def test_approved_receipt_allows_only_minor_findings(tmp_path: Path) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(
        attempt,
        digest,
        findings=[{"severity": "minor", "code": "wording_polish"}],
    )

    assert profile._read_review_receipt(receipt)["findings"] == [
        {"severity": "minor", "code": "wording_polish"}
    ]
    profile.register_review_receipt(campaign, "attempt-0001", receipt)
    profile.promote_reviewed_artifacts(
        campaign, "attempt-0001", receipt, tmp_path / "published"
    )

    assert (tmp_path / "published").is_dir()


@pytest.mark.parametrize("later_receipt", (False, True))
def test_receipt_bytes_and_reviews_directory_are_durable_before_identity_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, later_receipt: bool
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    if later_receipt:
        rejected = _write_review_receipt(
            attempt,
            digest,
            decision="changes_required",
            filename="review-001.json",
        )
        profile.register_review_receipt(campaign, "attempt-0001", rejected)
        (attempt / "README.md").write_text("# Corrected\n", encoding="utf-8")
        digest = profile.canonical_artifact_digest(attempt)
    filename = "review-002.json" if later_receipt else "review-001.json"
    receipt = _write_review_receipt(attempt, digest, filename=filename)
    reviews = attempt / "reviews"
    registry = reviews / ".registered"
    marker_name = f"{filename}--{hashlib.sha256(receipt.read_bytes()).hexdigest()}"
    marker = registry / marker_name
    events: list[str] = []
    real_file_fsync = profile._fsync_regular_file
    real_directory_fsync = profile._fsync_directory

    def record_file(path: Path) -> None:
        if path == receipt:
            events.append("receipt")
        elif path == marker:
            events.append("marker")
        real_file_fsync(path)

    def record_directory(path: Path) -> None:
        if path == reviews:
            events.append("reviews")
        elif path == registry:
            events.append("registry")
        real_directory_fsync(path)

    monkeypatch.setattr(profile, "_fsync_regular_file", record_file)
    monkeypatch.setattr(profile, "_fsync_directory", record_directory)

    profile.register_review_receipt(campaign, "attempt-0001", receipt)

    assert events[:2] == ["receipt", "reviews"]
    assert events[-2:] == ["marker", "registry"]
    assert marker.is_file()
    assert marker.read_bytes() == b""


@pytest.mark.parametrize("later_receipt", (False, True))
@pytest.mark.parametrize("failure", ("receipt", "reviews"))
def test_receipt_pre_marker_durability_failure_never_creates_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    later_receipt: bool,
    failure: str,
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    if later_receipt:
        rejected = _write_review_receipt(
            attempt,
            digest,
            decision="changes_required",
            filename="review-001.json",
        )
        profile.register_review_receipt(campaign, "attempt-0001", rejected)
        (attempt / "README.md").write_text("# Corrected\n", encoding="utf-8")
        digest = profile.canonical_artifact_digest(attempt)
    filename = "review-002.json" if later_receipt else "review-001.json"
    receipt = _write_review_receipt(attempt, digest, filename=filename)
    reviews = attempt / "reviews"
    registry = reviews / ".registered"
    real_file_fsync = profile._fsync_regular_file
    real_directory_fsync = profile._fsync_directory

    def fail_file(path: Path) -> None:
        if failure == "receipt" and path == receipt:
            raise OSError("injected receipt fsync")
        real_file_fsync(path)

    def fail_directory(path: Path) -> None:
        if failure == "reviews" and path == reviews:
            raise OSError("injected reviews fsync")
        real_directory_fsync(path)

    monkeypatch.setattr(profile, "_fsync_regular_file", fail_file)
    monkeypatch.setattr(profile, "_fsync_directory", fail_directory)

    with pytest.raises(RuntimeError, match="^review_receipt_durability_failed$"):
        profile.register_review_receipt(campaign, "attempt-0001", receipt)

    assert not registry.exists() or not any(
        path.name.startswith(f"{filename}--") for path in registry.iterdir()
    )


def test_receipt_changed_after_pre_marker_fsync_is_not_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    reviews = attempt / "reviews"
    real_directory_fsync = profile._fsync_directory
    mutated = False

    def mutate_after_reviews_fsync(path: Path) -> None:
        nonlocal mutated
        real_directory_fsync(path)
        if path == reviews and not mutated:
            mutated = True
            payload = json.loads(receipt.read_bytes())
            payload["reviewed_at"] = "2026-08-23T00:00:01Z"
            receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    monkeypatch.setattr(profile, "_fsync_directory", mutate_after_reviews_fsync)

    with pytest.raises(RuntimeError, match="^review_receipt_changed$"):
        profile.register_review_receipt(campaign, "attempt-0001", receipt)

    assert not (reviews / ".registered").exists()


@pytest.mark.parametrize("later_receipt", (False, True))
@pytest.mark.parametrize("decision", ("approved", "changes_required"))
def test_receipt_marker_fsync_failure_is_stable_and_preserves_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    later_receipt: bool,
    decision: str,
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    if later_receipt:
        rejected = _write_review_receipt(
            attempt,
            digest,
            decision="changes_required",
            filename="review-001.json",
        )
        profile.register_review_receipt(campaign, "attempt-0001", rejected)
        (attempt / "README.md").write_text("# Corrected\n", encoding="utf-8")
        digest = profile.canonical_artifact_digest(attempt)
    filename = "review-002.json" if later_receipt else "review-001.json"
    receipt = _write_review_receipt(
        attempt, digest, decision=decision, filename=filename
    )
    registry = attempt / "reviews" / ".registered"
    marker = registry / (
        f"{filename}--{hashlib.sha256(receipt.read_bytes()).hexdigest()}"
    )
    real_file_fsync = profile._fsync_regular_file

    def fail_marker(path: Path) -> None:
        if path == marker:
            raise OSError("injected marker fsync")
        real_file_fsync(path)

    monkeypatch.setattr(profile, "_fsync_regular_file", fail_marker)

    with pytest.raises(
        RuntimeError, match="^review_receipt_registry_durability_failed$"
    ):
        profile.register_review_receipt(campaign, "attempt-0001", receipt)

    assert marker.is_file()
    assert marker.read_bytes() == b""

    events: list[str] = []

    def record_file(path: Path) -> None:
        if path == marker:
            events.append("marker")
        real_file_fsync(path)

    real_directory_fsync = profile._fsync_directory

    def record_directory(path: Path) -> None:
        if path == registry:
            events.append("registry")
        real_directory_fsync(path)

    monkeypatch.setattr(profile, "_fsync_regular_file", record_file)
    monkeypatch.setattr(profile, "_fsync_directory", record_directory)

    assert profile.register_review_receipt(
        campaign, "attempt-0001", receipt
    )["decision"] == decision
    assert events[-2:] == ["marker", "registry"]


def test_concurrent_changes_required_registration_has_one_stable_loser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest, decision="changes_required")
    original = profile._register_review_receipt_locked
    entered = threading.Event()
    finish = threading.Event()

    def block_first(*args, **kwargs):
        entered.set()
        assert finish.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(profile, "_register_review_receipt_locked", block_first)
    with ThreadPoolExecutor(max_workers=2) as executor:
        winner = executor.submit(
            profile.register_review_receipt, campaign, "attempt-0001", receipt
        )
        assert entered.wait(timeout=5)
        loser = executor.submit(
            profile.register_review_receipt, campaign, "attempt-0001", receipt
        )
        with pytest.raises(RuntimeError, match="^campaign_lock_held$"):
            loser.result(timeout=5)
        finish.set()
        assert winner.result(timeout=5)["decision"] == "changes_required"

    assert profile.register_review_receipt(
        campaign, "attempt-0001", receipt
    )["decision"] == "changes_required"
    assert [
        event["state"] for event in profile.attempt_lineage(campaign / "attempts.jsonl")
    ] == ["running", "complete_pending_review", "changes_required"]
    assert not (campaign / ".campaign-lock").exists()


def test_changes_required_winner_serializes_against_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    approved = _write_review_receipt(attempt, digest, filename="review-001.json")
    changes = _write_review_receipt(
        attempt,
        digest,
        decision="changes_required",
        filename="review-002.json",
    )
    destination = tmp_path / "published"
    original = profile._register_review_receipt_locked
    entered = threading.Event()
    finish = threading.Event()

    def block_changes(campaign_root, attempt_id, receipt_path):
        if Path(receipt_path) == changes:
            entered.set()
            assert finish.wait(timeout=5)
        return original(campaign_root, attempt_id, receipt_path)

    monkeypatch.setattr(profile, "_register_review_receipt_locked", block_changes)
    with ThreadPoolExecutor(max_workers=2) as executor:
        changes_future = executor.submit(
            profile.register_review_receipt,
            campaign,
            "attempt-0001",
            changes,
        )
        assert entered.wait(timeout=5)
        promotion = executor.submit(
            profile.promote_reviewed_artifacts,
            campaign,
            "attempt-0001",
            approved,
            destination,
        )
        with pytest.raises(RuntimeError, match="^campaign_lock_held$"):
            promotion.result(timeout=5)
        finish.set()
        changes_future.result(timeout=5)

    assert not destination.exists()
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1]["state"] == (
        "changes_required"
    )


def test_promotion_winner_blocks_later_changes_required_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    approved = _write_review_receipt(attempt, digest, filename="review-001.json")
    changes = _write_review_receipt(
        attempt,
        digest,
        decision="changes_required",
        filename="review-002.json",
    )
    destination = tmp_path / "published"
    real_copy2 = shutil.copy2
    entered = threading.Event()
    finish = threading.Event()

    def block_first_copy(source: Path, target: Path) -> Path:
        if not entered.is_set():
            entered.set()
            assert finish.wait(timeout=5)
        return Path(real_copy2(source, target))

    monkeypatch.setattr(profile.shutil, "copy2", block_first_copy)
    with ThreadPoolExecutor(max_workers=2) as executor:
        promotion = executor.submit(
            profile.promote_reviewed_artifacts,
            campaign,
            "attempt-0001",
            approved,
            destination,
        )
        assert entered.wait(timeout=5)
        changes_future = executor.submit(
            profile.register_review_receipt,
            campaign,
            "attempt-0001",
            changes,
        )
        with pytest.raises(RuntimeError, match="^campaign_lock_held$"):
            changes_future.result(timeout=5)
        finish.set()
        promotion.result(timeout=5)

    assert destination.is_dir()
    with pytest.raises(RuntimeError, match="^review_attempt_already_approved$"):
        profile.register_review_receipt(campaign, "attempt-0001", changes)
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1]["state"] == (
        "complete_pending_review"
    )


@pytest.mark.parametrize("problem", ("missing", "non_approved", "attempt", "digest"))
def test_promotion_rejects_missing_nonapproved_or_mismatched_receipt(
    tmp_path: Path, problem: str
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    if problem == "missing":
        receipt.unlink()
    else:
        payload = json.loads(receipt.read_bytes())
        if problem == "non_approved":
            payload["decision"] = "changes_required"
            payload["findings"] = [
                {"severity": "important", "code": "report_verdict_mismatch"}
            ]
        elif problem == "attempt":
            payload["attempt_id"] = "attempt-0002"
        else:
            payload["artifact_set_sha256"] = "f" * 64
        receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    destination = tmp_path / "published"

    with pytest.raises(RuntimeError):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert not destination.exists()
    assert attempt.is_dir()


def test_promotion_rejects_any_preexisting_destination(tmp_path: Path) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "README.md").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="^review_destination_exists$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert (destination / "README.md").read_text(encoding="utf-8") == "occupied\n"


def test_atomic_directory_rename_never_replaces_existing_empty_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence").write_bytes(b"retained")
    destination = tmp_path / "destination"

    profile._atomic_rename_directory_noreplace(source, destination)

    assert not source.exists()
    assert (destination / "evidence").read_bytes() == b"retained"
    second_source = tmp_path / "second-source"
    second_source.mkdir()
    (second_source / "evidence").write_bytes(b"still staged")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(RuntimeError, match="^review_destination_exists$"):
        profile._atomic_rename_directory_noreplace(second_source, occupied)
    assert (second_source / "evidence").read_bytes() == b"still staged"
    assert list(occupied.iterdir()) == []


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin mode workaround only")
def test_native_setup_and_mode_restore_double_failure_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    stage = destination.parent / f".{destination.name}.task-20010-stage"
    real_open = profile.os.open
    real_close = profile.os.close
    real_fchmod = profile.os.fchmod
    stage_fd: int | None = None
    stage_fd_closed = False
    native_symbol_requested = False

    def track_open(path, flags, *args, **kwargs):
        nonlocal stage_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == stage and stage_fd is None:
            stage_fd = descriptor
        return descriptor

    def track_close(descriptor: int) -> None:
        nonlocal stage_fd_closed
        if descriptor == stage_fd:
            stage_fd_closed = True
        real_close(descriptor)

    def fail_restore(descriptor: int, mode: int) -> None:
        if (
            descriptor == stage_fd
            and native_symbol_requested
            and mode == 0o555
        ):
            raise OSError("injected source mode restore failure")
        real_fchmod(descriptor, mode)

    class MissingNativeLibrary:
        def __getattr__(self, name: str):
            nonlocal native_symbol_requested
            if name == "renamex_np":
                native_symbol_requested = True
            raise AttributeError(name)

    monkeypatch.setattr(profile.os, "open", track_open)
    monkeypatch.setattr(profile.os, "close", track_close)
    monkeypatch.setattr(profile.os, "fchmod", fail_restore)
    monkeypatch.setattr(
        profile.ctypes,
        "CDLL",
        lambda _name, *, use_errno: MissingNativeLibrary(),
    )

    with pytest.raises(RuntimeError, match="^review_promotion_rename_failed$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert stage_fd is not None and stage_fd_closed
    assert stage.is_dir()
    assert stat.S_IMODE(stage.stat().st_mode) == 0o755
    assert attempt.is_dir()
    assert not destination.exists()
    assert not (campaign / ".campaign-lock").exists()


def test_promotion_empty_destination_race_preserves_staging_and_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    real_rename = profile._atomic_rename_directory_noreplace

    def inject_empty_destination(source: Path, target: Path, **kwargs) -> None:
        target.mkdir()
        real_rename(source, target, **kwargs)

    monkeypatch.setattr(
        profile, "_atomic_rename_directory_noreplace", inject_empty_destination
    )

    with pytest.raises(RuntimeError, match="^review_destination_exists$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    stage = destination.parent / f".{destination.name}.task-20010-stage"
    assert stage.is_dir()
    assert {path.name for path in stage.iterdir()} == (
        REVIEWED_ARTIFACT_NAMES | {"confirmatory-review-receipt.json"}
    )
    assert destination.is_dir() and list(destination.iterdir()) == []
    assert attempt.is_dir()


def test_promotion_boundary_mutation_fails_final_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    original = profile._atomic_rename_directory_noreplace

    def mutate_at_boundary(source: Path, target: Path, **kwargs) -> None:
        staged_readme = source / "README.md"
        if sys.platform == "darwin":
            os.chflags(
                staged_readme,
                staged_readme.stat().st_flags & ~stat.UF_IMMUTABLE,
            )
        staged_readme.chmod(0o644)
        staged_readme.write_text("boundary mutation\n", encoding="utf-8")
        original(source, target, **kwargs)

    monkeypatch.setattr(
        profile, "_atomic_rename_directory_noreplace", mutate_at_boundary
    )

    with pytest.raises(RuntimeError, match="^review_promotion_copy_mismatch$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    stage = destination.parent / f".{destination.name}.task-20010-stage"
    assert stage.is_dir()
    assert not destination.exists()
    assert attempt.is_dir()


def test_promotion_seals_and_fsyncs_stage_before_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    stage = destination.parent / f".{destination.name}.task-20010-stage"
    events: list[str] = []
    real_file_fsync = profile._fsync_regular_file
    real_directory_fsync = profile._fsync_directory

    def record_file(path: Path) -> None:
        if path.parent == stage:
            assert stat.S_IMODE(path.stat().st_mode) == 0o444
            events.append(f"file:{path.name}")
        real_file_fsync(path)

    def record_directory(path: Path) -> None:
        if path == stage:
            assert stat.S_IMODE(path.stat().st_mode) == 0o555
            events.append("directory:stage")
        elif path == destination.parent and destination.exists():
            events.append("directory:destination-parent")
        real_directory_fsync(path)

    monkeypatch.setattr(profile, "_fsync_regular_file", record_file)
    monkeypatch.setattr(profile, "_fsync_directory", record_directory)

    profile.promote_reviewed_artifacts(campaign, "attempt-0001", receipt, destination)

    assert events == [
        *(f"file:{name}" for name in profile.REVIEWED_ARTIFACTS),
        "file:confirmatory-review-receipt.json",
        "directory:stage",
        "directory:destination-parent",
    ]
    assert all(
        stat.S_IMODE((destination / name).stat().st_mode) == 0o444
        for name in REVIEWED_ARTIFACT_NAMES | {"confirmatory-review-receipt.json"}
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555


def test_sealed_stage_blocks_post_verify_name_replacement_before_native_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    stage = destination.parent / f".{destination.name}.task-20010-stage"
    native_library = profile.ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        real_native = native_library.renamex_np
    elif sys.platform.startswith("linux"):
        real_native = native_library.renameat2
    else:
        pytest.skip("native no-replace proof is POSIX-only")
    attempted = False
    replaced = False

    class InjectingRename:
        argtypes = None
        restype = None

        def __call__(self, *args) -> int:
            nonlocal attempted, replaced
            attempted = True
            staged_readme = stage / "README.md"
            try:
                staged_readme.unlink()
                staged_readme.write_text("post-verify replacement\n", encoding="utf-8")
                replaced = True
            except PermissionError:
                pass
            return int(real_native(*args))

    class InjectingLibrary:
        pass

    injected_library = InjectingLibrary()
    setattr(
        injected_library,
        "renamex_np" if sys.platform == "darwin" else "renameat2",
        InjectingRename(),
    )
    monkeypatch.setattr(
        profile.ctypes, "CDLL", lambda _name, *, use_errno: injected_library
    )

    profile.promote_reviewed_artifacts(campaign, "attempt-0001", receipt, destination)

    assert attempted
    assert not replaced
    assert (destination / "README.md").read_bytes() == b"# Confirmation\n"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin immutable flags only")
def test_failed_native_rename_never_clears_foreign_destination_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    stage = destination.parent / f".{destination.name}.task-20010-stage"
    native_library = profile.ctypes.CDLL(None, use_errno=True)
    real_native = native_library.renamex_np
    foreign_names = REVIEWED_ARTIFACT_NAMES | {
        "confirmatory-review-receipt.json"
    }
    foreign_state: dict[str, tuple[bytes, int]] = {}

    class RacingRename:
        argtypes = None
        restype = None

        def __call__(self, *args) -> int:
            destination.mkdir()
            for name in foreign_names:
                path = destination / name
                path.write_bytes(f"foreign:{name}".encode())
                os.chflags(path, path.stat().st_flags | stat.UF_IMMUTABLE)
                foreign_state[name] = (path.read_bytes(), path.stat().st_flags)
            return int(real_native(*args))

    class RacingLibrary:
        renamex_np = RacingRename()

    monkeypatch.setattr(
        profile.ctypes, "CDLL", lambda _name, *, use_errno: RacingLibrary()
    )

    with pytest.raises(RuntimeError, match="^review_destination_exists$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert stage.is_dir()
    assert stat.S_IMODE(stage.stat().st_mode) == 0o555
    assert all(
        not ((stage / name).stat().st_flags & stat.UF_IMMUTABLE)
        for name in foreign_names
    )
    assert {
        name: ((destination / name).read_bytes(), (destination / name).stat().st_flags)
        for name in foreign_names
    } == foreign_state


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin immutable flags only")
def test_post_syscall_destination_swap_fails_without_mutating_foreign_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    detached = tmp_path / "published-detached"
    native_library = profile.ctypes.CDLL(None, use_errno=True)
    real_native = native_library.renamex_np
    foreign_state: dict[str, tuple[bytes, int, int]] = {}

    class SwappingRename:
        argtypes = None
        restype = None

        def __call__(self, *args) -> int:
            result = int(real_native(*args))
            assert result == 0
            destination.rename(detached)
            foreign_state.update(_write_immutable_foreign_publication(destination))
            return result

    class SwappingLibrary:
        renamex_np = SwappingRename()

    monkeypatch.setattr(
        profile.ctypes, "CDLL", lambda _name, *, use_errno: SwappingLibrary()
    )

    with pytest.raises(RuntimeError, match="^publication_durability_uncertain$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert _publication_state(destination) == foreign_state
    assert stat.S_IMODE(destination.stat().st_mode) == 0o711
    assert stat.S_IMODE(detached.stat().st_mode) == 0o555
    assert all(
        not ((detached / name).stat().st_flags & stat.UF_IMMUTABLE)
        for name in TASK5_PUBLISHED_NAMES
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin immutable flags only")
def test_failed_native_rename_stage_root_swap_mutates_only_owned_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    stage = destination.parent / f".{destination.name}.task-20010-stage"
    detached = tmp_path / "published-detached"
    native_library = profile.ctypes.CDLL(None, use_errno=True)
    real_native = native_library.renamex_np
    foreign_stage_state: dict[str, tuple[bytes, int, int]] = {}

    class SwappingRename:
        argtypes = None
        restype = None

        def __call__(self, *args) -> int:
            stage.rename(detached)
            foreign_stage_state.update(
                _write_immutable_foreign_publication(stage, root_mode=0o701)
            )
            destination.mkdir()
            return int(real_native(*args))

    class SwappingLibrary:
        renamex_np = SwappingRename()

    monkeypatch.setattr(
        profile.ctypes, "CDLL", lambda _name, *, use_errno: SwappingLibrary()
    )

    with pytest.raises(RuntimeError, match="^review_destination_exists$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert _publication_state(stage) == foreign_stage_state
    assert stat.S_IMODE(stage.stat().st_mode) == 0o701
    assert stat.S_IMODE(detached.stat().st_mode) == 0o555
    assert all(
        not ((detached / name).stat().st_flags & stat.UF_IMMUTABLE)
        for name in TASK5_PUBLISHED_NAMES
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin immutable flags only")
@pytest.mark.parametrize("native_success", (False, True))
def test_failed_native_rename_leaf_swap_never_mutates_foreign_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_success: bool
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    stage = destination.parent / f".{destination.name}.task-20010-stage"
    detached = tmp_path / "published-detached"
    native_library = profile.ctypes.CDLL(None, use_errno=True)
    real_native = native_library.renamex_np
    foreign_readme_state: tuple[bytes, int, int] | None = None

    class SwappingRename:
        argtypes = None
        restype = None

        def __call__(self, *args) -> int:
            nonlocal foreign_readme_state
            detached.mkdir()
            readme = stage / "README.md"
            os.chflags(readme, readme.stat().st_flags & ~stat.UF_IMMUTABLE)
            readme.rename(detached / "README.md")
            readme.write_bytes(b"foreign leaf")
            readme.chmod(0o620)
            os.chflags(readme, readme.stat().st_flags | stat.UF_IMMUTABLE)
            metadata = readme.stat()
            foreign_readme_state = (
                readme.read_bytes(),
                stat.S_IMODE(metadata.st_mode),
                metadata.st_flags,
            )
            if not native_success:
                destination.mkdir()
            return int(real_native(*args))

    class SwappingLibrary:
        renamex_np = SwappingRename()

    monkeypatch.setattr(
        profile.ctypes, "CDLL", lambda _name, *, use_errno: SwappingLibrary()
    )

    expected_error = (
        "publication_durability_uncertain"
        if native_success
        else "review_destination_exists"
    )
    with pytest.raises(RuntimeError, match=f"^{expected_error}$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert foreign_readme_state is not None
    readme = (destination if native_success else stage) / "README.md"
    metadata = readme.stat()
    assert (
        readme.read_bytes(),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_flags,
    ) == foreign_readme_state
    owned_readme = detached / "README.md"
    assert owned_readme.read_bytes() == b"# Confirmation\n"
    assert not (owned_readme.stat().st_flags & stat.UF_IMMUTABLE)


@pytest.mark.parametrize("failure", ("file", "stage_directory"))
def test_promotion_pre_rename_fsync_failure_preserves_source_and_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    stage = destination.parent / f".{destination.name}.task-20010-stage"
    real_directory_fsync = profile._fsync_directory

    if failure == "file":
        real_file_fsync = profile._fsync_regular_file

        def fail_staged_file(path: Path) -> None:
            if path.parent == stage:
                raise OSError("injected file fsync")
            real_file_fsync(path)

        monkeypatch.setattr(
            profile,
            "_fsync_regular_file",
            fail_staged_file,
        )
    else:

        def fail_stage_directory(path: Path) -> None:
            if path == stage:
                raise OSError("injected stage directory fsync")
            real_directory_fsync(path)

        monkeypatch.setattr(profile, "_fsync_directory", fail_stage_directory)

    with pytest.raises(RuntimeError, match="^review_promotion_stage_fsync_failed$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert attempt.is_dir()
    assert stage.is_dir()
    assert stat.S_IMODE(stage.stat().st_mode) == 0o555
    assert not destination.exists()
    assert not (campaign / ".campaign-lock").exists()


def test_promotion_parent_fsync_failure_retains_destination_as_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    real_directory_fsync = profile._fsync_directory

    def fail_published_parent(path: Path) -> None:
        if path == destination.parent and destination.exists():
            raise OSError("injected destination parent fsync")
        real_directory_fsync(path)

    monkeypatch.setattr(profile, "_fsync_directory", fail_published_parent)

    with pytest.raises(RuntimeError, match="^publication_durability_uncertain$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert destination.is_dir()
    assert attempt.is_dir()
    assert not (campaign / ".campaign-lock").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin immutable flags only")
@pytest.mark.parametrize("failure", ("mode", "flags"))
def test_post_rename_restore_failure_is_stable_and_closes_owned_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    stage = destination.parent / f".{destination.name}.task-20010-stage"
    real_atomic = profile._atomic_rename_directory_noreplace
    real_fchmod = profile.os.fchmod
    real_fchflags = profile._fchflags
    real_open = profile.os.open
    real_close = profile.os.close
    native_completed = False
    owned_descriptors: set[int] = set()
    closed_descriptors: set[int] = set()

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        candidate = Path(path)
        if (
            candidate in {stage, destination}
            or candidate.parent in {stage, destination}
        ):
            owned_descriptors.add(descriptor)
        return descriptor

    def track_close(descriptor: int) -> None:
        if descriptor in owned_descriptors:
            closed_descriptors.add(descriptor)
        real_close(descriptor)

    def mark_native_complete(*args, **kwargs) -> None:
        nonlocal native_completed
        real_atomic(*args, **kwargs)
        native_completed = True

    def fail_mode(descriptor: int, mode: int) -> None:
        if failure == "mode" and native_completed and mode == 0o555:
            raise OSError("injected published mode restore failure")
        real_fchmod(descriptor, mode)

    def fail_flags(descriptor: int, flags: int) -> None:
        if failure == "flags" and native_completed:
            raise OSError("injected published flag restore failure")
        real_fchflags(descriptor, flags)

    monkeypatch.setattr(profile.os, "open", track_open)
    monkeypatch.setattr(profile.os, "close", track_close)
    monkeypatch.setattr(
        profile, "_atomic_rename_directory_noreplace", mark_native_complete
    )
    monkeypatch.setattr(profile.os, "fchmod", fail_mode)
    monkeypatch.setattr(profile, "_fchflags", fail_flags)

    with pytest.raises(RuntimeError, match="^publication_durability_uncertain$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert destination.is_dir()
    assert attempt.is_dir()
    assert owned_descriptors
    assert owned_descriptors <= closed_descriptors
    assert not (campaign / ".campaign-lock").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin flag durability only")
@pytest.mark.parametrize("failure", (None, "leaf", "directory"))
def test_restored_publication_metadata_is_fsynced_before_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    stage = destination.parent / f".{destination.name}.task-20010-stage"
    real_atomic = profile._atomic_rename_directory_noreplace
    real_open = profile.os.open
    real_close = profile.os.close
    real_fsync = profile.os.fsync
    native_completed = False
    stage_closed = False
    stage_fd: int | None = None
    leaf_fds: dict[int, str] = {}
    opened_paths: dict[int, Path] = {}
    owned_descriptors: set[int] = set()
    closed_descriptors: set[int] = set()
    events: list[str] = []

    def track_open(path, flags, *args, **kwargs):
        nonlocal stage_fd
        descriptor = real_open(path, flags, *args, **kwargs)
        candidate = Path(path)
        opened_paths[descriptor] = candidate
        if candidate == stage and stage_fd is None:
            stage_fd = descriptor
            owned_descriptors.add(descriptor)
        elif candidate.parent == stage and candidate.name in TASK5_PUBLISHED_NAMES:
            if candidate.name not in leaf_fds.values():
                leaf_fds[descriptor] = candidate.name
                owned_descriptors.add(descriptor)
        elif candidate == destination or candidate.parent == destination:
            owned_descriptors.add(descriptor)
        return descriptor

    def track_close(descriptor: int) -> None:
        nonlocal stage_closed
        if descriptor in owned_descriptors:
            closed_descriptors.add(descriptor)
        if descriptor == stage_fd:
            stage_closed = True
        real_close(descriptor)

    def mark_native_complete(*args, **kwargs) -> None:
        nonlocal native_completed
        real_atomic(*args, **kwargs)
        native_completed = True

    def inject_metadata_fsync(descriptor: int) -> None:
        if native_completed and not stage_closed:
            if descriptor in leaf_fds:
                events.append(f"leaf:{leaf_fds[descriptor]}")
                if failure == "leaf":
                    raise OSError("injected restored leaf fsync failure")
            elif descriptor == stage_fd:
                events.append("directory:published")
                if failure == "directory":
                    raise OSError("injected restored directory fsync failure")
            elif opened_paths.get(descriptor) == destination.parent:
                events.append("directory:parent")
        real_fsync(descriptor)

    monkeypatch.setattr(profile.os, "open", track_open)
    monkeypatch.setattr(profile.os, "close", track_close)
    monkeypatch.setattr(profile.os, "fsync", inject_metadata_fsync)
    monkeypatch.setattr(
        profile, "_atomic_rename_directory_noreplace", mark_native_complete
    )

    if failure is None:
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )
        assert events == [
            *(f"leaf:{name}" for name in profile.REVIEWED_ARTIFACTS),
            "leaf:confirmatory-review-receipt.json",
            "directory:published",
            "directory:parent",
        ]
    else:
        with pytest.raises(RuntimeError, match="^publication_durability_uncertain$"):
            profile.promote_reviewed_artifacts(
                campaign, "attempt-0001", receipt, destination
            )
        assert "directory:parent" not in events

    assert destination.is_dir()
    assert attempt.is_dir()
    assert owned_descriptors
    assert owned_descriptors <= closed_descriptors
    assert not (campaign / ".campaign-lock").exists()


def test_windows_promotion_fails_closed_without_directory_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    monkeypatch.setattr(profile.sys, "platform", "win32")

    with pytest.raises(
        RuntimeError,
        match="^review_publication_directory_durability_unsupported$",
    ):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, tmp_path / "published"
        )

    assert attempt.is_dir()
    assert not (tmp_path / "published").exists()


def test_promotion_rejects_source_changed_after_review(tmp_path: Path) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    (attempt / "README.md").write_text("changed after review\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="^review_receipt_binding_mismatch$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, tmp_path / "published"
        )


def test_promotion_rejects_destination_copy_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    real_copy2 = shutil.copy2

    def corrupt_copy(source: Path, target: Path) -> Path:
        result = Path(real_copy2(source, target))
        if Path(source).name == "README.md":
            result.write_text("corrupt copy\n", encoding="utf-8")
        return result

    monkeypatch.setattr(profile.shutil, "copy2", corrupt_copy)

    with pytest.raises(RuntimeError, match="^review_promotion_copy_mismatch$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert not destination.exists()
    assert (attempt / "README.md").read_bytes() == b"# Confirmation\n"


def test_promotion_rejects_source_mutation_between_copy_and_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    real_copy2 = shutil.copy2

    def mutate_source_after_copy(source: Path, target: Path) -> Path:
        result = Path(real_copy2(source, target))
        if Path(source) == receipt:
            (attempt / "README.md").write_text(
                "mutated during copy\n", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(profile.shutil, "copy2", mutate_source_after_copy)

    with pytest.raises(RuntimeError, match="^review_promotion_source_changed$"):
        profile.promote_reviewed_artifacts(
            campaign, "attempt-0001", receipt, destination
        )

    assert not destination.exists()


def test_successful_promotion_publishes_exact_five_plus_receipt(
    tmp_path: Path,
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"

    profile.promote_reviewed_artifacts(campaign, "attempt-0001", receipt, destination)

    assert {path.name for path in destination.iterdir()} == (
        REVIEWED_ARTIFACT_NAMES | {"confirmatory-review-receipt.json"}
    )
    assert profile.canonical_artifact_digest(destination) == digest
    assert (
        json.loads((destination / "confirmatory-review-receipt.json").read_bytes())[
            "artifact_set_sha256"
        ]
        == digest
    )
    assert attempt.is_dir()


def test_digest_register_and_promote_actions_never_contact_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    campaign, attempt, digest, _raw_sha256 = _prepare_review_attempt(tmp_path)
    receipt = _write_review_receipt(attempt, digest)
    destination = tmp_path / "published"
    monkeypatch.setattr(
        profile,
        "preflight_provider",
        lambda *_args: pytest.fail("maintenance action contacted provider"),
    )

    assert (
        profile.main(
            [
                "--campaign-action",
                "digest",
                "--campaign-root",
                str(campaign),
                "--attempt-id",
                "attempt-0001",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["artifact_set_sha256"] == digest
    assert (
        profile.main(
            [
                "--campaign-action",
                "register-review",
                "--campaign-root",
                str(campaign),
                "--attempt-id",
                "attempt-0001",
                "--review-receipt",
                str(receipt),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        profile.main(
            [
                "--campaign-action",
                "promote",
                "--campaign-root",
                str(campaign),
                "--attempt-id",
                "attempt-0001",
                "--review-receipt",
                str(receipt),
                "--destination",
                str(destination),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["event"] == "campaign_promoted"


def test_parse_arguments_rejects_incomplete_or_unpinned_acquisition(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        profile.parse_arguments(["--output-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        profile.parse_arguments(
            [
                "--endpoint",
                "http://127.0.0.1:9099",
                "--model",
                "fixture.gguf",
                "--campaign-root",
                str(tmp_path),
                "--burn-in-blocks",
                "1",
            ]
        )


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
    assert sample_root.is_dir()
    assert list(sample_root.iterdir()) == []
    remove_successful_sample_root(run_root, sample_root)
    assert sample_root.is_dir()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(RuntimeError, match="sample_cleanup_refused"):
        remove_successful_sample_root(run_root, outside)
    assert outside.is_dir()


def test_remove_successful_sample_root_parent_swap_preserves_foreign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    samples_root = run_root / "samples"
    sample_root = samples_root / "000-measured-0-control"
    sample_root.mkdir(parents=True)
    (sample_root / "owned-sentinel").write_bytes(b"remove")
    detached_samples = tmp_path / "detached-samples"
    outside_samples = tmp_path / "outside-samples"
    foreign_sample = outside_samples / sample_root.name
    foreign_sample.mkdir(parents=True)
    foreign_sentinel = foreign_sample / "foreign-sentinel"
    foreign_sentinel.write_bytes(b"preserve")
    original_clear = profile._remove_directory_contents_fd
    swapped = False

    def swap_parent() -> None:
        nonlocal swapped
        if swapped:
            return
        swapped = True
        samples_root.rename(detached_samples)
        samples_root.symlink_to(outside_samples, target_is_directory=True)

    def clear_then_swap(descriptor: int, **kwargs) -> None:
        swap_parent()
        original_clear(descriptor, **kwargs)

    monkeypatch.setattr(
        profile, "_remove_directory_contents_fd", clear_then_swap
    )

    with pytest.raises(RuntimeError, match="^sample_cleanup_refused$"):
        profile.remove_successful_sample_root(run_root, sample_root)

    assert swapped
    assert foreign_sentinel.read_bytes() == b"preserve"
    retained = detached_samples / sample_root.name
    assert retained.is_dir()
    assert list(retained.iterdir()) == []


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
                "phase": spec["phase"],
                "iteration": spec["iteration"],
                "arm": spec["arm"],
                "target_revision_kind": (
                    "control" if spec["arm"] == "control" else "candidate"
                ),
                "schedule_position": spec["schedule_position"],
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


def test_child_mode_retains_exact_schedule_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    sample_root = run_root / "sample"
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
            "sample_id": "burn_in-0-enabled",
            "phase": "burn_in",
            "iteration": 0,
            "arm": "enabled",
            "schedule_position": 5,
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
        profile.TargetAdapter,
        "for_arm",
        classmethod(
            lambda _cls, _root, _arm: SimpleNamespace(revision_kind="candidate")
        ),
    )

    async def mounted(*_args, **_kwargs):
        return _valid_sample("enabled")

    monkeypatch.setattr(profile, "run_mounted_sample", mounted)
    monkeypatch.setattr(profile, "validate_sample", lambda _row: ())
    args = SimpleNamespace(
        child_spec=spec_path,
        output_root=run_root,
        endpoint="http://127.0.0.1:9099",
        model="fixture.gguf",
    )

    assert profile.run_child_mode(args) == 0
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert [row["schedule_position"] for row in rows] == [5, 5]


@pytest.mark.parametrize(
    ("iterations", "burn_in_blocks", "schedule_kind", "finalization_failure"),
    (
        (30, 5, "official", None),
        (30, 5, "official", "manifest"),
        (30, 5, "official", "summary"),
        (30, 5, "official", "run_complete"),
        (1, 1, "disposable_smoke", None),
    ),
)
def test_confirmatory_parent_preflights_before_samples_and_routes_exact_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    iterations: int,
    burn_in_blocks: int,
    schedule_kind: str,
    finalization_failure: str | None,
) -> None:
    campaign = tmp_path / "campaign"
    attempt = _acquire_attempt(campaign)
    output = attempt.root
    harness = {"revision": "1" * 40, "runner_sha256": "2" * 64}
    monkeypatch.setattr(profile, "current_harness_identity", lambda _root: harness)
    monkeypatch.setattr(
        profile,
        "resolve_benchmark_revisions",
        lambda *_args, **_kwargs: {
            "control": profile.CONTROL_SHA,
            "candidate": profile.CANDIDATE_SHA,
        },
    )
    monkeypatch.setattr(
        profile,
        "resolve_implementation_base_revision",
        lambda _root: profile.IMPLEMENTATION_BASE_SHA,
    )
    monkeypatch.setattr(profile, "preflight_provider", lambda *_args: {"status": "ready"})
    monkeypatch.setattr(
        profile,
        "provider_server_metadata",
        lambda *_args: {"model_alias": "fixture.gguf"},
    )
    monkeypatch.setattr(profile, "runtime_metadata", lambda: {"python": "fixture"})
    monkeypatch.setattr(profile, "host_load_snapshot", lambda: {})
    monkeypatch.setattr(profile, "listener_resource_snapshot", lambda *_args: {})
    monkeypatch.setattr(
        profile,
        "listener_identity",
        lambda *_args: {"listener_count": 1, "fingerprint_sha256": "3" * 64},
    )
    listener_checks: list[str] = []
    monkeypatch.setattr(
        profile,
        "verify_listener_identity",
        lambda _endpoint, fingerprint: listener_checks.append(fingerprint),
    )
    monkeypatch.setattr(profile, "_remove_target_worktrees", lambda *_a, **_k: None)

    roots: dict[str, Path] = {}

    def prepare(_repository, run_root, *, name, revision):
        target = run_root / name
        target.mkdir()
        roots[name] = target
        return target

    monkeypatch.setattr(profile, "prepare_target_worktree", prepare)
    expected_protocol = {key: key for key in profile._PROTOCOL_MISMATCH_CODES}
    monkeypatch.setattr(
        profile, "load_original_protocol", lambda *_args: expected_protocol
    )
    summary_inputs: list[list[dict[str, object]]] = []
    statistics = SimpleNamespace(
        validate_sample=lambda _row: (),
        validate_run=lambda _rows, **_kwargs: (),
        build_summary=lambda rows: summary_inputs.append(list(rows))
        or {"overall_verdict": "pass", "sample_count": len(rows)},
    )
    monkeypatch.setattr(profile, "load_original_runner", lambda _root: statistics)
    monkeypatch.setattr(
        profile, "confirmation_protocol", lambda **_kwargs: expected_protocol
    )
    monkeypatch.setattr(profile, "protocol_mismatches", lambda *_args: ())

    child_specs: list[tuple[dict[str, object], Path]] = []

    def child(command, *, evidence_path, cwd, **_kwargs):
        spec_path = Path(command[command.index("--child-spec") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        child_specs.append((spec, cwd))
        if spec.get("mode") == "protocol_preflight":
            expected_behaviors = profile.expected_protocol_preflight_behaviors()
            terminal = {
                "event": "protocol_preflight",
                "sample_id": spec["sample_id"],
                "phase": spec["phase"],
                "iteration": spec["iteration"],
                "arm": spec["arm"],
                "target_revision_kind": (
                    "legacy" if spec["arm"] == "control" else "candidate"
                ),
                "workspace_content_tree_digest": "4" * 64,
                "tool_definition_sha256": ("5" if spec["arm"] == "control" else "6") * 64,
                "behavior_sha256": expected_behaviors[str(spec["arm"])],
                "final_ownership": {"live_threads": 0},
            }
        else:
            terminal = _valid_sample(str(spec["arm"]), iteration=int(spec["iteration"]))
            terminal.update(
                {
                    "event": "sample",
                    "sample_id": spec["sample_id"],
                    "phase": spec["phase"],
                    "iteration": spec["iteration"],
                    "arm": spec["arm"],
                    "schedule_position": spec["schedule_position"],
                    "workspace_content_tree_digest": "4" * 64,
                    "expected_permission_definition_hash": (
                        "5" if spec["arm"] == "control" else "6"
                    )
                    * 64,
                }
            )
        with evidence_path.open("a", encoding="utf-8") as stream:
            profile.write_boundary_event(
                stream,
                {
                    "event": "child_start",
                    "sample_id": spec["sample_id"],
                    "phase": spec["phase"],
                    "iteration": spec["iteration"],
                    "arm": spec["arm"],
                    "schedule_position": spec["schedule_position"],
                },
            )
            profile.write_boundary_event(stream, terminal)
        return profile.ChildResult("complete", 0, terminal)

    monkeypatch.setattr(profile, "run_child_with_watchdog", child)
    finalization: list[str] = []
    real_write_json = profile._write_json

    def write_json(path: Path, value: dict[str, object]) -> None:
        kind = "manifest" if path.name.endswith("manifest.json") else "summary"
        finalization.append(kind)
        if finalization_failure == kind:
            raise RuntimeError(f"{kind}_finalization_failed")
        real_write_json(path, value)

    real_boundary = profile.write_boundary_event

    def boundary(destination, event):
        if event.get("event") == "run_complete":
            finalization.append("run_complete")
            if finalization_failure == "run_complete":
                raise RuntimeError("run_complete_finalization_failed")
        real_boundary(destination, event)

    monkeypatch.setattr(profile, "_write_json", write_json)
    monkeypatch.setattr(profile, "write_boundary_event", boundary)

    def pin(*, verdict: str, raw_sha256: str) -> dict[str, object]:
        finalization.append("pin")
        return profile.write_acquisition_pin(
            campaign,
            attempt.attempt_id,
            verdict=verdict,
            raw_sha256=raw_sha256,
        )

    args = SimpleNamespace(
        output_root=output,
        control_sha=profile.CONTROL_SHA,
        candidate_sha=profile.CANDIDATE_SHA,
        endpoint="http://127.0.0.1:9099",
        model="fixture.gguf",
        iterations=iterations,
        burn_in_blocks=burn_in_blocks,
        sample_timeout=1.0,
        attempt_id=attempt.attempt_id,
        campaign_root=campaign,
        acquisition_pin_callback=pin,
    )

    if finalization_failure is not None:
        with pytest.raises(RuntimeError, match=f"{finalization_failure}_finalization_failed"):
            profile.run_parent_mode(args)
        pinned = profile.read_acquisition_pin(campaign, attempt.attempt_id)
        assert pinned is not None
        assert finalization[0] == "pin"
        assert profile.recover_interrupted_attempt(
            campaign, process_start_probe=lambda _pid: None
        ) == pinned
        with pytest.raises(
            RuntimeError, match="campaign_acquisition_blocked:complete_pending_review"
        ):
            profile.require_campaign_acquisition(campaign / "attempts.jsonl")
        return

    assert profile.run_parent_mode(args) == 0
    assert finalization == ["pin", "manifest", "summary", "run_complete"]
    preflights = [spec for spec, _cwd in child_specs[:3]]
    samples = child_specs[3:]
    schedule = profile.sample_schedule(
        iterations, burn_in_blocks=burn_in_blocks
    )
    raw_rows = [
        json.loads(line)
        for line in (output / "real-provider-three-turn.raw.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(samples) == len(schedule)
    assert sum(
        row["event"] == "child_start" and row["phase"] != "protocol_preflight"
        for row in raw_rows
    ) == len(schedule)
    assert sum(row["event"] == "sample" for row in raw_rows) == len(schedule)
    assert [spec["mode"] for spec in preflights] == ["protocol_preflight"] * 3
    assert [spec["arm"] for spec in preflights] == list(profile.ARMS)
    assert [spec["schedule_position"] for spec, _cwd in samples] == list(
        range(len(schedule))
    )
    assert [cwd for spec, cwd in samples if spec["arm"] == "control"] == [
        roots["control"]
    ] * sum(item.arm == "control" for item in schedule)
    assert all(
        cwd == roots["candidate"]
        for spec, cwd in samples
        if spec["arm"] != "control"
    )
    assert len(listener_checks) == len(schedule) * 2
    summary = json.loads(
        (output / "real-provider-three-turn.summary.json").read_text()
    )
    statistics_row_count = len(schedule) - burn_in_blocks * len(profile.ARMS)
    assert summary["sample_count"] == statistics_row_count
    assert summary["excluded_burn_in_sample_count"] == (
        burn_in_blocks * len(profile.ARMS)
    )
    assert summary["burn_in_contract_status"] == "passed"
    if iterations >= 2:
        assert len(summary_inputs) == 1
        assert len(summary_inputs[0]) == statistics_row_count
        assert not any(row["phase"] == "burn_in" for row in summary_inputs[0])
    else:
        assert summary_inputs == []
        assert summary["overall_verdict"] == "smoke"
    manifest = json.loads(
        (output / "real-provider-three-turn.manifest.json").read_text()
    )
    assert manifest["burn_in_blocks"] == burn_in_blocks
    assert manifest["campaign_schedule"] == profile.campaign_schedule_contract(
        iterations, burn_in_blocks
    )
    assert manifest["campaign_schedule"]["kind"] == schedule_kind
    assert manifest["protocol_equivalence"]["status"] == "passed"
    assert manifest["implementation_base_revision"] == profile.IMPLEMENTATION_BASE_SHA
    assert [row["schedule_position"] for row in manifest["sample_schedule"]] == list(
        range(len(schedule))
    )


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


def _exact_protocol_preflight_rows() -> list[dict[str, object]]:
    expected = profile.expected_protocol_preflight_behaviors()
    return [
        {
            "arm": arm,
            "target_revision_kind": (
                "legacy" if arm == "control" else "candidate"
            ),
            "behavior_sha256": expected[arm],
            "workspace_content_tree_digest": "c" * 64,
            "tool_definition_sha256": f"{index + 1}" * 64,
            "final_ownership": {"live_threads": 0},
        }
        for index, arm in enumerate(profile.ARMS)
    ]


def test_protocol_preflight_behavior_contract_is_exact_and_arm_specific() -> None:
    expected = profile.expected_protocol_preflight_behaviors()

    assert set(expected) == set(profile.ARMS)
    assert len(set(expected.values())) == len(profile.ARMS)
    assert profile.validate_protocol_preflight_rows(
        _exact_protocol_preflight_rows()
    ) == (
        "c" * 64,
        {arm: f"{index + 1}" * 64 for index, arm in enumerate(profile.ARMS)},
    )


@pytest.mark.parametrize("arm", profile.ARMS)
def test_protocol_preflight_rejects_each_arm_behavior_mismatch(arm: str) -> None:
    rows = _exact_protocol_preflight_rows()
    next(row for row in rows if row["arm"] == arm)["behavior_sha256"] = "f" * 64

    with pytest.raises(
        RuntimeError, match=f"protocol_preflight_behavior_mismatch:{arm}"
    ):
        profile.validate_protocol_preflight_rows(rows)


@pytest.mark.parametrize(("mutation", "code"), [
    ("missing", "protocol_preflight_behavior_missing:disabled"),
    ("duplicate", "protocol_preflight_behavior_duplicate:disabled"),
])
def test_protocol_preflight_rejects_missing_or_duplicate_arm(
    mutation: str, code: str
) -> None:
    rows = _exact_protocol_preflight_rows()
    disabled = next(row for row in rows if row["arm"] == "disabled")
    if mutation == "missing":
        rows.remove(disabled)
    else:
        rows.append(dict(disabled))

    with pytest.raises(RuntimeError, match=code):
        profile.validate_protocol_preflight_rows(rows)


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


def test_main_acquisition_owns_attempt_through_pending_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    attempt_root = campaign / "attempts/attempt-0001"
    attempt_root.mkdir(parents=True)
    owner = profile.CampaignLockOwner(123, "a" * 64, "b" * 64)
    attempt = profile.CampaignAttempt("attempt-0001", attempt_root, owner)
    profile.append_attempt_state(
        campaign / "attempts.jsonl",
        {"attempt_id": attempt.attempt_id, "state": "running"},
    )
    calls: list[object] = []
    monkeypatch.setattr(
        profile,
        "acquire_campaign_attempt",
        lambda root: calls.append(("acquire", root)) or attempt,
    )

    def parent(args):
        calls.append(("parent", args.output_root, args.attempt_id))
        raw = b"raw\n"
        (args.output_root / "real-provider-three-turn.raw.jsonl").write_bytes(raw)
        (args.output_root / "real-provider-three-turn.summary.json").write_text(
            json.dumps({"overall_verdict": "smoke"}), encoding="utf-8"
        )
        (args.output_root / "real-provider-three-turn.manifest.json").write_text(
            "{}", encoding="utf-8"
        )
        args.completed_acquisition = {
            "verdict": "smoke",
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
        args.acquisition_pin_callback(**args.completed_acquisition)
        return 0

    monkeypatch.setattr(profile, "run_parent_mode", parent)
    real_complete = profile.complete_attempt_measurement

    def complete(ledger, attempt_id, **values):
        calls.append(("complete", ledger, attempt_id, values))
        return real_complete(ledger, attempt_id, **values)

    monkeypatch.setattr(profile, "complete_attempt_measurement", complete)
    monkeypatch.setattr(
        profile,
        "release_campaign_attempt",
        lambda root, owned: calls.append(("release", root, owned)),
    )

    assert profile.main(
        [
            "--endpoint",
            "http://127.0.0.1:9099",
            "--model",
            "fixture.gguf",
                "--campaign-root",
                str(campaign),
                "--iterations",
                "1",
                "--burn-in-blocks",
                "1",
            "--candidate-sha",
            profile.CANDIDATE_SHA,
        ]
    ) == 0
    assert [call[0] for call in calls] == ["acquire", "parent", "complete", "release"]
    assert calls[2][3] == {
        "verdict": "smoke",
        "raw_sha256": hashlib.sha256(b"raw\n").hexdigest(),
    }


def test_pre_acquisition_failure_preserves_terminal_append_failure_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    attempt = profile.acquire_campaign_attempt(
        campaign,
        pid=123,
        process_start_probe=lambda _pid: "a" * 64,
        owner_token_factory=lambda: "b" * 64,
    )
    monkeypatch.setattr(profile, "acquire_campaign_attempt", lambda _root: attempt)
    monkeypatch.setattr(
        profile,
        "run_parent_mode",
        lambda _args: (_ for _ in ()).throw(RuntimeError("preflight_failed")),
    )
    monkeypatch.setattr(
        profile,
        "append_attempt_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("terminal_append_failed")
        ),
    )

    with pytest.raises(BaseExceptionGroup) as caught:
        profile.run_acquisition_action(SimpleNamespace(campaign_root=campaign))

    assert str(caught.value) == (
        "confirmation_acquisition_and_terminal_state_failed (2 sub-exceptions)"
    )
    assert [str(error) for error in caught.value.exceptions] == [
        "preflight_failed",
        "terminal_append_failed",
    ]
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1]["state"] == (
        "running"
    )
    assert (campaign / ".campaign-lock").is_dir()


@pytest.mark.parametrize(
    ("code", "state", "reason_category"),
    (
        ("provider_preflight_models_failed", "failed", "provider"),
        ("harness_worktree_dirty", "failed", "acquisition"),
        ("confirmation_raw_invalid", "invalid", "raw"),
        ("protocol_preflight_contract_failed", "invalid", "product"),
        ("confirmation_run_invalid", "invalid", "completeness"),
        ("listener_identity_changed", "invalid", "isolation"),
        ("retained_evidence_privacy_violation", "invalid", "privacy"),
        ("protocol_preflight_thread_survivor", "invalid", "ownership"),
        ("unknown_acquisition_error", "failed", "acquisition"),
    ),
)
def test_predefinitive_failure_classification_is_explicit(
    code: str, state: str, reason_category: str
) -> None:
    assert profile.acquisition_failure_event(
        "attempt-0001", RuntimeError(code)
    ) == {
        "attempt_id": "attempt-0001",
        "state": state,
        "reason_category": reason_category,
    }


@pytest.mark.parametrize(
    ("code", "state", "reason_category"),
    (
        ("provider_preflight_models_failed", "failed", "provider"),
        ("retained_evidence_privacy_violation", "invalid", "privacy"),
    ),
)
def test_predefinitive_action_failure_records_classified_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    state: str,
    reason_category: str,
) -> None:
    campaign = tmp_path / "campaign"
    attempt = _acquire_attempt(campaign)
    monkeypatch.setattr(profile, "acquire_campaign_attempt", lambda _root: attempt)
    monkeypatch.setattr(
        profile,
        "run_parent_mode",
        lambda _args: (_ for _ in ()).throw(RuntimeError(code)),
    )

    with pytest.raises(RuntimeError, match=f"^{code}$"):
        profile.run_acquisition_action(SimpleNamespace(campaign_root=campaign))

    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == {
        "attempt_id": attempt.attempt_id,
        "state": state,
        "reason_category": reason_category,
    }
    assert not (campaign / ".campaign-lock").exists()


def test_acquisition_action_rejects_lexical_campaign_symlink(tmp_path: Path) -> None:
    external = tmp_path / "external-acquire"
    external.mkdir()
    campaign = tmp_path / "campaign-link"
    campaign.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="^campaign_root_invalid$"):
        profile.run_acquisition_action(SimpleNamespace(campaign_root=campaign))

    assert list(external.iterdir()) == []


def test_acquisition_action_rejects_campaign_with_symlink_ancestor(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external-acquire-parent"
    external.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="^campaign_root_invalid$"):
        profile.run_acquisition_action(
            SimpleNamespace(campaign_root=linked_parent / "campaign")
        )

    assert list(external.iterdir()) == []


def test_darwin_system_tmp_alias_supports_task6_campaign_path() -> None:
    if sys.platform != "darwin":
        pytest.skip("Darwin owns the /tmp -> /private/tmp compatibility alias")
    tmp_alias = Path("/tmp")
    if (
        not tmp_alias.is_symlink()
        or os.readlink(tmp_alias) not in {"private/tmp", "/private/tmp"}
        or tmp_alias.resolve(strict=True) != Path("/private/tmp")
    ):
        pytest.skip("host does not expose the standard Darwin /tmp alias")
    created = subprocess.run(
        ["mktemp", "-d", "/tmp/tldw-task6-campaign.XXXXXX"],
        check=True,
        capture_output=True,
        text=True,
    )
    campaign = Path(created.stdout.strip())
    try:
        arguments = profile.parse_arguments(
            [
                "--endpoint",
                "http://127.0.0.1:9099",
                "--model",
                "fixture.gguf",
                "--campaign-root",
                str(campaign),
                "--iterations",
                "1",
                "--burn-in-blocks",
                "1",
                "--candidate-sha",
                profile.CANDIDATE_SHA,
            ]
        )
        assert str(arguments.campaign_root).startswith("/tmp/")

        attempt = profile.acquire_campaign_attempt(
            arguments.campaign_root,
            pid=123,
            process_start_probe=lambda _pid: _OWNER_START,
            owner_token_factory=lambda: "b" * 64,
        )
        recovered = profile.recover_interrupted_attempt(
            arguments.campaign_root, process_start_probe=lambda _pid: None
        )

        assert recovered == {
            "attempt_id": attempt.attempt_id,
            "state": "failed",
            "reason_category": "interrupted",
        }
        assert not (campaign / ".campaign-lock").exists()
    finally:
        shutil.rmtree(campaign)


def test_recover_action_rejects_lexical_campaign_symlink(tmp_path: Path) -> None:
    external = tmp_path / "external-recover"
    owner = _acquire_lock(external)
    campaign = tmp_path / "campaign-link"
    campaign.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="^campaign_root_invalid$"):
        profile.run_campaign_maintenance_action(
            SimpleNamespace(campaign_action="recover", campaign_root=campaign)
        )

    assert profile._read_lock_owner(external / ".campaign-lock") == owner


def test_recover_action_rejects_campaign_with_symlink_ancestor(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external-recover-parent"
    campaign = external / "campaign"
    owner = _acquire_lock(campaign)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="^campaign_root_invalid$"):
        profile.run_campaign_maintenance_action(
            SimpleNamespace(
                campaign_action="recover",
                campaign_root=linked_parent / "campaign",
            )
        )

    assert profile._read_lock_owner(campaign / ".campaign-lock") == owner


def _completed_parent_artifacts(args: SimpleNamespace) -> None:
    raw = b"protocol-valid-raw\n"
    (args.output_root / "real-provider-three-turn.raw.jsonl").write_bytes(raw)
    (args.output_root / "real-provider-three-turn.summary.json").write_text(
        json.dumps({"overall_verdict": "pass"}), encoding="utf-8"
    )
    (args.output_root / "real-provider-three-turn.manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    completed = {
        "verdict": "pass",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }
    args.acquisition_pin_callback(**completed)
    args.completed_acquisition = completed


def test_acquisition_pin_is_atomic_empty_campaign_marker_not_attempt_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    attempt_root = profile.create_attempt_root(campaign, "attempt-0001")
    monkeypatch.setattr(
        profile.tempfile,
        "mkstemp",
        lambda *_args, **_kwargs: pytest.fail("pin used a staging tempfile"),
    )

    event = profile.write_acquisition_pin(
        campaign,
        "attempt-0001",
        verdict="pass",
        raw_sha256="a" * 64,
    )

    marker = campaign / (
        "acquisition-pins/attempt-0001--pass--" + "a" * 64
    )
    assert marker.is_dir()
    assert list(marker.iterdir()) == []
    assert list(marker.parent.iterdir()) == [marker]
    assert profile.read_acquisition_pin(campaign, "attempt-0001") == event
    assert list(attempt_root.iterdir()) == []


def test_pin_marker_survives_parent_fsync_failure_without_retryable_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    attempt = _acquire_attempt(campaign)
    monkeypatch.setattr(profile, "acquire_campaign_attempt", lambda _root: attempt)
    monkeypatch.setattr(
        profile,
        "run_parent_mode",
        lambda args: _completed_parent_artifacts(args) or 0,
    )
    pins_root = campaign / "acquisition-pins"
    real_fsync = profile._fsync_directory
    pin_fsync_calls = 0

    def fail_after_marker_mkdir(path: Path) -> None:
        nonlocal pin_fsync_calls
        if path == pins_root:
            pin_fsync_calls += 1
            if pin_fsync_calls == 1:
                raise OSError("injected pin parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(profile, "_fsync_directory", fail_after_marker_mkdir)
    args = SimpleNamespace(
        campaign_root=campaign, iterations=30, burn_in_blocks=5
    )

    assert profile.run_acquisition_action(args) == 0

    assert pin_fsync_calls == 2
    pending = profile.read_acquisition_pin(campaign, attempt.attempt_id)
    assert pending is not None
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == pending
    assert not (campaign / ".campaign-lock").exists()
    assert profile.recover_interrupted_attempt(
        campaign,
        process_start_probe=lambda _pid: pytest.fail("released owner was probed"),
    ) == pending
    with pytest.raises(
        RuntimeError, match="campaign_acquisition_blocked:complete_pending_review"
    ):
        profile.require_campaign_acquisition(campaign / "attempts.jsonl")


def test_persistent_pin_parent_fsync_failure_stays_running_until_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    attempt = _acquire_attempt(campaign)
    monkeypatch.setattr(profile, "acquire_campaign_attempt", lambda _root: attempt)
    monkeypatch.setattr(
        profile,
        "run_parent_mode",
        lambda args: _completed_parent_artifacts(args) or 0,
    )
    pins_root = campaign / "acquisition-pins"
    real_fsync = profile._fsync_directory
    persistent_failure = True

    def fail_pin_parent_fsync(path: Path) -> None:
        if persistent_failure and path == pins_root:
            raise OSError("persistent pin parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(profile, "_fsync_directory", fail_pin_parent_fsync)
    args = SimpleNamespace(
        campaign_root=campaign, iterations=30, burn_in_blocks=5
    )

    with pytest.raises(
        RuntimeError, match="^campaign_acquisition_pin_durability_failed$"
    ):
        profile.run_acquisition_action(args)

    pin = profile.read_acquisition_pin(campaign, attempt.attempt_id)
    assert pin is not None
    assert profile.attempt_lineage(campaign / "attempts.jsonl") == (
        {"attempt_id": attempt.attempt_id, "state": "running"},
    )
    assert (campaign / ".campaign-lock").is_dir()
    persistent_failure = False

    assert profile.recover_interrupted_attempt(
        campaign, process_start_probe=lambda _pid: None
    ) == pin
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == pin
    assert not (campaign / ".campaign-lock").exists()
    with pytest.raises(
        RuntimeError, match="campaign_acquisition_blocked:complete_pending_review"
    ):
        profile.require_campaign_acquisition(campaign / "attempts.jsonl")


@pytest.mark.parametrize("pin_outcome", ("exact", "absent", "mismatch"))
def test_pin_callback_failure_never_appends_retryable_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pin_outcome: str,
) -> None:
    campaign = tmp_path / "campaign"
    attempt = _acquire_attempt(campaign)
    monkeypatch.setattr(profile, "acquire_campaign_attempt", lambda _root: attempt)
    monkeypatch.setattr(
        profile,
        "run_parent_mode",
        lambda args: _completed_parent_artifacts(args) or 0,
    )
    real_write_pin = profile.write_acquisition_pin

    def fail_pin(*args, **kwargs):
        if pin_outcome == "exact":
            real_write_pin(*args, **kwargs)
        elif pin_outcome == "mismatch":
            real_write_pin(*args, **{**kwargs, "raw_sha256": "b" * 64})
        raise OSError("injected pin callback failure")

    monkeypatch.setattr(profile, "write_acquisition_pin", fail_pin)
    args = SimpleNamespace(
        campaign_root=campaign, iterations=30, burn_in_blocks=5
    )

    if pin_outcome == "exact":
        assert profile.run_acquisition_action(args) == 0
        pending = profile.read_acquisition_pin(campaign, attempt.attempt_id)
        assert pending is not None
        assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == pending
        assert not (campaign / ".campaign-lock").exists()
    else:
        with pytest.raises(OSError, match="injected pin callback failure"):
            profile.run_acquisition_action(args)
        assert profile.attempt_lineage(campaign / "attempts.jsonl") == (
            {"attempt_id": attempt.attempt_id, "state": "running"},
        )
        assert (campaign / ".campaign-lock").is_dir()


@pytest.mark.parametrize("create_namespace", (False, True))
def test_acquisition_pin_read_treats_only_absent_state_as_unpinned(
    tmp_path: Path, create_namespace: bool
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    if create_namespace:
        (campaign / "acquisition-pins").mkdir()

    assert profile.read_acquisition_pin(campaign, "attempt-0001") is None


@pytest.mark.parametrize("namespace_kind", ("file", "symlink"))
def test_acquisition_pin_rejects_unowned_state_namespace(
    tmp_path: Path, namespace_kind: str
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    namespace = campaign / "acquisition-pins"
    if namespace_kind == "file":
        namespace.write_bytes(b"")
    else:
        target = tmp_path / "elsewhere"
        target.mkdir()
        namespace.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="^campaign_acquisition_pin_invalid$"):
        profile.read_acquisition_pin(campaign, "attempt-0001")
    with pytest.raises(RuntimeError, match="^campaign_acquisition_pin_invalid$"):
        profile.write_acquisition_pin(
            campaign,
            "attempt-0001",
            verdict="pass",
            raw_sha256="a" * 64,
        )


def test_acquisition_pin_rejects_partial_namespace_write(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    pins = campaign / "acquisition-pins"
    pins.mkdir(parents=True)
    (pins / ".acquisition-pin-partial").write_bytes(b'{"attempt_id":')

    with pytest.raises(RuntimeError, match="^campaign_acquisition_pin_invalid$"):
        profile.read_acquisition_pin(campaign, "attempt-0001")


@pytest.mark.parametrize(
    "marker_name",
    (
        "attempt-0001--pass--bad-hash",
        "attempt-0001--unknown--" + "a" * 64,
        "attempt-0001--pass--" + "A" * 64,
        "attempt-0001--pass--" + "a" * 64 + "--pid-123",
    ),
)
def test_acquisition_pin_rejects_private_or_noncanonical_marker(
    tmp_path: Path, marker_name: str
) -> None:
    campaign = tmp_path / "campaign"
    pins = campaign / "acquisition-pins"
    pins.mkdir(parents=True)
    (pins / marker_name).mkdir()

    with pytest.raises(RuntimeError, match="^campaign_acquisition_pin_invalid$"):
        profile.read_acquisition_pin(campaign, "attempt-0001")


def test_pending_ledger_failure_keeps_raw_verdict_pin_blocking_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    attempt = profile.acquire_campaign_attempt(
        campaign,
        pid=123,
        process_start_probe=lambda _pid: "a" * 64,
        owner_token_factory=lambda: "b" * 64,
    )
    monkeypatch.setattr(profile, "acquire_campaign_attempt", lambda _root: attempt)

    def parent(args):
        _completed_parent_artifacts(args)
        return 0

    monkeypatch.setattr(profile, "run_parent_mode", parent)
    monkeypatch.setattr(
        profile,
        "complete_attempt_measurement",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("pending_ledger_write_failed")
        ),
    )
    args = SimpleNamespace(
        campaign_root=campaign, iterations=30, burn_in_blocks=5
    )

    with pytest.raises(RuntimeError, match="pending_ledger_write_failed"):
        profile.run_acquisition_action(args)

    pin = profile.read_acquisition_pin(campaign, attempt.attempt_id)
    assert pin == {
        "attempt_id": attempt.attempt_id,
        "state": "complete_pending_review",
        "verdict": "pass",
        "raw_sha256": hashlib.sha256(b"protocol-valid-raw\n").hexdigest(),
    }
    assert {path.name for path in attempt.root.iterdir()} == {
        "real-provider-three-turn.raw.jsonl",
        "real-provider-three-turn.summary.json",
        "real-provider-three-turn.manifest.json",
    }
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1]["state"] == "running"
    assert (campaign / ".campaign-lock").is_dir()
    with pytest.raises(RuntimeError, match="campaign_acquisition_blocked:running"):
        profile.require_campaign_acquisition(campaign / "attempts.jsonl")

    monkeypatch.undo()
    recovered = profile.recover_interrupted_attempt(
        campaign, process_start_probe=lambda _pid: None
    )
    assert recovered == pin
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == pin
    assert not (campaign / ".campaign-lock").exists()
    with pytest.raises(
        RuntimeError, match="campaign_acquisition_blocked:complete_pending_review"
    ):
        profile.require_campaign_acquisition(campaign / "attempts.jsonl")


def test_post_acquisition_manifest_failure_stays_pending_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    attempt = profile.acquire_campaign_attempt(
        campaign,
        pid=123,
        process_start_probe=lambda _pid: "a" * 64,
        owner_token_factory=lambda: "b" * 64,
    )
    monkeypatch.setattr(profile, "acquire_campaign_attempt", lambda _root: attempt)

    def parent(args):
        _completed_parent_artifacts(args)
        return 0

    monkeypatch.setattr(profile, "run_parent_mode", parent)
    monkeypatch.setattr(
        profile,
        "_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("manifest_finalization_failed")
        ),
    )

    with pytest.raises(RuntimeError, match="manifest_finalization_failed"):
        profile.run_acquisition_action(
            SimpleNamespace(
                campaign_root=campaign, iterations=30, burn_in_blocks=5
            )
        )

    lineage = profile.attempt_lineage(campaign / "attempts.jsonl")
    assert lineage[-1] == profile.read_acquisition_pin(campaign, attempt.attempt_id)
    assert lineage[-1]["state"] == "complete_pending_review"
    assert not (campaign / ".campaign-lock").exists()
    with pytest.raises(
        RuntimeError, match="campaign_acquisition_blocked:complete_pending_review"
    ):
        profile.require_campaign_acquisition(campaign / "attempts.jsonl")


def test_recovery_finishes_lock_release_when_pending_append_raised_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    attempt = profile.acquire_campaign_attempt(
        campaign,
        pid=123,
        process_start_probe=lambda _pid: "a" * 64,
        owner_token_factory=lambda: "b" * 64,
    )
    monkeypatch.setattr(profile, "acquire_campaign_attempt", lambda _root: attempt)

    def parent(args):
        _completed_parent_artifacts(args)
        return 0

    monkeypatch.setattr(profile, "run_parent_mode", parent)
    real_complete = profile.complete_attempt_measurement

    def complete_then_fail(*args, **kwargs):
        real_complete(*args, **kwargs)
        raise RuntimeError("pending_ledger_post_commit_failed")

    monkeypatch.setattr(profile, "complete_attempt_measurement", complete_then_fail)

    with pytest.raises(RuntimeError, match="pending_ledger_post_commit_failed"):
        profile.run_acquisition_action(
            SimpleNamespace(
                campaign_root=campaign, iterations=30, burn_in_blocks=5
            )
        )

    pin = profile.read_acquisition_pin(campaign, attempt.attempt_id)
    assert profile.attempt_lineage(campaign / "attempts.jsonl")[-1] == pin
    assert (campaign / ".campaign-lock").is_dir()
    monkeypatch.undo()
    assert profile.recover_interrupted_attempt(
        campaign, process_start_probe=lambda _pid: None
    ) == pin
    assert not (campaign / ".campaign-lock").exists()


def test_recover_action_never_contacts_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        profile,
        "preflight_provider",
        lambda *_args: pytest.fail("maintenance action contacted provider"),
    )
    monkeypatch.setattr(
        profile,
        "recover_interrupted_attempt",
        lambda root: {"attempt_id": "attempt-0001", "state": "failed", "reason_category": "interrupted"},
    )

    assert profile.main(
        [
            "--campaign-action",
            "recover",
            "--campaign-root",
            str(tmp_path),
        ]
    ) == 0


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


def test_prepare_target_worktree_preserves_partial_add_failure(tmp_path: Path) -> None:
    target = (tmp_path / "run/candidate").resolve()
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[2] == "add":
            target.mkdir(parents=True)
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="busy")
        assert command == ["git", "worktree", "list", "--porcelain"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(BaseExceptionGroup, match="target_worktree_add_failed"):
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert calls == [
        ["git", "worktree", "add", "--detach", str(target), profile.CANDIDATE_SHA],
        ["git", "worktree", "list", "--porcelain"],
    ]
    assert target.is_dir()
    assert not (target.parent / ".candidate-cleanup-receipt.json").exists()


def test_prepare_target_worktree_preserves_partial_when_git_proves_absent(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "run/candidate").resolve()

    def fake_run(command, **_kwargs):
        if command[2] == "add":
            target.mkdir(parents=True)
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="add failed")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(BaseExceptionGroup, match="target_worktree_add_failed"):
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert target.is_dir()
    assert not (target.parent / ".candidate-cleanup-receipt.json").exists()


def test_prepare_target_worktree_preserves_partial_directory_when_add_raises(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "run/candidate").resolve()

    def fake_run(command, **_kwargs):
        if command[2] == "add":
            target.mkdir(parents=True)
            raise RuntimeError("git-add-exploded")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(BaseExceptionGroup, match="target_worktree_add_failed"):
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert target.is_dir()
    assert not (target.parent / ".candidate-cleanup-receipt.json").exists()


def test_prepare_target_worktree_preserves_unregistered_foreign_add_failure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "run/candidate"
    sentinel = target / "foreign-sentinel"

    def fake_run(command, **_kwargs):
        if command[2] == "add":
            target.mkdir(parents=True)
            sentinel.write_bytes(b"preserve")
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="busy")
        assert command == ["git", "worktree", "list", "--porcelain"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(BaseExceptionGroup, match="^target_worktree_add_failed"):
        profile.prepare_target_worktree(
            tmp_path / "repository",
            tmp_path / "run",
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert sentinel.read_bytes() == b"preserve"
    assert not (target.parent / ".candidate-cleanup-receipt.json").exists()


@pytest.mark.parametrize(
    "reserved_name",
    (
        "candidate",
        ".candidate-cleanup",
        ".candidate-cleanup-receipt.json",
        ".candidate-cleanup-terminal.json",
        "..candidate-cleanup-receipt.json-stage-deadbeef",
        ".candidate-cleanup-orphan-deadbeef",
        ".campaign-worktree-cleanup-deadbeef",
    ),
)
def test_prepare_target_worktree_rejects_reserved_cleanup_namespace_before_git(
    tmp_path: Path,
    reserved_name: str,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / reserved_name).mkdir()
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        raise AssertionError("Git must not run for reserved cleanup evidence")

    with pytest.raises(RuntimeError, match="^target_worktree_failed:candidate"):
        profile.prepare_target_worktree(
            tmp_path / "repository",
            run_root,
            name="candidate",
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert calls == []
    assert (run_root / reserved_name).is_dir()


@pytest.mark.parametrize("name", ("control", "candidate"))
def test_prepare_target_worktree_rechecks_reserved_namespace_after_git_success(
    tmp_path: Path,
    name: str,
) -> None:
    run_root = tmp_path / "run"
    target = run_root / name
    quarantine = run_root / f".{name}-cleanup"

    def fake_run(command, **_kwargs):
        assert command[:4] == ["git", "worktree", "add", "--detach"]
        target.mkdir(parents=True)
        quarantine.mkdir()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(
        RuntimeError,
        match=rf"^target_worktree_failed:{name}:cleanup_reserved$",
    ):
        profile.prepare_target_worktree(
            tmp_path / "repository",
            run_root,
            name=name,
            revision=profile.CANDIDATE_SHA,
            run_command=fake_run,
        )

    assert target.is_dir()
    assert quarantine.is_dir()


def test_remove_target_worktree_partial_no_owned_rechecks_before_success(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    run_root = tmp_path / "run"
    run_root.mkdir()
    target = run_root / "candidate"
    injected = False

    def fake_run(command, **_kwargs):
        nonlocal injected
        assert command == ["git", "worktree", "list", "--porcelain"]
        if not injected:
            injected = True
            target.mkdir()
            (target / "foreign-sentinel").write_bytes(b"preserve")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository,
            run_root,
            name="candidate",
            run_command=fake_run,
            allow_unregistered_owned=True,
        )

    assert (target / "foreign-sentinel").read_bytes() == b"preserve"


@pytest.mark.parametrize("receipt_kind", ("receipt", "terminal"))
def test_remove_target_worktree_partial_no_owned_rejects_orphan_receipt(
    tmp_path: Path,
    receipt_kind: str,
) -> None:
    repository = tmp_path / "repository"
    run_root = tmp_path / "run"
    run_root.mkdir()
    receipt = run_root / f".candidate-cleanup-{receipt_kind}.json"
    receipt.write_bytes(b"{}\n")

    def fake_run(command, **_kwargs):
        assert command == ["git", "worktree", "list", "--porcelain"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository,
            run_root,
            name="candidate",
            run_command=fake_run,
            allow_unregistered_owned=True,
        )

    assert receipt.read_bytes() == b"{}\n"


@pytest.mark.parametrize("name", ("control", "candidate"))
def test_remove_target_worktree_partial_no_owned_rechecks_injected_receipt(
    tmp_path: Path,
    name: str,
) -> None:
    repository = tmp_path / "repository"
    run_root = tmp_path / "run"
    run_root.mkdir()
    receipt = run_root / f".{name}-cleanup-receipt.json"
    registration_checks = 0

    def fake_run(command, **_kwargs):
        nonlocal registration_checks
        assert command == ["git", "worktree", "list", "--porcelain"]
        registration_checks += 1
        if registration_checks == 2:
            receipt.write_bytes(b"{}\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(
        RuntimeError,
        match=rf"^target_worktree_admin_marker_conflict:{name}$",
    ):
        profile._remove_target_worktree(
            repository,
            run_root,
            name=name,
            run_command=fake_run,
            allow_unregistered_owned=True,
        )

    assert registration_checks == 2
    assert receipt.read_bytes() == b"{}\n"


@pytest.mark.parametrize("name", ("control", "candidate"))
def test_remove_target_worktree_final_verifier_rejects_injected_cleanup_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / name
    _add_real_worktree(repository, target, branch=name)
    original_registrations = profile._worktree_registrations
    orphan = run_root / f".{name}-cleanup-orphan-injected"
    registration_checks = 0

    def inject_on_final_registration(*args, **kwargs):
        nonlocal registration_checks
        registrations = original_registrations(*args, **kwargs)
        registration_checks += 1
        if registration_checks == 3:
            orphan.mkdir()
        return registrations

    monkeypatch.setattr(
        profile, "_worktree_registrations", inject_on_final_registration
    )
    with pytest.raises(
        RuntimeError,
        match=rf"^target_worktree_admin_marker_conflict:{name}$",
    ):
        profile._remove_target_worktree(
            repository,
            run_root,
            name=name,
        )

    assert orphan.is_dir()
    assert target.is_dir()


def test_prepare_target_worktree_preserves_add_when_registration_check_raises(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "run/candidate").resolve()
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[2] == "add":
            target.mkdir(parents=True)
            raise RuntimeError("git-add-exploded")
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
        ["git", "worktree", "list", "--porcelain"],
    ]
    assert target.is_dir()
    assert [str(error) for error in caught.value.exceptions] == [
        "git-add-exploded",
        "git-list-exploded",
    ]


def test_prepare_target_worktree_keeps_registered_partial_add_directory(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "run/candidate").resolve()

    def fake_run(command, **_kwargs):
        if command[2] == "add":
            target.mkdir(parents=True)
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="add failed")
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
    primary, cleanup = caught.value.exceptions
    assert str(primary) == "target_worktree_failed:candidate"
    assert isinstance(cleanup, BaseExceptionGroup)
    assert [str(error) for error in cleanup.exceptions] == [
        "target_worktree_remove_failed:candidate",
        "target_worktree_registration_check_failed",
    ]


def _init_real_worktree_repository(root: Path) -> Path:
    repository = root / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "codex@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Codex"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "base", "--allow-empty"],
        cwd=repository,
        check=True,
    )
    return repository


def _add_real_worktree(
    repository: Path, target: Path, *, branch: str
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", branch, str(target)],
        cwd=repository,
        check=True,
    )


def _linked_worktree_admin(target: Path) -> Path:
    prefix = "gitdir: "
    payload = (target / ".git").read_text(encoding="utf-8")
    assert payload.startswith(prefix) and payload.endswith("\n")
    return Path(payload[len(prefix) : -1])


def _assert_retained_target_tombstone(run_root: Path, name: str) -> None:
    target = run_root / name
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_remove_target_worktree_real_git_parent_swap_preserves_unrelated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    attempts = tmp_path / "attempts"
    run_root = attempts / "attempt-0001"
    target = run_root / "candidate"
    outside = tmp_path / "outside" / "attempt-0001" / "candidate"
    _add_real_worktree(repository, target, branch="intended")
    _add_real_worktree(repository, outside, branch="unrelated")
    sentinel = outside / "uncommitted-sentinel"
    sentinel.write_bytes(b"must survive")
    original_move = profile._move_worktree_admin_to_marker
    swapped = False

    def swap_after_admin_claim(*args, **kwargs):
        nonlocal swapped
        result = original_move(*args, **kwargs)
        swapped = True
        profile.os.rename(attempts, tmp_path / "original-attempts")
        profile.os.rename(tmp_path / "outside", attempts)
        return result

    monkeypatch.setattr(
        profile, "_move_worktree_admin_to_marker", swap_after_admin_claim
    )

    profile._remove_target_worktree(
        repository,
        run_root,
        name="candidate",
    )

    assert swapped
    swapped_sentinel = attempts / "attempt-0001/candidate/uncommitted-sentinel"
    assert swapped_sentinel.read_bytes() == b"must survive"
    assert profile._worktree_registrations(repository) == frozenset(
        {str(repository.resolve()), str(outside.resolve())}
    )
    _assert_retained_target_tombstone(
        tmp_path / "original-attempts/attempt-0001", "candidate"
    )


def test_remove_target_worktree_real_git_rejects_replaced_admin_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    original_admin = _linked_worktree_admin(target)
    detached_original = repository / ".git" / "detached-original-admin"
    replacement_identity: tuple[int, int] | None = None
    original_move = profile._move_worktree_admin_to_marker

    def replace_then_move(
        common: Path, target_text: str, admin_name: str, *claim
    ) -> None:
        nonlocal replacement_identity
        original_admin.rename(detached_original)
        shutil.copytree(detached_original, original_admin)
        (original_admin / "replacement-sentinel").write_bytes(b"preserve")
        metadata = original_admin.stat()
        replacement_identity = (metadata.st_dev, metadata.st_ino)
        original_move(common, target_text, admin_name, *claim)

    monkeypatch.setattr(
        profile, "_move_worktree_admin_to_marker", replace_then_move
    )

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_identity_changed:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert replacement_identity is not None
    restored = original_admin.stat()
    assert (restored.st_dev, restored.st_ino) == replacement_identity
    assert (original_admin / "replacement-sentinel").read_bytes() == b"preserve"
    assert detached_original.is_dir()
    marker = repository / ".git" / profile._worktree_admin_marker_name(
        str(target)
    )
    assert (marker / "identity-conflict").is_file()
    assert str(target) in profile._worktree_registrations(repository)


def test_remove_target_worktree_real_git_rejects_claim_replacement_before_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    original_delete = profile._delete_worktree_admin_marker
    detached_claim = repository / ".git" / "detached-claimed-admin"

    def replace_then_delete(
        common: Path, target_text: str, **kwargs
    ) -> None:
        marker = common / profile._worktree_admin_marker_name(target_text)
        claimed = marker / "admin"
        claimed.rename(detached_claim)
        shutil.copytree(detached_claim, claimed)
        (claimed / "replacement-sentinel").write_bytes(b"preserve")
        original_delete(common, target_text, **kwargs)

    monkeypatch.setattr(
        profile, "_delete_worktree_admin_marker", replace_then_delete
    )

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    marker = repository / ".git" / profile._worktree_admin_marker_name(
        str(target)
    )
    assert (marker / "admin/replacement-sentinel").read_bytes() == b"preserve"
    assert detached_claim.is_dir()
    assert target.is_dir()


def test_remove_target_worktree_identity_conflict_preserves_empty_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    canonical_admin = _linked_worktree_admin(target)
    marker = repository / ".git" / profile._worktree_admin_marker_name(
        str(target)
    )
    detached_claim = repository / ".git" / "detached-original-claim"
    original_rename = profile.os.rename
    empty_identity: tuple[int, int] | None = None
    injected = False

    def replace_claim_and_create_empty(source, destination, *args, **kwargs):
        nonlocal empty_identity, injected
        result = original_rename(source, destination, *args, **kwargs)
        if (
            not injected
            and source == canonical_admin.name
            and destination == "admin"
        ):
            injected = True
            original_rename(marker / "admin", detached_claim)
            shutil.copytree(detached_claim, marker / "admin")
            (marker / "admin/replacement-sentinel").write_bytes(b"preserve")
            canonical_admin.mkdir()
            metadata = canonical_admin.stat()
            empty_identity = (metadata.st_dev, metadata.st_ino)
        return result

    monkeypatch.setattr(profile.os, "rename", replace_claim_and_create_empty)

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_identity_changed:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert empty_identity is not None
    canonical = canonical_admin.stat()
    assert (canonical.st_dev, canonical.st_ino) == empty_identity
    assert (marker / "admin/replacement-sentinel").read_bytes() == b"preserve"


@pytest.mark.parametrize("boundary", ("admin", "marker"))
def test_remove_target_worktree_terminal_paths_are_never_unlinked_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    marker = repository / ".git" / profile._worktree_admin_marker_name(
        str(target)
    )
    detached = repository / ".git" / f"detached-{boundary}"
    original_remove_contents = profile._remove_directory_contents_fd
    replacement_identity: tuple[int, int] | None = None
    injected = False

    def replace_empty_admin(descriptor: int, **kwargs) -> None:
        nonlocal replacement_identity, injected
        original_remove_contents(descriptor, **kwargs)
        if boundary == "admin" and not injected and (marker / "admin").is_dir():
            injected = True
            (marker / "admin").rename(detached)
            (marker / "admin").mkdir()
            metadata = (marker / "admin").stat()
            replacement_identity = (metadata.st_dev, metadata.st_ino)
        if boundary == "marker" and not injected and marker.is_dir():
            injected = True
            marker.rename(detached)
            marker.mkdir()
            metadata = marker.stat()
            replacement_identity = (metadata.st_dev, metadata.st_ino)

    monkeypatch.setattr(
        profile, "_remove_directory_contents_fd", replace_empty_admin
    )

    with pytest.raises(RuntimeError):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert replacement_identity is not None
    replacement = marker / "admin" if boundary == "admin" else marker
    metadata = replacement.stat()
    assert (metadata.st_dev, metadata.st_ino) == replacement_identity


def test_remove_target_worktree_real_git_unregisters_absent_exact_target(
    tmp_path: Path,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    detached = tmp_path / "detached-candidate"
    _add_real_worktree(repository, target, branch="candidate")
    target.rename(detached)

    profile._remove_target_worktree(
        repository, run_root, name="candidate"
    )

    assert detached.is_dir()
    assert profile._worktree_registrations(repository) == frozenset(
        {str(repository.resolve())}
    )
    marker = repository / ".git" / profile._worktree_admin_marker_name(
        str(target)
    )
    assert {entry.name for entry in marker.iterdir()} == {
        "admin",
        "pending.json",
        "terminal.json",
    }
    assert list((marker / "admin").iterdir()) == []
    receipt = (marker / "terminal.json").read_bytes()
    assert (marker / "pending.json").read_bytes() == receipt
    assert str(target).encode() not in receipt
    assert set(json.loads(receipt)) == {
        "admin_dev",
        "admin_ino",
        "target_dev",
        "target_ino",
        "target_sha256",
    }
    assert json.loads(receipt)["target_dev"] is None
    assert json.loads(receipt)["target_ino"] is None
    assert receipt == (
        json.dumps(
            json.loads(receipt), sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n"
    )

    profile._remove_target_worktree(
        repository, run_root, name="candidate"
    )


@pytest.mark.parametrize("replacement", ("path", "registration"))
def test_remove_target_worktree_terminal_tombstone_rejects_new_generation(
    tmp_path: Path, replacement: str
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    target_metadata = target.stat()

    profile._remove_target_worktree(
        repository, run_root, name="candidate"
    )
    marker = repository / ".git" / profile._worktree_admin_marker_name(
        str(target)
    )
    receipt = json.loads((marker / "terminal.json").read_bytes())
    assert (receipt["target_dev"], receipt["target_ino"]) == (
        target_metadata.st_dev,
        target_metadata.st_ino,
    )
    detached_original = tmp_path / "detached-original-candidate"
    target.rename(detached_original)
    if replacement == "path":
        target.mkdir()
        (target / "sentinel").write_bytes(b"preserve")
    else:
        subprocess.run(
            ["git", "worktree", "add", "-q", "--detach", str(target), "HEAD"],
            cwd=repository,
            check=True,
        )

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert target.is_dir()
    assert detached_original.is_dir()
    if replacement == "path":
        assert (target / "sentinel").read_bytes() == b"preserve"


def test_remove_target_worktree_registered_does_not_replace_injected_empty_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    quarantine = run_root / ".candidate-cleanup"
    _add_real_worktree(repository, target, branch="candidate")
    original_backlink = profile._worktree_backlink_admin_name
    injected_identity: tuple[int, int] | None = None

    def validate_then_inject(*args, **kwargs):
        nonlocal injected_identity
        result = original_backlink(*args, **kwargs)
        quarantine.mkdir()
        metadata = quarantine.stat()
        injected_identity = (metadata.st_dev, metadata.st_ino)
        return result

    monkeypatch.setattr(
        profile, "_worktree_backlink_admin_name", validate_then_inject
    )
    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert injected_identity is not None
    current = quarantine.stat()
    assert (current.st_dev, current.st_ino) == injected_identity
    assert target.is_dir()


def test_remove_target_worktree_pending_receipt_rejects_foreign_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    quarantine = run_root / ".candidate-cleanup"
    detached_original = tmp_path / "detached-original-candidate"
    _add_real_worktree(repository, target, branch="candidate")
    original_delete = profile._delete_worktree_admin_marker

    def publish_then_interrupt(*args, **kwargs) -> None:
        original_delete(*args, **kwargs)
        raise KeyboardInterrupt("pending receipt checkpoint")

    monkeypatch.setattr(
        profile, "_delete_worktree_admin_marker", publish_then_interrupt
    )
    with pytest.raises(KeyboardInterrupt):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    marker = repository / ".git" / profile._worktree_admin_marker_name(
        str(target)
    )
    pending_before = (marker / "pending.json").read_bytes()
    target.rename(quarantine)
    quarantine.rename(detached_original)
    quarantine.mkdir()
    foreign_sentinel = quarantine / "foreign-sentinel"
    foreign_sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(
        profile, "_delete_worktree_admin_marker", original_delete
    )

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert detached_original.is_dir()
    assert foreign_sentinel.read_bytes() == b"preserve"
    assert (marker / "pending.json").read_bytes() == pending_before
    assert not (marker / "terminal.json").exists()


@pytest.mark.parametrize("replacement_content", ("empty", "nonempty"))
def test_remove_target_worktree_terminal_rejects_quarantine_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_content: str,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    detached_original = tmp_path / "detached-original-target"
    _add_real_worktree(repository, target, branch="candidate")
    target_metadata = target.stat()
    target_identity = (target_metadata.st_dev, target_metadata.st_ino)
    original_remove = profile._remove_directory_contents_fd
    swapped = False

    def clear_then_swap(descriptor: int, **kwargs) -> None:
        nonlocal swapped
        original_remove(descriptor, **kwargs)
        metadata = os.fstat(descriptor)
        if not swapped and (metadata.st_dev, metadata.st_ino) == target_identity:
            swapped = True
            target.rename(detached_original)
            target.mkdir()
            if replacement_content == "nonempty":
                (target / "foreign-sentinel").write_bytes(b"preserve")

    monkeypatch.setattr(
        profile, "_remove_directory_contents_fd", clear_then_swap
    )

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert swapped
    assert detached_original.is_dir()
    assert target.is_dir()
    if replacement_content == "nonempty":
        assert (target / "foreign-sentinel").read_bytes() == b"preserve"


def test_remove_target_worktree_terminal_rejects_nonempty_exact_quarantine(
    tmp_path: Path,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    profile._remove_target_worktree(repository, run_root, name="candidate")
    sentinel = target / "foreign-sentinel"
    sentinel.write_bytes(b"preserve")

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert sentinel.read_bytes() == b"preserve"


@pytest.mark.parametrize("reappeared", ("path", "registration"))
def test_remove_target_worktree_terminal_rechecks_reappeared_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reappeared: str,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    profile._remove_target_worktree(repository, run_root, name="candidate")
    original_delete = profile._delete_worktree_admin_marker
    detached_reappeared = tmp_path / "detached-reappeared-candidate"

    def complete_then_reappear(*args, **kwargs):
        result = original_delete(*args, **kwargs)
        if kwargs.get("complete"):
            target.rename(detached_reappeared)
            if reappeared == "path":
                target.mkdir()
                (target / "foreign-sentinel").write_bytes(b"preserve")
            else:
                subprocess.run(
                    [
                        "git",
                        "worktree",
                        "add",
                        "-q",
                        "--detach",
                        str(target),
                        "HEAD",
                    ],
                    cwd=repository,
                    check=True,
                )
        return result

    monkeypatch.setattr(
        profile, "_delete_worktree_admin_marker", complete_then_reappear
    )

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    if reappeared == "path":
        assert (target / "foreign-sentinel").read_bytes() == b"preserve"
    else:
        assert detached_reappeared.is_dir()
        assert str(target) in profile._worktree_registrations(repository)


@pytest.mark.parametrize(
    ("field", "malformed_identity"),
    (
        ("target_dev", True),
        ("target_dev", -1),
        ("target_dev", "1"),
        ("target_dev", []),
        ("target_dev", {}),
        ("target_dev", None),
        ("target_ino", True),
        ("target_ino", -1),
        ("target_ino", None),
    ),
)
def test_remove_target_worktree_rejects_malformed_target_receipt_identity(
    tmp_path: Path, field: str, malformed_identity: object
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    profile._remove_target_worktree(repository, run_root, name="candidate")
    marker = repository / ".git" / profile._worktree_admin_marker_name(
        str(target)
    )
    receipt_path = marker / "pending.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt[field] = malformed_identity
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )


@pytest.mark.parametrize("malformation", ("missing", "unknown", "noncanonical"))
def test_remove_target_worktree_rejects_malformed_target_receipt_schema(
    tmp_path: Path, malformation: str
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    profile._remove_target_worktree(repository, run_root, name="candidate")
    marker = repository / ".git" / profile._worktree_admin_marker_name(
        str(target)
    )
    receipt_path = marker / "pending.json"
    receipt = json.loads(receipt_path.read_bytes())
    if malformation == "missing":
        del receipt["target_ino"]
    elif malformation == "unknown":
        receipt["unknown"] = 1
    if malformation == "noncanonical":
        payload = json.dumps(receipt, indent=2) + "\n"
    else:
        payload = json.dumps(
            receipt, sort_keys=True, separators=(",", ":")
        ) + "\n"
    receipt_path.write_text(payload)

    with pytest.raises(
        RuntimeError,
        match="^target_worktree_admin_marker_conflict:candidate$",
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )


@pytest.mark.parametrize(
    "checkpoint",
    (
        "empty_admin_marker",
        "owned_admin_marker",
        "admin_content_removed",
        "receipt_published",
        "admin_cleared",
        "target_cleared",
    ),
)
def test_remove_target_worktree_real_git_resumes_namespace_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    admin = _linked_worktree_admin(target)
    removable_admin_entry = next(
        entry.name for entry in admin.iterdir() if entry.name != "gitdir"
    )
    admin_metadata = admin.stat()
    admin_identity = (admin_metadata.st_dev, admin_metadata.st_ino)
    target_metadata = target.stat()
    target_identity = (target_metadata.st_dev, target_metadata.st_ino)
    original_rename = profile.os.rename
    original_unlink = profile.os.unlink
    original_publish = profile._publish_admin_tombstone
    original_remove_contents = profile._remove_directory_contents_fd

    def interrupt_rename(source, destination, *args, **kwargs):
        if checkpoint == "empty_admin_marker" and destination == "admin":
            raise KeyboardInterrupt("empty admin marker checkpoint")
        result = original_rename(source, destination, *args, **kwargs)
        if checkpoint == "owned_admin_marker" and destination == "admin":
            raise KeyboardInterrupt("owned admin marker checkpoint")
        return result

    def interrupt_unlink(path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        if checkpoint == "admin_content_removed" and path == removable_admin_entry:
            raise KeyboardInterrupt("admin content removed checkpoint")
        return result

    def interrupt_publish(*args, **kwargs):
        original_publish(*args, **kwargs)
        if checkpoint == "receipt_published":
            raise KeyboardInterrupt("receipt published checkpoint")

    def interrupt_remove_contents(descriptor: int, **kwargs) -> None:
        original_remove_contents(descriptor, **kwargs)
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        if checkpoint == "admin_cleared" and identity == admin_identity:
            raise KeyboardInterrupt("admin cleared checkpoint")
        if checkpoint == "target_cleared" and identity == target_identity:
            raise KeyboardInterrupt("target cleared checkpoint")

    monkeypatch.setattr(profile.os, "rename", interrupt_rename)
    monkeypatch.setattr(profile.os, "unlink", interrupt_unlink)
    monkeypatch.setattr(
        profile, "_publish_admin_tombstone", interrupt_publish
    )
    monkeypatch.setattr(
        profile, "_remove_directory_contents_fd", interrupt_remove_contents
    )
    with pytest.raises(KeyboardInterrupt):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert target.is_dir()
    monkeypatch.setattr(profile.os, "rename", original_rename)
    monkeypatch.setattr(profile.os, "unlink", original_unlink)
    monkeypatch.setattr(
        profile, "_publish_admin_tombstone", original_publish
    )
    monkeypatch.setattr(
        profile, "_remove_directory_contents_fd", original_remove_contents
    )
    profile._remove_target_worktree(
        repository, run_root, name="candidate"
    )

    _assert_retained_target_tombstone(run_root, "candidate")
    assert profile._worktree_registrations(repository) == frozenset(
        {str(repository.resolve())}
    )


@pytest.mark.parametrize("name", ("control", "candidate"))
def test_remove_target_worktree_real_git_resumes_after_admin_tombstone_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / name
    _add_real_worktree(repository, target, branch=name)
    original_delete_marker = profile._delete_worktree_admin_marker

    def delete_then_interrupt(
        common: Path, target_text: str, **kwargs
    ) -> None:
        original_delete_marker(common, target_text, **kwargs)
        raise KeyboardInterrupt("post-unregister checkpoint")

    monkeypatch.setattr(
        profile, "_delete_worktree_admin_marker", delete_then_interrupt
    )
    with pytest.raises(KeyboardInterrupt):
        profile._remove_target_worktree(
            repository, run_root, name=name
        )

    assert target.is_dir()
    assert str(target) not in profile._worktree_registrations(repository)
    monkeypatch.setattr(
        profile, "_delete_worktree_admin_marker", original_delete_marker
    )
    profile._remove_target_worktree(
        repository, run_root, name=name
    )

    _assert_retained_target_tombstone(run_root, name)


def test_remove_target_worktree_resumes_legacy_registered_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    quarantine = run_root / ".candidate-cleanup"
    _add_real_worktree(repository, target, branch="candidate")
    original_delete = profile._delete_worktree_admin_marker

    def publish_then_interrupt(*args, **kwargs) -> None:
        original_delete(*args, **kwargs)
        raise KeyboardInterrupt("legacy pending receipt")

    monkeypatch.setattr(
        profile, "_delete_worktree_admin_marker", publish_then_interrupt
    )
    with pytest.raises(KeyboardInterrupt):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )
    target.rename(quarantine)
    monkeypatch.setattr(
        profile, "_delete_worktree_admin_marker", original_delete
    )

    profile._remove_target_worktree(
        repository, run_root, name="candidate"
    )

    assert quarantine.is_dir()
    assert list(quarantine.iterdir()) == []
    assert not target.exists()


def test_remove_target_worktree_real_git_refuses_admin_marker_collision(
    tmp_path: Path,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    marker = repository / ".git" / profile._worktree_admin_marker_name(str(target))
    marker.mkdir()
    (marker / "foreign").write_bytes(b"must survive")

    with pytest.raises(
        RuntimeError, match="^target_worktree_admin_marker_conflict:candidate$"
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert (marker / "foreign").read_bytes() == b"must survive"
    assert target.is_dir()
    assert str(target) in profile._worktree_registrations(repository)


@pytest.mark.parametrize(
    "malformation", ("backlink", "admin_gitdir", "duplicate", "admin_symlink")
)
def test_remove_target_worktree_real_git_rejects_malformed_admin_contract(
    tmp_path: Path, malformation: str
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    admin = _linked_worktree_admin(target)
    if malformation == "backlink":
        (target / ".git").write_bytes(b"invalid backlink\n")
    elif malformation == "admin_gitdir":
        (admin / "gitdir").write_bytes(b"invalid gitdir\n")
    elif malformation == "duplicate":
        duplicate = admin.parent / "duplicate"
        duplicate.mkdir()
        (duplicate / "gitdir").write_text(
            f"{target / '.git'}\n", encoding="utf-8"
        )
    else:
        detached_admin = repository / ".git" / "detached-admin"
        admin.rename(detached_admin)
        admin.symlink_to(detached_admin, target_is_directory=True)

    expected_error = (
        "target_worktree_remove_failed:candidate"
        if malformation == "admin_gitdir"
        else "target_worktree_admin_invalid"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert target.is_dir()
    assert not (run_root / ".candidate-cleanup").exists()


def test_remove_target_worktree_real_git_rejects_worktrees_symlink(
    tmp_path: Path,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    worktrees = repository / ".git/worktrees"
    detached = repository / ".git/detached-worktrees"
    worktrees.rename(detached)
    worktrees.symlink_to(detached, target_is_directory=True)

    with pytest.raises(RuntimeError, match="target_worktree_admin_invalid"):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert target.is_dir()
    assert not (run_root / ".candidate-cleanup").exists()


def test_remove_target_worktree_real_git_rejects_common_directory_symlink(
    tmp_path: Path,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    common = repository / ".git"
    detached = tmp_path / "detached-common"
    common.rename(detached)
    common.symlink_to(detached, target_is_directory=True)

    with pytest.raises(
        RuntimeError, match="^target_worktree_unregister_failed:candidate$"
    ):
        profile._remove_target_worktree(
            repository, run_root, name="candidate"
        )

    assert target.is_dir()
    assert not (run_root / ".candidate-cleanup").exists()


def test_remove_target_worktree_real_git_preserves_missing_unrelated_registration(
    tmp_path: Path,
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    unrelated = tmp_path / "unrelated"
    _add_real_worktree(repository, target, branch="candidate")
    _add_real_worktree(repository, unrelated, branch="unrelated")
    unrelated_detached = tmp_path / "unrelated-detached"
    unrelated.rename(unrelated_detached)

    profile._remove_target_worktree(
        repository, run_root, name="candidate"
    )

    assert unrelated_detached.is_dir()
    assert profile._worktree_registrations(repository) == frozenset(
        {str(repository.resolve()), str(unrelated.resolve())}
    )


def test_remove_target_worktree_real_git_preserves_unrelated_that_goes_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    unrelated = tmp_path / "unrelated"
    detached = tmp_path / "unrelated-detached"
    _add_real_worktree(repository, target, branch="candidate")
    _add_real_worktree(repository, unrelated, branch="unrelated")
    original_move = profile._move_worktree_admin_to_marker

    def move_then_detach(
        common: Path, target_text: str, admin_name: str, *claim
    ) -> int:
        claimed = original_move(common, target_text, admin_name, *claim)
        unrelated.rename(detached)
        return claimed

    monkeypatch.setattr(
        profile, "_move_worktree_admin_to_marker", move_then_detach
    )

    profile._remove_target_worktree(
        repository, run_root, name="candidate"
    )

    assert detached.is_dir()
    assert profile._worktree_registrations(repository) == frozenset(
        {str(repository.resolve()), str(unrelated.resolve())}
    )


def test_remove_target_worktree_real_git_same_target_contention_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    original_registrations = profile._worktree_registrations
    barrier = threading.Barrier(2)
    local = threading.local()

    def synchronized_registrations(*args, **kwargs):
        result = original_registrations(*args, **kwargs)
        if not getattr(local, "synchronized", False):
            local.synchronized = True
            barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(
        profile, "_worktree_registrations", synchronized_registrations
    )

    def cleanup() -> BaseException | None:
        try:
            profile._remove_target_worktree(
                repository, run_root, name="candidate"
            )
        except BaseException as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: cleanup(), range(2)))

    stable_errors = {
        "target_worktree_unregister_failed:candidate",
        "target_worktree_admin_marker_conflict:candidate",
    }
    for outcome in outcomes:
        if outcome is not None and str(outcome) not in stable_errors:
            raise outcome
    assert any(outcome is None for outcome in outcomes)
    assert all(
        outcome is None or str(outcome) in stable_errors
        for outcome in outcomes
    ), [repr(outcome) for outcome in outcomes]
    monkeypatch.setattr(
        profile, "_worktree_registrations", original_registrations
    )
    profile._remove_target_worktree(
        repository, run_root, name="candidate"
    )

    assert str(target) not in profile._worktree_registrations(repository)
    _assert_retained_target_tombstone(run_root, "candidate")


@pytest.mark.parametrize("iteration", range(5))
def test_remove_target_worktree_real_git_disappearance_race_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, iteration: int
) -> None:
    root = tmp_path / str(iteration)
    root.mkdir()
    repository = _init_real_worktree_repository(root)
    run_root = root / "run"
    target = run_root / "candidate"
    _add_real_worktree(repository, target, branch="candidate")
    original_strict = profile._strict_owned_directory
    original_move = profile._move_worktree_admin_to_marker
    loser_waiting = threading.Event()
    allow_loser = threading.Event()
    selection_lock = threading.Lock()
    loser_selected = False

    def delayed_strict(path: Path, *args, **kwargs):
        nonlocal loser_selected
        delay = False
        if path == target:
            with selection_lock:
                if not loser_selected:
                    loser_selected = True
                    delay = True
        if delay:
            loser_waiting.set()
            assert allow_loser.wait(timeout=5)
        return original_strict(path, *args, **kwargs)

    def release_after_admin_claim(*args, **kwargs):
        result = original_move(*args, **kwargs)
        allow_loser.set()
        return result

    monkeypatch.setattr(profile, "_strict_owned_directory", delayed_strict)
    monkeypatch.setattr(
        profile, "_move_worktree_admin_to_marker", release_after_admin_claim
    )

    def cleanup() -> BaseException | None:
        try:
            profile._remove_target_worktree(
                repository, run_root, name="candidate"
            )
        except BaseException as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cleanup)
        assert loser_waiting.wait(timeout=5)
        second = executor.submit(cleanup)
        outcomes = (first.result(timeout=10), second.result(timeout=10))

    assert any(outcome is None for outcome in outcomes)
    assert all(
        outcome is None
        or str(outcome)
        in {
            "target_worktree_unregister_failed:candidate",
            "target_worktree_admin_marker_conflict:candidate",
        }
        for outcome in outcomes
    ), [repr(outcome) for outcome in outcomes]


def test_remove_target_worktree_real_git_different_targets_are_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    for name in ("control", "candidate"):
        _add_real_worktree(repository, run_root / name, branch=name)
    original_registrations = profile._worktree_registrations
    barrier = threading.Barrier(2)
    local = threading.local()

    def synchronized_registrations(*args, **kwargs):
        result = original_registrations(*args, **kwargs)
        if not getattr(local, "synchronized", False):
            local.synchronized = True
            barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(
        profile, "_worktree_registrations", synchronized_registrations
    )

    def cleanup(name: str) -> BaseException | None:
        try:
            profile._remove_target_worktree(
                repository, run_root, name=name
            )
        except BaseException as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(cleanup, ("control", "candidate")))

    assert all(
        outcome is None
        or str(outcome).startswith("target_worktree_unregister_failed:")
        for outcome in outcomes
    )
    monkeypatch.setattr(
        profile, "_worktree_registrations", original_registrations
    )
    for name in ("control", "candidate"):
        profile._remove_target_worktree(
            repository, run_root, name=name
        )

    assert profile._worktree_registrations(repository) == frozenset(
        {str(repository.resolve())}
    )
    _assert_retained_target_tombstone(run_root, "control")
    _assert_retained_target_tombstone(run_root, "candidate")


def test_remove_target_worktree_rejects_non_target_name(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    with pytest.raises(RuntimeError, match="target_worktree_invalid"):
        profile._remove_target_worktree(
            tmp_path / "repository",
            run_root,
            name="outside",
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


def test_remove_target_worktrees_attempts_both_when_first_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _init_real_worktree_repository(tmp_path)
    run_root = tmp_path / "run"
    for name in ("control", "candidate"):
        _add_real_worktree(repository, run_root / name, branch=name)
    original_move_admin = profile._move_worktree_admin_to_marker

    def fail_candidate(
        common: Path, target: str, admin_name: str, *claim
    ) -> int:
        if target.endswith("/candidate"):
            raise RuntimeError("target_worktree_unregister_failed")
        return original_move_admin(common, target, admin_name, *claim)

    monkeypatch.setattr(
        profile, "_move_worktree_admin_to_marker", fail_candidate
    )

    with pytest.raises(RuntimeError, match="target_worktree_unregister_failed:candidate"):
        profile._remove_target_worktrees(
            repository,
            run_root,
            names=("candidate", "control"),
        )

    assert (run_root / "candidate").is_dir()
    _assert_retained_target_tombstone(run_root, "control")
    assert profile._worktree_registrations(repository) == frozenset(
        {str(repository.resolve()), str(run_root / "candidate")}
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
