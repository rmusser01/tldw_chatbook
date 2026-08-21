"""Revision-pinned three-turn Console latency benchmark for TASK-19641."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import random
import re
import os
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from threading import Lock
from typing import IO, Any, Mapping
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request


ARMS = ("control", "disabled", "enabled")
CONTROL_SHA = "5f720a40417eaa78f33619d5cbc82effc470104b"
FIXED_MUTATION = b"task-19641 deterministic mutation\n"
TURN_PROMPTS = (
    "Reply with exactly turn one complete and do not use any tools.",
    (
        "Use exactly two tool calls and no others. First call load_tools with "
        "ids containing local:fs_write. Then call fs_write with path "
        "measured/turn-two.txt and content exactly task-19641 deterministic "
        "mutation followed by one newline. After the tool result reply with "
        "exactly turn two complete."
    ),
    "Reply with exactly turn three complete and do not use any tools.",
)
EXPECTED_ROUND_COUNTS = {"1": 1, "2": 3, "3": 1}
ALL_REVIEW_EVENTS = frozenset(
    {
        "baseline_started",
        "baseline_ready",
        "finalization_scheduled",
        "review_e_started",
        "review_e_completed",
    }
)


@dataclass(frozen=True)
class ArmContract:
    """Required and prohibited Change Review boundaries for one benchmark arm."""

    required_review: tuple[str, ...]
    prohibited_review: frozenset[str]


ARM_CONTRACTS = {
    "control": ArmContract(
        required_review=(
            "baseline_started",
            "baseline_ready",
            "review_e_started",
            "review_e_completed",
        ),
        prohibited_review=frozenset({"finalization_scheduled"}),
    ),
    "disabled": ArmContract(
        required_review=(),
        prohibited_review=ALL_REVIEW_EVENTS,
    ),
    "enabled": ArmContract(
        required_review=(
            "baseline_started",
            "baseline_ready",
            "finalization_scheduled",
            "review_e_started",
            "review_e_completed",
        ),
        prohibited_review=frozenset(),
    ),
}

NON_REGRESSION_METRICS = (
    "third_send_to_worker_ns",
    "event_loop_lag_p95_ns",
)
APPLICATION_CRITICAL_PATH_METRICS = (
    "assistant_durable_to_release_ns",
    "terminal_to_third_provider_ns",
)
REQUIRED_METRICS = NON_REGRESSION_METRICS + APPLICATION_CRITICAL_PATH_METRICS + (
    "provider_total_ns",
    "conversation_wall_ns",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TARGET_MODULES = (
    "tldw_chatbook",
    "Tests.UI.test_destination_shells",
    "Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation",
)
_SAFE_INHERITED_ENVIRONMENT = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
)
_OWNED_CHILD_ENVIRONMENT = (
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "TMPDIR",
    "TLDW_CONFIG_PATH",
)
_CHILD_SPEC_KEYS = frozenset(
    {
        "sample_id",
        "phase",
        "iteration",
        "arm",
        "target_root",
        "sample_root",
        "run_root",
        "evidence_path",
    }
)


@dataclass(frozen=True)
class SamplePlan:
    """One warmup or measured arm invocation owned by the parent."""

    phase: str
    arm: str
    iteration: int


@dataclass(frozen=True)
class ChildResult:
    """Bounded child-process outcome with its last durable evidence event."""

    status: str
    returncode: int | None
    last_event: dict[str, Any] | None


@dataclass
class TargetAdapter:
    """Contain revision-specific Change Review discovery behind one seam."""

    target_root: Path
    arm: str
    revision_kind: str
    _review_events: dict[str, int] = field(default_factory=dict, init=False)
    _restorations: list[Callable[[], None]] = field(default_factory=list, init=False)
    _event_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @classmethod
    def for_arm(cls, target_root: Path, arm: str) -> "TargetAdapter":
        root = target_root.resolve()
        workspace_package = root / "tldw_chatbook" / "Workspaces"
        tracker = workspace_package / "change_turn_tracker.py"
        consent = workspace_package / "change_review_consent.py"
        finalization = workspace_package / "change_review_finalization.py"
        bridge = root / "tldw_chatbook" / "Chat" / "console_agent_bridge.py"
        tracker_text = tracker.read_text(encoding="utf-8") if tracker.is_file() else ""
        bridge_text = bridge.read_text(encoding="utf-8") if bridge.is_file() else ""
        has_legacy_tracker = (
            "class ChangeTurnTracker" in tracker_text
            and "def begin_turn" in tracker_text
            and "def end_turn" in tracker_text
        )
        has_candidate = consent.is_file() and finalization.is_file()
        if arm == "control":
            has_legacy_bridge = (
                "_change_tracker.begin_turn" in bridge_text
                and "_change_tracker.end_turn" in bridge_text
                and "_change_finalization_coordinator" not in bridge_text
            )
            if not has_legacy_tracker or not has_legacy_bridge or has_candidate:
                raise RuntimeError("target_fingerprint_mismatch:control")
            kind = "legacy"
        elif arm in {"disabled", "enabled"}:
            if not has_legacy_tracker or not has_candidate:
                raise RuntimeError(f"target_fingerprint_mismatch:{arm}")
            consent_text = consent.read_text(encoding="utf-8")
            finalization_text = finalization.read_text(encoding="utf-8")
            if (
                "class ChangeReviewConsentService" not in consent_text
                or "class ChangeReviewFinalizationCoordinator" not in finalization_text
                or "def finalize" not in finalization_text
                or "def _worker_loop" not in finalization_text
                or "def finish_turn" not in tracker_text
                or "_change_finalization_coordinator.register" not in bridge_text
                or "_change_finalization_coordinator.finalize" not in bridge_text
            ):
                raise RuntimeError(f"target_fingerprint_mismatch:{arm}")
            kind = "candidate"
        else:
            raise RuntimeError("target_fingerprint_mismatch:unknown_arm")
        return cls(root, arm, kind)

    def _require_shape(self, expected_arm: str, expected_kind: str) -> None:
        if self.arm != expected_arm or self.revision_kind != expected_kind:
            raise RuntimeError(f"target_adapter_mismatch:{expected_arm}")

    def configure_control_review(self, app: Any, runtime: "WorkspaceRuntime") -> None:
        """Configure the legacy control without importing candidate services."""
        self._require_shape("control", "legacy")
        if runtime.consent_service is not None or runtime.review_state != "legacy":
            raise RuntimeError("target_adapter_mismatch:control_runtime")
        app.change_review_consent_service = None

    def configure_candidate_disabled_review(
        self, app: Any, runtime: "WorkspaceRuntime"
    ) -> None:
        """Install the explicit disabled consent service on the candidate app."""
        self._require_shape("disabled", "candidate")
        if runtime.consent_service is None or runtime.review_state != "disabled":
            raise RuntimeError("target_adapter_mismatch:disabled_runtime")
        app.change_review_consent_service = runtime.consent_service

    def configure_candidate_enabled_review(
        self, app: Any, runtime: "WorkspaceRuntime"
    ) -> None:
        """Install the ready, explicitly enabled consent service on the app."""
        self._require_shape("enabled", "candidate")
        if (
            runtime.consent_service is None
            or runtime.review_state != "enabled"
            or not runtime.review_ready
        ):
            raise RuntimeError("target_adapter_mismatch:enabled_runtime")
        app.change_review_consent_service = runtime.consent_service

    def configure_review(self, app: Any, runtime: "WorkspaceRuntime") -> None:
        """Dispatch arm configuration without leaking revision checks to the driver."""
        methods = {
            "control": self.configure_control_review,
            "disabled": self.configure_candidate_disabled_review,
            "enabled": self.configure_candidate_enabled_review,
        }
        methods[self.arm](app, runtime)

    def _record_review_event(self, name: str) -> None:
        with self._event_lock:
            self._review_events.setdefault(name, time.perf_counter_ns())

    def _wrap_method(
        self,
        owner: type[Any],
        name: str,
        wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]],
    ) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            raise RuntimeError(f"target_fingerprint_mismatch:{self.arm}:{name}")
        replacement = wrapper_factory(original)
        setattr(owner, name, replacement)

        def restore() -> None:
            setattr(owner, name, original)

        self._restorations.append(restore)

    def install_timing_wrappers(
        self,
        *,
        tracker_type: type[Any] | None = None,
        coordinator_type: type[Any] | None = None,
    ) -> None:
        """Install observational review-boundary wrappers before runtime creation."""
        if self._restorations:
            raise RuntimeError("target_adapter_wrappers_already_installed")
        if tracker_type is None:
            from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

            tracker_type = ChangeTurnTracker

        def before(event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def factory(original: Callable[..., Any]) -> Callable[..., Any]:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    self._record_review_event(event)
                    return original(*args, **kwargs)

                return wrapped

            return factory

        def after(event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def factory(original: Callable[..., Any]) -> Callable[..., Any]:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    result = original(*args, **kwargs)
                    self._record_review_event(event)
                    return result

                return wrapped

            return factory

        def around(
            started: str, completed: str
        ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def factory(original: Callable[..., Any]) -> Callable[..., Any]:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    self._record_review_event(started)
                    try:
                        return original(*args, **kwargs)
                    finally:
                        self._record_review_event(completed)

                return wrapped

            return factory

        if self.revision_kind == "legacy":
            self._require_shape("control", "legacy")
            self._wrap_method(tracker_type, "begin_turn", before("baseline_started"))

            def legacy_end(
                original: Callable[..., Any],
            ) -> Callable[..., Any]:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    handle = args[1] if len(args) > 1 else kwargs.get("handle")
                    await_baseline = getattr(handle, "await_baseline", None)
                    if not callable(await_baseline):
                        raise RuntimeError(
                            "target_fingerprint_mismatch:control:await_baseline"
                        )
                    await_baseline()
                    self._record_review_event("baseline_ready")
                    self._record_review_event("review_e_started")
                    try:
                        return original(*args, **kwargs)
                    finally:
                        self._record_review_event("review_e_completed")

                return wrapped

            self._wrap_method(
                tracker_type,
                "end_turn",
                legacy_end,
            )
            return

        if self.revision_kind != "candidate" or self.arm not in {
            "disabled",
            "enabled",
        }:
            raise RuntimeError(f"target_adapter_mismatch:{self.arm}")
        if coordinator_type is None:
            from tldw_chatbook.Workspaces.change_review_finalization import (
                ChangeReviewFinalizationCoordinator,
            )

            coordinator_type = ChangeReviewFinalizationCoordinator
        self._wrap_method(
            coordinator_type, "register", before("baseline_started")
        )
        self._wrap_method(
            coordinator_type, "await_baseline", after("baseline_ready")
        )
        # Timestamp the scheduling boundary before delegating: finalize can
        # wake a worker immediately, so recording after return can invert E.
        self._wrap_method(
            coordinator_type,
            "finalize",
            before("finalization_scheduled"),
        )
        self._wrap_method(
            tracker_type,
            "finish_turn",
            around("review_e_started", "review_e_completed"),
        )

    def review_events(self) -> dict[str, int]:
        """Return the content-free first observation for each review boundary."""
        with self._event_lock:
            return dict(self._review_events)

    def reset_review_events(self) -> None:
        """Start a fresh timing window after an unmeasured setup turn settles."""
        with self._event_lock:
            self._review_events.clear()

    def close(self) -> None:
        """Restore every wrapped target method in reverse installation order."""
        for restore in reversed(self._restorations):
            restore()
        self._restorations.clear()


@dataclass
class WorkspaceRuntime:
    """Owned workspace, review, and local-tool resources for one child sample."""

    workspace_id: str
    workspace_root: Path
    shadow_root: Path
    database: Any
    registry: Any
    binding: Any
    consent_service: Any
    review_state: str
    review_ready: bool
    control_plane: Any
    local_provider: Any
    hub: Any
    gate: Any
    permission_definition_hash: str

    def close(self) -> None:
        if self.consent_service is not None:
            self.consent_service.shutdown(timeout=2.0)
        self.database.close()


def balanced_arm_order(iteration: int) -> tuple[str, str, str]:
    """Return one complete arm triple with a rotating starting arm."""
    offset = iteration % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    """Return the one-based nearest-rank percentile for ``values``."""
    if not values or not 0 < fraction <= 1:
        raise ValueError("percentile requires values and 0 < fraction <= 1")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def paired_p95_ratio_bounds(
    blocks: Sequence[Mapping[str, float]],
    candidate: str,
    *,
    resamples: int = 10_000,
    seed: int = 19_641,
) -> dict[str, tuple[float, float] | float]:
    """Bootstrap candidate/control p95 ratios by complete iteration block."""
    if len(blocks) < 2:
        raise ValueError("paired bootstrap requires at least two blocks")
    if candidate not in ARMS or candidate == "control":
        raise ValueError("candidate must identify a non-control arm")
    if any(set(block) != set(ARMS) for block in blocks):
        raise ValueError("paired bootstrap requires complete blocks")
    if resamples < 1:
        raise ValueError("paired bootstrap requires at least one resample")

    control_p95 = nearest_rank_percentile(
        [float(block["control"]) for block in blocks], 0.95
    )
    if control_p95 <= 0:
        raise ValueError("paired bootstrap requires a positive control p95")

    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        sampled = [blocks[generator.randrange(len(blocks))] for _ in blocks]
        sampled_control = nearest_rank_percentile(
            [float(block["control"]) for block in sampled], 0.95
        )
        if sampled_control <= 0:
            raise ValueError("paired bootstrap requires a positive control p95")
        sampled_candidate = nearest_rank_percentile(
            [float(block[candidate]) for block in sampled], 0.95
        )
        ratios.append(sampled_candidate / sampled_control)

    return {
        "two_sided_95": (
            nearest_rank_percentile(ratios, 0.025),
            nearest_rank_percentile(ratios, 0.975),
        ),
        "one_sided_lower_95": nearest_rank_percentile(ratios, 0.05),
        "one_sided_upper_95": nearest_rank_percentile(ratios, 0.95),
    }


def sample_heartbeat_p95_ns(tick_lateness_ns: Sequence[int]) -> float:
    """Reduce one sample's raw heartbeat ticks to one equally weighted p95."""
    if not tick_lateness_ns:
        raise ValueError("heartbeat vector must not be empty")
    return nearest_rank_percentile(tick_lateness_ns, 0.95)


def _append_error(errors: list[str], code: str) -> None:
    if code not in errors:
        errors.append(code)


def validate_sample(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable fail-closed error codes for one terminal sample record."""
    errors: list[str] = []
    arm = row.get("arm")
    if arm not in ARM_CONTRACTS:
        return ("arm_unknown",)
    if row.get("status") != "complete":
        _append_error(errors, "sample_incomplete")
    if row.get("terminal_third_assistant_ns") is None:
        _append_error(errors, "terminal_third_assistant_missing")
    if row.get("third_provider_started_ns") is None:
        _append_error(errors, "third_provider_missing")
    third_provider_completed = row.get("terminal_third_provider_completed_ns")
    if third_provider_completed is None:
        _append_error(errors, "terminal_third_provider_missing")
    elif not (
        isinstance(third_provider_completed, int)
        and isinstance(row.get("third_provider_started_ns"), int)
        and row["third_provider_started_ns"] <= third_provider_completed
        and isinstance(row.get("terminal_third_assistant_ns"), int)
        and third_provider_completed <= row["terminal_third_assistant_ns"]
    ):
        _append_error(errors, "terminal_third_provider_timing")
    if row.get("provider_round_counts") != EXPECTED_ROUND_COUNTS:
        _append_error(errors, "provider_round_contract")
    provider_usage = row.get("provider_usage")
    if not isinstance(provider_usage, list) or len(provider_usage) != 5:
        _append_error(errors, "provider_usage_contract")
    elif any(
        not isinstance(usage, Mapping)
        or set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"}
        or any(
            not isinstance(usage.get(key), int)
            or isinstance(usage.get(key), bool)
            or usage[key] < 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        or usage["total_tokens"]
        != usage["prompt_tokens"] + usage["completion_tokens"]
        for usage in provider_usage
    ):
        _append_error(errors, "provider_usage_contract")
    trigger = row.get("terminal_turn_2_provider_completed_ns")
    requested = row.get("third_send_requested_ns")
    released = row.get("turn_2_release_ns")
    if not all(isinstance(value, int) for value in (trigger, requested, released)):
        _append_error(errors, "third_send_timing_missing")
    elif not trigger <= requested < released:
        _append_error(errors, "third_send_not_queued")
    heartbeat = row.get("heartbeat_lateness_ns")
    if not isinstance(heartbeat, list) or not heartbeat:
        _append_error(errors, "heartbeat_missing")
    elif any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in heartbeat
    ):
        _append_error(errors, "heartbeat_contract")
    if row.get("prompt_loss_count") != 0:
        _append_error(errors, "prompt_loss")
    if row.get("selected_binding_access") != "rw":
        _append_error(errors, "workspace_binding_not_rw")

    calls = row.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 2:
        _append_error(errors, "tool_call_contract")
        calls = []
    expected_payload = row.get("expected_payload_sha256")
    expected_permission = row.get("expected_permission_definition_hash")
    if not (
        isinstance(expected_payload, str)
        and _SHA256.fullmatch(expected_payload)
        and isinstance(expected_permission, str)
        and _SHA256.fullmatch(expected_permission)
    ):
        _append_error(errors, "expected_hash_contract")
    if len(calls) == 2:
        load_call, write_call = calls
        if not isinstance(load_call, Mapping) or (
            load_call.get("name") != "load_tools"
            or load_call.get("turn") != 2
            or load_call.get("provider_round") != 1
            or load_call.get("requested_tool_id") != "local:fs_write"
        ):
            _append_error(errors, "load_tools_contract")
        if not isinstance(write_call, Mapping) or (
            write_call.get("name") != "fs_write"
            or write_call.get("turn") != 2
            or write_call.get("provider_round") != 2
            or write_call.get("tool_id") != "local:fs_write"
            or write_call.get("path") != "measured/turn-two.txt"
            or write_call.get("payload_sha256") != expected_payload
        ):
            _append_error(errors, "fs_write_contract")
        if any(
            not isinstance(call, Mapping)
            or call.get("permission") != "allow"
            or call.get("definition_hash") != expected_permission
            for call in calls
        ):
            _append_error(errors, "permission_contract")
    mutation = row.get("mutation")
    if not isinstance(mutation, Mapping) or (
        mutation.get("path") != "measured/turn-two.txt"
        or mutation.get("payload_sha256") != expected_payload
        or mutation.get("success") is not True
    ):
        _append_error(errors, "mutation_contract")

    review_events = row.get("review_events")
    if not isinstance(review_events, Mapping):
        _append_error(errors, "review_event_missing")
        review_keys: set[str] = set()
    else:
        review_keys = set(review_events)
    contract = ARM_CONTRACTS[str(arm)]
    if not set(contract.required_review).issubset(review_keys):
        _append_error(errors, "review_event_missing")
    if contract.prohibited_review.intersection(review_keys):
        _append_error(errors, "review_event_prohibited")
    if review_keys:
        if any(
            not isinstance(review_events.get(key), int)
            for key in contract.required_review
            if key in review_events
        ):
            _append_error(errors, "review_event_timing")
        if {
            "baseline_started",
            "baseline_ready",
        }.issubset(review_keys) and not (
            review_events["baseline_started"] <= review_events["baseline_ready"]
        ):
            _append_error(errors, "review_event_timing")
        if {"review_e_started", "review_e_completed"}.issubset(review_keys) and not (
            review_events["review_e_started"] <= review_events["review_e_completed"]
        ):
            _append_error(errors, "review_event_timing")
        if {
            "finalization_scheduled",
            "review_e_started",
        }.issubset(review_keys) and not (
            review_events["finalization_scheduled"] <= review_events["review_e_started"]
        ):
            _append_error(errors, "review_event_timing")
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping) or any(
        metric not in metrics
        or not isinstance(metrics[metric], (int, float))
        or isinstance(metrics[metric], bool)
        or metrics[metric] < 0
        for metric in REQUIRED_METRICS
    ):
        _append_error(errors, "metrics_contract")
    return tuple(errors)


def validate_run(
    rows: Sequence[Mapping[str, Any]], *, expected_iterations: int = 30
) -> tuple[str, ...]:
    """Validate measured sample cardinality, identity, blocks, and contracts."""
    errors: list[str] = []
    warmups = [row for row in rows if row.get("phase") == "warmup"]
    if (
        len(warmups) != len(ARMS)
        or {row.get("arm") for row in warmups} != set(ARMS)
        or any(validate_sample(row) for row in warmups)
    ):
        _append_error(errors, "warmup_contract")
    measured = [row for row in rows if row.get("phase") == "measured"]
    if len(measured) != expected_iterations * len(ARMS):
        _append_error(errors, "sample_count")
    sample_ids = [row.get("sample_id") for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        _append_error(errors, "sample_id_duplicate")
    for iteration in range(expected_iterations):
        block = [row for row in measured if row.get("iteration") == iteration]
        if len(block) != len(ARMS) or {row.get("arm") for row in block} != set(ARMS):
            _append_error(errors, "rotation_block_contract")
    for row in measured:
        for code in validate_sample(row):
            _append_error(errors, f"sample:{code}")
    return tuple(errors)


def _metric_blocks(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> list[dict[str, float]]:
    by_iteration: dict[int, dict[str, float]] = {}
    for row in rows:
        iteration = int(row["iteration"])
        metrics = row["metrics"]
        by_iteration.setdefault(iteration, {})[str(row["arm"])] = float(metrics[metric])
    return [by_iteration[index] for index in sorted(by_iteration)]


def _bound_verdict(bounds: Mapping[str, Any], *, ceiling: float) -> str:
    if float(bounds["one_sided_upper_95"]) <= ceiling:
        return "pass"
    if float(bounds["one_sided_lower_95"]) > ceiling:
        return "regression"
    return "inconclusive"


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 19_641,
) -> dict[str, Any]:
    """Build recomputable arm summaries and conservative benchmark verdicts."""
    validation_errors = validate_run(rows)
    if validation_errors:
        return {
            "overall_verdict": "invalid",
            "validation_errors": list(validation_errors),
            "arms": {},
            "critical_path_improvement_claims": {},
        }

    measured = [row for row in rows if row.get("phase") == "measured"]
    metric_names = tuple(measured[0]["metrics"])
    arm_summaries: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in measured if row["arm"] == arm]
        distributions = {}
        for metric in metric_names:
            values = [float(row["metrics"][metric]) for row in arm_rows]
            distributions[metric] = {
                "median": statistics.median(values),
                "p95": nearest_rank_percentile(values, 0.95),
            }
        arm_summaries[arm] = {"metrics": distributions}

    claims: dict[str, Any] = {}
    candidate_verdicts: list[str] = []
    for arm in ARMS[1:]:
        gates: dict[str, Any] = {}
        for metric in NON_REGRESSION_METRICS:
            bounds = paired_p95_ratio_bounds(
                _metric_blocks(measured, metric),
                arm,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            gates[metric] = {
                "bounds": bounds,
                "verdict": _bound_verdict(bounds, ceiling=1.10),
            }
        gate_verdicts = [gate["verdict"] for gate in gates.values()]
        arm_verdict = (
            "regression"
            if "regression" in gate_verdicts
            else "pass"
            if all(verdict == "pass" for verdict in gate_verdicts)
            else "inconclusive"
        )
        arm_summaries[arm]["gates"] = gates
        arm_summaries[arm]["verdict"] = arm_verdict
        candidate_verdicts.append(arm_verdict)
        for metric in APPLICATION_CRITICAL_PATH_METRICS:
            bounds = paired_p95_ratio_bounds(
                _metric_blocks(measured, metric),
                arm,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            if float(bounds["one_sided_upper_95"]) < 1.0:
                claims.setdefault(metric, {})[arm] = bounds
    arm_summaries["control"]["verdict"] = "reference"
    overall = (
        "regression"
        if "regression" in candidate_verdicts
        else "pass"
        if all(verdict == "pass" for verdict in candidate_verdicts)
        else "inconclusive"
    )
    return {
        "overall_verdict": overall,
        "validation_errors": [],
        "arms": arm_summaries,
        "critical_path_improvement_claims": claims,
    }


_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "headers",
        "environment",
        "environ",
        "command_line",
        "prompt",
        "response",
        "content",
        "tool_result",
        "file_content",
    }
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def privacy_violations(value: Any, *, location: str = "$") -> tuple[str, ...]:
    """Return sensitive-field and unnormalized absolute-path locations."""
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            child = f"{location}.{key}"
            if key_text in _FORBIDDEN_KEYS or key_text.endswith("_api_key"):
                violations.append(f"sensitive_key:{child}")
            violations.extend(privacy_violations(nested, location=child))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            violations.extend(
                privacy_violations(nested, location=f"{location}[{index}]")
            )
    elif isinstance(value, str) and (
        value.startswith("/") or _WINDOWS_ABSOLUTE.match(value)
    ):
        violations.append(f"absolute_path:{location}")
    return tuple(violations)


def normalize_text(text: str, roots: Mapping[str, Path]) -> str:
    """Replace known absolute roots with stable aliases, longest path first."""
    normalized = text
    ordered = sorted(
        ((alias, str(path.resolve())) for alias, path in roots.items()),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for alias, root in ordered:
        normalized = normalized.replace(root, alias)
    return normalized


def safe_error_code(error: BaseException) -> str:
    """Retain a stable machine code only; discard arbitrary exception copy."""
    value = str(error).strip()
    return value if re.fullmatch(r"[a-z0-9_.:-]{1,120}", value) else "unclassified"


def write_boundary_event(destination: IO[str], event: Mapping[str, Any]) -> None:
    """Write and immediately flush one low-frequency evidence boundary."""
    destination.write(json.dumps(dict(event), sort_keys=True, separators=(",", ":")))
    destination.write("\n")
    destination.flush()


class HeartbeatBuffer:
    """Fixed-capacity in-memory heartbeat storage with no I/O ownership."""

    def __init__(self, *, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("heartbeat capacity must be positive")
        self._values = [0] * capacity
        self._length = 0

    def record(self, tick_lateness_ns: int) -> None:
        if self._length >= len(self._values):
            raise OverflowError("heartbeat buffer capacity exceeded")
        self._values[self._length] = int(tick_lateness_ns)
        self._length += 1

    def values(self) -> list[int]:
        return self._values[: self._length]


def write_terminal_sample(
    destination: IO[str], sample: Mapping[str, Any], heartbeat: HeartbeatBuffer
) -> None:
    """Emit a terminal sample and its heartbeat vector exactly once."""
    terminal = dict(sample)
    terminal["heartbeat_lateness_ns"] = heartbeat.values()
    write_boundary_event(destination, terminal)


def build_child_environment(
    base_environment: Mapping[str, str], sample_root: Path
) -> dict[str, str]:
    """Construct a credential-free child environment rooted in one sample."""
    root = sample_root.resolve()
    environment = {
        key: base_environment[key]
        for key in _SAFE_INHERITED_ENVIRONMENT
        if base_environment.get(key)
    }
    environment.update(
        {
            "HOME": str(root / "home"),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "TMPDIR": str(root / "tmp"),
            "TLDW_CONFIG_PATH": str(root / "config" / "tldw_cli" / "config.toml"),
            "TLDW_TEST_MODE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def assert_child_environment(
    sample_root: Path, environment: Mapping[str, str]
) -> None:
    """Fail before target imports unless the child owns its complete environment."""
    root = sample_root.resolve()
    allowed = set(_SAFE_INHERITED_ENVIRONMENT) | set(_OWNED_CHILD_ENVIRONMENT) | {
        "TLDW_TEST_MODE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        # macOS inserts this locale descriptor even when subprocess.env is
        # otherwise exact. It contains no path, credential, or user content.
        "__CF_USER_TEXT_ENCODING",
    }
    if set(environment) - allowed:
        raise RuntimeError("child_environment_mismatch:unexpected_key")
    for key in _OWNED_CHILD_ENVIRONMENT:
        value = environment.get(key)
        if not value or not Path(value).resolve().is_relative_to(root):
            raise RuntimeError(f"child_environment_mismatch:{key}")
    expected_flags = {
        "TLDW_TEST_MODE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if any(environment.get(key) != value for key, value in expected_flags.items()):
        raise RuntimeError("child_environment_mismatch:flags")


def write_child_spec(path: Path, spec: Mapping[str, Any]) -> None:
    """Persist one parent-owned child specification with an exact schema."""
    if set(spec) != _CHILD_SPEC_KEYS:
        raise RuntimeError("child_spec_invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(spec), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def read_child_spec(path: Path) -> dict[str, Any]:
    """Read one child specification and reject schema/type drift."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("child_spec_invalid") from exc
    if not isinstance(value, dict) or set(value) != _CHILD_SPEC_KEYS:
        raise RuntimeError("child_spec_invalid")
    if (
        value.get("arm") not in ARMS
        or value.get("phase") not in {"warmup", "measured"}
        or not isinstance(value.get("iteration"), int)
        or not isinstance(value.get("sample_id"), str)
        or any(
            not isinstance(value.get(key), str)
            for key in ("target_root", "sample_root", "run_root", "evidence_path")
        )
    ):
        raise RuntimeError("child_spec_invalid")
    return value


def write_child_config(sample_root: Path, *, endpoint: str, model: str) -> Path:
    """Write the minimal isolated config before any target application import."""
    root = sample_root.resolve()
    for relative in ("home", "config", "data", "cache", "tmp"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config_path = root / "config" / "tldw_cli" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            (
                "[first_run]",
                "setup_completed = true",
                "",
                "[splash_screen]",
                "enabled = false",
                "",
                "[model_catalog]",
                "auto_refresh_enabled = false",
                "refresh_consent_recorded = false",
                "",
                "[subscriptions]",
                "enable_background_checking = false",
                "",
                "[console]",
                "local_tools_enabled = true",
                'workspace_root = ""',
                "",
                "[api_settings.llama_cpp]",
                f"api_url = {json.dumps(endpoint)}",
                f"model = {json.dumps(model)}",
                "temperature = 0.0",
                "max_tokens = 512",
                'reasoning_effort = "none"',
                "timeout = 120",
                "retries = 0",
                "streaming = true",
                "",
            )
        ),
        encoding="utf-8",
    )
    return config_path


def preflight_provider(
    endpoint: str,
    model: str,
    *,
    urlopen: Callable[..., Any] | None = None,
    probe_timeout: float = 10.0,
    completion_timeout: float = 120.0,
) -> dict[str, Any]:
    """Verify one credential-free loopback llama.cpp endpoint and model."""
    parsed = urlsplit(endpoint.strip())
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") not in {"", "/v1"}
        or not model.strip()
    ):
        raise ValueError("preflight_endpoint_refused")
    if urlopen is None:
        from urllib.request import urlopen as stdlib_urlopen

        urlopen = stdlib_urlopen
    base_path = parsed.path.rstrip("/")
    api_path = base_path if base_path == "/v1" else f"{base_path}/v1"
    authority = parsed.netloc
    api_base = urlunsplit((parsed.scheme, authority, api_path, "", ""))

    models_request = Request(
        f"{api_base}/models",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urlopen(models_request, timeout=probe_timeout) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError("provider_preflight_models_failed")
        models_payload = json.loads(response.read())
    data = models_payload.get("data") if isinstance(models_payload, Mapping) else None
    model_ids = [
        str(item.get("id"))
        for item in data
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    ] if isinstance(data, list) else []
    if model not in model_ids:
        raise RuntimeError("provider_preflight_model_mismatch")

    completion_body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly synthetic ready.",
                }
            ],
            "temperature": 0.0,
            "max_tokens": 16,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    completion_request = Request(
        f"{api_base}/chat/completions",
        data=completion_body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(completion_request, timeout=completion_timeout) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError("provider_preflight_completion_failed")
        completion_payload = json.loads(response.read())
    choices = (
        completion_payload.get("choices")
        if isinstance(completion_payload, Mapping)
        else None
    )
    message = (
        choices[0].get("message")
        if isinstance(choices, list)
        and choices
        and isinstance(choices[0], Mapping)
        else None
    )
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("provider_preflight_completion_empty")
    return {
        "status": "ready",
        "model": model,
        "model_count": len(model_ids),
        "completion_chars": len(content),
    }


def extract_sse_usage(line: str) -> dict[str, int] | None:
    """Extract only coherent aggregate token counts from one SSE line."""
    if not line.startswith("data: {"):
        return None
    try:
        payload = json.loads(line[6:])
    except (TypeError, ValueError):
        return None
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    if not isinstance(usage, Mapping):
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    values = {key: usage.get(key) for key in keys}
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values.values()
    ):
        return None
    if values["total_tokens"] != (
        values["prompt_tokens"] + values["completion_tokens"]
    ):
        return None
    return values


def runtime_metadata(
    *, version_lookup: Callable[[str], str] = importlib_metadata.version
) -> dict[str, Any]:
    """Return portable interpreter, SQLite, and direct dependency versions."""
    dependencies = {
        name: version_lookup(name)
        for name in ("httpx", "pydantic", "rich", "textual")
    }
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "sqlite": sqlite3.sqlite_version,
        "dependencies": dependencies,
    }


def host_load_snapshot() -> dict[str, Any]:
    """Return content-free host capacity and scheduler load facts."""
    return {
        "logical_cpu_count": os.cpu_count(),
        "load_average": [float(value) for value in os.getloadavg()],
    }


def provider_server_metadata(
    endpoint: str,
    model: str,
    *,
    urlopen: Callable[..., Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Read a strict allowlist of reproducibility facts from llama.cpp props."""
    parsed = urlsplit(endpoint.strip())
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") not in {"", "/v1"}
        or not model.strip()
    ):
        raise ValueError("provider_metadata_endpoint_refused")
    if urlopen is None:
        from urllib.request import urlopen as stdlib_urlopen

        urlopen = stdlib_urlopen
    server_base = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    request = Request(
        f"{server_base}/props",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        if getattr(response, "status", 200) != 200:
            raise RuntimeError("provider_metadata_failed")
        payload = json.loads(response.read())
    if not isinstance(payload, Mapping) or payload.get("model_alias") != model:
        raise RuntimeError("provider_metadata_model_mismatch")
    generation = payload.get("default_generation_settings")
    modalities = payload.get("modalities")
    result = {
        "build_info": payload.get("build_info"),
        "model_alias": payload.get("model_alias"),
        "total_slots": payload.get("total_slots"),
        "context_tokens": (
            generation.get("n_ctx") if isinstance(generation, Mapping) else None
        ),
        "endpoints": {
            "metrics": payload.get("endpoint_metrics"),
            "slots": payload.get("endpoint_slots"),
            "props": payload.get("endpoint_props"),
        },
        "is_sleeping": payload.get("is_sleeping"),
        "modalities": {
            "vision": modalities.get("vision") if isinstance(modalities, Mapping) else None,
            "audio": modalities.get("audio") if isinstance(modalities, Mapping) else None,
        },
    }
    if (
        not isinstance(result["build_info"], str)
        or not isinstance(result["total_slots"], int)
        or not isinstance(result["context_tokens"], int)
        or any(not isinstance(value, bool) for value in result["endpoints"].values())
        or not isinstance(result["is_sleeping"], bool)
        or any(not isinstance(value, bool) for value in result["modalities"].values())
    ):
        raise RuntimeError("provider_metadata_contract_failed")
    return result


def listener_resource_snapshot(
    endpoint: str,
    *,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    """Measure local listener RSS/CPU without retaining PID or command text."""
    parsed = urlsplit(endpoint.strip())
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
    ):
        raise ValueError("listener_endpoint_refused")
    lookup = run_command(
        [
            "lsof",
            "-nP",
            "-t",
            f"-iTCP:{parsed.port}",
            "-sTCP:LISTEN",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    pids = sorted({line.strip() for line in lookup.stdout.splitlines() if line.strip()})
    if lookup.returncode != 0 or not pids or any(not pid.isdecimal() for pid in pids):
        raise RuntimeError("listener_inventory_failed")
    processes = []
    for pid in pids:
        sample = run_command(
            ["ps", "-o", "rss=,%cpu=", "-p", pid],
            check=False,
            capture_output=True,
            text=True,
        )
        fields = sample.stdout.split()
        if sample.returncode != 0 or len(fields) != 2:
            raise RuntimeError("listener_resource_sample_failed")
        try:
            rss_kib = int(fields[0])
            cpu_percent = float(fields[1])
        except ValueError as exc:
            raise RuntimeError("listener_resource_sample_failed") from exc
        if rss_kib < 0 or not math.isfinite(cpu_percent) or cpu_percent < 0:
            raise RuntimeError("listener_resource_sample_failed")
        processes.append(
            {"rss_bytes": rss_kib * 1_024, "cpu_percent": cpu_percent}
        )
    return {"listener_count": len(processes), "processes": processes}


def sample_schedule(iterations: int) -> tuple[SamplePlan, ...]:
    """Return fail-fast warmups followed by complete rotated measured blocks."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    schedule = [SamplePlan("warmup", arm, -1) for arm in ARMS]
    for iteration in range(iterations):
        schedule.extend(
            SamplePlan("measured", arm, iteration)
            for arm in balanced_arm_order(iteration)
        )
    return tuple(schedule)


def install_target_root(target_root: Path) -> None:
    """Clear candidate modules and make one immutable target import-first."""
    resolved = target_root.resolve()
    if not resolved.is_dir():
        raise RuntimeError("target_import_mismatch: target root does not exist")
    sys.dont_write_bytecode = True
    for name in tuple(sys.modules):
        if name == "tldw_chatbook" or name.startswith("tldw_chatbook."):
            sys.modules.pop(name, None)
        elif name == "Tests" or name.startswith("Tests."):
            sys.modules.pop(name, None)
    resolved_text = str(resolved)
    sys.path[:] = [entry for entry in sys.path if entry != resolved_text]
    sys.path.insert(0, resolved_text)
    importlib.invalidate_caches()


def assert_target_modules(
    modules: Sequence[str] | Mapping[str, Path], target_root: Path
) -> dict[str, str]:
    """Import or inspect modules and require every file below ``target_root``."""
    target = target_root.resolve()
    paths: dict[str, Path] = {}
    if isinstance(modules, Mapping):
        paths = {str(name): Path(path).resolve() for name, path in modules.items()}
    else:
        for name in modules:
            module = importlib.import_module(name)
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                raise RuntimeError(f"target_import_mismatch: {name} has no file")
            paths[name] = Path(module_file).resolve()
    mismatches = {
        name: path for name, path in paths.items() if not path.is_relative_to(target)
    }
    if mismatches:
        names = ", ".join(sorted(mismatches))
        raise RuntimeError(f"target_import_mismatch: {names}")
    return {name: str(path) for name, path in paths.items()}


def _last_jsonl_event(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        return None
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run_child_with_watchdog(
    command: Sequence[str],
    *,
    evidence_path: Path,
    timeout_seconds: float,
    term_grace_seconds: float = 5.0,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> ChildResult:
    """Run one child in its own process group with TERM/KILL deadlines."""
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    status: str
    try:
        process.communicate(timeout=timeout_seconds)
        status = "complete" if process.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=term_grace_seconds)
            status = "timed_out_terminated"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            status = "timed_out_killed"
    return ChildResult(
        status=status,
        returncode=process.returncode,
        last_event=_last_jsonl_event(evidence_path),
    )


def resolve_benchmark_revisions(
    repository_root: Path,
    *,
    control_ref: str,
    candidate_ref: str,
    run_command: Any = subprocess.run,
) -> dict[str, str]:
    """Resolve and freeze the exact control and candidate commit hashes."""
    resolved: dict[str, str] = {}
    for arm, revision in (("control", control_ref), ("candidate", candidate_ref)):
        completed = run_command(
            ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        value = completed.stdout.strip() if completed.returncode == 0 else ""
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise RuntimeError(f"revision_resolution_failed:{arm}")
        resolved[arm] = value
    if resolved["control"] != CONTROL_SHA:
        raise RuntimeError("control_revision_mismatch")
    return resolved


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the application-import-free parent/child benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--control-sha", default=CONTROL_SHA)
    parser.add_argument("--candidate-sha", default="HEAD")
    parser.add_argument("--sample-timeout", type=float, default=900.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--child-spec", type=Path)
    return parser.parse_args(arguments)


def prepare_control_worktree(
    repository_root: Path,
    run_root: Path,
    *,
    control_sha: str,
    run_command: Any = subprocess.run,
) -> Path:
    """Create one detached control worktree beneath the explicit run root."""
    target = (run_root / "control-worktree").resolve()
    if target.exists():
        raise RuntimeError("control_worktree_failed: target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "worktree", "add", "--detach", str(target), control_sha]
    completed = run_command(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not target.is_dir():
        detail = completed.stderr.strip()
        raise RuntimeError(f"control_worktree_failed:{detail}")
    return target


def _fixed_bytes(label: str, size: int) -> bytes:
    seed = hashlib.sha256(label.encode("utf-8")).digest()
    repetitions, remainder = divmod(size, len(seed))
    return seed * repetitions + seed[:remainder]


def content_tree_digest(root: Path) -> str:
    """Hash relative paths, sizes, and file bytes without Git metadata."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def generate_corpus(
    root: Path,
    *,
    file_count: int = 1_024,
    file_size: int = 4 * 1_024,
    blob_size: int = 8 * 1_024 * 1_024,
) -> dict[str, Any]:
    """Create the fixed synthetic workspace corpus and its portable manifest."""
    corpus = root / "corpus"
    measured = root / "measured"
    corpus.mkdir(parents=True, exist_ok=True)
    measured.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for index in range(file_count):
        path = corpus / f"{index:04d}.bin"
        payload = _fixed_bytes(f"corpus-{index}", file_size)
        path.write_bytes(payload)
        manifest.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    blob = corpus / "tracked-blob.bin"
    payload = _fixed_bytes("tracked-blob", blob_size)
    blob.write_bytes(payload)
    manifest.append(
        {
            "path": blob.relative_to(root).as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    return {"files": manifest, "content_tree_digest": content_tree_digest(root)}


def prepare_workspace_runtime(
    sample_root: Path,
    *,
    arm: str,
    readiness_timeout: float = 30.0,
) -> WorkspaceRuntime:
    """Build real isolated workspace/review/permission services for one sample."""
    if arm not in ARMS:
        raise ValueError("unknown benchmark arm")
    root = sample_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    workspace_root = root / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    database_root = root / "database"
    database_root.mkdir(parents=True, exist_ok=True)
    shadow_root = root / "shadow" / "change_review"

    from types import SimpleNamespace

    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.MCP.permission_store import definition_hash
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
    from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService

    database = WorkspaceDB(
        database_root / "workspaces.sqlite", client_id="task-19641-benchmark"
    )
    registry = LocalWorkspaceRegistryService(database)
    workspace_id = "benchmark-workspace"
    registry.create_workspace(workspace_id=workspace_id, name="Benchmark")
    registry.set_active_workspace(workspace_id)
    binding = registry.add_folder_binding(
        workspace_id, workspace_root, allow_write=True
    )

    consent_service = None
    review_ready = False
    review_state = "legacy" if arm == "control" else arm
    shadow_service = ShadowRepoService(data_dir=shadow_root)
    if arm == "control":
        registry.set_change_review_enabled(workspace_id, True)
        shadow_service.repo_for_root(workspace_root).snapshot("root registered")
        review_ready = True
    else:
        from tldw_chatbook.Workspaces.change_review_consent import (
            ChangeReviewCapability,
            ChangeReviewConsentService,
            ChangeReviewState,
            RootReadinessState,
        )

        registry.set_change_review_enabled(workspace_id, arm == "enabled")
        consent_service = ChangeReviewConsentService(
            registry,
            initialize_root=lambda path: shadow_service.repo_for_root(path).snapshot(
                "root registered"
            ),
            capability_reader=lambda: ChangeReviewCapability(
                ChangeReviewState.ENABLED
            ),
            worker_count=1,
        )
        registry.attach_change_review_consent_service(consent_service)
        admission = consent_service.admit_turn(workspace_id)
        if arm == "disabled":
            if admission.ready_roots or admission.skipped_roots:
                raise RuntimeError("disabled_review_scheduled_work")
        else:
            deadline = time.monotonic() + readiness_timeout
            while True:
                status = consent_service.status(workspace_id)
                if status.roots and all(
                    item.state is RootReadinessState.READY for item in status.roots
                ):
                    review_ready = True
                    break
                if status.roots and any(
                    item.state is RootReadinessState.FAILED for item in status.roots
                ):
                    raise RuntimeError("change_review_initialization_failed")
                if time.monotonic() >= deadline:
                    raise RuntimeError("change_review_initialization_timed_out")
                time.sleep(0.01)

    store_root = root / "permissions"
    store_root.mkdir(parents=True, exist_ok=True)
    local_store = LocalMCPStore(store_root / "local_mcp.json")
    control_plane = UnifiedMCPControlPlaneService(
        target_store=None,
        context_store=None,
        local_service=SimpleNamespace(store=local_store),
        server_service=None,
    )
    local_provider = LocalToolProvider(workspace_root=workspace_root, allow_write=True)
    hub = local_provider.hub_tool_for("fs_write")
    control_plane.set_tool_state(
        hub.server_key,
        hub.name,
        "allow",
        tool=hub,
    )
    gate = control_plane.gate_tool_test(hub)
    if gate.state != "allow" or gate.origin != "tool_override":
        raise RuntimeError("fs_write_permission_not_allowed")
    return WorkspaceRuntime(
        workspace_id=workspace_id,
        workspace_root=workspace_root,
        shadow_root=shadow_root,
        database=database,
        registry=registry,
        binding=binding,
        consent_service=consent_service,
        review_state=review_state,
        review_ready=review_ready,
        control_plane=control_plane,
        local_provider=local_provider,
        hub=hub,
        gate=gate,
        permission_definition_hash=definition_hash(
            hub.description,
            hub.input_schema,
        ),
    )


async def run_scripted_mounted_sample(
    sample_root: Path,
    *,
    arm: str,
) -> dict[str, Any]:
    """Drive the real mounted composer/queue/tool path with no network calls."""
    import asyncio
    import threading

    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
        _visible_text,
    )
    from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Widgets.Console import ConsoleComposerBar

    runtime = prepare_workspace_runtime(sample_root, arm=arm)
    (runtime.workspace_root / "measured").mkdir(exist_ok=True)

    class ScriptedGateway:
        def __init__(self) -> None:
            self.calls = 0
            self.turn_two_started = threading.Event()
            self.turn_two_terminal = threading.Event()
            self.third_terminal = threading.Event()
            self.third_provider_started_ns: int | None = None
            self.tool_calls: list[str] = []

        async def resolve_for_send(self, selection: Any) -> ConsoleProviderResolution:
            return ConsoleProviderResolution(
                provider="llama_cpp",
                base_url=selection.base_url or "http://127.0.0.1:9099",
                model="scripted-model",
                ready=True,
                readiness_key="llama_cpp",
                execution_key="llama_cpp",
            )

        async def stream_chat(self, _resolution: Any, _messages: Any, **_kwargs: Any):
            self.calls += 1
            call = self.calls
            if call == 1:
                yield "turn-one-complete"
                return
            if call == 2:
                self.turn_two_started.set()
                self.tool_calls.append("load_tools")
                payload = {
                    "name": "load_tools",
                    "arguments": {"ids": ["local:fs_write"]},
                }
                yield f"{FENCE_OPEN}\n{json.dumps(payload)}\n```"
                return
            if call == 3:
                self.tool_calls.append("fs_write")
                payload = {
                    "name": "fs_write",
                    "arguments": {
                        "path": "measured/turn-two.txt",
                        "content": FIXED_MUTATION.decode("utf-8"),
                    },
                }
                yield f"{FENCE_OPEN}\n{json.dumps(payload)}\n```"
                return
            if call == 4:
                try:
                    yield "turn-two-complete"
                finally:
                    self.turn_two_terminal.set()
                return
            if call == 5:
                self.third_provider_started_ns = time.perf_counter_ns()
                try:
                    yield "turn-three-complete"
                finally:
                    self.third_terminal.set()
                return
            raise AssertionError(f"unexpected provider call {call}")

    gateway = ScriptedGateway()
    app = _build_test_app()
    old_consent = getattr(app, "change_review_consent_service", None)
    if old_consent is not None:
        old_consent.shutdown(timeout=1.0)
    old_workspace_db = getattr(app, "local_workspace_db", None)
    if old_workspace_db is not None:
        old_workspace_db.close()
    app.local_workspace_db = runtime.database
    app.workspace_registry_service = runtime.registry
    app.change_review_consent_service = runtime.consent_service
    app.unified_mcp_service = runtime.control_plane
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "scripted-model",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "scripted-model",
            "temperature": 0.0,
            "max_tokens": 512,
        }
    }
    app.app_config.setdefault("console", {})["workspace_root"] = str(
        runtime.workspace_root
    )
    # ChatScreen deliberately refreshes disk-shaped snapshots through
    # load_settings(). This benchmark owns an injected, sample-scoped
    # snapshot instead, so remove the two live-config marker sections and
    # make its workspace/provider values authoritative for the mounted run.
    app.app_config.pop("general", None)
    app.app_config.pop("logging", None)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "scripted-model"
    chat_db = CharactersRAGDB(
        sample_root / "database" / "chatbook.sqlite",
        client_id="task-19641-mounted",
    )
    app.chachanotes_db = chat_db
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    third_send_requested_ns: int | None = None
    turn_2_release_ns: int | None = None
    allow_turn_2_release = asyncio.Event()
    console_runtime: Any | None = None

    async def wait_until(predicate: Any, *, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError("mounted sample did not settle before timeout")
            await asyncio.sleep(0.005)

    try:
        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            composer = console.query_one("#console-native-composer", ConsoleComposerBar)
            controller = console._ensure_console_chat_controller()
            benchmark_session = next(
                session
                for session in controller.store.sessions()
                if session.id == controller.store.active_session_id
            )
            if benchmark_session.workspace_id != runtime.workspace_id:
                raise AssertionError(
                    "mounted session selected the wrong workspace: "
                    f"{benchmark_session.workspace_id!r} != {runtime.workspace_id!r}"
                )
            # Project-instruction disclosure is a separate, one-time consent
            # interaction. Disable it for benchmark-owned sessions; the
            # explicit Console workspace_root above keeps local tools confined
            # to this same isolated workspace without timing an unrelated
            # unattended modal.
            controller.store.set_session_project_instruction_state(
                controller.store.active_session_id,
                ProjectInstructionControlState.legacy_disabled(),
            )
            coordinator = controller.prompt_queue_coordinator
            original_after_turn = coordinator._after_turn

            async def observed_after_turn(session_id: str, result: Any) -> None:
                nonlocal turn_2_release_ns
                if gateway.calls == 4 and turn_2_release_ns is None:
                    await allow_turn_2_release.wait()
                    turn_2_release_ns = time.perf_counter_ns()
                await original_after_turn(session_id, result)

            coordinator._after_turn = observed_after_turn

            async def type_text(text: str) -> None:
                composer.focus()
                await pilot.press(*tuple(text.lower()))

            await type_text(TURN_PROMPTS[0])
            await pilot.press("enter")
            try:
                await wait_until(
                    lambda: "turn-one-complete" in _visible_text(console)
                )
            except AssertionError as exc:
                send_button = console.query_one("#console-send-message")
                setup_modals = list(console.query("#console-setup-modal"))
                raise AssertionError(
                    "turn-one diagnostics: "
                    f"gateway_calls={gateway.calls}, "
                    f"draft_length={len(composer.draft_text())}, "
                    f"focused={type(host.focused).__name__}, "
                    f"send_disabled={send_button.disabled}, "
                    f"setup={[modal.display for modal in setup_modals]}, "
                    f"run_status={controller.run_state.status.value}, "
                    f"visible_copy={controller.run_state.visible_copy!r}, "
                    "messages="
                    f"{[(message.role.value, message.status, len(message.content)) for message in controller.store.messages_for_session(controller.store.active_session_id)]!r}"
                ) from exc

            await type_text(TURN_PROMPTS[1])
            await pilot.press("enter")
            await wait_until(gateway.turn_two_started.is_set)
            await type_text(TURN_PROMPTS[2])
            assert composer.draft_text() == TURN_PROMPTS[2].lower()
            await wait_until(gateway.turn_two_terminal.is_set)
            third_send_requested_ns = time.perf_counter_ns()
            try:
                await pilot.press("enter")
            finally:
                allow_turn_2_release.set()
            await wait_until(gateway.third_terminal.is_set)
            await wait_until(lambda: "turn-three-complete" in _visible_text(console))
            await pilot.pause(0.05)

            store = console._ensure_console_chat_store()
            assistants = [
                message
                for message in store.messages_for_session(store.active_session_id)
                if message.role is ConsoleMessageRole.ASSISTANT
            ]
            terminal = assistants[-1].content
            mutation_path = runtime.workspace_root / "measured/turn-two.txt"
            if not mutation_path.exists():
                rows = store.messages_for_session(store.active_session_id)
                raise AssertionError(
                    "fs_write did not create the benchmark mutation: "
                    f"{[(message.role.value, message.status, len(message.content)) for message in rows]!r}"
                )
            console_runtime = console._console_runtime()
            await console_runtime.dispose()
    finally:
        if console_runtime is not None:
            await console_runtime.dispose()
        close_chat_db = getattr(chat_db, "close_connection", None)
        if callable(close_chat_db):
            close_chat_db()
        runtime.close()

    if third_send_requested_ns is None or turn_2_release_ns is None:
        raise RuntimeError("mounted sample missed queue timing boundaries")
    return {
        "provider_round_counts": {"1": 1, "2": 3, "3": 1},
        "tool_calls": gateway.tool_calls,
        "third_send_requested_ns": third_send_requested_ns,
        "turn_2_release_ns": turn_2_release_ns,
        "third_provider_started_ns": gateway.third_provider_started_ns,
        "terminal_third_assistant": terminal,
    }


class _MountedObservation:
    """Thread-safe, body-free observations for one mounted real-provider sample."""

    def __init__(
        self,
        *,
        ui_loop: Any,
        turn_two_terminal: Any,
        third_assistant_terminal: Any,
        permission_hash: str,
    ) -> None:
        self._lock = Lock()
        self._ui_loop = ui_loop
        self._turn_two_terminal = turn_two_terminal
        self._third_assistant_terminal = third_assistant_terminal
        self._permission_hash = permission_hash
        self.dispatch_count = 0
        self.accepted_count = 0
        self.drain_count = 0
        self.worker_count = 0
        self.after_turn_count = 0
        self.durable_count = 0
        self.provider_call_count = 0
        self.provider_round_counts = {"1": 0, "2": 0, "3": 0}
        self.provider_calls: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.conversation_started_ns: int | None = None
        self.third_send_requested_ns: int | None = None
        self.third_worker_started_ns: int | None = None
        self.turn_2_release_ns: int | None = None
        self.turn_2_assistant_durable_ns: int | None = None
        self.terminal_third_assistant_ns: int | None = None

    def dispatch_requested(self) -> None:
        now = time.perf_counter_ns()
        with self._lock:
            self.dispatch_count += 1
            if self.dispatch_count == 1:
                self.conversation_started_ns = now
            elif self.dispatch_count == 3:
                self.third_send_requested_ns = now

    def worker_started(self) -> None:
        now = time.perf_counter_ns()
        with self._lock:
            self.worker_count += 1
            if self.worker_count == 3:
                self.third_worker_started_ns = now

    def turn_accepted(self) -> None:
        with self._lock:
            self.accepted_count += 1

    def drain_started(self) -> None:
        with self._lock:
            self.drain_count += 1

    def after_turn_started(self) -> None:
        now = time.perf_counter_ns()
        with self._lock:
            self.after_turn_count += 1
            if self.after_turn_count == 2:
                self.turn_2_release_ns = now

    def assistant_durable(self) -> None:
        now = time.perf_counter_ns()
        terminal = False
        with self._lock:
            self.durable_count += 1
            if self.durable_count == 2:
                self.turn_2_assistant_durable_ns = now
            elif self.durable_count == 3:
                self.terminal_third_assistant_ns = now
                terminal = True
        if terminal:
            self._third_assistant_terminal.set()

    def provider_started(self) -> tuple[int, int, int]:
        now = time.perf_counter_ns()
        with self._lock:
            self.provider_call_count += 1
            call_index = self.provider_call_count
            turn = self.worker_count
            if str(turn) in self.provider_round_counts:
                self.provider_round_counts[str(turn)] += 1
                provider_round = self.provider_round_counts[str(turn)]
            else:
                provider_round = 0
            self.provider_calls.append(
                {
                    "call": call_index,
                    "turn": turn,
                    "round": provider_round,
                    "started_ns": now,
                    "first_chunk_ns": None,
                    "completed_ns": None,
                    "usage": None,
                }
            )
            return call_index, turn, provider_round

    def provider_first_chunk(self, call_index: int) -> None:
        now = time.perf_counter_ns()
        with self._lock:
            call = self.provider_calls[call_index - 1]
            if call["first_chunk_ns"] is None:
                call["first_chunk_ns"] = now

    def provider_completed(self, call_index: int) -> None:
        now = time.perf_counter_ns()
        with self._lock:
            self.provider_calls[call_index - 1]["completed_ns"] = now
        if call_index == 4:
            self._ui_loop.call_soon_threadsafe(self._turn_two_terminal.set)

    def provider_usage(self, call_index: int, usage: Mapping[str, int]) -> None:
        """Attach one content-free terminal usage envelope to its call."""
        with self._lock:
            call = self.provider_calls[call_index - 1]
            if call["usage"] is not None:
                raise RuntimeError("provider_usage_duplicate")
            call["usage"] = dict(usage)

    def schema_loaded(self, tool_id: str) -> None:
        if tool_id != "local:fs_write":
            return
        with self._lock:
            self.tool_calls.append(
                {
                    "name": "load_tools",
                    "turn": self.worker_count,
                    "provider_round": self.provider_round_counts.get(
                        str(self.worker_count), 0
                    ),
                    "requested_tool_id": tool_id,
                    "permission": "allow",
                    "definition_hash": self._permission_hash,
                }
            )

    def local_tool_invoked(self, tool_id: str, args: Mapping[str, Any]) -> None:
        name = tool_id.split(":", 1)[-1]
        if name != "fs_write":
            return
        raw_payload = args.get("content", "")
        payload = (
            raw_payload.encode("utf-8")
            if isinstance(raw_payload, str)
            else bytes(raw_payload)
            if isinstance(raw_payload, (bytes, bytearray))
            else b""
        )
        with self._lock:
            self.tool_calls.append(
                {
                    "name": "fs_write",
                    "turn": self.worker_count,
                    "provider_round": self.provider_round_counts.get(
                        str(self.worker_count), 0
                    ),
                    "tool_id": "local:fs_write",
                    "path": str(args.get("path", "")),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "permission": "allow",
                    "definition_hash": self._permission_hash,
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "dispatch_count": self.dispatch_count,
                "accepted_count": self.accepted_count,
                "drain_count": self.drain_count,
                "worker_count": self.worker_count,
                "after_turn_count": self.after_turn_count,
                "durable_count": self.durable_count,
                "provider_call_count": self.provider_call_count,
                "provider_round_counts": dict(self.provider_round_counts),
                "provider_calls": [dict(call) for call in self.provider_calls],
                "tool_calls": [dict(call) for call in self.tool_calls],
                "conversation_started_ns": self.conversation_started_ns,
                "third_send_requested_ns": self.third_send_requested_ns,
                "third_worker_started_ns": self.third_worker_started_ns,
                "turn_2_release_ns": self.turn_2_release_ns,
                "turn_2_assistant_durable_ns": self.turn_2_assistant_durable_ns,
                "terminal_third_assistant_ns": self.terminal_third_assistant_ns,
            }


def _target_tree_inventory(
    root: Path, excluded_roots: Sequence[Path]
) -> dict[str, tuple[int, int]]:
    """Return source-tree size/mtime inventory outside the owned sample root."""
    inventory: dict[str, tuple[int, int]] = {}
    target = root.resolve()
    excluded = tuple(
        path.resolve()
        for path in excluded_roots
        if path.resolve() != target and path.resolve().is_relative_to(target)
    )
    for directory, names, files in os.walk(target):
        current = Path(directory)
        names[:] = [
            name
            for name in names
            if name != ".git"
            and not any(
                (current / name).resolve().is_relative_to(item)
                for item in excluded
            )
        ]
        if any(current.resolve().is_relative_to(item) for item in excluded):
            names[:] = []
            continue
        for name in files:
            path = current / name
            if any(path.resolve().is_relative_to(item) for item in excluded):
                continue
            stat = path.stat()
            inventory[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
            )
    return inventory


def _install_common_timing_wrappers(
    observation: _MountedObservation,
) -> list[Callable[[], None]]:
    """Install shared target wrappers and fail before provider construction."""
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
        ConsolePromptQueueCoordinator,
    )
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptQueueUIController,
    )

    restorations: list[Callable[[], None]] = []

    def replace(owner: type[Any], name: str, replacement: Callable[..., Any]) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            raise RuntimeError(f"target_common_seam_missing:{name}")
        setattr(owner, name, replacement(original))

        def restore() -> None:
            setattr(owner, name, original)

        restorations.append(restore)

    def dispatch_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapped(owner: Any, *args: Any, **kwargs: Any) -> Any:
            observation.dispatch_requested()
            return await original(owner, *args, **kwargs)

        return wrapped

    def accepted_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(owner: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(owner, *args, **kwargs)
            observation.turn_accepted()
            return result

        return wrapped

    def after_turn_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapped(owner: Any, *args: Any, **kwargs: Any) -> Any:
            observation.after_turn_started()
            return await original(owner, *args, **kwargs)

        return wrapped

    def drain_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapped(owner: Any, *args: Any, **kwargs: Any) -> Any:
            observation.drain_started()
            return await original(owner, *args, **kwargs)

        return wrapped

    def worker_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapped(owner: Any, *args: Any, **kwargs: Any) -> Any:
            observation.worker_started()
            return await original(owner, *args, **kwargs)

        return wrapped

    def durable_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(owner: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(owner, *args, **kwargs)
            observation.assistant_durable()
            return result

        return wrapped

    def schema_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(owner: Any, tool_id: str, *args: Any, **kwargs: Any) -> Any:
            result = original(owner, tool_id, *args, **kwargs)
            observation.schema_loaded(tool_id)
            return result

        return wrapped

    def invoke_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(
            owner: Any,
            tool_id: str,
            args: Mapping[str, Any],
            *extra: Any,
            **kwargs: Any,
        ) -> Any:
            result = original(owner, tool_id, args, *extra, **kwargs)
            observation.local_tool_invoked(tool_id, args)
            return result

        return wrapped

    replace(ConsolePromptQueueUIController, "dispatch", dispatch_wrapper)
    replace(ConsolePromptQueueCoordinator, "turn_accepted", accepted_wrapper)
    replace(ConsolePromptQueueCoordinator, "_after_turn", after_turn_wrapper)
    replace(ConsolePromptQueueCoordinator, "_drain_waiting", drain_wrapper)
    replace(ConsoleChatController, "_run_agent_reply", worker_wrapper)
    replace(
        ConsoleChatController,
        "_record_run_assistant_message",
        durable_wrapper,
    )
    replace(ToolCatalogRegistry, "load_schema", schema_wrapper)
    replace(LocalToolProvider, "invoke", invoke_wrapper)
    return restorations


async def run_mounted_sample(
    sample_root: Path,
    *,
    arm: str,
    endpoint: str,
    model: str,
    adapter: TargetAdapter,
    isolated_env: Mapping[str, str],
    owned_run_root: Path | None = None,
) -> dict[str, Any]:
    """Drive one real gateway sample through the mounted composer and queue."""
    import asyncio
    import threading

    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        ConsoleRunStatus,
    )
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Chat import console_provider_gateway as gateway_module
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Tools import workspace_file_roots
    from tldw_chatbook.Widgets.Console import ConsoleComposerBar

    root = sample_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    inventory_exclusions = (root,) + (
        (owned_run_root.resolve(),) if owned_run_root is not None else ()
    )
    source_before = _target_tree_inventory(
        adapter.target_root, inventory_exclusions
    )
    baseline_threads = {id(thread) for thread in threading.enumerate()}
    ui_loop = asyncio.get_running_loop()
    turn_two_terminal = asyncio.Event()
    turn_two_provider_release = threading.Event()
    third_assistant_terminal = asyncio.Event()
    heartbeat_stop = asyncio.Event()
    heartbeat = HeartbeatBuffer(capacity=100_000)
    runtime: WorkspaceRuntime | None = None
    chat_db: Any | None = None
    gateway: Any | None = None
    console_runtime: Any | None = None
    registry_factory = workspace_file_roots._registry_factory
    common_restorations: list[Callable[[], None]] = []
    benchmark_restorations: list[Callable[[], None]] = []
    adapter.install_timing_wrappers()

    async def heartbeat_loop() -> None:
        interval = 0.01
        target = ui_loop.time() + interval
        while not heartbeat_stop.is_set():
            await asyncio.sleep(max(0.0, target - ui_loop.time()))
            now = ui_loop.time()
            heartbeat.record(max(0, int((now - target) * 1_000_000_000)))
            target += interval

    heartbeat_task = asyncio.create_task(heartbeat_loop())
    observation: _MountedObservation | None = None
    store: Any | None = None
    active_session_id: str | None = None
    try:
        runtime = prepare_workspace_runtime(root, arm=arm)
        corpus = generate_corpus(runtime.workspace_root)
        observation = _MountedObservation(
            ui_loop=ui_loop,
            turn_two_terminal=turn_two_terminal,
            third_assistant_terminal=third_assistant_terminal,
            permission_hash=runtime.permission_definition_hash,
        )
        common_restorations = _install_common_timing_wrappers(observation)
        workspace_file_roots._registry_factory = lambda: runtime.registry

        app = _build_test_app()
        old_consent = getattr(app, "change_review_consent_service", None)
        if old_consent is not None:
            old_consent.shutdown(timeout=1.0)
        old_workspace_db = getattr(app, "local_workspace_db", None)
        if old_workspace_db is not None:
            old_workspace_db.close()
        app.local_workspace_db = runtime.database
        app.workspace_registry_service = runtime.registry
        adapter.configure_review(app, runtime)
        app.unified_mcp_service = runtime.control_plane
        app.app_config["chat_defaults"] = {
            "provider": "llama_cpp",
            "model": model,
        }
        app.app_config["api_settings"] = {
            "llama_cpp": {
                "api_url": endpoint,
                "model": model,
                "temperature": 0.0,
                "max_tokens": 512,
                "reasoning_effort": "none",
                "timeout": 120,
                "retries": 0,
                "streaming": True,
            }
        }
        app.app_config.setdefault("console", {})["workspace_root"] = str(
            runtime.workspace_root
        )
        app.app_config.pop("general", None)
        app.app_config.pop("logging", None)
        app.chat_api_provider_value = "llama_cpp"
        app.chat_api_model_value = model
        chat_db = CharactersRAGDB(
            root / "database" / "chatbook.sqlite",
            client_id="task-19641-real-provider",
        )
        app.chachanotes_db = chat_db
        gateway = ConsoleProviderGateway(
            config_provider=lambda: app.app_config,
            environ=dict(isolated_env),
        )
        original_payload_builder = gateway_module.build_llamacpp_chat_payload

        def usage_enabled_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
            payload = original_payload_builder(*args, **kwargs)
            if payload.get("stream") is True:
                payload["stream_options"] = {"include_usage": True}
            return payload

        gateway_module.build_llamacpp_chat_payload = usage_enabled_payload
        benchmark_restorations.append(
            lambda: setattr(
                gateway_module,
                "build_llamacpp_chat_payload",
                original_payload_builder,
            )
        )
        original_stream_chat = gateway.stream_chat
        original_sse_parser = gateway._content_from_sse_line
        current_provider_call: int | None = None

        def observed_sse_line(line: str) -> str:
            usage = extract_sse_usage(line)
            if usage is not None:
                if current_provider_call is None:
                    raise RuntimeError("provider_usage_without_call")
                assert observation is not None
                observation.provider_usage(current_provider_call, usage)
            return original_sse_parser(line)

        gateway._content_from_sse_line = observed_sse_line

        async def observed_stream_chat(*args: Any, **kwargs: Any):
            nonlocal current_provider_call
            assert observation is not None
            call_index, _turn, _provider_round = observation.provider_started()
            current_provider_call = call_index
            try:
                async for chunk in original_stream_chat(*args, **kwargs):
                    observation.provider_first_chunk(call_index)
                    yield chunk
            finally:
                observation.provider_completed(call_index)
                current_provider_call = None
                if call_index == 4:
                    turn_two_provider_release.wait(timeout=30.0)

        gateway.stream_chat = observed_stream_chat
        app.console_provider_gateway_factory = lambda: gateway
        host = ConsoleHarness(app)

        async def wait_until(predicate: Callable[[], bool], timeout: float = 180.0) -> None:
            deadline = time.monotonic() + timeout
            while not predicate():
                if time.monotonic() >= deadline:
                    raise RuntimeError("mounted_sample_timeout")
                await asyncio.sleep(0.005)

        async def wait_review_idle(screen: Any) -> None:
            coordinator = getattr(screen._console_runtime(), "change_review_coordinator", None)
            if coordinator is not None:
                idle = await asyncio.to_thread(coordinator.wait_idle, 30.0)
                if not idle:
                    raise RuntimeError("change_review_idle_timeout")
                if coordinator.publication_signal.snapshot().pending:
                    raise RuntimeError("change_review_publication_pending")

        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            composer = console.query_one("#console-native-composer", ConsoleComposerBar)
            controller = console._ensure_console_chat_controller()
            store = console._ensure_console_chat_store()
            active_session_id = store.active_session_id
            session = next(
                item for item in store.sessions() if item.id == active_session_id
            )
            if session.workspace_id != runtime.workspace_id:
                raise RuntimeError("mounted_sample_workspace_mismatch")
            store.set_session_project_instruction_state(
                active_session_id,
                ProjectInstructionControlState.legacy_disabled(),
            )
            session_settings = store.session_settings(active_session_id)
            if session_settings is None:
                raise RuntimeError("mounted_sample_settings_missing")
            store.replace_session_settings(
                active_session_id,
                dataclass_replace(
                    session_settings,
                    temperature=0.0,
                    max_tokens=512,
                    reasoning_effort="none",
                    streaming=True,
                ),
            )

            async def type_prompt(text: str) -> None:
                composer.focus()
                await pilot.press(*tuple(text.lower()))

            await type_prompt(TURN_PROMPTS[0])
            await pilot.press("enter")
            await wait_until(lambda: observation.snapshot()["durable_count"] >= 1)
            await wait_review_idle(console)
            adapter.reset_review_events()

            await type_prompt(TURN_PROMPTS[1])
            await pilot.press("enter")
            await wait_until(lambda: observation.snapshot()["worker_count"] >= 2)
            await type_prompt(TURN_PROMPTS[2])
            if composer.draft_text() != TURN_PROMPTS[2].lower():
                raise RuntimeError("third_prompt_draft_loss")
            await asyncio.wait_for(turn_two_terminal.wait(), timeout=180.0)
            try:
                await pilot.press("enter")
            finally:
                turn_two_provider_release.set()
            await asyncio.wait_for(third_assistant_terminal.wait(), timeout=180.0)
            await wait_until(lambda: observation.snapshot()["provider_call_count"] >= 5)
            await wait_review_idle(console)
            await pilot.pause(0.05)

            snapshot = observation.snapshot()
            messages = store.messages_for_session(active_session_id)
            user_count = sum(
                message.role is ConsoleMessageRole.USER for message in messages
            )
            completed_assistants = [
                message
                for message in messages
                if message.role is ConsoleMessageRole.ASSISTANT
                and message.status == "complete"
            ]
            if (
                len(completed_assistants) != 3
                or controller.run_state.status is not ConsoleRunStatus.COMPLETED
            ):
                raise RuntimeError("terminal_assistant_contract_failed")
            queue_snapshot = controller.prompt_queue_registry.snapshot(
                active_session_id
            )
            prompt_loss_count = abs(3 - user_count) + queue_snapshot.total_count
            console_runtime = console._console_runtime()
            await console_runtime.dispose()

        heartbeat_stop.set()
        await heartbeat_task
        mutation_path = runtime.workspace_root / "measured" / "turn-two.txt"
        mutation_success = (
            mutation_path.is_file() and mutation_path.read_bytes() == FIXED_MUTATION
        )
        if not mutation_success:
            raise RuntimeError("benchmark_mutation_contract_failed")
        assert observation is not None
        snapshot = observation.snapshot()
        provider_calls = snapshot["provider_calls"]
        if (
            snapshot["dispatch_count"] != 3
            or snapshot["accepted_count"] != 3
            or snapshot["drain_count"] != 2
            or snapshot["worker_count"] != 3
            or snapshot["after_turn_count"] != 2
            or snapshot["durable_count"] != 3
        ):
            raise RuntimeError("mounted_queue_contract_failed")
        if len(provider_calls) != 5:
            raise RuntimeError("provider_round_contract_failed")
        provider_usage = [call["usage"] for call in provider_calls]
        if any(not isinstance(usage, Mapping) for usage in provider_usage):
            raise RuntimeError("provider_usage_contract_failed")
        required_times = (
            snapshot["conversation_started_ns"],
            snapshot["third_send_requested_ns"],
            snapshot["third_worker_started_ns"],
            snapshot["turn_2_release_ns"],
            snapshot["turn_2_assistant_durable_ns"],
            snapshot["terminal_third_assistant_ns"],
            provider_calls[3]["completed_ns"],
            provider_calls[4]["started_ns"],
            provider_calls[4]["completed_ns"],
        )
        if not all(isinstance(value, int) for value in required_times):
            raise RuntimeError("mounted_sample_timing_missing")
        provider_total_ns = sum(
            int(call["completed_ns"]) - int(call["started_ns"])
            for call in provider_calls
        )
        result = {
            "status": "complete",
            "provider_round_counts": snapshot["provider_round_counts"],
            "provider_usage": provider_usage,
            "terminal_turn_2_provider_completed_ns": provider_calls[3][
                "completed_ns"
            ],
            "third_send_requested_ns": snapshot["third_send_requested_ns"],
            "turn_2_release_ns": snapshot["turn_2_release_ns"],
            "third_worker_started_ns": snapshot["third_worker_started_ns"],
            "third_provider_started_ns": provider_calls[4]["started_ns"],
            "terminal_third_provider_completed_ns": provider_calls[4][
                "completed_ns"
            ],
            "terminal_third_assistant_ns": snapshot[
                "terminal_third_assistant_ns"
            ],
            "heartbeat_lateness_ns": heartbeat.values(),
            "prompt_loss_count": prompt_loss_count,
            "selected_binding_access": runtime.binding.metadata["access"],
            "expected_payload_sha256": hashlib.sha256(FIXED_MUTATION).hexdigest(),
            "expected_permission_definition_hash": (
                runtime.permission_definition_hash
            ),
            "tool_calls": snapshot["tool_calls"],
            "mutation": {
                "path": "measured/turn-two.txt",
                "payload_sha256": hashlib.sha256(FIXED_MUTATION).hexdigest(),
                "success": mutation_success,
            },
            "review_events": adapter.review_events(),
            "workspace_content_tree_digest": corpus["content_tree_digest"],
            "metrics": {
                "third_send_to_worker_ns": (
                    int(snapshot["third_worker_started_ns"])
                    - int(snapshot["third_send_requested_ns"])
                ),
                "event_loop_lag_p95_ns": sample_heartbeat_p95_ns(
                    heartbeat.values()
                ),
                "assistant_durable_to_release_ns": (
                    int(snapshot["turn_2_release_ns"])
                    - int(snapshot["turn_2_assistant_durable_ns"])
                ),
                "terminal_to_third_provider_ns": (
                    int(provider_calls[4]["started_ns"])
                    - int(provider_calls[3]["completed_ns"])
                ),
                "provider_total_ns": provider_total_ns,
                "conversation_wall_ns": (
                    int(snapshot["terminal_third_assistant_ns"])
                    - int(snapshot["conversation_started_ns"])
                ),
            },
        }
    finally:
        turn_two_provider_release.set()
        heartbeat_stop.set()
        if not heartbeat_task.done():
            await heartbeat_task
        if console_runtime is not None:
            try:
                await console_runtime.dispose()
            except Exception:
                pass
        if gateway is not None:
            try:
                await gateway.aclose()
            except Exception:
                pass
        if chat_db is not None:
            close_chat_db = getattr(chat_db, "close_connection", None)
            if callable(close_chat_db):
                close_chat_db()
        if runtime is not None:
            runtime.close()
        workspace_file_roots._registry_factory = registry_factory
        for restore in reversed(common_restorations):
            restore()
        for restore in reversed(benchmark_restorations):
            restore()
        adapter.close()

    # asyncio.run() would close these one frame after this coroutine returns,
    # but terminal evidence is emitted only after ownership is already zero.
    await ui_loop.shutdown_default_executor()
    deadline = time.monotonic() + 2.0
    while True:
        survivors = [
            thread
            for thread in threading.enumerate()
            if id(thread) not in baseline_threads and thread.is_alive()
        ]
        if not survivors or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.01)
    if survivors:
        names = ":".join(
            sorted(
                {
                    re.sub(r"[^a-z0-9_-]+", "-", thread.name.lower())[:40]
                    or "unnamed"
                    for thread in survivors
                }
            )
        )
        raise RuntimeError(f"benchmark_owned_thread_survivor:{names}")
    if (
        _target_tree_inventory(adapter.target_root, inventory_exclusions)
        != source_before
    ):
        raise RuntimeError("target_source_write_detected")
    result["final_ownership"] = {
        "live_threads": 0,
        "provider_closed": True,
        "sqlite_closed": True,
        "shadow_operations_pending": 0,
        "target_source_writes": 0,
    }
    return result


def run_child_mode(args: argparse.Namespace) -> int:
    """Bootstrap one isolated target revision and emit one terminal sample."""
    import asyncio

    spec = read_child_spec(args.child_spec)
    sample_root = Path(spec["sample_root"]).resolve()
    run_root = Path(spec["run_root"]).resolve()
    target_root = Path(spec["target_root"]).resolve()
    evidence_path = Path(spec["evidence_path"]).resolve()
    if (
        not sample_root.is_relative_to(run_root)
        or not evidence_path.is_relative_to(run_root)
        or args.output_root.resolve() != run_root
        or not target_root.is_dir()
    ):
        raise RuntimeError("child_spec_invalid")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("a", encoding="utf-8") as evidence:
        write_boundary_event(
            evidence,
            {
                "event": "child_start",
                "sample_id": spec["sample_id"],
                "phase": spec["phase"],
                "iteration": spec["iteration"],
                "arm": spec["arm"],
            },
        )
        try:
            assert_child_environment(sample_root, os.environ)
            if Path(os.environ["TLDW_CONFIG_PATH"]).resolve() != (
                sample_root / "config" / "tldw_cli" / "config.toml"
            ).resolve():
                raise RuntimeError("child_environment_mismatch:config")
            adapter = TargetAdapter.for_arm(target_root, str(spec["arm"]))
            install_target_root(target_root)
            imported = assert_target_modules(TARGET_MODULES, target_root)
            target_modules = {
                name: Path(path).relative_to(target_root).as_posix()
                for name, path in imported.items()
            }
            sample = asyncio.run(
                run_mounted_sample(
                    sample_root,
                    arm=str(spec["arm"]),
                    endpoint=args.endpoint,
                    model=args.model,
                    adapter=adapter,
                    isolated_env=os.environ,
                    owned_run_root=run_root,
                )
            )
            sample.update(
                {
                    "event": "sample",
                    "sample_id": spec["sample_id"],
                    "phase": spec["phase"],
                    "iteration": spec["iteration"],
                    "arm": spec["arm"],
                    "target_revision_kind": adapter.revision_kind,
                    "target_modules": target_modules,
                }
            )
            validation_errors = validate_sample(sample)
            if validation_errors:
                raise RuntimeError("child_sample_invalid")
            if privacy_violations(sample):
                raise RuntimeError("child_sample_privacy_violation")
            write_boundary_event(evidence, sample)
            return 0
        except Exception as exc:
            write_boundary_event(
                evidence,
                {
                    "event": "child_failure",
                    "sample_id": spec["sample_id"],
                    "phase": spec["phase"],
                    "iteration": spec["iteration"],
                    "arm": spec["arm"],
                    "error_type": type(exc).__name__,
                    "error_code": safe_error_code(exc),
                },
            )
            return 1


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _remove_control_worktree(repository_root: Path, target: Path) -> None:
    completed = subprocess.run(
        ["git", "worktree", "remove", "--force", str(target)],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("control_worktree_cleanup_failed")


def prepare_output_root(path: Path) -> None:
    """Create an output root or preserve its sole documentation README."""
    if path.exists():
        existing = list(path.iterdir())
        if any(item.name != "README.md" or not item.is_file() for item in existing):
            raise RuntimeError("output_root_not_empty")
    path.mkdir(parents=True, exist_ok=True)


def run_parent_mode(args: argparse.Namespace) -> int:
    """Own revisions, child lifecycles, validation, and retained smoke evidence."""
    repository_root = Path(__file__).resolve().parents[2]
    run_root = args.output_root.resolve()
    prepare_output_root(run_root)
    raw_path = run_root / "real-provider-three-turn.raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    revisions = resolve_benchmark_revisions(
        repository_root,
        control_ref=args.control_sha,
        candidate_ref=args.candidate_sha,
    )
    preflight = preflight_provider(args.endpoint, args.model)
    server_metadata = provider_server_metadata(args.endpoint, args.model)
    runtime = runtime_metadata()
    host_before = host_load_snapshot()
    listener_before = listener_resource_snapshot(args.endpoint)
    control_root = prepare_control_worktree(
        repository_root,
        run_root,
        control_sha=revisions["control"],
    )
    rows: list[dict[str, Any]] = []
    runner = Path(__file__).resolve()
    try:
        for index, plan in enumerate(sample_schedule(args.iterations)):
            sample_id = f"{plan.phase}-{plan.iteration}-{plan.arm}"
            sample_root = run_root / "samples" / f"{index:03d}-{sample_id}"
            target_root = control_root if plan.arm == "control" else repository_root
            environment = build_child_environment(os.environ, sample_root)
            write_child_config(sample_root, endpoint=args.endpoint, model=args.model)
            spec_path = sample_root / "child-spec.json"
            write_child_spec(
                spec_path,
                {
                    "sample_id": sample_id,
                    "phase": plan.phase,
                    "iteration": plan.iteration,
                    "arm": plan.arm,
                    "target_root": str(target_root.resolve()),
                    "sample_root": str(sample_root.resolve()),
                    "run_root": str(run_root),
                    "evidence_path": str(raw_path),
                },
            )
            command = [
                sys.executable,
                str(runner),
                "--endpoint",
                args.endpoint,
                "--model",
                args.model,
                "--iterations",
                str(args.iterations),
                "--output-root",
                str(run_root),
                "--control-sha",
                revisions["control"],
                "--candidate-sha",
                revisions["candidate"],
                "--sample-timeout",
                str(args.sample_timeout),
                "--child-spec",
                str(spec_path),
            ]
            child = run_child_with_watchdog(
                command,
                evidence_path=raw_path,
                timeout_seconds=args.sample_timeout,
                environment=environment,
                cwd=target_root,
            )
            last = child.last_event
            if (
                child.status != "complete"
                or child.returncode != 0
                or not isinstance(last, dict)
                or last.get("event") != "sample"
                or last.get("sample_id") != sample_id
            ):
                write_boundary_event(
                    sys.stdout,
                    {
                        "event": "sample_failed",
                        "sample_id": sample_id,
                        "child_status": child.status,
                        "returncode": child.returncode,
                    },
                )
                return 1
            errors = validate_sample(last)
            if errors or privacy_violations(last):
                raise RuntimeError("parent_sample_validation_failed")
            rows.append(last)
            write_boundary_event(
                sys.stdout,
                {
                    "event": "sample_complete",
                    "sample_id": sample_id,
                    "completed": len(rows),
                    "scheduled": len(sample_schedule(args.iterations)),
                },
            )
    finally:
        _remove_control_worktree(repository_root, control_root)

    validation_errors = validate_run(rows, expected_iterations=args.iterations)
    if validation_errors:
        raise RuntimeError("parent_run_validation_failed")
    corpus_digests = {
        str(row.get("workspace_content_tree_digest", "")) for row in rows
    }
    if len(corpus_digests) != 1 or not _SHA256.fullmatch(next(iter(corpus_digests))):
        raise RuntimeError("workspace_corpus_mismatch")
    permission_hashes_by_arm: dict[str, str] = {}
    for arm in ARMS:
        arm_hashes = {
            str(row.get("expected_permission_definition_hash", ""))
            for row in rows
            if row.get("arm") == arm
        }
        if len(arm_hashes) != 1 or not _SHA256.fullmatch(next(iter(arm_hashes))):
            raise RuntimeError("tool_schema_fixture_mismatch")
        permission_hashes_by_arm[arm] = next(iter(arm_hashes))
    host_after = host_load_snapshot()
    listener_after = listener_resource_snapshot(args.endpoint)
    manifest = {
        "schema": 1,
        "revisions": revisions,
        "model": args.model,
        "provider": "llama_cpp",
        "temperature": 0.0,
        "max_tokens": 512,
        "reasoning_effort": "none",
        "stream_options": {"include_usage": True},
        "fixture_ids": {
            "turn_prompts": "task-19641-three-turn-prompts-v1",
            "tool_schema": "local:fs_write-target-definition-v1",
            "mutation": "task-19641-confined-fs-write-v1",
            "workspace_corpus": "task-19641-workspace-corpus-v1",
        },
        "fixture_hashes": {
            "turn_prompts_sha256": hashlib.sha256(
                json.dumps(TURN_PROMPTS, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "mutation_sha256": hashlib.sha256(FIXED_MUTATION).hexdigest(),
            "workspace_content_tree_digest": next(iter(corpus_digests)),
            "tool_definition_sha256_by_arm": permission_hashes_by_arm,
        },
        "sample_schedule": [
            {
                "phase": item.phase,
                "arm": item.arm,
                "iteration": item.iteration,
            }
            for item in sample_schedule(args.iterations)
        ],
        "preflight": preflight,
        "runtime": runtime,
        "provider_server": server_metadata,
        "host_load": {
            "before": host_before,
            "after": host_after,
        },
        "listener_resources": {
            "before": listener_before,
            "after": listener_after,
        },
    }
    summary = (
        build_summary(rows)
        if args.iterations >= 2
        else {
            "overall_verdict": "smoke",
            "validation_errors": [],
            "sample_count": len(rows),
            "arms": {},
            "critical_path_improvement_claims": {},
        }
    )
    if privacy_violations(manifest) or privacy_violations(summary):
        raise RuntimeError("retained_evidence_privacy_violation")
    _write_json(run_root / "real-provider-three-turn.manifest.json", manifest)
    _write_json(run_root / "real-provider-three-turn.summary.json", summary)
    write_boundary_event(
        sys.stdout,
        {
            "event": "run_complete",
            "sample_count": len(rows),
            "verdict": summary["overall_verdict"],
        },
    )
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the import-free preflight or dispatch parent/child benchmark mode."""
    args = parse_arguments(arguments)
    if args.preflight_only:
        preflight = preflight_provider(args.endpoint, args.model)
        write_boundary_event(sys.stdout, {"event": "preflight", **preflight})
        return 0
    if args.child_spec is not None:
        return run_child_mode(args)
    return run_parent_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
