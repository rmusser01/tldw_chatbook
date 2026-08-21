"""Revision-pinned three-turn Console latency benchmark for TASK-19641."""

from __future__ import annotations

import argparse
import math
import json
import importlib
import random
import re
import os
import signal
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Mapping


ARMS = ("control", "disabled", "enabled")
CONTROL_SHA = "5f720a40417eaa78f33619d5cbc82effc470104b"
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
                "timeout = 120",
                "retries = 0",
                "streaming = true",
                "",
            )
        ),
        encoding="utf-8",
    )
    return config_path


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
