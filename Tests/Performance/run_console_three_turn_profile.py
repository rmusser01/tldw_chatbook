"""Revision-pinned three-turn Console latency benchmark for TASK-19641."""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import inspect
import json
import math
import random
import re
import os
import secrets
import signal
import shutil
import sqlite3
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from threading import Lock
from typing import IO, Any, Mapping
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request


ARMS = ("control", "disabled", "enabled")
CONTROL_SHA = "5f720a40417eaa78f33619d5cbc82effc470104b"
ORIGINAL_HARNESS_SHA = "eb8225a32f88ea43c337aff99804d360384e7668"
ORIGINAL_RUNNER_SHA256 = (
    "fbca69703b771f7b7b27fa78ef9bf095fb30712435743877e20fcb01bb6d06ae"
)
CANDIDATE_SHA = ORIGINAL_HARNESS_SHA
REQUEST_SETTINGS = {
    "temperature": 0.0,
    "max_tokens": 512,
    "reasoning_effort": "none",
    "streaming": True,
    "include_usage": True,
}
P95_FRACTION = 0.95
MEASURED_BLOCKS = 30
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 19_641
NON_REGRESSION_CEILING = 1.10
IMPROVEMENT_CEILING = 1.00
ORIGINAL_EVIDENCE_SHA256 = {
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
_ORIGINAL_EVIDENCE_RELATIVE = Path(
    "Docs/superpowers/qa/console-three-turn-real-provider"
)
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
ATTEMPT_STATES = frozenset(
    {
        "running",
        "failed",
        "invalid",
        "complete_pending_review",
        "changes_required",
    }
)
BLOCKING_ATTEMPT_STATES = frozenset(
    {"running", "complete_pending_review", "changes_required"}
)
_ATTEMPT_ID = re.compile(r"^attempt-(\d{4})$")
_MEASURED_VERDICTS = frozenset({"pass", "regression", "inconclusive"})
_ATTEMPT_REASON_CATEGORIES = {
    "failed": frozenset({"provider", "acquisition", "interrupted"}),
    "invalid": frozenset(
        {"raw", "product", "completeness", "isolation", "privacy", "ownership"}
    ),
    "changes_required": frozenset(
        {"manifest", "summary", "report", "readme", "digest", "receipt", "presentation"}
    ),
}
_ATTEMPT_FIELDS = {
    "running": frozenset({"attempt_id", "state"}),
    "failed": frozenset({"attempt_id", "state", "reason_category"}),
    "invalid": frozenset({"attempt_id", "state", "reason_category"}),
    "complete_pending_review": frozenset(
        {"attempt_id", "state", "verdict", "raw_sha256"}
    ),
    "changes_required": frozenset(
        {"attempt_id", "state", "verdict", "raw_sha256", "reason_category"}
    ),
}


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
_CHILD_SPEC_KEYS_WITH_MODE = _CHILD_SPEC_KEYS | {"mode"}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_original_evidence(candidate_root: Path) -> dict[str, str]:
    """Require the complete original TASK-20009 artifact set byte-for-byte."""
    evidence_root = candidate_root.resolve() / _ORIGINAL_EVIDENCE_RELATIVE
    for name in ORIGINAL_EVIDENCE_SHA256:
        path = evidence_root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"original_evidence_missing:{name}")
    observed_names = {path.name for path in evidence_root.iterdir()}
    if observed_names != set(ORIGINAL_EVIDENCE_SHA256):
        raise RuntimeError("original_evidence_set_mismatch")
    for name, expected in ORIGINAL_EVIDENCE_SHA256.items():
        if _sha256_file(evidence_root / name) != expected:
            raise RuntimeError(f"original_evidence_hash_mismatch:{name}")
    return dict(ORIGINAL_EVIDENCE_SHA256)


def load_original_runner(candidate_root: Path) -> Any:
    """Digest-check and isolate the original benchmark statistics module."""
    path = (
        candidate_root.resolve()
        / "Tests/Performance/run_console_three_turn_profile.py"
    )
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("original_runner_missing")
    if _sha256_file(path) != ORIGINAL_RUNNER_SHA256:
        raise RuntimeError("original_runner_hash_mismatch")
    name = "task_20009_original_runner"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("original_runner_load_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        for function_name in ("validate_sample", "validate_run", "build_summary"):
            if not callable(getattr(module, function_name, None)):
                raise RuntimeError("original_runner_contract_mismatch")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_PROTOCOL_MISMATCH_CODES = {
    "revisions": "protocol_revisions_mismatch",
    "provider_kind": "protocol_provider_kind_mismatch",
    "provider_server": "protocol_provider_server_mismatch",
    "runtime": "protocol_runtime_mismatch",
    "model_alias": "protocol_model_alias_mismatch",
    "request_settings": "protocol_request_settings_mismatch",
    "fixture_ids": "protocol_fixture_ids_mismatch",
    "fixture_hashes": "protocol_fixture_hashes_mismatch",
    "metric_names": "protocol_metric_names_mismatch",
    "primary_gate_names": "protocol_primary_gate_names_mismatch",
    "p95": "protocol_p95_mismatch",
    "measured_blocks": "protocol_measured_blocks_mismatch",
    "resampling": "protocol_resampling_mismatch",
    "confidence_bounds": "protocol_confidence_bounds_mismatch",
    "non_regression_ceiling": "protocol_non_regression_ceiling_mismatch",
    "improvement_ceiling": "protocol_improvement_ceiling_mismatch",
}


def _statistics_protocol(module: Any, *, error_code: str) -> dict[str, Any]:
    """Fingerprint percentile, paired-bootstrap, and summary behavior."""
    nearest = module.nearest_rank_percentile
    paired = module.paired_p95_ratio_bounds
    summary_builder = module.build_summary
    validate = module.validate_run
    fractions: list[float] = []

    def recording_nearest(values: Sequence[float], fraction: float) -> float:
        fractions.append(fraction)
        return nearest(values, fraction)

    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    try:
        summary_parameters = inspect.signature(summary_builder).parameters
        paired_parameters = inspect.signature(paired).parameters
        resamples = summary_parameters["bootstrap_resamples"].default
        seed = summary_parameters["bootstrap_seed"].default
        if (
            not isinstance(resamples, int)
            or isinstance(resamples, bool)
            or resamples < 1
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(paired_parameters["resamples"].default, int)
            or isinstance(paired_parameters["resamples"].default, bool)
            or paired_parameters["resamples"].default < 1
            or not isinstance(paired_parameters["seed"].default, int)
            or isinstance(paired_parameters["seed"].default, bool)
        ):
            raise RuntimeError(error_code)
        nearest_probe = (
            nearest([4.0, 1.0, 3.0, 2.0], 0.5),
            nearest([4.0, 1.0, 3.0, 2.0], 0.75),
        )
        if nearest_probe != (2.0, 3.0):
            raise RuntimeError(error_code)
        try:
            paired(
                [
                    {"control": 1.0, "disabled": 2.0},
                    {"control": 2.0, "disabled": 1.0},
                ],
                "disabled",
                resamples=2,
                seed=1,
            )
        except ValueError:
            pass
        else:
            raise RuntimeError(error_code)
        blocks = [
            {"control": 74.0, "disabled": 11.0, "enabled": 138.0},
            {"control": 25.0, "disabled": 127.0, "enabled": 68.0},
            {"control": 20.0, "disabled": 2.0, "enabled": 31.0},
        ]
        default_bounds = {
            candidate: paired(
                blocks, candidate, resamples=resamples, seed=seed
            )
            for candidate in module.ARMS[1:]
        }
        if default_bounds["disabled"] != paired(
            blocks, "disabled", resamples=resamples, seed=seed
        ):
            raise RuntimeError(error_code)
        alternate_disabled_bounds = paired(
            blocks, "disabled", resamples=resamples, seed=seed + 1
        )
        distribution = (
            1,
            1,
            1,
            2,
            2,
            3,
            5,
            8,
            13,
            21,
            34,
            55,
            89,
            144,
            233,
            377,
            610,
            987,
            1_597,
            2_584,
            4_181,
            6_765,
            10_946,
            17_711,
            28_657,
            46_368,
            75_025,
            121_393,
            196_418,
            999_999,
        )
        rows = [
            {
                "phase": "measured",
                "iteration": iteration,
                "arm": arm,
                "metrics": {
                    metric: float(
                        distribution[iteration] * (arm_index + 1) + metric_index
                    )
                    for metric_index, metric in enumerate(module.REQUIRED_METRICS)
                },
            }
            for iteration in range(30)
            for arm_index, arm in enumerate(module.ARMS)
        ]
        summary_calls: list[dict[str, Any]] = []

        def recording_paired(
            blocks: Sequence[Mapping[str, float]],
            candidate: str,
            *,
            resamples: int,
            seed: int,
        ) -> Any:
            call = {
                "candidate": candidate,
                "blocks": [
                    {arm: float(block[arm]) for arm in module.ARMS}
                    for block in blocks
                ],
                "resamples": resamples,
                "seed": seed,
            }
            summary_calls.append(call)
            sentinel = 0.75 + int(digest(call)[:8], 16) / 0xFFFFFFFF / 2
            return {
                "two_sided_95": (sentinel, sentinel + 0.02),
                "one_sided_lower_95": sentinel + 0.01,
                "one_sided_upper_95": sentinel + 0.02,
            }

        module.validate_run = lambda *_args, **_kwargs: ()
        module.nearest_rank_percentile = recording_nearest
        module.paired_p95_ratio_bounds = recording_paired
        summary = summary_builder(
            rows,
            bootstrap_resamples=resamples,
            bootstrap_seed=seed,
        )
        if not summary_calls or any(
            call["resamples"] != resamples or call["seed"] != seed
            for call in summary_calls
        ):
            raise RuntimeError(error_code)
        matching_fractions = {
            fraction
            for fraction in fractions
            if all(
                nearest(
                    [
                        row["metrics"][metric]
                        for row in rows
                        if row["arm"] == arm
                    ],
                    fraction,
                )
                == summary["arms"][arm]["metrics"][metric]["p95"]
                for arm in module.ARMS
                for metric in module.REQUIRED_METRICS
            )
        }
        if len(matching_fractions) != 1:
            raise RuntimeError(error_code)
        fraction = matching_fractions.pop()
        p95_payload = {
            "probe": nearest_probe,
            "summary_fraction": fraction,
            "summary": {
                arm: {
                    metric: summary["arms"][arm]["metrics"][metric]["p95"]
                    for metric in module.REQUIRED_METRICS
                }
                for arm in module.ARMS
            },
        }
        resampling_payload = {
            "resamples": resamples,
            "seed": seed,
            "default_seed_by_candidate": default_bounds,
            "alternate_seed_disabled": alternate_disabled_bounds,
            "summary_call_trace": summary_calls,
            "summary": summary,
        }
        return {
            "p95_method": "nearest_rank",
            "p95_fraction": fraction,
            "p95_behavior_sha256": digest(p95_payload),
            "resampling_method": "paired_complete_blocks",
            "resamples": resamples,
            "seed": seed,
            "resampling_behavior_sha256": digest(resampling_payload),
        }
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc) == error_code:
            raise
        raise RuntimeError(error_code) from exc
    finally:
        module.validate_run = validate
        module.nearest_rank_percentile = nearest
        module.paired_p95_ratio_bounds = paired


def confirmation_protocol(
    *,
    revisions: Mapping[str, Any],
    provider_kind: str,
    provider_server: Mapping[str, Any],
    runtime: Mapping[str, Any],
    model_alias: str,
    workspace_content_tree_digest: str,
    tool_definition_sha256_by_arm: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete machine-only confirmatory protocol contract."""
    if (
        not isinstance(revisions, Mapping)
        or not isinstance(provider_kind, str)
        or not isinstance(provider_server, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(model_alias, str)
        or not isinstance(workspace_content_tree_digest, str)
        or not isinstance(tool_definition_sha256_by_arm, Mapping)
    ):
        raise RuntimeError("confirmation_protocol_invalid")
    if (
        set(revisions) != {"control", "candidate"}
        or any(
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{40}", value)
            for value in revisions.values()
        )
        or not provider_kind
        or not model_alias
        or provider_server.get("model_alias") != model_alias
        or not _SHA256.fullmatch(workspace_content_tree_digest)
        or set(tool_definition_sha256_by_arm) != set(ARMS)
        or any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in tool_definition_sha256_by_arm.values()
        )
    ):
        raise RuntimeError("confirmation_protocol_invalid")

    def clone(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("confirmation_protocol_invalid") from exc

    statistics_protocol = _statistics_protocol(
        sys.modules[__name__], error_code="confirmation_protocol_statistics_invalid"
    )
    bounds = paired_p95_ratio_bounds(
        [
            {"control": float(index), "disabled": float(index), "enabled": float(index)}
            for index in range(1, 3)
        ],
        "disabled",
        resamples=1,
        seed=0,
    )

    return {
        "revisions": clone(revisions),
        "provider_kind": provider_kind,
        "provider_server": clone(provider_server),
        "runtime": clone(runtime),
        "model_alias": model_alias,
        "request_settings": clone(REQUEST_SETTINGS),
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
            "workspace_content_tree_digest": workspace_content_tree_digest,
            "tool_definition_sha256_by_arm": clone(
                tool_definition_sha256_by_arm
            ),
        },
        "metric_names": sorted(REQUIRED_METRICS),
        "primary_gate_names": sorted(NON_REGRESSION_METRICS),
        "p95": {
            "method": statistics_protocol["p95_method"],
            "fraction": statistics_protocol["p95_fraction"],
            "behavior_sha256": statistics_protocol["p95_behavior_sha256"],
        },
        "measured_blocks": MEASURED_BLOCKS,
        "resampling": {
            "method": statistics_protocol["resampling_method"],
            "resamples": statistics_protocol["resamples"],
            "seed": statistics_protocol["seed"],
            "behavior_sha256": statistics_protocol[
                "resampling_behavior_sha256"
            ],
        },
        "confidence_bounds": list(bounds),
        "non_regression_ceiling": NON_REGRESSION_CEILING,
        "improvement_ceiling": IMPROVEMENT_CEILING,
    }


def protocol_mismatches(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> tuple[str, ...]:
    """Compare the complete protocol without defaults or partial matching."""
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        return ("protocol_schema_mismatch",)
    try:
        json.dumps(expected)
        json.dumps(observed)
    except (TypeError, ValueError):
        return ("protocol_schema_mismatch",)
    if set(expected) != set(_PROTOCOL_MISMATCH_CODES) or set(observed) != set(
        _PROTOCOL_MISMATCH_CODES
    ):
        return ("protocol_schema_mismatch",)
    return tuple(
        code
        for key, code in _PROTOCOL_MISMATCH_CODES.items()
        if observed[key] != expected[key]
    )


def _original_thresholds(original: Any, rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Discover decision thresholds by discriminating the pinned summary builder."""
    real_bounds = original.paired_p95_ratio_bounds

    def summary_at(ratio: float) -> Mapping[str, Any]:
        original.paired_p95_ratio_bounds = lambda *_args, **_kwargs: {
            "two_sided_95": (ratio, ratio),
            "one_sided_lower_95": ratio,
            "one_sided_upper_95": ratio,
        }
        return original.build_summary(rows, bootstrap_resamples=1)

    def transition(
        predicate: Callable[[Mapping[str, Any]], bool],
        *,
        inclusive: bool,
    ) -> float:
        lower, upper = 0.0, 2.0
        while math.nextafter(lower, math.inf) < upper:
            middle = (lower + upper) / 2
            if predicate(summary_at(middle)):
                lower = middle
            else:
                upper = middle
        boundary = lower if inclusive else upper
        below = math.nextafter(boundary, -math.inf)
        above = math.nextafter(boundary, math.inf)
        if (
            not predicate(summary_at(below))
            or predicate(summary_at(boundary)) is not inclusive
            or predicate(summary_at(above))
        ):
            raise RuntimeError("original_protocol_invalid")
        return boundary

    try:
        non_regression = transition(
            lambda summary: summary["arms"]["disabled"]["verdict"] == "pass",
            inclusive=True,
        )
        improvement = transition(
            lambda summary: bool(summary["critical_path_improvement_claims"]),
            inclusive=False,
        )
    finally:
        original.paired_p95_ratio_bounds = real_bounds
    return non_regression, improvement


def load_original_protocol(
    original_runner_root: Path,
    evidence_repository_root: Path,
) -> dict[str, Any]:
    """Load independent runner and retained evidence roots as protocol input."""
    verify_original_evidence(evidence_repository_root)
    manifest_path = (
        evidence_repository_root.resolve()
        / _ORIGINAL_EVIDENCE_RELATIVE
        / "real-provider-three-turn.manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_bytes())
        evidence_root = manifest_path.parent
        machine_summary = json.loads(
            (evidence_root / "real-provider-three-turn.summary.json").read_bytes()
        )
        rows = [
            row
            for row in (
                json.loads(line)
                for line in (
                    evidence_root / "real-provider-three-turn.raw.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            )
            if row.get("event") == "sample"
        ]
        original = load_original_runner(original_runner_root)
        real_percentile = original.nearest_rank_percentile
        observed_fractions: list[float] = []

        def record_percentile(values: Sequence[float], fraction: float) -> float:
            observed_fractions.append(fraction)
            return real_percentile(values, fraction)

        original.nearest_rank_percentile = record_percentile
        try:
            built = original.build_summary(rows, bootstrap_resamples=1)
            bounds = original.paired_p95_ratio_bounds(
                [
                    {
                        "control": float(index),
                        "disabled": float(index),
                        "enabled": float(index),
                    }
                    for index in range(1, 3)
                ],
                "disabled",
                resamples=1,
                seed=0,
            )
        finally:
            original.nearest_rank_percentile = real_percentile
        non_regression, improvement = _original_thresholds(original, rows)
        statistics_protocol = _statistics_protocol(
            original, error_code="original_protocol_invalid"
        )
        hashes = manifest["fixture_hashes"]
        server = manifest["provider_server"]
        metric_names = list(built["arms"]["control"]["metrics"])
        gate_names = list(machine_summary["arms"]["disabled"]["gates"])
        matching_fractions = {
            fraction
            for fraction in observed_fractions
            if all(
                real_percentile(
                    [
                        row["metrics"][metric]
                        for row in rows
                        if row["phase"] == "measured" and row["arm"] == arm
                    ],
                    fraction,
                )
                == machine_summary["arms"][arm]["metrics"][metric]["p95"]
                for arm in ARMS
                for metric in metric_names
            )
        }
        if (
            set(metric_names) != set(machine_summary["arms"]["control"]["metrics"])
            or set(gate_names) != set(built["arms"]["disabled"]["gates"])
            or any(
                built["arms"][arm]["metrics"]
                != machine_summary["arms"][arm]["metrics"]
                for arm in ARMS
            )
            or len(matching_fractions) != 1
            or statistics_protocol["p95_fraction"] not in matching_fractions
        ):
            raise RuntimeError("original_protocol_invalid")
        return {
            "revisions": manifest["revisions"],
            "provider_kind": manifest["provider"],
            "provider_server": server,
            "runtime": manifest["runtime"],
            "model_alias": server["model_alias"],
            "request_settings": {
                "temperature": manifest["temperature"],
                "max_tokens": manifest["max_tokens"],
                "reasoning_effort": manifest["reasoning_effort"],
                "streaming": True,
                "include_usage": manifest["stream_options"]["include_usage"],
            },
            "fixture_ids": manifest["fixture_ids"],
            "fixture_hashes": hashes,
            "metric_names": metric_names,
            "primary_gate_names": gate_names,
            "p95": {
                "method": statistics_protocol["p95_method"],
                "fraction": statistics_protocol["p95_fraction"],
                "behavior_sha256": statistics_protocol[
                    "p95_behavior_sha256"
                ],
            },
            "measured_blocks": len(
                {
                    row["iteration"]
                    for row in manifest["sample_schedule"]
                    if row["phase"] == "measured"
                }
            ),
            "resampling": {
                "method": statistics_protocol["resampling_method"],
                "resamples": statistics_protocol["resamples"],
                "seed": statistics_protocol["seed"],
                "behavior_sha256": statistics_protocol[
                    "resampling_behavior_sha256"
                ],
            },
            "confidence_bounds": list(bounds),
            "non_regression_ceiling": non_regression,
            "improvement_ceiling": improvement,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("original_protocol_invalid") from exc


@dataclass(frozen=True)
class SamplePlan:
    """One warmup, burn-in, or measured arm invocation owned by the parent."""

    phase: str
    arm: str
    iteration: int


@dataclass(frozen=True)
class ChildResult:
    """Bounded child-process outcome with its last durable evidence event."""

    status: str
    returncode: int | None
    last_event: dict[str, Any] | None


@dataclass(frozen=True)
class CampaignLockOwner:
    """Private identity required to release one exact campaign lock."""

    pid: int
    process_start_sha256: str
    owner_token: str


@dataclass(frozen=True)
class CampaignAttempt:
    """One running attempt and the exact campaign lock that owns it."""

    attempt_id: str
    root: Path
    owner: CampaignLockOwner


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
        failures: list[BaseException] = []
        try:
            if self.consent_service is not None:
                self.consent_service.shutdown(timeout=2.0)
        except BaseException as exc:
            failures.append(exc)
        try:
            self.database.close()
        except BaseException as exc:
            failures.append(exc)
        _raise_failures("workspace_runtime_cleanup_failed", failures)


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
        [float(block["control"]) for block in blocks], P95_FRACTION
    )
    if control_p95 <= 0:
        raise ValueError("paired bootstrap requires a positive control p95")

    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        sampled = [blocks[generator.randrange(len(blocks))] for _ in blocks]
        sampled_control = nearest_rank_percentile(
            [float(block["control"]) for block in sampled], P95_FRACTION
        )
        if sampled_control <= 0:
            raise ValueError("paired bootstrap requires a positive control p95")
        sampled_candidate = nearest_rank_percentile(
            [float(block[candidate]) for block in sampled], P95_FRACTION
        )
        ratios.append(sampled_candidate / sampled_control)

    return {
        "two_sided_95": (
            nearest_rank_percentile(ratios, 0.025),
            nearest_rank_percentile(ratios, 0.975),
        ),
        "one_sided_lower_95": nearest_rank_percentile(ratios, 0.05),
        "one_sided_upper_95": nearest_rank_percentile(ratios, P95_FRACTION),
    }


def sample_heartbeat_p95_ns(tick_lateness_ns: Sequence[int]) -> float:
    """Reduce one sample's raw heartbeat ticks to one equally weighted p95."""
    if not tick_lateness_ns:
        raise ValueError("heartbeat vector must not be empty")
    return nearest_rank_percentile(tick_lateness_ns, P95_FRACTION)


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


def validate_confirmation_rows(
    rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[SamplePlan],
    *,
    validate_sample: Callable[[Mapping[str, Any]], tuple[str, ...]],
) -> tuple[tuple[str, ...], list[Mapping[str, Any]]]:
    """Validate the full confirmatory sequence before excluding burn-in rows."""
    expected = [
        (
            f"{plan.phase}-{plan.iteration}-{plan.arm}",
            plan.phase,
            plan.iteration,
            plan.arm,
            position,
        )
        for position, plan in enumerate(schedule)
    ]
    observed = [
        (
            row.get("sample_id"),
            row.get("phase"),
            row.get("iteration"),
            row.get("arm"),
            row.get("schedule_position"),
        )
        for row in rows
    ]
    errors: list[str] = []
    if observed != expected:
        errors.append("confirmation_schedule_contract")
    sample_ids = [row.get("sample_id") for row in rows]
    if any(
        sample_id in sample_ids[:position]
        for position, sample_id in enumerate(sample_ids)
    ):
        errors.append("confirmation_sample_id_duplicate")
    sample_errors = [validate_sample(row) for row in rows]
    if any(sample_errors):
        errors.append("confirmation_sample_contract")
    filtered = [row for row in rows if row.get("phase") != "burn_in"]
    return tuple(errors), filtered


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
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
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
                "p95": nearest_rank_percentile(values, P95_FRACTION),
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
                "verdict": _bound_verdict(bounds, ceiling=NON_REGRESSION_CEILING),
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
            if float(bounds["one_sided_upper_95"]) < IMPROVEMENT_CEILING:
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
        "pid",
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
            if (
                key_text in _FORBIDDEN_KEYS
                or key_text.endswith("_api_key")
                or key_text.endswith("_pid")
            ):
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


def _require_campaign_enum(
    value: Any, allowed: frozenset[str], error_code: str
) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise RuntimeError(error_code)
    return value


def _validate_attempt_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return one exact, content-free campaign event or fail closed."""
    record = dict(event)
    if privacy_violations(record):
        raise RuntimeError("campaign_event_privacy_violation")
    state = _require_campaign_enum(
        record.get("state"), ATTEMPT_STATES, "campaign_attempt_state_invalid"
    )
    if set(record) != _ATTEMPT_FIELDS[state]:
        raise RuntimeError("campaign_event_fields_invalid")
    attempt_id = record.get("attempt_id")
    if not isinstance(attempt_id, str) or not _ATTEMPT_ID.fullmatch(attempt_id):
        raise RuntimeError("campaign_attempt_id_invalid")
    if state in _ATTEMPT_REASON_CATEGORIES:
        _require_campaign_enum(
            record.get("reason_category"),
            _ATTEMPT_REASON_CATEGORIES[state],
            "campaign_reason_category_invalid",
        )
    if state in {"complete_pending_review", "changes_required"}:
        _require_campaign_enum(
            record.get("verdict"),
            _MEASURED_VERDICTS,
            "campaign_verdict_invalid",
        )
        raw_sha256 = record.get("raw_sha256")
        if not isinstance(raw_sha256, str) or not _SHA256.fullmatch(raw_sha256):
            raise RuntimeError("campaign_raw_hash_invalid")
    return record


def _validate_attempt_lineage(events: Sequence[Mapping[str, Any]]) -> None:
    current_number = 0
    current_id: str | None = None
    current_state: str | None = None
    raw_sha256: str | None = None
    measured_verdict: str | None = None
    transitions = {
        "running": {"failed", "invalid", "complete_pending_review"},
        "complete_pending_review": {"changes_required"},
        "changes_required": {"complete_pending_review"},
    }
    for candidate in events:
        event = _validate_attempt_event(candidate)
        attempt_id = event["attempt_id"]
        number = int(attempt_id.removeprefix("attempt-"))
        if attempt_id != current_id:
            if (
                event["state"] != "running"
                or number != current_number + 1
                or (current_state is not None and current_state not in {"failed", "invalid"})
            ):
                raise RuntimeError("campaign_attempt_sequence_invalid")
            current_number = number
            current_id = attempt_id
            current_state = "running"
            raw_sha256 = None
            measured_verdict = None
            continue
        if event["state"] not in transitions.get(str(current_state), set()):
            raise RuntimeError("campaign_attempt_transition_invalid")
        event_raw_sha256 = event.get("raw_sha256")
        if event_raw_sha256 is not None:
            if raw_sha256 is not None and event_raw_sha256 != raw_sha256:
                raise RuntimeError("campaign_raw_hash_mismatch")
            raw_sha256 = event_raw_sha256
        event_verdict = event.get("verdict")
        if event_verdict is not None:
            if measured_verdict is not None and event_verdict != measured_verdict:
                raise RuntimeError("campaign_verdict_mismatch")
            measured_verdict = event_verdict
        current_state = event["state"]


def attempt_lineage(ledger: Path) -> tuple[dict[str, Any], ...]:
    """Read and validate the complete ordered append-only campaign lineage."""
    if not ledger.exists():
        return ()
    if not ledger.is_file() or ledger.is_symlink():
        raise RuntimeError("campaign_ledger_invalid")
    payload = ledger.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise RuntimeError("campaign_ledger_malformed")
    records: list[dict[str, Any]] = []
    try:
        encoded_lines = payload.decode("utf-8").splitlines()
        for line in encoded_lines:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise RuntimeError("campaign_ledger_malformed")
            records.append(parsed)
        _validate_attempt_lineage(records)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("campaign_ledger_malformed") from exc
    for line, record in zip(encoded_lines, records, strict=True):
        if line != json.dumps(record, sort_keys=True):
            raise RuntimeError("campaign_ledger_malformed")
    return tuple(records)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_namespace(
    path: Path, *, parents: bool = False, exist_ok: bool = False
) -> None:
    existed = path.exists() or path.is_symlink()
    if parents or exist_ok:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    else:
        path.mkdir()
    if not existed:
        _fsync_directory(path.parent)


def _rename_namespace(source: Path, target: Path) -> None:
    source.rename(target)
    _fsync_directory(source.parent)
    if target.parent != source.parent:
        _fsync_directory(target.parent)


def _unlink_namespace(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


def _rmdir_namespace(path: Path) -> None:
    path.rmdir()
    try:
        _fsync_directory(path.parent)
    except BaseException:
        try:
            path.mkdir()
            _fsync_directory(path.parent)
        except OSError:
            pass
        raise


def append_attempt_state(ledger: Path, event: Mapping[str, Any]) -> None:
    """Validate and durably append one canonical JSONL campaign event."""
    record = _validate_attempt_event(event)
    existing = attempt_lineage(ledger)
    _validate_attempt_lineage((*existing, record))
    _mkdir_namespace(ledger.parent, parents=True, exist_ok=True)
    existed = ledger.exists()
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not existed:
        _fsync_directory(ledger.parent)


def next_attempt_id(events: Sequence[Mapping[str, Any]]) -> str:
    """Return the next four-digit campaign attempt identifier."""
    _validate_attempt_lineage(events)
    number = 1 if not events else int(events[-1]["attempt_id"].removeprefix("attempt-")) + 1
    if number > 9_999:
        raise RuntimeError("campaign_attempt_id_exhausted")
    return f"attempt-{number:04d}"


def create_attempt_root(campaign_root: Path, attempt_id: str) -> Path:
    """Create one fresh numbered staging root without removing prior evidence."""
    if not _ATTEMPT_ID.fullmatch(attempt_id):
        raise RuntimeError("campaign_attempt_id_invalid")
    if campaign_root.is_symlink():
        raise RuntimeError("campaign_root_invalid")
    _mkdir_namespace(campaign_root, parents=True, exist_ok=True)
    attempts_root = campaign_root / "attempts"
    if attempts_root.is_symlink() or (
        attempts_root.exists() and not attempts_root.is_dir()
    ):
        raise RuntimeError("campaign_attempts_root_invalid")
    _mkdir_namespace(attempts_root, exist_ok=True)
    attempt_root = attempts_root / attempt_id
    try:
        _mkdir_namespace(attempt_root)
    except FileExistsError as exc:
        raise RuntimeError("campaign_attempt_root_exists") from exc
    return attempt_root


def _legacy_orphan_attempt_id(
    campaign_root: Path, lineage: Sequence[Mapping[str, Any]]
) -> str | None:
    attempts_root = campaign_root / "attempts"
    if not attempts_root.exists():
        return None
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise RuntimeError("campaign_orphan_attempt_invalid")
    known = {event["attempt_id"] for event in lineage}
    unknown = [path for path in attempts_root.iterdir() if path.name not in known]
    if not unknown:
        return None
    orphan_id = next_attempt_id(lineage)
    if (
        len(unknown) != 1
        or unknown[0].name != orphan_id
        or unknown[0].is_symlink()
        or not unknown[0].is_dir()
    ):
        raise RuntimeError("campaign_orphan_attempt_invalid")
    return orphan_id


def require_campaign_acquisition(ledger: Path) -> str:
    """Return the next attempt ID only when the latest state permits retry."""
    lineage = attempt_lineage(ledger)
    if lineage and lineage[-1]["state"] in BLOCKING_ATTEMPT_STATES:
        raise RuntimeError(f"campaign_acquisition_blocked:{lineage[-1]['state']}")
    return next_attempt_id(lineage)


def complete_attempt_measurement(
    ledger: Path,
    attempt_id: str,
    *,
    verdict: str,
    raw_sha256: str,
) -> dict[str, Any]:
    """Record any protocol-valid measured verdict as pending independent review."""
    event = {
        "attempt_id": attempt_id,
        "state": "complete_pending_review",
        "verdict": verdict,
        "raw_sha256": raw_sha256,
    }
    append_attempt_state(ledger, event)
    return event


def process_start_identity(
    pid: int, *, run_command: Any = subprocess.run
) -> str | None:
    """Return a stable content-free fingerprint for one live PID's start time."""
    try:
        completed = run_command(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except BaseException as exc:
        raise RuntimeError("campaign_process_identity_failed") from exc
    if (
        returncode == 1
        and isinstance(stdout, str)
        and not stdout.strip()
        and isinstance(stderr, str)
        and not stderr.strip()
    ):
        return None
    if (
        not isinstance(returncode, int)
        or isinstance(returncode, bool)
        or returncode != 0
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or stderr.strip()
    ):
        raise RuntimeError("campaign_process_identity_failed")
    started = stdout.strip()
    try:
        time.strptime(started, "%a %b %d %H:%M:%S %Y")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("campaign_process_identity_failed") from exc
    return hashlib.sha256(started.encode("utf-8")).hexdigest()


def _validate_lock_owner(record: Mapping[str, Any]) -> CampaignLockOwner:
    if set(record) != {"pid", "process_start_sha256", "owner_token"}:
        raise RuntimeError("campaign_lock_owner_invalid")
    pid = record.get("pid")
    process_start_sha256 = record.get("process_start_sha256")
    owner_token = record.get("owner_token")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid < 1
        or not isinstance(process_start_sha256, str)
        or not _SHA256.fullmatch(process_start_sha256)
        or not isinstance(owner_token, str)
        or not _SHA256.fullmatch(owner_token)
    ):
        raise RuntimeError("campaign_lock_owner_invalid")
    return CampaignLockOwner(pid, process_start_sha256, owner_token)


def _read_lock_owner(lock_root: Path) -> CampaignLockOwner:
    owner_path = lock_root / "owner.json"
    try:
        if (
            not lock_root.is_dir()
            or lock_root.is_symlink()
            or {path.name for path in lock_root.iterdir()} != {"owner.json"}
            or not owner_path.is_file()
            or owner_path.is_symlink()
        ):
            raise RuntimeError("campaign_lock_owner_invalid")
        return _read_owner_file(owner_path)
    except OSError as exc:
        raise RuntimeError("campaign_lock_owner_invalid") from exc


def _read_owner_file(owner_path: Path) -> CampaignLockOwner:
    if not owner_path.is_file() or owner_path.is_symlink():
        raise RuntimeError("campaign_lock_owner_invalid")
    try:
        payload = owner_path.read_bytes()
        if not payload.endswith(b"\n"):
            raise RuntimeError("campaign_lock_owner_invalid")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise RuntimeError("campaign_lock_owner_invalid")
        owner = _validate_lock_owner(parsed)
        if payload != (json.dumps(parsed, sort_keys=True) + "\n").encode("utf-8"):
            raise RuntimeError("campaign_lock_owner_invalid")
        return owner
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("campaign_lock_owner_invalid") from exc


def _campaign_marker_error(campaign_root: Path) -> str | None:
    if (campaign_root / ".campaign-release").exists() or (
        campaign_root / ".campaign-release"
    ).is_symlink():
        return "campaign_release_in_progress"
    if (campaign_root / ".campaign-recovery").exists() or (
        campaign_root / ".campaign-recovery"
    ).is_symlink():
        return "campaign_recovery_in_progress"
    if (campaign_root / ".campaign-rollback").exists() or (
        campaign_root / ".campaign-rollback"
    ).is_symlink():
        return "campaign_recovery_in_progress"
    return None


def _is_empty_private_directory(path: Path) -> bool:
    try:
        return path.is_dir() and not path.is_symlink() and not any(path.iterdir())
    except OSError:
        return False


def _delete_exact_lock_root(
    lock_root: Path, owner: CampaignLockOwner
) -> None:
    if _read_lock_owner(lock_root) != owner:
        raise RuntimeError("campaign_lock_owner_mismatch")
    _unlink_namespace(lock_root / "owner.json")
    _rmdir_namespace(lock_root)


def _owner_payload(owner: CampaignLockOwner) -> str:
    return (
        json.dumps(
            {
                "pid": owner.pid,
                "process_start_sha256": owner.process_start_sha256,
                "owner_token": owner.owner_token,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _cleanup_owner_stage(stage: Path) -> None:
    try:
        owner_path = stage / "owner.json"
        if owner_path.is_file() and not owner_path.is_symlink():
            _unlink_namespace(owner_path)
        if _is_empty_private_directory(stage):
            _rmdir_namespace(stage)
    except OSError:
        pass


def _create_owner_stage(
    campaign_root: Path, owner: CampaignLockOwner
) -> Path:
    stage = Path(tempfile.mkdtemp(prefix=".campaign-owner-", dir=campaign_root))
    try:
        _fsync_directory(campaign_root)
        with (stage / "owner.json").open("x", encoding="utf-8") as stream:
            stream.write(_owner_payload(owner))
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(stage)
    except BaseException:
        _cleanup_owner_stage(stage)
        raise
    return stage


def _publish_campaign_lock(
    campaign_root: Path, owner: CampaignLockOwner
) -> CampaignLockOwner:
    stage = _create_owner_stage(campaign_root, owner)
    lock_root = campaign_root / ".campaign-lock"
    try:
        try:
            _rename_namespace(stage, lock_root)
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise RuntimeError("campaign_lock_held") from exc
            raise RuntimeError("campaign_lock_publish_failed") from exc
        return owner
    finally:
        _cleanup_owner_stage(stage)


def _restore_marker_to_canonical(
    campaign_root: Path,
    marker_root: Path,
    owner: CampaignLockOwner,
) -> bool:
    """Restore without replacing a concurrently created canonical lock."""
    lock_root = campaign_root / ".campaign-lock"
    try:
        if _read_lock_owner(marker_root) != owner:
            return False
    except RuntimeError:
        return False
    if lock_root.exists() or lock_root.is_symlink():
        try:
            if _is_empty_private_directory(lock_root):
                _rmdir_namespace(lock_root)
            elif _read_lock_owner(lock_root) != owner:
                return False
            else:
                _delete_exact_lock_root(marker_root, owner)
                return True
        except BaseException:
            return False
    try:
        if _read_lock_owner(marker_root) != owner:
            raise RuntimeError("campaign_lock_owner_mismatch")
        _publish_campaign_lock(campaign_root, owner)
        _delete_exact_lock_root(marker_root, owner)
    except BaseException:
        return False
    return True


def _preserve_recovery_rollback(
    campaign_root: Path,
    recovery_root: Path,
    owner: CampaignLockOwner,
) -> None:
    """Restore the owner, or preserve its marker when another lock won."""
    if _restore_marker_to_canonical(campaign_root, recovery_root, owner):
        return
    lock_root = campaign_root / ".campaign-lock"
    if not (lock_root.exists() or lock_root.is_symlink()) or (
        _is_empty_private_directory(lock_root)
    ):
        return
    try:
        if _read_lock_owner(lock_root) == owner:
            return
    except RuntimeError:
        pass
    try:
        _rename_namespace(recovery_root, campaign_root / ".campaign-rollback")
    except OSError:
        pass


def _probe_process_start_identity(
    process_start_probe: Callable[[int], str | None],
    pid: int,
    *,
    allow_dead: bool,
) -> str | None:
    try:
        observed = process_start_probe(pid)
    except BaseException as exc:
        raise RuntimeError("campaign_process_identity_failed") from exc
    if observed is None and allow_dead:
        return None
    if not isinstance(observed, str) or not _SHA256.fullmatch(observed):
        raise RuntimeError("campaign_process_identity_invalid")
    return observed


def acquire_campaign_lock(
    campaign_root: Path,
    *,
    pid: int | None = None,
    process_start_probe: Callable[[int], str | None] = process_start_identity,
    owner_token_factory: Callable[[], str] = lambda: secrets.token_hex(32),
) -> CampaignLockOwner:
    """Atomically acquire one campaign lock and write only private owner metadata."""
    if campaign_root.is_symlink():
        raise RuntimeError("campaign_root_invalid")
    _mkdir_namespace(campaign_root, parents=True, exist_ok=True)
    lock_root = campaign_root / ".campaign-lock"
    release_root = campaign_root / ".campaign-release"
    recovery_root = campaign_root / ".campaign-recovery"
    if _is_empty_private_directory(release_root) and not lock_root.exists():
        _rmdir_namespace(release_root)
    if _is_empty_private_directory(recovery_root) and not lock_root.exists():
        recovery_lineage = attempt_lineage(campaign_root / "attempts.jsonl")
        if not recovery_lineage or recovery_lineage[-1] == {
            "attempt_id": recovery_lineage[-1]["attempt_id"],
            "state": "failed",
            "reason_category": "interrupted",
        }:
            _rmdir_namespace(recovery_root)
    if _is_empty_private_directory(lock_root):
        try:
            _rename_namespace(lock_root, recovery_root)
        except OSError as exc:
            raise RuntimeError("campaign_lock_held") from exc
        if not _is_empty_private_directory(recovery_root):
            try:
                _rename_namespace(recovery_root, lock_root)
            except OSError:
                pass
            raise RuntimeError("campaign_lock_held")
        _rmdir_namespace(recovery_root)
    rollback_root = campaign_root / ".campaign-rollback"
    if rollback_root.exists() or rollback_root.is_symlink():
        if lock_root.exists() or lock_root.is_symlink():
            raise RuntimeError("campaign_recovery_in_progress")
        rollback_owner = _read_lock_owner(rollback_root)
        if _restore_marker_to_canonical(
            campaign_root, rollback_root, rollback_owner
        ):
            raise RuntimeError("campaign_recovery_rolled_back")
        raise RuntimeError("campaign_recovery_in_progress")
    marker_error = _campaign_marker_error(campaign_root)
    if marker_error is not None:
        raise RuntimeError(marker_error)
    if lock_root.exists() or lock_root.is_symlink():
        raise RuntimeError("campaign_lock_held")
    owner_pid = os.getpid() if pid is None else pid
    start_identity = _probe_process_start_identity(
        process_start_probe, owner_pid, allow_dead=False
    )
    try:
        owner_token = owner_token_factory()
    except BaseException as exc:
        raise RuntimeError("campaign_owner_token_failed") from exc
    owner = _validate_lock_owner(
        {
            "pid": owner_pid,
            "process_start_sha256": start_identity,
            "owner_token": owner_token,
        }
    )
    _publish_campaign_lock(campaign_root, owner)
    marker_error = _campaign_marker_error(campaign_root)
    if marker_error is not None:
        _delete_exact_lock_root(lock_root, owner)
        raise RuntimeError(marker_error)
    return owner


def release_campaign_lock(campaign_root: Path, owner: CampaignLockOwner) -> None:
    """Atomically own, validate, and remove only the exact caller-owned lock."""
    lock_root = campaign_root / ".campaign-lock"
    release_root = campaign_root / ".campaign-release"
    if release_root.exists() or release_root.is_symlink():
        if lock_root.exists() or lock_root.is_symlink():
            raise RuntimeError("campaign_release_in_progress")
        if _is_empty_private_directory(release_root):
            _rmdir_namespace(release_root)
            return
        observed_owner = _read_lock_owner(release_root)
        if observed_owner != owner:
            _restore_marker_to_canonical(
                campaign_root, release_root, observed_owner
            )
            raise RuntimeError("campaign_lock_owner_mismatch")
        _delete_exact_lock_root(release_root, owner)
        return
    marker_error = _campaign_marker_error(campaign_root)
    if marker_error is not None:
        raise RuntimeError(marker_error)
    if _read_lock_owner(lock_root) != owner:
        raise RuntimeError("campaign_lock_owner_mismatch")
    try:
        _rename_namespace(lock_root, release_root)
    except FileNotFoundError as exc:
        raise RuntimeError("campaign_lock_owner_invalid") from exc
    except OSError as exc:
        raise RuntimeError("campaign_release_in_progress") from exc
    observed_owner = _read_lock_owner(release_root)
    if observed_owner != owner:
        _restore_marker_to_canonical(campaign_root, release_root, observed_owner)
        raise RuntimeError("campaign_lock_owner_mismatch")
    _delete_exact_lock_root(release_root, owner)


def acquire_campaign_attempt(
    campaign_root: Path,
    *,
    pid: int | None = None,
    process_start_probe: Callable[[int], str | None] = process_start_identity,
    owner_token_factory: Callable[[], str] = lambda: secrets.token_hex(32),
) -> CampaignAttempt:
    """Acquire the campaign and durably start one fresh numbered attempt."""
    owner = acquire_campaign_lock(
        campaign_root,
        pid=pid,
        process_start_probe=process_start_probe,
        owner_token_factory=owner_token_factory,
    )
    attempt_id: str | None = None
    try:
        ledger = campaign_root / "attempts.jsonl"
        attempt_id = require_campaign_acquisition(ledger)
        append_attempt_state(
            ledger, {"attempt_id": attempt_id, "state": "running"}
        )
        try:
            root = create_attempt_root(campaign_root, attempt_id)
        except BaseException as primary:
            try:
                append_attempt_state(
                    ledger,
                    {
                        "attempt_id": attempt_id,
                        "state": "failed",
                        "reason_category": "acquisition",
                    },
                )
            except BaseException as terminal:
                raise BaseExceptionGroup(
                    "campaign_staging_and_terminal_state_failed",
                    [primary, terminal],
                ) from None
            raise
        return CampaignAttempt(attempt_id, root, owner)
    except BaseException as primary:
        try:
            current = attempt_lineage(campaign_root / "attempts.jsonl")
        except BaseException:
            current = ()
        if current and current[-1] == {
            "attempt_id": attempt_id,
            "state": "running",
        }:
            raise
        try:
            release_campaign_lock(campaign_root, owner)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "campaign_acquisition_and_release_failed", [primary, cleanup]
            ) from None
        raise


def release_campaign_attempt(campaign_root: Path, attempt: CampaignAttempt) -> None:
    """Release the exact lock held by a normally terminated attempt."""
    release_campaign_lock(campaign_root, attempt.owner)


def recover_interrupted_attempt(
    campaign_root: Path,
    *,
    process_start_probe: Callable[[int], str | None] = process_start_identity,
) -> dict[str, Any]:
    """Atomically own a dead running lock and append only ``failed:interrupted``."""
    lock_root = campaign_root / ".campaign-lock"
    recovery_root = campaign_root / ".campaign-recovery"
    ledger = campaign_root / "attempts.jsonl"
    lineage = attempt_lineage(ledger)
    latest = lineage[-1] if lineage else None
    orphan_id = _legacy_orphan_attempt_id(campaign_root, lineage)
    if (recovery_root.exists() or recovery_root.is_symlink()) and (
        lock_root.exists() or lock_root.is_symlink()
    ):
        canonical_owner = _read_lock_owner(lock_root)
        if _is_empty_private_directory(recovery_root):
            _rmdir_namespace(recovery_root)
        else:
            recovery_owner = _read_lock_owner(recovery_root)
            if recovery_owner != canonical_owner:
                raise RuntimeError("campaign_recovery_owner_conflict")
            _delete_exact_lock_root(recovery_root, recovery_owner)
    owner: CampaignLockOwner | None = None
    if recovery_root.exists() or recovery_root.is_symlink():
        if lineage and lineage[-1] == {
            "attempt_id": lineage[-1]["attempt_id"],
            "state": "failed",
            "reason_category": "interrupted",
        }:
            if _is_empty_private_directory(recovery_root):
                _rmdir_namespace(recovery_root)
            else:
                recovery_owner = _read_lock_owner(recovery_root)
                _delete_exact_lock_root(recovery_root, recovery_owner)
            return lineage[-1]
        if _is_empty_private_directory(recovery_root) and not lineage:
            _rmdir_namespace(recovery_root)
            return {"state": "failed", "reason_category": "interrupted"}
        if (latest is None or latest["state"] != "running") and orphan_id is None:
            raise RuntimeError("campaign_recovery_in_progress")
        owner = _read_lock_owner(recovery_root)
        observed_start = _probe_process_start_identity(
            process_start_probe, owner.pid, allow_dead=True
        )
        if observed_start == owner.process_start_sha256:
            _preserve_recovery_rollback(campaign_root, recovery_root, owner)
            raise RuntimeError("campaign_lock_owner_live")
    else:
        marker_error = _campaign_marker_error(campaign_root)
        if marker_error is not None:
            raise RuntimeError(marker_error)
        owner = _read_lock_owner(lock_root)
    if latest is not None and latest["state"] != "running" and orphan_id is None:
        if latest["state"] in BLOCKING_ATTEMPT_STATES:
            raise RuntimeError(
                f"campaign_recovery_state_blocked:{latest['state']}"
            )
        raise RuntimeError("campaign_recovery_state_invalid")
    if not recovery_root.exists():
        observed_start = _probe_process_start_identity(
            process_start_probe, owner.pid, allow_dead=True
        )
        if observed_start == owner.process_start_sha256:
            raise RuntimeError("campaign_lock_owner_live")
        try:
            _rename_namespace(lock_root, recovery_root)
        except OSError as exc:
            raise RuntimeError("campaign_recovery_lost") from exc
        taken_owner = _read_lock_owner(recovery_root)
        if taken_owner != owner:
            _preserve_recovery_rollback(
                campaign_root, recovery_root, taken_owner
            )
            raise RuntimeError("campaign_lock_owner_mismatch")
    event = (
        {
            "attempt_id": orphan_id or latest["attempt_id"],
            "state": "failed",
            "reason_category": "interrupted",
        }
        if latest is not None or orphan_id is not None
        else {"state": "failed", "reason_category": "interrupted"}
    )
    try:
        observed_start = _probe_process_start_identity(
            process_start_probe, owner.pid, allow_dead=True
        )
        if observed_start == owner.process_start_sha256:
            raise RuntimeError("campaign_lock_owner_live")
        current = attempt_lineage(ledger)
        if current != lineage or (
            orphan_id is None
            and latest is not None
            and current[-1]["state"] != "running"
        ):
            raise RuntimeError("campaign_recovery_state_changed")
        if orphan_id is not None:
            append_attempt_state(
                ledger, {"attempt_id": orphan_id, "state": "running"}
            )
            append_attempt_state(ledger, event)
        elif latest is not None:
            append_attempt_state(ledger, event)
    except BaseException:
        try:
            committed_lineage = (
                *lineage,
                *(
                    ({"attempt_id": orphan_id, "state": "running"},)
                    if orphan_id is not None
                    else ()
                ),
                *(
                    (event,)
                    if latest is not None or orphan_id is not None
                    else ()
                ),
            )
            appended = (
                latest is not None or orphan_id is not None
            ) and attempt_lineage(ledger) == committed_lineage
        except BaseException:
            appended = False
        if appended:
            raise
        _preserve_recovery_rollback(campaign_root, recovery_root, owner)
        raise
    _unlink_namespace(recovery_root / "owner.json")
    _rmdir_namespace(recovery_root)
    return event


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


def safe_error_origin(error: BaseException) -> str:
    """Return only the terminal traceback function name, never a file path."""
    frames = traceback.extract_tb(error.__traceback__)
    value = frames[-1].name if frames else "unknown"
    return value if re.fullmatch(r"[A-Za-z0-9_<>.-]{1,120}", value) else "unknown"


async def await_owned_cleanup(awaitable: Any) -> None:
    """Ignore child-operation cancellation while preserving caller cancellation."""
    import asyncio

    try:
        await awaitable
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if not should_suppress_owned_teardown_cancel(
            contract_complete=True,
            cancellation_count=current.cancelling() if current is not None else 0,
        ):
            raise


def should_suppress_owned_teardown_cancel(
    *, contract_complete: bool, cancellation_count: int
) -> bool:
    """Distinguish child-loop teardown cancellation from caller cancellation."""
    return contract_complete and cancellation_count == 0


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
    if set(spec) not in {_CHILD_SPEC_KEYS, _CHILD_SPEC_KEYS_WITH_MODE}:
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
    if not isinstance(value, dict) or set(value) not in {
        _CHILD_SPEC_KEYS,
        _CHILD_SPEC_KEYS_WITH_MODE,
    }:
        raise RuntimeError("child_spec_invalid")
    mode = value.get("mode", "sample")
    phases = (
        {"protocol_preflight"}
        if mode == "protocol_preflight"
        else {"warmup", "burn_in", "measured"}
    )
    if (
        mode not in {"sample", "protocol_preflight"}
        or value.get("arm") not in ARMS
        or value.get("phase") not in phases
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
                f"temperature = {REQUEST_SETTINGS['temperature']}",
                f"max_tokens = {REQUEST_SETTINGS['max_tokens']}",
                "reasoning_effort = "
                f"{json.dumps(REQUEST_SETTINGS['reasoning_effort'])}",
                "timeout = 120",
                "retries = 0",
                f"streaming = {json.dumps(REQUEST_SETTINGS['streaming'])}",
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


def current_harness_identity(
    repository_root: Path,
    *,
    runner_path: Path | None = None,
    run_command: Any = subprocess.run,
) -> dict[str, str]:
    """Require a clean harness and retain its full commit and runner digest."""
    status = run_command(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError("harness_status_failed")
    if status.stdout:
        raise RuntimeError("harness_worktree_dirty")
    revision_result = run_command(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = revision_result.stdout.strip()
    if revision_result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("harness_revision_failed")
    runner = (runner_path or Path(__file__)).resolve()
    if not runner.is_file() or runner.is_symlink():
        raise RuntimeError("harness_runner_missing")
    return {"revision": revision, "runner_sha256": _sha256_file(runner)}


def listener_identity(
    endpoint: str,
    *,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    """Hash listener PID/start identity without retaining either raw value."""
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
    inventory = {
        line.strip() for line in lookup.stdout.splitlines() if line.strip()
    }
    if (
        lookup.returncode != 0
        or not inventory
        or any(not pid.isdecimal() for pid in inventory)
    ):
        raise RuntimeError("listener_identity_failed")
    pids = sorted(inventory, key=int)
    identities = []
    for pid in pids:
        sample = run_command(
            ["ps", "-o", "pid=,lstart=", "-p", pid],
            check=False,
            capture_output=True,
            text=True,
        )
        fields = sample.stdout.strip().split(maxsplit=1)
        if (
            sample.returncode != 0
            or len(fields) != 2
            or fields[0] != pid
            or not fields[1].strip()
        ):
            raise RuntimeError("listener_identity_failed")
        identities.append(f"{pid}\0{fields[1].strip()}")
    fingerprint = hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()
    return {"listener_count": len(identities), "fingerprint_sha256": fingerprint}


def verify_listener_identity(
    endpoint: str,
    expected_fingerprint: str,
    *,
    run_command: Any = subprocess.run,
) -> dict[str, Any]:
    """Fail the attempt if the exact listener changes at a boundary."""
    if not _SHA256.fullmatch(expected_fingerprint):
        raise RuntimeError("listener_identity_invalid")
    observed = listener_identity(endpoint, run_command=run_command)
    if observed["fingerprint_sha256"] != expected_fingerprint:
        raise RuntimeError("listener_identity_changed")
    return observed


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


def sample_schedule(
    iterations: int, *, burn_in_blocks: int = 0
) -> tuple[SamplePlan, ...]:
    """Return warmups, optional burn-in, and complete rotated measured blocks."""
    if iterations < 1 or burn_in_blocks < 0:
        raise ValueError("schedule counts must be nonnegative with measured iterations")
    schedule = [SamplePlan("warmup", arm, -1) for arm in ARMS]
    for block in range(burn_in_blocks):
        schedule.extend(
            SamplePlan("burn_in", arm, block)
            for arm in balanced_arm_order(block)
        )
    for iteration in range(iterations):
        schedule.extend(
            SamplePlan("measured", arm, iteration)
            for arm in balanced_arm_order(burn_in_blocks + iteration)
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


def prepare_target_worktree(
    repository_root: Path,
    run_root: Path,
    *,
    name: str,
    revision: str,
    run_command: Any = subprocess.run,
) -> Path:
    """Create one named detached target owned by the explicit run root."""
    if name not in {"control", "candidate"} or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        raise RuntimeError("target_worktree_invalid")
    root = run_root.resolve()
    target = (root / name).resolve()
    if target.parent != root:
        raise RuntimeError("target_worktree_invalid")
    if target.exists():
        raise RuntimeError(f"target_worktree_failed:{name}:target_exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "worktree", "add", "--detach", str(target), revision]
    try:
        completed = run_command(
            command,
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except BaseException as exc:
        primary = exc
    else:
        if completed.returncode == 0 and target.is_dir():
            return target
        primary = RuntimeError(f"target_worktree_failed:{name}")

    cleanup_failures: list[BaseException] = []
    try:
        _remove_target_worktree(
            repository_root,
            root,
            name=name,
            run_command=run_command,
            allow_unregistered_owned=True,
        )
    except BaseException as exc:
        cleanup_failures.append(exc)
    _raise_failures(
        "target_worktree_add_failed", [primary, *cleanup_failures]
    )
    raise AssertionError("unreachable")


def _target_worktree_registered(
    repository_root: Path,
    target: Path,
    *,
    run_command: Any = subprocess.run,
) -> bool:
    registrations = _worktree_registrations(
        repository_root, run_command=run_command
    )
    expected = str(target) if target.is_absolute() else str(target.resolve())
    return expected in registrations


def _worktree_registrations(
    repository_root: Path, *, run_command: Any = subprocess.run
) -> frozenset[str]:
    completed = run_command(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("target_worktree_registration_check_failed")
    return frozenset(
        line.removeprefix("worktree ")
        for line in completed.stdout.splitlines()
        if line.startswith("worktree ")
    )


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


def _prepare_workspace_runtime_owned(
    sample_root: Path,
    *,
    arm: str,
    readiness_timeout: float = 30.0,
    _owned_resources: list[tuple[str, Any]],
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
    _owned_resources.append(("database", database))
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
        _owned_resources.append(("consent", consent_service))
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


def prepare_workspace_runtime(
    sample_root: Path,
    *,
    arm: str,
    readiness_timeout: float = 30.0,
) -> WorkspaceRuntime:
    """Build an isolated runtime and close every partial construction on failure."""
    owned_resources: list[tuple[str, Any]] = []
    try:
        return _prepare_workspace_runtime_owned(
            sample_root,
            arm=arm,
            readiness_timeout=readiness_timeout,
            _owned_resources=owned_resources,
        )
    except BaseException as primary:
        failures: list[BaseException] = [primary]
        for kind, resource in reversed(owned_resources):
            try:
                if kind == "consent":
                    resource.shutdown(timeout=2.0)
                else:
                    resource.close()
            except BaseException as exc:
                failures.append(exc)
        _raise_failures("workspace_runtime_construction_failed", failures)
        raise AssertionError("unreachable")


def protocol_preflight_ownership(
    baseline_thread_ids: set[int],
) -> dict[str, Any]:
    """Measure content-free ownership after protocol-preflight cleanup."""
    survivors = [
        thread
        for thread in threading.enumerate()
        if id(thread) not in baseline_thread_ids and thread.is_alive()
    ]
    if survivors:
        raise RuntimeError("protocol_preflight_thread_survivor")
    return {"live_threads": 0}


def _raise_failures(message: str, failures: Sequence[BaseException]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    raise BaseExceptionGroup(message, list(failures))


def protocol_preflight(
    target_root: Path,
    sample_root: Path,
    *,
    arm: str,
    adapter_factory: Callable[[Path, str], Any] = TargetAdapter.for_arm,
    runtime_factory: Callable[..., Any] = prepare_workspace_runtime,
    corpus_generator: Callable[[Path], Mapping[str, Any]] = generate_corpus,
    ownership_probe: Callable[[set[int]], Mapping[str, Any]] = (
        protocol_preflight_ownership
    ),
) -> dict[str, Any]:
    """Derive target behavior/tool fixtures without mounting a conversation."""
    baseline_thread_ids = {id(thread) for thread in threading.enumerate()}
    adapter = adapter_factory(target_root.resolve(), arm)
    runtime: Any | None = None
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    cleanup_failures: list[BaseException] = []
    try:
        runtime = runtime_factory(sample_root.resolve(), arm=arm)
        corpus = corpus_generator(runtime.workspace_root)
        corpus_digest = corpus.get("content_tree_digest")
        tool_digest = runtime.permission_definition_hash
        if (
            not isinstance(corpus_digest, str)
            or not _SHA256.fullmatch(corpus_digest)
            or not isinstance(tool_digest, str)
            or not _SHA256.fullmatch(tool_digest)
        ):
            raise RuntimeError("protocol_preflight_hash_invalid")
        behavior = {
            "target_revision_kind": adapter.revision_kind,
            "review_state": runtime.review_state,
            "review_ready": runtime.review_ready,
        }
        result = {
            "event": "protocol_preflight",
            "arm": arm,
            "target_revision_kind": adapter.revision_kind,
            "behavior_sha256": hashlib.sha256(
                json.dumps(
                    behavior, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "workspace_content_tree_digest": corpus_digest,
            "tool_definition_sha256": tool_digest,
        }
    except BaseException as exc:
        failure = exc
    finally:
        try:
            if runtime is not None:
                runtime.close()
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            adapter.close()
        except BaseException as exc:
            cleanup_failures.append(exc)
    _raise_failures(
        "protocol_preflight_failed",
        ([failure] if failure is not None else []) + cleanup_failures,
    )
    if result is None:
        raise RuntimeError("protocol_preflight_failed")
    result["final_ownership"] = dict(ownership_probe(baseline_thread_ids))
    return result


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
            await await_owned_cleanup(console_runtime.dispose())
            console_runtime = None
    finally:
        if console_runtime is not None:
            await await_owned_cleanup(console_runtime.dispose())
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
                "temperature": REQUEST_SETTINGS["temperature"],
                "max_tokens": REQUEST_SETTINGS["max_tokens"],
                "reasoning_effort": REQUEST_SETTINGS["reasoning_effort"],
                "timeout": 120,
                "retries": 0,
                "streaming": REQUEST_SETTINGS["streaming"],
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
                payload["stream_options"] = {
                    "include_usage": REQUEST_SETTINGS["include_usage"]
                }
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

        mounted_contract_complete = False

        @asynccontextmanager
        async def owned_host_test():
            try:
                async with host.run_test(size=(160, 48)) as pilot:
                    yield pilot
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if not should_suppress_owned_teardown_cancel(
                    contract_complete=mounted_contract_complete,
                    cancellation_count=(
                        current.cancelling() if current is not None else 0
                    ),
                ):
                    raise

        async with owned_host_test() as pilot:
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
                    temperature=REQUEST_SETTINGS["temperature"],
                    max_tokens=REQUEST_SETTINGS["max_tokens"],
                    reasoning_effort=REQUEST_SETTINGS["reasoning_effort"],
                    streaming=REQUEST_SETTINGS["streaming"],
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
            await await_owned_cleanup(console_runtime.dispose())
            console_runtime = None
            mounted_contract_complete = True

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
                await await_owned_cleanup(console_runtime.dispose())
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
            install_target_root(target_root)
            imported = assert_target_modules(TARGET_MODULES, target_root)
            target_modules = {
                name: Path(path).relative_to(target_root).as_posix()
                for name, path in imported.items()
            }
            if spec.get("mode", "sample") == "protocol_preflight":
                result = protocol_preflight(
                    target_root,
                    sample_root,
                    arm=str(spec["arm"]),
                )
                result.update(
                    {
                        "sample_id": spec["sample_id"],
                        "phase": spec["phase"],
                        "iteration": spec["iteration"],
                        "arm": spec["arm"],
                        "target_modules": target_modules,
                    }
                )
                if privacy_violations(result):
                    raise RuntimeError("protocol_preflight_privacy_violation")
                write_boundary_event(evidence, result)
                return 0
            adapter = TargetAdapter.for_arm(target_root, str(spec["arm"]))
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
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
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
                    "error_origin": safe_error_origin(exc),
                },
            )
            return 1


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _strict_owned_directory(
    path: Path,
    *,
    parent: Path | None,
    error_code: str,
) -> tuple[Path, tuple[int, int]]:
    try:
        if path.is_symlink():
            raise RuntimeError(error_code)
        before = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
        after = resolved.stat()
    except OSError as exc:
        raise RuntimeError(error_code) from exc
    identity = (before.st_dev, before.st_ino)
    if (
        not stat.S_ISDIR(before.st_mode)
        or identity != (after.st_dev, after.st_ino)
        or (parent is not None and resolved.parent != parent)
    ):
        raise RuntimeError(error_code)
    return resolved, identity


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _remove_directory_contents_fd(
    descriptor: int, *, preserve: frozenset[str] = frozenset()
) -> None:
    for entry in os.scandir(descriptor):
        if entry.name in preserve:
            continue
        if entry.is_dir(follow_symlinks=False):
            try:
                child = os.open(
                    entry.name, _directory_open_flags(), dir_fd=descriptor
                )
            except FileNotFoundError:
                continue
            try:
                _remove_directory_contents_fd(child)
            finally:
                os.close(child)
            try:
                os.rmdir(entry.name, dir_fd=descriptor)
            except FileNotFoundError:
                pass
        else:
            try:
                os.unlink(entry.name, dir_fd=descriptor)
            except FileNotFoundError:
                pass
    os.fsync(descriptor)


def _git_common_directory(
    repository_root: Path, *, run_command: Any
) -> Path:
    completed = run_command(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        completed.returncode != 0
        or not isinstance(completed.stdout, str)
        or len(completed.stdout.splitlines()) != 1
        or not completed.stdout.strip()
    ):
        raise RuntimeError("target_worktree_unregister_failed")
    common = Path(completed.stdout.strip())
    if not common.is_absolute():
        common = repository_root / common
    return Path(os.path.abspath(common))


def _open_strict_directory(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute():
        raise RuntimeError("target_worktree_unregister_failed")
    descriptor = os.open("/", _directory_open_flags())
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, _directory_open_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise RuntimeError("target_worktree_unregister_failed") from exc
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_fd(
    parent_descriptor: int, name: str, *, error_code: str
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
                raise RuntimeError(error_code)
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 4097):
                chunks.append(chunk)
                if sum(map(len, chunks)) > 4096:
                    raise RuntimeError(error_code)
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RuntimeError(error_code) from exc
    if len(payload) > 4096:
        raise RuntimeError(error_code)
    return payload


def _admin_gitdir_target(admin_descriptor: int) -> str:
    payload = _read_regular_file_fd(
        admin_descriptor,
        "gitdir",
        error_code="target_worktree_admin_invalid",
    )
    try:
        value = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("target_worktree_admin_invalid") from exc
    if not value.endswith("\n") or "\n" in value[:-1] or not value[:-1]:
        raise RuntimeError("target_worktree_admin_invalid")
    return os.path.abspath(value[:-1])


def _worktree_admin_marker_name(target: str) -> str:
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
    return f".campaign-worktree-cleanup-{digest}"


def _worktree_backlink_admin_name(
    owned_descriptor: int, common: Path
) -> str:
    payload = _read_regular_file_fd(
        owned_descriptor,
        ".git",
        error_code="target_worktree_admin_invalid",
    )
    prefix = b"gitdir: "
    if (
        not payload.startswith(prefix)
        or not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
    ):
        raise RuntimeError("target_worktree_admin_invalid")
    try:
        value = payload[len(prefix) : -1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("target_worktree_admin_invalid") from exc
    admin = Path(os.path.abspath(value))
    if admin.parent != common / "worktrees" or admin.name in {"", ".", ".."}:
        raise RuntimeError("target_worktree_admin_invalid")
    return admin.name


def _find_worktree_admin(common: Path, target: str) -> tuple[str, int]:
    """Match Git's linked-worktree admin by its exact ``gitdir`` backlink."""
    expected_gitfile = os.path.abspath(f"{target}/.git")
    common_descriptor = _open_strict_directory(common)
    match: tuple[str, int] | None = None
    try:
        worktrees_descriptor = os.open(
            "worktrees", _directory_open_flags(), dir_fd=common_descriptor
        )
        try:
            for entry in os.scandir(worktrees_descriptor):
                if not entry.is_dir(follow_symlinks=False):
                    raise RuntimeError("target_worktree_admin_invalid")
                admin_descriptor = os.open(
                    entry.name,
                    _directory_open_flags(),
                    dir_fd=worktrees_descriptor,
                )
                retained = False
                try:
                    if _admin_gitdir_target(admin_descriptor) == expected_gitfile:
                        if match is not None:
                            raise RuntimeError("target_worktree_admin_invalid")
                        match = (entry.name, admin_descriptor)
                        retained = True
                finally:
                    if not retained:
                        os.close(admin_descriptor)
        finally:
            os.close(worktrees_descriptor)
    except OSError as exc:
        if match is not None:
            os.close(match[1])
        raise RuntimeError("target_worktree_admin_invalid") from exc
    except BaseException:
        if match is not None:
            os.close(match[1])
        raise
    finally:
        os.close(common_descriptor)
    if match is None:
        raise RuntimeError("target_worktree_admin_invalid")
    return match


def _admin_marker_state(common: Path, target: str) -> str:
    marker_name = _worktree_admin_marker_name(target)
    common_descriptor = _open_strict_directory(common)
    try:
        try:
            marker_descriptor = os.open(
                marker_name,
                _directory_open_flags(),
                dir_fd=common_descriptor,
            )
        except FileNotFoundError:
            return "absent"
        try:
            entries = {entry.name for entry in os.scandir(marker_descriptor)}
            if not entries:
                return "empty"
            if "identity-conflict" in entries:
                raise RuntimeError("target_worktree_admin_marker_conflict")
            if entries in ({"retired"}, {"admin", "retired"}):
                receipt = _read_regular_file_fd(
                    marker_descriptor,
                    "retired",
                    error_code="target_worktree_admin_marker_conflict",
                )
                if receipt != f"{os.path.abspath(f'{target}/.git')}\n".encode():
                    raise RuntimeError(
                        "target_worktree_admin_marker_conflict"
                    )
                if "admin" in entries:
                    admin_descriptor = os.open(
                        "admin",
                        _directory_open_flags(),
                        dir_fd=marker_descriptor,
                    )
                    try:
                        if any(os.scandir(admin_descriptor)):
                            raise RuntimeError(
                                "target_worktree_admin_marker_conflict"
                            )
                    finally:
                        os.close(admin_descriptor)
                return "retiring"
            if entries != {"admin"}:
                raise RuntimeError("target_worktree_admin_marker_conflict")
            admin_descriptor = os.open(
                "admin", _directory_open_flags(), dir_fd=marker_descriptor
            )
            try:
                if _admin_gitdir_target(admin_descriptor) != os.path.abspath(
                    f"{target}/.git"
                ):
                    raise RuntimeError(
                        "target_worktree_admin_marker_conflict"
                    )
            finally:
                os.close(admin_descriptor)
            return "owned"
        finally:
            os.close(marker_descriptor)
    except OSError as exc:
        raise RuntimeError("target_worktree_admin_marker_conflict") from exc
    finally:
        os.close(common_descriptor)


def _move_worktree_admin_to_marker(
    common: Path,
    target: str,
    admin_name: str,
    expected_admin_descriptor: int,
) -> int:
    """Atomically unregister one linked worktree without touching its content."""
    marker_name = _worktree_admin_marker_name(target)
    common_descriptor = _open_strict_directory(common)
    try:
        worktrees_descriptor = os.open(
            "worktrees", _directory_open_flags(), dir_fd=common_descriptor
        )
        try:
            try:
                os.mkdir(marker_name, mode=0o700, dir_fd=common_descriptor)
                os.fsync(common_descriptor)
            except FileExistsError:
                pass
            marker_descriptor = os.open(
                marker_name,
                _directory_open_flags(),
                dir_fd=common_descriptor,
            )
            try:
                entries = {entry.name for entry in os.scandir(marker_descriptor)}
                if entries not in (set(), {"admin"}):
                    raise RuntimeError(
                        "target_worktree_admin_marker_conflict"
                    )
                if not entries:
                    try:
                        os.rename(
                            admin_name,
                            "admin",
                            src_dir_fd=worktrees_descriptor,
                            dst_dir_fd=marker_descriptor,
                        )
                        os.fsync(worktrees_descriptor)
                        os.fsync(marker_descriptor)
                    except FileNotFoundError:
                        if {
                            entry.name for entry in os.scandir(marker_descriptor)
                        } != {"admin"}:
                            raise RuntimeError(
                                "target_worktree_unregister_failed"
                            ) from None
                moved_descriptor = os.open(
                    "admin",
                    _directory_open_flags(),
                    dir_fd=marker_descriptor,
                )
                retain_moved_descriptor = False
                try:
                    expected = os.fstat(expected_admin_descriptor)
                    moved = os.fstat(moved_descriptor)
                    identity_changed = (
                        expected.st_dev,
                        expected.st_ino,
                    ) != (moved.st_dev, moved.st_ino)
                    metadata_changed = (
                        _admin_gitdir_target(moved_descriptor)
                        != os.path.abspath(f"{target}/.git")
                    )
                    if identity_changed or metadata_changed:
                        os.rename(
                            "admin",
                            "conflict",
                            src_dir_fd=marker_descriptor,
                            dst_dir_fd=marker_descriptor,
                        )
                        conflict_descriptor = os.open(
                            "identity-conflict",
                            os.O_WRONLY
                            | os.O_CREAT
                            | os.O_EXCL
                            | getattr(os, "O_CLOEXEC", 0),
                            0o600,
                            dir_fd=marker_descriptor,
                        )
                        try:
                            os.fsync(conflict_descriptor)
                        finally:
                            os.close(conflict_descriptor)
                        os.fsync(marker_descriptor)
                        try:
                            os.rename(
                                "conflict",
                                admin_name,
                                src_dir_fd=marker_descriptor,
                                dst_dir_fd=worktrees_descriptor,
                            )
                        except OSError:
                            pass
                        else:
                            os.fsync(worktrees_descriptor)
                            os.fsync(marker_descriptor)
                        raise RuntimeError(
                            "target_worktree_admin_identity_changed"
                        )
                    retain_moved_descriptor = True
                finally:
                    if not retain_moved_descriptor:
                        os.close(moved_descriptor)
            finally:
                os.close(marker_descriptor)
        finally:
            os.close(worktrees_descriptor)
    except OSError as exc:
        raise RuntimeError("target_worktree_unregister_failed") from exc
    finally:
        os.close(common_descriptor)
    try:
        marker_state = _admin_marker_state(common, target)
    except BaseException:
        os.close(moved_descriptor)
        raise
    if marker_state != "owned":
        os.close(moved_descriptor)
        raise RuntimeError("target_worktree_admin_marker_conflict")
    return moved_descriptor


def _delete_worktree_admin_marker(
    common: Path,
    target: str,
    *,
    claimed_admin_descriptor: int | None = None,
) -> None:
    marker_name = _worktree_admin_marker_name(target)
    common_descriptor = _open_strict_directory(common)
    try:
        marker_descriptor = os.open(
            marker_name,
            _directory_open_flags(),
            dir_fd=common_descriptor,
        )
        try:
            entries = {entry.name for entry in os.scandir(marker_descriptor)}
            if not entries:
                pass
            elif entries in ({"retired"}, {"admin", "retired"}):
                receipt = _read_regular_file_fd(
                    marker_descriptor,
                    "retired",
                    error_code="target_worktree_admin_marker_conflict",
                )
                if receipt != f"{os.path.abspath(f'{target}/.git')}\n".encode():
                    raise RuntimeError(
                        "target_worktree_admin_marker_conflict"
                    )
                if "admin" in entries:
                    admin_descriptor = os.open(
                        "admin",
                        _directory_open_flags(),
                        dir_fd=marker_descriptor,
                    )
                    try:
                        if any(os.scandir(admin_descriptor)):
                            raise RuntimeError(
                                "target_worktree_admin_marker_conflict"
                            )
                    finally:
                        os.close(admin_descriptor)
                    os.rmdir("admin", dir_fd=marker_descriptor)
                    os.fsync(marker_descriptor)
                os.unlink("retired", dir_fd=marker_descriptor)
                os.fsync(marker_descriptor)
            elif entries != {"admin"}:
                raise RuntimeError("target_worktree_admin_marker_conflict")
            else:
                admin_descriptor = os.open(
                    "admin", _directory_open_flags(), dir_fd=marker_descriptor
                )
                try:
                    if claimed_admin_descriptor is not None:
                        claimed = os.fstat(claimed_admin_descriptor)
                        observed = os.fstat(admin_descriptor)
                        if (claimed.st_dev, claimed.st_ino) != (
                            observed.st_dev,
                            observed.st_ino,
                        ):
                            raise RuntimeError(
                                "target_worktree_admin_marker_conflict"
                            )
                    if _admin_gitdir_target(
                        admin_descriptor
                    ) != os.path.abspath(f"{target}/.git"):
                        raise RuntimeError(
                            "target_worktree_admin_marker_conflict"
                        )
                    _remove_directory_contents_fd(
                        admin_descriptor, preserve=frozenset({"gitdir"})
                    )
                    os.rename(
                        "gitdir",
                        "retired",
                        src_dir_fd=admin_descriptor,
                        dst_dir_fd=marker_descriptor,
                    )
                    os.fsync(admin_descriptor)
                    os.fsync(marker_descriptor)
                finally:
                    os.close(admin_descriptor)
                os.rmdir("admin", dir_fd=marker_descriptor)
                os.fsync(marker_descriptor)
                receipt = _read_regular_file_fd(
                    marker_descriptor,
                    "retired",
                    error_code="target_worktree_admin_marker_conflict",
                )
                if receipt != f"{os.path.abspath(f'{target}/.git')}\n".encode():
                    raise RuntimeError(
                        "target_worktree_admin_marker_conflict"
                    )
                os.unlink("retired", dir_fd=marker_descriptor)
                os.fsync(marker_descriptor)
        finally:
            os.close(marker_descriptor)
        os.rmdir(marker_name, dir_fd=common_descriptor)
        os.fsync(common_descriptor)
    except OSError as exc:
        raise RuntimeError("target_worktree_unregister_failed") from exc
    finally:
        os.close(common_descriptor)


def _remove_target_worktree(
    repository_root: Path,
    run_root: Path,
    *,
    name: str,
    run_command: Any = subprocess.run,
    expected_root: Path | None = None,
    expected_root_identity: tuple[int, int] | None = None,
    confinement_error: str = "target_worktree_invalid",
    allow_unregistered_owned: bool = False,
) -> None:
    if name not in {"control", "candidate"}:
        raise RuntimeError("target_worktree_invalid")
    root, root_identity = _strict_owned_directory(
        run_root, parent=None, error_code=confinement_error
    )
    if (
        expected_root is not None
        and (
            root != expected_root
            or root_identity != expected_root_identity
        )
    ):
        raise RuntimeError(confinement_error)
    target_path = root / name
    quarantine_name = f".{name}-cleanup"
    quarantine_path = root / quarantine_name
    target_present = target_path.exists() or target_path.is_symlink()
    quarantine_present = (
        quarantine_path.exists() or quarantine_path.is_symlink()
    )
    if target_present and quarantine_present:
        if not (target_path.exists() or target_path.is_symlink()) or not (
            quarantine_path.exists() or quarantine_path.is_symlink()
        ):
            raise RuntimeError(
                f"target_worktree_unregister_failed:{name}"
            )
        raise RuntimeError(confinement_error)
    owned_name: str | None = None
    owned_identity: tuple[int, int] | None = None
    if target_present:
        try:
            _target, owned_identity = _strict_owned_directory(
                target_path, parent=root, error_code=confinement_error
            )
        except RuntimeError as exc:
            if not target_path.exists() and not target_path.is_symlink():
                raise RuntimeError(
                    f"target_worktree_unregister_failed:{name}"
                ) from exc
            raise
        owned_name = name
    elif quarantine_present:
        try:
            _quarantine, owned_identity = _strict_owned_directory(
                quarantine_path, parent=root, error_code=confinement_error
            )
        except RuntimeError as exc:
            if not quarantine_path.exists() and not quarantine_path.is_symlink():
                raise RuntimeError(
                    f"target_worktree_unregister_failed:{name}"
                ) from exc
            raise
        owned_name = quarantine_name
    target_text = str(target_path)
    try:
        registrations = _worktree_registrations(
            repository_root, run_command=run_command
        )
    except RuntimeError as registration_failure:
        raise BaseExceptionGroup(
            "target_worktree_cleanup_failed",
            [
                RuntimeError(f"target_worktree_remove_failed:{name}"),
                registration_failure,
            ],
        ) from None
    registered = target_text in registrations
    if not registered and owned_name is None:
        return
    if not registered and owned_name == name:
        if not allow_unregistered_owned:
            raise RuntimeError(f"target_worktree_remove_failed:{name}")
        root_descriptor: int | None = None
        try:
            root_descriptor = os.open(root, _directory_open_flags())
            owned_descriptor = os.open(
                name, _directory_open_flags(), dir_fd=root_descriptor
            )
        except OSError as exc:
            if root_descriptor is not None:
                os.close(root_descriptor)
            raise RuntimeError(confinement_error) from exc
        try:
            observed_root = os.fstat(root_descriptor)
            observed_owned = os.fstat(owned_descriptor)
            if (
                (observed_root.st_dev, observed_root.st_ino) != root_identity
                or (observed_owned.st_dev, observed_owned.st_ino)
                != owned_identity
            ):
                raise RuntimeError(confinement_error)
            os.rename(
                name,
                quarantine_name,
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
            )
            os.fsync(root_descriptor)
            _remove_directory_contents_fd(owned_descriptor)
            os.rmdir(quarantine_name, dir_fd=root_descriptor)
            os.fsync(root_descriptor)
        finally:
            os.close(owned_descriptor)
            os.close(root_descriptor)
        return
    try:
        common = _git_common_directory(
            repository_root, run_command=run_command
        )
        marker_state = _admin_marker_state(common, target_text)
    except RuntimeError as exc:
        if str(exc) == "target_worktree_admin_marker_conflict":
            raise RuntimeError(
                f"target_worktree_admin_marker_conflict:{name}"
            ) from exc
        raise RuntimeError(f"target_worktree_unregister_failed:{name}") from exc
    if not registered:
        if marker_state == "empty" and owned_name != quarantine_name:
            raise RuntimeError(
                f"target_worktree_admin_marker_conflict:{name}"
            )
        if marker_state == "absent" and owned_name is None:
            return
    elif marker_state == "owned":
        raise RuntimeError(f"target_worktree_admin_marker_conflict:{name}")
    admin_name: str | None = None
    admin_descriptor: int | None = None
    if registered:
        try:
            admin_name, admin_descriptor = _find_worktree_admin(
                common, target_text
            )
        except RuntimeError as exc:
            try:
                current_marker = _admin_marker_state(common, target_text)
                current_registrations = _worktree_registrations(
                    repository_root, run_command=run_command
                )
            except RuntimeError as current_exc:
                if str(current_exc) == "target_worktree_admin_marker_conflict":
                    raise RuntimeError(
                        f"target_worktree_admin_marker_conflict:{name}"
                    ) from current_exc
                raise RuntimeError(
                    f"target_worktree_unregister_failed:{name}"
                ) from current_exc
            if (
                current_marker != "absent"
                or target_text not in current_registrations
            ):
                raise RuntimeError(
                    f"target_worktree_unregister_failed:{name}"
                ) from exc
            raise
    try:
        root_descriptor = os.open(root, _directory_open_flags())
    except OSError as exc:
        if admin_descriptor is not None:
            os.close(admin_descriptor)
        raise RuntimeError(confinement_error) from exc
    owned_descriptor: int | None = None
    claimed_admin_descriptor: int | None = None
    try:
        observed_root = os.fstat(root_descriptor)
        if (observed_root.st_dev, observed_root.st_ino) != root_identity:
            raise RuntimeError(confinement_error)
        if owned_name is not None:
            try:
                owned_descriptor = os.open(
                    owned_name,
                    _directory_open_flags(),
                    dir_fd=root_descriptor,
                )
            except OSError as exc:
                owned_path = root / owned_name
                if not owned_path.exists() and not owned_path.is_symlink():
                    raise RuntimeError(
                        f"target_worktree_unregister_failed:{name}"
                    ) from exc
                raise RuntimeError(confinement_error) from exc
            observed_owned = os.fstat(owned_descriptor)
            if (observed_owned.st_dev, observed_owned.st_ino) != owned_identity:
                raise RuntimeError(confinement_error)
            if registered or marker_state in {"owned", "retiring"}:
                backlink_name = _worktree_backlink_admin_name(
                    owned_descriptor, common
                )
                if registered and backlink_name != admin_name:
                    raise RuntimeError("target_worktree_admin_invalid")
        if owned_name == name:
            try:
                os.rename(
                    name,
                    quarantine_name,
                    src_dir_fd=root_descriptor,
                    dst_dir_fd=root_descriptor,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"target_worktree_unregister_failed:{name}"
                ) from exc
            os.fsync(root_descriptor)
        try:
            if registered:
                assert admin_name is not None
                assert admin_descriptor is not None
                claimed_admin_descriptor = _move_worktree_admin_to_marker(
                    common,
                    target_text,
                    admin_name,
                    admin_descriptor,
                )
            after = _worktree_registrations(
                repository_root, run_command=run_command
            )
            expected = registrations - ({target_text} if registered else set())
            if after != expected:
                raise RuntimeError("target_worktree_unregister_failed")
            if registered or marker_state in {"owned", "retiring", "empty"}:
                _delete_worktree_admin_marker(
                    common,
                    target_text,
                    claimed_admin_descriptor=claimed_admin_descriptor,
                )
        except RuntimeError as exc:
            if str(exc) == "target_worktree_admin_marker_conflict":
                raise RuntimeError(
                    f"target_worktree_admin_marker_conflict:{name}"
                ) from exc
            if str(exc) == "target_worktree_admin_identity_changed":
                raise RuntimeError(
                    f"target_worktree_admin_identity_changed:{name}"
                ) from exc
            raise RuntimeError(
                f"target_worktree_unregister_failed:{name}"
            ) from exc
        if owned_descriptor is not None:
            _remove_directory_contents_fd(owned_descriptor)
            os.close(owned_descriptor)
            owned_descriptor = None
            try:
                os.rmdir(quarantine_name, dir_fd=root_descriptor)
            except FileNotFoundError:
                pass
            os.fsync(root_descriptor)
    finally:
        if owned_descriptor is not None:
            os.close(owned_descriptor)
        os.close(root_descriptor)
        if admin_descriptor is not None:
            os.close(admin_descriptor)
        if claimed_admin_descriptor is not None:
            os.close(claimed_admin_descriptor)


def _remove_target_worktrees(
    repository_root: Path,
    run_root: Path,
    *,
    names: Sequence[str],
    run_command: Any = subprocess.run,
    expected_root: Path | None = None,
    expected_root_identity: tuple[int, int] | None = None,
    confinement_error: str = "target_worktree_invalid",
) -> None:
    """Attempt every owned worktree removal before reporting cleanup failure."""
    failures: list[BaseException] = []
    for name in names:
        try:
            _remove_target_worktree(
                repository_root,
                run_root,
                name=name,
                run_command=run_command,
                expected_root=expected_root,
                expected_root_identity=expected_root_identity,
                confinement_error=confinement_error,
            )
        except BaseException as exc:
            failures.append(exc)
    _raise_failures("target_worktree_cleanup_failed", failures)


def cleanup_attempt_worktrees(
    repository_root: Path,
    campaign_root: Path,
    attempt_id: str,
    *,
    run_command: Any = subprocess.run,
) -> None:
    """Remove only the two detached target worktrees owned by one attempt."""
    if not _ATTEMPT_ID.fullmatch(attempt_id) or campaign_root.is_symlink():
        raise RuntimeError("campaign_attempt_cleanup_refused")
    campaign, _campaign_identity = _strict_owned_directory(
        campaign_root,
        parent=None,
        error_code="campaign_attempt_cleanup_refused",
    )
    attempts_root = campaign_root / "attempts"
    attempts, _attempts_identity = _strict_owned_directory(
        attempts_root,
        parent=campaign,
        error_code="campaign_attempt_cleanup_refused",
    )
    attempt_root = attempts_root / attempt_id
    attempt, attempt_identity = _strict_owned_directory(
        attempt_root,
        parent=attempts,
        error_code="campaign_attempt_cleanup_refused",
    )
    _remove_target_worktrees(
        repository_root,
        attempt_root,
        names=("control", "candidate"),
        run_command=run_command,
        expected_root=attempt,
        expected_root_identity=attempt_identity,
        confinement_error="campaign_attempt_cleanup_refused",
    )


def prepare_output_root(path: Path) -> None:
    """Create an output root or preserve its sole documentation README."""
    if path.exists():
        existing = list(path.iterdir())
        if any(item.name != "README.md" or not item.is_file() for item in existing):
            raise RuntimeError("output_root_not_empty")
    path.mkdir(parents=True, exist_ok=True)


def remove_successful_sample_root(run_root: Path, sample_root: Path) -> None:
    """Remove only a completed child root directly below ``run_root/samples``."""
    samples_root = run_root.resolve() / "samples"
    target = sample_root.resolve()
    if target.parent != samples_root or not target.is_dir():
        raise RuntimeError("sample_cleanup_refused")
    shutil.rmtree(target)


def run_parent_mode(args: argparse.Namespace) -> int:
    """Own revisions, child lifecycles, validation, and retained smoke evidence."""
    repository_root = Path(__file__).resolve().parents[2]
    harness = current_harness_identity(repository_root)
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
    initial_listener = listener_identity(args.endpoint)
    control_root = prepare_target_worktree(
        repository_root,
        run_root,
        name="control",
        revision=revisions["control"],
    )
    candidate_root: Path | None = None
    rows: list[dict[str, Any]] = []
    runner = Path(__file__).resolve()
    statistics_module: Any = sys.modules[__name__]
    try:
        candidate_root = prepare_target_worktree(
            repository_root,
            run_root,
            name="candidate",
            revision=revisions["candidate"],
        )
        if revisions["candidate"] == CANDIDATE_SHA:
            load_original_protocol(candidate_root, repository_root)
            statistics_module = load_original_runner(candidate_root)
        for index, plan in enumerate(sample_schedule(args.iterations)):
            verify_listener_identity(
                args.endpoint,
                initial_listener["fingerprint_sha256"],
            )
            sample_id = f"{plan.phase}-{plan.iteration}-{plan.arm}"
            sample_root = run_root / "samples" / f"{index:03d}-{sample_id}"
            target_root = control_root if plan.arm == "control" else candidate_root
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
            verify_listener_identity(
                args.endpoint,
                initial_listener["fingerprint_sha256"],
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
            errors = statistics_module.validate_sample(last)
            if errors or privacy_violations(last):
                raise RuntimeError("parent_sample_validation_failed")
            rows.append(last)
            remove_successful_sample_root(run_root, sample_root)
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
        primary_failure = sys.exception()
        cleanup_names = (
            ("candidate", "control")
            if candidate_root is not None
            else ("control",)
        )
        try:
            _remove_target_worktrees(
                repository_root,
                run_root,
                names=cleanup_names,
            )
        except BaseException as cleanup_failure:
            if primary_failure is not None:
                raise BaseExceptionGroup(
                    "parent_run_and_cleanup_failed",
                    [primary_failure, cleanup_failure],
                ) from None
            raise

    samples_root = run_root / "samples"
    if samples_root.is_dir():
        samples_root.rmdir()

    validation_errors = statistics_module.validate_run(
        rows, expected_iterations=args.iterations
    )
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
        "harness": harness,
        "revisions": revisions,
        "model": args.model,
        "provider": "llama_cpp",
        "temperature": REQUEST_SETTINGS["temperature"],
        "max_tokens": REQUEST_SETTINGS["max_tokens"],
        "reasoning_effort": REQUEST_SETTINGS["reasoning_effort"],
        "stream_options": {
            "include_usage": REQUEST_SETTINGS["include_usage"]
        },
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
        "listener_identity": initial_listener,
    }
    summary = (
        statistics_module.build_summary(rows)
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
