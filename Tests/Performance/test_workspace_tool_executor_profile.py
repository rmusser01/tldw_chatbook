"""Deterministic contracts for the one-shot workspace executor profiler."""

from __future__ import annotations

import json
import math
from pathlib import Path

from Tests.Performance import run_workspace_tool_executor_profile as profile


def test_nearest_rank_p95_uses_one_based_ceiling() -> None:
    assert profile.nearest_rank_p95(list(range(1, 31))) == 29
    assert profile.nearest_rank_p95([7.0]) == 7.0


def test_profile_uses_exact_operations_and_fake_clock_sample_injection(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, int]] = []
    ticks = iter(float(value) / 1000.0 for value in range(0, 240, 2))

    def fake_sample(
        _workspace: Path, operation: str, mode: str, sample_index: int
    ) -> None:
        calls.append((operation, mode, sample_index))

    report = profile.build_profile(
        tmp_path,
        samples=2,
        clock=lambda: next(ticks),
        sample_runner=fake_sample,
        metadata={
            "head_commit": "a" * 40,
            "platform": "test-platform",
            "python": "3.12.test",
        },
    )

    assert tuple(report["operations"]) == (
        "stat",
        "read",
        "write",
        "list",
        "git_status",
        "git_diff",
    )
    assert calls == [
        (operation, mode, sample_index)
        for operation in profile.OPERATIONS
        for sample_index in range(2)
        for mode in ("direct", "one_shot")
    ]
    for metrics in report["operations"].values():
        assert metrics == {
            "direct_ms": {"median": 2.0, "p95": 2.0},
            "one_shot_ms": {"median": 2.0, "p95": 2.0},
            "startup_overhead_ms": {"median": 0.0, "p95": 0.0},
        }


def test_profile_json_is_finite_content_free_metadata_without_timing_gate(
    tmp_path: Path,
) -> None:
    private_marker = "private-root-and-content-marker"

    def fake_sample(
        _workspace: Path, _operation: str, _mode: str, _sample_index: int
    ) -> None:
        return None

    ticks = iter(float(value) for value in range(1000))
    report = profile.build_profile(
        tmp_path / private_marker,
        samples=1,
        clock=lambda: next(ticks),
        sample_runner=fake_sample,
        metadata={
            "head_commit": "b" * 40,
            "platform": "bounded-platform",
            "python": "3.12.0",
        },
    )
    output = tmp_path / "profile.json"
    profile.write_profile(report, output)
    raw = output.read_text(encoding="utf-8")
    decoded = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    assert set(decoded) == {
        "schema_version",
        "head_commit",
        "platform",
        "python",
        "samples",
        "operations",
    }
    assert private_marker not in raw
    assert "path" not in decoded
    assert "content" not in decoded
    assert "threshold" not in decoded
    assert "passed" not in decoded
    assert "qualification" not in decoded
    for metrics in decoded["operations"].values():
        assert set(metrics) == {"direct_ms", "one_shot_ms", "startup_overhead_ms"}
        for summary in metrics.values():
            assert set(summary) == {"median", "p95"}
            assert all(math.isfinite(value) for value in summary.values())


def test_invalid_sample_count_is_refused() -> None:
    for value in (0, -1, True):
        try:
            profile.build_profile(
                Path("."),
                samples=value,
                sample_runner=lambda *_args: None,
            )
        except ValueError as error:
            assert str(error) == "samples must be a positive integer"
        else:
            raise AssertionError(f"accepted invalid sample count: {value!r}")

