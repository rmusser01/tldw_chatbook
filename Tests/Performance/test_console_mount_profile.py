"""Regression tests for the standalone Console mount profiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.Performance.run_console_mount_profile import (
    _outgoing_detached_elapsed_ms,
)


def test_outgoing_detached_elapsed_uses_unmount_completion_timestamp() -> None:
    assert (
        _outgoing_detached_elapsed_ms(
            {"completed_at": 12.5},
            started=10.0,
        )
        == 2500.0
    )


def test_outgoing_detached_elapsed_rejects_a_missing_unmount_observation() -> None:
    with pytest.raises(RuntimeError, match="outgoing unmount was not observed"):
        _outgoing_detached_elapsed_ms({}, started=10.0)


@pytest.mark.integration
def test_profiler_measures_a_warm_visit_on_the_reusable_console_route(
    tmp_path: Path,
) -> None:
    """The runner completes against today's reusable route (TASK-33260).

    It rotted silently once: route reuse (TASK-31520) meant ``on_mount`` no
    longer fired per visit and every run died "profile condition did not
    settle". One real warm iteration, in a fresh interpreter because the
    runner points ``os.environ`` at its own scratch profile.

    Args:
        tmp_path: pytest fixture; holds the runner's JSON report.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "profile.json"
    env = {**os.environ, "PYTHONPATH": str(repo_root), "TLDW_TEST_MODE": "1"}
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "Tests/Performance/run_console_mount_profile.py"),
            "--iterations",
            "1",
            "--output",
            str(output),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    summary = json.loads(output.read_text())["summary"]["warm_resume"]
    assert summary["iterations"] == 1
    assert summary["full_ready_ms"]["median"] > 0
    assert summary["widget_counts"]["composer"]["median"] > 0


@pytest.mark.integration
def test_composition_phases_measure_cold_mounts_of_their_own_variants(
    tmp_path: Path,
) -> None:
    """``--phase production`` evicts the reusable Console before each visit.

    Its variants patch ``compose``, which runs only on a cold mount. If the
    eviction stopped working, every sample would be a warm resume of one
    instance and the eager and deferred Context rails would mount the same
    widgets. Here the deferred rail must mount far fewer than the eager one.

    Args:
        tmp_path: pytest fixture; holds the runner's JSON report.
    """
    import json
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "profile.json"
    env = {**os.environ, "PYTHONPATH": str(repo_root), "TLDW_TEST_MODE": "1"}
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "Tests/Performance/run_console_mount_profile.py"),
            "--iterations",
            "1",
            "--phase",
            "production",
            "--output",
            str(output),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    rails = {
        sample["variant"]: sample["first_interactive_widget_counts"]["context_rail"]
        for sample in json.loads(output.read_text())["samples"]
    }
    assert set(rails) == {"eager", "deferred"}
    assert rails["deferred"] * 10 < rails["eager"], rails
