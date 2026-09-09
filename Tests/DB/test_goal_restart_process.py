"""Kill an owned process at real native boundaries; restart never redispatches."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("boundary", ["accepted", "checkpoint"])
def test_actual_process_restart_preserves_limits_and_has_no_duplicate_operations(
    tmp_path, boundary
):
    cwd = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["TLDW_AGENTS_GOAL_RUNS_ENABLED"] = "true"
    env["TLDW_AGENTS_AUTOWAKE_ENABLED"] = "false"
    env["PYTHONPATH"] = str(cwd)
    with (tmp_path / "child.log").open("w") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "Tests.DB.goal_restart_child",
                str(tmp_path),
                "produce",
                boundary,
            ],
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 20
            while (
                not (tmp_path / "boundary.json").exists()
                and child.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert (tmp_path / "boundary.json").exists(), (
                tmp_path / "child.log"
            ).read_text()[-8000:]
            before = (
                (tmp_path / "provider-operations").read_text(),
                (tmp_path / "tool-operations").read_text(),
            )
            assert before == ("call\ncall\n", "tool\n")
            child.kill()
            child.wait(5)
            restarted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "Tests.DB.goal_restart_child",
                    str(tmp_path),
                    "restart",
                    boundary,
                ],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=log,
                timeout=20,
                check=False,
            )
            assert restarted.returncode == 0, (tmp_path / "child.log").read_text()[
                -8000:
            ]
            assert (
                (tmp_path / "provider-operations").read_text(),
                (tmp_path / "tool-operations").read_text(),
            ) == before
            result = json.loads((tmp_path / "restarted.json").read_text())
            assert result["status"] == (
                "recovery_required" if boundary == "accepted" else "paused"
            )
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(5)
