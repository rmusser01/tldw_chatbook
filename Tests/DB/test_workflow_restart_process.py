"""Kill the actual producer and open its SQLite history in a separate interpreter."""

import pytest

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize(
    "boundary",
    [
        "prepared",
        "accepted",
        "quiesced",
        "quiesced_result",
        "thread_start",
        "effect",
        "result_commit",
        "during_result_commit",
        "settled",
    ],
)
def test_fresh_process_recovery_never_replays_native_work(tmp_path, boundary):
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    cwd = Path(__file__).resolve().parents[2]
    profile = tmp_path / "profile"
    profile.mkdir()
    env = os.environ.copy()
    env.update(
        HOME=str(profile),
        USERPROFILE=str(profile),
        TLDW_CONFIG_PATH=str(profile / "config.toml"),
        TLDW_TEST_CONFIG_ROOT=str(profile),
        XDG_DATA_HOME=str(profile / "data"),
        XDG_CONFIG_HOME=str(profile / "config"),
        PYTHONPATH=str(cwd),
    )
    env.pop("TLDW_TEST_CONFIG_ROOT_OWNER", None)
    command = [sys.executable, "-m", "Tests.DB.workflow_restart_child", str(tmp_path)]
    with (tmp_path / "child.log").open("w") as log:
        child = subprocess.Popen(
            [*command, "produce", boundary], cwd=cwd, env=env, stdout=log, stderr=log
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
            ).read_text()[-7000:]
            counter = tmp_path / "provider-operations"
            before = counter.read_bytes() if counter.exists() else b""
            stamp = json.loads((tmp_path / "boundary.json").read_text())
            assert stamp["pid"] == child.pid and stamp["boundary"] == boundary
            child.kill()
            killed = child.wait(5)
            assert killed != 0
            if os.name == "posix":
                import signal

                assert killed == -signal.SIGKILL
            restart = subprocess.run(
                [*command, "restart", boundary],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=log,
                timeout=20,
                check=False,
            )
            assert restart.returncode == 0, (tmp_path / "child.log").read_text()[-7000:]
            assert (counter.read_bytes() if counter.exists() else b"") == before
            data = json.loads((tmp_path / "restarted.json").read_text())
            assert data["pid"] != child.pid
            assert data["runs_before_audit"] == data["runs_after_audit"]
            if boundary == "during_result_commit":
                assert stamp["full_sync"] == 2 and stamp["uncommitted_results"] == 1
                assert data["results"] == 0 and data["checkpoints"] == 0
            if data["status"] == "recovery_required":
                assert data["unresolved_charge"] >= stamp["wall_limit"]
            (tmp_path / "process.json").write_text(
                json.dumps(
                    {
                        "producer_pid": child.pid,
                        "producer_wait": killed,
                        "restart_pid": data["pid"],
                        "restart_wait": restart.returncode,
                        "producer_pgid": stamp["pgid"],
                        "restart_pgid": data["pgid"],
                        "producer_argv": child.args,
                        "restart_argv": restart.args,
                        "boundary": boundary,
                        "recovered": data,
                        "provider_operations": before.count(b"call\n"),
                    },
                    indent=2,
                )
            )
            assert data["status"] == (
                "awaiting_resume"
                if boundary in {"prepared", "quiesced", "quiesced_result", "settled"}
                else "recovery_required"
            )
            if boundary in {
                "effect",
                "during_result_commit",
                "result_commit",
                "quiesced_result",
                "settled",
            }:
                assert (tmp_path / "0" / "source.py").read_text() == "actual effect"
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(5)
