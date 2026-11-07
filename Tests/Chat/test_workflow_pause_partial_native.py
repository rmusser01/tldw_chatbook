"""Pause and partial continuation through real native owners and storage."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize(
    "mode",
    [
        "bound-pause",
        "prebind-pause",
        "before-commit",
        "stop",
        "passive-pause",
        "live-limit",
        "remaining",
        "remaining-cancel",
    ],
)
def test_original_committed_child_control_and_required_wave_return(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_pause_partial_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize(
    "mode", ["remaining-prepared-stale", "remaining-shutdown", "unregistered-control"]
)
def test_remaining_acceptance_review_boundaries(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "success",
        script=Path(__file__).with_name("workflow_pause_partial_probe.py").read_text(),
        timeout=60,
    )


def test_remaining_original_suppressed_workspace_close_retains_custody(tmp_path):
    import json
    import os
    import subprocess
    import sys

    for name in ("home", "config", "data"):
        (tmp_path / name).mkdir(mode=0o700)
    environment = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config/config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    log = tmp_path / "remaining-close-output.log"
    with log.open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-c",
                Path(__file__).with_name("workflow_pause_partial_probe.py").read_text(),
                "remaining-close-fault",
                "retained",
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    stdout = log.read_text()
    assert result.returncode == 0, stdout[-4000:]
    line = next(
        line
        for line in stdout.splitlines()
        if line.startswith("E_FIX1_OWNED_CLOSE_FACTS ")
    )
    print(line)
    facts = json.loads(line.removeprefix("E_FIX1_OWNED_CLOSE_FACTS "))
    assert facts["fault_count"] == facts["cells"] == 1 and facts["fault_attempts"] > 0
    assert all(
        facts[name]
        for name in (
            "live_handle",
            "cached_original",
            "retained_lease",
            "retained_participant",
        )
    )
    assert not facts["delivered"], (
        "unclosed original storage delivered execution custody"
    )
    assert all(
        facts[name]
        for name in (
            "owner_retained",
            "cell_retained",
            "claim_retained",
            "phase_retained",
        )
    )
    assert facts["calls"] == {"left": 3}
