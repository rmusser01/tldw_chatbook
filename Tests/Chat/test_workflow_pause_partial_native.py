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
