"""Mounted review controls operate on the displayed immutable workflow."""

import pytest

from Tests.UI.test_console_workflow_command import run_workflow_ui

pytestmark = pytest.mark.bootstrap_profile


def test_approve_then_accept_real_phase_from_its_detail(tmp_path):
    run_workflow_ui(tmp_path, "controls")


@pytest.mark.parametrize(
    "mode",
    [
        "pause",
        "stale",
        "revise",
        "preset",
        "selection",
        "malformed",
        "pending",
        "metadata",
        "capture",
        "two-phase",
    ],
)
def test_native_workflow_control_boundaries(tmp_path, mode):
    run_workflow_ui(tmp_path, mode)
