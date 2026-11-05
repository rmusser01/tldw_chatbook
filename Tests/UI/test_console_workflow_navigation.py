"""Workflow history belongs to the selected Chat and never launches on selection."""

import pytest

from Tests.UI.test_console_workflow_command import run_workflow_ui

pytestmark = pytest.mark.bootstrap_profile


def test_narrow_history_reopens_same_run_after_detach(tmp_path):
    run_workflow_ui(tmp_path, "navigation")


def test_detach_preserves_planning_and_background_link_identifies_owner(tmp_path):
    run_workflow_ui(tmp_path, "detach")


def test_detach_waits_for_original_finite_workspace_read(tmp_path):
    run_workflow_ui(tmp_path, "read-detach")


def test_saved_history_paginates_without_starting_a_workflow(tmp_path):
    run_workflow_ui(tmp_path, "history-page")
