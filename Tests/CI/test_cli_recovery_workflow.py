"""CLI regressions must run on the existing bounded Linux PR matrix."""

from pathlib import Path


def test_cli_regressions_are_selected_by_the_pr_recovery_workflow():
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/tool-recovery.yml"
    ).read_text(encoding="utf-8")
    assert "python-version: ['3.12', '3.14']" in workflow
    assert "timeout-minutes: 15" in workflow
    for selected in (
        "Tests/Agents/test_raw_shell_recovery.py",
        "Tests/Tools/test_virtual_cli_listing_recovery.py",
        "Tests/Tools/test_workspace_atomic_writes.py",
        "Tests/Tools/test_raw_cli_executor_process.py",
        "Tests/Tools/test_remote_executor_loopback.py::test_fs_list_long_roundtrip_retains_exclusions_and_plain_output",
    ):
        assert selected in workflow, f"PR workflow does not select {selected}"
    assert "cli-agent-results.xml" in workflow
    assert "cli-worker-results.xml" in workflow
    assert "tldw_chatbook/Tools/**" in workflow
    assert "Tests/Tools/**" in workflow
