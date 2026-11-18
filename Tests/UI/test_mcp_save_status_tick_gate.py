"""The MCP save-status timer stats the config file on a coarse cadence only."""

import pytest

from tldw_chatbook.UI.MCP_Modules import mcp_workbench
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench

pytestmark = pytest.mark.bootstrap_profile


def _bare() -> MCPWorkbench:
    workbench = MCPWorkbench.__new__(MCPWorkbench)
    workbench._save_status_tick = 0
    workbench._save_file_identity_cache = None
    return workbench


def test_file_identity_stat_reads_once_until_forced(monkeypatch):
    workbench = _bare()
    reads = []
    monkeypatch.setattr(
        mcp_workbench,
        "current_config_identity",
        lambda: (reads.append("id"), (len(reads),))[1],
    )
    monkeypatch.setattr(
        mcp_workbench,
        "config_file_revision",
        lambda path: (reads.append("stat"), (len(reads), 0, 0, 0)),
    )

    first = workbench._save_status_file_identity()
    cached = workbench._save_status_file_identity()
    cached_again = workbench._save_status_file_identity()
    assert cached == first and cached_again == first
    assert len(reads) == 2, "cached reads must not stat the config file"

    forced = workbench._save_status_file_identity(refresh=True)
    assert len(reads) == 4, "a forced refresh must stat again"
    assert forced != first


def test_timer_requests_a_fresh_stat_only_every_eighth_tick():
    workbench = _bare()
    flags = []
    workbench._sync_root_save_status = lambda **kw: flags.append(kw["file_identity_refresh"])
    workbench._sync_master_save_status = lambda **kw: None

    for _ in range(9):
        workbench._sync_local_config_save_status()

    refresh_ticks = [index for index, flag in enumerate(flags) if flag]
    assert refresh_ticks == [0, 8], "fresh file identity only on ticks 1 and 9 (~2s)"


def test_direct_post_save_calls_keep_fresh_stats():
    import inspect

    for method in (
        MCPWorkbench._sync_root_save_status,
        MCPWorkbench._sync_master_save_status,
    ):
        default = inspect.signature(method).parameters["file_identity_refresh"].default
        assert default is True, f"{method.__name__} must default to a fresh stat"
