"""Changed Workspace/Kernel sources retain the original direct probe route."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SOURCE_FALLBACK_SCRIPT = r"""
import sys
from pathlib import Path

from pytest import MonkeyPatch

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB import store_kernel
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Tools import workspace_file_roots as roots
from tldw_chatbook.UI.Console_Modules import run_log_probe_scope as scope
from tldw_chatbook.Backup_Recovery import participants

route, outcome = sys.argv[1:]
assert outcome == "source"
runs = AgentRunsDB(Path.home() / "runs.sqlite", client_id="probe-source")
try:
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=None, provider_gateway=None
    )
    registry = roots._default_registry_factory()
    workspace = registry.db
    try:
        assert scope._stock_default_probe(bridge) is True
        with MonkeyPatch.context() as patch:
            def changed(self):
                raise AssertionError("changed database callback was invoked")

            if route == "inherited_code":
                original = store_kernel.SQLiteStoreKernel._get_connection.__wrapped__
                patch.setattr(original, "__code__", changed.__code__)
            elif route == "inherited_defaults":
                original = store_kernel.SQLiteStoreKernel._get_connection.__wrapped__
                patch.setattr(original, "__defaults__", (object(),))
            elif route == "kernel_global":
                patch.setattr(store_kernel, "_core_access", changed)
            elif route == "class_hook":
                patch.setattr(WorkspaceDB, "_configure_connection", changed)
            elif route == "instance_hook":
                patch.setattr(workspace, "_open_connection", changed.__get__(workspace))
            elif route == "inherited_alias":
                patch.setattr(WorkspaceDB, "_held_connection", WorkspaceDB._held_connection)
            else:
                assert route == "inherited_false_shadow"
                patch.setattr(WorkspaceDB, "is_memory_db", False)

            def unexpected_finite(read):
                raise AssertionError("changed source entered the finite worker")

            patch.setattr(participants, "run_finite_local_worker", unexpected_finite)
            value = object()
            original_error = LookupError("original direct callback error")
            calls = []

            def read():
                calls.append("read")
                return value

            def fail():
                calls.append("error")
                raise original_error

            assert scope._stock_default_probe(bridge) is False
            assert scope.finite_stock_probe(bridge, read) is value
            try:
                scope.finite_stock_probe(bridge, fail)
            except LookupError as error:
                assert error is original_error
            else:
                raise AssertionError("original direct error was swallowed")
            assert calls == ["read", "error"]
    finally:
        workspace.close()
finally:
    runs.close()
print("retired and reopened")
"""


@pytest.mark.parametrize(
    "route",
    [
        "inherited_code",
        "inherited_defaults",
        "kernel_global",
        "class_hook",
        "instance_hook",
        "inherited_alias",
        "inherited_false_shadow",
    ],
)
def test_changed_workspace_probe_source_keeps_direct_callback(tmp_path, route):
    _run(tmp_path, route, "source", script=_SOURCE_FALLBACK_SCRIPT)
