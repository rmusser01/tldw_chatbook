"""MCP preimport must defer its existing visual panels until real use."""

from pathlib import Path

from Tests.Packaging.test_chunking_import_closure import _run_isolated_python


def test_mcp_route_defers_visual_panels(tmp_path: Path) -> None:
    result = _run_isolated_python(
        tmp_path,
        """
import sys
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
assert MCPScreen.__module__ == "tldw_chatbook.UI.Screens.mcp_screen"
for name in ("mcp_inspector", "mcp_audit_mode", "mcp_profile_form", "mcp_rail",
             "mcp_server_mutations", "mcp_servers_mode", "mcp_tools_mode", "mcp_schema_form"):
    assert "tldw_chatbook.UI.MCP_Modules." + name not in sys.modules, name
# The permission DTO/gate owner stays exactly where it was.
assert "tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode" in sys.modules
print("MCP_DEFERRED_OK")
""",
    )
    assert result.returncode == 0, (result.stdout, result.stderr[-4000:])
    assert "MCP_DEFERRED_OK" in result.stdout


def test_mcp_lazy_exports_preserve_types_and_canonical_aliases(tmp_path: Path) -> None:
    result = _run_isolated_python(
        tmp_path,
        """
import importlib
import inspect
import typing
from tldw_chatbook.UI.MCP_Modules import mcp_workbench as workbench
assert workbench._workbench is workbench
# Resolve annotations before warming the public aliases explicitly.
hints = typing.get_type_hints(workbench.MCPWorkbench.on_mcp_inspector_hub_action_requested)
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
assert hints["event"] is MCPInspector.HubActionRequested
for _, method in inspect.getmembers(workbench.MCPWorkbench, inspect.isfunction):
    if method.__module__ == workbench.__name__:
        typing.get_type_hints(method)
expected = {
    "mcp_audit_mode": ["MCPAuditMode"],
    "mcp_inspector": ["MCPInspector", "_safe_diagnostic_message", "_safe_exception_text", "_safe_tool_test_text"],
    "mcp_profile_form": ["MCPImportPanel", "MCPProfileForm", "_import_summary", "_import_severity"],
    "mcp_rail": ["MCPRail", "_target_id_from_server_key"],
    "mcp_server_mutations": ["MCPServerMutationsPanel"],
    "mcp_servers_mode": ["MCPServersMode"],
    "mcp_tools_mode": ["MCPToolsMode"],
}
star = {}
exec("from tldw_chatbook.UI.MCP_Modules.mcp_workbench import *", star)
for owner_name, names in expected.items():
    owner = importlib.import_module("." + owner_name, workbench.__package__)
    for name in names:
        assert name in dir(workbench)
        assert getattr(workbench, name) is getattr(owner, name)
        direct = {}
        exec("from tldw_chatbook.UI.MCP_Modules.mcp_workbench import " + name, direct)
        assert direct[name] is getattr(owner, name)
        if not name.startswith("_"):
            assert name in workbench.__all__
            assert star[name] is getattr(owner, name)
try:
    workbench.no_such_export
except AttributeError:
    pass
else:
    raise AssertionError("unknown exports must raise AttributeError")
print("MCP_EXPORTS_OK")
""",
    )
    assert result.returncode == 0, (result.stdout, result.stderr[-4000:])
    assert "MCP_EXPORTS_OK" in result.stdout
