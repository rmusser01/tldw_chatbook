"""BuiltinToolProvider binds the run's workspace around tool execution."""

from __future__ import annotations

from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
from tldw_chatbook.Tools import workspace_file_roots as wfr


class _ProbeTool:
    name = "probe_workspace"
    description = "records the bound run workspace"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return {"workspace": wfr.current_run_workspace_id()}


class _OpenGate:
    def check(self, tool):
        return None


def test_invoke_binds_and_clears_run_workspace() -> None:
    provider = BuiltinToolProvider(gate=_OpenGate(), workspace_id="ws-a")
    provider._tools["probe_workspace"] = _ProbeTool()

    result = provider.invoke("builtin:probe_workspace", {})

    assert result.ok, result.error
    assert '"workspace": "ws-a"' in result.content
    assert wfr.current_run_workspace_id() is None  # cleared after invoke


def test_invoke_without_workspace_leaves_context_unset() -> None:
    provider = BuiltinToolProvider(gate=_OpenGate())
    provider._tools["probe_workspace"] = _ProbeTool()

    result = provider.invoke("builtin:probe_workspace", {})

    assert result.ok, result.error
    assert '"workspace": null' in result.content
