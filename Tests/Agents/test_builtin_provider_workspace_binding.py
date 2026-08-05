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


def test_concurrent_providers_keep_distinct_workspace_bindings() -> None:
    """Spec §7: two overlapping runs resolve different roots (ContextVar
    isolation across interleaved invokes)."""
    import threading

    results: dict[str, str | None] = {}

    class _EchoTool:
        name = "probe_workspace"
        description = "echo bound workspace"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            import asyncio
            from tldw_chatbook.Tools import workspace_file_roots as wfr
            await asyncio.sleep(0.05)  # force overlap window
            return {"workspace": wfr.current_run_workspace_id()}

    def run(workspace_id: str) -> None:
        provider = BuiltinToolProvider(gate=_OpenGate(), workspace_id=workspace_id)
        provider._tools["probe_workspace"] = _EchoTool()
        result = provider.invoke("builtin:probe_workspace", {})
        results[workspace_id] = result.content

    threads = [
        threading.Thread(target=run, args=("ws-alpha",)),
        threading.Thread(target=run, args=("ws-beta",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert '"workspace": "ws-alpha"' in (results["ws-alpha"] or "")
    assert '"workspace": "ws-beta"' in (results["ws-beta"] or "")
