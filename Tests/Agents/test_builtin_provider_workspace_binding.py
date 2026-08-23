"""BuiltinToolProvider binds the run's workspace around tool execution."""

from __future__ import annotations

from contextlib import nullcontext

from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.Tools import workspace_file_roots as wfr
from tldw_chatbook.Tools.file_operation_tools import ReadFileTool


class _ProbeTool:
    name = "probe_workspace"
    description = "records the bound run workspace"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return {"workspace": wfr.current_run_workspace_id()}


class _OpenGate:
    def check(self, tool, run_id):
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


def test_builtin_provider_cannot_read_another_chat_scratch(
    tmp_path,
    monkeypatch,
) -> None:
    root_a = tmp_path / "chat-a"
    root_b = tmp_path / "chat-b"
    root_a.mkdir()
    root_b.mkdir()
    marker = root_a / "marker.txt"
    marker.write_text("chat-a", encoding="utf-8")
    monkeypatch.setattr(
        "tldw_chatbook.Tools.file_operation_tools._resolve_sandbox_config",
        lambda: str(tmp_path),
    )
    provider = BuiltinToolProvider(
        gate=_OpenGate(),
        workspace_id="workspace-default",
        sandbox_root=root_b,
        sandbox_lease=lambda: nullcontext(root_b),
    )
    provider._tools["read_file"] = ReadFileTool()

    result = provider.invoke(
        "builtin:read_file",
        {"file_path": str(marker)},
    )

    assert result.ok is False
    assert "outside" in str(result.error).lower()


def test_builtin_provider_rejects_file_access_after_scratch_close(tmp_path) -> None:
    manager = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = manager.snapshot("chat-a")
    marker = snapshot.root / "marker.txt"
    marker.write_text("chat-a", encoding="utf-8")
    provider = BuiltinToolProvider(
        gate=_OpenGate(),
        workspace_id="workspace-default",
        sandbox_root=snapshot.root,
        sandbox_lease=lambda: manager.lease(snapshot),
    )
    provider._tools["read_file"] = ReadFileTool()

    manager.close("chat-a")
    result = provider.invoke("builtin:read_file", {"file_path": str(marker)})

    assert result.ok is False
    assert manager.wait_for_cleanup(timeout_seconds=2.0)
