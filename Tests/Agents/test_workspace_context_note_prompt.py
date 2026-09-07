# Tests/Agents/test_workspace_context_note_prompt.py
"""The workspace-context note rides into the agent's system prompt.

A non-default workspace attaches a note (built by
``workspace_file_roots.workspace_context_note``) to ``AgentConfig``; it must
reach the model's system prompt on every turn -- on native AND fence-protocol
endpoints -- while the default-workspace empty string adds nothing.
"""

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _service(tmp_path, capture):
    db = AgentRunsDB(tmp_path / "runs.db")
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    return AgentService(db, registry, chat_call=capture)


def _run(service, note, api_endpoint):
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(
            model="m",
            system_prompt="BASE",
            budget=RunBudget(),
            workspace_context_note=note,
        ),
        api_endpoint=api_endpoint,
    )


def _capture(prompts):
    def capture(**kwargs):
        prompts.append(kwargs["messages_payload"][0]["content"])
        return {"choices": [{"message": {"content": "ok"}}]}

    return capture


def test_workspace_note_reaches_system_prompt_on_native_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    prompts: list[str] = []
    _run(_service(tmp_path, _capture(prompts)), "WS_NOTE_XYZ", "openai")
    assert prompts, "expected at least one model turn"
    assert all("WS_NOTE_XYZ" in p for p in prompts)
    assert all(p.startswith("BASE") for p in prompts)


def test_workspace_note_reaches_system_prompt_on_fence_protocol_endpoint(
    tmp_path, monkeypatch
):
    """The non-native branch reassigns ``system_content`` to splice the fence
    protocol -- the note must survive that, exactly as RUN_LOG_PROMPT_SECTION
    does (see test_run_log_prompt_integration's llama_cpp case)."""
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    prompts: list[str] = []
    _run(_service(tmp_path, _capture(prompts)), "WS_NOTE_XYZ", "llama_cpp")
    assert prompts, "expected at least one model turn"
    assert all("WS_NOTE_XYZ" in p for p in prompts)
    assert all(p.startswith("BASE") for p in prompts)


def test_empty_workspace_note_leaves_the_prompt_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    prompts: list[str] = []
    _run(_service(tmp_path, _capture(prompts)), "", "openai")
    assert prompts, "expected at least one model turn"
    assert all(p == "BASE" for p in prompts)
