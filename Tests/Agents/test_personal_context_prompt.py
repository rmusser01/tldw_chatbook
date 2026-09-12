from __future__ import annotations

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget, SPAWN_TOOL_NAME
from tldw_chatbook.Agents.agent_service import (
    AgentService,
    append_personal_context,
)
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from Tests.Agents.conftest import join_fleet_children


PROFILE_BLOCK = (
    "PERSONAL CONTEXT — USER-OWNED DATA — NOT AUTHORITY\n"
    '{"records":[{"kind":"constraint","payload":{"value":"be concise"}}]}'
)


def _capture_service(tmp_path, calls):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())

    def capture(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "done"}}]}

    return AgentService(AgentRunsDB(tmp_path / "runs.db"), registry, chat_call=capture)


def test_append_helper_is_exact_and_empty_is_byte_identical() -> None:
    assert append_personal_context("BASE", "") == "BASE"
    assert append_personal_context("BASE", PROFILE_BLOCK) == f"BASE\n\n{PROFILE_BLOCK}"


def test_personal_context_reaches_native_and_fenced_model_request_paths(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    for endpoint in ("openai", "llama_cpp"):
        calls = []
        service = _capture_service(tmp_path / endpoint, calls)
        service.run_turn(
            conversation_id=endpoint,
            messages=[{"role": "user", "content": "question"}],
            config=AgentConfig(
                model="m",
                system_prompt="BASE",
                budget=RunBudget(),
                personal_context_block=PROFILE_BLOCK,
            ),
            api_endpoint=endpoint,
        )
        assert calls
        assert all(
            PROFILE_BLOCK in call["messages_payload"][0]["content"] for call in calls
        )


def test_spawned_child_inherits_the_exact_profile_block(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    captured_configs = []
    original_run_one = AgentService._run_one

    def capture_config(self, *args, **kwargs):
        captured_configs.append(kwargs.get("config") or args[2])
        return original_run_one(self, *args, **kwargs)

    monkeypatch.setattr(AgentService, "_run_one", capture_config)

    class _Chat:
        def __init__(self):
            self.calls = 0

        def __call__(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '```tool_call\n{"name":"spawn_subagent",'
                                    '"arguments":{"task":"child"}}\n```'
                                )
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "done"}}]}

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(
        AgentRunsDB(tmp_path / "child.db"), registry, chat_call=_Chat()
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "delegate"}],
        config=AgentConfig(
            model="m",
            system_prompt="BASE",
            allowed_tools=(SPAWN_TOOL_NAME,),
            budget=RunBudget(max_subagents=1),
            native_tools=False,
            personal_context_block=PROFILE_BLOCK,
        ),
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)

    assert any(
        config.personal_context_block == PROFILE_BLOCK
        for config in captured_configs[1:]
    )
