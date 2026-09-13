"""fork_chat / new_chat runtime-tool constants and schema shapes."""
from tldw_chatbook.Agents.agent_models import (
    FORK_CHAT_TOOL_NAME,
    NEW_CHAT_TOOL_NAME,
    RUNTIME_TOOL_NAMES,
)
from tldw_chatbook.Agents.tool_catalog import FORK_CHAT_TOOL_SCHEMA, NEW_CHAT_TOOL_SCHEMA


def test_names_are_runtime_tools():
    assert FORK_CHAT_TOOL_NAME == "fork_chat"
    assert NEW_CHAT_TOOL_NAME == "new_chat"
    assert FORK_CHAT_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert NEW_CHAT_TOOL_NAME in RUNTIME_TOOL_NAMES


def test_fork_chat_schema_shape():
    assert FORK_CHAT_TOOL_SCHEMA.id == "runtime:fork_chat"
    assert FORK_CHAT_TOOL_SCHEMA.name == FORK_CHAT_TOOL_NAME
    props = FORK_CHAT_TOOL_SCHEMA.parameters["properties"]
    assert set(props) == {"title", "opening_prompt", "instructions"}
    assert FORK_CHAT_TOOL_SCHEMA.parameters["required"] == []
    for text in (FORK_CHAT_TOOL_SCHEMA.description,):
        assert "user" in text and "confirm" in text  # documents the confirm contract


def test_new_chat_schema_shape():
    assert NEW_CHAT_TOOL_SCHEMA.id == "runtime:new_chat"
    assert NEW_CHAT_TOOL_SCHEMA.name == NEW_CHAT_TOOL_NAME
    props = NEW_CHAT_TOOL_SCHEMA.parameters["properties"]
    assert set(props) == {"title", "opening_prompt", "instructions"}
    assert NEW_CHAT_TOOL_SCHEMA.parameters["required"] == []


def test_first_request_schema_plan_includes_chat_create_tools():
    from tldw_chatbook.Agents.agent_service import build_first_request_schema_plan
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Agents.agent_models import AgentConfig

    registry = ToolCatalogRegistry()

    def _names(**flags):
        plan = build_first_request_schema_plan(
            registry, (),
            AgentConfig(model="m", system_prompt="s", allowed_tools=()),
            "llama_cpp", [],
            skill_file_enabled=False,
            install_skill_enabled=False,
            run_skill_script_enabled=False,
            run_log_active=False,
            **flags,
        )
        return {s.name for s in plan.runtime_schemas}

    on = _names(fork_chat_enabled=True, new_chat_enabled=True)
    assert "fork_chat" in on and "new_chat" in on
    off = _names(fork_chat_enabled=False, new_chat_enabled=False)
    assert "fork_chat" not in off and "new_chat" not in off
