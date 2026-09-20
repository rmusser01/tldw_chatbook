"""fork_chat / new_chat runtime-tool constants and schema shapes."""
from tldw_chatbook.Agents.agent_models import (
    AgentDefinition,
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


# ---------------------------------------------------------------------------
# TASK-32874: dynamic schema builders (ADR-147 pattern reused for the
# chat-creation tools).
# ---------------------------------------------------------------------------

def test_chat_create_schema_builder_identity_when_nothing_to_add():
    from tldw_chatbook.Agents.tool_catalog import (
        FORK_CHAT_TOOL_SCHEMA,
        NEW_CHAT_TOOL_SCHEMA,
        build_chat_create_schema,
    )

    assert build_chat_create_schema(FORK_CHAT_TOOL_SCHEMA, ()) is FORK_CHAT_TOOL_SCHEMA
    assert build_chat_create_schema(NEW_CHAT_TOOL_SCHEMA, ()) is NEW_CHAT_TOOL_SCHEMA
    unrouted = AgentDefinition(name="plain", description="no routing")
    assert (
        build_chat_create_schema(FORK_CHAT_TOOL_SCHEMA, (unrouted,))
        is FORK_CHAT_TOOL_SCHEMA
    )


def test_chat_create_schema_builder_adds_preset_for_routed_definitions():
    from tldw_chatbook.Agents.tool_catalog import (
        FORK_CHAT_TOOL_SCHEMA,
        build_chat_create_schema,
    )

    routed = AgentDefinition(name="local", description="d", provider="llama_cpp", model="m1")
    other = AgentDefinition(name="cloud", description="d", provider="openai")
    schema = build_chat_create_schema(FORK_CHAT_TOOL_SCHEMA, (routed, other))
    props = schema.parameters["properties"]
    assert set(props) == {"title", "opening_prompt", "instructions", "preset"}
    assert props["preset"]["enum"] == ["local", "cloud"]
    assert "runs on llama_cpp / m1" in props["preset"]["description"]
    assert schema.id == FORK_CHAT_TOOL_SCHEMA.id
    assert schema.name == FORK_CHAT_TOOL_SCHEMA.name
    assert schema.parameters["required"] == []


def test_chat_create_schema_builder_override_args_when_enabled():
    from tldw_chatbook.Agents.tool_catalog import (
        NEW_CHAT_TOOL_SCHEMA,
        build_chat_create_schema,
    )

    schema = build_chat_create_schema(
        NEW_CHAT_TOOL_SCHEMA,
        (),
        override_enabled=True,
        override_targets=(("llama_cpp", ("m1", "m2")),),
    )
    props = schema.parameters["properties"]
    assert {"provider", "model"} <= set(props)
    assert "llama_cpp" in props["provider"]["description"]
    assert "preset" not in props
