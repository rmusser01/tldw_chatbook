"""build_spawn_schema: the spawn tool's per-run schema with named agents."""

from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Agents.tool_catalog import SPAWN_TOOL_SCHEMA, build_spawn_schema

RESEARCHER = AgentDefinition(
    name="researcher",
    description="Searches and summarizes sources.",
    instructions="Research thoroughly.",
)
CRITIC = AgentDefinition(name="critic", instructions="Critique carefully.")


def test_no_definitions_returns_shipped_schema_object():
    # Identity, not equality: byte-identical behavior when no definitions
    # exist (spec §4 — phase 1 is purely additive).
    assert build_spawn_schema([]) is SPAWN_TOOL_SCHEMA


def test_definitions_add_optional_agent_enum_and_roster():
    schema = build_spawn_schema([RESEARCHER, CRITIC])
    assert schema.id == SPAWN_TOOL_SCHEMA.id
    assert schema.name == SPAWN_TOOL_SCHEMA.name
    props = schema.parameters["properties"]
    assert props["task"] == SPAWN_TOOL_SCHEMA.parameters["properties"]["task"]
    assert props["agent"]["enum"] == ["researcher", "critic"]
    # Prose roster for fence-protocol models (they read descriptions,
    # not enums): one "name — description" line each.
    assert "researcher — Searches and summarizes sources." in props["agent"]["description"]
    assert "- critic" in props["agent"]["description"]
    # agent stays OPTIONAL: required is untouched.
    assert schema.parameters["required"] == ["task"]
