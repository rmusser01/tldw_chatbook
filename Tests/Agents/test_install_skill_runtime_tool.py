"""install_skill: the fifth runtime tool — name, schema, dispatch, gating.

Model: Tests/Agents/test_skill_file_runtime_tool.py (the 4th runtime tool).
install_skill is NOT a ToolProvider — its schema is pinned into
runtime_schemas (never disclosure-gated) only for the top-level agent
(agent_kind == primary), and its closure lives on LoopDeps.install_skill.
"""

import json

from tldw_chatbook.Agents.agent_models import (
    INSTALL_SKILL_TOOL_NAME,
    RUNTIME_TOOL_NAMES,
    SPAWN_TOOL_NAME,
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
    SKILL_FILE_TOOL_NAME,
)
from tldw_chatbook.Agents.tool_catalog import INSTALL_SKILL_TOOL_SCHEMA


def test_install_skill_name_in_runtime_tool_names():
    assert INSTALL_SKILL_TOOL_NAME == "install_skill"
    assert RUNTIME_TOOL_NAMES == {
        SPAWN_TOOL_NAME,
        FIND_TOOLS_NAME,
        LOAD_TOOLS_NAME,
        SKILL_FILE_TOOL_NAME,
        INSTALL_SKILL_TOOL_NAME,
    }


def test_install_skill_schema_shape():
    s = INSTALL_SKILL_TOOL_SCHEMA
    assert s.id == "runtime:install_skill"
    assert s.name == INSTALL_SKILL_TOOL_NAME
    assert s.parameters["required"] == ["url"]
    assert s.parameters["properties"]["url"]["type"] == "string"
    # Description must tell the model the key facts.
    assert "pending" in s.description.lower()
    assert "confirm" in s.description.lower()
