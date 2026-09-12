"""Validation and authority-safe contents of built-in agent presets."""

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUNTIME_TOOL_NAMES,
    validate_agent_definition,
)
from tldw_chatbook.Agents.agent_presets import (
    AGENT_PRESETS,
    BULK_READER_PRESET,
    CRITIC_PRESET,
    INGEST_RUNNER_PRESET,
    RESEARCHER_PRESET,
)


@pytest.mark.parametrize(
    ("preset", "name", "tools"),
    [
        (
            BULK_READER_PRESET,
            "bulk-reader",
            ("fs_list", "fs_read", "fs_glob", "fs_grep"),
        ),
        (RESEARCHER_PRESET, "researcher", ()),
        (CRITIC_PRESET, "critic", ("fs_list", "fs_read", "fs_glob", "fs_grep")),
        (INGEST_RUNNER_PRESET, "ingest-runner", ()),
    ],
)
def test_starter_preset_is_valid_and_excludes_runtime_tools(preset, name, tools):
    assert preset.name == name
    assert preset.tool_allowlist == tools
    assert validate_agent_definition(preset) == []
    assert not set(preset.tool_allowlist).intersection(RUNTIME_TOOL_NAMES)


def test_agent_presets_have_fixed_display_order():
    assert AGENT_PRESETS == (
        BULK_READER_PRESET,
        RESEARCHER_PRESET,
        CRITIC_PRESET,
        INGEST_RUNNER_PRESET,
    )
