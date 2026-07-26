"""TASK-545 P2: the ported tools must declare risk tags.

Until this lands, no shipped tool overrides `risk_tags`, so the built-in
gate's entire `ask` path is exercised only by tests. Tagging is what makes
the approval machinery live for real users.
"""

import pytest

from tldw_chatbook.MCP.permission_store import BUILTIN_HIGH_RISK_TAGS
from tldw_chatbook.Tools.file_operation_tools import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from tldw_chatbook.Tools.note_management_tools import CreateNoteTool, UpdateNoteTool


@pytest.mark.parametrize(
    "factory,expected",
    [
        (WriteFileTool, ("mutates",)),
        (CreateNoteTool, ("mutates",)),
        (UpdateNoteTool, ("mutates",)),
        (ReadFileTool, ("reads",)),
        (ListDirectoryTool, ("reads",)),
    ],
)
def test_tool_declares_expected_risk_tags(factory, expected):
    assert factory().risk_tags == expected


@pytest.mark.parametrize(
    "factory",
    [WriteFileTool, CreateNoteTool, UpdateNoteTool, ReadFileTool, ListDirectoryTool],
)
def test_every_declared_tag_is_in_the_builtin_vocabulary(factory):
    """A typo'd tag would silently never floor anything to `ask`."""
    tags = set(factory().risk_tags)
    assert tags, "tool declares no risk tags"
    assert tags <= BUILTIN_HIGH_RISK_TAGS, (
        f"unrecognized risk tags: {tags - BUILTIN_HIGH_RISK_TAGS}"
    )


def test_read_only_search_tool_stays_untagged():
    """Scope guard: SearchNotesTool is explicitly out of P2's scope, so it
    must not acquire tags and start prompting."""
    from tldw_chatbook.Tools.note_management_tools import SearchNotesTool

    assert SearchNotesTool().risk_tags == ()
