"""Task 1 (student-workflow): the note-save action's policy id, pinned.

Spec §4/§8: ``library_save_note`` maps to the new ``library.notes``
resource with a ``save`` action -- the local Library agent-tools policy
home (the same ``library_collections`` capability that owns
``library.collections.*``, ``library.templates`` and ``library.media``),
NOT the notes UI's own ``notes.*`` CRUD verbs (those model the notes
screen's row operations; the agent tool is the Library write surface) and
NOT the derived notes read action the MCP mapping would otherwise resolve
a note-typed tool to.

Pinned here (the task-4/task-5 pin pattern):

* ``library.notes.save.local`` is registered with the local-only
  attributes (and present in the equality test's own literal, so the verb
  cannot drift from the registry);
* the MCP local-control mapping resolves the tool to EXACTLY that action
  id on BOTH seams (tool.execute previews and tools/call runtime
  requests), and the override map is EXACTLY the three writing tools;
* no server variant exists (local-only resource).
"""

from __future__ import annotations

from tldw_chatbook.MCP.local_control_service import (
    _LIBRARY_TOOL_ACTION_OVERRIDES,
    _TOOL_ACTION_IDS,
)
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY

#: The single action id the note-save tool is allowed to run under.
LIBRARY_NOTES_SAVE_ACTION_ID = "library.notes.save.local"


def test_library_notes_save_action_id_is_registered() -> None:
    entry = CAPABILITY_REGISTRY[LIBRARY_NOTES_SAVE_ACTION_ID]
    assert entry.capability_id == "library_collections"
    assert entry.domain_id == "library_collections"
    assert entry.required_source == "local"
    assert entry.authority_owner == "local"
    assert entry.enabled is True
    assert entry.action_kind == "update"  # create-or-update semantics


def test_library_notes_save_action_id_is_pinned_by_the_equality_literal() -> None:
    """The existing exact-equality test picks up the new verb; tie this pin
    to its literal so removing the verb fails BOTH tests."""
    from Tests.RuntimePolicy.test_runtime_policy_core import (
        EXPECTED_ACTION_IDS_BY_CAPABILITY,
    )

    library_ids = EXPECTED_ACTION_IDS_BY_CAPABILITY.get("library_collections")
    assert library_ids is not None
    assert LIBRARY_NOTES_SAVE_ACTION_ID in library_ids


def test_library_notes_save_is_local_only() -> None:
    assert f"{LIBRARY_NOTES_SAVE_ACTION_ID.removesuffix('.local')}.server" not in (
        CAPABILITY_REGISTRY
    )


def test_mcp_local_control_maps_save_note_to_the_write_action() -> None:
    """The save tool's MCP policy mapping resolves to the write action,
    never the derived notes read (``notes.list.local``) a note-typed tool
    would otherwise fall to."""
    assert _TOOL_ACTION_IDS["library_save_note"] == LIBRARY_NOTES_SAVE_ACTION_ID


def test_writing_tool_overrides_are_exactly_the_three_write_tools() -> None:
    """The override map stays descriptor-keyed and exactly the three writing
    operations -- every other descriptor keeps its type-owned read mapping."""
    assert _LIBRARY_TOOL_ACTION_OVERRIDES == {
        "spec_save": "library.templates.save.local",
        "rechunk": "library.media.rechunk.local",
        "save": LIBRARY_NOTES_SAVE_ACTION_ID,
    }
