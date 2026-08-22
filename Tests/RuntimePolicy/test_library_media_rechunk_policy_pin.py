"""Task 5 (chunking-agent-tools): the re-chunk action's policy id, pinned.

Spec §6 picked a DEDICATED verb deliberately: ``library_rechunk_media``
maps to the new ``library.media`` resource with a ``rechunk`` action --
the local Library agent-tools policy home (the same ``library_collections``
capability that owns ``library.collections.*`` and ``library.templates``),
NOT the RAG-admin surface's ``rag.admin.launch`` (that verb belongs to the
RAG-admin bulk action per ADR-003's verb-ownership precedent; this is a
Library-media item action, the #3 Task-13 semantic-owner precedent).

Pinned here (the task-13/task-4 pin pattern):

* ``library.media.rechunk.local`` is registered with the local-only
  attributes (and present in the equality test's own literal, so the verb
  cannot drift from the registry);
* the MCP local-control mapping resolves the tool to EXACTLY that action
  id, and the override map is EXACTLY the two writing tools (the
  tool-mapping pin);
* no server variant exists (local-only resource).
"""

from __future__ import annotations

from tldw_chatbook.MCP.local_control_service import (
    _LIBRARY_TOOL_ACTION_OVERRIDES,
    _TOOL_ACTION_IDS,
)
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY

#: The single action id the re-chunk tool is allowed to run under.
LIBRARY_MEDIA_RECHUNK_ACTION_ID = "library.media.rechunk.local"


def test_library_media_rechunk_action_id_is_registered() -> None:
    entry = CAPABILITY_REGISTRY[LIBRARY_MEDIA_RECHUNK_ACTION_ID]
    assert entry.capability_id == "library_collections"
    assert entry.domain_id == "library_collections"
    assert entry.required_source == "local"
    assert entry.authority_owner == "local"
    assert entry.enabled is True
    assert entry.action_kind == "launch"  # a regeneration run, not a CRUD write


def test_library_media_rechunk_action_id_is_pinned_by_the_equality_literal() -> None:
    """The existing exact-equality test picks up the new verb; tie this pin
    to its literal so removing the verb fails BOTH tests."""
    from Tests.RuntimePolicy.test_runtime_policy_core import (
        EXPECTED_ACTION_IDS_BY_CAPABILITY,
    )

    library_ids = EXPECTED_ACTION_IDS_BY_CAPABILITY.get("library_collections")
    assert library_ids is not None
    assert LIBRARY_MEDIA_RECHUNK_ACTION_ID in library_ids


def test_library_media_rechunk_is_local_only() -> None:
    assert f"{LIBRARY_MEDIA_RECHUNK_ACTION_ID.removesuffix('.local')}.server" not in (
        CAPABILITY_REGISTRY
    )


def test_mcp_local_control_maps_rechunk_to_the_write_action() -> None:
    """The re-chunk tool's MCP policy mapping resolves to the rechunk
    action, never the derived media read (``media.reading.list.local``)."""
    assert _TOOL_ACTION_IDS["library_rechunk_media"] == LIBRARY_MEDIA_RECHUNK_ACTION_ID


def test_writing_tool_overrides_are_exactly_the_two_write_tools() -> None:
    """The override map stays descriptor-keyed and exactly the two writing
    operations -- every other descriptor keeps its type-owned read mapping."""
    assert _LIBRARY_TOOL_ACTION_OVERRIDES == {
        "spec_save": "library.templates.save.local",
        "rechunk": LIBRARY_MEDIA_RECHUNK_ACTION_ID,
    }
