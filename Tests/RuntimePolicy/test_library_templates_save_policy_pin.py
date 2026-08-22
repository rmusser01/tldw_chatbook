"""Task 4 (chunking-agent-tools): the chunk-spec save action's policy id, pinned.

Spec §6 picked a DEDICATED verb deliberately: ``library_save_chunk_spec``
maps to the new ``library.templates`` resource with a ``save`` action --
the local Library agent-tools policy home (the same ``library_collections``
capability that owns ``library.collections.*``), NOT the RAG-admin surface's
``rag.template.*`` actions (those belong to the RAG-admin UI seam per
ADR-003's verb-ownership precedent) and NOT the derived media read action
the MCP mapping used provisionally while the tool refused everything.

Pinned here (the task-13 re-chunk-pin pattern):

* ``library.templates.save.local`` is registered with the local-only
  attributes (and present in the equality test's own literal, so the verb
  cannot drift from the registry);
* the MCP local-control mapping resolves the tool to EXACTLY that action
  id -- the Task-3 deadline carry (a live write resolving to a read
  action under policy would be wrong even with the CRUD gate behind it);
* no server variant exists (local-only resource).
"""

from __future__ import annotations

from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY

#: The single action id the chunk-spec save tool is allowed to run under.
LIBRARY_TEMPLATES_SAVE_ACTION_ID = "library.templates.save.local"


def test_library_templates_save_action_id_is_registered() -> None:
    entry = CAPABILITY_REGISTRY[LIBRARY_TEMPLATES_SAVE_ACTION_ID]
    assert entry.capability_id == "library_collections"
    assert entry.domain_id == "library_collections"
    assert entry.required_source == "local"
    assert entry.authority_owner == "local"
    assert entry.enabled is True
    assert entry.action_kind == "update"  # create-or-update semantics


def test_library_templates_save_action_id_is_pinned_by_the_equality_literal() -> None:
    """The existing exact-equality test picks up the new verb; tie this pin
    to its literal so removing the verb fails BOTH tests."""
    from Tests.RuntimePolicy.test_runtime_policy_core import (
        EXPECTED_ACTION_IDS_BY_CAPABILITY,
    )

    library_ids = EXPECTED_ACTION_IDS_BY_CAPABILITY.get("library_collections")
    assert library_ids is not None
    assert LIBRARY_TEMPLATES_SAVE_ACTION_ID in library_ids


def test_library_templates_save_is_local_only() -> None:
    assert f"{LIBRARY_TEMPLATES_SAVE_ACTION_ID.removesuffix('.local')}.server" not in (
        CAPABILITY_REGISTRY
    )


def test_mcp_local_control_maps_spec_save_to_the_write_action() -> None:
    """The Task-3 deadline carry, held: the save tool's MCP policy mapping
    resolves to the write action, never the provisional derived read
    (``media.reading.list.local``)."""
    from tldw_chatbook.MCP.local_control_service import _TOOL_ACTION_IDS

    assert _TOOL_ACTION_IDS["library_save_chunk_spec"] == (
        LIBRARY_TEMPLATES_SAVE_ACTION_ID
    )
