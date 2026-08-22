"""Task 13 (PR E, AC 46): the re-chunk action's policy id, pinned.

Spec §10.4 picked the LAZY option deliberately: the re-chunk action REUSES
the existing ``rag.admin.launch`` verb (exactly what the backfill-shaped
trigger already means) instead of adding a fifth ``rag.admin.*`` verb --
which would have meant editing ``runtime_policy/registry.py`` AND the
exact-equality literal block, and whose tempting shortcut (extending the
SHARED ``DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS`` tuple) would silently
grant the verb to other capabilities.

Pinned here:

* ``rag.admin.launch.local`` is registered (and present in the equality
  test's own literal, so the reuse cannot drift from the registry);
* the scope service routes the re-chunk through EXACTLY that action id;
* a policy denial reaches the caller (the worker surfaces it, never
  bypasses it);
* no fifth ``rag.admin`` verb was added (the lazy choice, held in place).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.RAG_Admin.rag_admin_scope_service import RAGAdminScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY

#: AC 46: the single action id this action is allowed to launch under.
RECHUNK_POLICY_ACTION_ID = "rag.admin.launch.local"


def test_rechunk_policy_action_id_is_registered() -> None:
    assert RECHUNK_POLICY_ACTION_ID in CAPABILITY_REGISTRY


def test_rechunk_policy_action_id_is_pinned_by_the_equality_literal() -> None:
    """The existing exact-equality test picks up the reused verb; tie this
    pin to its literal so removing the verb fails BOTH tests."""
    from Tests.RuntimePolicy.test_runtime_policy_core import (
        EXPECTED_ACTION_IDS_BY_CAPABILITY,
    )

    rag_ids = EXPECTED_ACTION_IDS_BY_CAPABILITY.get("rag_embeddings_chunking_admin")
    assert rag_ids is not None
    assert RECHUNK_POLICY_ACTION_ID in rag_ids


def test_no_fifth_rag_admin_verb_was_added() -> None:
    """The lazy choice, held: ``rag.admin`` local actions stay exactly
    list/configure/launch/observe."""
    local_rag_admin = {
        action_id
        for action_id in CAPABILITY_REGISTRY
        if action_id.startswith("rag.admin.") and action_id.endswith(".local")
    }
    assert local_rag_admin == {
        "rag.admin.list.local",
        "rag.admin.configure.local",
        "rag.admin.launch.local",
        "rag.admin.observe.local",
    }


class _RecordingEnforcer:
    def __init__(self, *, deny: bool = False) -> None:
        self.actions: list[str] = []
        self._deny = deny

    def require_allowed(self, *, action_id: str) -> None:
        self.actions.append(action_id)
        if self._deny:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code="capability_disabled",
                user_message=f"policy denies {action_id}",
                effective_source="local",
                authority_owner="test",
            )


class _RecordingLocalService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def rechunk_legacy_media(self, **kwargs) -> dict:
        self.calls.append(dict(kwargs))
        return {"rechunked": 0, "skipped": 0, "failed": 0}


@pytest.mark.asyncio
async def test_scope_service_routes_rechunk_under_rag_admin_launch_local() -> None:
    enforcer = _RecordingEnforcer()
    local = _RecordingLocalService()
    scope = RAGAdminScopeService(
        local_service=local, server_service=None, policy_enforcer=enforcer
    )

    summary = await scope.rechunk_legacy_media(
        mode="local", rag_service=None, indexing_db=None
    )

    assert enforcer.actions == [RECHUNK_POLICY_ACTION_ID]
    assert local.calls, "the scope service must delegate to the local backend"
    assert summary == {"rechunked": 0, "skipped": 0, "failed": 0}


@pytest.mark.asyncio
async def test_scope_service_surfaces_policy_denial_for_rechunk() -> None:
    enforcer = _RecordingEnforcer(deny=True)
    local = _RecordingLocalService()
    scope = RAGAdminScopeService(
        local_service=local, server_service=None, policy_enforcer=enforcer
    )

    with pytest.raises(PolicyDeniedError):
        await scope.rechunk_legacy_media(mode="local")

    assert local.calls == [], "a denied launch must never reach the backend"
