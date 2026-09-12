from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Agents.profile_tool_provider import (
    ProfileToolProvider,
    ProfileToolRunScope,
)
from tldw_chatbook.Personal_Context.proposal_service import (
    ProfileProposalQuota,
    ProposalQuotaExceeded,
)
from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority
from tldw_chatbook.Personal_Context.service import PersonalContextService


def test_quota_is_shared_across_fresh_providers_by_turn_and_session() -> None:
    quota = ProfileProposalQuota(per_turn=2, per_session=3)

    quota.reserve("turn-1", "session-1")
    quota.reserve("turn-1", "session-1")
    with pytest.raises(ProposalQuotaExceeded):
        quota.reserve("turn-1", "session-1")

    quota.reserve("turn-2", "session-1")
    with pytest.raises(ProposalQuotaExceeded):
        quota.reserve("turn-3", "session-1")


def test_failed_commit_releases_quota_reservation() -> None:
    quota = ProfileProposalQuota(per_turn=1, per_session=1)

    reservation = quota.reserve("turn-1", "session-1")
    reservation.release()

    quota.reserve("turn-1", "session-1")


def test_default_quota_is_five_per_turn_and_twenty_five_per_session() -> None:
    quota = ProfileProposalQuota()

    for turn_number in range(5):
        for _ in range(5):
            quota.reserve(f"turn-{turn_number}", "session-1")

    with pytest.raises(ProposalQuotaExceeded):
        quota.reserve("turn-0", "session-1")
    with pytest.raises(ProposalQuotaExceeded):
        quota.reserve("new-turn", "session-1")


def _scope(run_id: str) -> ProfileToolRunScope:
    return ProfileToolRunScope(
        run_id=run_id,
        session_id="session-1",
        profile_id="profile-1",
        scope_id="scope-1",
        authority=AgentAuthority.READ_ONLY,
        generation=0,
        authority_revision="revision-1",
    )


def test_run_scope_is_immutable() -> None:
    scope = _scope("turn-1")

    with pytest.raises(FrozenInstanceError):
        scope.run_id = "other"  # type: ignore[misc]


def test_stamp_scope_nesting_restores_the_directly_retained_scope() -> None:
    base = _scope("turn-base")
    outer = _scope("turn-outer")
    inner = _scope("turn-inner")
    provider = ProfileToolProvider(PersonalContextService.locked(), run_scope=base)

    assert provider._scope is base
    with provider.stamp_scope(outer.run_id, outer):
        assert provider._scope is outer
        with provider.stamp_scope(inner.run_id, inner):
            assert provider._scope is inner
        assert provider._scope is outer
    assert provider._scope is base
