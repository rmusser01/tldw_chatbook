"""Digest-pinned 'always allow scripts' grants + the run_script policy row."""

import pytest

from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY, get_capability_entry
from tldw_chatbook.runtime_policy.types import PolicyDeniedError


def test_run_script_policy_row_exists():
    assert "skills.run_script.launch.local" in CAPABILITY_REGISTRY
    entry = get_capability_entry("skills.run_script.launch.local")
    assert entry is not None


def test_unknown_action_id_still_fails_closed():
    with pytest.raises(PolicyDeniedError):
        get_capability_entry("skills.run_script.launch.nonsense")


def test_grant_records_current_digest(trust_service_with_skill):
    service, name = trust_service_with_skill
    assert service.script_execution_granted(name) is False
    service.grant_script_execution(name)
    assert service.script_execution_granted(name) is True
    assert service.script_grant_digest(name) == service.current_fingerprint_digest(name)


def test_grant_is_invalidated_when_content_changes(trust_service_with_skill, tmp_path):
    service, name = trust_service_with_skill
    service.grant_script_execution(name)
    assert service.script_execution_granted(name) is True
    (service.skills_dir / name / "scripts" / "hello.py").write_text(
        "print('mutated')", encoding="utf-8"
    )
    assert service.script_execution_granted(name) is False, (
        "a content change must drop the standing grant back to per-run confirm"
    )


def test_revoke_clears_the_grant(trust_service_with_skill):
    service, name = trust_service_with_skill
    service.grant_script_execution(name)
    service.revoke_script_execution(name)
    assert service.script_execution_granted(name) is False
    assert service.script_grant_digest(name) is None


def test_grant_persists_across_a_fresh_service_instance(
    trust_service_with_skill, make_trust_service
):
    service, name = trust_service_with_skill
    service.grant_script_execution(name)
    reloaded = make_trust_service()
    assert reloaded.script_execution_granted(name) is True


_MALFORMED_SKILL_NAMES = [
    "has spaces",
    "double--hyphen",
    "-leading-hyphen",
    "",
]


@pytest.mark.parametrize("bad_name", _MALFORMED_SKILL_NAMES)
def test_script_execution_granted_is_false_for_malformed_name(
    trust_service_with_skill, bad_name
):
    """A malformed name must resolve to "not granted", never raise.

    A later UI render path only catches (NoMatches, QueryError,
    AttributeError); an uncaught ValueError here would escape and break
    the panel render.
    """
    service, _name = trust_service_with_skill
    assert service.script_execution_granted(bad_name) is False


@pytest.mark.parametrize("bad_name", _MALFORMED_SKILL_NAMES)
def test_script_grant_digest_is_none_for_malformed_name(
    trust_service_with_skill, bad_name
):
    service, _name = trust_service_with_skill
    assert service.script_grant_digest(bad_name) is None


@pytest.mark.parametrize("bad_name", _MALFORMED_SKILL_NAMES)
def test_current_fingerprint_digest_raises_for_malformed_name(
    trust_service_with_skill, bad_name
):
    """Documented write/derive-side behavior: malformed names raise."""
    service, _name = trust_service_with_skill
    with pytest.raises(ValueError):
        service.current_fingerprint_digest(bad_name)


@pytest.mark.parametrize("bad_name", _MALFORMED_SKILL_NAMES)
def test_grant_script_execution_raises_for_malformed_name(
    trust_service_with_skill, bad_name
):
    """Documented write-side behavior: malformed names raise."""
    service, _name = trust_service_with_skill
    with pytest.raises(ValueError):
        service.grant_script_execution(bad_name)


@pytest.mark.parametrize("bad_name", _MALFORMED_SKILL_NAMES)
def test_revoke_script_execution_raises_for_malformed_name(
    trust_service_with_skill, bad_name
):
    """Documented write-side behavior: malformed names raise."""
    service, _name = trust_service_with_skill
    with pytest.raises(ValueError):
        service.revoke_script_execution(bad_name)


# ---------------------------------------------------------------------------
# Deleting a skill must revoke its standing script grant.
#
# The grant sidecar is keyed by skill NAME and pinned to a content digest, so
# an orphaned entry silently reactivates when a skill of the same name is
# reinstalled with byte-identical content: trust itself is re-reviewed on
# reinstall, but the script grant would not be -- handing an UNATTENDED run to
# an installation the user never granted.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_skill_revokes_the_standing_script_grant(script_service):
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    service, name = script_service
    trust = service.trust_service

    # `delete_skill` requires an index record, which the fixture's on-disk
    # bundle does not have; register one through the real create path.
    other = "grant-demo"
    await service.create_skill(
        name=other,
        content="---\nname: grant-demo\ndescription: demo\n---\nbody\n",
        supporting_files={"scripts/hello.py": "print('hello')"},
        trust_approved=True,
    )
    trust.grant_script_execution(other)
    assert trust.script_execution_granted(other) is True

    assert await service.delete_skill(other) is True
    assert trust.script_grant_digest(other) is None, (
        "a deleted skill's grant must not survive to be reactivated by a "
        "byte-identical reinstall of the same name"
    )

    # Reinstalling identical content re-derives the SAME digest, so a
    # surviving sidecar entry would have silently re-granted here.
    await service.create_skill(
        name=other,
        content="---\nname: grant-demo\ndescription: demo\n---\nbody\n",
        supporting_files={"scripts/hello.py": "print('hello')"},
        trust_approved=True,
    )
    assert trust.script_execution_granted(other) is False
    assert isinstance(service, LocalSkillsService)


@pytest.mark.asyncio
async def test_delete_skill_survives_a_failing_grant_revoke(script_service):
    """Revocation is best-effort: it must never fail an already-done delete."""
    service, _name = script_service
    other = "revoke-boom"
    await service.create_skill(
        name=other,
        content="---\nname: revoke-boom\ndescription: demo\n---\nbody\n",
        trust_approved=True,
    )

    real_trust = service.trust_service

    class _ExplodingRevokeTrustService:
        def __getattr__(self, item):
            return getattr(real_trust, item)

        def revoke_script_execution(self, skill_name):
            raise RuntimeError("sidecar unwritable")

    service.trust_service = _ExplodingRevokeTrustService()
    assert await service.delete_skill(other) is True
