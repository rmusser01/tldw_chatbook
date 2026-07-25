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
