"""Shared fixtures for `Tests/Skills/`.

Construction of `SkillTrustStore`/`SkillTrustService` follows the pattern
established in `test_skill_trust_service.py`: `SkillTrustStore` requires an
explicit `marker_store` (a file-backed marker store is fine for tests).

``make_trust_service``/``trust_service_with_skill`` used to live here, but
were promoted to the root ``Tests/conftest.py`` (task-7 of the
skills-script-execution SDD plan) once ``Tests/Library/`` needed them too --
pytest fixture discovery can't see a sibling directory's ``conftest.py``, and
duplicating them would let the two copies drift apart. They are still
available here unchanged, inherited from the root conftest.
"""

import pytest


def _bootstrap_and_trust(trust_service, skill_name: str) -> None:
    """Unlock + bootstrap trust so the current on-disk skill files are trusted.

    Reuses the exact idiom already established across `Tests/Skills/` for
    reaching a trusted `SkillTrustService` -- `test_skill_trust_service.py`'s
    local `_service()` helper and the `svc.bootstrap_trust("pw",
    salt=secrets.token_bytes(32))` calls throughout e.g.
    `test_verify_content_binary.py` and `test_skill_trust_service_reset_posture.py`.
    A fixed passphrase/salt pair is fine here: the trust manifest only needs
    to be internally self-consistent for the duration of one test, never
    secret.

    `bootstrap_trust` snapshots every skill directory that currently exists
    under `trust_service.skills_dir` as the trusted baseline -- so callers
    must finish writing every file `skill_name` needs *before* calling this.
    Writing a new file into an already-bootstrapped skill directory flips
    that skill's fingerprint diff (an "added" file) and re-quarantines it,
    exactly like a real skill being reviewed and then locally modified.

    Args:
        trust_service: An unlocked-or-not `SkillTrustService` to bootstrap.
        skill_name: The skill whose directory should already be fully
            populated on disk. Documented for caller intent only --
            `bootstrap_trust` itself trusts every skill directory present,
            not only this one.
    """
    trust_service.bootstrap_trust("test-passphrase", salt=b"3" * 32)


@pytest.fixture
def script_service(make_trust_service):
    """A LocalSkillsService with one TRUSTED skill carrying scripts.

    Args:
        make_trust_service: Factory fixture for building trust-service
            instances.

    Returns:
        A `(service, skill_name)` tuple. `skill_name`'s bundle already
        contains `scripts/hello.py` and `scripts/echo_args.py`, and the
        skill is trusted (see `_bootstrap_and_trust`). Tests that add MORE
        files to the bundle after this fixture returns must re-trust via
        `trust.trust_current_skill(name, audit_event=...)` before invoking
        `describe_skill_script`/`run_skill_script` again -- otherwise the
        newly-added file flips the skill to `quarantined_added` and every
        call raises `SkillTrustBlockedError`.
    """
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    trust = make_trust_service()
    name = "demo-skill"
    skill_dir = trust.skills_dir / name
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\nbody\n", encoding="utf-8"
    )
    (skill_dir / "scripts" / "hello.py").write_text("print('hello')", encoding="utf-8")
    (skill_dir / "scripts" / "echo_args.py").write_text(
        "import sys; print('|'.join(sys.argv[1:]))", encoding="utf-8"
    )
    service = LocalSkillsService(
        store_dir=trust.skills_dir.parent,
        trust_service=trust,
        allow_untrusted_without_trust_service=False,
    )
    _bootstrap_and_trust(trust, name)
    return service, name


@pytest.fixture
def script_service_untrusted(make_trust_service):
    """A LocalSkillsService with one skill whose trust was never bootstrapped.

    Args:
        make_trust_service: Factory fixture for building trust-service
            instances.

    Returns:
        A `(service, skill_name)` tuple where `skill_name` exists on disk
        (with a `scripts/hello.py`) but `_require_trusted_skill` must raise
        for it -- the backing `SkillTrustService` is left uninitialized
        (`trust_uninitialized`), mirroring
        `test_uninitialized_service_blocks_until_bootstrap` in
        `test_skill_trust_service.py`.
    """
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    trust = make_trust_service()
    name = "demo-skill"
    skill_dir = trust.skills_dir / name
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\nbody\n", encoding="utf-8"
    )
    (skill_dir / "scripts" / "hello.py").write_text("print('hello')", encoding="utf-8")
    service = LocalSkillsService(
        store_dir=trust.skills_dir.parent,
        trust_service=trust,
        allow_untrusted_without_trust_service=False,
    )
    return service, name


@pytest.fixture
def script_scope_service(script_service):
    """A SkillsScopeService wrapping the trusted `script_service`, no policy enforcer.

    Args:
        script_service: The trusted `(LocalSkillsService, skill_name)` fixture.

    Returns:
        A `(SkillsScopeService, skill_name)` tuple.
    """
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    service, name = script_service
    scope = SkillsScopeService(local_service=service, server_service=None)
    return scope, name


@pytest.fixture
def script_scope_service_denied(make_trust_service):
    """A SkillsScopeService with `skills.run_script.launch.local` policy-disabled.

    Wires a REAL `ServicePolicyEnforcer` bound to a REAL `PolicyEngine`, over
    a registry that is the production `CAPABILITY_REGISTRY` with exactly one
    row (`skills.run_script.launch.local`) swapped for a disabled copy --
    mirroring how `test_e2e_install_skill_from_github_tree_url_real_services`
    (`Tests/Skills/test_skill_remote_fetch.py`) wires a real enforcer end to
    end. A scope service built with `policy_enforcer=None` would make every
    policy check a silent no-op (see `SkillsScopeService._enforce_policy`),
    which would make a "denied" test pass vacuously; wiring a real enforcer
    with a real (if edited) registry is what makes the denial genuine.

    Args:
        make_trust_service: Factory fixture for building trust-service
            instances.

    Returns:
        A `(SkillsScopeService, skill_name)` tuple whose `local_service` and
        `scope_service` share the same enforcer, both gated by the disabled
        row.
    """
    import dataclasses

    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
    from tldw_chatbook.runtime_policy.engine import PolicyEngine
    from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
    from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    trust = make_trust_service()
    name = "demo-skill"
    skill_dir = trust.skills_dir / name
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\nbody\n", encoding="utf-8"
    )
    (skill_dir / "scripts" / "hello.py").write_text("print('hello')", encoding="utf-8")
    _bootstrap_and_trust(trust, name)

    action_id = "skills.run_script.launch.local"
    disabled_registry = dict(CAPABILITY_REGISTRY)
    disabled_registry[action_id] = dataclasses.replace(
        disabled_registry[action_id], enabled=False
    )
    policy_enforcer = ServicePolicyEnforcer(
        state_provider=lambda: RuntimeSourceState(active_source="local"),
        engine=PolicyEngine(disabled_registry),
    )
    local_service = LocalSkillsService(
        store_dir=trust.skills_dir.parent,
        trust_service=trust,
        policy_enforcer=policy_enforcer,
    )
    scope_service = SkillsScopeService(
        local_service=local_service,
        server_service=None,
        policy_enforcer=policy_enforcer,
    )
    return scope_service, name
