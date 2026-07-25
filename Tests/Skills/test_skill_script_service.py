"""Trust + resolution discipline for the script-execution seams."""

import os
import stat

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


@pytest.mark.asyncio
async def test_runs_a_text_script_via_the_interpreter_map(script_service):
    service, name = script_service
    result = await service.run_skill_script(name, "scripts/hello.py", [])
    assert result.exit_code == 0
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_passes_args_through(script_service):
    service, name = script_service
    result = await service.run_skill_script(name, "scripts/echo_args.py", ["a", "b"])
    assert "a|b" in result.stdout


@pytest.mark.asyncio
async def test_exec_bit_file_runs_direct(script_service):
    service, name = script_service
    path = service._skill_dir(name) / "scripts" / "direct.sh"
    path.write_text("#!/bin/sh\necho direct-ran\n", encoding="utf-8")
    os.chmod(path, path.stat().st_mode | stat.S_IXUSR)
    # A file added AFTER script_service's bootstrap flips the skill's
    # fingerprint diff to "added" (any new file quarantines the whole
    # skill -- see SkillTrustService.status_for_skill), so this must
    # explicitly re-approve before describe/run will see it as trusted.
    service.trust_service.trust_current_skill(name, audit_event="test_setup")
    plan = await service.describe_skill_script(name, "scripts/direct.sh")
    assert plan.mechanism == "direct-exec"
    result = await service.run_skill_script(name, "scripts/direct.sh", [])
    assert "direct-ran" in result.stdout


@pytest.mark.asyncio
async def test_untrusted_skill_refuses_without_spawning(script_service_untrusted):
    service, name = script_service_untrusted
    with pytest.raises(SkillTrustBlockedError):
        await service.run_skill_script(name, "scripts/hello.py", [])


@pytest.mark.asyncio
async def test_describe_also_refuses_when_untrusted(script_service_untrusted):
    service, name = script_service_untrusted
    with pytest.raises(SkillTrustBlockedError):
        await service.describe_skill_script(name, "scripts/hello.py")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_path",
    ["../outside.py", "/etc/passwd", "SKILL.md", "scripts/missing.py"],
)
async def test_bad_paths_are_rejected(script_service, bad_path):
    service, name = script_service
    with pytest.raises(ValueError):
        await service.run_skill_script(name, bad_path, [])


def test_symlink_escape_is_indistinguishable_from_missing(script_service, tmp_path):
    """Symlink-oracle safety: existing vs missing targets give the SAME error.

    Exercises `_resolve_script` directly rather than through
    `run_skill_script`: the fixture's skill is TRUSTED, but planting a
    symlink anywhere inside a Chatbook-managed skill directory trips
    `SkillTrustService.status_for_skill`'s unconditional
    `unsupported_paths` check (permanently, regardless of re-trusting --
    the check re-scans the live tree on every call). Going through
    `run_skill_script` would therefore always raise `SkillTrustBlockedError`
    (reason `unsupported_path`) before ever reaching containment logic,
    conflating an orthogonal trust property with the one this test targets.
    Calling `_resolve_script` in isolation verifies the actual property
    under test: containment resolution alone must not leak whether an
    escaping symlink's target exists.
    """
    service, name = script_service
    scripts = service._skill_dir(name) / "scripts"
    real_target = tmp_path / "real_outside.py"
    real_target.write_text("print('outside')", encoding="utf-8")
    (scripts / "to_existing.py").symlink_to(real_target)
    (scripts / "to_missing.py").symlink_to(tmp_path / "nope.py")

    errors = []
    for link in ("scripts/to_existing.py", "scripts/to_missing.py"):
        with pytest.raises(ValueError) as excinfo:
            service._resolve_script(name, link)
        errors.append(str(excinfo.value))
    assert errors[0] == errors[1], "symlink target existence must not leak"


@pytest.mark.asyncio
async def test_unrunnable_type_errors_clearly(script_service):
    service, name = script_service
    (service._skill_dir(name) / "notes.txt").write_text("just text", encoding="utf-8")
    # See test_exec_bit_file_runs_direct: a post-bootstrap file addition
    # quarantines the whole skill until re-approved.
    service.trust_service.trust_current_skill(name, audit_event="test_setup")
    with pytest.raises(ValueError) as excinfo:
        await service.run_skill_script(name, "notes.txt", [])
    assert "unrunnable_script_type" in str(excinfo.value)


@pytest.mark.asyncio
async def test_script_cannot_write_into_its_own_bundle(script_service):
    """Scratch cwd, not the skill dir — a script must not tamper its fingerprints."""
    service, name = script_service
    skill_dir = service._skill_dir(name)
    (skill_dir / "scripts" / "writer.py").write_text(
        "open('tampered.txt', 'w').write('x'); print('wrote')", encoding="utf-8"
    )
    # See test_exec_bit_file_runs_direct: a post-bootstrap file addition
    # quarantines the whole skill until re-approved.
    service.trust_service.trust_current_skill(name, audit_event="test_setup")
    result = await service.run_skill_script(name, "scripts/writer.py", [])
    assert "wrote" in result.stdout
    assert not (skill_dir / "tampered.txt").exists()


@pytest.mark.asyncio
async def test_scratch_root_config_knob_is_reachable(script_service, tmp_path, monkeypatch):
    """The 3-arg get_cli_setting form must actually reach [skills].

    Patches ``tldw_chatbook.config`` (not the skills module) because the
    helper imports get_cli_setting lazily, at call time.
    """
    import tldw_chatbook.config as config_module

    custom_root = tmp_path / "custom-scratch"
    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(custom_root)
            if (section, key) == ("skills", "script_scratch_root")
            else default
        ),
    )
    service, name = script_service
    (service._skill_dir(name) / "scripts" / "cwd.py").write_text(
        "import os; print(os.path.realpath(os.getcwd()))", encoding="utf-8"
    )
    # See test_exec_bit_file_runs_direct: a post-bootstrap file addition
    # quarantines the whole skill until re-approved.
    service.trust_service.trust_current_skill(name, audit_event="test_setup")
    result = await service.run_skill_script(name, "scripts/cwd.py", [])
    assert str(custom_root.resolve()) in result.stdout


@pytest.mark.asyncio
async def test_scope_service_rejects_server_mode(script_scope_service):
    scope, name = script_scope_service
    with pytest.raises(ValueError, match="local-only"):
        await scope.run_skill_script(name, "scripts/hello.py", [], mode="server")


@pytest.mark.asyncio
async def test_scope_enforce_run_script_denies_when_policy_off(script_scope_service_denied):
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError

    scope, _name = script_scope_service_denied
    with pytest.raises(PolicyDeniedError):
        scope.enforce_run_script()
