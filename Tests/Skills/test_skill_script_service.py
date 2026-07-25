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
    """Symlink-oracle safety: existing vs missing targets give the SAME error KIND.

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

    The two probes use DIFFERENT requested paths (`to_existing.py` vs.
    `to_missing.py`), so the messages themselves are expected to differ --
    `_resolve_script` echoes the caller's own `script_path` back (a pure
    function of that input, so it leaks nothing about the filesystem; see
    `local_skills_service._SCRIPT_NOT_FOUND_ERROR`). What must stay
    constant is the error KIND (the part before the colon): that is the
    actual oracle-safety property, and a naive `errors[0] == errors[1]`
    whole-string comparison across two different inputs would not pin it
    (a bug that echoed a totally different message per branch could satisfy
    that assertion by chance for other reasons, or a correctly-fixed
    implementation with paths in the text would fail it for the WRONG
    reason).
    """
    service, name = script_service
    scripts = service._skill_dir(name) / "scripts"
    real_target = tmp_path / "real_outside.py"
    real_target.write_text("print('outside')", encoding="utf-8")
    (scripts / "to_existing.py").symlink_to(real_target)
    (scripts / "to_missing.py").symlink_to(tmp_path / "nope.py")

    links = ("scripts/to_existing.py", "scripts/to_missing.py")
    errors = []
    for link in links:
        with pytest.raises(ValueError) as excinfo:
            service._resolve_script(name, link)
        errors.append(str(excinfo.value))

    kinds = [message.split(":", 1)[0] for message in errors]
    assert kinds[0] == kinds[1] == "local_skill_script_not_found", (
        "symlink target existence must not leak via a different error KIND"
    )
    # And pin the (now-restored) per-path detail alongside the kind, so a
    # regression back to a bare constant (which would make this whole test
    # vacuous again) is caught too.
    assert errors[0] == f"local_skill_script_not_found:{links[0]}"
    assert errors[1] == f"local_skill_script_not_found:{links[1]}"


@pytest.mark.asyncio
async def test_symlink_in_bundle_blocks_the_public_seam(script_service):
    """A symlinked bundle raises SkillTrustBlockedError from the PUBLIC seam.

    Complements `test_symlink_escape_is_indistinguishable_from_missing`
    (which pokes `_resolve_script` directly to isolate the containment
    property from trust, per that test's own docstring): this test proves
    the OTHER half of that claim -- that a real caller going through
    `describe_skill_script`/`run_skill_script` never even reaches
    containment logic for a symlink-bearing bundle, because the trust
    scanner's `unsupported_paths` check quarantines it first.
    """
    service, name = script_service
    scripts = service._skill_dir(name) / "scripts"
    (scripts / "escape.py").symlink_to(scripts / "hello.py")

    with pytest.raises(SkillTrustBlockedError):
        await service.describe_skill_script(name, "scripts/hello.py")
    with pytest.raises(SkillTrustBlockedError):
        await service.run_skill_script(name, "scripts/hello.py", [])


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
    """The 3-arg get_cli_setting form actually reaches [skills] via REAL config.

    Writes a real config.toml and points TLDW_CONFIG_PATH at it -- the
    established pattern in e.g. Tests/test_config_console_defaults.py --
    rather than monkeypatching `get_cli_setting` itself. Monkeypatching the
    function under test cannot prove this docstring's claim: it only shows
    that `_script_scratch_root` calls whatever `get_cli_setting` happens to
    be bound to, not that the REAL `get_cli_setting("skills", "<key>",
    default)` 3-arg call actually resolves `[skills]` out of a loaded TOML
    file (the exact thing the section-dict-form bug at config.py:3965 would
    otherwise silently defeat). This exercises the genuine
    `load_cli_config_and_ensure_existence()` -> section-lookup path.
    """
    custom_root = tmp_path / "custom-scratch"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[skills]\nscript_scratch_root = "{custom_root}"\n', encoding="utf-8"
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

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
async def test_scratch_root_inside_skill_directory_is_rejected(script_service, monkeypatch):
    """A configured scratch root inside a skill's own bundle must be rejected.

    PROBED regression: pointing `[skills] script_scratch_root` at a path
    under a skill's own directory would (pre-fix) make the run's cwd land
    INSIDE the trusted bundle, letting the script write into it -- exactly
    the "a script must never tamper with its own bundle" property this
    task exists to guarantee (and any residue left behind would permanently
    quarantine the skill, since any added file re-fingerprints it).
    """
    import tldw_chatbook.config as config_module

    service, name = script_service
    skill_dir = service._skill_dir(name)
    unsafe_root = skill_dir / "unsafe-scratch"
    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(unsafe_root)
            if (section, key) == ("skills", "script_scratch_root")
            else default
        ),
    )
    (skill_dir / "scripts" / "cwd.py").write_text(
        "import os; print(os.path.realpath(os.getcwd()))", encoding="utf-8"
    )
    # See test_exec_bit_file_runs_direct: a post-bootstrap file addition
    # quarantines the whole skill until re-approved.
    service.trust_service.trust_current_skill(name, audit_event="test_setup")

    result = await service.run_skill_script(name, "scripts/cwd.py", [])

    reported_cwd = result.stdout.strip()
    resolved_skill_dir = str(skill_dir.resolve())
    assert not reported_cwd.startswith(resolved_skill_dir), (
        f"script cwd {reported_cwd!r} must not be inside the skill bundle "
        f"{resolved_skill_dir!r}"
    )
    # The rejected root must never even be created, let alone hold residue.
    assert not unsafe_root.exists()


def test_plan_for_script_does_not_read_the_whole_file_into_memory(
    script_service, monkeypatch
):
    """Classification must sniff only the first 8KB, never read the whole file.

    Regression guard for the OOM defect: `path.read_bytes()[:8192]` reads
    the ENTIRE file into memory before the slice ever runs, so a large
    vendored binary or model inside a trusted bundle would OOM the app on a
    mere `describe`. Actually writing a huge file here would be slow and
    disk-heavy for a unit test, so this pins the STRUCTURAL fix instead:
    `_plan_for_script` must never call `Path.read_bytes` at all.

    Exercises `_plan_for_script` directly (after resolving via
    `_resolve_script`) rather than through `describe_skill_script`: the
    trust scanner legitimately whole-file-reads every bundled file
    (including this same script) to compute fingerprints as part of trust
    re-verification, so patching `Path.read_bytes` globally around the
    public seam would trip on THAT unrelated read instead of isolating the
    classification codepath under test.
    """
    from pathlib import Path as PathModule

    service, name = script_service
    _skill_dir, path = service._resolve_script(name, "scripts/hello.py")

    def _forbidden_read_bytes(self, *args, **kwargs):
        raise AssertionError(
            "_plan_for_script must not read_bytes() the whole file"
        )

    monkeypatch.setattr(PathModule, "read_bytes", _forbidden_read_bytes)
    plan = service._plan_for_script(name, "scripts/hello.py", path)
    assert plan.mechanism == "interpreter"


@pytest.mark.asyncio
async def test_run_reverifies_trust_after_describe_succeeds(script_service):
    """`run_skill_script` must re-check trust itself, not lean on a prior describe.

    Mutating the bundle after a successful `describe_skill_script` call
    flips the skill to quarantined (any added file re-fingerprints it,
    exactly like the setup step every other test in this file that adds a
    script must perform). `run_skill_script` must see that live state and
    refuse, even though `describe_skill_script` already vetted this exact
    script moments earlier -- proving `run` re-verifies rather than
    trusting a stale `describe` result.
    """
    service, name = script_service
    plan = await service.describe_skill_script(name, "scripts/hello.py")
    assert plan.mechanism == "interpreter"

    (service._skill_dir(name) / "revocation-marker.txt").write_text(
        "mutated after describe succeeded", encoding="utf-8"
    )

    with pytest.raises(SkillTrustBlockedError):
        await service.run_skill_script(name, "scripts/hello.py", [])


@pytest.mark.asyncio
async def test_string_args_are_rejected_not_exploded_into_characters(script_service):
    """A bare str for `args` must be rejected, not exploded into one argv per char."""
    service, name = script_service
    with pytest.raises(ValueError, match="invalid_skill_script_args"):
        await service.run_skill_script(name, "scripts/hello.py", "ab")


@pytest.mark.asyncio
async def test_non_str_args_elements_are_rejected(script_service):
    service, name = script_service
    with pytest.raises(ValueError, match="invalid_skill_script_args"):
        await service.run_skill_script(name, "scripts/hello.py", [1, 2])


@pytest.mark.asyncio
async def test_tuple_args_are_accepted(script_service):
    """`args` accepts a tuple of str, not only a list."""
    service, name = script_service
    result = await service.run_skill_script(name, "scripts/echo_args.py", ("a", "b"))
    assert "a|b" in result.stdout


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
