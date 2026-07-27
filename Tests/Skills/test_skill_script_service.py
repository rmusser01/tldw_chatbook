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


@pytest.mark.parametrize(
    "bad_path",
    [
        "../outside.py",  # traversal segment: validate_supporting_file_path's
        # OWN "Invalid path segment in '../outside.py'" message, pre-fix,
        # escaped `_resolve_script` unwrapped.
        "scripts/bad segment!.py",  # fails SEGMENT_PATTERN: validator's own
        # "Invalid path segment 'bad segment!.py' in ..." message.
        "x" * 5000,  # exceeds MAX_SUPPORTING_FILE_PATH_LEN: validator's own
        # "Supporting file path too long: ..." message.
    ],
)
def test_validator_rejection_is_indistinguishable_from_missing(script_service, bad_path):
    """A `validate_supporting_file_path` rejection must carry the SAME error
    KIND as a genuinely missing file (Qodo #871 finding 3).

    `_resolve_script`'s docstring promises every script-path rejection --
    unsafe, missing, symlink, untrusted, or the reserved body -- surfaces as
    the identical `local_skill_script_not_found` kind, so an escape attempt
    can never be told apart from a typo. `validate_supporting_file_path` is
    an independent validator with its OWN differently-worded `ValueError`
    messages ("Invalid path segment ...", "Supporting file path too long:
    ..."); calling it without a try/except would let those specific
    messages leak straight through, breaking that invariant for exactly the
    paths a security-conscious caller is most likely to probe with.
    """
    service, name = script_service

    with pytest.raises(ValueError) as validator_exc:
        service._resolve_script(name, bad_path)
    with pytest.raises(ValueError) as missing_exc:
        service._resolve_script(name, "scripts/definitely-missing.py")

    validator_kind = str(validator_exc.value).split(":", 1)[0]
    missing_kind = str(missing_exc.value).split(":", 1)[0]
    assert validator_kind == missing_kind == "local_skill_script_not_found", (
        f"validator-rejected path surfaced a different error kind: "
        f"{validator_exc.value!r} vs {missing_exc.value!r}"
    )
    # Pin the per-path echo alongside the kind, so a regression to a bare
    # constant (which would make the kind-equality assertion above
    # vacuously true) is still caught.
    assert str(validator_exc.value) == f"local_skill_script_not_found:{bad_path}"


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
async def test_scratch_root_config_knob_is_reachable(
    script_service, tmp_path, tmp_path_factory, monkeypatch
):
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

    `custom_root` is deliberately built from `tmp_path_factory`, NOT
    `tmp_path`: `script_service`'s `LocalSkillsService.store_dir` resolves to
    this same test's `tmp_path` (see `make_trust_service`), which is now
    (TASK-853) itself a protected container -- a scratch root nested
    anywhere under it, including a plain `tmp_path / "custom-scratch"`
    sibling of `skills`/`trust`, is correctly rejected as unsafe. A distinct
    `tmp_path_factory` directory is a genuinely unrelated location, the
    realistic shape of a real `[skills] script_scratch_root` value.
    """
    custom_root = tmp_path_factory.mktemp("custom-scratch")
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


def test_is_unsafe_scratch_root_rejects_both_containment_directions(
    script_service, tmp_path_factory
):
    """TASK-853: `_is_unsafe_scratch_root` must reject BOTH containment
    directions, not just the one that already worked.

    Pre-fix, this check only tested `get_safe_relative_path(root, container)`
    -- root NESTED INSIDE a container -- so a root that instead ENCLOSES a
    store (e.g. the skills service's own `store_dir`, which contains both
    `skills_dir` and the trust store) passed uncaught. A reproduction
    confirmed `store_dir` itself, and an ancestor of it, were both accepted
    pre-fix. Every candidate here is derived from the service's own live
    attributes (`service.store_dir`, `service.skills_dir`,
    `service.trust_service.trust_store.store_dir`) rather than a re-spelled
    literal, so this tracks the real containers instead of a copy that
    could silently drift from them.
    """
    service, name = script_service
    trust_store_dir = service.trust_service.trust_store.store_dir

    # Direction 1 (already worked pre-fix): root NESTED INSIDE a store.
    assert service._is_unsafe_scratch_root(service.skills_dir / "nested") is True
    assert service._is_unsafe_scratch_root(trust_store_dir / "nested") is True

    # Direction 2 (the bug): root that ENCLOSES a store. `store_dir` itself
    # (now an explicit container) contains both `skills_dir` and the trust
    # store, and its own parent encloses `store_dir` in turn.
    assert service._is_unsafe_scratch_root(service.store_dir) is True
    assert service._is_unsafe_scratch_root(service.store_dir.parent) is True

    # A genuinely unrelated root -- neither inside nor enclosing any store --
    # must still be accepted; the fix must not turn into "reject everything".
    legitimate_root = tmp_path_factory.mktemp("legit-scratch")
    assert service._is_unsafe_scratch_root(legitimate_root) is False


@pytest.mark.asyncio
async def test_cancelling_the_awaiting_task_lets_the_thread_finish_and_clean_up(
    script_service, monkeypatch
):
    """Cancelling `run_skill_script`'s awaiting task must not orphan the child.

    PROBED regression from the previous fix wave: once `run_script_subprocess`
    was offloaded via `asyncio.to_thread` with the scratch-dir `rmtree` left
    in the COROUTINE's own `finally`, cancelling the awaiting task raced that
    `finally` against the still-running thread -- the coroutine returned (and
    deleted the scratch dir) immediately on cancellation while the offloaded
    thread and its child process kept running for up to `wall_clock_seconds`
    against a now-unlinked cwd. `asyncio.to_thread` cannot interrupt a
    `concurrent.futures.Future` once it is RUNNING (only a PENDING future can
    actually be cancelled), so once the thread has started, cancelling the
    coroutine only detaches it from that future -- the thread keeps going
    regardless. The fix makes the scratch dir's whole create/run/cleanup
    lifecycle belong to that SAME offloaded callable, so a cancelled caller
    can never see (or cause) cleanup while the child is still alive; the
    thread is trusted to clean up after itself once `run_script_subprocess`
    (which itself SIGKILLs the process group before returning) finishes.
    """
    import asyncio
    import tempfile as tempfile_module
    from pathlib import Path as PathModule

    from tldw_chatbook.Skills_Interop.skill_script_runner import ScriptRunLimits

    service, name = script_service
    (service._skill_dir(name) / "scripts" / "slow.py").write_text(
        "import pathlib, time\n"
        "cwd = pathlib.Path.cwd()\n"
        "(cwd / 'started.marker').write_text('1')\n"
        "time.sleep(1.0)\n"
        "(cwd / 'finished.marker').write_text('1')\n",
        encoding="utf-8",
    )
    # See test_exec_bit_file_runs_direct: a post-bootstrap file addition
    # quarantines the whole skill until re-approved.
    service.trust_service.trust_current_skill(name, audit_event="test_setup")

    created_dirs: list[str] = []
    real_mkdtemp = tempfile_module.mkdtemp

    def _spy_mkdtemp(*args, **kwargs):
        created = real_mkdtemp(*args, **kwargs)
        created_dirs.append(created)
        return created

    monkeypatch.setattr(tempfile_module, "mkdtemp", _spy_mkdtemp)

    task = asyncio.create_task(
        service.run_skill_script(
            name,
            "scripts/slow.py",
            [],
            limits=ScriptRunLimits(wall_clock_seconds=10),
        )
    )

    # Wait until the offloaded thread has actually created the scratch dir
    # AND the child has actually started -- so the underlying
    # concurrent.futures.Future is RUNNING (not PENDING) by the time we
    # cancel, which is exactly the state the regression needs to be reached
    # through: a PENDING future WOULD be cancelled outright by
    # asyncio.to_thread, masking the bug this test pins.
    scratch = None
    for _ in range(500):
        if created_dirs:
            candidate = PathModule(created_dirs[0])
            if (candidate / "started.marker").exists():
                scratch = candidate
                break
        await asyncio.sleep(0.01)
    assert scratch is not None, "the sandboxed script never started"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The child is still mid-sleep right after the coroutine returns from
    # cancellation: the scratch dir must still be there, untouched, and the
    # child must not have finished yet.
    assert scratch.exists(), (
        "the scratch dir must not be removed while the child is still alive"
    )
    assert not (scratch / "finished.marker").exists(), (
        "the child should still be running at the moment of cancellation"
    )

    # The offloaded thread is unaffected by the coroutine's cancellation: it
    # keeps the child running to completion. task-584 changed what happens to
    # the directory afterwards -- a run that PRODUCED files (this one writes
    # both markers) is deliberately RETAINED as its output, so the property to
    # assert is no longer "it disappears" but "the child ran to completion
    # without cleanup racing it".
    for _ in range(500):
        if (scratch / "finished.marker").exists():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail(
            "the offloaded thread did not run the child to completion after "
            "the awaiting coroutine was cancelled"
        )
    # Both markers survive: nothing deleted the directory out from under a
    # still-live child, which is the race this test exists to catch.
    assert (scratch / "started.marker").exists()
    assert (scratch / "finished.marker").exists()


def test_plan_for_script_does_not_read_the_whole_file_into_memory(
    script_service, monkeypatch
):
    """Classification must sniff only the first 8KB, never read the whole file.

    Regression guard for the OOM defect: `path.read_bytes()[:8192]` reads
    the ENTIRE file into memory before the slice ever runs, so a large
    vendored binary or model inside a trusted bundle would OOM the app on a
    mere `describe`. Actually writing a huge file here would be slow and
    disk-heavy for a unit test, so this pins the STRUCTURAL fix instead:
    `_plan_for_script` must never call `Path.read_bytes` at all, AND the
    sniff `fh.read(...)` it does call must pass an explicit, small byte-count
    bound -- a regression to a bare `fh.read()` (no size, which reads to EOF)
    would satisfy the `read_bytes`-forbidden assertion alone while still
    being unbounded, so that check by itself is not sufficient.

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

    read_calls: list[tuple[tuple, dict]] = []
    real_open = PathModule.open

    def _spy_open(self, *args, **kwargs):
        fh = real_open(self, *args, **kwargs)
        real_read = fh.read

        def _spy_read(*read_args, **read_kwargs):
            read_calls.append((read_args, read_kwargs))
            return real_read(*read_args, **read_kwargs)

        fh.read = _spy_read
        return fh

    monkeypatch.setattr(PathModule, "open", _spy_open)

    plan = service._plan_for_script(name, "scripts/hello.py", path)

    assert plan.mechanism == "interpreter"
    assert read_calls, "_plan_for_script must sniff the file via fh.read(...)"
    call_args, call_kwargs = read_calls[0]
    size_arg = call_args[0] if call_args else call_kwargs.get("size")
    assert isinstance(size_arg, int) and 0 < size_arg <= 65536, (
        f"the sniff read must pass an explicit, small byte-count bound "
        f"(got args={call_args!r} kwargs={call_kwargs!r}); an unbounded "
        f"fh.read() would read the whole file into memory"
    )


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


# ---------------------------------------------------------------------------
# Fail-closed manifest membership: a script is runnable ONLY if the trust
# manifest actually fingerprints it.
#
# The trust scanner deliberately PRUNES VCS/OS/build junk (`node_modules/`,
# `.git/`, `__pycache__/`, `*.tmp`/`*.pyc`/`*~`/`.DS_Store`) so a real bundle's
# litter cannot make a skill permanently untrustable -- but
# `validate_supporting_file_path` ACCEPTS those same paths. Before the fix the
# run seam sat on the permissive side of that disagreement, so a file the
# human's trust review never saw (and whose bytes could be swapped afterwards
# without perturbing the digest a standing grant is pinned to) was fully
# runnable inside a "trusted" skill.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pruned_node_modules_script_is_not_runnable(script_service):
    """A file under a scanner-pruned dir is invisible to trust, so it cannot run."""
    service, name = script_service
    skill_dir = service._skill_dir(name)
    (skill_dir / "node_modules").mkdir()
    pwn = skill_dir / "node_modules" / "pwn.sh"
    pwn.write_text("#!/bin/sh\necho PWNED\n", encoding="utf-8")
    os.chmod(pwn, pwn.stat().st_mode | stat.S_IXUSR)
    # Re-approving cannot help: the scanner never fingerprints this path, so
    # the skill still reads as fully trusted with no changed files.
    service.trust_service.trust_current_skill(name, audit_event="test_setup")
    assert service.trust_service.status_for_skill(name).trust_status == "trusted"

    with pytest.raises(ValueError) as describe_exc:
        await service.describe_skill_script(name, "node_modules/pwn.sh")
    with pytest.raises(ValueError) as run_exc:
        await service.run_skill_script(name, "node_modules/pwn.sh", [])
    # Indistinguishable from a genuinely missing file: refusing with a
    # distinct "exists but untrusted" error would itself be an oracle.
    for exc in (describe_exc, run_exc):
        assert str(exc.value) == "local_skill_script_not_found:node_modules/pwn.sh"


@pytest.mark.asyncio
async def test_pruned_tmp_suffix_exec_script_is_not_runnable(script_service):
    """`*.tmp` is pruned by suffix; an exec bit must not make it runnable."""
    service, name = script_service
    backup = service._skill_dir(name) / "backup.sh.tmp"
    backup.write_text("#!/bin/sh\necho TMP_PWNED\n", encoding="utf-8")
    os.chmod(backup, backup.stat().st_mode | stat.S_IXUSR)
    service.trust_service.trust_current_skill(name, audit_event="test_setup")

    with pytest.raises(ValueError) as excinfo:
        await service.run_skill_script(name, "backup.sh.tmp", [])
    assert str(excinfo.value) == "local_skill_script_not_found:backup.sh.tmp"


@pytest.mark.asyncio
async def test_a_pruned_path_is_absent_from_the_human_trust_review(script_service):
    """Pins WHY the gate must be manifest membership, not path validation."""
    service, name = script_service
    skill_dir = service._skill_dir(name)
    (skill_dir / "node_modules").mkdir()
    (skill_dir / "node_modules" / "pwn.sh").write_text("echo hi\n", encoding="utf-8")
    service.trust_service.trust_current_skill(name, audit_event="test_setup")

    review = service.trust_service.capture_review(name)
    reviewed = {entry["relative_path"] for entry in review["current_fingerprints"]}
    assert "node_modules/pwn.sh" not in reviewed
    assert "scripts/hello.py" in reviewed
    trusted = service.trust_service.trusted_file_paths(name)
    assert "node_modules/pwn.sh" not in trusted
    assert "scripts/hello.py" in trusted


@pytest.mark.asyncio
async def test_a_normally_fingerprinted_script_still_runs(script_service):
    """The fail-closed gate must not break the ordinary case."""
    service, name = script_service
    plan = await service.describe_skill_script(name, "scripts/hello.py")
    assert plan.mechanism == "interpreter"
    result = await service.run_skill_script(name, "scripts/hello.py", [])
    assert result.exit_code == 0
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_a_trust_service_without_the_accessor_fails_closed(script_service):
    """A trust service that cannot answer "is this trusted?" refuses the run.

    Mirrors `_verify_exact_skill_content`'s handling of a trust service
    missing `verify_skill_content`: an unanswerable trust query is a refusal,
    never a permissive default.
    """
    service, name = script_service
    real_trust = service.trust_service

    class _NoAccessorTrustService:
        """Passes `ensure_skill_trusted` but exposes no `trusted_file_paths`."""

        def ensure_skill_trusted(self, skill_name):
            real_trust.ensure_skill_trusted(skill_name)

    service.trust_service = _NoAccessorTrustService()
    with pytest.raises(ValueError) as excinfo:
        await service.run_skill_script(name, "scripts/hello.py", [])
    assert str(excinfo.value) == "local_skill_script_not_found:scripts/hello.py"


@pytest.mark.asyncio
async def test_trusted_file_paths_fails_closed_when_trust_is_locked(script_service):
    """A locked trust service vouches for nothing (rather than everything)."""
    service, name = script_service
    trust = service.trust_service
    assert "scripts/hello.py" in trust.trusted_file_paths(name)
    trust._keys = None
    assert trust.trusted_file_paths(name) == frozenset()
    assert trust.is_trusted_file(name, "scripts/hello.py") is False


def test_trusted_file_paths_fails_closed_on_a_malformed_name(script_service):
    service, _name = script_service
    assert service.trust_service.trusted_file_paths("Not A Skill!") == frozenset()


def test_trusted_file_paths_fails_closed_for_an_unknown_skill(script_service):
    service, _name = script_service
    assert service.trust_service.trusted_file_paths("never-installed") == frozenset()


@pytest.mark.asyncio
async def test_no_trust_service_escape_hatch_keeps_its_existing_semantics(tmp_path):
    """`allow_untrusted_without_trust_service` is unchanged, not widened.

    With NO trust service there is no manifest to consult, so the manifest
    membership gate defers to exactly the flag `_require_trusted_skill`
    already keys off: True runs, False refuses (and refuses at the trust gate,
    before path resolution).
    """
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    skill_dir = tmp_path / "skills" / "demo-skill" / "scripts"
    skill_dir.mkdir(parents=True)
    (skill_dir.parent / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\nbody\n", encoding="utf-8"
    )
    (skill_dir / "hello.py").write_text("print('hello')", encoding="utf-8")

    permissive = LocalSkillsService(
        store_dir=tmp_path,
        trust_service=None,
        allow_untrusted_without_trust_service=True,
    )
    result = await permissive.run_skill_script("demo-skill", "scripts/hello.py", [])
    assert "hello" in result.stdout

    strict = LocalSkillsService(
        store_dir=tmp_path,
        trust_service=None,
        allow_untrusted_without_trust_service=False,
    )
    with pytest.raises(SkillTrustBlockedError):
        await strict.run_skill_script("demo-skill", "scripts/hello.py", [])


@pytest.mark.asyncio
async def test_plan_reports_the_canonical_skill_name_not_the_callers_spelling(
    script_service,
):
    """The plan feeds a human consent card, so it must name what will run."""
    service, name = script_service
    plan = await service.describe_skill_script("  DEMO-Skill ", "scripts/hello.py")
    assert plan.skill_name == name == "demo-skill"
