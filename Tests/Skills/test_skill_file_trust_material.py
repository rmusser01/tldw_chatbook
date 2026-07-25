"""task-578: skill_file reads must require trust-manifest membership.

The trust scanner deliberately prunes VCS/OS/build junk from fingerprinting so a
real bundle's litter cannot make a skill permanently untrustable. The
consequence is that a pruned file is never fingerprinted and never shown in the
human trust review -- so reading one hands the agent content the reviewer never
saw. These tests pin that the read seam asks the manifest, exactly as the
execution seam already does.
"""

import os
import stat

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


def _retrust(service, name):
    """Re-approve a skill after its on-disk content changed."""
    service.trust_service.trust_current_skill(name)


@pytest.mark.asyncio
async def test_reads_a_normal_fingerprinted_file(script_service):
    """AC#3: an ordinary supporting file still reads."""
    service, name = script_service
    (service._skill_dir(name) / "notes.md").write_text("hello notes", encoding="utf-8")
    _retrust(service, name)
    out = await service.read_skill_file(name, "notes.md")
    assert "hello notes" in out["content"]


@pytest.mark.asyncio
async def test_reads_a_nested_fingerprinted_file(script_service):
    """AC#3: nested paths still read."""
    service, name = script_service
    nested = service._skill_dir(name) / "references" / "deep"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "api.md").write_text("nested reference", encoding="utf-8")
    _retrust(service, name)
    out = await service.read_skill_file(name, "references/deep/api.md")
    assert "nested reference" in out["content"]


@pytest.mark.asyncio
async def test_reads_the_canonical_body(script_service):
    """AC#3: SKILL.md is trust material and stays readable through this seam."""
    service, name = script_service
    out = await service.read_skill_file(name, "SKILL.md")
    assert out["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "junk_path",
    [
        "vendored.tmp",
        "half-written.part",
        "session.swp",
    ],
)
async def test_pruned_non_vendor_paths_are_not_readable(script_service, junk_path):
    """AC#1: a present-but-unfingerprinted file is refused.

    These paths pass ``validate_supporting_file_path`` but the trust scanner
    prunes them, so they are invisible to trust review. They are transient
    editor/build artifacts, NOT vendored dependency data, so the vendored-read
    exemption deliberately does not cover them.
    """
    service, name = script_service
    target = service._skill_dir(name) / junk_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("instructions the reviewer never saw", encoding="utf-8")
    _retrust(service, name)

    with pytest.raises(ValueError) as excinfo:
        await service.read_skill_file(name, junk_path)
    assert "local_skill_file_not_found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_pruned_read_refusal_matches_a_missing_file(script_service):
    """AC#2: the refusal cannot be used to probe what a bundle contains."""
    service, name = script_service
    present = service._skill_dir(name) / "leftover.tmp"
    present.write_text("present but untrusted", encoding="utf-8")
    _retrust(service, name)

    errors = []
    for path in ("leftover.tmp", "absent.tmp"):
        with pytest.raises(ValueError) as excinfo:
            await service.read_skill_file(name, path)
        errors.append(str(excinfo.value).split(":", 1)[0])
    assert errors[0] == errors[1], "existence of a pruned file must not leak"


@pytest.mark.asyncio
async def test_vendored_data_is_readable_under_the_exemption(script_service):
    """A bundle may read its own vendored dependency data."""
    service, name = script_service
    target = service._skill_dir(name) / "node_modules" / "pkg" / "readme.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("vendored dependency docs", encoding="utf-8")
    _retrust(service, name)

    out = await service.read_skill_file(name, "node_modules/pkg/readme.md")
    assert "vendored dependency docs" in out["content"]


@pytest.mark.asyncio
async def test_vendored_read_is_labelled_as_outside_trust_review(script_service):
    """The exemption must not read as reviewed content.

    Vendored files are pruned from fingerprinting, so no human ever saw them
    in the trust review. The only channel that reaches the model is the
    content itself, so the notice rides there.
    """
    service, name = script_service
    target = service._skill_dir(name) / "node_modules" / "pkg" / "guide.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("do the thing", encoding="utf-8")
    _retrust(service, name)

    out = await service.read_skill_file(name, "node_modules/pkg/guide.md")
    assert out["trust_reviewed"] is False
    assert "not covered by trust review" in out["content"]


@pytest.mark.asyncio
async def test_ordinary_reads_are_marked_trust_reviewed(script_service):
    """A fingerprinted read carries no exemption notice."""
    service, name = script_service
    (service._skill_dir(name) / "plain.md").write_text("plain", encoding="utf-8")
    _retrust(service, name)

    out = await service.read_skill_file(name, "plain.md")
    assert out["trust_reviewed"] is True
    assert "not covered by trust review" not in out["content"]


@pytest.mark.asyncio
async def test_vendored_exemption_does_not_extend_to_execution(script_service):
    """READ-only: vendored code must never become runnable."""
    service, name = script_service
    target = service._skill_dir(name) / "node_modules" / "tool.sh"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("#!/bin/sh\necho pwned\n", encoding="utf-8")
    os.chmod(target, target.stat().st_mode | stat.S_IXUSR)
    _retrust(service, name)

    with pytest.raises(ValueError):
        await service.describe_skill_script(name, "node_modules/tool.sh")
    with pytest.raises(ValueError):
        await service.run_skill_script(name, "node_modules/tool.sh", [])


@pytest.mark.asyncio
async def test_untrusted_skill_still_refuses_before_the_manifest_check(
    script_service_untrusted,
):
    """Trust blocking still precedes membership: the error kind is unchanged."""
    service, name = script_service_untrusted
    with pytest.raises(SkillTrustBlockedError):
        await service.read_skill_file(name, "SKILL.md")
