import pytest

from tldw_chatbook.Skills_Interop.local_skills_service import (
    SKILL_FILE_READ_CAP_CHARS,
    LocalSkillsService,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


def _svc(tmp_path):
    return LocalSkillsService(
        store_dir=tmp_path, allow_untrusted_without_trust_service=True
    )


async def _make_skill(svc, name="demo"):
    await svc.create_skill(name=name, content=f"---\nname: {name}\n---\nbody\n")
    d = svc._skill_dir(name)
    (d / "references").mkdir(parents=True, exist_ok=True)
    (d / "references" / "api.md").write_text("# api docs\n", encoding="utf-8")
    (d / "assets").mkdir(exist_ok=True)
    (d / "assets" / "logo.png").write_bytes(b"\x89PNG\x00bin")
    return d


@pytest.mark.asyncio
async def test_read_happy_path_nested(tmp_path):
    svc = _svc(tmp_path)
    await _make_skill(svc)
    out = await svc.read_skill_file("demo", "references/api.md")
    assert out == {"content": "# api docs\n", "truncated": False, "size": len("# api docs\n")}


@pytest.mark.asyncio
async def test_read_traversal_and_bad_paths_rejected(tmp_path):
    svc = _svc(tmp_path)
    await _make_skill(svc)
    for bad in ("../escape.md", "/abs.md", "refs/SKILL.md"):
        with pytest.raises(ValueError):
            await svc.read_skill_file("demo", bad)


@pytest.mark.asyncio
async def test_intermediate_symlink_dir_yields_not_found_not_unsafe(tmp_path):
    # Qodo/PR#814 hardening: an intermediate symlinked directory inside the
    # bundle must never act as an existence/size oracle -- an escape that
    # resolves to a REAL file outside the bundle must raise the SAME
    # "local_skill_file_not_found" error as an escape that resolves to
    # nothing, never the "unsafe local skill path" message a downstream
    # containment check would otherwise leak only when the target exists.
    svc = _svc(tmp_path)
    d = await _make_skill(svc)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("top secret\n", encoding="utf-8")
    link = d / "link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError) as existing_exc:
        await svc.read_skill_file("demo", "link/secret.md")
    with pytest.raises(ValueError) as missing_exc:
        await svc.read_skill_file("demo", "link/nonexistent.md")

    assert "local_skill_file_not_found" in str(existing_exc.value)
    assert "local_skill_file_not_found" in str(missing_exc.value)
    assert "unsafe" not in str(existing_exc.value)
    assert "unsafe" not in str(missing_exc.value)


@pytest.mark.asyncio
async def test_read_binary_returns_refusal_not_bytes(tmp_path):
    svc = _svc(tmp_path)
    await _make_skill(svc)
    out = await svc.read_skill_file("demo", "assets/logo.png")
    assert out["truncated"] is False
    assert "binary file" in out["content"]
    assert "\x89" not in out["content"]


@pytest.mark.asyncio
async def test_read_truncates_over_cap(tmp_path):
    svc = _svc(tmp_path)
    d = await _make_skill(svc)
    (d / "references" / "big.md").write_text("x" * (SKILL_FILE_READ_CAP_CHARS + 500), encoding="utf-8")
    out = await svc.read_skill_file("demo", "references/big.md")
    assert out["truncated"] is True
    assert len(out["content"]) < SKILL_FILE_READ_CAP_CHARS + 200  # cap + marker line
    assert "truncated" in out["content"].rsplit("\n", 1)[-1]


@pytest.mark.asyncio
async def test_read_untrusted_raises_blocked(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    await _make_skill(svc)

    def _deny(name):
        raise SkillTrustBlockedError(
            skill_name=name, reason_code="skill_modified", trust_status="quarantined_modified"
        )

    monkeypatch.setattr(svc, "_require_trusted_skill", _deny)
    with pytest.raises(SkillTrustBlockedError):
        await svc.read_skill_file("demo", "references/api.md")


@pytest.mark.asyncio
async def test_read_missing_file_clean_error(tmp_path):
    svc = _svc(tmp_path)
    await _make_skill(svc)
    with pytest.raises(ValueError, match="local_skill_file_not_found"):
        await svc.read_skill_file("demo", "references/nope.md")


@pytest.mark.asyncio
async def test_read_skill_md_body_is_readable(tmp_path):
    # Qodo/PR#814: the spec says the skill body IS readable through this
    # seam -- validate_supporting_file_path's case-insensitive "skill.md"
    # rejection must not apply to the exact canonical body path.
    svc = _svc(tmp_path)
    await _make_skill(svc)
    out = await svc.read_skill_file("demo", "SKILL.md")
    assert out["content"] == "---\nname: demo\n---\nbody\n"
    assert out["truncated"] is False


@pytest.mark.asyncio
async def test_read_nested_or_wrongcase_skill_md_still_rejected(tmp_path):
    # Only the EXACT "SKILL.md" top-level path is exempted -- a nested
    # shadow file and any-case variants still go through the validator and
    # are rejected exactly as before (the validator's case-insensitive
    # "skill.md" basename check, not a not-found -- these paths never reach
    # the filesystem).
    svc = _svc(tmp_path)
    await _make_skill(svc)
    with pytest.raises(ValueError):
        await svc.read_skill_file("demo", "references/SKILL.md")
    with pytest.raises(ValueError):
        await svc.read_skill_file("demo", "skill.md")


@pytest.mark.asyncio
async def test_scope_service_server_mode_rejected(tmp_path):
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    scope = SkillsScopeService(local_service=_svc(tmp_path), server_service=None)
    with pytest.raises(ValueError, match="local-only"):
        await scope.read_skill_file("demo", "references/api.md", mode="server")


def test_policy_action_id_registered():
    # The engine denies unknown ids (fail-closed) — pin that the new id exists.
    from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY

    assert "skills.read_file.launch.local" in CAPABILITY_REGISTRY


@pytest.mark.asyncio
async def test_execute_skill_carries_reference_files(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    await _make_skill(svc)

    real_manifest = svc._read_bundle_manifest
    walks = []

    def _counting(skill_dir):
        walks.append(skill_dir)
        return real_manifest(skill_dir)

    monkeypatch.setattr(svc, "_read_bundle_manifest", _counting)
    result = await svc.execute_skill("demo")
    refs = {r["path"]: r for r in (result.get("reference_files") or [])}
    assert refs["references/api.md"]["is_text"] is True
    assert refs["assets/logo.png"]["is_text"] is False
    assert "executable" not in refs["references/api.md"]
    # Mechanism: exactly ONE bundle walk per invocation (get_skill's own) —
    # reference_files must be derived from the in-hand payload, not a re-walk.
    assert len(walks) == 1


@pytest.mark.asyncio
async def test_execute_skill_reference_files_none_when_no_bundle(tmp_path):
    svc = _svc(tmp_path)
    await svc.create_skill(name="bare", content="---\nname: bare\n---\nbody\n")
    result = await svc.execute_skill("bare")
    assert result.get("reference_files") is None
