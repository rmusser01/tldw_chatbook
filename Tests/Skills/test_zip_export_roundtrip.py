import io, os, stat, zipfile, pytest
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.mark.asyncio
async def test_export_roundtrip_preserves_tree_and_mode(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    await svc.create_skill(name="demo", content="---\nname: demo\n---\nbody\n")
    d = svc._skill_dir("demo")
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    (d / "scripts" / "run.sh").write_bytes(b"#!/bin/sh\n")
    os.chmod(d / "scripts" / "run.sh", 0o755)
    (d / "assets").mkdir(exist_ok=True)
    (d / "assets" / "logo.png").write_bytes(b"\x89PNG\x00bin")

    export = await svc.export_skill("demo")
    with zipfile.ZipFile(io.BytesIO(export["content"])) as z:
        names = set(z.namelist())
        assert {"SKILL.md", "scripts/run.sh", "assets/logo.png"} <= names
        assert z.read("assets/logo.png") == b"\x89PNG\x00bin"
        info = z.getinfo("scripts/run.sh")
        assert (info.external_attr >> 16) & stat.S_IXUSR


@pytest.mark.asyncio
async def test_export_skill_missing_body_raises_domain_error_not_raw_oserror(
    tmp_path,
):
    """Merge-gate review (PR #784, MINOR finding): the top-level SKILL.md
    read in ``export_skill`` had no ``is_file()``/symlink guard, unlike
    every other body-read path in this service. A corrupted store (index
    entry present, on-disk body missing) must fail export with a domain
    ``ValueError`` naming the skill -- never a raw ``FileNotFoundError``
    leaking out of ``read_bytes()``.
    """
    svc = LocalSkillsService(store_dir=tmp_path)
    await svc.create_skill(name="demo", content="---\nname: demo\n---\nbody\n")
    body = svc._skill_dir("demo") / "SKILL.md"
    body.unlink()

    with pytest.raises(ValueError, match="local_skill_missing_skill_md"):
        await svc.export_skill("demo")


@pytest.mark.asyncio
async def test_export_skill_symlinked_body_is_rejected_not_followed(tmp_path):
    """A SKILL.md body that is itself a symlink (to a file outside the
    bundle) must be rejected, not followed -- mirrors
    ``import_skill_directory``'s own symlinked-body guard."""
    outside = tmp_path / "outside.md"
    outside.write_text("---\nname: evil\n---\nleaked\n", encoding="utf-8")
    svc = LocalSkillsService(store_dir=tmp_path / "store")
    await svc.create_skill(name="demo", content="---\nname: demo\n---\nbody\n")
    body = svc._skill_dir("demo") / "SKILL.md"
    body.unlink()
    body.symlink_to(outside)

    with pytest.raises(ValueError, match="local_skill_missing_skill_md"):
        await svc.export_skill("demo")
