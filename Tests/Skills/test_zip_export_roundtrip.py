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
