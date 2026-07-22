import os, stat, pytest
from pathlib import Path
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.mark.asyncio
async def test_import_skill_directory_faithful(tmp_path):
    src = tmp_path / "src" / "demo"
    (src / "scripts").mkdir(parents=True)
    (src / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")
    (src / "scripts" / "run.sh").write_bytes(b"#!/bin/sh\necho hi\n")
    os.chmod(src / "scripts" / "run.sh", 0o755)
    (src / "assets").mkdir()
    (src / "assets" / "logo.png").write_bytes(b"\x89PNG\x00bin")
    (src / ".git").mkdir()
    (src / ".git" / "config").write_text("junk", encoding="utf-8")

    svc = LocalSkillsService(store_dir=tmp_path / "store")
    await svc.import_skill_directory(src, name="demo")
    d = svc._skill_dir("demo")
    assert (d / "scripts" / "run.sh").read_bytes() == b"#!/bin/sh\necho hi\n"
    assert d.joinpath("scripts", "run.sh").stat().st_mode & stat.S_IXUSR
    assert (d / "assets" / "logo.png").read_bytes() == b"\x89PNG\x00bin"
    assert not (d / ".git").exists()          # junk pruned
    skill = await svc.get_skill("demo")
    assert skill["trust_status"] != "trusted"  # trust-pending
