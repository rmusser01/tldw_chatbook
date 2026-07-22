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


@pytest.mark.asyncio
async def test_import_skill_directory_narrows_exec_bit_widening_to_owner_only(tmp_path):
    """The exec-bit-preservation chmod must only trust/add owner-exec
    (S_IXUSR), never widen group/other permissions. A freshly written dest
    file always starts with zero exec bits (Python's default open() mode is
    0o666, which carries no x bits at all), so any group/other exec bit
    present after import came from an over-broad ``| 0o755`` widening, not
    from the source."""
    src = tmp_path / "src" / "demo"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")
    (src / "run.sh").write_bytes(b"#!/bin/sh\necho hi\n")
    os.chmod(src / "run.sh", 0o700)  # owner-only exec; narrowly permissioned source

    svc = LocalSkillsService(store_dir=tmp_path / "store")
    await svc.import_skill_directory(src, name="demo")
    d = svc._skill_dir("demo")
    dest_mode = (d / "run.sh").stat().st_mode
    assert dest_mode & stat.S_IXUSR        # owner-exec preserved
    assert not dest_mode & stat.S_IXGRP    # group-exec NOT widened
    assert not dest_mode & stat.S_IXOTH    # other-exec NOT widened


@pytest.mark.asyncio
async def test_import_skill_directory_skips_symlink(tmp_path):
    """A symlink in the source (pointing out of the tree) is skipped, not
    copied or followed -- the import still succeeds and the link's name is
    absent from the stored skill dir and supporting files."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    src = tmp_path / "src" / "demo"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")
    (src / "real.txt").write_text("kept", encoding="utf-8")
    (src / "link.txt").symlink_to(outside)

    svc = LocalSkillsService(store_dir=tmp_path / "store")
    result = await svc.import_skill_directory(src, name="demo")
    d = svc._skill_dir("demo")
    assert (d / "real.txt").read_text(encoding="utf-8") == "kept"
    assert not (d / "link.txt").exists()  # symlink skipped, not copied
    supporting = result.get("supporting_files") or {}
    assert "link.txt" not in supporting


@pytest.mark.asyncio
async def test_import_skill_directory_rejects_symlinked_skill_md(tmp_path):
    """A top-level SKILL.md that is itself a symlink (to a file outside the
    bundle) is rejected -- its target must NOT be read into the skill body."""
    outside = tmp_path / "outside.md"
    outside.write_text("---\nname: evil\n---\nleaked\n", encoding="utf-8")
    src = tmp_path / "src" / "demo"
    src.mkdir(parents=True)
    (src / "SKILL.md").symlink_to(outside)

    svc = LocalSkillsService(store_dir=tmp_path / "store")
    with pytest.raises(ValueError):
        await svc.import_skill_directory(src, name="demo")
    assert not svc._skill_dir("demo").exists()  # no partial write


@pytest.mark.asyncio
async def test_import_skill_directory_enforces_per_file_cap(tmp_path):
    """A supporting file exceeding the 5MB per-file cap raises ValueError and
    leaves NO partial write -- the target skill dir is never created."""
    from tldw_chatbook.tldw_api.skills_schemas import MAX_SUPPORTING_FILE_BYTES

    src = tmp_path / "src" / "demo"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")
    big = src / "big.bin"
    with big.open("wb") as handle:  # sparse: seek past cap, write 1 byte
        handle.seek(MAX_SUPPORTING_FILE_BYTES)
        handle.write(b"\x00")
    assert big.stat().st_size > MAX_SUPPORTING_FILE_BYTES

    svc = LocalSkillsService(store_dir=tmp_path / "store")
    with pytest.raises(ValueError):
        await svc.import_skill_directory(src, name="demo")
    assert not svc._skill_dir("demo").exists()  # no partial write
