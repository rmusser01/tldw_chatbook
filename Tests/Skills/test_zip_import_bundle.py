import io, zipfile, stat, pytest
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data, mode in members:
            info = zipfile.ZipInfo(name)
            if mode:
                info.external_attr = mode << 16
            z.writestr(info, data)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_zip_import_nested_binary_exec(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip([
        ("SKILL.md", b"---\nname: z\n---\nbody\n", 0),
        ("scripts/run.sh", b"#!/bin/sh\n", 0o755),
        ("assets/logo.png", b"\x89PNG\x00bin", 0),
    ])
    await svc.import_skill_file(data, filename="z.zip", content_type="application/zip")
    d = svc._skill_dir("z")
    assert (d / "scripts" / "run.sh").read_bytes() == b"#!/bin/sh\n"
    assert d.joinpath("scripts", "run.sh").stat().st_mode & stat.S_IXUSR
    assert (d / "assets" / "logo.png").read_bytes() == b"\x89PNG\x00bin"


@pytest.mark.asyncio
async def test_zip_slip_rejected(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip([("SKILL.md", b"body", 0), ("../evil.md", b"x", 0)])
    with pytest.raises(ValueError):
        await svc.import_skill_file(data, filename="z.zip", content_type="application/zip")


# S_IFLNK (0o120000) | 0o777 -- a symlink member's unix mode in external_attr.
_SYMLINK_MODE = 0o120777


@pytest.mark.asyncio
async def test_zip_import_skips_symlink_member(tmp_path):
    """A symlink member is skipped (not materialized, not fatal); real siblings import."""
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip([
        ("SKILL.md", b"---\nname: sym\n---\nbody\n", 0),
        ("evil-link", b"/etc/passwd", _SYMLINK_MODE),
        ("real.txt", b"hi", 0),
    ])
    await svc.import_skill_file(data, filename="sym.zip", content_type="application/zip")
    d = svc._skill_dir("sym")
    assert not (d / "evil-link").exists()
    assert (d / "real.txt").read_bytes() == b"hi"


@pytest.mark.asyncio
async def test_zip_import_rejects_case_fold_collision(tmp_path):
    """Two members differing only in case would collide on a case-insensitive FS."""
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip([
        ("SKILL.md", b"---\nname: casec\n---\nbody\n", 0),
        ("Notes.md", b"a", 0),
        ("notes.md", b"b", 0),
    ])
    with pytest.raises(ValueError, match="case_collision"):
        await svc.import_skill_file(data, filename="casec.zip", content_type="application/zip")


@pytest.mark.asyncio
async def test_zip_import_prunes_junk_members(tmp_path):
    """VCS/build junk (``.git/…``, ``__pycache__/*.pyc``) is pruned; real files land."""
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip([
        ("SKILL.md", b"---\nname: junk\n---\nbody\n", 0),
        (".git/config", b"[core]", 0),
        ("__pycache__/x.pyc", b"\x00", 0),
        ("keep.txt", b"keep", 0),
    ])
    await svc.import_skill_file(data, filename="junk.zip", content_type="application/zip")
    d = svc._skill_dir("junk")
    assert not (d / ".git").exists()
    assert not (d / "__pycache__").exists()
    assert (d / "keep.txt").read_bytes() == b"keep"


@pytest.mark.asyncio
async def test_zip_import_rejects_oversized_member_leaving_no_skill_dir(tmp_path):
    """A member over the 5MB per-file cap is rejected BEFORE any skill dir exists.

    Uses a ~5MB+1 highly-compressible member (single repeated byte) so the zip
    stays tiny and the test is fast -- it exercises the ``member.file_size``
    per-file guard, not a decompression-ratio bomb.
    """
    svc = LocalSkillsService(store_dir=tmp_path)
    big = b"x" * (5 * 1024 * 1024 + 1)
    data = _zip([
        ("SKILL.md", b"---\nname: big\n---\nbody\n", 0),
        ("big.bin", big, 0),
    ])
    with pytest.raises(ValueError, match="too_large"):
        await svc.import_skill_file(data, filename="big.zip", content_type="application/zip")
    assert not svc._skill_dir("big").exists()


@pytest.mark.asyncio
async def test_zip_import_rejects_too_many_members_leaving_no_skill_dir(tmp_path):
    """Exceeding the 500-file count cap is rejected BEFORE any skill dir exists."""
    svc = LocalSkillsService(store_dir=tmp_path)
    members = [("SKILL.md", b"---\nname: many\n---\nbody\n", 0)]
    for i in range(501):
        members.append((f"f{i}.txt", b"x", 0))
    data = _zip(members)
    with pytest.raises(ValueError, match="too_many_files"):
        await svc.import_skill_file(data, filename="many.zip", content_type="application/zip")
    assert not svc._skill_dir("many").exists()


@pytest.mark.asyncio
async def test_zip_import_rejects_non_utf8_body_as_valueerror(tmp_path):
    """A non-UTF-8 SKILL.md body raises a clear ValueError, not a raw decode error."""
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip([("SKILL.md", b"\xff\xfe not utf8", 0)])
    with pytest.raises(ValueError, match="local_skill_invalid_archive"):
        await svc.import_skill_file(data, filename="bad.zip", content_type="application/zip")
