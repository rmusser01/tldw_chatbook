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
    """Junk pruning is ISOLATED from the name-pattern reject.

    Both members here have names the supporting-file NAME PATTERN would
    otherwise ACCEPT, so their absence proves ``_is_junk`` /
    ``SUPPORTING_JUNK_DIRS`` specifically (not the path validator):
      * ``notes.pyc`` -- top-level, pattern-valid, dropped by ``_is_junk``
        (``.pyc`` suffix).
      * ``node_modules/lib.js`` -- pattern-valid path, dropped because
        ``node_modules`` is in ``SUPPORTING_JUNK_DIRS``.
    """
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip([
        ("SKILL.md", b"---\nname: junk\n---\nbody\n", 0),
        ("notes.pyc", b"\x00", 0),
        ("node_modules/lib.js", b"x=1", 0),
        ("keep.txt", b"keep", 0),
    ])
    await svc.import_skill_file(data, filename="junk.zip", content_type="application/zip")
    d = svc._skill_dir("junk")
    assert not (d / "notes.pyc").exists()
    assert not (d / "node_modules").exists()
    assert (d / "keep.txt").read_bytes() == b"keep"


@pytest.mark.asyncio
async def test_zip_import_streams_and_rejects_member_whose_actual_content_exceeds_cap(
    tmp_path,
):
    """A member whose ACTUAL content exceeds the 5MB per-file cap is rejected
    BEFORE any skill dir exists.

    Uses a ~5MB+1 highly-compressible member (single repeated byte) so the zip
    stays tiny and the test is fast/CI-safe -- honest oversize content, no
    decompression-ratio bomb. The rejection surfaces as the standard
    ``ValueError`` contract (the ``file_size`` fast-path or the streaming
    cumulative abort) and leaves no partial skill on disk.
    """
    svc = LocalSkillsService(store_dir=tmp_path)
    big = b"x" * (5 * 1024 * 1024 + 1)
    data = _zip([
        ("SKILL.md", b"---\nname: big\n---\nbody\n", 0),
        ("big.bin", big, 0),
    ])
    with pytest.raises(ValueError, match="local_skill_file_too_large|corrupt_member"):
        await svc.import_skill_file(data, filename="big.zip", content_type="application/zip")
    assert not svc._skill_dir("big").exists()


def _zip_with_understated_file_size(*, declared: int = 5) -> bytes:
    """Build a zip whose ``big.bin`` central-directory ``file_size`` LIES.

    The member is written with real (larger) content, then its ZipInfo's
    ``file_size`` is mutated smaller before ``close()`` writes the central
    directory -- so ``infolist()`` reports the understated size (defeating a
    naive ``member.file_size`` pre-check) while the actual DEFLATE payload is
    bigger. Reading such a member trips a CRC mismatch (``BadZipFile``); the
    service must surface that as a ``ValueError``, never a raw ``BadZipFile``.
    """
    buf = io.BytesIO()
    z = zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED)
    z.writestr("SKILL.md", b"---\nname: forged\n---\nbody\n")
    info = zipfile.ZipInfo("big.bin")
    z.writestr(info, b"y" * 4096)   # local header + data written with real size
    info.file_size = declared        # mutate BEFORE close -> central dir lies
    z.close()
    return buf.getvalue()


@pytest.mark.asyncio
async def test_zip_import_wraps_forged_understated_member_as_valueerror(tmp_path):
    """A member with a forged/understated ``file_size`` header (a lying zip that
    a pre-check cannot catch) surfaces as the standard ``ValueError`` contract,
    not a raw ``zipfile.BadZipFile``, and leaves no skill dir behind."""
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip_with_understated_file_size()
    with pytest.raises(ValueError, match="corrupt_member|local_skill_file_too_large"):
        await svc.import_skill_file(
            data, filename="forged.zip", content_type="application/zip"
        )
    assert not svc._skill_dir("forged").exists()


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
