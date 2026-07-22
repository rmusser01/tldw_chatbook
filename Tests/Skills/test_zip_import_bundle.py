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
