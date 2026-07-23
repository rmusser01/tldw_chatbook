import os

import pytest

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.mark.asyncio
async def test_get_skill_with_nested_and_binary_never_raises(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    await svc.create_skill(name="demo", content="---\nname: demo\n---\nbody\n")
    d = svc._skill_dir("demo")
    (d / "references").mkdir(parents=True, exist_ok=True)
    (d / "references" / "api.md").write_text("# api\n", encoding="utf-8")
    (d / "assets").mkdir(parents=True, exist_ok=True)
    (d / "assets" / "logo.png").write_bytes(b"\x89PNG\x00binary")
    skill = await svc.get_skill("demo")      # must NOT raise
    assert skill["supporting_files"]["references/api.md"] == "# api\n"
    assert "assets/logo.png" not in skill["supporting_files"]   # binary excluded from text view
    paths = {b["path"]: b for b in skill["bundle_files"]}
    assert paths["assets/logo.png"]["is_text"] is False
    assert paths["references/api.md"]["is_text"] is True


@pytest.mark.asyncio
async def test_get_skill_with_toplevel_binary_never_raises(tmp_path):
    # Pin the literal bug this task fixes: a TOP-LEVEL binary sibling of
    # SKILL.md must not crash get_skill (previously raised UnicodeDecodeError).
    svc = LocalSkillsService(store_dir=tmp_path)
    await svc.create_skill(name="demo", content="---\nname: demo\n---\nbody\n")
    d = svc._skill_dir("demo")
    (d / "logo.png").write_bytes(b"\x89PNG\x00bin")
    skill = await svc.get_skill("demo")      # must NOT raise
    assert "logo.png" not in (skill["supporting_files"] or {})
    paths = {b["path"]: b for b in skill["bundle_files"]}
    assert paths["logo.png"]["is_text"] is False


def test_apply_supporting_files_rejects_traversal(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    d = svc._skill_dir("demo")
    d.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        svc._apply_supporting_files(d, {"../escape.md": "x"})


def test_apply_supporting_files_rejects_symlink_escape(tmp_path):
    # Exercise ONLY the containment guard: every path segment ("link",
    # "evil.md") is individually valid, so validate_supporting_file_path does
    # NOT reject the key. The write still escapes skill_dir because "link" is a
    # symlink to a directory outside it. This must be caught by the resolve()
    # within-skill_dir containment block, not by segment validation.
    svc = LocalSkillsService(store_dir=tmp_path)
    skill_dir = svc._skill_dir("demo")
    skill_dir.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(outside_dir, skill_dir / "link")
    with pytest.raises(ValueError):
        svc._apply_supporting_files(skill_dir, {"link/evil.md": "x"})
