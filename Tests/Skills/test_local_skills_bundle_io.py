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


def test_apply_supporting_files_rejects_traversal(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    d = svc._skill_dir("demo")
    d.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        svc._apply_supporting_files(d, {"../escape.md": "x"})
