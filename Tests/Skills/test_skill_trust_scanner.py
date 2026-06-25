import hashlib

from tldw_chatbook.Skills_Interop.skill_trust_models import SkillFileFingerprint
from tldw_chatbook.Skills_Interop.skill_trust_scanner import scan_skill_directory


def test_scan_skill_directory_fingerprints_skill_and_supporting_text_in_sorted_order(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    skill_bytes = b"# Demo\nUse safely.\n"
    notes_bytes = b"Trusted notes.\n"
    (skill_dir / "notes.md").write_bytes(notes_bytes)
    (skill_dir / "SKILL.md").write_bytes(skill_bytes)

    snapshot = scan_skill_directory("demo", skill_dir)

    assert snapshot.skill_name == "demo"
    assert snapshot.fingerprints == (
        SkillFileFingerprint(
            relative_path="SKILL.md",
            file_type="skill",
            byte_length=len(skill_bytes),
            sha256=hashlib.sha256(skill_bytes).hexdigest(),
        ),
        SkillFileFingerprint(
            relative_path="notes.md",
            file_type="supporting_text",
            byte_length=len(notes_bytes),
            sha256=hashlib.sha256(notes_bytes).hexdigest(),
        ),
    )
    assert snapshot.text_files["SKILL.md"] == skill_bytes.decode("utf-8")
    assert snapshot.text_files["notes.md"] == notes_bytes.decode("utf-8")
    assert snapshot.unsupported_paths == ()


def test_scan_skill_directory_marks_unsupported_paths_without_text_or_fingerprints(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (skill_dir / "SKILL.md.tmp").write_text("partial", encoding="utf-8")
    nested_dir = skill_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "ignored.md").write_text("ignored", encoding="utf-8")
    (skill_dir / "binary.dat").write_bytes(b"\xff\xfe\x00")
    (skill_dir / "unsafe name.md").write_text("unsafe", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (skill_dir / "linked.md").symlink_to(outside)

    snapshot = scan_skill_directory("demo", skill_dir)

    assert snapshot.unsupported_paths == (
        "SKILL.md.tmp",
        "binary.dat",
        "linked.md",
        "nested",
        "unsafe name.md",
    )
    assert [item.relative_path for item in snapshot.fingerprints] == ["SKILL.md"]
    assert snapshot.text_files == {"SKILL.md": "# Demo\n"}
