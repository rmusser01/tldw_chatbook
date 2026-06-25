import hashlib
import os
from pathlib import Path

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
    (skill_dir / "binary.md").write_bytes(b"\xff\xfe")
    (skill_dir / "nul.md").write_bytes(b"safe prefix\x00unsafe suffix")
    (skill_dir / "unsafe name.md").write_text("unsafe", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (skill_dir / "linked.md").symlink_to(outside)

    snapshot = scan_skill_directory("demo", skill_dir)

    assert snapshot.unsupported_paths == (
        "SKILL.md.tmp",
        "binary.md",
        "linked.md",
        "nested",
        "nul.md",
        "unsafe name.md",
    )
    assert [item.relative_path for item in snapshot.fingerprints] == ["SKILL.md"]
    assert snapshot.text_files == {"SKILL.md": "# Demo\n"}


def test_scan_skill_directory_rejects_case_variants_and_case_insensitive_temp_suffixes(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (skill_dir / "notes.md.TMP").write_text("partial", encoding="utf-8")

    snapshot = scan_skill_directory("demo", skill_dir)

    assert snapshot.unsupported_paths == ("notes.md.TMP",)
    assert [item.relative_path for item in snapshot.fingerprints] == ["SKILL.md"]
    assert snapshot.text_files == {"SKILL.md": "# Demo\n"}

    for index, reserved_variant in enumerate(("Skill.md", "skill.md")):
        variant_dir = tmp_path / f"reserved-{index}"
        variant_dir.mkdir()
        (variant_dir / reserved_variant).write_text("# Case variant\n", encoding="utf-8")

        variant_snapshot = scan_skill_directory("demo", variant_dir)

        assert variant_snapshot.unsupported_paths == (reserved_variant,)
        assert variant_snapshot.fingerprints == ()
        assert variant_snapshot.text_files == {}


def test_scan_skill_directory_treats_non_regular_and_read_error_paths_as_unsupported(
    tmp_path,
    monkeypatch,
):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (skill_dir / "racy.md").write_text("removed while scanning", encoding="utf-8")
    os.mkfifo(skill_dir / "pipe.md")
    original_read_bytes = Path.read_bytes

    def read_bytes_with_race(path):
        if path.name == "pipe.md":
            raise AssertionError("scanner attempted to read a non-regular path")
        if path.name == "racy.md":
            raise OSError("file disappeared")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes_with_race)

    snapshot = scan_skill_directory("demo", skill_dir)

    assert snapshot.unsupported_paths == ("pipe.md", "racy.md")
    assert [item.relative_path for item in snapshot.fingerprints] == ["SKILL.md"]
    assert snapshot.text_files == {"SKILL.md": "# Demo\n"}
