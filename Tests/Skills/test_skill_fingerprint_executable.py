from tldw_chatbook.Skills_Interop.skill_trust_models import SkillFileFingerprint


def test_manifest_entry_omits_executable_when_false():
    fp = SkillFileFingerprint(
        relative_path="notes.md", file_type="supporting_text", byte_length=3, sha256="ab",
    )
    # BACKWARD-COMPAT: byte-identical to the pre-change 4-key dict.
    assert fp.as_manifest_entry() == {
        "relative_path": "notes.md", "file_type": "supporting_text",
        "byte_length": 3, "sha256": "ab",
    }


def test_manifest_entry_includes_executable_when_true():
    fp = SkillFileFingerprint(
        relative_path="scripts/run.sh", file_type="supporting_text",
        byte_length=3, sha256="ab", executable=True,
    )
    assert fp.as_manifest_entry()["executable"] is True
