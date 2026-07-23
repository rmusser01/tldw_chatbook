import secrets, pytest
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore, SkillTrustStore,
)


def _svc(tmp_path):
    (tmp_path / "trust").mkdir(exist_ok=True)
    (tmp_path / "skills").mkdir(exist_ok=True)
    return SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "trust" / "m.json"),
        ),
        key_cache=None,
    )


def test_verify_content_tolerates_binary_and_exec(tmp_path):
    import os, stat
    d = tmp_path / "skills" / "demo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("body\n", encoding="utf-8")
    (d / "assets").mkdir()
    (d / "assets" / "logo.png").write_bytes(b"\x89PNG\x00bin")   # binary → in manifest, not in text view
    (d / "scripts").mkdir()
    (d / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(d / "scripts" / "run.sh", 0o755)                    # executable text
    svc = _svc(tmp_path)
    svc.bootstrap_trust("pw", salt=secrets.token_bytes(32))      # trusts the full bundle
    # verify_skill_content receives the TEXT view (no binary; run.sh present but mode unknown).
    svc.verify_skill_content(
        "demo",
        skill_content="body\n",
        supporting_files={"scripts/run.sh": "#!/bin/sh\n"},
    )   # must NOT raise (was raising: assets/logo.png "missing", scripts/run.sh "modified")


def test_verify_content_still_catches_modified_text_body(tmp_path):
    d = tmp_path / "skills" / "demo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("body\n", encoding="utf-8")
    svc = _svc(tmp_path)
    svc.bootstrap_trust("pw", salt=secrets.token_bytes(32))
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustBlockedError
    with pytest.raises(SkillTrustBlockedError):
        svc.verify_skill_content("demo", skill_content="TAMPERED\n", supporting_files=None)
