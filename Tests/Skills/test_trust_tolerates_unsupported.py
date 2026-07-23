import secrets

from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
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


def test_bootstrap_does_not_raise_on_residual_unsupported(tmp_path):
    d = tmp_path / "skills" / "demo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("body\n", encoding="utf-8")
    (d / "good.md").write_text("ok\n", encoding="utf-8")
    (d / "link.md").symlink_to(d / "good.md")   # residual unsupported (symlink)
    svc = _svc(tmp_path)
    svc.bootstrap_trust("pw", salt=secrets.token_bytes(32))   # must NOT raise
    status = svc.status_for_skill("demo")
    assert status.trust_status == "quarantined_unsupported_path"  # surfaced, recoverable
    # Removing the residual clears it.
    (d / "link.md").unlink()
    svc.trust_current_skill("demo")   # must NOT raise
    assert svc.status_for_skill("demo").trust_status == "trusted"
