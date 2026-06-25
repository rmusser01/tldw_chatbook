import base64
import json

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    KeyringSkillTrustKeyCache,
    SkillTrustStore,
)


class FakeSecureKeyring:
    __module__ = "keyring.backends.macOS"
    priority = 1

    def __init__(self):
        self.values = {}

    def get_password(self, service_name, username):
        return self.values.get((service_name, username))

    def set_password(self, service_name, username, password):
        self.values[(service_name, username)] = password


def _service(tmp_path, passphrase="passphrase", key_cache=None):
    skills_dir = tmp_path / "skills"
    trust_store = SkillTrustStore(
        store_dir=tmp_path / "trust",
        marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
    )
    service = SkillTrustService(
        skills_dir=skills_dir,
        trust_store=trust_store,
        key_cache=key_cache,
    )
    service.unlock_with_passphrase(passphrase, salt=b"6" * 32)
    return service, skills_dir


def _write_skill(skills_dir, name="demo", content="# Demo\n"):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def test_uninitialized_service_blocks_until_bootstrap(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)

    status = service.status_for_skill("demo")

    assert status.trust_status == "trust_uninitialized"
    assert status.trust_blocked is True
    assert service.overall_status() == "trust_uninitialized"
    assert service.keyring_convenience_enabled is False
    assert service.reduced_rollback_protection is False
    with pytest.raises(SkillTrustBlockedError, match="trust_uninitialized"):
        service.ensure_skill_trusted("demo")


def test_bootstrap_trusts_current_files_and_detects_modification(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)

    service.bootstrap_trust()
    trusted = service.status_for_skill("demo")
    assert trusted.trust_status == "trusted"
    assert service.overall_status() == "trusted"
    service.ensure_skill_trusted("demo")

    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nBackdoor\n", encoding="utf-8")
    modified = service.status_for_skill("demo")

    assert modified.trust_status == "quarantined_modified"
    assert modified.changed_files == ("SKILL.md",)
    with pytest.raises(SkillTrustBlockedError, match="skill_modified"):
        service.ensure_skill_trusted("demo")


def test_existing_manifest_without_unlock_reports_locked_status(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()

    locked_service = SkillTrustService(
        skills_dir=skills_dir,
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )
    status = locked_service.status_for_skill("demo")

    assert status.trust_status == "trust_locked"
    assert status.trust_blocked is True
    assert locked_service.overall_status() == "trust_locked"


def test_missing_marker_reports_global_manifest_error(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    (tmp_path / "marker.json").unlink()

    status = service.status_for_skill("demo")

    assert status.trust_status == "quarantined_manifest_error"
    assert status.trust_reason_code == "rollback_marker_unavailable"
    assert status.trust_blocked is True
    assert service.overall_status() == "quarantined_manifest_error"


def test_invalid_manifest_reports_blocked_status_without_raising(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    manifest_path = tmp_path / "trust" / "skill_trust_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["manifest"]["generation"] = 2
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    status = service.status_for_skill("demo")

    assert status.trust_status == "quarantined_manifest_error"
    assert status.trust_reason_code == "manifest_invalid"
    assert status.trust_blocked is True
    with pytest.raises(SkillTrustBlockedError, match="manifest_invalid"):
        service.ensure_skill_trusted("demo")


def test_review_approval_requires_live_files_to_match_reviewed_snapshot(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nChanged\n", encoding="utf-8")

    review = service.capture_review("demo")
    assert review["changed_files"] == ["SKILL.md"]
    assert review["current_files"] == {"SKILL.md": "# Demo\nChanged\n"}
    (skills_dir / "demo" / "SKILL.md").write_text(
        "# Demo\nChanged again\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="snapshot_mismatch"):
        service.trust_reviewed_snapshot(review["review_id"])


def test_review_approval_restores_trust_for_reviewed_snapshot(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nChanged\n", encoding="utf-8")

    review = service.capture_review("demo")
    service.trust_reviewed_snapshot(review["review_id"])

    status = service.status_for_skill("demo")
    assert status.trust_status == "trusted"
    assert status.trust_blocked is False
    service.ensure_skill_trusted("demo")


def test_keyring_convenience_loads_only_salt_bound_cached_keys(tmp_path):
    fake_keyring = FakeSecureKeyring()
    key_cache = KeyringSkillTrustKeyCache(keyring_backend=fake_keyring)
    service, skills_dir = _service(tmp_path, key_cache=key_cache)
    _write_skill(skills_dir)
    service.bootstrap_trust()

    service.enable_keyring_convenience()

    cached_service = SkillTrustService(
        skills_dir=skills_dir,
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
        key_cache=key_cache,
    )
    assert cached_service.unlock_from_keyring_convenience() is True
    assert cached_service.keyring_convenience_enabled is True
    assert cached_service.status_for_skill("demo").trust_status == "trusted"

    payload = json.loads((tmp_path / "trust" / "skill_trust_manifest.json").read_text())
    payload["kdf_salt"] = base64.b64encode(b"7" * 32).decode("ascii")
    (tmp_path / "trust" / "skill_trust_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    stale_service = SkillTrustService(
        skills_dir=skills_dir,
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
        key_cache=key_cache,
    )

    assert stale_service.unlock_from_keyring_convenience() is False
    assert stale_service.status_for_skill("demo").trust_status == "trust_locked"
