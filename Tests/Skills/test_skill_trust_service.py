import base64
import json
import shutil

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    KeyringSkillTrustKeyCache,
    SkillTrustStore,
    UnavailableSkillTrustGenerationMarkerStore,
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


def test_verify_skill_content_accepts_exact_trusted_content_and_supporting_files(tmp_path):
    service, skills_dir = _service(tmp_path)
    skill_dir = _write_skill(skills_dir, content="# Demo\nRender {{args}}\n")
    (skill_dir / "notes.md").write_text("trusted notes\n", encoding="utf-8")
    service.bootstrap_trust()

    service.verify_skill_content(
        "demo",
        skill_content="# Demo\nRender {{args}}\n",
        supporting_files={"notes.md": "trusted notes\n"},
    )


def test_verify_skill_content_rejects_in_memory_content_not_matching_trusted_manifest(tmp_path):
    service, skills_dir = _service(tmp_path)
    skill_dir = _write_skill(skills_dir, content="# Demo\nRender {{args}}\n")
    (skill_dir / "notes.md").write_text("trusted notes\n", encoding="utf-8")
    service.bootstrap_trust()

    with pytest.raises(SkillTrustBlockedError, match="skill_modified") as exc:
        service.verify_skill_content(
            "demo",
            skill_content="# Demo\nMALICIOUS {{args}}\n",
            supporting_files={"notes.md": "trusted notes\n"},
        )

    assert exc.value.trust_status == "quarantined_modified"
    assert exc.value.changed_files == ("SKILL.md",)


def test_verify_skill_content_rejects_extra_and_missing_supporting_files(tmp_path):
    service, skills_dir = _service(tmp_path)
    skill_dir = _write_skill(skills_dir, content="# Demo\nRender {{args}}\n")
    (skill_dir / "notes.md").write_text("trusted notes\n", encoding="utf-8")
    service.bootstrap_trust()

    with pytest.raises(SkillTrustBlockedError, match="skill_deleted") as missing:
        service.verify_skill_content(
            "demo",
            skill_content="# Demo\nRender {{args}}\n",
            supporting_files=None,
        )
    assert missing.value.changed_files == ("notes.md",)

    with pytest.raises(SkillTrustBlockedError, match="skill_added") as added:
        service.verify_skill_content(
            "demo",
            skill_content="# Demo\nRender {{args}}\n",
            supporting_files={"extra.md": "extra\n", "notes.md": "trusted notes\n"},
        )
    assert added.value.changed_files == ("extra.md",)


def test_status_for_unsafe_skill_name_returns_blocked_without_scanning_outside(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    outside = tmp_path / "outside"
    _write_skill(tmp_path, name="outside", content="# Secret\n")
    service.bootstrap_trust()

    traversal_status = service.status_for_skill("../outside")
    absolute_status = service.status_for_skill(str(outside.resolve()))

    assert traversal_status.trust_status == "quarantined_unsupported_path"
    assert traversal_status.trust_reason_code == "unsupported_path"
    assert traversal_status.trust_blocked is True
    assert absolute_status.trust_status == "quarantined_unsupported_path"
    with pytest.raises(SkillTrustBlockedError, match="unsupported_path"):
        service.ensure_skill_trusted("../outside")
    with pytest.raises(ValueError, match="unsupported_path"):
        service.capture_review("../outside")
    with pytest.raises(ValueError, match="unsupported_path"):
        service.capture_review(str(outside.resolve()))


def test_bootstrap_rejects_root_skill_directory_symlink(tmp_path):
    service, skills_dir = _service(tmp_path)
    outside_dir = tmp_path / "outside-demo"
    outside_dir.mkdir()
    (outside_dir / "SKILL.md").write_text("# Outside\n", encoding="utf-8")
    skills_dir.mkdir(parents=True)
    (skills_dir / "demo").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="unsupported_path"):
        service.bootstrap_trust()


def test_root_skill_directory_symlink_is_quarantined_after_bootstrap(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    shutil.rmtree(skills_dir / "demo")
    outside_dir = tmp_path / "outside-demo"
    outside_dir.mkdir()
    (outside_dir / "SKILL.md").write_text("# Outside\n", encoding="utf-8")
    (skills_dir / "demo").symlink_to(outside_dir, target_is_directory=True)

    status = service.status_for_skill("demo")

    assert status.trust_status == "quarantined_unsupported_path"
    assert status.trust_reason_code == "unsupported_path"
    assert status.changed_files == ("demo",)


def test_added_file_is_quarantined(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    (skills_dir / "demo" / "notes.md").write_text("new support text", encoding="utf-8")

    status = service.status_for_skill("demo")

    assert status.trust_status == "quarantined_added"
    assert status.trust_reason_code == "skill_added"
    assert status.changed_files == ("notes.md",)


def test_deleted_file_is_quarantined(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    (skills_dir / "demo" / "SKILL.md").unlink()

    status = service.status_for_skill("demo")

    assert status.trust_status == "quarantined_deleted"
    assert status.trust_reason_code == "skill_deleted"
    assert status.changed_files == ("SKILL.md",)


def test_unsupported_child_path_is_quarantined(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    (skills_dir / "demo" / "unsafe name.md").write_text("unsafe", encoding="utf-8")

    status = service.status_for_skill("demo")

    assert status.trust_status == "quarantined_unsupported_path"
    assert status.trust_reason_code == "unsupported_path"
    assert status.changed_files == ("unsafe name.md",)


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


def test_marker_mismatch_reports_global_manifest_error(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    service.trust_store.marker_store.save_marker(generation=99, manifest_digest="old")

    status = service.status_for_skill("demo")

    assert status.trust_status == "quarantined_manifest_error"
    assert status.trust_reason_code == "rollback_marker_unavailable"
    assert status.trust_blocked is True
    assert service.overall_status() == "quarantined_manifest_error"


def test_marker_store_unavailable_reports_global_manifest_error(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    unavailable_service = SkillTrustService(
        skills_dir=skills_dir,
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=UnavailableSkillTrustGenerationMarkerStore("offline"),
        ),
    )
    unavailable_service.unlock_with_passphrase("passphrase")

    status = unavailable_service.status_for_skill("demo")

    assert status.trust_status == "quarantined_manifest_error"
    assert status.trust_reason_code == "rollback_marker_unavailable"
    assert status.trust_blocked is True
    assert unavailable_service.overall_status() == "quarantined_manifest_error"


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


def test_overall_status_reports_quarantine_when_live_skill_is_modified(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nChanged\n", encoding="utf-8")

    assert service.overall_status() == "quarantined_modified"


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
    assert review["review_id"] not in service._reviews


def test_capture_review_does_not_retain_file_contents_in_pending_review(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    secret_text = "# Demo\nSECRET REVIEW TEXT\n"
    (skills_dir / "demo" / "SKILL.md").write_text(secret_text, encoding="utf-8")

    review = service.capture_review("demo")
    stored_review = service._reviews[review["review_id"]]

    assert review["current_files"] == {"SKILL.md": secret_text}
    assert set(stored_review) == {
        "review_id",
        "skill_name",
        "manifest_generation",
        "current_digest",
        "changed_files",
    }
    assert secret_text not in json.dumps(stored_review)


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
    assert review["review_id"] not in service._reviews
    service.ensure_skill_trusted("demo")


def test_review_approval_rejects_stale_manifest_generation(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    _write_skill(skills_dir, name="other", content="# Other\n")
    service.bootstrap_trust()
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nChanged\n", encoding="utf-8")
    review = service.capture_review("demo")
    (skills_dir / "other" / "SKILL.md").write_text("# Other\nChanged\n", encoding="utf-8")
    service.trust_current_skill("other")

    with pytest.raises(ValueError, match="snapshot_mismatch"):
        service.trust_reviewed_snapshot(review["review_id"])
    assert review["review_id"] not in service._reviews


def test_discard_review_removes_captured_review(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    (skills_dir / "demo" / "SKILL.md").write_text("# Demo\nChanged\n", encoding="utf-8")
    review = service.capture_review("demo")

    service.discard_review(review["review_id"])

    with pytest.raises(KeyError):
        service.trust_reviewed_snapshot(review["review_id"])


def test_trust_current_skill_rejects_missing_skill(tmp_path):
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()

    with pytest.raises(ValueError, match="skill_deleted"):
        service.trust_current_skill("missing")


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
