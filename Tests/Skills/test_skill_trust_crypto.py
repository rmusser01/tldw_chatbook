import pytest

from tldw_chatbook.Skills_Interop import SkillTrustBlockedError, SkillTrustStatus
from tldw_chatbook.Skills_Interop.skill_trust_crypto import (
    canonical_json,
    decrypt_json_blob,
    derive_skill_trust_keys,
    encrypt_json_blob,
    manifest_mac,
    sha256_hex,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import (
    TRUST_STATUS_TRUSTED,
    SkillDirectorySnapshot,
    SkillFileFingerprint,
)


def test_skill_trust_key_derivation_separates_key_purposes():
    keys = derive_skill_trust_keys("correct horse battery staple", salt=b"0" * 32)

    assert len(keys.manifest_mac_key) == 32
    assert len(keys.snapshot_key) == 32
    assert len(keys.audit_mac_key) == 32
    assert len(keys.wrapped_root_key) == 32
    assert keys.manifest_mac_key != keys.snapshot_key
    assert keys.manifest_mac_key != keys.audit_mac_key
    assert keys.snapshot_key != keys.audit_mac_key
    assert keys.wrapped_root_key not in {
        keys.manifest_mac_key,
        keys.snapshot_key,
        keys.audit_mac_key,
    }


def test_skill_trust_key_derivation_requires_32_byte_salt():
    with pytest.raises(ValueError, match="salt must be 32 bytes"):
        derive_skill_trust_keys("passphrase", salt=b"short")


def test_canonical_json_and_sha256_are_deterministic():
    first = canonical_json({"b": [2, 1], "a": {"z": True}})
    second = canonical_json({"a": {"z": True}, "b": [2, 1]})

    assert first == second
    assert first == b'{"a":{"z":true},"b":[2,1]}'
    assert sha256_hex(b"trusted skill") == (
        "c595ac6946639cb66901b97a07d21806dadbad0c5755d7a039385b750a6cfc65"
    )


def test_manifest_mac_rejects_tampered_manifest_payload():
    keys = derive_skill_trust_keys("passphrase", salt=b"1" * 32)
    manifest = {"version": 1, "generation": 1, "skills": {"demo": {"status": "trusted"}}}
    tag = manifest_mac(manifest, keys.manifest_mac_key)

    tampered = {"version": 1, "generation": 2, "skills": {"demo": {"status": "trusted"}}}

    assert manifest_mac(manifest, keys.manifest_mac_key) == tag
    assert manifest_mac(tampered, keys.manifest_mac_key) != tag


def test_snapshot_encryption_round_trips_and_authenticates_associated_data():
    keys = derive_skill_trust_keys("passphrase", salt=b"2" * 32)
    payload = {"files": {"SKILL.md": "# Demo\nTrusted"}}
    encrypted = encrypt_json_blob(payload, keys.snapshot_key, associated_data=b"demo:generation:1")

    assert encrypted["alg"] == "AES-256-GCM"
    assert (
        decrypt_json_blob(encrypted, keys.snapshot_key, associated_data=b"demo:generation:1")
        == payload
    )

    with pytest.raises(ValueError, match="snapshot authentication failed"):
        decrypt_json_blob(encrypted, keys.snapshot_key, associated_data=b"demo:generation:2")


def test_snapshot_decryption_rejects_non_object_payloads():
    keys = derive_skill_trust_keys("passphrase", salt=b"3" * 32)
    encrypted = encrypt_json_blob(  # type: ignore[arg-type]
        ["not", "an", "object"],
        keys.snapshot_key,
        associated_data=b"demo:generation:1",
    )

    with pytest.raises(ValueError, match="snapshot payload must be an object"):
        decrypt_json_blob(encrypted, keys.snapshot_key, associated_data=b"demo:generation:1")


def test_trust_models_produce_json_safe_response_fields():
    fingerprint = SkillFileFingerprint(
        relative_path="SKILL.md",
        file_type="text/markdown",
        byte_length=12,
        sha256="abc123",
    )
    snapshot = SkillDirectorySnapshot(
        skill_name="demo",
        fingerprints=(fingerprint,),
        text_files={"SKILL.md": "# Demo"},
    )
    status = SkillTrustStatus(
        skill_name="demo",
        trust_status=TRUST_STATUS_TRUSTED,
        trust_reason_code=None,
        trust_blocked=False,
        changed_files=("SKILL.md",),
        manifest_generation=3,
        last_verified_at="2026-06-25T00:00:00+00:00",
    )

    assert fingerprint.as_manifest_entry() == {
        "relative_path": "SKILL.md",
        "file_type": "text/markdown",
        "byte_length": 12,
        "sha256": "abc123",
    }
    assert snapshot.fingerprint_map == {"SKILL.md": fingerprint}
    assert status.response_fields() == {
        "trust_status": "trusted",
        "trust_reason_code": None,
        "trust_blocked": False,
        "trust_changed_files": ["SKILL.md"],
        "trust_manifest_generation": 3,
        "trust_last_verified_at": "2026-06-25T00:00:00+00:00",
    }


def test_trust_blocked_error_carries_safe_fields():
    error = SkillTrustBlockedError(
        skill_name="demo",
        reason_code="skill_modified",
        trust_status="quarantined_modified",
        changed_files=("SKILL.md",),
    )

    assert str(error) == "Local skill demo is trust-blocked: skill_modified"
    assert error.skill_name == "demo"
    assert error.reason_code == "skill_modified"
    assert error.trust_status == "quarantined_modified"
    assert error.changed_files == ("SKILL.md",)
