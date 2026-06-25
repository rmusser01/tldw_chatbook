import json
from types import MappingProxyType

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_crypto import derive_skill_trust_keys
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    KeyringSkillTrustGenerationMarkerStore,
    KeyringSkillTrustKeyCache,
    SkillTrustMarkerUnavailable,
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


class FakePlaintextKeyring(FakeSecureKeyring):
    __module__ = "keyring.backends.file"


def test_trust_store_round_trips_manifest_snapshot_marker_and_salt(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"3" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    manifest = {
        "version": 1,
        "generation": 1,
        "skills": {
            "demo": {
                "files": [
                    {
                        "relative_path": "SKILL.md",
                        "file_type": "skill",
                        "byte_length": 6,
                        "sha256": "abc",
                    }
                ],
                "snapshot_id": "demo-1",
                "trusted_at": "2026-06-25T00:00:00+00:00",
            }
        },
        "audit": [],
    }

    store.save_manifest(manifest, keys, salt=b"3" * 32)
    store.save_snapshot("demo-1", {"files": {"SKILL.md": "# Demo"}}, keys, generation=1)

    loaded = store.load_manifest(keys)
    snapshot = store.load_snapshot("demo-1", keys, generation=1)

    assert store.has_manifest() is True
    assert store.manifest_path == tmp_path / "trust" / "skill_trust_manifest.json"
    assert store.snapshots_dir == tmp_path / "trust" / "snapshots"
    assert loaded == manifest
    assert snapshot == {"files": {"SKILL.md": "# Demo"}}
    assert store.load_salt() == b"3" * 32
    assert marker.load_marker() == {
        "generation": 1,
        "manifest_digest": store.manifest_digest(loaded),
    }


def test_trust_store_rejects_tampered_manifest(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"4" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    store.save_manifest({"version": 1, "generation": 1, "skills": {}, "audit": []}, keys, salt=b"4" * 32)
    payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    payload["manifest"]["generation"] = 2
    store.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest authentication failed"):
        store.load_manifest(keys)


def test_trust_store_rejects_marker_mismatch(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"5" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    store.save_manifest({"version": 1, "generation": 3, "skills": {}, "audit": []}, keys, salt=b"5" * 32)
    marker.save_marker(generation=2, manifest_digest="old")

    with pytest.raises(ValueError, match="manifest generation marker mismatch"):
        store.load_manifest(keys)


def test_trust_store_rejects_missing_marker_after_manifest_exists(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"6" * 32)
    marker_path = tmp_path / "marker.json"
    marker = FileSkillTrustGenerationMarkerStore(marker_path)
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    store.save_manifest({"version": 1, "generation": 1, "skills": {}, "audit": []}, keys, salt=b"6" * 32)
    marker_path.unlink()

    with pytest.raises(ValueError, match="manifest generation marker mismatch"):
        store.load_manifest(keys)


def test_trust_store_load_salt_requires_32_bytes(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"7" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    store.save_manifest({"version": 1, "generation": 1, "skills": {}, "audit": []}, keys, salt=b"7" * 32)
    payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    payload["kdf_salt"] = "c2hvcnQ="
    store.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="skill trust salt invalid"):
        store.load_salt()


def test_trust_store_snapshot_accepts_immutable_mapping_payloads(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"8" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)
    payload = {"files": MappingProxyType({"SKILL.md": "# Demo"})}

    store.save_snapshot("demo-1", payload, keys, generation=1)

    assert store.load_snapshot("demo-1", keys, generation=1) == {"files": {"SKILL.md": "# Demo"}}


def test_keyring_generation_marker_store_round_trips_with_secure_backend():
    marker = KeyringSkillTrustGenerationMarkerStore(keyring_backend=FakeSecureKeyring())

    marker.save_marker(generation=3, manifest_digest="digest")

    assert marker.load_marker() == {"generation": 3, "manifest_digest": "digest"}


def test_keyring_generation_marker_store_rejects_insecure_backend():
    with pytest.raises(SkillTrustMarkerUnavailable, match="secure OS-backed"):
        KeyringSkillTrustGenerationMarkerStore(keyring_backend=FakePlaintextKeyring())


def test_keyring_key_cache_round_trips_without_storing_passphrase():
    fake = FakeSecureKeyring()
    keys = derive_skill_trust_keys("passphrase", salt=b"9" * 32)
    cache = KeyringSkillTrustKeyCache(keyring_backend=fake)

    cache.save_keys(keys)
    loaded = cache.load_keys()

    assert loaded == keys
    assert "passphrase" not in "\n".join(fake.values.values())
    assert repr(loaded).count("redacted") == 4
