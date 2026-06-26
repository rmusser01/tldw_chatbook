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
    build_skill_trust_marker_store_with_fallback,
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


class FailingMarkerStore:
    def __init__(self, wrapped, *, fail_on_save_number):
        self.wrapped = wrapped
        self.fail_on_save_number = fail_on_save_number
        self.save_count = 0

    def load_marker(self):
        return self.wrapped.load_marker()

    def save_marker(self, *, generation, manifest_digest):
        self.save_count += 1
        if self.save_count == self.fail_on_save_number:
            raise SkillTrustMarkerUnavailable("simulated marker failure")
        self.wrapped.save_marker(generation=generation, manifest_digest=manifest_digest)


class AlwaysFailingMarkerStore:
    def __init__(self, marker):
        self.marker = marker

    def load_marker(self):
        return self.marker

    def save_marker(self, *, generation, manifest_digest):
        raise SkillTrustMarkerUnavailable("original marker failure")


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


def test_trust_store_marker_failure_preserves_previous_manifest_and_marker(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"7" * 32)
    marker = FailingMarkerStore(
        FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        fail_on_save_number=2,
    )
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)
    previous_manifest = {"version": 1, "generation": 1, "skills": {}, "audit": []}
    advanced_manifest = {
        "version": 1,
        "generation": 2,
        "skills": {"demo": {"snapshot_id": "demo-2"}},
        "audit": [],
    }

    store.save_manifest(previous_manifest, keys, salt=b"7" * 32)

    with pytest.raises(SkillTrustMarkerUnavailable, match="simulated marker failure"):
        store.save_manifest(advanced_manifest, keys, salt=b"7" * 32)

    assert store.load_manifest(keys) == previous_manifest
    assert marker.load_marker() == {
        "generation": 1,
        "manifest_digest": store.manifest_digest(previous_manifest),
    }


def test_trust_store_marker_rollback_preserves_original_failure(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"7" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)
    previous_manifest = {"version": 1, "generation": 1, "skills": {}, "audit": []}
    advanced_manifest = {"version": 1, "generation": 2, "skills": {}, "audit": []}

    store.save_manifest(previous_manifest, keys, salt=b"7" * 32)
    store.marker_store = AlwaysFailingMarkerStore({"invalid": "previous-marker"})

    with pytest.raises(SkillTrustMarkerUnavailable, match="original marker failure"):
        store.save_manifest(advanced_manifest, keys, salt=b"7" * 32)

    store.marker_store = marker
    assert store.load_manifest(keys) == previous_manifest


def test_trust_store_rejects_snapshot_directory_symlink_escape(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"7" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store_dir = tmp_path / "trust"
    outside_dir = tmp_path / "outside"
    store_dir.mkdir()
    outside_dir.mkdir()
    (store_dir / "snapshots").symlink_to(outside_dir, target_is_directory=True)
    store = SkillTrustStore(store_dir=store_dir, marker_store=marker)

    with pytest.raises(ValueError, match="unsafe skill trust path"):
        store.save_snapshot("demo-1", {"files": {"SKILL.md": "# Demo"}}, keys, generation=1)

    assert not (outside_dir / "demo-1.json").exists()


def test_file_marker_store_rejects_marker_parent_symlink_escape(tmp_path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    marker_parent = tmp_path / "marker-parent"
    marker_parent.symlink_to(outside_dir, target_is_directory=True)
    marker = FileSkillTrustGenerationMarkerStore(marker_parent / "marker.json")

    with pytest.raises(ValueError, match="unsafe skill trust path"):
        marker.save_marker(generation=1, manifest_digest="digest")

    assert not (outside_dir / "marker.json").exists()


def test_marker_store_builder_falls_back_to_reduced_protection_file_marker(tmp_path):
    marker_store, reduced = build_skill_trust_marker_store_with_fallback(
        fallback_marker_path=tmp_path / "trust" / "marker.json",
        keyring_backend=FakePlaintextKeyring(),
    )

    assert isinstance(marker_store, FileSkillTrustGenerationMarkerStore)
    assert reduced is True
    marker_store.save_marker(generation=1, manifest_digest="digest")
    assert marker_store.load_marker() == {"generation": 1, "manifest_digest": "digest"}


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


def test_trust_store_rejects_non_string_mapping_keys(tmp_path):
    keys = derive_skill_trust_keys("passphrase", salt=b"9" * 32)
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker)

    with pytest.raises(ValueError, match="mapping keys must be strings"):
        store.save_snapshot("demo-1", {"files": {1: "numeric", "1": "string"}}, keys, generation=1)

    assert not (store.snapshots_dir / "demo-1.json").exists()


def test_keyring_generation_marker_store_round_trips_with_secure_backend():
    marker = KeyringSkillTrustGenerationMarkerStore(keyring_backend=FakeSecureKeyring())

    marker.save_marker(generation=3, manifest_digest="digest")

    assert marker.load_marker() == {"generation": 3, "manifest_digest": "digest"}


def test_keyring_generation_marker_store_rejects_insecure_backend():
    with pytest.raises(SkillTrustMarkerUnavailable, match="secure OS-backed"):
        KeyringSkillTrustGenerationMarkerStore(keyring_backend=FakePlaintextKeyring())


def test_keyring_key_cache_round_trips_without_storing_passphrase():
    fake = FakeSecureKeyring()
    keys = derive_skill_trust_keys("passphrase", salt=b"a" * 32)
    cache = KeyringSkillTrustKeyCache(keyring_backend=fake)

    cache.save_keys(keys, salt=b"a" * 32)
    loaded = cache.load_keys(expected_salt=b"a" * 32)

    assert loaded == keys
    assert "passphrase" not in "\n".join(fake.values.values())
    assert repr(loaded).count("redacted") == 4


def test_keyring_key_cache_rejects_stale_salt_binding():
    fake = FakeSecureKeyring()
    keys = derive_skill_trust_keys("passphrase", salt=b"b" * 32)
    cache = KeyringSkillTrustKeyCache(keyring_backend=fake)

    cache.save_keys(keys, salt=b"b" * 32)

    assert cache.load_keys(expected_salt=b"c" * 32) is None
