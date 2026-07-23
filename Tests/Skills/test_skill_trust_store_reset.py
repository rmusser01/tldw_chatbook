from pathlib import Path

from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    KeyringSkillTrustGenerationMarkerStore,
    KeyringSkillTrustKeyCache,
    SkillTrustStore,
    _MARKER_USERNAME,
)


class _FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, s, a):
        return self.store.get((s, a))

    def set_password(self, s, a, v):
        self.store[(s, a)] = v

    def delete_password(self, s, a):
        self.store.pop((s, a), None)


def test_file_marker_clear_is_idempotent(tmp_path):
    store = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store.save_marker(generation=1, manifest_digest="d")
    assert store.load_marker() is not None
    store.clear()
    assert store.load_marker() is None
    store.clear()  # idempotent, no raise
    assert store.load_marker() is None


def test_keyring_marker_clear_deletes_scoped_account(monkeypatch):
    kr = _FakeKeyring()
    monkeypatch.setattr(
        "tldw_chatbook.Skills_Interop.skill_trust_store.is_secure_keyring_backend",
        lambda backend: True,
    )
    store = KeyringSkillTrustGenerationMarkerStore(keyring_backend=kr, account_scope="aaaa")
    store.save_marker(generation=1, manifest_digest="d")
    store.clear()
    assert store.load_marker() is None
    assert kr.store == {}


def test_keyring_key_cache_clear(monkeypatch):
    kr = _FakeKeyring()
    monkeypatch.setattr(
        "tldw_chatbook.Skills_Interop.skill_trust_store.is_secure_keyring_backend",
        lambda backend: True,
    )
    cache = KeyringSkillTrustKeyCache(keyring_backend=kr, account_scope="aaaa")
    kr.set_password("tldw_chatbook.skill_trust.keys", cache._account, "payload")
    cache.clear()
    assert kr.store == {}


def test_store_delete_manifest_removes_file_and_snapshots(tmp_path):
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json")
    store = SkillTrustStore(store_dir=tmp_path, marker_store=marker)
    store.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    store.manifest_path.write_text("{}", encoding="utf-8")
    store.snapshots_dir.mkdir(parents=True, exist_ok=True)
    (store.snapshots_dir / "snap").write_text("x", encoding="utf-8")
    store.delete_manifest()
    assert not store.manifest_path.exists()
    assert not store.snapshots_dir.exists()
    store.delete_manifest()  # idempotent
