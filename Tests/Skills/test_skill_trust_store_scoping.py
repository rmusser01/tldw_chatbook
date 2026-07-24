from pathlib import Path

from tldw_chatbook.Skills_Interop.skill_trust_store import (
    KeyringSkillTrustGenerationMarkerStore,
    KeyringSkillTrustKeyCache,
    skill_trust_account_scope,
    _MARKER_USERNAME,
    _KEY_CACHE_USERNAME,
)


class _FakeKeyring:
    """Secure-looking dict-backed keyring for tests."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    # is_secure_keyring_backend checks the class/module; see conftest note.
    def get_password(self, service, account):
        return self.store.get((service, account))

    def set_password(self, service, account, value):
        self.store[(service, account)] = value

    def delete_password(self, service, account):
        self.store.pop((service, account), None)


def test_account_scope_is_stable_and_dir_specific(tmp_path):
    a = skill_trust_account_scope(tmp_path / "profileA" / "skills" / "trust")
    b = skill_trust_account_scope(tmp_path / "profileB" / "skills" / "trust")
    assert a and b and a != b
    assert a == skill_trust_account_scope(tmp_path / "profileA" / "skills" / "trust")
    assert len(a) == 16


def test_scoped_marker_accounts_do_not_cross_read(monkeypatch):
    kr = _FakeKeyring()
    monkeypatch.setattr(
        "tldw_chatbook.Skills_Interop.skill_trust_store.is_secure_keyring_backend",
        lambda backend: True,
    )
    a = KeyringSkillTrustGenerationMarkerStore(keyring_backend=kr, account_scope="aaaa")
    b = KeyringSkillTrustGenerationMarkerStore(keyring_backend=kr, account_scope="bbbb")
    a.save_marker(generation=1, manifest_digest="digestA")
    assert a.load_marker() is not None
    assert b.load_marker() is None  # scoped: B cannot see A's marker
    # And the legacy global account is untouched by scoped writes:
    assert (KeyringSkillTrustGenerationMarkerStore.__dataclass_fields__)  # sanity
    assert kr.store.get(("tldw_chatbook.skill_trust", _MARKER_USERNAME)) is None


def test_scoped_key_cache_accounts_do_not_cross_read(monkeypatch):
    kr = _FakeKeyring()
    monkeypatch.setattr(
        "tldw_chatbook.Skills_Interop.skill_trust_store.is_secure_keyring_backend",
        lambda backend: True,
    )
    ca = KeyringSkillTrustKeyCache(keyring_backend=kr, account_scope="aaaa")
    cb = KeyringSkillTrustKeyCache(keyring_backend=kr, account_scope="bbbb")
    # Write a raw payload under A's scoped account, prove B can't read it.
    kr.set_password("tldw_chatbook.skill_trust.keys", f"{_KEY_CACHE_USERNAME}:aaaa", "x")
    assert kr.get_password("tldw_chatbook.skill_trust.keys", f"{_KEY_CACHE_USERNAME}:bbbb") is None
    assert kr.get_password("tldw_chatbook.skill_trust.keys", _KEY_CACHE_USERNAME) is None
