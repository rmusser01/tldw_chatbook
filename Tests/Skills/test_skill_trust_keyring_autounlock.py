"""task-624: cached trust keys must unlock WITHOUT an explicit call.

`unlock_from_keyring_convenience()` worked but had zero callers, so enabling
keyring convenience persisted keys the app never loaded — every launch reported
"Skill trust is locked for this session". These tests pin the call site, not
the method: a fresh service that has cached keys must reach a usable state on
its own.
"""

import base64
import json

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    KeyringSkillTrustKeyCache,
    SkillTrustStore,
)

from .test_skill_trust_service import FakeSecureKeyring, _service, _write_skill


def _fresh_service(tmp_path, skills_dir, key_cache):
    """Build a service exactly as a new process would: no in-memory keys."""
    return SkillTrustService(
        skills_dir=skills_dir,
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
        key_cache=key_cache,
    )


@pytest.fixture
def cached(tmp_path):
    """A trust store whose derived keys are already in the key cache."""
    fake_keyring = FakeSecureKeyring()
    key_cache = KeyringSkillTrustKeyCache(keyring_backend=fake_keyring)
    service, skills_dir = _service(tmp_path, key_cache=key_cache)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    service.enable_keyring_convenience()
    return tmp_path, skills_dir, key_cache


def test_posture_is_ready_without_an_explicit_unlock(cached):
    """AC#1/#2: the header must not report 'locked' when keys are available."""
    tmp_path, skills_dir, key_cache = cached
    fresh = _fresh_service(tmp_path, skills_dir, key_cache)
    assert fresh.trust_posture() == "ready"


def test_skill_status_resolves_without_an_explicit_unlock(cached):
    tmp_path, skills_dir, key_cache = cached
    fresh = _fresh_service(tmp_path, skills_dir, key_cache)
    assert fresh.status_for_skill("demo").trust_status == "trusted"


def test_trusted_file_paths_resolves_without_an_explicit_unlock(cached):
    """The gate script execution and skill_file reads both depend on."""
    tmp_path, skills_dir, key_cache = cached
    fresh = _fresh_service(tmp_path, skills_dir, key_cache)
    assert "SKILL.md" in fresh.trusted_file_paths("demo")


def test_convenience_flag_reflects_the_cached_state(cached):
    """AC#5: the keychain entry IS the persistence; no config flag needed."""
    tmp_path, skills_dir, key_cache = cached
    fresh = _fresh_service(tmp_path, skills_dir, key_cache)
    assert fresh.keyring_convenience_enabled is False
    fresh.trust_posture()
    assert fresh.keyring_convenience_enabled is True


def test_no_key_cache_still_reports_locked(tmp_path):
    """AC#3: absent cache falls back cleanly to the existing locked path."""
    service, skills_dir = _service(tmp_path)
    _write_skill(skills_dir)
    service.bootstrap_trust()
    fresh = SkillTrustService(
        skills_dir=skills_dir,
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )
    assert fresh.trust_posture() == "locked"


def test_stale_cached_keys_do_not_unlock(cached):
    """AC#4: salt-bound — a manifest re-salted since caching must stay locked."""
    tmp_path, skills_dir, key_cache = cached
    manifest_path = tmp_path / "trust" / "skill_trust_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["kdf_salt"] = base64.b64encode(b"7" * 32).decode("ascii")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    fresh = _fresh_service(tmp_path, skills_dir, key_cache)
    assert fresh.trust_posture() == "locked"
    assert fresh.status_for_skill("demo").trust_status == "trust_locked"


def test_a_raising_key_cache_never_escapes(cached):
    """AC#3: a broken keyring backend must not crash a posture query."""
    tmp_path, skills_dir, _cache = cached

    class Exploding:
        def load_keys(self, *, expected_salt):
            raise RuntimeError("keychain exploded")

    fresh = _fresh_service(tmp_path, skills_dir, Exploding())
    assert fresh.trust_posture() == "locked"


def test_successful_unlock_reads_the_keychain_only_once(cached):
    """The happy path must not re-read the keychain on every render.

    No latch is needed for this: a successful load sets the in-memory keys,
    which short-circuits the attempt thereafter.
    """
    tmp_path, skills_dir, key_cache = cached

    class Counting:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def load_keys(self, *, expected_salt):
            self.calls += 1
            return self.inner.load_keys(expected_salt=expected_salt)

    counting = Counting(key_cache)
    fresh = _fresh_service(tmp_path, skills_dir, counting)
    for _ in range(5):
        fresh.trust_posture()
        fresh.status_for_skill("demo")
    assert counting.calls == 1


def test_a_transient_failure_does_not_suppress_a_later_unlock(cached):
    """Regression: a one-shot latch made a transient keyring error permanent.

    `trust_posture()` documents an unavailable keyring as recoverable by Retry,
    so a failed attempt must never bar the next one.
    """
    tmp_path, skills_dir, key_cache = cached

    class FlakyOnce:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def load_keys(self, *, expected_salt):
            self.calls += 1
            if self.calls == 1:
                raise OSError("keychain temporarily unavailable")
            return self.inner.load_keys(expected_salt=expected_salt)

    flaky = FlakyOnce(key_cache)
    fresh = _fresh_service(tmp_path, skills_dir, flaky)
    assert fresh.trust_posture() == "locked"
    assert fresh.trust_posture() == "ready", "a retry must be able to succeed"


def test_settings_posture_also_auto_unlocks(cached):
    """overall_status() is a lockedness decision point too (Settings surface)."""
    tmp_path, skills_dir, key_cache = cached
    fresh = _fresh_service(tmp_path, skills_dir, key_cache)
    assert fresh.overall_status() == "trusted"
