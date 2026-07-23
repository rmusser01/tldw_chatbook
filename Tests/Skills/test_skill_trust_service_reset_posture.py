import secrets

from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
    SkillTrustMarkerUnavailable,
)


def _service(tmp_path, marker=None):
    marker = marker or FileSkillTrustGenerationMarkerStore(tmp_path / "trust" / "marker.json")
    (tmp_path / "trust").mkdir(parents=True, exist_ok=True)
    (tmp_path / "skills").mkdir(parents=True, exist_ok=True)
    return SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(store_dir=tmp_path / "trust", marker_store=marker),
        key_cache=None,
    )


def test_posture_needs_setup_on_pristine(tmp_path):
    svc = _service(tmp_path)
    assert svc.trust_posture() == "needs_setup"


def test_posture_needs_resetup_when_marker_present_but_no_manifest(tmp_path):
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "trust" / "marker.json")
    (tmp_path / "trust").mkdir(parents=True, exist_ok=True)
    marker.save_marker(generation=1, manifest_digest="d")  # foreign/inherited marker
    svc = _service(tmp_path, marker=marker)
    assert svc.trust_posture() == "needs_resetup"


def test_posture_unavailable_when_marker_load_raises(tmp_path):
    class _Raising:
        def load_marker(self):
            raise SkillTrustMarkerUnavailable("locked keychain")

        def save_marker(self, **k):
            pass

        def clear(self):
            pass

    svc = _service(tmp_path, marker=_Raising())
    # Force has_manifest True by writing a manifest file placeholder path.
    svc.trust_store.manifest_path.write_text("{}", encoding="utf-8")
    assert svc.trust_posture() == "unavailable"


def test_posture_unavailable_beats_needs_setup_when_no_manifest_and_marker_raises(
    tmp_path,
):
    # THE ordering guarantee (Qodo review): an unreadable marker store (e.g. a
    # locked OS keyring) must surface as "unavailable" (Retry, non-destructive)
    # even with NO manifest -- never "needs_setup", whose bootstrap would fail
    # trying to persist a marker to the very store that just raised. The
    # `not available` check therefore precedes the `not has_manifest` branch.
    class _Raising:
        def load_marker(self):
            raise SkillTrustMarkerUnavailable("locked keychain")

        def save_marker(self, **k):
            pass

        def clear(self):
            pass

    svc = _service(tmp_path, marker=_Raising())
    assert svc.trust_store.has_manifest() is False
    assert svc.trust_posture() == "unavailable"


def test_reset_then_bootstrap_recovers_from_poison(tmp_path):
    marker = FileSkillTrustGenerationMarkerStore(tmp_path / "trust" / "marker.json")
    (tmp_path / "trust").mkdir(parents=True, exist_ok=True)
    marker.save_marker(generation=9, manifest_digest="stale")  # poison
    svc = _service(tmp_path, marker=marker)
    assert svc.trust_posture() == "needs_resetup"
    svc.reset_trust()
    assert svc.trust_posture() == "needs_setup"
    svc.bootstrap_trust("passphrase", salt=secrets.token_bytes(32))
    assert svc.trust_posture() == "ready"


def test_reset_trust_is_non_crashing_and_idempotent(tmp_path):
    svc = _service(tmp_path)
    svc.reset_trust()
    svc.reset_trust()  # idempotent, no raise
    assert svc.trust_posture() == "needs_setup"


def test_posture_locked_when_manifest_and_marker_present_but_keys_absent(tmp_path):
    # Bootstrap writes manifest + marker to disk; a FRESH service has keys=None.
    svc1 = _service(tmp_path)
    svc1.bootstrap_trust("pw", salt=secrets.token_bytes(32))
    assert svc1.trust_posture() == "ready"
    svc2 = _service(tmp_path)  # same on-disk store, no in-memory keys
    assert svc2.trust_posture() == "locked"


def test_posture_orphaned_manifest_beats_locked_when_keys_absent(tmp_path):
    # THE ordering guarantee: manifest present + marker cleanly absent + keys None
    # must be needs_resetup (branch 4), never locked (branch 5).
    svc1 = _service(tmp_path)
    svc1.bootstrap_trust("pw", salt=secrets.token_bytes(32))
    svc2 = _service(tmp_path)  # fresh: keys None
    svc2.trust_store.marker_store.clear()  # remove marker, leave manifest
    assert svc2.trust_store.has_manifest() is True
    assert svc2.trust_posture() == "needs_resetup"


def test_posture_error_when_manifest_corrupted_with_keys_loaded(tmp_path):
    svc = _service(tmp_path)
    svc.bootstrap_trust("pw", salt=secrets.token_bytes(32))
    # Corrupt the on-disk manifest so _load_valid_manifest raises while keys stay in memory.
    svc.trust_store.manifest_path.write_text('{"tampered": true}', encoding="utf-8")
    assert svc.trust_posture() == "error"
