import secrets
from pathlib import Path

import pytest

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
