"""Encrypted self-voiceprint store (TASK-31826, Task 1 of the Meetings
cross-meeting speaker enrollment SDD program). See
Docs/superpowers/specs/2026-09-06-meeting-voiceprint-design.md section 3.1.
"""
import json, os, stat
import pytest
from tldw_chatbook.Audio import voiceprint as vp

class FakeKeys:
    mode = "keyring"
    def __init__(self, key="k" * 32, blocked=False): self.key, self.blocked, self.created = key, blocked, 0
    def get_or_create(self): self.created += 1; return self.key
    def get(self, timeout_s): return None if self.blocked else self.key

def _rec(cent=(1.0, 0.0), n=1.0, model="ecapa@rev1"):
    return vp.Voiceprint(model_id=model, centroid=vp.unit_normalise(cent), sample_count=n, meetings_contributed=1,
                         created_at="2026-09-06T00:00:00", updated_at="2026-09-06T00:00:00", threshold_used=0.2)

def test_round_trip_is_encrypted_at_rest(tmp_path):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys())
    store.save(_rec())
    raw = (tmp_path / "voiceprint.json").read_text()
    assert "1.0" not in raw and '"mode": "keyring"' in raw          # payload ciphertext, envelope mode visible
    assert store.load().voiceprint.centroid == [1.0, 0.0]

def test_blocked_key_reports_keyring_locked_without_raising(tmp_path):
    keys = FakeKeys(); store = vp.VoiceprintStore(tmp_path / "voiceprint.json", keys); store.save(_rec())
    keys.blocked = True
    res = store.load(timeout_s=0.1)
    assert res.voiceprint is None and res.reason == "keyring_locked"

def test_model_mismatch_needs_reenrollment(tmp_path):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys()); store.save(_rec(model="ecapa@rev1"))
    assert store.load(expected_model_id="ecapa@rev2").reason == "needs_reenrollment"

def test_atomic_save_keeps_old_file_when_write_fails(tmp_path, monkeypatch):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys()); store.save(_rec((1.0, 0.0)))
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        store.save(_rec((0.0, 1.0)))
    assert store.load().voiceprint.centroid == [1.0, 0.0]

def test_merge_is_capped_normalised_weighted_mean(tmp_path):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys(), per_meeting_cap=2.0); store.save(_rec((1.0, 0.0), n=2.0))
    out = store.merge_sample((0.0, 10.0), weight=100.0, model_id="ecapa@rev1")   # weight capped to 2.0
    assert out.sample_count == pytest.approx(4.0) and out.meetings_contributed == 2
    assert out.centroid == pytest.approx(vp.unit_normalise((0.5, 0.5)))

def test_export_import_with_passphrase_and_model_gate(tmp_path):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys()); store.save(_rec())
    with pytest.raises(ValueError):
        store.export(tmp_path / "out.json", "")
    store.export(tmp_path / "out.json", "correct horse")
    other = vp.VoiceprintStore(tmp_path / "b" / "voiceprint.json", FakeKeys(key="j" * 32))
    other.save(_rec((0.0, 1.0), model="ecapa@rev9"))
    with pytest.raises(vp.ModelMismatch):
        other.import_(tmp_path / "out.json", "correct horse", replace=False)
    assert other.import_(tmp_path / "out.json", "correct horse", replace=True).model_id == "ecapa@rev1"

def test_keyfile_provider_refuses_broad_permissions(tmp_path):
    p = tmp_path / "voiceprint.key"; p.write_text("k" * 32); p.chmod(0o644)
    assert vp.KeyfileKeyProvider(p).get(timeout_s=0.1) is None

def test_delete_removes_file_only(tmp_path):
    keys = FakeKeys(); store = vp.VoiceprintStore(tmp_path / "voiceprint.json", keys); store.save(_rec())
    assert store.delete() is True and not (tmp_path / "voiceprint.json").exists() and keys.key


# --- Review fix-round regressions (2026-09-06) -----------------------------

def _export_donor_record(tmp_path, subdir, model):
    donor = vp.VoiceprintStore(tmp_path / subdir / "voiceprint.json", FakeKeys(key="j" * 32))
    donor.save(_rec(model=model))
    out = tmp_path / f"{subdir}-out.json"
    donor.export(out, "correct horse")
    return out

def test_import_locked_store_refuses_and_leaves_file_unchanged_either_way(tmp_path):
    # CRITICAL: a locked/unreadable existing record must never be silently
    # overwritten just because `current is None` looked the same as "empty".
    keys = FakeKeys(); store = vp.VoiceprintStore(tmp_path / "voiceprint.json", keys)
    store.save(_rec(model="ecapa@rev9"))
    raw_before = (tmp_path / "voiceprint.json").read_bytes()
    donor_out = _export_donor_record(tmp_path, "donor", "ecapa@rev1")

    keys.blocked = True
    with pytest.raises(vp.StoreUnavailable):
        store.import_(donor_out, "correct horse", replace=False)
    assert (tmp_path / "voiceprint.json").read_bytes() == raw_before

    # A locked key never justifies a replace either -- it might unlock.
    with pytest.raises(vp.StoreUnavailable):
        store.import_(donor_out, "correct horse", replace=True)
    assert (tmp_path / "voiceprint.json").read_bytes() == raw_before

def test_import_undecryptable_store_refuses_without_replace(tmp_path):
    keys = FakeKeys(); store = vp.VoiceprintStore(tmp_path / "voiceprint.json", keys)
    store.save(_rec(model="ecapa@rev9"))
    keys.key = "x" * 32  # wrong key now -> existing record is undecryptable
    donor_out = _export_donor_record(tmp_path, "donor", "ecapa@rev1")

    with pytest.raises(vp.StoreUnavailable):
        store.import_(donor_out, "correct horse", replace=False)

def test_import_undecryptable_store_may_be_overwritten_with_replace(tmp_path):
    keys = FakeKeys(); store = vp.VoiceprintStore(tmp_path / "voiceprint.json", keys)
    store.save(_rec(model="ecapa@rev9"))
    keys.key = "x" * 32  # wrong key now -> existing record is undecryptable
    donor_out = _export_donor_record(tmp_path, "donor", "ecapa@rev1")

    result = store.import_(donor_out, "correct horse", replace=True)
    assert result.model_id == "ecapa@rev1"

def test_load_non_dict_envelope_reports_cannot_decrypt_without_raising(tmp_path):
    path = tmp_path / "voiceprint.json"
    path.write_text("null")
    result = vp.VoiceprintStore(path, FakeKeys()).load()
    assert result.voiceprint is None and result.reason == "cannot_decrypt"

def test_keyfile_provider_creates_key_pre_restricted(tmp_path, monkeypatch):
    p = tmp_path / "voiceprint.key"
    modes_used = []
    real_open = os.open
    def spy_open(path, flags, mode=0o777, *a, **kw):
        modes_used.append(mode)
        return real_open(path, flags, mode, *a, **kw)
    monkeypatch.setattr(os, "open", spy_open)

    key = vp.KeyfileKeyProvider(p).get_or_create()

    assert modes_used == [0o600]
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert vp.KeyfileKeyProvider(p).get(timeout_s=0.1) == key

def test_keyfile_provider_get_or_create_refuses_unsafe_existing_file(tmp_path):
    p = tmp_path / "voiceprint.key"; p.write_text("k" * 32); p.chmod(0o644)
    with pytest.raises(vp.StoreUnavailable):
        vp.KeyfileKeyProvider(p).get_or_create()
    assert p.read_text() == "k" * 32  # untouched -- no silent new key minted

def test_atomic_write_removes_stray_tmp_on_replace_failure(tmp_path, monkeypatch):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys()); store.save(_rec())
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        store.save(_rec((0.0, 1.0)))
    assert not (tmp_path / "voiceprint.tmp").exists()

def test_keyfile_provider_removes_stray_tmp_on_replace_failure(tmp_path, monkeypatch):
    p = tmp_path / "voiceprint.key"
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        vp.KeyfileKeyProvider(p).get_or_create()
    assert not p.with_suffix(".tmp").exists()


# --- Task 4 fix round 1: a stat-only presence check (no Keychain prompt) ---

def test_exists_is_a_stat_and_never_touches_the_key(tmp_path):
    """The Meetings rail asks "is there a voiceprint?" at screen mount; that
    question must not raise a Keychain prompt (the decrypt at meeting Start
    does, on a thread, with a timeout)."""

    class CountingKeys(FakeKeys):
        def __init__(self):
            super().__init__()
            self.reads = 0

        def get(self, timeout_s):
            self.reads += 1
            return super().get(timeout_s)

    keys = CountingKeys()
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", keys)
    assert store.exists() is False
    store.save(_rec())
    assert store.exists() is True
    assert keys.reads == 0                      # save mints via get_or_create, never get
    store.delete()
    assert store.exists() is False
    assert keys.reads == 0


def test_mode_reports_the_key_provider_in_use(tmp_path):
    assert vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys()).mode == "keyring"
    keyfile_store = vp.VoiceprintStore(
        tmp_path / "b" / "voiceprint.json", vp.KeyfileKeyProvider(tmp_path / "b" / "voiceprint.key")
    )
    assert keyfile_store.mode == "keyfile"
