"""Creation-intent dependencies remain fresh in warm storage admission."""

import os

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import control_records
from tldw_chatbook.Backup_Recovery import storage_admission as storage


def _verdict(path):
    try:
        with storage.acquire_storage(path) as lease:
            lease.execution_context(path)
            return "allowed", None
    except (OSError, ValueError, RuntimeError) as error:
        return "refused", type(error).__name__


def test_warm_admission_observes_new_exact_creation_intent(tmp_path, monkeypatch):
    root = tmp_path / "bootstrap"
    selector = tmp_path / "config.toml"
    selector.write_text("[console]\n", encoding="utf-8")
    selector.chmod(0o600)
    target = tmp_path / "store.db"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    monkeypatch.setattr(storage, "_EVIDENCE_REUSE", True)
    startup = storage.acquire_storage()
    try:
        for _ in range(4):
            assert _verdict(target) == ("allowed", None)
        hold = storage._holds[(os.getpid(), str(root))]
        assert hold.evidence[str(selector)].confirmed, "warm evidence not reached"
        marker = root.parent / control_records._creation_name(root)
        marker.write_bytes(b"{malformed-private-creation-intent")
        marker.chmod(0o600)
        reused = _verdict(target)
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
        derived = _verdict(target)
        assert derived[0] == "refused", "full derivation did not exercise the fence"
        assert reused == derived
        assert marker.read_bytes() == b"{malformed-private-creation-intent"
    finally:
        startup.close()


def test_selector_evidence_stamps_exact_ancestor_creation_intents(
    tmp_path, monkeypatch
):
    root = tmp_path / "bootstrap"
    selector = tmp_path / "config.toml"
    selector.write_text("[console]\n", encoding="utf-8")
    selector.chmod(0o600)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    with storage.acquire_storage():
        evidence = storage._selector_evidence(
            root, selector, (control_records.UNBOUND_NAMESPACE,), None
        )
        assert evidence is not None
        stamped = dict(evidence.content)
        for child in storage._chain(root)[1:]:
            marker = child.parent / control_records._creation_name(child)
            assert marker in stamped
            assert stamped[marker] is None


def test_metadata_evidence_preserves_absent_creation_intents(tmp_path, monkeypatch):
    root = tmp_path / "bootstrap"
    selector = tmp_path / "config.toml"
    selector.write_text("[console]\n", encoding="utf-8")
    selector.chmod(0o600)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    with storage.acquire_storage() as lease:
        hold = storage._holds[lease._key]
        evidence = storage._metadata_evidence(hold)
        assert evidence is not None, "validated absent creation intents prevented reuse"
        stamped = dict(evidence.content)
        for child in storage._chain(root)[1:]:
            marker = child.parent / control_records._creation_name(child)
            assert marker in stamped and stamped[marker] is None
        marker = root.parent / control_records._creation_name(root)
        marker.write_bytes(b"{malformed-private-creation-intent")
        marker.chmod(0o600)
        assert evidence.observe() != evidence.stamps()
        assert storage._metadata_evidence(hold) is None


def test_metadata_evidence_refuses_marker_appearing_between_observations(
    tmp_path, monkeypatch
):
    root = tmp_path / "bootstrap"
    selector = tmp_path / "config.toml"
    selector.write_text("[console]\n", encoding="utf-8")
    selector.chmod(0o600)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    with storage.acquire_storage() as lease:
        original = storage._selector_evidence
        observed = []
        marker = root.parent / control_records._creation_name(root)

        def appear_after_selector(*args):
            base = original(*args)
            assert base is not None and dict(base.content)[marker] is None
            observed.append(base)
            marker.write_bytes(b"{malformed-private-creation-intent")
            marker.chmod(0o600)
            return base

        monkeypatch.setattr(storage, "_selector_evidence", appear_after_selector)
        assert storage._metadata_evidence(storage._holds[lease._key]) is None
        assert len(observed) == 1, "intervention missed the actual first observation"
