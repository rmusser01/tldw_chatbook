"""Actual Windows cold admissions; no forced warm storage authority."""

import os

import pytest

from Tests.Backup_Recovery.test_bootstrap import local_scope  # noqa: F401


@pytest.mark.skipif(os.name != "nt", reason="actual Windows cold admission")
def test_windows_cold_acquisition_rederives_and_refuses_changed_current_control(
    local_scope,  # noqa: F811 - shared fixture
    monkeypatch,
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile

    root, config, data, authority = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    calls = []
    original = storage._scope

    def scope(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage, "_scope", scope)
    owner = storage.acquire_storage()
    try:
        before = len(calls)
        for _ in range(2):
            with storage.acquire_storage(data / "native.db") as lease:
                assert lease.execution_context(data / "native.db")[1]
        assert len(calls) >= before + 2
        hold = storage._holds[owner._key]
        assert not hold.evidence and not hold.path_evidence
        registry = authority.control_root / "registry.json"
        original_bytes = registry.read_bytes()
        registry.write_bytes(b"{" + b" " * (len(original_bytes) - 1))
        try:
            with pytest.raises((bootstrap.RecoveryRequired, ValueError, OSError)):
                storage.acquire_storage(data / "must-not-write.db")
            assert not (data / "must-not-write.db").exists()
        finally:
            registry.write_bytes(original_bytes)
    finally:
        owner.close()
