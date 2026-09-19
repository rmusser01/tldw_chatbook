"""Credential processing uses only the native session's private staged copies."""

import pytest

from Tests.Backup_Recovery.test_core_owners import application_authority
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired


def test_staged_credentials_require_scope_and_preserve_source(tmp_path, monkeypatch):
    source = tmp_path / "source.toml"
    source.write_text('api_key="source-secret"\n')
    source.chmod(0o600)
    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700)
    candidate = staging / "config.toml"
    candidate.write_bytes(source.read_bytes())
    candidate.chmod(0o600)
    authority = application_authority(tmp_path, source, monkeypatch)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((source,), staging),
    ):
        assert (
            storage._read_staged_credential_file(candidate, max_bytes=1024)
            == source.read_bytes()
        )
        assert storage._write_staged_credential_file(candidate, 'api_key=""\n')
        assert candidate.read_text() == 'api_key=""\n'
        material = staging / "credential-material.json"
        assert storage._write_staged_credential_file(material, "{}")
        assert storage._read_staged_credential_file(material, max_bytes=1024) == b"{}"
        with pytest.raises(RecoveryRequired, match="capture_path_outside_scope"):
            storage._write_staged_credential_file(source, "changed")
        with pytest.raises(RecoveryRequired, match="capture_path_outside_scope"):
            storage._read_staged_credential_file(source, max_bytes=1024)
    assert source.read_text() == 'api_key="source-secret"\n'
    assert storage._read_staged_credential_file(candidate, max_bytes=1024) is None
    assert storage._write_staged_credential_file(candidate, "changed") is False


@pytest.mark.parametrize("invalid", ["symlink", "hardlink", "public", "oversized"])
def test_staged_credentials_reject_unqualified_files(tmp_path, monkeypatch, invalid):
    source = tmp_path / "source.toml"
    source.write_text("original")
    source.chmod(0o600)
    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700)
    candidate = staging / "config.toml"
    if invalid == "symlink":
        candidate.symlink_to(source)
    elif invalid == "hardlink":
        candidate.hardlink_to(source)
    else:
        candidate.write_text("0123456789")
        candidate.chmod(0o644 if invalid == "public" else 0o600)
    authority = application_authority(tmp_path, source, monkeypatch)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((source,), staging),
    ):
        with pytest.raises((ValueError, OSError, RecoveryRequired)):
            storage._read_staged_credential_file(candidate, max_bytes=5)
        if invalid != "oversized":
            with pytest.raises((ValueError, OSError, RecoveryRequired)):
                storage._write_staged_credential_file(candidate, "changed")
    assert source.read_text() == "original"


@pytest.mark.parametrize("changed_at", ["open", "read"])
def test_staged_credential_read_rechecks_opened_file_privacy(
    tmp_path, monkeypatch, changed_at
):
    source = tmp_path / "source.toml"
    source.write_text("original")
    source.chmod(0o600)
    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700)
    candidate = staging / "credentials.toml"
    candidate.write_text("staged-secret")
    candidate.chmod(0o600)
    authority = application_authority(tmp_path, source, monkeypatch)
    original_open, original_read = storage.os.open, storage.os.read
    credential_fd = None

    def open_file(path, flags, *args, **kwargs):
        nonlocal credential_fd
        if path == candidate.name:
            if changed_at == "open":
                candidate.chmod(0o644)
            credential_fd = original_open(path, flags, *args, **kwargs)
            return credential_fd
        return original_open(path, flags, *args, **kwargs)

    def read_file(fd, count):
        result = original_read(fd, count)
        if fd == credential_fd and changed_at == "read":
            candidate.chmod(0o644)
        return result

    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((source,), staging),
    ):
        monkeypatch.setattr(storage.os, "open", open_file)
        monkeypatch.setattr(storage.os, "read", read_file)
        with pytest.raises(RecoveryRequired, match="credential_staging_required"):
            storage._read_staged_credential_file(candidate, max_bytes=1024)
    assert source.read_text() == "original"
