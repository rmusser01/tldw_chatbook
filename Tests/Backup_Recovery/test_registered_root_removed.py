"""A registered folder that is later removed is named, not a generic failure.

TASK-34100.16.1 review v5r2 (V2). Each capture registers the roots it copies as
``backup.source.*`` admission entries. ADR-126 keeps an absent registered root
strict ("missing roots without such already owned coverage still refuse"), so
once such a folder is removed, every later Create backup and the next start
refuse until it is back. Live, Create ended in the generic
``backup_operation_failed`` (a bare ``FileNotFoundError``) and the start said
only ``storage_admission_unavailable``; neither named the folder, and the User
Guide's advice to ``rmdir`` the tokenizers folder walked users into it
(g161v5-v2a/24-28, g161v5-v2d). Treating such a root as absent instead is an
owner decision, held for TASK-34100.16.1.4. These tests pin what the user sees
until then, and that recreating the folder (the advice both messages give)
lets backup and start succeed again.
"""

import os
from threading import Event

import pytest

from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery import bootstrap, owner_registry
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture
from tldw_chatbook.Backup_Recovery.recovery_service import (
    RecoveryService,
    issue_code,
    issue_message,
)


@pytest.fixture
def removed_tokenizers(tmp_path, monkeypatch):
    """A fresh profile whose tokenizers folder a backup registered, then removed."""
    from tldw_chatbook.Backup_Recovery import capture_service

    home = tmp_path / "home"
    source = home / ".config" / "tldw_cli" / "config.toml"
    source.parent.mkdir(parents=True, mode=0o700)
    data = home / ".local" / "share" / "tldw_cli"
    source.write_text(
        f'[general]\nusers_name="default_user"\n[paths]\ndata_dir="{data}"\n'
    )
    source.chmod(0o600)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(source))
    root = source.parent / "recovery-bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setattr(owner_registry, "_adapters", {})
    monkeypatch.setattr(capture_service, "effective_config_path", lambda: source)
    monkeypatch.setattr(capture_service, "default_config_path", lambda: source)
    options = {"allow_partial": True, "staging_parent": tmp_path}
    tokenizers = source.parent / "tokenizers"
    tokenizers.mkdir(mode=0o700)  # what CustomTokenizerManager's first write makes
    preview = preview_capture((source,), options=options, include_known_profiles=True)
    _populate_required_dependencies(preview)
    preview = preview_capture((source,), options=options, include_known_profiles=True)
    assert {
        item.status for item in preview.items if item.owner == "tokenizers.custom"
    } == {"included_directory"}
    capture(
        (source,),
        preview.scope_digest,
        tmp_path / "first.tldw-backup.zip",
        options=options,
        cancel=Event(),
        include_known_profiles=True,
    )
    os.rmdir(tokenizers)  # the removal the User Guide used to recommend
    yield source, options, tokenizers
    lease = storage._startups.pop((os.getpid(), str(root)), None)
    if lease is not None:
        lease.close()


def test_create_backup_names_the_removed_folder_and_recovers_once_it_is_back(
    tmp_path, removed_tokenizers
):
    """Create backup names the missing folder; recreating it lets Create succeed.

    Before: ``FileNotFoundError`` from ``_capture_names``' strict resolve, which
    the screen showed as ``backup_operation_failed`` with no location.
    """
    source, options, tokenizers = removed_tokenizers
    service = RecoveryService(tmp_path / "control")
    try:
        preview = service.preview_backup(
            (source,), options=options, include_known_profiles=True
        )
        operation = service.start_backup(
            (source,),
            preview.scope_digest,
            tmp_path / "second.tldw-backup.zip",
            options=options,
            password=None,
            include_known_profiles=True,
        )
        status = service.wait(operation, timeout=120)

        assert (status["state"], status["issues"]) == (
            "failed",
            ("registered_storage_missing",),
        )
        assert status["result"].get("path") == str(tokenizers)
        message = issue_message("registered_storage_missing")
        assert "no longer exists" in message and "create it again" in message

        tokenizers.mkdir(mode=0o700)  # the remedy the message gives
        preview = service.preview_backup(
            (source,), options=options, include_known_profiles=True
        )
        operation = service.start_backup(
            (source,),
            preview.scope_digest,
            tmp_path / "third.tldw-backup.zip",
            options=options,
            password=None,
            include_known_profiles=True,
        )
        status = service.wait(operation, timeout=120)
        assert status["state"] == "succeeded", status["issues"]
        assert status["result"]["archive_verified"] is True
    finally:
        service.close()


def test_capture_refusal_keeps_a_bounded_code_and_carries_the_location(
    removed_tokenizers,
):
    """The refusal's text stays a bounded code; the location travels apart.

    ``RecoveryRequired`` text never carries a local locator, and only fixed
    codes reach the view: the path rides on the error, for the service's own
    result row.
    """
    source, options, tokenizers = removed_tokenizers
    preview = preview_capture((source,), options=options, include_known_profiles=True)
    with pytest.raises(Exception) as raised:
        capture(
            (source,),
            preview.scope_digest,
            source.parent.parent.parent / "second.tldw-backup.zip",
            options=options,
            cancel=Event(),
            include_known_profiles=True,
        )

    assert issue_code(raised.value, kind="backup") == "registered_storage_missing"
    assert str(raised.value) == "registered_storage_missing"
    assert getattr(raised.value, "path", None) == tokenizers


def test_next_start_names_the_removed_folder_and_starts_once_it_is_back(
    removed_tokenizers,
):
    """The refused start names the folder; recreating it lets Chatbook start.

    Before: "Recovery required: storage_admission_unavailable" and nothing else
    (g161v5-v2d/04), with the app unable to start until a guess fixed it.
    """
    _, _, tokenizers = removed_tokenizers
    with pytest.raises(SystemExit) as refused:
        storage.admit_startup()

    text = str(refused.value)
    assert text.startswith("Recovery required: storage_admission_unavailable")
    assert str(tokenizers) in text
    assert "create it again" in text

    tokenizers.mkdir(mode=0o700)  # the remedy the message gives
    storage.admit_startup()
