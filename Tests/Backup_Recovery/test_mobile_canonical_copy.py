"""Isolated native archive binding; these controls issue no operation authority."""

import hashlib
from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_archive_reader import archive, manifest
from tldw_chatbook.Backup_Recovery import archive_reader, crypto
from tldw_chatbook.Backup_Recovery.journal import (
    _MobileCaptureVerified,
    observe_artifact,
)
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import (
    _CanonicalCopy,
    _verify_canonical_archive,
)


def test_native_acquired_copy_binds_framed_cipher_and_plaintext(
    tmp_path, monkeypatch, helper_resource_root
):
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    document = manifest()
    document["credential_policy"] = "rollback"
    original = archive(tmp_path, document)
    original.chmod(0o600)
    cipher = tmp_path / "capture.zip.age"
    crypto.transform(
        original, cipher, password=b"isolated-copy", decrypt=False, cancel=Event()
    )
    acquired = archive_reader.acquire(
        cipher, tmp_path / "acquisition", ArchiveLimits(), b"isolated-copy", Event()
    )
    capture = _MobileCaptureVerified(
        purpose="canonical",
        archive=observe_artifact(cipher),
        manifest_digest=hashlib.sha256(acquired.manifest_bytes).hexdigest(),
        partition_digest="1" * 64,
    )
    # Only immutable source facts are inputs to this isolated verifier. No
    # prepared operation, journal completion, binding or restore plan is issued.
    proof = _CanonicalCopy(capture, "2" * 64, False, None)
    assert acquired.encrypted_source.path != cipher
    assert acquired.encrypted_source.identity[1] != cipher.stat().st_ino
    assert acquired.encrypted_source.digest != capture.archive.sha256
    assert (
        _verify_canonical_archive(proof, acquired, acquired.digest).credential_policy
        == "rollback"
    )
    for altered in (
        replace(
            acquired,
            encrypted_source=replace(acquired.encrypted_source, digest="0" * 64),
        ),
        replace(
            acquired,
            encrypted_source=replace(
                acquired.encrypted_source, plaintext_digest="0" * 64
            ),
        ),
        replace(
            acquired,
            encrypted_source=replace(
                acquired.encrypted_source, manifest_digest="0" * 64
            ),
        ),
    ):
        with pytest.raises(ValueError, match="local_snapshot_source_changed"):
            _verify_canonical_archive(proof, altered, altered.digest)
    with pytest.raises(ValueError, match="local_snapshot_source_changed"):
        _verify_canonical_archive(proof, acquired, "0" * 64)
    changed_capture = capture.model_copy(
        update={"archive": capture.archive.model_copy(update={"sha256": "0" * 64})}
    )
    with pytest.raises(ValueError, match="local_snapshot_source_changed"):
        _verify_canonical_archive(
            replace(proof, capture=changed_capture), acquired, acquired.digest
        )
    copied = acquired.encrypted_source.path
    content = copied.read_bytes()
    moved = copied.with_name("retained-original-cipher")
    copied.rename(moved)
    copied.write_bytes(content)
    copied.chmod(0o600)
    with pytest.raises(ValueError, match="encrypted_source_changed"):
        _verify_canonical_archive(proof, acquired, acquired.digest)


def test_fresh_selected_inventory_keeps_separate_selector_cohort(tmp_path, monkeypatch):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
    from tldw_chatbook.Backup_Recovery.capture_service import (
        _selectors,
        preview_capture,
    )

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        preview = values[4]
    selected = tuple(dict.fromkeys((preview.source_selector, preview.target_selector)))
    current = preview_capture(selected, options={})
    assert current.complete, [
        (item.owner, item.status)
        for item in current.items
        if item.status in {"missing_required", "unavailable", "unsupported"}
    ]
    assert {item.path for item in current.items if item.owner == "config"} == set(
        selected
    )
    assert _selectors(selected, include_known_profiles=True) == tuple(
        row.selector for row in preview.selector_provenance
    )
