"""Atomic, authority-fenced Actor Pack archive publication."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.export import ActorPackExportSnapshot
from tldw_chatbook.Actor_Packs.publication import (
    ActorPackPublicationError,
    capture_actor_pack_destination,
    publish_actor_pack,
)

from .conftest import PNG_1X1, PORTABLE_UUID, canonical_json


def _snapshot() -> ActorPackExportSnapshot:
    return ActorPackExportSnapshot(
        actor_kind="character",
        actor_revision=1,
        portable_uuid=PORTABLE_UUID,
        identity_version=1,
        portrait_name="portrait.png",
        portrait_sha256=hashlib.sha256(PNG_1X1).hexdigest(),
        local_actor_id="private-character-id",
        actor_payload=canonical_json(
            {
                "schema": "tldw.actor/v1",
                "actor_kind": "character",
                "portable_uuid": PORTABLE_UUID,
                "data": {"name": "Published"},
            }
        ),
        portrait_bytes=PNG_1X1,
    )


def test_publish_absent_destination_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "guide.tldw-actor-pack"
    contract = capture_actor_pack_destination(destination)

    result = publish_actor_pack(_snapshot(), contract)

    assert result.committed is True
    assert result.durability in {"durable", "unsupported"}
    assert result.archive_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert not list(tmp_path.glob(".guide.tldw-actor-pack.*.tmp"))
    assert str(tmp_path) not in repr(contract)


def test_publish_replaces_only_the_exact_confirmed_destination(tmp_path: Path) -> None:
    destination = tmp_path / "confirmed.tldw-actor-pack"
    destination.write_bytes(b"confirmed incumbent")
    contract = capture_actor_pack_destination(destination)

    result = publish_actor_pack(_snapshot(), contract)

    assert result.committed is True
    assert destination.read_bytes() != b"confirmed incumbent"
    assert result.archive_sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ("cancel", "authority"))
def test_precommit_refusal_preserves_existing_destination(
    tmp_path: Path, mode: str
) -> None:
    destination = tmp_path / "existing.tldw-actor-pack"
    destination.write_bytes(b"incumbent")
    contract = capture_actor_pack_destination(destination)

    with pytest.raises(ActorPackPublicationError) as caught:
        publish_actor_pack(
            _snapshot(),
            contract,
            cancelled=(lambda: mode == "cancel"),
            authority_guard=(lambda: mode != "authority"),
        )

    expected = (
        "actor_pack_export_cancelled"
        if mode == "cancel"
        else "actor_pack_export_authority_changed"
    )
    assert caught.value.category == expected
    assert destination.read_bytes() == b"incumbent"
    assert not list(tmp_path.glob(".*.tmp"))


def test_destination_substitution_is_refused_before_replace(tmp_path: Path) -> None:
    destination = tmp_path / "substitute.tldw-actor-pack"
    destination.write_bytes(b"expected")
    contract = capture_actor_pack_destination(destination)

    def substitute(phase: str) -> None:
        if phase == "archive_fsynced":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"replacement")
            os.replace(replacement, destination)

    with pytest.raises(
        ActorPackPublicationError, match="actor_pack_export_destination_changed"
    ):
        publish_actor_pack(_snapshot(), contract, phase_hook=substitute)

    assert destination.read_bytes() == b"replacement"


def test_destination_is_revalidated_after_final_authority_callback(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "post-authority.tldw-actor-pack"
    destination.write_bytes(b"expected")
    contract = capture_actor_pack_destination(destination)

    def substitute_and_accept() -> bool:
        replacement = tmp_path / "post-authority-replacement"
        replacement.write_bytes(b"replacement")
        os.replace(replacement, destination)
        return True

    with pytest.raises(
        ActorPackPublicationError, match="actor_pack_export_destination_changed"
    ):
        publish_actor_pack(_snapshot(), contract, authority_guard=substitute_and_accept)

    assert destination.read_bytes() == b"replacement"


def test_destination_symlink_is_never_followed(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"private")
    destination = tmp_path / "linked.tldw-actor-pack"
    destination.symlink_to(target)

    with pytest.raises(
        ActorPackPublicationError, match="actor_pack_export_destination_invalid"
    ):
        capture_actor_pack_destination(destination)

    assert target.read_bytes() == b"private"


def test_parent_substitution_is_refused_and_owned_temp_is_removed(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "exports"
    parent.mkdir()
    destination = parent / "guide.tldw-actor-pack"
    contract = capture_actor_pack_destination(destination)
    displaced = tmp_path / "displaced"

    def substitute_parent(phase: str) -> None:
        if phase == "archive_fsynced":
            os.rename(parent, displaced)
            parent.mkdir()

    with pytest.raises(
        ActorPackPublicationError, match="actor_pack_export_destination_changed"
    ):
        publish_actor_pack(_snapshot(), contract, phase_hook=substitute_parent)

    assert not destination.exists()
    assert not list(displaced.glob(".*.tmp"))


def test_temporary_substitution_is_not_deleted(tmp_path: Path) -> None:
    destination = tmp_path / "temp-race.tldw-actor-pack"
    contract = capture_actor_pack_destination(destination)
    replacement: Path | None = None

    def substitute_temp(phase: str) -> None:
        nonlocal replacement
        if phase == "archive_fsynced":
            temporary = next(tmp_path.glob(".temp-race.tldw-actor-pack.*.tmp"))
            issued = tmp_path / "issued"
            os.rename(temporary, issued)
            replacement = temporary
            replacement.write_bytes(b"unrelated")

    with pytest.raises(
        ActorPackPublicationError, match="actor_pack_export_cleanup_ambiguous"
    ):
        publish_actor_pack(_snapshot(), contract, phase_hook=substitute_temp)

    assert replacement is not None and replacement.read_bytes() == b"unrelated"
    assert not destination.exists()


def test_file_fsync_precedes_replace_and_parent_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "ordering.tldw-actor-pack"
    contract = capture_actor_pack_destination(destination)
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def tracked_replace(*args, **kwargs) -> None:
        events.append("replace")
        real_replace(*args, **kwargs)

    monkeypatch.setattr("tldw_chatbook.Actor_Packs.publication.os.fsync", tracked_fsync)
    monkeypatch.setattr(
        "tldw_chatbook.Actor_Packs.publication.os.replace", tracked_replace
    )

    publish_actor_pack(_snapshot(), contract)

    assert events == ["fsync", "replace", "fsync"]


def test_replace_failure_preserves_incumbent_and_removes_owned_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "replace-failure.tldw-actor-pack"
    destination.write_bytes(b"incumbent")
    contract = capture_actor_pack_destination(destination)

    def fail_replace(*_args, **_kwargs) -> None:
        raise OSError("private replacement detail")

    monkeypatch.setattr(
        "tldw_chatbook.Actor_Packs.publication.os.replace", fail_replace
    )

    with pytest.raises(
        ActorPackPublicationError, match="actor_pack_export_publication_failed"
    ) as caught:
        publish_actor_pack(_snapshot(), contract)

    assert "private replacement detail" not in str(caught.value)
    assert destination.read_bytes() == b"incumbent"
    assert not list(tmp_path.glob(".*.tmp"))


def test_parent_fsync_failure_reports_committed_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "uncertain.tldw-actor-pack"
    contract = capture_actor_pack_destination(destination)

    def fail_parent_fsync(_parent_fd: int) -> str:
        raise OSError("private path detail")

    monkeypatch.setattr(
        "tldw_chatbook.Actor_Packs.publication._fsync_parent", fail_parent_fsync
    )

    result = publish_actor_pack(_snapshot(), contract)

    assert result.committed is True
    assert result.durability == "actor_pack_export_durability_uncertain"
    assert destination.exists()
    assert "private path detail" not in repr(result)


def test_unsupported_secure_publication_fails_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "unsupported.tldw-actor-pack"
    contract = capture_actor_pack_destination(destination)
    monkeypatch.setattr(
        "tldw_chatbook.Actor_Packs.publication._secure_publication_supported",
        lambda: False,
    )

    with pytest.raises(
        ActorPackPublicationError, match="actor_pack_export_publication_unsupported"
    ):
        publish_actor_pack(_snapshot(), contract)

    assert not destination.exists()
