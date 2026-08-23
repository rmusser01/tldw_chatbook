from __future__ import annotations

import importlib.util
from collections.abc import Mapping

import pytest

from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_authority import (
    NotesScopeSyncAuthority,
    NotesSyncAuthorityError,
    NotesSyncNoteSnapshot,
)


def test_notes_sync_authority_module_is_importable() -> None:
    assert (
        importlib.util.find_spec("tldw_chatbook.Notes.notes_sync_authority") is not None
    )


class RecordingLocalNotes:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.record: dict[str, object] = {
            "id": "note-1",
            "title": "Before",
            "content": "before",
            "version": 4,
            "updated_at": "2026-08-22T12:30:00+00:00",
            "deleted": 0,
        }

    def get_note_by_id(self, user_id: str, note_id: str) -> Mapping[str, object]:
        self.calls.append(("get_note_by_id", user_id, note_id))
        return dict(self.record)

    def update_note(
        self,
        user_id: str,
        note_id: str,
        update_data: dict[str, object],
        expected_version: int,
    ) -> bool:
        self.calls.append(
            ("update_note", user_id, note_id, dict(update_data), expected_version)
        )
        if expected_version != self.record["version"]:
            return False
        self.record.update(update_data)
        self.record["version"] = expected_version + 1
        return True

    def add_note(
        self,
        user_id: str,
        title: str,
        content: str,
        *,
        note_id: str,
    ) -> str:
        self.calls.append(("add_note", user_id, title, content, note_id))
        self.record = {
            "id": note_id,
            "title": title,
            "content": content,
            "version": 1,
            "deleted": 0,
        }
        return note_id

    def soft_delete_note(
        self,
        user_id: str,
        note_id: str,
        expected_version: int,
    ) -> bool:
        self.calls.append(("soft_delete_note", user_id, note_id, expected_version))
        if expected_version != self.record["version"]:
            return False
        self.record["deleted"] = 1
        self.record["version"] = expected_version + 1
        return True


class NoCallServer:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected server call: {name}")


class HostileLocalNotes(RecordingLocalNotes):
    def get_note_by_id(self, user_id: str, note_id: str) -> Mapping[str, object]:
        raise RuntimeError("credential_secret")


@pytest.mark.asyncio
async def test_sync_note_read_and_replace_route_only_through_scope_service() -> None:
    local = RecordingLocalNotes()
    service = NotesScopeService(local, NoCallServer())

    before = await service.get_note_for_sync(
        scope=ScopeType.LOCAL_NOTE,
        note_id="note-1",
        user_id="user-1",
    )
    after = await service.replace_note_for_sync(
        scope=ScopeType.LOCAL_NOTE,
        note_id="note-1",
        title="After",
        content="after",
        expected_version=4,
        user_id="user-1",
    )

    assert before == {
        "id": "note-1",
        "title": "Before",
        "content": "before",
        "version": 4,
        "updated_at": "2026-08-22T12:30:00+00:00",
        "deleted": 0,
    }
    assert after["version"] == 5
    assert after["content"] == "after"
    assert local.calls == [
        ("get_note_by_id", "user-1", "note-1"),
        (
            "update_note",
            "user-1",
            "note-1",
            {"title": "After", "content": "after"},
            4,
        ),
        ("get_note_by_id", "user-1", "note-1"),
    ]


@pytest.mark.asyncio
async def test_sync_note_replace_rejects_stale_version_without_readback() -> None:
    local = RecordingLocalNotes()
    service = NotesScopeService(local, NoCallServer())

    with pytest.raises(RuntimeError, match="stale_note"):
        await service.replace_note_for_sync(
            scope=ScopeType.LOCAL_NOTE,
            note_id="note-1",
            title="After",
            content="after",
            expected_version=3,
            user_id="user-1",
        )

    assert local.record["content"] == "before"
    assert local.calls == [
        (
            "update_note",
            "user-1",
            "note-1",
            {"title": "After", "content": "after"},
            3,
        )
    ]


@pytest.mark.asyncio
async def test_sync_note_authority_fails_closed_for_unimplemented_server_claim_seam() -> (
    None
):
    service = NotesScopeService(RecordingLocalNotes(), NoCallServer())

    with pytest.raises(RuntimeError, match="server_contract_missing"):
        await service.get_note_for_sync(
            scope=ScopeType.SERVER_NOTE,
            note_id="note-1",
        )


@pytest.mark.asyncio
async def test_private_note_authority_observes_and_replaces_through_scope_service() -> (
    None
):
    local = RecordingLocalNotes()
    authority = NotesScopeSyncAuthority(
        NotesScopeService(local, NoCallServer()),
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
    )

    before = await authority.observe("note-1")
    after = await authority.replace(
        before,
        title="After",
        content="after",
    )

    assert before.note_id == "note-1"
    assert before.version == 4
    assert before.title == "Before"
    assert before.content == "before"
    assert before.updated_at == "2026-08-22T12:30:00+00:00"
    assert (
        before.content_digest
        == "6db7d803e74f1ffa7d8f5adc0bf95b3e15bf4c8373fffadf546227cc6c6742cb"
    )
    assert after.version == 5
    assert after.content == "after"
    assert "Before" not in repr(before)
    assert "before" not in repr(before)
    assert before.content_digest not in repr(before)


@pytest.mark.asyncio
async def test_private_note_authority_creates_caller_identified_note_through_service() -> (
    None
):
    local = RecordingLocalNotes()
    authority = NotesScopeSyncAuthority(
        NotesScopeService(local, NoCallServer()),
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
    )

    created = await authority.create(
        note_id="note-created",
        title="Created",
        content="from file",
    )

    assert created.note_id == "note-created"
    assert created.version == 1
    assert created.content == "from file"
    assert local.calls == [
        ("add_note", "user-1", "Created", "from file", "note-created"),
        ("get_note_by_id", "user-1", "note-created"),
    ]

    await authority.delete(created)

    assert local.record["deleted"] == 1
    assert local.calls[-2:] == [
        ("soft_delete_note", "user-1", "note-created", 1),
        ("get_note_by_id", "user-1", "note-created"),
    ]


@pytest.mark.asyncio
async def test_private_note_authority_rejects_mismatched_readback() -> None:
    local = RecordingLocalNotes()
    authority = NotesScopeSyncAuthority(
        NotesScopeService(local, NoCallServer()),
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
    )
    before = await authority.observe("note-1")
    local.record["id"] = "different-note"

    with pytest.raises(NotesSyncAuthorityError, match="note_identity_changed"):
        await authority.replace(before, title="After", content="after")


def test_private_note_snapshot_validates_and_redacts_all_authority_values() -> None:
    with pytest.raises(ValueError, match="opaque"):
        NotesSyncNoteSnapshot(
            note_scope_id="/private/scope",
            note_id="note-1",
            title="title",
            content="body",
            version=1,
            content_digest="a" * 64,
        )


def test_private_note_snapshot_updated_at_is_typed_optional_and_validated() -> None:
    digest = "230d8358dc8e8890b4c58deeb62912ee2f20357ae92a5cc861b98e68fe31acb5"
    snapshot = NotesSyncNoteSnapshot(
        note_scope_id="local_note",
        note_id="note-1",
        title="Title",
        content="body",
        version=1,
        content_digest=digest,
        updated_at=None,
    )

    assert snapshot.updated_at is None
    with pytest.raises(ValueError, match="updated_at"):
        NotesSyncNoteSnapshot(
            note_scope_id="local_note",
            note_id="note-1",
            title="Title",
            content="body",
            version=1,
            content_digest=digest,
            updated_at="not-a-timestamp",
        )


@pytest.mark.asyncio
async def test_authority_never_promotes_raw_exception_text_to_reason_code() -> None:
    authority = NotesScopeSyncAuthority(
        NotesScopeService(HostileLocalNotes(), NoCallServer()),
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
    )

    with pytest.raises(NotesSyncAuthorityError) as captured:
        await authority.observe("note-1")

    assert captured.value.reason_code == "note_observation_failed"
    assert captured.value.__cause__ is None
    assert "credential_secret" not in repr(captured.value)
