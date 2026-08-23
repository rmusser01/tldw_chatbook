from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import replace

import pytest

from tldw_chatbook.Notes.note_folder_models import (
    FolderCollisionError,
    NoteFolder,
    NoteFolderMembership,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_authority import (
    ConflictNoteRequest,
    ManualFolderRequest,
    ManualPlacementRequest,
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


class RecordingFolderRepository:
    def __init__(self) -> None:
        self.folders: dict[str, NoteFolder] = {}
        self.memberships: dict[tuple[str, str], NoteFolderMembership] = {}
        self.deleted_memberships: set[tuple[str, str]] = set()
        self.folder_creates = 0
        self.placement_creates = 0
        self.race_folder: NoteFolder | None = None
        self.managed_folder_ids: set[str] = set()

    def get_folder_by_path(self, segments: tuple[str, ...]) -> NoteFolder | None:
        normalized = "/" + "/".join(segment.strip().casefold() for segment in segments)
        return next(
            (
                folder
                for folder in self.folders.values()
                if not folder.deleted and folder.normalized_path == normalized
            ),
            None,
        )

    def get_folder(
        self, folder_id: str, *, include_deleted: bool = False
    ) -> NoteFolder | None:
        folder = self.folders.get(folder_id)
        if folder is None or (folder.deleted and not include_deleted):
            return None
        return folder

    def has_managed_folder_ownership(self, folder_id: str) -> bool:
        return folder_id in self.managed_folder_ids

    def create_folder(
        self, *, name: str, parent_id: str | None, folder_id: str
    ) -> NoteFolder:
        self.folder_creates += 1
        if self.race_folder is not None:
            self.folders[self.race_folder.folder_id] = self.race_folder
            raise FolderCollisionError("lost race")
        parent_path = "" if parent_id is None else self.folders[parent_id].path
        parent_normalized = (
            "" if parent_id is None else self.folders[parent_id].normalized_path
        )
        folder = NoteFolder(
            folder_id=folder_id,
            parent_id=parent_id,
            name=name,
            path="/".join(value for value in (parent_path, name) if value),
            normalized_path="/"
            + "/".join(
                value for value in (parent_normalized, name.casefold()) if value
            ).lstrip("/"),
            version=1,
            deleted=False,
        )
        self.folders[folder_id] = folder
        return folder

    def list_memberships(
        self, *, note_ids: tuple[str, ...], include_inactive: bool = False
    ) -> tuple[NoteFolderMembership, ...]:
        del include_inactive
        return tuple(
            membership
            for membership in self.memberships.values()
            if membership.note_id in note_ids
        )

    def get_exact_manual_membership(
        self,
        *,
        folder_id: str,
        note_id: str,
        include_deleted: bool = False,
    ) -> tuple[NoteFolderMembership, bool] | None:
        del include_deleted
        membership = self.memberships.get((folder_id, note_id))
        return (
            None
            if membership is None
            else (membership, (folder_id, note_id) in self.deleted_memberships)
        )

    def attach_manual(
        self,
        *,
        folder_id: str,
        note_id: str,
        expected_note_version: int | None = None,
    ) -> NoteFolderMembership:
        del expected_note_version
        self.placement_creates += 1
        membership = NoteFolderMembership(
            membership_id=f"membership-{self.placement_creates}",
            folder_id=folder_id,
            note_id=note_id,
            ownership="manual",
            owner_id="",
            owner_active=True,
            version=1,
        )
        self.memberships[(folder_id, note_id)] = membership
        return membership


def _folder(
    folder_id: str,
    name: str,
    *,
    parent_id: str | None = None,
    parent_path: str = "",
    version: int = 1,
    deleted: bool = False,
) -> NoteFolder:
    path = "/".join(value for value in (parent_path, name) if value)
    return NoteFolder(
        folder_id=folder_id,
        parent_id=parent_id,
        name=name,
        path=path,
        normalized_path=f"/{path.casefold()}",
        version=version,
        deleted=deleted,
    )


def _authority_with_folders() -> tuple[
    NotesScopeSyncAuthority, RecordingLocalNotes, RecordingFolderRepository
]:
    local = RecordingLocalNotes()
    folders = RecordingFolderRepository()
    authority = NotesScopeSyncAuthority(
        NotesScopeService(local, NoCallServer(), folder_repository=folders),
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
    )
    return authority, local, folders


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
async def test_private_note_authority_maps_production_last_modified_to_updated_at() -> (
    None
):
    local = RecordingLocalNotes()
    local.record.pop("updated_at")
    local.record["last_modified"] = "2026-08-22T13:45:00+00:00"
    authority = NotesScopeSyncAuthority(
        NotesScopeService(local, NoCallServer()),
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
    )

    observed = await authority.observe("note-1")

    assert observed.updated_at == "2026-08-22T13:45:00+00:00"


@pytest.mark.asyncio
async def test_private_note_authority_prefers_updated_at_over_last_modified() -> None:
    local = RecordingLocalNotes()
    local.record["last_modified"] = "2026-08-22T13:45:00+00:00"
    authority = NotesScopeSyncAuthority(
        NotesScopeService(local, NoCallServer()),
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
    )

    observed = await authority.observe("note-1")

    assert observed.updated_at == "2026-08-22T12:30:00+00:00"


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


@pytest.mark.asyncio
async def test_conflict_copy_manual_folder_reuses_actual_normalized_path_id() -> None:
    authority, _local, repository = _authority_with_folders()
    existing = _folder("actual-folder", "Conflict copies", version=7)
    repository.folders[existing.folder_id] = existing

    verified = await authority.create_or_verify_manual_folder(
        ManualFolderRequest(
            folder_id="deterministic-folder",
            parent_id=None,
            name=" conflict copies ",
            path_segments=("Conflict copies",),
        )
    )

    assert (verified.folder_id, verified.version) == ("actual-folder", 7)
    assert repository.folder_creates == 0


@pytest.mark.asyncio
async def test_conflict_copy_manual_folder_lost_race_rereads_exact_winner_once() -> (
    None
):
    authority, _local, repository = _authority_with_folders()
    repository.race_folder = _folder("winner-folder", "Conflict copies", version=3)

    verified = await authority.create_or_verify_manual_folder(
        ManualFolderRequest(
            folder_id="deterministic-folder",
            parent_id=None,
            name="Conflict copies",
            path_segments=("Conflict copies",),
        )
    )

    assert (verified.folder_id, verified.version) == ("winner-folder", 3)
    assert repository.folder_creates == 1


@pytest.mark.asyncio
async def test_conflict_copy_manual_folder_rejects_id_path_or_parent_collision() -> (
    None
):
    authority, _local, repository = _authority_with_folders()
    repository.folders["deterministic-folder"] = _folder(
        "deterministic-folder", "Different"
    )
    request = ManualFolderRequest(
        folder_id="deterministic-folder",
        parent_id=None,
        name="Conflict copies",
        path_segments=("Conflict copies",),
    )

    with pytest.raises(NotesSyncAuthorityError, match="folder_authority_changed"):
        await authority.create_or_verify_manual_folder(request)

    assert repository.folder_creates == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("winner_kind", ("deterministic_id", "normalized_path"))
@pytest.mark.parametrize("level", ("parent", "child"))
async def test_conflict_copy_manual_folder_rejects_managed_parent_or_child_winner(
    winner_kind: str,
    level: str,
) -> None:
    authority, _local, repository = _authority_with_folders()
    parent_id = None if level == "parent" else "verified-parent"
    name = "Conflict copies" if level == "parent" else "My synced notes"
    segments = (
        ("Conflict copies",)
        if level == "parent"
        else (
            "Conflict copies",
            "My synced notes",
        )
    )
    requested_id = f"deterministic-{level}"
    winner_id = requested_id if winner_kind == "deterministic_id" else f"actual-{level}"
    winner = _folder(
        winner_id,
        name,
        parent_id=parent_id,
        parent_path="" if level == "parent" else "Conflict copies",
        version=7,
    )
    repository.folders[winner.folder_id] = winner
    repository.managed_folder_ids.add(winner.folder_id)

    with pytest.raises(NotesSyncAuthorityError, match="folder_authority_changed"):
        await authority.create_or_verify_manual_folder(
            ManualFolderRequest(
                folder_id=requested_id,
                parent_id=parent_id,
                name=name,
                path_segments=segments,
            )
        )

    assert repository.folder_creates == 0


@pytest.mark.asyncio
async def test_conflict_copy_note_reuses_only_exact_active_content() -> None:
    authority, local, _repository = _authority_with_folders()
    local.record = {
        "id": "copy-note",
        "title": "Original",
        "content": "note side",
        "version": 9,
        "deleted": 0,
    }
    request = ConflictNoteRequest(
        note_id="copy-note",
        title="Original",
        content="note side",
    )

    reused = await authority.create_or_verify_conflict_note(request)

    assert (reused.note_id, reused.version) == ("copy-note", 9)
    assert not any(call[0] == "add_note" for call in local.calls)
    local.record["content"] = "collision"
    with pytest.raises(NotesSyncAuthorityError, match="conflict_copy_collision"):
        await authority.create_or_verify_conflict_note(request)


@pytest.mark.asyncio
async def test_conflict_copy_note_lost_uniqueness_race_verifies_winner() -> None:
    class RacingNotes(RecordingLocalNotes):
        def __init__(self) -> None:
            super().__init__()
            self.record = {}

        def get_note_by_id(
            self, user_id: str, note_id: str
        ) -> Mapping[str, object] | None:
            self.calls.append(("get_note_by_id", user_id, note_id))
            return dict(self.record) if self.record else None

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
                "version": 4,
                "deleted": 0,
            }
            raise RuntimeError("unique constraint secret")

    local = RacingNotes()
    authority = NotesScopeSyncAuthority(
        NotesScopeService(local, NoCallServer()),
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
    )

    verified = await authority.create_or_verify_conflict_note(
        ConflictNoteRequest("copy-note", "Original", "note side")
    )

    assert verified.version == 4
    assert sum(call[0] == "add_note" for call in local.calls) == 1


@pytest.mark.asyncio
async def test_conflict_copy_manual_placement_reuses_actual_id_and_rejects_managed() -> (
    None
):
    authority, _local, repository = _authority_with_folders()
    repository.memberships[("folder-1", "copy-note")] = NoteFolderMembership(
        membership_id="actual-placement",
        folder_id="folder-1",
        note_id="copy-note",
        ownership="manual",
        owner_id="",
        owner_active=True,
        version=6,
    )
    request = ManualPlacementRequest("folder-1", "copy-note", 9)

    reused = await authority.create_or_verify_manual_placement(request)

    assert (reused.membership_id, reused.version) == ("actual-placement", 6)
    assert repository.placement_creates == 0
    repository.memberships[("folder-1", "copy-note")] = replace(
        repository.memberships[("folder-1", "copy-note")],
        ownership="managed",
        owner_id="root-1",
    )
    with pytest.raises(NotesSyncAuthorityError, match="placement_authority_changed"):
        await authority.create_or_verify_manual_placement(request)


@pytest.mark.asyncio
async def test_conflict_copy_deleted_manual_placement_fails_without_reviving() -> None:
    authority, _local, repository = _authority_with_folders()
    pair = ("folder-1", "copy-note")
    repository.memberships[pair] = NoteFolderMembership(
        membership_id="deleted-placement",
        folder_id=pair[0],
        note_id=pair[1],
        ownership="manual",
        owner_id="",
        owner_active=True,
        version=4,
    )
    repository.deleted_memberships.add(pair)

    with pytest.raises(NotesSyncAuthorityError, match="placement_authority_changed"):
        await authority.create_or_verify_manual_placement(
            ManualPlacementRequest(*pair, 9)
        )

    assert repository.placement_creates == 0


@pytest.mark.asyncio
async def test_conflict_copy_create_or_verify_is_local_only() -> None:
    authority = NotesScopeSyncAuthority(
        NotesScopeService(RecordingLocalNotes(), NoCallServer()),
        scope=ScopeType.SERVER_NOTE,
        note_scope_id="server_note",
    )

    with pytest.raises(NotesSyncAuthorityError, match="server_contract_missing"):
        await authority.create_or_verify_conflict_note(
            ConflictNoteRequest("copy-note", "Original", "note side")
        )
