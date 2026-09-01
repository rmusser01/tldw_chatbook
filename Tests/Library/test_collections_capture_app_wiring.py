"""App composition contracts for the Collections capture authority."""

from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import tldw_chatbook.app as app_module
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.collections_capture_models import (
    ExternalMediaReference,
    ExternalNoteReference,
)
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingBackend
from tldw_chatbook.Notes.notes_scope_service import ScopeType
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import CLI_APP_CLIENT_ID


class _MediaScope:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def get_backing_media_item(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _NotesScope:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def get_note_detail(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _FailingMediaScope:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def get_backing_media_item(self, **_kwargs):
        raise self.error


class _FailingNotesScope:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def get_note_detail(self, **_kwargs):
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("authority_attr", "expected_mode"),
    (
        ("local_collections_capture_authority", MediaReadingBackend.LOCAL),
        ("server_collections_capture_authority", MediaReadingBackend.SERVER),
    ),
)
async def test_media_provenance_uses_backing_media_id_and_matching_authority(
    authority_attr: str,
    expected_mode: MediaReadingBackend,
) -> None:
    media = _MediaScope({"id": "media-7"})
    app = SimpleNamespace(
        media_reading_scope_service=media,
        local_collections_capture_authority=SimpleNamespace(key="local:key"),
        server_collections_capture_authority=SimpleNamespace(key="server:key"),
    )
    authority_key = getattr(app, authority_attr).key

    availability = await app_module._resolve_collections_media_reference(
        app,
        ExternalMediaReference(authority_key, "media-7"),
    )

    assert availability.state == "available"
    assert media.calls == [
        {
            "mode": expected_mode,
            "media_id": "media-7",
            "include_content": False,
            "include_versions": False,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("authority_attr", "expected_scope", "expected_user"),
    (
        ("local_collections_capture_authority", ScopeType.LOCAL_NOTE, "local-user"),
        ("server_collections_capture_authority", ScopeType.SERVER_NOTE, None),
    ),
)
async def test_note_provenance_never_sends_a_workspace_scope(
    authority_attr: str,
    expected_scope: ScopeType,
    expected_user: str | None,
) -> None:
    notes = _NotesScope({"id": "note-9"})
    app = SimpleNamespace(
        notes_scope_service=notes,
        notes_user_id="local-user",
        local_collections_capture_authority=SimpleNamespace(key="local:key"),
        server_collections_capture_authority=SimpleNamespace(key="server:key"),
    )
    authority_key = getattr(app, authority_attr).key

    availability = await app_module._resolve_collections_note_reference(
        app,
        ExternalNoteReference(authority_key, "note-9"),
    )

    assert availability.state == "available"
    assert notes.calls == [
        {
            "scope": expected_scope,
            "note_id": "note-9",
            "user_id": expected_user,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner", "scope", "reference", "expected_reason"),
    (
        (
            "media",
            _FailingMediaScope(KeyError("missing")),
            ExternalMediaReference("local:key", "media-7"),
            "media_reference_missing",
        ),
        (
            "media",
            _FailingMediaScope(PermissionError("denied")),
            ExternalMediaReference("local:key", "media-7"),
            "media_reference_unauthorized",
        ),
        (
            "note",
            _FailingNotesScope(KeyError("missing")),
            ExternalNoteReference("local:key", "note-9"),
            "note_reference_missing",
        ),
        (
            "note",
            _FailingNotesScope(PermissionError("denied")),
            ExternalNoteReference("local:key", "note-9"),
            "note_reference_unauthorized",
        ),
    ),
)
async def test_reference_failures_are_bounded_without_owner_content(
    owner: str,
    scope: object,
    reference: ExternalMediaReference | ExternalNoteReference,
    expected_reason: str,
) -> None:
    app = SimpleNamespace(
        media_reading_scope_service=scope,
        notes_scope_service=scope,
        notes_user_id="local-user",
        local_collections_capture_authority=SimpleNamespace(key="local:key"),
        server_collections_capture_authority=SimpleNamespace(key="server:key"),
    )

    if owner == "media":
        availability = await app_module._resolve_collections_media_reference(
            app, reference
        )
    else:
        availability = await app_module._resolve_collections_note_reference(
            app, reference
        )

    assert availability.state == "unavailable"
    assert availability.reason == expected_reason


def test_local_capture_wiring_reuses_configured_collections_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "chosen" / "collections.sqlite"
    database_path.parent.mkdir()
    database = LibraryCollectionsDB(database_path, CLI_APP_CLIENT_ID)
    app = SimpleNamespace(
        local_library_collections_db=database,
        media_reading_scope_service=object(),
        notes_scope_service=object(),
        notes_user_id="local-user",
        runtime_policy=SimpleNamespace(
            state=SimpleNamespace(active_source="local", active_server_id=None)
        ),
        server_context_provider=None,
    )
    monkeypatch.setattr(
        app_module,
        "get_library_collections_db_path",
        lambda: database_path,
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path / "profile")

    TldwCli._wire_collections_capture_services(app)

    assert app.collections_capture_repository.db is database
    assert app.local_collections_capture_service.repository.db is database
    assert app.collections_capture_scope_service.active_authority.kind == "local"
    assert app.collections_legacy_recovery_service is not None


def test_server_capture_authority_ignores_workspace_changes() -> None:
    context = SimpleNamespace(
        active_server_id="profile-a",
        auth_token="secret-token",
        credential_source="test",
    )
    client = object()
    provider = SimpleNamespace(
        get_active_context=lambda: context,
        build_client=lambda: client,
    )
    app = SimpleNamespace(
        app_config={
            "library": {
                "reader": {"library_open": False},
                "collections_reader": {"items_open": False, "items_width": 53},
            }
        },
        runtime_policy=SimpleNamespace(
            state=SimpleNamespace(active_source="server", active_server_id="profile-a")
        ),
        server_context_provider=provider,
        collections_capture_scope_service=SimpleNamespace(activate=lambda *_args: None),
        server_collections_capture_authority=None,
        server_collections_capture_service=None,
        active_workspace_id="workspace-a",
    )

    TldwCli._activate_collections_capture_authority(app)
    first = app.server_collections_capture_service
    first_authority = app.server_collections_capture_authority
    preferences_before = dict(app.app_config["library"]["collections_reader"])
    app.active_workspace_id = "workspace-b"
    TldwCli._activate_collections_capture_authority(app)

    assert app.server_collections_capture_service is first
    assert app.server_collections_capture_authority == first_authority
    assert first.client is client
    assert app.app_config["library"]["collections_reader"] == preferences_before


@pytest.mark.asyncio
async def test_startup_recovery_runs_blocking_capture_work_off_loop() -> None:
    loop_thread = threading.get_ident()
    calls: list[tuple[str, int, int | None]] = []

    class _Repository:
        def interrupt_stale_extractions(self) -> None:
            calls.append(("interrupt", threading.get_ident(), None))

    class _OfflineStore:
        def reconcile_batch(self, *, limit: int) -> None:
            calls.append(("offline", threading.get_ident(), limit))

    app = SimpleNamespace(
        collections_capture_repository=_Repository(),
        collections_offline_store=_OfflineStore(),
    )

    await TldwCli._reconcile_collections_capture_startup(app)

    assert [name for name, _thread, _limit in calls] == ["interrupt", "offline"]
    assert all(thread != loop_thread for _name, thread, _limit in calls)
    assert calls[1][2] == 25


@pytest.mark.asyncio
async def test_capture_shutdown_deactivates_scope_and_cancels_extractions() -> None:
    calls: list[str] = []

    class _Scope:
        def deactivate(self) -> None:
            calls.append("deactivate")

    class _Local:
        async def cancel_extractions(self) -> None:
            calls.append("cancel")

    app = SimpleNamespace(
        collections_capture_scope_service=_Scope(),
        local_collections_capture_service=_Local(),
    )

    await TldwCli._shutdown_collections_capture_runtime(app)

    assert calls == ["deactivate", "cancel"]
