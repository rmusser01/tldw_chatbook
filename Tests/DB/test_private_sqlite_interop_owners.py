from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.Kanban_Interop import local_kanban_db
from tldw_chatbook.Notifications import (
    client_notifications_db,
    event_state_repository,
)
from tldw_chatbook.Research_Interop import local_research_service
from tldw_chatbook.Sync_Interop import notes_mirror, sync_state_repository
from tldw_chatbook.Utils.private_paths import PrivatePathError
from tldw_chatbook.Writing_Interop import local_writing_service


FileOwnerFactory = Callable[[Path], tuple[object, Callable[[], None]]]


def _kanban_owner(path: Path) -> tuple[object, Callable[[], None]]:
    connection = local_kanban_db.open_connection(path)
    return connection, connection.close


def _writing_owner(path: Path) -> tuple[object, Callable[[], None]]:
    return local_writing_service.LocalWritingService(path), lambda: None


def _research_owner(path: Path) -> tuple[object, Callable[[], None]]:
    return local_research_service.LocalResearchService(path), lambda: None


def _notes_mirror_owner(path: Path) -> tuple[object, Callable[[], None]]:
    owner = notes_mirror.NotesMirror(path)
    return owner, owner.close


FILE_OWNER_CASES: tuple[tuple[ModuleType, str, FileOwnerFactory], ...] = (
    (local_kanban_db, "kanban.local", _kanban_owner),
    (local_writing_service, "writing.local", _writing_owner),
    (local_research_service, "research.local", _research_owner),
    (notes_mirror, "sync.notes_mirror", _notes_mirror_owner),
)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
@pytest.mark.parametrize(("module", "owner_id", "factory"), FILE_OWNER_CASES)
def test_file_owner_uses_registered_private_seam_and_creates_0600(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    owner_id: str,
    factory: FileOwnerFactory,
) -> None:
    database = tmp_path / f"{owner_id}.db"
    calls: list[tuple[str, object, bool]] = []

    def tracking_connect(
        selected_owner: str,
        target: object,
        *,
        read_only: bool = False,
        **kwargs: object,
    ):
        calls.append((selected_owner, target, read_only))
        return private_sqlite.connect_private_sqlite(
            selected_owner,
            target,
            read_only=read_only,
            **kwargs,
        )

    monkeypatch.setattr(
        module,
        "connect_private_sqlite",
        tracking_connect,
        raising=False,
    )

    _owner, close = factory(database)
    close()

    assert calls
    assert {call[0] for call in calls} == {owner_id}
    assert all(call[2] is False for call in calls)
    assert stat.S_IMODE(database.stat().st_mode) == 0o600


@pytest.mark.parametrize(("module", "owner_id", "factory"), FILE_OWNER_CASES)
def test_file_owner_rejects_missing_parent_before_raw_sqlite_access(
    tmp_path: Path,
    module: ModuleType,
    owner_id: str,
    factory: FileOwnerFactory,
) -> None:
    del module, owner_id
    database = tmp_path / "missing" / "owner.db"

    with patch.object(
        private_sqlite.sqlite3,
        "connect",
        side_effect=AssertionError("raw SQLite access"),
    ) as raw_connect:
        with pytest.raises(PrivatePathError):
            factory(database)

    raw_connect.assert_not_called()


@pytest.mark.skipif(os.name == "nt", reason="POSIX trust contract")
@pytest.mark.parametrize(("module", "owner_id", "factory"), FILE_OWNER_CASES)
def test_file_owner_rejects_unsafe_parent_before_raw_sqlite_access(
    tmp_path: Path,
    module: ModuleType,
    owner_id: str,
    factory: FileOwnerFactory,
) -> None:
    del module, owner_id
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir()
    unsafe_parent.chmod(0o777)
    database = unsafe_parent / "owner.db"

    with patch.object(
        private_sqlite.sqlite3,
        "connect",
        side_effect=AssertionError("raw SQLite access"),
    ) as raw_connect:
        with pytest.raises(PrivatePathError):
            factory(database)

    raw_connect.assert_not_called()


MEMORY_OWNER_CASES: tuple[tuple[ModuleType, str, type], ...] = (
    (
        client_notifications_db,
        "notifications.client",
        client_notifications_db.ClientNotificationsDB,
    ),
    (
        event_state_repository,
        "notifications.event_state",
        event_state_repository.EventStateRepository,
    ),
    (
        sync_state_repository,
        "sync.state",
        sync_state_repository.SyncStateRepository,
    ),
)


@pytest.mark.parametrize(("module", "owner_id", "owner_type"), MEMORY_OWNER_CASES)
def test_persistent_memory_owner_uses_registered_seam_once_and_reuses_connection(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    owner_id: str,
    owner_type: type,
) -> None:
    calls: list[str] = []

    def tracking_connect(
        selected_owner: str,
        target: object,
        *,
        read_only: bool = False,
        **kwargs: object,
    ):
        calls.append(selected_owner)
        return private_sqlite.connect_private_sqlite(
            selected_owner,
            target,
            read_only=read_only,
            **kwargs,
        )

    monkeypatch.setattr(
        module,
        "connect_private_sqlite",
        tracking_connect,
        raising=False,
    )

    owner = owner_type(":memory:")
    first = owner._get_connection()
    second = owner._get_connection()

    assert first is second
    assert calls == [owner_id]
    owner.close()


@pytest.mark.parametrize(
    ("module", "owner_id", "factory"),
    (
        (
            local_kanban_db,
            "kanban.local",
            lambda: local_kanban_db.open_connection(":memory:"),
        ),
        (
            local_writing_service,
            "writing.local",
            lambda: local_writing_service.LocalWritingService(Path(":memory:")),
        ),
        (
            local_research_service,
            "research.local",
            lambda: local_research_service.LocalResearchService(Path(":memory:")),
        ),
        (
            notes_mirror,
            "sync.notes_mirror",
            lambda: notes_mirror.NotesMirror(":memory:"),
        ),
    ),
)
def test_path_memory_forms_route_through_registered_owner(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    owner_id: str,
    factory: Callable[[], object],
) -> None:
    calls: list[str] = []

    def tracking_connect(
        selected_owner: str,
        target: object,
        *,
        read_only: bool = False,
        **kwargs: object,
    ):
        calls.append(selected_owner)
        return private_sqlite.connect_private_sqlite(
            selected_owner,
            target,
            read_only=read_only,
            **kwargs,
        )

    monkeypatch.setattr(
        module,
        "connect_private_sqlite",
        tracking_connect,
        raising=False,
    )

    owner = factory()

    assert calls
    assert set(calls) == {owner_id}
    close = getattr(owner, "close", None)
    if callable(close):
        close()


def test_writing_path_memory_reuses_connection_and_supports_crud() -> None:
    service = local_writing_service.LocalWritingService(Path(":memory:"))
    try:
        first = service._connect()
        second = service._connect()
        project = service.create_project(title="In-memory novel")

        assert first is second
        assert service.get_project(project["id"])["title"] == "In-memory novel"
        assert [row["id"] for row in service.list_projects()] == [project["id"]]
    finally:
        service.close()


def test_research_path_memory_reuses_connection_and_supports_crud() -> None:
    service = local_research_service.LocalResearchService(Path(":memory:"))
    try:
        first = service._connect()
        second = service._connect()
        session = service.create_session(
            title="In-memory research",
            query="Verify memory persistence",
        )

        assert first is second
        assert service.get_session(session["id"])["title"] == "In-memory research"
        assert [row["id"] for row in service.list_sessions()] == [session["id"]]
    finally:
        service.close()
