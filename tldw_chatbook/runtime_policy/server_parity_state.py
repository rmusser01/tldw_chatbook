"""Authoritative local repositories for server-parity runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


EVENT_STATE_DB_FILENAME = "tldw_chatbook_event_state.db"
SYNC_STATE_DB_FILENAME = "tldw_chatbook_sync_state.db"


@dataclass(slots=True)
class ServerParityStateRepositories:
    """Single bundle for local state that backs server-parity features.

    `ClientNotificationsDB` remains the local notification inbox authority.
    Server event cursors/dedupe and sync mirror state live in separate durable
    repositories to avoid mixing server-owned state into local notifications.
    """

    local_notifications_db: ClientNotificationsDB
    event_state_repository: EventStateRepository
    sync_state_repository: SyncStateRepository

    def clear_server_profile_state(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
    ) -> None:
        self.event_state_repository.clear_server_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
        )
        self.sync_state_repository.clear_server_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
        )

    def close(self) -> None:
        self.event_state_repository.close()
        self.sync_state_repository.close()


def build_server_parity_state_repositories(
    *,
    data_dir: str | Path,
    client_id: str,
    local_notifications_db: ClientNotificationsDB,
) -> ServerParityStateRepositories:
    resolved_data_dir = Path(data_dir).expanduser().resolve()
    resolved_data_dir.mkdir(parents=True, exist_ok=True)
    return ServerParityStateRepositories(
        local_notifications_db=local_notifications_db,
        event_state_repository=EventStateRepository(
            resolved_data_dir / EVENT_STATE_DB_FILENAME,
            client_id,
        ),
        sync_state_repository=SyncStateRepository(
            resolved_data_dir / SYNC_STATE_DB_FILENAME,
            client_id,
        ),
    )
