"""Server sync transport interoperability services."""

from .key_recovery_service import SyncKeyRecoveryService
from .chat_outbox_producer import ChatSyncV2OutboxProducer
from .local_first_sync_service import LocalFirstSyncService
from .manual_sync_control import ManualSyncControlService
from .restore_service import SyncRestoreService
from .server_sync_service import ServerSyncService
from .sync_scope_service import SyncBackend, SyncScopeService
from .sync_state_repository import SyncStateRepository


def __getattr__(name: str):
    """Load the Notes producer only when Notes mutation wiring requests it."""

    if name == "NotesSyncV2OutboxProducer":
        from .notes_outbox_producer import NotesSyncV2OutboxProducer

        return NotesSyncV2OutboxProducer
    raise AttributeError(name)

__all__ = [
    "LocalFirstSyncService",
    "ManualSyncControlService",
    "ChatSyncV2OutboxProducer",
    "NotesSyncV2OutboxProducer",
    "ServerSyncService",
    "SyncBackend",
    "SyncKeyRecoveryService",
    "SyncRestoreService",
    "SyncScopeService",
    "SyncStateRepository",
]
