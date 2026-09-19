"""Compatibility-preserving lazy public exports; recovery imports no services."""

from importlib import import_module

_EXPORTS = {
    "SyncKeyRecoveryService": (".key_recovery_service", "SyncKeyRecoveryService"),
    "ChatSyncV2OutboxProducer": (".chat_outbox_producer", "ChatSyncV2OutboxProducer"),
    "LocalFirstSyncService": (".local_first_sync_service", "LocalFirstSyncService"),
    "ManualSyncControlService": (".manual_sync_control", "ManualSyncControlService"),
    "NotesSyncV2OutboxProducer": (
        ".notes_outbox_producer",
        "NotesSyncV2OutboxProducer",
    ),
    "SyncRestoreService": (".restore_service", "SyncRestoreService"),
    "ServerSyncService": (".server_sync_service", "ServerSyncService"),
    "SyncBackend": (".sync_scope_service", "SyncBackend"),
    "SyncScopeService": (".sync_scope_service", "SyncScopeService"),
    "SyncStateRepository": (".sync_state_repository", "SyncStateRepository"),
}


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


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
