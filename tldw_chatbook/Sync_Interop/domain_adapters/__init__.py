"""Local Sync v2 domain adapters for Chatbook."""

from .chat import ChatSyncAdapter
from .media import MediaSyncAdapter
from .notes import NotesSyncAdapter
from .source_cache import SourceCacheSyncAdapter
from .workspaces import WorkspacesSyncAdapter


def __getattr__(name: str):
    """Load the optional Notes organization adapter on first use."""

    if name == "NotesOrganizationSyncAdapter":
        from .notes_organization import NotesOrganizationSyncAdapter

        return NotesOrganizationSyncAdapter
    raise AttributeError(name)

__all__ = [
    "ChatSyncAdapter",
    "MediaSyncAdapter",
    "NotesSyncAdapter",
    "NotesOrganizationSyncAdapter",
    "SourceCacheSyncAdapter",
    "WorkspacesSyncAdapter",
]
