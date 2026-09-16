"""Retained shadow Git bytes, independent of external workspace content."""

from tldw_chatbook.Backup_Recovery.models import OwnerAdapter
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


class _ChangeTracking(_RawDeclaration):
    def discover(self, config):
        # Exact ShadowRepoService default. No Git process, hooks, restoration,
        # garbage collection or stale lock takeover is invoked by this adapter.
        return self._tree(config, user_data_dir(config) / "change_review")


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_ChangeTracking("workspaces.change_tracking"),)
