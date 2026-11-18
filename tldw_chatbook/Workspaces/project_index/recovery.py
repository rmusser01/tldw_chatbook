"""Inert declaration for excluded cache namespaces, including detached bindings."""

from dataclasses import replace

from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


class ProjectIndexCacheDeclaration(_RawDeclaration):
    def discover(self, config):
        return tuple(
            replace(item, status="intentionally_excluded")
            if item.status in {"included", "included_directory", "unused"}
            else item
            for item in self._tree(config, user_data_dir(config) / "project_index")
        )

    def capture(self, item, destination, cancel):
        raise ValueError("project_index_payload_excluded")


def recovery_adapters():
    return (
        ProjectIndexCacheDeclaration("db.workspace_project_index", max_bytes=1024**3),
    )
