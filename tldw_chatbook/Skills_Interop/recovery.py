"""Installed local skill definitions and quarantined trust-byte inventory."""

from dataclasses import replace
from pathlib import Path

from tldw_chatbook.Backup_Recovery.config_adapter import _Definition
from tldw_chatbook.Backup_Recovery.models import OwnerAdapter


def default_local_skills_store_dir(user_data_dir: str | Path) -> Path:
    """Canonical app-owned skills root, shared with the runtime selector."""
    return Path(user_data_dir) / "skills"


class _Skills(_Definition):
    def discover(self, config):
        entries = super().discover(config)
        root = next(
            (
                item.path
                for item in entries
                if item.metadata and item.metadata.relative_path == ""
            ),
            None,
        )
        if root is None:
            return entries
        return tuple(
            replace(item, status="unsupported")
            if item.path
            and item.path != root
            and item.path.relative_to(root).parts[0]
            not in {"tldw_chatbook_skills.json", "skills", "trust"}
            else item
            for item in entries
        )


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    # Trust manifests/grants/snapshots remain historical bytes: importing a grant
    # never grants execution. Credential processing belongs to task14.
    return (_Skills("skills", leaf="skills", tree=True, participant_pending=True),)
