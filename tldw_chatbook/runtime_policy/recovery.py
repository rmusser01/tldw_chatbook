"""Local source-selection recovery; never reconnect imported server profiles."""

from tldw_chatbook.Backup_Recovery.models import OwnerAdapter, discovery_context
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


class _SourceState(_RawDeclaration):
    def discover(self, config):
        return (
            self._item(
                config,
                discovery_context(config).config_path.parent / "runtime_policy.json",
            ),
        )


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    # Event/sync SQLite and notifications have their actual domain adapters.
    # Credentials are managed by the credential cohort; no keyring access here.
    return (_SourceState("runtime.source_state", "json", 16 * 1024**2),)
