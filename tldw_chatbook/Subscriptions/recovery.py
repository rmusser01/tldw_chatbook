"""Managed briefing audio recovery; exported feeds are external user outputs."""

from tldw_chatbook.Backup_Recovery.models import OwnerAdapter
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


class _BriefingAudio(_RawDeclaration):
    def discover(self, config):
        return self._tree(config, user_data_dir(config) / "briefing_audio")


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    # SiteConfigManager stores definitions in the subscriptions DB owned by the
    # DB cohort; it is not a second physical capture of that file.
    return (_BriefingAudio("subscriptions.assets"),)
