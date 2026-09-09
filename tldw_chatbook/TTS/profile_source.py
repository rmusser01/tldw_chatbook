"""Exact app-created profile source checks, without descendant IO authority."""

from dataclasses import dataclass
from pathlib import Path
import sys

from tldw_chatbook.Backup_Recovery import profile_paths
from .profile_errors import ProfileRepositoryError


@dataclass(eq=False)
class _ConfiguredProfileSource:
    repository: object
    repository_class: type
    repository_module: object
    config: object
    functions: tuple
    config_path: Path
    profile: Path
    database: Path

    def check(self, repository) -> None:
        config = self.config
        cache = getattr(config, "_CONFIG_CACHE", None)
        if (
            repository is not self.repository
            or type(repository) is not self.repository_class
            or sys.modules.get("tldw_chatbook.TTS.profile_repository")
            is not self.repository_module
            or self.repository_module.TTSProfileRepository is not self.repository_class
            or sys.modules.get("tldw_chatbook.config") is not config
            or any(
                getattr(config, name, None) is not original
                for name, original in zip(_SELECTOR_FUNCTIONS, self.functions)
            )
            or cache is None
            or profile_paths.lexical_path(config._get_effective_config_path())
            != self.config_path
            or config._CONFIG_CACHE_SOURCE != self.config_path
            or profile_paths.user_data_dir(cache) != self.profile
            or profile_paths.database_path(cache, "tts_profiles_db_path")
            != self.database
            or repository._database_path != self.database
        ):
            raise ProfileRepositoryError("unavailable")


_SELECTOR_FUNCTIONS = (
    "_get_effective_config_path",
    "get_user_data_dir",
    "get_tts_profiles_db_path",
)


def bind_app_repository(app) -> None:
    """Bind only the original constructor's actual configured repository receiver."""
    app_module = sys.modules.get("tldw_chatbook.app")
    repository_module = sys.modules.get("tldw_chatbook.TTS.profile_repository")
    config = sys.modules.get("tldw_chatbook.config")
    # One exact composition edge, never stack discovery or shape-based enrollment.
    if (
        app_module is None
        or type(app) is not app_module.TldwCli
        or sys._getframe(1).f_code is not app_module.TldwCli.__init__.__code__
        or repository_module is None
        or config is None
        or getattr(config, "_CONFIG_CACHE", None) is None
    ):
        return
    repository = app._tts_profile_repository
    if type(repository) is not repository_module.TTSProfileRepository:
        return
    selected = profile_paths.database_path(config._CONFIG_CACHE, "tts_profiles_db_path")
    if repository._database_path != selected:
        return  # Explicit custom constructor paths retain ordinary behavior.
    if repository._configured_source is not None:
        repository._configured_source.check(repository)
        return
    source = _ConfiguredProfileSource(
        repository,
        type(repository),
        repository_module,
        config,
        tuple(getattr(config, name) for name in _SELECTOR_FUNCTIONS),
        profile_paths.lexical_path(config._get_effective_config_path()),
        profile_paths.user_data_dir(config._CONFIG_CACHE),
        selected,
    )
    source.check(repository)
    repository._configured_source = source


def check_repository_source(repository) -> None:
    source = repository._configured_source
    if source is not None:
        try:
            if type(source) is not _ConfiguredProfileSource:
                raise ProfileRepositoryError("unavailable")
            _ConfiguredProfileSource.check(source, repository)
        except (ValueError, TypeError, OSError, AttributeError):
            raise ProfileRepositoryError("unavailable") from None
