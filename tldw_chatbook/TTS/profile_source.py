"""Exact app-created profile source checks, without descendant IO authority."""

import sys
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(eq=False)
class _ConfiguredMaterializerSource:
    materializer: object
    materializer_module: object
    materializer_class: type
    bootstrap_module: object
    factory: object
    config: object
    selectors: tuple
    config_path: Path
    profile: Path
    root: Path

    def check(self, materializer) -> None:
        from .profile_reference_materialization import TTSCloneMaterializationError

        config = self.config
        cache = getattr(config, "_CONFIG_CACHE", None)
        if (
            materializer is not self.materializer
            or type(materializer) is not self.materializer_class
            or sys.modules.get("tldw_chatbook.TTS.profile_reference_materialization")
            is not self.materializer_module
            or self.materializer_module.TTSCloneReferenceMaterializer
            is not self.materializer_class
            or sys.modules.get("tldw_chatbook.TTS.adapter_bootstrap")
            is not self.bootstrap_module
            or self.bootstrap_module.build_default_tts_service is not self.factory
            or sys.modules.get("tldw_chatbook.config") is not config
            or self.bootstrap_module.get_user_data_dir is not self.selectors[1]
            or config._get_effective_config_path is not self.selectors[0]
            or config.get_user_data_dir is not self.selectors[1]
            or cache is None
            or profile_paths.lexical_path(config._get_effective_config_path())
            != self.config_path
            or config._CONFIG_CACHE_SOURCE != self.config_path
            or profile_paths.user_data_dir(cache) != self.profile
            or materializer._root != self.root
        ):
            raise TTSCloneMaterializationError("unavailable")


def bind_default_materializer(materializer) -> None:
    """Bind the original lazy factory receiver; explicit constructors stay ordinary."""
    bootstrap = sys.modules.get("tldw_chatbook.TTS.adapter_bootstrap")
    module = sys.modules.get("tldw_chatbook.TTS.profile_reference_materialization")
    config = sys.modules.get("tldw_chatbook.config")
    if (
        bootstrap is None
        or module is None
        or config is None
        or sys._getframe(1).f_code is not bootstrap.build_default_tts_service.__code__
        or type(materializer) is not module.TTSCloneReferenceMaterializer
        or getattr(config, "_CONFIG_CACHE", None) is None
        or bootstrap.get_user_data_dir is not config.get_user_data_dir
    ):
        return
    profile = profile_paths.user_data_dir(config._CONFIG_CACHE)
    root = profile / "tts_clone_materializations"
    if materializer._root != root:
        return
    source = _ConfiguredMaterializerSource(
        materializer,
        module,
        type(materializer),
        bootstrap,
        bootstrap.build_default_tts_service,
        config,
        (config._get_effective_config_path, config.get_user_data_dir),
        profile_paths.lexical_path(config._get_effective_config_path()),
        profile,
        root,
    )
    source.check(materializer)
    materializer._configured_source = source


def check_materializer_source(materializer) -> None:
    from .profile_reference_materialization import TTSCloneMaterializationError

    source = materializer._configured_source
    if source is not None:
        try:
            if type(source) is not _ConfiguredMaterializerSource:
                raise TTSCloneMaterializationError("unavailable")
            _ConfiguredMaterializerSource.check(source, materializer)
        except (ValueError, TypeError, OSError, AttributeError):
            raise TTSCloneMaterializationError("unavailable") from None
