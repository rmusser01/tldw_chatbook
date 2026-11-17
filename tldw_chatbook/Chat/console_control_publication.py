"""Named original publications after an applied local source fence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from tldw_chatbook import config
from tldw_chatbook.Utils.definition_records import (
    capture_definitions,
    definitions_current,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


@dataclass(frozen=True, slots=True)
class ReplaceConfiguration:
    identity: Any
    fence: Any
    serialized: str = field(repr=False)
    expected_snapshot: config.ConfigFileSnapshot | None = field(
        default=None, repr=False
    )
    create_backup: bool = True


@dataclass(frozen=True, slots=True)
class RefreshConfiguration:
    identity: Any
    fence: Any


@dataclass(frozen=True, slots=True)
class SetWorkspaceBindingAccess:
    identity: Any
    fence: Any
    binding_id: str
    allow_write: bool


@dataclass(frozen=True, slots=True)
class ApplySourceRevision:
    identity: Any
    publication: Any


@dataclass(frozen=True, slots=True, eq=False)
class ConsoleConfigurationPublication:
    fence: Any = field(repr=False)
    previous_sources: Any = field(repr=False)
    sources: Any = field(repr=False)
    view: str
    config_identity: Any = field(repr=False)
    original_result: Any = field(repr=False)
    _issuer: Any = field(repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class ConsoleWorkspacePublication:
    fence: Any = field(repr=False)
    sources: Any = field(repr=False)
    binding: Any = field(repr=False)
    _issuer: Any = field(repr=False)


def require_publication_definitions():
    if not definitions_current(_PUBLICATION_DEFINITIONS):
        raise RuntimeError("Console publication source changed")


async def publish_original_source(controls, command):
    """Keep the original setter, result kind and physical preparation owner."""
    from .console_preparation_reads import run_preparation_read

    fence = command.fence

    def current():
        require_publication_definitions()
        controls._require_source_publication(fence)

    def invoke():
        current()
        effect = controls._begin_publication_effect(fence)
        confirmed = False
        operation = state = None
        try:
            if type(command) is SetWorkspaceBindingAccess:
                from tldw_chatbook.Workspaces.project_index.runtime import (
                    project_index_registry_connections,
                )

                registry = fence.previous_sources.workspace_registry
                if type(registry) is not LocalWorkspaceRegistryService:
                    raise TypeError(
                        "Workspace publication requires the original registry"
                    )
                with project_index_registry_connections(registry.db):
                    binding = registry.set_folder_binding_access(
                        command.binding_id, allow_write=command.allow_write
                    )
                confirmed = True  # The original scope proved native/core closure.
                current()
                return ConsoleWorkspacePublication(
                    fence, fence.previous_sources, binding, controls
                )

            from tldw_chatbook.Backup_Recovery import (
                config_participants,
                raw_participants,
            )
            from tldw_chatbook.Backup_Recovery import storage_admission as storage
            from tldw_chatbook.MCP.local_permission_operation import (
                _state_resources_retired,
            )

            with config_participants.operation(config) as operation:
                state = raw_participants._states[operation]
                if type(command) is ReplaceConfiguration:
                    if command.expected_snapshot is None:
                        result = config.replace_cli_config_serialized(
                            command.serialized, create_backup=command.create_backup
                        )
                    else:
                        result = config.replace_cli_config_snapshot(
                            command.serialized,
                            command.expected_snapshot,
                            create_backup=command.create_backup,
                        )
                    configuration = result[0]
                    view = "loaded_bootstrap"
                else:
                    result = config.refresh_runtime_config_from_cli_config()
                    if not result.caches_reloaded or result.settings_view is None:
                        raise ConfigurationPublicationFailed(result)
                    configuration = result.settings_view
                    view = "normalized_settings"
                identity = config.current_config_identity()
            with storage._lock:
                confirmed = (
                    operation not in raw_participants._states
                    and operation not in storage._raw_operations
                    and _state_resources_retired(state, storage)
                )
            if not confirmed:
                raise RuntimeError("Configuration native retirement is unproven")
            current()
            return ConsoleConfigurationPublication(
                fence,
                fence.previous_sources,
                replace(fence.previous_sources, configuration=configuration),
                view,
                identity,
                result,
                controls,
            )
        finally:
            if operation is not None and state is not None:
                with storage._lock:
                    confirmed = (
                        operation not in raw_participants._states
                        and operation not in storage._raw_operations
                        and _state_resources_retired(state, storage)
                    )
            controls._finish_effect(effect, retired=confirmed)

    publication = await run_preparation_read(
        invoke,
        creator=controls,
        session_id=None,
        reads=controls._publication_reads,
        observers=(controls._runtime._preparation_reads,),
        source=fence,
        require_current=current,
    )
    current()
    controls._publications.add(publication)
    return publication


class ConfigurationPublicationFailed(RuntimeError):
    def __init__(self, original_result):
        super().__init__("Original configuration refresh did not publish")
        self.original_result = original_result


_PUBLICATION_DEFINITIONS = capture_definitions(
    tuple(
        (vars(config), name, getattr(config, name))
        for name in (
            "replace_cli_config_serialized",
            "replace_cli_config_snapshot",
            "_replace_cli_config_serialized",
            "_replace_cli_config_serialized_guarded",
            "refresh_runtime_config_from_cli_config",
            "_publish_runtime_config_unlocked",
            "_config_write_lock",
            "current_config_identity",
        )
    )
    + (
        (
            LocalWorkspaceRegistryService,
            "set_folder_binding_access",
            LocalWorkspaceRegistryService.set_folder_binding_access,
        ),
        (
            LocalWorkspaceRegistryService,
            "save_runtime_binding",
            LocalWorkspaceRegistryService.save_runtime_binding,
        ),
    )
)
