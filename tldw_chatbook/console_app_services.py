"""Public app service slots read by Console source owners.

A read returns the currently installed object, including an absent service on
a partial host. Each caller retains responsibility for checking custody.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


class ConsoleApplicationServices(Protocol):
    @property
    def app_config(self) -> Mapping[str, Any] | None: ...

    @property
    def chachanotes_db(self) -> CharactersRAGDB | None: ...

    @property
    def console_runtime(self) -> ConsoleRuntime | None: ...

    @property
    def raw_cli_runtime(self) -> RawCliRuntime | None: ...

    @property
    def unified_mcp_service(self) -> UnifiedMCPControlPlaneService | None: ...

    @property
    def workspace_registry_service(self) -> LocalWorkspaceRegistryService | None: ...


def read_app_config(app: ConsoleApplicationServices | None) -> Mapping[str, Any] | None:
    try:
        return app.app_config if app is not None else None
    except AttributeError:
        return None


def read_chat_database(
    app: ConsoleApplicationServices | None,
) -> CharactersRAGDB | None:
    try:
        return app.chachanotes_db if app is not None else None
    except AttributeError:
        return None


def read_console_runtime(
    app: ConsoleApplicationServices | None,
) -> ConsoleRuntime | None:
    try:
        return app.console_runtime if app is not None else None
    except AttributeError:
        return None


def read_raw_cli_runtime(
    app: ConsoleApplicationServices | None,
) -> RawCliRuntime | None:
    try:
        return app.raw_cli_runtime if app is not None else None
    except AttributeError:
        return None


def read_mcp_service(
    app: ConsoleApplicationServices | None,
) -> UnifiedMCPControlPlaneService | None:
    try:
        return app.unified_mcp_service if app is not None else None
    except AttributeError:
        return None


def read_workspace_registry(
    app: ConsoleApplicationServices | None,
) -> LocalWorkspaceRegistryService | None:
    try:
        return app.workspace_registry_service if app is not None else None
    except AttributeError:
        return None
