"""Server-side composition of the local agent tool provider for external MCP clients.

Non-Console MCP serving has no approval card and no session scope, so this
composition differs from the Console's ``_compose_local_provider``
(``Chat/console_chat_controller.py``) in exactly those seams:

- ``resolve_state`` loads the ``MCPPermissionStore`` payload FRESH per call
  (operator changes take effect immediately -- ``load()`` never raises, so
  a missing/corrupt store file resolves to the ask-default payload and the
  provider fails closed).
- ``approval_callback`` is None -- external clients can never approve, so an
  ask-state call fails closed with ``EXTERNAL_NO_CALLBACK_REFUSAL`` (the
  provider's ``no_callback_refusal`` override) instead of the Console's
  misleading timeout copy. The operator-grant path is the Console's
  "Always allow" (tool-level ``allow`` + ``definition_hash`` under
  ``local:__local__``), which resolves to "allow" here.
- ``kill_switch`` reads the store's kill switch guarded: a raising read is
  treated as engaged (fail closed), mirroring the controller's compose-time
  discipline.
- NO Console ``SessionTodoStore`` (so ``todo_create``, ``todo_update``,
  ``todo_get``, and ``todo_list`` are unregistered; the retired
  ``todo_write`` is also absent), NO session-approval seam, NO
  ``persist_approval`` (there is no approval to persist).

Deferred: ``record_decision`` is deliberately NOT wired. The server's audit
path for external local-tool refusals is a separate design question
(where the execution log lives for a headless server process), so refusals
here record nothing for now.

``_local_agent_tool_registrations`` turns a composed provider's catalog
into binding-ready ``LocalToolRegistration`` entries (name, description,
JSON parameters, handler); ``MCP/server.py`` stages them on the gateway when
``[mcp] expose_local_tools`` is enabled.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable, NamedTuple

from loguru import logger

from tldw_chatbook.Agents.agent_models import ToolResult
# task-24458: deferred to the one runtime construction site below. This
# module is reached by the screen pre-importer through
# `UI/MCP_Modules/mcp_workbench.py`, and a module-scope import here puts
# `Tools.workspace_tool_executor` plus ~9 further modules into the
# pre-import payload. This module has no `from __future__ import
# annotations`, so the three `LocalToolProvider` annotations are QUOTED --
# quoting three names is a smaller change than switching the whole module
# to string annotations.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.config import get_subscriptions_db_path
from tldw_chatbook.DB.Subscriptions_DB import (
    SubscriptionsDB,
    SubscriptionsDBReadError,
    SubscriptionsDBUnavailableError,
)
from tldw_chatbook.MCP.permission_store import resolve_effective_state
from tldw_chatbook.runtime_policy.bootstrap import (
    load_default_runtime_source_state,
)
from tldw_chatbook.Tools.watchlists_tool_service import WatchlistsToolService

EXTERNAL_NO_CALLBACK_REFUSAL = (
    "tool requires operator approval (permission state is 'ask' and external "
    "MCP clients cannot approve); an operator must grant 'allow' for this "
    "tool in the Console or permission store"
)


class _LazyWatchlistsDBResolver:
    """Open and retain one external-MCP read-only Watchlists database."""

    def __init__(self) -> None:
        self._database: SubscriptionsDB | None = None
        self._pending_cleanup: tuple[SubscriptionsDB, Exception] | None = None
        self._lock = threading.Lock()

    def __call__(self) -> SubscriptionsDB:
        database = self._database
        if database is not None:
            return database

        with self._lock:
            database = self._database
            if database is not None:
                return database

            if self._pending_cleanup is not None:
                candidate, failure = self._pending_cleanup
                try:
                    candidate.close()
                except Exception:  # noqa: BLE001 -- retry same cleanup later
                    raise failure from None
                self._pending_cleanup = None

            candidate: SubscriptionsDB | None = None
            try:
                candidate = SubscriptionsDB(get_subscriptions_db_path(), read_only=True)
                candidate.assert_agent_read_ready()
            except Exception as failure:
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:  # noqa: BLE001 -- preserve readiness failure
                        if isinstance(failure, SubscriptionsDBUnavailableError):
                            retained_failure = SubscriptionsDBUnavailableError()
                        elif isinstance(failure, SubscriptionsDBReadError):
                            retained_failure = SubscriptionsDBReadError()
                        elif isinstance(failure, FileNotFoundError):
                            retained_failure = FileNotFoundError()
                        elif isinstance(failure, ImportError):
                            retained_failure = ImportError()
                        else:
                            retained_failure = RuntimeError(
                                "Watchlists database initialization failed"
                            )
                        self._pending_cleanup = (candidate, retained_failure)
                raise

            self._database = candidate
            return candidate


def build_server_local_provider(
    workspace_root: Path, permission_store: Any
) -> "LocalToolProvider":
    """Compose a LocalToolProvider for non-Console (external) MCP serving.

    resolve_state loads the store payload FRESH per call (operator changes
    take effect immediately); approval_callback is None (fail closed);
    no_callback_refusal is EXTERNAL_NO_CALLBACK_REFUSAL; kill_switch from
    the store. Follows _compose_local_provider's discipline minus the
    Console-only seams (session approvals, persist, SessionTodoStore).

    Args:
        workspace_root: Confinement root for all path-taking tools.
        permission_store: An ``MCPPermissionStore``-shaped object (typed as
            ``Any`` so tests can hand in temp stores or minimal fakes);
            must provide ``load() -> dict`` and ``get_kill_switch() -> bool``.

    Returns:
        A ``LocalToolProvider`` whose catalog excludes ``todo_create``,
        ``todo_update``, ``todo_get``, ``todo_list``, and the retired
        ``todo_write``, and whose ask-state calls fail closed with
        ``EXTERNAL_NO_CALLBACK_REFUSAL``.
    """

    def _kill_switch() -> bool:
        # Guarded read: a raising read is treated as ENGAGED (fail closed),
        # mirroring _compose_local_provider's compose-time discipline.
        try:
            return bool(permission_store.get_kill_switch())
        except Exception:  # noqa: BLE001 -- fail closed on a store read failure
            logger.warning(
                "Local MCP tool kill-switch state unavailable; failing closed."
            )
            return True

    def _resolve_state(hub: Any) -> Any:
        try:
            payload = permission_store.load()
            return resolve_effective_state(payload, hub)
        except Exception:  # noqa: BLE001 -- provider maps the fixed failure safely
            raise RuntimeError("permission state unavailable") from None

    watchlists_service = WatchlistsToolService(
        db_resolver=_LazyWatchlistsDBResolver(),
        # Owner-module loader (TASK-18609): constructing the store here
        # violated the runtime-policy ownership boundary.
        runtime_source_loader=load_default_runtime_source_state,
    )
    # task-24458: every `local_tool_provider` name this function needs is
    # imported here rather than at module scope. That module pulls the whole
    # workspace tool-execution cluster, and this module is reached by the
    # screen pre-importer via `UI/MCP_Modules/mcp_workbench.py`.
    from tldw_chatbook.Agents.local_tool_provider import (
        LocalToolExposure,
        LocalToolProvider,
        WorkspaceToolExecutor,
        _default_specs,
    )

    resolved_root = Path(workspace_root).resolve()
    workspace_executor = WorkspaceToolExecutor(resolved_root)
    external_specs = [
        spec
        for spec in _default_specs(
            resolved_root,
            workspace_executor=workspace_executor,
            watchlists_service=watchlists_service,
        )
        if spec.exposure is LocalToolExposure.CONSOLE_AND_EXTERNAL_MCP
    ]

    return LocalToolProvider(
        workspace_root=resolved_root,
        specs=external_specs,
        resolve_state=_resolve_state,
        kill_switch=_kill_switch,
        approval_callback=None,
        no_callback_refusal=EXTERNAL_NO_CALLBACK_REFUSAL,
        workspace_executor=workspace_executor,
    )


class LocalToolRegistration(NamedTuple):
    """One local tool ready for all-or-none gateway publication."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]


def _make_registration_handler(
    provider: "LocalToolProvider", tool_id: str
) -> Callable[[dict[str, Any]], ToolResult]:
    """Return a handler that preserves the provider's canonical result."""

    def handler(arguments: dict[str, Any]) -> ToolResult:
        return provider.invoke(tool_id, arguments)

    return handler


def _local_agent_tool_registrations(
    provider: "LocalToolProvider",
) -> list[LocalToolRegistration]:
    """Build the binding-ready registration list for a provider's catalog.

    One registration per descriptor explicitly marked for Console and
    external MCP publication. Each handler returns the exact ``ToolResult``
    from ``provider.invoke`` for the gateway to classify.
    """
    # task-24458: deferred with the rest of this module's
    # `local_tool_provider` imports.
    from tldw_chatbook.Agents.local_tool_provider import LocalToolExposure

    registrations: list[LocalToolRegistration] = []
    for spec in provider.specs_for_exposure(
        LocalToolExposure.CONSOLE_AND_EXTERNAL_MCP
    ):
        schema = provider.load_schema(spec.name)
        registrations.append(
            LocalToolRegistration(
                name=schema.name,
                description=schema.description,
                parameters=schema.parameters,
                handler=_make_registration_handler(provider, spec.name),
            )
        )
    return registrations


def local_tools_exposure_enabled() -> bool:
    """The `[mcp] expose_local_tools` gate, coerced.

    Lives here (not in server.py) because Tests/MCP pin server.py to never
    call ``get_cli_setting`` directly. Coercion at the consumer matters:
    ``get_cli_setting`` reads the raw TOML tree, so a quoted ``"false"``
    would otherwise be truthy and fail this security-relevant gate OPEN.
    """
    from ..config import coerce_bool_setting, get_cli_setting

    return coerce_bool_setting(
        get_cli_setting("mcp", "expose_local_tools", False), False
    )


def resolve_server_workspace_root() -> Path:
    """The workspace root for external MCP serving (Console's rule).

    ``[console] workspace_root`` with ``~`` expanded, else the process cwd.
    """
    import os

    from ..config import get_cli_setting

    raw = (get_cli_setting("console", "workspace_root", "") or "").strip()
    return Path(raw).expanduser().resolve() if raw else Path(os.getcwd()).resolve()
