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
- NO ``todo_store`` (Console-session-scoped state; ``todo_write`` is simply
  absent from the composed catalog), NO session-approval seam, NO
  ``persist_approval`` (there is no approval to persist).

Deferred: ``record_decision`` is deliberately NOT wired. The server's audit
path for external local-tool refusals is a separate design question
(where the execution log lives for a headless server process), so refusals
here record nothing for now.

``_local_agent_tool_registrations`` turns a composed provider's catalog
into binding-ready ``LocalToolRegistration`` entries (name, description,
JSON parameters, handler); ``MCP/server.py`` binds them onto FastMCP when
``[mcp] expose_local_tools`` is enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple

from loguru import logger

from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.MCP.permission_store import resolve_effective_state

EXTERNAL_NO_CALLBACK_REFUSAL = (
    "tool requires operator approval (permission state is 'ask' and external "
    "MCP clients cannot approve); an operator must grant 'allow' for this "
    "tool in the Console or permission store"
)


def build_server_local_provider(
    workspace_root: Path, permission_store: Any
) -> LocalToolProvider:
    """Compose a LocalToolProvider for non-Console (external) MCP serving.

    resolve_state loads the store payload FRESH per call (operator changes
    take effect immediately); approval_callback is None (fail closed);
    no_callback_refusal is EXTERNAL_NO_CALLBACK_REFUSAL; kill_switch from
    the store. Follows _compose_local_provider's discipline minus the
    Console-only seams (session approvals, persist, todo store).

    Args:
        workspace_root: Confinement root for all path-taking tools.
        permission_store: An ``MCPPermissionStore``-shaped object (typed as
            ``Any`` so tests can hand in temp stores or minimal fakes);
            must provide ``load() -> dict`` and ``get_kill_switch() -> bool``.

    Returns:
        A ``LocalToolProvider`` whose catalog excludes ``todo_write`` and
        whose ask-state calls fail closed with
        ``EXTERNAL_NO_CALLBACK_REFUSAL``.
    """

    def _kill_switch() -> bool:
        # Guarded read: a raising read is treated as ENGAGED (fail closed),
        # mirroring _compose_local_provider's compose-time discipline.
        try:
            return bool(permission_store.get_kill_switch())
        except Exception as exc:  # noqa: BLE001 -- fail closed on a store read failure
            logger.warning(
                f"local_server_tools: kill-switch read failed (treating as engaged): {exc}"
            )
            return True

    return LocalToolProvider(
        workspace_root=Path(workspace_root).resolve(),
        resolve_state=lambda hub: resolve_effective_state(permission_store.load(), hub),
        kill_switch=_kill_switch,
        approval_callback=None,
        no_callback_refusal=EXTERNAL_NO_CALLBACK_REFUSAL,
    )


class LocalToolRegistration(NamedTuple):
    """One local tool ready to bind onto a FastMCP server.

    ``description``/``parameters`` come from the provider's ``load_schema``
    and are kept for introspection and future SDK versions -- FastMCP
    derives input schemas from Python type annotations, not JSON schema,
    so the binding layer registers each tool with a generic
    ``arguments: dict`` signature today.
    """

    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], Any]


def _make_registration_handler(
    provider: LocalToolProvider, tool_id: str
) -> Callable[[dict], Any]:
    """Build one tool handler with a clean ``handler(arguments: dict)`` signature.

    A factory (not a closure default-arg) keeps the signature free of
    extra parameters so FastMCP's annotation-derived schema sees only
    ``arguments``. Fail-safe: ``invoke()`` never raises, so a malformed
    (non-dict) arguments payload becomes an error dict, not an exception.
    """

    def handler(arguments: dict) -> Any:
        result = provider.invoke(tool_id, arguments)
        if result.ok:
            return result.content
        # server.py error-dict convention (server.py:187/:209).
        return {"error": result.error}

    return handler


def _parameter_summary(parameters: dict) -> str:
    """Render a compact parameter summary for appending to a tool description.

    FastMCP derives schemas from type annotations, so the generic
    ``arguments: dict`` binding leaves external clients with no parameter
    documentation; this carries the essentials (names, required, types) in
    the description instead. Returns "" when there are no properties.
    """
    properties = parameters.get("properties") or {}
    if not properties:
        return ""
    required = set(parameters.get("required") or ())
    parts = [
        f"{name}{' (required)' if name in required else ''}: {spec.get('type', 'any')}"
        for name, spec in properties.items()
    ]
    return " Parameters: " + "; ".join(parts) + "."


def _local_agent_tool_registrations(
    provider: LocalToolProvider,
) -> list[LocalToolRegistration]:
    """Build the binding-ready registration list for a provider's catalog.

    One registration per catalog entry (``todo_write`` is already absent
    from the server composition's catalog). Each handler calls
    ``provider.invoke`` and returns ``result.content`` on success or
    ``{"error": result.error}`` on refusal/failure.
    """
    registrations: list[LocalToolRegistration] = []
    for entry in provider.list_catalog():
        schema = provider.load_schema(entry.id)
        registrations.append(
            LocalToolRegistration(
                name=schema.name,
                description=schema.description,
                parameters=schema.parameters,
                handler=_make_registration_handler(provider, entry.id),
            )
        )
    return registrations
