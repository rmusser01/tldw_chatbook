"""Exact-input remembering application; permission storage confirms its own facts."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hub_tool_catalog import HubTool
    from .permission_store import MCPPermissionStore


def apply_exact_arg_rule(
    store: MCPPermissionStore | None,
    server_key: str,
    tool_name: str,
    *,
    args: Mapping[str, Any],
    tool: HubTool | None,
    profile_id: str,
    publish: Callable[..., None],
    hash_definition: Callable[..., str],
    hash_free_server_keys: Collection[str],
) -> None:
    """Write the captured exact inputs, then report only a confirmed owner fact.

    Args:
        store: Permission owner, or None when persistence is unavailable.
        server_key: Owning server's stable key.
        tool_name: Tool name within that server.
        args: Original captured arguments.
        tool: Captured definition, required for fingerprinted namespaces.
        profile_id: Exact profile receiving the rule.
        publish: Live optional observation callback from the service.
        hash_definition: Live fingerprint helper from the service.
        hash_free_server_keys: Existing service namespace exemptions.
    """
    if store is None:
        publish("not_applied", error_code="writer_unavailable")
        return
    hash_value: str | None = None
    if server_key not in hash_free_server_keys:
        if tool is None:
            raise ValueError("tool is required to add an arg rule")
        hash_value = hash_definition(tool.description, tool.input_schema)
    store.add_tool_arg_rule(
        server_key,
        tool_name,
        args=args,
        definition_hash=hash_value,
        profile_id=profile_id,
    )
    confirm = getattr(store, "has_current_tool_arg_rule", None)
    if (
        callable(confirm)
        and confirm(
            server_key,
            tool_name,
            args=args,
            definition_hash=hash_value,
            profile_id=profile_id,
        )
        is True
    ):
        publish("applied", actual_scope="allow_matching")
    else:
        publish("not_applied", error_code="authority_unavailable")
