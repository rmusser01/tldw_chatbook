"""Code-owned risk metadata for the built-in MCP catalog (ADR-183)."""

CHARACTER_WRITE_TOOLS = frozenset({"create_character", "update_character"})
BUILTIN_MCP_SERVER_KEY = "builtin:tldw_chatbook"


def builtin_tool_risk_tags(name: str) -> tuple[str, ...]:
    """Return trusted tags; inventory data cannot declare built-in policy.

    Args:
        name: Canonical built-in tool name.

    Returns:
        Code-owned risk tags, or an empty tuple for an untagged tool.
    """
    return ("mutates",) if name in CHARACTER_WRITE_TOOLS else ()


def standalone_character_write_refusal(name: str) -> dict[str, str] | None:
    """Check current operator grants; standalone clients cannot approve calls.

    Args:
        name: Canonical character-write tool name being invoked.

    Returns:
        None when the current permission store explicitly allows the call;
        otherwise an error_code/error mapping explaining the refusal. The kill
        switch and unavailable permissions both deny the call.
    """
    from ..config import get_user_data_dir
    from .permission_store import MCPPermissionStore, resolve_effective_state_by_key

    try:
        store = MCPPermissionStore(get_user_data_dir() / "mcp_permissions.json")
        if store.get_kill_switch():
            return {
                "error_code": "permission_denied",
                "error": "MCP tools are disabled.",
            }
        state = resolve_effective_state_by_key(
            store.load(), BUILTIN_MCP_SERVER_KEY, name
        ).state
    except Exception:  # noqa: BLE001 -- permission lookup failures must deny writes
        return {
            "error_code": "permission_denied",
            "error": "Tool permissions are unavailable.",
        }
    if state == "allow":
        return None
    if state == "ask":
        return {
            "error_code": "permission_required",
            "error": "An operator must allow this tool in MCP Permissions before an external client can write character cards.",
        }
    return {
        "error_code": "permission_denied",
        "error": "This tool is disabled in MCP Permissions.",
    }
