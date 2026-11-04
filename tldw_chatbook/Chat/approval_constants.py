"""Shared capture cap for tool descriptions (widget, controller, permission summary, local tool provider)."""

#: ADR-332 (Qodo review #7): single named cap for tool-description capture
#: at the three pending-row producers (MCP, local, builtin) and the
#: summarizer prompt -- one constant so the egress bound cannot drift
#: between tool owners.
TOOL_DESCRIPTION_CAPTURE_CAP = 300


APPROVAL_OPTIONS = (
    ("Once", "approve_once"),
    ("This session", "approve_session"),
    ("Always · these args", "allow_matching"),
    ("Always", "always_allow"),
    ("Deny", "deny"),
)


def approval_options(requested: object) -> list[tuple[str, str]]:
    """Apply the native default and supported-option filter for ordinary tools."""
    if not isinstance(requested, (list, tuple)) or not requested:
        return list(APPROVAL_OPTIONS)
    wanted = set(requested)
    narrowed = [pair for pair in APPROVAL_OPTIONS if pair[1] in wanted]
    return narrowed or list(APPROVAL_OPTIONS)
