"""Shared capture cap for tool descriptions (widget, controller, permission summary, local tool provider)."""

#: ADR-090 (Qodo review #7): single named cap for tool-description capture
#: at the three pending-row producers (MCP, local, builtin) and the
#: summarizer prompt -- one constant so the egress bound cannot drift
#: between tool owners.
TOOL_DESCRIPTION_CAPTURE_CAP = 300
