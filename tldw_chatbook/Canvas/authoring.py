"""Bounded Canvas authoring helpers; source is always HTML text, never script."""

from html import escape
from importlib.resources import files

from .limits import CanvasLimitError, validate_utf8_text
from .profiles import ProfileSnapshot, resolve_profile


def canvas_authoring_guide(snapshot: ProfileSnapshot, profile_id: str) -> str:
    """Return bounded guidance for the exact historical runtime profile.

    Args:
        snapshot: Immutable process-lifetime catalog and execution-policy view.
        profile_id: Exact runtime profile retained by the Canvas revision.

    Returns:
        Profile-specific authoring guidance when execution is admitted, or
        source-only preservation guidance when the profile is unavailable.

    Raises:
        ValueError: If ``profile_id`` is not a valid Canvas profile identifier.
        CanvasLimitError: If the packaged guide exceeds its fixed UTF-8 limit.
    """
    selected = resolve_profile(
        snapshot, operation="load", parent_profile=profile_id, has_diagrams=False
    )
    if not selected.executable:
        return (
            f"Canvas profile {selected.profile_id}: source-only; execution unavailable. "
            "Preserve source/history. Do not substitute a different profile. "
            "Repair with a current profile requires an explicit new Canvas."
        )
    if selected.profile_id == "canvas-v1":
        return (
            "Canvas profile canvas-v1: inline HTML/CSS and bounded classic scripts. "
            "No Mermaid declarations in this profile. Source acceptance does not "
            "prove preview success. No network, storage, modules, filesystem, "
            "parent DOM or Chatbook API access."
        )
    if selected.profile_id != "canvas-v2-mermaid-1":
        return (
            f"Canvas profile {selected.profile_id}: authoring guide unavailable. "
            "Preserve the exact profile; do not assume current Mermaid syntax."
        )
    guide = (
        files("tldw_chatbook.Canvas")
        .joinpath("static/mermaid-authoring.txt")
        .read_text(encoding="utf-8")
    )
    validate_utf8_text(guide, limit=8192, field_name="Canvas authoring guide")
    return guide


def wrap_mermaid_document(source: str) -> str:
    """Wrap exact Mermaid text in one complete, bounded HTML document."""
    validate_utf8_text(source, limit=8192, field_name="diagram source")
    if not source.strip():
        raise CanvasLimitError("diagram source must not be empty")
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Diagram</title></head><body>"
        '<pre data-canvas-diagram="mermaid">'
        + escape(source, quote=False)
        + "</pre></body></html>"
    )
