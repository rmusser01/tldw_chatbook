"""Pinned OpenAI packaging interpretation; inline overlays replace compatibility."""

from pathlib import Path

from ..models import PackageInspection


def overlay_for_openai(inline: dict | None, compatibility: dict | None) -> dict:
    """Select one complete overlay, including explicit empty declarations."""
    return dict(inline if inline is not None else (compatibility or {}))


def inspect_codex(root: Path, catalog_overlay: dict | None = None) -> PackageInspection:
    """Inspect the bounded OpenAI view through the common inspection owner."""
    from ..inspection import inspect_package

    return inspect_package(root, dialect="openai", catalog_overlay=catalog_overlay)
