"""Pinned Cursor packaging entry; component locations replace discovery."""

from pathlib import Path

from ..models import PackageInspection


def inspect_cursor(
    root: Path, catalog_overlay: dict | None = None
) -> PackageInspection:
    """Inspect the bounded Cursor view through the common inspection owner."""
    from ..inspection import inspect_package

    return inspect_package(root, dialect="cursor", catalog_overlay=catalog_overlay)
