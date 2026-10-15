"""Cursor package adapter pinned to the 2026-09-16 documented subset."""

from pathlib import Path

from ..models import PackageInspection

ADAPTER_VERSION = "chatbook-cursor/1"


def inspect_cursor(
    root: Path, catalog_overlay: dict | None = None
) -> PackageInspection:
    """Inspect Cursor paths with manifest fields replacing catalog defaults."""
    from ..inspection import inspect_package

    return inspect_package(root, dialect="cursor", catalog_overlay=catalog_overlay)
