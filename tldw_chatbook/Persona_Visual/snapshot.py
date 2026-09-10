"""Read-only, bounded native Buddy archive snapshots for explicit publication."""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from io import BytesIO
from types import MappingProxyType
from typing import Any

from . import importer
from .assets import (
    PersonaVisualAssetMetadata,
    _decode_selected_frame,
    validate_persona_visual_asset_set,
)
from .validation import validate_persona_visual_manifest


@dataclass(frozen=True, slots=True)
class BuddyAssetSnapshot:
    """Immutable validated source bytes and their native raster metadata."""

    metadata: PersonaVisualAssetMetadata
    data: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class BuddySnapshot:
    """Path-free reviewed content with private source-revalidation authority."""

    title: str
    manifest_json: str
    assets: tuple[BuddyAssetSnapshot, ...]
    artwork: Mapping[str, Any]
    source_sha256: str
    _guard: Callable[[], bool] = field(repr=False, compare=False)

    source_context: tuple[tuple[str, str], ...] = ()
    description: str = ""
    source_kind: str = "imported"

    def __post_init__(self) -> None:
        """Detach inert scalar artwork fields from mutable review inputs."""
        object.__setattr__(self, "artwork", MappingProxyType(dict(self.artwork)))

    def is_current(self) -> bool:
        """Fail closed if the reviewed source changed or cannot be revalidated."""
        try:
            return self._guard() is True
        except Exception:  # noqa: BLE001 - public boundary must fail closed without paths
            return False


def _validate_manifest(
    manifest_json: str, assets: tuple[BuddyAssetSnapshot, ...]
) -> None:
    validate_persona_visual_asset_set(tuple(asset.metadata for asset in assets))
    validate_persona_visual_manifest(
        manifest_json,
        {
            asset.metadata.asset_key: (asset.metadata.width, asset.metadata.height)
            for asset in assets
        },
    )


def read_buddy_archive(path: os.PathLike[str] | str) -> BuddySnapshot:
    """Read a pinned native archive without staging files or creating a Persona.

    Args:
        path: Absolute native archive filename. The source must be a regular,
            singly linked file; no-follow checks reject a symbolic link.

    Returns:
        Immutable validated artwork, assets and metadata with a private guard
        that revalidates the source before publication.

    Raises:
        PersonaVisualImportError: Source access or reading fails (the
            ``persona_visual_import_failed`` category), path or native archive
            validation fails, the format is unsupported, or the source changes
            during review. Exception categories contain no private path text.
    """
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    try:
        # The importer owns no-follow identity checks; do not resolve links here.
        validated_path = validate_path_simple(path, probe_existing=False)
        try:
            source = importer._pin_source(validated_path)
        except OSError:
            # A source read failure says nothing about the archive's validity.
            # Keep this narrow: decoder errors later still mean invalid content.
            raise importer.PersonaVisualImportError(
                "persona_visual_import_failed"
            ) from None
        with zipfile.ZipFile(BytesIO(source.data)) as archive:
            members, pack, records = importer._validated_archive(archive, lambda: False)
            assets = []
            for record in records:
                data = archive.read(members[record["asset_path"]])
                if (
                    len(data) != record["byte_count"]
                    or hashlib.sha256(data).hexdigest() != record["sha256"]
                ):
                    raise ValueError
                metadata = PersonaVisualAssetMetadata(
                    **{
                        key: record[key]
                        for key in (
                            "asset_key",
                            "role",
                            "mime_type",
                            "byte_count",
                            "sha256",
                            "width",
                            "height",
                        )
                    }
                )
                metadata = validate_persona_visual_asset_set((metadata,))[0]
                # Enforce the native decoded-pixel budget before decoding all frames.
                _decode_selected_frame(data, metadata, 0)
                frame_count, duration_ms = importer._inspect_image(
                    BytesIO(data), record
                )
                metadata = PersonaVisualAssetMetadata(
                    **{
                        key: getattr(metadata, key)
                        for key in (
                            "asset_key",
                            "role",
                            "mime_type",
                            "byte_count",
                            "sha256",
                            "width",
                            "height",
                        )
                    },
                    frame_count=frame_count,
                    duration_ms=duration_ms,
                )
                assets.append(BuddyAssetSnapshot(metadata, data))
            manifest_json = importer._canonical_text(pack["visual_manifest"])
            frozen_assets = tuple(assets)
            _validate_manifest(manifest_json, frozen_assets)
            raw_pack = importer._json_member(archive, members, "metadata/pack.json")[
                "pack"
            ]
            from .artwork import artwork_from_pack

            artwork = artwork_from_pack(raw_pack)
        source_path, source_identity, source_digest = (
            source.path,
            source.identity,
            source.sha256,
        )
        snapshot = BuddySnapshot(
            pack["title"],
            manifest_json,
            frozen_assets,
            artwork,
            source.sha256,
            lambda: importer._source_identity_current(
                source_path, source_identity, source_digest
            ),
            tuple(
                sorted(
                    {**pack["source_context"], "provenance": "untrusted-import"}.items()
                )
            ),
        )
        if not snapshot.is_current():
            raise importer.PersonaVisualImportError("persona_visual_import_stale")
        return snapshot
    except importer.PersonaVisualImportError:
        raise
    except Exception:  # noqa: BLE001 - public boundary must fail closed without paths
        raise importer.PersonaVisualImportError(
            "persona_visual_import_invalid"
        ) from None
