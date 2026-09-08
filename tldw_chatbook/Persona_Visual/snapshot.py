"""Validated, read-only Buddy sources for reviewed character conversion."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from io import BytesIO
from types import MappingProxyType
from typing import Any

from tldw_chatbook.Character_Chat.artwork_attribution import (
    ARTWORK_NAMESPACE,
    artwork_context,
)

from . import importer
from .assets import (
    PersonaVisualAssetMetadata,
    _decode_selected_frame,
    load_persona_visual_asset,
    validate_persona_visual_asset_set,
)
from .repository import PersonaVisualRepository, decode_native_artwork
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

    def __post_init__(self) -> None:
        """Detach inert scalar artwork fields from mutable review inputs."""
        object.__setattr__(self, "artwork", MappingProxyType(dict(self.artwork)))

    def is_current(self) -> bool:
        """Fail closed if the reviewed source changed or cannot be revalidated."""
        try:
            return self._guard() is True
        except Exception:  # noqa: BLE001 - public boundary must fail closed without paths
            return False


def _artwork(source: Mapping[str, Any]) -> dict[str, Any]:
    # Operational provenance and local source IDs never become creator credits.
    context = source.get("source_context", {})
    if isinstance(context, Mapping) and "artwork" in context:
        return decode_native_artwork(context["artwork"])
    if "artwork" in source:
        return decode_native_artwork(source["artwork"])
    if isinstance(context, Mapping) and ARTWORK_NAMESPACE in context:
        return artwork_context(context)[ARTWORK_NAMESPACE]
    if ARTWORK_NAMESPACE in source:
        return artwork_context(source)[ARTWORK_NAMESPACE]
    record = {
        "version": 1,
        "creator": source.get("creator"),
        "license": source.get("license"),
        "source_url": source.get("source_url"),
        "notices": source.get("notices", ""),
    }
    return artwork_context({ARTWORK_NAMESPACE: record})[ARTWORK_NAMESPACE]


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

    Raises:
        PersonaVisualImportError: Native validation fails or the source changes.
    """
    try:
        source = importer._pin_source(path)
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
            artwork = _artwork(raw_pack)
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


def read_saved_buddy(
    repository: PersonaVisualRepository,
    persona_id: str,
    profile_root: os.PathLike[str] | str,
) -> BuddySnapshot:
    """Pin an active saved graph and verified bytes; retain no runtime binding.

    The guard re-reads the graph and asset bytes, catching binding/version changes,
    metadata edits, file corruption and deletion before character publication.
    """
    try:
        exported = repository.get_active_persona_pack_for_export(persona_id)
        if exported is None:
            raise ValueError
        assets = []
        for item in exported.assets:
            record = item.record
            metadata = PersonaVisualAssetMetadata(
                **{
                    key: getattr(record, key)
                    for key in (
                        "asset_key",
                        "role",
                        "mime_type",
                        "byte_count",
                        "sha256",
                        "width",
                        "height",
                        "frame_count",
                        "duration_ms",
                    )
                }
            )
            loaded = load_persona_visual_asset(
                profile_root, storage_key=item.storage_key, metadata=metadata
            )
            assets.append(BuddyAssetSnapshot(loaded.metadata, loaded.data))
        frozen_assets = tuple(assets)
        manifest_json = exported.manifest_bytes.decode("utf-8")
        _validate_manifest(manifest_json, frozen_assets)
        artwork = _artwork(dict(exported.source_context))
        # A portable content digest excludes local identity, paths and timestamps.
        digest = hashlib.sha256(
            importer._canonical_text(
                {
                    "title": exported.graph.pack.title,
                    "manifest": json.loads(manifest_json),
                    "assets": sorted(
                        (asset.metadata.asset_key, asset.metadata.sha256)
                        for asset in frozen_assets
                    ),
                    "artwork": artwork,
                }
            ).encode()
        ).hexdigest()

        def current() -> bool:
            if repository.get_active_persona_pack_for_export(persona_id) != exported:
                return False
            for item, asset in zip(exported.assets, frozen_assets, strict=True):
                loaded = load_persona_visual_asset(
                    profile_root,
                    storage_key=item.storage_key,
                    metadata=asset.metadata,
                )
                if loaded.data != asset.data:
                    return False
            return True

        snapshot = BuddySnapshot(
            exported.graph.pack.title,
            manifest_json,
            frozen_assets,
            artwork,
            digest,
            current,
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
