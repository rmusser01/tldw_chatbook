"""Portable native Buddy archives built from pinned, verified local bytes."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from io import BytesIO

from . import importer
from .artwork import encode_native_artwork
from .repository import (
    PersonaVisualRepository,
    _source_context_json,
)
from .snapshot import BuddySnapshot, _validate_manifest, read_saved_buddy


def build_native_buddy_archive(
    snapshot: BuddySnapshot, *, source_context: dict[str, str] | None = None
) -> bytes:
    """Encode a current snapshot with original image bytes and public credits.

    Args:
        snapshot: Pinned source including its description and source context.
        source_context: Replace the snapshot context when supplied, including an
            explicitly empty mapping. Original artwork terms always remain attached.

    Raises:
        PersonaVisualImportError: Content is invalid or its source is stale.
    """
    try:
        if not snapshot.is_current():
            raise importer.PersonaVisualImportError("persona_visual_import_stale")
        _validate_manifest(snapshot.manifest_json, snapshot.assets)
        context = dict(
            snapshot.source_context if source_context is None else source_context
        )
        artwork_json = encode_native_artwork(dict(snapshot.artwork))
        if "artwork" in context and context["artwork"] != artwork_json:
            raise ValueError
        context["artwork"] = artwork_json
        _source_context_json(context)
        payloads: dict[str, bytes] = {}
        records = []
        for index, asset in enumerate(snapshot.assets):
            metadata = asset.metadata
            if (
                len(asset.data) != metadata.byte_count
                or hashlib.sha256(asset.data).hexdigest() != metadata.sha256
            ):
                raise ValueError
            suffix = {
                "image/png": "png",
                "image/jpeg": "jpg",
                "image/webp": "webp",
                "image/gif": "gif",
            }[metadata.mime_type]
            path = f"assets/buddy-{index:04d}.{suffix}"
            payloads[path] = asset.data
            record = {
                "source_asset_id": metadata.asset_key,
                "asset_role": metadata.role,
                "mime_type": metadata.mime_type,
                "asset_size_bytes": metadata.byte_count,
                "asset_sha256": metadata.sha256,
                "width": metadata.width,
                "height": metadata.height,
                "duration_ms": metadata.duration_ms,
                "asset_bytes_status": "present",
                "asset_path": path,
            }
            # Exercise the real native raster decoder before constructing an archive.
            inspected_record = {**record, "sha256": metadata.sha256}
            importer._inspect_image(BytesIO(asset.data), inspected_record)
            records.append(record)
        payloads["metadata/assets.json"] = _json({"assets": records})
        payloads["metadata/pack.json"] = _json(
            {
                "pack": {
                    "title": snapshot.title,
                    "description": snapshot.description,
                    "renderer_type": "sprite_frames",
                    "manifest_version": 1,
                    "visual_manifest": json.loads(snapshot.manifest_json),
                    "source_context": context,
                }
            }
        )
        checksums = {
            name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()
        }
        payloads["manifest.json"] = _json(
            {
                "schema_version": importer.PERSONA_VISUAL_PACK_SCHEMA,
                "encryption": {"encrypted": False},
                "sections": [
                    {"path": name, "sha256": digest}
                    for name, digest in sorted(checksums.items())
                ],
            }
        )
        checksums["manifest.json"] = hashlib.sha256(
            payloads["manifest.json"]
        ).hexdigest()
        payloads["checksums/sha256.json"] = _json(checksums)
        output = BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in sorted(payloads.items()):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100600 << 16
                archive.writestr(info, data)
        data = output.getvalue()
        with zipfile.ZipFile(BytesIO(data)) as archive:
            importer._validated_archive(archive, lambda: False)
        if not snapshot.is_current():
            raise importer.PersonaVisualImportError("persona_visual_import_stale")
        return data
    except importer.PersonaVisualImportError:
        raise
    except Exception:  # noqa: BLE001 - bounded public failure without source paths
        raise importer.PersonaVisualImportError(
            "persona_visual_import_invalid"
        ) from None


def export_persona_visual_archive(
    repository: PersonaVisualRepository,
    persona_id: str,
    profile_root: os.PathLike[str] | str,
) -> bytes:
    """Export one exact saved active Buddy without exposing local authority paths."""
    snapshot = read_saved_buddy(repository, persona_id, profile_root)
    return build_native_buddy_archive(snapshot)


def _json(value: object) -> bytes:
    return importer._canonical_text(value).encode("utf-8")
