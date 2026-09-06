"""Create-only pixel-migu Buddy content, installed after Actor Pack recovery."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any
from uuid import uuid4

from tldw_chatbook.Actor_Packs.activation import _cleanup_published, _publish_immutable
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Utils.private_paths import secure_private_directory

from .assets import PersonaVisualAssetMetadata, load_persona_visual_asset
from .repository import PersonaVisualRepository
from .validation import validate_persona_visual_manifest

PIXEL_MIGU_PERSONA_ID = "local-persona-builtin-pixel-migu"


def ensure_builtin_pixel_migu_buddy(
    local_service: LocalCharacterPersonaService,
    coordinator: PersonaActorPackCoordinator,
    *,
    profile_root: Path,
) -> dict[str, Any] | None:
    """Install once per profile without changing existing choices or tombstones.

    Run on the startup/readiness worker. Failed installations remain retryable;
    the Actor Pack coordinator compensates the Persona JSON on SQLite failure.

    Args:
        local_service: Profile-local character and Persona store.
        coordinator: Recovery and transaction coordinator for that store.
        profile_root: Profile directory receiving the immutable Buddy assets.

    Returns:
        The newly created or existing Persona record, including a retained
        tombstone, or ``None`` when recovery or character availability prevents
        installation.

    Raises:
        ValueError: If bundled assets, paths, or graph metadata are invalid.
        OSError: If package resources or profile storage cannot be accessed.
        PersonaActorPackCoordinatorError: If coordinated creation fails.
    """
    from tldw_chatbook.Character_Chat.builtin_pixel_migu import (
        ensure_builtin_pixel_migu,
        find_builtin_pixel_migu_character,
    )

    coordinator.ensure_recovered()
    if coordinator.recovery_error or coordinator.blocked_intent_ids:
        return None
    # Reuse the store's reentrant coordination lock to serialize racing startup
    # and first-library-read callers, including the coordinator's JSON writes.
    with local_service._persona_store_lock:
        if getattr(local_service, "_pixel_migu_seed_complete", False):
            return dict(
                local_service._find_persona_profile(
                    PIXEL_MIGU_PERSONA_ID, include_deleted=True
                )
            )
        try:
            current = local_service._find_persona_profile(
                PIXEL_MIGU_PERSONA_ID, include_deleted=True
            )
        except ValueError:
            current = None
        if current is not None:
            local_service._pixel_migu_seed_complete = True
            return dict(current)
        # A retained registry identity is also terminal; never replace a fork
        # or attempt to repair user-owned state by silently creating a Persona.
        if coordinator.repository.get_identity("persona", PIXEL_MIGU_PERSONA_ID):
            # Another service instance may have won while this one's list was
            # cached. Refresh that committed store without recreating anything.
            local_service._load_personas()
            try:
                current = local_service._find_persona_profile(
                    PIXEL_MIGU_PERSONA_ID, include_deleted=True
                )
            except ValueError:
                return None
            local_service._pixel_migu_seed_complete = True
            return dict(current)
        ensure_builtin_pixel_migu(local_service.db)
        character = find_builtin_pixel_migu_character(local_service.db)
        if character is None or character.get("deleted"):
            return None
        now = local_service._now()
        profile = {
            "id": PIXEL_MIGU_PERSONA_ID,
            "name": "pixel-migu",
            "description": "A small pixel companion for your work.",
            "character_card_id": character["id"],
            "is_active": True,
            "created_at": now,
            "last_modified": now,
            "version": 1,
            "deleted": False,
        }
        published = []
        relative = f"persona_visual/builtins/pixel_migu/{uuid4().hex}"
        repository = PersonaVisualRepository(local_service.db)
        try:
            source = files("tldw_chatbook").joinpath(
                "assets", "persona_visual", "pixel_migu"
            )
            manifest_bytes = source.joinpath("manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            declarations = json.loads(source.joinpath("assets.json").read_bytes())
            # Share the strict wire-schema validator used by import, authoring,
            # persistence and runtime instead of maintaining a second schema.
            validate_persona_visual_manifest(
                manifest, known_assets={row["asset_key"] for row in declarations}
            )
            # Each attempt owns its files. A racing loser can clean up without
            # removing bytes another service instance has already committed.
            digest = hashlib.sha256(manifest_bytes).hexdigest()
            target = Path(profile_root) / relative
            if not secure_private_directory(
                target, create=True, application_owned=True
            ).verified_private:
                raise ValueError("pixel_migu_publication_denied")
            assets = []
            for declaration in declarations:
                metadata_fields = dict(declaration)
                filename = metadata_fields.pop("filename")
                publication_path = validate_path(filename, target, redact_paths=True)
                if publication_path.name != filename:
                    raise ValueError("pixel_migu_asset_path_invalid")
                metadata = PersonaVisualAssetMetadata(**metadata_fields)
                # as_file supports individual resources on Python 3.11 as well
                # as 3.12, including ZIP-backed importlib Traversables.
                with as_file(source.joinpath(filename)) as materialized:
                    # Canonicalize temporary-directory aliases while leaving
                    # the resource leaf for the loader's no-follow admission.
                    source_parent = materialized.parent.resolve(strict=True)
                    asset = load_persona_visual_asset(
                        source_parent,
                        storage_key=materialized.name,
                        metadata=metadata,
                    )
                created = _publish_immutable(
                    publication_path, asset.data, metadata.sha256
                )
                if created is not None:
                    published.append(created)
                metadata_fields["bytes"] = metadata_fields.pop("byte_count")
                assets.append(
                    {
                        **metadata_fields,
                        "storage_relpath": f"{relative}/{filename}",
                    }
                )
            created = _publish_immutable(
                validate_path("manifest.json", target, redact_paths=True),
                manifest_bytes,
                digest,
            )
            if created is not None:
                published.append(created)
            coordinator.create_persona(
                profile,
                portable_uuid=str(uuid4()),
                sqlite_effect=lambda: repository._activate_new_pack_in_transaction(
                    persona_id=PIXEL_MIGU_PERSONA_ID,
                    title="pixel-migu",
                    manifest=manifest,
                    manifest_storage_relpath=f"{relative}/manifest.json",
                    assets=assets,
                    expected_persona_revision=1,
                    source_context={
                        "provenance": "bundled-pixel-migu",
                        "license": "LicenseRef-User-Supplied",
                    },
                ),
            )
        except Exception:
            # An ambiguous postcommit error must not remove files now owned by
            # the graph. If durable ownership cannot be read, retain the files.
            try:
                committed = repository.get_active_persona_pack_for_export(
                    PIXEL_MIGU_PERSONA_ID
                )
                owned = committed is not None and any(
                    asset.storage_key.startswith(f"{relative}/")
                    for asset in committed.assets
                )
            except Exception:  # noqa: BLE001 - retain files on uncertain ownership
                owned = True
            if not owned:
                _cleanup_published(published)
            raise
        local_service._pixel_migu_seed_complete = True
        return dict(profile)
