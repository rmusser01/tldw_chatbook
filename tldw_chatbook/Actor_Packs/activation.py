"""Transactional activation of one still-current Actor Pack review."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
from tldw_chatbook.Character_Chat.visual_identity import (
    compute_pack_content_sha256,
    validate_visual_identity_manifest,
)
from tldw_chatbook.Utils.private_paths import secure_private_directory

from .importer import (
    ActorPackImportError,
    ActorPackImportReview,
    ActorPackImportService,
    _ActorPackSectionMaterial,
)
from .contracts import canonical_json_bytes
from .persona_coordinator import (
    PersonaActorPackCoordinator,
    PersonaActorPackCoordinatorError,
)
from .repository import ActorPackRepository, ActorPackRepositoryError


class ActorPackActivationError(ValueError):
    """One fixed, path-free activation failure."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackActivationResult:
    """Path-free committed actor identity."""

    actor_kind: str
    local_actor_id: str
    portable_uuid: str
    cleanup_pending: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedSharedVisual:
    pack: dict[str, Any]
    manifest: dict[str, Any]
    assets: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _PreparedPersonaVisual:
    manifest: dict[str, Any]
    manifest_storage_relpath: str
    assets: tuple[dict[str, Any], ...]


class ActorPackActivationService:
    """Consume immutable reviews at the existing Character/Persona boundaries."""

    def __init__(
        self,
        db: CharactersRAGDB,
        local_service: LocalCharacterPersonaService,
        repository: ActorPackRepository,
        persona_coordinator: PersonaActorPackCoordinator,
        importer: ActorPackImportService,
    ) -> None:
        if (
            not isinstance(db, CharactersRAGDB)
            or not isinstance(local_service, LocalCharacterPersonaService)
            or not isinstance(repository, ActorPackRepository)
            or not isinstance(persona_coordinator, PersonaActorPackCoordinator)
            or not isinstance(importer, ActorPackImportService)
        ):
            raise ActorPackActivationError("actor_pack_import_activation_invalid")
        self.db = db
        self.local_service = local_service
        self.repository = repository
        self.persona_coordinator = persona_coordinator
        self.importer = importer
        self.visual_identity_repository = VisualIdentityRepository(db)
        self.persona_visual_repository = PersonaVisualRepository(db)

    def activate(
        self,
        review: ActorPackImportReview,
        action: str,
        *,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> ActorPackActivationResult:
        """Activate one action offered by the exact review."""

        if (
            type(review) is not ActorPackImportReview
            or type(action) is not str
            or action not in review.allowed_actions
        ):
            raise ActorPackActivationError("actor_pack_import_action_invalid")
        self._raise_if_cancelled(cancel_requested)
        try:
            material = self.importer._activation_material(review)
            if review.actor_kind == "character":
                result = self._activate_character(
                    review,
                    action,
                    dict(material.actor_fields),
                    material.portrait,
                    material.sections,
                    cancel_requested,
                )
            else:
                result = self._activate_persona(
                    review,
                    action,
                    dict(material.actor_fields),
                    material.sections,
                    cancel_requested,
                )
        except ActorPackActivationError:
            raise
        except ActorPackImportError as exc:
            raise ActorPackActivationError(exc.category) from None
        except PersonaActorPackCoordinatorError as exc:
            category = (
                "actor_pack_import_cancelled"
                if exc.category == "actor_pack_creation_cancelled"
                else "actor_pack_import_activation_failed"
            )
            raise ActorPackActivationError(category) from None
        except (ActorPackRepositoryError, sqlite3.Error, TypeError, ValueError):
            raise ActorPackActivationError(
                "actor_pack_import_activation_failed"
            ) from None
        try:
            self.importer.cleanup_review(review)
        except ActorPackImportError:
            return ActorPackActivationResult(
                result.actor_kind,
                result.local_actor_id,
                result.portable_uuid,
                cleanup_pending=True,
            )
        return result

    def _activate_character(
        self,
        review: ActorPackImportReview,
        action: str,
        fields: dict[str, Any],
        portrait: bytes,
        sections: tuple[_ActorPackSectionMaterial, ...],
        cancel_requested: Callable[[], bool],
    ) -> ActorPackActivationResult:
        prepared = self._prepare_character_sections(review, sections)
        if action == "update_existing":
            if (
                review._matched_local_actor_id is None
                or review._matched_actor_version is None
            ):
                raise ActorPackActivationError("actor_pack_import_review_stale")
            character_id = int(review._matched_local_actor_id)
            fields["image"] = portrait
            with self.db.transaction(immediate=True) as cursor:
                self._raise_if_cancelled(cancel_requested)
                self.importer.revalidate_review(review)
                self.db._update_character_card_in_transaction(
                    cursor,
                    character_id,
                    fields,
                    expected_version=review._matched_actor_version,
                    require_outermost=True,
                )
                self._activate_shared_visual(prepared, "character", character_id)
            return ActorPackActivationResult(
                "character", str(character_id), review.portable_uuid
            )
        assigned = (
            review.portable_uuid
            if action == "create_new"
            else self.repository._new_portable_uuid()
        )
        source_uuid = review.portable_uuid if action == "create_copy" else None
        fields["image"] = portrait
        with self.db.transaction(immediate=True) as cursor:
            self._raise_if_cancelled(cancel_requested)
            self.importer.revalidate_review(review)
            if action == "create_copy":
                fields["name"] = self._copy_character_name(cursor, str(fields["name"]))
            character_id = self.db._insert_character_card_in_transaction(
                cursor,
                fields,
                require_outermost=True,
            )
            identity = self.repository._assign_identity_in_transaction(
                "character",
                character_id,
                source_portable_uuid=source_uuid,
                portable_uuid=assigned,
            )
            self._activate_shared_visual(prepared, "character", character_id)
        return ActorPackActivationResult(
            "character", str(character_id), identity.portable_uuid
        )

    def _activate_persona(
        self,
        review: ActorPackImportReview,
        action: str,
        fields: dict[str, Any],
        sections: tuple[_ActorPackSectionMaterial, ...],
        cancel_requested: Callable[[], bool],
    ) -> ActorPackActivationResult:
        shared, persona_visual = self._prepare_persona_sections(review, sections)
        if action == "update_existing":
            if (
                review._matched_local_actor_id is None
                or review._matched_actor_version is None
            ):
                raise ActorPackActivationError("actor_pack_import_review_stale")
            persona_id = review._matched_local_actor_id
            current = dict(self.local_service._find_persona_profile(persona_id))
            current.update(fields)
            current["last_modified"] = self.local_service._now()
            current["version"] = review._matched_actor_version + 1
            new_authority = (
                current["version"],
                hashlib.sha256(canonical_json_bytes(current)).hexdigest(),
            )
            self.importer.revalidate_review(review)
            self._raise_if_cancelled(cancel_requested)
            committed = self.persona_coordinator.create_persona(
                current,
                portable_uuid=review.portable_uuid,
                operation="update",
                cancel_requested=cancel_requested,
                authority_guard=lambda: self._review_is_current(
                    review,
                    alternate_actor_authorities=(new_authority,),
                ),
                sqlite_effect=lambda: self._activate_persona_sections(
                    shared,
                    persona_visual,
                    persona_id,
                    current["version"],
                ),
            )
            return ActorPackActivationResult(
                "persona",
                persona_id,
                committed.identity.portable_uuid,
                cleanup_pending=committed.cleanup_pending,
            )
        self.importer.revalidate_review(review)
        self._raise_if_cancelled(cancel_requested)
        persona_id = f"local-persona-{uuid4().hex}"
        now = self.local_service._now()
        profile = {
            **fields,
            "id": persona_id,
            "created_at": now,
            "last_modified": now,
            "version": 1,
            "deleted": False,
        }
        portable_uuid = (
            review.portable_uuid
            if action == "create_new"
            else self.repository._new_portable_uuid()
        )
        committed = self.persona_coordinator.create_persona(
            profile,
            portable_uuid=portable_uuid,
            operation="create" if action == "create_new" else "copy",
            source_portable_uuid=(
                review.portable_uuid if action == "create_copy" else None
            ),
            cancel_requested=cancel_requested,
            authority_guard=lambda: self._review_is_current(review),
            sqlite_effect=lambda: self._activate_persona_sections(
                shared,
                persona_visual,
                persona_id,
                profile["version"],
            ),
        )
        return ActorPackActivationResult(
            "persona",
            persona_id,
            committed.identity.portable_uuid,
            cleanup_pending=committed.cleanup_pending,
        )

    def _prepare_character_sections(
        self,
        review: ActorPackImportReview,
        sections: tuple[_ActorPackSectionMaterial, ...],
    ) -> _PreparedSharedVisual | None:
        if not sections:
            return None
        if len(sections) != 1 or sections[0].kind != "shared-visual-identity":
            raise ActorPackActivationError(
                "actor_pack_import_section_activation_failed"
            )
        section = sections[0]
        try:
            manifest = json.loads(canonical_json_bytes(section.manifest))
            raw_assets = manifest["assets"]
            if type(raw_assets) is not list or len(raw_assets) != len(section.assets):
                raise ValueError
            publication_root = (
                self.importer._profile_root
                / "visual_identities"
                / "actor_packs"
                / review.content_digest
            )
            privacy = secure_private_directory(
                publication_root, create=True, application_owned=True
            )
            if not privacy.verified_private:
                raise ValueError
            asset_rows: list[dict[str, Any]] = []
            for index, ((asset, data), raw) in enumerate(
                zip(section.assets, raw_assets, strict=True), start=1
            ):
                suffix = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                }[asset.mime_type]
                filename = f"asset-{index:04d}{suffix}"
                target = publication_root / filename
                _publish_immutable(target, data, asset.sha256)
                storage_relpath = f"actor_packs/{review.content_digest}/{filename}"
                raw["storage_relpath"] = storage_relpath
                row = dict(raw)
                row["original_expression_key"] = row.pop("original_label")
                row["source_filename"] = filename
                row["source_context"] = {"provenance": "actor-pack-import"}
                asset_rows.append(row)
            manifest["pack_content_sha256"] = compute_pack_content_sha256(manifest)
            validated = validate_visual_identity_manifest(manifest)
            pack = {
                "title": validated.title,
                "default_expression_key": validated.default_expression_key,
                "source_kind": "manual",
                "source_context": {
                    **dict(review.provenance),
                    **{f"license_{key}": value for key, value in review.license},
                },
            }
            return _PreparedSharedVisual(pack, manifest, tuple(asset_rows))
        except (KeyError, OSError, TypeError, ValueError):
            raise ActorPackActivationError(
                "actor_pack_import_section_activation_failed"
            ) from None

    def _prepare_persona_sections(
        self,
        review: ActorPackImportReview,
        sections: tuple[_ActorPackSectionMaterial, ...],
    ) -> tuple[_PreparedSharedVisual | None, _PreparedPersonaVisual | None]:
        kinds = tuple(section.kind for section in sections)
        if len(kinds) != len(set(kinds)) or any(
            kind not in {"shared-visual-identity", "persona-runtime"} for kind in kinds
        ):
            raise ActorPackActivationError(
                "actor_pack_import_section_activation_failed"
            )
        shared_section = tuple(
            section for section in sections if section.kind == "shared-visual-identity"
        )
        shared = (
            None
            if not shared_section
            else self._prepare_character_sections(review, shared_section)
        )
        runtime = next(
            (section for section in sections if section.kind == "persona-runtime"),
            None,
        )
        if runtime is None:
            return shared, None
        try:
            publication_root = (
                self.importer._profile_root
                / "actor_packs"
                / review.content_digest
                / "persona-runtime"
            )
            privacy = secure_private_directory(
                publication_root, create=True, application_owned=True
            )
            if not privacy.verified_private:
                raise ValueError
            assets: list[dict[str, Any]] = []
            for index, (asset, data) in enumerate(runtime.assets, start=1):
                suffix = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                }[asset.mime_type]
                filename = f"asset-{index:04d}{suffix}"
                _publish_immutable(publication_root / filename, data, asset.sha256)
                assets.append(
                    {
                        "asset_key": asset.asset_key,
                        "role": "frame",
                        "storage_relpath": (
                            f"actor_packs/{review.content_digest}/"
                            f"persona-runtime/{filename}"
                        ),
                        "mime_type": asset.mime_type,
                        "bytes": asset.byte_count,
                        "sha256": asset.sha256,
                        "width": asset.width,
                        "height": asset.height,
                        "frame_count": 1,
                        "duration_ms": None,
                    }
                )
            manifest = json.loads(canonical_json_bytes(runtime.manifest))
            manifest_name = "manifest.json"
            manifest_bytes = canonical_json_bytes(manifest)
            _publish_immutable(
                publication_root / manifest_name,
                manifest_bytes,
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
            return (
                shared,
                _PreparedPersonaVisual(
                    manifest,
                    (
                        f"actor_packs/{review.content_digest}/"
                        f"persona-runtime/{manifest_name}"
                    ),
                    tuple(assets),
                ),
            )
        except (KeyError, OSError, TypeError, ValueError):
            raise ActorPackActivationError(
                "actor_pack_import_section_activation_failed"
            ) from None

    def _activate_persona_sections(
        self,
        shared: _PreparedSharedVisual | None,
        persona_visual: _PreparedPersonaVisual | None,
        persona_id: str,
        persona_revision: int,
    ) -> None:
        self._activate_shared_visual(shared, "persona", persona_id)
        if persona_visual is not None:
            self.persona_visual_repository._activate_new_pack_in_transaction(
                persona_id=persona_id,
                title="Imported Persona Visual",
                manifest=persona_visual.manifest,
                manifest_storage_relpath=persona_visual.manifest_storage_relpath,
                assets=persona_visual.assets,
                expected_persona_revision=persona_revision,
                source_context={"provenance": "actor-pack-import"},
            )

    def _activate_shared_visual(
        self,
        prepared: _PreparedSharedVisual | None,
        actor_kind: str,
        actor_id: int | str,
    ) -> None:
        if prepared is None:
            return
        self.visual_identity_repository.activate_pack(
            pack=prepared.pack,
            manifest=prepared.manifest,
            assets=prepared.assets,
            actor_kind=actor_kind,
            actor_id=actor_id,
        )

    def _review_is_current(
        self,
        review: ActorPackImportReview,
        *,
        alternate_actor_authorities: tuple[tuple[int, str], ...] = (),
    ) -> bool:
        try:
            self.importer.revalidate_review(
                review,
                alternate_actor_authorities=alternate_actor_authorities,
            )
        except ActorPackImportError:
            return False
        return True

    @staticmethod
    def _copy_character_name(cursor: sqlite3.Cursor, name: str) -> str:
        candidate = f"{name} (Copy)"
        suffix = 2
        while cursor.execute(
            "SELECT 1 FROM character_cards WHERE name = ? AND deleted = 0",
            (candidate,),
        ).fetchone():
            candidate = f"{name} (Copy {suffix})"
            suffix += 1
        return candidate

    @staticmethod
    def _raise_if_cancelled(checker: Callable[[], bool]) -> None:
        try:
            cancelled = checker()
        except Exception:
            cancelled = True
        if cancelled is True:
            raise ActorPackActivationError("actor_pack_import_cancelled")


def _publish_immutable(target: Path, data: bytes, expected_sha256: str) -> None:
    """Create or attest one content-addressed private immutable asset."""

    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        metadata = os.lstat(target)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(data)
        ):
            raise ValueError
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 64 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        if digest.hexdigest() != expected_sha256:
            raise ValueError
        return
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ActorPackActivationError",
    "ActorPackActivationResult",
    "ActorPackActivationService",
]
