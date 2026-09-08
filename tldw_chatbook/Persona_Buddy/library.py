"""Profile-local Buddy library; artwork ownership never creates a Persona."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Visual.artwork import (
    artwork_from_pack,
    encode_native_artwork,
)
from tldw_chatbook.Persona_Visual.assets import (
    PersonaVisualAssetMetadata,
    load_persona_visual_asset,
)
from tldw_chatbook.Persona_Visual.publication import (
    PersonaVisualPublicationAssetSource,
    PersonaVisualPublicationSnapshot,
    cleanup_persona_visual_publication_candidate,
    publish_persona_visual,
)
from tldw_chatbook.Persona_Visual.repository import (
    PersonaVisualGraph,
    PersonaVisualRepository,
)
from tldw_chatbook.Persona_Visual.runtime import (
    PersonaVisualResolution,
    resolve_active_buddy_visual,
)
from tldw_chatbook.Persona_Visual.snapshot import (
    BuddyAssetSnapshot,
    BuddySnapshot,
    read_buddy_archive,
)

if TYPE_CHECKING:
    from .preferences import PersonaBuddyPreferences

_PUBLICATION_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class BuddyRecord:
    """Path-free library identity, independent of prompts and Persona lifecycle."""

    id: str
    name: str
    revision: int
    source_key: str | None = None


class BuddyLibrary:
    """Blocking library operations; callers run I/O on their retained app worker."""

    def __init__(
        self,
        db: CharactersRAGDB,
        profile_root: Path,
        *,
        persona_reader: Callable[[str], Mapping[str, object]] | None = None,
    ) -> None:
        self.db = db
        self.profile_root = Path(profile_root)
        self.repository = PersonaVisualRepository(db)
        self.persona_reader = persona_reader

    def list_buddies(self) -> tuple[BuddyRecord, ...]:
        rows = self.db.execute_query(
            "SELECT buddy.id,buddy.name,buddy.version,buddy.source_key FROM buddy_profiles buddy "
            "JOIN buddy_visual_bindings binding ON binding.buddy_id=buddy.id "
            "WHERE buddy.status='active' AND binding.status='active' "
            "AND binding.buddy_revision=buddy.version ORDER BY buddy.name COLLATE NOCASE,buddy.id"
        ).fetchall()
        return tuple(BuddyRecord(*row) for row in rows)

    def get_buddy(self, buddy_id: str) -> BuddyRecord | None:
        return next((row for row in self.list_buddies() if row.id == buddy_id), None)

    def get_graph(self, buddy_id: str) -> PersonaVisualGraph | None:
        return self.repository.get_active_buddy_pack(buddy_id)

    def resolve_preview(
        self, buddy_id: str, *, state: str = "idle", reduced_motion: bool = False
    ) -> PersonaVisualResolution:
        return resolve_active_buddy_visual(
            self.repository,
            buddy_id,
            self.profile_root,
            state,
            reduced_motion=reduced_motion,
        )

    @staticmethod
    def review_archive(path: Path | str) -> BuddySnapshot:
        """Validate archive bytes and notices without staging or changing selection."""
        return read_buddy_archive(path)

    def import_archive(
        self, path: Path | str, *, name: str | None = None
    ) -> BuddyRecord:
        return self.publish_review(self.review_archive(path), name=name)

    def _source_record(self, source_key: str) -> BuddyRecord | None:
        row = self.db.execute_query(
            "SELECT id,name,version,source_key,status FROM buddy_profiles WHERE source_key=?",
            (source_key,),
        ).fetchone()
        if row is None:
            return None
        if row[4] != "active":
            raise ValueError("buddy_source_retired")
        return self.get_buddy(row[0])

    def publish_review(
        self,
        review: BuddySnapshot,
        *,
        name: str | None = None,
        source_key: str | None = None,
    ) -> BuddyRecord:
        """Publish one reviewed immutable copy; only a complete binding is listed."""
        if type(review) is not BuddySnapshot or not review.is_current():
            raise ValueError("buddy_source_changed")
        title = review.title if name is None else name
        if type(title) is not str or not title.strip() or len(title) > 256:
            raise ValueError("buddy_name_invalid")
        if source_key is not None and (
            type(source_key) is not str or not 1 <= len(source_key) <= 256
        ):
            raise ValueError("buddy_source_key_invalid")
        with _PUBLICATION_LOCK:
            if source_key is not None:
                existing = self._source_record(source_key)
                if existing is not None:
                    return existing
            buddy_id = str(uuid4())
            # A prior interrupted attempt may have only its unlisted owner row.
            with self.db.transaction(immediate=True):
                pending = (
                    None
                    if source_key is None
                    else self.db.execute_query(
                        "SELECT id FROM buddy_profiles WHERE source_key=?",
                        (source_key,),
                    ).fetchone()
                )
                if pending is not None:
                    buddy_id = pending[0]
                else:
                    self.db.execute_query(
                        "INSERT INTO buddy_profiles(id,name,source_key) VALUES (?,?,?)",
                        (buddy_id, title, source_key),
                    )
            try:
                with TemporaryDirectory(prefix="buddy-publication-") as folder:
                    source = Path(folder).resolve()
                    sources = []
                    for index, asset in enumerate(review.assets):
                        suffix = {
                            "image/png": ".png",
                            "image/jpeg": ".jpg",
                            "image/gif": ".gif",
                            "image/webp": ".webp",
                        }[asset.metadata.mime_type]
                        key = f"{index:03d}{suffix}"
                        (source / key).write_bytes(asset.data)
                        sources.append(
                            PersonaVisualPublicationAssetSource(key, asset.metadata)
                        )
                    context = dict(review.source_context)
                    context["artwork"] = encode_native_artwork(dict(review.artwork))
                    snapshot = PersonaVisualPublicationSnapshot(
                        persona_id=None,
                        persona_revision=0,
                        buddy_id=buddy_id,
                        buddy_revision=1,
                        title=title,
                        description=review.description,
                        source_kind=review.source_kind,
                        source_context=tuple(sorted(context.items())),
                        manifest_json=review.manifest_json,
                        assets=tuple(sources),
                    )
                    publish_persona_visual(
                        self.repository,
                        snapshot,
                        source_root=source,
                        profile_root=self.profile_root,
                        authority_guard=review.is_current,
                    )
            except Exception as exc:
                # Never remove an owner/files after a possibly committed write.
                if self.get_graph(buddy_id) is None:
                    with self.db.transaction(immediate=True):
                        self.db.execute_query(
                            "DELETE FROM buddy_profiles WHERE id=?", (buddy_id,)
                        )
                    candidate = getattr(exc, "cleanup_candidate", None)
                    if candidate:
                        try:
                            cleanup_persona_visual_publication_candidate(
                                self.repository,
                                profile_root=self.profile_root,
                                cleanup_candidate=candidate,
                            )
                        except Exception:  # noqa: BLE001 - stable path-free cleanup boundary
                            logger.warning("buddy_publication_cleanup_deferred")
                raise
            result = self.get_buddy(buddy_id)
            if result is None:
                raise ValueError("buddy_publication_failed")
            return result

    def copy_persona(
        self, persona_id: str, *, name: str | None = None, source_key: str | None = None
    ) -> BuddyRecord:
        """Copy a current local Persona graph; its subsequent lifecycle is irrelevant."""
        if source_key is not None:
            existing = self._source_record(source_key)
            if existing is not None:
                return existing
        if self.persona_reader is None:
            raise ValueError("buddy_persona_authority_unavailable")
        record = dict(self.persona_reader(persona_id))
        graph = self.repository.get_active_persona_pack_for_export(persona_id)
        if graph is None:
            raise ValueError("buddy_persona_visual_unavailable")

        def current() -> bool:
            fresh = self.persona_reader(persona_id)
            saved = self.repository.get_active_persona_pack(persona_id)
            return bool(
                fresh.get("id") == persona_id
                and fresh.get("version") == record.get("version")
                and type(record.get("version")) is int
                and fresh.get("deleted", False) is False
                and fresh.get("is_active", True) is True
                and saved is not None
                and saved.identity == graph.graph.identity
                and saved.identity.persona_revision == record["version"]
            )

        if not current():
            raise ValueError("buddy_persona_authority_changed")
        assets = []
        for item in graph.assets:
            row = item.record
            metadata = PersonaVisualAssetMetadata(
                asset_key=row.asset_key,
                role=row.role,
                mime_type=row.mime_type,
                byte_count=row.byte_count,
                sha256=row.sha256,
                width=row.width,
                height=row.height,
                frame_count=row.frame_count,
                duration_ms=row.duration_ms,
            )
            loaded = load_persona_visual_asset(
                self.profile_root, storage_key=item.storage_key, metadata=metadata
            )
            assets.append(BuddyAssetSnapshot(metadata, loaded.data))
        context = dict(graph.source_context)
        review = BuddySnapshot(
            graph.graph.pack.title,
            graph.manifest_bytes.decode("utf-8"),
            tuple(assets),
            artwork_from_pack({"source_context": context}),
            graph.graph.identity.manifest_sha256,
            current,
            graph.source_context,
            graph.graph.pack.description,
            source_kind=graph.graph.pack.source_kind,
        )
        return self.publish_review(review, name=name, source_key=source_key)

    def ensure_builtin(self, *, legacy_retired: bool = False) -> BuddyRecord | None:
        """Install bundled pixels once without Persona creation or selection changes."""
        key = "builtin:pixel-migu"
        try:
            existing = self._source_record(key)
        except ValueError:
            return None  # A retained user tombstone is terminal.
        if existing is not None:
            return existing
        if legacy_retired:
            return None
        source = files("tldw_chatbook").joinpath(
            "assets", "persona_visual", "pixel_migu"
        )
        manifest = json.dumps(
            json.loads(source.joinpath("manifest.json").read_bytes()),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assets = []
        for declaration in json.loads(source.joinpath("assets.json").read_bytes()):
            fields = dict(declaration)
            filename = fields.pop("filename")
            assets.append(
                BuddyAssetSnapshot(
                    PersonaVisualAssetMetadata(**fields),
                    source.joinpath(filename).read_bytes(),
                )
            )
        review = BuddySnapshot(
            "pixel-migu",
            manifest,
            tuple(assets),
            {
                "version": 1,
                "creator": None,
                "license": "LicenseRef-User-Supplied",
                "source_url": None,
                "notices": "",
            },
            hashlib.sha256(manifest.encode()).hexdigest(),
            lambda: True,
            (
                ("provenance", "bundled-pixel-migu"),
                ("license", "LicenseRef-User-Supplied"),
            ),
            source_kind="manual",
        )
        return self.publish_review(review, source_key=key)

    def migrate_legacy_selection(
        self,
        preferences: PersonaBuddyPreferences,
        *,
        writer: Callable[[PersonaBuddyPreferences], bool],
    ) -> PersonaBuddyPreferences:
        """Copy once, then persist; a failed copy/write leaves legacy choices intact."""
        from .preferences import BuddySelection, PersonaBuddySelection

        selection = preferences.selection
        if type(selection) is not PersonaBuddySelection:
            return preferences
        key = (
            "legacy:" + hashlib.sha256(selection.local_persona_id.encode()).hexdigest()
        )
        try:
            buddy = self.copy_persona(selection.local_persona_id, source_key=key)
            candidate = replace(preferences, selection=BuddySelection(buddy.id))
            return candidate if writer(candidate) is True else preferences
        except Exception:  # noqa: BLE001 - failed migration preserves the usable legacy selection
            return preferences
