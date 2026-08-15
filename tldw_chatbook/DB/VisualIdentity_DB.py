"""Local Visual Identity persistence.

``LOCAL_OWNER_ID`` is a local-only, profile-local sentinel, not a server user ID. Future
sync code must translate it rather than sending it to a server.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


LOCAL_OWNER_ID: Final = 0


class VisualIdentityRepository:
    """Store activated Visual Identity packs in the migrated local schema."""

    def __init__(self, db: CharactersRAGDB) -> None:
        """Initialize the repository without creating or migrating schema.

        Args:
            db: Open ChaChaNotes database whose migrations own the schema.
        """
        self.db = db

    def find_pack_by_source_id(
        self, source_id: str, *, include_deleted: bool = False
    ) -> dict[str, Any] | None:
        """Find a built-in pack by its stable source identifier.

        Args:
            source_id: Value stored as ``source_id`` in ``source_context_json``.
            include_deleted: Whether deleted pack tombstones may match.

        Returns:
            The first matching pack row as a plain dictionary, or ``None``.

        Raises:
            ValueError: If selected built-in metadata or its active version is invalid.
        """
        rows = self.db.execute_query(
            """
            SELECT *
              FROM visual_identity_packs
             WHERE owner_user_id = ?
               AND source_kind = ?
               AND (? = 1 OR status != ?)
             ORDER BY id
            """,
            (LOCAL_OWNER_ID, "builtin", int(include_deleted), "deleted"),
        ).fetchall()
        for row in rows:
            pack = dict(row)
            try:
                context = json.loads(
                    pack["source_context_json"],
                    parse_constant=_reject_nonstandard_json_constant,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("visual_identity_source_context_invalid") from exc
            if not isinstance(context, dict):
                raise ValueError("visual_identity_source_context_invalid")
            if context.get("source_id") == source_id:
                self._validate_pack_active_version(pack)
                return pack
        return None

    def get_active_actor_pack(
        self, actor_kind: str, actor_id: int | str
    ) -> dict[str, Any] | None:
        """Fetch an actor's active binding, pack, version, and live assets.

        Args:
            actor_kind: Server-aligned actor kind (``character`` or ``persona``).
            actor_id: Profile-local actor identifier.

        Returns:
            Nested plain dictionaries for the active graph, or ``None`` when no
            active binding or active pack exists.

        Raises:
            ValueError: If a stored active-version reference crosses pack rows.
        """
        with self.db.transaction():
            binding_row = self.db.execute_query(
                """
                SELECT *
                  FROM visual_identity_bindings
                 WHERE owner_user_id = ?
                   AND actor_kind = ?
                   AND actor_id = ?
                   AND status = ?
                """,
                (LOCAL_OWNER_ID, actor_kind, str(actor_id), "active"),
            ).fetchone()
            if binding_row is None:
                return None
            binding = dict(binding_row)

            pack_row = self.db.execute_query(
                """
                SELECT *
                  FROM visual_identity_packs
                 WHERE id = ? AND owner_user_id = ? AND status = ?
                """,
                (binding["pack_id"], LOCAL_OWNER_ID, "active"),
            ).fetchone()
            if pack_row is None:
                return None
            pack = dict(pack_row)
            self._validate_pack_active_version(pack)
            self._validate_binding_active_version(binding)

            version_row = self.db.execute_query(
                """
                SELECT *
                  FROM visual_identity_pack_versions
                 WHERE id = ? AND owner_user_id = ?
                """,
                (binding["active_version_id"], LOCAL_OWNER_ID),
            ).fetchone()
            if version_row is None:
                raise ValueError("visual_identity_binding_active_version_mismatch")
            version = dict(version_row)
            return {
                "binding": binding,
                "pack": pack,
                "version": version,
                "assets": self.list_version_assets(version["id"]),
            }

    def list_version_assets(self, version_id: int) -> list[dict[str, Any]]:
        """List a version's non-deleted assets in deterministic label order.

        Args:
            version_id: Immutable pack-version primary key.

        Returns:
            Plain asset dictionaries ordered by original expression key, then ID.

        Raises:
            ValueError: If the version is absent or an asset references another pack.
        """
        version = self.db.execute_query(
            """
            SELECT pack_id
              FROM visual_identity_pack_versions
             WHERE id = ? AND owner_user_id = ?
            """,
            (version_id, LOCAL_OWNER_ID),
        ).fetchone()
        if version is None:
            raise ValueError("visual_identity_pack_version_not_found")
        pack_id = int(version["pack_id"])
        assets = [
            dict(row)
            for row in self.db.execute_query(
                """
                SELECT *
                  FROM visual_identity_assets
                 WHERE pack_version_id = ?
                   AND owner_user_id = ?
                   AND deleted = 0
                 ORDER BY original_expression_key, id
                """,
                (version_id, LOCAL_OWNER_ID),
            ).fetchall()
        ]
        if any(
            asset["pack_id"] is not None and int(asset["pack_id"]) != pack_id
            for asset in assets
        ):
            raise ValueError("visual_identity_asset_pack_mismatch")
        return assets

    def activate_pack(
        self,
        *,
        pack: Mapping[str, Any],
        manifest: Mapping[str, Any],
        assets: Sequence[Mapping[str, Any]],
        actor_kind: str,
        actor_id: int | str,
        expected_active_identity: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Atomically create and activate one complete pack graph.

        Args:
            pack: Pack fields; ownership is restricted to ``LOCAL_OWNER_ID``.
            manifest: Immutable version manifest.
            assets: Activated asset fields without ``pack_version_id``.
            actor_kind: Actor kind to bind.
            actor_id: Profile-local actor identifier to bind.
            expected_active_identity: Optional ``(pack_id, version_id)`` that
                must still own the actor binding before a copy-on-write fork.

        Returns:
            Nested plain dictionaries for the activated graph.

        Raises:
            ValueError: If input ownership or same-pack relationships are invalid.
        """
        self._validate_actor(actor_kind, actor_id)
        self._validate_local_owner(pack)
        if pack.get("active_version_id") is not None:
            raise ValueError("visual_identity_pack_active_version_mismatch")

        with self.db.transaction():
            if expected_active_identity is not None:
                existing = self._validate_existing_actor_binding(
                    actor_kind, str(actor_id)
                )
                if (
                    existing is None
                    or (int(existing["pack_id"]), int(existing["active_version_id"]))
                    != expected_active_identity
                ):
                    raise ValueError("visual_identity_binding_changed")
            pack_id = int(
                self.db.execute_query(
                    """
                    INSERT INTO visual_identity_packs(
                        owner_user_id, title, description, status,
                        active_version_id, default_expression_key, source_kind,
                        source_context_json
                    ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        LOCAL_OWNER_ID,
                        pack["title"],
                        pack.get("description", ""),
                        "active",
                        pack.get("default_expression_key", "neutral"),
                        pack.get("source_kind", "manual"),
                        _json_dump(pack.get("source_context", {})),
                    ),
                ).lastrowid
            )
            self._validate_asset_candidates(assets, pack_id)
            version_id = self._insert_version(
                pack_id=pack_id,
                version_number=1,
                default_expression_key=str(
                    pack.get("default_expression_key", "neutral")
                ),
                manifest=manifest,
            )
            self._insert_assets(pack_id, version_id, assets)
            self.db.execute_query(
                """
                UPDATE visual_identity_packs
                   SET active_version_id = ?, updated_at = CURRENT_TIMESTAMP
                 WHERE id = ? AND owner_user_id = ?
                """,
                (version_id, pack_id, LOCAL_OWNER_ID),
            )
            self._upsert_active_binding(
                actor_kind=actor_kind,
                actor_id=str(actor_id),
                pack_id=pack_id,
                version_id=version_id,
            )
            active = self.get_active_actor_pack(actor_kind, actor_id)
            if active is None:
                raise RuntimeError("activated_visual_identity_pack_not_found")
            return active

    def publish_version(
        self,
        pack_id: int,
        *,
        manifest: Mapping[str, Any],
        assets: Sequence[Mapping[str, Any]],
        actor_kind: str,
        actor_id: int | str,
        default_expression_key: str | None = None,
        expected_active_version_id: int | None = None,
    ) -> dict[str, Any]:
        """Atomically publish the next immutable version of a user-owned pack.

        Args:
            pack_id: Existing profile-owned pack primary key.
            manifest: Immutable version manifest.
            assets: Activated asset fields without ``pack_version_id``.
            actor_kind: Actor kind whose active binding is updated.
            actor_id: Profile-local actor identifier.
            default_expression_key: Optional replacement default expression.
            expected_active_version_id: Optional version that must still be
                active for both pack and actor before appending.

        Returns:
            Nested plain dictionaries for the newly active graph.

        Raises:
            ValueError: If the pack is absent, built-in, inactive, or corrupted.
        """
        self._validate_actor(actor_kind, actor_id)
        self._validate_asset_candidates(assets, pack_id)

        with self.db.transaction():
            pack_row = self.db.execute_query(
                """
                SELECT *
                  FROM visual_identity_packs
                 WHERE id = ? AND owner_user_id = ?
                """,
                (pack_id, LOCAL_OWNER_ID),
            ).fetchone()
            if pack_row is None:
                raise ValueError("visual_identity_pack_not_found")
            pack = dict(pack_row)
            if pack["status"] != "active":
                raise ValueError("visual_identity_pack_not_active")
            if pack["source_kind"] == "builtin":
                raise ValueError("visual_identity_builtin_pack_immutable")
            self._validate_pack_active_version(pack)
            binding = self._validate_existing_actor_binding(actor_kind, str(actor_id))
            if expected_active_version_id is not None:
                if binding is None or int(binding["pack_id"]) != pack_id:
                    raise ValueError("visual_identity_binding_changed")
                if (
                    int(pack["active_version_id"]) != expected_active_version_id
                    or int(binding["active_version_id"]) != expected_active_version_id
                ):
                    raise ValueError("visual_identity_binding_changed")

            version_number = int(
                self.db.execute_query(
                    """
                    SELECT COALESCE(MAX(version_number), 0) + 1
                      FROM visual_identity_pack_versions
                     WHERE pack_id = ? AND owner_user_id = ?
                    """,
                    (pack_id, LOCAL_OWNER_ID),
                ).fetchone()[0]
            )
            resolved_default = default_expression_key or str(
                pack["default_expression_key"]
            )
            version_id = self._insert_version(
                pack_id=pack_id,
                version_number=version_number,
                default_expression_key=resolved_default,
                manifest=manifest,
            )
            self._insert_assets(pack_id, version_id, assets)
            self.db.execute_query(
                """
                UPDATE visual_identity_packs
                   SET active_version_id = ?,
                       default_expression_key = ?,
                       updated_at = CURRENT_TIMESTAMP,
                       version = version + 1
                 WHERE id = ? AND owner_user_id = ?
                """,
                (version_id, resolved_default, pack_id, LOCAL_OWNER_ID),
            )
            self._upsert_active_binding(
                actor_kind=actor_kind,
                actor_id=str(actor_id),
                pack_id=pack_id,
                version_id=version_id,
            )
            active = self.get_active_actor_pack(actor_kind, actor_id)
            if active is None:
                raise RuntimeError("published_visual_identity_pack_not_found")
            return active

    def archive_pack(self, pack_id: int) -> dict[str, Any]:
        """Soft-archive a pack.

        Args:
            pack_id: Pack primary key.

        Returns:
            Updated pack row.

        Raises:
            ValueError: If no eligible pack matches or its active version is invalid.
        """
        return self._set_pack_status(pack_id, "archived", include_deleted=False)

    def mark_pack_deleted(self, pack_id: int) -> dict[str, Any]:
        """Soft-delete a pack while retaining its rows.

        Args:
            pack_id: Pack primary key.

        Returns:
            Updated pack tombstone.

        Raises:
            ValueError: If no local pack matches or its active version is invalid.
        """
        return self._set_pack_status(pack_id, "deleted", include_deleted=True)

    def mark_binding_deleted(
        self, actor_kind: str, actor_id: int | str
    ) -> dict[str, Any]:
        """Tombstone an actor's active binding without deleting it.

        Args:
            actor_kind: Actor kind of the active binding.
            actor_id: Profile-local actor identifier.

        Returns:
            Updated binding tombstone.

        Raises:
            ValueError: If the active binding is absent or references another pack.
        """
        with self.db.transaction():
            row = self.db.execute_query(
                """
                SELECT *
                  FROM visual_identity_bindings
                 WHERE owner_user_id = ?
                   AND actor_kind = ?
                   AND actor_id = ?
                   AND status = ?
                """,
                (LOCAL_OWNER_ID, actor_kind, str(actor_id), "active"),
            ).fetchone()
            if row is None:
                raise ValueError("visual_identity_binding_not_found")
            binding = dict(row)
            self._validate_binding_active_version(binding)
            binding_id = int(binding["id"])
            self.db.execute_query(
                """
                UPDATE visual_identity_bindings
                   SET status = ?,
                       updated_at = CURRENT_TIMESTAMP,
                       version = version + 1
                 WHERE id = ? AND owner_user_id = ?
                """,
                ("deleted", binding_id, LOCAL_OWNER_ID),
            )
        return self._binding_by_id(binding_id)

    def _insert_version(
        self,
        *,
        pack_id: int,
        version_number: int,
        default_expression_key: str,
        manifest: Mapping[str, Any],
    ) -> int:
        return int(
            self.db.execute_query(
                """
                INSERT INTO visual_identity_pack_versions(
                    pack_id, owner_user_id, version_number,
                    default_expression_key, manifest_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    pack_id,
                    LOCAL_OWNER_ID,
                    version_number,
                    default_expression_key,
                    _json_dump(manifest),
                ),
            ).lastrowid
        )

    def _insert_assets(
        self,
        pack_id: int,
        version_id: int,
        assets: Sequence[Mapping[str, Any]],
    ) -> None:
        for asset in assets:
            self.db.execute_query(
                """
                INSERT INTO visual_identity_assets(
                    owner_user_id, pack_id, pack_version_id, expression_key,
                    original_expression_key, display_label, source_filename,
                    storage_relpath, content_type, bytes, sha256, width, height,
                    source_context_json, is_animated, frame_count, duration_ms,
                    preview_relpath, deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    LOCAL_OWNER_ID,
                    pack_id,
                    version_id,
                    asset["expression_key"],
                    asset.get("original_expression_key", ""),
                    asset.get("display_label", ""),
                    asset["source_filename"],
                    asset["storage_relpath"],
                    asset["content_type"],
                    asset["bytes"],
                    asset["sha256"],
                    asset["width"],
                    asset["height"],
                    _json_dump(asset.get("source_context", {})),
                    int(bool(asset.get("is_animated", False))),
                    asset.get("frame_count"),
                    asset.get("duration_ms"),
                    asset.get("preview_relpath"),
                    int(bool(asset.get("deleted", False))),
                ),
            )

    def _upsert_active_binding(
        self, *, actor_kind: str, actor_id: str, pack_id: int, version_id: int
    ) -> None:
        existing = self._validate_existing_actor_binding(actor_kind, actor_id)
        if existing is None:
            self.db.execute_query(
                """
                INSERT INTO visual_identity_bindings(
                    owner_user_id, actor_kind, actor_id, pack_id, active_version_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (LOCAL_OWNER_ID, actor_kind, actor_id, pack_id, version_id),
            )
            return
        self.db.execute_query(
            """
            UPDATE visual_identity_bindings
               SET pack_id = ?,
                   active_version_id = ?,
                   updated_at = CURRENT_TIMESTAMP,
                   version = version + 1
             WHERE id = ?
            """,
            (pack_id, version_id, existing["id"]),
        )

    def _validate_existing_actor_binding(
        self, actor_kind: str, actor_id: str
    ) -> dict[str, Any] | None:
        row = self.db.execute_query(
            """
            SELECT *
              FROM visual_identity_bindings
             WHERE owner_user_id = ?
               AND actor_kind = ?
               AND actor_id = ?
               AND status = ?
            """,
            (LOCAL_OWNER_ID, actor_kind, actor_id, "active"),
        ).fetchone()
        if row is None:
            return None
        binding = dict(row)
        self._validate_binding_active_version(binding)
        return binding

    def _validate_pack_active_version(self, pack: Mapping[str, Any]) -> None:
        version_id = pack.get("active_version_id")
        if version_id is None:
            raise ValueError("visual_identity_pack_active_version_mismatch")
        version_pack_id = self._version_pack_id(int(version_id))
        if version_pack_id != int(pack["id"]):
            raise ValueError("visual_identity_pack_active_version_mismatch")

    def _validate_binding_active_version(self, binding: Mapping[str, Any]) -> None:
        version_pack_id = self._version_pack_id(int(binding["active_version_id"]))
        if version_pack_id != int(binding["pack_id"]):
            raise ValueError("visual_identity_binding_active_version_mismatch")

    def _version_pack_id(self, version_id: int) -> int | None:
        row = self.db.execute_query(
            """
            SELECT pack_id
              FROM visual_identity_pack_versions
             WHERE id = ? AND owner_user_id = ?
            """,
            (version_id, LOCAL_OWNER_ID),
        ).fetchone()
        return int(row["pack_id"]) if row is not None else None

    @staticmethod
    def _validate_actor(actor_kind: str, actor_id: int | str) -> None:
        if actor_kind not in {"character", "persona"}:
            raise ValueError("visual_identity_actor_kind_invalid")
        if not str(actor_id):
            raise ValueError("visual_identity_actor_id_required")

    @staticmethod
    def _validate_local_owner(pack: Mapping[str, Any]) -> None:
        if int(pack.get("owner_user_id", LOCAL_OWNER_ID)) != LOCAL_OWNER_ID:
            raise ValueError("visual_identity_owner_must_be_local")

    @staticmethod
    def _validate_asset_candidates(
        assets: Sequence[Mapping[str, Any]], pack_id: int
    ) -> None:
        for asset in assets:
            if "pack_version_id" in asset:
                raise ValueError("visual_identity_asset_pack_version_not_allowed")
            candidate_pack_id = asset.get("pack_id")
            if candidate_pack_id is not None and int(candidate_pack_id) != pack_id:
                raise ValueError("visual_identity_asset_pack_mismatch")

    def _set_pack_status(
        self, pack_id: int, status: str, *, include_deleted: bool
    ) -> dict[str, Any]:
        with self.db.transaction():
            row = self.db.execute_query(
                """
                SELECT *
                  FROM visual_identity_packs
                 WHERE id = ?
                   AND owner_user_id = ?
                   AND (? = 1 OR status != ?)
                """,
                (pack_id, LOCAL_OWNER_ID, int(include_deleted), "deleted"),
            ).fetchone()
            if row is None:
                raise ValueError("visual_identity_pack_not_found")
            pack = dict(row)
            self._validate_pack_active_version(pack)
            if pack["status"] == status:
                return pack
            self.db.execute_query(
                """
                UPDATE visual_identity_packs
                   SET status = ?,
                       updated_at = CURRENT_TIMESTAMP,
                       version = version + 1
                 WHERE id = ? AND owner_user_id = ?
                """,
                (status, pack_id, LOCAL_OWNER_ID),
            )
            row = self.db.execute_query(
                """
                SELECT *
                  FROM visual_identity_packs
                 WHERE id = ? AND owner_user_id = ?
                """,
                (pack_id, LOCAL_OWNER_ID),
            ).fetchone()
            if row is None:
                raise ValueError("visual_identity_pack_not_found")
            return dict(row)

    def _binding_by_id(self, binding_id: int) -> dict[str, Any]:
        row = self.db.execute_query(
            """
            SELECT *
              FROM visual_identity_bindings
             WHERE id = ? AND owner_user_id = ?
            """,
            (binding_id, LOCAL_OWNER_ID),
        ).fetchone()
        if row is None:
            raise ValueError("visual_identity_binding_not_found")
        return dict(row)


def _json_dump(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise ValueError("nonstandard JSON constant")
