"""Create-only bundled pixel-migu character content (ADR-122)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

BUILTIN_ID = "pixel-migu"
PACK_ID = "pixel-migu.expressions"


def find_builtin_pixel_migu_character(db: CharactersRAGDB) -> dict[str, Any] | None:
    """Find stable character provenance, including renamed and deleted cards.

    Args:
        db: Initialized profile-local character database.

    Returns:
        The earliest matching card's ID, name and deletion flag, or ``None``.
    """
    row = db.execute_query(
        """SELECT id, name, deleted FROM character_cards
             WHERE CASE WHEN json_valid(extensions)
                        THEN json_extract(extensions, '$."tldw/builtin_id"')
                   END = ?
             ORDER BY id LIMIT 1""",
        (BUILTIN_ID,),
    ).fetchone()
    return dict(row) if row is not None else None


def _already_seeded(db: CharactersRAGDB) -> bool:
    if find_builtin_pixel_migu_character(db) is not None:
        return True
    # Retain a surviving pack's provenance even after a character was removed.
    return (
        db.execute_query(
            """SELECT id FROM visual_identity_packs
             WHERE owner_user_id = 0 AND source_kind = 'builtin'
               AND CASE WHEN json_valid(source_context_json)
                        THEN json_extract(source_context_json, '$.source_id')
                   END = ? LIMIT 1""",
            (PACK_ID,),
        ).fetchone()
        is not None
    )


def ensure_builtin_pixel_migu(db: CharactersRAGDB) -> None:
    """Atomically create the optional character and expression pack once.

    Existing provenance is terminal, including user edits, forks and tombstones.
    All resources are validated before mutation. No active selection is changed.

    Args:
        db: Initialized profile-local character database.

    Raises:
        ValueError: If bundled content is unavailable or invalid.
        Exception: Database failures propagate to the startup caller after rollback.
    """
    if _already_seeded(db):
        return
    card, manifest_data, manifest, assets = _load_bundle()

    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

    with db.transaction(immediate=True):
        # Another process can finish the seed while this one validates resources.
        if _already_seeded(db):
            return
        occupied = {
            row[0] for row in db.execute_query("SELECT name FROM character_cards")
        }
        name = BUILTIN_ID
        suffix = 1
        while name in occupied:
            name = f"{BUILTIN_ID} (Built-in)" + (f" {suffix}" if suffix > 1 else "")
            suffix += 1
        card["name"] = name
        character_id = db.add_character_card(card)
        if character_id is None:
            raise ValueError("pixel_migu_character_insert_failed")
        VisualIdentityRepository(db).activate_pack(
            pack={
                "title": manifest.title,
                "description": "Bundled pixel-migu expression pack.",
                "default_expression_key": manifest.default_expression_key,
                "source_kind": "builtin",
                "source_context": {
                    "source_id": PACK_ID,
                    "pack_content_sha256": manifest.pack_content_sha256,
                },
            },
            manifest=manifest_data,
            assets=assets,
            actor_kind="character",
            actor_id=character_id,
            require_unbound_actor=True,
        )


def _load_bundle() -> tuple[dict[str, Any], dict[str, Any], Any, list[dict[str, Any]]]:
    # These imports and all package I/O stay off the already-seeded startup path.
    from dataclasses import asdict
    from importlib import resources
    from io import BytesIO
    from pathlib import PurePosixPath

    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        extract_json_from_image_file,
        parse_v2_card,
    )
    from tldw_chatbook.Character_Chat.visual_identity import (
        _strict_json_object,
        _validate_image_bytes,
        load_visual_identity_asset,
        validate_visual_identity_manifest,
    )

    root = resources.files("tldw_chatbook").joinpath(
        "assets", "characters", "pixel_migu"
    )

    def read_resource(filename: str) -> bytes:
        with root.joinpath(filename).open("rb") as stream:
            data = stream.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("pixel_migu_resource_too_large")
        return data

    card_json = _strict_json_object(read_resource("pixel-migu.character.json"))
    portrait = read_resource("pixel-migu.character.png")
    embedded = extract_json_from_image_file(BytesIO(portrait))
    if embedded is None or _strict_json_object(embedded) != card_json:
        raise ValueError("pixel_migu_card_mismatch")
    card = parse_v2_card(card_json)
    if (
        card_json.get("spec") != "chara_card_v2"
        or card_json.get("spec_version") != "2.0"
        or card is None
        or card.get("extensions", {}).get("tldw/builtin_id") != BUILTIN_ID
        or card.get("extensions", {}).get("tldw/visual_identity_pack_id") != PACK_ID
    ):
        raise ValueError("pixel_migu_card_invalid")
    card["image"] = portrait
    manifest_data = _strict_json_object(read_resource("visual_identity_pack.json"))
    manifest = validate_visual_identity_manifest(manifest_data)
    if (
        manifest.pack_id != PACK_ID
        or manifest.license != "LicenseRef-User-Supplied"
        or len(manifest.assets) != 18
        or not {"neutral", "thinking", "custom:speaking", "custom:error"}.issubset(
            asset.expression_key for asset in manifest.assets
        )
    ):
        raise ValueError("pixel_migu_pack_invalid")
    assets = []
    decoded_pixels = 0
    for asset in manifest.assets:
        path = PurePosixPath(asset.storage_relpath)
        if path.parts[:3] != ("characters", "pixel_migu", "expressions"):
            raise ValueError("pixel_migu_asset_path_invalid")
        loaded = load_visual_identity_asset(asset, source_kind="builtin")
        decoded_pixels += _validate_image_bytes(
            loaded, decoded_pixels_before=decoded_pixels
        )
        row = asdict(asset)
        row["original_expression_key"] = row.pop("original_label")
        row["source_filename"] = path.name
        assets.append(row)
    return card, manifest_data, manifest, assets
