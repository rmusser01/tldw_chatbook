"""Structured Visual Identity resolver contracts."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat import visual_identity
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "resolver.db", client_id="resolver-test")
    yield database
    database.close_connection()


@pytest.fixture
def user_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "profile"


def _webp_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="WEBP", lossless=True)
    return output.getvalue()


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _animated_gif_bytes() -> bytes:
    output = BytesIO()
    frames = [
        Image.new("RGB", (8, 8), color) for color in ((20, 30, 40), (80, 90, 100))
    ]
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    return output.getvalue()


def _bmp_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(output, format="BMP")
    return output.getvalue()


def _add_character(
    db: CharactersRAGDB, *, image: bytes | None = None, name: str = "Resolver"
) -> int:
    character_id = db.add_character_card({"name": name, "image": image})
    assert character_id is not None
    return int(character_id)


def _asset(
    user_data_dir: Path,
    expression_key: str,
    color: tuple[int, int, int],
) -> dict[str, Any]:
    data = _webp_bytes(color)
    filename = expression_key.replace("custom:", "") + ".webp"
    relative_path = f"characters/resolver/{filename}"
    path = user_data_dir / "visual_identities" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "expression_key": expression_key,
        "original_expression_key": expression_key,
        "display_label": expression_key.title(),
        "source_filename": filename,
        "storage_relpath": relative_path,
        "content_type": "image/webp",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": 8,
        "height": 8,
        "source_context": {"fixture": True},
        "is_animated": False,
        "frame_count": 1,
    }


def _activate(
    db: CharactersRAGDB,
    user_data_dir: Path,
    character_id: int,
    keys: tuple[str, ...],
    *,
    default: str = "neutral",
    source_kind: str = "manual",
) -> dict[str, Any]:
    colors = [(index * 31 % 255, 80, 160) for index in range(1, len(keys) + 1)]
    return VisualIdentityRepository(db).activate_pack(
        pack={
            "title": "Resolver fixture",
            "default_expression_key": default,
            "source_kind": source_kind,
            "source_context": {"source_id": "resolver.fixture"},
        },
        manifest={"schema_id": "resolver/v1"},
        assets=[
            _asset(user_data_dir, key, color)
            for key, color in zip(keys, colors, strict=True)
        ],
        actor_kind="character",
        actor_id=character_id,
    )


def _resolve(
    db: CharactersRAGDB,
    user_data_dir: Path,
    character_id: int,
    requested_state: str,
    manual_expression_key: str | None = None,
) -> Any:
    return visual_identity.resolve_visual_identity(
        db,
        actor_kind="character",
        actor_id=character_id,
        requested_state=requested_state,
        manual_expression_key=manual_expression_key,
        user_data_dir=user_data_dir,
    )


def test_resolution_result_is_frozen_slotted_and_manual_wins(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db, image=_png_bytes((10, 20, 30)))
    activated = _activate(
        db,
        user_data_dir,
        character_id,
        ("neutral", "thinking", "custom:admiration"),
    )

    result = _resolve(db, user_data_dir, character_id, "thinking", "admiration")

    assert result.actor_kind == "character"
    assert result.actor_id == str(character_id)
    assert result.requested_expression_key == "thinking"
    assert result.manual_expression_key == "custom:admiration"
    assert result.resolved_expression_key == "custom:admiration"
    assert result.pack_id == activated["pack"]["id"]
    assert result.pack_version_id == activated["version"]["id"]
    assert result.asset_id is not None
    assert result.expression_id is None
    assert result.storage_source == "manual"
    assert result.storage_relpath.endswith("admiration.webp")
    assert result.content_type == "image/webp"
    assert result.is_animated is False
    assert result.resolution_source == "pack_manual"
    assert result.fallback_reason == "none"
    assert result.image_bytes
    assert result.cache_identity[-1] == (
        f"sha256={hashlib.sha256(result.image_bytes).hexdigest()}"
    )
    with pytest.raises((AttributeError, TypeError)):
        result.resolved_expression_key = "neutral"
    assert not hasattr(result, "__dict__")


@pytest.mark.parametrize(
    ("state", "expected_key"),
    [
        ("idle", "neutral"),
        ("thinking", "thinking"),
        ("speaking", "custom:speaking"),
        ("error", "custom:error"),
    ],
)
def test_operational_states_map_to_pack_keys(
    db: CharactersRAGDB,
    user_data_dir: Path,
    state: str,
    expected_key: str,
) -> None:
    character_id = _add_character(db, name=f"Resolver {state}")
    _activate(
        db,
        user_data_dir,
        character_id,
        ("neutral", "thinking", "custom:speaking", "custom:error"),
    )

    result = _resolve(db, user_data_dir, character_id, state)

    assert result.resolved_expression_key == expected_key
    assert result.resolution_source == "pack_operational"
    assert result.fallback_reason == "none"


@pytest.mark.parametrize(
    ("state", "expected_key"),
    [("happy", "happy"), ("custom:smug", "custom:smug")],
)
def test_live_explicit_states_map_to_pack_keys(
    db: CharactersRAGDB,
    user_data_dir: Path,
    state: str,
    expected_key: str,
) -> None:
    character_id = _add_character(db)
    _activate(db, user_data_dir, character_id, ("neutral", expected_key))

    result = _resolve(db, user_data_dir, character_id, state)

    assert result.requested_expression_key == expected_key
    assert result.resolved_expression_key == expected_key
    assert result.resolution_source == "pack_explicit"
    assert result.fallback_reason == "none"


def test_history_resolves_exact_recorded_asset_after_active_pack_changes(
    db: CharactersRAGDB, user_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    character_id = _add_character(db)
    old = _activate(db, user_data_dir, character_id, ("neutral", "happy"))
    old_asset = next(
        asset for asset in old["assets"] if asset["expression_key"] == "happy"
    )
    old_bytes = (
        user_data_dir / "visual_identities" / old_asset["storage_relpath"]
    ).read_bytes()
    _activate(db, user_data_dir, character_id, ("neutral", "sad"))
    transaction_calls = 0
    original_transaction = db.transaction

    @contextmanager
    def tracked_transaction():
        nonlocal transaction_calls
        transaction_calls += 1
        with original_transaction() as connection:
            yield connection

    monkeypatch.setattr(db, "transaction", tracked_transaction)

    result = visual_identity.resolve_historical_visual_identity(
        db,
        actor_id=character_id,
        pack_id=old["pack"]["id"],
        pack_version_id=old["version"]["id"],
        expression_key="happy",
        expression_id=None,
        asset_id=old_asset["id"],
        user_data_dir=user_data_dir,
    )

    assert result.resolution_source == "history_immutable"
    assert result.fallback_reason == "none"
    assert result.pack_id == old["pack"]["id"]
    assert result.pack_version_id == old["version"]["id"]
    assert result.asset_id == old_asset["id"]
    assert result.resolved_expression_key == "happy"
    assert result.image_bytes == old_bytes
    assert transaction_calls == 1


def test_history_fails_closed_when_recorded_identity_is_inconsistent(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db)
    activated = _activate(db, user_data_dir, character_id, ("neutral", "happy"))
    asset = next(
        item for item in activated["assets"] if item["expression_key"] == "happy"
    )

    result = visual_identity.resolve_historical_visual_identity(
        db,
        actor_id=character_id,
        pack_id=activated["pack"]["id"],
        pack_version_id=activated["version"]["id"],
        expression_key="sad",
        expression_id=None,
        asset_id=asset["id"],
        user_data_dir=user_data_dir,
    )

    assert result.resolution_source == "placeholder"
    assert result.fallback_reason == "history_unavailable"
    assert result.image_bytes is None

    invalid_expression_id = visual_identity.resolve_historical_visual_identity(
        db,
        actor_id=character_id,
        pack_id=activated["pack"]["id"],
        pack_version_id=activated["version"]["id"],
        expression_key="happy",
        expression_id="server-expression-id",
        asset_id=asset["id"],
        user_data_dir=user_data_dir,
    )

    assert invalid_expression_id.resolution_source == "placeholder"
    assert invalid_expression_id.fallback_reason == "history_unavailable"


def test_unknown_manual_falls_through_to_requested_operational(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db)
    _activate(db, user_data_dir, character_id, ("neutral", "thinking"))

    result = _resolve(db, user_data_dir, character_id, "thinking", "not-in-pack")

    assert result.manual_expression_key == "custom:not_in_pack"
    assert result.resolved_expression_key == "thinking"
    assert result.resolution_source == "pack_operational"
    assert result.fallback_reason == "manual_unavailable"


def test_missing_requested_uses_version_default_before_neutral(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db)
    _activate(
        db,
        user_data_dir,
        character_id,
        ("neutral", "happy"),
        default="happy",
    )

    result = _resolve(db, user_data_dir, character_id, "error")

    assert result.resolved_expression_key == "happy"
    assert result.resolution_source == "pack_default"
    assert result.fallback_reason == "requested_unavailable"


def test_missing_requested_and_default_use_neutral(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db)
    activated = _activate(db, user_data_dir, character_id, ("neutral",))
    db.execute_query(
        "UPDATE visual_identity_pack_versions SET default_expression_key = ? WHERE id = ?",
        ("happy", activated["version"]["id"]),
    )

    result = _resolve(db, user_data_dir, character_id, "error")

    assert result.resolved_expression_key == "neutral"
    assert result.resolution_source == "pack_neutral"
    assert result.fallback_reason == "default_unavailable"


def test_corrupt_and_missing_pack_assets_fall_through_to_matching_legacy_blob(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db, image=_png_bytes((10, 20, 30)))
    activated = _activate(
        db,
        user_data_dir,
        character_id,
        ("thinking", "neutral"),
    )
    assets = activated["assets"]
    for asset in assets:
        path = user_data_dir / "visual_identities" / asset["storage_relpath"]
        if asset["expression_key"] == "thinking":
            path.write_bytes(b"corrupt")
        else:
            path.unlink()
    legacy = _png_bytes((200, 20, 20))
    db.set_character_expression_image(character_id, "thinking", legacy, "image/png")

    result = _resolve(db, user_data_dir, character_id, "thinking")

    assert result.image_bytes == legacy
    assert result.resolution_source == "legacy_expression"
    assert result.fallback_reason == "pack_assets_unavailable"
    assert result.expression_id is not None
    assert result.storage_relpath is None


def test_corrupt_pack_metadata_falls_through_without_exposing_storage_path(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db)
    activated = _activate(db, user_data_dir, character_id, ("thinking",))
    db.execute_query(
        "UPDATE visual_identity_assets SET frame_count = 'invalid' WHERE id = ?",
        (activated["assets"][0]["id"],),
    )
    legacy = _png_bytes((8, 9, 10))
    db.set_character_expression_image(character_id, "thinking", legacy, "image/png")

    result = _resolve(db, user_data_dir, character_id, "thinking")

    assert result.resolution_source == "legacy_expression"
    assert result.image_bytes == legacy


@pytest.mark.parametrize(
    "inactive", ["binding", "pack_archived", "pack_deleted", "version", "asset"]
)
def test_inactive_or_mismatched_pack_skips_package_reads_and_uses_legacy(
    db: CharactersRAGDB,
    user_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    inactive: str,
) -> None:
    character_id = _add_character(db)
    activated = _activate(db, user_data_dir, character_id, ("thinking", "neutral"))
    legacy = _png_bytes((30, 40, 50))
    db.set_character_expression_image(character_id, "thinking", legacy, "image/png")
    if inactive == "binding":
        db.execute_query(
            "UPDATE visual_identity_bindings SET status = 'deleted' WHERE id = ?",
            (activated["binding"]["id"],),
        )
    elif inactive in {"pack_archived", "pack_deleted"}:
        db.execute_query(
            "UPDATE visual_identity_packs SET status = ? WHERE id = ?",
            (
                "archived" if inactive == "pack_archived" else "deleted",
                activated["pack"]["id"],
            ),
        )
    elif inactive == "version":
        mismatched_version_id = int(
            db.execute_query(
                """INSERT INTO visual_identity_pack_versions(
                       pack_id, owner_user_id, version_number,
                       default_expression_key, manifest_json
                   ) VALUES (?, 0, 2, 'neutral', '{}')""",
                (activated["pack"]["id"],),
            ).lastrowid
        )
        db.execute_query(
            "UPDATE visual_identity_bindings SET active_version_id = ? WHERE id = ?",
            (mismatched_version_id, activated["binding"]["id"]),
        )
    else:
        db.execute_query(
            "UPDATE visual_identity_assets SET deleted = 1 WHERE pack_version_id = ?",
            (activated["version"]["id"],),
        )
    reads = 0

    def forbidden_read(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal reads
        reads += 1
        raise AssertionError("inactive graph must not read an asset")

    monkeypatch.setattr(visual_identity, "load_visual_identity_asset", forbidden_read)

    result = _resolve(db, user_data_dir, character_id, "thinking")

    assert result.resolution_source == "legacy_expression"
    assert reads == 0


def test_soft_deleted_actor_is_placeholder_even_when_bound(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db, image=_png_bytes((1, 2, 3)))
    _activate(db, user_data_dir, character_id, ("neutral",))
    db.execute_query(
        "UPDATE character_cards SET deleted = 1 WHERE id = ?", (character_id,)
    )

    result = _resolve(db, user_data_dir, character_id, "idle")

    assert result.resolution_source == "placeholder"
    assert result.fallback_reason == "actor_unavailable"
    assert result.image_bytes is None


def test_legacy_only_character_uses_matching_state_then_card_portrait(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    portrait = _png_bytes((1, 30, 90))
    character_id = _add_character(db, image=portrait)
    legacy = _png_bytes((90, 30, 1))
    db.set_character_expression_image(character_id, "speaking", legacy, "image/png")

    speaking = _resolve(db, user_data_dir, character_id, "speaking")
    idle = _resolve(db, user_data_dir, character_id, "idle")

    assert speaking.resolution_source == "legacy_expression"
    assert speaking.resolved_expression_key == "custom:speaking"
    assert speaking.image_bytes == legacy
    assert idle.resolution_source == "card_portrait"
    assert idle.resolved_expression_key == "neutral"
    assert idle.image_bytes == portrait
    assert speaking.cache_identity != idle.cache_identity


def test_character_without_any_image_and_missing_character_use_placeholder(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db)

    no_image = _resolve(db, user_data_dir, character_id, "error")
    absent = _resolve(db, user_data_dir, character_id + 999, "error")

    assert no_image.resolution_source == "placeholder"
    assert no_image.fallback_reason == "portrait_unavailable"
    assert absent.resolution_source == "placeholder"
    assert absent.fallback_reason == "actor_unavailable"
    assert no_image.cache_identity != absent.cache_identity


def test_unknown_requested_state_uses_pack_default_without_legacy_lookup(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    character_id = _add_character(db)
    _activate(db, user_data_dir, character_id, ("neutral",))
    db.set_character_expression_image(
        character_id, "error", _png_bytes((20, 20, 20)), "image/png"
    )

    result = _resolve(db, user_data_dir, character_id, "unknown-state")

    assert result.requested_expression_key is None
    assert result.resolution_source == "pack_default"


def test_active_pack_query_is_bounded_and_happy_path_reads_one_selected_asset(
    db: CharactersRAGDB,
    user_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character_id = _add_character(db)
    _activate(
        db,
        user_data_dir,
        character_id,
        ("neutral", "thinking", "happy", "sad", "custom:error"),
    )
    queries: list[str] = []
    original_query = db.execute_query
    original_load = visual_identity.load_visual_identity_asset
    reads = 0

    def recording_query(query: str, params: tuple[Any, ...] | None = None) -> Any:
        queries.append(query)
        return original_query(query, params)

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal reads
        reads += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(db, "execute_query", recording_query)
    monkeypatch.setattr(visual_identity, "load_visual_identity_asset", recording_load)

    result = _resolve(db, user_data_dir, character_id, "thinking")

    resolver_queries = [
        query for query in queries if "visual_identity_resolver" in query
    ]
    assert len(resolver_queries) == 1
    assert "LIMIT 4" in resolver_queries[0]
    assert "c.image" not in resolver_queries[0]
    assert "legacy.image" not in resolver_queries[0]
    assert len(queries) == 1
    assert reads == 1
    assert result.resolved_expression_key == "thinking"


def test_duplicate_manual_rows_do_not_hide_valid_operational_candidate(
    db: CharactersRAGDB,
    user_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character_id = _add_character(db)
    activated = _activate(
        db,
        user_data_dir,
        character_id,
        (
            "custom:admiration",
            "custom:admiration",
            "custom:admiration",
            "custom:admiration",
            "thinking",
            "neutral",
        ),
    )
    manual_path = next(
        user_data_dir / "visual_identities" / asset["storage_relpath"]
        for asset in activated["assets"]
        if asset["expression_key"] == "custom:admiration"
    )
    manual_path.write_bytes(b"corrupt-manual")
    queries: list[tuple[str, tuple[Any, ...] | None]] = []
    original_query = db.execute_query

    def recording_query(query: str, params: tuple[Any, ...] | None = None) -> Any:
        queries.append((query, params))
        return original_query(query, params)

    monkeypatch.setattr(db, "execute_query", recording_query)

    result = _resolve(db, user_data_dir, character_id, "thinking", "admiration")

    resolver_query, params = next(
        item for item in queries if "visual_identity_resolver" in item[0]
    )
    assert result.resolved_expression_key == "thinking"
    assert result.resolution_source == "pack_operational"
    assert result.fallback_reason == "manual_unavailable"
    assert len(queries) == 1
    plan = (
        db.get_connection()
        .execute(f"EXPLAIN QUERY PLAN {resolver_query}", params or ())
        .fetchall()
    )
    plan_text = " ".join(str(row[3]) for row in plan)
    assert "CORRELATED SCALAR SUBQUERY" in plan_text
    assert "idx_visual_identity_assets_pack_expression" in plan_text


def test_duplicate_rows_have_a_bounded_sqlite_instruction_budget(
    db: CharactersRAGDB,
    user_data_dir: Path,
) -> None:
    character_id = _add_character(db)
    activated = _activate(
        db,
        user_data_dir,
        character_id,
        ("custom:admiration", "thinking", "neutral"),
    )
    manual_asset_id = next(
        int(asset["id"])
        for asset in activated["assets"]
        if asset["expression_key"] == "custom:admiration"
    )
    connection = db.get_connection()

    def measure() -> int:
        callbacks = 0

        def count_instructions() -> int:
            nonlocal callbacks
            callbacks += 1
            return int(callbacks > 100)

        connection.set_progress_handler(count_instructions, 100)
        try:
            result = _resolve(db, user_data_dir, character_id, "thinking", "admiration")
        finally:
            connection.set_progress_handler(None, 0)
        assert result.resolution_source == "pack_manual"
        return callbacks

    duplicate_sql = """
        INSERT INTO visual_identity_assets(
            owner_user_id, pack_id, pack_version_id, expression_key,
            original_expression_key, display_label, source_filename,
            storage_relpath, content_type, bytes, sha256, width, height,
            source_context_json, is_animated, frame_count, duration_ms,
            preview_relpath, deleted
        )
        SELECT owner_user_id, pack_id, pack_version_id, expression_key,
               original_expression_key, display_label, source_filename,
               storage_relpath, content_type, bytes, sha256, width, height,
               source_context_json, is_animated, frame_count, duration_ms,
               preview_relpath, deleted
          FROM visual_identity_assets WHERE id = ?
    """
    samples = [measure()]
    with db.transaction() as transaction:
        transaction.executemany(duplicate_sql, [(manual_asset_id,)] * 1_000)
    samples.append(measure())
    with db.transaction() as transaction:
        transaction.executemany(duplicate_sql, [(manual_asset_id,)] * 9_000)
    samples.append(measure())

    assert samples[2] <= 100
    assert max(samples) - min(samples) <= 10


@pytest.mark.parametrize(
    "invalid_legacy",
    [7, "text-image", b"", b"corrupt-image", pytest.param(_bmp_bytes(), id="bmp")],
)
def test_invalid_legacy_values_fall_through_to_valid_card(
    db: CharactersRAGDB,
    user_data_dir: Path,
    invalid_legacy: object,
) -> None:
    portrait = _png_bytes((11, 12, 13))
    character_id = _add_character(db, image=portrait)
    db.execute_query(
        """INSERT INTO character_expression_images(character_id, state_id, image, mime)
           VALUES (?, 'thinking', ?, 'text/html')""",
        (character_id, invalid_legacy),
    )

    result = _resolve(db, user_data_dir, character_id, "thinking")

    assert result.resolution_source == "card_portrait"
    assert result.content_type == "image/png"
    assert result.is_animated is False
    assert result.image_bytes == portrait


def test_oversized_legacy_blob_is_not_materialized_and_falls_through(
    db: CharactersRAGDB,
    user_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    portrait = _png_bytes((14, 15, 16))
    character_id = _add_character(db, image=portrait)
    db.execute_query(
        """INSERT INTO character_expression_images(character_id, state_id, image, mime)
           VALUES (?, 'thinking', zeroblob(?), 'image/png')""",
        (character_id, visual_identity.MAX_EXPRESSION_ASSET_BYTES + 1),
    )
    queries: list[tuple[str, tuple[Any, ...] | None]] = []
    original_query = db.execute_query

    def recording_query(query: str, params: tuple[Any, ...] | None = None) -> Any:
        queries.append((query, params))
        return original_query(query, params)

    monkeypatch.setattr(db, "execute_query", recording_query)

    result = _resolve(db, user_data_dir, character_id, "thinking")

    assert result.resolution_source == "card_portrait"
    assert result.image_bytes == portrait
    legacy_query, params = next(
        item for item in queries if "character_expression_images" in item[0]
    )
    assert "CASE WHEN typeof(image) = 'blob'" in legacy_query
    assert (
        params is not None and params[0] == visual_identity.MAX_EXPRESSION_ASSET_BYTES
    )


@pytest.mark.parametrize(
    "invalid_card",
    [9, "text-image", b"", b"corrupt-image", pytest.param(_bmp_bytes(), id="bmp")],
)
def test_invalid_card_values_fall_through_to_placeholder(
    db: CharactersRAGDB,
    user_data_dir: Path,
    invalid_card: object,
) -> None:
    character_id = _add_character(db)
    db.execute_query(
        "UPDATE character_cards SET image = ? WHERE id = ?",
        (invalid_card, character_id),
    )

    result = _resolve(db, user_data_dir, character_id, "idle")

    assert result.resolution_source == "placeholder"
    assert result.fallback_reason == "portrait_unavailable"
    assert result.image_bytes is None


def test_oversized_card_blob_is_not_materialized_and_uses_placeholder(
    db: CharactersRAGDB,
    user_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character_id = _add_character(db)
    db.execute_query(
        "UPDATE character_cards SET image = zeroblob(?) WHERE id = ?",
        (visual_identity.MAX_EXPRESSION_ASSET_BYTES + 1, character_id),
    )
    queries: list[tuple[str, tuple[Any, ...] | None]] = []
    original_query = db.execute_query

    def recording_query(query: str, params: tuple[Any, ...] | None = None) -> Any:
        queries.append((query, params))
        return original_query(query, params)

    monkeypatch.setattr(db, "execute_query", recording_query)

    result = _resolve(db, user_data_dir, character_id, "idle")

    assert result.resolution_source == "placeholder"
    assert result.image_bytes is None
    card_query, params = next(
        item for item in queries if "FROM character_cards WHERE" in item[0]
    )
    assert "CASE WHEN typeof(image) = 'blob'" in card_query
    assert (
        params is not None and params[0] == visual_identity.MAX_EXPRESSION_ASSET_BYTES
    )


def test_fallback_mime_and_animation_are_derived_from_decoded_bytes(
    db: CharactersRAGDB,
    user_data_dir: Path,
) -> None:
    static_id = _add_character(db, name="Static", image=_png_bytes((1, 2, 3)))
    static_png = _png_bytes((20, 21, 22))
    db.set_character_expression_image(static_id, "thinking", static_png, "text/html")
    animated_id = _add_character(db, name="Animated")
    animated_gif = _animated_gif_bytes()
    db.set_character_expression_image(
        animated_id, "thinking", animated_gif, "image/png"
    )

    static = _resolve(db, user_data_dir, static_id, "thinking")
    animated = _resolve(db, user_data_dir, animated_id, "thinking")

    assert (static.content_type, static.is_animated) == ("image/png", False)
    assert (animated.content_type, animated.is_animated) == ("image/gif", True)
    assert "content_type=image/png" in static.cache_identity
    assert "is_animated=0" in static.cache_identity
    assert "content_type=image/gif" in animated.cache_identity
    assert "is_animated=1" in animated.cache_identity
    assert (
        static.cache_identity[-1] == f"sha256={hashlib.sha256(static_png).hexdigest()}"
    )
    assert animated.cache_identity[-1] == (
        f"sha256={hashlib.sha256(animated_gif).hexdigest()}"
    )


def test_builtin_pack_reads_only_the_selected_package_asset(
    db: CharactersRAGDB,
    user_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character_id = _add_character(db)
    activated = _activate(
        db,
        user_data_dir,
        character_id,
        ("neutral", "thinking", "custom:error"),
        source_kind="builtin",
    )
    thinking = next(
        asset for asset in activated["assets"] if asset["expression_key"] == "thinking"
    )
    expected = (
        user_data_dir / "visual_identities" / thinking["storage_relpath"]
    ).read_bytes()
    reads: list[str] = []

    def package_read(
        asset: visual_identity.VisualIdentityManifestAsset,
        *,
        source_kind: str,
        user_data_dir: Path,
    ) -> visual_identity.LoadedVisualIdentityAsset:
        del user_data_dir
        reads.append(source_kind)
        return visual_identity.LoadedVisualIdentityAsset(asset=asset, data=expected)

    monkeypatch.setattr(visual_identity, "load_visual_identity_asset", package_read)

    result = _resolve(db, user_data_dir, character_id, "thinking")

    assert reads == ["builtin"]
    assert result.storage_source == "builtin"
    assert result.image_bytes == expected


def test_cache_identity_changes_with_manual_actor_version_asset_and_digest(
    db: CharactersRAGDB, user_data_dir: Path
) -> None:
    first_id = _add_character(db, name="First")
    second_id = _add_character(db, name="Second")
    first = _activate(
        db,
        user_data_dir,
        first_id,
        ("neutral", "thinking", "custom:admiration"),
    )
    _activate(db, user_data_dir, second_id, ("neutral", "thinking"))

    operational = _resolve(db, user_data_dir, first_id, "thinking")
    manual = _resolve(db, user_data_dir, first_id, "thinking", "admiration")
    other_actor = _resolve(db, user_data_dir, second_id, "thinking")
    thinking_asset = next(
        asset for asset in first["assets"] if asset["expression_key"] == "thinking"
    )

    assert (
        len(
            {
                operational.cache_identity,
                manual.cache_identity,
                other_actor.cache_identity,
            }
        )
        == 3
    )
    assert f"pack_version_id={first['version']['id']}" in operational.cache_identity
    assert f"asset_id={thinking_asset['id']}" in operational.cache_identity
    assert f"sha256={thinking_asset['sha256']}" in operational.cache_identity
