"""Profile-local region settings and atomic publication of composed portrait sets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from tldw_chatbook.Character_Chat.portrait_compositing import (
    PixelRect,
    PortraitCompositionRecipe,
    recover_components,
    validate_recipe,
    validate_recipe_geometry,
)
from tldw_chatbook.Character_Chat.portrait_performance import (
    FRAME_ROLES,
    MAX_SET_BYTES,
    PerformanceConflict,
    PerformanceFrame,
    PerformanceSource,
    _performance_write,
    _preflight_source,
    _source_stamp,
    _validate_source,
    capture_performance_source,
)

if TYPE_CHECKING:
    from .ChaChaNotes_DB import CharactersRAGDB


@dataclass(frozen=True, slots=True)
class PortraitRegionSettings:
    """One independently revisioned, source-bound selection."""

    source: PerformanceSource
    recipe: PortraitCompositionRecipe
    revision: int


def serialize_recipe(recipe: PortraitCompositionRecipe) -> str:
    """Serialize only the bounded, versioned geometry contract."""
    validate_recipe_geometry(recipe)
    raw = json.dumps(asdict(recipe), sort_keys=True, separators=(",", ":"))
    if len(raw.encode("utf-8")) > 2048:
        raise ValueError("portrait_composition_recipe_bytes_exceeded")
    return raw


def deserialize_recipe(raw: str) -> PortraitCompositionRecipe:
    """Reject malformed, unknown-version and extra-key persisted recipes."""
    if type(raw) is not str or len(raw.encode("utf-8")) > 2048:
        raise ValueError("portrait_composition_recipe_invalid")
    try:
        data = json.loads(raw)
        if type(data) is not dict:
            raise ValueError("portrait_composition_recipe_invalid")
        for key in ("left_eye", "right_eye", "mouth"):
            data[key] = PixelRect(**data[key])
        recipe = PortraitCompositionRecipe(**data)
        validate_recipe_geometry(recipe)
        return recipe
    except (TypeError, KeyError, RecursionError) as exc:
        raise ValueError("portrait_composition_recipe_invalid") from exc


def optional_recipe(raw: str | None) -> PortraitCompositionRecipe | None:
    """Invalid provenance cannot disable otherwise valid ordinary six-frame artwork."""
    try:
        return deserialize_recipe(raw) if raw is not None else None
    except ValueError:
        return None


def _row_source(row) -> PerformanceSource:
    reference = json.loads(row["source_ref_json"])
    if type(reference) is not list:
        raise ValueError("portrait_performance_source_invalid")
    source = PerformanceSource(
        row["character_id"],
        row["expression_key"],
        row["source_sha256"],
        row["source_kind"],
        tuple(reference),
    )
    _validate_source(source)
    return source


def _prepared_base(db: CharactersRAGDB, source: PerformanceSource) -> bytes:
    from tldw_chatbook.Character_Chat.portrait_performance_authoring import (
        performance_reference,
    )

    current, raw = capture_performance_source(
        db,
        character_id=source.character_id,
        requested_state={"custom:speaking": "speaking", "custom:error": "error"}.get(
            source.expression_key, source.expression_key
        ),
        manual_expression_key=source.expression_key,
    )
    if current != source:
        raise PerformanceConflict("portrait_performance_source_changed")
    return performance_reference(raw, source.source_sha256).content


def get_regions(
    db: CharactersRAGDB, *, character_id: int, expression_key: str
) -> PortraitRegionSettings | None:
    """Read settings without hiding a stale source from the next editor."""
    row = db.execute_query(
        """SELECT r.* FROM character_portrait_regions r
        JOIN character_cards c ON c.id=r.character_id AND c.deleted=0
        WHERE r.character_id=? AND r.expression_key=?""",
        (character_id, expression_key),
    ).fetchone()
    if row is None:
        return None
    return PortraitRegionSettings(
        _row_source(row), deserialize_recipe(row["recipe_json"]), row["revision"]
    )


def save_regions(
    db: CharactersRAGDB,
    source: PerformanceSource,
    recipe: PortraitCompositionRecipe,
    *,
    expected_revision: int,
) -> int:
    """Save independently of the active set, with source and settings CAS fences."""
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("portrait_regions_revision_invalid")
    stamp = _preflight_source(db, source)
    validate_recipe(_prepared_base(db, source), recipe)
    encoded = serialize_recipe(recipe)
    with _performance_write(db) as cursor:
        if _source_stamp(db, source) != stamp:
            raise PerformanceConflict("portrait_performance_source_changed")
        row = cursor.execute(
            "SELECT * FROM character_portrait_regions WHERE character_id=? AND expression_key=?",
            (source.character_id, source.expression_key),
        ).fetchone()
        actual = 0 if row is None else row["revision"]
        if actual != expected_revision:
            raise PerformanceConflict("portrait_regions_revision_changed")
        try:
            previous_source = _row_source(row) if row is not None else None
        except ValueError:
            # A confirmed CAS replacement repairs invalid optional settings.
            previous_source = None
        if previous_source == source and row["recipe_json"] == encoded:
            return actual
        revision = actual + 1
        cursor.execute(
            """INSERT INTO character_portrait_regions
            (character_id,expression_key,source_sha256,source_kind,source_ref_json,recipe_json,revision)
            VALUES (?,?,?,?,?,?,?) ON CONFLICT(character_id,expression_key) DO UPDATE SET
            source_sha256=excluded.source_sha256,source_kind=excluded.source_kind,
            source_ref_json=excluded.source_ref_json,recipe_json=excluded.recipe_json,revision=excluded.revision""",
            (
                source.character_id,
                source.expression_key,
                source.source_sha256,
                source.source_kind,
                json.dumps(source.source_ref),
                encoded,
                revision,
            ),
        )
    return revision


def apply_performance(
    db: CharactersRAGDB,
    source: PerformanceSource,
    frames: tuple[PerformanceFrame, ...],
    *,
    expected_revision: int,
    composition_recipe: PortraitCompositionRecipe | None,
    expected_regions_revision: int | None,
) -> int:
    """Validate before SQL and publish frames/provenance under both revision fences."""
    from tldw_chatbook.Character_Chat.portrait_performance import (
        normalize_performance_frames,
    )

    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("portrait_performance_revision_invalid")
    stamp = _preflight_source(db, source)
    if (
        type(frames) is not tuple
        or len(frames) != 6
        or any(
            not isinstance(frame, PerformanceFrame) or frame.role not in FRAME_ROLES
            for frame in frames
        )
        or {frame.role for frame in frames} != set(FRAME_ROLES)
    ):
        raise ValueError("portrait_performance_requires_six_roles")
    ordered = tuple(sorted(frames, key=lambda frame: FRAME_ROLES.index(frame.role)))
    normalized = normalize_performance_frames(
        {frame.role: frame.png for frame in frames}
    )
    previous = db.get_character_portrait_performance(
        character_id=source.character_id, expression_key=source.expression_key
    )
    old = (
        {frame.role: frame for frame in previous.frames}
        if previous
        and previous.source == source
        and previous.revision == expected_revision
        else {}
    )
    for frame, checked in zip(ordered, normalized, strict=True):
        if (
            type(frame.width) is not int
            or type(frame.height) is not int
            or frame.width != checked.width
            or frame.height != checked.height
            or frame.sha256 != hashlib.sha256(frame.png).hexdigest()
            or (frame != checked and frame != old.get(frame.role))
        ):
            raise ValueError("portrait_performance_frame_metadata_invalid")
    if sum(len(frame.png) for frame in frames) > MAX_SET_BYTES:
        raise ValueError("portrait_performance_set_bytes_exceeded")
    encoded = None
    if composition_recipe is not None:
        if type(expected_regions_revision) is not int or expected_regions_revision < 1:
            raise ValueError("portrait_regions_revision_invalid")
        recover_components(_prepared_base(db, source), composition_recipe, ordered)
        encoded = serialize_recipe(composition_recipe)
    elif expected_regions_revision is not None:
        raise ValueError("portrait_composition_recipe_required")
    with _performance_write(db) as cursor:
        if _source_stamp(db, source) != stamp:
            raise PerformanceConflict("portrait_performance_source_changed")
        row = cursor.execute(
            "SELECT id,revision FROM character_portrait_performances WHERE character_id=? AND expression_key=?",
            (source.character_id, source.expression_key),
        ).fetchone()
        actual = 0 if row is None else row["revision"]
        if actual != expected_revision:
            raise PerformanceConflict("portrait_performance_revision_changed")
        if encoded is not None:
            settings = cursor.execute(
                "SELECT * FROM character_portrait_regions WHERE character_id=? AND expression_key=?",
                (source.character_id, source.expression_key),
            ).fetchone()
            if (
                settings is None
                or settings["revision"] != expected_regions_revision
                or _row_source(settings) != source
                or settings["recipe_json"] != encoded
            ):
                raise PerformanceConflict("portrait_regions_revision_changed")
        if old:
            stored = cursor.execute(
                "SELECT role,png,width,height,sha256 FROM character_portrait_performance_frames WHERE performance_id=?",
                (row["id"],),
            ).fetchall()
            if {f["role"]: PerformanceFrame(*f) for f in stored} != old:
                raise PerformanceConflict("portrait_performance_revision_changed")
        revision = actual + 1
        cursor.execute(
            """INSERT INTO character_portrait_performances
            (character_id,expression_key,source_sha256,source_kind,source_ref_json,revision,active,composition_recipe_json)
            VALUES (?,?,?,?,?,?,1,?) ON CONFLICT(character_id,expression_key) DO UPDATE SET
            source_sha256=excluded.source_sha256,source_kind=excluded.source_kind,
            source_ref_json=excluded.source_ref_json,revision=excluded.revision,active=1,
            composition_recipe_json=excluded.composition_recipe_json""",
            (
                source.character_id,
                source.expression_key,
                source.source_sha256,
                source.source_kind,
                json.dumps(source.source_ref),
                revision,
                encoded,
            ),
        )
        parent_id = cursor.execute(
            "SELECT id FROM character_portrait_performances WHERE character_id=? AND expression_key=?",
            (source.character_id, source.expression_key),
        ).fetchone()[0]
        cursor.execute(
            "DELETE FROM character_portrait_performance_frames WHERE performance_id=?",
            (parent_id,),
        )
        cursor.executemany(
            "INSERT INTO character_portrait_performance_frames (performance_id,role,png,width,height,sha256) VALUES (?,?,?,?,?,?)",
            [(parent_id, f.role, f.png, f.width, f.height, f.sha256) for f in ordered],
        )
    return revision
