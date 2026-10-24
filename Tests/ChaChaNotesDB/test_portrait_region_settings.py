"""Real SQLite proof of independent settings CAS and paired-frame provenance."""

from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image

from Tests.Character_Chat.test_portrait_compositing import core, scene
from tldw_chatbook.Character_Chat.portrait_performance import (
    PerformanceConflict,
    capture_performance_source,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def case(tmp_path):
    base, recipe, components = scene()
    db = CharactersRAGDB(tmp_path / "regions.db", "region-test")
    cid = db.add_character_card({"name": "Paired", "image": base})
    source, _ = capture_performance_source(db, character_id=cid, requested_state="idle")
    frames = core().assemble_components(base, recipe, components)
    try:
        yield db, source, base, recipe, frames
    finally:
        db.close()


def get(db, source):
    return db.get_character_portrait_performance(
        character_id=source.character_id, expression_key=source.expression_key
    )


def regions(db, source):
    return db.get_character_portrait_regions(
        character_id=source.character_id, expression_key=source.expression_key
    )


def test_regions_save_is_independent_idempotent_and_revision_checked(case):
    db, source, _, recipe, frames = case
    db.apply_character_portrait_performance(source, frames, expected_revision=0)
    before = get(db, source)
    rev = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    assert rev == 1 and get(db, source) == before
    assert regions(db, source).source == source and regions(db, source).recipe == recipe
    assert (
        db.save_character_portrait_regions(source, recipe, expected_revision=rev) == rev
    )
    with pytest.raises(PerformanceConflict):
        db.save_character_portrait_regions(source, recipe, expected_revision=0)
    changed = replace(recipe, mouth=replace(recipe.mouth, width=19))
    assert (
        db.save_character_portrait_regions(source, changed, expected_revision=rev) == 2
    )
    with pytest.raises(PerformanceConflict):
        db.apply_character_portrait_performance(
            source,
            frames,
            expected_revision=1,
            composition_recipe=recipe,
            expected_regions_revision=rev,
        )
    assert get(db, source) == before


def test_composited_apply_round_trip_and_manual_apply_clears_provenance(case):
    db, source, base, recipe, frames = case
    rev = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    published = db.apply_character_portrait_performance(
        source,
        frames,
        expected_revision=0,
        composition_recipe=recipe,
        expected_regions_revision=rev,
    )
    record = get(db, source)
    assert record.composition_recipe == recipe and record.revision == published
    assert core().recover_components(base, record.composition_recipe, record.frames)
    assert (
        db.save_character_portrait_regions(source, recipe, expected_revision=rev) == rev
    )
    db.apply_character_portrait_performance(source, frames, expected_revision=published)
    assert get(db, source).composition_recipe is None
    assert regions(db, source).revision == rev


def test_reencoded_stored_role_is_validated_without_rewriting_unaffected_bytes(case):
    import hashlib

    db, source, base, recipe, frames = case
    rev = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    db.apply_character_portrait_performance(
        source,
        frames,
        expected_revision=0,
        composition_recipe=recipe,
        expected_regions_revision=rev,
    )
    with Image.open(BytesIO(frames[0].png)) as image, BytesIO() as output:
        image.save(output, format="PNG", compress_level=0)
        stored = output.getvalue()
    assert stored != frames[0].png
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE character_portrait_performance_frames SET png=?,sha256=? WHERE role='rest'",
            (stored, hashlib.sha256(stored).hexdigest()),
        )
    before = get(db, source)
    db.apply_character_portrait_performance(
        source,
        before.frames,
        expected_revision=before.revision,
        composition_recipe=recipe,
        expected_regions_revision=rev,
    )
    assert get(db, source).frames[0].png == stored
    with Image.open(BytesIO(base)) as image, BytesIO() as output:
        image.save(output, format="PNG", compress_level=0)
        core().validate_recipe(output.getvalue(), regions(db, source).recipe)
    assert regions(db, source).revision == rev


def test_source_change_and_failed_pairing_leave_settings_and_frames_intact(case):
    db, source, _, recipe, frames = case
    rev = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    with pytest.raises(ValueError):
        db.apply_character_portrait_performance(
            source,
            frames[:-1] + (frames[0],),
            expected_revision=0,
            composition_recipe=recipe,
            expected_regions_revision=rev,
        )
    assert get(db, source) is None
    card = db.get_character_card_by_id(source.character_id)
    db.update_character_card(
        source.character_id, {"image": frames[2].png}, card["version"]
    )
    with pytest.raises(PerformanceConflict):
        db.save_character_portrait_regions(source, recipe, expected_revision=rev)
    assert regions(db, source).revision == rev


def test_remove_and_character_lifetime_preserve_only_intended_rows(case):
    db, source, _, recipe, frames = case
    rev = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    first = db.apply_character_portrait_performance(
        source,
        frames,
        expected_revision=0,
        composition_recipe=recipe,
        expected_regions_revision=rev,
    )
    db.remove_character_portrait_performance(source, expected_revision=first)
    assert (
        regions(db, source).recipe == recipe
        and get(db, source).composition_recipe is None
    )
    card = db.get_character_card_by_id(source.character_id)
    assert db.soft_delete_character_card(source.character_id, card["version"])
    assert regions(db, source) is None
    assert db.restore_character_card(source.character_id, card["version"] + 1)
    assert regions(db, source).recipe == recipe
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM character_cards WHERE id=?", (source.character_id,))
    assert (
        db.execute_query("SELECT count(*) FROM character_portrait_regions").fetchone()[
            0
        ]
        == 0
    )


def test_invalid_applied_provenance_does_not_disable_valid_ordinary_frames(case):
    db, source, _, recipe, frames = case
    rev = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    db.apply_character_portrait_performance(
        source,
        frames,
        expected_revision=0,
        composition_recipe=recipe,
        expected_regions_revision=rev,
    )
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE character_portrait_performances SET composition_recipe_json=?",
            ('{"algorithm_version":999}',),
        )
    record = get(db, source)
    assert (
        record.active and record.frames == frames and record.composition_recipe is None
    )


@pytest.mark.parametrize(
    "damage",
    ["unknown_version", "extra_key", "bool_coordinate", "too_long", "not_object"],
)
def test_recipe_deserialization_fails_closed(case, damage):
    import json

    from tldw_chatbook.DB.portrait_region_settings import (
        deserialize_recipe,
        serialize_recipe,
    )

    _, _, _, recipe, _ = case
    raw = serialize_recipe(recipe)
    data = json.loads(raw)
    if damage == "unknown_version":
        data["algorithm_version"] = 999
    elif damage == "extra_key":
        data["path"] = "/private/source.png"
    elif damage == "bool_coordinate":
        data["mouth"]["x"] = True
    elif damage == "too_long":
        data["base_pixel_sha256"] = "a" * 2049
    else:
        data = []
    with pytest.raises(ValueError):
        deserialize_recipe(json.dumps(data))


def test_save_and_composed_apply_have_no_sync_or_partial_commit(case):
    db, source, _, recipe, frames = case
    before_sync = db.execute_query("SELECT count(*) FROM sync_log").fetchone()[0]
    revision = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    first = db.apply_character_portrait_performance(
        source,
        frames,
        expected_revision=0,
        composition_recipe=recipe,
        expected_regions_revision=revision,
    )
    before = get(db, source)
    with db.transaction() as cursor:
        cursor.execute(
            "CREATE TEMP TRIGGER fail_pair BEFORE INSERT ON character_portrait_performance_frames WHEN NEW.role='blink_small' BEGIN SELECT RAISE(ABORT, 'fixture'); END"
        )
        with pytest.raises(Exception, match="fixture"):
            db.apply_character_portrait_performance(
                source, frames, expected_revision=first
            )
    assert get(db, source) == before
    assert regions(db, source).revision == revision
    assert (
        db.execute_query("SELECT count(*) FROM sync_log").fetchone()[0] == before_sync
    )


def test_actual_pair_corruption_and_base_pixel_change_cannot_publish(case):
    from tldw_chatbook.Character_Chat.portrait_performance import (
        normalize_performance_frame,
    )

    db, source, _, recipe, frames = case
    revision = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    bad = frames[:-1] + (normalize_performance_frame("blink_open", frames[1].png),)
    with pytest.raises(ValueError):
        db.apply_character_portrait_performance(
            source,
            bad,
            expected_revision=0,
            composition_recipe=recipe,
            expected_regions_revision=revision,
        )
    with pytest.raises(ValueError):
        db.save_character_portrait_regions(
            source,
            replace(recipe, base_pixel_sha256="0" * 64),
            expected_revision=revision,
        )
    assert get(db, source) is None and regions(db, source).recipe == recipe


def test_settings_two_connections_cannot_overwrite_same_revision(case):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    db, source, _, recipe, _ = case
    barrier = Barrier(2)

    def save():
        other = CharactersRAGDB(db.db_path_str, "region-racer")
        try:
            barrier.wait(timeout=10)
            try:
                return other.save_character_portrait_regions(
                    source, recipe, expected_revision=0
                )
            except PerformanceConflict:
                return "conflict"
        finally:
            other.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: save(), range(2)))
    assert results.count("conflict") == 1 and results.count(1) == 1


@pytest.mark.parametrize("change", ["regions", "source"])
def test_change_during_composed_validation_cannot_publish(case, monkeypatch, change):
    from tldw_chatbook.DB import portrait_region_settings as storage

    db, source, _, recipe, frames = case
    revision = db.save_character_portrait_regions(source, recipe, expected_revision=0)
    original = storage.recover_components

    def changed(*args):
        result = original(*args)
        if change == "regions":
            db.save_character_portrait_regions(
                source,
                replace(recipe, mouth=replace(recipe.mouth, width=19)),
                expected_revision=revision,
            )
        else:
            card = db.get_character_card_by_id(source.character_id)
            db.update_character_card(
                source.character_id, {"image": frames[2].png}, card["version"]
            )
        return result

    monkeypatch.setattr(storage, "recover_components", changed)
    with pytest.raises(PerformanceConflict):
        db.apply_character_portrait_performance(
            source,
            frames,
            expected_revision=0,
            composition_recipe=recipe,
            expected_regions_revision=revision,
        )
    assert get(db, source) is None
