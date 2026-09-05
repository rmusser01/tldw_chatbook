"""Record edits retain authored authority and immutable captured provenance."""

import pytest

from tldw_chatbook.Chunking import lab_state as state


def test_record_edit_preserves_invalid_and_pending_authority():
    assert hasattr(state, "edit_record_fields")
    session = state.new_session("test")
    candidate_id = next(iter(session.candidates))
    session = state.edit_json(session, candidate_id, '{"bad":')
    before = session.candidates[candidate_id]["draft"]
    changed = state.edit_record_fields(
        session, candidate_id, {"name": "Draft", "tags_text": "kept,"}
    )
    draft = changed.candidates[candidate_id]["draft"]
    assert draft["raw_json"] == '{"bad":'
    assert draft["parsed_json"] == before["parsed_json"]
    assert draft["parse_error"] == before["parse_error"]
    assert draft["record_fields"]["tags_text"] == "kept,"
    assert state.undo_edit(changed).candidates[candidate_id]["draft"] == before
    valid = state.discard_pending_edit(changed, candidate_id)
    pending = state.edit_control(valid, candidate_id, "chunking.config.max_size", "-")
    changed = state.edit_record_fields(
        pending, candidate_id, {"description": "still editing"}
    )
    assert changed.candidates[candidate_id]["draft"]["pending_controls"] == {
        "chunking.config.max_size": "-"
    }
    assert (
        changed.candidates[candidate_id]["draft"]["record_fields"]["tags_text"]
        == "kept,"
    )


def test_unsaved_run_captures_record_fields_without_catalog_identity():
    assert hasattr(state, "edit_record_fields")
    session = state.new_session("test")
    candidate_id = next(iter(session.candidates))
    session = state.edit_record_fields(
        session, candidate_id, {"name": "Captured", "tags_text": "original,"}
    )
    request = state.capture_batch(session, (candidate_id,))[0]
    assert request.template_record == {
        "name": "Captured",
        "description": "",
        "tags": ["original"],
    }
    state.edit_record_fields(session, candidate_id, {"tags": ["new"]})
    assert request.template_record["tags"] == ["original"]
    assert (
        session.candidates[candidate_id]["draft"]["record_fields"]["tags_text"]
        == "original,"
    )


def test_saved_association_keeps_new_edits_and_refuses_replaced_draft():
    assert hasattr(state, "associate_saved_record")
    session = state.new_session("test")
    candidate_id = next(iter(session.candidates))
    captured = session.candidates[candidate_id]["draft"]
    changed = state.edit_json(session, candidate_id, '{"bad":')
    saved = {
        "id": 3,
        "uuid": "uuid",
        "version": 1,
        "name": "saved",
        "description": "",
        "tags": [],
    }
    associated = state.associate_saved_record(
        changed, candidate_id, saved, captured_draft=captured
    )
    assert associated.candidates[candidate_id]["draft"]["raw_json"] == '{"bad":'
    assert associated.candidates[candidate_id]["draft"]["expected_record"] == {
        "id": 3,
        "uuid": "uuid",
        "version": 1,
    }
    assert associated.undo == changed.undo
    replaced = state.replace_template(
        changed,
        candidate_id,
        {"chunking": {"method": "words"}},
        record_fields={},
        expected_record={"id": 7, "uuid": "other", "version": 1},
    )
    with pytest.raises(ValueError, match="replaced"):
        state.associate_saved_record(
            replaced, candidate_id, saved, captured_draft=captured
        )
    imported = state.replace_template(
        changed,
        candidate_id,
        {"chunking": {"method": "words"}},
        record_fields={},
        expected_record=None,
    )
    with pytest.raises(ValueError, match="replaced"):
        state.associate_saved_record(
            imported,
            candidate_id,
            saved,
            captured_draft=captured,
            captured_generation=session.candidates[candidate_id].get(
                "draft_generation"
            ),
        )
