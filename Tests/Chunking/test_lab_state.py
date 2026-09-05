import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_chatbook.Chunking.lab_models import (
    DraftState,
    ExecutionReport,
    LabSession,
    RunRequest,
    RunResult,
)
from tldw_chatbook.Chunking.lab_state import (
    accept_result,
    can_execute,
    capture_batch,
    discard_pending_edit,
    edit_control,
    edit_json,
    install_batch,
    is_result_stale,
    new_session,
    pin_baseline,
    replace_sample,
    replace_template,
    undo_edit,
    update_view,
)


def test_invalid_json_is_the_current_draft():
    session = new_session("test-profile")
    candidate_id = next(iter(session.candidates))
    changed = edit_json(session, candidate_id, '{"chunking":')
    assert changed.candidates[candidate_id]["draft"]["raw_json"] == '{"chunking":'
    assert changed.revision == session.revision + 1
    assert not can_execute(changed, candidate_id)


def _candidate_b(session: LabSession) -> str:
    return next(
        candidate_id
        for candidate_id, candidate in session.candidates.items()
        if candidate["role"] == "B"
    )


def _completed(request: RunRequest, text: str = "chunk") -> RunResult:
    return RunResult(
        request=request,
        status="completed",
        report=ExecutionReport(
            chunks=(
                {
                    "text": text,
                    "metadata": {},
                    "provenance": {},
                    "span": None,
                },
            ),
            transformed_text=text,
        ),
        started_at="2026-09-04T00:00:00Z",
        finished_at="2026-09-04T00:00:01Z",
        elapsed_ms=1.0,
        error=None,
    )


def _run_b(session: LabSession) -> tuple[LabSession, RunRequest]:
    request = capture_batch(session, (_candidate_b(session),))[0]
    installed = install_batch(session, (request,))
    return accept_result(installed, _completed(request)), request


def test_invalid_json_keeps_last_valid_document_until_explicit_discard():
    session = new_session("profile")
    candidate_id = _candidate_b(session)
    valid = edit_json(
        session,
        candidate_id,
        '{"chunking":{"method":"words"},"metadata":{"kept":[1]}}',
    )
    invalid = edit_json(valid, candidate_id, '{"chunking":')
    draft = invalid.candidates[candidate_id]["draft"]
    assert json.loads(draft["parsed_json"])["metadata"] == {"kept": [1]}
    assert draft["parse_error"] == {
        "message": "Expecting value",
        "line": 1,
        "column": 13,
    }
    restored = discard_pending_edit(invalid, candidate_id)
    assert (
        restored.candidates[candidate_id]["draft"]["raw_json"] == draft["parsed_json"]
    )
    assert can_execute(restored, candidate_id)
    assert undo_edit(restored).candidates[candidate_id]["draft"]["raw_json"] == (
        '{"chunking":'
    )


def test_invalid_control_owns_exact_raw_text_and_blocks_json_until_discarded():
    session = new_session("profile")
    candidate_id = _candidate_b(session)
    pending = edit_control(session, candidate_id, "chunking.config.max_size", "12e")
    draft = pending.candidates[candidate_id]["draft"]
    assert draft["pending_controls"] == {"chunking.config.max_size": "12e"}
    assert draft["authority"] == "controls"
    assert not can_execute(pending, candidate_id)
    with pytest.raises(ValueError, match="pending control"):
        edit_json(pending, candidate_id, "{}")
    restored = discard_pending_edit(pending, candidate_id)
    assert restored.candidates[candidate_id]["draft"]["pending_controls"] == {}
    assert can_execute(restored, candidate_id)


@given(
    extension=st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(),
        lambda children: (
            st.lists(children, max_size=4)
            | st.dictionaries(st.text(min_size=1, max_size=8), children, max_size=4)
        ),
        max_leaves=12,
    )
)
def test_control_patch_round_trip_preserves_nested_metadata(extension):
    session = new_session("profile")
    candidate_id = _candidate_b(session)
    body = {
        "chunking": {"method": "words", "config": {"future_option": extension}},
        "metadata": {"extension": extension},
        "classifier": {"rules": [{"when": extension}]},
        "preprocessing": [
            {"operation": "detect_language"},
            {"operation": "normalize_whitespace"},
        ],
    }
    loaded = edit_json(session, candidate_id, json.dumps(body, ensure_ascii=False))
    changed = edit_control(loaded, candidate_id, "chunking.config.max_size", "240")
    actual = json.loads(changed.candidates[candidate_id]["draft"]["parsed_json"])
    assert actual["metadata"] == body["metadata"]
    assert actual["classifier"] == body["classifier"]
    assert actual["chunking"]["config"]["future_option"] == extension
    assert actual["preprocessing"] == body["preprocessing"]
    assert actual["chunking"]["config"]["max_size"] == 240


def test_method_switch_preserves_options_and_order_instead_of_normalizing():
    session = new_session("profile")
    candidate_id = _candidate_b(session)
    body = {
        "preprocessing": [
            {"operation": "clean_markdown"},
            {"operation": "detect_language"},
        ],
        "chunking": {
            "method": "words",
            "config": {"preserve_sentences": True, "future": {"x": 1}},
        },
        "postprocessing": [
            {"operation": "add_overlap"},
            {"operation": "filter_empty"},
        ],
    }
    loaded = edit_json(session, candidate_id, json.dumps(body))
    switched = edit_control(loaded, candidate_id, "chunking.method", "fixed_size")
    actual = json.loads(switched.candidates[candidate_id]["draft"]["parsed_json"])
    assert actual["chunking"] == {
        "method": "fixed_size",
        "config": {"preserve_sentences": True, "future": {"x": 1}},
    }
    assert actual["preprocessing"] == body["preprocessing"]
    assert actual["postprocessing"] == body["postprocessing"]
    assert not can_execute(switched, candidate_id)


@pytest.mark.parametrize(
    "raw",
    [
        '[{"stage":"chunk"}]',
        '{"parent":"legacy","stages":[]}',
        '{"chunking":{"method":"unknown"},"extension":{"nested":true}}',
    ],
)
def test_unknown_shapes_remain_lossless_but_are_not_executable(raw):
    session = new_session("profile")
    candidate_id = _candidate_b(session)
    changed = edit_json(session, candidate_id, raw)
    assert changed.candidates[candidate_id]["draft"]["raw_json"] == raw
    assert not can_execute(changed, candidate_id)


def test_sample_hashes_exact_utf8_and_sample_replacement_is_undoable():
    session = new_session("profile")
    changed = replace_sample(session, "café\n", {"kind": "file", "name": "x.txt"})
    assert changed.view["sample_hash"] == (
        "7b49b9e063bd91a4f9252b413261f5557b9c570aa61516989499f64a62dbcdd6"
    )
    assert changed.samples[changed.view["sample_hash"]]["text"] == "café\n"
    restored = undo_edit(changed)
    assert restored.view["sample_hash"] == session.view["sample_hash"]
    assert restored.revision == changed.revision + 1


def test_same_text_source_replacement_is_undoable_without_changing_identity():
    session = replace_sample(
        new_session("profile"), "same", {"kind": "file", "name": "first.txt"}
    )
    changed = replace_sample(session, "same", {"kind": "library", "media_id": 7})
    assert changed.view["sample_hash"] == session.view["sample_hash"]
    assert changed.samples[changed.view["sample_hash"]]["source"] == {
        "kind": "library",
        "media_id": 7,
    }
    restored = undo_edit(changed)
    assert restored.samples[restored.view["sample_hash"]]["source"] == {
        "kind": "file",
        "name": "first.txt",
    }


def test_batch_capture_is_pure_and_freezes_sample_recipe_and_loaded_record():
    source = {"kind": "library", "locator": {"media_id": 7}}
    body = {"chunking": {"method": "words"}, "metadata": {"x": [1]}}
    fields = {"name": "Loaded", "description": "before", "tags": ["one"]}
    identity = {"id": 9, "uuid": "template-uuid", "version": 3}
    session = new_session("profile")
    candidate_id = _candidate_b(session)
    session = replace_sample(session, "same sample", source)
    session = replace_template(
        session,
        candidate_id,
        body,
        record_fields=fields,
        expected_record=identity,
    )
    before = session.model_dump(mode="json")
    request = capture_batch(session, (candidate_id,))[0]
    source["locator"]["media_id"] = 99
    body["metadata"]["x"].append(2)
    fields["tags"].append("two")
    identity["version"] = 4
    assert session.model_dump(mode="json") == before
    assert json.loads(request.recipe.authored_json)["metadata"] == {"x": [1]}
    assert request.sample.source["locator"]["media_id"] == 7
    assert request.template_record == {
        "id": 9,
        "uuid": "template-uuid",
        "version": 3,
        "name": "Loaded",
        "description": "before",
        "tags": ["one"],
    }


def test_same_sample_different_configs_have_distinct_recipe_identities():
    session = replace_sample(new_session("profile"), "same", {"kind": "paste"})
    session, _ = _run_b(session)
    session = pin_baseline(session)
    candidate_id = _candidate_b(session)
    session = edit_control(session, candidate_id, "chunking.config.max_size", "99")
    requests = capture_batch(
        session,
        tuple(session.candidates),
    )
    assert [
        session.candidates[request.candidate_id]["role"] for request in requests
    ] == [
        "A",
        "B",
    ]
    assert len({request.sample.sample_hash for request in requests}) == 1
    assert len({request.recipe.recipe_hash for request in requests}) == 2


def test_result_staleness_tracks_sample_and_candidate_recipe_not_template_name():
    session = replace_sample(new_session("profile"), "sample", {"kind": "paste"})
    candidate_id = _candidate_b(session)
    session = replace_template(
        session,
        candidate_id,
        {"chunking": {"method": "words"}},
        record_fields={"name": "old name", "description": "old", "tags": ["old"]},
        expected_record={"id": 1, "uuid": "same", "version": 1},
    )
    session, request = _run_b(session)
    assert not is_result_stale(session, candidate_id, request.run_id)
    renamed = replace_template(
        session,
        candidate_id,
        json.loads(request.recipe.authored_json),
        record_fields={"name": "new name", "description": "", "tags": []},
        expected_record={"id": 1, "uuid": "same", "version": 1},
    )
    assert not is_result_stale(renamed, candidate_id, request.run_id)
    pinned = pin_baseline(renamed)
    baseline = next(
        candidate
        for candidate in pinned.candidates.values()
        if candidate["role"] == "A"
    )
    assert baseline["template_record"] == {
        "id": 1,
        "uuid": "same",
        "version": 1,
        "name": "old name",
        "description": "old",
        "tags": ["old"],
    }
    edited = edit_control(renamed, candidate_id, "chunking.config.max_size", "88")
    assert is_result_stale(edited, candidate_id, request.run_id)
    sample_changed = replace_sample(renamed, "sample ", {"kind": "paste"})
    assert is_result_stale(sample_changed, candidate_id, request.run_id)


def test_pin_requires_current_completed_b_and_replacement_is_deliberate_and_undoable():
    session = new_session("profile")
    with pytest.raises(ValueError, match="completed current B"):
        pin_baseline(session)
    session, request = _run_b(session)
    pinned = pin_baseline(session)
    baseline_id = next(
        key for key, candidate in pinned.candidates.items() if candidate["role"] == "A"
    )
    assert pinned.candidates[baseline_id]["pinned_recipe"] == request.recipe.model_dump(
        mode="json"
    )
    with pytest.raises(ValueError, match="replace"):
        pin_baseline(pinned)
    candidate_id = _candidate_b(pinned)
    edited = edit_control(pinned, candidate_id, "chunking.config.max_size", "75")
    edited, newer_request = _run_b(edited)
    replaced = pin_baseline(edited, replace=True)
    assert baseline_id in replaced.candidates
    assert replaced.candidates[baseline_id]["pinned_recipe"] == (
        newer_request.recipe.model_dump(mode="json")
    )
    restored = undo_edit(replaced)
    assert restored.candidates[baseline_id]["pinned_recipe"] == (
        request.recipe.model_dump(mode="json")
    )


def test_loaded_template_replacement_is_explicit_and_undoable():
    session = new_session("profile")
    candidate_id = _candidate_b(session)
    original = session.candidates[candidate_id]["draft"]
    loaded = replace_template(
        session,
        candidate_id,
        {"chunking": {"method": "fixed_size"}},
        record_fields={"name": "Saved", "description": "desc", "tags": ["tag"]},
        expected_record={"id": 4, "uuid": "u", "version": 2},
    )
    assert (
        json.loads(loaded.candidates[candidate_id]["draft"]["parsed_json"])["chunking"][
            "method"
        ]
        == "fixed_size"
    )
    assert undo_edit(loaded).candidates[candidate_id]["draft"] == original


def test_view_only_transition_keeps_content_undo_available():
    session = new_session("profile")
    changed = replace_sample(session, "new sample", {"kind": "paste"})
    viewed = update_view(changed, {"result_mode": "compare", "selected_chunk": 2})
    assert viewed.revision == changed.revision + 1
    assert viewed.undo is changed.undo
    assert undo_edit(viewed).view["sample_hash"] == session.view["sample_hash"]
    with pytest.raises(ValueError, match="retained sample"):
        update_view(viewed, {"sample_hash": "missing"})


def test_install_rejects_capture_after_inputs_change_and_late_results_are_fenced():
    session = new_session("profile")
    candidate_id = _candidate_b(session)
    request = capture_batch(session, (candidate_id,))[0]
    changed = edit_control(session, candidate_id, "chunking.config.max_size", "22")
    with pytest.raises(ValueError, match="captured inputs"):
        install_batch(changed, (request,))
    installed = install_batch(session, (request,))
    newer_draft = edit_control(
        installed, candidate_id, "chunking.config.max_size", "33"
    )
    accepted = accept_result(newer_draft, _completed(request))
    assert is_result_stale(accepted, candidate_id, request.run_id)
    foreign = _completed(
        request.model_copy(update={"epoch": new_session("profile").epoch})
    )
    with pytest.raises(ValueError, match="epoch"):
        accept_result(installed, foreign)
    not_member = _completed(request.model_copy(update={"run_id": "not-a-member"}))
    with pytest.raises(ValueError, match="membership"):
        accept_result(installed, not_member)


def test_failed_current_batch_member_cannot_borrow_previous_completed_result():
    session, first = _run_b(new_session("profile"))
    candidate_id = _candidate_b(session)
    second = capture_batch(session, (candidate_id,))[0]
    installed = install_batch(session, (second,))
    failed = RunResult(
        request=second,
        status="failed",
        report=None,
        started_at="start",
        finished_at="end",
        elapsed_ms=1,
        error={"message": "failed"},
    )
    accepted = accept_result(installed, failed)
    assert accepted.candidates[candidate_id]["current_run_id"] == second.run_id
    assert accepted.candidates[candidate_id]["previous_run_id"] == first.run_id
    with pytest.raises(ValueError, match="completed current B"):
        pin_baseline(accepted)


def test_undo_pin_invalidates_batch_that_captured_removed_baseline():
    session, _ = _run_b(new_session("profile"))
    pinned = pin_baseline(session)
    baseline_id = next(
        candidate_id
        for candidate_id, candidate in pinned.candidates.items()
        if candidate["role"] == "A"
    )
    requests = capture_batch(pinned, tuple(pinned.candidates))
    installed = install_batch(pinned, requests)

    undone = undo_edit(installed)

    assert baseline_id not in undone.candidates
    assert undone.batch is None
    json.loads(undone.model_dump_json())
    for request in requests:
        with pytest.raises(ValueError, match="not active"):
            accept_result(undone, _completed(request))


def test_pure_draft_edits_reuse_large_result_and_sample_maps():
    session, _ = _run_b(new_session("profile"))
    changed = edit_json(session, _candidate_b(session), '{"chunking":')
    assert changed.results is session.results
    assert changed.samples is session.samples


def test_models_detach_nested_inputs_and_publication_outputs():
    fields = {"name": "n", "description": "", "tags": [{"nested": [1]}]}
    draft = DraftState(
        raw_json="{}",
        parsed_json="{}",
        parse_error=None,
        pending_controls={},
        authority="synced",
        record_fields=fields,
        expected_record=None,
    )
    fields["tags"][0]["nested"].append(2)
    assert draft.record_fields["tags"] == [{"nested": [1]}]
    dumped = draft.model_dump(mode="json")
    dumped["record_fields"]["tags"][0]["nested"].append(3)
    assert draft.record_fields["tags"] == [{"nested": [1]}]

    session = new_session("profile")
    published = session.model_dump(mode="json")
    published["candidates"][_candidate_b(session)]["draft"]["record_fields"][
        "tags"
    ].append("outside")
    assert (
        session.candidates[_candidate_b(session)]["draft"]["record_fields"]["tags"]
        == []
    )

    payload = session.model_dump(mode="json")
    restored = LabSession.model_validate(payload)
    payload["candidates"][_candidate_b(session)]["draft"]["record_fields"][
        "tags"
    ].append("mutated input")
    assert (
        restored.candidates[_candidate_b(session)]["draft"]["record_fields"]["tags"]
        == []
    )


def test_publication_revalidates_nested_snapshot_mutation():
    session = new_session("profile")
    candidate_id = _candidate_b(session)
    session.candidates[candidate_id]["role"] = "A"
    with pytest.raises(ValueError, match="editable B"):
        session.model_dump(mode="json")


def test_run_result_status_and_session_reference_integrity_are_validated():
    session = new_session("profile")
    request = capture_batch(session, (_candidate_b(session),))[0]
    with pytest.raises(ValueError, match="completed"):
        RunResult(
            request=request,
            status="failed",
            report=_completed(request).report,
            started_at="start",
            finished_at="end",
            elapsed_ms=1,
            error={"message": "failed"},
        )
    payload = session.model_dump(mode="json")
    candidate = next(iter(payload["candidates"].values()))
    payload["candidates"]["third"] = {
        **candidate,
        "candidate_id": "third",
        "role": "A",
        "editable": False,
        "draft": None,
        "pinned_recipe": capture_batch(session, (_candidate_b(session),))[
            0
        ].recipe.model_dump(mode="json"),
    }
    payload["candidates"]["fourth"] = {
        **payload["candidates"]["third"],
        "candidate_id": "fourth",
    }
    with pytest.raises(ValueError, match="at most two"):
        LabSession.model_validate(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"authority": "synced", "parse_error": {"message": "bad"}}, "JSON"),
        ({"authority": "json", "pending_controls": {"x": "-"}}, "control"),
        ({"authority": "controls", "pending_controls": {}}, "pending"),
    ],
)
def test_draft_model_rejects_competing_or_inconsistent_authority(changes, message):
    payload = {
        "raw_json": "{}",
        "parsed_json": "{}",
        "parse_error": None,
        "pending_controls": {},
        "authority": "synced",
        "record_fields": {},
        "expected_record": None,
    }
    payload.update(changes)
    with pytest.raises(ValueError, match=message):
        DraftState.model_validate(payload)
