"""Non-executing recovery transfer and bounded authoring evidence."""

import hashlib
import json

import pytest

from Tests.DB.test_chunking_lab_db import completed_session
from tldw_chatbook.Chunking.lab_models import canonical_json
from tldw_chatbook.Chunking.lab_recovery import (
    RecoveryImportError,
    export_recovery,
    parse_recovery,
)
from tldw_chatbook.Chunking.lab_state import edit_json, new_session, replace_sample


def test_recovery_export_preserves_invalid_authoring_text():
    session = new_session("test-profile")
    candidate_id = next(iter(session.candidates))
    session = edit_json(session, candidate_id, '{"chunking":')
    restored = parse_recovery(export_recovery(session))
    assert restored.candidates == session.candidates
    assert restored.undo == ()


@pytest.mark.parametrize(
    "payload",
    [
        b"{",
        b"[]",
        b'{"version":2}',
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b"\xff",
        b"[" * 65 + b"0" + b"]" * 65,
    ],
)
def test_malformed_recovery_is_rejected(payload):
    with pytest.raises(RecoveryImportError):
        parse_recovery(payload)


def test_duplicate_key_in_otherwise_valid_envelope_is_rejected():
    payload = export_recovery(new_session("profile"))
    with pytest.raises(RecoveryImportError):
        parse_recovery(payload[:-1] + b',"version":1}')


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_value_in_otherwise_valid_envelope_is_rejected(invalid):
    envelope = json.loads(export_recovery(new_session("profile")))
    envelope["session"]["view"]["invalid"] = invalid
    with pytest.raises(RecoveryImportError):
        parse_recovery(json.dumps(envelope).encode())


def test_newer_version_in_otherwise_valid_envelope_is_rejected():
    envelope = json.loads(export_recovery(new_session("profile")))
    envelope["version"] = 2
    with pytest.raises(RecoveryImportError):
        parse_recovery(json.dumps(envelope).encode())


def test_changed_sample_digest_is_rejected():
    envelope = json.loads(export_recovery(new_session("profile")))
    sample = next(iter(envelope["session"]["samples"].values()))
    sample["text"] = "changed"
    with pytest.raises(RecoveryImportError):
        parse_recovery(json.dumps(envelope).encode())


@pytest.mark.parametrize("text", ["é" * (1024 * 1024), "\x00" * (2 * 1024 * 1024)])
def test_exact_sample_text_limit_survives_json_escaping(text):
    session = replace_sample(new_session("profile"), text, {"path": "/missing/source"})
    restored = parse_recovery(export_recovery(session))
    assert restored.samples[restored.view["sample_hash"]]["text"] == text


def test_oversized_raw_draft_is_rejected_on_import():
    envelope = json.loads(export_recovery(new_session("profile")))
    next(iter(envelope["session"]["candidates"].values()))["draft"]["raw_json"] = (
        "x" * (2 * 1024 * 1024 + 1)
    )
    with pytest.raises(RecoveryImportError):
        parse_recovery(json.dumps(envelope).encode())


def test_pending_controls_and_old_captured_runtime_remain_readable(monkeypatch):
    from tldw_chatbook.Chunking import lab_preflight, lab_state
    from tldw_chatbook.Chunking.lab_state import edit_control, pin_baseline

    session = pin_baseline(completed_session())
    candidate_id = next(
        key for key, candidate in session.candidates.items() if candidate["role"] == "B"
    )
    session = edit_control(session, candidate_id, "chunking.config.max_size", "12e")

    def unavailable(*args, **kwargs):
        raise AssertionError("Import must never call executable preflight")

    monkeypatch.setattr(lab_preflight, "prepare_recipe", unavailable)
    monkeypatch.setattr(lab_state, "prepare_recipe", unavailable)
    restored = parse_recovery(export_recovery(session))
    assert restored.candidates == session.candidates
    assert restored.results == session.results


def test_old_unsupported_captured_recipe_uses_original_identity():
    from tldw_chatbook.Chunking.lab_models import LabSession

    document = completed_session().model_dump(mode="json")
    result = next(iter(document["results"].values()))
    recipe = result["request"]["recipe"]
    recipe["runtime"]["engine_version"] = "old-engine-no-longer-installed"
    recipe["effective_json"] = '{"legacy_operation":{"removed":true}}'
    identity = {
        "authored": json.loads(recipe["authored_json"]),
        "effective": json.loads(recipe["effective_json"]),
        "runtime": recipe["runtime"],
    }
    recipe["recipe_hash"] = hashlib.sha256(
        canonical_json(identity).encode()
    ).hexdigest()
    document["batch"]["requests"][result["request"]["run_id"]] = result["request"]
    session = LabSession.model_validate(document)
    restored = parse_recovery(export_recovery(session))
    assert restored.results == session.results


@pytest.mark.parametrize(
    "change", ["digest", "recipe", "membership", "dangling", "batch_outcome"]
)
def test_result_integrity_and_membership_are_checked_independently(change):
    envelope = json.loads(export_recovery(completed_session()))
    session = envelope["session"]
    run_id, result = next(iter(session["results"].items()))
    if change == "digest":
        result["report"]["transformed_text"] = "tampered"
    elif change == "recipe":
        result["request"]["recipe"]["effective_json"] = "{}"
    elif change == "membership":
        result["request"]["candidate_id"] = "outsider"
    elif change == "dangling":
        next(iter(session["candidates"].values()))["current_run_id"] = "missing"
    else:
        session["batch"]["outcomes"][run_id] = "failed"
    if change != "digest":
        envelope["digests"]["results"][run_id] = hashlib.sha256(
            canonical_json(result).encode()
        ).hexdigest()
    with pytest.raises(RecoveryImportError):
        parse_recovery(json.dumps(envelope).encode())


def test_unfinished_import_is_interrupted_without_execution():
    from tldw_chatbook.Chunking.lab_state import capture_batch, install_batch

    session = new_session("profile")
    request = capture_batch(session, tuple(session.candidates))[0]
    session = install_batch(session, (request,))
    restored = parse_recovery(export_recovery(session))
    assert restored.results[request.run_id]["status"] == "interrupted"
    assert restored.batch["outcomes"] == {request.run_id: "interrupted"}


@pytest.mark.parametrize("boundary", ["parse", "load", "rebase"])
def test_recovery_boundaries_share_partial_batch_interruption_policy(
    tmp_path, boundary
):
    from Tests.Chunking.test_lab_state import _completed
    from tldw_chatbook.Chunking.lab_recovery import rebase_recovery
    from tldw_chatbook.Chunking.lab_state import (
        accept_result,
        capture_batch,
        install_batch,
        pin_baseline,
    )
    from tldw_chatbook.DB.Chunking_Lab_DB import CheckpointStore

    session = pin_baseline(completed_session())
    requests = capture_batch(session, tuple(session.candidates))
    session = accept_result(install_batch(session, requests), _completed(requests[0]))
    completed = session.results[requests[0].run_id]
    if boundary == "parse":
        restored = parse_recovery(export_recovery(session))
    elif boundary == "load":
        store = CheckpointStore(tmp_path / "lab.sqlite3", session.profile_key)
        try:
            token = store.save(session, expected=None)
            restored, loaded_token = store.load()
            assert loaded_token == token
        finally:
            store.close()
    else:
        restored = rebase_recovery(session, "another-profile", "new-authority")
    assert restored.results[requests[0].run_id] == completed
    interrupted = restored.results[requests[1].run_id]
    assert interrupted == {
        "request": requests[1].model_dump(mode="json"),
        "status": "interrupted",
        "report": None,
        "started_at": "",
        "finished_at": "",
        "elapsed_ms": 0.0,
        "error": {"message": "Preview interrupted before recovery"},
    }
    assert restored.revision == session.revision + 1
    assert restored.content_revision == session.content_revision + 1
    if boundary == "rebase":
        assert restored.batch is None
        assert (restored.profile_key, restored.epoch) == (
            "another-profile",
            "new-authority",
        )
    else:
        assert restored.batch["outcomes"] == {
            requests[0].run_id: "completed",
            requests[1].run_id: "interrupted",
        }
        assert (restored.profile_key, restored.epoch) == (
            session.profile_key,
            session.epoch,
        )


def test_total_envelope_limit_is_checked_before_decoding(monkeypatch):
    from tldw_chatbook.Chunking import lab_recovery

    monkeypatch.setattr(lab_recovery, "MAX_ENVELOPE_BYTES", 32)
    with pytest.raises(RecoveryImportError):
        parse_recovery(b" " * 33)


def test_export_rechecks_mutated_retained_blob():
    session = completed_session()
    export_recovery(session)
    next(iter(session.results.values()))["request"]["sample"]["text"] = (
        "changed after admission"
    )
    with pytest.raises(ValueError):
        export_recovery(session)


def test_pre_content_revision_envelope_defaults_to_zero():
    envelope = json.loads(export_recovery(new_session("profile")))
    del envelope["session"]["content_revision"]
    assert parse_recovery(json.dumps(envelope).encode()).content_revision == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("content_revision", -1),
        ("content_revision", 1),
        ("content_revision", True),
        ("revision", "0"),
    ],
)
def test_invalid_content_or_overall_revision_is_rejected(field, value):
    envelope = json.loads(export_recovery(new_session("profile")))
    envelope["session"][field] = value
    with pytest.raises(RecoveryImportError):
        parse_recovery(json.dumps(envelope).encode())


def test_matching_result_limit_counts_entire_snapshot(monkeypatch):
    from Tests.Chunking.test_lab_state import _completed
    from tldw_chatbook.Chunking import lab_recovery
    from tldw_chatbook.Chunking.lab_models import RunResult
    from tldw_chatbook.Chunking.lab_state import (
        accept_result,
        capture_batch,
        install_batch,
    )

    # A smaller resource budget exercises the same complete-snapshot accounting.
    monkeypatch.setattr(lab_recovery, "MAX_RESULT_BYTES", 8192)
    session = new_session("profile")
    request = capture_batch(session, tuple(session.candidates))[0]
    session = install_batch(session, (request,))
    value = _completed(request).model_dump(mode="json")
    value["report"]["transformed_text"] = ""
    overhead = len(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )
    value["report"]["transformed_text"] = "x" * (8192 - overhead)
    accepted = accept_result(session, RunResult.model_validate(value))
    assert len(parse_recovery(export_recovery(accepted)).results) == 1
    value["report"]["transformed_text"] += "x"
    with pytest.raises(ValueError, match="Result"):
        accept_result(session, RunResult.model_validate(value))
    assert session.results == {}
    envelope = json.loads(export_recovery(accepted))
    envelope["session"]["results"][request.run_id] = value
    envelope["digests"]["results"][request.run_id] = hashlib.sha256(
        canonical_json(value).encode()
    ).hexdigest()
    with pytest.raises(RecoveryImportError):
        parse_recovery(json.dumps(envelope).encode())


def test_empty_containers_obey_exact_envelope_depth():
    from tldw_chatbook.Chunking.lab_state import update_view

    nested = {}
    for _ in range(60):
        nested = {"x": nested}
    session = update_view(new_session("profile"), {"nested": nested})
    assert parse_recovery(export_recovery(session)).view == session.view
    with pytest.raises(ValueError, match="depth"):
        update_view(session, {"nested": {"x": nested}})


def test_string_run_revision_is_rejected_without_coercing_captured_snapshot():
    envelope = json.loads(export_recovery(completed_session()))
    run_id, result = next(iter(envelope["session"]["results"].items()))
    result["request"]["revision"] = str(result["request"]["revision"])
    envelope["session"]["batch"] = None
    envelope["digests"]["requests"] = {}
    envelope["digests"]["results"][run_id] = hashlib.sha256(
        canonical_json(result).encode()
    ).hexdigest()
    with pytest.raises(RecoveryImportError):
        parse_recovery(json.dumps(envelope).encode())


def test_exact_checkpoint_limit_matches_active_and_export_accounting():
    from tldw_chatbook.Chunking.lab_state import update_view

    session = update_view(new_session("profile"), {"padding": ""})
    small = json.loads(export_recovery(session))["session"]
    small["samples"] = {key: None for key in small["samples"]}
    overhead = len(
        json.dumps(
            small, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )
    padding = "x" * (8 * 1024 * 1024 - overhead)
    accepted = update_view(session, {"padding": padding})
    assert parse_recovery(export_recovery(accepted)).view["padding"] == padding
    with pytest.raises(ValueError, match="checkpoint"):
        update_view(accepted, {"padding": padding + "x"})


def test_blob_allowance_counts_referenced_samples_and_results(monkeypatch):
    from Tests.Chunking.test_lab_state import _completed
    from tldw_chatbook.Chunking import lab_recovery
    from tldw_chatbook.Chunking.lab_state import (
        accept_result,
        capture_batch,
        install_batch,
    )

    monkeypatch.setattr(lab_recovery, "MAX_BLOBS", 2)
    session = completed_session().model_copy(update={"undo": ()})
    assert len(parse_recovery(export_recovery(session)).results) == 1
    request = capture_batch(session, tuple(session.candidates))[0]
    installed = install_batch(session, (request,))
    with pytest.raises(ValueError, match="referenced"):
        accept_result(installed, _completed(request))
    assert installed.results == session.results


def test_too_many_chunks_are_rejected_before_active_publication():
    from Tests.Chunking.test_lab_state import _completed
    from tldw_chatbook.Chunking.lab_models import RunResult
    from tldw_chatbook.Chunking.lab_state import (
        accept_result,
        capture_batch,
        install_batch,
    )

    session = new_session("profile")
    request = capture_batch(session, tuple(session.candidates))[0]
    installed = install_batch(session, (request,))
    result = _completed(request).model_dump(mode="json")
    result["report"]["chunks"] *= 10_001
    with pytest.raises(ValueError, match="10,000"):
        accept_result(installed, RunResult.model_validate(result))
    assert installed.results == {}
