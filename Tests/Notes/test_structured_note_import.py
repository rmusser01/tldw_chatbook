"""Explicit source imports retain source and bind language to approved effects."""

import hashlib
import json
from dataclasses import replace

import pytest

from Tests.Notes.test_note_import_executor import (
    _approved_execution_plan,
    _execution_bounds,
    _execution_item,
    _expected_note_id,
)
from tldw_chatbook.Notes.note_import_discovery import (
    ImportSelectionError,
    discover_import_sources,
)
from tldw_chatbook.Notes.note_import_execution_models import (
    _private_payload_fingerprint as execution_fingerprint,
)
from tldw_chatbook.Notes.note_import_parsers import parse_import_sources
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ParsedNotePayload,
    ProposedFolderMembership,
)
from tldw_chatbook.Notes.note_import_planner import _private_payload_fingerprint
from tldw_chatbook.Notes.note_import_receipts import ImportReceiptConflictError

pytest_plugins = ["Tests.Notes.test_note_import_executor"]


@pytest.mark.parametrize(
    ("extension", "language"),
    [
        ("json", "json"),
        ("yaml", "yaml"),
        ("yml", "yaml"),
        ("jsonl", "jsonl"),
        ("txt", "text"),
        ("md", "markdown"),
    ],
)
def test_explicit_source_import_keeps_exact_representation(
    tmp_path, extension, language
):
    source = '\ufeff  {"title":"This is source","x":9007199254740993}\r\n  '
    path = tmp_path / f"Data.{extension}"
    path.write_bytes(source.encode("utf-8"))
    bounds = _execution_bounds()
    batch = parse_import_sources(
        discover_import_sources([path], bounds),
        bounds,
        destination_folder_segments=("Imported",),
        source_mode=True,
    )
    assert batch.issues == ()
    payload = batch.parsed[0].payloads[0]
    assert (payload.title, payload.content, payload.content_language) == (
        "Data",
        source,
        language,
    )


def test_legacy_container_path_remains_available(tmp_path):
    path = tmp_path / "notes.json"
    path.write_text(json.dumps([{"title": "Old note", "content": "body"}]))
    bounds = _execution_bounds()
    batch = parse_import_sources(
        discover_import_sources([path], bounds),
        bounds,
        destination_folder_segments=("Imported",),
    )
    assert (batch.parsed[0].payloads[0].title, batch.parsed[0].payloads[0].content) == (
        "Old note",
        "body",
    )


def test_language_changes_approval_and_effect_identity(real_executor):
    executor, _, receipts, _, _, db = real_executor
    payload = ParsedNotePayload(
        title="Data", content='{"x":9007199254740993}\r\n', content_language="json"
    )
    item = _execution_item(
        item_id="source",
        payloads=(payload,),
        action=ImportAction.CREATE_NEW,
        memberships=(ProposedFolderMembership(0, ("Imported",)),),
        add_membership=True,
    )
    approved = _approved_execution_plan(item, proposed_folder_paths=(("Imported",),))
    receipt = executor.execute(approved)
    assert receipt.imported == 1
    note = db.get_note_by_id(_expected_note_id("source", 0))
    assert (note["content"], note["content_language"]) == (payload.content, "json")
    assert executor.execute(approved).imported == 1
    changed = replace(payload, content_language="future/mode")
    assert _private_payload_fingerprint((payload,)) != _private_payload_fingerprint(
        (changed,)
    )
    assert execution_fingerprint((payload,)) != execution_fingerprint((changed,))
    with pytest.raises(ImportReceiptConflictError):
        receipts.begin(
            _approved_execution_plan(
                replace(item, payloads=(changed,)),
                proposed_folder_paths=(("Imported",),),
            ),
            batch_size=25,
        )


def test_local_update_preserves_omitted_language_and_resets_explicit_null(
    target_harness,
):
    target, _, _, db = target_harness
    payload = ParsedNotePayload(title="Data", content="source", content_language="yaml")
    target.create_note(note_id="portable-note", payload=payload)
    legacy = ParsedNotePayload(title="Data", content="legacy body")
    target.replace_note(note_id="portable-note", expected_version=1, payload=legacy)
    assert db.get_note_by_id("portable-note")["content_language"] == "yaml"
    target.replace_note(
        note_id="portable-note",
        expected_version=2,
        payload=ParsedNotePayload(title="Data", content="reset", content_language=None),
    )
    assert db.get_note_by_id("portable-note")["content_language"] is None


def test_omitted_language_keeps_historical_fingerprint_bytes():
    payload = ParsedNotePayload(title="Legacy", content="Body", keywords=("tag",))
    legacy = {
        "payloads": [
            {
                "type": "parsed_note_payload",
                "content": "Body",
                "keywords": ["tag"],
                "template_name": None,
                "title": "Legacy",
            }
        ],
        "type": "tldw_note_import_payload_set",
        "version": 1,
    }
    digest = hashlib.sha256(
        json.dumps(
            legacy, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    assert (
        _private_payload_fingerprint((payload,))
        == execution_fingerprint((payload,))
        == digest
    )
    assert (
        _private_payload_fingerprint((replace(payload, content_language=None),))
        != digest
    )


def test_portable_container_preserves_title_body_and_opaque_mode(tmp_path):
    path = tmp_path / "portable.json"
    source = "\ufeff  <literal>9007199254740993\r\n  "
    path.write_text(
        json.dumps(
            {
                "format_version": "tldw.note.v1",
                "title": "  Data  ",
                "content": source,
                "content_language": " future/😀 ",
            }
        )
    )
    bounds = _execution_bounds()
    batch = parse_import_sources(
        discover_import_sources([path], bounds),
        bounds,
        destination_folder_segments=("Imported",),
    )
    payload = batch.parsed[0].payloads[0]
    assert (payload.title, payload.content, payload.content_language) == (
        "  Data  ",
        source,
        " future/😀 ",
    )


def test_unsupported_portable_contract_refuses_selected_batch(tmp_path):
    old = tmp_path / "legacy.json"
    old.write_text(json.dumps({"title": "Legacy", "content": "body"}))
    future = tmp_path / "future.yaml"
    future.write_text(
        "format_version: tldw.note.v99\ntitle: Future\ncontent: source\ncontent_language: yaml"
    )
    bounds = _execution_bounds()
    with pytest.raises(ImportSelectionError):
        parse_import_sources(
            discover_import_sources([old, future], bounds),
            bounds,
            destination_folder_segments=("Imported",),
        )
