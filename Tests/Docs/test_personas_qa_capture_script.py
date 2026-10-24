"""Smoke tests for the Personas workbench QA capture script (TASK-33910.33).

The end-to-end run (every step's SVG) is exercised by hand; these pin the two
seams that broke silently: that the script seeds a real ChaChaNotes DB the
paged loader reads, and that its outputs stay out of tracked paths unless a
run names an approval set.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "Docs/superpowers/qa/personas-workbench/capture_personas_workbench.py"
)


@pytest.fixture(scope="module")
def capture():
    spec = importlib.util.spec_from_file_location("capture_personas_workbench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seeding_gives_the_paged_loader_rows_and_a_clickable_conversation(
    capture, tmp_path
):
    db = CharactersRAGDB(tmp_path / "qa.db", "qa-capture-test")
    try:
        sam_id, first_conversation = capture.seed_character_db(db)

        names = {c["name"] for c in db.list_character_cards()}
        assert {"Detective Sam", "Lab Assistant"} <= names
        assert db.get_character_card_by_id(sam_id)["name"] == "Detective Sam"
        listed = [c["id"] for c in db.get_conversations_for_character(sam_id)]
        assert first_conversation == listed[0]
        assert {c["id"] for c in capture.CONVERSATIONS} == set(listed)
        messages = db.get_messages_for_conversation(first_conversation)
        assert len(messages) == 2 * len(capture.MESSAGES)
    finally:
        db.close_connection()


def test_default_output_is_scratch_and_approval_set_is_the_only_tracked_target(
    capture, tmp_path
):
    default = capture._resolve_out(capture._parse_args([]), tmp_path)
    assert default.is_relative_to(tmp_path)
    assert not default.is_relative_to(capture.REPO_ROOT)

    approved = capture._resolve_out(
        capture._parse_args(["--approval-set", "zz-test-approval"]), tmp_path
    )
    try:
        assert approved == capture.QA_DIR / "zz-test-approval"
    finally:
        approved.rmdir()

    with pytest.raises(SystemExit):
        capture._resolve_out(
            capture._parse_args(["--approval-set", "../escape"]), tmp_path
        )
