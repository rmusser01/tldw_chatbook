"""Smoke tests for the Personas workbench QA capture script (TASK-33910.33).

The end-to-end run (every step's SVG) is exercised by hand; these pin the two
seams that broke silently: that the script seeds a real ChaChaNotes DB the
paged loader reads, and that its outputs stay out of tracked paths unless a
run names an approval set.

Lives in Tests/Docs (not Tests/UI): it uses no UI, and a Tests/UI file that
lands first on an xdist worker trips the app-import `RecoveryRequired` bootstrap
(Tests/UI/conftest.py) -- seen in review of this file's first version.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tldw_chatbook.Character_Chat.Character_Chat_Lib import get_character_page_for_ui
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


def test_seeding_is_visible_through_the_paged_loader_and_has_a_clickable_conversation(
    capture, tmp_path
):
    db = CharactersRAGDB(tmp_path / "qa.db", "qa-capture-test")
    try:
        sam_id, first_conversation = capture.seed_character_db(db)

        names = {c["name"] for c in db.list_character_cards()}
        assert {"Detective Sam", "Lab Assistant"} <= names
        assert db.get_character_card_by_id(sam_id)["name"] == "Detective Sam"
        # The read the screen's DB-paged loader uses (personas_screen).
        page = get_character_page_for_ui(db, limit=50, offset=0)
        assert {"Detective Sam", "Lab Assistant"} <= {r["name"] for r in page}
        assert sam_id in {r["id"] for r in page}
        listed = [c["id"] for c in db.get_conversations_for_character(sam_id)]
        assert first_conversation == listed[0]
        assert {c["id"] for c in capture.CONVERSATIONS} == set(listed)
        messages = db.get_messages_for_conversation(first_conversation)
        assert len(messages) == 2 * len(capture.MESSAGES)
    finally:
        db.close_connection()


def test_default_output_is_scratch_and_approval_set_is_the_only_tracked_target(
    capture, tmp_path, monkeypatch
):
    default = capture._resolve_out(capture._parse_args([]), tmp_path)
    assert default.is_relative_to(tmp_path)
    assert not default.is_relative_to(capture.REPO_ROOT)

    monkeypatch.setattr(capture, "QA_DIR", tmp_path / "qa")
    approved = capture._resolve_out(
        capture._parse_args(["--approval-set", "zz-test-approval"]), tmp_path
    )
    assert approved == tmp_path / "qa" / "zz-test-approval"
    assert approved.is_dir()

    with pytest.raises(SystemExit):
        capture._resolve_out(
            capture._parse_args(["--approval-set", "../escape"]), tmp_path
        )


def test_relative_out_resolves_against_the_callers_cwd_not_the_repo(
    capture, tmp_path, monkeypatch
):
    """run() chdirs to the repo root; --out must be resolved before that."""
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    seen = {}

    async def fake_main(out, scratch):
        seen["out"] = out

    monkeypatch.setattr(capture, "main", fake_main)
    monkeypatch.setattr(capture, "_isolate_config", lambda scratch: None)
    monkeypatch.setattr(capture.sys, "path", list(capture.sys.path))
    capture.run(["--out", "rel-out"])
    assert seen["out"] == caller / "rel-out"
