"""TASK-34100.13 (fulltrack-05): the Full track has no Notes step.

The step had no control and committed nothing; lasting Notes sync is set up
from Library ▸ Notes ▸ Add from files…, which the Summary now names. Drafts
saved by the eleven-step Full track (at the Notes step, or with the old RAG
and splash-card fields) still resume instead of being silently dropped.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state


def test_full_track_has_no_notes_step() -> None:
    full = wizard_state.active_step_ids(wizard_state.TRACK_FULL, key_entered=False)
    assert "notes" not in full
    assert len(full) == 10
    assert full[-1] == wizard_state.STEP_SUMMARY


def test_the_full_track_is_the_same_whatever_happens_mid_run() -> None:
    assert wizard_state.active_step_ids(
        wizard_state.TRACK_FULL, key_entered=False
    ) == wizard_state.active_step_ids(wizard_state.TRACK_FULL, key_entered=True)


def test_wizard_no_longer_owns_the_notes_section() -> None:
    assert "notes" not in wizard_state.WIZARD_OWNED_SECTIONS
    assert wizard_state.commit_sections_allowed({"notes": {"x": 1}}) is False


def test_summary_notes_row_points_at_add_from_files() -> None:
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows({}, {}, rag_deps_installed=True)
    }
    row = rows["Notes folder sync"]
    assert row.state == wizard_state.ROW_DEFAULT
    assert row.detail == "not set up · Library ▸ Notes ▸ Add from files…"


def _draft_config(active_step: str, values: dict) -> dict:
    return {
        "first_run": {
            "draft_version": wizard_state.SETUP_DRAFT_VERSION,
            "draft_track": wizard_state.TRACK_FULL,
            "active_step_id": active_step,
            "draft_values": values,
            "resume_attempted": False,
        }
    }


def test_a_draft_saved_at_the_retired_notes_step_resumes_at_the_next_step() -> None:
    draft = wizard_state.read_setup_draft(
        _draft_config("notes", {"welcome": {"track": "full"}, "notes": {}})
    )
    assert draft is not None
    assert draft.active_step_id == wizard_state.STEP_APPEARANCE
    assert "notes" not in draft.values


@pytest.mark.parametrize(
    ("step_id", "retired_field"),
    [("rag", "embedding_model"), ("appearance", "splash_card")],
)
def test_a_draft_with_a_retired_field_keeps_its_other_values(
    step_id: str, retired_field: str
) -> None:
    values = {
        "welcome": {"track": "full"},
        step_id: {retired_field: "something"},
        "model": {"model_id": "gpt-4.1-mini"},
    }
    draft = wizard_state.read_setup_draft(_draft_config("tools", values))
    assert draft is not None
    assert draft.values["model"] == {"model_id": "gpt-4.1-mini"}
    assert retired_field not in draft.values.get(step_id, {})
