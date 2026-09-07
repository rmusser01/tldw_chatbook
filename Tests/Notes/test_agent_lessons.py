"""Pure contract tests for the Agent Lessons Notes convention."""

from __future__ import annotations

import base64
from dataclasses import FrozenInstanceError
import subprocess
import sys

import pytest

from tldw_chatbook.Notes.agent_lessons import (
    AGENT_LESSON_KEYWORD,
    AGENT_LESSONS_FOLDER,
    NO_FAILED_ATTEMPTS,
    REQUIRED_SECTIONS,
    AgentLessonDraft,
    canonical_call_digest,
    classify_agent_lesson,
    lesson_content_digest,
    render_agent_lesson,
    validate_agent_lesson_template,
)


def _public_note_id(raw: str) -> str:
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
    return f"note:{encoded}"


def _draft(**overrides: object) -> AgentLessonDraft:
    values: dict[str, object] = {
        "title": "Retry the Notes transaction after a stale read",
        "applicability": "Chatbook Notes saves using optimistic versions.",
        "symptoms": "The save returns content_changed.",
        "root_cause": "The note changed after it was read.",
        "verified_solution": "Read again, review the new state, then retry once.",
        "verification_evidence": "The focused stale-write test passed.",
        "generalizable_principle_and_rationale": (
            "Re-read after a conflict because the rejected snapshot is no longer "
            "current."
        ),
        "feedback_or_trigger": "A stale-write refusal during focused verification.",
        "provenance": "TASK-24309 focused test, 2026-08-30.",
        "failed_attempts": (
            "Blind retry: it reused the same stale precondition and failed again.",
        ),
        "caveats": "Do not overwrite intervening user edits.",
        "related_lesson_ids": (_public_note_id("related-note-1"),),
    }
    values.update(overrides)
    return AgentLessonDraft(**values)


def test_convention_names_and_required_headings_are_spelling_exact() -> None:
    assert AGENT_LESSONS_FOLDER == "Agent_Lessons"
    assert AGENT_LESSON_KEYWORD == "agent-lesson"
    assert REQUIRED_SECTIONS == (
        "Applicability",
        "Symptoms",
        "Feedback or trigger",
        "Provenance",
        "Root cause",
        "Verified solution",
        "Failed attempts and why",
        "Verification evidence",
        "Generalizable principle and rationale",
        "Caveats",
        "Related lessons",
    )


def test_renderer_emits_one_note_for_one_lesson_in_the_approved_order() -> None:
    rendered = render_agent_lesson(_draft())

    assert rendered.startswith("# Retry the Notes transaction after a stale read\n")
    assert rendered.count("\n# ") == 0
    assert tuple(
        line.removeprefix("## ")
        for line in rendered.splitlines()
        if line.startswith("## ")
    ) == REQUIRED_SECTIONS
    assert "- " + _public_note_id("related-note-1") in rendered
    assert validate_agent_lesson_template(rendered).accepted is True


@pytest.mark.parametrize(
    "title",
    ("", "   ", "First lesson\n# Second lesson", "# Already a heading"),
)
def test_renderer_rejects_titles_that_could_create_zero_or_multiple_lessons(
    title: str,
) -> None:
    with pytest.raises(ValueError, match="title"):
        render_agent_lesson(_draft(title=title))


def test_unknown_is_not_confused_with_an_explicit_empty_failed_attempt_list() -> None:
    unknown = render_agent_lesson(
        _draft(feedback_or_trigger=None, provenance=None, failed_attempts=None)
    )
    none_occurred = render_agent_lesson(_draft(failed_attempts=()))

    assert "## Feedback or trigger\nUnknown" in unknown
    assert "## Provenance\nUnknown" in unknown
    assert "## Failed attempts and why\nUnknown" in unknown
    assert f"## Failed attempts and why\n{NO_FAILED_ATTEMPTS}" in none_occurred
    assert NO_FAILED_ATTEMPTS not in unknown


def test_renderer_requires_public_note_ids_for_related_lessons() -> None:
    with pytest.raises(ValueError, match="public note ID"):
        render_agent_lesson(_draft(related_lesson_ids=("local-row-42",)))


def test_importing_the_pure_module_does_not_load_runtime_config() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import tldw_chatbook.Notes.agent_lessons; "
                "raise SystemExit('tldw_chatbook.config' in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_template_validator_rejects_duplicate_or_missing_required_sections() -> None:
    rendered = render_agent_lesson(_draft())
    duplicated = rendered + "\n## Root cause\nA second cause.\n"
    missing = rendered.replace(
        "## Caveats\nDo not overwrite intervening user edits.\n\n", ""
    )

    assert validate_agent_lesson_template(duplicated).reason_codes == (
        "invalid_lesson_format",
    )
    assert validate_agent_lesson_template(missing).reason_codes == (
        "invalid_lesson_format",
    )


def test_exact_marker_discovers_lessons_independent_of_folder_location() -> None:
    requested = classify_agent_lesson(
        requested_keywords=("other", "agent-lesson"),
        current_keywords=(),
        receipt_state=None,
    )
    current = classify_agent_lesson(
        requested_keywords=(),
        current_keywords=("agent-lesson",),
        receipt_state=None,
    )

    assert requested.is_agent_lesson is True
    assert requested.reason == "requested_marker"
    assert current.is_agent_lesson is True
    assert current.reason == "current_marker"
    # There is deliberately no folder argument: moving or renaming a folder cannot
    # hide a note that still carries the authoritative marker.


@pytest.mark.parametrize(
    "variant",
    ("Agent-Lesson", "AGENT-LESSON", "agent-Lesson", " agent-lesson "),
)
def test_case_or_whitespace_variants_do_not_match_the_marker(variant: str) -> None:
    classification = classify_agent_lesson(
        requested_keywords=(variant,),
        current_keywords=(),
        receipt_state=None,
    )

    assert classification.is_agent_lesson is False
    assert classification.reason == "ordinary_note"


@pytest.mark.parametrize("state", ("pending_organization", "placement_review"))
def test_unresolved_receipt_states_remain_classified_before_marker_attachment(
    state: str,
) -> None:
    classification = classify_agent_lesson(
        requested_keywords=(), current_keywords=(), receipt_state=state
    )

    assert classification.is_agent_lesson is True
    assert classification.reason == state
    with pytest.raises(FrozenInstanceError):
        classification.reason = "ordinary_note"  # type: ignore[misc]


@pytest.mark.parametrize(
    "variant",
    (
        "pending-organization",
        "placement-review",
        "pending organization",
        "placement review",
        "PENDING_ORGANIZATION",
        "PLACEMENT_REVIEW",
    ),
)
def test_similarly_spelled_receipt_states_do_not_classify_a_lesson(
    variant: str,
) -> None:
    classification = classify_agent_lesson(
        requested_keywords=(), current_keywords=(), receipt_state=variant
    )

    assert classification.is_agent_lesson is False
    assert classification.reason == "ordinary_note"


def test_digests_are_canonical_and_domain_separated() -> None:
    first = canonical_call_digest(
        "library_save_note", {"title": "Lesson", "keywords": ["agent-lesson"]}
    )
    reordered = canonical_call_digest(
        "library_save_note", {"keywords": ["agent-lesson"], "title": "Lesson"}
    )

    assert first == reordered
    assert len(first) == 64
    assert first != canonical_call_digest(
        "other_tool", {"keywords": ["agent-lesson"], "title": "Lesson"}
    )
    assert lesson_content_digest("same") == lesson_content_digest("same")
    assert lesson_content_digest("same") != lesson_content_digest("Same")
