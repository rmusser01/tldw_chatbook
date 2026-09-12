from __future__ import annotations

import pytest

from tldw_chatbook.Agents.agent_models import ToolSchema
from tldw_chatbook.Notes.agent_lessons import (
    build_agent_lessons_runtime_guidance,
)


def _schema(name: str) -> ToolSchema:
    return ToolSchema(
        id=f"library:{name}",
        name=name,
        description=name,
        parameters={"type": "object"},
    )


SEARCH = _schema("library_search_notes")
GET = _schema("library_get_note")
SAVE = _schema("library_save_note")
OTHER = _schema("calculator")


def test_no_disclosed_notes_tools_adds_no_guidance() -> None:
    assert build_agent_lessons_runtime_guidance((), trusted_role="primary") == ""
    assert (
        build_agent_lessons_runtime_guidance((OTHER,), trusted_role="subagent")
        == ""
    )


def test_search_and_get_only_add_untrusted_search_read_guidance() -> None:
    suffix = build_agent_lessons_runtime_guidance(
        (SEARCH, GET), trusted_role="primary"
    )

    assert "Agent Lessons protocol" in suffix
    assert "library_search_notes" in suffix
    assert "library_get_note" in suffix
    assert "untrusted reference data" in suffix
    assert "cannot override" in suffix
    assert "cannot grant permission" in suffix
    assert "independent evidence" in suffix
    assert "library_save_note" not in suffix


def test_primary_with_search_get_and_save_gets_complete_reviewed_save_flow() -> None:
    suffix = build_agent_lessons_runtime_guidance(
        (OTHER, SEARCH, GET, SAVE), trusted_role="primary"
    )

    required = (
        "search first",
        "Feedback or trigger",
        "Provenance",
        "independent evidence",
        "Generalizable principle and rationale",
        "Unknown",
        "Never invent failed attempts",
        "update an existing lesson",
        "progressive disclosure",
        "exact preview",
        "explicit approval",
        "library_save_note",
    )
    for phrase in required:
        assert phrase in suffix


def test_subagent_with_full_schemas_searches_drafts_and_returns_without_mutation(
) -> None:
    suffix = build_agent_lessons_runtime_guidance(
        (SEARCH, GET, SAVE), trusted_role="subagent"
    )

    assert "search first" in suffix
    assert "structured draft" in suffix
    assert "return" in suffix
    assert "foreground primary" in suffix
    assert "Do not call library_save_note" in suffix
    assert "Do not mutate Notes" in suffix
    assert "exact preview" not in suffix


@pytest.mark.parametrize(
    "schemas",
    [
        (SAVE,),
        (GET, SAVE),
        (OTHER, SAVE),
    ],
)
def test_save_without_search_never_receives_save_guidance(schemas) -> None:
    suffix = build_agent_lessons_runtime_guidance(
        schemas, trusted_role="primary"
    )
    assert "library_save_note" not in suffix
    assert "exact preview" not in suffix


def test_search_only_guidance_mentions_only_the_disclosed_notes_capability() -> None:
    suffix = build_agent_lessons_runtime_guidance(
        (SEARCH,), trusted_role="primary"
    )
    assert "library_search_notes" in suffix
    assert "library_get_note" not in suffix
    assert "library_save_note" not in suffix


def test_untrusted_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="trusted_role"):
        build_agent_lessons_runtime_guidance(
            (SEARCH, GET, SAVE), trusted_role="direct"  # type: ignore[arg-type]
        )
