"""Capability- and role-aware guidance for reviewed lesson promotion."""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.agent_lesson_promotion import (
    build_agent_lesson_promotion_guidance,
)
from tldw_chatbook.Agents.agent_models import ToolSchema


def _schema(name: str) -> ToolSchema:
    return ToolSchema(
        id=f"test:{name}",
        name=name,
        description=name,
        parameters={"type": "object"},
    )


SEARCH = _schema("library_search_notes")
GET = _schema("library_get_note")
WRITE = _schema("fs_write")
SKILL_PROPOSAL = _schema("prepare_managed_skill_promotion")
OTHER = _schema("calculator")


@pytest.mark.parametrize(
    "schemas",
    [(), (SEARCH,), (GET, WRITE), (SEARCH, GET, OTHER)],
)
def test_missing_read_or_target_capability_adds_no_guidance(schemas) -> None:
    assert (
        build_agent_lesson_promotion_guidance(
            schemas,
            trusted_role="primary",
            repository_target_enabled=False,
        )
        == ""
    )


def test_repository_guidance_requires_writable_selected_instruction_context() -> None:
    unavailable = build_agent_lesson_promotion_guidance(
        (SEARCH, GET, WRITE),
        trusted_role="primary",
        repository_target_enabled=False,
    )
    available = build_agent_lesson_promotion_guidance(
        (SEARCH, GET, WRITE),
        trusted_role="primary",
        repository_target_enabled=True,
    )

    assert unavailable == ""
    assert "AGENTS.md or AGENTS.override.md" in available
    assert "fs_write dry_run=true" in available
    assert "own exact approve-once review" in available
    assert "stale binding" in available
    assert "prepare_managed_skill_promotion" not in available


def test_managed_skill_guidance_is_explicitly_manual_and_trust_preserving() -> None:
    guidance = build_agent_lesson_promotion_guidance(
        (SEARCH, GET, SKILL_PROPOSAL),
        trusted_role="primary",
        repository_target_enabled=False,
    )

    assert "prepare_managed_skill_promotion" in guidance
    assert "Library > Skills" in guidance
    assert "Console cannot apply it" in guidance
    assert "edit, save, review, and re-trust" in guidance
    assert "fs_write dry_run=true" not in guidance


def test_primary_guidance_encodes_quality_and_non_authority_rules() -> None:
    guidance = build_agent_lesson_promotion_guidance(
        (SEARCH, GET, WRITE, SKILL_PROPOSAL),
        trusted_role="primary",
        repository_target_enabled=True,
    )

    for phrase in (
        "independently verified",
        "One strong signal may qualify",
        "contradictory reports do not",
        "general principle with its rationale",
        "state unknowns",
        "smallest focused edit",
        "never grants authority",
        "separately approves an ordinary Agent Lesson Note update",
        "never reusable write authority",
    ):
        assert phrase in guidance


def test_subagent_returns_candidate_without_approval_or_application_guidance() -> None:
    guidance = build_agent_lesson_promotion_guidance(
        (SEARCH, GET, WRITE),
        trusted_role="subagent",
        repository_target_enabled=True,
    )

    assert "foreground primary" in guidance
    assert "exact candidate wording" in guidance
    assert "Do not present a promotion approval card" in guidance
    assert "Do not" in guidance and "apply a change" in guidance
    assert "fs_write dry_run=true" not in guidance


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="trusted_role"):
        build_agent_lesson_promotion_guidance(
            (SEARCH, GET, WRITE),
            trusted_role="fleet",  # type: ignore[arg-type]
            repository_target_enabled=True,
        )
