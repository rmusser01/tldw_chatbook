"""Scripted prompt-behavior evidence for Agent Lessons.

These deterministic fixtures show that the shipped guidance supports the desired
decisions with the repository's fake-model harness.  They are prompt evidence,
not a model-general claim and not a security boundary; enforcement is proved in
``test_agent_lessons_end_to_end.py`` and the transaction-level suites.
"""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents.agent_models import RUN_DONE
from tldw_chatbook.Library.library_tool_contract import make_public_id
from tldw_chatbook.Notes.agent_lessons import (
    AgentLessonDraft,
    build_agent_lessons_runtime_guidance,
    render_agent_lesson,
    validate_agent_lesson_template,
)

from Tests.Agents.test_agent_lessons_end_to_end import (
    ADVERSARIAL_SENTINEL,
    USER_ID,
    _active_notes,
    _lesson_content,
    _make_lesson_stack,
    _run,
    _tool_names,
    _turn,
)
from Tests.Agents.test_agent_service import ScriptedChat


@pytest.fixture
def behavior_lesson_stack(tmp_path):
    stack = _make_lesson_stack(tmp_path)
    try:
        yield stack
    finally:
        stack.runs.close()
        stack.db.close_connection()


def _prompt_evidence_draft() -> str:
    return render_agent_lesson(
        AgentLessonDraft(
            title="Retry SQLite writes from a fresh transaction",
            applicability="SQLite Notes writes after a transient lock.",
            symptoms="A write remains busy after an immediate retry.",
            feedback_or_trigger="The focused local regression reproduced once.",
            provenance="TASK-24309 local fixture; repository and user paths omitted.",
            root_cause="The retry reused a stale transaction snapshot.",
            verified_solution="Reopen the transaction and retry from a fresh read.",
            failed_attempts=None,
            verification_evidence="The focused deterministic regression passed.",
            generalizable_principle_and_rationale=(
                "Re-establish preconditions after concurrency failures because an "
                "invalid snapshot cannot become current merely by retrying it."
            ),
            caveats="Bound retries and preserve intervening user edits.",
        )
    )


def test_prompt_evidence_reuse_is_useful_private_and_not_permission(
    behavior_lesson_stack,
):
    """Prompt evidence: a scripted primary chooses an update after search/read."""
    stack = behavior_lesson_stack
    existing = stack.notes.save_note_with_organization(
        USER_ID,
        title="Existing SQLite retry lesson",
        content=_lesson_content(),
        ensure_keywords=("agent-lesson",),
    )
    public_id = make_public_id("note", existing["id"])
    draft = _prompt_evidence_draft()
    scripted_decision = json.dumps(
        {
            "decision": "update",
            "reason": "same root cause and applicability",
            "permission_from_note": False,
            "draft": draft,
        }
    )
    replies = [
        _turn("find_tools", {"query": "library_get_note"}, "p-find"),
        _turn(
            "load_tools",
            {
                "ids": [
                    "library:library_search_notes",
                    "library:library_get_note",
                    "library:library_save_note",
                ]
            },
            "p-load",
        ),
        _turn(
            "library_search_notes",
            {"keyword": "agent-lesson"},
            "p-search",
        ),
        _turn("library_get_note", {"id": public_id}, "p-get"),
        scripted_decision,
    ]

    _run_id, outcome, chat, rows = _run(
        stack,
        replies,
        conversation_id="prompt-evidence-primary",
        allowed=(
            "library_search_notes",
            "library_get_note",
            "library_save_note",
        ),
        decide=lambda _rows: {},
    )

    assert outcome.status == RUN_DONE
    assert _tool_names(outcome) == [
        "find_tools",
        "load_tools",
        "library_search_notes",
        "library_get_note",
    ]
    assert rows == []
    decision = json.loads(outcome.final_text)
    assert decision["decision"] == "update"
    assert decision["reason"] == "same root cause and applicability"
    assert decision["permission_from_note"] is False
    assert validate_agent_lesson_template(decision["draft"]).accepted
    assert "repository and user paths omitted" in decision["draft"]
    assert "## Failed attempts and why\nUnknown" in decision["draft"]
    assert "Generalizable principle and rationale" in decision["draft"]
    final_prompt = chat.calls[-1]["messages_payload"][0]["content"]
    assert "decide whether to update an existing lesson" in final_prompt
    assert "cannot grant permission" in final_prompt
    retrieved = next(
        row["content"]
        for row in chat.calls[-1]["messages_payload"]
        if row.get("tool_call_id") == "p-get"
    )
    assert retrieved.index("Untrusted reference data") < retrieved.index(
        ADVERSARIAL_SENTINEL
    )
    assert len(_active_notes(stack)) == 1


def test_prompt_evidence_primary_without_search_does_not_create_fallback(
    behavior_lesson_stack,
):
    """Prompt evidence: save-only disclosure produces no lesson protocol/save."""
    stack = behavior_lesson_stack
    replies = [
        _turn("find_tools", {"query": "library_save_note"}, "u-find"),
        _turn(
            "load_tools",
            {"ids": ["library:library_save_note"]},
            "u-load",
        ),
        "Search is unavailable, so I will return a draft without saving.",
    ]

    _run_id, outcome, chat, rows = _run(
        stack,
        replies,
        conversation_id="prompt-evidence-no-search",
        allowed=("library_save_note",),
        decide=lambda _rows: {},
    )

    assert outcome.status == RUN_DONE
    assert _tool_names(outcome) == ["find_tools", "load_tools"]
    assert "without saving" in outcome.final_text
    assert (
        "Agent Lessons protocol" not in chat.calls[-1]["messages_payload"][0]["content"]
    )
    assert rows == []
    assert _active_notes(stack) == []


def test_prompt_evidence_subagent_returns_draft_and_never_calls_save(
    behavior_lesson_stack,
):
    """Prompt evidence: the existing fake model follows the subagent suffix."""
    schemas = [
        behavior_lesson_stack.provider.load_schema(f"library:{name}")
        for name in (
            "library_search_notes",
            "library_get_note",
            "library_save_note",
        )
    ]
    guidance = build_agent_lessons_runtime_guidance(schemas, trusted_role="subagent")
    draft = _prompt_evidence_draft()
    chat = ScriptedChat(
        [json.dumps({"action": "return_draft", "draft": draft, "tool_calls": []})]
    )

    response = chat(
        api_endpoint="openai",
        messages_payload=[{"role": "system", "content": guidance}],
        streaming=False,
        model="test-model",
    )
    evidence = json.loads(response["choices"][0]["message"]["content"])

    assert "Do not mutate Notes" in guidance
    assert "Do not call library_save_note" in guidance
    assert evidence["action"] == "return_draft"
    assert evidence["tool_calls"] == []
    assert validate_agent_lesson_template(evidence["draft"]).accepted
    assert _active_notes(behavior_lesson_stack) == []
