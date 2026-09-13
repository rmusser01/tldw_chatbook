"""Deterministic end-to-end proof for reviewed lesson promotion boundaries."""

from __future__ import annotations

import json
import time

from tldw_chatbook.Agents.agent_models import RUN_DONE
from tldw_chatbook.Agents.project_instruction_resolver import (
    ProjectInstructionResolver,
)
from tldw_chatbook.Agents.run_context import (
    CurrentRunActor,
    use_run_actor,
    use_tool_call_id,
)
from tldw_chatbook.Library.library_tool_contract import make_public_id

from Tests.Agents.test_agent_lessons_end_to_end import (
    USER_ID,
    _active_notes,
    _lesson_content,
    _make_lesson_stack,
    _run,
    _turn,
)
from Tests.Chat.test_console_agent_lesson_promotion_approval import (
    RUN,
    _LiveInstructionContext,
    _prepare_args,
    _provider,
    _review_and_invoke,
)


def test_repository_apply_is_exact_and_outcome_note_needs_separate_approval(
    tmp_path,
) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text("# Existing instruction\n", encoding="utf-8")
    context = _LiveInstructionContext(tmp_path)
    provider = _provider(tmp_path, context)
    notes_state = tmp_path / "notes-state"
    notes_state.mkdir()
    lesson_stack = _make_lesson_stack(notes_state)
    original_note_content = _lesson_content()
    original_title = "Preserve concurrent edits"
    original_lesson = lesson_stack.notes.save_note_with_organization(
        USER_ID,
        title=original_title,
        content=original_note_content,
        ensure_keywords=("agent-lesson",),
    )
    def approve(rows):
        return {row.call_id: "approve_once" for row in rows}

    try:
        _, prepared = _review_and_invoke(
            provider,
            _prepare_args("# Existing instruction\n\nCheck current state before write.\n"),
            "prepare-e2e",
            approve,
        )
        proposal = json.loads(prepared.content)
        assert target.read_text(encoding="utf-8") == "# Existing instruction\n"
        assert _active_notes(lesson_stack)[0]["content"] == original_note_content

        apply_args = {
            "path": proposal["target_path"],
            "content": proposal["replacement_content"],
            "expected_sha256": proposal["expected_sha256"],
            "proposal_digest": proposal["proposal_digest"],
        }
        _, applied = _review_and_invoke(
            provider, apply_args, "apply-e2e", approve
        )

        assert applied.ok
        assert target.read_text(encoding="utf-8") == proposal["replacement_content"]
        assert _active_notes(lesson_stack)[0]["content"] == original_note_content
        later = ProjectInstructionResolver().resolve_startup(
            binding_id="binding-1",
            binding_root=tmp_path,
            locator_fingerprint="fingerprint-1",
            max_bytes=32_768,
            dispatch_started_wall_ns=time.time_ns(),
        )
        assert later.source is not None
        assert later.source.body == proposal["replacement_content"]

        with use_run_actor(CurrentRunActor("primary", RUN, None)):
            with use_tool_call_id("replay-e2e"):
                replay = provider.invoke("fs_write", apply_args)
        assert not replay.ok

        outcome_content = original_note_content.replace(
            "Bound retries and preserve intervening user edits.",
            "Bound retries and preserve intervening user edits. "
            "Promotion outcome: Applied after exact review and verification.",
        )
        public_id = make_public_id("note", original_lesson["id"])
        save_args = {
            "title": original_title,
            "content": outcome_content,
            "note_id": public_id,
            "expected_version": original_lesson["version"],
            "expected_organization_version": original_lesson[
                "organization_version"
            ],
        }
        replies = [
            _turn("find_tools", {"query": "library_save_note"}, "o-find"),
            _turn(
                "load_tools",
                {"ids": ["library:library_save_note"]},
                "o-load",
            ),
            _turn("library_save_note", save_args, "o-save"),
            "Recorded the separately reviewed outcome.",
        ]
        _run_id, outcome, _chat, approval_rows = _run(
            lesson_stack,
            replies,
            conversation_id="promotion-outcome",
            allowed=(
                "library_search_notes",
                "library_get_note",
                "library_save_note",
            ),
            decide=approve,
        )

        assert outcome.status == RUN_DONE
        assert [row.call_id for row in approval_rows] == ["o-save"]
        active_content = _active_notes(lesson_stack)[0]["content"]
        assert "Promotion outcome: Applied" in active_content, [
            (step.kind, step.tool_name, step.result, step.tool_outcome)
            for step in outcome.steps
        ]
    finally:
        lesson_stack.runs.close()
        lesson_stack.db.close_connection()
