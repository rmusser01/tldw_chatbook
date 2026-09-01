"""Tests for the `recurring_question` execution core (schedules-handoff PR-2,
task 3): `resolve_execution_target`'s precedence, and the six-row
`classify_rag_response`-parity ladder in `execute_recurring_question`.

`run_library_rag_search`/`generate_library_rag_answer` are the only faked
seams (monkeypatched as `automation_execution` module attributes, per the
plan's Step 1) -- everything else (scope normalization, finding-policy ->
top_k, classification) runs for real.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from tldw_chatbook.Library.library_rag_answer_service import (
    ANSWER_STATUS_ABSTAINED,
    ANSWER_STATUS_FAILED,
    ANSWER_STATUS_NO_EVIDENCE,
    ANSWER_STATUS_READY,
    LibraryRagAnswer,
)
from tldw_chatbook.Library.library_rag_service import LibraryRagSearchOutcome
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow
from tldw_chatbook.Scheduling import automation_execution
from tldw_chatbook.Scheduling.automation_execution import (
    RESULT_SUMMARY_MAX_CHARS,
    ExecutionOutcome,
    execute_recurring_question,
    resolve_execution_target,
)

pytestmark = pytest.mark.unit


def _row(source_type: str = "media", score: float | None = 0.9, result_id: str = "r1") -> LibraryRagResultRow:
    return LibraryRagResultRow(
        result_id=result_id,
        title=f"Title {result_id}",
        snippet="snippet",
        score=score,
        source_id="src-1",
        chunk_id="chunk-1",
        citations=(),
        provenance=MappingProxyType({"source_type": source_type}),
    )


def _definition_row(**overrides: Any) -> dict:
    row: dict[str, Any] = {
        "input": {"question": "What changed?"},
        "config": {},
        "finding_policy": {"preset": "balanced_findings"},
    }
    row.update(overrides)
    return row


class _FakeApp:
    """Stand-in `app` -- `run_library_rag_search` is monkeypatched, so this
    never actually reads `library_rag_search_service`; it exists only
    because `execute_recurring_question`'s signature takes an `app`.
    """


# --- resolve_execution_target: precedence -----------------------------------


def test_resolve_execution_target_definition_wins():
    row = _definition_row(input={"question": "q", "provider": "anthropic", "model": "opus", "max_tokens": 500})
    target = resolve_execution_target(row)
    assert target == {"provider": "anthropic", "model": "opus", "max_tokens": 500}


def test_resolve_execution_target_falls_through_blank_definition_to_config(monkeypatch):
    row = _definition_row(input={"question": "q", "provider": "  ", "model": None, "max_tokens": 0})

    def fake_get_cli_setting(section, key, default=None):
        assert section == "scheduling"
        return {
            "executor_provider": "openai",
            "executor_model": "gpt-5",
            "executor_max_tokens": 700,
        }.get(key, default)

    monkeypatch.setattr(automation_execution, "get_cli_setting", fake_get_cli_setting)
    target = resolve_execution_target(row)
    assert target == {"provider": "openai", "model": "gpt-5", "max_tokens": 700}


def test_resolve_execution_target_falls_through_to_library_provider(monkeypatch):
    row = _definition_row(input={"question": "q"})
    monkeypatch.setattr(automation_execution, "get_cli_setting", lambda *a, **k: None)
    monkeypatch.setattr(
        automation_execution,
        "resolve_library_rag_answer_provider",
        lambda: ("deepseek", None),
    )
    target = resolve_execution_target(row)
    assert target == {"provider": "deepseek", "model": None, "max_tokens": 1000}


def test_resolve_execution_target_max_tokens_capped_at_4000(monkeypatch):
    row = _definition_row(input={"question": "q", "max_tokens": 999999})
    monkeypatch.setattr(automation_execution, "get_cli_setting", lambda *a, **k: None)
    monkeypatch.setattr(
        automation_execution, "resolve_library_rag_answer_provider", lambda: (None, None)
    )
    target = resolve_execution_target(row)
    assert target["max_tokens"] == 4000


# --- execute_recurring_question: question_empty ------------------------------


@pytest.mark.asyncio
async def test_question_empty_returns_degraded_without_calling_retrieval(monkeypatch):
    called = False

    async def fake_search(app, request):
        nonlocal called
        called = True
        return LibraryRagSearchOutcome(status="ready", results=(_row(),))

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    outcome = await execute_recurring_question(_FakeApp(), _definition_row(input={"question": "   "}))
    assert outcome.outcome == "degraded"
    assert outcome.answer_mode == "none"
    assert outcome.failure_reason == {"code": "question_empty"}
    assert called is False


# --- the six-row classification ladder ---------------------------------------


@pytest.mark.asyncio
async def test_row1_retrieval_blocked_is_degraded(monkeypatch):
    async def fake_search(app, request):
        return LibraryRagSearchOutcome(status="blocked", results=())

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    outcome = await execute_recurring_question(_FakeApp(), _definition_row())
    assert outcome.outcome == "degraded"
    assert outcome.answer_mode == "none"
    assert outcome.failure_reason == {"code": "retrieval_blocked"}
    assert outcome.source_refs == []


@pytest.mark.asyncio
async def test_row1_retrieval_failed_is_degraded(monkeypatch):
    async def fake_search(app, request):
        return LibraryRagSearchOutcome(status="failed", results=())

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    outcome = await execute_recurring_question(_FakeApp(), _definition_row())
    assert outcome.outcome == "degraded"
    assert outcome.failure_reason == {"code": "retrieval_failed"}


@pytest.mark.asyncio
async def test_row2_zero_results_is_no_match(monkeypatch):
    async def fake_search(app, request):
        return LibraryRagSearchOutcome(status="empty", results=())

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    outcome = await execute_recurring_question(_FakeApp(), _definition_row())
    assert outcome.outcome == "no_match"
    assert outcome.answer_mode == "none"
    assert outcome.title == "No matching sources found"
    assert outcome.failure_reason is None


@pytest.mark.asyncio
async def test_row3_generation_disabled_is_evidence_only_without_generation_call(monkeypatch):
    generate_called = False

    async def fake_search(app, request):
        return LibraryRagSearchOutcome(status="ready", results=(_row(),))

    async def fake_generate(**kwargs):
        nonlocal generate_called
        generate_called = True
        raise AssertionError("generation must not be called when generation_mode is disabled")

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    monkeypatch.setattr(automation_execution, "generate_library_rag_answer", fake_generate)
    outcome = await execute_recurring_question(
        _FakeApp(), _definition_row(config={"generation_mode": "disabled"})
    )
    assert outcome.outcome == "finding"
    assert outcome.answer_mode == "evidence_only"
    assert outcome.title == "Relevant evidence found"
    assert generate_called is False
    assert outcome.source_refs == [{"source": "media", "id": "r1", "title": "Title r1"}]


@pytest.mark.asyncio
async def test_row4_ready_answer_is_synthesized_finding(monkeypatch):
    async def fake_search(app, request):
        return LibraryRagSearchOutcome(status="ready", results=(_row(),))

    async def fake_generate(**kwargs):
        return LibraryRagAnswer(status=ANSWER_STATUS_READY, text="The answer is 42.", citation_status="validated")

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    monkeypatch.setattr(automation_execution, "generate_library_rag_answer", fake_generate)
    outcome = await execute_recurring_question(_FakeApp(), _definition_row())
    assert outcome.outcome == "finding"
    assert outcome.answer_mode == "synthesized"
    assert outcome.title == "Possible answer found"
    assert outcome.answer == "The answer is 42."
    assert outcome.summary == "The answer is 42."
    assert outcome.confidence == {"citation_status": "validated"}
    assert outcome.evidence_summary["answer_present"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [ANSWER_STATUS_ABSTAINED, ANSWER_STATUS_NO_EVIDENCE, ANSWER_STATUS_FAILED]
)
async def test_row5_optional_not_ready_is_evidence_only_with_answer_dropped(monkeypatch, status):
    async def fake_search(app, request):
        return LibraryRagSearchOutcome(status="ready", results=(_row(),))

    async def fake_generate(**kwargs):
        return LibraryRagAnswer(status=status, text="ignored")

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    monkeypatch.setattr(automation_execution, "generate_library_rag_answer", fake_generate)
    outcome = await execute_recurring_question(
        _FakeApp(), _definition_row(config={"generation_mode": "optional"})
    )
    assert outcome.outcome == "finding"
    assert outcome.answer_mode == "evidence_only"
    assert outcome.answer is None
    assert outcome.evidence_summary["answer_present"] is False
    assert outcome.evidence_summary["generation_status"] == status
    assert outcome.failure_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [ANSWER_STATUS_ABSTAINED, ANSWER_STATUS_NO_EVIDENCE, ANSWER_STATUS_FAILED]
)
async def test_row6_required_not_ready_is_degraded(monkeypatch, status):
    async def fake_search(app, request):
        return LibraryRagSearchOutcome(status="ready", results=(_row(),))

    async def fake_generate(**kwargs):
        return LibraryRagAnswer(status=status, text="ignored")

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    monkeypatch.setattr(automation_execution, "generate_library_rag_answer", fake_generate)
    outcome = await execute_recurring_question(
        _FakeApp(), _definition_row(config={"generation_mode": "required"})
    )
    assert outcome.outcome == "degraded"
    assert outcome.answer_mode == "none"
    assert outcome.failure_reason == {"code": "generation_required_unavailable"}
    assert outcome.source_refs == []


# --- finding_policy: top_k + high_confidence_only score post-filter ---------


@pytest.mark.asyncio
async def test_finding_policy_top_k_override_is_passed_through_in_range(monkeypatch):
    captured = {}

    async def fake_search(app, request):
        captured["top_k"] = request.top_k
        return LibraryRagSearchOutcome(status="empty", results=())

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    await execute_recurring_question(
        _FakeApp(), _definition_row(finding_policy={"preset": "balanced_findings", "top_k": 25})
    )
    assert captured["top_k"] == 25


@pytest.mark.asyncio
async def test_finding_policy_top_k_override_out_of_range_falls_back_to_default(monkeypatch):
    """`coerce_int_setting`'s bounds semantics (the `_coerce_int` precedent
    this reuses) reject an out-of-range value back to the default rather
    than clamping it -- 500 is outside the 1-100 ceiling, so the preset's
    own top_k (10) is used instead of 100."""
    captured = {}

    async def fake_search(app, request):
        captured["top_k"] = request.top_k
        return LibraryRagSearchOutcome(status="empty", results=())

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    await execute_recurring_question(
        _FakeApp(), _definition_row(finding_policy={"preset": "balanced_findings", "top_k": 500})
    )
    assert captured["top_k"] == 10


@pytest.mark.asyncio
async def test_high_confidence_only_drops_weak_scored_rows(monkeypatch):
    strong = _row(result_id="strong", score=0.9)
    weak = _row(result_id="weak", score=0.1)
    unscored = LibraryRagResultRow(
        result_id="unscored",
        title="Unscored",
        snippet="",
        score=None,
        source_id="s",
        chunk_id="c",
        citations=(),
        provenance=MappingProxyType({"source_type": "note"}),
    )

    async def fake_search(app, request):
        return LibraryRagSearchOutcome(status="ready", results=(strong, weak, unscored))

    async def fake_generate(**kwargs):
        assert {row.result_id for row in kwargs["results"]} == {"strong", "unscored"}
        return LibraryRagAnswer(status=ANSWER_STATUS_READY, text="answer")

    monkeypatch.setattr(automation_execution, "run_library_rag_search", fake_search)
    monkeypatch.setattr(automation_execution, "generate_library_rag_answer", fake_generate)
    outcome = await execute_recurring_question(
        _FakeApp(),
        _definition_row(finding_policy={"preset": "high_confidence_only"}),
    )
    assert outcome.outcome == "finding"
    assert {ref["id"] for ref in outcome.source_refs} == {"strong", "unscored"}


# --- _bounded --------------------------------------------------------------


def test_bounded_caps_at_1000_chars_with_ellipsis():
    long_text = "x" * 2000
    bounded = automation_execution._bounded(long_text)
    assert len(bounded) == RESULT_SUMMARY_MAX_CHARS
    assert bounded.endswith("…")


def test_bounded_leaves_short_text_untouched():
    assert automation_execution._bounded("short") == "short"


# --- _classify: missing-answer guard (was `assert answer is not None`) -----


def test_classify_missing_answer_after_attempted_generation_degrades_honestly():
    """`_classify`'s "generation was attempted" branch used to `assert
    answer is not None` -- unreachable through `execute_recurring_question`
    (it only reaches this branch after actually calling
    `generate_library_rag_answer`), but a production `assert` is stripped
    under `-O` and is not a real guard. It must degrade honestly rather than
    crash or silently fall through."""
    outcome = automation_execution._classify(
        retrieval_status="ok",
        results=(_row(),),
        generation_mode="optional",
        answer=None,
    )
    assert outcome.outcome == "degraded"
    assert outcome.failure_reason == {"code": "generation_unavailable"}
    assert outcome.evidence_summary["answer_present"] is False
