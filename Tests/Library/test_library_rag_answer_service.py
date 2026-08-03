"""Tests for the Library RAG grounded-answer service (PR-3 task 1).

Live UAT found the Library's "RAG Answer" mode never answered: it returned
evidence rows and admitted generation "remains downstream work" -- and when
it did return rows, it surfaced 6%-relevance fixtures for an unrelated
query. The owner's ruling: answer honestly, always favouring accuracy over
assumption. "Nothing in your library supports an answer" is a first-class
answer here, not an error, and every claim must be grounded in the
retrieved evidence with citations.

**Exactly one seam is faked: `chat`.** The evidence bundle, the prompt
formatter and the citation validator are all the real shipping ones
(`search_handoff.build_library_rag_evidence_bundle`,
`answer_citations.format_evidence_for_cited_answer` /
`build_answer_citation_validation`) -- reusing them wholesale is the point
of the design, so faking them would test a parallel universe.

No test here can reach a network: every call injects a recording fake.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.Library.library_rag_answer_service import (
    ANSWER_MAX_TOKENS,
    ANSWER_STATUS_ABSTAINED,
    ANSWER_STATUS_FAILED,
    ANSWER_STATUS_NO_EVIDENCE,
    ANSWER_STATUS_READY,
    ANSWER_TEMPERATURE,
    EMPTY_ANSWER_ERROR,
    ERROR_CHAR_CAP,
    LIBRARY_RAG_ANSWER_SYSTEM_PROMPT,
    LIBRARY_RAG_NO_EVIDENCE_TEXT,
    _is_abstention,
    generate_library_rag_answer,
)

pytestmark = pytest.mark.unit


QUERY = "Why did the incident happen?"

GROUNDED_ANSWER = "An expired credential caused the incident [S1]."


def _row(**overrides):
    """One Library RAG result row, in the shape the panel already builds."""
    row = {
        "result_id": "note-42:chunk-7",
        "title": "Incident Review",
        "snippet": "Expired credential caused the incident.",
        "source_id": "note-42",
        "chunk_id": "chunk-7",
        "source_type": "note",
        "runtime_backend": "local-fts",
        "score": 0.93,
    }
    row.update(overrides)
    return row


def _blocked_row():
    """A row whose evidence is real but ineligible to ground an answer.

    Cross-workspace rows stay visible in browse but are `blocked` in the
    bundle -- nothing in them may be cited.
    """
    return _row(
        title="Workspace B Note",
        workspace_ids=("workspace-b",),
        active_workspace_id="workspace-a",
    )


class _FakeChat:
    """The one faked seam: a stand-in for `Chat_Functions.chat_api_call`.

    Records every call's kwargs so a test can assert what reached the
    provider boundary -- and, crucially, that nothing did on the paths that
    must never call a provider.
    """

    def __init__(self, *, reply: object = GROUNDED_ANSWER, error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


# --- Contract 1: no eligible evidence never reaches a provider -----------


async def test_zero_results_answers_honestly_without_calling_the_provider():
    chat = _FakeChat()

    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[],
        coverage_note="",
        provider="openai",
        model="gpt-x",
        chat=chat,
    )

    assert answer.status == ANSWER_STATUS_NO_EVIDENCE
    assert answer.text == "Nothing in your library supports an answer to that."
    assert answer.text == LIBRARY_RAG_NO_EVIDENCE_TEXT
    assert answer.error == ""
    assert chat.calls == [], "zero evidence must never spend a provider call"


async def test_results_with_no_citable_evidence_never_reach_the_provider():
    """All-blocked evidence is the same honest answer as no evidence at all.

    Asking a model to answer from a block that says "no available evidence
    references are eligible for grounding" can only produce an abstention or
    a guess; spending a call to find out which is exactly the assumption the
    ruling forbids.
    """
    chat = _FakeChat()

    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_blocked_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=chat,
    )

    assert answer.status == ANSWER_STATUS_NO_EVIDENCE
    assert answer.text == LIBRARY_RAG_NO_EVIDENCE_TEXT
    assert chat.calls == []
    # The bundle still travels back, so the panel can show why each row was
    # ineligible rather than silently dropping it.
    assert answer.evidence_bundle is not None
    assert answer.evidence_bundle.references[0].status == "blocked"


# --- Contract 2: the shape of the one provider call ---------------------


async def test_the_provider_call_is_one_shot_non_streaming_and_low_temperature():
    chat = _FakeChat()

    await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="anthropic",
        model="claude-x",
        chat=chat,
    )

    assert len(chat.calls) == 1
    kwargs = chat.calls[0]
    assert kwargs["api_endpoint"] == "anthropic"
    assert kwargs["model"] == "claude-x"
    assert kwargs["streaming"] is False
    assert kwargs["temp"] == ANSWER_TEMPERATURE == 0.2
    assert kwargs["max_tokens"] == ANSWER_MAX_TOKENS == 1200
    # The honesty prompt travels in `system_message`, not as a message role:
    # each provider handler decides how its API wants the system turn.
    assert kwargs["system_message"] == LIBRARY_RAG_ANSWER_SYSTEM_PROMPT
    assert [message["role"] for message in kwargs["messages_payload"]] == ["user"]


async def test_an_async_chat_seam_is_awaited():
    calls: list[dict] = []

    async def _async_chat(**kwargs):
        calls.append(kwargs)
        return GROUNDED_ANSWER

    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=_async_chat,
    )

    assert len(calls) == 1
    assert answer.status == ANSWER_STATUS_READY
    assert answer.text == GROUNDED_ANSWER


# --- Contract 3: what the model is actually shown -----------------------


async def test_the_user_message_carries_evidence_question_and_coverage_note():
    chat = _FakeChat()
    coverage_note = "All matches are weak. Semantic search found nothing from: Notes."

    await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note=coverage_note,
        provider="openai",
        model=None,
        chat=chat,
    )

    user = chat.calls[0]["messages_payload"][0]["content"]
    # The staged-evidence block from the shipping formatter, not a parallel one.
    assert "[Staged evidence]" in user
    assert "[S1] Incident Review" in user
    assert "Expired credential caused the incident." in user
    assert QUERY in user
    # Retrieval honesty reaches the model, not just the screen.
    assert "Retrieval coverage" in user
    assert coverage_note in user
    # The evidence is fenced as untrusted data.
    assert "UNTRUSTED EVIDENCE BEGIN" in user
    assert "UNTRUSTED EVIDENCE END" in user


async def test_an_empty_coverage_note_adds_no_coverage_claim():
    chat = _FakeChat()

    await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="   ",
        provider="openai",
        model=None,
        chat=chat,
    )

    user = chat.calls[0]["messages_payload"][0]["content"]
    assert "Retrieval coverage" not in user


# --- Contract 4: provider failure is recorded, never raised -------------


@pytest.mark.parametrize(
    "error",
    [
        ChatAuthenticationError("bad key", provider="openai"),
        ChatRateLimitError("slow down", provider="openai"),
        ChatBadRequestError("bad params", provider="openai"),
        ChatProviderError("upstream 503", provider="openai"),
        ChatConfigurationError("no model configured", provider="openai"),
        ChatAPIError("generic failure"),
        ValueError("unsupported provider"),
    ],
    ids=[
        "authentication",
        "rate_limit",
        "bad_request",
        "provider",
        "configuration",
        "api",
        "value_error",
    ],
)
async def test_provider_failures_become_a_failed_answer_and_never_raise(error):
    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=_FakeChat(error=error),
    )

    assert answer.status == ANSWER_STATUS_FAILED
    assert answer.error == str(error)
    assert answer.text == ""


async def test_a_huge_provider_error_is_capped_not_pasted_whole():
    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=_FakeChat(error=ChatProviderError("x" * 5000)),
    )

    assert answer.status == ANSWER_STATUS_FAILED
    assert len(answer.error) <= ERROR_CHAR_CAP == 500
    assert answer.error.endswith(" [...]")


async def test_an_error_with_no_message_still_names_the_failure():
    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=_FakeChat(error=ValueError()),
    )

    assert answer.status == ANSWER_STATUS_FAILED
    assert answer.error == "ValueError"


# --- Contract 5: an empty reply is a failure, not a silent success ------


@pytest.mark.parametrize("reply", ["", "   \n  ", None, {}], ids=["empty", "blank", "none", "no_content"])
async def test_an_empty_model_reply_is_a_failure(reply):
    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=_FakeChat(reply=reply),
    )

    assert answer.status == ANSWER_STATUS_FAILED
    assert answer.error == EMPTY_ANSWER_ERROR == "The model returned an empty answer."
    assert answer.text == ""


# --- Contract 6: citation validation decides ready vs abstained ---------


async def test_a_cited_answer_is_ready_and_carries_validated_citations():
    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=_FakeChat(reply=GROUNDED_ANSWER),
    )

    assert answer.status == ANSWER_STATUS_READY
    assert answer.text == GROUNDED_ANSWER
    assert answer.citation_status == "validated"
    assert answer.citation_recovery == ""
    assert answer.evidence_bundle.references[0].evidence_id == "S1"


async def test_an_uncited_answer_stays_ready_but_carries_the_recovery_copy():
    """An answer that cites nothing is a grounding failure the UI must show.

    It is deliberately NOT relabelled as an abstention: calling an uncited
    claim "abstained" would dress a possible hallucination up as honesty.
    """
    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=_FakeChat(reply="An expired credential caused the incident."),
    )

    assert answer.status == ANSWER_STATUS_READY
    assert answer.citation_status == "uncited"
    assert answer.citation_recovery == "The answer does not cite available staged evidence."


async def test_an_invented_citation_label_is_reported_as_unverified():
    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=_FakeChat(reply="The credential expired [S7]."),
    )

    assert answer.status == ANSWER_STATUS_READY
    assert answer.citation_status == "unverified"
    assert answer.citation_recovery


async def test_a_model_that_declines_is_recorded_as_abstained():
    answer = await generate_library_rag_answer(
        query=QUERY,
        results=[_row()],
        coverage_note="",
        provider="openai",
        model=None,
        chat=_FakeChat(reply=f"  {LIBRARY_RAG_NO_EVIDENCE_TEXT}\n"),
    )

    assert answer.status == ANSWER_STATUS_ABSTAINED
    assert answer.text == LIBRARY_RAG_NO_EVIDENCE_TEXT


def test_insufficient_evidence_validation_counts_as_an_abstention():
    """The mapping contract, tested at its own seam.

    `build_answer_citation_validation` only reports `insufficient_evidence`
    when the bundle holds no citable reference -- a state
    `generate_library_rag_answer` refuses to send to a provider at all (see
    the no-evidence tests above), so this branch is unreachable end-to-end
    today. It is kept, and pinned here, because the mapping is part of the
    published contract: if a caller ever hands in a pre-built bundle, an
    answer validated as insufficient must never be presented as grounded.
    """
    assert _is_abstention("Some text with no support.", "insufficient_evidence") is True
    assert _is_abstention(LIBRARY_RAG_NO_EVIDENCE_TEXT, "uncited") is True
    assert _is_abstention("Nothing in your library supports an answer to that", "uncited") is True
    assert _is_abstention("An expired credential caused the incident [S1].", "validated") is False


# --- The honesty prompt itself ------------------------------------------


def test_the_system_prompt_pins_the_honesty_contract():
    """A future prompt edit must not silently drop any of these four.

    Each line here is a behaviour UAT caught the old path getting wrong.
    """
    prompt = LIBRARY_RAG_ANSWER_SYSTEM_PROMPT

    # (a) answer only from the staged evidence
    assert "using only the evidence staged" in prompt
    # (b) cite with the given bracketed labels, never invent one
    assert "[S1]" in prompt
    assert "never invent" in prompt
    # (c) plain abstention, in the exact words the no-evidence path uses
    assert LIBRARY_RAG_NO_EVIDENCE_TEXT in prompt
    # (d) the evidence is untrusted data, not instructions
    assert "untrusted" in prompt
    assert "ignore" in prompt
