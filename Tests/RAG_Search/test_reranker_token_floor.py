"""The reasoning token floor (TASK-17365).

TASK-17065 (AC#11) turned `include_reasoning` OFF on the two shipped
read-only profiles that carried it, because the shipped `max_tokens = 100`
leaves roughly sixty tokens for the JSON once the model has written its
free-form explanation: the payload truncates, `json.loads` raises, the row
comes back `scored=False` -- and the call was still BILLED. Built-in
profiles never persist, so every install picked that fix up.

**A profile a user CLONED before the fix did not.** `ProfileConfig.to_dict`
writes `asdict(reranking_config)` and `from_dict` rebuilds
`RerankingConfig(**...)`, so a clone carries `include_reasoning = true`
alongside `max_tokens = 100` in the user's own saved JSON, where no shipped
default can reach it.

**The fix is a FLOOR, not a migration** (spec decision, pre-registered): a
migration has to rewrite a file the user owns and then GUESS whether a large
`max_tokens` was deliberate. A floor cannot guess wrong -- it only ever
raises a budget that is too small to hold what the config asked the model to
produce, and it reaches profiles this arc will never see, including ones
cloned after it ships.

**No live provider calls.** Every test fakes `chat_api_call`, the reranker's
single provider seam, and BINDS what it is handed against the real
signature (the TASK-17065 rule: a fake that declares the caller's own
argument list agrees with the caller's bugs). The budget under assertion is
therefore the value that actually reaches the dispatcher, not a config
attribute read back.
"""

import inspect
from typing import Callable, List

import pytest

from tldw_chatbook.Chat.Chat_Functions import chat_api_call as real_chat_api_call
from tldw_chatbook.RAG_Search import reranker as reranker_module
from tldw_chatbook.RAG_Search.reranker import (
    REASONING_TOKEN_FLOOR,
    PointwiseReranker,
    RerankingConfig,
)
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult


#: The fake provider's chars-per-token approximation. Only its EXISTENCE
#: matters to these tests: a provider stops emitting when the budget runs
#: out, and a truncated JSON object does not parse.
_CHARS_PER_TOKEN = 4


def _install_seam(monkeypatch, responder: Callable[[dict], str]) -> List[dict]:
    """Fake `chat_api_call`, bound to the real signature; return the landings.

    `responder` receives the BoundArguments mapping (so it can react to
    `max_tokens`, which is the whole subject here) and returns the raw model
    string. Positional arguments are refused: this seam is keyword-only by
    contract, and a positional list at this signature is what once routed a
    credential into `api_endpoint`.
    """
    signature = inspect.signature(real_chat_api_call)
    landings: List[dict] = []

    def fake_chat_api_call(*args, **kwargs):
        assert not args, (
            "the reranker must call chat_api_call by keyword; positional "
            f"arguments landed here: {args!r}"
        )
        bound = signature.bind(*args, **kwargs)
        landings.append(dict(bound.arguments))
        return responder(dict(bound.arguments))

    monkeypatch.setattr(reranker_module, "chat_api_call", fake_chat_api_call)
    return landings


def _reasoning_response(landed: dict) -> str:
    """A model that writes its explanation, then stops at the token budget.

    The explanation is emitted FIRST -- the shape the task describes, and the
    shape the shipped pointwise template asks for -- so a budget too small to
    hold it truncates the object before its closing brace and the whole
    payload is unparseable. Nothing here is provider-specific; it is the
    mechanism (a budget is a hard stop) rendered at the seam.
    """
    reasoning = (
        "The document is a close topical match for the query: it names the "
        "same entity, uses the same domain vocabulary, and answers the "
        "question directly in its opening paragraph rather than in passing. "
        "Ranking it above the neighbouring rows is therefore justified on "
        "content and not merely on lexical overlap with the query terms. "
        "The competing rows mention the entity only in a list of related "
        "topics, which is why their similarity scores are close but their "
        "usefulness to this particular question is not. I have also checked "
        "that the passage is not a navigational stub or an index page, since "
        "those score well on vocabulary overlap while carrying no answer."
    )
    full = '{"reasoning": "%s", "score": 0.9}' % reasoning
    # ~700 chars: about 175 tokens of explanation. Comfortably inside the
    # floor and just as comfortably outside the shipped 100-token budget --
    # which is the entire point, and is why the budget must be the thing
    # under test rather than a fixed truncation length.
    budget_chars = landed["max_tokens"] * _CHARS_PER_TOKEN
    return full[:budget_chars]


def _fixed_score(_landed: dict) -> str:
    return '{"score": 0.5}'


def _results(n: int) -> List[SearchResult]:
    return [
        SearchResult(
            id=f"doc-{i}",
            score=round(0.9 - 0.1 * i, 3),
            document=f"body of document number {i}",
            metadata={"doc_title": f"doc-{i}", "source_type": "media"},
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_reasoning_raises_the_token_floor(monkeypatch):
    """The cloned-profile shape: reasoning on, `max_tokens = 100`. The budget
    that reaches the dispatcher is the floor, not the 100 the saved profile
    carries."""
    reranker = PointwiseReranker(
        RerankingConfig(include_reasoning=True, max_tokens=100, cache_results=False)
    )
    landings = _install_seam(monkeypatch, _fixed_score)

    await reranker.rerank("quokka marsupial", _results(1))

    assert landings, "the reranker must have reached the provider seam"
    assert landings[0]["max_tokens"] >= REASONING_TOKEN_FLOOR


@pytest.mark.asyncio
async def test_a_deliberate_large_max_tokens_is_untouched(monkeypatch):
    """It is a FLOOR, not an assignment. A user who deliberately set 4000
    keeps 4000 -- this is the property a migration could not have had, since
    a migration must decide whether that value was deliberate."""
    reranker = PointwiseReranker(
        RerankingConfig(include_reasoning=True, max_tokens=4000, cache_results=False)
    )
    landings = _install_seam(monkeypatch, _fixed_score)

    await reranker.rerank("quokka marsupial", _results(1))

    assert landings[0]["max_tokens"] == 4000


@pytest.mark.asyncio
async def test_no_floor_without_reasoning(monkeypatch):
    """No reasoning, no floor: 100 tokens is ample for `{"score": 0.9}`, and
    raising it would buy nothing while widening every provider's ceiling."""
    reranker = PointwiseReranker(
        RerankingConfig(include_reasoning=False, max_tokens=100, cache_results=False)
    )
    landings = _install_seam(monkeypatch, _fixed_score)

    await reranker.rerank("quokka marsupial", _results(1))

    assert landings[0]["max_tokens"] == 100


@pytest.mark.asyncio
async def test_a_cloned_reasoning_profile_still_comes_back_scored(monkeypatch):
    """AC#2/AC#3 end to end: the saved-profile shape against a provider that
    truncates at the budget. Pre-floor every row is billed and unscored
    (`failed == total`, no `rerank_score` stamp); with the floor the JSON
    survives and every row is scored."""
    reranker = PointwiseReranker(
        RerankingConfig(
            include_reasoning=True,
            max_tokens=100,
            cache_results=False,
            retry_on_failure=False,
        )
    )
    landings = _install_seam(monkeypatch, _reasoning_response)

    outcome = await reranker.rerank("quokka marsupial", _results(3))

    assert landings, "the reranker must have reached the provider seam"
    assert outcome.total == 3
    assert len(landings) == 3, "one billed call per row, floor or no floor"
    # `rerank_score` is the production "this number is no longer a
    # similarity" stamp (`Library/library_rag_score_kinds.py` reads it), and
    # `_apply_scores` withholds it from any row whose `RerankingResult` came
    # back `scored=False`. Its presence on every row is what "billed AND
    # scored" means here.
    #
    # NOT asserted: `outcome.failed == 0`. `PointwiseReranker.rerank` counts
    # only scoring attempts that RAISED, and `_score_result` swallows
    # `json.JSONDecodeError` and returns `scored=False` instead -- so on
    # exactly this path (a response that arrived and would not parse) the
    # counter stays at zero either way, and asserting it would be vacuous.
    # That undercount makes `RerankOutcome.degraded` read False over rows no
    # model scored; it is a separate defect from this one and is recorded
    # rather than quietly leaned on.
    assert all("rerank_score" in r.metadata for r in outcome.results), (
        "a reasoning-enabled profile with the shipped max_tokens produced "
        "billed-but-unscored rows: the model's explanation truncated the JSON"
    )
    assert {r.metadata["rerank_score"] for r in outcome.results} == {0.9}


def test_the_floor_is_large_enough_to_hold_a_reasoning_payload():
    """A floor smaller than the thing it exists to fit is a placebo. 400
    tokens is the task's own figure and roughly four times the shipped
    default; pinned so a later 'tidy-up' cannot shrink it back."""
    assert REASONING_TOKEN_FLOOR >= 400
