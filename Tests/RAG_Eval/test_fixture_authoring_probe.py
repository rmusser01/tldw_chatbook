# Tests/RAG_Eval/test_fixture_authoring_probe.py
"""Env-gated: the authoring probe against the real stack, and the prompts
before-pin it produced.

Two things live here because they are the same measurement seen from two
sides.

`harness/fixture_probe.py` is the tool the P2ab fail-first classes were
admitted with. Its verdict logic is pure and pinned always-on
(`test_fixture_probe.py`); what cannot be pinned there is that the tool,
driven against a real runtime, reports what the pipeline actually does. A
probe that quietly returned "miss" for everything would have admitted every
candidate it was shown, and the corpus would now be full of fixtures nobody
can improve because nothing was ever wrong.

The second test is the prompts sub-leg's before-number. Prompt fixtures are
admitted STRUCTURALLY, not by a measured ranking: the harness has no prompts
writer, the engine's keyword leg has no prompts sub-leg, and prompts have no
vector index, so those documents are absent from retrieval by construction.
This pins both halves — absent from the runtime, and missed by all three
modes — so that when the prompts sub-leg lands, the flip is a failing test
with an explanation rather than a number that moved.

Skipped unless `RAG_EVAL=1` plus the embeddings extras plus a warm model
cache — see `harness/environment.py`.
"""
from __future__ import annotations

from Tests.RAG_Eval.harness.environment import harness_gate

pytestmark = harness_gate()

K = 10

#: A shipped keyword fixture today's hybrid answers at rank 1. The probe must
#: REJECT it: that is the direction of the admission rule that is easy to get
#: wrong and impossible to notice, because a tool that admits everything
#: still produces a plausible-looking table.
ANSWERED_QUERY_ID = "kw-obsidian-spindle"


def test_the_probe_rejects_a_fixture_todays_pipeline_answers(tmp_path, capsys):
    from Tests.RAG_Eval.harness.fixture_probe import (
        VECTOR_MODES,
        admission_comment,
        format_probe_report,
        probe_candidates,
        verdict,
    )
    from Tests.RAG_Eval.harness.goldenset import load_fixtures

    corpus, golden = load_fixtures()
    query = next((q for q in golden if q.id == ANSWERED_QUERY_ID), None)
    assert query is not None, (
        f"{ANSWERED_QUERY_ID!r} is no longer in the golden set; this smoke test "
        "needs a fixture the pipeline demonstrably answers"
    )

    results = probe_candidates(corpus, [query], tmp_path, k=K)
    assert len(results) == 1
    result = results[0]

    with capsys.disabled():
        print("\n" + format_probe_report(results, k=K))

    outcome = verdict(result)
    assert not outcome.admitted, (
        f"the probe would have ADMITTED {ANSWERED_QUERY_ID!r}, which hybrid "
        "answers — the admission rule is inverted or the ranks are not being "
        f"read: {outcome.reason}"
    )
    for mode in VECTOR_MODES:
        cell = result.cell(mode)
        assert cell is not None and cell.error is None, f"{mode}: {cell}"
        assert cell.docs_returned > 0, f"{mode}: returned nothing at all"
    hybrid = result.cell("hybrid")
    assert hybrid.best_rank is not None, (
        "hybrid did not return the target for a keyword fixture it has always "
        "answered; the probe is measuring something else"
    )
    # The receipt an authoring session pastes into the fixture file has to
    # carry the ranks, not merely the fact that somebody ran it.
    comment = admission_comment(result, date="2026-08-10")
    assert comment.startswith("# admitted: 2026-08-10 ")
    assert f"hybrid={hybrid.best_rank}" in comment


def test_every_prompt_fixture_is_invisible_to_all_three_modes(tmp_path, capsys):
    """THE BEFORE-PIN of the prompts sub-leg.

    State as of 2026-08-10: every `prompt` golden query scores recall 0.000
    in semantic, plain and hybrid, because its target was never written into
    any DB (`ingest.UNWRITABLE_SOURCE_TYPES`) and no seam serves prompts.

    When the prompts keyword sub-leg lands, this test FAILS — deliberately.
    The fix is to flip the expectation (prompt queries become findable in
    plain and hybrid, and stay missing in semantic, which has no prompt
    index) and keep the absence assertion on `runtime.unwritable` only until
    the harness gains a prompts writer. Do not delete the test: what it
    guards is that a prompt cell reading 0.000 is explained rather than
    assumed.
    """
    from Tests.RAG_Eval.harness.fixture_probe import format_probe_report, probe_candidates
    from Tests.RAG_Eval.harness.goldenset import load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import MODES

    corpus, golden = load_fixtures()
    prompts = [query for query in golden if query.category == "prompt"]
    assert len(prompts) >= 4, (
        f"only {len(prompts)} prompt fixtures; the before-number this pins "
        "needs the category's floor"
    )

    runtime = build_eval_runtime(corpus, tmp_path)
    try:
        # Absence FIRST: it is the reason for the misses below, and a miss
        # whose cause is unknown is not a before-number.
        prompt_slugs = {doc.slug for doc in corpus if doc.source_type == "prompt"}
        assert set(runtime.unwritable) == prompt_slugs, (
            f"ingestion wrote or skipped something unexpected: "
            f"unwritable={sorted(runtime.unwritable)}"
        )
        assert not (prompt_slugs & set(runtime.slug_to_source)), (
            "a prompt fixture reached the runtime's slug map, so it is "
            "retrievable and this pin no longer measures absence"
        )
        results = probe_candidates(corpus, prompts, tmp_path, k=K, runtime=runtime)
    finally:
        try:
            runtime.close()
        except Exception as exc:  # pragma: no cover - reported, not raised
            print(f"NOTE: runtime.close() failed after the pin: {exc!r}")

    with capsys.disabled():
        print("\n" + format_probe_report(results, k=K))

    for result in results:
        for mode in MODES:
            cell = result.cell(mode)
            assert cell is not None and cell.error is None, (
                f"{result.query_id}/{mode}: the query erred, so its miss "
                f"proves nothing: {cell}"
            )
            assert cell.is_miss, (
                f"{result.query_id}: {mode} returned {result.target_slugs} at "
                f"rank {cell.best_rank}. If a prompts seam has landed, flip "
                "this pin (see the docstring) rather than deleting it."
            )
        # Vacuity check on the mode that could otherwise miss for the wrong
        # reason: plain returns nothing for many queries, so its miss alone
        # would be weak evidence. Hybrid returning rows proves retrieval ran.
        assert result.cell("hybrid").docs_returned > 0, (
            f"{result.query_id}: hybrid returned no rows at all"
        )
