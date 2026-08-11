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

The second test was the prompts sub-leg's before-number and is now its
after-number (TASK-15020/B2, 2026-08-11). Prompt fixtures were admitted
STRUCTURALLY rather than by a measured ranking — the harness had no prompts
writer, the engine's keyword leg had no prompts sub-leg, and prompts have no
vector index, so those documents were absent from retrieval by construction.
B2 shipped the first two; the third is deliberately still true. The test now
pins the three-way split that produces (hybrid finds, semantic cannot, plain
is a harness gap), keeping the same job: a prompt cell is EXPLAINED rather
than assumed, in whichever direction it reads.

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


#: A keyword-SHAPED query for the same prompt `PROMPT_REACHABILITY_SLUG`
#: names. Every token of it occurs in that fixture, which is the whole
#: point: see the reachability half of the pin below.
PROMPT_REACHABILITY_QUERY = "shift log summary supervisor"
PROMPT_REACHABILITY_SLUG = "prompt-shift-summary"


def test_prompt_fixtures_are_reachable_but_their_golden_queries_are_not(
    tmp_path, capsys
):
    """THE PROMPTS SUB-LEG'S OUTCOME — both states, dated, and the surprise.

    **Before (2026-08-10 → 2026-08-11, TASK-15020/B2):** every `prompt`
    golden query scored recall 0.000 in semantic, plain AND hybrid, because
    its target was never written into any DB (`ingest.UNWRITABLE_SOURCE_
    TYPES`, as it was then called) and no seam served prompts. That was
    absence, not a retrieval failure, and the before-pin asserted the
    absence first so the misses had a stated cause.

    **After (this commit):** the harness writes prompts through
    `PromptsDatabase.add_prompt` and the engine has a prompts keyword
    sub-leg — and **the category's numbers did not move**. All three modes
    still miss all five queries. The cause changed completely, which is why
    this test changed shape rather than expectation:

    * **semantic** misses structurally, as before and by design: nothing
      indexes prompts into the vector store (`ingest.UNINDEXED_SOURCE_
      TYPES`); B2 deliberately left semantic indexing of prompts out of
      scope.
    * **plain** misses because this harness wires
      `prompt_scope_service=None`. Plain mode never touches the engine — it
      fans out over the Library's own four seams — so its prompt column is a
      HARNESS gap, not a pipeline one; the shipped app's plain mode does
      find prompts.
    * **hybrid** misses for the finding this task actually produced, filed
      as **TASK-15400**: the engine's keyword leg builds its MATCH as an
      implicit AND over EVERY query token (`_escape_fts5_query`, TASK-3995)
      with no plural/singular widening. The five prompt queries are
      natural-language sentences, so each is one to five absent tokens away
      from matching and the leg returns NOTHING for them. Measured across
      the whole golden set at authoring time: the keyword leg returns zero
      rows for 40 of 60 queries, firing only for the two categories whose
      queries are keyword-shaped (`keyword` 13/16, `scoped` 7/7). The
      dominant cause is AND-strictness over CONTENT words (`template`,
      `building`, `rough`, `turns`, `pulls`, `builds`), not function words:
      a stopword-trimmed AND rescues exactly 1 of those 40. For media,
      notes and conversations the semantic leg hides all of it. Prompts
      have no semantic leg, so they are where it becomes visible.

    **The reachability half is what keeps this from being indistinguishable
    from "B2 never shipped".** A keyword-shaped query for the same fixture,
    through the same runtime and the same production seam, returns it at
    the top of hybrid. The sub-leg works; the golden prompt queries measure
    a capability prompts structurally do not have.

    The queries are deliberately NOT re-authored to make the number move:
    fixtures rewritten after seeing the result measure the rewrite. Which
    side should change — the engine's MATCH construction or the prompt
    fixtures — is a decision with its own measurement, filed as TASK-15400
    rather than smuggled in here. (The Library's four-seam
    `build_fts_match_query` is NOT the ready-made answer: measured, it
    rescues 1 of the 40 zero-row queries, and it is AND-joined too.)
    """
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )

    from Tests.RAG_Eval.harness.fixture_probe import format_probe_report, probe_candidates
    from Tests.RAG_Eval.harness.goldenset import GoldenQuery, load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import SOURCE_TYPES

    corpus, golden = load_fixtures()
    prompts = [query for query in golden if query.category == "prompt"]
    assert len(prompts) >= 4, (
        f"only {len(prompts)} prompt fixtures; the number this pins needs "
        "the category's floor"
    )
    reachability = GoldenQuery(
        "pm-reachability-probe",
        PROMPT_REACHABILITY_QUERY,
        "prompt",
        (PROMPT_REACHABILITY_SLUG,),
    )

    runtime = build_eval_runtime(corpus, tmp_path)
    try:
        # Presence FIRST, the mirror image of the before-pin's absence
        # assertion: a miss can only be attributed to retrieval once the
        # target is known to be in the runtime at all.
        prompt_slugs = {doc.slug for doc in corpus if doc.source_type == "prompt"}
        assert prompt_slugs <= set(runtime.slug_to_source), (
            "a prompt fixture never reached the runtime's slug map, so these "
            "misses would be absence again rather than the new finding"
        )
        # ...and the surviving structural absence, stated rather than
        # inferred from a semantic column of zeros.
        assert set(runtime.unindexed) == prompt_slugs, (
            f"ingestion indexed or skipped something unexpected: "
            f"unindexed={sorted(runtime.unindexed)}"
        )
        results = probe_candidates(corpus, prompts, tmp_path, k=K, runtime=runtime)
        reachable = probe_candidates(
            corpus, [reachability], tmp_path, k=K, runtime=runtime
        )[0]
        # The fusion provenance for that same hit, READ off the engine's own
        # `hybrid_fusion` block rather than recomputed. Task 6's lesson: a
        # mechanism sentence is an oracle, and paper arithmetic has already
        # refuted itself twice in this arc.
        runtime.service.config.search.default_search_mode = "hybrid"
        payload = runtime.run(
            LibraryLocalRagSearchService(runtime.app).search(
                PROMPT_REACHABILITY_QUERY, SOURCE_TYPES, "rag", top_k=K
            )
        )
        prompt_provenance = [
            (position, row["source_id"], row["provenance"].get("hybrid_fusion"))
            for position, row in enumerate(payload["results"], start=1)
            if (row.get("provenance") or {}).get("source_type") == "prompt"
        ]
    finally:
        try:
            runtime.close()
        except Exception as exc:  # pragma: no cover - reported, not raised
            print(f"NOTE: runtime.close() failed after the pin: {exc!r}")

    with capsys.disabled():
        print("\n" + format_probe_report([*results, reachable], k=K))
        for position, source_id, fusion in prompt_provenance:
            print(
                f"reachability provenance: rank {position} prompt {source_id} "
                f"{fusion}"
            )

    for result in results:
        for mode in ("semantic", "plain", "hybrid"):
            cell = result.cell(mode)
            assert cell is not None and cell.error is None, (
                f"{result.query_id}/{mode}: the query erred, so its result "
                f"proves nothing: {cell}"
            )
            assert cell.is_miss, (
                f"{result.query_id}: {mode} returned {result.target_slugs} at "
                f"rank {cell.best_rank}. The docstring explains why each mode "
                "misses; whichever reason stopped being true, update it here "
                "rather than deleting the pin."
            )
        # Vacuity check: hybrid returning rows proves retrieval ran and the
        # prompt simply was not among them.
        assert result.cell("hybrid").docs_returned > 0, (
            f"{result.query_id}: hybrid returned no rows at all"
        )

    # THE REACHABILITY HALF. Same runtime, same seam, keyword-shaped query.
    hybrid = reachable.cell("hybrid")
    assert hybrid is not None and hybrid.error is None, hybrid
    assert not hybrid.is_miss, (
        f"a keyword-shaped query ({PROMPT_REACHABILITY_QUERY!r}) did not "
        f"return {PROMPT_REACHABILITY_SLUG!r} within k={K}. Then the prompts "
        "sub-leg itself is broken, and the misses above are NOT the "
        "query-shape finding this test describes."
    )
    assert reachable.cell("semantic").is_miss, (
        "the semantic leg returned a prompt, so something is indexing "
        "prompts into the vector store and this whole class measures "
        "something else now"
    )
    # ...and it got there as an FTS-ONLY row. This is the property the whole
    # sub-leg rests on: prompts have no vector twin, so every prompt row in a
    # fused result is one the weighting rescued.
    assert prompt_provenance, (
        "the production seam returned no prompt row for the keyword-shaped "
        "query, though the probe found one — the two paths disagree"
    )
    for position, source_id, fusion in prompt_provenance:
        assert fusion is not None, (
            f"prompt {source_id} at rank {position} carries no fusion "
            "provenance, so it cannot be shown to be FTS-only"
        )
        assert fusion["vector_rank"] is None and fusion["vector_score"] is None, (
            f"prompt {source_id} at rank {position} carries a vector "
            f"contribution: {fusion}"
        )
        assert fusion["fts_rank"] is not None
