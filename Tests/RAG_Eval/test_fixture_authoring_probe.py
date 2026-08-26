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
than assumed, in whichever direction it reads. It has since read a THIRD way
— TASK-15400's Task 4 (2026-08-11) shipped `and_stopword_trim` as the
keyword leg's MATCH construction and hybrid now answers one of the five
golden prompt queries (recall 0.000 → 0.200); the test's docstring carries
all three states and the pin asserts the split, one hit and four misses,
rather than a direction. TASK-15700's Task 4 (2026-08-13) then moved the
default again, to `and_then_prefix`, WITHOUT moving this cell — a fourth
state in which only the mechanism behind the single hit changed (trim →
prefix fallback). That is recorded in the test's docstring too, because a
pin whose number holds for a new reason is the easiest kind to misread.

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

#: The ONE prompt golden query the shipped construction answers. Its query
#: is "saved prompt for chasing a supplier about a late order"; the full AND
#: over every token cannot be satisfied (`_FTS5_STOPWORDS`' own comment
#: records it as blocked solely by "about"). This single id IS the gated
#: `prompt` cell: 1 of 5 = recall 0.200.
#:
#: RENAMED 2026-08-13 (TASK-15700 Task 4) from
#: `STOPWORD_TRIM_RESCUED_PROMPT_ID`, because the MECHANISM changed under
#: the new default even though the CELL did not. Under
#: `and_stopword_trim` (2026-08-11 → 2026-08-13) it was rescued by the
#: TRIM: the content-token AND satisfied what the full AND could not. Under
#: the shipped `and_then_prefix` the primary IS the full AND, so it returns
#: zero rows and the per-sub-leg PREFIX FALLBACK rescues it instead. One
#: query, two mechanisms — which is exactly why this cell is unmoved across
#: the flip, and why the old name would now assert a false attribution.
RESCUED_PROMPT_ID = "pm-vendor-chaser"


def test_prompt_fixtures_are_reachable_but_four_of_five_golden_queries_are_not(
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
    * **hybrid** missed for the finding this task actually produced, filed
      as **TASK-15400**: the engine's keyword leg built its MATCH as an
      implicit AND over EVERY query token (`_escape_fts5_query`, TASK-3995)
      with no plural/singular widening. The five prompt queries are
      natural-language sentences, so each was one to five absent tokens
      away from matching and the leg returned NOTHING for them. Measured
      across the whole golden set at authoring time: the keyword leg
      returned zero rows for 40 of 60 queries, firing only for the two
      categories whose queries are keyword-shaped (`keyword` 13/16,
      `scoped` 7/7). The dominant cause is AND-strictness over CONTENT
      words (`template`, `building`, `rough`, `turns`, `pulls`, `builds`),
      not function words. For media, notes and conversations the semantic
      leg hides all of it. Prompts have no semantic leg, so they are where
      it becomes visible.

    **THIRD STATE — DISCLOSED ORACLE FLIP (2026-08-11, TASK-15400 Task 4,
    sweep row `and_trim`): hybrid now answers ONE of the five.** The arc
    swept four MATCH constructions and shipped the winner:
    `SearchConfig.fts_match_construction` went from `"and"` to
    `"and_stopword_trim"`, so the leg ANDs the CONTENT tokens. That rescues
    exactly `RESCUED_PROMPT_ID` — the category's gated cell
    moves recall 0.000 -> 0.200 (mrr 0.022, ndcg 0.060, precision 0.020),
    and it is the ONLY gated cell family that moves in any mode. So this
    test keeps its job in its third form: the four remaining misses are
    still EXPLAINED by the paragraph above — absent CONTENT words, which no
    stopword list removes — and they are part of the residual "39 of 60
    zero-row queries" bound the arc's re-scoped merge-level follow-up owns
    (**TASK-15700**), because
    widening the MATCH form further was measured to break the leg's
    round-robin merge rather than help. The one hit is asserted rather than
    tolerated: if it silently reverted, the winner would not be doing what
    the sweep measured. semantic and
    plain are untouched by the construction (measured byte-identical across
    all four) and still miss all five for the two reasons above.

    **FOURTH STATE — DISCLOSED (2026-08-13, TASK-15700 Task 4): the default
    moved again and this cell did NOT.** TASK-15700 fixed the round-robin
    merge the paragraph above blames, re-ran the sweep as SIX rows, and
    shipped `"and_then_prefix"` — by OWNER RULING, not by the rule's own
    output (the rule tied `prefix` and `and_then_prefix` at census 23 and
    its tie-break selected `prefix`; the owner overrode it for structural
    self-displacement immunity). **Nothing here flips**, and the reason is
    worth stating because it is the trap: the gained census hits vs the
    outgoing default are `kw-quillon-mast` and `kw-thimble-relay`, both in
    the `keyword` category, so the prompt category still reads exactly one
    hit of five. What DID change is this cell's MECHANISM — the new
    default's PRIMARY is the FULL AND, which returns zero rows here, and the
    per-sub-leg PREFIX FALLBACK is what now reaches `RESCUED_PROMPT_ID`.
    Hence the constant's rename (see its comment): the assertion is
    unchanged, its attribution is not. The residual bound tightens 39 → 36
    of 60 zero-row queries, with the leg answering 23 of 53.

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
    # DENOMINATOR (2026-08-11, TASK-15400 Task 4): pinned EXACTLY, where it
    # used to be a `>= 4` floor. The cell this test stands behind is
    # `prompt/recall == 0.200`, which is 1 of 5 — a floor pins the numerator
    # while letting the denominator drift, and 1-of-6 would keep every
    # assertion below green while the baselined cell moved to 0.167.
    assert len(prompts) == 5, (
        f"{len(prompts)} prompt fixtures, not 5; the gated prompt cell is "
        "1/5 = 0.200. Re-run the eval and re-stamp before moving this."
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

    # THE SPLIT, asserted BEFORE the per-query loop so that it is the oracle
    # that speaks first about the category's shape: exactly one of the five
    # is a hybrid hit, and it is the one the shipped construction was chosen
    # for. Ordered deliberately — the per-mode loop below reds on a second
    # rescue too, but it reports it as "a mode stopped missing", which reads
    # like a broken pin rather than a moved cell.
    hybrid_hits = {
        result.query_id for result in results if not result.cell("hybrid").is_miss
    }
    assert hybrid_hits == {RESCUED_PROMPT_ID}, (
        f"hybrid answers {sorted(hybrid_hits)} of the prompt category, not "
        f"just {RESCUED_PROMPT_ID!r}. With the denominator "
        "pinned at 5 above, this set IS the committed prompt/recall 0.200. "
        "Re-run the construction sweep and re-stamp before moving this pin."
    )

    for result in results:
        # The flip: this one id is a hybrid HIT under the shipped
        # `and_stopword_trim` and was a miss under the pre-arc `and`.
        rescued = result.query_id == RESCUED_PROMPT_ID
        for mode in ("semantic", "plain", "hybrid"):
            cell = result.cell(mode)
            assert cell is not None and cell.error is None, (
                f"{result.query_id}/{mode}: the query erred, so its result "
                f"proves nothing: {cell}"
            )
            if mode == "hybrid" and rescued:
                assert not cell.is_miss, (
                    f"{result.query_id}: hybrid MISSED the query the shipped "
                    "construction was chosen for. Either the default reverted "
                    "to `and` or `and_stopword_trim` stopped rescuing it — "
                    "the gated prompt cell is 0.200 because of this one hit."
                )
                continue
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


#: The EXACT census hit-set the SHIPPED construction scores — all 23 ids,
#: not a count and not a sample. A bare `== 23`, or a spot-check of the two
#: gained ids, would both stay green if the leg swapped one hit for another;
#: that silent-swap failure is precisely what the sweep's `lost` column
#: exists to catch, and a set equality is the only assertion that catches it
#: here. Derived from a real census run at the shipped default (TASK-15700
#: Task 4), not transcribed from a report.
SHIPPED_LEG_CENSUS_IDS = frozenset({
    "kw-ashgrove-pump",
    "kw-drayton-conveyor",
    "kw-fennimore-changeover",
    "kw-halcyon-ledger",
    "kw-larkspur-turbine",
    "kw-marlstone-kiln",
    "kw-nimbus-rollback",
    "kw-obsidian-spindle",
    "kw-pellucid-gauge",
    "kw-plant-maintenance-record",
    "kw-quillon-mast",
    "kw-thimble-relay",
    "kw-verdigris-coating",
    "kw-zephyr-asset-tag",
    "kw-zephyr-flywheel",
    "pm-vendor-chaser",
    "sc-duty-board-notice",
    "sc-intake-screen-survey",
    "sc-meter-box-key",
    "sc-pump-chamber-inspection",
    "sc-sample-point-sign",
    "sc-storm-overflow-record",
    "sc-valve-pit-access",
})
SHIPPED_LEG_CENSUS = len(SHIPPED_LEG_CENSUS_IDS)
SHIPPED_LEG_CENSUS_SCOREABLE = 53
#: The residual bound AC#7 owns: golden queries the leg returns NOTHING for.
SHIPPED_LEG_ZERO_ROW = 36
#: Gained vs the construction that shipped 2026-08-11 -> 2026-08-13. A
#: SUBSET of the set above, called out by name because these two ids are
#: what the 2026-08-13 flip actually bought.
PREFIX_FALLBACK_GAINED_IDS = frozenset({"kw-quillon-mast", "kw-thimble-relay"})
#: The vector-blind fixture — hard constraint (a) at the LEG level, where
#: the sweep measured it. Also a member of the set above.
VECTOR_BLIND_FIXTURE_ID = "kw-plant-maintenance-record"


def test_the_shipped_construction_scores_the_census_the_owner_ruling_bought(
    tmp_path, capsys
):
    """The keyword leg answers 23 of 53, and WHICH 23 — the flip's oracle.

    DISCLOSED NEW PIN (2026-08-13, TASK-15700 Task 4). The arc's headline
    number had no always-on defence: the census lives in the gated sweep
    (`test_fusion_sweep.py`), which the standard gated run excludes, so a
    default that silently reverted would have left every other pin green
    except the two that assert the default's NAME. This pin asserts the
    CONSEQUENCE instead, which is what the flip was actually for.

    Read the value correctly: `and_then_prefix` ships by OWNER RULING, not
    as the pre-registered rule's output. The rule tied it with `prefix` at
    census 23 — measurement-identical on every captured axis — and its
    tie-break (fewest extra FTS statements, 240 vs 460) selected `prefix`;
    the owner overrode that for structural immunity to intra-sub-leg
    self-displacement. So this census is the number BOTH qualifiers score,
    and it is not evidence for the ruling — the ruling's evidence is
    structural, and its price is statements, neither of which a census can
    see.

    Reverting the default to `and_stopword_trim` reds the hit-set assertion
    FIRST, naming the construction it actually found and the exact ids that
    went missing (`kw-quillon-mast`, `kw-thimble-relay`) — the ordering is
    deliberate, because a bare "the default is not `and_then_prefix`" says
    what changed without saying what it cost.
    """
    from Tests.RAG_Eval.harness.fusion_sweep import keyword_leg_census
    from Tests.RAG_Eval.harness.goldenset import load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime

    corpus, golden = load_fixtures()
    runtime = build_eval_runtime(corpus, tmp_path)
    try:
        # Read off the runtime, NOT set here: this pin is about what the
        # SHIPPED default does, so a construction assigned locally would
        # make it green under any default at all.
        construction = runtime.service.config.search.fts_match_construction
        census = keyword_leg_census(runtime, golden, k=K)
    finally:
        runtime.close()

    with capsys.disabled():
        print(f"\nkeyword-leg census under {construction!r}: "
              f"{census.hits}/{census.scoreable}, "
              f"{len(census.zero_row_queries)} zero-row of {census.queries}")

    # THE HIT-SET, asserted FIRST and as a set EQUALITY. Ordered ahead of
    # the construction guard on purpose: a revert must red with the ids it
    # COST, not merely with the name that changed. Set equality (not `<=`,
    # not a count) is what closes the silent-swap hole — a leg that traded
    # one hit for another keeps `census.hits == 23` and would sail past any
    # weaker form of this assertion.
    hit_set = set(census.hit_queries)
    lost = SHIPPED_LEG_CENSUS_IDS - hit_set
    gained = hit_set - SHIPPED_LEG_CENSUS_IDS
    assert hit_set == SHIPPED_LEG_CENSUS_IDS, (
        f"the keyword leg's census hit-set moved under {construction!r}: "
        f"lost {sorted(lost) or '-'}, gained {sorted(gained) or '-'} "
        f"({census.hits} hits, was {SHIPPED_LEG_CENSUS}). If the default "
        "reverted, the two lost ids are what the 2026-08-13 flip bought; if "
        "it did not, the leg changed under a fixed construction and the "
        "sweep needs re-running."
    )
    # ...and only then the guard, which explains why the number above is
    # meaningful at all.
    assert construction == "and_then_prefix", (
        f"the shipped default is {construction!r}; this census was measured "
        "for `and_then_prefix` and means nothing under another construction"
    )
    assert (census.hits, census.scoreable) == (
        SHIPPED_LEG_CENSUS,
        SHIPPED_LEG_CENSUS_SCOREABLE,
    ), f"census moved: {census.hits}/{census.scoreable}"
    # The two ids the flip bought, named separately from the set above so a
    # future edit to the set cannot quietly drop them without saying so.
    assert PREFIX_FALLBACK_GAINED_IDS <= hit_set, (
        "the leg lost one of the two ids the 2026-08-13 flip bought "
        f"({sorted(PREFIX_FALLBACK_GAINED_IDS)})"
    )
    # The vector-blind fixture's own leg row — hard constraint (a) at the
    # LEG level, where the sweep measured it.
    assert VECTOR_BLIND_FIXTURE_ID in hit_set, (
        "the vector-blind fixture left the keyword leg's top-10; its hybrid "
        "rescue is the constraint that disqualified `or` outright"
    )
    assert len(census.zero_row_queries) == SHIPPED_LEG_ZERO_ROW, (
        f"the residual zero-row bound moved to "
        f"{len(census.zero_row_queries)} of {census.queries}"
    )
