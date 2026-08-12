# Tests/RAG_Eval/test_harness_scoped.py
"""Env-gated: what a SCOPED query does, measured through the seam.

This file pinned a constraint before it was removed, and now pins its
removal. Both states, with their dates, because the value of a flip is that
someone can read what flipped:

* **before — 2026-08-10.** The P1 arc deferred scope-aware hybrid retrieval,
  so the Library seam diverted a hybrid profile to semantic whenever a scope
  was active: `RAGService.search` raised for a non-empty `metadata_allowlist`
  with any non-semantic search type. The hybrid column of every scoped
  measurement was therefore a second semantic column.
* **after — 2026-08-11 (TASK-15020/B1).** The engine's FTS sub-legs take
  their entry's ids as a parameterized filter (B1a, `f0f2ac793`), so the
  guard narrowed from "semantic only" to "not keyword only"; the seam then
  stopped routing on the scope at all (B1b, `f5352f2b8`) and the
  `ROUTE_NOTE_HYBRID_SCOPED` disclosure was retired with it. A scoped query
  under a hybrid profile now runs the engine's fused hybrid, with **no
  routing disclosure** — the absence is the assertion, because a disclosure
  is exactly what a divert would leave behind.

The test below records the after-state with the real stack — real writers, a
real index, the production seam — so the flip is a visible change in a
committed test rather than a number that moved for unexplained reasons. A
pin nobody can read afterwards is just a passing test.

The corpus is NOT touched. The shipped scoped FIXTURES (seven, in
`golden.toml`, over one 100-document scope) are what the harness scores; this
file is the routing pin, and it keeps its own two-document inline scenario so
it stays a statement about routing that a fixture edit cannot move.

Skipped unless `RAG_EVAL=1` plus the embeddings extras plus a warm model
cache — see `harness/environment.py`.
"""
from __future__ import annotations

from Tests.RAG_Eval.harness.environment import harness_gate

pytestmark = harness_gate()

K = 10

#: A keyword query the shipped corpus answers, reused verbatim for both the
#: control and the scoped variant so the ONLY difference between them is the
#: scope. (`kw-obsidian-spindle` in the golden set.)
QUERY_TEXT = "Obsidian-3 lathe spindle bearing"
TARGET_SLUG = "media-obsidian-lathe"

#: The scope: one media document (the target) and one note. Two source types
#: on purpose — a scope with both exercises the per-source-type allowlist
#: split (`rag_scope.build_semantic_allowlists` returns one entry per type,
#: because a flat allowlist dict cannot express a union across types).
SCOPE_SLUGS = (TARGET_SLUG, "note-zephyr-flywheel")


def test_a_scoped_query_under_a_hybrid_profile_runs_fused_hybrid(
    tmp_path, capsys
):
    """THE AFTER-PIN of this arc's routing change (flipped 2026-08-11).

    On the hybrid profile, through `LibraryLocalRagSearchService`:

      * unscoped -> `rag-hybrid`, no routing disclosure
      * scoped   -> `rag-hybrid`, no routing disclosure

    Until TASK-15020/B1 the second line read `rag-semantic`, disclosed as
    `ROUTE_NOTE_HYBRID_SCOPED` ("scope active — semantic only until
    scope-aware hybrid lands"). This test asserted exactly that, and went red
    on all three of its routing assertions the moment B1b landed; flipping
    them — rather than deleting the test — is what makes the change legible
    afterwards. See the module docstring for both states and their dates.

    **The scope, not the route, is now what separates the two arms.** Before
    the flip they were distinguishable by their backend label; now they route
    identically, and the only thing that still distinguishes them is what
    they were allowed to return. So the containment assertion is no longer a
    supporting detail — it is the assertion, and it is only meaningful while
    the CONTROL reaches outside the scope, which is asserted too. Without
    that, a scope that had quietly stopped reaching the store would look
    exactly like a scope that worked.

    Both arms run in ONE `run_eval` over one runtime: the runtime (real
    writers, real embeddings) is the expensive part, and running the control
    in a second one would compare across two indexes.
    """
    from Tests.RAG_Eval.harness.goldenset import SCOPED_CATEGORY, GoldenQuery, load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.runner import run_eval

    corpus, _golden = load_fixtures()
    by_slug = {doc.slug: doc for doc in corpus}
    for slug in SCOPE_SLUGS:
        assert slug in by_slug, (
            f"{slug!r} is no longer in the corpus; this pin's scoped scenario "
            "was built from documents the shipped corpus contains"
        )

    control = GoldenQuery(
        id="pin-control-unscoped",
        query=QUERY_TEXT,
        category="keyword",
        relevant_slugs=(TARGET_SLUG,),
    )
    scoped = GoldenQuery(
        id="pin-scoped",
        query=QUERY_TEXT,
        category=SCOPED_CATEGORY,
        relevant_slugs=(TARGET_SLUG,),
        scope_slugs=SCOPE_SLUGS,
    )

    runtime = build_eval_runtime(corpus, tmp_path)
    close_error: Exception | None = None
    try:
        report = run_eval(runtime, [control, scoped], k=K, modes=("hybrid",))
    finally:
        # Never raise a close failure over a real one.
        try:
            runtime.close()
        except Exception as exc:  # pragma: no cover - reported, not raised
            close_error = exc

    hybrid = report.modes["hybrid"]
    outcomes = {outcome.query_id: outcome for outcome in hybrid.queries}

    with capsys.disabled():
        print("\n" + report.format_summary())
    if close_error is not None:
        print(f"NOTE: runtime.close() failed after the run: {close_error!r}")

    assert not hybrid.errors, (
        f"the run erred, so its routing proves nothing: {hybrid.errors}"
    )

    # --- control: no scope, the profile's own route -------------------------
    unscoped = outcomes[control.id]
    assert unscoped.runtime_backend == "rag-hybrid", (
        f"the unscoped control routed to {unscoped.runtime_backend!r}, not the "
        "hybrid profile's own fused path — the scoped assertion below would "
        "then be pinning something other than the scope's effect"
    )
    assert unscoped.route_notes == (), (
        f"the unscoped control carried routing disclosures {unscoped.route_notes}"
    )
    assert TARGET_SLUG in unscoped.retrieved_doc_ids, (
        "the control did not even find the target; the scoped arm's result "
        "would be uninterpretable"
    )
    # The control must reach OUTSIDE the scope, or the containment assertion
    # below is satisfied by a search that returned nothing in particular.
    assert set(unscoped.retrieved_doc_ids) - set(SCOPE_SLUGS), (
        f"the unscoped control returned only {sorted(set(unscoped.retrieved_doc_ids))}, "
        "all of it inside the scope — containment below would then prove "
        "nothing about the allowlist"
    )

    # --- the pin: same query, scoped -> the profile's own fused path ---------
    pinned = outcomes[scoped.id]
    assert pinned.rows_returned > 0, (
        "the scoped search returned nothing at all, so its route says nothing "
        "about scoped retrieval"
    )
    assert pinned.runtime_backend == "rag-hybrid", (
        f"a scoped query under the hybrid profile routed to "
        f"{pinned.runtime_backend!r}. If this is 'rag-semantic', the B1 divert "
        "is back: a scope is once again costing the user their profile's "
        "retrieval mode, and every scoped hybrid number the harness reports "
        "is a semantic number (see this test's docstring for both states)."
    )
    assert pinned.route_notes == (), (
        f"the scoped query carried routing disclosures {pinned.route_notes}. "
        "After B1 there is nothing to disclose — the scope no longer changes "
        "the route — so a disclosure here means some arm diverted the search "
        "again, quietly, behind a hybrid label."
    )

    # --- and the scope really restricted retrieval ---------------------------
    assert set(pinned.retrieved_doc_ids) <= set(SCOPE_SLUGS), (
        f"scoped retrieval returned {sorted(set(pinned.retrieved_doc_ids) - set(SCOPE_SLUGS))} "
        "from OUTSIDE the scope — the allowlist did not reach BOTH engine "
        "legs (B1a pushed it into the FTS sub-legs; a leg that lost it "
        "searches the whole corpus and fuses the result in)"
    )
    assert TARGET_SLUG in pinned.retrieved_doc_ids, (
        "the scoped search missed its in-scope target, so a later 'scoped "
        "recall improved' claim would have nothing to improve on"
    )

    # The report must carry the route for review, not only this assertion.
    # One backend for both arms now: before B1 this read
    # `("rag-hybrid", "rag-semantic")` because the scoped arm was diverted.
    assert hybrid.runtime_backends == ("rag-hybrid",), (
        f"the mode's recorded backends {hybrid.runtime_backends} show more "
        "than the hybrid profile's own route in one pass"
    )
