# Tests/RAG_Eval/test_harness_scoped.py
"""Env-gated: what a SCOPED query does today, measured through the seam.

This file exists to pin a constraint before it is removed. The P1 arc
deferred scope-aware hybrid retrieval; the Library seam therefore diverts a
hybrid profile to semantic whenever a scope is active, because
`RAGService.search` raises for a non-empty `metadata_allowlist` with any
non-semantic search type. That is a hard engine constraint, not a
preference, and it means the hybrid column of every scoped measurement is
today a second semantic column.

The pin below records that state with the real stack — real writers, a real
index, the production seam — so the arc's later routing change is a visible
flip in a committed test rather than a number that moved for unexplained
reasons. A pin nobody can read afterwards is just a passing test.

The corpus is NOT touched: there are no scoped fixtures yet (they are
authored later in this arc, together with their quota), and the committed
corpus must stay byte-identical until then or every committed baseline's
fingerprint changes. The scoped scenario here is therefore built inline from
documents the shipped corpus already contains.

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


def test_a_scoped_query_under_a_hybrid_profile_routes_semantic_today(
    tmp_path, capsys
):
    """THE BEFORE-PIN of this arc's routing change.

    State as of 2026-08-10, on the hybrid profile, through
    `LibraryLocalRagSearchService`:

      * unscoped -> `rag-hybrid`   (the engine's RRF fusion)
      * scoped   -> `rag-semantic` (diverted; `_search_rag`'s
        `hybrid, scoped -> semantic` arm, disclosed as
        `ROUTE_NOTE_HYBRID_SCOPED`)

    The AFTER state, once scope-aware hybrid lands later in this arc (the
    engine's semantic-only allowlist guard is removed for hybrid and the
    allowlists reach the FTS legs), is:

      * scoped   -> `rag-hybrid`, with no divert disclosure

    When that lands, this test FAILS — deliberately. The fix is to flip the
    expectation here to `rag-hybrid` and keep the control assertion, not to
    delete the test: the control is what proves the two arms are still
    distinguishable at all.

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

    # --- the pin: same query, scoped -> diverted to semantic ----------------
    pinned = outcomes[scoped.id]
    assert pinned.rows_returned > 0, (
        "the scoped search returned nothing at all, so its route says nothing "
        "about scoped retrieval"
    )
    assert pinned.runtime_backend == "rag-semantic", (
        f"a scoped query under the hybrid profile routed to "
        f"{pinned.runtime_backend!r}. If this is 'rag-hybrid', scope-aware "
        "hybrid has landed and this before-pin should be flipped to expect it "
        "(see this test's docstring) — do not delete the test."
    )
    assert pinned.route_notes, (
        "the divert was not disclosed; a silently diverted mode is exactly "
        "what this harness must never report as a hybrid measurement"
    )
    assert any("scope" in note.lower() for note in pinned.route_notes), (
        f"routing disclosures {pinned.route_notes} do not name the scope"
    )

    # --- and the scope really restricted retrieval ---------------------------
    assert set(pinned.retrieved_doc_ids) <= set(SCOPE_SLUGS), (
        f"scoped retrieval returned {sorted(set(pinned.retrieved_doc_ids) - set(SCOPE_SLUGS))} "
        "from OUTSIDE the scope — the allowlist did not reach the store"
    )
    assert TARGET_SLUG in pinned.retrieved_doc_ids, (
        "the scoped search missed its in-scope target, so a later 'scoped "
        "recall improved' claim would have nothing to improve on"
    )

    # The report must carry the route for review, not only this assertion.
    assert hybrid.runtime_backends == ("rag-hybrid", "rag-semantic"), (
        f"the mode's recorded backends {hybrid.runtime_backends} no longer "
        "show both routes in one pass"
    )
