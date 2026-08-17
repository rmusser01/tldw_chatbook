"""What a Library Search/RAG retrieval score MEANS, and what may be banded.

The Library evidence list renders a match band ("match: strong") instead of
a raw number, and the band is a claim about COSINE SIMILARITY. Nothing else
the retrieval stack produces lives on that scale:

* hybrid retrieval fuses by RANK (RRF). A fused score's theoretical maximum
  is ``1 / (rrf_k + 1)`` -- about 0.17 at the shipped ``rrf_k = 5``
  (TASK-4110), and it was ~0.016 at the previous ``rrf_k = 60``. Either way
  it is below the 0.2 weak boundary, though no longer by an order of
  magnitude: the number is not a similarity at any k, which is why the kind
  and not the magnitude decides. Banding it on similarity thresholds
  rendered a wall of "match: weak (0.02)" on every hybrid search, perfect
  matches included (RAG-port P0/Task 6).
* reranker scores are unbounded (cross-encoder logits, 0-10 LLM scales); a
  value that happens to land inside [0, 1] is not a similarity either.

This module owns that vocabulary and the one rule for resolving it from a
result's provenance. It lives apart from ``library_rag_state`` so the
Console handoff builder (``UI/Views/RAGSearch/search_handoff.py``) can share
the rule: ``library_rag_state`` imports ``library_rag_answer_service``,
which imports that builder, so a direct import would close an import cycle.
It deliberately depends on nothing but the standard library.
"""

from __future__ import annotations

from typing import Any, Mapping


#: A plain vector-store cosine similarity -- the only kind that existed
#: before hybrid retrieval and reranking became reachable, and therefore the
#: default every pre-existing call site keeps.
LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY = "vector_similarity"
#: An RRF-fused rank score. Its similarity, when one exists, is the vector
#: leg preserved in ``metadata["hybrid_fusion"]["vector_score"]``.
LIBRARY_RAG_SCORE_KIND_HYBRID_FUSION = "hybrid_fusion"
#: A reranker model's output. Unbounded; never banded.
LIBRARY_RAG_SCORE_KIND_RERANKER = "reranker"
LIBRARY_RAG_SCORE_KINDS = frozenset(
    {
        LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY,
        LIBRARY_RAG_SCORE_KIND_HYBRID_FUSION,
        LIBRARY_RAG_SCORE_KIND_RERANKER,
    }
)

#: ``RAGService._fuse_hybrid_results``' per-leg provenance block.
HYBRID_FUSION_METADATA_KEY = "hybrid_fusion"
#: What a REAL reranked row carries. ``BaseReranker._apply_scores``
#: (``RAG_Search/reranker.py``) is the only reranking path that REPLACES a
#: row's score -- with the scoring model's relevance score, or by default a
#: weighted blend of it and the original similarity -- and it stamps this key
#: while doing so. It is therefore the production signal that "this score is
#: no longer a similarity". Two strategies take that path: ``pointwise``
#: (an LLM's 0-1 score) and ``cross_encoder`` (a local cross-encoder's
#: logits, min-max normalised into ``score_scale``; TASK-16965).
#:
#: It matters that this is keyed on score REPLACEMENT rather than on "a
#: reranker ran": ``PairwiseReranker`` and ``ListwiseReranker`` only
#: REORDER results, leaving ``score`` and ``metadata`` untouched, so their
#: rows still carry the retrieval similarity they came in with and must
#: keep being banded. Suppressing their band would throw away a true
#: number.
#:
#: For the same reason the stamp is PER ROW, not per rerank() call: a
#: pointwise run whose provider call failed for some candidates keeps those
#: rows' original scores, and ``_apply_scores`` leaves them unstamped
#: (``RerankingResult.scored is False``). It used to stamp them with the
#: original score, so a 14/15-failed rerank rendered " | reranked" on
#: fourteen rows no model ever looked at -- conservative about the NUMBER
#: but an over-claim about what happened (TASK-3502 note-b). A partly
#: degraded search therefore mixes kinds across its rows, which is the
#: honest rendering: each row states what was actually done to it.
#:
#: The danger this closes is quiet: ``RerankingConfig.score_scale``
#: defaults to ``(0.0, 1.0)``, so a default-configured pointwise reranker
#: emits scores INSIDE the similarity band range -- a 0.95 relevance score
#: would have rendered "match: strong", a cosine claim about a number that
#: is not a cosine.
RERANK_SCORE_METADATA_KEY = "rerank_score"
#: ``RAG_Search/local_citation_capture.py``'s ``FINAL_SCORE_KIND_KEY`` --
#: the canonical "what scale is the final score on" channel. Accepted as an
#: additional reranker signal (it costs nothing and is the vocabulary the
#: citation trace already speaks), but nothing in the app writes it today,
#: which is why the ``rerank_score`` stamp above is the primary marker.
#:
#: Deliberately NOT read as a kind signal: the ``reranking_skipped`` /
#: ``reranking_degraded`` tags (``enhanced_rag_service_v2.py``), which
#: disclose that reranking FAILED or partly failed. The scores on those
#: rows are the base retrieval scores, so treating either tag as a reranker
#: kind would hide a real similarity behind " | reranked".
FINAL_SCORE_KIND_METADATA_KEY = "_final_score_kind"
#: An explicit, already-resolved kind stated by a producer.
SCORE_KIND_METADATA_KEY = "score_kind"
VECTOR_SCORE_METADATA_KEY = "vector_score"


def coerce_optional_float(value: Any) -> float | None:
    """Return `value` as a float, or `None` when it isn't numeric.

    Args:
        value: Any candidate score value.

    Returns:
        The float, or `None` for `None`, `""`, and anything non-numeric.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_library_rag_score_kind(value: Any) -> str:
    """Canonicalize a score-kind label, defaulting to vector similarity.

    Args:
        value: A raw kind label from provenance, config, or a caller.

    Returns:
        A member of `LIBRARY_RAG_SCORE_KINDS`. An unrecognized value falls
        back to `LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY` -- the historical
        contract of every call site that predates score kinds, all of which
        pass a cosine similarity and no kind at all.
    """
    kind = str(value or "").strip().lower()
    if kind in LIBRARY_RAG_SCORE_KINDS:
        return kind
    return LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY


def library_rag_similarity_input(
    score: float | None,
    *,
    score_kind: str = LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY,
    vector_score: float | None = None,
) -> float | None:
    """Return the cosine similarity to band on, or `None` when none exists.

    The one seam that answers "is this number a similarity?" -- shared by
    `library_rag_score_suffix` (the per-row band) and
    `library_rag_all_matches_weak` (the all-weak coverage claim), so the
    band on screen and the sentence above it can never disagree.

    Args:
        score: The row's own retrieval score, whatever scale it is on.
        score_kind: One of `LIBRARY_RAG_SCORE_KINDS`; anything else is
            treated as `vector_similarity`.
        vector_score: The vector leg's preserved cosine similarity, for
            `hybrid_fusion` rows.

    Returns:
        The similarity to band, or `None` for a row that carries no
        similarity at all (reranked rows; FTS-leg-only hybrid rows;
        unscored keyword rows).
    """
    kind = normalize_library_rag_score_kind(score_kind)
    if kind == LIBRARY_RAG_SCORE_KIND_RERANKER:
        return None
    if kind == LIBRARY_RAG_SCORE_KIND_HYBRID_FUSION:
        return vector_score
    return score


def library_rag_result_score_kind(*candidates: Any) -> tuple[str, float | None]:
    """Resolve `(score_kind, vector_score)` from a result's metadata blocks.

    One derivation rule, shared by the panel row
    (`LibraryRagResultRow.from_result`) and the Console evidence bundle
    (`UI/Views/RAGSearch/search_handoff.py`), so the band on screen and the
    score staged into an answer can never disagree about what a number
    means.

    Resolution order, first match wins, across `candidates` in the order
    given: an explicit `score_kind`; the reranker markers (`rerank_score`,
    which the pointwise reranker stamps when it replaces the score, or the
    `_final_score_kind` channel); the presence of a `hybrid_fusion`
    provenance block. Anything else is a plain vector similarity.

    Reranking is checked BEFORE fusion because it runs after it: a hybrid
    search whose results were then reranked carries both blocks, and the
    later stage owns what the final score means.

    Args:
        *candidates: Mappings to consult in priority order (typically a
            row's `provenance`, then its engine `metadata`, then the raw
            result mapping). Non-mappings are skipped, so callers may pass
            whatever they have.

    Returns:
        The canonical score kind, plus the vector leg's similarity for
        `hybrid_fusion` rows (`None` for every other kind, and for an
        FTS-leg-only hybrid row).
    """
    mappings = [item for item in candidates if isinstance(item, Mapping)]

    def _first(key: str) -> Any:
        for mapping in mappings:
            if key in mapping:
                return mapping[key]
        return None

    explicit_kind = _first(SCORE_KIND_METADATA_KEY)
    final_kind = _first(FINAL_SCORE_KIND_METADATA_KEY)
    reranked = any(RERANK_SCORE_METADATA_KEY in mapping for mapping in mappings)

    if explicit_kind not in (None, ""):
        kind = normalize_library_rag_score_kind(explicit_kind)
    elif (
        reranked
        or str(final_kind or "").strip().lower() == LIBRARY_RAG_SCORE_KIND_RERANKER
    ):
        kind = LIBRARY_RAG_SCORE_KIND_RERANKER
    elif any(HYBRID_FUSION_METADATA_KEY in mapping for mapping in mappings):
        # Key present with ANY value, including a malformed one: the score
        # went through fusion, so it is not a similarity. Falling back to
        # "vector_similarity" on a malformed block would reopen the exact
        # defect this resolves -- banding a fused ~0.016 on cosine
        # thresholds.
        kind = LIBRARY_RAG_SCORE_KIND_HYBRID_FUSION
    else:
        kind = LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY

    if kind != LIBRARY_RAG_SCORE_KIND_HYBRID_FUSION:
        return kind, None
    fusion = _first(HYBRID_FUSION_METADATA_KEY)
    if isinstance(fusion, Mapping) and VECTOR_SCORE_METADATA_KEY in fusion:
        # Authoritative even when it is `None`: the fusion block states
        # per-leg presence, and `None` means the vector leg never returned
        # this chunk (an FTS-leg-only row -- no similarity exists).
        return kind, coerce_optional_float(fusion[VECTOR_SCORE_METADATA_KEY])
    return kind, coerce_optional_float(_first(VECTOR_SCORE_METADATA_KEY))


__all__ = [
    "FINAL_SCORE_KIND_METADATA_KEY",
    "HYBRID_FUSION_METADATA_KEY",
    "RERANK_SCORE_METADATA_KEY",
    "LIBRARY_RAG_SCORE_KINDS",
    "LIBRARY_RAG_SCORE_KIND_HYBRID_FUSION",
    "LIBRARY_RAG_SCORE_KIND_RERANKER",
    "LIBRARY_RAG_SCORE_KIND_VECTOR_SIMILARITY",
    "SCORE_KIND_METADATA_KEY",
    "VECTOR_SCORE_METADATA_KEY",
    "coerce_optional_float",
    "library_rag_result_score_kind",
    "library_rag_similarity_input",
    "normalize_library_rag_score_kind",
]
