"""Research-report self-eval scorer (task-16327).

Deterministic scoring of deep-search research reports from the verification
outcomes the pipeline already produces: task-16331's ``citation_verification``
payload (marker resolution, quote grounding, uncited sentences) and
task-16325's claims (supported/unverified). No LLM is consulted -- the
metrics are computed from the stored payload, so the same run always scores
the same way and pipeline changes are measurable against a baseline.

Metrics (all in [0, 1]):
- ``citation_accuracy``  -- resolved [n] markers / total markers (0.0 when
  the report cited nothing).
- ``quote_grounding``   -- verbatim-verified quotes / checked quotes (0.0
  when the report quoted nothing).
- ``claim_support_rate``-- supported claims / claims when per-claim detail
  exists; otherwise falls back to marker accuracy.
- ``cited_sentence_ratio`` -- cited sentences / all sentences (from marker
  and uncited-sentence counts).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

__all__ = [
    "BASELINE_VERIFICATION_PAYLOAD",
    "BASELINE_METRICS",
    "score_research_report",
]


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        num = float(numerator or 0)
        den = float(denominator or 0)
    except (TypeError, ValueError):
        return 0.0
    if den <= 0:
        return 0.0
    return max(0.0, min(1.0, num / den))


def score_research_report(verification: Mapping[str, Any]) -> Dict[str, float]:
    """Score one research report from its verification payload.

    Accepts either the flat ``citation_verification`` block or a full
    ``verification_summary`` artifact (the ``citation_verification`` key is
    unwrapped when present, so the live-baseline flow can hand over the
    whole summary including the gate block)."""
    if not isinstance(verification, Mapping):
        verification = {}
    if isinstance(verification.get("citation_verification"), Mapping):
        nested = dict(verification["citation_verification"])
        gate = verification.get("gate")
        if isinstance(gate, Mapping):
            nested.setdefault("gate", gate)
        verification = nested
    markers_total = verification.get("markers_total") or 0
    markers_resolved = verification.get("markers_resolved") or 0
    citation_accuracy = _ratio(markers_resolved, markers_total)

    quotes_checked = verification.get("quotes_checked") or 0
    quotes_verified = verification.get("quotes_verified") or 0
    quote_grounding = _ratio(quotes_verified, quotes_checked)

    claims = [
        claim
        for claim in (verification.get("claims") or [])
        if isinstance(claim, Mapping)
    ]
    if claims:
        supported = sum(1 for claim in claims if claim.get("status") == "supported")
        claim_support_rate = _ratio(supported, len(claims))
    else:
        claim_support_rate = citation_accuracy

    uncited_sentences = verification.get("uncited_sentences") or 0
    cited_sentence_ratio = _ratio(
        markers_total, float(markers_total or 0) + float(uncited_sentences or 0)
    )

    metrics = {
        "citation_accuracy": citation_accuracy,
        "quote_grounding": quote_grounding,
        "claim_support_rate": claim_support_rate,
        "cited_sentence_ratio": cited_sentence_ratio,
    }
    # Gate pass-rate (task-16333): relevant/raw from the gate block; present
    # only when the pipeline reported gate outcomes.
    gate = verification.get("gate")
    if isinstance(gate, Mapping):
        raw = gate.get("raw") or 0
        relevant = gate.get("relevant") or 0
        if raw:
            metrics["gate_pass_rate"] = _ratio(relevant, raw)
    return metrics


# Baseline (task-16327): the synthetic verification payload the recorded
# baseline was computed from. It pins the metric definitions -- a scorer
# regression moves these numbers -- and stands in until a live pipeline run
# (network + configured LLMs) records a production baseline the same way.
BASELINE_VERIFICATION_PAYLOAD: Dict[str, Any] = {
    "markers_total": 10,
    "markers_resolved": 8,
    "unknown_marker_ids": [11, 12],
    "quotes_checked": 4,
    "quotes_verified": 3,
    "quotes_misquoted": 1,
    "uncited_sentences": 6,
    "claims": [
        {"claim_id": "claim-1", "status": "supported"},
        {"claim_id": "claim-2", "status": "supported"},
        {"claim_id": "claim-3", "status": "supported"},
        {"claim_id": "claim-4", "status": "unverified"},
    ],
}

BASELINE_METRICS: Dict[str, float] = score_research_report(
    BASELINE_VERIFICATION_PAYLOAD
)


def aggregate_metrics(payloads: list) -> Dict[str, float]:
    """Mean metrics across a list of verification payloads (task-16330 --
    live-baseline aggregation); an empty list scores all zeros with
    ``sample_count`` 0."""
    scored = [score_research_report(payload) for payload in payloads]
    aggregate: Dict[str, float] = {"sample_count": float(len(scored))}
    keys = [
        "citation_accuracy",
        "quote_grounding",
        "claim_support_rate",
        "cited_sentence_ratio",
        "gate_pass_rate",
    ]
    for key in keys:
        # Per-key means over the payloads that CARRY the key (gate counts
        # only exist once the pipeline reports them) -- always-defined
        # metrics still average over every payload.
        values = [metric[key] for metric in scored if key in metric]
        if key == "gate_pass_rate":
            if values:
                aggregate[key] = sum(values) / len(values)
            continue
        aggregate[key] = (sum(values) / len(values)) if values else 0.0
    return aggregate
