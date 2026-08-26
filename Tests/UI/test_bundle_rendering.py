"""Bundle/artifact rendering for the Research window (task-16483)."""

from tldw_chatbook.UI.Research_Modules.bundle_rendering import (
    default_artifact_for_bundle,
    render_artifact,
    render_bundle_summary,
)

_LOCAL_BUNDLE = {
    "run": {
        "id": "run-1", "query": "What is RAG?", "status": "completed",
        "phase": "completed", "confidence": None,
    },
    "artifacts": [
        {"artifact_name": "plan.json", "content_type": "application/json",
         "content": {"query": "What is RAG?"}},
        {"artifact_name": "report_v1.md", "content_type": "text/markdown",
         "content": "# Report\nAnswer[1]."},
        {"artifact_name": "verification_summary.json", "content_type": "application/json",
         "content": {"confidence": 0.9, "relevant_count": 2,
                     "citation_verification": {"markers_total": 4, "markers_resolved": 4,
                                               "quotes_checked": 1, "quotes_verified": 1,
                                               "uncited_sentences": 1},
                     "gate": {"relevant": 4, "raw": 5, "fallback": False}}},
        {"artifact_name": "claims.json", "content_type": "application/json",
         "content": {"claims": [
             {"claim_id": "claim-1", "text": "Supported fact[1].", "status": "supported"},
             {"claim_id": "claim-2", "text": "Shaky[2?].", "status": "unverified"}],
             "claim_count": 2}},
        {"artifact_name": "sources.json", "content_type": "application/json",
         "content": {"evidence": [{"id": 1, "title": "One", "url": "https://one.example/"}]}},
        {"artifact_name": "budget_ledger.json", "content_type": "application/json",
         "content": {"searches_used": 2, "docs_used": 5, "tokens_settled": 40,
                     "runtime_elapsed_s": 12.5}},
    ],
}


def test_bundle_summary_local_shape_lists_run_and_inventory():
    out = render_bundle_summary(_LOCAL_BUNDLE)

    assert "run-1" in out and "completed" in out
    for name in ("plan.json", "report_v1.md", "verification_summary.json"):
        assert name in out


def test_bundle_summary_server_shape_and_empty():
    out = render_bundle_summary({"report_v1.md": "# R", "bundle.json": {}})
    assert "report_v1.md" in out and "bundle.json" in out

    assert render_bundle_summary(None) == "No bundle loaded."
    assert render_bundle_summary({}) != "No bundle loaded."


def test_default_artifact_prefers_report_and_never_the_run_record():
    assert default_artifact_for_bundle(_LOCAL_BUNDLE) == "report_v1.md"
    assert default_artifact_for_bundle({"run": {}, "artifacts": [
        {"artifact_name": "plan.json", "content": {}}]}) == "plan.json"
    assert default_artifact_for_bundle({"bundle.json": {}}) == "bundle.json"
    assert default_artifact_for_bundle(None) is None


def test_render_artifact_structured_per_known_type():
    def artifact(name, content, content_type="application/json"):
        return {"artifact_name": name, "content_type": content_type,
                "artifact_version": 1, "content": content}

    report = render_artifact(artifact("report_v1.md", "# Report\nAnswer[1].", "text/markdown"))
    assert "# Report" in report and "Answer[1]." in report

    verification = render_artifact(
        _LOCAL_BUNDLE["artifacts"][2]
    )
    assert "confidence: 0.9" in verification
    assert "markers 4/4" in verification and "quotes 1/1" in verification
    assert "gate: 4/5" in verification

    claims = render_artifact(_LOCAL_BUNDLE["artifacts"][3])
    assert "claim-1" in claims and "supported" in claims and "Supported fact[1]." in claims

    sources = render_artifact(_LOCAL_BUNDLE["artifacts"][4])
    assert "[1] One — https://one.example/" in sources

    budget = render_artifact(_LOCAL_BUNDLE["artifacts"][5])
    assert "searches 2" in budget and "docs 5" in budget and "40" in budget


def test_render_artifact_falls_back_to_pretty_json():
    out = render_artifact({"artifact_name": "custom.json", "content_type": "application/json",
                           "content": {"z": 1, "a": [1, 2]}})
    assert '"a"' in out and '"z"' in out

    assert render_artifact(None) == "No artifact loaded."


def test_render_artifact_namespace_tolerant():
    from types import SimpleNamespace

    out = render_artifact(SimpleNamespace(
        artifact_name="report_v1.md", content_type="text/markdown",
        artifact_version=2, content="Namespaced report."))
    assert "Namespaced report." in out and "Version: 2" in out
