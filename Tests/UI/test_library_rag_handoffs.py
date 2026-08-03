from __future__ import annotations

import inspect

import pytest

from tldw_chatbook.UI.Views.RAGSearch.search_handoff import (
    build_library_rag_evidence_bundle,
    build_library_rag_console_live_work_payload,
)


def test_library_rag_console_payload_preserves_evidence_fields():
    result = {
        "result_id": "note-42:chunk-7",
        "title": "Incident Review",
        "snippet": "Expired credential caused the incident.",
        "source_id": "note-42",
        "chunk_id": "chunk-7",
        "score": 0.93,
        "runtime_backend": "local-fts",
        "citations": [{"label": "Incident Review p.2"}],
    }

    payload = build_library_rag_console_live_work_payload(
        result,
        query="Why did the incident happen?",
    )

    assert {
        key: payload[key]
        for key in (
            "target_id",
            "result_id",
            "query",
            "title",
            "source_id",
            "chunk_id",
            "snippet",
            "citations",
            "score",
            "runtime_backend",
            "source_authority",
            "source_selector_state",
        )
    } == {
        "target_id": "local:library-rag:note-42:chunk-7",
        "result_id": "note-42:chunk-7",
        "query": "Why did the incident happen?",
        "title": "Incident Review",
        "source_id": "note-42",
        "chunk_id": "chunk-7",
        "snippet": "Expired credential caused the incident.",
        "citations": ["Incident Review p.2"],
        "score": 0.93,
        "runtime_backend": "local-fts",
        "source_authority": "local",
        "source_selector_state": "local",
    }
    bundle = payload["evidence_bundle"]
    reference = bundle["references"][0]
    assert bundle["query"] == "Why did the incident happen?"
    assert bundle["status"] == "available"
    assert reference["evidence_id"] == "S1"
    assert reference["source_id"] == "note-42"
    assert reference["snippet"] == "Expired credential caused the incident."
    assert reference["authority_label"] == "Source authority: local"
    assert reference["metadata"]["active_context_eligible"] is True
    assert reference["metadata"]["global_browse_visible"] is True


def test_library_rag_evidence_bundle_blocks_cross_workspace_context():
    bundle = build_library_rag_evidence_bundle(
        {
            "result_id": "note-42:chunk-7",
            "title": "Workspace B Note",
            "snippet": "Workspace B evidence remains visible.",
            "source_id": "note-42",
            "chunk_id": "chunk-7",
            "source_type": "note",
            "runtime_backend": "local-fts",
            "workspace_ids": ("workspace-b",),
            "active_workspace_id": "workspace-a",
        },
        query="Can I use this in Workspace A?",
    )

    payload = bundle.to_payload()
    reference = payload["references"][0]
    assert payload["status"] == "blocked"
    assert reference["status"] == "blocked"
    assert reference["workspace_id"] == "workspace-b"
    assert reference["authority_label"] == (
        "Workspace: workspace-b (blocked for active workspace workspace-a)"
    )
    assert reference["metadata"]["global_browse_visible"] is True
    assert reference["metadata"]["active_context_eligible"] is False
    assert reference["metadata"]["eligibility_reason"] == "cross_workspace"
    assert reference["metadata"]["active_workspace_id"] == "workspace-a"


def test_library_rag_evidence_bundle_preserves_provenance_identity():
    bundle = build_library_rag_evidence_bundle(
        {
            "title": "Server Transcript",
            "snippet": "The server transcript contains the source evidence.",
            "provenance": {
                "source_id": "media-9",
                "chunk_id": "chunk-3",
                "source_type": "media",
                "runtime_backend": "server-rag",
            },
        },
        query="What does the transcript say?",
    )

    payload = bundle.to_payload()
    reference = payload["references"][0]
    assert reference["source_id"] == "media-9"
    assert reference["source_type"] == "media"
    assert reference["source_owner"] == "server"
    assert reference["content_ref"] == "server:library-rag:media-9:chunk-3"
    assert reference["metadata"]["chunk_id"] == "chunk-3"
    assert reference["metadata"]["runtime_backend"] == "server-rag"


def test_library_rag_evidence_bundle_marks_empty_results_missing():
    bundle = build_library_rag_evidence_bundle(
        [],
        query="What evidence is available?",
    )

    payload = bundle.to_payload()
    assert payload["status"] == "missing"
    assert payload["references"] == []
    assert payload["metadata"]["eligible_reference_count"] == 0
    assert payload["metadata"]["blocked_reference_count"] == 0


@pytest.mark.parametrize("status", ("stale", "unknown"))
def test_library_rag_evidence_bundle_preserves_non_missing_reference_status(status):
    bundle = build_library_rag_evidence_bundle(
        {
            "title": "Indexed source",
            "snippet": "Existing evidence has a non-ready state.",
            "source_id": f"{status}-source",
            "evidence_status": status,
        },
        query=f"Show {status} evidence",
    )

    payload = bundle.to_payload()
    reference = payload["references"][0]
    assert payload["status"] == status
    assert reference["status"] == status


def test_library_rag_evidence_bundle_drops_out_of_range_scores():
    payload = build_library_rag_console_live_work_payload(
        {
            "title": "Out of range score",
            "snippet": "Score should not prevent staging evidence.",
            "source_id": "note-99",
            "score": 1.5,
        },
        query="Can this evidence stage?",
    )

    reference = payload["evidence_bundle"]["references"][0]
    assert payload["score"] is None
    assert "score" not in reference


def test_library_rag_console_payload_uses_shared_validation_for_unsafe_text():
    result = {
        "result_id": "note-42:chunk-7",
        "title": "<script>alert('bad')</script>",
        "snippet": "javascript:alert(1)",
        "source_id": "note-42<script>",
        "chunk_id": "chunk-7",
        "runtime_backend": "server-rag",
        "citations": [{"label": "onclick=bad"}],
    }

    payload = build_library_rag_console_live_work_payload(
        result,
        query="javascript:alert(1)",
    )

    assert payload["target_id"] == "server:library-rag:note-42:chunk-7"
    assert payload["result_id"] == "note-42:chunk-7"
    assert payload["query"] == ""
    assert payload["title"] == "Untitled source"
    assert payload["source_id"] == ""
    assert payload["chunk_id"] == "chunk-7"
    assert payload["snippet"] == ""
    assert payload["citations"] == []
    assert payload["source_authority"] == "server"
    assert payload["target_id"].startswith(f"{payload['source_authority']}:")


def test_library_rag_console_payload_target_prefix_matches_source_authority():
    local_payload = build_library_rag_console_live_work_payload(
        {
            "result_id": "local-note:chunk-1",
            "title": "Local Note",
            "runtime_backend": "local-fts",
        },
        query="local query",
    )
    server_payload = build_library_rag_console_live_work_payload(
        {
            "result_id": "server-note:chunk-1",
            "title": "Server Note",
            "runtime_backend": "server-rag",
        },
        query="server query",
    )

    assert local_payload["source_authority"] == "local"
    assert local_payload["target_id"].startswith("local:library-rag:")
    assert local_payload["target_id"].startswith(
        f"{local_payload['source_authority']}:"
    )
    assert server_payload["source_authority"] == "server"
    assert server_payload["target_id"].startswith("server:library-rag:")
    assert server_payload["target_id"].startswith(
        f"{server_payload['source_authority']}:"
    )


def test_library_rag_console_payload_helper_documents_contract():
    docstring = inspect.getdoc(build_library_rag_console_live_work_payload)

    assert docstring is not None
    assert "Args:" in docstring
    assert "Returns:" in docstring
