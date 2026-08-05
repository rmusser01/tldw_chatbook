from __future__ import annotations

import inspect

import pytest

from tldw_chatbook.Chat.console_display_state import (
    console_prompted_source_count,
    console_staged_source_count,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
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


def _library_use_in_console_launch(result: dict, *, query: str) -> ConsoleLiveWorkLaunch:
    """Build the launch exactly as `library_screen.py::_stage_library_rag_result_in_console`
    does for one selected Library Search/RAG result: `opener(...,
    payload=build_library_rag_console_live_work_payload(selected_result,
    query=...), ...)`.
    """
    return ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title=result.get("title", ""),
        payload=build_library_rag_console_live_work_payload(result, query=query),
        status="staged",
        recovery="Review citations before sending.",
        action_label="Review evidence in Console",
    )


def test_library_use_in_console_launch_carries_a_non_empty_evidence_bundle():
    """D1c: Library's 'Use in Console' must never stage a bundleless launch.

    A bundleless launch makes `console_staged_source_count` fall through to
    the historical literal `1` and makes `console_prompted_source_count`
    return `0` (a Library-staged send could never be prompt-captured).
    """
    result = {
        "result_id": "note-42:chunk-7",
        "title": "Incident Review",
        "snippet": "Expired credential caused the incident.",
        "source_id": "note-42",
        "chunk_id": "chunk-7",
        "score": 0.93,
        "runtime_backend": "local-fts",
    }
    launch = _library_use_in_console_launch(
        result, query="Why did the incident happen?"
    )

    assert launch.payload.get("evidence_bundle")
    assert launch.payload["evidence_bundle"]["references"]


def test_library_use_in_console_chip_and_prompted_counts_are_honest():
    """The staged-context chip and the send-time prompt count must both be
    DERIVED from the real bundle for a Library-staged launch, not a
    fallthrough literal.

    `console_staged_source_count` reading `1` is ambiguous by itself (the
    pre-fix literal-`1` fallback and an honest 1-reference bundle look the
    same at n=1). `console_prompted_source_count` is not: it reads `0` for
    a bundleless launch and only counts *available* local references
    otherwise, so pairing an available result with a blocked one proves
    both counts come from the bundle's actual reference statuses.
    """
    available_result = {
        "result_id": "note-42:chunk-7",
        "title": "Incident Review",
        "snippet": "Expired credential caused the incident.",
        "source_id": "note-42",
        "chunk_id": "chunk-7",
        "runtime_backend": "local-fts",
    }
    launch = _library_use_in_console_launch(
        available_result, query="Why did the incident happen?"
    )
    assert console_staged_source_count(launch) == 1
    assert console_prompted_source_count(launch) == 1

    blocked_result = {
        "result_id": "note-9:chunk-1",
        "title": "Other Workspace Note",
        "snippet": "This source belongs to another workspace.",
        "source_id": "note-9",
        "chunk_id": "chunk-1",
        "runtime_backend": "local-fts",
        "workspace_ids": ["workspace-b"],
        "active_workspace_id": "workspace-a",
    }
    blocked_launch = _library_use_in_console_launch(
        blocked_result, query="What does this workspace note say?"
    )
    # Still one staged reference (the tray/chip must still say "1 staged") --
    # but zero of it can reach the model, proving `console_prompted_source_
    # count` reads the bundle's own status, not a hardcoded pass-through.
    assert console_staged_source_count(blocked_launch) == 1
    assert console_prompted_source_count(blocked_launch) == 0


def test_library_use_in_console_bundle_matches_console_run_builder_shape():
    """Library and Console-Run staging must use the SAME builder so the
    payload shape is byte-compatible, not a hand-rolled equivalent."""
    result = {
        "result_id": "note-42:chunk-7",
        "title": "Incident Review",
        "snippet": "Expired credential caused the incident.",
        "source_id": "note-42",
        "chunk_id": "chunk-7",
        "score": 0.93,
        "runtime_backend": "local-fts",
    }
    query = "Why did the incident happen?"
    launch = _library_use_in_console_launch(result, query=query)

    # Console's own Run path builds
    # `build_library_rag_evidence_bundle(outcome.results, query=...).to_payload()`
    # for a single-result outcome; Library's path embeds the identical bundle
    # via the shared `build_library_rag_console_live_work_payload` builder.
    console_run_bundle_payload = build_library_rag_evidence_bundle(
        [result], query=query
    ).to_payload()

    assert launch.payload["evidence_bundle"] == console_run_bundle_payload
