"""Post-UX product roadmap handoff documentation regressions."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = Path("Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
BACKLOG_POINTER = Path("backlog/docs/product-maturity-roadmap.md")
PUBLIC_ROADMAP = Path("Docs/Product_Roadmap.md")
POST_UX_QA_README = Path("Docs/superpowers/qa/product-maturity/post-ux/README.md")
POST_UX_QA_TEMPLATE = Path("Docs/superpowers/qa/product-maturity/post-ux/walkthrough-template.md")


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_post_ux_spec_preserves_tracker_handoff_rules():
    text = _text(SPEC)

    assert "Relationship To Existing Roadmaps" in text
    assert "Verified work reuse rule" in text
    assert "Operational handoff rule" in text
    for task_id in ("TASK-10", "TASK-11", "TASK-12", "TASK-13"):
        assert task_id in text
    assert "Post-UX Reliability Rebaseline" in text
    assert "not reimplemented" in text


def test_product_tracker_maps_post_ux_roadmap_to_existing_tasks():
    text = _text(TRACKER)

    assert "Post-UX Roadmap Handoff" in text
    assert "Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md" in text
    assert "Product Maturity Phase 1 and Phase 2 are not reopened" in text
    for task_id in ("TASK-10", "TASK-11", "TASK-12", "TASK-13"):
        assert task_id in text
    for stage in (
        "Post-UX Reliability Rebaseline",
        "Source, Knowledge, And Artifact Loops",
        "Controlled Agent Configuration And Run Loops",
        "Monitoring And Cross-Loop Recovery",
        "Server Parity And Live Integrations",
        "Release Hardening And Distribution",
    ):
        assert stage in text


def test_backlog_pointer_names_execution_source_of_truth():
    text = _text(BACKLOG_POINTER)

    assert "2026-05-06-post-ux-product-roadmap-design.md" in text
    assert "canonical product-maturity tracker" in text
    assert "Do not create a parallel phase tree" in text


def test_public_roadmap_is_directional_and_commitment_free():
    text = _text(PUBLIC_ROADMAP)

    assert "local-first agentic knowledge console" in text
    assert "Now: Reliability And Product Confidence" in text
    assert "Next: Complete Workflow Loops" in text
    assert "Later: Server-Backed And Live Capabilities" in text
    assert "Deferred Tranche Gates" in text
    assert "ACP runtime launch" in text
    assert "write sync promotion" in text
    assert "Workspaces and Library depth" in text
    assert "citation and snippet carry-through" in text
    assert "optional dependency and package polish" in text
    assert "Always: Local-First Control" in text
    forbidden_patterns = (
        r"\bETA\b",
        r"\bdeadline\b",
        r"\bwill\s+ship\s+on\b",
        r"\bTASK-",
        r"\bPhase\s+1\.1\b",
        r"\bPhase\s+2\.5\b",
    )
    assert not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in forbidden_patterns)
