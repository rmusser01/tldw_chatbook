"""Mechanical completeness checks for the Speech & TTS closeout evidence."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / (
    "Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md"
)
EVIDENCE_PATH = REPO_ROOT / (
    "Docs/superpowers/qa/speech-tts-settings-ownership-2026-08-01/release-evidence.md"
)
MODULE_GUIDE_PATH = REPO_ROOT / "Docs/Development/TTS/TTS_MODULE_GUIDE.md"
SPEECH_GUIDE_PATH = REPO_ROOT / "Docs/Features/Speech-Services-Guide.md"

REQUIREMENT_HEADING = re.compile(
    r"^### ((?:IA|OWN|CFG|CAT|STATE|MIG|SEC|A11Y)-\d{3}) — ",
    re.MULTILINE,
)
EVIDENCE_ROW = re.compile(
    r"^\|\s*((?:IA|OWN|CFG|CAT|STATE|MIG|SEC|A11Y)-\d{3})\s*\|"
    r"\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$",
    re.MULTILINE,
)
TEST_NODE = re.compile(r"(Tests/[A-Za-z0-9_./-]+\.py)::(test_[A-Za-z0-9_]+)")
UAT_REFERENCE = re.compile(r"\bUAT-(?:0[1-9]|10)\b")


def _requirement_ids() -> tuple[str, ...]:
    return tuple(REQUIREMENT_HEADING.findall(SPEC_PATH.read_text(encoding="utf-8")))


def _assert_test_node_exists(relative_path: str, test_name: str) -> None:
    test_path = REPO_ROOT / relative_path
    assert test_path.is_file(), (
        f"Evidence references missing test file: {relative_path}"
    )
    source = test_path.read_text(encoding="utf-8")
    definition = re.compile(
        rf"^(?:async\s+)?def\s+{re.escape(test_name)}\s*\(",
        re.MULTILINE,
    )
    assert definition.search(source), (
        f"Evidence references missing test node: {relative_path}::{test_name}"
    )


def test_release_evidence_maps_every_approved_requirement_once() -> None:
    """Every stable PRD ID has one auditable evidence row and no invented IDs."""

    requirements = _requirement_ids()
    assert len(requirements) == len(set(requirements))
    assert EVIDENCE_PATH.is_file(), "TASK-1988 release evidence has not been created"

    evidence = EVIDENCE_PATH.read_text(encoding="utf-8")
    rows = EVIDENCE_ROW.findall(evidence)
    row_ids = [requirement_id for requirement_id, *_rest in rows]

    assert len(row_ids) == len(set(row_ids)), (
        "Evidence contains duplicate requirement rows"
    )
    assert set(row_ids) == set(requirements), (
        f"Missing: {sorted(set(requirements) - set(row_ids))}; "
        f"unknown: {sorted(set(row_ids) - set(requirements))}"
    )

    for requirement_id, evidence_kind, references, result in rows:
        test_nodes = TEST_NODE.findall(references)
        uat_references = UAT_REFERENCE.findall(references)
        assert test_nodes or uat_references, (
            f"{requirement_id} has no focused test, end-to-end test, or UAT journey"
        )
        assert evidence_kind.strip() in {"Automated", "Automated + UAT", "Manual UAT"}
        kind = evidence_kind.strip()
        result_copy = result.strip()
        if kind == "Automated":
            assert result_copy == "Passing"
        elif kind == "Automated + UAT":
            assert result_copy.startswith("Automated passing; live UAT passed"), (
                f"{requirement_id} must distinguish passing automated and live evidence"
            )
        else:
            assert result_copy.startswith(
                "Passed with explicit human audible confirmation"
            ), f"{requirement_id} must retain explicit human audible evidence"
        for relative_path, test_name in test_nodes:
            _assert_test_node_exists(relative_path, test_name)

    assert "UAT pending TASK-1989" not in evidence
    assert "TASK-1989 has passed UAT-01 through UAT-10" in evidence
    assert "`pocket-tts-en` and\n`supertonic-3`" in evidence
    assert "literal multi-model clause remains blocked" not in evidence


def test_release_evidence_keeps_headless_and_audible_claims_separate() -> None:
    """Deterministic CI must not be presented as human audible acceptance."""

    evidence = EVIDENCE_PATH.read_text(encoding="utf-8")
    assert "Headless complete-WAV and playback-handoff proof: Passing" in evidence
    assert (
        "Human audible playback proof: Passed in TASK-1989 with explicit user"
        in evidence
    )
    assert "separately identified" in evidence
    assert (
        "No provider process, provider network, model download, or audio hardware"
        in (evidence)
    )
    assert "does not claim audible output or incremental streaming" in evidence


def test_clone_reference_guides_document_privacy_downgrade_and_scope_truth() -> None:
    module_guide = MODULE_GUIDE_PATH.read_text(encoding="utf-8")
    speech_guide = SPEECH_GUIDE_PATH.read_text(encoding="utf-8")
    combined = f"{module_guide}\n{speech_guide}"
    normalized = combined.casefold()

    for required in (
        "local plaintext",
        "not encryption",
        "best-effort deletion",
        "not forensic erasure",
        "TASK-13208",
        "retained v2 pre-migration backup",
        "loss of post-migration profile changes",
        "explicit clone transfer",
        "warning-gated",
        "voice-bundle portability",
    ):
        assert required.casefold() in normalized
    assert "source path is never persisted" in normalized
    assert "reference audio and transcript" in normalized
    assert "profile database backups contain" in normalized
