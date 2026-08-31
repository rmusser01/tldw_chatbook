from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Agents.agent_lesson_promotion import (
    ManagedSkillProposal,
    PromotionEvidence,
    RepositoryInstructionProposal,
    assess_promotion_evidence,
    classify_managed_skill_target,
    classify_repository_instruction_target,
    sha256_text,
)


def _evidence(**changes: object) -> PromotionEvidence:
    values: dict[str, object] = {
        "lesson_note_ids": ("note_1",),
        "summary": "Use an expected digest before replacing a reviewed file.",
        "provenance": "Reproduced against the production file-write boundary.",
        "verification": "The stale-state regression passes deterministically.",
        "principle": "Bind approval to the state the reviewer actually saw.",
        "rationale": "Otherwise an intervening edit can be overwritten.",
        "procedural": True,
        "reusable": True,
        "independently_verified": True,
    }
    values.update(changes)
    return PromotionEvidence(**values)


def test_verified_reusable_evidence_is_eligible_without_incident_threshold() -> None:
    outcome = assess_promotion_evidence(_evidence())

    assert outcome.eligible is True
    assert outcome.reason_code == "eligible"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"independently_verified": False}, "unverified_evidence"),
        ({"contradictory": True}, "contradictory_evidence"),
        ({"procedural": False}, "not_procedural"),
        ({"reusable": False}, "not_reusable"),
        ({"interaction_specific": True}, "interaction_specific"),
        ({"rationale": ""}, "missing_rationale"),
    ],
)
def test_ineligible_evidence_has_stable_non_sensitive_reason(
    changes: dict[str, object], reason: str
) -> None:
    outcome = assess_promotion_evidence(_evidence(**changes))

    assert outcome.eligible is False
    assert outcome.reason_code == reason


def test_credential_bearing_evidence_is_ineligible() -> None:
    outcome = assess_promotion_evidence(
        _evidence(summary="secret_key: abcdefghijklmnop")
    )

    assert outcome.eligible is False
    assert outcome.reason_code == "credential_material_detected"


@pytest.mark.parametrize("name", ["AGENTS.md", "AGENTS.override.md"])
def test_repository_target_is_limited_to_instruction_files_in_writable_binding(
    tmp_path: Path, name: str
) -> None:
    outcome = classify_repository_instruction_target(
        binding_root=tmp_path,
        target_path=tmp_path / "nested" / name,
        binding_id="binding-1",
        locator_fingerprint="fingerprint",
        writable=True,
    )

    assert outcome.eligible is True
    assert outcome.mode == "reviewed_apply"


@pytest.mark.parametrize(
    ("target", "writable", "reason"),
    [
        ("README.md", True, "ineligible_target"),
        ("../AGENTS.md", True, "outside_binding"),
        ("AGENTS.md", False, "binding_read_only"),
    ],
)
def test_repository_target_refuses_ineligible_authority(
    tmp_path: Path, target: str, writable: bool, reason: str
) -> None:
    outcome = classify_repository_instruction_target(
        binding_root=tmp_path,
        target_path=tmp_path / target,
        binding_id="binding-1",
        locator_fingerprint="fingerprint",
        writable=writable,
    )

    assert outcome.eligible is False
    assert outcome.reason_code == reason


def test_managed_local_skills_are_proposal_only() -> None:
    outcome = classify_managed_skill_target(
        owner="local",
        readable=True,
        managed=True,
    )

    assert outcome.eligible is True
    assert outcome.mode == "proposal_only"


@pytest.mark.parametrize("owner", ["builtin", "runtime", "server"])
def test_non_local_skill_owners_are_ineligible(owner: str) -> None:
    outcome = classify_managed_skill_target(
        owner=owner,
        readable=True,
        managed=True,
    )

    assert outcome.eligible is False
    assert outcome.reason_code == "ineligible_skill_owner"


def test_repository_proposal_digest_binds_complete_exact_preview() -> None:
    proposal = RepositoryInstructionProposal.build(
        evidence=_evidence(),
        binding_id="binding-1",
        locator_fingerprint="fingerprint",
        root_identity="root-id",
        target_path="AGENTS.md",
        effective_chain=(("AGENTS.md", "standard", "a" * 64),),
        effective_chain_digest="b" * 64,
        expected_sha256="c" * 64,
        expected_absent=False,
        replacement_content="# Instructions\n",
        bounded_diff="--- AGENTS.md\n+++ AGENTS.md\n",
        verification_command="pytest -q Tests/Agents/test_agent_lesson_promotion.py",
        verification_text="Focused promotion contract passes.",
    )

    assert proposal.replacement_sha256 == sha256_text("# Instructions\n")
    assert len(proposal.proposal_digest) == 64
    changed = RepositoryInstructionProposal.build(
        evidence=_evidence(),
        binding_id="binding-1",
        locator_fingerprint="fingerprint",
        root_identity="root-id",
        target_path="AGENTS.md",
        effective_chain=(("AGENTS.md", "standard", "a" * 64),),
        effective_chain_digest="b" * 64,
        expected_sha256="c" * 64,
        expected_absent=False,
        replacement_content="# Different\n",
        bounded_diff="--- AGENTS.md\n+++ AGENTS.md\n",
        verification_command="pytest -q Tests/Agents/test_agent_lesson_promotion.py",
        verification_text="Focused promotion contract passes.",
    )
    assert changed.proposal_digest != proposal.proposal_digest


def test_managed_skill_proposal_binds_version_trust_and_exact_replacement() -> None:
    proposal = ManagedSkillProposal.build(
        evidence=_evidence(),
        skill_public_id="skill-1",
        skill_name="example",
        expected_version=4,
        expected_trust_state="trusted",
        current_content="# Existing\n",
        replacement_content="# Improved\n",
        verification="Read through the Library editor after manual application.",
    )

    assert proposal.mode == "proposal_only"
    assert proposal.expected_version == 4
    assert proposal.current_sha256 == sha256_text("# Existing\n")
    assert proposal.replacement_sha256 == sha256_text("# Improved\n")
    assert len(proposal.proposal_digest) == 64
