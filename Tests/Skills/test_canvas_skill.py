"""The optional Canvas skill uses Chatbook metadata and real local trust."""

from pathlib import Path

import pytest

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
)

SKILL_DIR = Path(__file__).resolve().parents[2] / "Docs/Examples/skills/canvas"
SKILL_PATH = SKILL_DIR / "SKILL.md"


async def _import_canvas_skill(tmp_path):
    assert SKILL_PATH.is_file(), "The optional Canvas example skill is missing"
    trust = SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(
                tmp_path / "marker.json", store_dir=tmp_path
            ),
        ),
    )
    trust.unlock_with_passphrase("canvas-test-passphrase", salt=b"8" * 32)
    # Initialize an empty test store; importing must not implicitly trust files.
    trust.bootstrap_trust()
    service = LocalSkillsService(store_dir=tmp_path, trust_service=trust)
    imported = await service.import_skill_directory(SKILL_DIR, name="canvas")
    return service, trust, imported


@pytest.mark.asyncio
async def test_canvas_skill_requires_review_and_rechecks_exact_imported_content(
    tmp_path,
):
    service, trust, imported = await _import_canvas_skill(tmp_path)
    assert imported["content"] == SKILL_PATH.read_text(encoding="utf-8")
    assert imported["trust_blocked"] is True
    with pytest.raises(SkillTrustBlockedError):
        await service.execute_skill("canvas", args="Create a savings calculator")

    review = trust.capture_review("canvas")
    assert set(review["current_files"]) == {"SKILL.md"}
    trust.trust_reviewed_snapshot(review["review_id"])
    result = await service.execute_skill("canvas", args="Create a savings calculator")
    assert "Create a savings calculator" in result["rendered_prompt"]
    assert "{{args}}" not in result["rendered_prompt"]
    assert result["execution_mode"] == "inline"
    assert result["allowed_tools"] is None
    assert result["model_override"] is None
    assert result["fork_output"] is None
    assert "reference_files" not in result

    installed = tmp_path / "skills/canvas/SKILL.md"
    installed.write_text(
        imported["content"] + "\nChanged after review.\n", encoding="utf-8"
    )
    with pytest.raises(SkillTrustBlockedError, match="skill_modified"):
        await service.execute_skill("canvas", args="Revise the calculator")
    context = await service.get_context()
    assert context["available_skills"] == []
    assert context["blocked_skills"][0]["name"] == "canvas"


@pytest.mark.asyncio
async def test_canvas_skill_native_metadata_and_small_progressive_disclosure_body(
    tmp_path,
):
    _service, _trust, imported = await _import_canvas_skill(tmp_path)
    assert imported["name"] == "canvas"
    assert imported["description"] == (
        "Create or revise a requested Canvas using Chatbook's supported HTML, "
        "interactive controls, and offline diagrams."
    )
    assert imported["validation_status"] == "valid"
    assert imported["validation_errors"] == []
    assert imported["argument_hint"] == "requested Canvas or change"
    assert imported["context"] == "inline"
    assert imported["user_invocable"] is True
    assert imported["disable_model_invocation"] is True
    assert imported["model"] is None
    assert imported["allowed_tools"] is None
    frontmatter, body = imported["content"].split("---", 2)[1:]
    assert "model:" not in frontmatter
    assert "allowed_tools:" not in frontmatter
    assert "allowed-tools:" not in frontmatter
    assert 0 < len(body.encode("utf-8")) <= 4096
    assert "## User request\n\n{{args}}" in body
    assert {
        path.relative_to(SKILL_DIR).as_posix() for path in SKILL_DIR.rglob("*")
    } == {"SKILL.md"}
    assert "<!doctype html>" not in body.lower()
    assert "```" not in body


def test_canvas_skill_keeps_consent_profiles_and_recovery_explicit():
    assert SKILL_PATH.is_file(), "The optional Canvas example skill is missing"
    body = " ".join(SKILL_PATH.read_text(encoding="utf-8").split())
    for instruction in (
        "Offer Canvas only when a substantial visual or interaction materially helps.",
        "wait for acceptance before loading detailed guides, delegating, or generating artifact source",
        "A decline or no answer does not authorize creation",
        "do not repeat the same declined offer",
        "Explicit Canvas requests and requested edits already authorize that work",
        "bounded corrections",
        "unrelated artifacts or unrequested redesigns need a new offer",
        "If context does not establish consent, clarify",
        "Bare activation without a concrete request requires clarification before authoring",
        "find_tools",
        "load_tools",
        "canvas_guide",
        "only relevant topics after consent",
        "expected_parent_revision_id",
        "Exact-profile guidance and runtime checks take precedence",
        "only available evidence",
        "after one failed repair attempt, stop",
        "Never silently replace a refused skill invocation",
    ):
        assert instruction in body
