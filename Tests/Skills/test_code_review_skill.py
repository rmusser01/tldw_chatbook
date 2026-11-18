"""The optional review skill uses Chatbook metadata and real local trust."""

from pathlib import Path

import pytest

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
)

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "Docs/Examples/skills/multi-model-review"
)
SKILL_PATH = SKILL_DIR / "SKILL.md"


async def _import_review_skill(tmp_path):
    assert SKILL_PATH.is_file(), "The optional review example skill is missing"
    trust = SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(
                tmp_path / "marker.json", store_dir=tmp_path
            ),
        ),
    )
    trust.unlock_with_passphrase("multi-model-review-test-passphrase", salt=b"8" * 32)
    # Initialize an empty test store; importing must not implicitly trust files.
    trust.bootstrap_trust()
    service = LocalSkillsService(store_dir=tmp_path, trust_service=trust)
    imported = await service.import_skill_directory(
        SKILL_DIR, name="multi-model-review"
    )
    return service, trust, imported


@pytest.mark.asyncio
async def test_review_skill_requires_review_and_rechecks_exact_imported_content(
    tmp_path,
):
    service, trust, imported = await _import_review_skill(tmp_path)
    assert imported["content"] == SKILL_PATH.read_text(encoding="utf-8")
    assert imported["trust_blocked"] is True
    with pytest.raises(SkillTrustBlockedError):
        await service.execute_skill(
            "multi-model-review", args="Review the working tree"
        )

    review = trust.capture_review("multi-model-review")
    assert set(review["current_files"]) == {"SKILL.md"}
    trust.trust_reviewed_snapshot(review["review_id"])
    result = await service.execute_skill(
        "multi-model-review", args="Review the working tree"
    )
    assert "Review the working tree" in result["rendered_prompt"]
    assert "{{args}}" not in result["rendered_prompt"]
    assert result["execution_mode"] == "inline"
    assert result["allowed_tools"] is None
    assert result["model_override"] is None
    assert result["fork_output"] is None
    assert "reference_files" not in result

    installed = tmp_path / "skills/multi-model-review/SKILL.md"
    installed.write_text(
        imported["content"] + "\nChanged after review.\n", encoding="utf-8"
    )
    with pytest.raises(SkillTrustBlockedError, match="skill_modified"):
        await service.execute_skill("multi-model-review", args="Verify the review")
    context = await service.get_context()
    assert context["available_skills"] == []
    assert context["blocked_skills"][0]["name"] == "multi-model-review"


@pytest.mark.asyncio
async def test_review_skill_native_metadata_and_small_progressive_disclosure_body(
    tmp_path,
):
    _service, _trust, imported = await _import_review_skill(tmp_path)
    assert imported["name"] == "multi-model-review"
    assert imported["description"] == (
        "Coordinate independent reviews of an accepted captured target using the configured reviewer presets."
    )
    assert imported["validation_status"] == "valid"
    assert imported["validation_errors"] == []
    assert (
        imported["argument_hint"] == "review target and optional verification request"
    )
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
    assert "code_review_guide" in body
    assert "```" not in body
