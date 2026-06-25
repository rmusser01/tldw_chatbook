import asyncio
import io
import zipfile

import pytest

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
)


SKILL_WITH_METADATA = """---
description: Summarize notes
argument_hint: note id
allowed_tools:
  - notes.read
model: local-model
context: fork
user_invocable: true
disable_model_invocation: false
---
# Summarize
Summarize {{args}}.
"""

AGENT_SKILL_WITH_METADATA = """---
name: summarize-notes
description: Summarize note collections. Use when a user asks for a concise notes summary.
allowed-tools: Read Bash(git:*)
---
# Summarize
Summarize {{args}}.
"""

INVALID_AGENT_SKILL = """---
name: Summarize Notes
description: ""
---
# Invalid
Missing valid Agent Skills metadata.
"""


def _trusted_local_service(tmp_path):
    trust_service = SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )
    trust_service.unlock_with_passphrase("passphrase", salt=b"7" * 32)
    return LocalSkillsService(store_dir=tmp_path, trust_service=trust_service), trust_service


@pytest.mark.asyncio
async def test_local_skills_service_persists_skill_metadata(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)

    created = await service.create_skill(name="summarize-notes", content=SKILL_WITH_METADATA)
    reloaded = LocalSkillsService(store_dir=tmp_path)
    loaded = await reloaded.get_skill("summarize-notes")
    listed = await reloaded.list_skills()
    context = await reloaded.get_context()

    assert created["version"] == 1
    assert loaded["description"] == "Summarize notes"
    assert loaded["argument_hint"] == "note id"
    assert loaded["allowed_tools"] == ["notes.read"]
    assert loaded["model"] == "local-model"
    assert loaded["context"] == "fork"
    assert loaded["user_invocable"] is True
    assert loaded["disable_model_invocation"] is False
    assert loaded["trust_status"] == "trusted"
    assert loaded["trust_blocked"] is False
    assert listed["skills"][0]["name"] == "summarize-notes"
    assert listed["skills"][0]["description"] == "Summarize notes"
    assert listed["skills"][0]["trust_status"] == "trusted"
    assert listed["skills"][0]["trust_blocked"] is False
    assert context["available_skills"][0]["argument_hint"] == "note id"
    assert "- summarize-notes" in context["context_text"]


@pytest.mark.asyncio
async def test_local_skills_service_validates_agent_skill_metadata_contract(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)

    created = await service.create_skill(name="summarize-notes", content=AGENT_SKILL_WITH_METADATA)
    listed = await service.list_skills()
    context = await service.get_context()

    assert created["validation_status"] == "valid"
    assert created["validation_errors"] == []
    assert created["agent_skill_name"] == "summarize-notes"
    assert created["description"] == "Summarize note collections. Use when a user asks for a concise notes summary."
    assert created["allowed_tools"] == ["Read", "Bash(git:*)"]
    assert listed["skills"][0]["validation_status"] == "valid"
    assert listed["skills"][0]["validation_errors"] == []
    assert context["available_skills"][0]["validation_status"] == "valid"


@pytest.mark.asyncio
async def test_local_skills_service_sanitizes_agent_skill_frontmatter_fields(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)
    content = """---
name: summarize-notes
description: Summarize notes.
license: "Apache-2.0\u0000"
compatibility: "<script>alert(1)</script>"
metadata:
  author: "Example\u0000Org"
  nested:
    ignored: true
  long: "{}"
---

Summarize the selected notes.
""".format("x" * 1200)

    created = await service.create_skill(name="summarize-notes", content=content)

    assert created["license"] == "Apache-2.0"
    assert "compatibility" not in created
    assert created["metadata"] == {
        "author": "ExampleOrg",
        "long": "x" * 1000,
    }


@pytest.mark.asyncio
async def test_local_skills_service_accepts_agent_skill_name_starting_with_number(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)
    content = """---
name: 1-summary
description: Summarize notes. Use when a user asks for a concise notes summary.
---

Summarize the selected notes.
"""

    created = await service.create_skill(name="1-summary", content=content)

    assert created["name"] == "1-summary"
    assert created["agent_skill_name"] == "1-summary"
    assert created["validation_status"] == "valid"
    assert created["validation_errors"] == []


@pytest.mark.asyncio
async def test_local_skills_service_import_preserves_digit_leading_agent_skill_name(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)
    content = b"""---
name: 1-summary
description: Summarize notes. Use when a user asks for a concise notes summary.
---

Summarize the selected notes.
"""

    imported = await service.import_skill_file(
        content,
        filename="1-summary.md",
        content_type="text/markdown",
    )

    assert imported["name"] == "1-summary"
    assert imported["agent_skill_name"] == "1-summary"
    assert imported["validation_status"] == "valid"
    assert imported["validation_errors"] == []


@pytest.mark.asyncio
async def test_local_skills_service_marks_over_schema_limit_description_invalid_without_crashing(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)
    oversized_description = "x" * 1001
    content = f"""---
name: summarize-notes
description: {oversized_description}
---

Summarize the selected notes.
"""

    created = await service.create_skill(name="summarize-notes", content=content)

    assert created["description"] == "x" * 1000
    assert created["validation_status"] == "invalid"
    assert "description must be 1000 characters or fewer" in created["validation_errors"]


@pytest.mark.asyncio
async def test_local_skills_service_reports_invalid_agent_skill_metadata_without_mutating_content(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)

    created = await service.create_skill(name="summarize-notes", content=INVALID_AGENT_SKILL)
    listed = await service.list_skills()
    loaded = await service.get_skill("summarize-notes")

    assert created["validation_status"] == "invalid"
    assert "name must use lowercase letters, numbers, and hyphens" in created["validation_errors"]
    assert "description is required" in created["validation_errors"]
    assert listed["skills"][0]["validation_status"] == "invalid"
    assert loaded["content"] == INVALID_AGENT_SKILL


@pytest.mark.asyncio
async def test_local_skills_service_uses_deterministic_metadata_defaults(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)

    await service.create_skill(name="draft-helper", content="# Draft Helper\n\nHelps rewrite local drafts.")
    loaded = await service.get_skill("draft-helper")

    assert loaded["description"] == "Helps rewrite local drafts."
    assert loaded["argument_hint"] is None
    assert loaded["allowed_tools"] is None
    assert loaded["model"] is None
    assert loaded["context"] == "inline"
    assert loaded["user_invocable"] is True
    assert loaded["disable_model_invocation"] is False


@pytest.mark.asyncio
async def test_local_skills_service_blocks_stale_expected_version(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nInitial")

    await service.update_skill("demo-skill", content="# Demo\nUpdated", expected_version=1)

    with pytest.raises(ValueError, match="local_skill_version_conflict:demo-skill"):
        await service.update_skill("demo-skill", content="# Demo\nStale", expected_version=1)


@pytest.mark.asyncio
async def test_local_skills_service_serializes_concurrent_updates(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nInitial")

    await asyncio.gather(
        service.update_skill("demo-skill", content="# Demo\nA"),
        service.update_skill("demo-skill", supporting_files={"a.md": "A"}),
    )

    loaded = await service.get_skill("demo-skill")
    assert loaded["version"] == 3
    assert loaded["supporting_files"]["a.md"] == "A"


@pytest.mark.asyncio
async def test_local_skills_service_rejects_unsafe_supporting_file_names(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)

    with pytest.raises(ValueError, match="Invalid supporting file name"):
        await service.create_skill(
            name="unsafe-skill",
            content="# Unsafe",
            supporting_files={"../escape.md": "bad"},
        )


@pytest.mark.asyncio
async def test_local_skills_service_import_export_round_trip(tmp_path):
    source = LocalSkillsService(store_dir=tmp_path / "source")
    created = await source.import_skill(
        name="rewrite-draft",
        content="# Rewrite\nRewrite {{args}}.",
        supporting_files={"style.md": "Use concise language."},
    )

    exported = await source.export_skill("rewrite-draft")
    assert exported["filename"] == "rewrite-draft.zip"
    assert exported["content_type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(exported["content"]), "r") as archive:
        assert sorted(archive.namelist()) == ["SKILL.md", "style.md"]

    target = LocalSkillsService(store_dir=tmp_path / "target")
    imported = await target.import_skill_file(
        exported["content"],
        filename=exported["filename"],
        content_type=exported["content_type"],
    )

    assert created["name"] == "rewrite-draft"
    assert imported["name"] == "rewrite-draft"
    assert (await target.get_skill("rewrite-draft"))["supporting_files"] == {"style.md": "Use concise language."}


@pytest.mark.asyncio
async def test_local_skills_service_import_skill_file_derives_name_from_markdown_filename(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)

    imported = await service.import_skill_file(
        b"# File Skill\nRender {{args}}",
        filename="file-skill.md",
        content_type="text/markdown",
    )

    assert imported["name"] == "file-skill"


@pytest.mark.asyncio
async def test_local_skills_service_execute_renders_prompt_without_model_invocation(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)
    await service.create_skill(name="summarize-notes", content=SKILL_WITH_METADATA)

    result = await service.execute_skill("summarize-notes", args="note-1")

    assert result == {
        "skill_name": "summarize-notes",
        "rendered_prompt": "# Summarize\nSummarize note-1.",
        "allowed_tools": ["notes.read"],
        "model_override": "local-model",
        "execution_mode": "fork",
        "fork_output": None,
    }


@pytest.mark.asyncio
async def test_local_skills_service_seed_builtin_skills_is_deterministic_when_empty(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)

    assert await service.seed_builtin_skills(overwrite=True) == {"seeded": [], "count": 0}


@pytest.mark.asyncio
async def test_local_skills_service_exposes_trust_state_and_blocks_uninitialized_context(tmp_path):
    service, _trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")

    listed = await service.list_skills()
    loaded = await service.get_skill("demo-skill")
    context = await service.get_context()

    assert listed["skills"][0]["trust_status"] == "trust_uninitialized"
    assert listed["skills"][0]["trust_blocked"] is True
    assert loaded["trust_status"] == "trust_uninitialized"
    assert loaded["trust_blocked"] is True
    assert context["available_skills"] == []
    assert context["blocked_skills"][0]["name"] == "demo-skill"
    assert context["blocked_skills"][0]["trust_reason_code"] == "trust_uninitialized"
    assert "demo-skill" not in context["context_text"]


@pytest.mark.asyncio
async def test_local_skills_service_blocks_execute_when_skill_changes_on_disk_after_bootstrap(tmp_path):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")
    trust.bootstrap_trust()
    (tmp_path / "skills" / "demo-skill" / "SKILL.md").write_text(
        "# Demo\nChanged {{args}}",
        encoding="utf-8",
    )

    with pytest.raises(SkillTrustBlockedError, match="skill_modified"):
        await service.execute_skill("demo-skill", args="x")


@pytest.mark.asyncio
async def test_local_skills_service_retrusts_explicitly_approved_update(tmp_path):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")
    trust.bootstrap_trust()

    updated = await service.update_skill(
        "demo-skill",
        content="# Demo\nChanged {{args}}",
        trust_approved=True,
    )
    listed = await service.list_skills()

    assert updated["trust_status"] == "trusted"
    assert updated["trust_blocked"] is False
    assert listed["skills"][0]["trust_status"] == "trusted"


@pytest.mark.asyncio
async def test_local_skills_service_unapproved_update_remains_trust_blocked(tmp_path):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")
    trust.bootstrap_trust()

    updated = await service.update_skill("demo-skill", content="# Demo\nChanged {{args}}")
    context = await service.get_context()

    assert updated["trust_status"] == "quarantined_modified"
    assert updated["trust_blocked"] is True
    assert updated["trust_changed_files"] == ["SKILL.md"]
    assert context["available_skills"] == []
    assert context["blocked_skills"][0]["trust_status"] == "quarantined_modified"
    assert "demo-skill" not in context["context_text"]
