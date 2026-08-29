import asyncio
import io
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_models import (
    SkillTrustBlockedError,
    SkillTrustStatus,
)
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
    marker_path = tmp_path / "marker.json"
    trust_service = SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(
                marker_path, store_dir=marker_path.parent
            ),
        ),
    )
    trust_service.unlock_with_passphrase("passphrase", salt=b"7" * 32)
    return LocalSkillsService(
        store_dir=tmp_path, trust_service=trust_service
    ), trust_service


def _compat_local_service(store_dir):
    return LocalSkillsService(
        store_dir=store_dir,
        allow_untrusted_without_trust_service=True,
    )


def _skill_trust_status(
    skill_name: str,
    *,
    trust_status: str = "trusted",
    trust_reason_code: str | None = None,
    trust_blocked: bool = False,
    changed_files: tuple[str, ...] = (),
) -> SkillTrustStatus:
    return SkillTrustStatus(
        skill_name=skill_name,
        trust_status=trust_status,
        trust_reason_code=trust_reason_code,
        trust_blocked=trust_blocked,
        changed_files=changed_files,
        manifest_generation=1,
        last_verified_at="2026-06-25T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_local_skills_service_persists_skill_metadata(tmp_path):
    service = _compat_local_service(tmp_path)

    created = await service.create_skill(
        name="summarize-notes", content=SKILL_WITH_METADATA
    )
    reloaded = _compat_local_service(tmp_path)
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
    service = _compat_local_service(tmp_path)

    created = await service.create_skill(
        name="summarize-notes", content=AGENT_SKILL_WITH_METADATA
    )
    listed = await service.list_skills()
    context = await service.get_context()

    assert created["validation_status"] == "valid"
    assert created["validation_errors"] == []
    assert created["agent_skill_name"] == "summarize-notes"
    assert (
        created["description"]
        == "Summarize note collections. Use when a user asks for a concise notes summary."
    )
    assert created["allowed_tools"] == ["Read", "Bash(git:*)"]
    assert listed["skills"][0]["validation_status"] == "valid"
    assert listed["skills"][0]["validation_errors"] == []
    assert context["available_skills"][0]["validation_status"] == "valid"


@pytest.mark.asyncio
async def test_local_skills_service_sanitizes_agent_skill_frontmatter_fields(tmp_path):
    service = _compat_local_service(tmp_path)
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
async def test_local_skills_service_accepts_agent_skill_name_starting_with_number(
    tmp_path,
):
    service = _compat_local_service(tmp_path)
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
async def test_local_skills_service_import_preserves_digit_leading_agent_skill_name(
    tmp_path,
):
    service = _compat_local_service(tmp_path)
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
async def test_local_skills_service_marks_over_schema_limit_description_invalid_without_crashing(
    tmp_path,
):
    service = _compat_local_service(tmp_path)
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
    assert (
        "description must be 1000 characters or fewer" in created["validation_errors"]
    )


@pytest.mark.asyncio
async def test_local_skills_service_reports_invalid_agent_skill_metadata_without_mutating_content(
    tmp_path,
):
    service = _compat_local_service(tmp_path)

    created = await service.create_skill(
        name="summarize-notes", content=INVALID_AGENT_SKILL
    )
    listed = await service.list_skills()
    loaded = await service.get_skill("summarize-notes")

    assert created["validation_status"] == "invalid"
    assert (
        "name must use lowercase letters, numbers, and hyphens"
        in created["validation_errors"]
    )
    assert "description is required" in created["validation_errors"]
    assert listed["skills"][0]["validation_status"] == "invalid"
    assert loaded["content"] == INVALID_AGENT_SKILL


@pytest.mark.asyncio
async def test_local_skills_service_uses_deterministic_metadata_defaults(tmp_path):
    service = _compat_local_service(tmp_path)

    await service.create_skill(
        name="draft-helper", content="# Draft Helper\n\nHelps rewrite local drafts."
    )
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
    service = _compat_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nInitial")

    await service.update_skill(
        "demo-skill", content="# Demo\nUpdated", expected_version=1
    )

    with pytest.raises(ValueError, match="local_skill_version_conflict:demo-skill"):
        await service.update_skill(
            "demo-skill", content="# Demo\nStale", expected_version=1
        )


@pytest.mark.asyncio
async def test_local_skills_service_serializes_concurrent_updates(tmp_path):
    service = _compat_local_service(tmp_path)
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
    service = _compat_local_service(tmp_path)

    with pytest.raises(ValueError, match="Invalid path segment"):
        await service.create_skill(
            name="unsafe-skill",
            content="# Unsafe",
            supporting_files={"../escape.md": "bad"},
        )


@pytest.mark.asyncio
async def test_local_skills_service_import_export_round_trip(tmp_path):
    source = _compat_local_service(tmp_path / "source")
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

    target = _compat_local_service(tmp_path / "target")
    imported = await target.import_skill_file(
        exported["content"],
        filename=exported["filename"],
        content_type=exported["content_type"],
    )

    assert created["name"] == "rewrite-draft"
    assert imported["name"] == "rewrite-draft"
    assert (await target.get_skill("rewrite-draft"))["supporting_files"] == {
        "style.md": "Use concise language."
    }


@pytest.mark.asyncio
async def test_local_skills_service_import_skill_file_derives_name_from_markdown_filename(
    tmp_path,
):
    service = _compat_local_service(tmp_path)

    imported = await service.import_skill_file(
        b"# File Skill\nRender {{args}}",
        filename="file-skill.md",
        content_type="text/markdown",
    )

    assert imported["name"] == "file-skill"


@pytest.mark.asyncio
async def test_local_skills_service_execute_renders_prompt_without_model_invocation(
    tmp_path,
):
    service = _compat_local_service(tmp_path)
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
async def test_local_skills_service_seed_builtin_skills_is_deterministic_when_empty(
    tmp_path,
):
    service = _compat_local_service(tmp_path)

    assert await service.seed_builtin_skills(overwrite=True) == {
        "seeded": [],
        "count": 0,
    }


@pytest.mark.asyncio
async def test_local_skills_service_without_trust_service_fails_closed_by_default(
    tmp_path,
):
    service = LocalSkillsService(store_dir=tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")

    listed = await service.list_skills()
    loaded = await service.get_skill("demo-skill")
    context = await service.get_context()

    assert listed["skills"][0]["trust_status"] == "trust_locked"
    assert listed["skills"][0]["trust_reason_code"] == "trust_service_unavailable"
    assert listed["skills"][0]["trust_blocked"] is True
    assert loaded["trust_status"] == "trust_locked"
    assert loaded["trust_reason_code"] == "trust_service_unavailable"
    assert context["available_skills"] == []
    assert context["blocked_skills"][0]["name"] == "demo-skill"
    assert "demo-skill" not in context["context_text"]
    with pytest.raises(SkillTrustBlockedError, match="trust_service_unavailable"):
        await service.execute_skill("demo-skill", args="x")


@pytest.mark.asyncio
async def test_list_skills_filters_name_and_description_before_exact_page_slice(
    tmp_path,
):
    service = _compat_local_service(tmp_path)
    for index in range(45):
        match = "Needle description" if index % 2 == 0 else "Ordinary description"
        await service.create_skill(
            name=f"skill-{index:02d}",
            content=f"---\ndescription: {match}\n---\nBody {index}",
        )

    page = await service.list_skills(
        query="nEeDlE",
        sort="name",
        limit=20,
        offset=20,
    )

    assert page["total"] == 23
    assert page["count"] == 3
    assert page["limit"] == 20
    assert page["offset"] == 20
    assert [item["name"] for item in page["skills"]] == [
        "skill-40",
        "skill-42",
        "skill-44",
    ]


@pytest.mark.asyncio
async def test_list_skills_filter_excludes_non_name_description_surfaces(tmp_path):
    service = _compat_local_service(tmp_path)
    await service.create_skill(
        name="ordinary-skill",
        content=(
            "---\n"
            "description: Ordinary helper\n"
            "argument_hint: private-needle\n"
            "metadata:\n  private-needle: present\n"
            "---\n"
            "Body contains private-needle."
        ),
        supporting_files={"private-needle.md": "private-needle"},
    )

    page = await service.list_skills(
        query="private-needle",
        sort="name",
        limit=20,
        offset=0,
    )

    assert page["total"] == 0
    assert page["skills"] == []


@pytest.mark.asyncio
async def test_list_skills_status_sort_and_source_wide_blocked_recovery_are_stable(
    tmp_path,
):
    class MixedTrustService:
        def status_for_skill(self, skill_name):
            blocked = skill_name in {"blocked-alpha", "blocked-zulu"}
            return _skill_trust_status(
                skill_name,
                trust_status="quarantined_modified" if blocked else "trusted",
                trust_reason_code="skill_modified" if blocked else None,
                trust_blocked=blocked,
                changed_files=("SKILL.md",) if blocked else (),
            )

    service = LocalSkillsService(
        store_dir=tmp_path,
        trust_service=MixedTrustService(),
    )
    for name in ("trusted-beta", "blocked-zulu", "trusted-alpha", "blocked-alpha"):
        await service.create_skill(
            name=name,
            content=f"---\ndescription: {name}\n---\nBody",
        )

    page = await service.list_skills(
        query="zulu",
        sort="status",
        limit=1,
        offset=0,
    )

    assert [item["name"] for item in page["skills"]] == ["blocked-zulu"]
    assert page["skills"][0]["trust_blocked"] is True
    assert page["total"] == 1
    assert page["blocked_total"] == 2
    assert page["first_blocked_skill_name"] == "blocked-alpha"


@pytest.mark.asyncio
async def test_local_skills_service_exposes_trust_state_and_blocks_uninitialized_context(
    tmp_path,
):
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
async def test_local_skills_service_blocks_execute_when_skill_changes_on_disk_after_bootstrap(
    tmp_path,
):
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
async def test_local_skills_service_rejects_symlinked_skill_file_reads(tmp_path):
    service = _compat_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\nsecret", encoding="utf-8")
    skill_path = tmp_path / "skills" / "demo-skill" / "SKILL.md"
    skill_path.unlink()
    skill_path.symlink_to(outside)

    with pytest.raises(ValueError, match="unsafe local skill path"):
        await service.get_skill("demo-skill")


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

    updated = await service.update_skill(
        "demo-skill", content="# Demo\nChanged {{args}}"
    )
    context = await service.get_context()

    assert updated["trust_status"] == "quarantined_modified"
    assert updated["trust_blocked"] is True
    assert updated["trust_changed_files"] == ["SKILL.md"]
    assert context["available_skills"] == []
    assert context["blocked_skills"][0]["trust_status"] == "quarantined_modified"
    assert "demo-skill" not in context["context_text"]


@pytest.mark.asyncio
async def test_local_skills_service_create_requires_explicit_trust_approval_after_bootstrap(
    tmp_path,
):
    service, trust = _trusted_local_service(tmp_path)
    trust.bootstrap_trust()

    unapproved = await service.create_skill(
        name="draft-skill", content="# Draft\nRender {{args}}"
    )
    approved = await service.create_skill(
        name="approved-skill",
        content="# Approved\nRender {{args}}",
        trust_approved=True,
    )
    context = await service.get_context()

    assert unapproved["trust_status"] == "quarantined_added"
    assert unapproved["trust_blocked"] is True
    assert approved["trust_status"] == "trusted"
    assert approved["trust_blocked"] is False
    assert [item["name"] for item in context["available_skills"]] == ["approved-skill"]
    assert [item["name"] for item in context["blocked_skills"]] == ["draft-skill"]
    with pytest.raises(SkillTrustBlockedError, match="skill_added"):
        await service.execute_skill("draft-skill", args="x")
    assert (await service.execute_skill("approved-skill", args="x"))[
        "rendered_prompt"
    ] == "# Approved\nRender x"


@pytest.mark.asyncio
async def test_local_skills_service_import_requires_explicit_trust_approval_after_bootstrap(
    tmp_path,
):
    service, trust = _trusted_local_service(tmp_path)
    trust.bootstrap_trust()

    unapproved = await service.import_skill(
        name="imported-draft",
        content="# Imported Draft\nRender {{args}}",
    )
    approved = await service.import_skill(
        name="imported-approved",
        content="# Imported Approved\nRender {{args}}",
        trust_approved=True,
    )

    assert unapproved["trust_status"] == "quarantined_added"
    assert unapproved["trust_blocked"] is True
    assert approved["trust_status"] == "trusted"
    assert approved["trust_blocked"] is False
    with pytest.raises(SkillTrustBlockedError, match="skill_added"):
        await service.execute_skill("imported-draft", args="x")
    assert (await service.execute_skill("imported-approved", args="x"))[
        "rendered_prompt"
    ] == ("# Imported Approved\nRender x")


@pytest.mark.asyncio
async def test_local_skills_service_import_file_requires_explicit_trust_approval_after_bootstrap(
    tmp_path,
):
    service, trust = _trusted_local_service(tmp_path)
    trust.bootstrap_trust()

    unapproved = await service.import_skill_file(
        b"# File Draft\nRender {{args}}",
        filename="file-draft.md",
        content_type="text/markdown",
    )
    approved = await service.import_skill_file(
        b"# File Approved\nRender {{args}}",
        filename="file-approved.md",
        content_type="text/markdown",
        trust_approved=True,
    )

    assert unapproved["trust_status"] == "quarantined_added"
    assert unapproved["trust_blocked"] is True
    assert approved["trust_status"] == "trusted"
    assert approved["trust_blocked"] is False
    with pytest.raises(SkillTrustBlockedError, match="skill_added"):
        await service.execute_skill("file-draft", args="x")
    assert (await service.execute_skill("file-approved", args="x"))[
        "rendered_prompt"
    ] == "# File Approved\nRender x"


@pytest.mark.asyncio
async def test_local_skills_service_rebaseline_failure_leaves_mutation_applied_but_blocked(
    tmp_path,
):
    class FailingTrustService:
        def status_for_skill(self, skill_name):
            return _skill_trust_status(
                skill_name,
                trust_status="quarantined_modified",
                trust_reason_code="trust_rebaseline_failed",
                trust_blocked=True,
                changed_files=("SKILL.md",),
            )

        def trust_current_skill(self, skill_name, *, audit_event):
            raise RuntimeError("trust store unavailable")

        def ensure_skill_trusted(self, skill_name):
            raise SkillTrustBlockedError(
                skill_name=skill_name,
                reason_code="trust_rebaseline_failed",
                trust_status="quarantined_modified",
                changed_files=("SKILL.md",),
            )

        def verify_skill_content(self, skill_name, *, skill_content, supporting_files):
            self.ensure_skill_trusted(skill_name)

    service = LocalSkillsService(
        store_dir=tmp_path, trust_service=FailingTrustService()
    )
    await service.create_skill(name="demo-skill", content="# Demo\nInitial")

    with pytest.raises(RuntimeError, match="trust store unavailable"):
        await service.update_skill(
            "demo-skill", content="# Demo\nUpdated", trust_approved=True
        )

    loaded = await service.get_skill("demo-skill")
    listed = await service.list_skills()
    context = await service.get_context()

    assert loaded["content"] == "# Demo\nUpdated"
    assert loaded["trust_status"] == "quarantined_modified"
    assert listed["skills"][0]["trust_status"] == "quarantined_modified"
    assert context["available_skills"] == []
    assert context["blocked_skills"][0]["name"] == "demo-skill"
    with pytest.raises(SkillTrustBlockedError, match="trust_rebaseline_failed"):
        await service.execute_skill("demo-skill", args="x")


@pytest.mark.asyncio
async def test_local_skills_service_execute_rechecks_trust_after_reading_content(
    tmp_path,
):
    class MutatingTrustService:
        def __init__(self, skill_path):
            self.skill_path = skill_path
            self.checks = 0

        def status_for_skill(self, skill_name):
            return _skill_trust_status(skill_name)

        def trust_current_skill(self, skill_name, *, audit_event):
            return None

        def ensure_skill_trusted(self, skill_name):
            self.checks += 1
            if self.checks == 1:
                self.skill_path.write_text("# Demo\nChanged {{args}}", encoding="utf-8")
                return
            raise SkillTrustBlockedError(
                skill_name=skill_name,
                reason_code="skill_modified",
                trust_status="quarantined_modified",
                changed_files=("SKILL.md",),
            )

        def verify_skill_content(self, skill_name, *, skill_content, supporting_files):
            self.ensure_skill_trusted(skill_name)

    skill_path = tmp_path / "skills" / "demo-skill" / "SKILL.md"
    trust_service = MutatingTrustService(skill_path)
    service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")

    with pytest.raises(SkillTrustBlockedError, match="skill_modified"):
        await service.execute_skill("demo-skill", args="x")
    assert trust_service.checks == 2


@pytest.mark.asyncio
async def test_local_skills_service_execute_preserves_crlf_for_exact_trust_verification(
    tmp_path,
):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(
        name="crlf-skill",
        content="# CRLF\r\nRender {{args}}\r\n",
        supporting_files={"notes.md": "trusted notes\r\n"},
    )
    trust.bootstrap_trust()

    assert trust.status_for_skill("crlf-skill").trust_status == "trusted"
    result = await service.execute_skill("crlf-skill", args="x")

    assert result["rendered_prompt"] == "# CRLF\r\nRender x"


@pytest.mark.asyncio
async def test_local_skills_service_execute_rejects_read_and_restore_content_race(
    tmp_path, monkeypatch
):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nRender {{args}}")
    trust.bootstrap_trust()
    skill_path = tmp_path / "skills" / "demo-skill" / "SKILL.md"
    original_read_bytes = Path.read_bytes
    trusted_content = original_read_bytes(skill_path)
    malicious_content = b"# Demo\nMALICIOUS {{args}}"

    def read_malicious_then_restored(self, *args, **kwargs):
        if self == skill_path:
            assert original_read_bytes(self) == trusted_content
            return malicious_content
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_malicious_then_restored)

    with pytest.raises(SkillTrustBlockedError, match="skill_modified"):
        await service.execute_skill("demo-skill", args="x")
    assert original_read_bytes(skill_path) == trusted_content


# ---------------------------------------------------------------------------
# Library read seams (task-1337 plan Task 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_skills_list_enumerates_with_exact_total_and_safe_projection(
    tmp_path,
):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="alpha-skill", content="# Alpha\nbody")
    await service.create_skill(name="beta-skill", content="# Beta\nbody")
    trust.bootstrap_trust()

    page = await service.list_library_skills(limit=1, offset=0)
    assert page["total"] == 2
    assert page["offset"] == 0
    assert page["limit"] == 1
    assert [item["name"] for item in page["items"]] == ["alpha-skill"]
    item = page["items"][0]
    assert item["trust_blocked"] is False
    for forbidden in ("content", "body", "supporting_files", "preview", "files"):
        assert forbidden not in item

    page_two = await service.list_library_skills(limit=1, offset=1)
    assert page_two["total"] == 2
    assert [item["name"] for item in page_two["items"]] == ["beta-skill"]


@pytest.mark.asyncio
async def test_library_skills_search_matches_casefolded_name_description_body(tmp_path):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="alpha-skill", content="# Alpha\nbody")
    await service.create_skill(
        name="beta-notes",
        content="---\ndescription: Quarterly Digest helper\n---\n# Beta\nbody",
    )
    await service.create_skill(name="gamma-deep", content="# Gamma\nquarterly body hit")
    trust.bootstrap_trust()

    payload = await service.search_library_skills(query="QUARTERLY", limit=10, offset=0)
    assert payload["total"] == 2
    by_name = {item["name"]: item for item in payload["items"]}
    assert set(by_name) == {"beta-notes", "gamma-deep"}
    assert "description" in by_name["beta-notes"]["matched_fields"]
    assert "body" in by_name["gamma-deep"]["matched_fields"]

    exact = await service.search_library_skills(query="alpha-skill", limit=10, offset=0)
    assert exact["items"][0]["name"] == "alpha-skill"
    assert "name" in exact["items"][0]["matched_fields"]


@pytest.mark.asyncio
async def test_library_skills_search_never_matches_blocked_bodies(tmp_path):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="blocked-skill", content="# Blocked\noriginal body")
    trust.bootstrap_trust()
    # Out-of-band tamper: the skill is now trust-blocked (skill_modified).
    (tmp_path / "skills" / "blocked-skill" / "SKILL.md").write_text(
        "# Blocked\nbody now contains uniqueterm"
    )

    payload = await service.search_library_skills(
        query="uniqueterm", limit=10, offset=0
    )
    assert payload["total"] == 0

    # Safe fields (name/description/trust) still surface the blocked skill.
    by_name = await service.search_library_skills(
        query="blocked-skill", limit=10, offset=0
    )
    assert [item["name"] for item in by_name["items"]] == ["blocked-skill"]
    item = by_name["items"][0]
    assert item["trust_blocked"] is True
    assert "body" not in item["matched_fields"]


@pytest.mark.asyncio
async def test_library_skills_list_and_search_skip_eager_supporting_reads(
    tmp_path, monkeypatch
):
    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="alpha-skill", content="# Alpha\nbody")
    trust.bootstrap_trust()

    def _explode(skill_dir):
        raise AssertionError(
            "eager _read_supporting_files must not run for Library reads"
        )

    monkeypatch.setattr(
        "tldw_chatbook.Skills_Interop.local_skills_service.LocalSkillsService._read_supporting_files",
        staticmethod(_explode),
    )

    await service.list_library_skills(limit=10, offset=0)
    await service.search_library_skills(query="alpha", limit=10, offset=0)


@pytest.mark.asyncio
async def test_library_skill_detail_returns_bounded_manifest_with_file_tokens(tmp_path):
    service, trust = _trusted_local_service(tmp_path)
    body = "# Alpha\n" + "body text " * 200  # > 2000 chars
    await service.create_skill(
        name="alpha-skill",
        content=body,
        supporting_files={"references/guide.md": "guide content " * 50},
    )
    trust.bootstrap_trust()

    detail = await service.get_library_skill("alpha-skill")
    assert detail["name"] == "alpha-skill"
    assert detail["trust_blocked"] is False
    assert detail["body_total_chars"] == len(body)
    assert len(detail["body_preview"]) <= 241
    paths = {entry["path"] for entry in detail["files"]}
    assert "SKILL.md" in paths
    assert "references/guide.md" in paths
    for entry in detail["files"]:
        assert entry["file_token"].startswith("file:")
        assert isinstance(entry["size"], int)
        # The manifest never inlines content.
        assert "content" not in entry
        assert "preview" not in entry
    assert "content" not in detail
    assert "supporting_files" not in detail


@pytest.mark.asyncio
async def test_library_skill_file_windows_text_and_revision_tracks_content(tmp_path):
    service, trust = _trusted_local_service(tmp_path)
    body = "abcdef" * 900  # 5400 chars
    await service.create_skill(name="alpha-skill", content=body)
    trust.bootstrap_trust()

    detail = await service.get_library_skill("alpha-skill")
    main_token = next(
        entry["file_token"] for entry in detail["files"] if entry["path"] == "SKILL.md"
    )

    page = await service.get_library_skill_file(
        "alpha-skill", main_token, start=1200, max_chars=2000
    )
    assert page["path"] == "SKILL.md"
    assert page["text"] == body[1200:3200]
    assert page["total_chars"] == len(body)
    assert page["start"] == 1200
    assert page["returned_chars"] == 2000
    assert page["has_more"] is True
    assert page["revision"]

    tail = await service.get_library_skill_file(
        "alpha-skill", main_token, start=5000, max_chars=2000
    )
    assert tail["text"] == body[5000:]
    assert tail["has_more"] is False
    assert tail["revision"] == page["revision"]

    # A legitimate update re-trusts and changes the content revision.
    await service.update_skill(
        "alpha-skill", content=body + "MORE", trust_approved=True
    )
    updated = await service.get_library_skill_file(
        "alpha-skill", main_token, start=0, max_chars=10
    )
    assert updated["revision"] != page["revision"]


@pytest.mark.asyncio
async def test_library_skill_file_rejects_traversal_and_garbage_tokens(tmp_path):
    import base64

    service, trust = _trusted_local_service(tmp_path)
    await service.create_skill(name="alpha-skill", content="# A\nbody")
    trust.bootstrap_trust()

    traversal = "file:" + base64.urlsafe_b64encode(b"../../etc/passwd").decode(
        "ascii"
    ).rstrip("=")
    with pytest.raises(ValueError):
        await service.get_library_skill_file(
            "alpha-skill", traversal, start=0, max_chars=100
        )
    with pytest.raises(ValueError):
        await service.get_library_skill_file(
            "alpha-skill", "file:not-valid-base64!!!", start=0, max_chars=100
        )
    with pytest.raises(ValueError):
        await service.get_library_skill_file(
            "alpha-skill", "absolute:/etc/passwd", start=0, max_chars=100
        )
    with pytest.raises(ValueError):
        await service.get_library_skill_file(
            "alpha-skill", "file:bm8tc3VjaC1maWxlLm1k", start=0, max_chars=100
        )


@pytest.mark.asyncio
async def test_library_skill_detail_and_file_are_fail_closed_for_blocked_skills(
    tmp_path,
):
    service = LocalSkillsService(store_dir=tmp_path)  # fail-closed: all blocked
    await service.create_skill(name="blocked-skill", content="# Blocked\nsecret body")

    detail = await service.get_library_skill("blocked-skill")
    assert detail["name"] == "blocked-skill"
    assert detail["trust_blocked"] is True
    for forbidden in ("body_preview", "body_total_chars", "files", "content"):
        assert forbidden not in detail

    detail_list = await service.list_library_skills(limit=10, offset=0)
    assert detail_list["items"][0]["trust_blocked"] is True

    with pytest.raises((SkillTrustBlockedError, ValueError)):
        await service.get_library_skill_file(
            "blocked-skill", "file:U0tJTEwubWQ", start=0, max_chars=100
        )
