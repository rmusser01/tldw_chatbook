"""Smoke test for the bundled `web-research` example skill.

Parses the skill through the public LocalSkillsService import/execute path
(rather than the private ``_parse_front_matter``) so the test exercises the
same validation, metadata extraction, and prompt rendering a real user hits.
"""

from pathlib import Path

import pytest

from tldw_chatbook.Agents.local_tool_provider import _default_specs
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_scanner import scan_skill_directory

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "Docs" / "Examples" / "skills" / "web-research"
)
SKILL_PATH = SKILL_DIR / "SKILL.md"


class _NoopWorkspaceExecutor:
    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
        raise AssertionError(f"unexpected workspace operation: {operation}")


def _compat_service(store_dir: Path) -> LocalSkillsService:
    return LocalSkillsService(
        store_dir=store_dir,
        allow_untrusted_without_trust_service=True,
    )


@pytest.mark.asyncio
async def test_web_research_skill_imports_and_executes_with_expected_metadata(tmp_path):
    service = _compat_service(tmp_path)
    content = SKILL_PATH.read_text(encoding="utf-8")

    imported = await service.import_skill(name="web-research", content=content)
    result = await service.execute_skill("web-research", args="test question")

    assert imported["name"] == "web-research"
    assert imported["validation_status"] == "valid"
    assert imported["validation_errors"] == []
    assert imported["argument_hint"]
    assert result["allowed_tools"] == ["web_search", "web_fetch"]

    rendered = result["rendered_prompt"]
    assert "web_search" in rendered
    assert "web_fetch" in rendered
    assert "citation" in rendered.lower()
    assert "test question" in rendered


@pytest.mark.asyncio
async def test_web_research_allowed_tools_are_registered_local_tools(tmp_path):
    service = _compat_service(tmp_path)
    content = SKILL_PATH.read_text(encoding="utf-8")

    imported = await service.import_skill(name="web-research", content=content)

    registered_names = {
        spec.name
        for spec in _default_specs(
            tmp_path, workspace_executor=_NoopWorkspaceExecutor()
        )
    }
    assert {"web_search", "web_fetch"} <= registered_names
    for tool_name in imported["allowed_tools"]:
        assert tool_name in registered_names


def test_web_research_skill_directory_scans_without_unsupported_paths():
    snapshot = scan_skill_directory("web-research", SKILL_DIR)

    assert snapshot.unsupported_paths == ()
    assert "SKILL.md" in snapshot.text_files
