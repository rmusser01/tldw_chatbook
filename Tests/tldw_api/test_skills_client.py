from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ReadingExportResponse,
    SkillCreate,
    SkillExecuteRequest,
    SkillExecutionResult,
    SkillImportRequest,
    SkillResponse,
    SkillUpdate,
    SkillsListResponse,
    SkillContextPayload,
    TLDWAPIClient,
)


NOW = "2026-04-25T12:00:00Z"


def _skill_payload(name: str = "summarize-notes", **overrides) -> dict:
    payload = {
        "id": "skill-1",
        "name": name,
        "description": "Summarize notes",
        "argument_hint": "[note-id]",
        "disable_model_invocation": False,
        "user_invocable": True,
        "allowed_tools": ["notes.read"],
        "model": None,
        "context": "inline",
        "content": "# Skill\nSummarize {{args}}",
        "supporting_files": {"reference.md": "Use concise bullets."},
        "directory_path": "/users/42/skills/summarize-notes",
        "created_at": NOW,
        "last_modified": NOW,
        "version": 2,
    }
    payload.update(overrides)
    return payload


def _skill_summary(name: str = "summarize-notes") -> dict:
    return {
        "name": name,
        "description": "Summarize notes",
        "argument_hint": "[note-id]",
        "user_invocable": True,
        "disable_model_invocation": False,
        "context": "inline",
    }


@pytest.mark.asyncio
async def test_skills_client_routes_crud_import_execute_and_context(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"skills": [_skill_summary()], "count": 1, "total": 1, "limit": 25, "offset": 5},
            {"available_skills": [_skill_summary()], "context_text": "- summarize-notes: Summarize notes"},
            _skill_payload(),
            _skill_payload(version=1),
            _skill_payload(content="# Updated"),
            {},
            _skill_payload("rewrite-draft"),
            _skill_payload("file-skill"),
            {
                "skill_name": "summarize-notes",
                "rendered_prompt": "Summarize note-1",
                "allowed_tools": ["notes.read"],
                "model_override": None,
                "execution_mode": "inline",
                "fork_output": None,
            },
            {"seeded": ["summarize-notes"], "count": 1},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    listed = await client.list_skills(include_hidden=True, limit=25, offset=5)
    context = await client.get_skills_context()
    fetched = await client.get_skill("summarize-notes")
    created = await client.create_skill(
        SkillCreate(
            name="summarize-notes",
            content="# Skill\nSummarize {{args}}",
            supporting_files={"reference.md": "Use concise bullets."},
        )
    )
    updated = await client.update_skill(
        "summarize-notes",
        SkillUpdate(content="# Updated", supporting_files={"reference.md": None}),
        expected_version=2,
    )
    deleted = await client.delete_skill("summarize-notes", expected_version=3)
    imported = await client.import_skill(
        SkillImportRequest(name="rewrite-draft", content="# Skill\nRewrite draft", overwrite=True)
    )
    imported_file = await client.import_skill_file(
        b"# Skill\nFile import",
        filename="file-skill.md",
        content_type="text/markdown",
        overwrite=True,
    )
    executed = await client.execute_skill("summarize-notes", SkillExecuteRequest(args="note-1"))
    seeded = await client.seed_builtin_skills(overwrite=True)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/skills/")
    assert mocked.await_args_list[0].kwargs["params"] == {"include_hidden": True, "limit": 25, "offset": 5}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/skills/context")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/skills/summarize-notes")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/skills/")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "name": "summarize-notes",
        "content": "# Skill\nSummarize {{args}}",
        "supporting_files": {"reference.md": "Use concise bullets."},
    }
    assert mocked.await_args_list[4].args[:2] == ("PUT", "/api/v1/skills/summarize-notes")
    assert mocked.await_args_list[4].kwargs["json_data"] == {
        "content": "# Updated",
        "supporting_files": {"reference.md": None},
    }
    assert mocked.await_args_list[4].kwargs["headers"] == {"If-Match": "2"}
    assert mocked.await_args_list[5].args[:2] == ("DELETE", "/api/v1/skills/summarize-notes")
    assert mocked.await_args_list[5].kwargs["headers"] == {"If-Match": "3"}
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/skills/import")
    assert mocked.await_args_list[6].kwargs["json_data"] == {
        "name": "rewrite-draft",
        "content": "# Skill\nRewrite draft",
        "overwrite": True,
    }
    assert mocked.await_args_list[7].args[:2] == ("POST", "/api/v1/skills/import/file")
    assert mocked.await_args_list[7].kwargs["params"] == {"overwrite": True}
    assert mocked.await_args_list[7].kwargs["files"] == [
        ("file", ("file-skill.md", b"# Skill\nFile import", "text/markdown"))
    ]
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/skills/summarize-notes/execute")
    assert mocked.await_args_list[8].kwargs["json_data"] == {"args": "note-1"}
    assert mocked.await_args_list[9].args[:2] == ("POST", "/api/v1/skills/seed")
    assert mocked.await_args_list[9].kwargs["params"] == {"overwrite": True}

    assert isinstance(listed, SkillsListResponse)
    assert isinstance(context, SkillContextPayload)
    assert isinstance(fetched, SkillResponse)
    assert isinstance(created, SkillResponse)
    assert isinstance(updated, SkillResponse)
    assert deleted is True
    assert imported.name == "rewrite-draft"
    assert imported_file.name == "file-skill"
    assert isinstance(executed, SkillExecutionResult)
    assert seeded["count"] == 1


@pytest.mark.asyncio
async def test_skills_client_exports_zip_with_binary_helper(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value=ReadingExportResponse(
            content=b"zip-bytes",
            content_type="application/zip",
            content_disposition='attachment; filename="summarize-notes.zip"',
            filename="summarize-notes.zip",
        )
    )
    monkeypatch.setattr(client, "_binary_request", mocked)

    exported = await client.export_skill("summarize-notes")

    assert mocked.await_args.args[:2] == ("GET", "/api/v1/skills/summarize-notes/export")
    assert exported.content == b"zip-bytes"
    assert exported.content_type == "application/zip"
    assert exported.filename == "summarize-notes.zip"
