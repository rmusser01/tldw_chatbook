from unittest.mock import Mock

import pytest

from tldw_chatbook.Skills_Interop import ServerSkillsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


NOW = "2026-04-25T12:00:00Z"


def _skill_payload(name="summarize-notes", **overrides):
    payload = {
        "id": f"skill-{name}",
        "name": name,
        "description": "Summarize notes",
        "argument_hint": "note id",
        "disable_model_invocation": False,
        "user_invocable": True,
        "allowed_tools": ["notes.read"],
        "model": None,
        "context": "inline",
        "content": "# Skill\nSummarize {{args}}",
        "supporting_files": None,
        "directory_path": f"/users/42/skills/{name}",
        "created_at": NOW,
        "last_modified": NOW,
        "version": 1,
    }
    payload.update(overrides)
    return payload


class FakeSkillsClient:
    def __init__(self):
        self.calls = []

    async def list_skills(self, **kwargs):
        self.calls.append(("list_skills", kwargs))
        return {"skills": [_skill_payload()], "count": 1, "total": 1, "limit": 25, "offset": 5}

    async def get_skills_context(self):
        self.calls.append(("get_skills_context",))
        return {"available_skills": [_skill_payload()], "context_text": "- summarize-notes"}

    async def get_skill(self, skill_name):
        self.calls.append(("get_skill", skill_name))
        return _skill_payload(skill_name)

    async def create_skill(self, request_data):
        self.calls.append(("create_skill", request_data.model_dump(exclude_none=True, mode="json")))
        return _skill_payload(request_data.name)

    async def update_skill(self, skill_name, request_data, **kwargs):
        self.calls.append(("update_skill", skill_name, request_data.model_dump(exclude_none=True, mode="json"), kwargs))
        return _skill_payload(skill_name, content="# Updated", version=2)

    async def delete_skill(self, skill_name, **kwargs):
        self.calls.append(("delete_skill", skill_name, kwargs))
        return True

    async def import_skill(self, request_data):
        self.calls.append(("import_skill", request_data.model_dump(exclude_none=True, mode="json")))
        return _skill_payload(request_data.name or "imported-skill")

    async def import_skill_file(self, file_content, **kwargs):
        self.calls.append(("import_skill_file", file_content, kwargs))
        return _skill_payload("file-skill")

    async def export_skill(self, skill_name):
        self.calls.append(("export_skill", skill_name))
        return {"content": b"zip-bytes", "filename": f"{skill_name}.zip"}

    async def execute_skill(self, skill_name, request_data=None):
        self.calls.append(("execute_skill", skill_name, request_data.model_dump(exclude_none=True, mode="json")))
        return {"skill_name": skill_name, "rendered_prompt": "Summarize note-1", "execution_mode": "inline"}

    async def seed_builtin_skills(self, **kwargs):
        self.calls.append(("seed_builtin_skills", kwargs))
        return {"seeded": ["summarize-notes"], "count": 1}


@pytest.mark.asyncio
async def test_server_skills_service_routes_core_skill_surface_with_policy_actions():
    client = FakeSkillsClient()
    policy = Mock()
    service = ServerSkillsService(client=client, policy_enforcer=policy)

    await service.list_skills(include_hidden=True, limit=25, offset=5)
    await service.get_context()
    await service.get_skill("summarize-notes")
    await service.create_skill(
        name="summarize-notes",
        content="# Skill\nSummarize {{args}}",
        supporting_files={"reference.md": "Use concise bullets."},
    )
    await service.update_skill(
        "summarize-notes",
        content="# Updated",
        supporting_files={"reference.md": None},
        expected_version=2,
    )
    await service.delete_skill("summarize-notes", expected_version=3)
    await service.import_skill(name="rewrite-draft", content="# Skill\nRewrite draft", overwrite=True)
    await service.import_skill_file(b"# Skill\nFile import", filename="file-skill.md", overwrite=True)
    await service.export_skill("summarize-notes")
    await service.execute_skill("summarize-notes", args="note-1")
    await service.seed_builtin_skills(overwrite=True)

    assert client.calls == [
        ("list_skills", {"include_hidden": True, "limit": 25, "offset": 5}),
        ("get_skills_context",),
        ("get_skill", "summarize-notes"),
        (
            "create_skill",
            {
                "name": "summarize-notes",
                "content": "# Skill\nSummarize {{args}}",
                "supporting_files": {"reference.md": "Use concise bullets."},
            },
        ),
        ("update_skill", "summarize-notes", {"content": "# Updated", "supporting_files": {"reference.md": None}}, {"expected_version": 2}),
        ("delete_skill", "summarize-notes", {"expected_version": 3}),
        ("import_skill", {"name": "rewrite-draft", "content": "# Skill\nRewrite draft", "overwrite": True}),
        ("import_skill_file", b"# Skill\nFile import", {"filename": "file-skill.md", "content_type": "text/markdown", "overwrite": True}),
        ("export_skill", "summarize-notes"),
        ("execute_skill", "summarize-notes", {"args": "note-1"}),
        ("seed_builtin_skills", {"overwrite": True}),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "skills.list.server",
        "skills.context.list.server",
        "skills.detail.server",
        "skills.create.server",
        "skills.update.server",
        "skills.delete.server",
        "skills.import.launch.server",
        "skills.import.launch.server",
        "skills.export.launch.server",
        "skills.execute.launch.server",
        "skills.seed.launch.server",
    ]


@pytest.mark.asyncio
async def test_server_skills_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_auth_required",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeSkillsClient()
    service = ServerSkillsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_skills()

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []
