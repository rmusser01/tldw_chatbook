"""Real native selected-source tools keep permissions and instruction scope separate."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Agents.test_goal_iteration_report import report
from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_console_local_review_hook import ALLOW, _FakeService
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from tldw_chatbook.Agents.goal_models import GoalToolScope
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

stores = _stores_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [None, "registry", "root", "symlink", "scope"])
async def test_native_selected_source_reads_and_refusals(
    stores, tmp_path, monkeypatch, change
):
    import tldw_chatbook.Chat.console_chat_controller as controller_module

    runs, persistence, registry, req = stores
    primary = Path(req.binding.locator)
    (primary / "fixture.txt").write_text("invalid")
    source, sibling = tmp_path / "source", tmp_path / "sibling"
    for root in (source, sibling):
        root.mkdir()
    (source / "data.txt").write_text("MODEL VISIBLE SOURCE DATA")
    (source / "AGENTS.md").write_text("SOURCE AGENTS MUST NEVER ACTIVATE")
    (sibling / "secret").write_text("UNSELECTED SECRET")
    binding = WorkspaceRuntimeBinding(
        workspace_id="workspace",
        binding_id="source",
        binding_kind="local-filesystem",
        label="Source",
        locator=str(source),
        status="ready",
        metadata={"access": "ro"},
    )
    registry.save_runtime_binding(binding)
    reference = req.binding.model_copy(
        update={"binding_id": "source", "locator": str(source), "access": "ro"}
    )
    req = req.model_copy(
        update={
            "source_bindings": (reference,),
            "tool_scope": GoalToolScope(
                catalog_tools=("local:fs_edit",)
                if change == "scope"
                else (
                    "local:fs_read",
                    "local:fs_list",
                    "local:fs_edit",
                    "local:fs_write",
                )
            ),
        }
    )
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        if count <= 2 and not any(
            m.get("role") == "tool"
            and "Deferred because" not in str(m.get("content", ""))
            for m in kwargs["messages_payload"]
        ):
            context = next(
                m["content"] for m in kwargs["messages_payload"] if m["role"] == "user"
            )
            resources = json.loads(
                context.split("Selected launch resources (JSON):\n", 1)[1].split(
                    "\n\nPrivate checkpoint memory", 1
                )[0]
            )
            root = resources["read_only_sources"][0]["locator"]
            if count == 1 and change == "registry":
                registry.save_runtime_binding(
                    binding.model_copy(update={"metadata": {"access": "rw"}})
                )
            if count == 1 and change in ("root", "symlink"):
                source.rename(tmp_path / "old-source")
                if change == "root":
                    source.mkdir()
                    (source / "data.txt").write_text("REPLACEMENT MUST NOT BE READ")
                else:
                    source.symlink_to(tmp_path / "old-source", target_is_directory=True)
            actions = [("fs_read", {"path": str(Path(root) / "data.txt")})]
            if change is None:
                actions += [
                    ("fs_list", {"path": root}),
                    ("fs_read", {"path": str(sibling / "secret")}),
                    (
                        "fs_write",
                        {"path": str(Path(root) / "data.txt"), "content": "bad"},
                    ),
                    (
                        "fs_edit",
                        {
                            "path": "fixture.txt",
                            "old_string": "invalid",
                            "new_string": "valid",
                        },
                    ),
                ]
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": str(i),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for i, (name, args) in enumerate(actions)
                ],
            }
        else:
            message = {"content": report()}
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    goal, store, session, controller, co, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    store.set_session_project_instruction_state(
        session.id, ProjectInstructionControlState.new_session()
    )
    controller.app = SimpleNamespace(unified_mcp_service=_FakeService(state=ALLOW))
    setting = controller_module.get_cli_setting
    monkeypatch.setattr(
        controller_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True
            if (section, key) == ("console", "local_tools_enabled")
            else setting(section, key, default)
        ),
    )
    try:
        await co.dispatch_once(goal.id)
        text = str(calls)
        assert "SOURCE AGENTS MUST NEVER ACTIVATE" not in text
        assert "UNSELECTED SECRET" not in text
        assert "REPLACEMENT MUST NOT BE READ" not in text
        if change is None:
            assert "MODEL VISIBLE SOURCE DATA" in text
            assert (primary / "fixture.txt").read_text() == "valid"
            assert (source / "data.txt").read_text() == "MODEL VISIBLE SOURCE DATA"
            assert "outside the workspace root" in text
        else:
            assert "MODEL VISIBLE SOURCE DATA" not in text
            assert (primary / "fixture.txt").read_text() == "invalid"
    finally:
        await co.shutdown()
        await gateway.aclose()
