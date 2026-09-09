"""Owned child-process driver for goal restart evidence (invoked with python -m)."""

import asyncio
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Chat.test_automatic_provider_budget import resolution
from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_console_goal_scheduling import progress
from Tests.Chat.test_goal_cli_verification import trusted_skill
from Tests.Chat.test_goal_conversation_provisioning import stores
from tldw_chatbook.Agents.goal_models import VerificationSpec
from tldw_chatbook.Agents.goal_run_service import GoalRunService
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_goal_runs import ConsoleGoalCoordinator
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

root, phase, boundary = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
assert (
    Path(os.environ["TLDW_CONFIG_PATH"]).is_relative_to(
        Path(os.environ["TMPDIR"]).parent
    )
    or "pytest" in os.environ["TLDW_CONFIG_PATH"]
)
assert Path(__import__("tldw_chatbook").__file__).resolve().is_relative_to(Path.cwd())


def stamp(path, payload):
    pending = path.with_suffix(".pending")
    with pending.open("w") as stream:
        json.dump(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def count_provider():
    with (root / "provider-operations").open("a") as stream:
        stream.write("call\n")
        stream.flush()
        os.fsync(stream.fileno())


async def main():
    if phase == "produce":
        fixture = stores.__wrapped__(root)
        runs, persistence, registry, req = next(fixture)
        (Path(req.binding.locator) / "fixture.txt").write_text("input")
        script = f"from pathlib import Path\np=Path({str(root / 'tool-operations')!r})\nwith p.open('a') as f: f.write('tool\\n'); f.flush()\nprint('checked')\n"
        scope, path, trust = trusted_skill(root, script)
        req = req.model_copy(
            update={
                "verifiers": (
                    VerificationSpec(
                        id="check",
                        executor_tool_id="run_skill_script",
                        verifier_path=str(path),
                        verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        skill_trust_ref=trust.current_fingerprint_digest("verifier"),
                        input_paths=("fixture.txt",),
                    ),
                )
            }
        )
        attempts = 0

        def provider(**kw):
            nonlocal attempts
            count_provider()
            attempts += 1
            if attempts == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "check",
                                        "type": "function",
                                        "function": {
                                            "name": "run_skill_script",
                                            "arguments": json.dumps(
                                                {
                                                    "skill_name": "verifier",
                                                    "script_path": "scripts/check.py",
                                                    "args": [],
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            if boundary == "accepted":
                saved = co.service.get(goal.id)
                stamp(
                    root / "boundary.json",
                    {
                        "goal": goal.id,
                        "used": dict(saved.accounting.used),
                        "reserved": dict(saved.accounting.reserved),
                        "deadline": saved.accounting.deadline_at,
                    },
                )
                threading.Event().wait(60)
            return progress(attempts)

        patch = pytest.MonkeyPatch()
        goal, store, session, controller, co, gateway, _ = build_goal_rig(
            (runs, persistence, registry, req), patch, provider
        )
        controller._agent_bridge._skills_service = scope
        controller.set_pending_skill_script = lambda *a, **kw: None
        result = await co.dispatch_once(goal.id)
        saved = co.service.checkpoint(result)
        assert saved.status == "ready", saved
        stamp(
            root / "boundary.json",
            {
                "goal": goal.id,
                "used": dict(saved.accounting.used),
                "reserved": dict(saved.accounting.reserved),
                "deadline": saved.accounting.deadline_at,
            },
        )
        await asyncio.Event().wait()
    else:
        state = json.loads((root / "boundary.json").read_text())
        runs = AgentRunsDB(root / "runs.db")
        chat = CharactersRAGDB(root / "chat.db", "test")
        registry = LocalWorkspaceRegistryService(WorkspaceDB(root / "workspace.db"))
        persistence = ChatPersistenceService(chat, workspace_registry=registry)
        service = GoalRunService(runs, persistence)

        def provider(**kw):
            count_provider()
            return progress(100)

        gateway = ConsoleProviderGateway(chat_api_call_fn=provider)

        async def resolve(_selection):
            return resolution()

        gateway.resolve_for_send = resolve
        store = ConsoleChatStore(persistence=persistence)
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            agent_bridge=ConsoleAgentBridge(
                agent_runs_db=runs, store=store, provider_gateway=gateway
            ),
            agent_runtime_enabled=True,
        )
        co = ConsoleGoalCoordinator(controller, service)
        controller._goal_coordinator = co
        controller.fleet_wake.start_recovery()
        await controller.fleet_wake.wait_for_recovery()
        service.project_recovery(controller.fleet_wake.recovery_result)
        app = SimpleNamespace(
            chachanotes_db=chat,
            chat_conversation_scope_service=ChatConversationScopeService(
                local_service=ChatConversationService(chat), server_service=None
            ),
        )
        session = await co.restore_session(state["goal"], app=app)
        assert store.messages_for_session(session.id)
        saved = await co.start(state["goal"])
        assert saved.status == (
            "recovery_required" if boundary == "accepted" else "paused"
        ), saved
        for resource in state["used"]:
            assert (
                saved.accounting.used[resource] + saved.accounting.reserved[resource]
                == state["used"][resource] + state["reserved"][resource]
            )
        if boundary == "accepted":
            assert (
                saved.accounting.uncertain and saved.accounting.reserved["tokens"] > 0
            )
        else:
            assert dict(saved.accounting.used) == state["used"]
            assert dict(saved.accounting.reserved) == state["reserved"]
        assert saved.accounting.deadline_at == state["deadline"]
        assert saved.accounting.limits.generations == 3
        if boundary == "checkpoint":
            resumed = service.resume(saved.id, expected_revision=saved.revision)
            assert resumed.status == "ready"
            assert resumed.accounting.deadline_at == state["deadline"]
        stamp(
            root / "restarted.json",
            {
                "status": saved.status,
                "messages": len(store.messages_for_session(session.id)),
            },
        )
        await co.shutdown()
        await controller.shutdown()
        await gateway.aclose()


asyncio.run(main())
