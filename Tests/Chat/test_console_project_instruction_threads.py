"""Selected-project sends retain Textual owning/worker thread affinity."""

import asyncio
import threading

import pytest
from textual.app import App

from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls
from Tests.Chat.test_console_agent_project_instructions import (
    SENTINEL,
    _binding,
    _RefusalProbeGateway,
)
from Tests.console_provider_doubles import persisted_console_store
from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionChainDelivery,
    InstructionSnapshot,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunStatus,
    ProjectInstructionActivationEvent,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

pytestmark = pytest.mark.bootstrap_profile

mcp_sources = snapshot_controls.mcp_sources
local_root = snapshot_controls.local_root
catalog_store = snapshot_controls.catalog_store
snapshot_case = snapshot_controls.snapshot_case


class _ThreadCheckingApp(App):
    """Keep real Textual scheduling and observe both sides of each hop."""

    def __init__(self):
        super().__init__()
        self.thread_hops = []

    def call_from_thread(self, callback, *args, **kwargs):
        caller = threading.get_ident()

        def observed():
            self.thread_hops.append((caller, threading.get_ident()))
            return callback(*args, **kwargs)

        return super().call_from_thread(observed)


@pytest.mark.asyncio
async def test_selected_project_send_composes_on_app_loop_and_rechecks_from_worker(
    snapshot_case, tmp_path
):
    # Keep the original composition methods: a replaced composer bypasses the
    # stock capture_run_tool_read currentness check that stranded accepted sends.
    case = snapshot_case
    snapshot_controls._loop_projection(case)
    (tmp_path / "AGENTS.md").write_text(SENTINEL, encoding="utf-8")
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.db", client_id="instruction-threads")
    )
    registry.create_workspace(workspace_id="w1", name="Workspace 1")
    registry.save_runtime_binding(_binding(tmp_path))
    store = persisted_console_store(
        db_path=tmp_path / "chat.db", workspace_registry=registry
    )
    session = store.create_session(workspace_id="w1")
    captured = {}
    event = ProjectInstructionActivationEvent(
        relative_sources=("AGENTS.md",), scopes=(".",), outcome_codes=()
    )

    class Bridge:
        def run_reply(self, **kwargs):
            captured.update(kwargs)
            captured["worker_thread"] = threading.get_ident()
            candidate = kwargs["startup_instruction_candidate"]
            assert candidate.source is not None
            snapshot = InstructionSnapshot(
                binding_id=candidate.binding_id,
                binding_root=candidate.binding_root,
                locator_fingerprint=candidate.locator_fingerprint,
                dispatch_started_wall_ns=candidate.dispatch_started_wall_ns,
                startup_source=candidate.source,
                global_outcomes=candidate.outcomes,
                primary_delivery=InstructionChainDelivery(
                    source_digests=(candidate.source.digest,),
                    outcomes=candidate.outcomes,
                ),
                warning_codes=(),
            )
            assert kwargs["confirm_project_instruction_dispatch"](snapshot) == "proceed"
            kwargs["on_project_instruction_activation"](event)
            return "run-1", RunOutcome(status=RUN_DONE, steps=[], final_text="done")

    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RefusalProbeGateway(),
        provider="openai",
        model="test-model",
        agent_bridge=Bridge(),
        agent_runtime_enabled=True,
        confirm_project_instruction_dispatch=lambda _notice: "proceed",
    )
    app = _ThreadCheckingApp()
    app.workspace_registry_service = registry
    app.unified_mcp_service = case.service
    controller.app = app

    async with app.run_test():
        result = await controller.submit_draft("question")

        assert result.accepted is True
        assert controller.run_state_for(session.id).status is ConsoleRunStatus.COMPLETED
        assert store.dispatch_recovery_for_session(session.id) is None
        assert captured["worker_thread"] != app._thread_id
        assert controller.project_instruction_activation_events(session.id) == (event,)
        assert (
            controller.project_instruction_display_metadata(session.id).outcome
            == "active"
        )

        guard = captured["local_provider"]._root_guard
        previous_hops = len(app.thread_hops)
        assert guard() is True
        assert len(app.thread_hops) == previous_hops
        assert await asyncio.to_thread(guard) is True
        assert len(app.thread_hops) == previous_hops + 1

        store.set_session_project_instruction_state(
            session.id, ProjectInstructionControlState.legacy_disabled()
        )
        assert await asyncio.to_thread(guard) is False
        assert len(app.thread_hops) == previous_hops + 2
        assert app.thread_hops
        assert all(
            caller != app._thread_id and owner == app._thread_id
            for caller, owner in app.thread_hops
        )
