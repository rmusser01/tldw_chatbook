"""An approval cannot extend accepted automatic authority or its deadline."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Agents.test_automatic_child_scope import accepted_context
from Tests.Agents.test_mcp_tool_provider import (
    FakeMCPService,
    _catalog_record,
    _compose,
    _tool_dict,
)
from Tests.Agents.test_mcp_tool_provider import (
    running_loop as _source_running_loop,
)
from Tests.Chat.test_console_agent_bridge import (
    _ChunkGateway,
    _fence,
    _install_skills_service,
    _run,
)
from Tests.DB.test_automatic_work_budget import _automatic_work_db  # noqa: F401
from Tests.MCP.test_local_control_service import FakeLocalStore, FakeMCPClient
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work
from tldw_chatbook.Agents.execution_capacity import WorkOrigin
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore, ConsoleMessageRole
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.permission_store import EffectiveToolState


@pytest.fixture(name="running_loop")
def _automatic_mcp_loop():
    yield from _source_running_loop.__wrapped__()


def mcp_provider(loop, *, approval=None, allow=False):
    service = FakeMCPService(
        catalog_records=[_catalog_record("srv", [_tool_dict("run")])],
        default_state=EffectiveToolState(
            state="allow" if allow else "ask", origin="global_default"
        ),
    )
    provider = MCPToolProvider(
        service=service, main_loop=loop, approval_callback=approval
    )
    _compose(provider)
    return provider, service, provider.list_catalog()[0].name


def test_mcp_approval_arriving_after_chain_pause_cannot_execute(db, running_loop):
    context = accepted_context(db)
    confirmations = []

    def approve(pending):
        confirmations.append(pending[0].llm_name)
        db.automatic_work.pause(context.chain_id, "wall_budget")
        return {pending[0].llm_name: "approve_once"}

    provider, backend, name = mcp_provider(running_loop, approval=approve)
    with context.scope():
        result = provider.invoke(name, {})
    assert confirmations == [name]
    assert not result.ok and "wall_budget" in result.error
    assert backend.execute_calls == []
    assert db.automatic_work.snapshot(context.chain_id).used["generation"] == 1


def test_mcp_checks_again_when_queued_coroutine_reaches_execution(
    db, running_loop, monkeypatch
):
    context = accepted_context(db)
    provider, backend, name = mcp_provider(running_loop, allow=True)
    entered, release, queued = threading.Event(), threading.Event(), threading.Event()
    results = []
    original_submit = asyncio.run_coroutine_threadsafe

    def submit(coro, loop):
        future = original_submit(coro, loop)
        queued.set()
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)

    def block_loop():
        entered.set()
        assert release.wait(5)

    def invoke():
        with context.scope():
            results.append(provider.invoke(name, {}))

    running_loop.call_soon_threadsafe(block_loop)
    thread = threading.Thread(target=invoke)
    try:
        assert entered.wait(5)
        thread.start()
        assert queued.wait(5)
        db.automatic_work.pause(context.chain_id, "wall_budget")
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert len(results) == 1
        assert not results[0].ok and "wall_budget" in results[0].error
        assert backend.execute_calls == []
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(5)


@pytest.mark.asyncio
async def test_external_mcp_connection_cannot_outlive_authority_then_dispatch(
    db, monkeypatch
):
    context = accepted_context(db)
    monkeypatch.setenv("API_KEY", "fixture-placeholder")
    sent = []

    class Client(FakeMCPClient):
        async def connect_to_server(self, *args, **kwargs):
            result = await super().connect_to_server(*args, **kwargs)
            db.automatic_work.pause(context.chain_id, "wall_budget")
            return result

        async def call_tool(self, profile_id, tool_name, args):
            sent.append((profile_id, tool_name, args, current_automatic_work()))
            return {"content": "called"}

    client = Client()
    service = LocalMCPControlService(
        store=FakeLocalStore(), client=client, manifest_provider=dict
    )
    with context.scope(), pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        await service.execute_external_tool("profile-a", "run", {})
    assert sent == []
    assert await service.execute_external_tool("profile-a", "run", {}) == {
        "content": "called"
    }
    assert sent == [("profile-a", "run", {}, None)]


def bridge_run(db, context, scope, tool_name, args, **kwargs):
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="go")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    gateway = _ChunkGateway([[_fence(tool_name, args)], ["must not continue"]])
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=gateway, skills_service=scope
    )
    with context.scope():
        outcome = _run(
            bridge,
            store,
            session,
            assistant.id,
            conversation_id="conversation",
            work_origin=WorkOrigin.AUTOMATIC,
            work_chain_id=context.chain_id,
            **kwargs,
        )
    return outcome, gateway


def test_install_skill_late_confirmation_cannot_fetch_or_install(db, monkeypatch):
    from tldw_chatbook.Skills_Interop import skill_remote_fetch

    context = accepted_context(db)
    installed, confirmed = [], []

    async def install(url, **kwargs):
        installed.append(url)
        return {"name": "demo"}

    def confirm(url):
        confirmed.append(url)
        db.automatic_work.pause(context.chain_id, "wall_budget")
        return True

    monkeypatch.setattr(skill_remote_fetch, "install_skill_from_url", install)
    outcome, gateway = bridge_run(
        db,
        context,
        _install_skills_service(),
        "install_skill",
        {"url": "https://github.com/o/r"},
        request_skill_install_confirm=confirm,
    )
    assert confirmed == ["https://github.com/o/r"]
    assert installed == []
    assert outcome.status == "cancelled"
    assert gateway.calls == 1


@pytest.mark.parametrize("standing_grant", [False, True])
def test_skill_script_rechecks_authority_after_confirmation_or_describe(
    db, standing_grant
):
    context = accepted_context(db)
    executed, confirmed = [], []

    class Scope:
        local_service = SimpleNamespace(
            trust_service=SimpleNamespace(
                script_execution_granted=lambda name: standing_grant,
            )
        )

        async def get_context(self, *, mode="local"):
            return {"available_skills": [], "blocked_skills": []}

        def enforce_run_script(self):
            pass

        async def describe_skill_script(self, name, path):
            if standing_grant:
                db.automatic_work.pause(context.chain_id, "wall_budget")
            return SimpleNamespace(
                skill_name=name,
                mechanism="interpreter",
                interpreter_display="python",
                is_binary=False,
            )

        async def run_skill_script(self, name, path, args):
            executed.append((name, path, args))
            return SimpleNamespace(
                exit_code=0,
                timed_out=False,
                output_capped=False,
                sandbox_warnings=(),
                stdout="",
                stderr="",
                output_files=(),
            )

    def confirm(payload):
        confirmed.append(payload)
        db.automatic_work.pause(context.chain_id, "wall_budget")
        return {"allow": True, "remember": False}

    outcome, gateway = bridge_run(
        db,
        context,
        Scope(),
        "run_skill_script",
        {"skill_name": "demo", "script_path": "scripts/run.py", "args": []},
        request_skill_script_confirm=confirm,
    )
    assert len(confirmed) == (0 if standing_grant else 1)
    assert executed == []
    assert outcome.status == "cancelled"
    assert gateway.calls == 1


def test_prepared_mcp_context_does_not_prompt_or_execute(db, running_loop):
    context = accepted_context(db, accepted=False)
    confirmations = []

    def approve(pending):
        confirmations.append(pending[0].llm_name)
        return {pending[0].llm_name: "approve_once"}

    provider, backend, name = mcp_provider(running_loop, approval=approve)
    with context.scope():
        result = provider.invoke(name, {})
    assert confirmations == []
    assert backend.execute_calls == []
    assert not result.ok and "acceptance_required" in result.error
