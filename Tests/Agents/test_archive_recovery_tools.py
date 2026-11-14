"""Recovery registration requires live application authority, never tool names."""

import pytest

from tldw_chatbook.Agents.archive_recovery_tool_provider import (
    RECOVERY_RESERVED_TOOL_NAMES,
    ArchiveRecoveryToolProvider,
)
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.archive_recovery.runtime import recovery_eligibility


@pytest.mark.parametrize(
    "access,direct,interactive,primary,expected",
    [
        ("allowed", True, True, True, True),
        ("allowed", False, True, True, True),
        ("blocked", True, True, True, False),
        ("allowed", False, False, True, False),
        ("allowed", True, True, False, False),
    ],
)
def test_recovery_admission(access, direct, interactive, primary, expected):
    assert (
        recovery_eligibility(
            assistant_access=access,
            direct_library_tools=direct,
            user_originated=interactive,
            primary=primary,
            temporary=False,
            local_session=True,
            answerable=True,
        )
        is expected
    )


def test_no_generic_constructor_or_reserved_name_can_issue_authority():
    registry = ToolCatalogRegistry()
    provider = ArchiveRecoveryToolProvider(object(), object())
    assert not registry.register_archive_recovery_provider(provider, object())
    registry.register_provider(provider)
    assert not (
        RECOVERY_RESERVED_TOOL_NAMES & {e.name for e in registry.list_catalog()}
    )
    for name in RECOVERY_RESERVED_TOOL_NAMES:
        assert not registry.invoke_by_name(name, {}).ok


import asyncio
from copy import copy
from types import SimpleNamespace

from Tests.Agents.test_library_name_reservation import _MCPProvider, _provider_for
from Tests.Chat.test_archive_recovery_operations import (
    recovery_stores as recovery_stores,  # noqa: PLC0414 - pytest fixture export
)
from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor
from tldw_chatbook.Chat.archive_recovery.runtime import ArchiveRecoveryRuntime
from tldw_chatbook.Chat.console_agent_bridge import _compose_run_registry_and_allowed
from tldw_chatbook.Chat.console_library_policy import ConsoleAssistantLibraryAccess


def admitted(
    runtime, *, sid="source", allowed=True, temporary=False, local=True, proof=True
):
    token = runtime.issue_user_proof(sid) if proof else object()
    runtime.accept_user_proof(token, sid, "accepted-turn")
    context = SimpleNamespace(
        configuration=SimpleNamespace(recovery_user_origin=token),
        library_authority=SimpleNamespace(
            direct_library_tools=False,
            policy=SimpleNamespace(
                assistant_access=ConsoleAssistantLibraryAccess.ALLOWED
                if allowed
                else ConsoleAssistantLibraryAccess.BLOCKED
            ),
        ),
    )
    return runtime.admit(
        context,
        SimpleNamespace(
            id=sid, ephemeral=temporary, runtime_backend="local" if local else "server"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["direct", "rag"])
async def test_registration_coexists_and_pending_native_teardown_are_one_way(
    recovery_stores, mode
):
    app, cid, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    runtime.is_answerable = lambda _: True
    provider = admitted(runtime)
    registry = ToolCatalogRegistry()
    library, authority = _provider_for(mode)
    assert registry.register_builtin_library_provider(library, authority)
    assert not registry.register_archive_recovery_provider(
        provider, copy(provider.authority)
    )
    assert registry.register_archive_recovery_provider(provider, provider.authority)
    search, _ = provider.build_closures()

    def call():
        with use_run_actor(CurrentRunActor("primary", "native-run", None)):
            return search({})

    assert not (await asyncio.to_thread(call)).ok
    assert await asyncio.to_thread(provider.activate, "native-run")
    assert not await asyncio.to_thread(provider.activate, "native-run")
    assert not await asyncio.to_thread(provider.activate, "other-run")
    assert cid in (await asyncio.to_thread(call)).content
    await asyncio.to_thread(provider.close_run, "native-run")
    assert not (await asyncio.to_thread(call)).ok
    assert not (
        RECOVERY_RESERVED_TOOL_NAMES & {e.name for e in registry.list_catalog()}
    )
    await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [{"allowed": False}, {"temporary": True}, {"local": False}, {"proof": False}],
)
async def test_admission_cannot_be_minted_by_generic_turn_flags(
    recovery_stores, overrides
):
    app, _, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    runtime.is_answerable = lambda _: True
    assert admitted(runtime, **overrides) is None
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_foreign_issuer_subclass_and_spoofed_mcp_names_fail_closed(
    recovery_stores,
):
    app, _, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    runtime.is_answerable = lambda _: True
    provider = admitted(runtime)
    other = ArchiveRecoveryRuntime(app)
    registry = ToolCatalogRegistry()
    foreign = ArchiveRecoveryToolProvider(other, provider.authority)
    assert not registry.register_archive_recovery_provider(foreign, provider.authority)

    class Subclass(ArchiveRecoveryToolProvider):
        pass

    assert not registry.register_archive_recovery_provider(
        Subclass(runtime, provider.authority), provider.authority
    )
    spoof = _MCPProvider(RECOVERY_RESERVED_TOOL_NAMES)
    registry, allowed, _, _ = _compose_run_registry_and_allowed({}, mcp_provider=spoof)
    assert not (set(allowed) & RECOVERY_RESERVED_TOOL_NAMES)
    for name in RECOVERY_RESERVED_TOOL_NAMES:
        assert not registry.invoke_by_name(name, {}).ok
    assert not spoof.invoke_calls
    await runtime.shutdown()
    await other.shutdown()


def test_spoofed_issuer_cannot_authenticate_even_exact_provider_class():
    issuer = SimpleNamespace(authenticates=lambda _: True)
    authority = object()
    provider = ArchiveRecoveryToolProvider(issuer, authority)
    assert not ToolCatalogRegistry().register_archive_recovery_provider(
        provider, authority
    )


@pytest.mark.asyncio
async def test_child_and_foreign_actor_cannot_use_activated_primary(recovery_stores):
    app, _, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    runtime.is_answerable = lambda _: True
    provider = admitted(runtime)
    await asyncio.to_thread(provider.activate, "primary")
    search, restore = provider.build_closures()
    for actor in (
        CurrentRunActor("subagent", "child", "primary"),
        CurrentRunActor("primary", "foreign", None),
    ):

        def invoke(actor=actor):
            with use_run_actor(actor):
                return search({}), restore({"result_ref": "forged"}, call_id="call")

        results = await asyncio.to_thread(invoke)
        assert all(not result.ok for result in results)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_native_recovery_uses_existing_persona_call_cap(recovery_stores):
    from tldw_chatbook.Agents.run_tool_policy import RunToolPolicy

    app, _, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    runtime.is_answerable = lambda _: True
    provider = admitted(runtime)
    registry = ToolCatalogRegistry()
    assert registry.register_archive_recovery_provider(provider, provider.authority)
    registry.set_run_tool_policy(RunToolPolicy({"search_archived_conversations": 1}))
    await asyncio.to_thread(provider.activate, "native-cap")
    with use_run_actor(CurrentRunActor("primary", "native-cap", None)):
        assert (
            registry.check_archive_recovery_call("search_archived_conversations")
            is None
        )
        refusal = registry.check_archive_recovery_call("search_archived_conversations")
        assert not refusal.ok and "call_cap_reached" in refusal.error
    await runtime.shutdown()
