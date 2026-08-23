"""ADR-079 runtime provider selection and immutable authority tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.library_rag_tool_provider import (
    LibraryRagToolProvider,
    RAG_TOOL_NAME,
)
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Chat.console_agent_bridge import (
    _compose_run_registry_and_allowed,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS


class _DirectService:
    def invoke(self, _name, _arguments):
        return {"items": [], "total": 0}


def _context(
    access: ConsoleAssistantLibraryAccess,
    *,
    direct: bool,
    source: str = "durable",
) -> ConsoleTurnExecutionContext:
    policy = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=access,
        policy_revision=1 if source == "durable" else None,
        source=source,
        error_code="policy_read_error" if source == "unavailable" else None,
    )
    authority = ConsoleTurnLibraryAuthority(
        policy=policy,
        direct_library_tools=direct,
        source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
        scope_snapshot=ConsoleLibraryItemScopeSnapshot(
            note_ids=(), media_ids=(), conversations_allowed=True
        ),
        provider_intent=ConsoleProviderIntent(
            provider="openai", model="model-a", endpoint=None
        ),
        attempt_id=f"attempt-{source}-{direct}",
    )
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-a",
        provider_selection=ConsoleProviderSelection(
            provider="openai", explicit_model="model-a"
        ),
        # Deliberately opposite: the final authority, not this compatibility
        # mapping or a later global, owns Direct/RAG selection.
        tool_configuration={"direct_library_tools": not direct},
    )
    return ConsoleTurnExecutionContext(
        configuration=configuration,
        library_authority=authority,
        resolved_destination=ConsoleResolvedDestination(
            provider="openai",
            model="model-a",
            endpoint_identity="https://api.openai.com",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )


def _factory(context):
    if context.library_authority.direct_library_tools:
        return LibraryToolProvider(_DirectService())
    return LibraryRagToolProvider(None)


def _controller(factory=_factory):
    return ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=SimpleNamespace(),
        library_provider_factory=factory,
    )


@pytest.mark.parametrize(
    ("access", "source"),
    [
        (ConsoleAssistantLibraryAccess.BLOCKED, "durable"),
        (ConsoleAssistantLibraryAccess.BLOCKED, "unavailable"),
    ],
)
@pytest.mark.parametrize("direct", [False, True], ids=["rag", "direct"])
def test_blocked_or_unavailable_context_never_constructs_provider(
    access, source, direct
):
    calls = []

    def forbidden_factory(context):
        calls.append(context)
        raise AssertionError("blocked authority reached provider construction")

    controller = _controller(forbidden_factory)
    context = _context(access, direct=direct, source=source)

    assert controller._library_provider_for_context(context) is None
    assert calls == []

    registry, allowed, _builtins, _locals = _compose_run_registry_and_allowed(
        {}, library_provider=None, library_authority=None
    )
    reserved = set(LIBRARY_TOOL_DESCRIPTORS) | {RAG_TOOL_NAME}
    assert not reserved.intersection(allowed)
    assert not reserved.intersection(entry.name for entry in registry.list_catalog())
    for name in reserved:
        assert registry.resolve_name(name) is None
        assert registry.invoke_by_name(name, {}).ok is False


@pytest.mark.parametrize(
    ("direct", "expected_type", "expected_names"),
    [
        (True, LibraryToolProvider, frozenset(LIBRARY_TOOL_DESCRIPTORS)),
        (False, LibraryRagToolProvider, frozenset({RAG_TOOL_NAME})),
    ],
    ids=["direct-18", "rag-1"],
)
def test_allowed_context_selects_exact_captured_provider(
    direct, expected_type, expected_names
):
    context = _context(ConsoleAssistantLibraryAccess.ALLOWED, direct=direct)
    provider = _controller()._library_provider_for_context(context)

    assert type(provider) is expected_type
    assert frozenset(entry.name for entry in provider.list_catalog()) == expected_names
    authority = provider.builtin_authority
    assert authority.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED

    registry, allowed, _builtins, _locals = _compose_run_registry_and_allowed(
        {},
        library_provider=provider,
        library_authority=authority,
        ephemeral=True,
    )
    assert expected_names.issubset(allowed)
    assert frozenset(
        entry.name for entry in registry.list_catalog() if entry.source == "library"
    ) == expected_names


def test_provider_unavailable_or_wrong_for_captured_selector_fails_closed():
    direct = _context(ConsoleAssistantLibraryAccess.ALLOWED, direct=True)
    assert _controller(lambda _context: None)._library_provider_for_context(direct) is None
    assert (
        _controller(lambda _context: LibraryRagToolProvider(None))
        ._library_provider_for_context(direct)
        is None
    )

    rag = _context(ConsoleAssistantLibraryAccess.ALLOWED, direct=False)
    assert (
        _controller(lambda _context: LibraryToolProvider(_DirectService()))
        ._library_provider_for_context(rag)
        is None
    )


def test_allowed_then_blocked_builds_fresh_registry_without_cached_provider():
    controller = _controller()
    allowed_context = _context(ConsoleAssistantLibraryAccess.ALLOWED, direct=True)
    allowed_provider = controller._library_provider_for_context(allowed_context)
    allowed_registry, allowed_names, _builtins, _locals = (
        _compose_run_registry_and_allowed(
            {},
            library_provider=allowed_provider,
            library_authority=allowed_provider.builtin_authority,
        )
    )
    assert "library_list_notes" in allowed_names
    assert allowed_registry.resolve_name("library_list_notes") is not None

    blocked_context = _context(ConsoleAssistantLibraryAccess.BLOCKED, direct=True)
    blocked_provider = controller._library_provider_for_context(blocked_context)
    blocked_registry, blocked_names, _builtins, _locals = (
        _compose_run_registry_and_allowed(
            {}, library_provider=blocked_provider, library_authority=None
        )
    )

    assert blocked_provider is None
    assert "library_list_notes" not in blocked_names
    assert blocked_registry.resolve_name("library_list_notes") is None


def test_child_authority_can_narrow_but_never_widen_parent_registry():
    context = _context(ConsoleAssistantLibraryAccess.ALLOWED, direct=True)
    provider = _controller()._library_provider_for_context(context)
    registry, parent_allowed, _builtins, _locals = (
        _compose_run_registry_and_allowed(
            {},
            library_provider=provider,
            library_authority=provider.builtin_authority,
        )
    )
    reserved = frozenset(LIBRARY_TOOL_DESCRIPTORS)
    child_requested = {"library_get_note", "library_future_write"}
    child_allowed = tuple(name for name in parent_allowed if name in child_requested)

    assert set(parent_allowed).issuperset(reserved)
    assert child_allowed == ("library_get_note",)
    assert registry.resolve_name("library_future_write") is None


def test_mismatched_parent_child_authorities_do_not_cross_authenticate():
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    parent = LibraryToolProvider(_DirectService())
    child = LibraryToolProvider(_DirectService())
    from tldw_chatbook.Agents.tool_catalog import LIBRARY_RESERVED_TOOL_NAMES

    parent_authority = parent.issue_builtin_authority(
        reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
    )
    registry = ToolCatalogRegistry(ephemeral=True)

    assert registry.register_builtin_library_provider(child, parent_authority) is False
    assert registry.list_catalog() == []
