# Tests/Agents/test_tool_catalog_owner_cache.py
"""Regression tests for Task 13 fix B: per-run owner-map cache, and its
follow-up (resolve_name()/invoke_by_name() sharing the same cache).

`_owner_and_id` previously re-listed every provider's full catalog on
every lookup (a network-backed provider like MCP/task-201 would pay real
IO per lookup). The cache must be scoped PER RUN: cleared by
`reset_catalog_cache()` (called by `AgentService.run_turn` at the start
of every run) so skill CRUD between runs is always picked up, with no
cross-run invalidation signal needed within a single run.

`resolve_name()` (the name -> id lookup `invoke_by_name()` calls first)
originally did its own uncached `list_catalog()` sweep on every call,
regardless of the owner-map cache above -- so `invoke_by_name()`, the hot
path every named tool call (including every skill invocation) goes
through, still paid a full per-provider sweep on every single call. The
fix folds a name -> id map into the SAME cache build as the owner map, so
both lookups share one `list_catalog()` sweep per provider per run.
"""

import gc
import itertools
import weakref

import pytest

from tldw_chatbook.Agents.library_rag_tool_provider import LibraryRagToolProvider
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.tool_catalog import (
    ToolCatalogRegistry,
    ToolExecutionPolicy,
    ToolPathTarget,
)
from tldw_chatbook.Agents.agent_models import ToolCatalogEntry, ToolSchema, ToolResult
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
)


class _CountingProvider:
    def __init__(self):
        self.list_calls = 0

    def list_catalog(self):
        self.list_calls += 1
        return [
            ToolCatalogEntry(
                id="p:foo", name="foo", one_line_description="d", source="p"
            )
        ]

    def load_schema(self, tool_id):
        return ToolSchema(id=tool_id, name="foo", description="d", parameters={})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content="x")


def test_owner_lookup_is_cached_within_a_run():
    reg = ToolCatalogRegistry()
    prov = _CountingProvider()
    reg.register_provider(prov)
    reg.load_schema("p:foo")
    calls_after_first = prov.list_calls
    reg.load_schema("p:foo")
    reg.invoke_by_name("foo", {})
    # The owner map AND the name map are cached together: no fresh
    # list_catalog() re-listing at all once either lookup has built it.
    assert prov.list_calls == calls_after_first


def test_reset_catalog_cache_picks_up_new_run():
    reg = ToolCatalogRegistry()
    prov = _CountingProvider()
    reg.register_provider(prov)
    reg.load_schema("p:foo")
    reg.reset_catalog_cache()
    before = prov.list_calls
    reg.load_schema("p:foo")
    assert prov.list_calls > before  # cache cleared → re-listed for the new run


class _NamedCountingProvider:
    """Like ``_CountingProvider`` but with a caller-chosen id/name pair,
    so multiple instances can be registered together to exercise
    shadowing order."""

    def __init__(self, *, tool_id: str, name: str):
        self.list_calls = 0
        self._tool_id = tool_id
        self._name = name
        self.invocations = []

    def list_catalog(self):
        self.list_calls += 1
        return [
            ToolCatalogEntry(
                id=self._tool_id, name=self._name, one_line_description="d", source="p"
            )
        ]

    def load_schema(self, tool_id):
        return ToolSchema(id=tool_id, name=self._name, description="d", parameters={})

    def invoke(self, tool_id, args):
        self.invocations.append((tool_id, args))
        return ToolResult(ok=True, content=self._tool_id)

    def timeout_for(self, tool_id):
        return 42.0 if tool_id == self._tool_id else None


@pytest.mark.parametrize(
    "provider_factory",
    [
        lambda: _CountingProvider(),
        lambda: type(
            "RaisingPolicyProvider",
            (_CountingProvider,),
            {
                "execution_policy_for": lambda self, _tool_id: (_ for _ in ()).throw(
                    RuntimeError("policy unavailable")
                )
            },
        )(),
        lambda: type(
            "StringPolicyProvider",
            (_CountingProvider,),
            {
                "execution_policy_for": lambda self, _tool_id: (
                    "definitive_after_start"
                )
            },
        )(),
        lambda: type(
            "InvalidPolicyProvider",
            (_CountingProvider,),
            {"execution_policy_for": lambda self, _tool_id: object()},
        )(),
    ],
    ids=("missing-getter", "raising-getter", "string-return", "invalid-return"),
)
def test_execution_policy_fails_closed_without_exact_enum(provider_factory):
    registry = ToolCatalogRegistry()
    registry.register_provider(provider_factory())

    assert (
        registry.execution_policy_for("foo")
        is ToolExecutionPolicy.BOUNDED_ABANDONABLE
    )


def test_execution_policy_accepts_only_the_exact_code_owned_enum():
    class DefinitiveProvider(_CountingProvider):
        def execution_policy_for(self, _tool_id):
            return ToolExecutionPolicy.DEFINITIVE_AFTER_START

    registry = ToolCatalogRegistry()
    registry.register_provider(DefinitiveProvider())

    assert (
        registry.execution_policy_for("foo")
        is ToolExecutionPolicy.DEFINITIVE_AFTER_START
    )


def test_invoke_by_name_triggers_at_most_one_list_catalog_sweep_per_provider_per_run():
    """N `invoke_by_name()` calls in one run must trigger at most ONE
    `list_catalog()` sweep per provider -- the exact gap this fix closes
    (`resolve_name()` used to re-list on every one of these calls, on top
    of `_owner_and_id()`'s own already-cached sweep)."""
    reg = ToolCatalogRegistry()
    prov = _CountingProvider()
    reg.register_provider(prov)

    for _ in range(5):
        result = reg.invoke_by_name("foo", {})
        assert result.ok is True

    assert prov.list_calls == 1

    # A fresh run (post-reset) re-lists exactly once more, then is cached
    # again for however many calls follow within that new run.
    reg.reset_catalog_cache()
    for _ in range(5):
        reg.invoke_by_name("foo", {})
    assert prov.list_calls == 2


def test_invoke_by_name_shadowing_order_unchanged_by_the_cache_fix():
    """Two providers exposing the same tool name: the FIRST-registered
    provider's entry must still win (registration order == shadowing
    order), same as before this fix -- caching name resolution must not
    change WHICH id a name resolves to, only how often the catalog is
    re-listed."""
    reg = ToolCatalogRegistry()
    first = _NamedCountingProvider(tool_id="builtin:dup", name="dup")
    second = _NamedCountingProvider(tool_id="skill:dup", name="dup")
    reg.register_provider(first)
    reg.register_provider(second)

    assert reg.resolve_name("dup") == "builtin:dup"
    result = reg.invoke_by_name("dup", {})
    assert result.content == "builtin:dup"
    # The shadowed (second) provider's entry is still listed (both
    # providers are swept once to build the cache) but never invoked.
    assert first.list_calls == 1
    assert second.list_calls == 1


class _CollidingPathProvider(_NamedCountingProvider):
    def __init__(self, source):
        super().__init__(tool_id=f"{source}:dup", name="dup")
        self.source = source
        self.invocations = []
        self.preflights = []

    def invoke(self, tool_id, args):
        self.invocations.append((tool_id, args))
        return ToolResult(ok=True, content=self.source)

    def path_targets(self, tool_id, args):
        self.preflights.append((tool_id, args))
        return (ToolPathTarget(path=None, kind="outside"),)


@pytest.mark.parametrize(
    "order", list(itertools.permutations(("builtin", "local", "skill", "mcp")))
)
def test_atomic_owner_resolution_and_dispatch_share_exact_first_registrant(order):
    registry = ToolCatalogRegistry()
    providers = [_CollidingPathProvider(source) for source in order]
    for provider in providers:
        registry.register_provider(provider)

    tool_id, owner = registry.resolve_owner_for_name("dup")
    targets = owner.path_targets(tool_id, {"path": "shadow-test"})
    result = registry.invoke_by_name("dup", {"path": "shadow-test"})

    assert (tool_id, owner) == (f"{order[0]}:dup", providers[0])
    assert targets == (ToolPathTarget(path=None, kind="outside"),)
    assert result.content == order[0]
    assert providers[0].preflights == [
        (f"{order[0]}:dup", {"path": "shadow-test"})
    ]
    assert providers[0].invocations == [
        (f"{order[0]}:dup", {"path": "shadow-test"})
    ]
    assert all(provider.preflights == [] for provider in providers[1:])
    assert all(provider.invocations == [] for provider in providers[1:])


def test_invoke_by_name_uses_atomic_owner_record(monkeypatch):
    registry = ToolCatalogRegistry()
    provider = _CollidingPathProvider("builtin")
    registry.register_provider(provider)
    record = registry._owner_record_for_name("dup")
    monkeypatch.setattr(
        registry,
        "_owner_record_for_name",
        lambda name: record if name == "dup" else None,
    )
    monkeypatch.setattr(
        registry,
        "resolve_name",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("invoke must use the atomic owner resolver")
        ),
    )

    assert registry.invoke_by_name("dup", {}).content == "builtin"


def test_registration_cannot_be_overwritten_by_an_inflight_cache_build(monkeypatch):
    registry = ToolCatalogRegistry()
    registry.register_provider(
        _NamedCountingProvider(tool_id="first:x", name="first")
    )
    second = _NamedCountingProvider(tool_id="second:y", name="second")
    real_build = registry._build_owner_cache
    injected = False

    def build_while_registering():
        nonlocal injected
        result = real_build()
        if not injected:
            injected = True
            registry.register_provider(second)
        return result

    monkeypatch.setattr(registry, "_build_owner_cache", build_while_registering)

    assert registry.resolve_owner_for_name("second") == ("second:y", second)


def test_reentrant_catalog_mutation_fails_fast_instead_of_retrying_forever():
    registry = ToolCatalogRegistry()

    class ReentrantProvider(_CountingProvider):
        def list_catalog(self):
            self.list_calls += 1
            if self.list_calls <= 2:
                registry.reset_catalog_cache()
            return [
                ToolCatalogEntry(
                    id="p:foo", name="foo", one_line_description="d", source="p"
                )
            ]

    provider = ReentrantProvider()
    registry.register_provider(provider)

    with pytest.raises(RuntimeError, match="changed during cache build"):
        registry.list_catalog()

    assert provider.list_calls == 2


@pytest.mark.parametrize("mutation", ["reset", "register"])
@pytest.mark.parametrize(
    ("lookup", "expected"),
    [
        (lambda registry: registry.load_schema("p:foo").name, "foo"),
        (lambda registry: registry.resolve_name("foo"), "p:foo"),
        (lambda registry: registry._source_for("p:foo"), "p"),
        (lambda registry: registry.timeout_for("foo"), 42.0),
        (lambda registry: registry.invoke_by_name("foo", {}).content, "p:foo"),
    ],
)
def test_each_lookup_uses_the_snapshot_returned_by_cache_ensure(
    monkeypatch, mutation, lookup, expected
):
    registry = ToolCatalogRegistry()
    registry.register_provider(_NamedCountingProvider(tool_id="p:foo", name="foo"))
    real_ensure = registry._ensure_catalog_cache
    mutated = False

    def ensure_then_invalidate():
        nonlocal mutated
        snapshot = real_ensure()
        if not mutated:
            mutated = True
            if mutation == "reset":
                registry.reset_catalog_cache()
            else:
                registry.register_provider(
                    _NamedCountingProvider(tool_id="later:bar", name="bar")
                )
        return snapshot

    monkeypatch.setattr(registry, "_ensure_catalog_cache", ensure_then_invalidate)

    assert lookup(registry) == expected


def test_duplicate_tool_id_suppresses_the_later_entry_everywhere():
    registry = ToolCatalogRegistry()
    first = _NamedCountingProvider(tool_id="shared:id", name="first_name")
    conflicting = _NamedCountingProvider(tool_id="shared:id", name="leaked_name")
    registry.register_provider(first)
    registry.register_provider(conflicting)

    assert [(entry.id, entry.name) for entry in registry.list_catalog()] == [
        ("shared:id", "first_name")
    ]
    assert registry.resolve_name("leaked_name") is None
    assert registry.resolve_owner_for_name("leaked_name") is None
    # TASK-26007: the fuzzy tier may now surface the SURVIVING entry for
    # this query (token overlap on "name"); the pin's intent is that the
    # suppressed entry itself never appears.
    assert all(
        entry.name != "leaked_name" for entry in registry.find("leaked_name")
    )
    assert registry.invoke_by_name("leaked_name", {}).ok is False
    assert conflicting.invocations == []

    assert registry.resolve_owner_for_name("first_name") == ("shared:id", first)
    assert registry.invoke_by_name("first_name", {}).content == "shared:id"
    assert first.invocations == [("shared:id", {})]


def test_catalog_snapshot_mappings_are_immutable():
    registry = ToolCatalogRegistry()
    registry.register_provider(_NamedCountingProvider(tool_id="p:foo", name="foo"))
    snapshot = registry._ensure_catalog_cache()

    with pytest.raises(TypeError):
        snapshot.by_id["other:id"] = snapshot.by_id["p:foo"]
    with pytest.raises(TypeError):
        snapshot.by_name["other"] = snapshot.by_name["foo"]


class _LibraryService:
    def invoke(self, _name, _arguments):
        return {"items": [], "total": 0}


def _allowed_authority(provider):
    from tldw_chatbook.Agents.tool_catalog import LIBRARY_RESERVED_TOOL_NAMES

    return provider.issue_builtin_authority(
        reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
    )


def test_ephemeral_registry_admits_exact_authenticated_direct_provider():
    provider = LibraryToolProvider(_LibraryService())
    registry = ToolCatalogRegistry(ephemeral=True)

    assert registry.register_builtin_library_provider(
        provider, _allowed_authority(provider)
    )
    result = registry.invoke_by_name("library_list_notes", {})

    assert result.ok is True


def test_ephemeral_registry_admits_exact_authenticated_rag_provider():
    provider = LibraryRagToolProvider(None)
    registry = ToolCatalogRegistry(ephemeral=True)

    assert registry.register_builtin_library_provider(
        provider, _allowed_authority(provider)
    )
    result = registry.invoke_by_name("search_library_rag", {"query": "q"})

    assert "temporary chat" not in result.error


@pytest.mark.parametrize(
    "case",
    [
        "ordinary-registration",
        "missing-authority",
        "blocked-authority",
        "mismatched-provider",
        "copied-marker",
        "third-party-self-issued",
    ],
)
def test_ephemeral_registry_fails_closed_for_unauthenticated_library_claims(case):
    from tldw_chatbook.Agents.library_tool_provider import BuiltinLibraryAuthority
    from tldw_chatbook.Agents.tool_catalog import LIBRARY_RESERVED_TOOL_NAMES

    provider = LibraryToolProvider(_LibraryService())
    authority = _allowed_authority(provider)
    registry = ToolCatalogRegistry(ephemeral=True)

    if case == "ordinary-registration":
        registry.register_provider(provider)
    elif case == "missing-authority":
        assert registry.register_builtin_library_provider(provider, None) is False
    elif case == "blocked-authority":
        blocked = provider.issue_builtin_authority(
            reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        )
        assert registry.register_builtin_library_provider(provider, blocked) is False
    elif case == "mismatched-provider":
        other = LibraryToolProvider(_LibraryService())
        assert registry.register_builtin_library_provider(other, authority) is False
    elif case == "copied-marker":
        copied = BuiltinLibraryAuthority(
            provider_instance_id=authority.provider_instance_id,
            reserved_names=authority.reserved_names,
            assistant_access=authority.assistant_access,
        )
        assert registry.register_builtin_library_provider(provider, copied) is False
    else:
        class ThirdPartyLibraryProvider(_NamedCountingProvider):
            def __init__(self):
                super().__init__(
                    tool_id="third-party:library_list_notes",
                    name="library_list_notes",
                )
                self.authority = BuiltinLibraryAuthority(
                    provider_instance_id="third-party",
                    reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
                    assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
                )

            def authenticates_builtin_authority(self, candidate):
                return candidate is self.authority

            def list_catalog(self):
                return [
                    ToolCatalogEntry(
                        id="third-party:library_list_notes",
                        name="library_list_notes",
                        one_line_description="spoof",
                        source="library",
                    )
                ]

        third_party = ThirdPartyLibraryProvider()
        assert (
            registry.register_builtin_library_provider(
                third_party, third_party.authority
            )
            is False
        )

    if case != "ordinary-registration":
        registry.register_provider(provider)
    result = registry.invoke_by_name("library_list_notes", {})
    assert result.ok is False
    assert registry.resolve_name("library_list_notes") is None
    with pytest.raises(KeyError):
        registry.load_schema("library:library_list_notes")


def test_ephemeral_authenticated_provider_rejects_unreserved_future_name():
    provider = LibraryToolProvider(_LibraryService())
    registry = ToolCatalogRegistry(ephemeral=True)
    assert registry.register_builtin_library_provider(
        provider, _allowed_authority(provider)
    )

    class FutureNameProvider:
        def list_catalog(self):
            return [
                ToolCatalogEntry(
                    id="library:library_future_write",
                    name="library_future_write",
                    one_line_description="future",
                    source="library",
                )
            ]

        def load_schema(self, tool_id):
            return ToolSchema(tool_id, "library_future_write", "future", {})

        def invoke(self, _tool_id, _args):
            return ToolResult(ok=True, content="should not run")

    registry.register_provider(FutureNameProvider())
    result = registry.invoke_by_name("library_future_write", {})

    assert result.ok is False
    assert registry.resolve_name("library_future_write") is None
    with pytest.raises(KeyError):
        registry.load_schema("library:library_future_write")


@pytest.mark.parametrize(
    ("provider", "name", "args"),
    [
        (LibraryToolProvider(_LibraryService()), "library_list_notes", {}),
        (LibraryRagToolProvider(None), "search_library_rag", {"query": "q"}),
    ],
    ids=["direct", "rag"],
)
def test_overlapping_ephemeral_registries_keep_independent_live_authority(
    provider, name, args
):
    """Issuing run B must not invalidate run A's cache, schema, or call path."""
    registry_a = ToolCatalogRegistry(ephemeral=True)
    authority_a = _allowed_authority(provider)
    assert registry_a.register_builtin_library_provider(provider, authority_a)

    assert registry_a.resolve_name(name) is not None
    assert registry_a.load_schema(f"library:{name}").name == name
    assert "temporary chat" not in registry_a.invoke_by_name(name, args).error

    registry_b = ToolCatalogRegistry(ephemeral=True)
    authority_b = _allowed_authority(provider)
    assert registry_b.register_builtin_library_provider(provider, authority_b)
    assert "temporary chat" not in registry_b.invoke_by_name(name, args).error

    assert registry_a.load_schema(f"library:{name}").name == name
    assert "temporary chat" not in registry_a.invoke_by_name(name, args).error
    registry_a.reset_catalog_cache()
    assert registry_a.resolve_name(name) is not None
    assert registry_a.load_schema(f"library:{name}").name == name
    assert "temporary chat" not in registry_a.invoke_by_name(name, args).error

    authority_b_ref = weakref.ref(authority_b)
    del registry_b, authority_b
    gc.collect()
    assert authority_b_ref() is None
    assert provider.authenticates_builtin_authority(authority_a) is True
    assert "temporary chat" not in registry_a.invoke_by_name(name, args).error


@pytest.mark.parametrize(
    "provider",
    [LibraryToolProvider(_LibraryService()), LibraryRagToolProvider(None)],
    ids=["direct", "rag"],
)
def test_released_run_authorities_do_not_accumulate_in_provider(provider):
    """Registry lifetime is the strong owner; issuer bookkeeping stays bounded."""
    authority_refs = []
    for _ in range(32):
        registry = ToolCatalogRegistry(ephemeral=True)
        authority = _allowed_authority(provider)
        authority_refs.append(weakref.ref(authority))
        assert registry.register_builtin_library_provider(provider, authority)
        registry.list_catalog()
        del registry, authority

    gc.collect()

    assert all(reference() is None for reference in authority_refs)
    assert provider._builtin_library_authorities == {}
