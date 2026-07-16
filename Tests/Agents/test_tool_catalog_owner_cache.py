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
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Agents.agent_models import ToolCatalogEntry, ToolSchema, ToolResult


class _CountingProvider:
    def __init__(self):
        self.list_calls = 0

    def list_catalog(self):
        self.list_calls += 1
        return [ToolCatalogEntry(id="p:foo", name="foo", one_line_description="d", source="p")]

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

    def list_catalog(self):
        self.list_calls += 1
        return [ToolCatalogEntry(id=self._tool_id, name=self._name,
                                 one_line_description="d", source="p")]

    def load_schema(self, tool_id):
        return ToolSchema(id=tool_id, name=self._name, description="d", parameters={})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content=self._tool_id)


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
