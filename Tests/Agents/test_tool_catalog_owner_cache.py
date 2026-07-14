# Tests/Agents/test_tool_catalog_owner_cache.py
"""Regression tests for Task 13 fix B: per-run owner-map cache.

`_owner_and_id` previously re-listed every provider's full catalog on
every lookup (a network-backed provider like MCP/task-201 would pay real
IO per lookup). The cache must be scoped PER RUN: cleared by
`reset_catalog_cache()` (called by `AgentService.run_turn` at the start
of every run) so skill CRUD between runs is always picked up, with no
cross-run invalidation signal needed within a single run.
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
    # The owner map is cached: no fresh list_catalog() re-listing per lookup.
    assert prov.list_calls <= calls_after_first + 1  # resolve_name may list once; owner is cached


def test_reset_catalog_cache_picks_up_new_run():
    reg = ToolCatalogRegistry()
    prov = _CountingProvider()
    reg.register_provider(prov)
    reg.load_schema("p:foo")
    reg.reset_catalog_cache()
    before = prov.list_calls
    reg.load_schema("p:foo")
    assert prov.list_calls > before  # cache cleared → re-listed for the new run
