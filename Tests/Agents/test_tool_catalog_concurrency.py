# Tests/Agents/test_tool_catalog_concurrency.py
"""Registry cache under concurrent lookups (fleet PR 2a).

The registry's own comment (tool_catalog.py:893-907) documents that
`_owner_cache` and `_name_to_id_cache` are rebuilt without a lock, so two
concurrent lookups can observe different generations. With N children on
their own threads sharing the bridge's long-lived registry, that stops
being exotic.
"""

import threading

from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry


def test_concurrent_resolution_never_sees_a_torn_cache():
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    known = [e.name for e in registry.list_catalog()]
    assert known, "catalog must be non-empty for this test to mean anything"

    errors = []
    barrier = threading.Barrier(8)

    def hammer():
        barrier.wait()
        for _ in range(200):
            registry.reset_catalog_cache()
            for name in known:
                tool_id = registry.resolve_name(name)
                if tool_id is None:
                    errors.append(f"resolve_name({name}) -> None")
                    return

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
