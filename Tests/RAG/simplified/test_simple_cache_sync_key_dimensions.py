"""TASK-15701: the SYNC cache API must key on the same search-defining
dimensions the async one does.

`_make_key` takes eight inputs, three of which describe *what search was
actually run* — `keyword_source_types` (which sub-legs the keyword leg
queried), `hybrid_fusion` (alpha / rrf_k / pool) and `fts_match_construction`
(how the FTS MATCH expression was built). The async path passes all three;
the sync `get`/`put` did not accept them at all, so they rendered the legacy
key.

The failure that matters is not a missed hit (harmless) but a **wrong hit**:
two searches differing only in an omitted dimension collide, and the second
is served the first one's rows. Before TASK-15400 the legacy key was
accidentally truthful — `and` was both the legacy value and the shipped
default. The default has since moved twice (`and_stopword_trim`, then
`and_then_prefix`), so an omitted part now asserts a construction that did
not run.
"""

import asyncio

import pytest

from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache


def _cache() -> SimpleRAGCache:
    return SimpleRAGCache(max_size=32, ttl_seconds=300, enabled=True)


@pytest.mark.parametrize(
    "dimension,first,second",
    [
        ("fts_match_construction", "and", "and_then_prefix"),
        ("keyword_source_types", ("notes",), ("notes", "media")),
        ("hybrid_fusion", (0.7, 5, 2), (0.3, 60, 4)),
    ],
)
def test_sync_round_trips_differing_in_one_dimension_do_not_collide(
    dimension, first, second
):
    """AC#2: a search that differs ONLY in one search-defining dimension must
    not be served the other's rows.

    Args:
        dimension: The `_make_key` parameter under test.
        first: The value stored under.
        second: A different value that must not hit the first entry.
    """
    cache = _cache()
    rows_a = [{"id": "a"}]

    cache.put("q", "hybrid", 10, rows_a, "ctx-a", **{dimension: first})
    hit_same = cache.get("q", "hybrid", 10, **{dimension: first})
    hit_other = cache.get("q", "hybrid", 10, **{dimension: second})

    assert hit_same is not None, "the identical search must still hit"
    assert hit_same[0] == rows_a
    assert hit_other is None, (
        f"WRONG HIT: a search differing in {dimension} was served the other "
        f"search's rows ({first!r} vs {second!r})"
    )


def test_sync_and_async_render_the_same_key_for_the_same_search():
    """AC#3: the two paths must agree, shown rather than asserted separately.

    A sync `put` followed by an async `get_async` for the same search is the
    only check that cannot pass while the two key renderings differ.
    """
    cache = _cache()
    rows = [{"id": "shared"}]
    kwargs = dict(
        keyword_source_types=("notes", "media"),
        hybrid_fusion=(0.7, 5, 2),
        fts_match_construction="and_then_prefix",
    )

    cache.put("q", "hybrid", 10, rows, "ctx", **kwargs)
    hit = asyncio.run(cache.get_async("q", "hybrid", 10, **kwargs))

    assert hit is not None, "the async path did not find what the sync path stored"
    assert hit[0] == rows
