"""Media-type vocabulary check (spec §5 / §6.9, Task 3, AC 15).

Chatbook's ingest paths produce a fixed set of media-type strings; the
vendored planner switches its method choice on its own normalized
vocabulary. The frozen ``MEDIA_TYPE_MAP`` in
``tldw_chatbook.Chunking.auto_selection`` is the alignment between the two:
every chatbook-produced string maps (identity entries count), and every
mapping target is either identity or one of the planner's recognized types.
A vocabulary drift on either side fails here loudly instead of silently
disabling tier-2 per-type plans.
"""
from __future__ import annotations

from tldw_chatbook.Chunking.auto_selection import (
    KNOWN_INGEST_MEDIA_TYPES,
    MEDIA_TYPE_MAP,
)

#: The values ``engine.auto_planner._choose_method`` actually switches on
#: (plus "web", the target of its own ``_normalize_media_type`` set). A map
#: entry may only be identity or point at one of these — anything else is a
#: mapping onto a type the planner treats as generic, i.e. a no-op mapping
#: masquerading as alignment.
PLANNER_RECOGNIZED_TYPES = frozenset(
    {"web", "email", "ebook", "audio", "video", "document", "pdf"}
)


def test_every_ingest_media_type_is_mapped():
    assert set(KNOWN_INGEST_MEDIA_TYPES) <= set(MEDIA_TYPE_MAP)  # total coverage, identity entries count
    assert "web_document" not in MEDIA_TYPE_MAP.values() or MEDIA_TYPE_MAP["web_document"] == "web"  # normalization preserved


def test_known_ingest_media_types_is_a_frozen_tuple():
    # The plan freezes the vocabulary as a tuple: hashable, immutable,
    # order-stable for report generation. A drift to list/set here would
    # weaken the loudness contract of the subset pin above.
    assert isinstance(KNOWN_INGEST_MEDIA_TYPES, tuple)
    assert len(set(KNOWN_INGEST_MEDIA_TYPES)) == len(KNOWN_INGEST_MEDIA_TYPES)


def test_every_mapping_targets_a_planner_type_or_identity():
    # Identity entries are intentional ("no planner equivalent — keep the
    # generic plan"); non-identity entries must land on a type the planner
    # actually gives special treatment, or the entry is dead weight.
    for source, target in MEDIA_TYPE_MAP.items():
        assert target == source or target in PLANNER_RECOGNIZED_TYPES, (
            f"MEDIA_TYPE_MAP[{source!r}] = {target!r} is neither identity "
            "nor a planner-recognized type"
        )


def test_web_aliases_normalize_to_web():
    # The load-bearing alignment (spec §5): chatbook's web-content names
    # that the planner's own _normalize_media_type set does NOT cover.
    # Without these, every web_article/web_scraping item silently got the
    # generic sentence plan forever.
    for alias in ("web_article", "web_scraping", "webpage", "web_document", "html", "article"):
        assert MEDIA_TYPE_MAP.get(alias, alias) == "web"


def test_planner_recognized_types_pass_through_unchanged():
    # The types the planner natively understands must not be renamed by the
    # table — renaming one would change tier-2 behavior for a whole family.
    for native in ("pdf", "document", "ebook", "audio", "video", "email"):
        assert MEDIA_TYPE_MAP.get(native, native) == native
