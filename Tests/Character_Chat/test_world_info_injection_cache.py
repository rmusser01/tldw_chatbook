"""Task 3 (TASK-34415 / ADR-221): world-info injection cold-start cache.

Three layers of evidence:

1. GOLDEN PINS — activation output captured from the pre-change processor on a
   fixture covering primary+secondary keys, case sensitivity, regex entries,
   disabled/keyless entries and a recursive activation CYCLE
   (castle → dragon → keep → castle). These pin byte-identical behavior across
   the caching/precompilation refactor. The duplicate-entries probe pins the
   equality-vs-identity dedup equivalence for the recursion change.
2. NEW-BEHAVIOR TESTS (3a-3e) — generation counter, resolver cache, precompiled
   keys, single entry processing, pre-collected-books seam.
3. ACCEPTANCE SPIES — a 1000-entry book across two sends: the second send must
   perform zero book queries and zero re.compile calls.
"""

import re

import pytest

import tldw_chatbook.Character_Chat.world_info_processor as wip_module
import tldw_chatbook.Character_Chat.world_info_resolver as resolver_module
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor
from tldw_chatbook.Character_Chat.world_info_regex import regex_search
from tldw_chatbook.Character_Chat.world_info_resolver import (
    apply_world_info_to_message,
    resolve_world_info_injection,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError


@pytest.fixture
def wb_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "wi_cache.db", "test-client")
    yield db
    db.close_connection()


@pytest.fixture(autouse=True)
def _isolated_resolver_cache():
    """The resolver cache is module-level; keep every test hermetic."""
    resolver_module._clear_world_info_cache()
    wip_module._compiled_keyword.cache_clear()
    yield
    resolver_module._clear_world_info_cache()
    wip_module._compiled_keyword.cache_clear()


def _attach(db, conv_id, key, content, name="Lore", book_extra=None):
    db.add_conversation({"id": conv_id, "title": "C"})
    wb = WorldBookManager(db)
    book_id = wb.create_world_book(name)
    wb.create_world_book_entry(book_id, keys=[key], content=content)
    wb.associate_world_book_with_conversation(conv_id, book_id)
    return wb, book_id


# ---------------------------------------------------------------------------
# Golden activation pins (captured from the pre-change implementation)
# ---------------------------------------------------------------------------


def _golden_book():
    return {
        "name": "Golden",
        "enabled": True,
        "scan_depth": 2,
        "token_budget": 100000,
        "recursive_scanning": True,
        "entries": [
            {
                "id": 1,
                "keys": ["castle"],
                "content": "The castle is guarded by an ancient dragon.",
                "enabled": True,
                "position": "before_char",
                "insertion_order": 1,
            },
            {
                "id": 2,
                "keys": ["dragon", "wyrm"],
                "content": "A dragon hoards gold inside the keep.",
                "enabled": True,
                "position": "before_char",
                "insertion_order": 2,
            },
            {
                "id": 3,
                "keys": ["keep"],
                "content": "The keep stands at the heart of the castle.",
                "enabled": True,
                "position": "after_char",
                "insertion_order": 3,
            },
            {
                "id": 4,
                "keys": ["sword", "blade"],
                "secondary_keys": ["chosen"],
                "selective": True,
                "content": "Only the chosen may draw the blade.",
                "enabled": True,
                "position": "at_start",
                "insertion_order": 4,
            },
            {
                "id": 5,
                "keys": ["castle"],
                "content": "DISABLED NEVER APPEARS",
                "enabled": False,
                "position": "before_char",
                "insertion_order": 5,
            },
            {
                "id": 6,
                "keys": ["HTML"],
                "content": "HTML is case-sensitive.",
                "enabled": True,
                "case_sensitive": True,
                "position": "before_char",
                "insertion_order": 6,
            },
            {
                "id": 7,
                "keys": ["gr(a|e)y"],
                "content": "Grey or gray scales shimmer.",
                "enabled": True,
                "regex": True,
                "position": "at_end",
                "insertion_order": 7,
            },
            {
                "id": 8,
                "keys": [],
                "content": "KEYLESS NEVER APPEARS",
                "enabled": True,
                "insertion_order": 8,
            },
        ],
    }


GOLDEN_MESSAGE = "The chosen knight drew his sword at the castle. HTML and grAy scales."

# Captured pre-change: direct matches (castle e1, sword+chosen e4, HTML e6,
# regex grAy e7) plus the recursive cycle e1→e2→e3 (and e3's content mentions
# castle → e1 again, exercising the recursion dedup), ordered by insertion
# order after the (priority, insertion_order) sort.
GOLDEN_MATCHED_ORDER = [
    "The castle is guarded by an ancient dragon.",
    "A dragon hoards gold inside the keep.",
    "The keep stands at the heart of the castle.",
    "Only the chosen may draw the blade.",
    "HTML is case-sensitive.",
    "Grey or gray scales shimmer.",
]
GOLDEN_INJECTIONS = {
    "before_char": [
        "The castle is guarded by an ancient dragon.",
        "A dragon hoards gold inside the keep.",
        "HTML is case-sensitive.",
    ],
    "after_char": ["The keep stands at the heart of the castle."],
    "at_start": ["Only the chosen may draw the blade."],
    "at_end": ["Grey or gray scales shimmer."],
}


def test_golden_activation_unchanged_recursive_cycle_selective_case_regex():
    proc = WorldInfoProcessor(world_books=[_golden_book()])
    result = proc.process_messages(GOLDEN_MESSAGE, [])
    assert [e["content"] for e in result["matched_entries"]] == GOLDEN_MATCHED_ORDER
    assert {k: v for k, v in result["injections"].items()} == GOLDEN_INJECTIONS


def test_golden_secondary_key_alone_still_does_not_fire():
    proc = WorldInfoProcessor(world_books=[_golden_book()])
    result = proc.process_messages("a lone sword lies here", [])
    assert result["matched_entries"] == []


def test_golden_duplicate_entries_and_recursive_dedup_equivalence():
    """Two byte-identical entries that both match directly must BOTH stay
    (the initial match loop never deduped — equality- and identity-based
    recursion dedup agree here), and recursion must add the downstream entry
    exactly once. Captured pre-change."""
    book = {
        "name": "Dup",
        "enabled": True,
        "token_budget": 100000,
        "recursive_scanning": True,
        "entries": [
            {
                "id": 11,
                "keys": ["alpha"],
                "content": "alpha lore mentions beta.",
                "enabled": True,
                "insertion_order": 1,
            },
            {
                "id": 12,
                "keys": ["alpha"],
                "content": "alpha lore mentions beta.",
                "enabled": True,
                "insertion_order": 2,
            },
            {
                "id": 13,
                "keys": ["beta"],
                "content": "beta lore.",
                "enabled": True,
                "insertion_order": 3,
            },
        ],
    }
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("alpha strikes", [])
    assert [e["content"] for e in result["matched_entries"]] == [
        "alpha lore mentions beta.",
        "alpha lore mentions beta.",
        "beta lore.",
    ]


# ---------------------------------------------------------------------------
# 3a — generation counter
# ---------------------------------------------------------------------------


def test_generation_starts_at_zero_and_is_shared_across_manager_instances(wb_db):
    assert WorldBookManager(wb_db).generation == 0
    wb = WorldBookManager(wb_db)
    wb.create_world_book("B")
    assert wb.generation == 1
    # The send path constructs a fresh manager per call: it must see the bump.
    assert WorldBookManager(wb_db).generation == 1


def test_generation_bumps_on_every_write_path(wb_db):
    wb = WorldBookManager(wb_db)
    db = wb_db
    db.add_conversation({"id": "gen-conv", "title": "C"})
    char_id = db.add_character_card({"name": "GenChar"})

    book_id = wb.create_world_book("W")  # 1
    entry_id = wb.create_world_book_entry(book_id, keys=["k"], content="c")  # 2
    wb.update_world_book(book_id, description="d")  # 3
    wb.update_world_book_entry(entry_id, content="c2")  # 4
    wb.associate_world_book_with_conversation("gen-conv", book_id)  # 5
    wb.disassociate_world_book_from_conversation("gen-conv", book_id)  # 6
    wb.associate_world_book_with_conversation("gen-conv", book_id)  # 7
    wb.attach_world_book_to_character(book_id, char_id)  # 8 (snapshot write)
    wb.detach_world_book_from_character(char_id, "W")  # 9
    wb.delete_world_book_entry(entry_id)  # 10
    wb.delete_world_book(book_id)  # 11
    assert wb.generation == 11

    # Import path bumps through its underlying creates.
    before = wb.generation
    wb.import_world_book(
        {"name": "Imported", "entries": [{"keys": ["x"], "content": "y"}]}
    )
    assert wb.generation > before


def test_generation_stable_on_reads_and_failed_writes(wb_db):
    wb = WorldBookManager(wb_db)
    wb_db.add_conversation({"id": "gen-ro", "title": "C"})
    book_id = wb.create_world_book("R")
    wb.create_world_book_entry(book_id, keys=["k"], content="c")
    g = wb.generation

    wb.get_world_book(book_id)
    wb.get_world_book_by_name("R")
    wb.list_world_books(include_disabled=True)
    wb.get_world_book_entries(book_id, enabled_only=False)
    wb.get_world_books_for_conversation("gen-ro", enabled_only=False)
    wb.get_conversations_for_world_book(book_id)
    wb.export_world_book(book_id)
    assert wb.generation == g  # reads never bump

    assert wb.update_world_book(999999, name="nope") is False  # no rows
    assert wb.update_world_book(book_id) is True  # no-op update, no write
    assert wb.update_world_book_entry(999999, content="x") is False
    assert wb.delete_world_book_entry(999999) is False
    assert wb.disassociate_world_book_from_conversation("gen-ro", book_id) is False
    with pytest.raises(InputError):
        wb.attach_world_book_to_character(book_id, 999999)  # unknown character
    assert wb.generation == g  # failed writes never bump


# ---------------------------------------------------------------------------
# 3b — resolver cache
# ---------------------------------------------------------------------------


class _Spy:
    def __init__(self):
        self.count = 0


def _spy_manager_fetch(monkeypatch):
    spy = _Spy()
    orig = WorldBookManager.get_world_books_for_conversation

    def wrapper(self, *a, **k):
        spy.count += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(WorldBookManager, "get_world_books_for_conversation", wrapper)
    return spy


def _spy_processor_init(monkeypatch):
    spy = _Spy()
    orig = WorldInfoProcessor.__init__

    def wrapper(self, *a, **k):
        spy.count += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(WorldInfoProcessor, "__init__", wrapper)
    return spy


def test_second_resolve_with_no_edits_fetches_and_builds_once(wb_db, monkeypatch):
    _attach(wb_db, "cache-1", "dragon", "Dragons breathe fire.")
    fetch = _spy_manager_fetch(monkeypatch)
    builds = _spy_processor_init(monkeypatch)

    r1 = resolve_world_info_injection(wb_db, "cache-1", None, "a dragon appears", [])
    r2 = resolve_world_info_injection(wb_db, "cache-1", None, "a dragon appears", [])
    r3 = resolve_world_info_injection(wb_db, "cache-1", None, "griffin flies", [])

    assert fetch.count == 1  # one query total across three sends
    assert builds.count == 1  # one processor build total
    assert r1 == r2 and "Dragons breathe fire." in r1[0]
    assert r3 == ("griffin flies", 0)  # cached processor still matches correctly


def test_book_edit_bumps_generation_and_refetches(wb_db, monkeypatch):
    wb, book_id = _attach(wb_db, "cache-2", "dragon", "old lore")
    entry_id = wb.get_world_book_entries(book_id)[0]["id"]
    fetch = _spy_manager_fetch(monkeypatch)

    resolve_world_info_injection(wb_db, "cache-2", None, "a dragon", [])
    assert fetch.count == 1
    wb.update_world_book_entry(entry_id, content="new lore")  # generation bump
    text, count = resolve_world_info_injection(wb_db, "cache-2", None, "a dragon", [])
    assert fetch.count == 2  # invalidated → refetched
    assert "new lore" in text and count == 1


def test_cache_entry_association_change_invalidates(wb_db, monkeypatch):
    wb, book_id = _attach(wb_db, "cache-3", "dragon", "Dragons breathe fire.")
    fetch = _spy_manager_fetch(monkeypatch)
    assert resolve_world_info_injection(wb_db, "cache-3", None, "a dragon", [])[1] == 1
    assert fetch.count == 1
    wb.disassociate_world_book_from_conversation("cache-3", book_id)  # bump
    assert resolve_world_info_injection(wb_db, "cache-3", None, "a dragon", []) == (
        "a dragon",
        0,
    )
    assert fetch.count == 2


def test_cache_is_keyed_by_character_slot(wb_db, monkeypatch):
    _attach(wb_db, "cache-4", "dragon", "Dragons breathe fire.")
    fetch = _spy_manager_fetch(monkeypatch)
    char_a = {"id": 7, "name": "A", "extensions": {}}

    resolve_world_info_injection(wb_db, "cache-4", None, "a dragon", [])
    resolve_world_info_injection(wb_db, "cache-4", char_a, "a dragon", [])
    assert fetch.count == 2  # (conv, None) and (conv, 7) are separate slots
    resolve_world_info_injection(wb_db, "cache-4", char_a, "a dragon", [])
    assert fetch.count == 2  # second hit on the character slot


def test_cache_lru_evicts_beyond_8_conversations(wb_db, monkeypatch):
    wb = WorldBookManager(wb_db)
    for i in range(1, 10):
        conv = f"lru-{i}"
        wb_db.add_conversation({"id": conv, "title": "C"})
        book_id = wb.create_world_book(f"Book{i}")
        wb.create_world_book_entry(book_id, keys=["key"], content=f"lore-{i}")
        wb.associate_world_book_with_conversation(conv, book_id)
    fetch = _spy_manager_fetch(monkeypatch)

    for i in range(1, 10):  # fill: 9 misses
        text, count = resolve_world_info_injection(
            wb_db, f"lru-{i}", None, "a key appears", []
        )
        assert f"lore-{i}" in text and count == 1
    assert fetch.count == 9
    assert len(resolver_module._processor_cache) == 8  # bounded

    # Reverse order: conversations 9..2 are hot; conversation 1 was evicted.
    for i in range(9, 1, -1):
        _, count = resolve_world_info_injection(
            wb_db, f"lru-{i}", None, "a key appears", []
        )
        assert count == 1
    assert fetch.count == 9  # all hits so far
    text, count = resolve_world_info_injection(
        wb_db, "lru-1", None, "a key appears", []
    )
    assert fetch.count == 10  # exactly one rebuild after eviction
    assert "lore-1" in text and count == 1


def test_cache_never_serves_one_db_instance_to_another(wb_db, tmp_path, monkeypatch):
    """The generation cell lives on the db object: a second connection must
    never inherit the first connection's cached processor (its own generation
    counter starts at 0 and cannot see the first connection's bumps)."""
    _attach(wb_db, "cache-5", "dragon", "Dragons breathe fire.")
    assert resolve_world_info_injection(wb_db, "cache-5", None, "a dragon", [])[1] == 1

    db2 = CharactersRAGDB(tmp_path / "wi_cache.db", "test-client-2")
    try:
        wb2 = WorldBookManager(db2)
        entry_id = wb2.get_world_book_entries(wb2.list_world_books()[0]["id"])[0]["id"]
        wb2.delete_world_book_entry(entry_id)  # bumps db2's own counter
        # db1's cached entry has the same generation number but belongs to a
        # different db object: the identity check must force a rebuild.
        assert resolve_world_info_injection(db2, "cache-5", None, "a dragon", []) == (
            "a dragon",
            0,
        )
    finally:
        db2.close_connection()


def test_cached_processor_output_matches_fresh_build(wb_db):
    _attach(wb_db, "cache-6", "dragon", "Dragons breathe fire.")
    cached = resolve_world_info_injection(wb_db, "cache-6", None, "a dragon", [])
    resolver_module._clear_world_info_cache()
    fresh = resolve_world_info_injection(wb_db, "cache-6", None, "a dragon", [])
    assert cached == fresh


# ---------------------------------------------------------------------------
# 3c — precompiled keys
# ---------------------------------------------------------------------------


def test_processed_entries_carry_compiled_patterns():
    proc = WorldInfoProcessor(
        world_books=[
            {
                "name": "P",
                "enabled": True,
                "entries": [
                    {
                        "id": 1,
                        "keys": ["Dragon", "Wyrm"],
                        "secondary_keys": ["Fire"],
                        "selective": True,
                        "content": "c",
                        "enabled": True,
                    },
                    {
                        "id": 2,
                        "keys": ["HTML"],
                        "content": "c",
                        "enabled": True,
                        "case_sensitive": True,
                    },
                    {
                        "id": 3,
                        "keys": ["gr(a|e)y"],
                        "content": "c",
                        "enabled": True,
                        "regex": True,
                    },
                ],
            }
        ]
    )
    ci = proc.entries[0]
    assert len(ci["compiled_primary_keys"]) == 2
    assert all(isinstance(p, re.Pattern) for p in ci["compiled_primary_keys"])
    assert all(isinstance(p, re.Pattern) for p in ci["compiled_secondary_keys"])
    # Case-insensitive entries match lowered text with the lowered-key pattern.
    assert ci["compiled_primary_keys"][0].search("a dragon flew")
    assert not ci["compiled_primary_keys"][0].search("dragons fly")  # word boundary

    cs = proc.entries[1]
    assert cs["compiled_primary_keys"][0].search("learning HTML now")
    assert not cs["compiled_primary_keys"][0].search("learning html now")

    rx = proc.entries[2]
    assert isinstance(rx["compiled_primary_keys"][0], re.Pattern)
    assert rx["compiled_primary_keys"][0].search("the grAy sky")  # flag baked in


def test_keyword_in_text_accepts_precompiled_pattern_or_raw_string():
    proc = WorldInfoProcessor()
    pattern = wip_module._compiled_keyword("dragon")
    assert proc._keyword_in_text(pattern, "a dragon flew")
    assert not proc._keyword_in_text(pattern, "dragons flew")
    # Ad-hoc raw strings keep working via the module-level fallback.
    assert proc._keyword_in_text("wyrm", "a WYRM flew".lower())


def test_compiled_keyword_lru_cache_is_stable():
    a = wip_module._compiled_keyword("castle")
    b = wip_module._compiled_keyword("castle")
    assert a is b and isinstance(a, re.Pattern)


def test_matching_pass_performs_zero_compiles_with_stored_patterns(monkeypatch):
    """A full match pass over stored patterns must not compile anything."""
    book = {
        "name": "M",
        "enabled": True,
        "entries": [
            {
                "id": i,
                "keys": [f"kw{i}", f"alt{i}"],
                "secondary_keys": [f"sec{i}"],
                "selective": True,
                "content": f"content {i}",
                "enabled": True,
            }
            for i in range(50)
        ],
    }
    proc = WorldInfoProcessor(world_books=[book])
    compiles = []
    orig = re.compile

    def counting_compile(*a, **k):
        compiles.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(wip_module.re, "compile", counting_compile)
    proc.process_messages("kw7 plus sec7 in the text", [])
    assert compiles == []


def test_second_processor_build_with_same_keys_compiles_nothing(monkeypatch):
    entries = [
        {"id": i, "keys": [f"zz{i}"], "content": f"c{i}", "enabled": True}
        for i in range(20)
    ]
    WorldInfoProcessor(
        world_books=[{"name": "Z", "enabled": True, "entries": entries}]
    )  # warms the module-level key cache

    compiles = []
    orig = re.compile

    def counting_compile(*a, **k):
        compiles.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(wip_module.re, "compile", counting_compile)
    WorldInfoProcessor(world_books=[{"name": "Z", "enabled": True, "entries": entries}])
    assert compiles == []


def test_regex_search_accepts_a_precompiled_pattern():
    pattern = re.compile("gr(a|e)y", re.IGNORECASE)
    assert regex_search(pattern, "the grAy sky", ignore_case=True) is True
    assert regex_search(pattern, "no match here", ignore_case=True) is False
    # Raw-string behavior unchanged.
    assert regex_search("gr(a|e)y", "the gray sky", ignore_case=False) is True


def test_invalid_regex_still_downgraded_to_literal():
    proc = WorldInfoProcessor(
        world_books=[
            {
                "name": "Bad",
                "enabled": True,
                "entries": [
                    {
                        "id": 1,
                        "keys": ["(a+)+"],
                        "content": "catastrophic",
                        "enabled": True,
                        "regex": True,
                    }
                ],
            }
        ]
    )
    entry = proc.entries[0]
    assert entry["regex"] is False  # fail-closed downgrade, unchanged
    # The stored pattern is then the escaped LITERAL word-boundary pattern
    # (a punctuation-only keyword can never self-match under word boundaries —
    # same semantics as the pre-change string path).
    assert entry["compiled_primary_keys"][0].pattern == r"\b\(a\+\)\+\b"
    assert entry["compiled_primary_keys"][0] is wip_module._compiled_keyword("(a+)+")


# ---------------------------------------------------------------------------
# 3d — one _process_entry per entry; id-set recursion dedup
# ---------------------------------------------------------------------------


def test_process_entry_runs_exactly_once_per_entry_per_build(monkeypatch):
    book = {
        "name": "S",
        "enabled": True,
        "entries": [
            {"id": 1, "keys": ["a"], "content": "c1", "enabled": True},
            {"id": 2, "keys": ["b"], "content": "c2", "enabled": False},  # disabled
            {"id": 3, "keys": [], "content": "c3", "enabled": True},  # keyless
        ],
    }
    calls = []
    orig = WorldInfoProcessor._process_entry

    def counting(self, entry):
        calls.append(entry.get("id"))
        return orig(self, entry)

    monkeypatch.setattr(WorldInfoProcessor, "_process_entry", counting)
    proc = WorldInfoProcessor(world_books=[book])
    # Pre-change this book cost 2 enabled-list calls (ids 1, 3) + 3 candidate
    # calls (all ids) = 5; the contract is now exactly one call per raw entry,
    # in entry order.
    assert len(calls) == len(book["entries"])
    assert len(proc.entries) == 1  # only enabled keyed entries activate


def test_candidates_keep_metadata_after_single_process_refactor():
    """The diagnostics candidate list must still carry source metadata for
    every entry (enabled or not), with the priority offset applied."""
    book = {
        "name": "CandBook",
        "id": 77,
        "enabled": True,
        "priority": 2,  # priority_offset 2000
        "entries": [
            {
                "id": 5,
                "keys": ["a"],
                "content": "on",
                "enabled": True,
                "insertion_order": 10,
            },
            {
                "id": 6,
                "keys": ["b"],
                "content": "off",
                "enabled": False,
                "insertion_order": 20,
            },
        ],
    }
    proc = WorldInfoProcessor(world_books=[book])
    assert len(proc._candidate_entries) == 2
    by_id = {c["_entry_id"]: c for c in proc._candidate_entries}
    assert by_id[5]["_book_id"] == 77 and by_id[5]["_book_name"] == "CandBook"
    assert by_id[5]["_enabled"] is True and by_id[6]["_enabled"] is False
    assert by_id[5]["insertion_order"] == 10 + 2000
    assert by_id[6]["insertion_order"] == 20 + 2000
    # Active entries got the same offset.
    assert proc.entries[0]["insertion_order"] == 10 + 2000


# ---------------------------------------------------------------------------
# 3e interface — pre-collected books (console wiring deferred)
# ---------------------------------------------------------------------------


def test_resolve_accepts_precaptured_books_without_fetching(wb_db, monkeypatch):
    _attach(wb_db, "books-1", "dragon", "Dragons breathe fire.")
    precaptured = WorldBookManager(wb_db).get_world_books_for_conversation("books-1")
    fetch = _spy_manager_fetch(monkeypatch)  # spy AFTER the pre-collection

    text, count = resolve_world_info_injection(
        wb_db, "books-1", None, "a dragon appears", [], books=precaptured
    )
    assert fetch.count == 0  # no fetch when books are handed in
    assert "Dragons breathe fire." in text and count == 1

    # The books= path bypasses the cache: a later self-fetch resolve still
    # queries (the cache was not populated by the pre-collected call).
    text2, _ = resolve_world_info_injection(
        wb_db, "books-1", None, "a dragon appears", []
    )
    assert fetch.count == 1 and text2 == text


def test_resolve_with_empty_precaptured_books_returns_unchanged(wb_db, monkeypatch):
    _attach(wb_db, "books-2", "dragon", "Dragons breathe fire.")
    fetch = _spy_manager_fetch(monkeypatch)
    assert resolve_world_info_injection(
        wb_db, "books-2", None, "a dragon appears", [], books=[]
    ) == ("a dragon appears", 0)
    assert fetch.count == 0


def test_apply_wrapper_accepts_books(wb_db, monkeypatch):
    _attach(wb_db, "books-3", "dragon", "Dragons breathe fire.")
    precaptured = WorldBookManager(wb_db).get_world_books_for_conversation("books-3")
    fetch = _spy_manager_fetch(monkeypatch)  # spy AFTER the pre-collection
    text = apply_world_info_to_message(
        wb_db, "books-3", None, "a dragon appears", [], books=precaptured
    )
    assert fetch.count == 0 and "Dragons breathe fire." in text


# ---------------------------------------------------------------------------
# Acceptance spies — the 1000-entry book across two sends
# ---------------------------------------------------------------------------


def _build_1000_entry_book(db, conv_id):
    db.add_conversation({"id": conv_id, "title": "C"})
    wb = WorldBookManager(db)
    book_id = wb.create_world_book("Big")
    for i in range(1000):
        wb.create_world_book_entry(book_id, keys=[f"kw-{i:04d}"], content=f"lore {i}")
    wb.associate_world_book_with_conversation(conv_id, book_id)
    return wb, book_id


def test_second_send_on_1000_entry_book_zero_queries_zero_compiles(wb_db, monkeypatch):
    _build_1000_entry_book(wb_db, "big-1")
    fetch = _spy_manager_fetch(monkeypatch)

    compiles = []
    orig_compile = re.compile

    def counting_compile(*a, **k):
        compiles.append(1)
        return orig_compile(*a, **k)

    monkeypatch.setattr(wip_module.re, "compile", counting_compile)
    text1, count1 = resolve_world_info_injection(
        wb_db, "big-1", None, "nothing matches this message", []
    )
    first_fetches, first_compiles = fetch.count, len(compiles)
    assert first_fetches == 1
    assert first_compiles == 1000  # one per key, once

    compiles.clear()
    fetch.count = 0
    text2, count2 = resolve_world_info_injection(
        wb_db, "big-1", None, "nothing matches this message", []
    )
    assert fetch.count == 0  # zero book queries on the second send
    assert compiles == []  # zero re.compile calls on the second send
    assert (text1, count1) == (text2, count2) == ("nothing matches this message", 0)
