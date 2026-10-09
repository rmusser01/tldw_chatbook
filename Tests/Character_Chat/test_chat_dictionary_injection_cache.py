"""Task 4 (TASK-34416 / ADR-221): chat-dictionary injection cold-start cache.

Three layers of evidence, mirroring the world-info cache tests:

1. GOLDEN PINS — replacement output captured from the pre-change pipeline on a
   fixture covering literal keys, ``/regex/`` keys (with and without ``/i``),
   ``max_replacements > 1`` (including the self-cascading replacement trap),
   whole-word boundaries (a key embedded in a larger word must NOT match),
   case-sensitivity variants, and multiple dictionaries active at once
   (conversation + character-embedded, same-key collisions across dicts).
   These pin byte-identical behavior across the caching/precompilation
   refactor.
2. NEW-BEHAVIOR TESTS (4a-4d) — store generation counters on both dictionary
   services, the resolved-bundle cache, lazy per-entry compiled keys, the
   pre-collected-entries seam.
3. ACCEPTANCE SPIES — 3 dictionaries totaling 300 entries across two sends:
   the second send must perform zero dictionary DB loads, zero entry JSON
   parses, zero ``ChatDictionary.from_dict`` calls and zero ``re.compile``
   calls.
"""

import asyncio
import re
from datetime import datetime, timedelta

import pytest

import tldw_chatbook.Character_Chat.Chat_Dictionary_Lib as cdl
from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
    ChatDictionary,
    _resolve_active_dictionaries,
    apply_active_chatdicts_to_text,
    apply_replacement_once,
    collect_active_chatdict_entries,
    match_whole_words,
    process_user_input_with_diagnostics,
)
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
)
from tldw_chatbook.Character_Chat.server_chat_dictionary_service import (
    ServerChatDictionaryService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError


@pytest.fixture
def dict_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "dict_cache.db", "test-client")
    yield db
    db.close_connection()


@pytest.fixture(autouse=True)
def _isolated_caches():
    """The bundle cache and compiled-key caches are module-level; keep every
    test hermetic."""
    cdl._clear_dictionary_bundle_cache()
    cdl._compiled_whole_word.cache_clear()
    yield
    cdl._clear_dictionary_bundle_cache()
    cdl._compiled_whole_word.cache_clear()


def _attach(db, conv_id, key, content, name="Slang"):
    """Conversation-attach one single-entry dictionary via the service."""
    dict_id = cdl.save_chat_dictionary(
        db, name, entries=[cdl.ChatDictionary(key=key, content=content)]
    )
    LocalChatDictionaryService(db).attach_to_conversation(dict_id, conv_id)
    return dict_id


# ---------------------------------------------------------------------------
# Golden replacement pins (captured from the pre-change implementation)
# ---------------------------------------------------------------------------


def _golden_fixture(db):
    """3 dictionaries (2 conversation + 1 character-embedded) exercising
    literal keys, regex keys, max_replacements, whole-word boundaries, case
    variants and cross-dictionary same-key collisions."""
    conv_id = db.add_conversation({"title": "Golden"})
    service = LocalChatDictionaryService(db)

    med_id = cdl.save_chat_dictionary(
        db,
        "Med",
        entries=[
            cdl.ChatDictionary(key="BP", content="blood pressure"),
            cdl.ChatDictionary(key="Doctor", content="physician", case_sensitive=True),
            cdl.ChatDictionary(key="/spo2|o2sat/i", content="oxygen saturation"),
            cdl.ChatDictionary(key="ward", content="the Ward", max_replacements=3),
            cdl.ChatDictionary(key="q4h", content="every 4 hours"),
        ],
    )
    slang_id = cdl.save_chat_dictionary(
        db,
        "Slang",
        entries=[
            cdl.ChatDictionary(key="chill", content="relax"),
            cdl.ChatDictionary(key="/gr(a|e)y/", content="grey"),
            cdl.ChatDictionary(key="BP", content="SLANG-BP"),
        ],
    )
    char_id = db.add_character_card({"name": "Noir"})
    rec = db.get_character_card_by_id(char_id)
    ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    ext.setdefault("chat_dictionaries", []).append(
        {
            "name": "CharDict",
            "enabled": True,
            "entries": [
                {"key": "Noir", "content": "Detective Noir"},
                {"key": "/[0-9]+mg/i", "content": "a dose"},
            ],
        }
    )
    db.update_character_card(
        char_id, {"extensions": ext}, expected_version=rec["version"]
    )
    char_data = db.get_character_card_by_id(char_id)

    service.attach_to_conversation(med_id, conv_id)
    service.attach_to_conversation(slang_id, conv_id)
    return conv_id, char_data


GOLDEN_MESSAGES = [
    # whole-word + case-insensitive literal; xBPx must stay; Med's BP takes
    # the first occurrence, Slang's BP the second (max_replacements=1 each)
    "The BP was high, the bp note agrees, and xBPx stays put.",
    # case-sensitive literal: only exact "Doctor"
    "doctor DOCTOR Doctor reviewed it.",
    # regex /i: SpO2 and O2SAT both match, max_rep 1 -> first only
    "SpO2 and O2SAT were checked.",
    # max_replacements=3 on 5 occurrences — replacement output re-matches the
    # same case-insensitive pattern (self-cascade), exactly 3 passes
    "ward ward ward ward ward",
    # case-insensitive literal on uppercase
    "Give it Q4H.",
    # case-SENSITIVE regex: Gray stays, gray/grey replaced
    "Gray, gray, and grey scales.",
    # char-embedded literal (CI): noir + Noir, max_rep 1
    "the noir met Noir at noon.",
    # char-embedded regex /i, max_rep 1
    "100mg then 50MG later.",
    # multi-dictionary at once
    "BP ward chill Gray Noir 100mg q4h Doctor SpO2 xBPx",
    # nothing matches
    "Nothing to replace here.",
]

# Captured pre-change via apply_active_chatdicts_to_text (full send path).
GOLDEN_SEND_PATH = [
    "The blood pressure was high, the SLANG-BP note agrees, and xBPx stays put.",
    "doctor DOCTOR physician reviewed it.",
    "oxygen saturation and O2SAT were checked.",
    "the the the Ward ward ward ward ward",
    "Give it every 4 hours.",
    "Gray, grey, and grey scales.",
    "the Detective Noir met Noir at noon.",
    "a dose then 50MG later.",
    "blood pressure the the the Ward relax Gray Detective Noir a dose every 4 hours physician oxygen saturation xBPx",
    "Nothing to replace here.",
]

GOLDEN_COLLECTED_KEYS = [
    "BP",
    "Doctor",
    "/spo2|o2sat/i",
    "ward",
    "q4h",
    "chill",
    "/gr(a|e)y/",
    "BP",
    "Noir",
    "/[0-9]+mg/i",
]

GOLDEN_DIAGNOSTICS_M08 = {
    "matched": 9,
    "fired": 8,
    "skipped": 1,
    "total_replacements": 10,
    "tokens_used": 16,
    "budget_exceeded": False,
    "statuses": [
        ("BP", "fired", 1, "blood pressure"),
        ("Doctor", "fired", 1, "physician"),
        ("/spo2|o2sat/i", "fired", 1, "oxygen saturation"),
        ("ward", "fired", 3, "the Ward"),
        ("q4h", "fired", 1, "every 4 hours"),
        ("chill", "fired", 1, "relax"),
        ("BP", "no_replacement", 0, "SLANG-BP"),
        ("Noir", "fired", 1, "Detective Noir"),
        ("/[0-9]+mg/i", "fired", 1, "a dose"),
    ],
}


def test_golden_send_path_byte_identical(dict_db):
    conv_id, char_data = _golden_fixture(dict_db)
    for message, expected in zip(GOLDEN_MESSAGES, GOLDEN_SEND_PATH):
        out = apply_active_chatdicts_to_text(
            dict_db, conv_id, char_data, message, max_tokens=5000
        )
        assert out == expected, f"message={message!r}"


def test_golden_pipeline_and_union_order_byte_identical(dict_db):
    conv_id, char_data = _golden_fixture(dict_db)
    flat = collect_active_chatdict_entries(dict_db, conv_id, char_data)
    assert [e.raw_key for e in flat] == GOLDEN_COLLECTED_KEYS
    for message, expected in zip(GOLDEN_MESSAGES, GOLDEN_SEND_PATH):
        assert (
            process_user_input_with_diagnostics(message, flat, max_tokens=5000)[0]
            == expected
        )


def test_golden_diagnostics_stage_accounting_byte_identical(dict_db):
    conv_id, char_data = _golden_fixture(dict_db)
    flat = collect_active_chatdict_entries(dict_db, conv_id, char_data)
    _, diag = process_user_input_with_diagnostics(
        GOLDEN_MESSAGES[8], flat, max_tokens=5000
    )
    payload = diag.to_dict()
    summary = {
        k: payload[k]
        for k in (
            "matched",
            "fired",
            "skipped",
            "total_replacements",
            "tokens_used",
            "budget_exceeded",
        )
    }
    assert summary == {
        k: v for k, v in GOLDEN_DIAGNOSTICS_M08.items() if k != "statuses"
    }
    assert [
        (r["pattern"], r["status"], r["replacements"], r["content_preview"])
        for r in payload["entries"]
    ] == GOLDEN_DIAGNOSTICS_M08["statuses"]


def test_golden_whole_word_boundary_and_case_pins(dict_db):
    """Byte-level behavior pins for the whole-word / case-sensitivity axes:
    a literal key embedded in a larger word must NOT match; case variants
    respect entry.case_sensitive for literals and the pattern's own flags for
    regex keys."""
    ci = ChatDictionary(key="BP", content="blood pressure")
    assert match_whole_words([ci], "xBPx stays") == []
    assert match_whole_words([ci], "check bp now") == [ci]  # CI by default
    cs = ChatDictionary(key="Doctor", content="physician", case_sensitive=True)
    assert match_whole_words([cs], "the doctor") == []
    assert match_whole_words([cs], "the Doctor") == [cs]
    rx_i = ChatDictionary(key="/spo2|o2sat/i", content="x")
    assert match_whole_words([rx_i], "SpO2 low") == [rx_i]
    rx_cs = ChatDictionary(key="/gr(a|e)y/", content="x")
    assert match_whole_words([rx_cs], "Gray sky") == []
    assert match_whole_words([rx_cs], "grey sky") == [rx_cs]


# ---------------------------------------------------------------------------
# 4a — store generation counters
# ---------------------------------------------------------------------------


def test_generation_starts_at_zero_and_is_shared_across_service_instances(dict_db):
    assert LocalChatDictionaryService(dict_db).generation == 0
    service = LocalChatDictionaryService(dict_db)
    service.create_dictionary({"name": "D", "entries": []})
    assert service.generation == 1
    # The send path never sees a service; a fresh service over the same db
    # (and the lib's own view) must observe the same shared counter.
    assert LocalChatDictionaryService(dict_db).generation == 1
    assert cdl.dictionary_store_generation(dict_db) == 1


def test_generation_bumps_on_every_write_path(dict_db):
    service = LocalChatDictionaryService(dict_db)
    conv_id = dict_db.add_conversation({"title": "gen"})
    char_id = dict_db.add_character_card({"name": "GenChar"})

    dict_id = cdl.save_chat_dictionary(  # lib write choke point: create
        dict_db, "W", entries=[cdl.ChatDictionary(key="k", content="c")]
    )  # 1
    cdl.update_chat_dictionary(dict_db, dict_id, description="d")  # 2 (lib update)
    service.attach_to_conversation(dict_id, conv_id)  # 3 (attachment seam)
    service.attach_to_character(dict_id, char_id)  # 4 (embedded snapshot write)
    service.detach_from_character(char_id, "W")  # 5
    service.detach_from_conversation(dict_id, conv_id)  # 6
    service.attach_to_conversation(dict_id, conv_id)  # 7 (idempotent re-attach)
    service.add_entry(dict_id, {"pattern": "k2", "replacement": "c2"})  # 8
    entry_id = f"local:chat_dictionary_entry:{dict_id}:1"
    service.update_entry(entry_id, {"replacement": "c3"})  # 9
    service.delete_entry(entry_id)  # 10
    service.reorder_entries(dict_id, {"entry_ids": []})  # 11
    service.import_markdown({"name": "Imported", "content": "a: b\n"})  # 12
    service.import_json(
        {"name": "JsonImported", "entries": [{"pattern": "j", "replacement": "v"}]}
    )  # 13
    reverted = service.create_dictionary(
        {"name": "Rev", "entries": [{"pattern": "r", "replacement": "v"}]}
    )["id"]  # 14
    service.revert_version(reverted, 1)  # 15
    service.update_dictionary(dict_id, {"is_active": False})  # 16
    service.delete_dictionary(dict_id)  # 17
    assert service.generation == 17


def test_generation_stable_on_reads_and_failed_writes(dict_db):
    service = LocalChatDictionaryService(dict_db)
    conv_id = dict_db.add_conversation({"title": "ro"})
    dict_id = service.create_dictionary(
        {"name": "R", "entries": [{"pattern": "k", "replacement": "v"}]}
    )["id"]
    service.attach_to_conversation(dict_id, conv_id)
    g = service.generation

    # Reads never bump.
    service.list_dictionaries()
    service.get_dictionary(dict_id)
    service.list_entries(dict_id)
    service.process_text({"text": "k", "dictionary_id": dict_id})
    service.export_markdown(dict_id)
    service.export_json(dict_id)
    service.list_activity(dict_id)
    service.list_versions(dict_id)
    service.get_version(dict_id, 1)
    service.list_dictionary_conversations(dict_id)
    service.list_character_dictionaries(dict_db.add_character_card({"name": "C"}))
    service.summarize_active_dictionaries(conv_id, None)
    cdl.load_chat_dictionary(dict_db, dict_id)
    cdl.list_chat_dictionaries(dict_db)
    collect_active_chatdict_entries(dict_db, conv_id, None)
    assert service.generation == g

    # Failed / no-op writes never bump.
    assert cdl.update_chat_dictionary(dict_db, dict_id) is True  # no fields -> no write
    assert cdl.update_chat_dictionary(dict_db, 999999, description="x") is False
    assert cdl.delete_chat_dictionary(dict_db, 999999) is False
    with pytest.raises(ConflictError):
        cdl.save_chat_dictionary(dict_db, "R")  # duplicate name -> IntegrityError path
    assert service.generation == g


def test_server_service_generation_bumps_on_remote_mutations():
    class _FakeClient:
        async def create_chat_dictionary(self, request_data):
            return {"id": 1}

        async def update_chat_dictionary(self, dictionary_id, request_data):
            return {"id": dictionary_id}

        async def delete_chat_dictionary(self, dictionary_id, **kwargs):
            return {}

        async def add_chat_dictionary_entry(self, dictionary_id, request_data):
            return {"id": 12}

        async def update_chat_dictionary_entry(self, entry_id, request_data):
            return {"id": entry_id}

        async def delete_chat_dictionary_entry(self, entry_id):
            return {}

        async def bulk_chat_dictionary_entry_operations(self, request_data):
            return {}

        async def reorder_chat_dictionary_entries(self, dictionary_id, request_data):
            return {}

        async def import_chat_dictionary_markdown(self, request_data):
            return {"dictionary_id": 2}

        async def import_chat_dictionary_json(self, request_data):
            return {"dictionary_id": 3}

        async def revert_chat_dictionary_version(self, dictionary_id, revision):
            return {}

        async def list_chat_dictionaries(self, **kwargs):
            return {"dictionaries": []}

        async def get_chat_dictionary(self, dictionary_id):
            return {"id": dictionary_id}

        async def export_chat_dictionary_json(self, dictionary_id):
            return {"data": {}}

    service = ServerChatDictionaryService(client=_FakeClient())
    assert service.generation == 0
    asyncio.run(service.create_dictionary({"name": "X"}))  # 1
    asyncio.run(service.update_dictionary(1, {"name": "Y"}))  # 2
    asyncio.run(service.add_entry(1, {"pattern": "p"}))  # 3
    asyncio.run(service.update_entry(1, {"replacement": "r"}))  # 4
    asyncio.run(service.delete_entry(1))  # 5
    asyncio.run(service.bulk_entries({"operations": []}))  # 6
    asyncio.run(service.reorder_entries(1, {"entry_ids": []}))  # 7
    asyncio.run(service.import_markdown({"name": "M"}))  # 8
    asyncio.run(service.import_json({"name": "J"}))  # 9
    asyncio.run(service.revert_version(1, 1))  # 10
    asyncio.run(service.delete_dictionary(1))  # 11
    assert service.generation == 11

    g = service.generation
    asyncio.run(service.list_dictionaries())
    asyncio.run(service.get_dictionary(1))
    asyncio.run(service.export_json(1))
    assert service.generation == g  # remote reads never bump


# ---------------------------------------------------------------------------
# 4b — resolved-bundle cache
# ---------------------------------------------------------------------------


class _Spy:
    def __init__(self):
        self.count = 0


def _spy_load(monkeypatch):
    spy = _Spy()
    orig = cdl.load_chat_dictionary

    def wrapper(*a, **k):
        spy.count += 1
        return orig(*a, **k)

    monkeypatch.setattr(cdl, "load_chat_dictionary", wrapper)
    return spy


def _spy_from_dict(monkeypatch):
    spy = _Spy()
    orig = ChatDictionary.from_dict

    def wrapper(*a, **k):
        spy.count += 1
        return orig(*a, **k)

    monkeypatch.setattr(ChatDictionary, "from_dict", wrapper)
    return spy


def _embed_char_dict(db, char_id, name, entries=None):
    rec = db.get_character_card_by_id(char_id)
    ext = rec["extensions"] if isinstance(rec["extensions"], dict) else {}
    ext.setdefault("chat_dictionaries", []).append(
        {
            "name": name,
            "enabled": True,
            "entries": entries or [{"key": name, "content": name.lower()}],
        }
    )
    db.update_character_card(
        char_id, {"extensions": ext}, expected_version=rec["version"]
    )


def test_second_resolve_with_no_edits_loads_and_instantiates_once(dict_db, monkeypatch):
    conv_id = dict_db.add_conversation({"title": "cache-1"})
    _attach(dict_db, conv_id, "dragon", "grim", name="Dict1")
    char_id = dict_db.add_character_card({"name": "Noir"})
    _embed_char_dict(dict_db, char_id, "CharDict")
    char_data = dict_db.get_character_card_by_id(char_id)
    load = _spy_load(monkeypatch)
    fd = _spy_from_dict(monkeypatch)

    r1 = _resolve_active_dictionaries(dict_db, conv_id, char_data)
    r2 = _resolve_active_dictionaries(dict_db, conv_id, char_data)
    r3 = _resolve_active_dictionaries(dict_db, conv_id, char_data)

    assert load.count == 1  # one dictionary DB load total across three resolves
    assert fd.count == 2  # conv entry + char entry, built once
    # Ready to use on hit: the SAME ChatDictionary instances, no rebuilds.
    assert r1 == r2 == r3
    assert r2[0]["entries"][0] is r1[0]["entries"][0]
    assert r3[1]["entries"][0] is r1[1]["entries"][0]
    # The send-path collect shares the same bundle.
    e1 = collect_active_chatdict_entries(dict_db, conv_id, char_data)
    e2 = collect_active_chatdict_entries(dict_db, conv_id, char_data)
    assert e1[0] is e2[0] and load.count == 1


def test_dictionary_edit_bumps_generation_and_refetches(dict_db, monkeypatch):
    conv_id = dict_db.add_conversation({"title": "cache-2"})
    dict_id = _attach(dict_db, conv_id, "dragon", "old lore", name="Dict2")
    load = _spy_load(monkeypatch)

    r1 = _resolve_active_dictionaries(dict_db, conv_id, None)
    assert load.count == 1
    cdl.update_chat_dictionary(
        dict_db,
        dict_id,
        entries=[cdl.ChatDictionary(key="dragon", content="new lore")],
    )  # generation bump
    r2 = _resolve_active_dictionaries(dict_db, conv_id, None)
    assert load.count == 2  # invalidated -> refetched
    assert r2[0]["entries"][0].content == "new lore"
    # The bump rebuilds INSTANCES, not just rows: the cached entry object
    # (with its compiled pattern) must never be reused across a generation.
    assert r2[0]["entries"][0] is not r1[0]["entries"][0]


def test_attachment_change_invalidates_cached_bundle(dict_db, monkeypatch):
    conv_id = dict_db.add_conversation({"title": "cache-3"})
    dict_id = _attach(dict_db, conv_id, "dragon", "Dragons breathe fire.")
    load = _spy_load(monkeypatch)
    assert len(_resolve_active_dictionaries(dict_db, conv_id, None)) == 1
    assert load.count == 1

    LocalChatDictionaryService(dict_db).detach_from_conversation(dict_id, conv_id)
    # Attachment-seam bump: a stale cache hit would still serve the detached
    # dictionary's row -- an empty bundle proves the cache was invalidated
    # (the rebuild itself needs no dictionary loads; the active list is empty).
    assert _resolve_active_dictionaries(dict_db, conv_id, None) == []


def test_character_embedded_change_invalidates_cached_bundle(dict_db, monkeypatch):
    conv_id = dict_db.add_conversation({"title": "cache-4"})
    _attach(dict_db, conv_id, "unrelated", "x", name="Dict4")
    char_id = dict_db.add_character_card({"name": "Noir"})
    service = LocalChatDictionaryService(dict_db)
    dict_id = service.create_dictionary(
        {"name": "Embedded", "entries": [{"pattern": "k", "replacement": "v"}]}
    )["id"]
    char_data = dict_db.get_character_card_by_id(char_id)
    assert len(_resolve_active_dictionaries(dict_db, conv_id, char_data)) == 1

    service.attach_to_character(dict_id, char_id)  # embedded-snapshot write bump
    load = _spy_load(monkeypatch)  # spy AFTER setup (attach itself loads the dict)
    char_data = dict_db.get_character_card_by_id(char_id)
    rows = _resolve_active_dictionaries(dict_db, conv_id, char_data)
    assert [r["name"] for r in rows] == ["Dict4", "Embedded"]
    assert load.count == 1  # rebuild reloaded the conversation dict only


def test_cache_is_keyed_by_character_slot(dict_db, monkeypatch):
    conv_id = dict_db.add_conversation({"title": "cache-5"})
    _attach(dict_db, conv_id, "dragon", "grim", name="Dict5")
    load = _spy_load(monkeypatch)
    char_a = {"id": 7, "name": "A", "extensions": {}}

    _resolve_active_dictionaries(dict_db, conv_id, None)
    _resolve_active_dictionaries(dict_db, conv_id, char_a)
    assert load.count == 2  # (conv, None) and (conv, 7) are separate slots
    _resolve_active_dictionaries(dict_db, conv_id, char_a)
    assert load.count == 2  # second hit on the character slot


def test_cache_lru_evicts_beyond_8_conversations(dict_db, monkeypatch):
    service = LocalChatDictionaryService(dict_db)
    conv_ids = [dict_db.add_conversation({"title": f"lru-{i}"}) for i in range(1, 10)]
    for i in range(1, 10):
        dict_id = service.create_dictionary(
            {
                "name": f"Dict{i}",
                "entries": [{"pattern": f"kw{i}", "replacement": f"v{i}"}],
            }
        )["id"]
        service.attach_to_conversation(dict_id, conv_ids[i - 1])
    load = _spy_load(monkeypatch)

    for i in range(1, 10):  # fill: 9 misses
        rows = _resolve_active_dictionaries(dict_db, conv_ids[i - 1], None)
        assert len(rows) == 1
    assert load.count == 9
    assert len(cdl._dictionary_bundle_cache) == 8  # bounded

    # Reverse order: conversations 9..2 are hot; conversation 1 was evicted.
    for i in range(9, 1, -1):
        assert len(_resolve_active_dictionaries(dict_db, conv_ids[i - 1], None)) == 1
    assert load.count == 9  # all hits so far
    assert len(_resolve_active_dictionaries(dict_db, conv_ids[0], None)) == 1
    assert load.count == 10  # exactly one rebuild after eviction


def test_cache_never_serves_one_db_instance_to_another(dict_db, tmp_path, monkeypatch):
    """The generation cell lives on the db object: a second connection must
    never inherit the first connection's cached bundle (its own generation
    counter starts at 0 and cannot see the first connection's bumps)."""
    conv_id = dict_db.add_conversation({"title": "cache-6"})
    dict_id = _attach(dict_db, conv_id, "dragon", "grim", name="Dict6")
    _resolve_active_dictionaries(dict_db, conv_id, None)  # caches at generation 2
    cached_generation = cdl.dictionary_store_generation(dict_db)

    db2 = CharactersRAGDB(tmp_path / "dict_cache.db", "test-client-2")
    try:
        # Soft-delete the dictionary THROUGH db2 (same file): this both
        # changes db2's own view (the dict is gone) and bumps db2's counter.
        assert cdl.delete_chat_dictionary(db2, dict_id) is True
        # Raise db2's counter to the same NUMBER the first connection cached
        # at, so only the db-identity check can force the rebuild.
        while cdl.dictionary_store_generation(db2) < cached_generation:
            cdl._bump_generation(db2)
        # A generation-only match must NOT serve db1's stale bundle (which
        # still contains the dictionary).
        assert _resolve_active_dictionaries(db2, conv_id, None) == []
    finally:
        db2.close_connection()


def test_cached_bundle_rows_equal_fresh_resolve(dict_db):
    conv_id = dict_db.add_conversation({"title": "cache-7"})
    _attach(dict_db, conv_id, "dragon", "grim", name="Dict7")
    char_id = dict_db.add_character_card({"name": "N"})
    _embed_char_dict(dict_db, char_id, "CharDict7")
    char_data = dict_db.get_character_card_by_id(char_id)

    def _snapshot(rows):
        return [
            (
                r["name"],
                r["source"],
                r["enabled"],
                r["shadowed"],
                [e.to_dict() for e in r["entries"]],
            )
            for r in rows
        ]

    cached = _snapshot(_resolve_active_dictionaries(dict_db, conv_id, char_data))
    cdl._clear_dictionary_bundle_cache()
    fresh = _snapshot(_resolve_active_dictionaries(dict_db, conv_id, char_data))
    assert cached == fresh


def test_empty_bundle_is_not_cached(dict_db, monkeypatch):
    conv_id = dict_db.add_conversation({"title": "empty"})
    assert _resolve_active_dictionaries(dict_db, conv_id, None) == []
    assert len(cdl._dictionary_bundle_cache) == 0


def test_summarize_shares_the_cached_bundle(dict_db, monkeypatch):
    from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
        summarize_active_dictionaries,
    )

    conv_id = dict_db.add_conversation({"title": "sum"})
    _attach(dict_db, conv_id, "dragon", "grim", name="DictSum")
    load = _spy_load(monkeypatch)

    s1 = summarize_active_dictionaries(dict_db, conv_id, None)
    s2 = summarize_active_dictionaries(dict_db, conv_id, None)
    assert s1 == s2
    assert load.count == 1  # read model rides the send-path bundle cache


# ---------------------------------------------------------------------------
# 4c — compile once
# ---------------------------------------------------------------------------


def _counting_compile(monkeypatch):
    compiles = []
    orig = re.compile

    def counting_compile(*a, **k):
        compiles.append(a[0])
        return orig(*a, **k)

    monkeypatch.setattr(cdl.re, "compile", counting_compile)
    return compiles


def test_construction_compiles_nothing_but_still_classifies(monkeypatch):
    compiles = _counting_compile(monkeypatch)
    regex_entry = ChatDictionary(key="/sp(o|0)2/i", content="x")
    bad_entry = ChatDictionary(key="/(a+)+/", content="y")  # ReDoS screen rejects
    literal_entry = ChatDictionary(key="BP", content="z")
    plain_slash = ChatDictionary(key="/not-a-regex", content="w")  # no closing form
    assert compiles == []  # construction classifies only; compile+validation is lazy
    # The cheap slash-form classification is eager (internal state + debugging
    # fields), and compiling nothing:
    assert regex_entry._is_regex is True
    assert regex_entry.key_pattern_str == "sp(o|0)2"
    assert regex_entry.key_flags == re.IGNORECASE
    assert bad_entry._is_regex is True
    assert literal_entry._is_regex is False
    assert (
        plain_slash._is_regex is False and plain_slash.key_pattern_str == "/not-a-regex"
    )
    assert literal_entry.to_dict()["is_regex"] is False  # literal: still no compile
    assert compiles == []


def test_is_regex_reflects_post_validation_state_lazily(monkeypatch):
    """is_regex keeps its eager-build contract (False once a /regex/ key fails
    the ReDoS/syntax screen) while staying free until first read: reading it
    materializes the compiled key exactly once."""
    compiles = _counting_compile(monkeypatch)
    good = ChatDictionary(key="/colou?r/i", content="hue")
    bad = ChatDictionary(key="/(a+)+$/", content="x")
    assert compiles == []  # nothing compiled yet, even for the bad pattern
    assert bad.is_regex is False  # catastrophic pattern downgraded (lazy screen)
    assert compiles == ["(a+)+$"]  # only the validator's internal syntax probe
    assert good.is_regex is True
    assert compiles == ["(a+)+$", "colou?r", "colou?r"]  # probe + stored compile
    assert good.key.pattern == "colou?r"
    assert bad.key == "/(a+)+$/"  # downgraded to the literal raw key


def test_key_compiles_once_per_instance_and_is_stored_on_the_entry(monkeypatch):
    compiles = _counting_compile(monkeypatch)
    entry = ChatDictionary(key="/sp(o|0)2/i", content="x")
    k1 = entry.key
    k2 = entry.key
    k3 = entry.key
    assert isinstance(k1, re.Pattern)
    assert k1 is k2 is k3  # the compiled pattern lives on the entry instance
    # Exactly one validation (whose internal syntax probe compiles once) plus
    # one stored compile -- the same two re.compile calls the eager build
    # always paid; repeated access adds ZERO.
    assert compiles == ["sp(o|0)2", "sp(o|0)2"]


def test_literal_key_returns_raw_string_without_compiling(monkeypatch):
    compiles = _counting_compile(monkeypatch)
    entry = ChatDictionary(key="BP", content="blood pressure")
    assert entry.key == "BP"
    assert entry.key is entry.raw_key
    assert compiles == []


def test_invalid_regex_downgrade_semantics_unchanged(monkeypatch):
    compiles = _counting_compile(monkeypatch)
    entry = ChatDictionary(key="/(a+)+/", content="y")
    # Fail-closed downgrade: raw key as a literal, is_regex flipped off.
    assert entry.key == "/(a+)+/"
    assert entry.is_regex is False
    # The validator's internal syntax probe compiled once; the catastrophic
    # screen then rejected it before the stored compile.
    assert compiles == ["(a+)+"]
    # Under whole-word semantics a punctuation-leading fallback key can never
    # self-match (no \b between a space and "/") -- unchanged from pre-change.
    assert match_whole_words([entry], "has /(a+)+/ inside") == []


def test_compiled_whole_word_helper_is_lru_cached_and_flag_keyed():
    p1 = cdl._compiled_whole_word("bp", re.IGNORECASE)
    p2 = cdl._compiled_whole_word("bp", re.IGNORECASE)
    assert p1 is p2 and isinstance(p1, re.Pattern)
    assert p1.pattern == r"\bbp\b"
    p3 = cdl._compiled_whole_word("bp", 0)  # flag variants are distinct slots
    assert p3 is not p1


def test_match_pass_with_warm_helper_performs_zero_compiles(monkeypatch):
    entries = [ChatDictionary(key=f"kw{i:02d}", content=f"c{i}") for i in range(50)]
    assert match_whole_words(entries, "nothing matches here") == []  # warm
    compiles = _counting_compile(monkeypatch)
    assert match_whole_words(entries, "nothing matches here either") == []
    hit = match_whole_words(entries, "kw07 appears")
    assert [e.raw_key for e in hit] == ["kw07"]
    assert compiles == []


def test_replacement_loop_compiles_once_per_literal_key_not_per_replacement(
    monkeypatch,
):
    """apply_replacement_once must not recompile the whole-word pattern for
    every max_replacements iteration (previously one re.compile per call)."""
    # Content "X" never re-matches the key, so the loop is pure consumption.
    entry = ChatDictionary(key="ward", content="X", max_replacements=3)
    text = " ".join(["ward"] * 6)
    first_text, count = apply_replacement_once(text, entry)  # warms the helper
    assert count == 1

    compiles = _counting_compile(monkeypatch)
    current = first_text
    budget = entry.max_replacements - 1  # mirror the pipeline's bounded loop
    done = 0
    while budget > 0:
        current, replaced = apply_replacement_once(current, entry)
        if not replaced:
            break
        done += 1
        budget -= 1
    assert done == 2  # max_replacements=3 total (1 warm + 2 here), 3 wards left
    assert current.count("ward") == 3
    assert compiles == []  # zero re.compile inside the loop


def test_regex_entry_reuse_across_sends_never_recompiles(monkeypatch):
    entry = ChatDictionary(key="/sp(o|0)2/i", content="oxygen saturation")
    assert match_whole_words([entry], "spo2 low") == [entry]  # compile once
    compiles = _counting_compile(monkeypatch)
    assert match_whole_words([entry], "SPO2 low") == [entry]
    text, count = apply_replacement_once("check spo2 now", entry)
    assert count == 1 and text == "check oxygen saturation now"
    assert compiles == []


# ---------------------------------------------------------------------------
# 4d interface — pre-collected entries (console wiring deferred, as in T3)
# ---------------------------------------------------------------------------


def test_apply_accepts_precollected_entries_without_collecting(dict_db, monkeypatch):
    conv_id = dict_db.add_conversation({"title": "seam-1"})
    _attach(dict_db, conv_id, "Warden", "grim jailer", name="Seam")
    precaptured = collect_active_chatdict_entries(dict_db, conv_id, None)
    cdl._clear_dictionary_bundle_cache()  # drop what the pre-collection cached
    load = _spy_load(monkeypatch)  # spy AFTER the pre-collection

    collect_calls = []
    orig_collect = cdl.collect_active_chatdict_entries

    def counting_collect(*a, **k):
        collect_calls.append(1)
        return orig_collect(*a, **k)

    monkeypatch.setattr(cdl, "collect_active_chatdict_entries", counting_collect)

    out = apply_active_chatdicts_to_text(
        dict_db, conv_id, None, "The Warden nods.", entries=precaptured
    )
    assert out == "The grim jailer nods."
    assert collect_calls == []  # no collection when entries are handed in
    assert load.count == 0  # and no DB loads

    # The entries= path bypasses the cache (the caller owns collection): a
    # later self-collect still loads from the store.
    collect_active_chatdict_entries(dict_db, conv_id, None)
    assert load.count == 1


def test_apply_with_empty_precollected_entries_returns_unchanged(dict_db, monkeypatch):
    conv_id = dict_db.add_conversation({"title": "seam-2"})
    _attach(dict_db, conv_id, "Warden", "grim jailer", name="Seam2")
    load = _spy_load(monkeypatch)
    out = apply_active_chatdicts_to_text(
        dict_db, conv_id, None, "The Warden nods.", entries=[]
    )
    assert out == "The Warden nods."
    assert load.count == 0


def test_apply_entries_none_keeps_the_collecting_path(dict_db, monkeypatch):
    conv_id = dict_db.add_conversation({"title": "seam-3"})
    _attach(dict_db, conv_id, "Warden", "grim jailer", name="Seam3")
    load = _spy_load(monkeypatch)
    out = apply_active_chatdicts_to_text(dict_db, conv_id, None, "The Warden nods.")
    assert out == "The grim jailer nods."
    assert load.count == 1


# ---------------------------------------------------------------------------
# Acceptance spies — 3 dictionaries x 100 entries across two sends
# ---------------------------------------------------------------------------


def _spy_entries_json_loads(monkeypatch):
    """Count json.loads calls whose payload is an entries JSON array."""
    spy = _Spy()
    orig = cdl.json.loads

    def wrapper(s, *a, **k):
        if isinstance(s, str) and s[:1] == "[":
            spy.count += 1
        return orig(s, *a, **k)

    monkeypatch.setattr(cdl.json, "loads", wrapper)
    return spy


def _build_300_entry_store(db, conv_id):
    """3 conversation dictionaries x 100 entries (75 literal + 25 regex each)."""
    service = LocalChatDictionaryService(db)
    for i in range(3):
        entries = [
            cdl.ChatDictionary(key=f"lit{i}-{j:04d}", content=f"value {i}-{j}")
            for j in range(75)
        ] + [
            cdl.ChatDictionary(key=f"/re{i}-{j:02d}[a-z]+/i", content=f"regex {i}-{j}")
            for j in range(25)
        ]
        dict_id = cdl.save_chat_dictionary(db, f"Big{i}", entries=entries)
        service.attach_to_conversation(dict_id, conv_id)


def test_second_send_zero_loads_zero_parses_zero_reinstantiation_zero_compiles(
    dict_db, monkeypatch
):
    conv_id = dict_db.add_conversation({"title": "big"})
    _build_300_entry_store(dict_db, conv_id)

    load = _spy_load(monkeypatch)
    fd = _spy_from_dict(monkeypatch)
    jl = _spy_entries_json_loads(monkeypatch)
    compiles = _counting_compile(monkeypatch)

    message = "lit0-0007 met lit2-0042 while re1-03abc watched"
    text1 = apply_active_chatdicts_to_text(dict_db, conv_id, None, message)
    first = (load.count, fd.count, jl.count, len(compiles))
    # Cold start: 3 dictionary loads, 300 from_dict builds, 3 entries-array
    # JSON parses, and 225 literal whole-word helper compiles + 75 regex keys
    # x 2 compiles each (ReDoS validator syntax probe + stored compile).
    assert first == (3, 300, 3, 225 + 75 * 2)
    assert "value 0-7" in text1 and "regex 1-3" in text1

    load.count = fd.count = jl.count = 0
    compiles.clear()
    text2 = apply_active_chatdicts_to_text(dict_db, conv_id, None, message)
    assert (load.count, fd.count, jl.count, len(compiles)) == (0, 0, 0, 0)
    assert text2 == text1  # byte-identical second send

    # A different message still rides the same bundle: zero cold-start work.
    text3 = apply_active_chatdicts_to_text(
        dict_db, conv_id, None, "nothing at all matches here"
    )
    assert (load.count, fd.count, jl.count, len(compiles)) == (0, 0, 0, 0)
    assert text3 == "nothing at all matches here"


# ---------------------------------------------------------------------------
# Fix round 1 — timed-effect state across sends (sanctioned, now pinned)
# ---------------------------------------------------------------------------


def test_timed_effect_cooldown_persists_across_sends_and_resets_on_generation_bump(
    dict_db, monkeypatch
):
    """Pin the bundle cache's cross-send timed-effect semantics.

    Cached ``ChatDictionary`` instances persist ``last_triggered`` within a
    generation, so ``apply_timed_effects`` cooldowns actually suppress
    subsequent sends inside the window -- pre-change, per-send ``from_dict``
    re-instantiation reset the state and made cooldowns/delays inert across
    sends. A generation bump rebuilds the instances and resets the timing
    state. Asserted on replacement output (and the load spy proving one
    bundle rode both sends), never on ``last_triggered`` internals."""
    conv_id = dict_db.add_conversation({"title": "cooldown"})
    dict_id = cdl.save_chat_dictionary(
        dict_db,
        "Cooldown",
        entries=[
            cdl.ChatDictionary(
                key="boom",
                content="BANG",
                timed_effects={"sticky": 0, "cooldown": 60, "delay": 0},
            )
        ],
    )
    LocalChatDictionaryService(dict_db).attach_to_conversation(dict_id, conv_id)

    class _Clock:
        """Replaces cdl.datetime for the send path (only user: the
        pipeline's ``current_time = datetime.now()``); no sleeping."""

        def __init__(self):
            # Match the send path's legacy naive local clock.
            self.value = datetime(2026, 1, 1, 12, 0, 0)  # noqa: DTZ001

        def now(self):
            return self.value

    clock = _Clock()
    monkeypatch.setattr(cdl, "datetime", clock)
    load = _spy_load(monkeypatch)
    message = "the boom fires"

    # Send 1 at t0: fires and stamps the entry's timing state.
    assert apply_active_chatdicts_to_text(dict_db, conv_id, None, message) == (
        "the BANG fires"
    )
    # Send 2 at t0+1s: SAME cached instances (still one bundle), the
    # cooldown suppresses the replacement entirely.
    clock.value += timedelta(seconds=1)
    assert apply_active_chatdicts_to_text(dict_db, conv_id, None, message) == message
    assert load.count == 1  # both sends rode one bundle -- cooldown is cross-send

    # Send 3 at t0+61s: cooldown elapsed, fires again.
    clock.value += timedelta(seconds=60)
    assert apply_active_chatdicts_to_text(dict_db, conv_id, None, message) == (
        "the BANG fires"
    )

    # Send 4 at t0+62s would be inside the fresh 60s cooldown -- but a
    # dictionary edit bumps the generation: instances rebuild with the
    # timing state reset, so the entry fires immediately.
    clock.value += timedelta(seconds=1)
    cdl.update_chat_dictionary(dict_db, dict_id, description="bump")  # generation bump
    assert apply_active_chatdicts_to_text(dict_db, conv_id, None, message) == (
        "the BANG fires"
    )
    assert load.count == 2  # the bump rebuilt the bundle
