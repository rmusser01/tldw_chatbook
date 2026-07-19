import copy

from tldw_chatbook.Character_Chat.world_info_diagnostics import (
    WorldBookEntryDiagnostic,
    WorldBookScanDiagnostics,
)
from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor


def test_entry_diagnostic_to_dict_round_trips_fields():
    rec = WorldBookEntryDiagnostic(
        entry_id=7, source_book_id=3, source_book_name="Blackreach",
        keys=["Warden"], activation_reason="matched key 'Warden'", status="fired",
        token_cost=12, injection_order=0, position="before_char",
        content_preview="The grim jailer…", depth_level=0,
    )
    assert rec.to_dict() == {
        "entry_id": 7, "source_book_id": 3, "source_book_name": "Blackreach",
        "keys": ["Warden"], "priority": 0, "activation_reason": "matched key 'Warden'", "status": "fired",
        "token_cost": 12, "injection_order": 0, "position": "before_char",
        "content_preview": "The grim jailer…", "depth_level": 0,
    }


def test_scan_diagnostic_to_dict_nests_entries_and_summary():
    rec = WorldBookEntryDiagnostic(
        entry_id=1, source_book_id=1, source_book_name="B", keys=["k"],
        activation_reason="disabled", status="skipped:disabled",
        token_cost=0, injection_order=None, position="before_char",
        content_preview="", depth_level=0,
    )
    diag = WorldBookScanDiagnostics(
        entries=[rec], matched=1, fired=0, skipped=1,
        tokens_used=0, token_budget=500, budget_exceeded=False, books_scanned=1,
    )
    out = diag.to_dict()
    assert out["matched"] == 1 and out["fired"] == 0 and out["skipped"] == 1
    assert out["token_budget"] == 500 and out["books_scanned"] == 1
    assert out["entries"] == [rec.to_dict()]


def test_scan_diagnostic_defaults():
    diag = WorldBookScanDiagnostics()
    d = diag.to_dict()
    assert d["entries"] == [] and d["matched"] == 0 and d["fired"] == 0
    assert d["budget_exceeded"] is False and d["books_scanned"] == 0


def _book(book_id, name, entries, **kw):
    return {"id": book_id, "name": name, "enabled": True, "scan_depth": 3,
            "token_budget": 500, "recursive_scanning": False, "entries": entries, **kw}


def _entry(entry_id, keys, content, **kw):
    return {"id": entry_id, "keys": keys, "content": content, "enabled": True,
            "position": "before_char", "insertion_order": 0, "selective": False,
            "secondary_keys": [], "case_sensitive": False, **kw}


def test_candidate_entries_include_disabled_and_source_meta():
    book = _book(3, "Blackreach", [
        _entry(7, ["Warden"], "grim jailer", enabled=True),
        _entry(8, ["Ghost"], "pale figure", enabled=False),
    ])
    proc = WorldInfoProcessor(world_books=[book])
    # plain path: only the enabled entry loaded
    assert len(proc.entries) == 1
    # diagnostics candidate list: BOTH, tagged with source + enabled + id
    cand = {c["_entry_id"]: c for c in proc._candidate_entries}
    assert set(cand) == {7, 8}
    assert cand[7]["_book_id"] == 3 and cand[7]["_book_name"] == "Blackreach"
    assert cand[7]["_enabled"] is True and cand[8]["_enabled"] is False


def test_plain_process_messages_byte_identical_disabled_and_selective():
    book = _book(1, "B", [
        _entry(1, ["Warden"], "grim jailer", enabled=True),
        _entry(2, ["Ghost"], "pale figure", enabled=False),
        _entry(3, ["Vault"], "sealed door", selective=True, secondary_keys=["gold"]),
    ])
    proc = WorldInfoProcessor(world_books=[book])
    before = copy.deepcopy(proc.process_messages("The Warden guards the Vault of gold.", []))
    # Re-run to ensure determinism/no mutation of internal state.
    after = proc.process_messages("The Warden guards the Vault of gold.", [])
    assert before == after
    # Warden fires, Ghost disabled (never), Vault selective+secondary 'gold' present → fires.
    contents = before["injections"]["before_char"]
    assert "grim jailer" in contents and "sealed door" in contents
    assert "pale figure" not in contents


def test_classify_entry_match_decomposes_reason():
    book = _book(1, "B", [_entry(3, ["Vault"], "x", selective=True, secondary_keys=["gold"])])
    proc = WorldInfoProcessor(world_books=[book])
    entry = next(c for c in proc._candidate_entries if c["_entry_id"] == 3)
    text = "The Vault is sealed."
    primary_hit, pk, sec_req, sec_hit, sk = proc._classify_entry_match(entry, text, text.lower())
    assert primary_hit is True and pk == "Vault"
    assert sec_req is True and sec_hit is False and sk is None
    text2 = "The Vault of gold."
    p2, pk2, sr2, sh2, sk2 = proc._classify_entry_match(entry, text2, text2.lower())
    assert p2 and sr2 and sh2 and sk2 == "gold"


def test_diagnostics_result_equals_plain_process_messages():
    book = _book(1, "B", [
        _entry(1, ["Warden"], "grim jailer"),
        _entry(2, ["Ghost"], "pale figure", enabled=False),
        _entry(3, ["Vault"], "sealed door", selective=True, secondary_keys=["gold"]),
    ])
    proc = WorldInfoProcessor(world_books=[book])
    msg = "The Warden guards the Vault of gold."
    plain = proc.process_messages(msg, [])
    result, diag = proc.process_messages_with_diagnostics(msg, [])
    assert result == plain                       # byte-identical result (the fired-set pin)
    assert isinstance(diag, WorldBookScanDiagnostics)


def test_diagnostics_classifies_disabled_secondary_and_budget():
    book = _book(1, "B", [
        _entry(1, ["Warden"], "AAAA " * 200),    # ~200 tokens, fires first
        _entry(2, ["Warden"], "BBBB " * 200),    # matched but past the budget break
        _entry(3, ["Ghost"], "pale", enabled=False),
        _entry(4, ["Vault"], "sealed", selective=True, secondary_keys=["gold"]),
    ], token_budget=250)
    proc = WorldInfoProcessor(world_books=[book])
    # `_process_world_books` takes max(self.token_budget, book_budget), and the
    # constructor's own default is already 500, so a single book with a *lower*
    # budget than 500 can never pull the effective budget down (pre-existing
    # behavior, unrelated to this task). Set it directly to exercise the break.
    proc.token_budget = 250
    _result, diag = proc.process_messages_with_diagnostics("Warden Ghost Vault", [])
    by_id = {e.entry_id: e for e in diag.entries}
    assert by_id[1].status == "fired"
    assert by_id[2].status == "skipped:budget"          # hard break drops everything after
    assert by_id[3].status == "skipped:disabled"        # disabled but key matched
    assert by_id[4].status == "skipped:secondary"       # 'gold' absent
    assert diag.budget_exceeded is True
    assert by_id[1].source_book_name == "B" and by_id[1].injection_order is not None
    # an entry whose key never appears is NOT reported at all
    assert all(e.status != "no_match" for e in diag.entries)


def test_diagnostics_multi_book_priority_offset_agrees_with_plain():
    hi = _book(10, "Hi", [_entry(1, ["Warden"], "hi-content")], priority=1)   # offset 1000
    lo = _book(11, "Lo", [_entry(2, ["Warden"], "lo-content")], priority=0)
    proc = WorldInfoProcessor(world_books=[lo, hi])
    plain = proc.process_messages("Warden", [])
    result, _diag = proc.process_messages_with_diagnostics("Warden", [])
    assert result == plain


def test_diagnostics_reports_recursively_fired_entries():
    """recursive_scanning=True: 'castle' is keyed off the user message, and
    its content mentions 'dragon' — a second entry whose key never appears
    in the user message or history, only inside 'castle's content. On a real
    send this entry legitimately fires (mirrors test_recursive_scanning in
    test_world_info.py). Diagnostics must report it as fired (not silently
    drop it), with fired == len(matched_entries)."""
    book = _book(1, "B", [
        _entry(1, ["castle"], "The castle is protected by a dragon."),
        _entry(2, ["dragon"], "Dragons breathe fire."),
    ], recursive_scanning=True)
    proc = WorldInfoProcessor(world_books=[book])
    msg = "Tell me about the castle."

    plain = proc.process_messages(msg, [])
    result, diag = proc.process_messages_with_diagnostics(msg, [])

    # The byte-identical pin still holds.
    assert result == plain
    # Both castle (primary hit) and dragon (recursive-only hit) fired.
    assert len(result["matched_entries"]) == 2
    assert diag.fired == len(result["matched_entries"])
    assert diag.fired == 2

    by_key = {tuple(e.keys): e for e in diag.entries}
    castle = by_key[("castle",)]
    assert castle.status == "fired" and castle.depth_level == 0

    dragon = by_key[("dragon",)]
    assert dragon.status == "fired"
    assert dragon.depth_level == 1
    assert "recursive" in dragon.activation_reason


def test_classify_entry_match_tolerates_null_keys():
    """A candidate/entry whose keys or secondary_keys are None (e.g. an explicit
    null DB field) must not raise in the diagnostics classifier (Gemini #673 —
    `.get('keys', [])` returns None, not [], when the key exists with value None)."""
    proc = WorldInfoProcessor(world_books=[_book(1, "B", [_entry(1, ["Warden"], "x")])])
    # keys=None -> no primary match, no TypeError
    assert proc._classify_entry_match(
        {"keys": None, "secondary_keys": None}, "The Warden.", "the warden."
    ) == (False, None, False, False, None)
    # selective with secondary_keys=None -> primary hit, secondary not required (falsy)
    p, pk, sr, sh, sk = proc._classify_entry_match(
        {"keys": ["Warden"], "selective": True, "secondary_keys": None},
        "The Warden.", "the warden.",
    )
    assert p is True and pk == "Warden" and sr is False


def test_high_priority_entry_survives_budget_over_low_priority():
    """Priority (not FIFO) decides budget survival: the high-priority entry
    fires even though a low-priority one comes first by insertion_order."""
    book = _book(1, "B", [
        _entry(1, ["low"], "AAAA " * 200, insertion_order=0, priority=0),    # ~250 tok, first by order
        _entry(2, ["high"], "BBBB " * 200, insertion_order=1, priority=90),  # ~250 tok, high priority
    ], token_budget=300)   # fits exactly one entry, not two
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("low high", [])
    fired = result["injections"]["before_char"]
    assert any("BBBB" in c for c in fired)   # high-priority survives
    assert all("AAAA" not in c for c in fired)  # low-priority dropped


def test_low_book_token_budget_is_honored():
    """A book token_budget below the old 500 default is honored (floor fix)."""
    book = _book(1, "B", [
        _entry(1, ["a"], "AAAA " * 30, insertion_order=0),   # ~30 tok
        _entry(2, ["b"], "BBBB " * 30, insertion_order=1),   # ~30 tok, over a 40-token budget
    ], token_budget=40)
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("a b", [])
    fired = result["injections"]["before_char"]
    assert any("AAAA" in c for c in fired) and all("BBBB" not in c for c in fired)


def test_injection_order_reflects_priority():
    book = _book(1, "B", [
        _entry(1, ["a"], "content-a", insertion_order=0, priority=1),
        _entry(2, ["b"], "content-b", insertion_order=1, priority=99),
    ])
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("a b", [])
    injected = result["injections"]["before_char"]
    assert injected == ["content-b", "content-a"]   # priority 99 before priority 1


def test_recursive_entry_reorders_by_insertion_order():
    """Recursive-scan normalization: an entry that fires via recursion but has
    a LOWER insertion_order than a direct match now orders ahead of it."""
    book = _book(1, "B", [
        _entry(1, ["castle"], "the castle guards a dragon", insertion_order=5),
        _entry(2, ["dragon"], "a fearsome dragon", insertion_order=0),
    ], recursive_scanning=True)
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("the castle", [])
    injected = result["injections"]["before_char"]
    # entry 2 (insertion_order 0, fired via recursion) now precedes entry 1 (order 5).
    assert injected.index("a fearsome dragon") < injected.index("the castle guards a dragon")


def test_diagnostics_duplicate_signature_entries_keep_distinct_ids():
    """Two entries sharing an identical (insertion_order, content, position)
    signature that BOTH fire must each be attributed to their own entry_id, not
    collapsed onto the first candidate (Qodo #673)."""
    book = _book(1, "B", [
        _entry(1, ["alpha"], "SAME"),
        _entry(2, ["beta"], "SAME"),
    ])
    proc = WorldInfoProcessor(world_books=[book])
    _result, diag = proc.process_messages_with_diagnostics("alpha and beta here", [])
    fired = [e for e in diag.entries if e.status == "fired"]
    assert len(fired) == 2
    assert {e.entry_id for e in fired} == {1, 2}


def test_diagnostic_record_carries_priority():
    book = _book(1, "B", [_entry(1, ["Warden"], "grim jailer", priority=42)])
    proc = WorldInfoProcessor(world_books=[book])
    _result, diag = proc.process_messages_with_diagnostics("The Warden.", [])
    fired = next(e for e in diag.entries if e.status == "fired")
    assert fired.priority == 42
    assert fired.to_dict()["priority"] == 42


def test_null_insertion_order_does_not_crash_sort():
    """An imported/hand-edited entry with insertion_order=None (or a non-numeric
    priority) must not raise a TypeError. Equal priorities force the sort to
    compare the insertion_order key element; None must be coerced first (Qodo
    #682). The processor also sorts self.entries at construction, so a bad value
    would otherwise crash before process_messages() even runs."""
    book = _book(1, "B", [
        _entry(1, ["a"], "content-a", insertion_order=None, priority="bad"),
        _entry(2, ["b"], "content-b", insertion_order=1, priority=0),
    ])
    proc = WorldInfoProcessor(world_books=[book])   # must not raise
    result = proc.process_messages("a b", [])       # must not raise
    injected = result["injections"]["before_char"]
    assert "content-a" in injected and "content-b" in injected
