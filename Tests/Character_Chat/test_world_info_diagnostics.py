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
        "keys": ["Warden"], "activation_reason": "matched key 'Warden'", "status": "fired",
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
