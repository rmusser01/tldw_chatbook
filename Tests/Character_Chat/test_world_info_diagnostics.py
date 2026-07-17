from tldw_chatbook.Character_Chat.world_info_diagnostics import (
    WorldBookEntryDiagnostic,
    WorldBookScanDiagnostics,
)


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
