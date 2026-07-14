"""Pure unit tests for the dictionary validation module (P1c)."""

from tldw_chatbook.Widgets.Persona_Widgets.personas_dictionary_validation import (
    ValidationFinding,
    validate_entries,
)


def _entry(pattern, *, etype="literal", probability=1.0, case_sensitive=False,
           entry_id="local:chat_dictionary_entry:1:0"):
    return {"id": entry_id, "pattern": pattern, "replacement": "x", "type": etype,
            "probability": probability, "case_sensitive": case_sensitive}


def test_clean_entries_yield_no_findings():
    assert validate_entries([_entry("BP"), _entry("/spo2/i", etype="regex")]) == []


def test_invalid_regex_detected_via_real_parser():
    findings = validate_entries([_entry("/[unclosed/", etype="regex")])
    assert [f.code for f in findings] == ["invalid_regex"]
    assert findings[0].field == "pattern"
    assert findings[0].entry_id == "local:chat_dictionary_entry:1:0"


def test_duplicate_pattern_same_type_flagged_once_per_extra():
    entries = [
        _entry("BP", entry_id="local:chat_dictionary_entry:1:0"),
        _entry("BP", entry_id="local:chat_dictionary_entry:1:1"),
        _entry("BP", etype="regex", entry_id="local:chat_dictionary_entry:1:2"),  # different type: ok
    ]
    findings = validate_entries(entries)
    dups = [f for f in findings if f.code == "duplicate_pattern"]
    assert len(dups) == 1 and dups[0].entry_id.endswith(":1")


def test_probability_zero_flagged():
    findings = validate_entries([_entry("BP", probability=0.0)])
    assert [f.code for f in findings] == ["probability_zero"]


def test_case_flag_on_regex_flagged():
    findings = validate_entries([_entry("/spo2/i", etype="regex", case_sensitive=True)])
    assert [f.code for f in findings] == ["case_flag_on_regex"]
