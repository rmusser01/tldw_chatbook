"""Model + behavior tests for per-entry enabled/case_sensitive/priority (P1c)."""

from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
    apply_replacement_once,
    group_scoring,
    match_whole_words,
    ChatDictionary,
)


def _entry(key, content, **kwargs):
    return ChatDictionary(key=key, content=content, **kwargs)


class TestEntryModelFields:
    def test_defaults_and_roundtrip(self):
        entry = _entry("BP", "blood pressure")
        assert entry.enabled is True
        assert entry.case_sensitive is False
        assert entry.priority == 0
        payload = entry.to_dict()
        assert payload["enabled"] is True
        assert payload["case_sensitive"] is False
        assert payload["priority"] == 0
        clone = ChatDictionary.from_dict(payload)
        assert (clone.enabled, clone.case_sensitive, clone.priority) == (True, False, 0)

    def test_explicit_values_roundtrip(self):
        entry = _entry("BP", "bp", enabled=False, case_sensitive=True, priority=7)
        clone = ChatDictionary.from_dict(entry.to_dict())
        assert (clone.enabled, clone.case_sensitive, clone.priority) == (False, True, 7)

    def test_legacy_stored_dict_without_new_keys_parses(self):
        legacy = {"key": "BP", "content": "blood pressure", "probability": 100,
                  "group": None, "timed_effects": None, "max_replacements": 1, "is_regex": False}
        clone = ChatDictionary.from_dict(legacy)
        assert (clone.enabled, clone.case_sensitive, clone.priority) == (True, False, 0)


class TestCaseSensitivity:
    def test_literal_default_stays_case_insensitive(self):
        entry = _entry("BP", "blood pressure")
        assert match_whole_words([entry], "check bp now") == [entry]
        text, count = apply_replacement_once("check bp now", entry)
        assert count == 1 and text == "check blood pressure now"

    def test_literal_case_sensitive_matches_exact_only(self):
        entry = _entry("BP", "blood pressure", case_sensitive=True)
        assert match_whole_words([entry], "check bp now") == []
        assert match_whole_words([entry], "check BP now") == [entry]
        text, count = apply_replacement_once("check bp now", entry)
        assert count == 0 and text == "check bp now"
        text, count = apply_replacement_once("check BP now", entry)
        assert count == 1 and text == "check blood pressure now"

    def test_regex_keys_ignore_case_sensitive_flag(self):
        # Regex case comes from the pattern's own /i flag, not the field.
        entry = _entry("/spo2/i", "oxygen saturation", case_sensitive=True)
        assert match_whole_words([entry], "SpO2 low") == [entry]


class TestGroupWinner:
    def test_priority_wins_over_length(self):
        long_low = _entry("blood pressure cuff", "sphygmomanometer", group="med", priority=0)
        short_high = _entry("cuff", "wrap", group="med", priority=5)
        winners = group_scoring([long_low, short_high])
        assert winners == [short_high]

    def test_legacy_equal_priority_keeps_length_winner(self):
        long_e = _entry("blood pressure cuff", "sphygmomanometer", group="med")
        short_e = _entry("cuff", "wrap", group="med")
        assert group_scoring([long_e, short_e]) == [long_e]
