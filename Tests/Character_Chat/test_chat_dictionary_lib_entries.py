"""Model + behavior tests for per-entry enabled/case_sensitive/priority (P1c)."""

from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import ChatDictionary


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
