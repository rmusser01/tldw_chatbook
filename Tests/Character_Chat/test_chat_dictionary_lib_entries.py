"""Model + behavior tests for per-entry enabled/case_sensitive/priority (P1c)."""

from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
    apply_replacement_once,
    group_scoring,
    match_whole_words,
    ChatDictionary,
    process_user_input,
    process_user_input_with_diagnostics,
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


class TestUnifiedOrdering:
    def test_disabled_entry_is_near_miss_not_fired(self):
        on = _entry("BP", "blood pressure")
        off = _entry("HR", "heart rate", enabled=False)
        text, diag = process_user_input_with_diagnostics("BP and HR", [on, off])
        assert text == "blood pressure and HR"
        by = {r.pattern: r for r in diag.entries}
        assert by["HR"].status == "skipped:disabled"
        assert by["BP"].status == "fired"
        assert diag.matched == 2 and diag.fired == 1 and diag.skipped == 1

    def test_priority_governs_budget_survival(self):
        # Strategy order (alphabetical) would put "aa" first; priority must trump it.
        cheap_low = _entry("aa", "w1 w2 w3", priority=0)      # 3 tokens
        pricey_high = _entry("zz", "w1 w2 w3 w4", priority=5)  # 4 tokens
        _, diag = process_user_input_with_diagnostics("aa zz", [cheap_low, pricey_high], max_tokens=4)
        by = {r.pattern: r for r in diag.entries}
        assert by["zz"].status == "fired"                      # high priority survived
        assert by["aa"].status == "skipped:token_budget"       # walk stopped after zz
        assert diag.budget_exceeded is True
        assert diag.tokens_used == 4

    def test_priority_governs_application_order(self):
        first = _entry("zz", "Z", priority=9)
        second = _entry("aa", "A", priority=0)
        _, diag = process_user_input_with_diagnostics("aa zz", [first, second])
        by = {r.pattern: r for r in diag.entries}
        assert by["zz"].applied_order == 0                     # -priority beats alphabetical
        assert by["aa"].applied_order == 1

    def test_legacy_zero_priority_keeps_strategy_application_order(self):
        a = _entry("bb", "B")
        b = _entry("aa", "A")
        _, diag = process_user_input_with_diagnostics("aa bb", [a, b])
        by = {r.pattern: r for r in diag.entries}
        assert by["aa"].applied_order == 0 and by["bb"].applied_order == 1  # alphabetical

    def test_legacy_budget_survival_now_follows_strategy_order(self):
        # THE spec'd legacy delta: stored order was [zz, aa]; survival now walks
        # strategy (alphabetical) order, so "aa" survives a 1-token budget.
        stored_first = _entry("zz", "w1")   # 1 token
        stored_second = _entry("aa", "w1")  # 1 token
        _, diag = process_user_input_with_diagnostics("aa zz", [stored_first, stored_second], max_tokens=1)
        by = {r.pattern: r for r in diag.entries}
        assert by["aa"].status == "fired"
        assert by["zz"].status == "skipped:token_budget"

    def test_content_preview_flattens_whitespace(self):
        entry = _entry("BP", "blood\npressure\treading")
        _, diag = process_user_input_with_diagnostics("BP", [entry])
        assert diag.entries[0].content_preview == "blood pressure reading"

    def test_wrapper_still_matches_core(self):
        entries = [
            _entry("BP", "blood pressure", priority=3),
            _entry("HR", "heart rate", enabled=False),
            _entry("bp2", "exact", case_sensitive=True),
        ]
        sample = "BP HR bp2 end"
        assert process_user_input(sample, entries) == \
            process_user_input_with_diagnostics(sample, entries)[0]


class TestLooseTypedCoercion:
    def test_priority_none_and_garbage_default_to_zero(self):
        assert ChatDictionary.from_dict(
            {"key": "BP", "content": "x", "priority": None}
        ).priority == 0  # RED-first: TypeError pre-fix
        assert _entry("BP", "x", priority="garbage").priority == 0

    def test_string_false_is_false(self):
        entry = _entry("BP", "x", enabled="false", case_sensitive="False")
        assert entry.enabled is False        # RED-first: True pre-fix
        assert entry.case_sensitive is False

    def test_string_true_and_unrecognized(self):
        assert _entry("BP", "x", enabled="yes").enabled is True
        assert _entry("BP", "x", enabled="maybe").enabled is True   # default
        assert _entry("BP", "x", case_sensitive="maybe").case_sensitive is False  # default
