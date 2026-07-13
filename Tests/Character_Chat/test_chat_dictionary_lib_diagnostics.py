"""Engine-level tests for the additive diagnostics path (P1b)."""

from datetime import datetime, timedelta

from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
    ChatDictionary,
    process_user_input,
    process_user_input_with_diagnostics,
)


def _entry(key, content, **kwargs):
    return ChatDictionary(key=key, content=content, **kwargs)


class TestDiagnosticsCore:
    def test_fired_entry_recorded_with_counts_and_order(self):
        entries = [_entry("BP", "blood pressure"), _entry("HR", "heart rate")]
        text, diag = process_user_input_with_diagnostics("BP and HR now", entries)
        assert text == "blood pressure and heart rate now"
        assert diag.matched == 2 and diag.fired == 2 and diag.skipped == 0
        by_pattern = {r.pattern: r for r in diag.entries}
        assert by_pattern["BP"].status == "fired"
        assert by_pattern["BP"].replacements == 1
        assert by_pattern["BP"].input_index == 0
        # applied_order follows the post-strategy (alphabetical) sequence.
        assert by_pattern["BP"].applied_order == 0
        assert by_pattern["HR"].applied_order == 1
        assert by_pattern["BP"].token_cost == 2  # "blood pressure".split()
        assert by_pattern["BP"].content_preview.startswith("blood pressure")
        assert diag.total_replacements == 2

    def test_never_matched_entries_are_omitted(self):
        entries = [_entry("BP", "blood pressure"), _entry("XYZZY", "plugh")]
        _, diag = process_user_input_with_diagnostics("BP only", entries)
        assert diag.matched == 1
        assert [r.pattern for r in diag.entries] == ["BP"]

    def test_group_scoring_loser_is_near_miss(self):
        # Named group: longest raw_key wins; the shorter one is skipped.
        winner = _entry("blood pressure cuff", "sphygmomanometer", group="med")
        loser = _entry("cuff", "wrap", group="med")
        _, diag = process_user_input_with_diagnostics(
            "the blood pressure cuff is here", [winner, loser]
        )
        by_pattern = {r.pattern: r for r in diag.entries}
        assert by_pattern["cuff"].status == "skipped:group_scoring"
        assert by_pattern["blood pressure cuff"].status == "fired"
        assert diag.matched == 2 and diag.fired == 1 and diag.skipped == 1

    def test_probability_zero_is_near_miss_and_hundred_fires(self):
        entries = [
            _entry("BP", "blood pressure", probability=100),
            _entry("HR", "heart rate", probability=0),
        ]
        _, diag = process_user_input_with_diagnostics("BP and HR", entries)
        by_pattern = {r.pattern: r for r in diag.entries}
        assert by_pattern["BP"].status == "fired"
        assert by_pattern["HR"].status == "skipped:probability"

    def test_cooldown_is_near_miss(self):
        entry = _entry("BP", "blood pressure", timed_effects={"sticky": 0, "cooldown": 3600, "delay": 0})
        entry.last_triggered = datetime.now() - timedelta(seconds=10)
        _, diag = process_user_input_with_diagnostics("BP now", [entry])
        assert diag.entries[0].status == "skipped:timed_effects"
        assert diag.fired == 0 and diag.skipped == 1

    def test_budget_truncation_marks_survivor_accounting_and_flag(self):
        # Costs: big=6 tokens, small=1 token. Budget 5: big fits? 6 > 5 -> break
        # at big; small (after the break) is ALSO dropped (walk-and-stop).
        cheap = _entry("aa", "one")                     # cost 1 - fits first (alphabetical strategy is later; budget uses MATCH order)
        big = _entry("bb", "w1 w2 w3 w4 w5 w6")         # cost 6
        small = _entry("cc", "tiny")                    # cost 1, after the break
        _, diag = process_user_input_with_diagnostics("aa bb cc", [cheap, big, small], max_tokens=5)
        by_pattern = {r.pattern: r for r in diag.entries}
        assert by_pattern["aa"].status == "fired"
        assert by_pattern["bb"].status == "skipped:token_budget"
        assert by_pattern["cc"].status == "skipped:token_budget"  # post-break drop
        assert diag.budget_exceeded is True
        assert diag.tokens_used == 1  # budget-stage accounting: survivors only
        assert diag.token_budget == 5

    def test_sequential_consumption_yields_no_replacement(self):
        # Post-strategy order is alphabetical by raw_key: "aa bb" applies before "bb".
        eater = _entry("aa bb", "eaten")   # replaces the whole phrase first
        victim = _entry("bb", "late")      # matched original text; nothing left to replace
        text, diag = process_user_input_with_diagnostics("aa bb here", [eater, victim])
        assert text == "eaten here"
        by_pattern = {r.pattern: r for r in diag.entries}
        assert by_pattern["aa bb"].status == "fired"
        assert by_pattern["bb"].status == "no_replacement"
        assert by_pattern["bb"].replacements == 0
        assert by_pattern["bb"].applied_order is not None  # it reached the loop
        # no_replacement counts as skipped in the totals invariant.
        assert diag.matched == 2 and diag.fired == 1 and diag.skipped == 1

    def test_totals_invariant_and_to_dict_shape(self):
        entries = [
            _entry("BP", "blood pressure", probability=100),
            _entry("HR", "heart rate", probability=0),
        ]
        _, diag = process_user_input_with_diagnostics("BP and HR", entries)
        assert diag.matched == diag.fired + diag.skipped
        payload = diag.to_dict()
        assert set(payload) == {
            "entries", "matched", "fired", "skipped", "total_replacements",
            "tokens_used", "token_budget", "budget_exceeded",
        }
        record = payload["entries"][0]
        assert set(record) == {
            "input_index", "pattern", "status", "replacements",
            "token_cost", "applied_order", "content_preview",
        }


class TestWrapperContract:
    def test_process_user_input_returns_identical_string(self):
        # Deterministic mix: fired, probability-0 skip, budget truncation, regex.
        entries = [
            _entry("BP", "blood pressure", probability=100),
            _entry("HR", "heart rate", probability=0),
            _entry("/spo2/i", "oxygen saturation"),
            _entry("zz", "w1 w2 w3 w4 w5 w6 w7 w8"),
        ]
        sample = "BP HR SpO2 zz end"
        wrapped = process_user_input(sample, entries, max_tokens=6)
        core_text, _ = process_user_input_with_diagnostics(sample, entries, max_tokens=6)
        assert wrapped == core_text

    def test_none_entries_degrades_to_unchanged_input(self):
        # Pre-refactor behavior: iteration failure degrades inside stage 1.
        assert process_user_input("hello world", None) == "hello world"
        text, diag = process_user_input_with_diagnostics("hello world", None)
        assert text == "hello world"
        assert diag.matched == 0

    def test_single_pass_iterable_still_substitutes(self):
        class OneShot:
            def __init__(self, items):
                self._items = items
                self._used = False
            def __iter__(self):
                assert not self._used, "consumed twice"
                self._used = True
                return iter(self._items)
            def __len__(self):
                return len(self._items)

        one_shot = OneShot([_entry("BP", "blood pressure")])
        assert process_user_input("BP now", one_shot) == "blood pressure now"

    def test_generator_entries_degrades_to_unchanged_input(self):
        # Length-less generators degraded to no-op pre-refactor (len() raised
        # inside stage 1); the fix must preserve exactly that.
        gen = (e for e in [_entry("BP", "blood pressure")])
        assert process_user_input("BP now", gen) == "BP now"
