# Roleplay P1b — Try-it Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Dictionaries Try-it pane real diagnostics — which entries fired, which near-missed and why, and the token-budget picture — via an additive diagnostics path in the substitution engine, with the chat-time `str` contract byte-compatible.

**Architecture:** Refactor-into-core: `process_user_input`'s body moves into `process_user_input_with_diagnostics(...) -> tuple[str, DictionaryProcessDiagnostics]`, instrumented purely by identity-preserving stage-boundary diffs (no helper changes); the old function becomes a 2-line wrapper. The local service adds ONE additive `diagnostics` response key with append-time entry-id enrichment; the Try-it widget renders summary + fired + near-miss sections with graceful degrade.

**Tech Stack:** Python ≥3.11 dataclasses, Textual, pytest (+asyncio for UI), real `CharactersRAGDB` for service tests.

**Spec:** `Docs/superpowers/specs/2026-07-14-roleplay-p1b-tryit-diagnostics-design.md` (committed `ccd1e6f5`; its "Engine reality" and ACs are binding — read once before Task 1).

## Global Constraints

- **Chat-time contract byte-compatible:** `process_user_input(user_input, entries, max_tokens=5000, strategy="sorted_evenly") -> str` keeps its exact signature, return type, and behavior (call sites `Chat/Chat_Functions.py:1024` and `:1281` unchanged). A wrapper-contract test pins it.
- **No helper-function changes** in `Chat_Dictionary_Lib.py` (`match_whole_words`, `group_scoring`, `filter_by_probability`, `apply_timed_effects`, `enforce_token_budget`, `apply_strategy`, `apply_replacement_once`, `calculate_token_usage`, `alert_token_budget_exceeded`) — diagnostics come from stage-boundary diffs only. The vestigial `except TokenBudgetExceededWarning` and the dead `alert_token_budget_exceeded` call are preserved **verbatim** (behavior-preserving refactor; no cleanup).
- **`budget_exceeded` is truncation-derived** (≥1 matched entry dropped at the budget stage), never from the alert path.
- **`tokens_used` is budget-stage accounting**: sum of `token_cost` over entries that SURVIVED the budget stage (includes `no_replacement` survivors).
- **Totals invariant:** `matched == fired + skipped` (`no_replacement` counts in `skipped`; the UI shows it under near-misses).
- **Entry-id enrichment is append-time tracking** (a parallel `entry_ids` list built as `process_text` appends entries) — never `input_index == stored index` arithmetic (breaks under the `group` filter and the `dictionary_id=None` path).
- **All existing `process_text` response keys byte-identical**; `diagnostics` is the only addition, and its assembly failure must omit the key, never fail the response.
- **UI degrade (AC5):** a response without `diagnostics` renders the P1a diff-only view + a dim "diagnostics unavailable" note. Each diagnostics section guards independently.
- **The test fake mirrors the REAL `to_dict()` shape exactly** (the P1a fake-divergence lesson) — never a friendlier invented shape.
- Only these files change: `Character_Chat/Chat_Dictionary_Lib.py`, `Character_Chat/local_chat_dictionary_service.py`, `Widgets/Persona_Widgets/personas_dictionary_tryit.py`, `UI/Screens/personas_screen.py`, the three test files, and the spec's status line. No shared-shell files.
- Google-style docstrings on new public callables. Widget `DEFAULT_CSS` structure-only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (isolated HOME; from this worktree use the main checkout's absolute `.venv/bin/python`):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    .venv/bin/python -m pytest <target> \
    -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
  ```

---

### Task 1: Engine — diagnostics dataclasses + instrumented core + wrapper

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` (add dataclasses + `process_user_input_with_diagnostics`; shrink `process_user_input` at `:503-660`)
- Test: `Tests/Character_Chat/test_chat_dictionary_lib_diagnostics.py` (create)

**Interfaces:**
- Consumes: the existing pipeline helpers (unchanged) and `ChatDictionary(key, content, probability=100, group=None, timed_effects=None, max_replacements=1)` (`/pat/flags` key form marks regex; `entry.last_triggered` settable for cooldown tests).
- Produces (Tasks 2-3 rely on these EXACT names):
  - `@dataclass DictionaryEntryDiagnostic`: `input_index: int`, `pattern: str`, `status: str`, `replacements: int = 0`, `token_cost: int = 0`, `applied_order: Optional[int] = None`, `content_preview: str = ""`; method `to_dict() -> Dict[str, Any]`.
  - `@dataclass DictionaryProcessDiagnostics`: `entries: List[DictionaryEntryDiagnostic]`, `matched: int`, `fired: int`, `skipped: int`, `total_replacements: int`, `tokens_used: int`, `token_budget: int`, `budget_exceeded: bool`; method `to_dict() -> Dict[str, Any]` (`{"entries": [rec.to_dict()...], ...totals}`).
  - `process_user_input_with_diagnostics(user_input: str, entries: List[ChatDictionary], max_tokens: int = 5000, strategy: str = "sorted_evenly") -> Tuple[str, DictionaryProcessDiagnostics]`.
  - `status` values: `"fired"`, `"skipped:group_scoring"`, `"skipped:probability"`, `"skipped:timed_effects"`, `"skipped:token_budget"`, `"skipped:strategy_error"` (defensive — only reachable if `apply_strategy` raises and the pipeline's existing `except` empties the list), `"no_replacement"`. Records are stored in **input order**; fired records carry `applied_order` (0-based post-strategy application position).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Character_Chat/test_chat_dictionary_lib_diagnostics.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run (Global Constraints command): target `Tests/Character_Chat/test_chat_dictionary_lib_diagnostics.py`
Expected: FAIL — `ImportError: cannot import name 'process_user_input_with_diagnostics'`.

- [ ] **Step 3: Implement**

In `Chat_Dictionary_Lib.py`:

(a) Add to the imports: `from dataclasses import dataclass, field`.

(b) Immediately above `process_user_input` (`:503`), add:

```python
@dataclass
class DictionaryEntryDiagnostic:
    """One matched entry's outcome in the substitution pipeline.

    Args:
        input_index: Position of the entry in the caller-provided list.
        pattern: The entry's raw key (slash-delimited for regex entries).
        status: ``"fired"``, ``"skipped:<stage>"``, or ``"no_replacement"``.
        replacements: Number of replacements this entry performed.
        token_cost: Approximate token cost of the entry's content.
        applied_order: 0-based position in the post-strategy application
            sequence for entries that reached the replacement loop, else None.
        content_preview: The first 40 characters of the entry's content.
    """

    input_index: int
    pattern: str
    status: str
    replacements: int = 0
    token_cost: int = 0
    applied_order: Optional[int] = None
    content_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Returns the record as a plain JSON-safe dict."""
        return {
            "input_index": self.input_index,
            "pattern": self.pattern,
            "status": self.status,
            "replacements": self.replacements,
            "token_cost": self.token_cost,
            "applied_order": self.applied_order,
            "content_preview": self.content_preview,
        }


@dataclass
class DictionaryProcessDiagnostics:
    """Aggregate diagnostics for one substitution run.

    Totals maintain the invariant ``matched == fired + skipped`` —
    ``no_replacement`` entries count as skipped. ``tokens_used`` is
    budget-stage accounting: the summed cost of entries that survived the
    token-budget stage (including no_replacement survivors), and
    ``budget_exceeded`` is truncation-derived (at least one matched entry
    was dropped at the budget stage).
    """

    entries: List[DictionaryEntryDiagnostic] = field(default_factory=list)
    matched: int = 0
    fired: int = 0
    skipped: int = 0
    total_replacements: int = 0
    tokens_used: int = 0
    token_budget: int = 0
    budget_exceeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Returns the diagnostics as a plain JSON-safe dict."""
        return {
            "entries": [record.to_dict() for record in self.entries],
            "matched": self.matched,
            "fired": self.fired,
            "skipped": self.skipped,
            "total_replacements": self.total_replacements,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "budget_exceeded": self.budget_exceeded,
        }
```

(c) Add `process_user_input_with_diagnostics` containing the MOVED body of `process_user_input` (`:539-660`) with recording added. The stage code, logging, and every `try/except` (including the vestigial `except TokenBudgetExceededWarning` and the dead `alert_token_budget_exceeded` call) stay **verbatim**; only the marked additions are new:

```python
def process_user_input_with_diagnostics(
    user_input: str,
    entries: List[ChatDictionary],
    max_tokens: int = 5000,
    strategy: str = "sorted_evenly"
) -> Tuple[str, DictionaryProcessDiagnostics]:
    """Runs the substitution pipeline and reports per-entry diagnostics.

    Identical behavior to :func:`process_user_input` (which now wraps this
    function); diagnostics are collected purely from stage-boundary diffs,
    so no pipeline stage behaves differently.

    Args:
        user_input: The text input from the user.
        entries: A list of ``ChatDictionary`` objects to apply.
        max_tokens: The maximum token budget for applied entries' content.
        strategy: The sorting strategy for entries before replacement.

    Returns:
        A ``(processed_text, diagnostics)`` tuple. On critical pipeline
        failure the original input is returned with whatever diagnostics
        were collected before the failure.
    """
    current_time = datetime.now()
    original_input_for_fallback = user_input
    temp_user_input = user_input

    diagnostics = DictionaryProcessDiagnostics(token_budget=max_tokens)
    # First-wins for pathological duplicate objects in the input list.
    index_by_id: Dict[int, int] = {}
    for input_index, candidate in enumerate(entries):
        index_by_id.setdefault(id(candidate), input_index)
    matched_snapshot: List[ChatDictionary] = []
    skip_reason_by_id: Dict[int, str] = {}
    replacements_by_id: Dict[int, int] = {}
    applied_order_by_id: Dict[int, int] = {}
    budget_survivor_ids: Set[int] = set()

    def _record_stage_drops(
        before: List[ChatDictionary], after: List[ChatDictionary], stage: str
    ) -> None:
        surviving = {id(e) for e in after}
        for candidate in before:
            if id(candidate) not in surviving and id(candidate) not in skip_reason_by_id:
                skip_reason_by_id[id(candidate)] = f"skipped:{stage}"

    def _finalize() -> None:
        for candidate in matched_snapshot:
            entry_id = id(candidate)
            replacements = replacements_by_id.get(entry_id, 0)
            if entry_id in skip_reason_by_id:
                status = skip_reason_by_id[entry_id]
            elif replacements > 0:
                status = "fired"
            else:
                status = "no_replacement"
            diagnostics.entries.append(
                DictionaryEntryDiagnostic(
                    input_index=index_by_id.get(entry_id, -1),
                    pattern=str(candidate.raw_key),
                    status=status,
                    replacements=replacements,
                    token_cost=calculate_token_usage([candidate]),
                    applied_order=applied_order_by_id.get(entry_id),
                    content_preview=str(candidate.content or "")[:40],
                )
            )
        diagnostics.matched = len(matched_snapshot)
        diagnostics.fired = sum(1 for r in diagnostics.entries if r.status == "fired")
        diagnostics.skipped = diagnostics.matched - diagnostics.fired
        diagnostics.total_replacements = sum(r.replacements for r in diagnostics.entries)
        # Budget-stage accounting: survivors of the budget stage, including
        # no_replacement survivors (they consumed budget without firing).
        diagnostics.tokens_used = sum(
            calculate_token_usage([candidate])
            for candidate in matched_snapshot
            if id(candidate) in budget_survivor_ids
        )

    try:
        # 1. Match entries  (verbatim from the original body)
        logging.debug(f"Chat Dictionary: Initial matching for: {user_input[:100]}")
        try:
            valid_initial_entries = [e for e in entries if isinstance(e, ChatDictionary)]
            if len(valid_initial_entries) != len(entries):
                logging.warning("Some provided entries were not ChatDictionary instances and were skipped.")
            matched_entries = match_whole_words(valid_initial_entries, user_input)
        except re.error as e:
            log_counter("chat_dict_regex_error", labels={"key": "compilation_phase"})
            logging.error(f"Invalid regex pattern during initial matching. Error: {str(e)}")
            matched_entries = []
        except Exception as e_match:
            log_counter("chat_dict_match_error")
            logging.error(f"Error during initial matching: {str(e_match)}", exc_info=True)
            matched_entries = []

        matched_snapshot = list(matched_entries)                     # ADDED
        logging.debug(f"Matched entries after initial filtering: {[e.raw_key for e in matched_entries]}")

        # 2. Group scoring (verbatim try/except, with a before-list diff)
        stage_before = list(matched_entries)                          # ADDED
        try:
            logging.debug(f"Chat Dictionary: Applying group scoring for {len(matched_entries)} entries")
            matched_entries = group_scoring(matched_entries)
        except Exception as e_gs:
            log_counter("chat_dict_group_scoring_error")
            logging.error(f"Error in group scoring: {str(e_gs)}")
            matched_entries = []
        _record_stage_drops(stage_before, matched_entries, "group_scoring")   # ADDED

        # 3. Probability filter (same pattern)
        stage_before = list(matched_entries)                          # ADDED
        try:
            logging.debug(f"Chat Dictionary: Filtering by probability for {len(matched_entries)} entries")
            matched_entries = filter_by_probability(matched_entries)
        except Exception as e_prob:
            log_counter("chat_dict_probability_error")
            logging.error(f"Error in probability filtering: {str(e_prob)}")
            matched_entries = []
        _record_stage_drops(stage_before, matched_entries, "probability")     # ADDED

        # 4. Timed effects (same pattern around the original loop)
        stage_before = list(matched_entries)                          # ADDED
        active_timed_entries = []
        try:
            logging.debug("Chat Dictionary: Applying timed effects")
            for entry in matched_entries:
                if apply_timed_effects(entry, current_time):
                    active_timed_entries.append(entry)
            matched_entries = active_timed_entries
        except Exception as e_time:
            log_counter("chat_dict_timed_effects_error")
            logging.error(f"Error applying timed effects: {str(e_time)}")
            matched_entries = []
        _record_stage_drops(stage_before, matched_entries, "timed_effects")   # ADDED

        # 5. Token budget (same pattern; truncation drives budget_exceeded)
        stage_before = list(matched_entries)                          # ADDED
        try:
            logging.debug(f"Chat Dictionary: Enforcing token budget for {len(matched_entries)} entries")
            matched_entries = enforce_token_budget(matched_entries, max_tokens)
        except TokenBudgetExceededWarning as e:
            log_counter("chat_dict_token_limit")
            logging.warning(str(e))
            matched_entries = []
        except Exception as e_budget:
            log_counter("chat_dict_token_budget_error")
            logging.error(f"Error enforcing token budget: {str(e_budget)}")
            matched_entries = []
        _record_stage_drops(stage_before, matched_entries, "token_budget")    # ADDED
        budget_survivor_ids = {id(e) for e in matched_entries}               # ADDED
        diagnostics.budget_exceeded = len(matched_entries) != len(stage_before)  # ADDED

        # Alert (dead code in practice — preserved verbatim, not used for diagnostics)
        try:
            alert_token_budget_exceeded(matched_entries, max_tokens)
        except Exception as e_alert:
            log_counter("chat_dict_token_alert_error")
            logging.error(f"Error in token budget alert: {str(e_alert)}")

        # 6. Strategy sort (sort-only; drops are only possible via its except)
        stage_before = list(matched_entries)                          # ADDED
        try:
            logging.debug("Chat Dictionary: Applying replacement strategy")
            matched_entries = apply_strategy(matched_entries, strategy)
        except Exception as e_strategy:
            log_counter("chat_dict_strategy_error")
            logging.error(f"Error applying strategy: {str(e_strategy)}")
            matched_entries = []
        _record_stage_drops(stage_before, matched_entries, "strategy_error")  # ADDED (defensive)

        # 7. Replacements (verbatim loop + order/count recording)
        for applied_position, entry in enumerate(matched_entries):    # ADDED enumerate
            applied_order_by_id[id(entry)] = applied_position         # ADDED
            try:
                logging.debug("Chat Dictionary: Applying replacements")
                replacements_done_for_this_entry = 0
                current_max_replacements = entry.max_replacements
                while current_max_replacements > 0:
                    temp_user_input, replaced_count = apply_replacement_once(temp_user_input, entry)
                    if replaced_count > 0:
                        replacements_done_for_this_entry += 1
                        current_max_replacements -= 1
                        entry.last_triggered = current_time
                    else:
                        break
                if replacements_done_for_this_entry > 0:
                    logging.debug(f"Replaced {replacements_done_for_this_entry} occurrences of '{entry.raw_key}'")
                replacements_by_id[id(entry)] = replacements_done_for_this_entry  # ADDED
            except Exception as e_replace:
                log_counter("chat_dict_replacement_error", labels={"key": entry.raw_key})
                logging.error(f"Error applying replacement for entry {entry.raw_key}: {str(e_replace)}", exc_info=True)
                continue

    except Exception as e_crit:
        log_counter("chat_dict_processing_error")
        logging.error(f"Critical error in process_user_input: {str(e_crit)}", exc_info=True)
        _finalize()                                                   # ADDED
        return original_input_for_fallback, diagnostics               # CHANGED (tuple)

    _finalize()                                                       # ADDED
    return temp_user_input, diagnostics                               # CHANGED (tuple)
```

**Implementer note on `_finalize`:** the plan's inline sketch above contains one placeholder line flagged by the adjacent NOTE — implement `tokens_used` exactly as the NOTE shows (`sum(calculate_token_usage([c]) for c in matched_snapshot if id(c) in budget_survivor_ids)`), reading `budget_survivor_ids` from the enclosing scope. Everything else in `_finalize` is literal.

(d) Replace `process_user_input`'s body (keep its docstring, appending one line: `Diagnostics-aware callers should use process_user_input_with_diagnostics.`):

```python
def process_user_input(
    user_input: str,
    entries: List[ChatDictionary],
    max_tokens: int = 5000,
    strategy: str = "sorted_evenly"
) -> str:
    processed_text, _diagnostics = process_user_input_with_diagnostics(
        user_input, entries, max_tokens=max_tokens, strategy=strategy
    )
    return processed_text
```

- [ ] **Step 4: Run the new test file — expect all PASS.** Then run the chat-path pins: `Tests/Chat/test_chat_functions.py Tests/Character_Chat/test_local_chat_dictionary_service.py Tests/Character_Chat/test_chat_dictionary_scope_service.py` — expect all PASS (wrapper behavior identical).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py Tests/Character_Chat/test_chat_dictionary_lib_diagnostics.py
git commit -m "feat(chat-dictionaries): additive diagnostics path in the substitution engine

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Service — additive `diagnostics` key with append-time id enrichment

**Files:**
- Modify: `tldw_chatbook/Character_Chat/local_chat_dictionary_service.py` (`process_text`, currently the block ending in the four-key return)
- Test: `Tests/Character_Chat/test_local_chat_dictionary_service.py`

**Interfaces:**
- Consumes: Task 1's `process_user_input_with_diagnostics` + `DictionaryProcessDiagnostics.to_dict()`.
- Produces: `process_text` response gains `"diagnostics": {...}` whose entry records each gain `"entry_id": "local:chat_dictionary_entry:<dict_id>:<stored_index>"`. Existing keys (`text`, `processed_text`, `dictionary_id`, `source`) byte-identical. Assembly failure omits the key.

- [ ] **Step 1: Write the failing tests** (append to `test_local_chat_dictionary_service.py`):

```python
def _create_two_entry_dictionary(service, *, name="Diagnose Me"):
    return service.create_dictionary(
        {
            "name": name,
            "entries": [
                {"pattern": "BP", "replacement": "blood pressure"},
                {"pattern": "HR", "replacement": "heart rate", "group": "vitals"},
            ],
        }
    )


def test_process_text_carries_enriched_diagnostics(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    created = _create_two_entry_dictionary(service)

    response = service.process_text({"text": "BP and HR", "dictionary_id": created["id"]})

    # Existing keys byte-identical in name and meaning.
    assert response["text"] == "BP and HR"
    assert response["processed_text"] == "blood pressure and heart rate"
    assert response["dictionary_id"] == created["id"]
    assert response["source"] == "local"

    diagnostics = response["diagnostics"]
    assert diagnostics["matched"] == 2 and diagnostics["fired"] == 2
    ids = {r["pattern"]: r["entry_id"] for r in diagnostics["entries"]}
    assert ids["BP"] == f"local:chat_dictionary_entry:{created['id']}:0"
    assert ids["HR"] == f"local:chat_dictionary_entry:{created['id']}:1"


def test_process_text_group_filter_keeps_ids_correct(dictionary_db):
    # The group filter drops entry 0 BEFORE the engine runs; append-time
    # tracking must still map the surviving entry to stored index 1.
    service = LocalChatDictionaryService(dictionary_db)
    created = _create_two_entry_dictionary(service, name="Grouped")

    response = service.process_text(
        {"text": "BP and HR", "dictionary_id": created["id"], "group": "vitals"}
    )

    records = response["diagnostics"]["entries"]
    assert [r["pattern"] for r in records] == ["HR"]
    assert records[0]["entry_id"] == f"local:chat_dictionary_entry:{created['id']}:1"
    assert records[0]["input_index"] == 0  # engine saw a 1-element list


def test_process_text_all_dictionaries_path_ids_carry_own_dict(dictionary_db):
    # dictionary_id=None concatenates entries across ALL dictionaries; each
    # record's entry_id must carry its OWN dictionary's id + stored index.
    service = LocalChatDictionaryService(dictionary_db)
    first = _create_two_entry_dictionary(service, name="First")
    second = service.create_dictionary(
        {"name": "Second", "entries": [{"pattern": "RR", "replacement": "respiratory rate"}]}
    )

    response = service.process_text({"text": "BP and RR"})

    ids = {r["pattern"]: r["entry_id"] for r in response["diagnostics"]["entries"]}
    assert ids["BP"] == f"local:chat_dictionary_entry:{first['id']}:0"
    assert ids["RR"] == f"local:chat_dictionary_entry:{second['id']}:0"
    assert response["dictionary_id"] is None


def test_process_text_omits_diagnostics_on_assembly_failure(dictionary_db, monkeypatch):
    service = LocalChatDictionaryService(dictionary_db)
    created = _create_two_entry_dictionary(service, name="Degrade")

    import tldw_chatbook.Character_Chat.Chat_Dictionary_Lib as cdl_module

    monkeypatch.setattr(
        cdl_module.DictionaryProcessDiagnostics,
        "to_dict",
        lambda self: (_ for _ in ()).throw(RuntimeError("assembly boom")),
    )
    response = service.process_text({"text": "BP now", "dictionary_id": created["id"]})
    assert response["processed_text"] == "blood pressure now"
    assert "diagnostics" not in response
```

- [ ] **Step 2: Run — expect FAIL** (`KeyError: 'diagnostics'`).

- [ ] **Step 3: Implement.** Replace the entry-collection loop and return in `process_text` (the current body builds `entries` + `strategy` then returns four keys):

```python
        entries: list[cdl.ChatDictionary] = []
        entry_ids: list[str] = []
        strategy = "sorted_evenly"
        for dictionary in dictionaries:
            strategy = dictionary.get("strategy") or strategy
            record_id = int(dictionary.get("id") or 0)
            for stored_index, entry in enumerate(dictionary.get("entries") or []):
                if group is not None and entry.group != group:
                    continue
                entries.append(entry)
                # Append-time id tracking: input_index == len-1 at append time,
                # correct under the group filter and the all-dictionaries path.
                entry_ids.append(f"local:chat_dictionary_entry:{record_id}:{stored_index}")
        processed_text, diagnostics = cdl.process_user_input_with_diagnostics(
            text, entries, max_tokens=token_budget, strategy=strategy
        )
        response = {
            "text": text,
            "processed_text": processed_text,
            "dictionary_id": dictionary_id,
            "source": "local",
        }
        try:
            diagnostics_payload = diagnostics.to_dict()
            for record in diagnostics_payload.get("entries") or []:
                input_index = record.get("input_index")
                if isinstance(input_index, int) and 0 <= input_index < len(entry_ids):
                    record["entry_id"] = entry_ids[input_index]
            response["diagnostics"] = diagnostics_payload
        except Exception:
            logger.opt(exception=True).warning(
                "Chat dictionary diagnostics assembly failed; returning substitution only."
            )
        return response
```

(Verify `logger` is imported in this module — it uses loguru elsewhere; if the module uses a different logging handle, match it.)

- [ ] **Step 4: Run the service file — all PASS** (new + existing).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/local_chat_dictionary_service.py Tests/Character_Chat/test_local_chat_dictionary_service.py
git commit -m "feat(chat-dictionaries): process_text returns enriched diagnostics (additive key)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Try-it renders the story (widget + screen pass-through + fake)

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_tryit.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (`_handle_dictionary_tryit_run`'s final `render_result` call only)
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: Task 2's response shape (`response["diagnostics"]` present or absent).
- Produces: `render_result(original: str, processed: str, diagnostics: dict | None = None)` (additive default — P1a callers/tests unchanged); DOM ids `#personas-dict-tryit-summary`, `#personas-dict-tryit-details` (scroll container), `#personas-dict-tryit-fired`, `#personas-dict-tryit-nearmiss`.

- [ ] **Step 1: Update the fake to mirror the REAL shape, then write failing tests.**

In `Tests/UI/test_personas_dictionaries.py`, replace `FakeDictScopeService.process_text` with a version that emits the exact `to_dict()`+enrichment shape (mini walk-and-stop for budget; probability 0 marks a skip; literal-only like before). Add a class attribute `emit_diagnostics = True` for the degrade test:

```python
    emit_diagnostics = True

    async def process_text(self, request_data: Any, mode: str = "local") -> dict:
        payload = dict(request_data)
        text = payload["text"]
        record = self.records.get(int(payload.get("dictionary_id") or 0))
        if record is None:
            raise ValueError("Local chat dictionary was not found.")
        token_budget = int(payload.get("token_budget") or 5000)
        processed = text
        diag_entries = []
        tokens_used = 0
        budget_exceeded = False
        applied_order = 0
        for stored_index, entry in enumerate(record["entries"]):
            pattern = entry.get("pattern") or ""
            if entry.get("type") == "regex" or not pattern or pattern not in text:
                continue  # never-matched entries are omitted (real shape)
            token_cost = len((entry.get("replacement") or "").split())
            base = {
                "input_index": stored_index,
                "pattern": pattern,
                "replacements": 0,
                "token_cost": token_cost,
                "applied_order": None,
                "content_preview": (entry.get("replacement") or "")[:40],
                "entry_id": f"local:chat_dictionary_entry:{record['id']}:{stored_index}",
            }
            if float(entry.get("probability") or 1.0) == 0.0:
                diag_entries.append({**base, "status": "skipped:probability"})
                continue
            if tokens_used + token_cost > token_budget:
                budget_exceeded = True
                diag_entries.append({**base, "status": "skipped:token_budget"})
                continue
            tokens_used += token_cost
            count = processed.count(pattern)
            processed = processed.replace(pattern, entry.get("replacement") or "")
            diag_entries.append(
                {**base, "status": "fired", "replacements": count, "applied_order": applied_order}
            )
            applied_order += 1
        self.calls.append(("process", text))
        response = {
            "text": text,
            "processed_text": processed,
            "dictionary_id": record["id"],
            "source": "local",
        }
        if self.emit_diagnostics:
            fired = sum(1 for r in diag_entries if r["status"] == "fired")
            response["diagnostics"] = {
                "entries": diag_entries,
                "matched": len(diag_entries),
                "fired": fired,
                "skipped": len(diag_entries) - fired,
                "total_replacements": sum(r["replacements"] for r in diag_entries),
                "tokens_used": tokens_used,
                "token_budget": token_budget,
                "budget_exceeded": budget_exceeded,
            }
        return response
```

Then append the tests (to `TestDictionaryTryIt`; reuse its `_select_first` helper and the `size=(200, 60)` pattern):

```python
    async def test_tryit_renders_summary_and_fired_lines(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "check BP now"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            summary = str(screen.query_one("#personas-dict-tryit-summary", Static).renderable)
            assert "1 fired" in summary and "0 skipped" in summary
            assert "/1000 tokens" in summary  # dict max_tokens=1000 rode along
            fired = str(screen.query_one("#personas-dict-tryit-fired", Static).renderable)
            assert "BP" in fired and "blood pressure" in fired and "×1" in fired

    async def test_tryit_renders_near_miss_with_reason(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        fake_dict_service.records[1]["entries"][1]["probability"] = 0.0  # HR never fires
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "BP and HR"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            nearmiss = str(screen.query_one("#personas-dict-tryit-nearmiss", Static).renderable)
            assert "HR" in nearmiss and "probability roll" in nearmiss
            summary = str(screen.query_one("#personas-dict-tryit-summary", Static).renderable)
            assert "1 fired" in summary and "1 skipped" in summary

    async def test_tryit_budget_flag_shows(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        fake_dict_service.records[1]["max_tokens"] = 2  # both entries cost 2 -> second is dropped
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "BP and HR"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            summary = str(screen.query_one("#personas-dict-tryit-summary", Static).renderable)
            assert "over budget" in summary
            nearmiss = str(screen.query_one("#personas-dict-tryit-nearmiss", Static).renderable)
            assert "token budget" in nearmiss

    async def test_tryit_degrades_without_diagnostics(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        fake_dict_service.emit_diagnostics = False
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "check BP now"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            # Diff still renders (P1a behavior)...
            processed = str(screen.query_one("#personas-dict-tryit-processed", Static).renderable)
            assert "blood pressure" in processed
            # ...and the summary carries the honest unavailable note.
            summary = str(screen.query_one("#personas-dict-tryit-summary", Static).renderable)
            assert "diagnostics unavailable" in summary
```

- [ ] **Step 2: Run — expect FAIL** (`NoMatches: #personas-dict-tryit-summary`).

- [ ] **Step 3: Implement the widget.** In `personas_dictionary_tryit.py`:

(a) Compose — after the `#personas-dict-tryit-processed` Static:

```python
        yield Static("", id="personas-dict-tryit-summary", markup=False)
        with Vertical(id="personas-dict-tryit-details"):
            yield Static("", id="personas-dict-tryit-fired")
            yield Static("", id="personas-dict-tryit-nearmiss")
```

(b) `DEFAULT_CSS` additions (structure-only; the scroll container is the P1a clipping lesson):

```css
    PersonasDictionaryTryItWidget #personas-dict-tryit-summary {
        height: 1;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-details {
        height: auto;
        max-height: 12;
        overflow-y: auto;
    }
    PersonasDictionaryTryItWidget #personas-dict-tryit-fired,
    PersonasDictionaryTryItWidget #personas-dict-tryit-nearmiss {
        height: auto;
    }
```

(c) Reason copy map + renderer (module-level dict, then methods):

```python
_SKIP_REASON_COPY = {
    "skipped:group_scoring": "skipped: lost group scoring",
    "skipped:probability": "skipped: probability roll — re-running may differ",
    "skipped:timed_effects": "skipped: cooldown or delay",
    "skipped:token_budget": "skipped: token budget",
    "no_replacement": "no replacement — text changed by an earlier entry",
}
```

```python
    def render_result(
        self, original: str, processed: str, diagnostics: dict | None = None
    ) -> None:
        """Renders the word-diff and, when available, the diagnostics story.

        Args:
            original: The sample text before substitution.
            processed: The sample text after substitution.
            diagnostics: The service's diagnostics dict, or None to degrade
                to the diff-only view with an "unavailable" note.
        """
        left, right = word_diff(original, processed)
        status = self.query_one("#personas-dict-tryit-status", Static)
        if original == processed:
            status.update("No differences - no entry changed the sample.")
        else:
            status.update("Changed spans highlighted below.")
        self.query_one("#personas-dict-tryit-original", Static).update(left)
        self.query_one("#personas-dict-tryit-processed", Static).update(right)
        self._render_diagnostics(diagnostics)

    def _render_diagnostics(self, diagnostics: dict | None) -> None:
        summary = self.query_one("#personas-dict-tryit-summary", Static)
        fired_area = self.query_one("#personas-dict-tryit-fired", Static)
        nearmiss_area = self.query_one("#personas-dict-tryit-nearmiss", Static)
        if not isinstance(diagnostics, dict):
            summary.update(Text("diagnostics unavailable", style="dim"))
            fired_area.update("")
            nearmiss_area.update("")
            return
        # Each section guards independently: render what parses, skip what doesn't.
        try:
            line = (
                f"{int(diagnostics.get('fired') or 0)} fired · "
                f"{int(diagnostics.get('skipped') or 0)} skipped · "
                f"{int(diagnostics.get('tokens_used') or 0)}"
                f"/{int(diagnostics.get('token_budget') or 0)} tokens"
            )
            text = Text(line)
            if diagnostics.get("budget_exceeded"):
                text.append(" · over budget", style="bold")
            summary.update(text)
        except Exception:
            summary.update(Text("diagnostics unavailable", style="dim"))
        records = diagnostics.get("entries")
        records = records if isinstance(records, list) else []
        try:
            fired = sorted(
                (r for r in records if r.get("status") == "fired"),
                key=lambda r: (r.get("applied_order") is None, r.get("applied_order") or 0),
            )
            fired_text = Text()
            for record in fired:
                fired_text.append(
                    f"{record.get('pattern')} → {record.get('content_preview')}"
                    f" · ×{int(record.get('replacements') or 0)}"
                    f" · {int(record.get('token_cost') or 0)} tok\n"
                )
            fired_area.update(fired_text)
        except Exception:
            fired_area.update("")
        try:
            misses = sorted(
                (r for r in records if r.get("status") != "fired"),
                key=lambda r: int(r.get("input_index") or 0),
            )
            miss_text = Text()
            for record in misses:
                reason = _SKIP_REASON_COPY.get(str(record.get("status")), str(record.get("status")))
                miss_text.append(f"{record.get('pattern')} — {reason}\n", style="dim")
            nearmiss_area.update(miss_text)
        except Exception:
            nearmiss_area.update("")
```

(d) Screen — in `_handle_dictionary_tryit_run`, the final call becomes:

```python
        tryit.render_result(
            str(response.get("text") or message.text),
            str(response.get("processed_text") or ""),
            response.get("diagnostics"),
        )
```

- [ ] **Step 4: Run the whole UI file — all PASS** (the four new tests + every prior test; the P1a Try-it tests exercise the default-None path).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_tryit.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_dictionaries.py
git commit -m "feat(personas): Try-it renders fired entries, near-misses, and the budget story

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Full gate + spec status + spec consistency touch-up

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-14-roleplay-p1b-tryit-diagnostics-design.md` (status line + two consistency touch-ups)

- [ ] **Step 1: Full gate**

```
HOME=... .venv/bin/python -m pytest \
  Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_workbench.py \
  Tests/Character_Chat/ Tests/Chat/test_chat_functions.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

Expected: all pass (report exact counts). Then import smoke on the three touched modules.

- [ ] **Step 2: Spec touch-ups** (documenting two implementation-level details the plan added):
  - AC2's per-entry field list: add `content_preview` (first 40 chars of the entry's content — AC4's fired line needs it).
  - AC2's status enum: add the defensive `skipped:strategy_error` (only reachable if `apply_strategy` raises; UI falls back to rendering the raw status string).
  - Status line: `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented (P1b).`

- [ ] **Step 3: Commit**

```bash
git add Docs/superpowers/specs/2026-07-14-roleplay-p1b-tryit-diagnostics-design.md
git commit -m "docs(roleplay): mark P1b Try-it-diagnostics spec implemented (+field-list touch-ups)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
