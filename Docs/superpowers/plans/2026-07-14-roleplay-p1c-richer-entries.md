# Roleplay P1c — Richer Entries + Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-entry `enabled` / `case_sensitive` / `priority` end-to-end (model → engine → service → form), the unified priority ordering that fixes the budget-before-strategy quirk, and a warn-not-block validation panel with jump-to-entry.

**Architecture:** Additive model fields with name-aligned round-trips; two helper amendments (conditional case flags, priority-first group winner); one pipeline reorder in `process_user_input_with_diagnostics` (disabled filter after match; strategy+priority sort moved BEFORE budget; post-budget sort removed); a pure validation module probing through the real parser; widget-owned validation panel + the AC5b scroll container paying the P1a clipping debt.

**Tech Stack:** Python ≥3.11, Textual (`OptionList`, `VerticalScroll`), pytest.

**Spec:** `Docs/superpowers/specs/2026-07-14-roleplay-p1c-richer-entries-design.md` (committed `471f6c40`) — its "Ground truths" and ACs are binding; read once before Task 1.

## Global Constraints

- **Legacy equivalence (AC3):** for data with all `priority=0`, all `enabled=True`, all `case_sensitive=False` — matching, filtering, group winners, and application order are byte-identical to pre-P1c. The ONLY legacy delta: under token-budget pressure, survival follows the strategy order instead of stored order (the spec'd quirk fix). Pinned by dedicated tests.
- **Name-aligned fields:** `enabled` / `case_sensitive` / `priority` use the SAME key in stored dicts and API payloads/responses. `_entry_from_payload` reads them with plain `.get(name, default)` — no dual-name fallback.
- **`case_sensitive` affects LITERAL keys only** (both `match_whole_words` and `apply_replacement_once`); regex keys keep deriving case from their own `/pat/i` flags. `ChatDictionary.matches()` is dead code — do NOT touch it.
- **Group winner = `max(entries, key=lambda e: (e.priority, len(raw_key)))`** — priority first, length tie-break (legacy: all 0 → length, unchanged).
- **Unified ordering:** strategy sort, then STABLE sort by `-priority`, both BEFORE the budget walk; no post-budget re-sort. Diagnostics stage-diffs follow the new sequence; status enum gains `skipped:disabled`.
- **Validation is warn-not-block, wholly Entries-tab-local** (inspector untouched); findings probe through `_entry_from_payload` → `ChatDictionary.is_regex` (never re-implement slash/wrap parsing).
- **Duplicate payload carries all NINE entry fields**; its round-trip test asserts every field.
- **AC5b:** the Entries `TabPane` content gets a scrollable container; a geometry test proves the button row + validation list are reachable at a constrained height.
- Only Personas-owned + Character_Chat dictionary files change. No DB schema bump. Widget CSS structure-only. Google docstrings on new public callables. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (isolated HOME; from this worktree use the main checkout's absolute `.venv/bin/python`):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    .venv/bin/python -m pytest <target> \
    -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
  ```

---

### Task 1: Model + service seam — the three fields round-trip

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` (`ChatDictionary.__init__` ~:67, `to_dict` ~:136, `from_dict` ~:147)
- Modify: `tldw_chatbook/Character_Chat/local_chat_dictionary_service.py` (`_entry_from_payload`, `_entry_to_response`)
- Test: `Tests/Character_Chat/test_chat_dictionary_lib_entries.py` (create), `Tests/Character_Chat/test_local_chat_dictionary_service.py`

**Interfaces:**
- Produces: `ChatDictionary(key, content, probability=100, group=None, timed_effects=None, max_replacements=1, enabled=True, case_sensitive=False, priority=0)`; `to_dict()` gains the three keys; `from_dict` reads them with defaults. `_entry_to_response` emits real `enabled` + `case_sensitive` + `priority` (int); `_entry_from_payload` consumes them. Tasks 2-6 rely on the attribute names exactly.

- [ ] **Step 1: Write the failing tests.** Create `Tests/Character_Chat/test_chat_dictionary_lib_entries.py`:

```python
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
```

Append to `Tests/Character_Chat/test_local_chat_dictionary_service.py`:

```python
def test_entry_new_fields_roundtrip_through_service(dictionary_db):
    service = LocalChatDictionaryService(dictionary_db)
    created = service.create_dictionary(
        {
            "name": "Fields",
            "entries": [
                {"pattern": "BP", "replacement": "blood pressure",
                 "enabled": False, "case_sensitive": True, "priority": 9},
            ],
        }
    )
    record = service.get_dictionary(created["id"])
    entry = record["entries"][0]
    assert entry["enabled"] is False          # no longer hardcoded True
    assert entry["case_sensitive"] is True
    assert entry["priority"] == 9
    # Partial update touching only the replacement preserves the three fields.
    updated = service.update_entry(entry["id"], {"replacement": "arterial pressure"})
    assert (updated["enabled"], updated["case_sensitive"], updated["priority"]) == (False, True, 9)
```

- [ ] **Step 2: Run — expect FAIL** (`TypeError: unexpected keyword 'enabled'` / `assert entry["enabled"] is False` fails on hardcoded True).

- [ ] **Step 3: Implement.**

(a) `ChatDictionary.__init__` signature gains `, enabled: bool = True, case_sensitive: bool = False, priority: int = 0` and the body sets `self.enabled = bool(enabled)`, `self.case_sensitive = bool(case_sensitive)`, `self.priority = int(priority)` (after the existing assignments).

(b) `to_dict()` adds:

```python
            'enabled': self.enabled,
            'case_sensitive': self.case_sensitive,
            'priority': self.priority,
```

(c) `from_dict()` passes `enabled=data.get('enabled', True), case_sensitive=data.get('case_sensitive', False), priority=data.get('priority', 0)`.

(d) `_entry_from_payload` (service) passes the same three via `data.get("enabled", True)`, `data.get("case_sensitive", False)`, `int(data.get("priority", 0) or 0)` into the constructor.

(e) `_entry_to_response` replaces `"enabled": True,` with `"enabled": entry.enabled,` and adds `"case_sensitive": entry.case_sensitive,` and `"priority": entry.priority,`.

- [ ] **Step 4: Run both test files — PASS.** Also run `Tests/Character_Chat/test_chat_dictionary_scope_service.py` (regression).

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): per-entry enabled/case_sensitive/priority round-trip (model + seam)` + trailer.

---

### Task 2: Helpers — conditional case flags + priority-first group winner

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` (`match_whole_words` ~:455, `apply_replacement_once` ~:498, `group_scoring` ~:335)
- Test: `Tests/Character_Chat/test_chat_dictionary_lib_entries.py`

**Interfaces:**
- Consumes: Task 1's attributes. Produces: literal-key matching/replacement honor `entry.case_sensitive`; group winner = `max(key=lambda e: (e.priority, len(str(e.raw_key)) if e.raw_key else 0))`.

- [ ] **Step 1: Failing tests** (append):

```python
from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
    apply_replacement_once,
    group_scoring,
    match_whole_words,
)


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
```

- [ ] **Step 2: Run — expect FAIL** (case-sensitive still matches lowercase; priority loser wins group).

- [ ] **Step 3: Implement.**

(a) `match_whole_words` literal branch — replace the hardcoded-flag search:

```python
        elif isinstance(entry.key, str): # Plain string key
            # Ensure whole word match for plain strings; case per entry.case_sensitive
            flags = 0 if getattr(entry, "case_sensitive", False) else re.IGNORECASE
            if re.search(rf'\b{re.escape(entry.key)}\b', text, flags):
```

(b) `apply_replacement_once` literal branch:

```python
    else: # Plain string key
        flags = 0 if getattr(entry, "case_sensitive", False) else re.IGNORECASE
        pattern = re.compile(rf'\b{re.escape(str(entry.key))}\b', flags) # Ensure entry.key is str
```

(`getattr` defaults keep any hand-constructed legacy object safe.)

(c) `group_scoring` winner line:

```python
            best_entry_in_group = max(
                group_entries_list,
                key=lambda e: (getattr(e, "priority", 0), len(str(e.raw_key)) if e.raw_key else 0),
            )
```

- [ ] **Step 4: Run the entries test file — PASS.** Then run `Tests/Character_Chat/test_chat_dictionary_lib_diagnostics.py` + `Tests/Chat/test_chat_functions.py` (legacy behavior pins — all defaults reproduce old outcomes).

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): per-entry case-sensitivity (both literal sites) + priority-first group winner` + trailer.

---

### Task 3: Pipeline — disabled filter + unified ordering + preview flatten

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py` (`process_user_input_with_diagnostics` — current stage anchors: budget block `:719-733`, strategy block `:743-751`, replacement loop `:754+`, `_finalize` content_preview line)
- Test: `Tests/Character_Chat/test_chat_dictionary_lib_entries.py`, `Tests/Character_Chat/test_chat_dictionary_lib_diagnostics.py`

**Interfaces:**
- Consumes: Tasks 1-2. Produces the NEW pipeline order: match → disabled filter (`skipped:disabled`) → group → probability → timed → strategy sort + stable `-priority` sort (one "ordering" stage, defensive diff `skipped:strategy_error`) → budget (walk-and-stop; `budget_exceeded` inside the try as today) → replacements. `content_preview` becomes whitespace-flattened. Status enum consumers (Task 6 fake/UI) rely on `skipped:disabled`.

- [ ] **Step 1: Failing tests.** Append to the entries test file:

```python
from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import (
    process_user_input,
    process_user_input_with_diagnostics,
)


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
```

- [ ] **Step 2: Run — expect FAIL** (`skipped:disabled` unknown; priority ignored; preview unflattened).

- [ ] **Step 3: Implement** in `process_user_input_with_diagnostics` (every pre-existing statement keeps its exact text; this task MOVES blocks and ADDS marked lines):

(a) **Disabled filter** — insert immediately after `matched_snapshot = list(matched_entries)` and its logging line:

```python
        # P1c: disabled entries stay visible as near-misses (filtered after match).
        stage_before = list(matched_entries)                          # ADDED
        matched_entries = [e for e in matched_entries if getattr(e, "enabled", True)]  # ADDED
        _record_stage_drops(stage_before, matched_entries, "disabled")               # ADDED
```

(b) **Unified ordering** — MOVE the entire strategy block (currently `:743-751`: `stage_before = list(...)` / try `apply_strategy` / except / `_record_stage_drops(..., "strategy_error")`) to sit immediately BEFORE the budget block (`:719`), keeping its text verbatim, and append one ADDED line after it:

```python
        matched_entries.sort(key=lambda e: -int(getattr(e, "priority", 0) or 0))  # ADDED: stable — strategy order breaks ties
```

(c) **Budget block** stays textually as-is (it now walks the unified order); the replacement loop follows it directly — confirm no second `apply_strategy` call remains anywhere after the budget block.

(d) **`_finalize`** — `content_preview` line becomes:

```python
                    content_preview=" ".join(str(candidate.content or "").split())[:40],
```

(e) The pipeline docstring's stage list updates to the new order (keep Google style).

- [ ] **Step 4: Run** the entries file + the FULL diagnostics file + `Tests/Chat/test_chat_functions.py` + `Tests/Character_Chat/test_local_chat_dictionary_service.py`. Expected: all pass — the P1b fixtures happen to be alphabetical-equals-stored so their pins hold; if any diagnostics test fails, inspect whether it pinned STORED-order budget survival (the spec'd delta) and update ONLY such assertions, documenting each in the report.

- [ ] **Step 5: Commit** — `feat(chat-dictionaries): unified priority ordering + disabled near-misses (spec'd quirk fix)` + trailer.

---

### Task 4: Validation module

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_validation.py`
- Test: `Tests/UI/test_personas_dictionary_validation.py` (create — pure unit tests, no Textual app)

**Interfaces:**
- Produces: `@dataclass ValidationFinding(code: str, field: str, message: str, entry_id: str | None)`; `validate_entries(entries: list[dict]) -> list[ValidationFinding]` over API-named entry dicts (as `get_dictionary` returns). Codes exactly: `invalid_regex`, `duplicate_pattern`, `probability_zero`, `case_flag_on_regex`. Task 6 renders `[{code}] {pattern} — {message}`.

- [ ] **Step 1: Failing tests.** Create the test file:

```python
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
```

- [ ] **Step 2: Run — expect FAIL** (module missing).

- [ ] **Step 3: Implement:**

```python
"""Warn-not-block validation for dictionary entries (Roleplay P1c).

Pure functions over API-named entry dicts. The regex probe goes through the
real parser chain (``_entry_from_payload`` -> ``ChatDictionary``) so the
wrap/slash/flag rules can never drift from the engine's.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...Character_Chat.local_chat_dictionary_service import _entry_from_payload


@dataclass(frozen=True)
class ValidationFinding:
    """One advisory finding about a dictionary entry.

    Args:
        code: Stable machine code (invalid_regex, duplicate_pattern,
            probability_zero, case_flag_on_regex).
        field: The entry field the finding is about.
        message: Human-readable explanation.
        entry_id: The positional entry id, or None when unavailable.
    """

    code: str
    field: str
    message: str
    entry_id: str | None


def validate_entries(entries: list[dict]) -> list[ValidationFinding]:
    """Returns advisory findings for a dictionary's entries.

    Args:
        entries: API-named entry dicts (as ``get_dictionary`` returns).

    Returns:
        Findings in entry order; empty when everything is clean.
    """
    findings: list[ValidationFinding] = []
    seen: dict[tuple[str, str], str | None] = {}
    for entry in entries:
        entry_id = entry.get("id")
        pattern = str(entry.get("pattern") or "")
        etype = str(entry.get("type") or "literal")

        if etype == "regex":
            probe = _entry_from_payload(entry)
            if not probe.is_regex:
                findings.append(ValidationFinding(
                    code="invalid_regex", field="pattern", entry_id=entry_id,
                    message="Pattern does not compile; the engine will treat it as a literal.",
                ))
            if entry.get("case_sensitive"):
                findings.append(ValidationFinding(
                    code="case_flag_on_regex", field="case_sensitive", entry_id=entry_id,
                    message="Case-sensitive is ignored for regex entries; use the /i flag instead.",
                ))

        key = (pattern, etype)
        if key in seen:
            findings.append(ValidationFinding(
                code="duplicate_pattern", field="pattern", entry_id=entry_id,
                message="Same pattern and type as an earlier entry; only one will usually fire.",
            ))
        else:
            seen[key] = entry_id

        probability = entry.get("probability")
        if probability is not None and float(probability) == 0.0:
            findings.append(ValidationFinding(
                code="probability_zero", field="probability", entry_id=entry_id,
                message="Probability 0 means this entry can never fire.",
            ))
    return findings


__all__ = ["ValidationFinding", "validate_entries"]
```

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit** — `feat(personas): dictionary validation module (warn-not-block, real-parser probe)` + trailer.

---

### Task 5: Entries form + table + Duplicate integrity

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py` (compose form rows, `on_mount` columns, `update_entries`, `form_payload`, `_fill_form_from_entry`)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (Duplicate payload `:1661-1672`)
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: Tasks 1's field names. Produces: form ids `#personas-dict-entry-enabled` (Switch, default True), `#personas-dict-entry-case` (Switch, default False), `#personas-dict-entry-priority` (Input, default "0"); `form_payload()` adds `enabled`/`case_sensitive`/`priority` (int; invalid → inline error "Priority must be a whole number."); table columns become `pattern · replacement · type · prob % · group · pri` with disabled rows rendered dim + `off` appended to the pattern cell (`Text` styling, markup-inert).

- [ ] **Step 1: Failing tests** (append to `TestDictionaryEntries` / `TestDictionaryNewDuplicate`; `size=(200, 60)`):

```python
    async def test_form_roundtrips_new_fields(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import Switch, TextArea

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-entry-pattern", Input).value = "ICU"
            screen.query_one("#personas-dict-entry-replacement", TextArea).text = "intensive care"
            screen.query_one("#personas-dict-entry-enabled", Switch).value = False
            screen.query_one("#personas-dict-entry-case", Switch).value = True
            screen.query_one("#personas-dict-entry-priority", Input).value = "5"
            await pilot.click("#personas-dict-entry-add")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            added = fake_dict_service.records[1]["entries"][-1]
            assert (added["enabled"], added["case_sensitive"], added["priority"]) == (False, True, 5)

    async def test_priority_input_validates_integer(self, mock_app_instance, stub_characters, fake_dict_service):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            screen.query_one("#personas-dict-entry-pattern", Input).value = "X"
            screen.query_one("#personas-dict-entry-priority", Input).value = "high"
            detail = screen.query_one("#personas-dictionary-detail")
            assert detail.form_payload() is None
            error = str(screen.query_one("#personas-dict-entry-error", Static).renderable)
            assert "whole number" in error.lower()
```

And in `TestDictionaryNewDuplicate`, EXTEND `test_duplicate_copies_entries_and_strategy` — before the duplicate click add:

```python
        fake_dict_service.records[1]["entries"][0].update(
            {"enabled": False, "case_sensitive": True, "priority": 4}
        )
```

and after locating `copy_rec`, replace the pattern-only assertion with the all-nine-fields check:

```python
            src = fake_dict_service.records[1]["entries"][0]
            dup = copy_rec["entries"][0]
            for field in ("pattern", "replacement", "probability", "group", "timed_effects",
                          "max_replacements", "type", "enabled", "case_sensitive", "priority"):
                assert dup.get(field) == src.get(field), field
```

- [ ] **Step 2: Run — FAIL** (`NoMatches: #personas-dict-entry-enabled`; duplicate drops the fields).

- [ ] **Step 3: Implement.**

(a) Compose — the first form row gains, after the max-repl Input:

```python
                    yield Switch(value=True, id="personas-dict-entry-enabled", tooltip="Entry enabled")
                    yield Switch(value=False, id="personas-dict-entry-case", tooltip="Case-sensitive (literal keys)")
                    yield Input(placeholder="Priority", id="personas-dict-entry-priority", value="0")
```

(b) `on_mount` columns: `table.add_columns("pattern", "replacement", "type", "prob %", "group", "pri")`.

(c) `update_entries` row build adds the `pri` cell and disabled treatment:

```python
        from rich.text import Text  # move to module imports

        for entry in self._entries:
            probability = entry.get("probability")
            prob_pct = round(float(probability if probability is not None else 1.0) * 100)
            enabled = bool(entry.get("enabled", True))
            style = "dim" if not enabled else ""
            pattern_cell = Text(str(entry.get("pattern") or ""), style=style)
            if not enabled:
                pattern_cell.append("  off", style="dim")
            table.add_row(
                pattern_cell,
                Text(str(entry.get("replacement") or ""), style=style),
                Text(str(entry.get("type") or "literal"), style=style),
                Text(str(prob_pct), style=style),
                Text(str(entry.get("group") or ""), style=style),
                Text(str(int(entry.get("priority") or 0)), style=style),
                key=str(entry.get("id")),
            )
```

(This replaces the existing plain-string `add_row` body; the cell CONTENT is unchanged, only wrapped in `Text` so disabled rows dim uniformly. `Text` is imported from `rich.text` at module level.)

(d) `form_payload()` adds (before the return, following the existing numeric-validation pattern):

```python
        raw_priority = self.query_one("#personas-dict-entry-priority", Input).value.strip() or "0"
        try:
            priority = int(raw_priority)
        except ValueError:
            error.update("Priority must be a whole number.")
            return None
```

and the returned dict gains `"enabled": bool(...#personas-dict-entry-enabled Switch value), "case_sensitive": bool(...#personas-dict-entry-case value), "priority": priority`.

(e) `_fill_form_from_entry` fills the three widgets from the entry (`enabled` default True, `case_sensitive` default False, `priority` default 0 → str).

(f) Screen Duplicate payload (`:1661-1672`) adds:

```python
                    "enabled": e.get("enabled", True),
                    "case_sensitive": e.get("case_sensitive", False),
                    "priority": e.get("priority", 0),
```

(g) Test fake: `make_dict_record` entries + `_entry_response` gain the three fields with defaults (`True/False/0`) so shapes stay real (Task 6 adds the SEMANTICS).

- [ ] **Step 4: Run the whole UI file — PASS** (all prior + new).

- [ ] **Step 5: Commit** — `feat(personas): entry form/table carry enabled+case+priority; Duplicate keeps all nine fields` + trailer.

---

### Task 6: Validation panel + AC5b scroll + Try-it reason + fake semantics

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py` (validation OptionList + jump + recompute; Entries tab scroll container)
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_tryit.py` (`_SKIP_REASON_COPY` + `skipped:disabled`)
- Test: `Tests/UI/test_personas_dictionaries.py`

**Interfaces:**
- Consumes: Task 4's `validate_entries`/`ValidationFinding`; Task 5's fields. Produces: `#personas-dict-validation` (OptionList; each option id = the finding's `entry_id`), recomputed in `load_dictionary` and `update_entries`; `OptionList.OptionSelected` moves the entries-table cursor to the row whose key equals the option id. Entries tab content wrapped in `VerticalScroll(id="personas-dict-entries-scroll")`. Fake `process_text` honors disabled (`skipped:disabled`), priority ordering, walk-and-stop `break`.

- [ ] **Step 1: Failing tests** (append):

```python
class TestDictionaryValidationPanel:
    async def _select_first(self, pilot, screen):
        rows = screen.query_one("#personas-library-rows", ListView)
        rows.index = 0
        rows.action_select_cursor()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    async def test_findings_listed_and_jump_moves_cursor(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import DataTable, OptionList

        fake_dict_service.records[1]["entries"].append(
            {"pattern": "BP", "replacement": "dup", "probability": 1.0, "group": None,
             "timed_effects": None, "max_replacements": 1, "type": "literal",
             "enabled": True, "case_sensitive": False, "priority": 0}
        )
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            panel = screen.query_one("#personas-dict-validation", OptionList)
            assert panel.option_count == 1  # the duplicate BP
            panel.highlighted = 0
            panel.action_select()
            await pilot.pause()
            table = screen.query_one("#personas-dict-entries-table", DataTable)
            assert table.cursor_row == 2  # jumped to the duplicate (index 2)

    async def test_panel_clears_on_clean_dictionary(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import OptionList

        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            assert screen.query_one("#personas-dict-validation", OptionList).option_count == 0

    async def test_entries_tab_scrolls_buttons_reachable_when_short(self, mock_app_instance, stub_characters, fake_dict_service):
        """AC5b geometry: at a height that can't fit the whole tab, the scroll
        container exists and the button row is scrollable into view."""
        from textual.containers import VerticalScroll
        from textual.widgets import Button

        app = StyledPersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 34)) as pilot:
            screen = await _enter_dictionaries(pilot)
            await self._select_first(pilot, screen)
            scroll = screen.query_one("#personas-dict-entries-scroll", VerticalScroll)
            add = screen.query_one("#personas-dict-entry-add", Button)
            add.scroll_visible(animate=False)
            await pilot.pause()
            region = add.region
            assert region.height > 0 and region.width > 0  # rendered, reachable


class TestTryItDisabledReason:
    async def test_disabled_entry_renders_reason(self, mock_app_instance, stub_characters, fake_dict_service):
        from textual.widgets import TextArea

        fake_dict_service.records[1]["entries"][1]["enabled"] = False  # HR off
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test(size=(200, 60)) as pilot:
            screen = await _enter_dictionaries(pilot)
            rows = screen.query_one("#personas-library-rows", ListView)
            rows.index = 0
            rows.action_select_cursor()
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            screen.query_one("#personas-dict-tryit-sample", TextArea).text = "BP and HR"
            await pilot.click("#personas-dict-tryit-run")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            nearmiss = str(screen.query_one("#personas-dict-tryit-nearmiss", Static).renderable)
            assert "HR" in nearmiss and "skipped: disabled" in nearmiss
```

- [ ] **Step 2: Run — FAIL** (`NoMatches: #personas-dict-validation` / no disabled semantics in the fake).

- [ ] **Step 3: Implement.**

(a) Detail widget — imports gain `OptionList`, `VerticalScroll` (`from textual.containers import ... VerticalScroll`), `from .personas_dictionary_validation import validate_entries`. Compose: wrap the ENTIRE Entries `TabPane` body in `with VerticalScroll(id="personas-dict-entries-scroll"):` and append after the button row:

```python
                    yield OptionList(id="personas-dict-validation")
```

CSS (structure-only): `#personas-dict-entries-scroll { height: 1fr; }` and `#personas-dict-validation { height: auto; max-height: 5; }`. The DataTable's `1fr` height moves to a fixed `min-height`/`max-height` pair so the scroll container owns the flex (implementer tunes; geometry test is the arbiter).

(b) Recompute — at the end of `update_entries` (which `load_dictionary` already calls):

```python
        self._refresh_validation()
```

```python
    def _refresh_validation(self) -> None:
        """Recompute advisory findings for the current entry list."""
        panel = self.query_one("#personas-dict-validation", OptionList)
        panel.clear_options()
        try:
            findings = validate_entries(self._entries)
        except Exception:
            logger.opt(exception=True).warning("Dictionary validation failed; panel left empty.")
            return
        for finding in findings:
            pattern = next(
                (str(e.get("pattern") or "") for e in self._entries if str(e.get("id")) == str(finding.entry_id)),
                "",
            )
            panel.add_option(Option(f"[{finding.code}] {pattern} — {finding.message}", id=str(finding.entry_id)))
```

Module imports gain `from loguru import logger` and `from textual.widgets.option_list import Option`. Validation must never raise out of the widget.

(c) Jump handler:

```python
    @on(OptionList.OptionSelected, "#personas-dict-validation")
    def _validation_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        entry_id = str(event.option.id or "")
        ids = self.entry_ids_in_order()
        if entry_id in ids:
            table = self.query_one("#personas-dict-entries-table", DataTable)
            table.move_cursor(row=ids.index(entry_id))
            self._fill_form_from_entry(entry_id)
```

(d) Try-it reason map gains `"skipped:disabled": "skipped: disabled",`.

(e) Fake `process_text` semantics (keeping the exact response shape): before the probability check, `if not entry.get("enabled", True): diag_entries.append({**base, "status": "skipped:disabled"}); continue`. Before iterating, order candidates: `ordered = sorted(enumerate(record["entries"]), key=lambda pair: (-int(pair[1].get("priority") or 0), pair[0]))` and iterate that (keeping `stored_index` from the enumerate pair). Budget branch changes from `continue` to walk-and-stop: on the first non-fitting entry mark it `skipped:token_budget`, set `budget_exceeded = True`, and mark every REMAINING candidate `skipped:token_budget` too, then `break`.

- [ ] **Step 4: Run the whole UI file — PASS** (all prior + new; the P1b budget-flag test still passes under walk-and-stop with its two 2-token entries and budget 2).

- [ ] **Step 5: Commit** — `feat(personas): validation panel with jump-to-entry, Entries-tab scroll (AC5b), disabled near-miss reason, faithful fake semantics` + trailer.

---

### Task 7: Full gate + spec status

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-14-roleplay-p1c-richer-entries-design.md` (status line)

- [ ] **Step 1: Full gate**

```
HOME=... .venv/bin/python -m pytest \
  Tests/UI/test_personas_dictionaries.py Tests/UI/test_personas_dictionary_validation.py \
  Tests/UI/test_personas_workbench.py Tests/Character_Chat/ Tests/Chat/test_chat_functions.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```

Expected: all pass (report exact counts). Import smoke on the four touched modules + the new validation module.

- [ ] **Step 2: Spec status + cross-doc enum touch-up** — in the P1c spec: `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented (P1c).` And in `Docs/superpowers/specs/2026-07-14-roleplay-p1b-tryit-diagnostics-design.md`, AC2's status enum gains `` `skipped:disabled` (P1c: entry disabled — filtered after matching) `` in the same inline style (keeps the two specs' enums consistent).

- [ ] **Step 3: Commit** — `docs(roleplay): mark P1c richer-entries spec implemented (+P1b enum touch-up)` + trailer (both spec files in one commit).
