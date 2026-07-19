# Roleplay P2a — Lore foundation + Try-it diagnostics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Lore (world-book) authoring surface in the Roleplay workbench — List → Detail → Try-it — where the Try-it shows which entries fire, why, at what token cost, and which near-miss.

**Architecture:** Add a diagnostics data model + an instrumented `process_messages_with_diagnostics` to `world_info_processor.py`, keeping the live `process_messages` byte-identical (parallel candidate list, plain path untouched). Two new I/O-free widgets (`PersonasLoreDetailWidget`, `PersonasLoreTryItWidget`) mirror the Dictionaries widgets; `personas_screen.py` wires the "lore" mode to them, owning `WorldBookManager` CRUD off-thread.

**Tech Stack:** Python 3.11+, Textual, `WorldBookManager` + `WorldInfoProcessor` (ChaChaNotes DB), pytest.

**Spec:** `Docs/superpowers/specs/2026-07-16-roleplay-p2a-lore-foundation-design.md` (`51bbd142`). Branch `claude/roleplay-p2-lore` off dev `9de2a1c5`.

## Global Constraints

- **Byte-compatible plain path.** `process_messages` is live on the legacy send (`chat_events.py:1000`); its output (`injections`, `matched_entries` count+shape, `tokens_used`) must be byte-identical before/after. Do NOT modify `self.entries`, `_find_matching_entries`, `_entry_matches`, `_apply_token_budget`, or `_organize_by_position`. Add the diagnostics path in parallel.
- **Conversation dictionaries scope of P2a:** local-only, over `WorldBookManager` + `world_info_processor` directly — **no service facade**, no `ccp_lore_handler` (the live seam is direct access, mirroring `_dictionary_scope_service`).
- **Deferred — do NOT build:** entry priority / priority-aware budget (P2c); richer editor (selective/secondary/case/regex) / import-export (P2d); attach (P2e/f); Console "what's in play" / native-send / conversation-only-send-bug fix (P2g); legacy-UI retirement (later); constant/sticky entries (never). *The diagnostics engine still READS selective/secondary_keys/case_sensitive so near-misses display; the P2a editor just doesn't expose editing them.*
- **DB off the UI loop.** `WorldBookManager` methods are synchronous; call them via `await asyncio.to_thread(...)` from the screen (native sends/personas-io run on the event loop).
- **Escape user text** in DataTable cells / labels (the P1 `escape_markup` lesson).
- **Widgets are I/O-free**: they emit message classes + expose load/render methods; the screen owns all DB I/O.
- **Test env** (prefix every pytest run):
  `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`
- **Staging:** stage only each task's own files. Never `git add -A`; never stage anything under `.superpowers/`.

## File structure

- `tldw_chatbook/Character_Chat/world_info_diagnostics.py` — NEW: `WorldBookEntryDiagnostic` + `WorldBookScanDiagnostics` dataclasses. (Task 1)
- `tldw_chatbook/Character_Chat/world_info_processor.py` — MODIFY: parallel candidate list + `_classify_entry_match` (Task 2); `process_messages_with_diagnostics` (Task 3).
- `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` — NEW: `PersonasLoreDetailWidget` + message classes. (Task 4)
- `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_tryit.py` — NEW: `PersonasLoreTryItWidget`. (Task 5)
- `tldw_chatbook/UI/Screens/personas_screen.py` — MODIFY: lore-mode wiring + CRUD handlers + Try-it build. (Task 6)
- Tests: `Tests/Character_Chat/test_world_info_diagnostics.py` (Tasks 1–3), `Tests/UI/test_personas_lore.py` (Tasks 4–6).

---

## Task 1: Lore diagnostics data model

**Files:**
- Create: `tldw_chatbook/Character_Chat/world_info_diagnostics.py`
- Test: `Tests/Character_Chat/test_world_info_diagnostics.py` (create)

**Interfaces:**
- Produces: `WorldBookEntryDiagnostic(...)` + `WorldBookScanDiagnostics(...)` dataclasses, each with `to_dict()`. Consumed by Tasks 3 (populates them) and 5 (renders `.to_dict()`).

- [ ] **Step 1: Write the failing test.** Create `Tests/Character_Chat/test_world_info_diagnostics.py`:

```python
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
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

Run: `... -m pytest Tests/Character_Chat/test_world_info_diagnostics.py -q`

- [ ] **Step 3: Implement.** Create `tldw_chatbook/Character_Chat/world_info_diagnostics.py`:

```python
"""Diagnostics data model for world-info (Lore) activation — the read model the
Try-it panel renders. Mirrors ``Chat_Dictionary_Lib``'s
``DictionaryProcessDiagnostics``/``DictionaryEntryDiagnostic`` shape, with
world-info-specific fields (source book, injection position, recursion depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorldBookEntryDiagnostic:
    entry_id: Optional[int]
    source_book_id: Optional[int]
    source_book_name: str
    keys: List[str]
    activation_reason: str
    status: str  # "fired" | "skipped:disabled" | "skipped:secondary" | "skipped:budget"
    token_cost: int = 0
    injection_order: Optional[int] = None
    position: str = "before_char"
    content_preview: str = ""
    depth_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "source_book_id": self.source_book_id,
            "source_book_name": self.source_book_name,
            "keys": list(self.keys),
            "activation_reason": self.activation_reason,
            "status": self.status,
            "token_cost": self.token_cost,
            "injection_order": self.injection_order,
            "position": self.position,
            "content_preview": self.content_preview,
            "depth_level": self.depth_level,
        }


@dataclass
class WorldBookScanDiagnostics:
    entries: List[WorldBookEntryDiagnostic] = field(default_factory=list)
    matched: int = 0
    fired: int = 0
    skipped: int = 0
    tokens_used: int = 0
    token_budget: int = 0
    budget_exceeded: bool = False
    books_scanned: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [record.to_dict() for record in self.entries],
            "matched": self.matched,
            "fired": self.fired,
            "skipped": self.skipped,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "budget_exceeded": self.budget_exceeded,
            "books_scanned": self.books_scanned,
        }
```

- [ ] **Step 4: Run — PASS** (3 passed).
- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/Character_Chat/world_info_diagnostics.py Tests/Character_Chat/test_world_info_diagnostics.py
git commit -m "feat(lore): world-info activation diagnostics data model"
```

---

## Task 2: Byte-compat refactor — candidate entries + decomposed matcher

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_info_processor.py` (`__init__`/`_process_character_book`/`_process_world_books`/`_process_entry` build a parallel candidate list; add `_classify_entry_match`)
- Test: `Tests/Character_Chat/test_world_info_diagnostics.py` (add byte-compat + classifier tests)

**Interfaces:**
- Produces: `self._candidate_entries: list[dict]` (ALL entries incl. disabled, each = the processed entry dict PLUS keys `_entry_id`, `_book_id`, `_book_name`, `_enabled`); `_classify_entry_match(entry, scan_text, scan_text_lower) -> tuple[bool, str|None, bool, bool, str|None]` = `(primary_hit, primary_key, secondary_required, secondary_hit, secondary_key)`. Consumed by Task 3.
- **Unchanged (must stay byte-identical):** `self.entries`, `process_messages`, `_find_matching_entries`, `_entry_matches`, `_apply_token_budget`, `_organize_by_position`, `_estimate_entry_tokens`.

- [ ] **Step 1: Write the failing tests.** Append to `Tests/Character_Chat/test_world_info_diagnostics.py`:

```python
import copy

from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor


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
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: _candidate_entries` / `_classify_entry_match`).

- [ ] **Step 3: Implement.** In `world_info_processor.py`:
  - In `__init__`, after the existing entry-loading, initialize `self._candidate_entries: List[Dict[str, Any]] = []` **before** the character/world-book processing calls, and have the processing methods append to it.
  - Modify `_process_character_book` and `_process_world_books` so that for **every** raw entry (not just `enabled`), they build a candidate via a new helper and append it to `self._candidate_entries`, while STILL only appending enabled+processed entries to `self.entries` exactly as today. Concretely, add a helper and call it alongside the existing loop:

```python
    def _make_candidate(self, entry, book_id, book_name, priority_offset=0):
        """Build a diagnostics candidate for ANY entry (enabled or not), carrying
        source-book + id + enabled metadata. Does NOT affect self.entries / the
        plain path. Returns None only if the entry has no usable keys."""
        processed = self._process_entry(entry)
        if processed is None:
            return None
        processed = dict(processed)
        processed["insertion_order"] = processed.get("insertion_order", 0) + priority_offset
        processed["_entry_id"] = entry.get("id")
        processed["_book_id"] = book_id
        processed["_book_name"] = book_name
        processed["_enabled"] = bool(entry.get("enabled", True))
        return processed
```

  In `_process_character_book`, alongside the existing `for entry in raw_entries:` loop, append `cand = self._make_candidate(entry, None, self.character_book.get("name", "Character book")); ` and if not None append to `self._candidate_entries`. In `_process_world_books`, inside the per-book loop use `priority_offset = book.get("priority", 0) * 1000` and `book_id = book.get("id")`, `book_name = book.get("name", "")`; for each entry append `self._make_candidate(entry, book_id, book_name, priority_offset)` (skip None). Sort `self._candidate_entries` by `insertion_order` at the end of each method, mirroring the existing `self.entries` sort. **Leave the existing `self.entries` construction exactly as-is** (do not route it through `_make_candidate`).

  - Add the decomposed matcher (reuses `_keyword_in_text`, does not change `_entry_matches`):

```python
    def _classify_entry_match(self, entry, scan_text, scan_text_lower):
        """Decompose an entry's match for diagnostics. Returns
        (primary_hit, primary_key, secondary_required, secondary_hit, secondary_key).
        Mirrors _entry_matches' logic exactly but reports WHICH key matched / why not."""
        case = entry.get("case_sensitive", False)

        def hit(key):
            return (self._keyword_in_text(key, scan_text) if case
                    else self._keyword_in_text(key.lower(), scan_text_lower))

        primary_key = next((k for k in entry.get("keys", []) if hit(k)), None)
        primary_hit = primary_key is not None
        if not primary_hit:
            return (False, None, False, False, None)
        secondary_required = bool(entry.get("selective", False) and entry.get("secondary_keys"))
        if not secondary_required:
            return (True, primary_key, False, False, None)
        secondary_key = next((k for k in entry.get("secondary_keys", []) if hit(k)), None)
        return (True, primary_key, True, secondary_key is not None, secondary_key)
```

- [ ] **Step 4: Run — PASS** (the 3 new tests + Task 1's 3). Then confirm no world-info regression:
  `... -m pytest Tests/Character_Chat/test_world_info_diagnostics.py $(git ls-files 'Tests/**/*world*info*' 'Tests/**/*world*book*') -q`
  Expected: all pass (byte-compat preserved).

- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/Character_Chat/world_info_processor.py Tests/Character_Chat/test_world_info_diagnostics.py
git commit -m "refactor(lore): parallel candidate list + decomposed matcher (plain path byte-unchanged)"
```

---

## Task 3: `process_messages_with_diagnostics` + classification

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_info_processor.py` (add the method)
- Test: `Tests/Character_Chat/test_world_info_diagnostics.py` (classification + fired==plain pin)

**Interfaces:**
- Consumes: `self._candidate_entries`, `_classify_entry_match`, `_build_scan_text`, `_apply_token_budget`, `_estimate_entry_tokens`, `_organize_by_position` (Task 2 + existing); `WorldBookScanDiagnostics`/`WorldBookEntryDiagnostic` (Task 1).
- Produces: `process_messages_with_diagnostics(current_message, conversation_history, scan_depth=None, apply_token_budget=True) -> tuple[dict, WorldBookScanDiagnostics]`. The `dict` equals plain `process_messages`. Consumed by the screen (Task 6).

- [ ] **Step 1: Write the failing tests.** Append classification + agreement tests:

```python
from tldw_chatbook.Character_Chat.world_info_diagnostics import WorldBookScanDiagnostics


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
```

- [ ] **Step 2: Run — expect FAIL** (`AttributeError: process_messages_with_diagnostics`).

- [ ] **Step 3: Implement.** Add to `WorldInfoProcessor`:

```python
    def process_messages_with_diagnostics(self, current_message, conversation_history,
                                          scan_depth=None, apply_token_budget=True):
        """Instrumented sibling of process_messages. Returns (result, diagnostics)
        where `result` is byte-identical to process_messages (the fired-set pin)
        and `diagnostics` classifies every candidate as fired / skipped:disabled /
        skipped:secondary / skipped:budget. Never raises on a bad entry."""
        from .world_info_diagnostics import WorldBookScanDiagnostics, WorldBookEntryDiagnostic

        # Authoritative result — literally the plain path (guarantees agreement).
        result = self.process_messages(current_message, conversation_history,
                                       scan_depth, apply_token_budget)
        fired_list = result.get("matched_entries", [])

        depth = scan_depth if scan_depth is not None else self.scan_depth
        scan_text = self._build_scan_text(current_message, conversation_history, depth)
        scan_text_lower = scan_text.lower()

        # Identify the fired entries by insertion_order + content + position (the
        # stable non-meta fields), so a candidate can be marked fired and given
        # its injection order. (matched_entries carry no _entry_id.)
        def sig(e):
            return (e.get("insertion_order", 0), e.get("content", ""), e.get("position", "before_char"))
        fired_sig_order = {sig(e): i for i, e in enumerate(fired_list)}

        records = []
        seen = set()
        books = set()
        for cand in self._candidate_entries:
            books.add(cand.get("_book_id"))
            try:
                primary_hit, pk, sec_req, sec_hit, sk = self._classify_entry_match(
                    cand, scan_text, scan_text_lower)
            except Exception:
                continue
            if not primary_hit:
                continue  # key never appeared → not reported
            key = sig(cand)
            if cand.get("_enabled", True) is False:
                status, reason, order = "skipped:disabled", f"disabled (key '{pk}' matched)", None
            elif sec_req and not sec_hit:
                status, reason, order = "skipped:secondary", "secondary key not found", None
            elif key in fired_sig_order and key not in seen:
                status = "fired"
                order = fired_sig_order[key]
                reason = (f"matched key '{pk}'" + (f" + secondary '{sk}'" if sk else ""))
                seen.add(key)
            else:
                status = "skipped:budget"
                order = None
                reason = "dropped by token budget"
            records.append(WorldBookEntryDiagnostic(
                entry_id=cand.get("_entry_id"), source_book_id=cand.get("_book_id"),
                source_book_name=str(cand.get("_book_name") or ""),
                keys=list(cand.get("keys", [])), activation_reason=reason, status=status,
                token_cost=self._estimate_entry_tokens(cand), injection_order=order,
                position=cand.get("position", "before_char"),
                content_preview=(cand.get("content", "") or "")[:80], depth_level=0,
            ))

        fired = sum(1 for r in records if r.status == "fired")
        diagnostics = WorldBookScanDiagnostics(
            entries=records, matched=len(records), fired=fired,
            skipped=len(records) - fired, tokens_used=result.get("tokens_used", 0),
            token_budget=self.token_budget,
            # Truncation-derived (mirrors the dictionary diagnostics): true when the
            # budget dropped at least one matched entry. The fired set's tokens_used
            # is always <= budget by construction, so don't compare against it.
            budget_exceeded=any(r.status == "skipped:budget" for r in records),
            books_scanned=len({b for b in books if b is not None}),
        )
        return result, diagnostics
```

  Note: `depth_level` is left 0 in P2a (recursion depth reporting is a small later enhancement; the plain recursion still runs in `result`). `matched` counts *reported* candidates (primary-key hits), not the send's `matched_entries`; the fired subset equals the send.

- [ ] **Step 4: Run — PASS** (3 new + prior). Then the full processor/diagnostics suite:
  `... -m pytest Tests/Character_Chat/test_world_info_diagnostics.py -q`

- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/Character_Chat/world_info_processor.py Tests/Character_Chat/test_world_info_diagnostics.py
git commit -m "feat(lore): process_messages_with_diagnostics (fired/near-miss classification, result==plain)"
```

---

## Task 4: `PersonasLoreDetailWidget`

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py`
- Test: `Tests/UI/test_personas_lore.py` (create; widget-in-isolation via a minimal `_Host(App)`)

**Template to adapt:** `Widgets/Persona_Widgets/personas_dictionary_detail.py` (`PersonasDictionaryDetailWidget`). Read it as the structural template — a `Vertical` subclass, plain `Message` subclasses (no dataclass), a `TabbedContent` with an Entries pane (`DataTable(cursor_type="row")` + an inline add/edit form of Input/TextArea/Switch/Select + Add/Update/Delete/Move-up/Move-down buttons) and a Settings pane, plain instance attributes for state (no `reactive()`), a manual dirty-diff, and `Move up/down` posting the **full reordered id list**. Lore differs only in the FIELDS and the pared-down tab set.

**Interfaces:**
- Produces (message classes, all `Message` subclasses): `LoreBookCreateRequested()`, `LoreBookDuplicateRequested()`, `LoreBookDeleteRequested()`, `LoreBookEnableToggled(enabled: bool)`, `LoreEntryAddRequested(payload: dict)`, `LoreEntryUpdateRequested(entry_id: str, payload: dict)`, `LoreEntryDeleteRequested(entry_id: str)`, `LoreEntriesReorderRequested(entry_ids: list[str])`, `LoreBookSettingsSaveRequested(payload: dict)`.
- Public API: `load_book(record: dict)`, `update_entries(entries: list[dict])`, `apply_enabled(enabled: bool)`, `clear()`, `entry_form_payload() -> dict | None`, `settings_payload() -> dict`, `set_status(message: str)`, `selected_entry_id` (property), `entry_ids_in_order() -> list[str]`.
- Entry payload shape (matches `WorldBookManager.create_world_book_entry` kwargs subset P2a edits): `{"keys": list[str], "content": str, "position": str, "enabled": bool, "insertion_order": int}`.
- Settings payload: `{"name": str, "description": str, "scan_depth": int, "token_budget": int, "recursive_scanning": bool, "enabled": bool}`.

- [ ] **Step 1: Write the failing test.** Create `Tests/UI/test_personas_lore.py` with an isolated-widget harness (mirror `Tests/UI/test_dictionary_picker.py`'s `_Host(App)` pattern):

```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_detail import (
    PersonasLoreDetailWidget,
    LoreEntryAddRequested,
)


class _DetailHost(App):
    def __init__(self):
        super().__init__()
        self.posted = []

    def compose(self) -> ComposeResult:
        yield PersonasLoreDetailWidget(id="personas-lore-detail")

    def on_lore_entry_add_requested(self, message: LoreEntryAddRequested) -> None:
        self.posted.append(message.payload)


@pytest.mark.asyncio
async def test_detail_loads_book_and_lists_entries():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "Blackreach", "description": "",
                          "scan_depth": 3, "token_budget": 500,
                          "recursive_scanning": False, "enabled": True})
        widget.update_entries([
            {"id": 7, "keys": ["Warden"], "content": "grim jailer",
             "position": "before_char", "enabled": True, "insertion_order": 0},
        ])
        await pilot.pause()
        table = app.query_one("#personas-lore-entries-table", DataTable)
        assert table.row_count == 1
        assert widget.settings_payload()["name"] == "Blackreach"


@pytest.mark.asyncio
async def test_add_entry_posts_payload_from_form():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        # fill the form
        app.query_one("#personas-lore-entry-keys").value = "Warden, Jailer"
        app.query_one("#personas-lore-entry-content").text = "grim jailer"
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        assert app.posted, "add must post LoreEntryAddRequested"
        payload = app.posted[-1]
        assert payload["keys"] == ["Warden", "Jailer"] and payload["content"] == "grim jailer"


@pytest.mark.asyncio
async def test_reorder_posts_full_id_list():
    app = _DetailHost()
    posted = []
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        widget.update_entries([
            {"id": 1, "keys": ["a"], "content": "x", "position": "before_char",
             "enabled": True, "insertion_order": 0},
            {"id": 2, "keys": ["b"], "content": "y", "position": "before_char",
             "enabled": True, "insertion_order": 1},
        ])
        await pilot.pause()
        assert widget.entry_ids_in_order() == ["1", "2"]
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `PersonasLoreDetailWidget`** by adapting `personas_dictionary_detail.py`. Required concrete pieces:
  - The message classes named in Interfaces (plain `Message` subclasses; `LoreEntryAddRequested.__init__(self, payload: dict)`, `LoreEntryUpdateRequested.__init__(self, entry_id: str, payload: dict)`, etc.).
  - `compose()`: a `TabbedContent(id="personas-lore-tabs")` with two `TabPane`s — **Entries** and **Settings** — plus a trailing `Static(id="personas-lore-status")`.
    - Entries pane: `DataTable(id="personas-lore-entries-table", cursor_type="row")` (columns added in `on_mount`: `keys`, `content`, `position`, `enabled`); below it the form — `Input(id="personas-lore-entry-keys")` (comma-separated), `TextArea(id="personas-lore-entry-content")`, `Select(id="personas-lore-entry-position", options=[("before_char",…),("after_char",…),("at_start",…),("at_end",…)])`, `Switch(id="personas-lore-entry-enabled")`; buttons `#personas-lore-entry-add`, `#-update`, `#-delete`, `#-move-up`, `#-move-down`.
    - Settings pane: `Input(#personas-lore-name)`, `TextArea(#personas-lore-description)`, `Input(#personas-lore-scan-depth)`, `Input(#personas-lore-token-budget)`, `Switch(#personas-lore-recursive)`, `Switch(#personas-lore-enabled)`, a `#personas-lore-settings-save` button.
  - `entry_form_payload()`: parse keys as comma-split+strip+drop-empty; content from the TextArea; position from the Select; enabled from the Switch; `insertion_order` = current count (append) for Add, or the edited entry's order for Update. Return `None` if keys empty or content empty.
  - `settings_payload()`: read the Settings fields; coerce `scan_depth`/`token_budget` to int (default 3/500 on bad input).
  - `_fill_form_from_entry` on row selection; `_post_reorder(offset)` swaps adjacent ids in `entry_ids_in_order()` and posts `LoreEntriesReorderRequested(ids)`.
  - **Escape user text** when building DataTable rows (use `rich.markup.escape` on keys/content/preview), mirroring the dictionary detail widget.
  - Add/Update/Delete/Save buttons post the corresponding messages with the payloads.

- [ ] **Step 4: Run — PASS** (3 tests). 
- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): PersonasLoreDetailWidget (List/Entries/Settings, I/O-free)"
```

---

## Task 5: `PersonasLoreTryItWidget`

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_tryit.py`
- Test: `Tests/UI/test_personas_lore.py` (add Try-it rendering tests)

**Template to adapt:** `Widgets/Persona_Widgets/personas_dictionary_tryit.py`. Reuse its **diagnostics story** structure (summary strip + fired list + near-miss list, each in its own `try/except`, degrading gracefully; `if not isinstance(diagnostics, dict)` → "diagnostics unavailable"). The MAIN result view differs: **not** a word-diff — show injected content grouped by position.

**Interfaces:**
- Produces: `LoreTryItRunRequested(text: str, pull_history: bool)` message; `PersonasLoreTryItWidget(Vertical)` with `set_ready(ready, hint="")`, `sample_text() -> str`, `pull_history() -> bool`, `render_result(injections: dict, diagnostics: dict | None)`, `show_error(message: str)`, `action_run_preview()`.
- Consumes: the diagnostics dict is `WorldBookScanDiagnostics.to_dict()` from Task 3 (`entries` list with `status`/`keys`/`content_preview`/`token_cost`/`injection_order`/`activation_reason`; top-level `fired`/`skipped`/`tokens_used`/`token_budget`/`budget_exceeded`). `injections` is `world_info_processor` `injections` (`{position: [content,...]}`).

- [ ] **Step 1: Write the failing tests.** Add to `Tests/UI/test_personas_lore.py`:

```python
from tldw_chatbook.Widgets.Persona_Widgets.personas_lore_tryit import PersonasLoreTryItWidget


class _TryItHost(App):
    def compose(self) -> ComposeResult:
        yield PersonasLoreTryItWidget(id="personas-lore-tryit")


@pytest.mark.asyncio
async def test_tryit_renders_injections_by_position_and_diagnostics():
    app = _TryItHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreTryItWidget)
        injections = {"before_char": ["grim jailer"], "after_char": [],
                      "at_start": [], "at_end": []}
        diagnostics = {
            "entries": [
                {"entry_id": 1, "keys": ["Warden"], "activation_reason": "matched key 'Warden'",
                 "status": "fired", "token_cost": 3, "injection_order": 0,
                 "position": "before_char", "content_preview": "grim jailer", "depth_level": 0},
                {"entry_id": 2, "keys": ["Ghost"], "activation_reason": "disabled (key 'Ghost' matched)",
                 "status": "skipped:disabled", "token_cost": 0, "injection_order": None,
                 "position": "before_char", "content_preview": "pale", "depth_level": 0},
            ],
            "matched": 2, "fired": 1, "skipped": 1, "tokens_used": 3,
            "token_budget": 500, "budget_exceeded": False, "books_scanned": 1,
        }
        widget.render_result(injections, diagnostics)
        await pilot.pause()
        from textual.widgets import Static
        summary = app.query_one("#personas-lore-tryit-summary", Static)
        assert "1 fired" in str(summary.renderable)
        # injection preview shows the fired content under before_char
        preview = app.query_one("#personas-lore-tryit-injections", Static)
        assert "grim jailer" in str(preview.renderable)


@pytest.mark.asyncio
async def test_tryit_degrades_on_bad_diagnostics():
    app = _TryItHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreTryItWidget)
        widget.render_result({"before_char": [], "after_char": [], "at_start": [], "at_end": []}, None)
        await pilot.pause()  # must not raise
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** `PersonasLoreTryItWidget(Vertical)`:
  - `compose()`: `Static` title; `TextArea(id="personas-lore-tryit-sample")`; a `Switch(id="personas-lore-tryit-pull-history")` labeled "Include recent turns"; `Button("Run preview", id="personas-lore-tryit-run", disabled=True)`; `Static(id="personas-lore-tryit-status")`; `Static(id="personas-lore-tryit-injections")` (the injection-by-position preview); `Static(id="personas-lore-tryit-summary")`; a `Vertical(id="personas-lore-tryit-details")` containing `Static(id="personas-lore-tryit-fired")` and `Static(id="personas-lore-tryit-nearmiss")`.
  - `BINDINGS = [Binding("ctrl+enter", "run_preview", "Run preview", show=False, priority=True)]`; `action_run_preview` posts `LoreTryItRunRequested(self.sample_text(), self.pull_history())` if ready.
  - `render_result(injections, diagnostics)`: build the injections preview by iterating positions in fixed order (`before_char`, `at_start`, `at_end`, `after_char`) and listing each content line under a position header (escape markup); then call `_render_diagnostics(diagnostics)`.
  - `_render_diagnostics(diagnostics)`: mirror the dictionary widget — `if not isinstance(diagnostics, dict)` → set summary "diagnostics unavailable" and clear lists; else summary = `f"{fired} fired · {skipped} near-miss · {tokens_used}/{token_budget} tokens"` + `" · over budget"` when `budget_exceeded`; fired list = entries with `status == "fired"` sorted by `(injection_order is None, injection_order or 0)`, each row `f"{', '.join(keys)} → {content_preview} · {token_cost} tok"`; near-miss list = entries with `status != "fired"` (there are only fired/near-miss here), each row `f"{', '.join(keys)} — {activation_reason}"`. Wrap each of the three sections in `try/except Exception`.
  - `set_ready`, `sample_text`, `pull_history`, `show_error`, and the `@on(Button.Pressed, "#personas-lore-tryit-run")` handler.

- [ ] **Step 4: Run — PASS** (2 tests + Task 4's 3).
- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_lore_tryit.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): PersonasLoreTryItWidget (injection preview + diagnostics story)"
```

---

## Task 6: `personas_screen.py` wiring + integration

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_lore.py` (add a mounted-screen integration test)

**Interfaces:**
- Consumes: Task 4/5 widgets + their message classes; `WorldBookManager` (`Character_Chat/world_book_manager.py`); Task 3 `process_messages_with_diagnostics`.

- [ ] **Step 1: Write the failing integration test.** Add to `Tests/UI/test_personas_lore.py`, mirroring `Tests/UI/test_personas_dictionaries.py`'s `PersonasTestApp` harness (delegating `App` + `mock_app_instance` from `Tests/UI/conftest.py`), but inject a **real** `CharactersRAGDB` as `mock_app_instance.chachanotes_db` seeded via `WorldBookManager` (so the screen's direct-manager CRUD is exercised):

```python
# (full harness modeled on Tests/UI/test_personas_dictionaries.py:493-531 —
#  PersonasTestApp(mock_app_instance) that pushes PersonasScreen; a real
#  CharactersRAGDB set as mock_app_instance.chachanotes_db; seed one world book
#  + entry via WorldBookManager before run_test.)
#
# Assertions:
#  1. clicking "#personas-mode-lore" shows "#personas-lore-detail" (not the
#     coming-soon placeholder) and the book appears in the library rail.
#  2. selecting the book loads its entries into PersonasLoreDetailWidget.
#  3. running Try-it with sample text matching an entry's key renders a fired
#     row (the screen builds WorldInfoProcessor(world_books=[{**book, entries}])
#     and calls process_messages_with_diagnostics).
```

(Author the harness concretely from the dictionaries template; the load-bearing assertions are the three above.)

- [ ] **Step 2: Run — expect FAIL** (lore mode still shows the placeholder; no lore handlers).

- [ ] **Step 3: Implement the wiring.**
  - Remove `"lore"` from `_COMING_SOON_MODES` (line 128 → `frozenset()`), and remove the `"lore"` entry from `_MODE_PLACEHOLDER_BODY` (line 131-134).
  - Add `"#personas-lore-detail"` to `_CENTER_VIEW_IDS` (line 151).
  - In `compose()` inside `#personas-detail-stack` (near line 485), add `yield PersonasLoreDetailWidget(id="personas-lore-detail")`; and near the dictionary Try-it mount (line 489-491) add `yield PersonasLoreTryItWidget(id="personas-lore-tryit")` with `display = False`.
  - In `_apply_mode` (line 972-1015): add `elif mode == "lore": await self._render_lore_rows(); self._show_center(None)`; generalize the Try-it visibility gate (line 991-996) so the lore try-it shows when `mode == "lore"` and the dict try-it when `mode == "dictionaries"`.
  - Add a `_lore_manager()` helper: `db = getattr(self.app_instance, "chachanotes_db", None); return WorldBookManager(db) if db is not None else None` (local import of `WorldBookManager`).
  - Add `_render_lore_rows(query=…)` (list books via `await asyncio.to_thread(manager.list_world_books, True)` → populate the library rail with `entity_kind="world_book"`), a `_select_lore_entry(entity_id)` (load the book + entries via `asyncio.to_thread`, call `detail.load_book(...)`/`detail.update_entries(...)`, set the Try-it ready), and the `@on(...)` handlers for the Lore message classes — mirroring `_run_dictionary_entry_op` but with `await asyncio.to_thread(manager.<method>, …)` for the sync manager, and the `run_worker(group="personas-io")` + `_io_dialog_active` guard for Create/Duplicate/Delete-book (Duplicate = `import_world_book(export_world_book(id), name_override=f"{name} (copy)")`; reorder = one `update_world_book_entry(entry_id, insertion_order=i)` per entry in the posted order).
  - In `_handle_entity_selected` (line 1067-1084), add an `elif message.entity_kind == "world_book":` branch → `self._run_guarded(lambda: self._select_lore_entry(message.entity_id))`.
  - `@on(LoreTryItRunRequested)`: build `WorldInfoProcessor(world_books=[{**book, "entries": entries}])` (from the currently-selected book + its entries, already in memory) and call `process_messages_with_diagnostics(sample, history)` off-thread; on the result call `tryit.render_result(result["injections"], diagnostics.to_dict())`. History = last-N turns only if the pull-history switch is on (P2a can pass `[]` when off; the source of "recent turns" is out of P2a's authoring scope — use `[]` unless a conversation is trivially available, and note it).

- [ ] **Step 4: Run — PASS** (integration test). Then the personas regression:
  `... -m pytest Tests/UI/test_personas_lore.py Tests/UI/test_personas_dictionaries.py -q`

- [ ] **Step 5: Commit.**
```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): wire Lore mode into the Roleplay workbench (List/Detail/Try-it)"
```

---

## Task 7: Full gate + spec status

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-16-roleplay-p2a-lore-foundation-design.md` (status line)

- [ ] **Step 1: Full gate.**
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Character_Chat/ Tests/UI/test_personas_lore.py Tests/UI/test_personas_dictionaries.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass (record counts). Then the import smoke:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('IMPORT OK')"
```

- [ ] **Step 2: Flip spec status** — `**Status:** Design approved (brainstorm), pending spec review.` → `**Status:** Implemented (P2a).`

- [ ] **Step 3: Commit.**
```bash
git add Docs/superpowers/specs/2026-07-16-roleplay-p2a-lore-foundation-design.md
git commit -m "docs(roleplay): mark P2a Lore foundation spec implemented"
```

---

## Notes for the executor

- **Load-bearing tests:** the Task 2/3 byte-compat pin (`process_messages` unchanged; diagnostics `result == plain` incl. disabled, selective/secondary, budget hard-break, multi-book priority-offset) and the Task 6 integration (lore mode shows the real widget; Try-it renders a real fired row from `process_messages_with_diagnostics`). These pin the two invariants: the live send is unaffected, and "shown = what the engine computes."
- **Do NOT touch the plain path.** `self.entries`, `process_messages`, `_find_matching_entries`, `_entry_matches`, `_apply_token_budget`, `_organize_by_position` stay byte-identical; the diagnostics path is parallel (`self._candidate_entries` + `_classify_entry_match` + `process_messages_with_diagnostics`).
- **Widgets are I/O-free**; the screen owns `WorldBookManager` CRUD, always via `asyncio.to_thread` (the manager is synchronous, native/personas-io workers run on the loop). Duplicate = export+import(name_override); reorder = per-entry `insertion_order` update; delete-book confirms via the `_io_dialog_active`-guarded worker.
- **Scope:** no entry priority, no selective/secondary/case/regex editing, no import-export, no attach, no Console/native-send. The diagnostics engine still READS selective/secondary/case so near-misses display for imported/existing entries.
- **Widget tasks (4, 5)** adapt the named dictionary widgets structurally — read `personas_dictionary_detail.py` / `personas_dictionary_tryit.py` as templates; the plan gives the message classes, ids, payload shapes, and the Lore-specific deltas (fields; injection-preview instead of word-diff).
