# Roleplay P2d-regex — optional regex matching for Lore entries

**Status:** Implemented (P2d-regex).

**Program:** Roleplay (Personas) redesign — P2 (Lore mode). The regex piece deliberately deferred from P2d-1; follows P2d-1 (matching editor #694) and P2d-2 (import/export #701), both merged. Schema currently v21.

## Why

A Lore entry matches its keys as word-boundary literals today. Power users want an entry's keys treated as **regex patterns** (case-insensitive by default, no word-boundary wrapping). The blocker was safety: matching runs on the UI event loop, so a catastrophic pattern could freeze the whole app. P2d-regex adds an opt-in per-entry `regex` flag plus a fail-closed validation layer so bad patterns never reach the send path.

## Load-bearing safety facts (verified at dev tip)

- **Matching runs on the UI loop.** `handle_chat_send_button_pressed` (`chat_events.py:420`) calls `world_info_processor.process_messages(...)` synchronously at `:1020` — before the API-call worker (`:1279`). A catastrophic regex freezes the TUI.
- **`re` cannot be portably time-bounded.** The `regex` third-party module (the only one with a match `timeout=`) is NOT a dependency; `signal.alarm` is POSIX-only and the app targets Windows; and CPython's `re` does not release the GIL during matching, so moving matching off-thread would not prevent a freeze either.
- **Therefore validation is the only lever.** Patterns are validated **fail-closed at save and at import**; the send path additionally never raises.

## Behavior-change framing

Purely opt-in and inert by default. Every existing entry gets `regex = 0` (column default); the matcher's `regex=False` path is **byte-identical to today**, so P2a/P2c/P2d pins hold and the live legacy send is unchanged for all existing data. Only an entry the user explicitly flags `regex` changes behavior.

## Residual-risk disclosure (accepted)

The ReDoS heuristic is **best-effort, not a hard guarantee**. It catches nested unbounded-quantifier shapes (`(a+)+`) and the trivial identical-alternation case (`(a|a)*`), but **not** general alternation-overlap ReDoS (e.g. `(a|ab)*`). Because `re` can't be timed out here, a pattern that slips past the heuristic can still freeze the app. This is an accepted trade-off for a single-user local TUI whose user authors their own lore (and whose imports are validated by the same heuristic). Documented so future work can add a timeout-capable engine (`regex`/`re2`) if the threat model changes.

## Ground truths (verified post-#701)

- **Schema:** `world_book_entries` (`ChaChaNotes_DB.py:1194`) has `…, selective, secondary_keys, case_sensitive, extensions, …` — no `regex`. `_CURRENT_SCHEMA_VERSION = 21`. Migration precedent: `_MIGRATE_V20_TO_V21_SQL` (`:2343`) + `_migrate_from_v20_to_v21` (`:3358`, PRAGMA-`table_info` guard + `ALTER TABLE … ADD COLUMN` + `executescript` + version check) + `migration_steps[20]` (`:3511`).
- **Matcher:** `_entry_matches` and `_classify_entry_match` (`world_info_processor.py`) both check each key via, effectively, `case_sensitive ? _keyword_in_text(key, scan_text) : _keyword_in_text(key.lower(), scan_text_lower)`. `_keyword_in_text` = `re.search(r'\b'+re.escape(keyword)+r'\b', text)`. `_process_entry` builds the processed entry dict.
- **Manager:** `create_world_book_entry(..., selective, secondary_keys, case_sensitive, extensions, priority)` (flat columns), `update_world_book_entry(**kwargs)` field loop, `get_world_book_entries` row dict, `export_world_book` per-entry dict — all enumerate the matching fields explicitly.
- **Import adapter:** `Character_Chat/world_book_import.py::normalize_world_book_import` maps external → tldw fields and raises `ValueError` (fail-closed) on bad entries; `import_world_book` reads `entry.get('regex', …)` won't exist yet.
- **Editor:** `personas_lore_detail.py` entry form has the P2d-1 matching row (`#personas-lore-entry-case-sensitive`, `-selective`, `-secondary-keys`); `entry_form_payload`, `_fill_form_from_entry`, and the `_handle_lore_entry_add` explicit-kwarg call are the P2d-1 seams.

## Architecture

### 1. Schema migration v21→v22 (`ChaChaNotes_DB.py` + doc-mirror)

Mirror P2c's v20→v21 exactly: add `regex BOOLEAN DEFAULT 0` to `world_book_entries`.
- `_MIGRATE_V21_TO_V22_SQL` + `_migrate_from_v21_to_v22(conn)` (PRAGMA-`table_info` idempotent guard → `ALTER TABLE world_book_entries ADD COLUMN regex BOOLEAN DEFAULT 0`; `DROP/CREATE` the two sync triggers `world_book_entries_sync_{create,update}` adding `'regex', NEW.regex` to both `json_object(...)` payloads and `OR OLD.regex IS NOT NEW.regex` to the update `WHEN`; `UPDATE db_schema_version SET version = 22 … AND version = 21`). FTS triggers untouched.
- Bump `_CURRENT_SCHEMA_VERSION = 22`; register `migration_steps[21] = self._migrate_from_v21_to_v22`.
- Update the base DDL (`world_book_entries` table + its 2 sync triggers) so fresh == migrated.
- Doc-mirror `tldw_chatbook/DB/migrations/chachanotes_v21_to_v22_world_book_entry_regex.sql`. **Next migration = v22→v23.** Re-verify at branch/merge time that no parallel branch claimed v22 (the v19/v20 collision lesson).

### 2. ReDoS-safe regex validator — new pure module `Character_Chat/world_info_regex.py`

Pure, DB-free, unit-testable.

- `MAX_REGEX_PATTERN_LENGTH = 500`
- `validate_regex_pattern(pattern: str) -> None` — raises `ValueError` (user-facing message) when a pattern is unusable, used at **save** and **import**:
  1. length > `MAX_REGEX_PATTERN_LENGTH` → `ValueError("Regex pattern is too long (max 500 characters).")`
  2. `re.compile(pattern)` fails → `ValueError(f"Invalid regex: {err}")`
  3. catastrophic-pattern heuristic → `ValueError("Regex pattern is too complex (nested quantifiers can hang matching).")`. The heuristic (best-effort, see residual-risk) flags: (a) an **unbounded** quantifier (`+`, `*`, `{n,}`) applied to a group whose body itself contains an unbounded quantifier — i.e. `(…+…)+`, `(…*…)*`, `(…+…)*`, … ; (b) the trivial identical-alternation `(x|x)…` followed by an unbounded quantifier. Bounded quantifiers (`{n}`, `{n,m}`) are NOT flagged.
- `regex_search(pattern: str, text: str, ignore_case: bool) -> bool` — send-path matcher: `re.search(pattern, text, re.IGNORECASE if ignore_case else 0)` in a `try/except Exception: return False`. **Never raises** — a bad/uncompilable pattern simply doesn't fire (defense-in-depth beyond the authoring/import gate).

### 3. Matcher regex branch (`world_info_processor.py`)

- `_process_entry` carries `'regex': bool(entry.get('regex', False))` in the processed entry dict (DB returns 0/1; character-book/imported dicts default to False; the import adapter already coerces booleans before persistence).
- **Load-time backstop (source-independent safety):** `_process_entry` is the single choke point where *all* entry sources converge before matching — standalone world books (`_process_world_books`), a character's embedded `character_book` (`_process_character_book`), and imports. When `regex` is True, validate every key **and secondary key** via `validate_regex_pattern`; if any raises, **downgrade the processed entry to `regex=False`** (literal) so a catastrophic/invalid pattern can never reach the send-path matcher regardless of origin. This closes the character-card vector that the editor/import gates don't cover (a crafted `character_book` entry never passes through them), and the send-path never-raise guard only stops exceptions, not a hang. It preserves the stable-no-op: only `regex=True`+bad entries are affected; every `regex=False` entry (all existing data) is byte-identical, and the check costs nothing when `regex` is False. Downgrading to literal is safe — the pattern text is `re.escape`'d and won't match normal chat.
- Introduce **one** private helper both matcher methods call, so the regex branch lives in a single place and `_entry_matches`/`_classify_entry_match` cannot drift (the P2a diagnostics-mirror invariant, by construction):
  ```
  def _key_hits(self, entry, key, scan_text, scan_text_lower) -> bool:
      if entry.get('regex', False):
          return regex_search(key, scan_text, ignore_case=not entry.get('case_sensitive', False))
      if entry.get('case_sensitive', False):
          return self._keyword_in_text(key, scan_text)
      return self._keyword_in_text(key.lower(), scan_text_lower)
  ```
  Refactor `_entry_matches` (both primary and secondary loops) and `_classify_entry_match`'s `hit(key)` closure to call `self._key_hits(entry, key, scan_text, scan_text_lower)`. When `regex=False` the literal path is byte-identical to today (still lowercases key+text when not case-sensitive). When `regex=True`, the key is a pattern matched against the **original** `scan_text` with `re.IGNORECASE` (when not case-sensitive) — no `\b`, no `re.escape`.

### 4. Manager + import/export (`world_book_manager.py`, `world_book_import.py`)

- `create_world_book_entry(..., regex: bool = False)` writes it; `update_world_book_entry` field loop handles `regex` (bool); `get_world_book_entries` returns it; `export_world_book` serializes it.
- `import_world_book` passes `regex=entry.get('regex', False)`.
- `normalize_world_book_import`: map a tldw `regex` bool (via `_coerce_bool(entry.get('regex'), False)`; SillyTavern's classic World Info has no per-entry regex-key flag, so external files default to literal). **For a regex entry, validate every key (and secondary key) via `validate_regex_pattern`** — a bad/too-complex pattern raises `ValueError` naming the entry (fail-closed, consistent with the adapter's existing strict per-entry validation, so an imported world book can't smuggle a pattern the heuristic recognizes as catastrophic; the documented best-effort residual still applies). Add `regex` to the normalized entry field set.

### 5. Editor (`personas_lore_detail.py`, `personas_screen.py`)

- Add a **Regex** `Switch(id="personas-lore-entry-regex")` to the entry form. Default off. It joins the P2d-1 matching row (Row 2, beside Case-sensitive/Selective), making three labeled switches — **verify the row still fits at the 2050×1240 QA size (the fr-width-wrap lesson); if it crowds, move Regex to its own row** below the switches.
- `entry_form_payload` includes `regex` (bool); `_fill_form_from_entry` populates it.
- **Save-time validation in the widget** (validate_regex_pattern is pure — the I/O-free widget may import it): a helper the add/update pressed-handlers call — when `regex` is on, validate every key **and secondary key**; on the first invalid/too-complex pattern, `set_status(<message>)` and do **not** post. Immediate feedback before anything is persisted.
- `_handle_lore_entry_add` threads `regex=payload.get("regex", False)` (the explicit-kwarg-drop lesson); update already forwards `**payload`.
- **Try-it: no change** — invalid patterns can't persist (save + import both validate), so there's no persisted-invalid entry to warn about; regex entries fire and display like any other match.

## Error handling / safety

- Fail-closed at authoring/import: an invalid/too-complex pattern is rejected before it can persist. The send-path `regex_search` never raises (belt-and-suspenders). `regex=0` is a stable no-op. Migration is idempotent (PRAGMA guard). No new dependency.

## Testing

- **Validator (pure unit):** length-cap rejection; syntax-error rejection; nested-unbounded-quantifier rejection (`(a+)+`, `(a*)*`); trivial identical-alternation rejection (`(a|a)*`); a bounded quantifier (`(\d{3}-)+`) and an ordinary pattern PASS; `regex_search` returns bool and never raises on a bad pattern; IGNORECASE on/off behavior.
- **Matcher:** a regex entry fires on a pattern that a literal match would miss (e.g. `w[ao]rden`); respects `case_sensitive`; a non-regex entry is unchanged (stable no-op); `_entry_matches` and `_classify_entry_match` agree (diagnostics mirror) for a regex entry.
- **Load-time backstop:** a processor built from a book/character-book entry with `regex=True` and an invalid *or* too-complex pattern (simulating an unvalidated `character_book` source) is downgraded to `regex=False` at `_process_entry` — the processed entry has `regex=False`, `process_messages` does not hang, and the entry is matched literally (not as a live regex).
- **Migration:** v21→v22 adds the column + recreates the 2 sync triggers with `regex`; idempotent (re-run no-op); downgrade-replay; fresh DDL == migrated (named-column reads survive ALTER-append).
- **Manager:** create/get/update round-trip `regex`; export includes it; import restores it (default False when absent).
- **Adapter:** maps `regex`; validates regex patterns and raises `ValueError` naming the entry on a bad/too-complex pattern; a non-regex entry with a would-be-bad "pattern" is untouched (not validated).
- **Editor:** the Regex switch round-trips through `entry_form_payload`/`_fill_form_from_entry`; the real screen add-handler persists `regex` (regression against a dropped kwarg); save-time validation blocks an invalid pattern (status set, nothing persisted).
- Full gate: `test_world_info_regex`, `test_world_info`, `test_world_info_diagnostics`, `test_world_book_manager`, `test_world_book_import`, `test_personas_lore`, the ChaChaNotes v21→v22 migration test, plus `import tldw_chatbook.app`.

## Decomposition

Single plan (~6 tasks: validator module; migration; matcher `_key_hits` + regex branch; manager+adapter; editor; gate) — one cohesive feature, P2c-sized. Could split into engine (validator+migration+matcher+manager+import-validation) and surface (editor) if two smaller PRs are preferred.

## Acceptance criteria

- [ ] `world_book_entries` has `regex BOOLEAN DEFAULT 0` on fresh (base DDL) and migrated (v21→v22) DBs, both sync triggers carry `regex`, migration idempotent; `_CURRENT_SCHEMA_VERSION = 22`.
- [ ] `world_info_regex.validate_regex_pattern` rejects over-length, invalid-syntax, and nested-unbounded-quantifier / identical-alternation patterns with a user-facing message; `regex_search` never raises.
- [ ] A `regex=True` entry matches its keys as case-insensitive-by-default regex (no `\b`, no escape); a `regex=False` entry is byte-identical to today; `_entry_matches` and `_classify_entry_match` agree.
- [ ] `_process_entry` downgrades a `regex=True` entry with an invalid/too-complex pattern to `regex=False` at load, so no catastrophic pattern reaches the matcher regardless of source (editor, world-book file, or character card).
- [ ] Manager create/update/get/export/import carry `regex`; the import adapter validates regex patterns fail-closed.
- [ ] The editor has a Regex switch that round-trips and is persisted by the real add-handler; an invalid pattern is rejected at save (status shown, not persisted).
- [ ] Existing (regex=0) data reproduces today's send output (stable no-op); no Markdown/attach/native-send changes.
- [ ] Full gate green; `import tldw_chatbook.app` OK.
