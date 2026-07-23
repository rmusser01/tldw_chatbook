# Roleplay P2d-regex — optional regex matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Lore entry opt into regex matching (`regex` flag) — its keys are matched as case-insensitive-by-default regex patterns — safely, never crashing or hanging the on-UI-loop send path.

**Architecture:** A pure `world_info_regex` validator (fail-closed at save + import) plus a load-time backstop in `_process_entry` (downgrade a bad-pattern entry to literal) form a source-independent safety net; a per-entry `regex` column (v21→v22 migration) drives a single `_key_hits` matcher branch. `regex=0` default is byte-identical to today.

**Tech Stack:** Python ≥3.11 `re` (no new dependency), Textual, pytest, SQLite (`CharactersRAGDB`).

## Global Constraints

- Matching runs on the UI loop, and `re` can't be portably timed out — so safety is: validate fail-closed at save + import, a load-time backstop in `_process_entry`, and a send-path matcher that never raises. NEVER hang or crash the send.
- `regex=0` is a stable no-op: the `regex=False` matcher path is byte-identical to today; all P2a/P2c/P2d pins must still pass unchanged.
- Migration mirrors P2c v20→v21 exactly: idempotent PRAGMA guard, recreate the 2 sync triggers, base DDL == migrated (named-column reads). `_CURRENT_SCHEMA_VERSION = 22`. Re-verify v22 uncontested at merge time.
- The ReDoS heuristic targets UNBOUNDED quantifiers only (`+`, `*`, `{n,}`) — never `?` or bounded `{n}`/`{n,m}`.
- No new dependency.
- Implementers stage ONLY their task's files (`git add <paths>`; never `git add -A`; never `.superpowers/`).
- **Test environment (from the worktree root):**
  ```bash
  HOME=/private/tmp/tldw-chatbook-test-home \
  XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```
  The venv is in the MAIN checkout; `import tldw_chatbook` resolves to the worktree source (cwd on `sys.path`).

## File Structure

- **Create** `tldw_chatbook/Character_Chat/world_info_regex.py` — pure validator + never-raising `regex_search`.
- **Modify** `tldw_chatbook/DB/ChaChaNotes_DB.py` — v21→v22 migration + base DDL.
- **Create** `tldw_chatbook/DB/migrations/chachanotes_v21_to_v22_world_book_entry_regex.sql` — doc-mirror.
- **Modify** `tldw_chatbook/Character_Chat/world_info_processor.py` — `_process_entry` (regex + backstop), `_key_hits`, refactor `_entry_matches`/`_classify_entry_match`.
- **Modify** `tldw_chatbook/Character_Chat/world_book_manager.py` — create/update/get/export/import carry `regex`.
- **Modify** `tldw_chatbook/Character_Chat/world_book_import.py` — normalize + validate `regex`.
- **Modify** `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` + `tldw_chatbook/UI/Screens/personas_screen.py` — Regex switch, payload/fill, save-validation, add-handler.
- Tests: `Tests/Character_Chat/test_world_info_regex.py`, `Tests/DB/test_chachanotes_world_book_regex_migration.py`, and additions to `Tests/Character_Chat/test_world_info_diagnostics.py`, `test_world_book_manager.py`, `test_world_book_import.py`, `Tests/UI/test_personas_lore.py`.

---

### Task 1: `world_info_regex` validator module

**Files:**
- Create: `tldw_chatbook/Character_Chat/world_info_regex.py`
- Test: `Tests/Character_Chat/test_world_info_regex.py`

**Interfaces:**
- Produces: `MAX_REGEX_PATTERN_LENGTH = 500`; `validate_regex_pattern(pattern: str) -> None` (raises `ValueError` with a user-facing message); `regex_search(pattern: str, text: str, ignore_case: bool) -> bool` (never raises).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Character_Chat/test_world_info_regex.py`:

```python
import pytest

from tldw_chatbook.Character_Chat.world_info_regex import (
    validate_regex_pattern,
    regex_search,
    MAX_REGEX_PATTERN_LENGTH,
)


@pytest.mark.parametrize("pattern", ["w[ao]rden", r"https?://", r"(\d{3}-)+\d{4}", "(a|b)*", "hello"])
def test_valid_patterns_pass(pattern):
    validate_regex_pattern(pattern)  # must not raise


def test_over_length_rejected():
    with pytest.raises(ValueError, match="too long"):
        validate_regex_pattern("a" * (MAX_REGEX_PATTERN_LENGTH + 1))


def test_syntax_error_rejected():
    with pytest.raises(ValueError, match="Invalid regex"):
        validate_regex_pattern("(unclosed")


@pytest.mark.parametrize("pattern", ["(a+)+", "(a*)*", "(a+)*$", "(a|a)*"])
def test_catastrophic_patterns_rejected(pattern):
    with pytest.raises(ValueError, match="too complex"):
        validate_regex_pattern(pattern)


def test_regex_search_matches_case_insensitive():
    assert regex_search("w[ao]rden", "The WARDEN speaks", ignore_case=True) is True
    assert regex_search("w[ao]rden", "The WARDEN speaks", ignore_case=False) is False
    assert regex_search("w[ao]rden", "the warden", ignore_case=False) is True


def test_regex_search_never_raises_on_bad_pattern():
    # An uncompilable pattern must return False, not raise.
    assert regex_search("(unclosed", "anything", ignore_case=True) is False


def test_regex_search_returns_bool():
    assert regex_search("x", "no match here", ignore_case=True) is False
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_info_regex.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Character_Chat.world_info_regex'`.

- [ ] **Step 3: Write the module**

Create `tldw_chatbook/Character_Chat/world_info_regex.py`:

```python
"""Fail-closed regex validation + a never-raising matcher for Lore entries.

World-info matching runs on the UI event loop and Python ``re`` cannot be
portably time-bounded, so a catastrophic pattern could freeze the app. Patterns
are validated fail-closed at save and import; the send-path matcher additionally
never raises. The catastrophic-pattern heuristic is best-effort: it flags nested
unbounded-quantifier shapes and trivial identical alternation, but not general
alternation-overlap ReDoS (documented residual risk).
"""
from __future__ import annotations

import re

MAX_REGEX_PATTERN_LENGTH = 500

# An UNBOUNDED quantifier only: +, *, or {n,}. Never ? or bounded {n}/{n,m}.
_UNBOUNDED = r"(?:[*+]|\{\d+,\})"
# (a) a (flat) group whose body contains an unbounded quantifier, itself
#     immediately followed by an unbounded quantifier: (…+…)+ , (…*…)* , …
_NESTED_QUANT_RE = re.compile(r"\([^()]*" + _UNBOUNDED + r"[^()]*\)" + _UNBOUNDED)
# (b) a trivial identical two-way alternation (x|x) followed by an unbounded
#     quantifier: (a|a)* .
_IDENTICAL_ALT_RE = re.compile(r"\(([^()|]+)\|\1\)" + _UNBOUNDED)


def _looks_catastrophic(pattern: str) -> bool:
    return bool(_NESTED_QUANT_RE.search(pattern) or _IDENTICAL_ALT_RE.search(pattern))


def validate_regex_pattern(pattern: str) -> None:
    """Raise ValueError (user-facing) if a pattern is unusable or dangerous.

    Args:
        pattern: The regex pattern string to validate.

    Raises:
        ValueError: If the pattern is too long, has invalid syntax, or matches
            the catastrophic-pattern heuristic.
    """
    if len(pattern) > MAX_REGEX_PATTERN_LENGTH:
        raise ValueError(
            f"Regex pattern is too long (max {MAX_REGEX_PATTERN_LENGTH} characters)."
        )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex: {exc}") from exc
    if _looks_catastrophic(pattern):
        raise ValueError(
            "Regex pattern is too complex (nested quantifiers can hang matching)."
        )


def regex_search(pattern: str, text: str, ignore_case: bool) -> bool:
    """Search ``text`` for ``pattern``; never raises (bad pattern → False).

    Args:
        pattern: The regex pattern.
        text: The text to search.
        ignore_case: Whether to match case-insensitively.

    Returns:
        True if the pattern matches anywhere in the text, else False (also False
        on any error — a bad pattern simply does not fire).
    """
    try:
        return bool(re.search(pattern, text, re.IGNORECASE if ignore_case else 0))
    except Exception:
        return False
```

- [ ] **Step 4: Run to verify it passes**

Run the Step-2 command. Expected: PASS (all parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/world_info_regex.py Tests/Character_Chat/test_world_info_regex.py
git commit -m "feat(lore): fail-closed regex validator + never-raising matcher"
```

---

### Task 2: Schema migration v21→v22 (`regex` column)

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (base DDL `world_book_entries` `:1199-1214` + its 2 sync triggers `:1352-1387`; new `_MIGRATE_V21_TO_V22_SQL` near `:2343`; new `_migrate_from_v21_to_v22` near `:3358`; `_CURRENT_SCHEMA_VERSION` `:143`; `migration_steps` `:3511`)
- Create: `tldw_chatbook/DB/migrations/chachanotes_v21_to_v22_world_book_entry_regex.sql`
- Test: `Tests/DB/test_chachanotes_world_book_regex_migration.py`

**Interfaces:**
- Produces: `world_book_entries.regex BOOLEAN DEFAULT 0` on fresh + migrated DBs; `_CURRENT_SCHEMA_VERSION == 22`.

- [ ] **Step 1: Write the failing migration test**

Create `Tests/DB/test_chachanotes_world_book_regex_migration.py` (mirrors the P2c v20→v21 test):

```python
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_world_book_entries_regex_migrate_v21_to_v22(tmp_path):
    db_path = tmp_path / "chacha.sqlite"
    db = CharactersRAGDB(str(db_path), client_id="test-client")
    conn = db.get_connection()
    # Simulate a V21-shaped DB: drop the V22 additions.
    conn.execute("DROP TRIGGER IF EXISTS world_book_entries_sync_create")
    conn.execute("DROP TRIGGER IF EXISTS world_book_entries_sync_update")
    conn.execute("ALTER TABLE world_book_entries DROP COLUMN regex")
    conn.execute(
        "UPDATE db_schema_version SET version = 21 WHERE schema_name = ?",
        (db._SCHEMA_NAME,),
    )
    conn.commit()
    db.close_connection()

    migrated = CharactersRAGDB(str(db_path), client_id="test-client")
    mconn = migrated.get_connection()
    version = mconn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (migrated._SCHEMA_NAME,),
    ).fetchone()
    assert version["version"] == migrated._CURRENT_SCHEMA_VERSION == 22
    cols = {r[1] for r in mconn.execute("PRAGMA table_info(world_book_entries)").fetchall()}
    assert "regex" in cols
    for trig in ("world_book_entries_sync_create", "world_book_entries_sync_update"):
        sql = mconn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (trig,)
        ).fetchone()["sql"]
        assert "regex" in sql


def test_fresh_db_has_regex_column_and_triggers(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "fresh.sqlite"), client_id="test-client")
    conn = db.get_connection()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(world_book_entries)").fetchall()}
    assert "regex" in cols
    create_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'world_book_entries_sync_create'"
    ).fetchone()["sql"]
    assert "regex" in create_sql
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/DB/test_chachanotes_world_book_regex_migration.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `_CURRENT_SCHEMA_VERSION` is 21 (fresh test) / the DROP COLUMN + reopen doesn't migrate to 22.

- [ ] **Step 3: Bump the version + register the step**

In `ChaChaNotes_DB.py`, change `:143`:
```python
    _CURRENT_SCHEMA_VERSION = 22  # Adds world_book_entries.regex (P2d-regex).
```
In `migration_steps` (`:3494-3511` region), add after the `20:` entry:
```python
                    20: self._migrate_from_v20_to_v21,
                    21: self._migrate_from_v21_to_v22,
```

- [ ] **Step 4: Add the migration SQL constant + method**

Add next to `_MIGRATE_V20_TO_V21_SQL` (after it, ~`:2389`):
```python
    _MIGRATE_V21_TO_V22_SQL = """
DROP TRIGGER IF EXISTS world_book_entries_sync_create;
DROP TRIGGER IF EXISTS world_book_entries_sync_update;

CREATE TRIGGER world_book_entries_sync_create
AFTER INSERT ON world_book_entries BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('world_book_entries', CAST(NEW.id AS TEXT), 'create', NEW.last_modified,
         (SELECT client_id FROM world_books WHERE id = NEW.world_book_id), 1,
         json_object('id', NEW.id, 'world_book_id', NEW.world_book_id, 'keys', NEW.keys,
                     'content', NEW.content, 'enabled', NEW.enabled, 'position', NEW.position,
                     'insertion_order', NEW.insertion_order, 'priority', NEW.priority,
                     'selective', NEW.selective, 'secondary_keys', NEW.secondary_keys,
                     'case_sensitive', NEW.case_sensitive, 'regex', NEW.regex,
                     'extensions', NEW.extensions,
                     'created_at', NEW.created_at, 'last_modified', NEW.last_modified));
END;

CREATE TRIGGER world_book_entries_sync_update
AFTER UPDATE ON world_book_entries
WHEN OLD.keys IS NOT NEW.keys OR
     OLD.content IS NOT NEW.content OR
     OLD.enabled IS NOT NEW.enabled OR
     OLD.position IS NOT NEW.position OR
     OLD.insertion_order IS NOT NEW.insertion_order OR
     OLD.priority IS NOT NEW.priority OR
     OLD.selective IS NOT NEW.selective OR
     OLD.secondary_keys IS NOT NEW.secondary_keys OR
     OLD.case_sensitive IS NOT NEW.case_sensitive OR
     OLD.regex IS NOT NEW.regex OR
     OLD.extensions IS NOT NEW.extensions
BEGIN
  INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
  VALUES('world_book_entries', CAST(NEW.id AS TEXT), 'update', NEW.last_modified,
         (SELECT client_id FROM world_books WHERE id = NEW.world_book_id), 1,
         json_object('id', NEW.id, 'world_book_id', NEW.world_book_id, 'keys', NEW.keys,
                     'content', NEW.content, 'enabled', NEW.enabled, 'position', NEW.position,
                     'insertion_order', NEW.insertion_order, 'priority', NEW.priority,
                     'selective', NEW.selective, 'secondary_keys', NEW.secondary_keys,
                     'case_sensitive', NEW.case_sensitive, 'regex', NEW.regex,
                     'extensions', NEW.extensions,
                     'created_at', NEW.created_at, 'last_modified', NEW.last_modified));
END;

UPDATE db_schema_version
   SET version = 22
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 21;
"""
```
Add the method next to `_migrate_from_v20_to_v21` (after it, ~`:3393`):
```python
    def _migrate_from_v21_to_v22(self, conn: sqlite3.Connection):
        """Migrate schema V21→V22: add ``regex`` to ``world_book_entries`` and
        redefine the sync triggers so edits to it reach ``sync_log``."""
        logger.info(f"Migrating schema from V21 to V22 for '{self._SCHEMA_NAME}' in DB: {self.db_path_str}...")
        try:
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(world_book_entries)").fetchall()
            }
            if "regex" not in existing_columns:
                conn.execute("ALTER TABLE world_book_entries ADD COLUMN regex BOOLEAN DEFAULT 0")
            conn.executescript(self._MIGRATE_V21_TO_V22_SQL)
            logger.debug(f"[{self._SCHEMA_NAME} V21→V22] Migration script executed.")
            final_version = self._get_db_version(conn)
            if final_version != 22:
                raise SchemaError(
                    f"[{self._SCHEMA_NAME} V21→V22] Migration version check failed. Expected 22, got: {final_version}"
                )
            logger.info(f"[{self._SCHEMA_NAME} V21→V22] Migration completed successfully for DB: {self.db_path_str}.")
        except sqlite3.Error as e:
            logger.opt(exception=True).error(f"[{self._SCHEMA_NAME} V21→V22] Migration failed: {e}")
            raise SchemaError(f"Migration from V21 to V22 failed for '{self._SCHEMA_NAME}': {e}") from e
        except Exception as e:
            logger.opt(exception=True).error(f"[{self._SCHEMA_NAME} V21→V22] Unexpected error during migration: {e}")
            raise SchemaError(f"Unexpected error migrating from V21 to V22 for '{self._SCHEMA_NAME}': {e}") from e
```

- [ ] **Step 5: Update the base DDL (fresh == migrated)**

In the base DDL `world_book_entries` table (`:1210`), add `regex` after `case_sensitive`:
```python
  case_sensitive  BOOLEAN  DEFAULT 0,
  regex           BOOLEAN  DEFAULT 0,
  extensions      TEXT,    -- JSON for future extensibility
```
In the base DDL `world_book_entries_sync_create` trigger (`:1361`) and `world_book_entries_sync_update` trigger (`:1385`), add `'regex', NEW.regex,` to both `json_object(...)` payloads (right after `'case_sensitive', NEW.case_sensitive,`), and add `OLD.regex IS NOT NEW.regex OR` to the update trigger's `WHEN` clause (right after the `case_sensitive` line, `:1375`). (Use the exact text from `_MIGRATE_V21_TO_V22_SQL` above so fresh == migrated.)

- [ ] **Step 6: Add the doc-mirror SQL file**

Create `tldw_chatbook/DB/migrations/chachanotes_v21_to_v22_world_book_entry_regex.sql` with:
```sql
-- Keep this aligned with ChaChaNotes_DB._MIGRATE_V21_TO_V22_SQL.
-- Adds world_book_entries.regex (opt-in regex matching, P2d-regex).
ALTER TABLE world_book_entries ADD COLUMN regex BOOLEAN DEFAULT 0;
```
(followed by the same two `DROP TRIGGER` + `CREATE TRIGGER` blocks and the `UPDATE db_schema_version` from `_MIGRATE_V21_TO_V22_SQL`.)

- [ ] **Step 7: Run to verify it passes**

Run the Step-2 command. Expected: PASS (2 tests). Then confirm no other DB tests regressed:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/DB/test_chachanotes_world_book_priority_migration.py Tests/DB/test_chachanotes_world_book_regex_migration.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/DB/migrations/chachanotes_v21_to_v22_world_book_entry_regex.sql Tests/DB/test_chachanotes_world_book_regex_migration.py
git commit -m "feat(db): schema v22 — add world_book_entries.regex + sync triggers"
```

---

### Task 3: Matcher regex branch + load-time backstop

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_info_processor.py` (`_process_entry`; new `_key_hits`; `_entry_matches`; `_classify_entry_match`)
- Test: `Tests/Character_Chat/test_world_info_diagnostics.py`

**Interfaces:**
- Consumes: `world_info_regex.validate_regex_pattern`, `regex_search` (Task 1).
- Produces: processed entries carry `'regex'` (bool, downgraded to False if any pattern is bad); `_key_hits(entry, key, scan_text, scan_text_lower) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Character_Chat/test_world_info_diagnostics.py` (uses the existing `_book`/`_entry` helpers):

```python
def test_regex_entry_fires_on_pattern_a_literal_would_miss():
    book = _book(1, "B", [_entry(1, ["w[ao]rden"], "grim jailer", regex=True)])
    proc = WorldInfoProcessor(world_books=[book])
    result = proc.process_messages("The Wardon appears.", [])  # matches w[ao]rden
    assert any("grim jailer" in c for c in result["injections"]["before_char"])
    # A literal (non-regex) entry with the same key would NOT match "Wardon".
    lit = WorldInfoProcessor(world_books=[_book(1, "B", [_entry(1, ["w[ao]rden"], "x")])])
    assert lit.process_messages("The Wardon appears.", [])["matched_entries"] == []


def test_regex_entry_respects_case_sensitive():
    book = _book(1, "B", [_entry(1, ["WARD.N"], "c", regex=True, case_sensitive=True)])
    proc = WorldInfoProcessor(world_books=[book])
    assert proc.process_messages("the warden", [])["matched_entries"] == []
    assert proc.process_messages("the WARDEN", [])["matched_entries"]


def test_regex_backstop_downgrades_bad_pattern_to_literal():
    # Simulates an unvalidated source (e.g. a character_book) with a catastrophic
    # pattern: the processor downgrades it to literal so matching can't hang.
    book = _book(1, "B", [_entry(1, ["(a+)+"], "content", regex=True)])
    proc = WorldInfoProcessor(world_books=[book])
    assert proc.entries[0]["regex"] is False           # downgraded at load
    # Matched literally: the literal "(a+)+" won't appear in normal text.
    assert proc.process_messages("aaaaaaaaaa!", [])["matched_entries"] == []
    # It DOES match the literal pattern text (proves literal matching, no hang).
    assert proc.process_messages("see (a+)+ here", [])["matched_entries"]


def test_non_regex_entry_unchanged_stable_no_op():
    book = _book(1, "B", [_entry(1, ["Warden"], "grim jailer")])  # regex defaults off
    proc = WorldInfoProcessor(world_books=[book])
    assert proc.process_messages("The Warden.", [])["matched_entries"]
    assert proc.process_messages("nothing here", [])["matched_entries"] == []
```

Note: the `_entry` helper already spreads `**kw`, so `regex=True`/`case_sensitive=True` flow through.

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_info_diagnostics.py -k "regex or stable_no_op" \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL (regex not honored; backstop absent).

- [ ] **Step 3: Add the import + `_process_entry` regex + backstop**

In `world_info_processor.py`, add the import at the top (after `import re`):
```python
from tldw_chatbook.Character_Chat.world_info_regex import validate_regex_pattern, regex_search
```
In `_process_entry`, add `regex` to the returned dict and a load-time backstop. Change the return to compute `regex` first:
```python
        regex_flag = bool(entry.get('regex', False))
        if regex_flag:
            # Load-time backstop: an invalid/too-complex pattern from ANY source
            # (editor, import, or a character_book) is downgraded to literal so
            # matching can never hang the on-loop send path.
            try:
                for pat in list(keys) + list(secondary_keys):
                    validate_regex_pattern(pat)
            except ValueError:
                regex_flag = False
        return {
            'keys': keys,
            'secondary_keys': secondary_keys,
            'content': entry.get('content', ''),
            'selective': entry.get('selective', False),
            'position': entry.get('position', 'before_char'),
            'insertion_order': _coerce_int(entry.get('insertion_order'), 0),
            'case_sensitive': entry.get('case_sensitive', False),
            'extensions': entry.get('extensions', {}),
            'priority': _coerce_int(entry.get('priority'), 0),
            'regex': regex_flag,
        }
```
(Keep the existing `keys`/`secondary_keys`/`_coerce_int` lines above the return exactly as they are; only add the `regex_flag` block and the `'regex': regex_flag,` entry.)

- [ ] **Step 4: Add `_key_hits` and route both matchers through it**

Add the helper (place it right above `_entry_matches`):
```python
    def _key_hits(self, entry: Dict[str, Any], key: str, scan_text: str, scan_text_lower: str) -> bool:
        """Does one key match the scan text? Single branch point for literal vs
        regex so _entry_matches and _classify_entry_match cannot drift."""
        if entry.get('regex', False):
            return regex_search(key, scan_text, ignore_case=not entry.get('case_sensitive', False))
        if entry.get('case_sensitive', False):
            return self._keyword_in_text(key, scan_text)
        return self._keyword_in_text(key.lower(), scan_text_lower)
```
Refactor `_entry_matches` — replace the primary-keys loop and the secondary-keys loop so each uses `_key_hits`:
```python
        primary_match = False
        for key in entry['keys']:
            if self._key_hits(entry, key, scan_text, scan_text_lower):
                primary_match = True
                break
        if not primary_match:
            return False
        if not entry.get('selective', False):
            return True
        if not entry['secondary_keys']:
            return True
        for key in entry['secondary_keys']:
            if self._key_hits(entry, key, scan_text, scan_text_lower):
                return True
        return False
```
Refactor `_classify_entry_match`'s `hit(key)` closure:
```python
        def hit(key):
            return self._key_hits(entry, key, scan_text, scan_text_lower)
```
(Remove the now-unused `case = entry.get('case_sensitive', False)` line in `_classify_entry_match`.) Leave `_keyword_in_text` unchanged.

- [ ] **Step 5: Run to verify it passes + no regressions**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_info_diagnostics.py Tests/Character_Chat/test_world_info.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass — the new regex tests AND every existing pin (stable no-op at `regex=0`).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Character_Chat/world_info_processor.py Tests/Character_Chat/test_world_info_diagnostics.py
git commit -m "feat(lore): regex matcher branch (_key_hits) + load-time backstop"
```

---

### Task 4: Manager + import/export carry `regex`

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_book_manager.py` (create sig+INSERT `:306-385`, get SELECT+row `:401-434`, update loop `:454-455`, export `:610-622`, import `:655-668`)
- Modify: `tldw_chatbook/Character_Chat/world_book_import.py` (`_normalize_entry`)
- Test: `Tests/Character_Chat/test_world_book_manager.py`, `Tests/Character_Chat/test_world_book_import.py`

**Interfaces:**
- Consumes: `world_info_regex.validate_regex_pattern` (Task 1).
- Produces: `create_world_book_entry(..., regex: bool = False)`; `get_world_book_entries` rows include `'regex'`; export/import carry `regex`; `normalize_world_book_import` maps + validates `regex`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Character_Chat/test_world_book_manager.py`:
```python
def test_entry_regex_round_trips(wb_manager):
    book_id = wb_manager.create_world_book("B")
    eid = wb_manager.create_world_book_entry(book_id, ["w[ao]rden"], "c", regex=True)
    assert wb_manager.get_world_book_entries(book_id)[0]["regex"] is True
    wb_manager.update_world_book_entry(eid, regex=False)
    assert wb_manager.get_world_book_entries(book_id)[0]["regex"] is False


def test_entry_regex_defaults_false(wb_manager):
    book_id = wb_manager.create_world_book("B")
    wb_manager.create_world_book_entry(book_id, ["k"], "c")
    assert wb_manager.get_world_book_entries(book_id)[0]["regex"] is False


def test_export_import_preserves_regex(wb_manager):
    book_id = wb_manager.create_world_book("B")
    wb_manager.create_world_book_entry(book_id, ["w[ao]rden"], "c", regex=True)
    data = wb_manager.export_world_book(book_id)
    assert data["entries"][0]["regex"] is True
    new_id = wb_manager.import_world_book(data, name_override="B copy")
    assert wb_manager.get_world_book_entries(new_id)[0]["regex"] is True
```

Append to `Tests/Character_Chat/test_world_book_import.py`:
```python
def test_regex_flag_normalized():
    e = normalize_world_book_import({"entries": [{"keys": ["w[ao]rden"], "content": "c", "regex": True}]})["entries"][0]
    assert e["regex"] is True


def test_bad_regex_pattern_rejects_file():
    with pytest.raises(ValueError, match="Entry 1"):
        normalize_world_book_import({"entries": [{"keys": ["(a+)+"], "content": "c", "regex": True}]})


def test_bad_pattern_ignored_when_not_regex():
    # A would-be-bad "pattern" in a non-regex entry is a literal keyword — never validated.
    e = normalize_world_book_import({"entries": [{"keys": ["(a+)+"], "content": "c"}]})["entries"][0]
    assert e["keys"] == ["(a+)+"] and e["regex"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_book_manager.py Tests/Character_Chat/test_world_book_import.py \
-k "regex" -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL (regex not carried / not validated).

- [ ] **Step 3: Thread `regex` through the manager**

In `world_book_manager.py`:
- `create_world_book_entry` signature (`:312`): add `regex: bool = False` after `priority: int = 0`:
  ```python
                                 priority: int = 0,
                                 regex: bool = False) -> int:
  ```
- INSERT query (`:366-369`): add `regex` to the column list and one more `?`:
  ```python
        INSERT INTO world_book_entries (world_book_id, keys, content, enabled, position,
                                       insertion_order, selective, secondary_keys,
                                       case_sensitive, extensions, priority, regex)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  ```
  and add the param after `_coerce_int(priority, 0)` (`:384`):
  ```python
                _coerce_int(priority, 0),
                bool(regex)
  ```
- `get_world_book_entries` SELECT (`:402-404`): append `, regex`:
  ```python
        SELECT id, world_book_id, keys, content, enabled, position, insertion_order,
               selective, secondary_keys, case_sensitive, extensions, created_at, last_modified,
               priority, regex
  ```
  and add to the row dict (`:433`):
  ```python
                    'priority': row[13],
                    'regex': bool(row[14])
  ```
- `update_world_book_entry` field loop (`:454-455`): add `'regex'` to the list:
  ```python
        for field in ['keys', 'content', 'enabled', 'position', 'insertion_order',
                     'selective', 'secondary_keys', 'case_sensitive', 'extensions', 'priority', 'regex']:
  ```
  (`regex` is a plain bool — no special coercion branch needed.)
- `export_world_book` entry dict (`:621`): add:
  ```python
                'priority': entry['priority'],
                'regex': entry['regex']
  ```
- `import_world_book` call (`:667`): add:
  ```python
                priority=entry.get('priority', 0),
                regex=entry.get('regex', False)
  ```

- [ ] **Step 4: Map + validate `regex` in the import adapter**

In `world_book_import.py`, add the import at the top (near the other imports):
```python
from tldw_chatbook.Character_Chat.world_info_regex import validate_regex_pattern
```
In `_normalize_entry`, after computing `keys`, `content`, `raw_secondary`, and before building the return dict, compute `regex` and validate patterns:
```python
    regex_on = _coerce_bool(entry.get("regex"), False)
    if regex_on:
        for pat in list(keys) + _as_str_list(raw_secondary):
            try:
                validate_regex_pattern(pat)
            except ValueError as exc:
                raise ValueError(f"Entry {index + 1}: {exc}") from exc
```
and add `"regex": regex_on,` to the returned dict.

- [ ] **Step 5: Run to verify they pass**

Run the Step-2 command. Expected: PASS. Then run both full files:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_book_manager.py Tests/Character_Chat/test_world_book_import.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Character_Chat/world_book_manager.py tldw_chatbook/Character_Chat/world_book_import.py Tests/Character_Chat/test_world_book_manager.py Tests/Character_Chat/test_world_book_import.py
git commit -m "feat(lore): thread regex through manager + import/export (fail-closed on import)"
```

---

### Task 5: Editor Regex switch + save-time validation

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` (compose matching row; `entry_form_payload`; `_fill_form_from_entry`; `_add_pressed`/`_update_pressed`)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (`_handle_lore_entry_add`)
- Test: `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: `world_info_regex.validate_regex_pattern` (Task 1); `entry_form_payload` now includes `regex`.
- Produces: `#personas-lore-entry-regex` Switch; payload key `regex`; the add-handler persists it.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_personas_lore.py` (`Switch` already imported):
```python
@pytest.mark.asyncio
async def test_regex_switch_round_trips_through_form():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "w[ao]rden"
        app.query_one("#personas-lore-entry-content", TextArea).text = "grim jailer"
        app.query_one("#personas-lore-entry-regex", Switch).value = True
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        assert app.posted[-1]["regex"] is True


@pytest.mark.asyncio
async def test_invalid_regex_pattern_blocks_save():
    app = _DetailHost()
    async with app.run_test(size=(140, 40)) as pilot:
        widget = app.query_one(PersonasLoreDetailWidget)
        widget.load_book({"id": 1, "name": "B", "description": "", "scan_depth": 3,
                          "token_budget": 500, "recursive_scanning": False, "enabled": True})
        app.query_one("#personas-lore-entry-keys", Input).value = "(a+)+"
        app.query_one("#personas-lore-entry-content", TextArea).text = "x"
        app.query_one("#personas-lore-entry-regex", Switch).value = True
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        assert not app.posted   # nothing posted
        status = str(app.query_one("#personas-lore-status", Static).renderable)
        assert "too complex" in status
```

Append the real-DB regression (uses `LorePersonasTestApp`, after the priority one):
```python
@pytest.mark.asyncio
async def test_add_entry_persists_regex_through_real_screen_handler(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        screen.query_one("#personas-lore-entry-keys", Input).value = "w[ao]rden"
        screen.query_one("#personas-lore-entry-content", TextArea).text = "a pale spirit"
        screen.query_one("#personas-lore-entry-regex", Switch).value = True
        await pilot.pause()
        await pilot.click("#personas-lore-entry-add")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        entries = WorldBookManager(lore_db).get_world_book_entries(seeded_lore_book["book_id"])
        added = next(e for e in entries if e["keys"] == ["w[ao]rden"])
        assert added["regex"] is True
```

- [ ] **Step 2: Run to verify they fail**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -k "regex" -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `#personas-lore-entry-regex` does not exist.

- [ ] **Step 3: Add the Regex switch + payload/fill + save-validation**

In `personas_lore_detail.py`:
- Add the import at the top:
  ```python
  from tldw_chatbook.Character_Chat.world_info_regex import validate_regex_pattern
  ```
- In compose, add the Regex switch in **its own** `Horizontal` row directly below the existing matching row (the `Horizontal` containing `#personas-lore-entry-case-sensitive` + `#personas-lore-entry-selective`) — its own row avoids crowding three labeled switches at the 2050×1240 QA size (the fr-width-wrap lesson):
  ```python
                    with Horizontal(classes="personas-lore-form-row"):
                        yield Static("Regex", markup=False)
                        yield Switch(value=False, id="personas-lore-entry-regex")
  ```
  (Place it after the case-sensitive/selective `Horizontal` and before the secondary-keys `Input`.)
- In `entry_form_payload`, after reading `selective`, read regex and add to the dict:
  ```python
        regex = bool(self.query_one("#personas-lore-entry-regex", Switch).value)
  ```
  and add `"regex": regex,` to the returned dict.
- In `_fill_form_from_entry`, populate it:
  ```python
        self.query_one("#personas-lore-entry-regex", Switch).value = bool(entry.get("regex", False))
  ```
- Add a save-validation helper and call it from both pressed handlers:
  ```python
      def _regex_payload_error(self, payload: dict) -> str | None:
          """Return a user-facing error if the payload is a regex entry with an
          invalid/too-complex key or secondary-key pattern, else None."""
          if not payload.get("regex"):
              return None
          for pat in list(payload.get("keys", [])) + list(payload.get("secondary_keys", [])):
              try:
                  validate_regex_pattern(pat)
              except ValueError as exc:
                  return str(exc)
          return None
  ```
  In `_add_pressed`, after `payload = self.entry_form_payload()` and the `if payload is not None:` guard, before posting:
  ```python
          if payload is not None:
              err = self._regex_payload_error(payload)
              if err:
                  self.set_status(err)
                  return
              payload["insertion_order"] = self._next_insertion_order()
              self.post_message(LoreEntryAddRequested(payload))
  ```
  In `_update_pressed`, likewise guard before posting:
  ```python
          payload = self.entry_form_payload()
          if payload is not None:
              err = self._regex_payload_error(payload)
              if err:
                  self.set_status(err)
                  return
              self.post_message(LoreEntryUpdateRequested(entry_id, payload))
  ```

- [ ] **Step 4: Thread `regex` through the screen add-handler**

In `personas_screen.py`, `_handle_lore_entry_add`'s `create_world_book_entry` call, add the explicit kwarg (beside `case_sensitive=...`):
```python
                selective=payload.get("selective", False),
                secondary_keys=payload.get("secondary_keys", []),
                case_sensitive=payload.get("case_sensitive", False),
                regex=payload.get("regex", False),
```
`_handle_lore_entry_update` needs no change (`**payload`).

- [ ] **Step 5: Run to verify they pass + full file**

Run the Step-2 command (PASS), then the whole file:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): Regex switch in the entry editor with save-time validation"
```

---

### Task 6: Full gate + spec status

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-18-roleplay-p2d-regex-design.md` (status → Implemented)

- [ ] **Step 1: Run the full gate**

```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_info_regex.py Tests/Character_Chat/test_world_info.py \
Tests/Character_Chat/test_world_info_diagnostics.py Tests/Character_Chat/test_world_book_manager.py \
Tests/Character_Chat/test_world_book_import.py Tests/UI/test_personas_lore.py \
Tests/DB/test_chachanotes_world_book_regex_migration.py Tests/DB/test_chachanotes_world_book_priority_migration.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass. Then the import smoke:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('APP IMPORT OK')"
```
Expected: `APP IMPORT OK`.

- [ ] **Step 2: Flip the spec status + commit**

Change the spec's `**Status:** Design.` to `**Status:** Implemented (P2d-regex).`
```bash
git add Docs/superpowers/specs/2026-07-18-roleplay-p2d-regex-design.md
git commit -m "docs(roleplay): mark P2d-regex spec implemented"
```

---

## Notes for the reviewer

- **The safety story is source-independent:** authoring validation (Task 5) + import validation (Task 4) give immediate feedback, but the load-time backstop in `_process_entry` (Task 3) is what guarantees no catastrophic pattern reaches the on-loop matcher regardless of source (editor, file import, or a character card). The backstop test (Task 3 Step 1) must pass.
- **`regex=0` must stay byte-identical:** the `_key_hits` literal branch reproduces today's exact logic (lowercase key + text when not case-sensitive). Every pre-existing world-info pin must pass unchanged.
- **Migration mirrors P2c exactly:** idempotent PRAGMA guard, both sync triggers recreated with `regex`, base DDL == migrated. Re-verify `v22` is uncontested at merge time.
- **Heuristic is best-effort by design** (documented residual risk): it targets unbounded quantifiers only and does not catch general alternation-overlap. Do not "fix" it into rejecting safe bounded patterns.
- Any change to `world_book_manager.py`'s existing coercion or `world_info_processor.py`'s literal path beyond what's specified is out of scope.
