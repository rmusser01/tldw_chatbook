# Bulk Prompt Chatbook Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export every active local Prompt and Recipe through the existing Library Chatbook workflow and import it losslessly into a fresh profile without restoring source identity, source history, collections, or timestamp values.

**Architecture:** Keep Chatbook v1 and add one strict versioned Prompt-record codec shared by the existing creator/importer. Add one privacy-safe, coherent Prompt export snapshot query to `PromptsDatabase`, extend the existing uncapped Library export scope with Prompts, and restore one Prompt-toolbar button that opens the existing export canvas. Prompt collection is all-or-nothing and uses a dedicated sanitized failure path before the creator's legacy broad exception handler.

**Tech Stack:** Python 3.11+, stdlib `json`/`sqlite3`/`zipfile`, existing `PromptsDatabase`, Chatbook v1 dataclasses, Textual 8.x, pytest/Hypothesis, Ruff, mypy.

**ADR required:** yes
**ADR path:** `backlog/decisions/057-portable-chatbook-prompt-records.md`
**Reason:** The task establishes a durable portable Prompt schema, backward-compatible dispatch, privacy/identity exclusions, and a cross-module all-or-nothing export contract.

---

## File map

- Create `tldw_chatbook/Prompt_Management/prompt_chatbook_record.py`: pure strict encoder/decoder and bounded validation exception.
- Create `Tests/Prompt_Management/test_prompt_chatbook_record.py`: truth table, property coverage, legacy compatibility, and repr/privacy tests for the codec.
- Modify `tldw_chatbook/DB/Prompts_DB.py`: uncapped active Prompt IDs plus coherent privacy-safe Prompt export snapshots.
- Create `Tests/Prompts_DB/test_prompts_db_chatbook_export.py`: real file-backed SQLite coverage for snapshot consistency, deleted exclusion, ordering, and privacy.
- Modify `tldw_chatbook/Chatbooks/Chatbook_Creator.py`: archive-local Prompt identities, lossless collection, all-or-nothing failure, and sanitized creator settlement.
- Modify `tldw_chatbook/Chatbooks/Chatbook_Importer.py`: strict new/legacy decode, ordinary destination mutation, bounded failures, and no content-bearing logs.
- Create `Tests/Chatbooks/test_chatbook_prompt_round_trip.py`: real ZIP/source/destination round trips and mutation/failure boundaries.
- Modify `Tests/Chatbooks/test_chatbook_creator.py`: focused sanitized creator failure tests where cheaper than a full ZIP fixture.
- Modify `Tests/Chatbooks/test_chatbook_importer.py`: focused legacy/invalid import status tests where incumbent fixtures are reusable.
- Modify `tldw_chatbook/Library/library_export_scope.py`: `prompts` scope, fourth count, uncapped selection, and truthful labels.
- Modify `Tests/Library/test_library_export_scope.py`: Prompt-only/Everything/count/selection/label contracts.
- Modify `Tests/Library/test_library_export_scope_ids.py`: explicit Prompt-ID mapping and no unrelated database access.
- Modify `Tests/Library/test_library_export_roundtrip.py`: update every resolver caller and prove Everything includes a real Prompt through service/ZIP/import.
- Modify `tldw_chatbook/Library/library_export_state.py`: four-source documentation and Prompt non-media form expectations only; no new state model.
- Modify `Tests/Library/test_library_export_state.py`: Prompt scope rendering and media-quality hiding.
- Modify `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`: restore the compact `Export…` toolbar button.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: Prompt DB resolution, four-source count/selection calls, privacy-safe recovery, Prompt export handler, and focus identity.
- Modify `Tests/Library/test_library_export_execution.py`: four-source request payload and failure-copy boundaries.
- Modify `Tests/UI/test_library_prompts_canvas.py`: mounted Prompt export, server refusal, stale/error/cancel/retry, focus, and compositor geometry.
- Modify `Docs/User_Guide/library/prompts.md`: bulk workflow, fields, compatibility, local-only behavior, and exclusions.
- Modify `Docs/User_Guide/library/import-and-export.md`: replace the obsolete three-source summary/label with the four-source contract.
- Modify `Docs/security/production-diagnostic-inventory.json`: review only changed Chatbook/Library diagnostic owner digests when call sites change.
- Modify `backlog/tasks/task-197 - Bulk-export-of-prompts-via-the-chatbook-format.md`: implementation notes, checked acceptance criteria, and Done status only after final verification.

## Constraints for every task

- Use `@superpowers:test-driven-development`: add focused RED evidence before production edits.
- Use `@ponytail` full: no new modal/controller/service/dependency, no Chatbook container bump, and no generic archive framework.
- Preserve source Prompt text literally; never place it in widget IDs, paths, exceptions, logs, or test node IDs.
- Use real file-backed SQLite for transaction/snapshot and archive round-trip tests.
- Do not weaken or bypass the existing server-mode, cancellation, overwrite, or atomic archive boundaries.
- Commit each task only after focused GREEN, Ruff on owned files, `py_compile` for changed production, and `git diff --check`.

### Task 1: Strict portable Prompt-record codec

**Files:**
- Create: `tldw_chatbook/Prompt_Management/prompt_chatbook_record.py`
- Create: `Tests/Prompt_Management/test_prompt_chatbook_record.py`

- [x] **Step 1: Write the new-record and legacy RED truth table**

Cover exact `None`/empty/non-empty lanes, Unicode/RTL/emoji/literal markup, Prompt/Recipe, structured/legacy/foreign/malformed definitions, keyword order, and the exact output key set. Add legacy cases with required `name`/`description`/`content` plus optional exact-int `id` and string-or-null timestamps.

```python
def test_new_record_preserves_exact_portable_fields() -> None:
    encoded = encode_chatbook_prompt_record(DETAIL)
    assert tuple(encoded) == CHATBOOK_PROMPT_RECORD_KEYS
    assert encoded["system_prompt"] == "[bold]\n研究🙂"
    assert decode_chatbook_prompt_record(encoded) == EXPECTED_ADD_PROMPT_FIELDS


def test_legacy_record_ignores_known_identity_metadata() -> None:
    decoded = decode_chatbook_prompt_record({
        "id": 42,
        "name": "Legacy",
        "description": "Old",
        "content": "System only",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": None,
    })
    assert decoded["system_prompt"] == "System only"
    assert decoded["user_prompt"] is None
    assert "id" not in decoded
```

- [x] **Step 2: Run the codec RED**

Run: `../../.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_chatbook_record.py -q --tb=short`
Expected: collection fails because `prompt_chatbook_record` does not exist.

- [x] **Step 3: Add adversarial validation/privacy REDs**

Parameterize missing/one-sided markers, unknown/bool versions, extra keys, blank names, invalid optional types, invalid enums, bool schema version, invalid keyword container/items, and unknown legacy keys. Assert the public exception has fixed `str`/`repr` and caplog does not contain body/definition sentinels. Add bounded Hypothesis generation for accepted strings and rejected non-string leaf values.

- [x] **Step 4: Implement the minimal pure codec**

```python
CHATBOOK_PROMPT_RECORD_SCHEMA = "tldw-chatbook-prompt"
CHATBOOK_PROMPT_RECORD_VERSION = 1
CHATBOOK_PROMPT_RECORD_KEYS = (...)


class PromptChatbookRecordError(ValueError):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__("Invalid Chatbook Prompt record.")


def encode_chatbook_prompt_record(detail: Mapping[str, Any]) -> dict[str, Any]:
    record = {key: detail.get(key) for key in PORTABLE_DETAIL_KEYS}
    record["record_schema"] = CHATBOOK_PROMPT_RECORD_SCHEMA
    record["record_version"] = CHATBOOK_PROMPT_RECORD_VERSION
    return _validate_new_record(record)


def decode_chatbook_prompt_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "record_schema" in payload or "record_version" in payload:
        return _to_add_prompt_fields(_validate_new_record(payload))
    return _decode_legacy(payload)
```

Keep `_validate_new_record`, `_to_add_prompt_fields`, and `_decode_legacy` as small helpers. The encoder returns the canonical versioned record; the decoder returns only `add_prompt` fields. Never parse or reserialize `prompt_definition`.

- [x] **Step 5: Run focused GREEN and mutation checks**

Run: `../../.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_chatbook_record.py -q --tb=short`
Expected: all tests pass. Temporarily bypass version dispatch and the bool/int guard; each mutation must make a focused test fail, then restore.

- [x] **Step 6: Static checks and commit**

Run:

```bash
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Prompt_Management/prompt_chatbook_record.py \
  Tests/Prompt_Management/test_prompt_chatbook_record.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Prompt_Management/prompt_chatbook_record.py \
  Tests/Prompt_Management/test_prompt_chatbook_record.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/Prompt_Management/prompt_chatbook_record.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Prompt_Management/prompt_chatbook_record.py
git diff --check
```

Expected: each exits 0.
Commit: `feat(prompts): define portable Chatbook records`

### Task 2: Coherent privacy-safe Prompt database snapshots

**Files:**
- Modify: `tldw_chatbook/DB/Prompts_DB.py`
- Create: `Tests/Prompts_DB/test_prompts_db_chatbook_export.py`

- [x] **Step 1: Write uncapped active-ID RED tests**

Seed 207 active Prompts/Recipes plus deleted controls in a real file-backed database. Assert one ordered uncapped result and exact-int IDs.

```python
assert db.get_all_active_prompt_ids() == expected_active_ids
assert deleted_id not in db.get_all_active_prompt_ids()
```

- [x] **Step 2: Write coherent snapshot/privacy RED tests**

Assert `fetch_prompt_chatbook_snapshot(id)` returns all portable columns plus canonical active keywords, rejects deleted/missing rows with bounded copy, and emits no source-ID/body/exception sentinel or traceback. Use WAL plus a second connection/trace callback to prove the row and keywords share one explicit read transaction.

- [x] **Step 3: Run database RED**

Run: `../../.venv/bin/python -m pytest Tests/Prompts_DB/test_prompts_db_chatbook_export.py -q --tb=short`
Expected: `AttributeError` for both missing methods.

- [x] **Step 4: Implement the minimal ID query and snapshot boundary**

```python
def get_all_active_prompt_ids(self) -> list[int]:
    rows = self.get_connection().execute(
        "SELECT id FROM Prompts WHERE deleted = 0 ORDER BY id"
    ).fetchall()
    return [int(row["id"]) for row in rows]


def fetch_prompt_chatbook_snapshot(self, prompt_id: int) -> dict[str, Any] | None:
    if type(prompt_id) is not int or prompt_id <= 0:
        raise ValueError("prompt_id must be a positive integer.")
    conn = self.get_connection()
    owns_transaction = not conn.in_transaction
    try:
        if owns_transaction:
            conn.execute("BEGIN")
        row = conn.execute(EXPORT_PROMPT_SQL, (prompt_id,)).fetchone()
        keywords = conn.execute(EXPORT_KEYWORDS_SQL, (prompt_id,)).fetchall()
        result = _detach_export_snapshot(row, keywords)
        if owns_transaction:
            conn.commit()
        return result
    except (sqlite3.Error, TypeError, ValueError, KeyError, IndexError):
        if owns_transaction:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
        raise DatabaseError("Failed to read Prompt export snapshot.") from None
```

Do not call `transaction`, `execute_query`, `get_prompt_by_id`, or `fetch_keywords_for_prompt`; do not log from this method. Keep SQL parameterized and keyword ordering identical to the incumbent canonical read.

- [x] **Step 5: Run GREEN and transaction/privacy mutations**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Prompts_DB/test_prompts_db_chatbook_export.py \
  Tests/Prompts_DB/test_prompts_db_retained_history.py::test_new_prompt_snapshot_captures_canonical_keywords_before_link_events \
  Tests/Prompts_DB/test_prompts_db_retained_history.py::test_update_prompt_snapshot_captures_final_keywords_without_rewriting_legacy_row \
  Tests/Prompts_DB/test_prompts_db_pytest.py::TestPromptOperations::test_add_prompt_with_keywords \
  -q --tb=short
```

Expected: all pass. Temporarily remove `BEGIN` and the deleted predicate; the snapshot/concurrency tests must fail. Temporarily route through `execute_query`; the privacy capture must fail. Inject a row missing each required key and prove `KeyError`/`IndexError` become fixed pre-mutation `DatabaseError`. Restore all mutations.

- [x] **Step 6: Static checks and commit**

Run:

```bash
../../.venv/bin/python -m ruff format --check tldw_chatbook/DB/Prompts_DB.py Tests/Prompts_DB/test_prompts_db_chatbook_export.py
../../.venv/bin/python -m ruff check tldw_chatbook/DB/Prompts_DB.py Tests/Prompts_DB/test_prompts_db_chatbook_export.py
../../.venv/bin/python -m py_compile tldw_chatbook/DB/Prompts_DB.py
../../.venv/bin/python -m mypy --follow-imports=skip tldw_chatbook/DB/Prompts_DB.py
git diff --check
```

Expected: each exits 0.
Commit: `feat(prompts): add Chatbook export snapshots`

### Task 3: Lossless all-or-nothing Chatbook Prompt create/import

**Files:**
- Modify: `tldw_chatbook/Chatbooks/Chatbook_Creator.py`
- Modify: `tldw_chatbook/Chatbooks/Chatbook_Importer.py`
- Create: `Tests/Chatbooks/test_chatbook_prompt_round_trip.py`
- Modify: `Tests/Chatbooks/test_chatbook_creator.py`
- Modify: `Tests/Chatbooks/test_chatbook_importer.py`

- [x] **Step 1: Write the real ZIP round-trip RED**

Create separate source/destination file-backed Prompt DBs and a real Chatbook ZIP. Seed legacy Prompt, structured-v2 Recipe, compatibility-only definition, distinct multiline lanes, Unicode/literal markup, keywords, one deleted row, and >50 active rows. For one exported Prompt, create multiple retained-history snapshots and attach it to a real local collection before export. Inspect manifest/payload exact keys/paths/null timestamp slots, then import and compare all portable fields plus fresh identity and exactly one destination `create` history snapshot. Assert no source history row, collection definition, or membership is represented in the ZIP or restored.

- [x] **Step 2: Write all-or-nothing and sanitized failure REDs**

Delete one selected row after resolution, inject database/encoding/write failures containing adversarial sentinels, and assert: no finalized new archive, any pre-existing destination remains intact, fixed result/status copy, no prompt data/source ID/exception text/traceback in Loguru or caplog.

- [x] **Step 3: Write legacy and invalid-import REDs**

Build actual legacy `prompt_<numeric-id>.json` files including `id`/timestamps. Assert exact legacy System mapping. Parameterize unknown/missing versions, mixed shapes, bad types, and extra keys; each fails before `add_prompt`, increments status once, and emits fixed recovery copy without payload values.

- [x] **Step 4: Run Chatbook RED**

Run: `../../.venv/bin/python -m pytest Tests/Chatbooks/test_chatbook_prompt_round_trip.py Tests/Chatbooks/test_chatbook_creator.py Tests/Chatbooks/test_chatbook_importer.py -q -k 'prompt or chatbook_prompt' --tb=short`
Expected: lossless/identity/all-or-nothing cases fail against the legacy collector/importer.

- [x] **Step 5: Implement archive-local collection and sanitized settlement**

Add a repr-safe `PromptChatbookExportError` with `archive_item_id` and `category` only. Enumerate selected source IDs to `item-000001`, write `prompt_item-000001.json`, and put that item ID/file path in the manifest. Call `fetch_prompt_chatbook_snapshot`, then the codec. On any non-cancellation failure, raise the sanitized type `from None`; do not catch-and-continue.

Catch `PromptChatbookExportError` before the broad creator exception branch:

```python
except PromptChatbookExportError as exc:
    dependency_info = {
        "missing_dependencies": list(self.missing_dependencies),
        "auto_included": list(self.auto_included_characters),
    }
    logger.error(
        "ChatbookCreator.create_chatbook: Prompt export failed "
        "item={} category={}",
        exc.archive_item_id,
        exc.category,
    )
    return False, "Unable to export one or more Prompts.", dependency_info
```

In the importer, derive the existing `prompt_{manifest_id}.json` filename, decode before mutation, apply optional name prefix, and pass every decoded portable field to `add_prompt`. Replace task-touched Prompt success/failure diagnostics with fixed operation + archive item + category only; status copy remains bounded.

- [x] **Step 6: Run GREEN and non-vacuity mutations**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Chatbooks/test_chatbook_prompt_round_trip.py \
  Tests/Chatbooks/test_chatbook_creator.py \
  Tests/Chatbooks/test_chatbook_importer.py \
  Tests/Chatbooks/test_local_chatbook_service_export.py \
  Tests/Chatbooks/test_local_chatbook_service.py \
  -q -k 'prompt or chatbook_prompt or export_chatbook or import_chatbook' --tb=short
```

Expected: zero failures. Mutate the collector back to source IDs, catch-and-continue, collapse lanes, parse/reserialize definitions, or bypass decode-before-write; each targeted test must fail. Restore all mutations and rerun the same command GREEN.

- [x] **Step 7: Static checks and commit**

Run:

```bash
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Chatbooks/Chatbook_Creator.py \
  tldw_chatbook/Chatbooks/Chatbook_Importer.py \
  Tests/Chatbooks/test_chatbook_prompt_round_trip.py \
  Tests/Chatbooks/test_chatbook_creator.py \
  Tests/Chatbooks/test_chatbook_importer.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Chatbooks/Chatbook_Creator.py \
  tldw_chatbook/Chatbooks/Chatbook_Importer.py \
  Tests/Chatbooks/test_chatbook_prompt_round_trip.py \
  Tests/Chatbooks/test_chatbook_creator.py \
  Tests/Chatbooks/test_chatbook_importer.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/Chatbooks/Chatbook_Creator.py \
  tldw_chatbook/Chatbooks/Chatbook_Importer.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Prompt_Management/prompt_chatbook_record.py
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
git diff --check
```

Expected: format/lint/compile/mypy/diff exit 0. Before changing diagnostics, record the inventory baseline; after changes, update only the exact changed-owner entries in `Docs/security/production-diagnostic-inventory.json` and prove each changed owner equals the scanner. If the whole scanner has unrelated baseline drift, keep that result non-gating and document the exact unrelated owners rather than regenerating them.
Commit: `feat(chatbooks): round-trip portable prompts`

### Task 4: Add Prompts to the uncapped Library export scope

**Files:**
- Modify: `tldw_chatbook/Library/library_export_scope.py`
- Modify: `tldw_chatbook/Library/library_export_state.py`
- Modify: `Tests/Library/test_library_export_scope.py`
- Modify: `Tests/Library/test_library_export_scope_ids.py`
- Modify: `Tests/Library/test_library_export_roundtrip.py`
- Modify: `Tests/Library/test_library_export_state.py`

- [x] **Step 1: Write scope/count/label RED tests**

Extend real/fake sources with 207 Prompt IDs. Assert Prompt-only scope never touches Media/ChaChaNotes, Everything returns four stable count keys/selections, explicit Prompt IDs map to `ContentType.PROMPT`, deleted rows are absent through the DB seam, and labels are truthful for zero/one/many. Update `_seed_source_dbs` in `Tests/Library/test_library_export_roundtrip.py` to construct/seed a real `PromptsDatabase`; update all five resolver calls to pass it and assert the Everything archive/import contains the seeded Prompt while single-source archives do not.

- [x] **Step 2: Run Library scope RED**

Run: `../../.venv/bin/python -m pytest Tests/Library/test_library_export_scope.py Tests/Library/test_library_export_scope_ids.py Tests/Library/test_library_export_state.py Tests/Library/test_library_export_roundtrip.py -q --tb=short`
Expected: `ExportScope(kind="prompts")` rejects, resolver calls fail on the missing Prompt source argument, and Everything lacks Prompt counts/selections/import.

- [x] **Step 3: Implement the fourth source without another abstraction**

Add `prompts` to `_VALID_KINDS`, `_KIND_TO_CONTENT_TYPE`, stable count dictionaries, resolver branches, and labels. Add one required one-method `PromptIdSource` protocol with `get_all_active_prompt_ids() -> list[int]`, and pass the Prompt DB as the third source argument. Keep `ids` generic and add no UI selection feature. Update all five incumbent resolver calls in `Tests/Library/test_library_export_roundtrip.py` to supply the seeded real Prompt DB.

```python
counts = {"media": 0, "conversations": 0, "notes": 0, "prompts": 0}
if scope.kind in ("everything", "prompts"):
    prompt_ids = [str(value) for value in prompts_db.get_all_active_prompt_ids()]
    if prompt_ids:
        selections[ContentType.PROMPT] = prompt_ids
```

Update state docstrings and assert `show_media_fields is False` for Prompt scope; do not add state fields.

- [x] **Step 4: Run GREEN and rendered-page bypass mutation**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Library/test_library_export_scope.py \
  Tests/Library/test_library_export_scope_ids.py \
  Tests/Library/test_library_export_state.py \
  Tests/Library/test_library_export_roundtrip.py \
  -q --tb=short
```

Expected: all pass, including the real Everything service/ZIP/import test with a Prompt. Temporarily cap IDs at 50 or route through browse results; the 207-row test must fail. Restore.

- [x] **Step 5: Static checks and commit**

Run:

```bash
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Library/library_export_scope.py \
  tldw_chatbook/Library/library_export_state.py \
  Tests/Library/test_library_export_scope.py \
  Tests/Library/test_library_export_scope_ids.py \
  Tests/Library/test_library_export_state.py \
  Tests/Library/test_library_export_roundtrip.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Library/library_export_scope.py \
  tldw_chatbook/Library/library_export_state.py \
  Tests/Library/test_library_export_scope.py \
  Tests/Library/test_library_export_scope_ids.py \
  Tests/Library/test_library_export_state.py \
  Tests/Library/test_library_export_roundtrip.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/Library/library_export_scope.py \
  tldw_chatbook/Library/library_export_state.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Library/library_export_scope.py \
  tldw_chatbook/Library/library_export_state.py
git diff --check
```

Expected: each exits 0.
Commit: `feat(library): scope Chatbook exports to prompts`

### Task 5: Wire the existing Library export canvas from Prompts

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/Library/test_library_export_execution.py`
- Modify: `Tests/UI/test_library_prompts_canvas.py`

- [x] **Step 1: Read UI craft guidance immediately before edits**

Read `@impeccable` `reference/craft-floor.md` and apply `@textual-tui`. Reuse the incumbent `ds-toolbar`, compact `library-canvas-action` Buttons, export canvas, worker, and focus restoration. Do not add CSS unless real compositor evidence proves it necessary.

- [x] **Step 2: Write mounted Prompt-export RED tests**

Replace the obsolete “no dead export button” assertion with: Sort/Import/Export are one horizontal toolbar; `Export…` opens `ExportScope(kind="prompts")`; server mode warns before count/service calls; Prompt count/selection uses `app_instance.prompts_db`; media quality is absent; ready/error/Retry/cancel and stale tokens retain existing behavior.

- [x] **Step 3: Write privacy and 64x24 focus/compositor REDs**

Inject explicit source-ID and exception sentinels in count and both selection branches. Assert fixed user copy and category-only logs with no traceback. At 64x24 and normal size under the generated stylesheet, assert all three toolbar Buttons are painted/focusable/inside the toolbar, use compositor strips/frames to prove labels are visible, and exercise `library-prompts-export` focus restoration.

- [x] **Step 4: Run UI RED**

Run: `../../.venv/bin/python -m pytest Tests/Library/test_library_export_execution.py Tests/UI/test_library_prompts_canvas.py -q -k 'prompt and export' --tb=short`
Expected: missing toolbar button/handler/Prompt DB scope and privacy assertions fail.

- [x] **Step 5: Implement minimal canvas/screen wiring**

Add one compact Button after Import:

```python
yield Button(
    "Export…",
    id="library-prompts-export",
    classes="library-canvas-action",
    compact=True,
)
```

Add one event handler that stops the event and awaits `_open_library_export_canvas(ExportScope(kind="prompts"))`. Add the ID to `_library_prompts_focus_identity`. Resolve `app_instance.prompts_db` beside existing DB handles and pass it to count/selection helpers. Include a memory-backed Prompt DB in the existing inline-versus-worker decision, and extend zero-count fallback to all four keys.

Change the three task-touched recovery branches to fixed user copy and parameterized Loguru fields:

```python
logger.warning(
    "Library export selection resolution failed scope_kind={} category={}",
    scope.kind,
    type(exc).__name__,
)
```

Never use `scope!r`, `str(exc)`, or `exception=True` there.

- [x] **Step 6: Run GREEN, mutation checks, and real visual verification**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Library/test_library_export_execution.py \
  Tests/UI/test_library_prompts_canvas.py \
  -q -k 'prompt and export' --tb=short
```

Expected: zero failures. Mutate server refusal, Prompt DB forwarding, focus allowlist, and sanitized logging; each focused regression must fail, then restore and rerun the same command GREEN. Create temporary `Tests/UI/test_task197_prompt_export_visual_closeout.py` using the incumbent real-bundle Textual host and write SVG/JSON evidence to `.superpowers/sdd/2026-08-12-task-197-bulk-prompt-chatbook-export/visual-closeout/`, then run:

```bash
../../.venv/bin/python -m pytest \
  Tests/UI/test_task197_prompt_export_visual_closeout.py \
  -q --tb=short
```

Expected: both 64x24 and normal-size cases pass with Sort/Import/Export painted, focusable, inside the toolbar, and export actions reachable. Inspect compositor strips and rasterized frames, record observations, then remove the temporary test file before staging. If no defect appears, make no CSS change.

- [x] **Step 7: Static checks and commit**

Run:

```bash
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Library/test_library_export_execution.py \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Library/test_library_export_execution.py \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs --json \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py
git diff --check
```

Expected: each exits 0; detector returns `[]` or only verified pre-existing findings. Run the detector once, after final UI edits.
Commit: `feat(library): export prompts in Chatbooks`

### Task 6: Integrated regression and documentation closeout

**Files:**
- Modify: `Docs/User_Guide/library/prompts.md`
- Modify: `Docs/User_Guide/library/import-and-export.md`
- Modify: `backlog/tasks/task-197 - Bulk-export-of-prompts-via-the-chatbook-format.md`
- Modify: this plan's checkboxes as work lands

- [x] **Step 1: Run the integrated Prompt Chatbook matrix**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Prompt_Management/test_prompt_chatbook_record.py \
  Tests/Prompts_DB/test_prompts_db_chatbook_export.py \
  Tests/Chatbooks/test_chatbook_prompt_round_trip.py \
  Tests/Chatbooks/test_chatbook_creator.py \
  Tests/Chatbooks/test_chatbook_importer.py \
  Tests/Chatbooks/test_local_chatbook_service_export.py \
  Tests/Library/test_library_export_scope.py \
  Tests/Library/test_library_export_scope_ids.py \
  Tests/Library/test_library_export_roundtrip.py \
  Tests/Library/test_library_export_state.py \
  Tests/Library/test_library_export_execution.py \
  Tests/UI/test_library_prompts_canvas.py \
  -q --tb=short
```

Expected: all selected TASK-197 cases pass. Classify any unrelated baseline failure with an unchanged isolated rerun rather than weakening tests.

- [x] **Step 2: Run proportionate regression gates**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Chatbooks \
  Tests/Prompts_DB/test_prompts_db_chatbook_export.py \
  Tests/Prompts_DB/test_prompts_db_retained_history.py \
  Tests/Prompts_DB/test_prompts_db_pytest.py \
  Tests/Prompt_Management/test_prompt_chatbook_record.py \
  Tests/Library/test_library_export_scope.py \
  Tests/Library/test_library_export_scope_ids.py \
  Tests/Library/test_library_export_roundtrip.py \
  Tests/Library/test_library_export_state.py \
  Tests/Library/test_library_export_execution.py \
  Tests/UI/test_library_prompts_canvas.py \
  -q --tb=short
```

Expected: zero TASK-197 regressions. Do not run broad CI-only or unrelated suites after the user's explicit instruction to ignore CI.

- [x] **Step 3: Update user documentation**

Document local Prompt/Recipe bulk export, Everything inclusion, exact portable fields, legacy import, new destination identity/history, and exclusions for source timestamps/deleted/history/collections. Use literal UI labels (`Export…`, `Export bundle (.zip)`). Update every old three-source statement/label in `Docs/User_Guide/library/import-and-export.md` to the four-source contract.

- [x] **Step 4: Complete static/security verification**

Use `@superpowers:verification-before-completion`. Run:

```bash
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Prompt_Management/prompt_chatbook_record.py \
  tldw_chatbook/DB/Prompts_DB.py \
  tldw_chatbook/Chatbooks/Chatbook_Creator.py \
  tldw_chatbook/Chatbooks/Chatbook_Importer.py \
  tldw_chatbook/Library/library_export_scope.py \
  tldw_chatbook/Library/library_export_state.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Prompt_Management/test_prompt_chatbook_record.py \
  Tests/Prompts_DB/test_prompts_db_chatbook_export.py \
  Tests/Chatbooks/test_chatbook_prompt_round_trip.py \
  Tests/Chatbooks/test_chatbook_creator.py \
  Tests/Chatbooks/test_chatbook_importer.py \
  Tests/Library/test_library_export_scope.py \
  Tests/Library/test_library_export_scope_ids.py \
  Tests/Library/test_library_export_roundtrip.py \
  Tests/Library/test_library_export_state.py \
  Tests/Library/test_library_export_execution.py \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Prompt_Management/prompt_chatbook_record.py \
  tldw_chatbook/DB/Prompts_DB.py \
  tldw_chatbook/Chatbooks/Chatbook_Creator.py \
  tldw_chatbook/Chatbooks/Chatbook_Importer.py \
  tldw_chatbook/Library/library_export_scope.py \
  tldw_chatbook/Library/library_export_state.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Prompt_Management/test_prompt_chatbook_record.py \
  Tests/Prompts_DB/test_prompts_db_chatbook_export.py \
  Tests/Chatbooks/test_chatbook_prompt_round_trip.py \
  Tests/Chatbooks/test_chatbook_creator.py \
  Tests/Chatbooks/test_chatbook_importer.py \
  Tests/Library/test_library_export_scope.py \
  Tests/Library/test_library_export_scope_ids.py \
  Tests/Library/test_library_export_roundtrip.py \
  Tests/Library/test_library_export_state.py \
  Tests/Library/test_library_export_execution.py \
  Tests/UI/test_library_prompts_canvas.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/Prompt_Management/prompt_chatbook_record.py \
  tldw_chatbook/DB/Prompts_DB.py \
  tldw_chatbook/Chatbooks/Chatbook_Creator.py \
  tldw_chatbook/Chatbooks/Chatbook_Importer.py \
  tldw_chatbook/Library/library_export_scope.py \
  tldw_chatbook/Library/library_export_state.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Prompt_Management/prompt_chatbook_record.py \
  tldw_chatbook/Library/library_export_scope.py \
  tldw_chatbook/Library/library_export_state.py
../../.venv/bin/python tldw_chatbook/css/build_css.py
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
rg -n "scope!r|str\(exc\)|exception=True" \
  tldw_chatbook/Chatbooks/Chatbook_Creator.py \
  tldw_chatbook/Chatbooks/Chatbook_Importer.py \
  tldw_chatbook/UI/Screens/library_screen.py
git diff --check
git status --short
```

Expected: formatter/lint/compile/mypy/CSS/diff exit 0; revert a timestamp-only generated CSS change if source CSS is unchanged. The inventory's TASK-197-touched owner entries equal the scanner exactly; any unrelated whole-tree drift is enumerated, not regenerated. The privacy grep is manually classified against only TASK-197-touched branches and adversarial privacy tests remain green. Status lists only intended TASK-197 files.

- [x] **Step 5: Request independent code/spec review**

Use `@superpowers:requesting-code-review` for a bounded review of the complete TASK-197 range. Address all evidenced Critical/Important/Minor findings RED-first and rerun affected gates until approved.

- [x] **Step 6: Close task hygiene**

Check all seven acceptance criteria, add concise Implementation Notes with ADR-057 and exact evidence, decide whether a genuine new lesson was learned, and set TASK-197 to Done via Backlog CLI only after every DoD item passes.

- [x] **Step 7: Commit closeout**

Commit the reviewed privacy correction and docs/task closeout as `fix(prompts): close bulk Chatbook export`. Confirm clean status and record final SHAs and evidence. Do not push or open a PR unless separately authorized.
