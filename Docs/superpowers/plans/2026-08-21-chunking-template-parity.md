# Chunking Template Parity & Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make chunking templates real in chatbook — one canonical server-flat shape, one store (Media DB v7), the vendored processor running all three stages, an ingest picker that actually changes chunks, and a Library-surface re-chunk action that clears the legacy-engine stamp — closing #1's two deferred rulings.

**Architecture:** Six dependency-ordered PRs (spec §10.6): PR 0 clears #1 residue; PR A vendors the template processor read-side (`templates.py` + `template_runtime.py` + local validation); PR B lands schema v6→v7 as a table rebuild shipping atomically with the CRUD rewrite; PR C deletes the second (file) store and its package-root exports; PR D wires templates into all six ingest seams with a precedence fix and a picker; PR E ships the report renderer and the re-chunk action with forced re-index and a worker mutual-exclusion guard.

**Tech Stack:** Python ≥3.11, SQLite (media DB v7, ADR-030 transactional migration), vendored `templates.py` from pin `dev@385afa95` (NOT moved), Textual 8.2.8 (`app.run_test()` for UI ACs).

**Spec:** `Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md` — all 8 maintainer decisions are ruled (§13); ACs are numbered §12.1–54 and cited per task below. The plan argues from the spec; conflicts resolve against it.

## Global Constraints

- **Never move the vendor pin** (`dev` @ `385afa951922c8a9dc2002c675bb6cad65e4ac23`); `templates.py` is a **move from `excluded` to `vendored`** in `VENDOR_MANIFEST.toml`, never both lists.
- Vendored files are never hand-edited; the only rewrites are the sync script's three mechanical rules.
- **ADR-030:** migration DDL + conversion + seeding + version bump run in ONE transaction, statement-by-statement via `_execute_transactional_script`; `BEGIN IMMEDIATE`; a seeded mid-rebuild failure must leave the DB at v6 intact.
- **§10.5 verification:** migrations proven against `tmp_path`/in-memory DBs only; UI via `app.run_test()` or a scratch data dir; **never launch the app against the shared library DB while v7 is in flight** (schema bump is a one-way door for every concurrent worktree).
- Historical v6 fixtures come from bootstrapping at a patched `_CURRENT_SCHEMA_VERSION` (the `Tests/ChaChaNotesDB/historical_bootstrap.py` pattern) — never hand-dropping tables.
- Repo rule: targeted test runs only; full sweep only on maintainer opt-in.
- Backlog/ADR filing: **write task/ADR files directly** (the CLI corrupts ids via `-p` and joins `--ac` commas; five-digit ids break `backlog task <id>`); sweep refs AND untracked worktree files at the instant of writing; verify with `backlog task <id> --plain` + `git status backlog/`. In zsh, brace ref expressions: `git show "${b}:path"`.
- Validation matches the server's endpoint semantics **including its three warts** (§7.1) — parity, not endorsement; warts are pinned by tests so a later "fix" can't silently break parity.
- `test_app_import_weight.py` stays green — no DB handle reaches `Chunk_Lib` (§8.2 layering).
- PR ordering is load-bearing: **A before B before C**; D and E after C. PR 0 is independent and mergeable immediately.
- Rolling-summarize fail-closed changes the failure path only; the payload-dict `llm_call_function` caller contract (pinned by `Tests/Chunking/test_chunk_lib_shim.py:146-175`) is unchanged.

---

## PR 0 — #1 residue (independently mergeable)

### Task 1: ADR + governance filings (BEFORE any implementation — spec AC 48–50, 51-partial)

**Files:**
- Create: `backlog/decisions/0NN-chunking-template-convergence.md` (number swept at execution time)
- Create: `tldw_chatbook/Chunking/engine/UPSTREAM_DEFECTS.md`
- Create: `backlog/tasks/task-<id> - *.md` × N (per §11 chatbook items 1–8, re-verified live)
- Modify: `backlog/docs/lessons-backlog-hygiene.md` (zsh `${b}:path` sweep-trap entry, with the incident)

**Interfaces:**
- Produces: the ADR number, the UPSTREAM_DEFECTS.md ledger (items 9–14 with upstream file:line + chatbook impact), the §11 task files (each: one-line outcome + 2–3 AC bullets).

Steps:
- [ ] Sweep ADR number across `backlog/decisions/` in every remote ref AND untracked worktree files (`git for-each-ref` + `ls .worktrees/*/backlog/decisions/` + main checkout untracked); pick the next free number; write the ADR: decision (full convergence to the server's flat shape; vendored processor; v7 table rebuild), and **explicitly state that v7 makes downgrade impossible, voiding ADR-073's `git revert` safety net**. Link spec + plan.
- [ ] Re-verify every §11 chatbook item live on `origin/dev` (`git log -S` for the symbol; grep the board for existing filings) — items fixed since exploration become notes, not tasks. File each survivor directly as a task file (hand-authored AC block; ids swept refs+untracked+board at the instant of writing; five-digit ids expected — direct file writes only).
- [ ] Write `UPSTREAM_DEFECTS.md` beside `VENDOR_MANIFEST.toml`: six entries (template_library phantom; mapper triplication; validate/runtime asymmetries; duplicate definitions divergent spelling; stale method fallback list; discarded preprocessing metadata), each with upstream file:line + chatbook impact.
- [ ] Append the zsh sweep-trap lesson to `lessons-backlog-hygiene.md` (incident: the §5 sweep returned clean-everywhere because `git show "$b:path"` applied `:t`).
- [ ] Commit: `docs(chunking): ADR for template convergence; file §11 follow-ups; upstream defects ledger`

### Task 2: Rolling-summarize fail-closed (AC 1–3)

**Files:**
- Modify: `tldw_chatbook/Chunking/Chunk_Lib.py` (the three marker branches: `:1103`, `:1112`, `:1120` — verify line anchors at execution)
- Test: `Tests/Chunking/test_chunk_lib_shim.py` (replace the untested marker branches with mutation-verified pins)

**Interfaces:**
- Consumes: engine `ProcessingError` (already re-exported).
- Produces: provider failure raises `ChunkingError` (subclass of the engine's, message naming the failed part) through BOTH the engine strategy path and the shim's legacy port; `llm_call_function` payload-dict contract unchanged.

- [ ] **Step 1 — failing tests** (add to `test_chunk_lib_shim.py`):

```python
def test_rolling_summarize_provider_failure_raises_both_prefixes():
    import pytest
    from tldw_chatbook.Chunking import Chunk_Lib

    def failing_llm(payload):
        raise RuntimeError("provider down")

    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    for method_opts in ({"method": "rolling_summarize", "max_size": 2},):
        with pytest.raises(Chunk_Lib.ChunkingError, match="summariz"):
            Chunk_Lib.improved_chunking_process(
                text, method_opts, llm_call_function_for_chunker=failing_llm,
            )


def test_marker_strings_never_persisted():
    # AC 2: BOTH prefixes pinned — they are different strings.
    from tldw_chatbook.Chunking import Chunk_Lib
    import inspect
    src = inspect.getsource(Chunk_Lib)
    assert "[Summarization failed for this part" not in src.replace('f"[Summarization failed for this part: {chunk_for_llm[:100]}...]"', '', 1) or True
    # The load-bearing pin: grep the module source for either f-string prefix.
    assert "Summarization failed for this part" not in _chunk_lib_runtime_source()
    assert "Summarization error for this part" not in _chunk_lib_runtime_source()
```

(Implement `_chunk_lib_runtime_source()` as `inspect.getsource(sys.modules['tldw_chatbook.Chunking.Chunk_Lib'])`; drop the vestigial first assert — shown here only to flag the trap the AC names: pin **both** prefixes.)

- [ ] **Step 2 — red:** `pytest Tests/Chunking/test_chunk_lib_shim.py -k rolling_summarize -v` → FAIL (markers still emitted).
- [ ] **Step 3 — implement:** the three `accumulated_summaries.append(f"[Summarization …")` branches become `raise ChunkingError(f"Rolling-summarize LLM call failed for part {i + 1}: {exc}") from exc` (keep the per-part index and cause). Update the module docstring at `:1443`.
- [ ] **Step 4 — mutation-verify (AC 3):** for EACH of the three guard sites, temporarily revert the raise back to an append (seeded mutation), run the pins, confirm red, restore. Record each mutation run in the commit message body.
- [ ] **Step 5 — green + regression:** `pytest Tests/Chunking/ -q --ignore=Tests/Chunking/test_sync_script.py` (the ported `test_rolling_summarize_fail_closed.py` — engine path — must stay green).
- [ ] **Step 6 — commit:** `feat(chunking): rolling-summarize fails closed — marker branches raise (both prefixes)`

### Task 3: #1-residue deletions (AC 4–5)

**Files:**
- Delete: `tldw_chatbook/RAG_Search/table_serializer.py`, `tldw_chatbook/Event_Handlers/template_events.py`, `DB/migrations/add_chunking_config.sql`
- Modify: `enhanced_chunking_service.py:61,76,90` + `parent_child_adapter.py:437,455` (remove `serialize_tables` kwarg threading), `Docs/superpowers/ENHANCED_RAG_FEATURES.md` (`:35`, `:149`, `:206` sections), `Tests/test_enhanced_rag.py` (`test_table_serialization` + the `:145` kwarg site)
- Modify: `Docs/security/production-diagnostic-inventory.json` (regenerate)

- [ ] **Step 1 — red pins first:** run the suites that touch these files (`Tests/test_enhanced_rag.py` — currently env-skipped; `pytest Tests/RAG/test_parent_child_adapter.py -q`) to record the pre-state.
- [ ] **Step 2 — delete + de-thread:** remove the four files, the kwarg threading (accept-and-ignore stays ONLY if a straggler caller exists — spec says remove the threading entirely; verify zero non-test callers first), doc sections, and the two test sites.
- [ ] **Step 3 — inventory regen (AC 5):** `python scripts/check_persistent_diagnostic_inventory.py` then hand-review the row diff — the inventory carries rows for the two deleted modules; the diff review is recorded in the commit body.
- [ ] **Step 4 — green:** `pytest Tests/Chunking/ Tests/RAG/test_parent_child_adapter.py Tests/Chunking/test_callsite_characterization.py -q --ignore=Tests/Chunking/test_sync_script.py`
- [ ] **Step 5 — commit:** `chore(chunking): delete #1 residue — table_serializer, template_events, orphan SQL; regenerate diagnostic inventory`

---

## PR A — processor (read-side only; no schema, no user-visible change)

### Task 4: Vendor `templates.py` (AC 6, 50)

**Files:**
- Modify: `Helper_Scripts/sync_chunking_engine.py` + `tldw_chatbook/Chunking/engine/VENDOR_MANIFEST.toml` (move `templates.py` from `excluded` → `vendored`)
- Create (via sync): `tldw_chatbook/Chunking/engine/templates.py`
- Test: `Tests/Chunking/test_sync_script.py` (extend tree-completeness to 36 files)

- [ ] **Step 1 — failing test:** extend `test_engine_tree_complete` to assert `templates.py` exists and `excluded` no longer names it; assert `_shims.testing.is_truthy` satisfies its import (spec §6.1: no new shim).
- [ ] **Step 2 — red, then manifest move + sync:** create the pinned worktree (`git -C ~/Documents/GitHub/tldw_server2 worktree add /tmp/tldw_server_sync 385afa95…`), move the list entry, run `python Helper_Scripts/sync_chunking_engine.py --source /tmp/tldw_server_sync`, verify 36 files, byte-faith modulo rewrite rules, zero `tldw_Server_API` refs.
- [ ] **Step 3 — licence recheck (AC 50):** both licence files still ship; pyproject `license-files` still names them (manifest `extra` untouched).
- [ ] **Step 4 — green + import weight:** `pytest Tests/Chunking/test_sync_script.py -q` (3 non-network tests + idempotency w/ local source); `pytest Tests/Performance/test_app_import_weight.py -q`.
- [ ] **Step 5 — commit:** `feat(chunking): vendor templates.py from the existing pin (manifest move, 35→36 files)`

### Task 5: `Chunking/template_runtime.py` — mapper, resolver, apply (AC 7–8, 10–13)

**Files:**
- Create: `tldw_chatbook/Chunking/template_runtime.py`
- Test: `Tests/Chunking/test_template_runtime.py`

**Interfaces:**
- Consumes: vendored `TemplateProcessor`, `ChunkingTemplate` (dataclasses); `Chunk_Lib._synthesize_flat_offsets` (`:604`) for offsets; engine `Chunker`.
- Produces (exact signatures later tasks rely on):
  - `template_from_record(record: dict) -> ChunkingTemplate` — the ONE flat→internal mapper; raises `TemplateError` (clear message, not `KeyError`) when `chunking` is missing.
  - `resolve_template(db, name: str) -> dict | None` — the ONLY name→template resolution (DB handle in, template dict out; `deleted = 0` filtered).
  - `apply_template(template: dict, text: str, options: dict | None = None) -> list[dict]` — runs pre+chunk+post via `TemplateProcessor.process_template`, then synthesizes the flat contract.

- [ ] **Step 1 — failing tests** (core contracts; full list per ACs):

```python
# Tests/Chunking/test_template_runtime.py
import pytest
from tldw_chatbook.Chunking import template_runtime as tr
from tldw_chatbook.Chunking.engine.exceptions import TemplateError

FLAT = {
    "preprocessing": [{"operation": "normalize_whitespace", "config": {"max_line_breaks": 1}}],
    "chunking": {"method": "sentences", "config": {"max_size": 2, "overlap": 0}},
    "postprocessing": [{"operation": "filter_empty", "config": {}}],
    "classifier": {"media_types": ["document"], "min_score": 0.5, "priority": 10},
    "metadata": {},
}


def test_mapper_guards_missing_chunking():           # AC 7
    with pytest.raises(TemplateError, match="chunking"):
        tr.template_from_record({"name": "x", "template_json": '{"preprocessing": []}'})


def test_apply_runs_pre_and_post_acid():             # AC 10 — pinned exact output
    text = "First  sentence.\n\n\n\nSecond sentence here.  Third one."
    out = tr.apply_template(FLAT, text)
    # normalize_whitespace collapses the quad newline; sentences max_size 2
    # → exactly two chunks; values PINNED, not "different from chunk-only":
    assert [c["text"] for c in out] == ["First sentence.", "Second sentence here.  Third one."]
    # AC 11: full flat contract synthesized
    for c in out:
        assert {"text", "start_char", "end_char", "word_count", "chunk_index",
                "total_chunks", "metadata"} <= set(c)
        assert c["metadata"]["offset_basis"].startswith(("source", "preprocessed:"))
    assert [c["chunk_index"] for c in out] == [0, 1]      # 0-based top-level (§7-task-7 convention)
    assert out[0]["metadata"]["offset_basis"] == "preprocessed:normalize_whitespace"


def test_unsynthesizable_offsets_omitted_never_none():   # AC 12 (unit half)
    out = tr.apply_template(FLAT_DELETING, "a b c d e f g h")  # template whose postprocess deletes chunks
    for c in out:
        assert c.get("start_char") is not None or "start_char" not in c  # never present-and-None
```

Also: AC 7/8 enumeration guards — one test asserts `templates.py`'s mapper is imported by exactly one chatbook module (grep-based guard + `test_the_guard_can_see_what_it_guards` self-check that the guard finds a seeded second import); same style for `resolve_template`.

- [ ] **Step 2 — red**, then implement `template_runtime.py` (mapper with guard; resolver `SELECT … WHERE name = ? AND deleted = 0` against the v6 table for now — PR B changes the columns; `apply_template` calling `TemplateProcessor.process_template` then synthesizing: offsets via `_synthesize_flat_offsets` against the **preprocessed** text (run pre-stage separately first to capture the basis string), `offset_basis` per §6.4, `chunk_index`/`total_chunks`/`word_count` reconstructed, and a final normalization pass that **omits** offset keys rather than emitting `None`).
- [ ] **Step 3 — AC 12 integration half + AC 13:** index a template-chunked fixture item and run a RAG search (no `TypeError`); media navigation returns chunk-sized content (`local_media_reading_service` path) — in `Tests/Chunking/test_template_runtime.py` with an in-memory Media DB + the scope-service RAG path used by existing tests (find the precedent in `Tests/RAG/`).
- [ ] **Step 4 — green + fencing (AC 9):** add the no-`TemplateManager`-construction test — probe the templates directory does not exist after suite runs + positive control (construct one in the test, watch the probe flip, clean up).
- [ ] **Step 5 — commit:** `feat(chunking): template_runtime — one mapper, one resolver, apply_template with flat-contract synthesis`

### Task 6: Local validation (AC 14–15) + scope wiring

**Files:**
- Create: `RAG_Admin/template_validation.py`
- Modify: `tldw_chatbook/RAG_Admin/rag_admin_scope_service.py:306-309` (`validate_template_config` routes to local instead of hard-raising)
- Test: `Tests/RAG_Admin/test_template_validation.py` + fixture table

**Interfaces:**
- Produces: `validate_template(template: dict) -> {"valid": bool, "errors": [{"field", "message"}], "warnings": [...]}` — never raises on invalid input; methods resolved against `Chunking.engine.chunker.Chunker().get_available_methods()` (the ENGINE class — not the shim).

- [ ] **Step 1 — fixture table first (AC 14's provability clause):** `Tests/RAG_Admin/template_validation_fixtures.json` — generate once by reading the pinned endpoint source (`endpoints/chunking_templates.py:782-992` at the pin via the temp worktree), recording input → expected `{valid, errors, warnings}` with upstream line ranges in the file header. Hand-write ~15 rows covering: missing chunking / missing method / unknown method / stale-fallback methods (`fixed_size` must VALIDATE — live registry) / boundary count > 20 / unsafe pattern (ReDoS) / `flags: "s"` rejected, `"i"`/`"m"` ok / unanchored `.*` → warning-not-error / classifier strict-key reject / `min_score` 1.5 reject / unknown operation name **validates clean** (wart) / `{type, params}` op **fails** validate (wart) / non-JSON-serializable reject.
- [ ] **Step 2 — failing test:** parametrize over the fixture table; `pytest Tests/RAG_Admin/test_template_validation.py -q` → FAIL (module absent).
- [ ] **Step 3 — implement** `template_validation.py` per spec §7 check-for-check (regex safety via vendored `engine/regex_safety.check_pattern`, max_len 256; classifier regexes ≤ 128; `flags ∈ {i, m}`).
- [ ] **Step 4 — warts pinned (AC 15):** three dedicated tests asserting the §7.1 asymmetries EXACTLY (op-name not checked → unknown op valid; `operation` required though runtime takes `type`; unknown top-level keys ignored) with comments naming them parity pins.
- [ ] **Step 5 — wire the scope service:** local-mode `validate_template_config` calls the new validator (unit-test: no more "Server retrieval-admin backend is required" raise in local mode).
- [ ] **Step 6 — green:** full `Tests/RAG_Admin/ -q`.
- [ ] **Step 7 — commit:** `feat(chunking): local template validation matching the server endpoint (warts pinned); scope wiring`

---

## PR B — storage (schema v7 ships atomically with its only reader)

### Task 7: v6→v7 migration + conversion + quarantine + seeds (AC 16–23, 29)

**Files:**
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py` (v7 rebuild migration + seed data)
- Create: `Tests/DB/historical_bootstrap_v6.py` (genuine-fixture helper, `Tests/ChaChaNotesDB/historical_bootstrap.py` pattern)
- Test: `Tests/DB/test_media_db_schema_v7.py`

**Interfaces:**
- Produces: `_CURRENT_SCHEMA_VERSION = 7`; `ChunkingTemplates(uuid, name, description, template_json, tags, is_builtin, version, deleted, created_at, updated_at)` + partial unique index `idx_chunking_templates_name_live` + `update_chunking_templates_timestamp` trigger recreated; conversion honoring §5.3 precedence (`is_system=1` rows dropped+re-seeded from the six built-ins; `general`/`conversational`/`contextual` converted+kept non-builtin; unconvertible → quarantined `deleted=1`, renamed `<name> (needs review)`, body under `metadata._unconverted`; dropped ops in `metadata._dropped_operations`; method repair `structural→structure_aware`, `hierarchical→structure_aware` + `config.hierarchical=true`, `contextual→sentences`).

- [ ] **Step 1 — re-sweep v7 (spec §5):** assert zero refs/worktrees claim v7 at execution start AND at merge time; record both sweeps in the ledger.
- [ ] **Step 2 — failing tests** (write `test_media_db_schema_v7.py` covering ACs 16–23, 29; the six seeds are the §5.5 data with provenance comments — pin SHA + upstream line range):
  - fresh install → v7, six built-ins present, zero old seeds (AC 29);
  - genuine v6 fixture (historical bootstrap) upgrades: trigger survives (UPDATE bumps `updated_at`, AC 17); seeded mid-rebuild failure leaves v6 intact (monkeypatch a conversion row to raise mid-flight; assert version 6 + original rows, AC 18); fixture provenance assert (AC 19);
  - conversion precedence (AC 20): fixture with 5 seeds + one custom row + one garbage row → seeds replaced, custom converted, `general`/`conversational`/`contextual` kept non-builtin;
  - quarantine (AC 21) + dropped ops (AC 22);
  - six seeds all execute against the live engine (AC 23 — call `apply_template` per seed on a small fixture).
- [ ] **Step 3 — implement:** `_CHUNKING_TEMPLATES_V7_MIGRATION_SQL` (CREATE `ChunkingTemplates_v7` per §5.2 DDL, INSERT…SELECT with conversion via Python row loop inside the same transaction, DROP, RENAME, indices, trigger) executed via `_execute_transactional_script` under `BEGIN IMMEDIATE`; per-row conversion in `Chunking/_template_conversion.py` (pure functions — unit-testable in isolation) with `uuid4` per row, tags extracted-and-removed, method repair, op mapping (`section_detection→extract_sections`, the other three dropped+recorded).
- [ ] **Step 4 — green:** `pytest Tests/DB/ -q` (fix `test_media_db_schema_v6.py`'s pin per AC 28 — assert `== _CURRENT_SCHEMA_VERSION` and subset deltas).
- [ ] **Step 5 — commit:** `feat(db): schema v7 — ChunkingTemplates rebuild with conversion, quarantine, six server built-ins`

### Task 8: CRUD rewrite + soft delete + validate-on-write (AC 24–28)

**Files:**
- Modify: `tldw_chatbook/Chunking/chunking_interop_library.py` (CRUD against v7; drop `_template_cache`; writes via `transaction()`), `tldw_chatbook/RAG_Admin/rag_admin_normalizers.py` (`uuid`/`version` from DB), `local_rag_admin_service.py:189,195` (`is_system` gone), `Utils/sql_validation.py` (`ChunkingTemplates` + columns in `VALID_TABLES`)
- Test: `Tests/RAG_Admin/` + `Tests/Chunking/` extensions

- [ ] **Step 1 — failing tests:** no `is_system` anywhere in tree (grep pin); every listing filters `deleted = 0`; create supplies `uuid` and **refuses invalid templates** (validation from Task 6); update validates the NEW body only (stored-invalid row editable — AC 24); soft delete end-to-end (row leaves listings, name reusable, `version` increments — AC 25); `sql_validation` accepts the table (AC 27).
- [ ] **Step 2 — implement + green + commit:** `feat(chunking): CRUD rewrite for v7 — soft delete, validate-on-write, DB-sourced uuid/version`

---

## PR C — convergence (the breaking change, isolated)

### Task 9: Delete the file store + package-root change + five packaging sites (AC 30–33)

**Files:**
- Delete: `Chunking/templates/` (13 JSON + README + example_usage.py), `Chunking/chunking_templates.py`
- Modify: `Chunking/__init__.py:19` (exports removed; vendored `ChunkingTemplate` NOT re-exported), `Chunk_Lib.py:711` (name resolution → pre-resolved dict only), `pyproject.toml:479,497`, `MANIFEST.in:12`, `Packaging/check_manifest.py:134-155,242`, `Tests/Packaging/test_installed_distribution.py:1475-1485` (data) + `:285` (import pin), `Tests/integration/test_core_functionality_integration.py:61`, `Tests/Chunking/test_chunking_templates.py`, `CHANGELOG.md` (breaking-export entry)

- [ ] **Step 1 — failing tests first:** the bare-name raise pin (AC 32):

```python
def test_bare_name_template_raises_named_error():      # AC 32
    from tldw_chatbook.Chunking import Chunk_Lib
    from tldw_chatbook.Chunking.engine.exceptions import TemplateError
    with pytest.raises(TemplateError, match="resolve_template"):
        Chunk_Lib.Chunker(template="academic_paper")          # name string → named raise
    pre = {"chunking": {"method": "words", "config": {"max_size": 3}}}
    assert Chunk_Lib.Chunker(template=pre).chunk_text("a b c d e")   # dict works
```

plus `template_manager=` accepted-and-ignored pin, and the packaging import-pin update.
- [ ] **Step 2 — delete + re-point + all five packaging sites in ONE commit** (spec §8.1.2); rebuild CSS bundle not needed here; **re-prove wheel/sdist against freshly built artifacts** (`python -m build` + `check_manifest.py`).
- [ ] **Step 3 — green:** `pytest Tests/Packaging/ Tests/Chunking/ Tests/integration/test_core_functionality_integration.py Tests/Performance/test_app_import_weight.py -q` (AC 33).
- [ ] **Step 4 — commit:** `feat(chunking)!: delete the file template store; package-root export change (five packaging sites)`

---

## PR D — consumers

### Task 10: Ingest seams + precedence (AC 34–37)

**Files:**
- Modify: `RAG_Search/chunking_service.py` (`improved_chunking_process` gains `template: dict | None = None` kwarg, forwarded), `Local_Ingestion/local_file_ingestion.py` (six seams per §9.2: `:1033/1058/1084` pass-through; `:1186-1192`, `:1309-1315` widen key-by-key projections; `:607-614` widen the plain-text dict; server mode hides the picker), `app.py:2907-2918` (`_ingest_job_options` precedence)
- Test: `Tests/Local_Ingestion/test_ingest_template_resolution.py`

- [ ] **Step 1 — failing tests** (AC 34–37): resolution-order tests per path (picker/batch → config `[chunking] default_template` → plain); precedence pin (template chunk-stage options beat builder defaults; user-changed form value beats template — the inert-picker trap AC 35); two-template governance fixture (AC 36 — same fixture, two templates, different persisted rows, per media family pdf/document/ebook + audio/video + plain); unresolvable name fails the item with a named error (AC 37); "None" default byte-identical to today (AC 36 tail).
- [ ] **Step 2 — implement + green:** `pytest Tests/Local_Ingestion/ Tests/Chunking/test_callsite_characterization.py Tests/RAG/test_chunking_service.py -q`
- [ ] **Step 3 — commit:** `feat(chunking): templates honored on all six ingest seams; precedence fix`

### Task 11: Persistence + picker + config (AC 38–40) + docs

**Files:**
- Modify: `Local_Ingestion/local_file_ingestion.py` (write `chunking_template`/`chunking_params` + `Media.chunking_config` in the `LIKE`-and-`json_extract`-compatible shape), `Widgets/Library/library_ingest_canvas.py` (Select: DB-populated off mount path, `escape_markup` labels, default "None (manual settings)", hidden in server mode), config template + defaults (`[chunking]` section), `Docs/User_Guide/library/*.md`
- Test: `Tests/UI/` picker tests via `app.run_test()` + config-loader test

- [ ] **Step 1 — failing tests:** AC 38 (persisted columns + both-reader-compatible `chunking_config` shape); AC 39 (picker contract, four properties); AC 40 (real loader emits `[chunking]`).
- [ ] **Step 2 — implement + docs with re-verified stamps + green + commit:** `feat(chunking): ingest template picker; chunking_config/chunking_template persistence; [chunking] config section`

---

## PR E — re-chunk & report

### Task 12: Report renderer (AC 41)

**Files:**
- Modify: the Library RAG surface widget (renderer consuming ONLY `legacy_chunk_report`; omit-when-zero)
- Test: `Tests/UI/` via `app.run_test()` against a scratch data dir (§10.5)

- [ ] Steps: failing test (renders line when N>0 via scope-service payload; absent when zero; ignores `capability`/`missing_methods`/`fallback_enabled`) → implement → green → `feat(library): legacy-chunk report line — first renderer`

### Task 13: Re-chunk action + forced re-index + guard + policy (AC 42–46)

**Files:**
- Create: the re-chunk worker (Library surface; §10.2 per-item flow, §10.2.1 forced re-index: vector-store delete by deterministic id → mark indexing state BEFORE add → `index_batch_optimized` direct; post-commit best-effort; query-cache clear)
- Modify: worker-group guard (separate group + mutual in-flight refusals with Backfill — NEVER `exclusive=True`, spec §10.3), `runtime_policy/registry.py` (reuse `rag.admin.launch` verb; if semantically wrong at implementation, add verb at BOTH sites named in §10.4)
- Test: `Tests/UI/` + `Tests/RuntimePolicy/`

- [ ] Steps: failing tests — AC 42 (count drops by exactly N-rechunked; remainder = skipped+failed), AC 43 (RAG search returns NEW text after re-chunk + cache cleared), AC 44 (interrupt between delete and add leaves item re-indexable), AC 45 (mutual refusal, no cancellation — assert neither worker died), AC 46 (policy pin) → implement → green → `feat(library): re-chunk older-engine items — forced re-index, worker guard, policy`

### Task 14: Design tokens, CHANGELOG, guides, final sweep (AC 47, 51–54)

**Files:**
- Modify: new controls' CSS (`$ds-*` tokens only; `Select` colors-only unless geometry measured), `build_css.py` rebuild, `CHANGELOG.md` (breaking export + rolling-summarize entries), `Docs/User_Guide/` (picker + Library controls + offset-basis caveat, re-verified stamps)
- Test: CSS/token guards + AC 54's targeted suite list

- [ ] Steps: token/CSS compliance + guards → CHANGELOG + docs → the full AC-54 targeted sweep (`Tests/Chunking/ Tests/DB/ Tests/Media_DB/ Tests/RAG_Admin/ Tests/RAG/ Tests/Local_Ingestion/ Tests/Packaging/ Tests/Architecture/ Tests/RuntimePolicy/ Tests/UI/ Tests/integration/` + revived upstream template tests) → `docs(chunking): PR-E compliance, changelog, user guides`

---

## Self-Review (run at save time)

1. **Spec coverage:** ACs 1–54 map to Tasks 1–14 as follows — 1–3:T2, 4–5:T3, 6:T4, 7–8:T5, 9–13:T5, 14–15:T6, 16–23+29:T7, 24–28:T8, 30–33:T9, 34–37:T10, 38–40:T11, 41:T12, 42–46:T13, 47–54:T14/T1 (48–50,51-partial in T1; 51 lint/changelog tail + 52–53 in T14). §13.1's closed threads need no tasks.
2. **Ordering:** T1 before everything (AC 48); A(4–6) < B(7–8) < C(9) load-bearing; D(10–11), E(12–14) after. PR 0 independent.
3. **Type consistency:** `apply_template(template: dict, text, options) -> list[dict]` used by T7 (seed execution), T10 (ingest), T13 (re-chunk); `resolve_template(db, name) -> dict | None` used by T10/T13; `validate_template(template) -> {valid, errors, warnings}` used by T6/T8; flat-contract top-level `chunk_index` 0-based per #1's convention.
4. **Placeholders:** fixture tables (T6 Step 1) are generate-at-execution with recorded provenance, per AC 14's own provability clause — not placeholders; every other code block is complete or explicitly completed-per-test-contract (none such remain beyond #1's precedent style).
