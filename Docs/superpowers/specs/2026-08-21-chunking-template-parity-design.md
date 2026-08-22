# Chunking Template Parity & Convergence — Design Spec

**Date:** 2026-08-21
**Status:** Draft, maintainer-approved in brainstorming (eight decisions in
§13), then revised against three adversarial reviews — fact-check, design
critique, and repo-lessons audit. What they changed is recorded in §13.1.
**Sub-project:** 2 of 6 in the Chunking Parity & Agent Tools program
**Depends on:** sub-project #1 (PR #1852, merged `f557195bb`) — the vendored
engine, the manifest sync, `Chunk_Lib` as a compat shim, Media DB schema v6
(`chunk_engine_version`), ADR-073
**Author:** brainstormed with the maintainer. Every claim was verified against
the two working trees named in §0; claims the review disproved are corrected
in place and flagged, so a reader can trust what survives.

---

## 0. Provenance & upstream pin

Provenance is split by side, as in #1's spec:

- **chatbook-side facts** were verified against `origin/dev` at `e31a18d45`
  (worktree `.worktrees/chunking-template-parity`).
- **`tldw_server`-side facts** were verified against the **same pin #1
  vendored from**: `dev` @ `385afa951922c8a9dc2002c675bb6cad65e4ac23`,
  read via a temporary worktree at `/tmp/tldw_server_sync`.

This sub-project does **not** move the pin. Vendoring a new file from the
existing pin keeps the tree reproducible by the existing sync script; a pin
bump is its own task with its own parity re-proof.

### 0.1 The program's premise for #2 was wrong — corrected here

#1's spec (§2) described this sub-project as:

> **Template v2 parity.** Server's flat `{preprocessing, chunking,
> postprocessing, tags, version: 2}` schema, `ChunkingTemplates` migration,
> DB↔internal-stages mapper, `template_library` port, validation matching the
> server's validate endpoint.

Three of those five clauses do not describe the server at this pin:

1. **There is no `version: 2` in any template.** The only `version` in the
   server's template system is `ChunkingTemplates.version`, an `INTEGER NOT
   NULL DEFAULT 1` **row-revision counter** incremented on every update
   (`DB_Management/media_db/runtime/chunk_template_ops.py:217`). It is a
   column, never a key inside `template_json`. The "v2" in the program's
   prose traces to `Chunking/__init__.py:351` `__version__ = '2.0.0'` (the
   engine module version) and the `test_chunker_v2.py` filename — the engine
   rewrite, not a template schema.
2. **`template_library/` does not exist.** Not in git (`git ls-tree -r HEAD --
   .../Chunking/` returns 45 files, zero JSON), not on disk, not gitignored.
   The server's own `README.md:21,31,142` and `Docs/Chunking/
   Chunking_Templates.md` describe it as shipping built-ins — the directory is
   phantom. `TemplateManager.__init__` *creates* it empty
   (`templates.py:546`), and `load_builtin_templates()` falls through both
   filesystem strategies to a **hardcoded Python fallback**
   (`template_initialization.py:129-209`). There is nothing to "port".
3. **There is no v1→v2 migration to mirror.** The server accepts **two**
   template shapes concurrently and permanently, branching on the presence of
   a `"stages"` key (`templates.py:648` vs `:658`), with no version field, no
   migration, and no rejection.

The two clauses that do hold — a `ChunkingTemplates` migration and validation
matching the validate endpoint — are specced below. The parity target is the
server's **flat shape**, not a fictional v2. §11 files the documentation
defect back to the server.

---

## 1. Why

Three separate problems converge on one surface.

**chatbook's chunking templates are a dead, broken, forked surface.** Today:

- **Two stores in two incompatible shapes.** The Media DB's
  `ChunkingTemplates` table (`Client_Media_DB_v2.py:1284-1304`) holds five
  seeded templates in a chatbook-only shape (`{name, description,
  base_method, pipeline[], metadata}`); a *second*, file-based store
  (`Chunking/templates/*.json`, 13 files, read by
  `Chunking/chunking_templates.py`) holds a different set in the same
  chatbook shape. Nothing reconciles them. `academic_paper` exists in both
  with **different definitions** (DB: `base_method: "structural"`, max_size
  500; file: `base_method: "semantic"`, max_size 800, threshold 0.7, plus
  pre/postprocess ops). `LocalRAGAdminService` reads only the DB;
  `Chunker(template=…)` reads only the files.
- **Three of the five seeded templates crash on apply.** `academic_paper`
  (`structural`), `code_documentation` (`hierarchical`), and `contextual`
  (`contextual`) name methods that exist in neither
  `_LEGACY_METHOD_MAP` (`Chunk_Lib.py:307-317`), the engine-native passthrough
  set (`:343-350`), nor `ChunkingMethod` (`engine/base.py:18-32` — which has
  `structure_aware`, **not** `structural`). Applying any of them raises
  `InvalidChunkingMethodError`. Only `general` and `conversational` work.
- **Apply silently drops two thirds of every template.**
  `_chunking_options_from_template` (`local_rag_admin_service.py:350-376`)
  returns `(method, options)` only; `preprocessing` and `postprocessing`
  stages are dropped on the floor. It also constructs `Chunker(options=…,
  template_manager=object())` (`:329`) — a decoy that is never consulted,
  because `Chunker.__init__` only touches `template_manager` when `template`
  is truthy (`Chunk_Lib.py:709`).
- **Zero UI, zero production callers.** The editor widget was deleted in
  `551193f86` (task-253). `Event_Handlers/template_events.py` (a base
  class plus 6 subclasses) has zero importers repo-wide. `MediaDetailsWidget` — the only
  widget that reads templates — is itself unreachable (no production
  importer; its template `Select` is hardcoded to `[("Default","default"),
  ("Custom Configuration","custom")]` and never populated from the DB). No
  ingest path offers a template. `apply_template` has no production caller.
- **A validator nobody calls.** `ChunkingInteropService.validate_template_json`
  (`chunking_interop_library.py:670-721`) validates only the chatbook pipeline
  shape and is invoked by nothing — not `create_template`, not
  `update_template`. Meanwhile `validate_template_config` hard-raises
  "Server retrieval-admin backend is required" in local mode
  (`rag_admin_scope_service.py:306-309`).

**#1 left two rulings explicitly deferred to this sub-project.** Spec Q3
deferred the user-triggered **re-chunk/re-index action**; the final
whole-branch review asked for an explicit ruling on the **rolling-summarize
dual contract** ("the promised Phase-B convergence never landed").

**The engine-version stamp #1 shipped has no way to be seen or acted on.**
`get_legacy_chunk_report_line()` returns `"Chunked by an older engine: N
items"` onto a diagnostics dict that **no renderer reads** — verified
exhaustively: the only consumers of `get_template_diagnostics` are the
services themselves, the scope router, and tests. And the one live re-index
trigger (Settings → Library & RAG → "Backfill RAG index") re-chunks into the
**vector store only**; it never touches `UnvectorizedMediaChunks`, the table
that carries the stamp. The two stores are disjoint, so today no user action
can clear a legacy count.

---

## 2. Goals

1. One canonical template shape in chatbook — the server's flat shape — with
   one store (the Media DB table) and the server's six working built-ins.
2. Templates that actually run: preprocessing, chunking, and postprocessing
   stages all execute, via the server's own vendored processor.
3. Templates that a user can pick and that ingestion honors.
4. A user-reachable path from "N items were chunked by an older engine" to
   "they aren't any more", touching both the persistent chunk table and the
   vector index.
5. The two deferred #1 rulings closed: re-chunk shipped, rolling-summarize
   fail-closed.
6. The discovered defects on both sides recorded as backlog tasks rather than
   silently absorbed.

## 3. Non-goals

- **The classifier / auto-selection engine** (`auto_planner.py`, scoring,
  `min_score`/`priority` matching) — that is sub-project #3. The `classifier`
  **block** is stored and validated here (it is part of the shape); nothing
  *acts* on it.
- **Agent chunking tools** (#4), **student workflow** (#5), **LLM-dependent
  extras** (#6: `propositions`, `auto_boundary_assistant`, `async_chunker`).
- **A template editor UI.** #2 ships a *picker*; authoring stays service-layer
  (§8.2 explains the layering reason: resolution needs a DB handle, so it
  lives above the import-light shim).
- **Moving the vendor pin.**
- **Reviving `MediaDetailsWidget`** or the other dead widgets found during
  exploration — filed as tasks (§11), not fixed here.

---

## 4. The canonical shape

### 4.1 What chatbook adopts

`template_json` holds exactly the server's flat shape:

```json
{
  "preprocessing": [{"operation": "normalize_whitespace", "config": {"max_line_breaks": 1}}],
  "chunking": {"method": "sentences", "config": {"max_size": 10, "overlap": 2}},
  "postprocessing": [{"operation": "add_overlap", "config": {"size": 100, "marker": "---"}}],
  "classifier": {"media_types": ["document"], "min_score": 0.5, "priority": 10},
  "metadata": {}
}
```

`name`, `description`, and `tags` are **columns**, not JSON keys (§5).
`chunking` is the only required block; `chunking.method` is required within
it. `Operation` is `{operation: str, config?: object}`.

The server also accepts `{type, params}` spellings and a stage-based shape
(§0.1); chatbook **writes** only the canonical spelling above. Its *reader*
inherits the server processor's tolerance for free (the vendored code accepts
both), which is a compatibility bonus, not a contract chatbook maintains.

### 4.2 The three shapes in play

| Shape | Where it lived | Fate |
|---|---|---|
| chatbook pipeline (`base_method` + `pipeline[]`) | Media DB seeds + `Chunking/templates/*.json` | **converted** (DB rows) / **deleted** (files) |
| server flat | inbound from `tldw_api` server mode | **becomes canonical** |
| server stage-based (`stages[]`) | server-side only, never in chatbook | not adopted; the vendored reader still accepts it |

### 4.3 The one mapper

The server implements flat→`ChunkingTemplate` **three times** — `templates.py:658-691`,
`endpoints/chunking_templates.py:712-744`, `endpoints/chunking.py:293-321` —
and two of the three copies do not guard a missing `chunking` key (raising
`KeyError` into a generic handler). chatbook implements it **once**, in
`Chunking/template_runtime.py` (§6.2), with the guard. The triplication is
filed back to the server (§11).

---

## 5. Data model — Media DB v6 → v7

**Version swept 2026-08-21:** `_CURRENT_SCHEMA_VERSION = 6` on `origin/dev`
and on every one of the 204 local/remote refs (80 at v4, 98 at v5, 26 at v6); **zero** branches claim v7, so
v7 is free at spec time. Per the repo's collision history this must be
**re-verified at implementation start and again at merge** — a cancelled CI
run cannot catch a version collision, and a merge that lands between those two
points is exactly how the v19/v20 collision happened.

*(Sweep trap, learned here: in zsh, `git show "$b:path"` silently applies the
`:t` history modifier and mangles the ref — the sweep returns clean for every
branch. Brace it: `git show "${b}:path"`.)*

### 5.1 Current state

```sql
CREATE TABLE ChunkingTemplates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    template_json TEXT NOT NULL,
    is_system BOOLEAN DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```
(`Client_Media_DB_v2.py:1284-1304`, arriving in migration v1→v2.) No `uuid`,
no `version`, no `tags`, no soft-delete — while the client-side response
schema `ChunkingTemplateResponse` (`tldw_api/rag_admin_schemas.py:46-57`)
requires `uuid: UUID` (`version: int` has a default of 1). Locally `version` is fabricated
as `1` (`rag_admin_normalizers.py:83`), `uuid` is never produced, and tags are
smuggled into `template_json["metadata"]["tags"]` plus a duplicate
`template_json["tags"]` (`local_rag_admin_service.py:123-135`).

### 5.2 Target state

```sql
CREATE TABLE ChunkingTemplates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    template_json TEXT NOT NULL,
    tags TEXT,                                   -- JSON-encoded list
    is_builtin BOOLEAN NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX idx_chunking_templates_name_live
    ON ChunkingTemplates(name) WHERE deleted = 0;
CREATE INDEX idx_chunking_templates_is_builtin ON ChunkingTemplates(is_builtin);
CREATE INDEX idx_chunking_templates_deleted ON ChunkingTemplates(deleted);
```

Column names follow the **server's** (`is_builtin`, not `is_system`) so the
local and server records normalize identically. The partial unique index
replaces the bare `UNIQUE(name)` so a soft-deleted row never blocks a re-add
— the server's semantics (`idx_template_name_not_deleted`).

### 5.2.1 The migration breaks the only CRUD layer — they ship together

`ChunkingInteropService` (`Chunking/chunking_interop_library.py`) is the sole
CRUD path to this table and is **raw SQL against the v6 column set**. After
v7, verified breakages: `create_template` inserts `is_system` (gone) and omits
the now-`NOT NULL UNIQUE` `uuid` → every create raises; `get_all_templates`
filters and orders on `is_system` → `OperationalError`; `delete_template`,
`duplicate_template`, `_row_to_template_dict`, and `get_template_statistics`
all read `row["is_system"]` → `KeyError`. `LocalRAGAdminService:189,195` and
`rag_admin_normalizers.py:79` read it too.

Additionally **no query anywhere filters `deleted = 0`**, so the moment soft
delete ships (§5.2), deleted templates reappear in every listing and name
lookup.

Therefore the CRUD rewrite is **part of the migration change, in the same
PR** — a schema and its only reader cannot ship apart. That rewrite also:

- adds `ChunkingTemplates` and its columns to `sql_validation.VALID_TABLES`
  (it is absent today, so any use of the media DB's version helper for the new
  `version` column would raise `InputError: Invalid table name`);
- routes writes through `MediaDatabase.transaction()` — the current code does
  `conn = self.media_db.get_connection(); … conn.commit()`, which the house
  style forbids;
- makes the normalizer source `uuid` and `version` from the DB instead of
  fabricating `version: 1` (`rag_admin_normalizers.py:83`), which was the
  original motivation for the new columns.

### 5.3 Migration mechanics (the parts that bite)

SQLite cannot drop/rename columns portably at the versions in play, so v7 is
a **table rebuild**: create `ChunkingTemplates_v7`, copy with conversion, drop
the old, rename, then recreate indices **and the update-timestamp trigger**.
The trigger (`update_chunking_templates_timestamp`, `:1300-1304`) is dropped
with the table and must be recreated by name — a rebuild that forgets it
leaves `updated_at` silently frozen. This is an explicit AC (§12).

**ADR-030 governs how this executes.** *"Every versioned media schema
transition, including its `schema_version` update and seed data, executes in
one real SQLite transaction… Media migration scripts are executed
statement-by-statement using SQLite's own complete-statement parser while the
normal transaction context is active"* — because `executescript()` implicitly
commits and defeats rollback. So the DDL, the per-row conversion, the seeding,
and the version bump all run inside the single ADR-030 transaction via
`_execute_transactional_script` (`:963-1001`); a stray `conn.executescript`
would silently destroy the step's atomicity. A seeded mid-rebuild failure must
leave the DB at v6 with the original table intact, and that is an AC.

Three verified specifics: this would be the **first table rebuild in this
file** (`grep "RENAME TO\|DROP TABLE" Client_Media_DB_v2.py` → zero hits;
precedents exist only in `Subscriptions_DB.py:765` and
`Library_Ingest_Jobs_DB.py:155`), so it gets extra review rather than
pattern-matching; foreign keys are ON (`:290`, `:766`) but **no table
references `ChunkingTemplates`**, so no FK dance is needed (assert it rather
than trust it); and `_execute_transactional_script` splits on
`sqlite3.complete_statement`, so the trigger's `BEGIN…END` body survives the
split intact. Use `BEGIN IMMEDIATE` for the migration — this machine routinely
runs concurrent sessions, and a DROP+RENAME widens the two-instance race from
"one failed ALTER" to "unopenable DB".

**Downgrade becomes impossible, and that voids one of ADR-073's safety nets.**
`_initialize_schema` hard-raises on a newer DB (`:1478-1480`), so after v7 a
`git revert` of the code leaves the media DB unopenable — and ADR-073 named
`git revert` as a program-wide safety net. Stated here, in the new ADR, and in
the release notes; it is a real cost of the schema change, not an oversight.

**The historical fixture must be genuine.** The v6-shaped DB the conversion
tests run against is produced by patching `_CURRENT_SCHEMA_VERSION` to 6 and
letting the production chain build it (the `Tests/ChaChaNotesDB/
historical_bootstrap.py` pattern). Hand-dropping a table and stamping the
version back is explicitly forbidden — that fixture style broke serially
across four repair tasks in this repo.

Per-row conversion:

- `uuid` ← generated (`uuid4`) per existing row.
- `is_builtin` ← old `is_system`.
- `tags` ← extracted from `template_json` (`tags`, else `metadata.tags`),
  then **removed from the JSON body** so the column is the only home.
- `template_json` ← pipeline→flat conversion (§5.4).
- `version` ← 1, `deleted` ← 0.

**Convert vs replace — the precedence, since both rules cover the same rows.**
Rows with `is_system = 1` (the five original seeds) are **dropped and
re-seeded** from §5.5; every other row is **converted** per §5.4. `general`
and `conversational` — the only two seeds that work today, and the only sane
default — do not exist in the upstream six, so they are **converted and kept
as non-builtin rows** rather than deleted: nothing a user could already have
selected disappears. `contextual` is converted and kept the same way (minus
its lost `add_context` op). Replacement otherwise uses the server's idempotent
seeding semantics (`media_db/api.py:876-900`): a built-in name that already
exists as a **custom** row is left alone and logged, never overwritten.

**Fresh-DB path.** A fresh database runs every migration in order, so it
creates the v2 table with the five old seeds and then rebuilds it at v7.
That is correct but non-obvious; v1→v2 is deliberately left untouched
(rewriting history to "optimize" a fresh install is how migration chains
diverge between users who upgrade and users who install fresh).

### 5.4 Converting existing rows

The conversion is mechanical:
`{base_method, pipeline: [{stage, method, options, operations}]}` →
`{preprocessing: [ops from stage=="preprocess"], chunking: {method: chunk
stage's method or base_method, config: chunk stage's options},
postprocessing: [ops from stage=="postprocess"]}`, with `{type, params}`
operation keys rewritten to `{operation, config}`.

A row that cannot be converted (no chunk stage and no `base_method`, or
unparseable JSON) is **not silently dropped**: it is preserved with its
original body under `metadata._unconverted` plus a `chunking` block naming the
default method, and a WARNING names the row. In practice this affects nobody
— no UI to author custom rows has ever shipped — but a migration that eats
user data on a path "nobody uses" is exactly the migration that eats the one
user who did.

**Method-name repair is part of conversion.** `structural` →
`structure_aware`, `hierarchical` → `structure_aware` (with `hierarchical:
true` in config), `contextual` → `sentences`. Rows are converted, then their
methods validated against the live registry.

**Four operations are lost, and that is an accepted regression.** The deleted
file store registered `extract_metadata`, `section_detection`,
`code_block_detection`, and `add_context` (`chunking_templates.py:98-107`);
the vendored processor's registry has **none of them** (its ten are
`normalize_whitespace, remove_headers, extract_sections, clean_markdown,
detect_language, add_overlap, filter_empty, merge_small, add_metadata,
format_chunks`). *(An earlier draft claimed `contextual`'s `add_context`
"survives as a real postprocessing stage" — false.)* Per §7.1's own note, an
unregistered operation validates clean and is then **warned-and-skipped at
runtime**, so a converted row naming one becomes a silent no-op. The
conversion therefore maps what it can (`section_detection` → `extract_sections`
where the intent matches) and **drops the rest explicitly**, recording them in
`metadata._dropped_operations` so the loss is visible in the row rather than
only in a log line.

**Stored-invalid rows must stay repairable.** §7 makes create/update refuse
invalid templates, while conversion can mint rows that fail validation (an
unrepairable method, a `_unconverted` body). Without a ruling those rows could
never be saved again — the migration would mint templates the user cannot fix
through the product. The ruling: an invalid stored row is **listed with a
flag**, **refused at apply** with a named error, and **editable** — update
validates the *new* body only, so correcting a bad row is always possible.

**An unconvertible row is quarantined, not silently re-pointed.** An earlier
draft kept its name and gave it a default `chunking` block, which swaps a
template's behavior invisibly. Instead it is soft-deleted (`deleted = 1`) and
renamed `<name> (needs review)`, its original body preserved under
`metadata._unconverted`, and the count surfaced on the same Library
diagnostics line §10.1 adds. A row a user may have selected never silently
chunks differently.

### 5.5 The six built-ins

Lifted **as data** from `template_initialization.py:132-208` (the hardcoded
fallback that actually runs at this pin): `academic_paper`,
`code_documentation`, `chat_conversation`, `book_chapters`,
`transcript_dialogue`, `legal_document`. Each seed carries a provenance
comment naming the pin SHA and the upstream line range, so a future sync can
diff them.

These are *data*, not vendored code — they are not in the manifest's file
list and the sync script does not manage them. §11 files the upstream
`template_library/` phantom so this can eventually become a real port.

**All six were executed against chatbook's engine before this spec was
accepted**, because seeding templates that crash is precisely the bug §1
describes. Result: **six of six run clean.**

| Template | method | verdict |
|---|---|---|
| `academic_paper` | `sentences` | works; `extract_sections` metadata computed then discarded (see below) |
| `code_documentation` | `structure_aware` | works |
| `chat_conversation` | `sentences` | works |
| `book_chapters` | `ebook_chapters` | works (degenerates to 1 chunk on non-ebook input — behavioral, not a bug) |
| `transcript_dialogue` | `sentences` | works |
| `legal_document` | `paragraphs` | works |

All four distinct methods resolve against the live 14-method registry, and
every operation the six use is registered. Contrast the current seeds, whose
`structural` / `hierarchical` / `contextual` raise `InvalidChunkingMethodError`.

**Known degradation to document:** operations that *return* metadata
(`extract_sections`, `detect_language`) have it merged into `data["metadata"]`,
which `process_template` then **discards** (`templates.py:159-167`) — so
`academic_paper`'s section extraction runs and is thrown away. Filed upstream
(§11) and noted in the user guide rather than papered over.

**Seed validation runs at build/test time, not during a user's migration.**
Under ADR-030's single-transaction rule a validation failure at runtime would
roll back the entire migration; so a seed that fails validation fails the
test suite, and runtime seeding treats the six as pre-proven.

---

## 6. Vendoring the processor

### 6.1 What is vendored

`templates.py` (812 lines) joins the manifest's file list; the engine tree
goes 35 → 36 files, reproducible by the existing
`Helper_Scripts/sync_chunking_engine.py`. Its imports resolve as:

| Import | Resolution |
|---|---|
| stdlib, `loguru` | already available |
| `from .chunker import Chunker` | already vendored (#1) |
| `from .exceptions import TemplateError` | already vendored |
| `from .regex_safety import check_pattern, compile_flags` | already vendored |
| `from tldw_Server_API.app.core.testing import is_truthy` | **already satisfied** — `_shims/testing.py:9` exports `is_truthy`, and the sync script's second rewrite rule maps `tldw_Server_API.app.core` → `tldw_chatbook.Chunking._shims` (`sync_chunking_engine.py:376`) |

**No shim work is required.** An earlier draft of this spec claimed
`is_truthy` was a gap to fill; verification against the tree showed #1's shim
already exports it (used by `templates.py:53` via the existing rewrite). The
vendoring is therefore a manifest change plus a sync run — nothing else.

One mechanical detail: `templates.py` currently sits in the manifest's
**`excluded`** list, so vendoring is a **move between two lists**, not an
append. Leaving the name in both would make a sync run ambiguous.

**`template_initialization.py` is NOT vendored.** It imports
`DB_Management.db_path_utils` and `DB_Management.media_db.api`, which drag the
server's entire config + path + Media-DB stack. Its only genuinely portable
content is the six built-in dicts (§5.5) and `load_builtin_templates()`, a
stdlib-only function whose two filesystem strategies both dead-end at the
phantom directory. Shimming a DB layer to reach six dicts is shim work with
no parity value.

### 6.2 What chatbook adds — `Chunking/template_runtime.py`

One new module, the seam between the DB and the vendored processor:

- `template_from_record(record) -> ChunkingTemplate` — the single mapper
  (§4.3), with the missing-`chunking` guard the server's copies lack.
- `resolve_template(db, name) -> dict | None` — the **only** name→template
  resolution in the codebase.
- `apply_template(template, text, options) -> list[dict]` — invokes
  `TemplateProcessor.process_template` and normalizes the result to
  chatbook's flat chunk contract (§6.4).

### 6.3 Fencing the vendored surface

`templates.py` also contains `TemplateManager`, `TemplateClassifier`, and
`TemplateLearner`. chatbook uses **`TemplateProcessor` and the two dataclasses
only**:

- `TemplateManager.__init__` **mkdirs a templates directory on construction**
  (`templates.py:546`) and carries its own divergent 3-template in-memory
  store (`:556-614`) whose `academic_paper` differs from the DB-seeded one.
  Constructing it would resurrect exactly the two-stores problem §4 deletes.
- `TemplateClassifier` is #3's business (§3).
- `TemplateLearner.learn_boundaries` returns five hardcoded patterns and
  ignores its input entirely (`:797-812`) — nothing to consume.

A test pins that no chatbook production module constructs `TemplateManager`
(§12). The classes stay vendored-but-unused, exactly like other engine
surface #1 vendored without consuming.

### 6.4 The chunk contract — the processor supplies none of it

**An earlier draft of this spec got this badly wrong and was corrected by
measurement.** It ruled that offsets survive "non-rewriting" postprocessing.
They do not, and the distinction it drew does not exist.

What the vendored processor actually returns, measured by executing it
against chatbook's engine:

- `_run_chunk_stage`'s normal branch calls `chunker.chunk_text(...)`, which
  returns **bare strings** for every method, and wraps them as
  `{'text': c, 'metadata': {}}` (`templates.py:296-298`). Offsets live on a
  different engine method (`chunk_text_with_metadata` →
  `ChunkMetadata(start_char, end_char, …)`) that the processor never calls.
- Measured across the four cases the old ruling distinguished — no
  postprocess stage, an empty one, `filter_empty` only, `add_metadata` only —
  **every chunk came back metadata-empty**. There was never anything to
  preserve.
- The `:313-314` early return protects *hierarchical* metadata
  (`start_offset`/`end_offset`, different key names) on a branch reached only
  when `hierarchical` is set; the flatten at `:318-323` fires on **any**
  non-empty operation list, rewriting or not.
- Two of the old ruling's "non-rewriting" classifications were wrong anyway:
  `filter_empty` **deletes chunks** (breaking index alignment), and
  `add_metadata` **rewrites text** when `prefix`/`suffix` is configured
  (`templates.py:501-517`).
- And all six built-ins run a text-rewriting **preprocessing** op
  (`normalize_whitespace` or `clean_markdown`), so chunk offsets are relative
  to the transformed string, never to the stored `Media.content`.

So the template path yields **no chunk metadata at all** unless chatbook
supplies it. That collides with #1's flat contract in three concrete ways,
each verified: `Tests/Chunking/test_callsite_characterization.py:52-76`
asserts the DB round-trip leaves `start_char`/`end_char` non-NULL; a
*present-but-`None`* offset flows into `indexing_helpers.py:263` and then
`int(metadata.get("chunk_start", 0))` at `vector_store.py:611`, raising
**`TypeError` at search time**; and `local_media_reading_service.py:2442-2446`
silently falls back to `text[0:len(text)]`, so media navigation degrades to
"the whole document" with no error.

**Ruling (maintainer, 2026-08-21): synthesize the contract after the
pipeline.** `template_runtime.apply_template` reconstructs the flat chunk
contract rather than inheriting upstream's shape:

1. Offsets are computed with the **existing** helper
   `Chunk_Lib._synthesize_flat_offsets` (`Chunk_Lib.py:607-669`) — the same
   code that keeps today's ingest non-NULL (`:1561-1568`) — against the
   **preprocessed** text the chunks actually came from.
2. Because that basis may differ from `Media.content`, each chunk carries
   `metadata.offset_basis` = `"source"` when no preprocessing op rewrote the
   text, or `"preprocessed:<op>"` naming the first op that did. Consumers
   that need source-relative spans (navigation, citations) can test one key
   instead of guessing.
3. When offsets cannot be synthesized at all (a postprocessing op deleted or
   merged chunks so the mapping is lost), the keys are **omitted entirely** —
   never present-and-`None`. One normalization point in `apply_template`
   guarantees the in-memory contract never carries `None`, which is what
   prevents the `TypeError` above.
4. `chunk_index`, `total_chunks`, and `word_count` are likewise reconstructed,
   so template chunks are indistinguishable in shape from plain ones.

This keeps navigation and citations working for the common case, at the cost
of one normalization step this spec owns. The user guide states the caveat
for the uncommon case (a template that merges or drops chunks loses precise
source positions).

---

## 7. Validation

The server's validate logic lives in a FastAPI endpoint
(`endpoints/chunking_templates.py:782-992`) entangled with pydantic response
models, `get_media_db_for_user`, and `core.Metrics`. It is not vendorable, so
chatbook **re-implements its semantics** in `RAG_Admin/template_validation.py`,
matching the endpoint check-for-check:

- `chunking` present; `chunking.method` present; method in the **live**
  registry (`Chunking.engine.chunker.Chunker().get_available_methods()` — the
  ENGINE class; the `Chunk_Lib` shim `Chunker` every other citation in this
  spec means has no such method. Not a hardcoded list —
  upstream's fallback list at `:832` is stale, omitting `fixed_size`, `code`,
  `code_ast`; §11 files it).
- `hierarchical` bool; `hierarchical_template.boundaries` a list, ≤ 20 rules;
  each boundary has a `pattern`; pattern regex-safety via the already-vendored
  `engine/regex_safety.check_pattern` (max_len 256); `flags` limited to
  `i`/`m`; unanchored `.*`/`.+` → **warning**, not error.
- `classifier` strict-key allowlist (`media_types`, `filename_regex`,
  `title_regex`, `url_regex`, `tags`, `min_score`, `priority`); `min_score`
  numeric in [0,1]; `priority` int; regexes ≤ 128 chars and safety-checked.
- `preprocessing`/`postprocessing` are lists; each operation carries an
  `operation` key; the whole template is JSON-serializable.

Result shape mirrors the endpoint's: `{valid: bool, errors: [{field,
message}], warnings: [...]}` — never an exception for invalid input.

**Wiring:** `RAGAdminScopeService.validate_template_config` stops hard-raising
in local mode (`rag_admin_scope_service.py:306-309`) and routes to this.
`create_template` and `update_template` call it and **refuse** invalid
templates — closing the "validator nobody calls" gap (§1). Seeding calls it
too (§5.5).

### 7.1 Deliberate parity, including the warts

The endpoint has three known asymmetries: it requires `operation` while the
runtime also accepts `type` (so a `{type, params}` template runs but fails
validation); it never checks that an operation **name** is registered (an
unknown op validates clean, then is warned-and-skipped at runtime); and its
pydantic pass **silently drops unknown top-level keys** before the hand-rolled
checks see them.

chatbook **matches these** rather than diverging — parity means a template
that validates here validates there. All three are filed upstream (§11); if
the server fixes them, the fix arrives through the same channel as any other
parity update. The one thing chatbook does *not* copy is the silent drop of
`name`/`description`/`tags`: those are columns here, so they never enter the
validated body in the first place.

---

## 8. Convergence — what runs, what dies

### 8.1 Deletions

| Deleted | Why it is safe |
|---|---|
| `Chunking/templates/` (14 JSON files + `README.md` + `example_usage.py`) | the file store; its only production reader is the module below |
| `Chunking/chunking_templates.py` (`ChunkingTemplateManager`, `ChunkingPipeline`, the Pydantic-v1 `.dict()` calls at `:156,159,206`) | replaced by the vendored processor + `template_runtime`; also mkdirs a user dir merely on construction (`:87-93`). **Not importer-free — see §8.1.1** |
| `Event_Handlers/template_events.py` (6 classes) | zero importers repo-wide |
| `DB/migrations/add_chunking_config.sql` | orphan duplicate of inlined DDL; nothing reads `DB/migrations/*.sql` |
| `RAG_Search/table_serializer.py` (612 lines) + the dead `serialize_tables` kwarg threading (`enhanced_chunking_service.py:61,76,90` → `parent_child_adapter.py:437,455`) + its `ENHANCED_RAG_FEATURES.md:149` section | zero production importers; its only caller (`ECS.serialize_table`) died in #1's task 8. #1's deferred-minor, now collected |
| `ChunkingInteropService._template_cache` | incoherent by construction: `get_chunking_service()` returns a fresh instance per call while `LocalRAGAdminService` holds its own, so a second instance's writes leave the first stale |

`Tests/test_enhanced_rag.py`'s `test_table_serialization` goes with the
module it tests.

### 8.1.1 The file store is a published import — deletion is an API change

Unlike the other deletions, `chunking_templates.py` is **re-exported from the
package root**: `Chunking/__init__.py:19` imports `ChunkingTemplateManager`,
`ChunkingPipeline`, and friends into `tldw_chatbook.Chunking`'s public
namespace. Four call sites depend on that:

| Site | Handling |
|---|---|
| `Chunking/__init__.py:19` | exports removed; the module's replacements (`template_runtime`) are **not** added to the package root — nothing outside the service layer should resolve templates (§8.2) |
| `Chunk_Lib.py:711` | the name-resolution path being replaced anyway (§8.2) |
| `Chunking/templates/example_usage.py:168` | deleted with the file store |
| `Tests/Packaging/test_installed_distribution.py:285` | **pins the installed distribution's importable surface** — must be updated in the same commit, or the packaging suite fails on a name that no longer exists |
| `Tests/integration/test_core_functionality_integration.py:61`, `Tests/Chunking/test_chunking_templates.py` | rewritten against `template_runtime` + the vendored processor, or deleted where they only exercise deleted behavior |

This is the one deletion in §8.1 that changes a public import surface. It is
still the right call — the module is the *second* store this sub-project
exists to eliminate — but it is a breaking change to
`tldw_chatbook.Chunking`'s namespace and the CHANGELOG says so.

**Name collision to rule:** `Chunking/__init__.py:19` also exports
`ChunkingTemplate`, `ChunkingStage`, and `ChunkingOperation`, and the vendored
`templates.py` defines its **own** `ChunkingTemplate` dataclass — same public
name, different class. Only the vendored one survives, and it is **not**
re-exported at the package root (§8.2: nothing outside the service layer
resolves templates). "Both, briefly" is how import-time confusion ships.

### 8.1.2 Deleting the template JSONs is a packaging change (ADR-032)

ADR-032 makes runtime-owned template data a **distribution obligation** and
names *"all thirteen built-in `Chunking/templates/*.json` definitions"*
explicitly. An earlier draft named exactly one packaging site; there are five,
all verified:

- `pyproject.toml:497` — package-data `"tldw_chatbook.Chunking" = ["templates/*.json"]` (plus the `:479` exclude entry)
- `MANIFEST.in:12` — `recursive-include tldw_chatbook/Chunking/templates *.json`
- `Packaging/check_manifest.py:134,146,154,155,242` — required sdist/wheel contents, including `README.md` and `example_usage.py`
- `Tests/Packaging/test_installed_distribution.py:1475,1484,1485` — the **data** contract, distinct from the import pin at `:285`
- `Tests/test_enhanced_rag.py:145` — passes the `serialize_tables` kwarg being removed (a second site beyond `test_table_serialization`); `ENHANCED_RAG_FEATURES.md` references the serializer at `:35`, `:149`, `:206`

All of it moves in the same commit as the deletion, and the wheel/sdist
contract is re-proven against freshly built artifacts.

*(Count correction: the directory holds **13** JSON files, not 14 as an
earlier draft said twice; `template_events.py` defines **7** classes — a base
plus six subclasses — not 6.)*

### 8.2 Name resolution moves to the service layer

`Chunk_Lib.Chunker` and `improved_chunking_process` keep their `template=` /
`template_manager=` parameters (a #1 signature-compatibility AC), but the
semantics change: they accept a **pre-resolved template dict**. A bare name
string raises a clear error naming `template_runtime.resolve_template`.

The reason is layering. Resolving a name requires a Media DB handle;
`Chunk_Lib` is the import-light shim that #1's `test_app_import_weight.py`
guards. Threading a database into it to satisfy a parameter **no production
caller uses** (verified: only `example_usage.py`, deleted above, and tests)
would trade a guarded boot-time invariant for nothing.

### 8.3 Rolling-summarize: fail-closed everywhere

The shim's legacy port appends `[Summarization failed for this part: …]`
markers into chunk text on provider error (`Chunk_Lib.py:1103`, `:1112`,
`:1120`) and never raises, while the engine strategy raises `ProcessingError`
(`engine/strategies/rolling_summarize.py:271-288`).

The three marker branches become **raises**, with a clear message. The
caller contract (the payload-dict `llm_call_function`, pinned by
`Tests/Chunking/test_chunk_lib_shim.py:146-175`) is unchanged — only the
failure path is. Evidence this is safe: **zero** tests reference the marker
strings, and **no production caller supplies an `llm_call_function` at all**
(every one of the 14 call sites passes none), so the marker branches are
unreachable outside tests and external library use. Silently persisting
`[Summarization failed]` as document text is data corruption with a friendly
face.

Mutation-verified pin tests replace the untested branches.

---

## 9. Making templates real

### 9.1 Resolution order, per path

The order differs by path, because at **first ingest** no media row exists
yet — an earlier draft stated one order "at ingest" that could not apply
there:

```
ingest      picker choice / batch default        (§9.3)
   else     config [chunking] default_template   (new, optional)
   else     plain method/size/overlap options    (today's behavior)

re-chunk    Media.chunking_config["template"]    (the stored per-media choice)
   else     config [chunking] default_template
   else     plain options
```

Read via `get_cli_setting("chunking", "default_template")`. *(An earlier draft
justified this by calling the dotted form "silently broken repo-wide" —
**stale**: `config.py:5841+` now resolves dotted paths against the nested TOML
tree, and both forms were measured returning the same value. The two-arg form
is still the house style; the false rationale is removed.)* The `[chunking]`
section does not exist in the shipped config today, so it is added to the
config template and defaults, with a test asserting the real loader emits it.

**Precedence ruling.** Today `Chunker.__init__` applies template options first
and then `template_options.update(options)` — *"Template options have lower
priority than explicit options"* (`Chunk_Lib.py:733-735`) — while
`_ingest_job_options` **always** sets `method`/`max_size`/`overlap`
(`app.py:2907-2918` plus per-group method injection). Left alone, a picked
template would be overridden on every path, every time: **the picker would be
inert.** The ruling: when a template is resolved, its chunk-stage options win
over the ingest builder's *defaults*, and only a value the user explicitly
changed in the ingest form overrides the template. An AC pins that picking a
template actually changes the persisted chunks.

**Not-found handling.** A per-media or configured template name that no
longer resolves (soft-deleted, renamed) does **not** silently fall through:
ingest fails the item with a named error, and re-chunk skips it and counts it.
Silent fallback to different chunking is how a user gets a library chunked
two ways without knowing.

### 9.2 The ingest seam — six sites, not one

**An earlier draft claimed "per-site change: none". That was wrong**, traced
end to end:

| Path | Site | Today |
|---|---|---|
| pdf / document / ebook | `local_file_ingestion.py:1033/1058/1084` | pass `chunk_options` whole — but land in `RAG_Search/chunking_service.improved_chunking_process(text, options)`, whose signature has **no `template` parameter** (`:188-190`) |
| audio / video | `:1186-1192`, `:1309-1315` | re-project `chunk_options` **key-by-key** into six scalar kwargs; an unknown key is dropped |
| image | `:1131` | passes `chunk_options=None` |
| plain text | `_chunk_text_for_ingest`, `:607-614` | rebuilds a **fresh three-key dict** |
| server mode | `Library/server_ingest_request.py:350-486` | an independent options builder that forwards only detected-type extras |

A `template` key *inside* `chunk_options` is inert even where the dict
survives, because `Chunker.__init__` gates on the **keyword** (`if template:`,
`Chunk_Lib.py:709`), never on `options["template"]`.

The work is therefore: add a `template` parameter to
`RAG_Search.chunking_service.improved_chunking_process` and forward it as the
kwarg (covers pdf/document/ebook); widen the audio/video projections and the
plain-text dict; and decide server mode explicitly — **the picker is hidden in
server mode** rather than accepted and ignored. Image ingestion does not chunk
and is documented as unaffected.

Persisted chunks fill the `chunking_template` / `chunking_params` columns that
migration v1→v2 added and **nothing has ever written** (verified:
`Client_Media_DB_v2.py:1307-1312`), alongside #1's `chunk_engine_version`
stamp. The `Media.chunking_config` writer is likewise new — the only existing
writer is the dead `MediaDetailsWidget` — and its JSON shape must satisfy both
existing readers: `get_documents_using_template`'s
`LIKE '%"template": "<name>"%'` and `get_template_statistics`'
`json_extract(chunking_config, '$.template')`.

### 9.3 The ingest picker

The Library ingest flow (`Widgets/Library/library_ingest_canvas.py`) gains a
template `Select`, populated from the DB via the scope service, defaulting to
**"None (manual settings)"** — which preserves today's behavior exactly. The
choice is stored per-media in `Media.chunking_config`.

Two rules from repo lessons apply: template names are user-authorable, so
they are `escape_markup`-ed in the Select's option labels (the Button-label
trap from the Home/Library redesign); and the populate call is off the mount
path (mount-time DB populate is the documented cause of "(0)" count bugs in
the Notes rebuild). Per repo policy, the matching `Docs/User_Guide/` page is
updated with a re-verified stamp.

---

## 10. Re-chunk and the report

### 10.0 Where this lives — ADR-003, not Settings

**An earlier draft put both surfaces in Settings → Library & RAG. That
contradicts an Accepted ADR.** ADR-003 states: *"RAG indexing, embedding model
lifecycle, **chunking templates**, collection management, and workspace
eligibility remain outside this Settings slice"*, and its rejected-alternatives
table names *"add full RAG indexing, embeddings, and chunk-template management
to Settings"* as rejected because it *"would flatten workflow ownership into
Settings and produce a large, hard-to-review PR."*

The existing "Backfill RAG index" button does sit in that slice, but it landed
under task-541 with **no ADR amendment** — undocumented drift, not sanctioned
precedent (verified: no ADR supersedes or amends 003).

**Ruling (maintainer, 2026-08-21): honor ADR-003 as written.** The report line
and the re-chunk action live on the **Library** RAG/media surface, which
ADR-003 names as the owner of source browsing and RAG execution. No amendment
is written. The Backfill button's drift is filed as its own task (§11) so the
boundary stops being ambiguous — this spec does not move or re-home it.

### 10.1 The report gets its first renderer

The Library RAG surface renders the legacy-chunk line
(`"Chunked by an older engine: N items"`) sourced through
`RAGAdminScopeService.get_template_diagnostics` — the `rag.admin.observe.local`
action. The line is **omit-when-empty** (the service already returns `""` and
omits the key on a fully-stamped library), so a clean library shows nothing
rather than a zero.

The renderer consumes **only** the `legacy_chunk_report` field. The same
payload's `capability` / `missing_methods` / `fallback_enabled` are
**hardcoded** and never probe the backing service (§11 item 4), so surfacing
them would render a fabricated health claim.

Note the count semantics fixed in #1's Qodo round: it counts media **items**
(`COUNT(DISTINCT media_id)`), not chunk rows.

### 10.2 The action

A **"Re-chunk older-engine items"** control on the Library RAG surface. Its
worker, per item:

1. re-chunk the source text through the template-aware path (§9.1 re-chunk
   resolution order, so a stored per-media template is honored);
2. **replace** that item's `UnvectorizedMediaChunks` rows in one transaction,
   the new rows carrying the current `chunk_engine_version` stamp;
3. re-index that item — see §10.2.1, which is not the obvious call.

Items whose source content is empty are skipped and counted. Failures are
per-item: one bad item does not abort the batch, and the summary reports
`N re-chunked, M skipped, K failed` — never a bare "done".

**Hard delete is forced, and that is a ruling, not a detail.**
`UnvectorizedMediaChunks` has `UNIQUE(media_id, chunk_index, chunk_type)`, so
soft-deleted old rows would still occupy the index and collide with the
re-insert. Replacement is therefore a hard `DELETE` on a **synced** table,
leaving no sync-log record for other clients. Accepted deliberately: these are
**derived** rows regenerated from an intact source, which is also why this
action is outside ADR-055's destructive-action patterns (ADR-055 governs
destroying user-authored content, not regenerating a projection). The spec
states that rather than leaving it to a reviewer to infer.

### 10.2.1 The re-index step is not the obvious call

The obvious implementation — reuse `ingestion_indexing.index_entries`, as #1
recommended for indexing work — is a **silent no-op**. That function opens
with `if not indexing_db.needs_reindexing(item_id, item_type, last_modified):
skipped += 1; continue` (`ingestion_indexing.py:706-717`), and re-chunking
does not touch the `Media` row, so `last_modified` is unchanged and **every
item skips**. The summary would honestly report "N re-chunked" while the
vector store never moved — and an AC written against a mock would pass.

So the worker must force the re-index explicitly: delete the document from the
vector store by its deterministic id (`ingestion_indexing.py:558-563`), mark
the indexing-state row **before** the add, then call `index_batch_optimized`
directly. Marking first matters: if the process dies between delete and add,
the item is re-indexable on the next run instead of being permanently absent
from search while `needs_reindexing` reports it current.

Two more consequences, both from ADR-030's derived-index contract: the index
write is **post-commit and best-effort** (source write commits first), and the
owning RAG service's **query cache is cleared** after a successful re-index —
otherwise a search immediately after re-chunking still serves pre-re-chunk
snippets. The whole step is conditional on the semantic index being enabled
and present, skipped with a note otherwise.

### 10.3 Worker exclusion — the correction that matters

The obvious design ("share the backfill's worker group, mark it exclusive")
is **wrong**. Textual's `exclusive=True` within a group **cancels** the
running worker rather than refusing to start — so a user pressing Re-chunk
mid-backfill would silently kill the backfill. (This is the task-228 lesson:
never `run_worker(exclusive=True)` without understanding the group's
cancellation semantics.)

Instead: a **separate** worker group plus an explicit mutual in-flight guard.
The re-chunk control refuses with a notice while a backfill is running, and
vice versa. Both workers pre-resolve their services outside the transient
event loop, per the #700-hardened pattern the backfill worker documents
(`settings_screen.py:13939,13969,13984`) — that pattern is copied even though
this surface now lives in Library (§10.0).

*(Note for future readers: CLAUDE.md gotcha 9 says "mark `exclusive=True` to
prevent duplicates". This is a deliberate, measured deviation — Textual 8.2.8's
`WorkerManager._new_worker` documents `exclusive: Cancel all workers in the
same group`. Do not "fix" it back.)*

### 10.4 Policy surface

The action launches through the scope service and therefore needs a policy
action id. Two options, and the spec picks the lazy one: **reuse the existing
`rag.admin.launch` verb**, which is exactly what the backfill-shaped trigger
already means. Adding a fifth `rag.admin.*` verb would require editing
`runtime_policy/registry.py:1260-1279` **and** the exact-equality literal block
at `Tests/RuntimePolicy/test_runtime_policy_core.py:1216-1241`, and the
tempting shortcut there — extending `DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS`
(`registry.py:185`) — is **shared**, so it would silently grant the verb to
other capabilities. If implementation finds `launch` semantically wrong, adding
a verb is a documented decision with those two edit sites, not an incidental
one.

---

## 10.5 Verification — and the constraint that shapes it

**A schema bump is a one-way door for every other worktree on this machine.**
`lessons-live-verification.md`: *"a single live launch of a schema-bumping
branch migrates the shared database and every concurrent worktree still on the
old version stops opening it… `TLDW_CONFIG_PATH` alone does NOT protect you
here."* This repo routinely runs concurrent sessions, and #2 both bumps the
schema **and** adds UI whose ACs are behavioral.

The rules, therefore:

- **Migration** is proven against `tmp_path` / in-memory Media DBs only, with
  the genuine v6 bootstrap fixture (§5.3). Never against the user's library.
- **UI** is verified against a **scratch data dir** — a scratch config that
  redirects `[paths] data_dir`, not merely `TLDW_CONFIG_PATH` — or deferred
  until the branch is what dev is on.
- **Do not launch the app against the shared library DB while v7 is in
  flight.** This is stated in the plan, not just here.
- Automated UI coverage uses `app.run_test()` (AppTest is unavailable here),
  which is how the picker and the Library controls get ACs without a live
  launch.

## 10.6 Phasing — six PRs

The design critic's assessment, adopted: this is a strict superset of what #1
split into three PRs, so it ships as **six dependency-ordered PRs**, each
independently mergeable, revertable, and reviewable.

| PR | Contents | Why it stands alone |
|---|---|---|
| **0** | §8.3 rolling-summarize fail-closed; delete `table_serializer.py` + dead kwarg + doc sections; `template_events.py`; orphan SQL | Pure #1 residue, zero template coupling. Small and mergeable immediately. |
| **A** | Vendor `templates.py` (manifest move); `Chunking/template_runtime.py` (one mapper, resolver, `apply_template` with the §6.4 synthesis); `template_validation.py`; `TemplateManager` fencing | Read-side only. No schema, no user-visible change. Provable with fixtures. |
| **B** | v6→v7 migration + conversion + quarantine + seeds **+ the CRUD/normalizer rewrite (§5.2.1)** + soft delete + validate-on-write + `sql_validation` rows | Storage and its only reader are atomic — they cannot ship apart. |
| **C** | Delete the file store; `Chunking/__init__` export change; `Chunk_Lib` name-resolution change; **all five packaging sites** (§8.1.2); inventory regeneration | The breaking API change, isolated so a revert is surgical. |
| **D** | Ingest write path (all six seams, §9.2); precedence fix; picker; `Media.chunking_config` writer; docs | First user-visible change. |
| **E** | Library report renderer; re-chunk action + forced re-index (§10.2.1); worker guard; policy; docs | Largest net-new UI; depends on D for resolution. |

Ordering is load-bearing: **A before B** (the mapper must exist before rows
are converted to a shape only it can read), and **B before C** (deleting the
file store while `Chunk_Lib` still resolves from it breaks resolution).

---

## 11. Backlog tasks to file

Filed, not fixed here. The filing procedure matters as much as the list —
`lessons-backlog-hygiene.md` records this recurring **ten-plus times**:

- **Every id here will be five digits** (true max across remote refs is
  **19601**), and the CLI **silently misbehaves on five-digit ids**: `backlog
  task <id>` / `task edit <id>` do nothing, and can create a file literally
  named `task-task- - .md`. So: **write the task file directly**, with a
  hand-authored AC block. Never pass `-p` (it corrupts the assigned id) and
  never comma-join `--ac` (it writes one un-checkable run-on criterion).
- **Sweep refs *and* untracked files.** A ref-only sweep misses uncommitted
  claims in sibling worktrees — which is exactly how ADR-042 ended up
  double-claimed (a tracked `042-watchlists-reader-first-ia.md` plus an
  untracked `042-design-token-system…` in the main checkout). Scan
  `.worktrees/*/backlog/` for untracked files too, derive the id at the
  instant of filing, and verify afterwards with `backlog task <id> --plain`
  plus `git status backlog/`.
- **Re-verify each finding is still live** on `origin/dev` before it becomes a
  task (`git log -S'<symbol>' origin/dev -- <path>`), and grep the board for an
  existing filing. A finding about untouched code is a *report* until then;
  several items below were observed during exploration and may have been
  fixed since.
- Each item below gets a **one-line outcome plus 2–3 AC bullets** at filing
  time, so nobody invents run-on criteria later.

**The five upstream items need a home this repo can act on.** There is no
existing convention for cross-repo tasks (#1 filed none), and an id on this
board cannot be verified against the server's tracker. Ruling: record items
8–12 in a tracked artifact beside the pin —
`tldw_chatbook/Chunking/engine/UPSTREAM_DEFECTS.md`, adjacent to
`VENDOR_MANIFEST.toml` — with the upstream file:line, the chatbook-side
impact, and an issue link once filed upstream. That file, not a task id, is
what the AC requires.

**chatbook:**

1. `MediaDetailsWidget` — **decide its fate** (a revive-or-delete fork is not
   an outcome; the task's outcome is "a decision is recorded and executed").
   Unreachable; hardcoded template Select; the only existing writer of
   `Media.chunking_config`.
2. `chunk_preview_modal.py` — orphan with no live importer.
3a. `Utils/embedding_templates.py` — dead module, no importer outside tests.
3b. `Widgets/embedding_template_selector.py` — dead widget; pairs with 3a but
   files separately (one task, one unit of work).
4. `LocalRAGAdminService.get_template_diagnostics` returns **hardcoded**
   `capability: "native"` / `missing_methods: []` — it never probes the
   backing service, so a broken backend reports healthy.
5. `get_documents_using_template` matches by raw substring
   (`LIKE '%"template": "<name>"%'`) while `get_template_statistics` uses
   `json_extract` on the same column in the same file.
6. `_parse_template_config` swallows malformed JSON to `{}`, yielding a
   silent `("words", {})` default instead of an error.
7. The Settings "Backfill RAG index" control is ADR-003 drift — it sits in a
   slice the ADR excludes, with no amendment. Decide: amend the ADR, or move
   the control to Library beside this sub-project's own (§10.0).
8. The two dead-skipped upstream test files
   (`test_upstream_chunking_templates.py`,
   `test_chunking_templates_validate_schema.py`) — partially revived by this
   sub-project (§12); the residue (the initialization half, the
   `_shims/DB_Management` / `_shims/AuthNZ` imports they expect) needs its own
   ruling.

**tldw_server** — recorded in `Chunking/engine/UPSTREAM_DEFECTS.md` (above),
each with the upstream file:line and its chatbook-side consequence:

9. `template_library/` is documented as shipping built-ins but has never
   existed in git; `TemplateManager` creates it empty and the real built-ins
   are a hardcoded Python fallback.
10. The flat→internal mapper is copy-pasted three times, and two copies lack
    the missing-`chunking` guard (raising `KeyError` into a generic handler).
11. Validate/runtime asymmetries: `operation` required at validate but `type`
    accepted at runtime; no unknown-operation-name check; unknown top-level
    keys silently dropped by the pydantic pass. *(This is why §7 replicates
    them deliberately — parity, not endorsement.)*
12. `academic_paper` / `code_documentation` / `chat_conversation` are defined
    **twice** — same content, divergent **schema spelling**
    (`template_initialization.py` `{operation, config}` flat vs
    `templates.py` `{type, params}` stage-based). *(An earlier draft said the
    content diverged; it does not.)*
13. The hardcoded method fallback list (`chunking_templates.py:832`) is stale:
    11 names, omitting `fixed_size`, `code`, `code_ast`. *(This is why §7 uses
    the live registry instead.)*
14. `process_template` collects preprocessing-produced metadata into
    `data["metadata"]` and never merges it into the returned chunks
    (`templates.py:159-168`), so `extract_sections` / `detect_language`
    produce nothing observable — the degradation §5.5 documents.

---

## 12. Acceptance criteria

Grouped by the PR that owns them (§10.6). Each is one checkable assertion —
run-on criteria cannot be ticked off independently, which is how a Definition
of Done becomes all-or-nothing.

**PR 0 — #1 residue**

1. Rolling-summarize raises on provider failure through both the engine and
   the shim.
2. Neither marker string can be persisted — **both** prefixes are pinned
   (`[Summarization failed…]` *and* `[Summarization error…]`; they are
   different strings, and an AC naming only the first leaves a branch alive).
3. Each new guard is shown red under a seeded mutation before being accepted.
4. `table_serializer.py`, its `serialize_tables` kwarg threading, its doc
   sections, `template_events.py`, and the orphan SQL are gone; the named
   suites are green without them.
5. The persistent diagnostic inventory is regenerated
   (`scripts/check_persistent_diagnostic_inventory.py`) and its row diff
   hand-reviewed in the same commit — `Docs/security/
   production-diagnostic-inventory.json` carries rows for two deleted modules.

**PR A — processor**

6. `templates.py` is vendored from the existing pin, **moved** from the
   manifest's `excluded` list to its vendored list, and reproduced
   byte-faithfully by an idempotent sync (import-rewrite lines excepted); no
   new shim module is required.
7. Exactly one flat→`ChunkingTemplate` mapper exists, and it raises a clear
   error (not `KeyError`) on a template missing `chunking`. Pinned by an
   enumeration guard with its own `test_the_guard_can_see_what_it_guards`
   self-check.
8. `template_runtime.resolve_template` is the only name→template resolution in
   the codebase (same guard style).
9. No production module constructs `TemplateManager`, `TemplateClassifier`, or
   `TemplateLearner`. Observed the way the code cannot fake: the templates
   directory does **not** exist on disk after a full boot-and-ingest run, with
   a positive control that constructs one and watches the probe flip.
10. `apply_template` runs preprocessing **and** postprocessing: a fixed input
    under a fixed template produces an **exact expected output** that differs
    from the chunking stage alone (a pinned value, not "demonstrably
    different").
11. Template chunks carry the full flat contract — synthesized offsets,
    `chunk_index`, `total_chunks`, `word_count` — and `metadata.offset_basis`
    names `"source"` or `"preprocessed:<op>"` (§6.4).
12. No chunk dict ever carries a present-but-`None` offset; when offsets are
    unsynthesizable the keys are absent. Pinned by a test that indexes a
    template-chunked item and runs a RAG search **without** `TypeError`.
13. A template-chunked item's media navigation returns chunk-sized content,
    not the whole document.
14. Local validation implements every check in §7, returns
    `{valid, errors, warnings}` rather than raising, and resolves methods
    against the **live engine registry**. Pinned by a fixture table generated
    once from the pinned endpoint source (input → expected result, with
    upstream line ranges recorded) — otherwise "matches the server" is
    unprovable and undriftable.
15. §7.1's three deliberate parity warts are pinned by tests, so a later
    "fix" cannot silently break parity.

**PR B — storage**

16. Media DB migrates v6 → v7 with `uuid`, `tags`, `is_builtin`, `version`,
    `deleted`, and the partial unique index on live names.
17. The update-timestamp trigger survives the rebuild (an update bumps
    `updated_at`).
18. DDL, row conversion, seeding, and the version bump execute in **one**
    ADR-030 transaction, statement-by-statement; a seeded mid-rebuild failure
    leaves the DB at v6 with the original table and rows intact.
19. The historical v6 fixture is produced by bootstrapping at a patched
    `_CURRENT_SCHEMA_VERSION`, not by dropping tables and stamping a version
    back.
20. `is_system = 1` rows are dropped and re-seeded; every other row converts;
    `general`, `conversational`, and `contextual` survive as non-builtin rows
    (nothing a user could have selected disappears).
21. An unconvertible row is quarantined — soft-deleted, renamed
    `<name> (needs review)`, body preserved — never silently re-pointed at a
    default method.
22. Dropped operations are recorded in `metadata._dropped_operations`.
23. The six built-ins seed and **all six execute** against the live engine.
24. A stored-invalid template is listed with a flag, refused at apply with a
    named error, and still **editable** (update validates the new body only).
25. Soft delete works end to end: the row leaves listings, the name is
    re-usable, and `version` increments on update.
26. `ChunkingInteropService` and the normalizers are rewritten with the
    schema: no `is_system` anywhere in the tree, every read filters
    `deleted = 0`, every write supplies `uuid`, writes go through
    `transaction()`, and `uuid`/`version` are sourced from the DB rather than
    fabricated.
27. `ChunkingTemplates` and its columns are registered in `sql_validation`.
28. `Tests/DB/test_media_db_schema_v6.py`'s version pin is updated, and new
    migration tests assert `== _CURRENT_SCHEMA_VERSION` and subset deltas
    rather than literals.
29. A fresh install lands at v7 with the six built-ins and none of the five
    old seeds.

**PR C — convergence**

30. The file store, `chunking_templates.py`, and `Chunking/templates/` are
    gone; `tldw_chatbook.Chunking`'s exports no longer name them; only the
    vendored `ChunkingTemplate` survives, and it is not re-exported at the
    package root.
31. All five packaging sites move in the same commit (§8.1.2), and the
    wheel/sdist contract is re-proven against freshly built artifacts.
32. A bare-name `template=` on `Chunker` / `improved_chunking_process` raises
    a **named exception type** (pinned, not "a clear error") pointing at
    `resolve_template`; a pre-resolved dict works. `template_manager=` is
    documented as accepted-and-ignored and pinned as such.
33. `test_app_import_weight.py` stays green (no DB dependency reaches the
    shim).

**PR D — consumers**

34. Ingest resolves picker/batch → config default → plain options; re-chunk
    resolves stored per-media → config default → plain options (§9.1).
35. A resolved template's chunk-stage options beat the ingest builder's
    defaults; only a user-changed form value overrides the template.
36. **Governance, not arrival:** ingesting one fixture under two different
    templates produces demonstrably different persisted chunk rows, and the
    "None" default produces byte-identical output to today's path. Asserted
    per media-type family (pdf/document/ebook, audio/video, plain text).
37. An unresolvable template name fails the ingest item with a named error and
    is skipped-and-counted by re-chunk — never a silent fallback.
38. Persisted chunks carry `chunking_template` / `chunking_params` alongside
    `chunk_engine_version`, and `Media.chunking_config` is written in a shape
    both existing readers understand (`LIKE '%"template": …'` and
    `json_extract($.template)`).
39. The picker lists DB templates, defaults to "None (manual settings)",
    escapes markup in labels, populates **off** the mount path, and is hidden
    in server mode.
40. `[chunking]` exists in the shipped config template/defaults, and a test
    asserts the real loader emits the section.

**PR E — re-chunk & report**

41. The Library surface renders the legacy-chunk line when non-zero, omits it
    when zero, and consumes **only** `legacy_chunk_report`.
42. Re-chunk replaces chunk rows **stamped**, and the legacy count drops by
    exactly the number reported re-chunked — the remainder explained by the
    skipped/failed counts (never "drops to zero", which contradicts skipping).
43. After a re-chunk, a RAG search returns the **new** chunk text — proving
    the forced re-index (§10.2.1), not the `needs_reindexing` no-op — and the
    owning service's query cache is cleared.
44. An interrupted re-chunk leaves the item re-indexable on the next run
    rather than permanently absent from search.
45. Re-chunk and Backfill cannot run simultaneously, and neither cancels the
    other; the second press is refused with a notice.
46. The re-chunk action's policy id is registered and pinned in
    `Tests/RuntimePolicy/`.
47. New controls define rest/hover/focus/disabled states using `$ds-*` tokens
    (no raw hex); the new `Select` commits to colors-only styling unless
    geometry is measured; new classes are styled or registered, the CSS bundle
    is rebuilt with `build_css.py`, and the CSS/token guards pass.

**Process (all PRs)**

48. The ADR is created **before implementation begins** (AGENTS.md), records
    the convergence decision, and states that v7 makes downgrade impossible
    (voiding ADR-073's revert net). Number swept across refs **and untracked
    worktree files** at the moment of writing.
49. Every §11 chatbook item is re-verified live on `origin/dev`, grepped
    against the board for an existing filing, then written directly as a task
    file with per-item ACs; the five upstream items are recorded in
    `Chunking/engine/UPSTREAM_DEFECTS.md`.
50. Licence hygiene re-checked after the manifest change (ADR-073): both
    licence files still ship and `pyproject`'s `license-files` still names
    them.
51. Linter and formatter succeed; the CHANGELOG records the breaking export
    change and the rolling-summarize behavior change; the zsh `${b}:path`
    sweep trap is added to `lessons-backlog-hygiene.md` with its incident.
52. `Docs/User_Guide/` pages for the picker and the Library controls are
    updated with re-verified stamps, including the offset-basis caveat.
53. Verification honors §10.5: migrations against temp DBs, UI against a
    scratch data dir or deferred, no live launch against the shared library
    while v7 is in flight.
54. Targeted suites pass: `Tests/Chunking/`, `Tests/DB/`, `Tests/Media_DB/`,
    `Tests/RAG_Admin/`, `Tests/RAG/`, `Tests/Local_Ingestion/`,
    `Tests/Packaging/`, `Tests/Architecture/`, `Tests/RuntimePolicy/`,
    `Tests/UI/`, `Tests/integration/`, plus the revived upstream template
    tests.

---

## 13. Decisions taken, and how this spec was corrected

Recorded so later sub-projects inherit them.

**Maintainer decisions (brainstorming, 2026-08-21):**

1. **Full convergence** over additive parity: the server's flat shape becomes
   chatbook's one shape; one store; the broken seeds are replaced, not fixed.
2. **Re-chunk ships in #2** (closing #1's Q3), touching **both** stores, with
   the report line as its first renderer.
3. **Templates get real consumers in #2**: the ingest write path **and** a
   picker UI — not headless parity.
4. **Rolling-summarize is fail-closed everywhere** (closing #1's review
   ruling); markers are data corruption, nothing depends on them.
5. **Vendor `templates.py`** from the existing pin rather than
   re-implementing; `template_initialization.py` stays out; validation is
   re-implemented locally. Discovered defects on both sides are **filed**.

**Maintainer decisions after adversarial review (2026-08-21):**

6. **The surface honors ADR-003** — report line and re-chunk action live in
   **Library**, not Settings. No ADR amendment; the existing Backfill button's
   drift is filed separately (§10.0).
7. **Offsets are synthesized after the pipeline**, not inherited from the
   processor and not abandoned (§6.4) — navigation and citations keep working.
8. **Scope holds, delivered as six phased PRs** (§10.6) rather than one.

### 13.1 What the review changed — and why the first draft was wrong

Three independent reviews (fact-check against both trees, design critique,
repo-lessons audit) ran against the first draft. The fact-check cleared the
biggest risk — **all six upstream built-ins execute cleanly on chatbook's
engine**, so §5.5 does not recreate the crash bug §1 describes — but four
rulings did not survive:

| First draft said | Reality | Now |
|---|---|---|
| Offsets survive "non-rewriting" postprocessing (§6.4) | The chunk stage returns **bare strings** on every path; all four measured cases were metadata-empty. Two "non-rewriting" ops were misclassified. All six built-ins rewrite text in *pre*processing anyway. | §6.4 synthesizes the whole flat contract, with `offset_basis` |
| "One seam, not four; per-site change: none" (§9.2) | **Six** seams; the public wrapper has no `template` parameter; 3 media types re-project options key-by-key; and explicit options **override** template options, so the picker would be inert | §9.2 names every seam; §9.1 adds a precedence ruling |
| `contextual`'s `add_context` "survives as a real stage" (§5.4) | `add_context` is not in the vendored registry — convergence **drops four operations** | §5.4 records them in `_dropped_operations` |
| Re-index via the existing indexing path (§10.2) | `needs_reindexing` skips every item because re-chunking doesn't touch `last_modified` — the feature would be a **silent no-op** | §10.2.1 forces it explicitly |

Structural gaps the review also closed: the migration **breaks the only CRUD
layer** (§5.2.1, unmentioned before); deleting the template JSONs touches
**five** packaging sites and an ADR-032 obligation, not one (§8.1.2); ADR-030
governs how the migration executes (§5.3); two repo gates the AC list never
ran (the diagnostic inventory, `Tests/DB`'s version pin); the schema-bump ↔
live-UI collision had no verification story (§10.5); and the five-digit task
CLI trap would have silently broken the §11 filing plan.

Factual drift corrected: 13 template JSONs not 14; a base class plus 6 event
subclasses not 6; 204 refs not 192; `Chunker().get_available_methods()` names
the **engine** class, not the shim; the dotted `get_cli_setting` form is no
longer broken (the prohibition stands on style, not on the false rationale);
`templates.py` is **moved** out of the manifest's `excluded` list, not
appended; and the two upstream `academic_paper` definitions differ in schema
spelling, not content.

Earlier self-review corrections that did survive: service-layer-only name
resolution (§8.2), the worker-exclusion inversion (§10.3), trigger recreation
and the fresh-DB path (§5.3), and fencing the vendored-but-unused classes
(§6.3).

**Closed thread:** #1's review asked to revisit `Tests/Chunking/conftest.py`'s
namespace injection "when #2 vendors templates (upstream's `__init__` would
then export these natively)". It does not apply: chatbook's
`engine/__init__.py` is **chatbook-authored** per #1's spec §5.1, not synced,
so vendoring `templates.py` changes nothing about what the package root
exports. The injection stays as-is.
