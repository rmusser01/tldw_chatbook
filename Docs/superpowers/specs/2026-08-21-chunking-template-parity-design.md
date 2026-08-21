# Chunking Template Parity & Convergence — Design Spec

**Date:** 2026-08-21
**Status:** Draft, maintainer-approved in brainstorming (five decisions + a
self-review pass recorded in §13)
**Sub-project:** 2 of 6 in the Chunking Parity & Agent Tools program
**Depends on:** sub-project #1 (PR #1852, merged `f557195bb`) — the vendored
engine, the manifest sync, `Chunk_Lib` as a compat shim, Media DB schema v6
(`chunk_engine_version`), ADR-073
**Author:** brainstormed with the maintainer; every claim below was verified
against the two working trees named in §0.

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
   (`DB_Management/media_db/runtime/chunk_template_ops.py:216`). It is a
   column, never a key inside `template_json`. The "v2" in the program's
   prose traces to `Chunking/__init__.py:351` `__version__ = '2.0.0'` (the
   engine module version) and the `test_chunker_v2.py` filename — the engine
   rewrite, not a template schema.
2. **`template_library/` does not exist.** Not in git (`git ls-tree -r HEAD --
   .../Chunking/` returns 46 files, zero JSON), not on disk, not gitignored.
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
  (`Chunking/templates/*.json`, 14 files, read by
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
  `_chunking_options_from_template` (`local_rag_admin_service.py:350-377`)
  returns `(method, options)` only; `preprocessing` and `postprocessing`
  stages are dropped on the floor. It also constructs `Chunker(options=…,
  template_manager=object())` (`:329`) — a decoy that is never consulted,
  because `Chunker.__init__` only touches `template_manager` when `template`
  is truthy (`Chunk_Lib.py:709`).
- **Zero UI, zero production callers.** The editor widget was deleted in
  `551193f86` (task-253). `Event_Handlers/template_events.py` (6 event
  classes) has zero importers repo-wide. `MediaDetailsWidget` — the only
  widget that reads templates — is itself unreachable (no production
  importer; its template `Select` is hardcoded to `[("Default","default"),
  ("Custom Configuration","custom")]` and never populated from the DB). No
  ingest path offers a template. `apply_template` has no production caller.
- **A validator nobody calls.** `ChunkingInteropService.validate_template_json`
  (`chunking_interop_library.py:670-721`) validates only the chatbook pipeline
  shape and is invoked by nothing — not `create_template`, not
  `update_template`. Meanwhile `validate_template_config` hard-raises
  "Server retrieval-admin backend is required" in local mode
  (`rag_admin_scope_service.py:303-306`).

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
  (§8.3 explains why that is not a half-measure).
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

### 4.2 The three shapes this replaces

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
and on every one of the 192 local/remote refs; **zero** branches claim v7, so
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
**requires** `uuid: UUID` and `version: int`. Locally `version` is fabricated
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

### 5.3 Migration mechanics (the parts that bite)

SQLite cannot drop/rename columns portably at the versions in play, so v7 is
a **table rebuild**: create `ChunkingTemplates_v7`, copy with conversion, drop
the old, rename, then recreate indices **and the update-timestamp trigger**.
The trigger (`update_chunking_templates_timestamp`, `:1300-1304`) is dropped
with the table and must be recreated by name — a rebuild that forgets it
leaves `updated_at` silently frozen. This is an explicit AC (§12).

Per-row conversion:

- `uuid` ← generated (`uuid4`) per existing row.
- `is_builtin` ← old `is_system`.
- `tags` ← extracted from `template_json` (`tags`, else `metadata.tags`),
  then **removed from the JSON body** so the column is the only home.
- `template_json` ← pipeline→flat conversion (§5.4).
- `version` ← 1, `deleted` ← 0.

Seeds: the five old rows are replaced by the server's six built-ins (§5.5).
Replacement uses the server's idempotent seeding semantics
(`media_db/api.py:876-900`): a built-in name that already exists as a
**custom** (user-authored) row is left alone and logged, never overwritten.

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

**Method-name repair is part of conversion.** `structural` → `structure_aware`,
`hierarchical` → `structure_aware` (with `hierarchical: true` in config),
`contextual` → `sentences` (its distinguishing behavior was the `add_context`
postprocess op, which now survives as a real postprocessing stage). Rows are
converted, then their methods validated against the live registry; an
unrepairable method is left as-is and reported by the validator, never
silently rewritten to `words`.

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

**Note the method-name divergence** these seeds carry: `book_chapters` uses
`ebook_chapters`, which chatbook supports; every built-in's method is
validated at seed time by the same validator users get (§7), and a seed that
fails validation fails the migration test — not the user's startup.

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
vendoring is therefore a manifest-list change plus a sync run — nothing else.

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

### 6.4 The offset contract — a real degradation, stated loudly

#1 established that persisted chunks carry non-NULL `start_char`/`end_char`
(the DB round-trip AC; the media navigation reader consumes those columns).
The server's postprocess stage **flattens chunks to bare strings**
(`templates.py:341`), destroying offsets, unless the stage has no operations
(`:313-314`, which early-returns to preserve hierarchical metadata).

So a template whose postprocessing **rewrites chunk text** cannot preserve
source offsets — the text no longer corresponds to a source span. The ruling:

- Postprocessing absent, empty, or composed only of **non-rewriting**
  operations (`filter_empty`, `add_metadata`) → offsets **preserved**.
- Postprocessing containing a **rewriting** operation (`add_overlap`,
  `merge_small`, `format_chunks`) → offsets persisted as **NULL**, with
  `metadata.offsets_dropped = "<operation>"` naming the cause.

Chunks with NULL offsets are already a supported state (every pre-v6 row is
one). The alternative — refusing rewriting operations outright — would make
two of the six built-ins unusable. This is documented in the user guide, not
just here: a user who picks `chat_conversation` (which uses `add_overlap`)
should not be surprised that the media viewer cannot highlight its chunks.

---

## 7. Validation

The server's validate logic lives in a FastAPI endpoint
(`endpoints/chunking_templates.py:782-992`) entangled with pydantic response
models, `get_media_db_for_user`, and `core.Metrics`. It is not vendorable, so
chatbook **re-implements its semantics** in `RAG_Admin/template_validation.py`,
matching the endpoint check-for-check:

- `chunking` present; `chunking.method` present; method in the **live**
  registry (`Chunker().get_available_methods()`, not a hardcoded list —
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
in local mode (`rag_admin_scope_service.py:303-306`) and routes to this.
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

### 9.1 Resolution order at ingest

```
per-media  Media.chunking_config["template"]     (column exists today)
   else    config [chunking] default_template     (new, optional)
   else    plain method/size/overlap options      (today's behavior)
```

Read via `get_cli_setting("chunking", "default_template")` — the **two-arg**
form. The dotted-path form (`get_cli_setting("chunking.default_template")`) is
silently broken repo-wide (flat lookup; recorded in the Console inline-images
memory) and must not be used.

### 9.2 One seam, not four

Rather than teaching `Book_Ingestion_Lib`, `PDF_Processing_Lib`,
`Image_Processing_Lib`, and `local_file_ingestion` about templates,
`app.py:_ingest_job_options` (`:2812-3160`) resolves the name to a template
dict **once** and threads it through the existing `chunk_options` payload
into `improved_chunking_process(template=…)` (which already forwards to
`Chunker`, `Chunk_Lib.py:1476-1481`). Per-site change: none.

Persisted chunks fill the `chunking_template` / `chunking_params` columns
that migration v1→v2 added and nothing has ever written
(`Client_Media_DB_v2.py:1307-1312`), alongside #1's `chunk_engine_version`
stamp.

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

### 10.1 The report gets its first renderer

Settings → Library & RAG renders the legacy-chunk line
(`"Chunked by an older engine: N items"`) sourced through
`RAGAdminScopeService.get_template_diagnostics` — the `rag.admin.observe.local`
action. The line is **omit-when-empty** (the service already returns `""` and
omits the key on a fully-stamped library), so a clean library shows nothing
rather than a zero.

Note the count semantics fixed in #1's Qodo round: it counts media **items**
(`COUNT(DISTINCT media_id)`), not chunk rows.

### 10.2 The action

A **"Re-chunk older-engine items"** control sits next to "Backfill RAG index"
in the same Library & RAG group. Its worker, per item:

1. re-chunk the source text through the template-aware path (§9.1 resolution
   order, so a per-media template is honored);
2. **replace** that item's `UnvectorizedMediaChunks` rows in one transaction,
   the new rows carrying the current `chunk_engine_version` stamp;
3. re-index that item in the vector store (stale-chunk delete + re-add), the
   step that makes the two disjoint stores agree — **conditional** on the
   semantic index being enabled and present, skipped with a note otherwise.

Items whose source content is empty are skipped and counted. Failures are
per-item: one bad item does not abort the batch, and the summary reports
`N re-chunked, M skipped, K failed` — never a bare "done".

### 10.3 Worker exclusion — the correction that matters

The obvious design ("share the backfill's worker group, mark it exclusive")
is **wrong**. Textual's `exclusive=True` within a group **cancels** the
running worker rather than refusing to start — so a user pressing Re-chunk
mid-backfill would silently kill the backfill. (This is the task-228 lesson:
never `run_worker(exclusive=True)` without understanding the group's
cancellation semantics.)

Instead: a **separate** worker group plus an explicit mutual in-flight guard.
Each control refuses with a notice ("Backfill is still running") while the
other is active. Both workers pre-resolve their services outside the
transient event loop, per the #700-hardened pattern the backfill worker
already documents (`settings_screen.py:13939,13969,13984`).

### 10.4 Policy surface

The action launches through the scope service and therefore needs a policy
action id in the `rag.admin.*` family (alongside `observe` and `launch`).
It is named in the spec so `Tests/RuntimePolicy/` stays honest, and so the
capability is declarable rather than incidentally allowed.

---

## 11. Backlog tasks to file

Filed, not fixed here. IDs assigned only after a **cross-branch, cross-worktree
sweep** (the repo's seven-collision history; most recently three branches
contending for ADR-072 during #1).

**chatbook:**

1. `MediaDetailsWidget` — revive or delete (unreachable; hardcoded template
   Select; the only widget that reads `Media.chunking_config`).
2. `chunk_preview_modal.py` — orphan with no live importer.
3. `Utils/embedding_templates.py` + `Widgets/embedding_template_selector.py`
   — a dead pair, unrelated to chunking templates but found adjacent.
4. `LocalRAGAdminService.get_template_diagnostics` returns **hardcoded**
   `capability: "native"` / `missing_methods: []` — it never probes the
   backing service, so a broken backend reports healthy.
5. `get_documents_using_template` matches by raw substring
   (`LIKE '%"template": "<name>"%'`) while `get_template_statistics` uses
   `json_extract` on the same column in the same file.
6. `_parse_template_config` swallows malformed JSON to `{}`, yielding a
   silent `("words", {})` default instead of an error.
7. The two dead-skipped upstream test files
   (`test_upstream_chunking_templates.py`,
   `test_chunking_templates_validate_schema.py`) — partially revived by this
   sub-project (§12); the residue (the initialization half, the
   `_shims/DB_Management` / `_shims/AuthNZ` imports they expect) needs its own
   ruling.

**tldw_server** (filed in that repo's tracker):

8. `template_library/` is documented as shipping built-ins but has never
   existed in git; `TemplateManager` creates it empty and the real built-ins
   are a hardcoded Python fallback.
9. The flat→internal mapper is copy-pasted three times, and two copies lack
   the missing-`chunking` guard (raising `KeyError` into a generic handler).
10. Validate/runtime asymmetries: `operation` required at validate but `type`
    accepted at runtime; no unknown-operation-name check; unknown top-level
    keys silently dropped by the pydantic pass.
11. `academic_paper` / `code_documentation` / `chat_conversation` are defined
    **twice** with divergent content (`template_initialization.py` flat vs
    `templates.py` stage-based).
12. The hardcoded method fallback list (`chunking_templates.py:832`) is stale:
    11 names, omitting `fixed_size`, `code`, `code_ast`.

---

## 12. Acceptance criteria

**Shape & storage**

1. Media DB migrates v6 → v7: `ChunkingTemplates` carries `uuid`, `tags`,
   `is_builtin`, `version`, `deleted`; the partial unique index on live names
   exists; **the update-timestamp trigger survives the rebuild** (an update
   bumps `updated_at`).
2. Existing rows convert pipeline→flat with method-name repair; an
   unconvertible row is preserved under `metadata._unconverted` and warned,
   never dropped.
3. The six upstream built-ins are seeded; a built-in name already present as a
   custom row is left alone and logged.
4. Soft delete works end to end: `delete_template(hard_delete=False)` sets
   `deleted`, the row vanishes from listings, and the name can be re-used.
   `version` increments on update.

**Processor**

5. `templates.py` is vendored from the **existing** pin, listed in the
   manifest, and reproduced byte-faithfully by the sync script (import-rewrite
   lines excepted); the sync remains idempotent and no new shim module is
   needed (`is_truthy` already resolves through `_shims/testing.py`).
6. Exactly one flat→`ChunkingTemplate` mapper exists in chatbook, and it
   raises a clear error (not `KeyError`) on a template missing `chunking`.
7. No chatbook production module constructs `TemplateManager`,
   `TemplateClassifier`, or `TemplateLearner`.
8. `apply_template` runs preprocessing **and** postprocessing: a template with
   a `normalize_whitespace` preprocess and an `add_overlap` postprocess
   produces output demonstrably different from the chunking stage alone.
9. Offsets survive non-rewriting postprocessing; a rewriting operation yields
   NULL offsets plus `metadata.offsets_dropped` naming the operation.

**Validation**

10. Local validation matches the server endpoint check-for-check (§7),
    returns `{valid, errors, warnings}` rather than raising, and uses the
    **live** method registry.
11. `validate_template_config` no longer hard-raises in local mode;
    `create_template` / `update_template` refuse invalid templates.

**Convergence**

12. The file-based store, `chunking_templates.py`, `template_events.py`, the
    orphan SQL, and `table_serializer.py` (plus its dead kwarg threading and
    doc section) are gone; the suite is green without them.
13. A bare-name `template=` on `Chunker` / `improved_chunking_process` raises
    a clear error naming the resolver; a pre-resolved dict works.
14. `test_app_import_weight.py` stays green (no DB dependency reaches the
    shim).
15. Rolling-summarize raises on provider failure through both the engine and
    the shim; no code path can persist a `[Summarization failed…]` marker.

**Consumers**

16. Ingest honors per-media template → config default → plain options, and
    persisted chunks carry `chunking_template` / `chunking_params` alongside
    `chunk_engine_version`.
17. The Library ingest picker lists DB templates, defaults to "None (manual
    settings)" (preserving today's behavior), escapes markup in labels, and
    persists the choice per media item.

**Re-chunk & report**

18. Settings → Library & RAG renders the legacy-chunk line when non-zero and
    omits it when zero.
19. "Re-chunk older-engine items" re-chunks legacy-stamped items, replaces
    their chunk rows **stamped**, and re-indexes them in the vector store when
    the index is present; the legacy count drops to zero afterward.
20. Re-chunk and Backfill cannot run simultaneously, and neither cancels the
    other — the second press is refused with a notice.
21. Per-item failures do not abort the batch; the summary reports
    re-chunked / skipped / failed counts.

**Process**

22. Every §11 task is filed with IDs verified against all branches and
    worktrees; a short ADR records the convergence decision — **077 at spec
    time** (074/075/076 are taken, and 076 is *already* double-claimed by two
    branches), re-swept immediately before the commit that adds it.
23. `tldw_chatbook.Chunking`'s package exports no longer name the deleted file
    store, and `Tests/Packaging/test_installed_distribution.py` is updated in
    the same commit (§8.1.1).
24. `Docs/User_Guide/` pages for the ingest picker and the Settings controls
    are updated with re-verified stamps, including the offset caveat (§6.4).
25. Targeted suites pass: Chunking, Media_DB, RAG_Admin, Local_Ingestion,
    RAG, Packaging, plus the revived upstream template tests.

---

## 13. Decisions taken during brainstorming

Recorded so later sub-projects inherit them.

1. **Full convergence** over additive parity: the server's flat shape becomes
   chatbook's one shape; one store; the broken seeds are replaced, not fixed.
2. **Re-chunk ships in #2** (closing #1's Q3), as a Settings control that
   touches **both** stores, with the report line as its first renderer.
3. **Templates get real consumers in #2**: the ingest write path **and** a
   picker UI — not headless parity.
4. **Rolling-summarize is fail-closed everywhere** (closing #1's review
   ruling); markers are data corruption, nothing depends on them.
5. **Vendor `templates.py`** from the existing pin rather than
   re-implementing; `template_initialization.py` stays out; validation is
   re-implemented locally to match the endpoint. Discovered defects on both
   sides are **filed** (§11).

Self-review corrections folded in before writing (each is a fact-driven
change, not a preference): the offset degradation (§6.4), service-layer-only
name resolution (§8.2), the worker-exclusion inversion (§10.3), the
trigger-recreation and fresh-DB migration hazards (§5.3), fencing the
vendored-but-unused classes (§6.3), and the two-arg config accessor (§9.1).

**Closed thread:** #1's review asked to revisit `Tests/Chunking/conftest.py`'s
namespace injection "when #2 vendors templates (upstream's `__init__` would
then export these natively)". It does not apply: chatbook's
`engine/__init__.py` is **chatbook-authored** per #1's spec §5.1, not synced,
so vendoring `templates.py` changes nothing about what the package root
exports. The injection stays as-is.
