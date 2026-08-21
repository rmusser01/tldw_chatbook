# Chunking Engine Parity — Design Spec

**Date:** 2026-08-18
**Status:** Draft; independently re-verified 2026-08-18 and corrected in place —
tokens fallback chain (§5.5/Q2/§12), chunk-dict shape at the persistence seam and the
flat-contract requirement (§6/§6.3.2), shim re-export of module constants (§6.1/§6.2),
chatbook-authored `engine/__init__.py` (§5.1), defusedxml posture (§5.5/Q9), `decks`
table name (§2), matched size-ceiling defaults (§9), import-weight guard (§10.8),
§1 table count fixes, and the end-state surface table (§5.6, maintainer review)
**Sub-project:** 1 of 6 in the Chunking Parity & Agent Tools program
**Author:** brainstormed with the maintainer. Provenance is split by side (see §0):
chatbook-side facts were verified against the chatbook working tree; `tldw_server`-side
facts were re-anchored to `dev` for this revision.

---

## 0. Provenance & upstream pin

Verification provenance differs by which codebase a fact lives in — the earlier blanket
claim "verified against the working tree" conflated the two and is corrected here.

- **Chatbook-side facts** (call sites, `Chunk_Lib`, `RAG_Search` services, schema, deps):
  verified against the chatbook working tree. Independently re-checked during review;
  the call-site inventory (§6.1.1) and schema facts (§8) hold. A second review pass
  (2026-08-18) re-verified the §6.1.1 table, the §5.3 shim inventory, and the dev-side
  drift claims against `dev` @ `385afa95`, and corrected the errors listed in the
  Status line.
- **`tldw_server`-side facts** (16.5k LOC, strategy roster, `process_text/metadata.py`
  keys, `_sanitize_input`, import graph): these were originally verified against a **local
  checkout on a codex research branch** (`f0f8e5af`, `codex/research-discovery-phase2a-
  planning`, 2026-07-20) — **not `dev`.** For this revision they were re-anchored to `dev`.

**Pin — the manifest and `sync_chunking_engine.py` MUST target this exact ref:**

- repo: `https://github.com/rmusser01/tldw_server.git`
- branch: `dev`
- commit: `385afa951922c8a9dc2002c675bb6cad65e4ac23` (2026-08-13)

**Dev re-anchor result.** Between the codex branch the facts were first checked on and
`dev` @ `385afa95`, the in-scope Chunking core is unchanged for every file this
sub-project's facts rest on (`chunker.py`, `process_text/`, `strategies/__init__.py`,
`splitters/`, `semantic.py`, and the full strategy roster — no adds/deletes/renames).
The only in-scope drift is `strategies/rolling_summarize.py`, which now **fail-closes**
(see §9). Files deferred to #6 (`auto_boundary_assistant.py`, `propositions.py`) drifted
substantially and must be re-verified when #6 is specced. The upstream test suite grew
from 43 to 46 files (§10).

**Wrong-tree hazard.** More than one `tldw_server` checkout can exist locally, and they
diverge — one observed local worktree was missing the entire `process_text/` package and
had a different LOC/test count. The sync script must pin repo + branch + SHA and verify
the synced tree matches that SHA; it must never sync from a bare local path.

---

## 1. Why

A student wants per-chapter notes from a book they ingested, or flashcards per section
of a PDF they attached. Today a Console agent cannot do this: `library_get_media` pages
media content by **blind character cursor** (`DEFAULT_MAX_CHARS = 8_000`,
`MAX_MAX_CHARS = 16_000`, `MAX_RESULT_BYTES = 32 KiB`), so an agent asked for "chapter 7"
walks fixed windows and guesses where chapters begin.

chatbook already owns a chapter chunker (`Chunking/Chunk_Lib.py`,
`_chunk_ebook_by_chapters`) — it is simply not reachable from the agent runtime, and
the surrounding engine has drifted badly from `tldw_server`'s. Users reasonably expect
media to behave the same in both products; it does not.

**Scope note:** that opening agent/student story is delivered by sub-projects **#4 and #5**
(the agent tool surface and the workflow), which are non-goals here (§4). What **this**
sub-project (#1) contributes to it is narrow but real: it makes `ebook_chapters` reachable
on the PDF path (§7.2). #1's own payoff stands on different ground — engine parity for
**ingestion and RAG quality for every user** (§2.1), plus sweeping up three latent bugs
(§7). Judge #1 on that, not on the per-chapter-notes demo.

| | chatbook today | tldw_server |
|---|---|---|
| Core engine | ~2.3k LOC monolithic `Chunk_Lib.py` | 16.5k LOC, 14 strategy modules, splitters, `process_text` pipeline |
| Methods | words, sentences, paragraphs, tokens, semantic, json, xml, ebook_chapters, rolling_summarize | + `structure_aware`, `propositions`, `code`, `code_ast`, `fixed_size`, hierarchical boundaries |
| Templates | stage-based `{base_method, pipeline}`, 13 JSON files, **no editor UI** | v2 flat schema + `classifier` + `version` + `tags`, DB-backed, CRUD API |
| Safety | none specific | `regex_safety` (ReDoS), `security_logger`, offset property tests |
| Test suite | 1 file in `Tests/Chunking/` + ~7 more touching chunking across `Tests/{Local_Ingestion,RAG,Internal_Prompts}` (second-review count; an earlier draft said "5 files") | 46 files (`dev` @ `385afa95`) |
| Auto-selection | none | `auto_planner` + `auto_boundary_assistant` |

This spec covers **sub-project 1 only**: replacing chatbook's chunking engine with
`tldw_server`'s behind a compatibility shim. Two distinct kinds of "compatibility" must
be kept separate, because they have opposite answers:

- **API compatibility — preserved.** No caller's *signature* changes; the shim absorbs
  the difference (§6.2).
- **Output compatibility — deliberately NOT preserved.** Because §6.3 converges the regex
  splitter that today handles `words`/`sentences`/`paragraphs` — "the overwhelming
  majority of all chunking" (§6.1) — the *chunks produced* change for most newly-ingested
  content. That is the point of parity, and it is the single largest behavioral risk in
  the sub-project (§12). This is not an "invisible" swap; it is an invisible-**interface**
  swap with a visible-**output** change. The version stamp (§8) makes stored data honest
  about it; live ingestion changes on day one for every user, with no opt-in.

## 2. Program context

The full ask decomposes into six sub-projects. Each is independently valuable and gets
its own spec → plan → PR cycle. This document specs #1.

1. **Engine parity** ← *this spec*. Vendor the server's `Chunking/` core behind
   chatbook's existing API. Pays off immediately in ingestion and RAG quality for every
   user, not just agent users.
2. **Template v2 parity.** Server's flat `{preprocessing, chunking, postprocessing,
   tags, version: 2}` schema, `ChunkingTemplates` migration, DB↔internal-stages mapper,
   `template_library` port, validation matching the server's validate endpoint.
3. **Classifier / auto-selection.** The `classifier` block (`media_types`,
   `filename_regex`, `title_regex`, `min_score`, `priority`) plus `auto_planner`
   scoring, so a media item gets the right template automatically.
4. **Agent chunking tools.** The original ask: a bounded `library_*`-shaped tool
   surface — structure map, fetch-unit, list/save spec, reuse-stored-chunks, opt-in
   re-chunk-and-persist.
5. **Student workflow.** Per-chapter fan-out ergonomics: sub-agent-per-unit, note
   conventions, flashcard/Q&A output format.
6. **LLM-dependent extras.** `propositions`, `auto_boundary_assistant`,
   `async_chunker`, telemetry depth. Parallel to #3–#5.

Decisions already taken during brainstorming, recorded here so later specs inherit them:

- **Source surface (#4):** ingested Library media only. Attached-but-not-ingested files
  are follow-up tasks (§11).
- **Write posture (#4):** reuse stored chunks when the method matches, **plus** an
  opt-in re-chunk-and-persist mode, **plus** saved chunk specs.
- **Template schema (#2/#3):** full server v2 parity including `classifier`.
- **Flashcards:** *(corrected — the earlier claim "chatbook has no flashcard subsystem…
  no DB table" was false.)* chatbook **does** have a flashcards data layer in
  `DB/ChaChaNotes_DB.py`: tables `decks` *(second-review correction: the deck table is
  named `decks`, not `flashcard_decks` — `flashcard_decks` exists only as a server-API
  client method, `tldw_api/client.py:9247`)*, `flashcards`, `flashcards_fts`,
  `flashcard_templates`, `flashcard_assets`. What it lacks is a **screen route** (none in
  `UI/Navigation/screen_registry.py`); `UI/Study_Modules/flashcards_handler.py` and
  `tldw_api/flashcards_schemas.py` are a client for a server API. So "per-chapter
  flashcards" (#5) has two viable targets — real `flashcards` rows, or *notes containing*
  Q/A markdown via `create_note` — and that is a **#5 decision to make deliberately**, not
  something settled here. It is not in scope for this sub-project either way; the point is
  that the choice must not be made on the false premise that no flashcard tables exist.

## 3. Goals

- chatbook and `tldw_server` produce **the same chunks for the same input and options**
  across every non-LLM method.
- Every existing chunking caller keeps working with no signature change.
- By the end of the program, **one** module performs text splitting (§5.6); #1's
  shims are transitional scaffolding with a demolition date, not permanent
  architecture.
- Chapter/section-aware chunking becomes reachable from **all** ingestion paths,
  including PDF (it is not today — see §7.2).
- Future upstream changes re-sync with one command plus a diff review, rather than a
  re-read of 16.5k lines.
- Existing persisted chunks stay valid; users can see which are stale and re-chunk on
  their own schedule.

## 4. Non-goals

- Template v2 schema, `classifier`, auto-selection — sub-projects #2/#3.
- Any agent-facing tool — sub-project #4.
- Attachment text extraction — §11 follow-ups.
- A template editor UI. chatbook has none today
  (`Widgets/chunking_templates_widget.py` and `chunking_template_editor.py` exist as
  **stale `.pyc` only**; the sources are deleted). Not restored here.
- Forcing a re-chunk of anyone's library.

## 5. Architecture

### 5.1 Vendored tree

The server's `app/core/Chunking/` tree is mirrored under
`tldw_chatbook/Chunking/engine/` — a **curated subset** of upstream paths (upstream's
`__init__.py`, README, SECURITY.md, and the #2/#3/#6-deferred modules are excluded),
kept path-for-path for everything it *does* include, so upstream files stay diffable:

```
tldw_chatbook/Chunking/
├── engine/                  # VENDORED — mechanical import rewrite only
│   ├── VENDOR_MANIFEST.toml # upstream repo + commit SHA + exact file list
│   ├── __init__.py          # chatbook-authored package init (NOT upstream's — see below)
│   ├── base.py  chunker.py  constants.py  exceptions.py  error_policy.py
│   ├── option_utils.py  regex_safety.py  security_logger.py
│   ├── multilingual.py  llm_context.py
│   ├── process_text/{models,options,preparation,dispatch,pipeline,metadata}.py
│   ├── splitters/{regex,blingfire}.py
│   ├── strategies/{words,sentences,paragraphs,tokens,json_xml,ebook_chapters,
│   │               ebook_chapters_patch,structure_aware,code,code_ast,
│   │               fixed_size,semantic,rolling_summarize}.py
│   └── utils/metrics.py
├── _shims/                  # chatbook-authored adapter layer (§5.3)
├── Chunk_Lib.py             # becomes a compatibility shim (§6)
├── chunking_templates.py    # untouched this sub-project (#2 replaces)
├── chunking_interop_library.py  # untouched
├── language_chunkers.py     # retained; superseded by engine/multilingual in #6
├── token_chunker.py         # retained behind the shim
└── templates/               # untouched (#2 replaces)
```

**`engine/__init__.py` must be chatbook-authored, not vendored.** Upstream's
`Chunking/__init__.py` pulls in the #2/#3/#6-deferred modules (`templates`,
`template_initialization`, `auto_planner`, `async_chunker`,
`auto_boundary_assistant`), imports `app.core.config` at line 58 (behind a
try/except that would silently swallow a missing shim if it were vendored), and
defines its *own* back-compat `improved_chunking_process`
(line 75) that would confusingly coexist with the `Chunk_Lib` shim's. The
chatbook-authored `engine/__init__.py` re-exports the phase-1 surface
(`Chunker`, `ChunkerConfig`, the exception classes) and nothing else; it lives in
the manifest's excluded list, not its vendored list.

### 5.2 Manifest-driven sync

`VENDOR_MANIFEST.toml` pins the upstream repo, commit SHA, and the exact list of
vendored paths. `Helper_Scripts/sync_chunking_engine.py` re-copies exactly those paths
and re-applies one rewrite rule:

- `tldw_Server_API.app.core.Chunking.X` → `tldw_chatbook.Chunking.engine.X`
- any other `tldw_Server_API.app.core.*` → `tldw_chatbook.Chunking._shims.*`

**Vendored files are never hand-edited.** Anything chatbook needs differently goes in a
shim or a subclass, or goes upstream first and comes back through a re-sync. Later
sub-projects extend the manifest rather than re-deriving the port. The script must be
idempotent and must fail loudly if a vendored file has local modifications.

> **Amendment (2026-08-20, task-19321).** Reviewed diagnostic-privacy repairs
> (ADR-029) may ship as chatbook-side *scripted* patches: `ENGINE_PATCHES` in
> `sync_chunking_engine.py`, recorded in the manifest's `[patches]` table. The
> canonical vendored state becomes upstream-at-pin + rewrite + patches; the
> sync stays idempotent, still fails loudly on hand-edits (the modification
> check compares against the patched state), and fails loudly when upstream
> drifts under a patch anchor (`_replace_once`). Hand-edits remain forbidden,
> and anything beyond log-record content still goes upstream or into a
> shim/subclass.

### 5.3 Shim inventory

Derived by tracing which **file** needs each import, not by counting grep hits. Most
of the server's outside reach belongs to modules this sub-project defers, so phase 1
needs far fewer shims than the raw import list suggests.

| Server import | Needed by | Phase 1 shim |
|---|---|---|
| `app.core.testing` | `chunker:21`, `base:15,130`, `option_utils:7`, `ebook_chapters:15`, `json_xml:22` | **yes** — `is_truthy` + `is_test_mode`, ~20 lines |
| `app.core.config` | `chunker:2533`, `base:133,469`, `regex_safety:132`, `ebook_chapters:91`, `json_xml:25` (upstream's own `Chunking/__init__.py:58` also imports it, but that file is chatbook-authored here so the reference is moot — §5.1) | **yes** — delegates to `tldw_chatbook.config.get_cli_setting` |
| `app.core.Utils.prompt_loader` | `strategies/rolling_summarize:13`, `strategies/propositions:31` | **yes, only because `rolling_summarize` ships** (§5.4) — maps to `Internal_Prompts/resolver.py`. Note the ID mapping: the server loads `load_prompt("chunking", "Rolling Summarization")` (`rolling_summarize.py:195`); the resolver carries the equivalent prompt as `summarization.rolling_summarize_system` (`summarization_prompts.py:60`) |
| `app.core.Metrics` | `chunker:39` | **no** — `chunker.py:38` already wraps the import in `try/except CHUNKER_NONCRITICAL_EXCEPTIONS` (which includes `ImportError`) with a complete no-op fallback. Providing a shim would add code for behavior the vendored module already has. |
| `app.core.Chat.chat_helpers`, `app.core.LLM_Calls.{adapter_utils,adapter_registry,provider_metadata}`, `app.core.AuthNZ.{llm_provider_overrides,provider_credential_runtime}`, `app.core.Chat.bounded_daemon` | `auto_boundary_assistant:16,19,23,28,29,30,35` | no — deferred to #6 |
| `app.core.http_client`, `app.core.exceptions` | `async_chunker:18,19` | no — deferred to #6 |
| `app.core.DB_Management.{db_path_utils, media_db.api}` | `template_initialization:17,18` | no — deferred to #2 |

**Phase 1 needs three shim modules** (`testing`, `config`, `prompt_loader`). It is not
"two, or three if `rolling_summarize` ships": §5.4 shows `rolling_summarize` is
force-imported at `chunker.py:31` and cannot be excluded, so its `prompt_loader`
dependency is mandatory, not conditional. Everything else arrives with the sub-project
that needs it.

### 5.4 Sub-project boundary

**In scope:** every non-LLM strategy, plus `semantic` and `rolling_summarize`.

What can be excluded is decided by the **import graph**, not by preference:

- `chunker.py:30-35` imports `fixed_size`, `rolling_summarize`, `sentences`,
  `structure_aware`, `tokens`, `words` at module level.
- Any `from .strategies.X import …` executes `strategies/__init__.py`, which
  additionally imports `semantic` and `json_xml`.
- Therefore `rolling_summarize` and `semantic` **cannot be excluded** without editing a
  vendored file, which §5.2 forbids. Their inclusion is forced, not chosen. (It is also
  desirable — chatbook ships both today — but the import graph settles it.)
- `propositions` is **not** eagerly imported, so it defers cleanly to #6 with no stub.
- `semantic.py` makes **no LLM calls**: it uses sklearn TF-IDF + cosine similarity, with
  sklearn imported *inside functions* rather than at module scope, so importing it on a
  base install (which has no sklearn — see §5.5) must not raise. That is an acceptance
  criterion, not an assumption.

**Deferred:** `templates.py`, `template_initialization.py`, `template_library/` (#2);
`auto_planner.py` (#3); `propositions.py`, `utils/proposition_eval.py`,
`async_chunker.py`, `auto_boundary_assistant.py` (#6).

### 5.5 Dependencies

Checked against `pyproject.toml`, not assumed.

- **`tiktoken` is the main new package — and the fallback question is still live.**
  An earlier draft claimed the vendored `strategies/tokens.py` "does not degrade —
  it `raise ImportError("tiktoken not available")` (lines 154/159/323)" and that
  hard-requiring tiktoken makes the degrade-to-what question "disappear." **That
  characterization was wrong.** The `raise` statements at 154/159 sit in
  `TiktokenTokenizer.encode/decode`, and the one at 323 sits *inside* the strategy's
  tokenizer-resolution `try`, immediately caught by `except ImportError` at line 324
  (tokens.py @ `385afa95`). The strategy's resolution is a **three-level chain**:
  tiktoken → transformers → `FallbackTokenizer` (word-based approximation, class at
  line 167). On an install with neither tiktoken nor transformers, `tokens` therefore
  **silently degrades to word approximation** — exactly the behavior Q2's resolution
  declares disallowed. Two consequences:
  1. Hard-requiring `tiktoken` in chatbook's `dependencies` (Q2) still works — with
     it installed the chain stops at level 1 — but the *reason* recorded in Q2 is
     factually wrong and must not be used to justify future decisions.
  2. The no-silent-swap guarantee must be enforced **in the shim**, not assumed from
     the vendored code: the `tokens` path in the shim must detect the
     word-approximation fallback (e.g. by checking the strategy's resolved tokenizer
     type before returning chunks) and raise a clear "install tiktoken" error instead.
     If a future re-sync or a broken install reintroduces the fallback, the shim is
     the only place that catches it.
  (`semantic.py` also prefers tiktoken inside its tokenizer init at line 416, with
  its own transformers fallback.)
  (Note: the earlier draft's claim that "chatbook's current `tokens` path … already
  fails on a base install today" was **also false** — `TokenBasedChunker.tokenizer`
  at `token_chunker.py:119-127` catches the transformers failure and falls back to
  `FallbackTokenizer`/word approximation, logging "Falling back to word-based
  approximation." Both engines degrade today; neither raises. The port is therefore
  *more* behavior-compatible on this axis than the draft assumed.)
- **Two more optional packages the engine reaches, previously unlisted here:**
  `blingfire` (`splitters/blingfire.py`, an *optional* sentence splitter — the default is
  the pure-Python `regex` splitter via `get_sentence_splitter("regex")`, and a missing
  `blingfire` falls back to regex, so this is low-risk but in phase-1 scope) and `spacy`
  (used **only** by `strategies/propositions.py`, which is deferred to #6 — so `spacy` is
  out of phase-1 scope, but note it for #6). Neither is a core chatbook dependency.
- **`defusedxml` is already declared** (`pyproject.toml:105,169,321,370`) — in extras,
  not core. **Consequence, previously unstated:** on a base install without defusedxml,
  the vendored engine's `xml` method **hard-fails** — `strategies/json_xml.py` raises
  `InvalidInputError("Secure XML parsing requires the 'defusedxml' package.")` (lines
  780/847 @ `385afa95`); there is no stdlib fallback. The legacy path has no such
  requirement (`Chunk_Lib.py:14` uses bare `xml.etree.ElementTree`). Since §7.1/§14
  make `chunk_xml` importable again, this is reachable, not hypothetical. **Resolved
  (Q9): `defusedxml` is promoted to core `dependencies`** — the `xml` method always
  works; the AC's absent-list in §14 stays as written.
- **`sklearn`, `nltk`, `transformers`, `numpy`, `langdetect` are NOT core dependencies.**
  chatbook's `dependencies = [...]` block contains none of them; they live in optional
  extras (`embeddings_rag`, etc.). A base install therefore has none of them, and the
  vendored engine must import and run on that install — degrading `semantic`, token
  counting, and language detection rather than raising.
- CJK/Thai tokenizers (`jieba`, `fugashi`, `konlpy`, `pythainlp`) stay optional and
  degrade, matching how the server's `multilingual.py` already guards them.
- `loguru` is shared and already core.

### 5.6 End-state surface (added on maintainer review)

Convergence (§6.3) is about the *engine*; this section is about the *import surface*,
so the compat shims are visibly transitional scaffolding with a demolition date
rather than permanent architecture. End state of the full six-sub-project program:

| Module | Fate | When |
|---|---|---|
| `Chunking/engine/` + `_shims/` | **The one engine.** Permanent public API surface: `Chunker`, `ChunkerConfig`, engine exceptions. | #1 |
| `Chunking/Chunk_Lib.py` | Compat shim — **transitional**. Imports re-point to `engine` as callers migrate; deleted once no importer remains (Q4 tracks this; the §6.1 inventory + tests are the checklist). | #1 → delete by #6 |
| `RAG_Search/chunking_service.py` | Thin wrapper — **transitional**. Keeps its name/import path in #1 (§6.3) only so its importers don't churn in the same PR; becomes a pure re-export, then a deprecation alias. | #1 → alias in #4 |
| `RAG_Search/enhanced_chunking_service.py` | Retired or reduced to a thin adapter over `structure_aware` (Q5). The vendored engine is the only structure-aware chunker afterward. | #1, own PR |
| `Media/local_media_reading_service._chunk_text` | Routes through the engine or is explicitly descoped (Q6). No independent splitter survives the program. | #1 or descope |
| `Chunking/token_chunker.py` | Retained behind the shim in #1 (an exported symbol, §6.1 Group A); superseded by the engine's `tokens` strategy — delete once `Chunk_Lib`'s `token_chunker` consumers migrate. | #1 → delete by #6 |
| `Chunking/language_chunkers.py` | Retained in #1 (`Chunk_Lib.py:778` calls it); superseded by `engine/multilingual.py`. | #1 → delete in #6 |
| `Chunking/chunking_templates.py` + `templates/` (13 JSON) | Untouched in #1; replaced wholesale by the server v2 schema. | #2 |
| `Chunking/chunking_interop_library.py` | Untouched in #1 — it is a **template-CRUD service layer** (DB-backed template storage used by RAG Admin and the media-details widget; imports no `Chunk_Lib` symbol), not a chunking entry point. #2 re-evaluates it alongside the template migration. | #2 |
| `Chunking/__init__.py` | Re-exports stay stable through #1; collapse to the engine surface as the shims delete. | #6 |

**The invariant this table exists to enforce:** after the program completes, exactly
one module performs text splitting (`Chunking/engine/`), and every other name in the
table is either deleted or a pure re-export with no logic. Any sub-project that adds
chunking logic outside `engine/` or `_shims/` violates the program's reason to exist.
In #1 specifically, the deletions are `_chunk_text_in_process` and the legacy
`Chunk_Lib` implementation; the rest of the surface narrows later — pretending #1
must delete every facade at once would make the PR unreviewable, which is why the
staged demolition is written down rather than left implicit.

## 6. Compatibility contract

The engine swap must be **interface**-invisible to callers (signatures unchanged); its
**output** is not invisible (§1, §6.3, §12). The most de-risking fact found during design
is specifically about the *shape* of the output, not its values:

> **`Chunker.process_text()`'s output is a superset of `improved_chunking_process()`'s —
> at the level of metadata *keys*.**

Read this precisely: every metadata key chatbook reads is emitted by the server pipeline,
so no caller breaks on a *missing* key. It does **not** mean the values match — chunk
*content* and metadata *values* legitimately differ (input sanitization §9, size ceilings
§9, boundary placement §12, language detection §7.4). "Superset of keys" is what makes the
shim safe; it is not a claim of value-level parity, which §8's version stamp exists
precisely because it does not hold for stored data.

Every metadata key chatbook's callers read is emitted by the server pipeline
(`process_text/metadata.py`): `chunk_index` (1-based), `total_chunks`, `chunk_method`,
`max_size_setting`, `overlap_setting`, `language`, `adaptive_chunking_used`,
`relative_position`, `initial_document_json_metadata`,
`initial_document_header_text`, `chunk_content_hash` — plus extras (`start_char` /
`end_char` offsets, `origin`, `max_size`, `overlap`). These two codebases are the same
lineage; chatbook's `Chunk_Lib` is an older fork.

**Shape mismatch at the persistence seam (second-review finding).** The
superset-of-keys argument covers keys *inside* `metadata`. It does **not** cover the
chunk-dict shape itself, and at one load-bearing seam the two differ:

- The engine's `process_text` returns `{"text": …, "metadata": {…}}` per chunk
  (`process_text/metadata.py:71`), with `start_char`/`end_char` **inside** the
  metadata dict.
- `MediaDatabase._persist_chunks` (`DB/Client_Media_DB_v2.py:3741-3765`) reads
  **top-level** keys — `ch["text"]`, `ch.get("start_char")`, `ch.get("end_char")`,
  `ch.get("chunk_type")`, `ch.get("metadata")` — writing them into the
  `UnvectorizedMediaChunks` columns.
- Today's regex path (`RAG_Search/chunking_service.py:259-324`) emits **flat** chunks
  with top-level `start_char`/`end_char`/`word_count`, so those columns are populated
  today.

If the §6.3.2 re-export returns engine-shaped chunks unchanged, every new ingest
**silently writes `NULL` into `start_char`/`end_char`** — and those columns are read
back by the chunk-navigation builder (`Media/local_media_reading_service.py:4689`,
`SELECT … start_char, end_char … FROM UnvectorizedMediaChunks`). "No caller breaks on
a missing key" is true only at the metadata level; at this seam keys go silently NULL,
which is worse than breaking. Similarly, `word_count` is top-level in today's flat
chunks and read by `chunk_preview_modal.py:189,199` (`chunk.get("word_count", 0)`);
engine output has no such key, so the preview's word counts silently zero unless the
shim adds it.

**Required decision (§6.3.2):** the `RAG_Search.chunking_service` re-export must
specify its output shape. Recommendation: **preserve today's flat contract** —
top-level `text`, `start_char`, `end_char`, `word_count` (derived, as today) — and
put the engine's rich metadata under `metadata`. The DB seam, the navigation reader,
and the preview modal then keep working unchanged. Characterization tests (§10.3)
must include the DB round-trip (`add_media_with_keywords` → read back columns) and
the navigation-builder read, not just the chunker calls.

### 6.1 chatbook has SEVEN chunking implementations, not one

The engine swap cannot be scoped as "replace `Chunk_Lib`". A tree-wide sweep found
**three** independent text-splitting implementations that do their own splitting outside
`Chunker` — `_chunk_text_in_process`, `EnhancedChunkingService`'s structural/hierarchical
path, and `local_media_reading_service._chunk_text` — and they sit on the
**most-travelled** paths. (An earlier draft listed a fourth, `audio_processing._chunk_text`,
here; it is *not* independent — it delegates to `ChunkingService.chunk_text`, so it belongs
with the Group C wrappers below. And "never touch `Chunker` at all" is imprecise for the
first two: both live on or under `ChunkingService`, which *does* construct `Chunker` for
the methods the regex path does not handle.) Any plan that only shims `Chunk_Lib` will
leave the majority of real chunking on the old code while appearing to succeed.

**Group A — engine-backed (a `Chunk_Lib` shim covers these):**

| Implementation | Role |
|---|---|
| `Chunking/Chunk_Lib.py` | `Chunker`, `improved_chunking_process`, module-level constants |
| `Chunking/token_chunker.py:108` `TokenBasedChunker` | backs `Chunker.token_chunker`; also exported from `Chunking/__init__.py` |
| `Chunking/language_chunkers.py` `LanguageChunkerFactory` | CJK splitting, called at `Chunk_Lib.py:778`; server equivalent is `multilingual.py` |

> **Constants are part of the import surface.** `Chunking/__init__.py:15` imports
> `DEFAULT_CHUNK_OPTIONS` from `Chunk_Lib`, and `Chunking/Chunk_Lib` module constants
> have importers outside the §6.1.1 table: `MAX_CHUNK_SIZE_PARAGRAPHS`
> (`Tests/RAG/test_config_profiles.py:12`), `DEFAULT_CHUNK_OPTIONS`/`Chunker`
> option-resolution behavior (`Tests/Internal_Prompts/test_summarization_migration.py`),
> plus `MAX_CHUNK_SIZE_WORDS/SENTENCES/TOKENS`, `MAX_DOCUMENT_SIZE_MB/BYTES`, and
> `ensure_nltk_data` (`Chunk_Lib.py:221-226,240,355`). The shim (§6.2) must re-export
> all of them; §14's "every call site works unchanged" AC does not cover these, since
> they are import-time consumers, not call sites. `Chunking/templates/example_usage.py`
> also calls `improved_chunking_process` — dead example code, listed here only so the
> inventory's completeness claim stays honest.

**Group B — independent implementations that bypass the engine (a `Chunk_Lib` shim does NOT cover these):**

| Implementation | What it is | Why it matters |
|---|---|---|
| `RAG_Search/chunking_service.py:259` `_chunk_text_in_process` | a **regex splitter** with its own sentence/paragraph patterns and its own validation | `ChunkingService.chunk_text:91` routes `words` / `sentences` / `paragraphs` here — **the overwhelming majority of all chunking**. Its docstring ("without initializing the full user-template stack") indicates it exists to dodge `Chunker` construction cost. |
| `RAG_Search/enhanced_chunking_service.py` (~770 lines) | `EnhancedChunkingService(ChunkingService)` with `DocumentStructureParser`, `_hierarchical_chunking`, `_structural_chunking`, `_sub_chunk_element`, `chunk_with_parent_retrieval`, `StructuredChunk`, `ChunkType` | A **second, complete structure-aware chunking system** that duplicates what the server's `structure_aware` strategy and `hierarchical_template.boundaries` do. Reachable from `Widgets/chunk_preview_modal.py:111` (see §7.4). |
| `Media/local_media_reading_service.py:1556` `_chunk_text` | naive fixed-size character slicer with overlap; no word or sentence awareness | splits mid-word by construction |

**Group C — orchestration wrappers (no splitting of their own; they inherit whatever B does):**
`simplified/rag_service.py:2961` `_chunk_document` → `ChunkingService.chunk_text`;
`simplified/indexing_helpers.py:33` `chunk_documents_batch` → `_chunk_document`;
`simplified/enhanced_indexing_helpers.py:38` `chunk_documents_with_parents`;
`RAG_Search/parallel_processor.py:436` `chunk_documents_batch`;
`Local_Ingestion/audio_processing.py:1064` `_chunk_text` → `ChunkingService.chunk_text`
(default `method="sentences"`) — transcript splitting that inherits Group B's behavior, so
converging the regex splitter (§6.3) fixes it automatically; no separate work needed.

**Out of scope — not document chunking**, listed so a later reader does not "fix" them:
`TTS/text_processing.py:186` `TextChunker` and the TTS backends (`higgs`, `kokoro`,
`chatterbox`) plus `Subscriptions/briefing_audio.py:391` are speech-synthesis
segmentation; `Notes/note_folder_repository.py:2061` `_chunks` and
`Sync_Interop/local_first_sync_service.py:380` `_chunk_push_items` are list batching;
`UI/Screens/chat_screen.py:7899` `_chunks` is stream handling;
`local_media_reading_service.py:4746` `_chunk_navigation_title` is string truncation.

### 6.1.1 Call sites

| Caller | Entry point |
|---|---|
| `Book_Ingestion_Lib.py:1036, 2290, 2486` | `Chunk_Lib.improved_chunking_process` (kwargs form) |
| `Summarization_General_Lib.py:630, 693` | `Chunk_Lib.improved_chunking_process` (positional) |
| `Book_Ingestion_Lib.py:1793` | `RAG_Search.chunking_service.improved_chunking_process` → regex path |
| `PDF_Processing_Lib.py:753` | same → regex path |
| `Image_Processing_Lib.py:475` | same → regex path |
| `Local_Ingestion/local_file_ingestion.py:1533` → `_chunk_text_for_ingest:576` | same → regex path |
| `RAG_Admin/local_rag_admin_service.py:287` | `Chunk_Lib.Chunker(options=…, template_manager=object())`, `.chunk_text(text, method=…, use_template=False)` |
| `RAG_Search/chunking_service.py:103` | `Chunk_Lib.Chunker(options)` — only for methods the regex path does *not* handle |
| `Widgets/chunk_preview_modal.py:111,134` | `EnhancedChunkingService()` **and** `Chunker()` |
| `RAG_Search/simplified/enhanced_indexing_helpers.py:66` | `EnhancedChunkingService()` → `chunk_with_parent_retrieval(...)` — **RAG indexing hot path** |
| `RAG_Search/simplified/enhanced_rag_service.py:51` | `EnhancedChunkingService()` → `chunk_with_parent_retrieval(...)` — **RAG indexing hot path** |
| `XML_Ingestion.py:12` | `Chunk_Lib.chunk_xml` — **broken import** (§7.1) |

> **Inventory correction.** An earlier draft named only the preview modal as an
> `EnhancedChunkingService` consumer. It has **three** consumers; the two RAG-indexing ones
> above use `chunk_with_parent_retrieval`, which returns parent/child structure
> (`parent_chunks`), a materially harder migration than flat chunks. The
> "every call site works unchanged, proven by characterization tests" AC (§14) and the §6.3
> retire-or-adapt decision (§13.5) must cover these two, not just the modal.

### 6.2 Signature mismatch to absorb

Legacy chatbook and vendored server signatures differ; the shim absorbs the difference.

```
# legacy (must keep working)
Chunker(options: Optional[dict] = None, tokenizer_name_or_path='gpt2', template=None, template_manager=None)  # options is optional, not a required dict
  .chunk_text(text, method=None, llm_call_function=None, llm_api_config=None, use_template=None)
improved_chunking_process(text, chunk_options_dict=None, tokenizer_name_or_path='gpt2',
                          template=None, template_manager=None,
                          llm_call_function_for_chunker=None, llm_api_config_for_chunker=None)

# vendored engine
Chunker(config: ChunkerConfig | None, …)
  .chunk_text(text, method=None, max_size=None, overlap=None, language=None, **options)
  .process_text(text, options=None, *, tokenizer_name_or_path=None, llm_call_func=None, llm_config=None)
```

`Chunk_Lib.py` is rewritten as a thin shim that:

- exports `improved_chunking_process(...)` delegating to
  `Chunker(config).process_text(text, options, tokenizer_name_or_path=…)`, translating
  the legacy kwargs (`llm_call_function_for_chunker` → `llm_call_func`, etc.);
- exports a `Chunker` adapter class accepting the legacy kwargs, translating the
  `options` dict into a `ChunkerConfig`, and mapping legacy `chunk_text(...)` arguments
  (including `use_template`) onto the engine's;
- re-exports `chunk_for_embedding`, `process_document_with_metadata`, `load_document`,
  a restored module-level `chunk_xml` (the capability itself is not missing — it lives as
  `Chunker._chunk_xml`, `Chunk_Lib.py:1592`; only the module-level *name* is gone, §7.1),
  the module-level constants `DEFAULT_CHUNK_OPTIONS`, `MAX_CHUNK_SIZE_WORDS`,
  `MAX_CHUNK_SIZE_SENTENCES`, `MAX_CHUNK_SIZE_PARAGRAPHS`, `MAX_CHUNK_SIZE_TOKENS`,
  `MAX_DOCUMENT_SIZE_MB`, `MAX_DOCUMENT_SIZE_BYTES` and `ensure_nltk_data`
  (import-time consumers exist outside §6.1.1 — see the Group A note in §6.1), and the
  exception classes `ChunkingError`, `InvalidChunkingMethodError`,
  `InvalidInputError`, `LanguageDetectionError`, `MemoryLimitError`.
  **Exception-name mismatch to reconcile (see §9):** two of those legacy names do **not**
  exist in the engine. The vendored `Chunking/exceptions.py` (on `dev` @ `385afa95`)
  defines `ChunkingError`, `InvalidInputError`, `InvalidChunkingMethodError`,
  `TokenizerError`, `TemplateError`, `LanguageNotSupportedError`, `ChunkSizeError`,
  `ProcessingError`, `ConfigurationError`, `CacheError` — but **no** `LanguageDetectionError`
  and **no** `MemoryLimitError`. The shim must define those two legacy names as aliases
  (mapping to `LanguageNotSupportedError` and to whatever `_enforce_text_size` raises,
  respectively) so existing `except` blocks keep matching.

The legacy `Chunk_Lib` implementation is deleted; only the shim remains at that path.

### 6.3 Converging the implementations

Parity requires every Group B implementation to route through the engine. Ordered by
how much content flows through it:

1. **`_chunk_text_in_process` (highest traffic).** Delete it; `ChunkingService.chunk_text`
   delegates to the engine for *all* methods. Its bespoke validation messages
   (`"max_words must be positive"`, overlap-versus-size rules) are part of the observed
   contract and must be preserved or deliberately changed — several tests and callers
   depend on the raised `ChunkingError`. Because it exists to avoid `Chunker`
   construction cost, the engine's construction cost must be measured before deleting it
   (§12) — the engine's `LRUCache` and a module-level shared `Chunker` are the mitigation
   if construction proves expensive.
2. **`RAG_Search.chunking_service.improved_chunking_process`.** Becomes a re-export of
   the `Chunk_Lib` shim, dropping its hardcoded method whitelist (§7.2) and its bespoke
   character-position recomputation (the engine emits real `start_char`/`end_char`).
   **Output shape (second-review finding, see §6):** the re-export must preserve the
   **flat** per-chunk contract today's callers and the DB consume — top-level `text`,
   `start_char`, `end_char`, `word_count`, with the engine's metadata dict under
   `metadata`. The engine's nested `{"text", "metadata"}` shape puts offsets inside
   `metadata`; passing it through unchanged makes `MediaDatabase._persist_chunks`
   (`Client_Media_DB_v2.py:3741-3765`) write `NULL` into the `start_char`/`end_char`
   columns and zeroes the preview modal's word counts. Flattening happens here, once,
   so every Group B/C consumer keeps its shape. Note the same module defines its own
   module-local `ChunkingError`/`InvalidChunkingMethodError` classes
   (`chunking_service.py:19-28`), distinct from `Chunk_Lib`'s; the re-export must alias
   those names to the engine's exceptions so existing `except` clauses on either name
   keep matching — the characterization tests must pin *which* class is raised.
3. **`EnhancedChunkingService`.** Its `_structural_chunking` / `_hierarchical_chunking`
   are chatbook's home-grown version of the server's `structure_aware` strategy and
   hierarchical boundaries. **Decision required (§13.5):** retire it in favor of the
   engine's equivalents, or keep it as a thin adapter over them. Retiring is the only
   option consistent with parity, but `chunk_with_parent_retrieval` (and the
   `StructuredChunk`/`parent_chunks` shape it returns) has **three** live consumers —
   `chunk_preview_modal.py:111`, and the two RAG-indexing consumers
   `enhanced_indexing_helpers.py:66` and `enhanced_rag_service.py:51` (§6.1.1). The two
   RAG consumers depend on parent/child retrieval, which the engine's `structure_aware`
   does not emit in the same shape, so this needs a real migration path and **warrants its
   own PR** — it is the (b)-split's hardest piece, not a footnote.
4. **`Media/local_media_reading_service._chunk_text`.** Lower traffic and structurally
   simpler (a raw-character slicer). Either route it through the engine or **explicitly
   declare it out of scope in this spec** — what is not acceptable is leaving it
   undocumented, as the first draft did. (`audio_processing._chunk_text` is **not** in this
   list: it already delegates to `ChunkingService.chunk_text`, so steps 1–2 above converge it
   for free — it is a Group C wrapper, §6.1.)

`ChunkingService` keeps its name and import path so its importers do not change; only
its body moves to the engine. It is a transitional facade per §5.6 — slated to become a
pure re-export and then a deprecation alias, on the pacing Q4 rules.

## 7. Bugs this sweeps up

### 7.1 `XML_Ingestion` is dead code

`Local_Ingestion/XML_Ingestion.py:12` imports a name that does not exist:

```
ImportError: cannot import name 'chunk_xml' from 'tldw_chatbook.Chunking.Chunk_Lib'
```

`Chunk_Lib.py:1976` is a leftover comment (`# chunk_xml, extract_xml_structure (done)`)
where the module-level function used to live. Nothing imports `XML_Ingestion`, so the
module is dead rather than crashing users. **Note the XML chunking *capability* is not
missing** — it lives as the private `Chunker._chunk_xml` (`Chunk_Lib.py:1592`, a live
dispatch target at :719); only the module-level `chunk_xml` *name* the import wants is
gone. Restoring a module-level `chunk_xml` in the shim (wrapping the engine's XML path)
fixes the import incidentally. **Reachability of XML ingestion is not restored here** —
that is a separate question about the ingestion UI.

### 7.2 PDF ingestion cannot chunk by chapter

Verified by execution against the working tree:

```
RESTRICTED RAISED: InvalidChunkingMethodError Invalid chunking method: ebook_chapters.
                   Valid methods are: ['words','sentences','paragraphs','tokens','semantic']
FULL OK chunks= 2 ['Chapter 1', 'Chapter 2']
```

`RAG_Search/chunking_service.improved_chunking_process` hardcodes
`['words','sentences','paragraphs','tokens','semantic']` and raises for anything else —
and it is the entry point used by PDF ingestion, image ingestion, and the
markdown/document path of book ingestion. §6.3 removes the whitelist, which makes
chapter/section chunking reachable for PDFs. This is a direct prerequisite for the
student workflow that motivated the program.

### 7.3 The chunk preview does not show what ingestion produces

`Widgets/chunk_preview_modal.py` builds its preview from **`EnhancedChunkingService`**
(line 111) and a bare **`Chunker()`** (line 134), while PDF, image, document and
`local_file_ingestion` paths all chunk via the **regex splitter** in
`RAG_Search/chunking_service.py:259`. The preview a user inspects before ingesting is
therefore produced by different code than the chunks that get stored. Converging the
implementations (§6.3) fixes this as a side effect; it is called out here so the fix is
verified deliberately rather than assumed.

### 7.4 Unreliable language detection (noted, not fixed)

During verification, a short English fixture was detected as Swedish (`language: sv`).
This is pre-existing behavior in the legacy path. The vendored engine has its own
detection; the ported property tests should reveal whether it behaves better. Not a
goal of this sub-project — recorded so it is not mistaken for a regression introduced
here.

## 8. Data & migration

Chunks persisted by the old engine remain valid and untouched. The corpus becomes
mixed, and is made **honest** about it.

- **Stamp.** Add `chunk_engine_version TEXT` to `UnvectorizedMediaChunks` and bump
  `Client_Media_DB_v2._CURRENT_SCHEMA_VERSION` from `5` to `6` with a migration, per
  the project's schema rules. A dedicated column (rather than a key inside the existing
  `metadata` JSON) is chosen so staleness is efficiently queryable. The same value is
  also stamped into the chunk metadata dict the shim returns, so in-memory consumers
  see it without a DB read.
- **Backfill.** The migration leaves existing rows `NULL`, which reads as "pre-parity
  engine". No row rewriting at upgrade time.
- **Report.** RAG Admin gains a read-only "chunked by an older engine (N items)"
  indicator.
- **Action — DEFERRED to sub-project #2 (Q3 resolved).** The user-triggered re-chunk +
  re-index for a selected media item is **not** built in #1. #1 ships stamp + report only;
  the action (the largest piece of net-new UI here) moves to #2, which owns the template
  work it naturally sits beside. Its acceptance criterion is removed from §14.
  **For #2's planning:** confirm which existing service owns re-index
  (`RAG_Search/ingestion_indexing.py` and the RAG Admin indexing controls are the likely
  seam) before designing the action's plumbing, and reuse the existing indexing controls
  rather than adding a parallel path.

Nothing re-chunks automatically. No blocking migration. In #1 nothing re-chunks at all;
once #2 adds the action, embeddings are regenerated only for items the user explicitly
re-chunks.

## 9. Error handling

- Vendored exception types are the source of truth; the shim re-exports legacy names as
  aliases so existing `except ChunkingError:` blocks keep matching.
- `regex_safety` (ReDoS protection) and `security_logger` come across with the engine.
  Because chapter patterns are user-supplied (`custom_chapter_pattern`) and will later
  be agent-supplied (#4), this protection is load-bearing, not incidental.
- **No `Metrics` shim is written.** `chunker.py:38` already guards the import with
  `try/except CHUNKER_NONCRITICAL_EXCEPTIONS` (which contains `ImportError`) and
  installs complete no-op fallbacks. Adding a shim would duplicate behavior the
  vendored module already has.
- **The engine mutates input text.** `_sanitize_input` (`chunker.py:1355`) replaces
  null bytes with spaces and applies NFC normalization when doing so preserves string
  length. The legacy path does neither, so text persisted to the media DB can differ
  after the port. This is a deliberate, stated behavior change — not a silent one — and
  belongs in the release notes.
- `MemoryLimitError` / document-size ceilings: the legacy path enforces
  `MAX_DOCUMENT_SIZE_MB`; the engine enforces its own via `_enforce_text_size`.
  **The defaults already match** — legacy `MAX_DOCUMENT_SIZE_MB = 100`
  (`Chunk_Lib.py:225`) and engine `ChunkerConfig.max_text_size = 100_000_000`
  (`base.py:389`) are both 100 MB — so divergence is only possible through explicit
  configuration, and the shim's job is to map legacy config onto `max_text_size`
  rather than reconcile differing defaults. One behavioral nuance to carry into the
  characterization tests: the legacy path raises its own `MemoryLimitError` class;
  `_enforce_text_size` raises `InvalidInputError` (`chunker.py:337`), a *different*
  class. The shim must reconcile these so a document accepted before is not rejected
  after — or, if the ceilings genuinely differ under some configuration, the change
  is stated in the release notes.
- LLM-dependent methods with no configured provider must fail with a clear,
  actionable error rather than a shim `AttributeError`.
- **`rolling_summarize` now fail-closes (dev drift — see §0).** On `dev` @ `385afa95`,
  `strategies/rolling_summarize.py` was changed so `_call_llm` returns `str` (no longer
  `Optional[str]`) and **raises `ProcessingError`** (`Chunking/exceptions.py:92`) on LLM
  failure instead of silently returning `None`. This aligns with the bullet above, but the
  shim must let `ProcessingError` surface as an actionable error and not swallow or
  mis-map it, and the ported suite gains `test_rolling_summarize_fail_closed.py` (§10).
  Any legacy caller that relied on a `None`/empty return from a failed rolling-summarize
  will now see a raise.
- Legacy names `LanguageDetectionError` and `MemoryLimitError` have **no engine equivalent
  by that name** (§6.2). The vendored `Chunking/exceptions.py` has `LanguageNotSupportedError`
  (not `LanguageDetectionError`) and enforces size via `_enforce_text_size` (no
  `MemoryLimitError` class). The shim defines both legacy names as aliases so existing
  `except` blocks keep matching — this is the concrete content of the first bullet's
  "re-exports legacy names as aliases." **Known over-breadth, accepted deliberately:**
  aliasing `MemoryLimitError` to `InvalidInputError` means a legacy
  `except MemoryLimitError:` block will also catch *non-size* `InvalidInputError`s
  (e.g. non-string input at `chunker.py:340`); aliasing `LanguageDetectionError` to
  `LanguageNotSupportedError` is exact. If the over-broad catch proves problematic in
  the characterization tests, the alternative is a shim-defined subclass raised only
  on the size path — decide at implementation, not spec, time.

## 10. Testing — how parity is proven

Claiming parity requires evidence, not a passing smoke test.

1. **Ported upstream suite.** Bring the server's `tests/Chunking/` suite across via the
   same import rewrite, into `Tests/Chunking/`. On `dev` @ `385afa95` this suite is **46
   `test_*.py` files** (the 43 on the codex branch the spec was first written against, plus
   three added on dev: `test_chunking_runtime_lifecycle.py`,
   `test_propositions_runtime_snapshot.py`, `test_rolling_summarize_fail_closed.py` — the
   last is **in scope** and must be ported; the propositions one lands with #6). Files
   depending on server-only fixtures (FastAPI `TestClient`, DB pools) are excluded and land
   with their own sub-projects — notably `test_chunking_endpoint.py`,
   `test_chunking_template_endpoint_errors.py`,
   `test_chunking_templates_endpoint_sanitization.py`, and the `test_async_*` files.
   The offset/property tests (`test_chunking_offsets_property.py`,
   `test_hierarchical_rewrite_offsets.py`, `test_chunking_overlap_properties.py`) are
   the highest-value ones and are **required** — port the **dev** versions
   (`test_chunking_overlap_properties.py` gained +74 lines on dev).
2. **Golden parity fixtures.** Generate expected outputs from the server engine once
   over a fixed corpus (prose, markdown with ATX headings, an ebook with chapter
   headings, JSON, XML, source code, a CJK sample), commit them as JSON, and assert
   chatbook reproduces them exactly. Re-run at every `sync_chunking_engine.py`
   execution.

   **Critical caveat:** `_sanitize_input` relaxes itself under test — it checks
   `PYTEST_CURRENT_TEST` and `is_test_mode()` and skips null-byte replacement when
   either is set. Fixtures generated or asserted under pytest therefore exercise a
   *different code path than production*, and a green suite would not prove production
   parity. Fixture generation and at least one verification run must execute with test
   mode explicitly disabled.

   **This caveat also undercuts item 1.** The "required" offset/overlap property tests run
   under pytest, i.e. in the *relaxed* sanitization path — so on their own they prove
   parity only in that path, which this caveat says is not production. Close the loop:
   run at least the offset/property suite **also** with test mode explicitly disabled, or
   state plainly that the golden fixtures (this item) are the *sole* production-path
   parity evidence and the property tests are structural-only. As written, AC "ported
   offset and overlap property tests pass" does not by itself establish production parity.
3. **Call-site characterization tests.** For every call site in §6.1.1, capture current
   output shape *before* the swap and assert it still holds *after*. These are what
   actually protect ingestion. The four call sites that route through the regex splitter
   (`Book_Ingestion_Lib:1793`, `PDF_Processing_Lib:753`, `Image_Processing_Lib:475`,
   `local_file_ingestion:1533`) are the highest-risk of the set, because their output
   changes the most. The set must also cover the two consumers the §6.1.1 table does
   not name: the **DB round-trip** (`add_media_with_keywords` → read back
   `start_char`/`end_char`/`chunk_index` columns from `UnvectorizedMediaChunks`) and the
   **navigation-builder read** (`local_media_reading_service.py:4689`), both of which
   consume top-level chunk keys that the engine's nested shape would silently NULL
   (§6).
4. **Thread-safety.** Port `test_thread_safety.py`. chatbook runs chunking on Textual
   workers, and the engine carries a module-level `LRUCache` plus `_thread_local` state.
5. **Regression test for §7.2.** `ebook_chapters` through the
   `RAG_Search.chunking_service` entry point must return chapter chunks, not raise.
6. **Import test for §7.1.** `import tldw_chatbook.Local_Ingestion.XML_Ingestion`
   must succeed.
7. **Preview/ingest agreement (§7.3).** The preview modal and the ingestion path must
   produce identical chunks for the same input and options. The preview's word-count
   column (`chunk_preview_modal.py:189,199`, `chunk.get("word_count", 0)`) must remain
   non-zero — it reads a top-level key the engine's nested shape does not emit (§6).
8. **Import-weight guard.** `Tests/Performance/test_app_import_weight.py` asserts that
   `import tldw_chatbook.app` loads no nltk/scipy/sklearn/pandas, and the engine will
   load at app boot via exactly the chain that test documents
   (`RAG_Admin/local_rag_admin_service.py` → `chunking_interop_library` →
   `Chunking/__init__` → shim → engine). Verified at the pinned SHA: the engine's
   module scope is clean — every heavy import (sklearn, nltk, tiktoken, transformers,
   defusedxml, blingfire, spacy) sits inside a function, guard, or try/except — so the
   guard should keep passing. But a future re-sync that introduces a module-scope heavy
   import would break **app boot**, not just chunking; this existing test is the
   tripwire, and it must stay green in every PR of this sub-project.

Run the full suite before the PR, per the project's testing rules.

## 11. Follow-up tasks to file

Not part of this sub-project; filed so they are not lost.

- **Attachment text extraction.** `Utils/file_handlers.py:313` `PDFFileHandler.process()`
  returns a literal placeholder (`"[PDF File: …] To process this PDF file, please use
  the Media Ingestion tab."`) instead of text. Same for `DocumentFileHandler`
  (`.doc/.docx/.rtf/.odt`) and `EbookFileHandler` (`.epub/.mobi/.azw/.azw3/.fb2`). An
  attached PDF therefore puts **zero** document text in front of the model. The
  extraction machinery already exists (`Local_Ingestion/PDF_Processing_Lib.process_pdf`,
  `Book_Ingestion_Lib`) and is simply not wired to the attachment path.
- **Attached file as a chunking source.** Once extraction works, let the sub-project #4
  tool address an attached (non-ingested) file, including the staging/caching decision
  for where extracted text lives.
- **XML ingestion reachability.** §7.1 makes the module importable; whether XML
  ingestion should be reachable from the ingestion UI is a separate product question.

Task IDs are deliberately **not** assigned in this draft. This repo has a documented
history of backlog ID collisions across worktrees and branches; IDs get chosen against
`origin/dev` at filing time, after this spec is reviewed.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Boundary drift changes RAG retrieval for existing users | Version stamp + report (§8) make the mixed corpus visible in #1; stored chunks are untouched, and the opt-in re-chunk remediation lands in #2 (Q3). Nothing auto-re-chunks |
| Vendored code looks foreign next to chatbook idioms | Accepted deliberately — it is the price of a diffable, re-syncable port |
| **Converging the regex splitter changes chunking for most existing content** | The single largest behavioral risk in the sub-project. Stored chunks stay valid and the version stamp (§8) marks them; new ingests use the new engine immediately (no opt-in — the re-chunk of *old* data is the #2 remediation); characterization tests (§10.3) contain it for callers |
| Deleting `_chunk_text_in_process` reintroduces the `Chunker` construction cost it was written to avoid | Measure before deleting; shared module-level `Chunker` + the engine's `LRUCache` are the mitigation (§6.3) |
| `EnhancedChunkingService` has **three** consumers, two on the RAG-indexing hot path (`enhanced_indexing_helpers.py:66`, `enhanced_rag_service.py:51`) using `chunk_with_parent_retrieval` | Resolve §13.5 before implementation; the parent-retrieval consumers make this its own PR. Enumerated in §6.1.1 |
| New dep `tiktoken` — the vendored `tokens.py` strategy **silently falls back to word approximation** (tiktoken → transformers → `FallbackTokenizer`, a caught chain), and §5.2 forbids editing it | **Resolved (Q2, corrected): hard-require `tiktoken` AND enforce it in the shim** — the shim's `tokens` path must detect the word-approximation fallback and raise a clear error; declaration alone does not prevent the silent swap. `defusedxml` already declared (see Q9); `blingfire` optional (regex default); sklearn/nltk/transformers/spacy are extras, so the engine must run without them (§5.5) |
| **Engine's nested chunk shape silently NULLs DB offset columns** — `_persist_chunks` reads top-level `start_char`/`end_char`; the engine puts them inside `metadata` (§6) | The `chunking_service` re-export flattens to today's flat contract (§6.3.2); DB round-trip + navigation-reader characterization tests (§10.3) catch any regression |
| Base install (no sklearn/nltk/transformers) cannot import the engine | `semantic.py` imports sklearn inside functions; verified by an explicit AC |
| Golden fixtures prove nothing because sanitization relaxes under pytest | Generate and verify with test mode explicitly disabled (§10.2) |
| Upstream and vendored copies drift again over time | `VENDOR_MANIFEST.toml` pins repo + branch + SHA (`dev` @ `385afa95`, §0); golden fixtures fail loudly when behavior moves |
| **Vendoring from the wrong local checkout.** Multiple `tldw_server` checkouts exist locally and diverge — one is missing the entire `process_text/` package (§0) | Sync script verifies the synced tree matches the pinned SHA; never sync from a bare local path |
| Facts were first verified against a codex research branch, not `dev` | Re-anchored to `dev` @ `385afa95` (§0); only `rolling_summarize` drifted in-scope. #6-deferred files (`auto_boundary_assistant`, `propositions`) must be re-verified when #6 is specced |
| Licence compatibility of vendored server code | **Resolved (Q1): compatible.** `GPL-3.0-only` → `AGPL-3.0-or-later` is permitted by GPLv3 §13; preserve upstream GPLv3 notices + ship the `LICENSE` in the vendored subtree (§13.1) |
| Document-size ceilings differ between engines | Defaults already match (100 MB both sides, §9); only explicit configuration can diverge — the shim maps legacy config onto `max_text_size` |

## 13. Open questions for review

Q1, Q2, Q3, Q4, Q5, Q6, and Q7 were decided by the maintainer during review and are
recorded as **RESOLVED** below; Q8 is adopted as the plan's phase structure. Q9 is
newly opened by the second review and is also resolved.

1. **RESOLVED — licence is compatible; vendor it.** chatbook is `AGPL-3.0-or-later`;
   `tldw_server` is `GPL-3.0-only`. GPLv3 §13 explicitly permits combining a GPLv3 work
   with an AGPLv3 work into one distributable combined work: the vendored files stay
   GPLv3, the AGPL §13 network-use clause applies to the combination, and chatbook conveys
   as AGPL overall. `GPL-3.0-only` (no "or later") is fine because §13 names AGPL *v3*
   specifically. Obligations: **preserve the upstream GPLv3 license headers in every
   vendored file, ship `tldw_server`'s `LICENSE` (GPLv3) inside the vendored subtree
   (`Chunking/engine/LICENSE`), record the licence + source in `VENDOR_MANIFEST.toml`, and
   add `tldw_chatbook.Chunking.engine = ["LICENSE"]` to pyproject's vendored-license
   `license-files` block** (same pattern as `Third_Party/aider`, `textual_fspicker`).
2. **RESOLVED (default) — `tiktoken` is a hard requirement; never silent-swap.** Add
   `tiktoken` to chatbook's dependencies so the `tokens` method's preferred tokenizer
   always works. **Correction (second review):** the original rationale — that the
   vendored `tokens.py` *raises* rather than degrades, so the fallback question
   disappears — was **wrong**. The strategy's tokenizer resolution is a caught,
   three-level chain (tiktoken → transformers → `FallbackTokenizer` word approximation;
   the `raise ImportError` at `tokens.py:323` is immediately caught at line 324), so
   with tiktoken merely *declared*, a broken or absent install still silently
   degrades. The decision stands, but its enforcement moves to the shim: the `tokens`
   path must verify the resolved tokenizer is real (not the word-approximation
   fallback) and raise a clear "install tiktoken" error otherwise — silently returning
   word-approximation chunks breaks parity and is disallowed. (For symmetry: the claim
   that chatbook's current `tokens` path "already fails on a base install today" was
   also false — `TokenBasedChunker` likewise falls back to word approximation at
   `token_chunker.py:119-127`; hard-requiring tiktoken therefore changes existing
   degraded behavior into always-correct behavior, a deliberate improvement, and
   should be stated in release notes.) (`defusedxml` is already declared, §5.5;
   `blingfire`/`spacy` remain optional engine deps, §5.5.)
3. **RESOLVED — the re-chunk + re-index action defers to #2.** #1 ships **stamp + report
   only** (§8): the `chunk_engine_version` column, the migration, and the RAG Admin
   "chunked by an older engine (N items)" indicator. The user-triggered re-chunk/re-index
   **action** moves to sub-project #2; its acceptance criterion drops from this spec (§14).
4. **RESOLVED — facades stay silent until migration.** *(Ruled by the maintainer,
   2026-08-19: no `DeprecationWarning` in #1.)* The `Chunk_Lib` shim and the
   `ChunkingService` wrapper emit no warnings; imports re-point to `engine` as later
   sub-projects migrate callers, and warnings (if any) appear only once that migration
   begins. Demolition is tracked mechanically by the §5.6 fate table plus the plan's
   task list — no dedicated shrinking-inventory test in #1. `ChunkingService` keeps
   its wrapper body through #1 and becomes a pure re-export when the (b) convergence
   lands (§6.3.2).
5. **RESOLVED — retire `EnhancedChunkingService`; adapt at the seam.** *(Ruled by the
   maintainer, 2026-08-19.)* The home-grown `_structural_chunking` /
   `_hierarchical_chunking` logic is deleted; the engine's `structure_aware` strategy
   plus hierarchical boundaries become the only structure-aware implementation. The
   two RAG-indexing consumers keep their working shape via a **parent/child adapter**
   — a thin module that derives `parent_chunks` (the `chunk_with_parent_retrieval`
   return shape) from the engine's hierarchical output, so
   `enhanced_indexing_helpers.py` and `enhanced_rag_service.py` keep working with no
   signature change. `StructuredChunk`/`ChunkType` are re-exported from the adapter
   for `chunk_preview_modal.py`. This lands as its own PR inside phase (b) — the
   spec's own "hardest piece" warning — and its characterization tests pin the
   parent/child contract before the swap.
6. **RESOLVED — converge it.** *(Ruled by the maintainer, 2026-08-19.)*
   `Media/local_media_reading_service._chunk_text` routes through the engine in
   phase (b), like every other splitter. (Correction: `audio_processing._chunk_text`
   is **not** independent — it delegates to `ChunkingService.chunk_text`, so §6.3
   converges it automatically; it is no longer part of this question. See §6.1
   Group C.)
7. **RESOLVED — no engine-selection flag; accept the risk explicitly.** The proposed
   `[chunking] engine = "legacy" | "parity"` toggle (which would keep *both* the old and
   new chunking implementations shipping so behavior could be flipped back via config —
   an *engine-implementation* switch, unrelated to any user chunking config/template) is
   **not** added. Keeping both engines alive would re-create the very
   two-implementations duplication this sub-project exists to delete. The safety nets are:
   (i) parity is proven before merge (§10) and this is not deployed to a fleet mid-release;
   (ii) `git revert` — the vendor+shim swap is a small, self-contained commit set;
   (iii) the §8 version stamp identifies exactly what the new engine touched, with opt-in
   re-chunk. `Chunk_Lib`'s legacy implementation is deleted as stated in §6.2.
8. **Split this sub-project into three PRs (recommendation, not an open question).**
   As written it carries the vendored port, the convergence of three independent
   implementations, a schema migration, **and** RAG Admin UI — too much for one reviewable
   PR, and the reason Q3/Q7 keep hedging. Recommended split, with **ordering that matters**:
   (a) vendor + shim + `Chunk_Lib` callers; (b) converge Group B (`_chunk_text_in_process`,
   `EnhancedChunkingService`, `local_media_reading_service._chunk_text`); (c) stamp +
   migration + read-only "older engine (N items)" report. **(c) must come after (b)**,
   because the `chunk_engine_version` stamp is only meaningful once every write path routes
   through the engine — stamping while the regex path still writes chunks would mislabel
   them. Per Q3, the re-chunk/re-index **action** is **not** part of (c) — it defers to #2 —
   so (c) is stamp + migration + report only, which keeps #1's net-new UI minimal.
9. **RESOLVED — promote `defusedxml` to a core dependency.** *(Ruled by the
   maintainer, 2026-08-19.)* `defusedxml` moves from extras into chatbook's core
   `dependencies` (small, pure-Python, security-positive — the engine treats it as a
   security requirement for XML parsing). The `xml` method then always works, and
   §14's base-install AC's absent-list stays as written (defusedxml present, like
   tiktoken).

## 14. Acceptance criteria

- [ ] `tldw_chatbook/Chunking/engine/` contains the vendored tree with a
      `VENDOR_MANIFEST.toml` pinning repo + branch + SHA (`dev` @ `385afa95`, §0) and the
      file list
- [ ] Licence obligations met (§13.1): upstream GPLv3 headers preserved in every vendored
      file, `tldw_server`'s `LICENSE` shipped at `Chunking/engine/LICENSE`, licence recorded
      in `VENDOR_MANIFEST.toml`, and `tldw_chatbook.Chunking.engine = ["LICENSE"]` added to
      pyproject's `license-files`
- [ ] `Helper_Scripts/sync_chunking_engine.py` reproduces the vendored tree from a
      clean upstream checkout **at the pinned SHA**, idempotently, verifies the source tree
      matches that SHA (not a bare local path, §0), and fails loudly on local modifications
- [ ] No vendored file contains a hand edit; every chatbook-specific behavior lives in
      `Chunking/_shims/` or a subclass
- [ ] The three shims exist (`testing`, `config`, `prompt_loader`); `prompt_loader` is
      mandatory, not conditional (§5.3/§5.4)
- [ ] Every call site in §6.1.1 works unchanged, proven by characterization tests —
      **including** the two `EnhancedChunkingService`/`chunk_with_parent_retrieval`
      consumers (`enhanced_indexing_helpers.py:66`, `enhanced_rag_service.py:51`) — and
      the characterization set also covers the DB round-trip and navigation-reader
      seams (§10.3), not just chunker calls
- [ ] The shim aliases legacy exceptions with no engine equivalent (`LanguageDetectionError`
      → `LanguageNotSupportedError`, `MemoryLimitError` → size-ceiling error) and lets
      `rolling_summarize`'s `ProcessingError` surface as an actionable error (§6.2/§9)
- [ ] `tiktoken` is a declared chatbook dependency (Q2 resolved: hard-require) **and the
      shim enforces it**: a `tokens` request that would resolve to the vendored chain's
      word-approximation fallback raises a clear "install tiktoken" error instead of
      returning approximation chunks — there is **no** silent fall-back to words
- [ ] The shim re-exports the module-level constants with import-time consumers
      outside §6.1.1 — `DEFAULT_CHUNK_OPTIONS`, `MAX_CHUNK_SIZE_{WORDS,SENTENCES,
      PARAGRAPHS,TOKENS}`, `MAX_DOCUMENT_SIZE_{MB,BYTES}`, `ensure_nltk_data` (§6.1/§6.2)
- [ ] `RAG_Search/chunking_service._chunk_text_in_process` is deleted and
      `ChunkingService.chunk_text` routes **all** methods through the engine, with its
      existing validation errors preserved or their change documented, and its output
      keeps the **flat** per-chunk contract — top-level `text`/`start_char`/`end_char`/
      `word_count` (§6.3.2) — proven by a DB round-trip characterization test
      (`add_media_with_keywords` → non-NULL `start_char`/`end_char`/`chunk_index`
      columns, §10.3)
- [ ] After #1, `Chunking/engine/` is the only module in the repo containing
      chunk-splitting logic; the legacy `Chunk_Lib` implementation is deleted (§6.2),
      and every remaining facade per §5.6 is a shim, a re-export, or explicitly
      tracked as deferred demolition (Q4/Q5/Q6) — none contains splitting logic of
      its own (the Group B implementations §6.3 routes through the engine are the
      test: each one's bespoke splitter is gone or descoped in writing)
- [ ] `Widgets/chunk_preview_modal` previews the chunks ingestion actually produces
      (§7.3), including a non-zero word-count column (§6 shape note)
- [ ] The engine imports and chunks successfully on a **base install** with no sklearn,
      nltk, transformers, numpy, langdetect or blingfire present — degrading (e.g. `semantic`
      and blingfire sentence-splitting), not raising, on import. (`tiktoken` and `defusedxml`
      are core per Q2/Q9, so `tokens` and `xml` work; the science stack stays optional.)
- [ ] Golden fixtures are generated and verified at least once with test mode explicitly
      disabled, so they reflect production sanitization behavior (§10.2)
- [ ] Ingest throughput for a representative book is within an agreed margin of the
      current implementation, measured before and after
- [ ] `ebook_chapters` returns chapter chunks through the `RAG_Search.chunking_service`
      entry point instead of raising `InvalidChunkingMethodError`
- [ ] `import tldw_chatbook.Local_Ingestion.XML_Ingestion` succeeds
- [ ] Golden parity fixtures match server output exactly for every non-LLM method
      across the fixed corpus
- [ ] Ported offset and overlap property tests pass (the **dev** versions, §10.1) — run at
      least once with test mode explicitly disabled so they exercise the production
      sanitization path, not only the pytest-relaxed one (§10.2)
- [ ] `test_rolling_summarize_fail_closed.py` is ported and passes (dev-added, in scope)
- [ ] `UnvectorizedMediaChunks.chunk_engine_version` exists, the media DB schema version
      is bumped to 6 with a migration, and existing rows are left `NULL`
- [ ] RAG Admin reports how many media items were chunked by the older engine (read-only)
- [ ] No media item is re-chunked in #1 (the re-chunk/re-index **action** defers to #2, Q3);
      the stamp + report ship here, the action does not
- [ ] Full test suite and linters pass
- [ ] `Tests/Performance/test_app_import_weight.py` stays green — the engine loads at
      app boot via `chunking_interop_library` → `Chunking/__init__` → shim → engine,
      and no re-sync may introduce a module-scope heavy import (§10.8)
- [ ] `Docs/User_Guide/` updated where chunking behavior is user-visible
