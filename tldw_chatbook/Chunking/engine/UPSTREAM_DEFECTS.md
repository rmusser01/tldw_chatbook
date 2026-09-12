# Upstream defects observed at the vendor pin

Server-side defects found while building chunking template parity
(sub-project #2). Recorded here — not as chatbook backlog tasks — because
an id on this board cannot be verified against the server's tracker
(design spec §11 ruling). This file sits beside `VENDOR_MANIFEST.toml`
so a future pin bump can re-check each item against the new SHA.

- **Pin:** `tldw_server` `dev` @ `385afa951922c8a9dc2002c675bb6cad65e4ac23`
- **Engine root:** `tldw_Server_API/app/core/Chunking/`
- **Endpoints root:** `tldw_Server_API/app/api/v1/endpoints/`
- **Source spec:** `Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md` §11 items 9–14
- **Governance:** ADR-078 (convergence decision that surfaced these)

Line numbers are as-read at the pin; re-verify before filing upstream.

---

## 1. Phantom `template_library/` directory (spec §11 item 9)

- **Where:** documented as shipping built-ins in
  `tldw_Server_API/app/core/Chunking/README.md:21,31,142` and
  `Docs/Chunking/Chunking_Templates.md`; `TemplateManager.__init__`
  creates it empty at
  `tldw_Server_API/app/core/Chunking/templates.py:546`; the built-ins
  that actually run are a hardcoded Python fallback at
  `tldw_Server_API/app/core/Chunking/template_initialization.py:129-209`.
- **Defect:** the directory has never existed in git (verified at the
  pin: `git ls-tree -r HEAD` over the Chunking tree returns no JSON),
  yet the docs describe it as the built-in source and both filesystem
  load strategies target it before dead-ending at the fallback.
- **chatbook impact:** there was nothing to "port" — chatbook lifts the
  six built-ins **as data** from the hardcoded fallback (spec §5.5),
  each seed carrying a provenance comment. Until upstream ships a real
  library, a pin bump must diff chatbook's seeds against
  `template_initialization.py`, not against JSON files.
- **Filed upstream:** no — <link>

## 2. Flat→internal mapper copy-pasted three times, two copies unguarded (spec §11 item 10)

- **Where:**
  `tldw_Server_API/app/core/Chunking/templates.py:658-691`;
  `tldw_Server_API/app/api/v1/endpoints/chunking_templates.py:712-744`;
  `tldw_Server_API/app/api/v1/endpoints/chunking.py:293-321`.
- **Defect:** the flat-shape→`ChunkingTemplate` mapping is implemented
  three times, and the two endpoint copies do not guard a missing
  `chunking` key — a template without it raises `KeyError` into a
  generic handler instead of a named validation error.
- **chatbook impact:** chatbook implements the mapper once
  (`Chunking/template_runtime.py`) with the guard (spec §4.3). The
  triplication is why parity had to be proven against one specific copy
  rather than "the server's mapper".
- **Filed upstream:** no — <link>

## 3. Validate/runtime asymmetries (spec §11 item 11)

- **Where:** validate endpoint
  `tldw_Server_API/app/api/v1/endpoints/chunking_templates.py:782-992`
  vs runtime processor
  `tldw_Server_API/app/core/Chunking/templates.py`.
- **Defect (three):** (a) validation requires the `operation` key while
  the runtime also accepts `{type, params}`, so a valid-at-runtime
  template can fail validation; (b) validation never checks that an
  operation **name** is registered — an unknown op validates clean and
  is warned-and-skipped at runtime; (c) the pydantic pass silently
  drops unknown top-level keys before the hand-rolled checks run.
- **chatbook impact:** chatbook's local validator **deliberately
  replicates all three** (spec §7.1, pinned by tests) so that a
  template validating here validates there. If upstream fixes any of
  them, the fix must arrive through a pin bump plus a parity-test
  update — the pins are what stop a local "fix" from silently
  diverging from the server.
- **Filed upstream:** no — <link>

## 4. Duplicate built-in definitions, divergent spelling (spec §11 item 12)

- **Where:**
  `tldw_Server_API/app/core/Chunking/template_initialization.py:132-208`
  (flat `{operation, config}`) vs
  `tldw_Server_API/app/core/Chunking/templates.py:556-614`
  (`TemplateManager`'s in-memory store, stage-based `{type, params}`).
- **Defect:** `academic_paper`, `code_documentation`, and
  `chat_conversation` are defined twice with the same content but a
  divergent schema spelling. Which definition a caller gets depends on
  which loader won.
- **chatbook impact:** chatbook adopts the flat spelling only and fences
  `TemplateManager` entirely (no production module may construct it —
  constructing it would resurrect a two-store drift, spec §6.3). The
  duplication is upstream's own instance of the two-stores problem this
  sub-project deletes on the chatbook side.
- **Filed upstream:** no — <link>

## 5. Stale hardcoded method fallback list (spec §11 item 13)

- **Where:**
  `tldw_Server_API/app/api/v1/endpoints/chunking_templates.py:832`.
- **Defect:** the endpoint's fallback list of valid chunking methods
  has 11 names and omits `fixed_size`, `code`, `code_ast` — methods the
  engine actually registers. A template using one of them can be
  rejected by validation yet run fine at runtime.
- **chatbook impact:** chatbook's validator resolves methods against
  the **live** engine registry
  (`Chunking.engine.chunker.Chunker().get_available_methods()`),
  never a hardcoded list (spec §7). Any drift between the endpoint's
  list and the registry is a template that validates on one side only.
- **Filed upstream:** no — <link>

## 6. Preprocessing metadata collected then discarded (spec §11 item 14)

- **Where:**
  `tldw_Server_API/app/core/Chunking/templates.py:159-168`.
- **Defect:** `process_template` merges metadata returned by
  preprocessing operations (`extract_sections`, `detect_language`) into
  `data["metadata"]` and never carries it into the returned chunks —
  the work runs and is thrown away.
- **chatbook impact:** chatbook's `academic_paper` seed uses
  `extract_sections`, so its section extraction currently produces
  nothing observable. This is a known degradation documented in the
  user guide (spec §5.5) rather than papered over; chatbook's
  synthesized chunk contract (spec §6.4) does not depend on it.
- **Filed upstream:** no — <link>

## 15. Validate endpoint converts any warning into a hard failure (warnings typed as `list[str]` but objects appended)

- **Upstream:** `tldw_Server_API/app/api/v1/endpoints/chunking_templates.py`
  (~:893-897 append, ~:981 `warnings if warnings else None`) and
  `tldw_Server_API/app/api/v1/schemas/chunking_templates_schemas.py:172`
  (`warnings: Optional[list[str]]`). The endpoint appends
  `TemplateValidationError` **objects** into `warnings`; response-model
  construction then raises `pydantic.ValidationError` (a `ValueError`),
  which the endpoint's outer `except` catches — so on the as-executed
  server, ANY template that produces a warning (e.g. an unanchored `.*`
  boundary) returns `valid=False` with "Validation error: …" and leaks
  pydantic internals.
- **chatbook impact:** chatbook's local validation deliberately diverges
  (spec §7 ruling: a warning never flips validity); the fixture row
  `unanchored-wildcard-warns-not-errors` pins that divergence. At the next
  pin bump, re-check whether upstream fixed the typing; if they did,
  chatbook's behavior already matches and only the fixture note changes.
- **Discovered:** sub-project #2, PR A task 6 (implementation review).
- **Filed upstream:** no — <link>

## 16. Auto/explicit template paths apply only the hierarchical block (all other template stages silently dropped)

- **Where:**
  `tldw_Server_API/app/core/Ingestion_Media_Processing/chunking_options.py:519-648`
  (`apply_chunking_template_if_any`, the shared helper every process-*
  endpoint calls; explicit-by-name branch :543-579, auto-apply branch
  :586-643).
- **Defect:** after picking a winner (auto loop, `score <= 0: continue`,
  best `(score, priority)` strictly-greater) or loading an explicit
  template by name, the helper extracts ONLY
  `chunking.config.hierarchical_template` + `chunking.method` +
  `chunking.config.{max_size, overlap}` into the options dict. A winning
  or explicitly-named template whose hierarchical block is absent (or not
  a dict) is **silently ignored** — nothing of it is applied — and the
  template's preprocessing/postprocessing stages never run on either
  path: templates with non-hierarchical bodies are inert on ingestion no
  matter how they were chosen.
- **chatbook impact:** chatbook deliberately diverges (auto-selection
  spec `2026-08-22-chunking-auto-selection-design.md` §0.2 divergence 1,
  ruling §8.6): a winning template runs IN FULL — preprocessing,
  chunking, postprocessing — through sub-project #2's template engine,
  indistinguishable from a manual pick; `Chunking/auto_selection.py`'s
  tier 1 returns the whole resolved template for exactly that reason.
- **Discovered:** 2026-08-22, spec review of sub-project #3 (read at the
  pin via the local `tldw_server2` checkout).
- **Filed upstream:** no — <link>
