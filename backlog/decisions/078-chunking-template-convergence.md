# ADR-078: Converge chunking templates on tldw_server's flat shape

- **Status:** Accepted
- **Date:** 2026-08-21
- **Tasks:** task-1 (ADR + governance filings; this ADR precedes all
  implementation) and the PR 0/A–E tasks of the Chunking Template Parity
  sub-project (#2 of the parity program)
- **Related:** `Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`
  (design; §13's eight maintainer rulings are the long-form version of this
  ADR), `Docs/superpowers/plans/2026-08-21-chunking-template-parity.md`
  (execution plan), ADR-073 (vendored engine this builds on), ADR-030
  (atomic media migrations), ADR-003 (Library owns the RAG surface),
  ADR-032 (distribution obligations for deleted template data)
- **Number sweep (2026-08-21):** remote refs
  (`git for-each-ref refs/remotes/` + `ls-tree`) max 076 — two ADRs
  currently numbered 076 on dev; all 15 sibling worktrees max 076;
  main-checkout untracked claims 042/067 (existing numbers). 077 is
  **spoken for**: TASK-19610 (To Do, `origin/docs/lesson-adr-number-collisions`)
  renumbers the later-claimed 076 (server-offload) to 077, verified free
  by that task. This ADR therefore takes 078. 047/048 are historical
  gaps, not reused.

## Context

ADR-073 vendored the chunking **engine** but left templates forked: two
stores in two incompatible shapes (the Media DB `ChunkingTemplates` table
with five seeded templates vs `Chunking/templates/*.json` with thirteen
files), three of the five seeds naming methods that do not exist (they
raise `InvalidChunkingMethodError` on apply), apply silently dropping
preprocessing/postprocessing, zero UI, zero production callers, and a
validator nothing invokes. The server's template system at the same pin
(385afa95) has none of these problems: one flat
`{preprocessing, chunking, postprocessing, classifier, metadata}` shape,
six built-ins that all execute cleanly on chatbook's vendored engine
(measured, spec §5.5), and a validate endpoint.

The program's original premise ("template v2", a `template_library/` port,
a v1→v2 migration) was wrong in three of five clauses (spec §0.1): there
is no version key inside any template, the library directory is a phantom
that has never existed in git, and the server accepts two shapes
concurrently with no migration. Parity therefore means converging on the
**flat shape as it actually is**, warts included.

## Decision

1. **Full convergence to the server's flat shape — one shape, one store.**
   `template_json` holds exactly `{preprocessing, chunking, postprocessing,
   classifier, metadata}`; `name`/`description`/`tags` are columns. The
   file-based store (`Chunking/templates/`, `chunking_templates.py`) is
   deleted; the Media DB table is the only store. Additive parity (keeping
   the chatbook pipeline shape alongside) was rejected — it preserves the
   two-store drift this sub-project exists to end. The server's
   stage-based shape is tolerated by the (vendored) reader but never
   written.
2. **Media DB v6 → v7 is a table rebuild of `ChunkingTemplates`**, adding
   `uuid`, `tags`, `is_builtin`, `version`, `deleted` plus a partial
   unique index on live names, executed under ADR-030's single-transaction
   rules via `_execute_transactional_script`, recreating the
   update-timestamp trigger. The schema and its only CRUD layer
   (`ChunkingInteropService`, normalizers) ship in the same PR — a schema
   and its sole reader cannot ship apart (spec §5.2.1).
3. **The template processor is vendored, not re-implemented.** The
   server's `templates.py` (`TemplateProcessor` + dataclasses) joins
   ADR-073's manifest at the existing pin; `TemplateManager`,
   `TemplateClassifier`, and `TemplateLearner` are vendored-but-fenced
   (no production module may construct them — `TemplateManager` mkdirs a
   directory and carries a divergent second in-memory store).
   chatbook adds one seam module, `Chunking/template_runtime.py`: the
   single flat→internal mapper (with the missing-`chunking` guard two of
   the server's three copies lack), the only name→template resolver, and
   `apply_template`, which synthesizes the flat chunk contract
   (offsets, `chunk_index`, `total_chunks`, `word_count`,
   `metadata.offset_basis`) the processor does not supply (spec §6.4).
4. **Local validation re-implements the server's validate endpoint
   check-for-check, including its warts.** `operation` required at
   validate while the runtime also accepts `type`; no unknown-operation
   check; unknown top-level keys silently dropped — chatbook replicates
   all three deliberately (pinned by tests), because parity means a
   template that validates here validates there. Methods resolve against
   the **live** engine registry, not the endpoint's stale hardcoded
   fallback. Create/update/seeding call the validator and refuse invalid
   bodies.
5. **The re-chunk action and the legacy-engine report line live on the
   Library RAG surface, honoring ADR-003 as written** (Settings is
   explicitly excluded there; the existing Backfill button's placement is
   undocumented drift, filed separately). Re-chunk replaces an item's
   chunk rows stamped with the current engine version and forces
   re-indexing explicitly (`needs_reindexing` would skip every item).
   Hard-delete of derived chunk rows is accepted; ADR-055 does not apply
   to regenerating a projection from an intact source.

## Consequences

- **v7 makes downgrade impossible, voiding ADR-073's `git revert` safety
  net.** `_initialize_schema` hard-raises `SchemaError` when the DB
  version exceeds the code's (`Client_Media_DB_v2.py:1477-1480`), so after
  v7 ships, reverting the code leaves every migrated media DB unopenable.
  ADR-073 named `git revert` as a program-wide safety net; for this
  sub-project that net does not exist. The remaining safety nets are the
  ADR-030 single-transaction migration (a seeded mid-rebuild failure
  leaves the DB at v6 intact) and pre-proven seeds (validated at
  build/test time, never during a user's migration). This cost is stated
  here and in the release notes; it is a real cost of the schema change,
  not an oversight.
- The vendored processor returns no chunk metadata, so chatbook owns one
  normalization step (`apply_template`) that synthesizes the flat
  contract; offsets are relative to preprocessed text when preprocessing
  rewrites it (`metadata.offset_basis` says which).
- Deleting the file store is a breaking change to
  `tldw_chatbook.Chunking`'s public namespace and a distribution change
  under ADR-032 (five packaging sites move in the same commit); the
  CHANGELOG records both, and the rollback story for that PR is surgical
  revert of an isolated PR, not the schema.
- Discovered defects are filed, not absorbed: chatbook-side findings as
  backlog tasks (spec §11 items 1–8), server-side findings in
  `tldw_chatbook/Chunking/engine/UPSTREAM_DEFECTS.md` beside the pin
  (spec §11 items 9–14), so a future sync can act on them.
- Rolling-summarize becomes fail-closed everywhere (shim markers were
  data corruption with a friendly face), and name resolution moves to the
  service layer so the import-light `Chunk_Lib` shim stays DB-free.
