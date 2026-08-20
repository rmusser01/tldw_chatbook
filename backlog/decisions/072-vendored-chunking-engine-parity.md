# ADR-072: Vendor tldw_server's Chunking engine behind a compat shim

- **Status:** Accepted
- **Date:** 2026-08-20
- **Tasks:** task-18905 / task-18906 / task-18907 (Chunking Engine Parity,
  sub-project #1 of the parity program)
- **Related:** `Docs/superpowers/specs/2026-08-18-chunking-engine-parity-design.md`
  (design; the Q1–Q9 maintainer rulings recorded there are the long-form
  version of this ADR), `Docs/superpowers/plans/2026-08-19-chunking-engine-parity.md`
  (execution plan), `tldw_chatbook/Chunking/engine/VENDOR_MANIFEST.toml` (the pin)

## Context

chatbook accumulated **seven** chunking implementations (spec §6.1): the
2,268-line legacy `Chunk_Lib`, a regex splitter inside `chunking_service`,
`EnhancedChunkingService`, a char slicer in `local_media_reading_service`,
and three delegators. They drifted from each other and from `tldw_server`'s
maintained `Chunking` engine, so the same document chunked differently
depending on which code path ingested it, and server-side fixes never
reached chatbook. Five further sub-projects (#2–#6: templates, re-chunk
actions, agent surfaces, deferred strategies) build on whichever engine
survives.

## Decision

1. **Vendor the server engine at a pinned SHA; never hand-edit it.** The 35
   engine files live at `tldw_chatbook/Chunking/engine/`, reproduced from
   `tldw_server` `dev` @ `385afa95` solely by `Helper_Scripts/sync_chunking_engine.py`,
   which is driven by `VENDOR_MANIFEST.toml` (repo + branch + SHA + file
   list), verifies the source checkout is at the pin, and fails loudly on
   local modifications. Import paths are the only rewrite the sync applies.
2. **chatbook-specific behavior lives in shims, not in engine files.** Three
   `_shims` modules (`testing`, `config`, `Utils/prompt_loader`) satisfy the
   engine's imports; `Chunk_Lib.py` remains as a compat shim that preserves
   legacy signatures, exception aliases, constants, and the flat chunk
   contract while delegating all splitting to the engine.
3. **One engine, no selection flag (spec Q7).** Every splitter converged
   onto the engine and the legacy implementations were deleted (regex
   splitter, `EnhancedChunkingService` internals — now a pure delegation
   adapter, the char slicer). A `legacy|parity` config toggle was explicitly
   rejected: it would re-create the duplication this project deletes. Safety
   nets are the pre-merge golden parity proof (70/70 fixtures byte-identical
   to the real server engine), `git revert`, and the `chunk_engine_version`
   stamp (schema v6) that identifies exactly which chunks the new engine
   wrote — with re-chunk deferred to #2 (Q3).
4. **GPLv3-in-AGPL vendoring is licence-compatible (Q1, GPLv3 §13).** The
   vendored files stay `GPL-3.0-only`; chatbook conveys the combination as
   AGPL-3.0-or-later. Obligations shipped in-tree: upstream `LICENSE` scope
   map plus the full `LICENSES/GPL-3.0-only.txt` text under
   `Chunking/engine/`, licence recorded in the manifest, both files listed
   in pyproject's vendored `license-files` block (aider precedent).
5. **`tiktoken` and `defusedxml` are core dependencies (Q2, Q9).** The
   `tokens` method must never silently degrade to word approximation — the
   shim probes the engine's resolved tokenizer and raises a clear
   "install tiktoken" error if it is the fallback.

## Consequences

- Upstream fixes arrive by editing the manifest's SHA and re-running the
  sync script; hand edits to `engine/*.py` are forbidden and rejected by
  the sync's local-modification check.
- Sub-projects #2–#6 build on this boundary: templates/prompts vendoring
  (#2), re-chunk/re-index action (#2, per Q3), deferred modules
  (`propositions`, `token_chunker`/`language_chunkers` deletion) tracked by
  the manifest's excluded list and the spec §5.6 fate table.
- The shim layer is transitional: facades stay silent (Q4) and shrink as
  later sub-projects migrate callers to engine imports directly.
- Licence hygiene must survive future syncs: the sync script copies both
  licence files; `Tests/Chunking/test_sync_script.py` pins the manifest and
  tree completeness.
