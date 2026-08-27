# ADR-093: Ship immutable tiktoken tables for offline runtime use

Status: Accepted

Date: 2026-08-27

Related Task: [TASK-2526](../tasks/task-2526%20-%20tiktoken-is-not-a-declared-dependency.md)

Related Spec: [Offline tiktoken Runtime Assets](../../Docs/superpowers/specs/2026-08-27-offline-tiktoken-runtime-assets-design.md)

Extends: [ADR-032](032-immutable-installed-distribution-assets.md) and
[ADR-073](073-vendored-chunking-engine-parity.md)

## Context

ADR-073 makes `tiktoken` a core dependency for token-based chunking. Chatbook
also uses it for Console token and cost estimates. The Python package does not
embed its BPE tables; it downloads them from OpenAI's public blob host on first
use and caches them outside Chatbook's authority.

TASK-21968 demonstrated that a cold cache causes repeated hidden network
attempts and established the GPT-2, cl100k, and o200k tables requested by the
test suite. It intentionally kept those assets test-only. Production therefore
still cannot promise real tokenization during a cold offline start. Chatbook's
explicit model map also names p50k and r50k, which were absent from the observed
test workload.

## Decision

1. GPT-2's two data-gym files and the r50k, p50k, cl100k, and o200k BPE tables
   are reviewed application runtime assets under
   `tldw_chatbook/assets/tiktoken_cache/`.
2. The files retain tiktoken's native `sha1(source URL)` cache names. A checked
   manifest records readable names, source URLs, tiktoken's expected content
   hashes, and the update procedure.
3. The exact reviewed inventory ships in both source and wheel artifacts. The
   installed package is an immutable read owner under ADR-032; Chatbook never
   seeds, repairs, or writes these assets at runtime.
4. At package import, Chatbook uses `setdefault` to point
   `TIKTOKEN_CACHE_DIR` at the bundled directory. This is early enough for both
   token estimates and direct chunking-engine imports.
5. A caller-supplied `TIKTOKEN_CACHE_DIR` remains authoritative. That override
   may intentionally restore tiktoken's writable-cache and download behavior.
6. The default bundle is closed and deterministic. A new encoding requires an
   explicit Chatbook asset update; absent encodings follow the consuming
   caller's existing error or approximation policy rather than silently
   expanding the runtime network surface.
7. Tests and production use one asset inventory. Core-dependency tests never
   skip when `tiktoken` is missing, and tier-specific fallback tests isolate the
   global estimate cache before changing tokenizer availability.

## Alternatives Considered

### Copy bundled tables into each user's writable cache

Rejected because it duplicates large immutable files and introduces import- or
first-use writes, concurrency, partial-copy recovery, and private-directory
ownership work without improving current encoding coverage.

### Allow first-use downloads and document the fallback

Rejected because token accounting and local chunking should not acquire a
hidden network dependency, and it leaves cold offline installs on approximate
counts despite declaring the tokenizer as core.

### Point tests at vendored tables but leave production unchanged

Rejected because this is the TASK-21968 state: it stabilizes CI but does not
solve the user-visible offline runtime gap.

## Consequences

- Distribution size grows by the reviewed table inventory, roughly eight
  megabytes.
- Standard token estimates and GPT-2 token chunking no longer require first-use
  network access.
- The application package sets one additional process environment default, but
  never overrides an explicit caller value.
- New tiktoken encodings do not become available merely by upgrading the
  dependency; the asset inventory must be reviewed and updated in the same
  change.
- The existing character estimator remains the degraded path when `tiktoken`
  itself is unavailable.
