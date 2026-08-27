# ADR-093: Ship immutable tiktoken tables for offline runtime use

Status: Accepted

Date: 2026-08-27

Related Task: [TASK-2526](../tasks/task-2526%20-%20Ship-tiktoken-and-its-encoding-tables-for-offline-token-estimates.md)

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
   manifest pins tiktoken `0.14.0`, readable names, source URLs, expected
   content hashes, constructor/cache API assumptions, and the update procedure.
   The core dependency is exactly pinned to the same reviewed version.
3. Tiktoken's MIT license applies to the encoding files. In
   [tiktoken issue #92](https://github.com/openai/tiktoken/issues/92#issuecomment-1497875652),
   an OpenAI collaborator stated that the repository license applies to the
   encoding files, and the Chatbook repository owner accepts that statement as
   sufficient redistribution evidence. The GPT-2 pair also has a clear
   [`openai/gpt-2` MIT source](https://github.com/openai/gpt-2/blob/master/LICENSE),
   and the exact tables, manifest, license, clarification link, and provenance
   notice are mandatory in source and wheel artifacts. Source URLs and hashes
   prove provenance and integrity; the preserved MIT terms and upstream
   clarification supply the redistribution permission.
4. The installed package is an immutable read owner under ADR-032. The
   canonical release checker requires the exact cache inventory, rejects
   unexpected cache entries, and verifies source-built and sdist-rebuilt wheels
   from read-only installed trees outside the checkout.
5. At package import, when neither supported cache environment variable was
   supplied, Chatbook points `TIKTOKEN_CACHE_DIR` at the bundle and replaces
   tiktoken's reviewed `read_file_cached` seam with a bundled-only reader. The
   reader verifies the requested table and hash and rejects missing, corrupt,
   or unmanifested data before any fetch, delete, directory creation, or write.
6. A pre-import caller-supplied `TIKTOKEN_CACHE_DIR` or legacy
   `DATA_GYM_CACHE_DIR` bypasses the guard and remains byte-for-byte
   authoritative, intentionally restoring upstream writable-cache/download
   behavior.
7. The default bundle is closed and deterministic. A new encoding requires an
   explicit Chatbook asset update. Token estimates log the load failure and use
   their existing character approximation; the `Chunk_Lib` tokens method keeps
   ADR-073's real-tokenizer probe and raises before word-approximate chunks.
   Direct vendored-engine imports retain upstream fallback behavior and are not
   the supported Chatbook chunking boundary; vendored files remain unmodified.
8. Tests and production use one asset inventory. Core-dependency tests never
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
  never overrides a pre-import explicit caller value.
- The design intentionally depends on tiktoken's reviewed internal load seam;
  dependency upgrades require a compatibility and asset audit before widening
  the exact pin.
- New tiktoken encodings do not become available merely by upgrading the
  dependency; the asset inventory must be reviewed and updated in the same
  change.
- The existing character estimator remains the degraded path when `tiktoken`
  itself is unavailable.
