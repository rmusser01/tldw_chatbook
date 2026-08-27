# Offline tiktoken Runtime Assets Design

**Task:** [TASK-2526](../../../backlog/tasks/task-2526%20-%20Ship-tiktoken-and-its-encoding-tables-for-offline-token-estimates.md)

**Decision:** [ADR-093](../../../backlog/decisions/093-offline-tiktoken-runtime-assets.md)

## Goal

Make real token estimates and token-based chunking work without network access
in a normal Chatbook installation, while preserving an explicit user cache
override and the existing no-`tiktoken` character fallback.

**Status:** Approved for implementation. The repository owner accepts the
upstream collaborator's statement that tiktoken's MIT license covers the
encoding files as sufficient redistribution evidence.

## Context

ADR-073 already made `tiktoken` a core dependency, so TASK-2526's original
packaging defect is no longer present. The dependency alone is insufficient:
`tiktoken` downloads BPE tables on first use. TASK-21968 observed the actual
requests, vendored GPT-2, `cl100k_base`, and `o200k_base` for tests, and proved
that a cold cache otherwise causes hidden network attempts.

Production still uses tiktoken's download-on-first-use default. The application
also explicitly maps legacy models to `p50k_base` and `r50k_base`, so a complete
runtime inventory must include those tables even though the current test sample
did not request them.

## Chosen Design

### Redistribution basis

The MIT license in the `tiktoken` repository covers the Python package and the
encoding files. OpenAI's public tiktoken issue #92 records the
[question and a collaborator's answer](https://github.com/openai/tiktoken/issues/92#issuecomment-1497875652)
that the repository license applies to those files. The Chatbook repository
owner accepts that upstream statement as the redistribution basis.
The GPT-2 vocabulary pair has a clear
[MIT source in `openai/gpt-2`](https://github.com/openai/gpt-2/blob/master/LICENSE)
as an additional provenance record.

The bundle will preserve tiktoken's MIT license, the collaborator's
clarification link, exact source URLs, and integrity hashes. The latter prove
provenance and integrity; the license and upstream clarification supply the
redistribution permission.

### Immutable cache ownership

Move TASK-21968's four cache entries from `Tests/fixtures/` into
`tldw_chatbook/assets/tiktoken_cache/` and add the `p50k_base` and `r50k_base`
entries. The directory is a tiktoken-native cache: each filename remains
`sha1(source URL)`. A human-readable manifest records encoding name, source URL,
tiktoken's expected SHA-256, and opaque cache filename.

The exact six binary files and manifest will be declared in `pyproject.toml`
and `MANIFEST.in`. The canonical `Packaging/check_manifest.py` contract will
also require every table, manifest, and notice, and reject unexpected entries
under the cache prefix. They will be immutable installed-distribution resources
under ADR-032; Chatbook will never write beneath its package root.

The manifest will pin tiktoken `0.14.0`, the constructor URL and expected
SHA-256 for every table, and the GPT-2 two-file hashes. `pyproject.toml` will
pin the same reviewed dependency version because the design relies on tiktoken's
`read_file_cached(blobpath, expected_hash)` seam, cache-key algorithm, and model
registry. An upgrade must re-audit those APIs, constructors, URLs, hashes, and
model-to-encoding mappings before changing the pin.

The application ships the exact required notices and an asset-provenance record
naming each source, hash, and redistribution basis. Packaging checks make those
records mandatory in sdist and wheel. The preserved MIT terms are compatible
with Chatbook's AGPL-3.0-or-later distribution.

### Runtime selection

At package import, a small `Utils/tiktoken_runtime.py` bootstrap will first record
whether `TIKTOKEN_CACHE_DIR` or legacy `DATA_GYM_CACHE_DIR` was supplied by the
caller. If either exists, it does nothing: tiktoken retains its upstream
writable-cache and download behavior byte-for-byte.

With no override, the bootstrap will point `TIKTOKEN_CACHE_DIR` at the bundled
directory and replace tiktoken `0.14.0`'s internal `read_file_cached` seam with
a bundled-only reader. The reader derives the native SHA-1 cache key from the
requested URL, reads that exact package file, verifies tiktoken's supplied
SHA-256, and returns bytes. Missing, corrupt, or unmanifested entries raise a
specific runtime error before any fetch, delete, directory creation, or write.
This happens before any Chatbook submodule can load an encoding and covers both
`Utils/token_counter.py` and direct vendored-engine imports.

The guard is installed only for the default bundle. Supporting a new encoding
requires an intentional asset/manifest update or a pre-import explicit cache
override. Import-time tests pin the tiktoken seam signature so an incompatible
dependency cannot silently bypass the guard.

### Failure matrix

| Condition | Token estimates | `Chunk_Lib` tokens method | Direct vendored engine |
|---|---|---|---|
| `tiktoken` library absent | Existing conservative character estimate | Existing real-tokenizer probe may use installed Transformers; otherwise raises `ChunkingError` before approximate chunks | Upstream engine behavior is unchanged and may select Transformers or its word fallback; it is not the supported Chatbook service boundary |
| Bundled table missing/corrupt | Guard raises before I/O; `get_tiktoken_encoding` logs and the existing OpenAI character approximation is returned | Probe may use installed Transformers; otherwise ADR-073 raises `ChunkingError` | Upstream engine may select Transformers or its word fallback; no fetch or package mutation occurs |
| Unknown model with a supported table | Existing cl100k model fallback remains real tokenization | Engine's existing cl100k fallback remains real tokenization | Same upstream cl100k behavior |
| New encoding absent from bundle | Same explicit logged approximation as a missing table | Same fail-closed compatibility behavior | Same upstream fallback behavior, without network |
| Explicit pre-import cache override | Original tiktoken cache/download/error behavior | Existing real-tokenizer enforcement | Original upstream engine behavior |

The direct engine is vendored internals, not the public Chatbook chunking
contract; ADR-073's no-word-approximation guarantee remains enforced at the
`Chunk_Lib` compatibility/service boundary. This task does not edit vendored
engine files.

### Test ownership

`Tests/conftest.py` will remove its test-only cache override; normal package
bootstrap will select the same runtime asset directory in tests and production,
with no second test-only copy. The existing cache-integrity test expands to all
six entries and continues proving that tokenization succeeds while the network
guard is active.

`Tests/Chat/test_token_counter.py` will stop skipping the real tokenizer test when
the core dependency is absent. Character-fallback tests explicitly disable
both tokenizer tiers and clear `_ESTIMATE_CACHE` before and after the tier
override. This fixes the clean-dev baseline failure without changing production
cache behavior: the failure came from a test that claimed to exercise fallback
while actually using tiktoken, then leaked that cached result into the next
fixture.

Installed-distribution coverage will verify that the exact files exist in both
sdist and wheel, imports Chatbook without a cache override, prohibits network
reads, and successfully loads every supported encoding from the installed
package. Missing- and corrupt-entry mutations prove zero fetches, zero package
tree changes, and an explicit error. Separate subprocesses prove both package
and direct-engine imports install the guard before encoding resolution, while a
pre-set override remains unchanged.

Artifact tests will cover both a source-built wheel and a wheel rebuilt from the
sdist. Each is installed outside the checkout, made read-only, and exercised
with checkout paths excluded so source files cannot satisfy missing assets.
Wheel and PKG-INFO metadata must retain the exact mandatory
`tiktoken==0.14.0` requirement.

## Alternatives Considered

### Seed a writable per-user cache

Rejected. It duplicates roughly eight megabytes per profile and adds first-run
filesystem mutation, concurrent seeding, partial-copy recovery, and ownership
checks. The installed package already provides an immutable read target.

### Keep runtime downloads and only improve fallback

Rejected. It keeps estimates approximate on cold offline installations and
retains an unexpected network side effect in a local accounting function.

### Bundle only the tables observed by TASK-21968

Rejected. Chatbook's model map explicitly requests `p50k_base` and `r50k_base`.
Test observation alone cannot prove coverage for models absent from the sample.

## Documentation and Compatibility

The user guide will state that standard tokenization is offline and bundled,
that pre-import `TIKTOKEN_CACHE_DIR`/`DATA_GYM_CACHE_DIR` overrides the bundle,
and that newly introduced encodings require a Chatbook update or explicit cache
configuration. The asset manifest documents the upstream version, source URLs,
expected hashes, redistribution notices, and repeatable update procedure. No
config schema, database migration, or UI setting is introduced; the existing
dependency becomes exactly pinned to its reviewed runtime contract.

## Verification

- Mutation-prove the real-token and fallback tests independently.
- Run focused token-counter and vendored-cache tests.
- Run focused chunking token-strategy tests using GPT-2.
- Build sdist and wheel, inspect the exact asset inventory, and tokenize from
  both source-built and sdist-rebuilt installed wheels with network access
  prohibited and the package tree read-only.
- Mutation-prove missing/corrupt assets, unexpected artifact entries, required
  notices, explicit override authority, dependency metadata, and checkout
  isolation.
- Run Ruff on changed Python files and `git diff --check`.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/093-offline-tiktoken-runtime-assets.md`

Reason: this changes the long-lived dependency runtime, package-data inventory,
network policy, and installed-asset ownership boundary.
