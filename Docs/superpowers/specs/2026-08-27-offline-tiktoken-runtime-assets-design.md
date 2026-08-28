# Offline tiktoken Runtime Assets Design

**Task:** [TASK-2526](../../../backlog/tasks/task-2526%20-%20Ship-tiktoken-and-its-encoding-tables-for-offline-token-estimates.md)

**Decision:** [ADR-093](../../../backlog/decisions/093-offline-tiktoken-runtime-assets.md)

## Goal

Make real token estimates and token-based chunking work without network access
in a normal Chatbook installation, while preserving an explicit user cache
override and the existing no-`tiktoken` character fallback.

**Status:** Implemented and verified. The repository owner accepts the
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

The runtime bundle contains TASK-21968's four former test cache entries plus
the `p50k_base` and `r50k_base` entries under
`tldw_chatbook/assets/tiktoken_cache/`. The directory is a tiktoken-native
cache: each filename remains
`sha1(source URL)`. A human-readable manifest records encoding name, source URL,
tiktoken's expected SHA-256, and opaque cache filename.

The exact six binary files plus manifest, license, and notice are declared in
`pyproject.toml` and `MANIFEST.in`. The canonical
`Packaging/check_manifest.py` contract requires all nine entries and rejects
unexpected entries under the cache prefix. They are immutable
installed-distribution resources under ADR-032; Chatbook never writes beneath
its package root.

The manifest pins tiktoken `0.14.0`, the constructor URL and expected
SHA-256 for every table, and the GPT-2 two-file hashes. `pyproject.toml` pins
the same reviewed dependency version because the design relies on tiktoken's
`read_file_cached(blobpath, expected_hash)` seam, cache-key algorithm, and model
registry. An upgrade must re-audit those APIs, constructors, URLs, hashes, and
model-to-encoding mappings before changing the pin.

The application ships the exact required notices and an asset-provenance record
naming each source, hash, and redistribution basis. Packaging checks make those
records mandatory in sdist and wheel. The preserved MIT terms are compatible
with Chatbook's AGPL-3.0-or-later distribution.

### Runtime selection

At package import, `Utils/tiktoken_runtime.py` first checks whether
`TIKTOKEN_CACHE_DIR` or legacy `DATA_GYM_CACHE_DIR` was supplied by the caller.
If either exists, it returns before importing tiktoken: tiktoken retains its
upstream writable-cache and download behavior byte-for-byte.

With no override, the bootstrap points `TIKTOKEN_CACHE_DIR` at the bundled
directory and replaces tiktoken `0.14.0`'s internal `read_file_cached` seam with
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

`Tests/conftest.py` has no test-only cache override; normal package bootstrap
selects the same runtime asset directory in tests and production, with no
second test-only copy. The cache-integrity test covers all six entries and
proves that tokenization succeeds while the network guard is active.

`Tests/Chat/test_token_counter.py` does not skip the real tokenizer test when
the core dependency is absent. Character-fallback tests explicitly disable
both tokenizer tiers and clear `_ESTIMATE_CACHE` before and after the tier
override. This fixes the clean-dev baseline failure without changing production
cache behavior: the failure came from a test that claimed to exercise fallback
while actually using tiktoken, then leaked that cached result into the next
fixture.

Installed-distribution coverage verifies that the exact files exist in both
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

### Portable release archives

The release checker validates extraction safety before trusting archive
contents. Wheel and sdist names must be canonical relative POSIX paths and
unique by both exact spelling and case-folded extraction path. Absolute and
drive-qualified paths, backslashes, dot/parent segments, repeated separators,
trailing-slash aliases, control and Windows-invalid characters, components
ending in a dot or space, and reserved Windows device stems are rejected. The
device table includes COM1-9/LPT1-9 and the COM¹/²/³ and LPT¹/²/³ aliases,
including names with extensions. Sdist entries are restricted to regular files
and directories, and wheel cache entries must be regular files, preventing
links or duplicate/alias members from replacing validated metadata or cache
assets during extraction.

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

Token/runtime/chunking evidence is run without a keyword filter so every test in
the four behavior-owning files executes:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Chat/test_token_counter.py \
  Tests/test_tiktoken_vendored_cache.py \
  Tests/Chunking/test_tokens_offsets.py \
  Tests/Chunking/test_chunk_lib_shim.py
```

Packaging evidence is a separate selection. It builds the sdist and source
wheel, rebuilds a wheel from the sdist, checks the exact inventory and metadata,
exercises read-only/offline/missing/corrupt paths, and runs the complete release
checker hardening matrix:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Packaging/test_installed_distribution.py \
  -k 'tiktoken or built_artifacts_match_distribution_contract or release_checker'
```

Completion also requires Ruff over every changed Python file, `py_compile` on
the loader and checker, and `git diff --check`. A full repository suite is not
part of this focused gate without separate owner opt-in.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/093-offline-tiktoken-runtime-assets.md`

Reason: this changes the long-lived dependency runtime, package-data inventory,
network policy, and installed-asset ownership boundary.
