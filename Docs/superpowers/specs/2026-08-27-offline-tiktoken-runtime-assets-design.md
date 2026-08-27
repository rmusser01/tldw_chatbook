# Offline tiktoken Runtime Assets Design

**Task:** [TASK-2526](../../../backlog/tasks/task-2526%20-%20tiktoken-is-not-a-declared-dependency.md)

**Decision:** [ADR-093](../../../backlog/decisions/093-offline-tiktoken-runtime-assets.md)

## Goal

Make real token estimates and token-based chunking work without network access
in a normal Chatbook installation, while preserving an explicit user cache
override and the existing no-`tiktoken` character fallback.

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

### Immutable cache ownership

Move TASK-21968's four cache entries from `Tests/fixtures/` into
`tldw_chatbook/assets/tiktoken_cache/` and add the `p50k_base` and `r50k_base`
entries. The directory is a tiktoken-native cache: each filename remains
`sha1(source URL)`. A human-readable manifest records encoding name, source URL,
tiktoken's expected SHA-256, and opaque cache filename.

The exact six binary files and manifest are declared in `pyproject.toml` and
`MANIFEST.in`. They are immutable installed-distribution resources under
ADR-032; Chatbook never writes beneath its package root.

### Runtime selection

`tldw_chatbook/__init__.py` sets `TIKTOKEN_CACHE_DIR` to the bundled directory
with `os.environ.setdefault`. This happens before any Chatbook submodule can
load a tiktoken encoding and covers both `Utils/token_counter.py` and direct
imports of the vendored chunking engine.

An environment value supplied before package import, including a custom
writable cache, remains authoritative. The default path never downloads: an
encoding absent from the reviewed bundle fails tiktoken resolution and follows
the caller's existing error/fallback policy. Supporting a new encoding requires
an intentional asset and manifest update or an explicit cache override.

### Test ownership

`Tests/conftest.py` points its cache guard at the same runtime asset directory;
there is no second test-only copy. The existing cache-integrity test expands to
all six entries and continues proving that tokenization succeeds while the
network guard is active.

`Tests/Chat/test_token_counter.py` stops skipping the real tokenizer test when
the core dependency is absent. Character-fallback tests explicitly disable
both tokenizer tiers and clear `_ESTIMATE_CACHE` before and after the tier
override. This fixes the clean-dev baseline failure without changing production
cache behavior: the failure came from a test that claimed to exercise fallback
while actually using tiktoken, then leaked that cached result into the next
fixture.

Installed-distribution coverage verifies that the exact files exist in both
sdist and wheel, imports Chatbook without a cache override, prohibits network
reads, and successfully loads every supported encoding from the installed
package.

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
that `TIKTOKEN_CACHE_DIR` overrides the bundle, and that newly introduced
encodings require a Chatbook update or explicit cache configuration. No config
schema, database migration, UI setting, or new dependency is introduced.

## Verification

- Mutation-prove the real-token and fallback tests independently.
- Run focused token-counter and vendored-cache tests.
- Run focused chunking token-strategy tests using GPT-2.
- Build sdist and wheel, inspect the exact asset inventory, and tokenize from
  the installed wheel with network access prohibited.
- Run Ruff on changed Python files and `git diff --check`.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/093-offline-tiktoken-runtime-assets.md`

Reason: this changes the long-lived dependency runtime, package-data inventory,
network policy, and installed-asset ownership boundary.
