---
id: TASK-2526
title: Ship tiktoken and its encoding tables for offline token estimates
status: Done
assignee: []
created_date: '2026-08-06 02:22'
updated_date: '2026-08-28 01:11'
labels:
  - packaging
  - tokens
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Utils/token_counter.py` prefers `tiktoken` and falls back to a conservative
character estimate. Since this task was filed, ADR-073 made `tiktoken` a core
dependency, satisfying the original dependency-placement concern. TASK-21968
then proved that `tiktoken` still downloads its encoding tables on first use
and vendored the observed tables for tests only.

Promote that proven offline test inventory into immutable application assets so
a normal installation performs real subword tokenization without network
access. Preserve explicit `TIKTOKEN_CACHE_DIR` authority and the existing
character fallback for environments where `tiktoken` is genuinely absent.

An OpenAI collaborator confirmed that tiktoken's MIT repository license applies
to the encoding files, and the repository owner accepts that statement as the
redistribution basis. The bundle must preserve the license, clarification link,
source URLs, and hashes.

Baseline investigation on 2026-08-27 also found that the old fallback test did
not actually select the fallback tier when `tiktoken` was installed. Its real
token count then leaked through the process-global estimate cache into a later
fixture. The task must make those tier-specific tests explicit and cache-clean.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `tiktoken==0.14.0` is a mandatory reviewed base dependency and its
      absence fails the standard tokenizer test instead of silently skipping
      it; built metadata retains that exact requirement.
- [x] #2 The exact GPT-2, r50k, p50k, cl100k, and o200k tables requested by
      Chatbook and its chunking engine ship in both the wheel and source
      distribution with the accepted redistribution evidence.
- [x] #3 With no explicit cache override, source and installed-distribution
      tokenization uses a guarded immutable bundle; missing/corrupt entries make
      zero fetch attempts and cannot mutate the package tree.
- [x] #4 An explicitly supplied `TIKTOKEN_CACHE_DIR` or legacy
      `DATA_GYM_CACHE_DIR` remains byte-for-byte authoritative and uses
      tiktoken's normal cache/download behavior.
- [x] #5 Tier-specific tests explicitly select and isolate the real-tokenizer
      and character-fallback paths; token estimates retain their existing
      character fallback when `tiktoken` is absent, while token chunking keeps
      ADR-073's fail-closed compatibility-shim contract.
- [x] #6 The runtime-asset policy, update procedure, provenance, and offline
      limitation for newly introduced encodings are documented, including the
      redistribution basis and required notice/license files.
- [x] #7 The canonical release checker requires the exact cache inventory and
      notices, rejects unexpected entries, and validates source-built plus
      sdist-rebuilt wheels from a read-only installed tree.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for the closed runtime bundle, explicit override authority,
   immutable failure behavior, and every supported encoding.
2. Add the minimal import-time tiktoken loader guard and the reviewed six-file
   cache inventory with manifest, MIT license, and provenance notice.
3. Pin `tiktoken==0.14.0`; extend package data, source distribution, metadata,
   and canonical release-checker contracts with exact-inventory rejection.
4. Fix the existing token-counter fallback test isolation without changing the
   production estimate cache contract.
5. Prove source-built and sdist-rebuilt wheels tokenize offline from read-only
   installed trees, including missing/corrupt and unexpected-entry mutations.
6. Run focused tests, static checks, artifact verification, independent review,
   and complete the task documentation before opening the PR.

Detailed plan: [Offline tiktoken Runtime Assets Implementation Plan](../../Docs/superpowers/plans/2026-08-27-offline-tiktoken-runtime-assets.md)

ADR required: yes

ADR path: `backlog/decisions/093-offline-tiktoken-runtime-assets.md`

Reason: this changes the dependency runtime, package-data inventory, network
policy, redistribution record, and immutable installed-asset boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-093's closed, immutable tiktoken runtime bundle. Package import
now preserves an explicit pre-import cache override or installs a
manifest-checked, no-fetch reader for the six reviewed blobs; packaging pins
`tiktoken==0.14.0`, ships the exact nine-entry cache directory, and validates
portable wheel/sdist extraction paths and metadata. Token-counter tests now
select the real and character tiers deterministically without changing the
production estimate cache contract.

Acceptance evidence:

1. The mandatory real-token test asserts tiktoken availability and its loaded
   encoding; built METADATA and PKG-INFO plus checker mutations require exactly
   `tiktoken==0.14.0`.
2. Source inventory/hash tests and built-artifact tests require the two GPT-2
   files and r50k, p50k, cl100k, and o200k in both artifact paths, with the MIT
   license, provenance notice, and accepted collaborator clarification.
3. Source and installed missing/corrupt mutations prohibit upstream reads and
   writes; source-built and sdist-rebuilt wheels tokenize from read-only trees.
4. Fresh subprocess tests cover both pre-import override variables and prove
   their values and upstream reader remain unchanged.
5. Tier-isolated token-counter tests prove real tokenization and character
   fallback; chunk-shim tests prove the tokens method raises before word
   approximation when no real tokenizer succeeds.
6. The manifest/notice, Packaging design, Library user guide, implementation
   spec, plan, and ADR document the closed inventory, hashes, update procedure,
   offline limitation, immutable ownership, and redistribution basis.
7. Built-artifact acceptance and the full release-checker mutation matrix cover
   exact cache contents, portable archive paths, canonical metadata, both build
   paths, and read-only installed execution.

Fresh verification on 2026-08-27: the unfiltered token/runtime/chunking gate
passed 85 tests; the separate packaging gate passed 120 tests with 19 unrelated
tests deselected. Ruff passed on all seven Python files changed by TASK-2526;
`py_compile` passed for the loader and checker; `git diff --check` passed. The
full repository suite was not run, per repository instruction.

PR review closeout tightened the manifest boundary with strict Pydantic models
and normalized schema failures, restored conservative character estimates when
the bundled reader fails, documented the installer exception, and added the
cache directory explicitly to the supported macOS Nuitka command. Focused
regressions cover malformed manifests, short ASCII/CJK fallback estimates, and
the generated Nuitka data-directory argument. A generic CLI path-validation
suggestion was declined because the canonical checker is a read-only,
standard-library-only tool whose documented contract permits explicit artifact
directories outside the repository; archive member paths remain strictly
validated.

ADR: [ADR-093](../decisions/093-offline-tiktoken-runtime-assets.md), extending
ADR-032 and ADR-073. Independent spec and code-quality reviews were incorporated
through focused regression commits before this closeout.
<!-- SECTION:NOTES:END -->
