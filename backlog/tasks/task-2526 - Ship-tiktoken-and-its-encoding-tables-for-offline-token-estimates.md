---
id: TASK-2526
title: Ship tiktoken and its encoding tables for offline token estimates
status: In Progress
assignee: []
created_date: '2026-08-06 02:22'
updated_date: '2026-08-27 19:23'
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
- [ ] #1 `tiktoken==0.14.0` is a mandatory reviewed base dependency and its
      absence fails the standard tokenizer test instead of silently skipping
      it; built metadata retains that exact requirement.
- [ ] #2 The exact GPT-2, r50k, p50k, cl100k, and o200k tables requested by
      Chatbook and its chunking engine ship in both the wheel and source
      distribution with the accepted redistribution evidence.
- [ ] #3 With no explicit cache override, source and installed-distribution
      tokenization uses a guarded immutable bundle; missing/corrupt entries make
      zero fetch attempts and cannot mutate the package tree.
- [ ] #4 An explicitly supplied `TIKTOKEN_CACHE_DIR` or legacy
      `DATA_GYM_CACHE_DIR` remains byte-for-byte authoritative and uses
      tiktoken's normal cache/download behavior.
- [ ] #5 Tier-specific tests explicitly select and isolate the real-tokenizer
      and character-fallback paths; token estimates retain their existing
      character fallback when `tiktoken` is absent, while token chunking keeps
      ADR-073's fail-closed compatibility-shim contract.
- [ ] #6 The runtime-asset policy, update procedure, provenance, and offline
      limitation for newly introduced encodings are documented, including the
      redistribution basis and required notice/license files.
- [ ] #7 The canonical release checker requires the exact cache inventory and
      notices, rejects unexpected entries, and validates source-built plus
      sdist-rebuilt wheels from a read-only installed tree.
<!-- AC:END -->

## Implementation Plan

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
