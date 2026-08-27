---
id: TASK-2526
title: Ship tiktoken and its encoding tables for offline token estimates
status: In Progress
assignee: []
created_date: '2026-08-06 02:22'
updated_date: '2026-08-27 18:10'
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

Baseline investigation on 2026-08-27 also found that the old fallback test did
not actually select the fallback tier when `tiktoken` was installed. Its real
token count then leaked through the process-global estimate cache into a later
fixture. The task must make those tier-specific tests explicit and cache-clean.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `tiktoken` remains a mandatory base dependency and its absence fails
      the standard tokenizer test instead of silently skipping it.
- [ ] #2 The exact GPT-2, r50k, p50k, cl100k, and o200k tables requested by
      Chatbook and its chunking engine ship in both the wheel and source
      distribution.
- [ ] #3 With no explicit cache override, source and installed-distribution
      tokenization uses the immutable bundled tables and makes no encoding
      download attempt.
- [ ] #4 An explicitly supplied `TIKTOKEN_CACHE_DIR` remains authoritative.
- [ ] #5 Tier-specific tests explicitly select and isolate the real-tokenizer
      and character-fallback paths; the existing fallback behavior remains
      available when `tiktoken` is absent.
- [ ] #6 The runtime-asset policy, update procedure, provenance, and offline
      limitation for newly introduced encodings are documented.
<!-- AC:END -->

## Design

- [Approved design](../../Docs/superpowers/specs/2026-08-27-offline-tiktoken-runtime-assets-design.md)
- [ADR-093](../decisions/093-offline-tiktoken-runtime-assets.md)
