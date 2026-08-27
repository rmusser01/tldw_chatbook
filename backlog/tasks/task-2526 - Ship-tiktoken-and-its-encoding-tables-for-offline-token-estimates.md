---
id: TASK-2526
title: Ship tiktoken and its encoding tables for offline token estimates
status: To Do
assignee: []
created_date: '2026-08-06 02:22'
updated_date: '2026-08-27 18:56'
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

Implementation is currently gated on documenting redistribution permission for
the separately hosted non-GPT-2 BPE tables. Tiktoken's package license, source
URLs, and hashes do not by themselves establish that permission.

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
- [ ] #2 After a defensible redistribution basis is documented for each asset,
      the exact GPT-2, r50k, p50k, cl100k, and o200k tables requested by
      Chatbook and its chunking engine ship in both the wheel and source
      distribution.
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

## Design

- [Proposed design (licensing-gated)](../../Docs/superpowers/specs/2026-08-27-offline-tiktoken-runtime-assets-design.md)
- [ADR-093](../decisions/093-offline-tiktoken-runtime-assets.md)

## Blocker

Paused on 2026-08-27 by owner direction. Preserve the full offline-tokenization
goal and do not implement the narrower GPT-2-only fallback design. Resume only
after a defensible redistribution basis is documented for the separately
hosted r50k, p50k, cl100k, and o200k tables. An OpenAI collaborator stated in
[tiktoken issue #92](https://github.com/openai/tiktoken/issues/92#issuecomment-1497875652)
that the repository license applies to the encoding files; a follow-up asked
for that statement in the repository license or another non-issue artifact.
The Chatbook repository owner has not yet accepted the issue comment alone as
sufficient release evidence. Source URLs and integrity hashes are not by
themselves redistribution permission.
