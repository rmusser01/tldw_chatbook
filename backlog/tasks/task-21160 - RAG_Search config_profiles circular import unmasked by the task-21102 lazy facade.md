---
id: TASK-21160
title: >-
  RAG_Search config_profiles circular import unmasked by the task-21102 lazy facade
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - regression
  - rag
  - imports
priority: high
dependencies: []
---

## Description

Source: TASK-21103's adversarial review origin-traced a circular-import failure that the
implementer had labeled "pre-existing": `import tldw_chatbook.RAG_Search.config_profiles`
as the FIRST RAG_Search import raised
`ImportError: cannot import name 'get_profile_manager' from partially initialized module`,
and standalone collection of `Tests/UI/test_console_runtime_ownership.py` failed the same
way. Decisive A/B (read-only `git archive` of pre-21102 `56e2de875`): both work before
TASK-21102 (`d60ebe1d0`), both fail after — **the lazy `RAG_Search/__init__` facade
unmasked latent cycle edges** that the old eager init had front-loaded in the safe order.

Cycle: `config_profiles.py:20` → `.simplified.config` → executes `simplified/__init__` →
eagerly executes `enhanced_rag_service_v2` / `rag_factory` / `active_config`, each of which
imported `..config_profiles` back at module level → partially-initialized module →
ImportError on any `config_profiles`-first order.

## Acceptance Criteria (the what)

- [x] `import tldw_chatbook.RAG_Search.config_profiles` succeeds as the first RAG_Search
  import (fresh subprocess), and the historically-safe simplified-first order still works
- [x] Standalone `pytest Tests/UI/test_console_runtime_ownership.py --collect-only`
  collects again (13 tests)
- [x] A regression guard pins both import orders AND statically censuses `simplified/*`
  for module-level `config_profiles` imports so the edge class cannot silently return
- [x] No behavior change: profile loading, active-config, and factory paths keep their
  semantics (existing RAG suites green; annotations preserved via TYPE_CHECKING /
  future-annotations)

## Implementation Notes

All three cycle edges broken by deferring the `config_profiles` import to use-site:

- `simplified/enhanced_rag_service_v2.py` — added `from __future__ import annotations`
  (module has no dataclass/pydantic/get_type_hints annotation introspection — verified);
  the five-name module-level import became TYPE_CHECKING (typing names) + a
  `_config_profiles()` lazy accessor for the two runtime uses (4× `get_profile_manager()`,
  1× `isinstance(config, ProfileConfig)`). No production or test importer pulls the
  config_profiles names from this module (census: only `EnhancedRAGServiceV2`,
  `_tag_first_result`, factory functions).
- `simplified/rag_factory.py` — `get_profile_manager` function-local at its two use sites
  (in-file precedent: `pipeline_loader.py:336`).
- `simplified/active_config.py` — module already had future-annotations; import became
  TYPE_CHECKING (`ProfileConfig` annotation) + function-local at the three runtime sites
  (`get_profile_manager`, `_slugify`, `ProfileConfig` constructor).

Red-first: on the pre-fix tree the new guard fails 2/3 (config_profiles-first subprocess +
static census; simplified-first passes, as expected — that order was always safe); 3/3 with
the fix. Standalone collection of the ownership file: uncollectable → 13 collected.

Evidence for the origin trace lives in TASK-21103's review record; the fix comment in
`enhanced_rag_service_v2.py` carries the full account for future readers.
