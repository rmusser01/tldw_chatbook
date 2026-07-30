---
id: TASK-1452
title: >-
  Central env-scaled Hypothesis profile; stop module-level load_profile() leaks that made example counts collection-order-dependent
status: In Progress
assignee: []
created_date: '2026-07-30 09:05'
labels:
  - testing
  - performance
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hypothesis binds `settings.default` at decoration time (verified by probe on installed 6.152.3), and four test modules call `settings.load_profile()` at import with no restore — so every module imported after one of them binds *that* module's profile instead of the central `tldw` profile TASK-1260 added. In a full run, `Tests/Utils/test_path_validation_properties.py` and the other 30+ unannotated `@given` sites run under `RAG_Search`'s "embeddings" profile; in a directory run they run under `tldw`. Effective example counts and deadlines depended on which files happened to be collected. Also: the suite had no example-count scaling at all (default 100 everywhere), part of audit driver #7, and `test_chachanotes_db_properties.py` registered "db_friendly" twice, with the second registration silently dropping the `function_scoped_fixture` suppression the first called "THE FIX".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] The central profile scales `max_examples`/`stateful_step_count` via `TLDW_HYPOTHESIS_PROFILE` (dev=25/20 default, ci=50/30, thorough=300/100), keeping TASK-1260's `deadline=None` policy
- [ ] No module leaves a non-central profile active after import: property modules needing extra health-check suppressions register a child profile (`parent=settings.default`) and restore `tldw` at end of module
- [ ] Redundant module profiles (RAG_Search "embeddings") are removed rather than inherited
- [ ] The ChaChaNotesDB double-registration is consolidated with the fixture suppression retained
- [ ] Affected property suites (ChaChaNotesDB, Media_DB, RAG_Search, Utils path-validation) pass; junit outcome diff vs baseline shows no regressions

## Implementation Plan

1. Probe decoration-time vs run-time binding on the installed Hypothesis (decides the mechanism)
2. Extend the central `tldw` profile with env-scaled example counts
3. Media_DB + ChaChaNotesDB + Prompts_DB properties: child profile via `parent=settings.default`, restore `tldw` at end of module; consolidate the ChaChaNotes double-registration
4. RAG_Search properties: delete the module profile (fully superseded by the central one)
5. Verify affected directories + junit diff

## Implementation Notes

Mechanism chosen after probing: decoration-time binding means each module's own
tests keep their intended settings while the *leak* is what harmed everyone else,
so per-module child profiles + end-of-module restore preserves current per-module
behavior exactly while making every other module's binding deterministic.
Deadline overrides (1000/1500/2000/5000) were dropped in favor of the central
deadline=None per TASK-1260's policy (deadlines measure the machine, not the
code). `tests_prompts_db_properties.py` is not currently collected (task-1463)
but got the same hygiene so enabling it later cannot reintroduce the leak.
Modified: `Tests/conftest.py`, `Tests/ChaChaNotesDB/test_chachanotes_db_properties.py`,
`Tests/Media_DB/test_media_db_properties.py`, `Tests/RAG_Search/test_embeddings_properties.py`,
`Tests/Prompts_DB/tests_prompts_db_properties.py`.
